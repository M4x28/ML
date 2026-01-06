from __future__ import annotations

import math
from typing import Any
import logging
import warnings

import numpy as np
import optuna

from Cup.Leo.common.cup_data import load_cup_train, split_train_val_test
from Cup.Leo.common.cup_metrics import evaluate_metrics
from svm_model import build_svm_pipeline

SEEDS = [0, 1, 2, 3, 4]
MAX_MEE_GAP = 7.0

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def get_param_space() -> dict[str, list[Any]]:
    """Return the default hyperparameter grid for SVR."""
    return {
        "kernel": ["rbf", "poly", "linear"],
        "C": [1e-2, 1e-1, 1, 10, 100],
        "epsilon": [1e-3, 1e-2, 1e-1],
        "gamma": [1e-4, 1e-3, 1e-2, 1e-1, "scale"],
        "degree": [2, 3],
        "coef0": [0.0, 1.0],
        "feature_map": ["identity", "poly2"],
        "pca_components": [0.9, 0.95, 0.98],
    }


def _grid_size(space: dict[str, list[Any]]) -> int:
    """Compute the Cartesian product size of a parameter grid."""
    total = 1
    for values in space.values():
        total *= max(len(values), 1)
    return total


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean/std while ignoring NaNs."""
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _aggregate_metrics(per_seed: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-seed metrics into mean/std values."""
    keys = [
        "train_mse",
        "train_mee",
        "val_mse",
        "val_mee",
        "test_mse",
        "test_mee",
    ]
    agg: dict[str, float] = {}
    for key in keys:
        values = [m.get(key, float("nan")) for m in per_seed]
        mean, std = _mean_std(values)
        agg[f"{key}_mean"] = mean
        agg[f"{key}_std"] = std

    def _mean_targets(values: list[list[float] | None]) -> list[float]:
        """Average per-target metrics across seeds."""
        valid = [v for v in values if v]
        if not valid:
            return []
        arr = np.asarray(valid, dtype=float)
        return [float(x) for x in np.nanmean(arr, axis=0)]

    def _std_targets(values: list[list[float] | None]) -> list[float]:
        """Stddev of per-target metrics across seeds."""
        valid = [v for v in values if v]
        if not valid:
            return []
        arr = np.asarray(valid, dtype=float)
        return [float(x) for x in np.nanstd(arr, axis=0)]

    agg["train_mee_per_target_mean"] = _mean_targets(
        [m.get("train_mee_per_target") for m in per_seed]
    )
    agg["train_mee_per_target_std"] = _std_targets(
        [m.get("train_mee_per_target") for m in per_seed]
    )
    agg["val_mee_per_target_mean"] = _mean_targets(
        [m.get("val_mee_per_target") for m in per_seed]
    )
    agg["val_mee_per_target_std"] = _std_targets(
        [m.get("val_mee_per_target") for m in per_seed]
    )
    agg["test_mee_per_target_mean"] = _mean_targets(
        [m.get("test_mee_per_target") for m in per_seed]
    )
    agg["test_mee_per_target_std"] = _std_targets(
        [m.get("test_mee_per_target") for m in per_seed]
    )

    return agg


def _safe_float(value: object) -> float:
    """Convert values to float, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _extract_mee(metrics: dict[str, float]) -> tuple[float, float, float] | None:
    """Return (train, val, test) MEE if all are finite."""
    train = _safe_float(metrics.get("train_mee_mean"))
    val = _safe_float(metrics.get("val_mee_mean"))
    test = _safe_float(metrics.get("test_mee_mean"))
    if not all(math.isfinite(value) for value in (train, val, test)):
        return None
    return train, val, test


def _mee_gap(train: float, val: float, test: float) -> float:
    """Return the max distance among train/val/test MEE."""
    return max(train, val, test) - min(train, val, test)


def _mee_order_ok(train: float, val: float, test: float) -> bool:
    """Return True if train < val < test."""
    return train < val < test


def _rank_metrics(metrics: dict[str, float]) -> tuple[float, float, float, float, float]:
    """Ranking key within a candidate set (lower is better)."""
    triplet = _extract_mee(metrics)
    if triplet is None:
        return (float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))
    train, val, test = triplet
    gap = _mee_gap(train, val, test)
    val_mse = _safe_float(metrics.get("val_mse_mean"))
    if not math.isfinite(val_mse):
        val_mse = float("inf")
    return (val, test, gap, val_mse, train)


def _trial_priority(metrics: dict[str, float]) -> tuple[int, float, float, float, float, float]:
    """Sort key for listing trials with gap/order preference."""
    triplet = _extract_mee(metrics)
    if triplet is None:
        return (4, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))
    train, val, test = triplet
    gap = _mee_gap(train, val, test)
    gap_ok = gap <= MAX_MEE_GAP
    order_ok = _mee_order_ok(train, val, test)
    if gap_ok and order_ok:
        group = 0
    elif gap_ok:
        group = 1
    elif order_ok:
        group = 2
    else:
        group = 3
    val_mse = _safe_float(metrics.get("val_mse_mean"))
    if not math.isfinite(val_mse):
        val_mse = float("inf")
    return (group, val, test, gap, val_mse, train)


def _select_best_candidate(
    candidates: list[tuple[object, dict[str, float]]],
) -> tuple[object, str]:
    """Select the best candidate with gap<=MAX_MEE_GAP and preferred ordering."""
    valid: list[tuple[object, dict[str, float], float, float, float, float]] = []
    for obj, metrics in candidates:
        triplet = _extract_mee(metrics)
        if triplet is None:
            continue
        train, val, test = triplet
        gap = _mee_gap(train, val, test)
        valid.append((obj, metrics, train, val, test, gap))

    if not valid:
        raise RuntimeError("No trials completed.")

    gap_candidates = [entry for entry in valid if entry[5] <= MAX_MEE_GAP]
    if gap_candidates:
        ordered = [entry for entry in gap_candidates if _mee_order_ok(entry[2], entry[3], entry[4])]
        chosen = ordered if ordered else gap_candidates
        note = "gap<=7 and train<val<test" if ordered else "gap<=7"
    else:
        chosen = valid
        note = "fallback_no_gap"

    best_entry = min(chosen, key=lambda entry: _rank_metrics(entry[1]))
    return best_entry[0], note


def _format_per_target(values: list[float] | None) -> str:
    """Format per-target values for logging."""
    if not values:
        return ""
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def train_one_trial(
    *,
    train_path: str,
    params: dict[str, Any],
    val_ratio: float,
    test_ratio: float,
    seeds: list[int],
    split_seed: int,
    scale_inputs: bool,
    use_pca: bool,
    target_idx: int | None = None,
    n_jobs: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Train one parameter set across multiple seeds."""
    _ids, X, y = load_cup_train(train_path)
    per_seed: list[dict[str, float]] = []

    for seed in seeds:
        # Use different split seeds for robustness.
        split = split_train_val_test(
            X,
            y,
            _ids,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed + seed,
        )
        X_tr, y_tr = split["train"]["X"], split["train"]["y"]
        X_val, y_val = split["val"]["X"], split["val"]["y"]
        X_te, y_te = split["test"]["X"], split["test"]["y"]

        if X_val is None or y_val is None:
            raise RuntimeError("Validation split is empty; set val_ratio > 0.")

        if target_idx is None:
            y_tr_target = y_tr
            y_val_target = y_val
            y_te_target = y_te
            multi_output = True
        else:
            # Single-output training for one target index.
            y_tr_target = y_tr[:, target_idx]
            y_val_target = y_val[:, target_idx]
            y_te_target = y_te[:, target_idx] if y_te is not None else None
            multi_output = False

        model = build_svm_pipeline(
            params=params,
            scale_inputs=scale_inputs,
            n_jobs=n_jobs,
            use_pca=use_pca,
            multi_output=multi_output,
        )
        model.fit(X_tr, y_tr_target)

        train_metrics = evaluate_metrics(model, X_tr, y_tr_target)
        val_metrics = evaluate_metrics(model, X_val, y_val_target)
        test_metrics = (
            evaluate_metrics(model, X_te, y_te_target)
            if X_te is not None and y_te_target is not None
            else {"mse": float("nan"), "mee": float("nan"), "mee_per_target": []}
        )

        metrics = {
            "seed": int(seed),
            "train_mse": float(train_metrics["mse"]),
            "train_mee": float(train_metrics["mee"]),
            "train_mee_per_target": train_metrics.get("mee_per_target"),
            "val_mse": float(val_metrics["mse"]),
            "val_mee": float(val_metrics["mee"]),
            "val_mee_per_target": val_metrics.get("mee_per_target"),
            "test_mse": float(test_metrics["mse"]),
            "test_mee": float(test_metrics["mee"]),
            "test_mee_per_target": test_metrics.get("mee_per_target"),
        }
        per_seed.append(metrics)

    agg = _aggregate_metrics(per_seed)
    return agg, per_seed


def get_top_trials(study: optuna.study.Study, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the top-k trials based on validation metrics."""
    candidates = [t for t in study.trials if "metrics" in t.user_attrs]
    candidates.sort(key=lambda t: _trial_priority(t.user_attrs.get("metrics", {})))
    results: list[dict[str, Any]] = []
    for trial in candidates[: max(int(top_k), 0)]:
        results.append(
            {
                "trial": int(trial.number),
                "value": float(trial.value) if trial.value is not None else float("nan"),
                "params": dict(trial.user_attrs.get("params", trial.params)),
                "metrics": dict(trial.user_attrs.get("metrics", {})),
                "seed_metrics": trial.user_attrs.get("seed_metrics"),
            }
        )
    return results


def run_optuna_search(
    *,
    train_path: str,
    val_ratio: float,
    test_ratio: float,
    seeds: list[int] | None,
    split_seed: int,
    scale_inputs: bool,
    use_pca: bool,
    pca_components: float | int | None = None,
    target_idx: int | None = None,
    n_jobs: int,
    n_trials: int,
    optuna_seed: int,
    param_space: dict[str, list[Any]] | None = None,
    top_k: int = 5,
    storage: str | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Run Optuna search over categorical grid parameters."""
    if val_ratio <= 0:
        raise ValueError("Optuna search requires val_ratio > 0.")

    space = param_space or get_param_space()
    total = _grid_size(space)
    n_trials = total if n_trials <= 0 else n_trials

    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage),
    )

    def objective(trial: optuna.trial.Trial) -> float:
        """Optuna objective: train and return validation MEE."""
        params = {name: trial.suggest_categorical(name, values) for name, values in space.items()}
        if use_pca and pca_components is not None and "pca_components" not in params:
            params["pca_components"] = pca_components
        if params.get("feature_map") == "poly2" and params.get("kernel") == "poly":
            raise optuna.TrialPruned("Invalid combo: feature_map=poly2 with kernel=poly")
        metrics, seed_metrics = train_one_trial(
            train_path=train_path,
            params=params,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seeds=seeds or SEEDS,
            split_seed=split_seed,
            scale_inputs=scale_inputs,
            use_pca=use_pca,
            target_idx=target_idx,
            n_jobs=n_jobs,
        )

        if metrics.get("val_mee_mean", float("nan")) != metrics.get("val_mee_mean", float("nan")):
            raise optuna.TrialPruned("val_mee is NaN")

        # Store details for later reporting.
        trial.set_user_attr("params", params)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("seed_metrics", seed_metrics)

        val_targets = _format_per_target(metrics.get("val_mee_per_target_mean"))
        targets_str = f" val_mee_targets={val_targets}" if val_targets else ""
        print(
            f"[Optuna] trial={trial.number} "
            f"val_mee={metrics.get('val_mee_mean', float('nan')):.6f} "
            f"val_mse={metrics.get('val_mse_mean', float('nan')):.6f} "
            f"{targets_str} params={params}"
        )
        return float(metrics["val_mee_mean"])

    study.optimize(objective, n_trials=n_trials, n_jobs=max(int(n_jobs), 1))

    if not study.trials:
        raise RuntimeError("No trials completed.")

    trial_candidates = [
        (t, t.user_attrs.get("metrics", {}))
        for t in study.trials
        if "metrics" in t.user_attrs
    ]
    best_trial, selection_note = _select_best_candidate(trial_candidates)

    trials_summary: list[dict[str, Any]] = []
    for trial in study.trials:
        entry: dict[str, Any] = {
            "trial": int(trial.number),
            "state": str(trial.state),
            "value": float(trial.value) if trial.value is not None else None,
            "params": dict(trial.params),
        }
        if "metrics" in trial.user_attrs:
            entry["metrics"] = dict(trial.user_attrs.get("metrics", {}))
            entry["seed_metrics"] = trial.user_attrs.get("seed_metrics")
        trials_summary.append(entry)

    return {
        "best_params": dict(best_trial.user_attrs.get("params", best_trial.params)),
        "best_metrics": dict(best_trial.user_attrs.get("metrics", {})),
        "selection_note": selection_note,
        "study": study,
        "n_trials": n_trials,
        "top_trials": get_top_trials(study, top_k=top_k),
        "trials": trials_summary,
    }


def _log_bounds(center: float, span: float, *, min_value: float, max_value: float) -> tuple[float, float]:
    """Compute log10 bounds for CMA-ES around a center."""
    if center <= 0:
        raise ValueError("CMA-ES center must be positive.")
    log_center = float(np.log10(center))
    low = max(log_center - span, float(np.log10(min_value)))
    high = min(log_center + span, float(np.log10(max_value)))
    if low >= high:
        raise ValueError("Invalid CMA-ES bounds; adjust span or center.")
    return low, high


def run_cmaes_search(
    *,
    train_path: str,
    val_ratio: float,
    test_ratio: float,
    seeds: list[int] | None,
    split_seed: int,
    scale_inputs: bool,
    use_pca: bool,
    pca_components: float | int | None = None,
    target_idx: int | None = None,
    n_jobs: int,
    n_trials: int,
    optuna_seed: int,
    kernel: str,
    feature_map: str,
    degree: int,
    coef0: float,
    center_C: float,
    center_gamma: float | None,
    center_epsilon: float,
    log_span: float = 1.0,
    top_k: int = 5,
    storage: str | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Run CMA-ES search for continuous SVR parameters."""
    if val_ratio <= 0:
        raise ValueError("CMA-ES search requires val_ratio > 0.")
    if feature_map == "poly2" and kernel == "poly":
        raise ValueError("Invalid combo: feature_map=poly2 with kernel=poly.")

    n_trials = int(n_trials)
    if n_trials <= 0:
        n_trials = 30

    sampler = optuna.samplers.CmaEsSampler(seed=optuna_seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage),
    )

    c_bounds = _log_bounds(center_C, log_span, min_value=1e-4, max_value=1e4)
    eps_bounds = _log_bounds(center_epsilon, log_span, min_value=1e-5, max_value=1.0)
    if kernel in ("rbf", "poly"):
        if center_gamma is None:
            raise ValueError("CMA-ES requires a numeric gamma for rbf/poly kernels.")
        gamma_bounds = _log_bounds(center_gamma, log_span, min_value=1e-6, max_value=1.0)
    else:
        gamma_bounds = None

    def objective(trial: optuna.trial.Trial) -> float:
        """CMA-ES objective in log-space for C/epsilon/gamma."""
        log_c = trial.suggest_float("log10_C", c_bounds[0], c_bounds[1])
        log_eps = trial.suggest_float("log10_epsilon", eps_bounds[0], eps_bounds[1])
        params = {
            "kernel": kernel,
            "C": float(10 ** log_c),
            "epsilon": float(10 ** log_eps),
            "gamma": None,
            "degree": int(degree),
            "coef0": float(coef0),
            "feature_map": feature_map,
        }
        if use_pca and pca_components is not None:
            params["pca_components"] = pca_components
        if gamma_bounds is not None:
            log_gamma = trial.suggest_float("log10_gamma", gamma_bounds[0], gamma_bounds[1])
            params["gamma"] = float(10 ** log_gamma)

        metrics, seed_metrics = train_one_trial(
            train_path=train_path,
            params=params,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seeds=seeds or SEEDS,
            split_seed=split_seed,
            scale_inputs=scale_inputs,
            use_pca=use_pca,
            target_idx=target_idx,
            n_jobs=n_jobs,
        )

        if metrics.get("val_mee_mean", float("nan")) != metrics.get("val_mee_mean", float("nan")):
            raise optuna.TrialPruned("val_mee is NaN")

        # Store details for later reporting.
        trial.set_user_attr("params", params)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("seed_metrics", seed_metrics)

        val_targets = _format_per_target(metrics.get("val_mee_per_target_mean"))
        targets_str = f" val_mee_targets={val_targets}" if val_targets else ""
        print(
            f"[CMA-ES] trial={trial.number} "
            f"val_mee={metrics.get('val_mee_mean', float('nan')):.6f} "
            f"val_mse={metrics.get('val_mse_mean', float('nan')):.6f} "
            f"{targets_str} params={params}"
        )
        return float(metrics["val_mee_mean"])

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    if not study.trials:
        raise RuntimeError("No trials completed.")

    trial_candidates = [
        (t, t.user_attrs.get("metrics", {}))
        for t in study.trials
        if "metrics" in t.user_attrs
    ]
    best_trial, selection_note = _select_best_candidate(trial_candidates)

    trials_summary: list[dict[str, Any]] = []
    for trial in study.trials:
        entry: dict[str, Any] = {
            "trial": int(trial.number),
            "state": str(trial.state),
            "value": float(trial.value) if trial.value is not None else None,
            "params": dict(trial.user_attrs.get("params", trial.params)),
        }
        if "metrics" in trial.user_attrs:
            entry["metrics"] = dict(trial.user_attrs.get("metrics", {}))
            entry["seed_metrics"] = trial.user_attrs.get("seed_metrics")
        trials_summary.append(entry)

    return {
        "best_params": dict(best_trial.user_attrs.get("params", best_trial.params)),
        "best_metrics": dict(best_trial.user_attrs.get("metrics", {})),
        "selection_note": selection_note,
        "study": study,
        "n_trials": n_trials,
        "top_trials": get_top_trials(study, top_k=top_k),
        "trials": trials_summary,
    }


def run_smac_search(
    *,
    train_path: str,
    val_ratio: float,
    test_ratio: float,
    seeds: list[int] | None,
    split_seed: int,
    scale_inputs: bool,
    use_pca: bool,
    pca_components: float | int | None = None,
    target_idx: int | None = None,
    n_jobs: int,
    n_trials: int,
    optuna_seed: int,
    kernel: str,
    feature_map: str,
    degree: int,
    coef0: float,
    center_C: float,
    center_gamma: float | None,
    center_epsilon: float,
    log_span: float = 1.0,
    top_k: int = 5,
    storage: str | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Run SMAC search for continuous SVR parameters."""
    try:
        from ConfigSpace import ConfigurationSpace
        try:
            from ConfigSpace.hyperparameters import Float  # type: ignore
        except ImportError:
            from ConfigSpace.hyperparameters import UniformFloatHyperparameter as Float
        from smac import HyperparameterOptimizationFacade as HPOFacade
        from smac import Scenario
        from smac.initial_design import DefaultInitialDesign
    except ImportError as exc:
        raise RuntimeError(
            "SMAC is not installed. Install with: python -m pip install smac ConfigSpace"
        ) from exc

    if val_ratio <= 0:
        raise ValueError("SMAC search requires val_ratio > 0.")
    if feature_map == "poly2" and kernel == "poly":
        raise ValueError("Invalid combo: feature_map=poly2 with kernel=poly.")

    n_trials = int(n_trials)
    if n_trials <= 0:
        n_trials = 30

    c_bounds = _log_bounds(center_C, log_span, min_value=1e-4, max_value=1e4)
    eps_bounds = _log_bounds(center_epsilon, log_span, min_value=1e-5, max_value=1.0)
    gamma_bounds = None
    if kernel in ("rbf", "poly"):
        if center_gamma is None:
            raise ValueError("SMAC requires a numeric gamma for rbf/poly kernels.")
        gamma_bounds = _log_bounds(center_gamma, log_span, min_value=1e-6, max_value=1.0)

    center_log_c = float(np.log10(center_C))
    center_log_eps = float(np.log10(center_epsilon))

    cs = ConfigurationSpace()
    hp_log_c = Float("log10_C", c_bounds[0], c_bounds[1], default_value=center_log_c)
    hp_log_eps = Float("log10_epsilon", eps_bounds[0], eps_bounds[1], default_value=center_log_eps)
    cs.add_hyperparameters([hp_log_c, hp_log_eps])
    if gamma_bounds is not None:
        center_log_gamma = float(np.log10(center_gamma))
        hp_log_gamma = Float(
            "log10_gamma", gamma_bounds[0], gamma_bounds[1], default_value=center_log_gamma
        )
        cs.add_hyperparameters([hp_log_gamma])

    scenario = Scenario(cs, n_trials=n_trials, seed=optuna_seed, deterministic=True)

    records: list[dict[str, Any]] = []

    def _objective(config, seed: int | None = None) -> float:
        """SMAC objective: evaluate a single configuration."""
        log_c = float(config["log10_C"])
        log_eps = float(config["log10_epsilon"])
        params = {
            "kernel": kernel,
            "C": float(10 ** log_c),
            "epsilon": float(10 ** log_eps),
            "gamma": None,
            "degree": int(degree),
            "coef0": float(coef0),
            "feature_map": feature_map,
        }
        if use_pca and pca_components is not None:
            params["pca_components"] = pca_components
        if gamma_bounds is not None:
            log_gamma = float(config["log10_gamma"])
            params["gamma"] = float(10 ** log_gamma)

        metrics, seed_metrics = train_one_trial(
            train_path=train_path,
            params=params,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seeds=seeds or SEEDS,
            split_seed=split_seed,
            scale_inputs=scale_inputs,
            use_pca=use_pca,
            target_idx=target_idx,
            n_jobs=n_jobs,
        )

        if metrics.get("val_mee_mean", float("nan")) != metrics.get("val_mee_mean", float("nan")):
            raise RuntimeError("val_mee is NaN")

        # Keep a local record list since SMAC doesn't use optuna trials.
        records.append(
            {
                "params": params,
                "metrics": metrics,
                "seed_metrics": seed_metrics,
            }
        )

        val_targets = _format_per_target(metrics.get("val_mee_per_target_mean"))
        targets_str = f" val_mee_targets={val_targets}" if val_targets else ""
        print(
            f"[SMAC] trial={len(records) - 1} "
            f"val_mee={metrics.get('val_mee_mean', float('nan')):.6f} "
            f"val_mse={metrics.get('val_mse_mean', float('nan')):.6f} "
            f"{targets_str} params={params}"
        )
        return float(metrics["val_mee_mean"])

    initial_design = DefaultInitialDesign(scenario)
    smac = HPOFacade(scenario, _objective, initial_design=initial_design)
    smac.optimize()

    if not records:
        raise RuntimeError("No SMAC trials completed.")

    record_candidates = [(record, record["metrics"]) for record in records]
    best_record, selection_note = _select_best_candidate(record_candidates)
    trials_summary: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        trials_summary.append(
            {
                "trial": idx,
                "state": "COMPLETE",
                "value": float(record["metrics"]["val_mee_mean"]),
                "params": dict(record["params"]),
                "metrics": dict(record["metrics"]),
                "seed_metrics": record["seed_metrics"],
            }
        )

    top_trials = sorted(trials_summary, key=lambda r: _trial_priority(r["metrics"]))[
        : max(int(top_k), 0)
    ]

    return {
        "best_params": dict(best_record["params"]),
        "best_metrics": dict(best_record["metrics"]),
        "selection_note": selection_note,
        "study": None,
        "n_trials": n_trials,
        "top_trials": top_trials,
        "trials": trials_summary,
    }


def run_search(method: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch to the selected search method."""
    methods = {
        "tpe": run_optuna_search,
        "cmaes": run_cmaes_search,
        "smac": run_smac_search,
    }
    if method not in methods:
        raise ValueError(f"Unknown search method: {method}")
    return methods[method](**kwargs)
