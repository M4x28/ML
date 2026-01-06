from __future__ import annotations

"""Hyperparameter search utilities for KNN CUP models."""

from typing import Any
import logging
import warnings

import numpy as np
import optuna

from cup_data import load_cup_train, split_train_val_test
from cup_metrics import evaluate_metrics
from knn_model import build_knn_pipeline

# Seeds used to average metrics across multiple splits.
SEEDS = [0, 1, 2, 3, 4]

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def get_param_space() -> dict[str, list[Any]]:
    """Grid search space used by Optuna (categorical sampling)."""
    return {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "metric": ["minkowski", "manhattan", "chebyshev"],
        "algorithm": ["auto"],
        "leaf_size": [30],
        "feature_map": ["identity", "poly2"],
        "pca_components": [None, 0.9, 0.95, 0.98],
        "scaler_type": ["none", "standard", "robust", "power"],
    }


def _grid_size(space: dict[str, list[Any]]) -> int:
    """Total number of combinations in a categorical grid."""
    total = 1
    for values in space.values():
        total *= max(len(values), 1)
    return total


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean/std while ignoring NaNs."""
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan"), float("nan")
    # Ignore NaNs to support missing test metrics.
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _aggregate_metrics(per_seed: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics across seeds with mean and std."""
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
        """Compute per-target mean metrics across seeds."""
        valid = [v for v in values if v]
        if not valid:
            return []
        arr = np.asarray(valid, dtype=float)
        return [float(x) for x in np.nanmean(arr, axis=0)]

    def _std_targets(values: list[list[float] | None]) -> list[float]:
        """Compute per-target std metrics across seeds."""
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


def _selection_key(metrics: dict[str, float]) -> tuple[float, float]:
    """Sort trials by validation MEE first, then validation MSE."""
    def _safe(name: str) -> float:
        """Return a finite metric for sorting, defaulting to inf."""
        value = metrics.get(name, float("inf"))
        if value != value:
            return float("inf")
        return float(value)

    return (_safe("val_mee_mean"), _safe("val_mse_mean"))


def _format_per_target(values: list[float] | None) -> str:
    """Format per-target metrics for logging."""
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
    target_idx: int | None = None,
    n_jobs: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Train a single set of params across multiple seeds."""
    _ids, X, y = load_cup_train(train_path)
    per_seed: list[dict[str, float]] = []

    for seed in seeds:
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
            # Multi-target training uses the full target matrix.
            y_tr_target = y_tr
            y_val_target = y_val
            y_te_target = y_te
        else:
            # Per-target training uses a single output column.
            y_tr_target = y_tr[:, target_idx]
            y_val_target = y_val[:, target_idx]
            y_te_target = y_te[:, target_idx] if y_te is not None else None

        model = build_knn_pipeline(
            params=params,
            scale_inputs=scale_inputs,
            n_jobs=n_jobs,
        )
        model.fit(X_tr, y_tr_target)

        # Evaluate consistently on train/val/test.
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
    """Return the best trials with stored metrics in sorted order."""
    candidates = [t for t in study.trials if "metrics" in t.user_attrs]
    candidates.sort(key=lambda t: _selection_key(t.user_attrs.get("metrics", {})))
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
    target_idx: int | None = None,
    n_jobs: int,
    n_trials: int,
    optuna_seed: int,
    param_space: dict[str, list[Any]] | None = None,
    top_k: int = 5,
    storage: str | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Run Optuna search over the categorical grid."""
    if val_ratio <= 0:
        raise ValueError("Optuna search requires val_ratio > 0.")

    space = param_space or get_param_space()
    # If n_trials is 0, evaluate the full categorical grid.
    total = _grid_size(space)
    n_trials = total if n_trials <= 0 else n_trials

    # Use TPE for efficient sampling over categorical values.
    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage),
    )

    def objective(trial: optuna.trial.Trial) -> float:
        """Optuna objective returning validation MEE for sampled params."""
        params = {name: trial.suggest_categorical(name, values) for name, values in space.items()}
        metrics, seed_metrics = train_one_trial(
            train_path=train_path,
            params=params,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seeds=seeds or SEEDS,
            split_seed=split_seed,
            scale_inputs=scale_inputs,
            target_idx=target_idx,
            n_jobs=n_jobs,
        )

        if metrics.get("val_mee_mean", float("nan")) != metrics.get("val_mee_mean", float("nan")):
            raise optuna.TrialPruned("val_mee is NaN")

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

    # Run the search, optionally in parallel.
    study.optimize(objective, n_trials=n_trials, n_jobs=max(int(n_jobs), 1))

    if not study.trials:
        raise RuntimeError("No trials completed.")

    # Select the best trial using the same sorting used for reporting.
    best_trial = min(
        (t for t in study.trials if "metrics" in t.user_attrs),
        key=lambda t: _selection_key(t.user_attrs["metrics"]),
    )

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
        "study": study,
        "n_trials": n_trials,
        "top_trials": get_top_trials(study, top_k=top_k),
        "trials": trials_summary,
    }


def run_search(method: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch to the requested search implementation."""
    if method != "tpe":
        raise ValueError(f"Unknown search method: {method}")
    return run_optuna_search(**kwargs)
