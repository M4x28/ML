from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import warnings

import optuna
import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch.utils.tensorboard import SummaryWriter

from cup_data import CupDataModule
from cup_model import CupLinearModel
from cup_metrics import evaluate_regression_metrics

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for name in ("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning"):
    logging.getLogger(name).setLevel(logging.ERROR)
logging.disable(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def get_param_space() -> dict[str, list[Any]]:
    """Return the default grid of hyperparameters for the search."""
    return {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "batch_size": [16, 32, 64, 128, 256],
        "epochs": [200, 400, 800],
        "l2": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "feature_map": ["identity", "poly2"],
    }


def _grid_size(space: dict[str, list[Any]]) -> int:
    """Compute the Cartesian product size of the search space."""
    total = 1
    for values in space.values():
        total *= max(len(values), 1)
    return total


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) ignoring NaNs; NaN if all values are NaN."""
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
    return agg


class BestMetricsTracker(Callback):
    """Track the best validation metric and optionally store model state."""

    def __init__(self, *, monitor: str = "val_mee", mode: str = "min", track_state: bool = True) -> None:
        """Initialize the tracker with the metric name and mode."""
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.track_state = track_state
        self.best_epoch: int | None = None
        self.best_metrics: dict[str, float] = {}
        self.best_state: dict[str, torch.Tensor] | None = None
        if mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = float("-inf")

    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Capture metrics and weights when the monitored value improves."""
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return

        value = metrics[self.monitor]
        current = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        improved = current < self.best_score if self.mode == "min" else current > self.best_score
        if not improved:
            return

        self.best_score = current
        self.best_epoch = trainer.current_epoch + 1

        def _get(name: str) -> float:
            """Read a scalar metric from callback_metrics with NaN fallback."""
            metric = metrics.get(name)
            if metric is None:
                return float("nan")
            if isinstance(metric, torch.Tensor):
                return float(metric.detach().cpu())
            return float(metric)

        # Snapshot metrics at the best epoch for reporting.
        self.best_metrics = {
            "train_loss": _get("train_loss"),
            "train_mse": _get("train_mse"),
            "train_mee": _get("train_mee"),
            "val_loss": _get("val_loss"),
            "val_mse": _get("val_mse"),
            "val_mee": _get("val_mee"),
        }

        if self.track_state:
            # Clone weights to restore the best model later.
            self.best_state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}


class OptunaPruningCallback(Callback):
    """Report metrics to Optuna and prune unpromising trials."""

    def __init__(self, trial: optuna.trial.Trial, *, monitor: str) -> None:
        """Initialize pruning callback with the given trial and metric."""
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Report the monitored metric and stop the trial if pruned."""
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        current = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        step = trainer.current_epoch
        self.trial.report(current, step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at step {step} ({self.monitor}={current:.6f})")


def train_one_trial(
    *,
    train_path: str,
    test_path: str | None,
    params: dict[str, Any],
    val_ratio: float,
    test_ratio: float,
    seeds: list[int],
    num_workers: int,
    pin_memory: bool,
    scale_inputs: bool,
    split_seed: int,
    patience: int,
    accelerator: str,
    target_index: int | None = None,
    trial: optuna.trial.Trial | None = None,
) -> tuple[dict[str, float], list[dict[str, float]], int | None]:
    """Train one hyperparameter configuration across multiple seeds."""
    per_seed: list[dict[str, float]] = []
    best_epochs: list[int] = []

    for seed in seeds:
        # Ensure deterministic splits and weight initialization per seed.
        seed_everything(seed)

        data_module = CupDataModule(
            train_path=train_path,
            test_path=test_path,
            batch_size=int(params["batch_size"]),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_seed=split_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            scale_inputs=scale_inputs,
            feature_map=str(params.get("feature_map", "identity")),
            target_index=target_index,
        )
        data_module.setup("fit")
        assert data_module.input_dim is not None
        assert data_module.target_dim is not None

        model = CupLinearModel(
            input_dim=data_module.input_dim,
            output_dim=data_module.target_dim,
            lr=float(params["lr"]),
            l2_reg=float(params["l2"]),
        )

        metrics_cb = BestMetricsTracker(monitor="val_mee", mode="min", track_state=True)
        callbacks: list[Callback] = [metrics_cb]
        if patience > 0 and val_ratio > 0:
            callbacks.append(EarlyStopping(monitor="val_mee", patience=patience, mode="min"))
        if trial is not None:
            callbacks.append(OptunaPruningCallback(trial, monitor="val_mee"))

        trainer = Trainer(
            max_epochs=int(params["epochs"]),
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
            accelerator=accelerator,
            enable_progress_bar=False,
        )
        trainer.fit(model, datamodule=data_module)

        metrics: dict[str, float] = {}
        if metrics_cb.best_epoch is not None:
            best_epochs.append(metrics_cb.best_epoch)

        if metrics_cb.best_state is not None:
            # Restore the best validation weights before metric evaluation.
            model.load_state_dict(metrics_cb.best_state)

        train_metrics = evaluate_regression_metrics(
            model,
            data_module.train_dataloader(),
            l2_reg=float(params["l2"]),
        )
        metrics["train_loss"] = float(train_metrics["loss"])
        metrics["train_mse"] = float(train_metrics["mse"])
        metrics["train_mee"] = float(train_metrics["mee"])

        val_metrics = evaluate_regression_metrics(
            model,
            data_module.val_dataloader(),
            l2_reg=float(params["l2"]),
        )
        metrics["val_loss"] = float(val_metrics["loss"])
        metrics["val_mse"] = float(val_metrics["mse"])
        metrics["val_mee"] = float(val_metrics["mee"])

        test_metrics = evaluate_regression_metrics(
            model,
            data_module.test_dataloader(),
            l2_reg=float(params["l2"]),
        )
        metrics["test_loss"] = float(test_metrics["loss"])
        metrics["test_mse"] = float(test_metrics["mse"])
        metrics["test_mee"] = float(test_metrics["mee"])

        metrics["seed"] = int(seed)
        per_seed.append(metrics)

    agg = _aggregate_metrics(per_seed)
    best_epoch = None
    if best_epochs:
        # Use the average best epoch for reporting.
        best_epoch = int(round(sum(best_epochs) / len(best_epochs)))
    return agg, per_seed, best_epoch


def _selection_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Sort key for selecting the best trial (val_mee, val_mse, val_loss)."""
    def _safe(name: str) -> float:
        """Return a numeric metric, treating NaN as infinity."""
        value = metrics.get(name, float("inf"))
        if value != value:
            return float("inf")
        return float(value)

    return (_safe("val_mee_mean"), _safe("val_mse_mean"), _safe("val_loss_mean"))


def get_top_trials(study: optuna.study.Study, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Return top-k trials based on validation metrics."""
    candidates = [t for t in study.trials if "metrics" in t.user_attrs]
    # Sort with the same selection policy used for best trial.
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
                "best_epoch": trial.user_attrs.get("best_epoch"),
            }
        )
    return results


def _as_scalar_dict(values: dict[str, Any] | None) -> dict[str, float]:
    """Cast a dict of metrics to float values when possible."""
    if not values:
        return {}
    out: dict[str, float] = {}
    for key, value in values.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def log_trial_summary(
    *,
    log_dir: str | Path,
    trial_number: int,
    params: dict[str, Any],
    metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    best_epoch: int | None,
) -> None:
    """Write trial parameters and metrics to TensorBoard."""
    trial_dir = Path(log_dir) / f"trial_{trial_number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(trial_dir))

    hparams = _as_scalar_dict(params)
    metric_scalars = _as_scalar_dict(metrics)
    if test_metrics:
        for key, value in _as_scalar_dict(test_metrics).items():
            metric_scalars[f"{key}"] = value
    if best_epoch is not None:
        writer.add_text("best_epoch", str(best_epoch), 0)

    if hparams and metric_scalars:
        # TensorBoard hparams summary helps compare trials quickly.
        writer.add_hparams(hparams, metric_scalars)
    for key, value in metric_scalars.items():
        writer.add_scalar(key, value, 0)

    writer.flush()
    writer.close()


def run_optuna_search(
    *,
    train_path: str,
    test_path: str | None,
    val_ratio: float,
    test_ratio: float,
    seeds: list[int] | None,
    num_workers: int,
    pin_memory: bool,
    scale_inputs: bool,
    split_seed: int,
    patience: int,
    accelerator: str,
    n_trials: int,
    n_jobs: int,
    optuna_seed: int,
    target_index: int | None = None,
    param_space: dict[str, list[Any]] | None = None,
    top_k: int = 5,
    tb_log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run Optuna search over a categorical grid and return summaries."""
    if val_ratio <= 0:
        raise ValueError("Optuna search requires val_ratio > 0.")

    space = param_space or get_param_space()
    total = _grid_size(space)
    # n_trials=0 means evaluate the full grid.
    n_trials = total if n_trials <= 0 else n_trials

    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.trial.Trial) -> float:
        """Optuna objective: train and return validation MEE."""
        # Use categorical suggestions to emulate grid search.
        params = {name: trial.suggest_categorical(name, values) for name, values in space.items()}
        metrics, seed_metrics, best_epoch = train_one_trial(
            train_path=train_path,
            test_path=test_path,
            params=params,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seeds=seeds or SEEDS,
            num_workers=num_workers,
            pin_memory=pin_memory,
            scale_inputs=scale_inputs,
            split_seed=split_seed,
            patience=patience,
            accelerator=accelerator,
            target_index=target_index,
            trial=trial,
        )
        if not metrics or "val_mee_mean" not in metrics:
            raise RuntimeError("Validation metrics unavailable; check val_ratio and logging.")

        # Persist rich info for later analysis.
        trial.set_user_attr("params", params)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("seed_metrics", seed_metrics)
        trial.set_user_attr("best_epoch", best_epoch)
        if tb_log_dir is not None:
            log_trial_summary(
                log_dir=tb_log_dir,
                trial_number=trial.number,
                params=params,
                metrics=metrics,
                test_metrics=None,
                best_epoch=best_epoch,
            )
        print(
            f"[Optuna] trial={trial.number} "
            f"val_mee={metrics.get('val_mee_mean', float('nan')):.6f} "
            f"val_mse={metrics.get('val_mse_mean', float('nan')):.6f} "
            f"val_loss={metrics.get('val_loss_mean', float('nan')):.6f} "
            f"params={params}"
        )
        return float(metrics["val_mee_mean"])

    study.optimize(objective, n_trials=n_trials, n_jobs=max(int(n_jobs), 1))

    if not study.trials:
        raise RuntimeError("No trials completed.")

    # Select the best trial using the same ordering as in summaries.
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
            entry["best_epoch"] = trial.user_attrs.get("best_epoch")
        trials_summary.append(entry)

    return {
        "best_params": dict(best_trial.user_attrs.get("params", best_trial.params)),
        "best_metrics": dict(best_trial.user_attrs.get("metrics", {})),
        "best_epoch": best_trial.user_attrs.get("best_epoch"),
        "study": study,
        "n_trials": n_trials,
        "top_trials": get_top_trials(study, top_k=top_k),
        "trials": trials_summary,
        "target_index": target_index,
    }
