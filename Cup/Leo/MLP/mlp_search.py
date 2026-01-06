from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping
from sklearn.model_selection import ParameterGrid

from cup_data import CupDataModule
from cup_autoencoder import AutoencoderConfig, apply_autoencoder_to_datamodule
from mlp_model import CupMLPModel

SEEDS = [0]


def get_param_space() -> dict[str, list[Any]]:
    """Return the default grid of hyperparameters."""
    return {
        "hidden_sizes": [
            [256, 128, 64],
            [256, 128],
        ],
        "activation": ["relu", "leakyrelu", "tanh"],
        "dropout": [0.0, 0.1],
        "lr": [1e-2, 7.5e-3, 5e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "optimizer": ["adam", "sgd"],
        "momentum": [None, 0.8],
        "batch_size": [64],
    }


def _apply_lr_scale(params: dict[str, Any], lr_scale: float) -> dict[str, Any]:
    """Scale the learning rate and annotate the scale factor."""
    scaled = dict(params)
    factor = float(lr_scale)
    if factor != 1.0:
        base_lr = float(scaled.get("lr", 1e-3))
        scaled["lr"] = base_lr * factor
        scaled["lr_scale"] = factor
    return scaled


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean/std while ignoring NaNs."""
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _aggregate_metrics(per_seed: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-seed metrics into mean/std values."""
    keys = [
        "train_loss",
        "train_mse",
        "train_mee",
        "val_loss",
        "val_mse",
        "val_mee",
        "test_loss",
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


def _aggregate_target_metrics(target_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-target metrics by averaging."""
    keys = [
        "train_loss",
        "train_mse",
        "train_mee",
        "val_loss",
        "val_mse",
        "val_mee",
    ]
    agg: dict[str, float] = {}
    for key in keys:
        values = [m.get(key, float("nan")) for m in target_metrics]
        mean, _ = _mean_std(values)
        agg[key] = mean
    return agg


def _resolve_batch_size(value: object, train_size: int, fallback: int) -> int:
    """Interpret batch size tokens like 'full' and fallback to defaults."""
    if isinstance(value, str):
        if value.lower() in ("full", "batch", "full_batch"):
            return max(int(train_size), 1)
        try:
            return int(value)
        except ValueError:
            return int(fallback)
    if value is None:
        return int(fallback)
    return int(value)


def _autoencoder_device(accelerator: str) -> torch.device:
    """Select device for autoencoder training."""
    if accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_autoencoder_config(
    autoencoder_params: dict[str, Any],
    *,
    batch_size: int,
    seed: int,
) -> AutoencoderConfig:
    """Build an AutoencoderConfig from a plain parameter dict."""
    hidden_dims = autoencoder_params.get("hidden_dims", (64, 32))
    latent_dims = autoencoder_params.get("latent_dims")
    return AutoencoderConfig(
        latent_dim=int(autoencoder_params.get("latent_dim", 8)),
        latent_dims=tuple(int(x) for x in latent_dims) if latent_dims else None,
        hidden_dims=tuple(int(x) for x in hidden_dims),
        activation=str(autoencoder_params.get("activation", "relu")),
        lr=float(autoencoder_params.get("lr", 1e-3)),
        weight_decay=float(autoencoder_params.get("weight_decay", 0.0)),
        epochs=int(autoencoder_params.get("epochs", 300)),
        patience=int(autoencoder_params.get("patience", 20)),
        batch_size=int(autoencoder_params.get("batch_size", batch_size)),
        seed=int(seed),
    )


class BestMetricsTracker(Callback):
    def __init__(self, *, monitor: str = "val_loss", mode: str = "min", track_state: bool = True) -> None:
        """Track the best validation metric and optionally keep model weights."""
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
        """Update the best checkpoint when monitored metric improves."""
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
            """Read a scalar metric from callback metrics."""
            metric = metrics.get(name)
            if metric is None:
                return float("nan")
            if isinstance(metric, torch.Tensor):
                return float(metric.detach().cpu())
            return float(metric)

        self.best_metrics = {
            "train_loss": _get("train_loss"),
            "train_mse": _get("train_mse"),
            "train_mee": _get("train_mee"),
            "val_loss": _get("val_loss"),
            "val_mse": _get("val_mse"),
            "val_mee": _get("val_mee"),
        }

        if self.track_state:
            # Clone weights to restore best validation model later.
            self.best_state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}


def train_one_trial(
    *,
    train_path: str,
    test_path: str,
    params: dict[str, Any],
    val_ratio: float,
    test_ratio: float,
    seeds: list[int],
    split_seed: int,
    num_workers: int,
    pin_memory: bool,
    scale_inputs: bool,
    patience: int,
    accelerator: str,
    max_epochs: int,
    grad_clip: float = 0.0,
    trial_label: str | None = None,
    verbose: bool = False,
    per_target: bool = False,
    use_autoencoder: bool = False,
    autoencoder_params: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[dict[str, float]], int | None]:
    """Train one hyperparameter configuration across multiple seeds."""
    per_seed: list[dict[str, float]] = []
    best_epochs: list[int] = []
    seed_total = len(seeds)
    label_prefix = f"{trial_label} " if trial_label else ""
    grad_clip_val = float(grad_clip) if grad_clip and float(grad_clip) > 0 else 0.0

    for seed_idx, seed in enumerate(seeds, start=1):
        if verbose:
            print(
                f"[Grid] {label_prefix}seed {seed_idx}/{seed_total} start (seed={seed})",
                flush=True,
            )
        seed_start = time.perf_counter()
        # Fix random seeds for deterministic splits and init.
        seed_everything(seed)

        raw_batch = params.get("batch_size", 32)
        init_batch = int(raw_batch) if not isinstance(raw_batch, str) else 32
        data_module = CupDataModule(
            train_path=train_path,
            test_path=test_path,
            batch_size=init_batch,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=int(split_seed + seed),
            seed=int(seed),
            num_workers=num_workers,
            pin_memory=pin_memory,
            scale_inputs=scale_inputs,
        )
        data_module.setup("fit")
        train_size = len(data_module.train_dataset) if data_module.train_dataset is not None else init_batch
        resolved_batch = _resolve_batch_size(raw_batch, train_size, init_batch)
        data_module.batch_size = resolved_batch
        data_module.predict_batch_size = resolved_batch
        autoencoder_metrics = None
        if use_autoencoder:
            # Train autoencoder and replace datasets with latent codes.
            ae_params = autoencoder_params or {}
            ae_config = _build_autoencoder_config(
                ae_params,
                batch_size=resolved_batch,
                seed=seed,
            )
            autoencoder_metrics = apply_autoencoder_to_datamodule(
                data_module,
                config=ae_config,
                device=_autoencoder_device(accelerator),
            )
        assert data_module.input_dim is not None
        assert data_module.target_dim is not None

        optimizer = str(params.get("optimizer", "adam")).lower()
        momentum = params.get("momentum")
        if optimizer == "sgd" and momentum is None:
            momentum = 0.9

        if per_target:
            # Train a separate single-output model for each target.
            target_metrics_list: list[dict[str, float]] = []
            target_epochs: list[int] = []
            for target_idx in range(int(data_module.target_dim)):
                model = CupMLPModel(
                    input_dim=data_module.input_dim,
                    output_dim=1,
                    hidden_sizes=params.get("hidden_sizes", [64, 64]),
                    activation=str(params.get("activation", "relu")),
                    dropout=float(params.get("dropout", 0.0)),
                    lr=float(params.get("lr", 1e-3)),
                    weight_decay=float(params.get("weight_decay", 0.0)),
                    optimizer=optimizer,
                    momentum=float(momentum) if momentum is not None else 0.0,
                )

                metrics_cb = BestMetricsTracker(monitor="val_loss", mode="min", track_state=True)
                callbacks: list[Callback] = [metrics_cb]
                if patience > 0 and val_ratio > 0:
                    callbacks.append(EarlyStopping(monitor="val_mee", patience=patience, mode="min"))

                epoch_limit = int(params.get("epochs", max_epochs))
                trainer = Trainer(
                    max_epochs=epoch_limit,
                    deterministic=True,
                    logger=False,
                    enable_checkpointing=False,
                    callbacks=callbacks,
                    accelerator=accelerator,
                    precision="16-mixed" if accelerator == "gpu" else "32-true",
                    gradient_clip_val=grad_clip_val,
                    gradient_clip_algorithm="norm",
                )

                train_loader = data_module.target_dataloader(
                    "train",
                    target_idx,
                    shuffle=True,
                    batch_size=resolved_batch,
                )
                val_loader = data_module.target_dataloader(
                    "val",
                    target_idx,
                    shuffle=False,
                    batch_size=resolved_batch,
                )
                if val_loader is not None:
                    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                else:
                    trainer.fit(model, train_dataloaders=train_loader)

                target_metrics = dict(metrics_cb.best_metrics)
                target_metrics["target_idx"] = int(target_idx)
                target_metrics_list.append(target_metrics)
                if metrics_cb.best_epoch is not None:
                    target_epochs.append(metrics_cb.best_epoch)

            metrics = _aggregate_target_metrics(target_metrics_list)
            # test_* metrics are computed later during final training.
            metrics["test_loss"] = float("nan")
            metrics["test_mse"] = float("nan")
            metrics["test_mee"] = float("nan")
            metrics["seed"] = int(seed)
            metrics["autoencoder"] = autoencoder_metrics
            metrics["per_target"] = target_metrics_list
            per_seed.append(metrics)

            if target_epochs:
                best_epochs.append(int(round(sum(target_epochs) / len(target_epochs))))
        else:
            # Multi-output model for all 4 targets.
            model = CupMLPModel(
                input_dim=data_module.input_dim,
                output_dim=data_module.target_dim,
                hidden_sizes=params.get("hidden_sizes", [64, 64]),
                activation=str(params.get("activation", "relu")),
                dropout=float(params.get("dropout", 0.0)),
                lr=float(params.get("lr", 1e-3)),
                weight_decay=float(params.get("weight_decay", 0.0)),
                optimizer=optimizer,
                momentum=float(momentum) if momentum is not None else 0.0,
            )

            metrics_cb = BestMetricsTracker(monitor="val_loss", mode="min", track_state=True)
            callbacks: list[Callback] = [metrics_cb]
            if patience > 0 and val_ratio > 0:
                callbacks.append(EarlyStopping(monitor="val_mee", patience=patience, mode="min"))

            epoch_limit = int(params.get("epochs", max_epochs))
            trainer = Trainer(
                max_epochs=epoch_limit,
                deterministic=True,
                logger=False,
                enable_checkpointing=False,
                callbacks=callbacks,
                accelerator=accelerator,
                precision="16-mixed" if accelerator == "gpu" else "32-true",
                gradient_clip_val=grad_clip_val,
                gradient_clip_algorithm="norm",
            )
            trainer.fit(model, datamodule=data_module)

            metrics = dict(metrics_cb.best_metrics)
            if metrics_cb.best_epoch is not None:
                best_epochs.append(metrics_cb.best_epoch)

            if metrics_cb.best_state is not None:
                # Restore best validation weights for metric reporting.
                model.load_state_dict(metrics_cb.best_state)

            metrics["test_loss"] = float("nan")
            metrics["test_mse"] = float("nan")
            metrics["test_mee"] = float("nan")

            metrics["seed"] = int(seed)
            metrics["autoencoder"] = autoencoder_metrics
            per_seed.append(metrics)
        if verbose:
            seed_elapsed = time.perf_counter() - seed_start
            best_epoch_label = metrics_cb.best_epoch if metrics_cb.best_epoch is not None else "n/a"
            print(
                f"[Grid] {label_prefix}seed {seed_idx}/{seed_total} done in {seed_elapsed:.1f}s | "
                f"best_epoch={best_epoch_label} "
                f"val_loss={metrics.get('val_loss', float('nan')):.6f} "
                f"val_mse={metrics.get('val_mse', float('nan')):.6f} "
                f"val_mee={metrics.get('val_mee', float('nan')):.6f} "
                f"test_mee={metrics.get('test_mee', float('nan')):.6f}",
                flush=True,
            )

    agg = _aggregate_metrics(per_seed)
    best_epoch = None
    if best_epochs:
        # Use average best epoch across seeds.
        best_epoch = int(round(sum(best_epochs) / len(best_epochs)))
    return agg, per_seed, best_epoch


def _selection_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Sort key used to rank trials (val_loss, val_mse, val_mee)."""
    def _safe(name: str) -> float:
        """Return a numeric metric, treating NaNs as infinity."""
        value = metrics.get(name, float("inf"))
        if value != value:
            return float("inf")
        return float(value)

    return (_safe("val_loss_mean"), _safe("val_mse_mean"), _safe("val_mee_mean"))


def _build_grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Filter and expand a parameter grid with optimizer constraints."""
    grid: list[dict[str, Any]] = []
    for params in ParameterGrid(space):
        optimizer = str(params.get("optimizer", "adam")).lower()
        momentum = params.get("momentum")
        if optimizer == "adam":
            if momentum is not None:
                continue
            params = dict(params)
            params["momentum"] = None
        elif optimizer == "sgd":
            if momentum is None:
                continue
        else:
            continue
        grid.append(dict(params))
    return grid


def _top_trials(trials: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    """Return the top-k trials sorted by validation metrics."""
    ordered = sorted(trials, key=lambda t: _selection_key(t.get("metrics", {})))
    return ordered[: max(int(top_k), 0)]


def run_grid_search(
    *,
    train_path: str,
    test_path: str,
    val_ratio: float,
    test_ratio: float,
    seeds: list[int] | None,
    split_seed: int,
    num_workers: int,
    pin_memory: bool,
    scale_inputs: bool,
    patience: int,
    accelerator: str,
    max_epochs: int,
    n_trials: int,
    param_space: dict[str, list[Any]] | None = None,
    top_k: int = 5,
    verbose: bool = True,
    per_target: bool = False,
    use_autoencoder: bool = False,
    autoencoder_params: dict[str, Any] | None = None,
    lr_scale: float = 1.0,
    grad_clip: float = 0.0,
) -> dict[str, Any]:
    """Run a grid search over hyperparameters and return summaries."""
    if val_ratio <= 0:
        raise ValueError("Grid search requires val_ratio > 0.")

    space = param_space or get_param_space()
    grid = _build_grid(space)
    total = len(grid)
    if total == 0:
        raise ValueError("Grid search has no valid combinations.")
    if n_trials > 0:
        grid = grid[: min(int(n_trials), total)]
    total_trials = len(grid)
    if verbose:
        mode = "per-target" if per_target else "multi-output"
        print(f"[Grid] combinations: {total} | running {total_trials} trials | mode={mode}", flush=True)

    trials: list[dict[str, Any]] = []
    for idx, params in enumerate(grid, start=1):
        trial_label = f"trial {idx}/{total_trials}"
        trial_params = _apply_lr_scale(params, lr_scale)
        if verbose:
            print(f"[Grid] {trial_label} params={trial_params}", flush=True)
        trial_start = time.perf_counter()
        metrics, seed_metrics, best_epoch = train_one_trial(
            train_path=train_path,
            test_path=test_path,
            params=trial_params,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seeds=seeds or SEEDS,
            split_seed=split_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            scale_inputs=scale_inputs,
            patience=patience,
            accelerator=accelerator,
            max_epochs=max_epochs,
            grad_clip=grad_clip,
            trial_label=trial_label,
            verbose=verbose,
            per_target=per_target,
            use_autoencoder=use_autoencoder,
            autoencoder_params=autoencoder_params,
        )
        if verbose:
            elapsed = time.perf_counter() - trial_start
            best_epoch_label = best_epoch if best_epoch is not None else "n/a"
            print(
                f"[Grid] {trial_label} done in {elapsed:.1f}s | "
                f"best_epoch={best_epoch_label} "
                f"val_loss={metrics.get('val_loss_mean', float('nan')):.6f} "
                f"val_mse={metrics.get('val_mse_mean', float('nan')):.6f} "
                f"val_mee={metrics.get('val_mee_mean', float('nan')):.6f} "
                f"test_mee={metrics.get('test_mee_mean', float('nan')):.6f}",
                flush=True,
            )
        trials.append(
            {
                "trial": int(idx - 1),
                "value": float(metrics.get("val_loss_mean", float("nan"))),
                "params": dict(trial_params),
                "metrics": dict(metrics),
                "seed_metrics": seed_metrics,
                "best_epoch": best_epoch,
            }
        )

    top_trials = _top_trials(trials, top_k=top_k)
    if not top_trials:
        raise RuntimeError("No trials completed.")

    best_trial = top_trials[0]
    return {
        "best_params": dict(best_trial.get("params", {})),
        "best_metrics": dict(best_trial.get("metrics", {})),
        "best_epoch": best_trial.get("best_epoch"),
        "n_trials": len(grid),
        "top_trials": top_trials,
        "trials": trials,
    }
