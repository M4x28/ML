from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

warnings.filterwarnings("ignore")

from cup_data import CupDataModule
from cup_io import DEFAULT_HEADER, export_artifacts, write_predictions_csv
from cup_metrics import evaluate_regression_metrics
from cup_autoencoder import AutoencoderConfig, apply_autoencoder_to_datamodule
from mlp_curves import CurveTracker, save_mee_curve, save_mse_curve
from mlp_model import CupMLPModel
from mlp_search import SEEDS, BestMetricsTracker, run_grid_search

FINAL_SEEDS = [0]

class VerboseEpochLogger(Callback):
    def __init__(self, every_n: int = 1) -> None:
        """Print per-epoch metrics every N epochs."""
        super().__init__()
        self.every_n = max(int(every_n), 1)

    def on_validation_end(self, trainer, pl_module) -> None:
        """Log train/val metrics at validation end for debugging."""
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n != 0:
            return
        metrics = trainer.callback_metrics

        def _get(name: str) -> float | None:
            """Fetch a scalar metric if available."""
            value = metrics.get(name)
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        train_loss = _get("train_loss")
        train_mse = _get("train_mse")
        train_mee = _get("train_mee")
        val_loss = _get("val_loss")
        val_mse = _get("val_mse")
        val_mee = _get("val_mee")
        if train_loss is None or val_loss is None:
            return

        print(
            "      "
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} train_mse={train_mse:.6f} train_mee={train_mee:.6f} | "
            f"val_loss={val_loss:.6f} val_mse={val_mse:.6f} val_mee={val_mee:.6f}",
            flush=True,
        )


def default_data_paths() -> tuple[Path, Path]:
    """Return default train/test dataset paths."""
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "ML-CUP25-TR.csv"
    test_path = project_root / "data" / "ML-CUP25-TS.csv"
    return train_path, test_path


def now_run_id() -> str:
    """Return a timestamped run identifier."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_hidden_sizes(value: str | None) -> list[int]:
    """Parse a comma-separated list of hidden sizes."""
    if value is None:
        return [64, 64]
    items = [v.strip() for v in value.split(",") if v.strip()]
    out = []
    for item in items:
        try:
            out.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid hidden size: {item}") from exc
    if not out:
        raise ValueError("hidden_sizes cannot be empty")
    return out


def parse_hidden_dims(value: str | None, *, fallback: list[int]) -> list[int]:
    """Parse hidden dims list with a fallback."""
    if value is None:
        return list(fallback)
    items = [v.strip() for v in value.split(",") if v.strip()]
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid hidden dim: {item}") from exc
    return out or list(fallback)


def parse_int_list(value: str | None, *, fallback: list[int]) -> list[int]:
    """Parse a comma-separated list of ints with a fallback."""
    if value is None:
        return list(fallback)
    items = [v.strip() for v in value.split(",") if v.strip()]
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid integer: {item}") from exc
    return out or list(fallback)


def _apply_lr_scale(params: dict[str, object], lr_scale: float) -> dict[str, object]:
    """Apply a multiplicative scale to the learning rate if unset."""
    scaled = dict(params)
    if scaled.get("lr_scale") is not None:
        return scaled
    factor = float(lr_scale)
    if factor != 1.0:
        base_lr = float(scaled.get("lr", 1e-3))
        scaled["lr"] = base_lr * factor
        scaled["lr_scale"] = factor
    return scaled


def _resolve_grad_clip(value: float | None) -> float:
    """Normalize gradient clip value, treating invalid/negative as 0."""
    if value is None:
        return 0.0
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    return val if val > 0 else 0.0


def _autoencoder_device(accelerator: str) -> torch.device:
    """Choose device for autoencoder training."""
    if accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_autoencoder_config(
    args: argparse.Namespace,
    *,
    batch_size: int,
    seed: int,
) -> AutoencoderConfig:
    """Build AutoencoderConfig from CLI args."""
    hidden_dims = parse_hidden_dims(args.ae_hidden_dims, fallback=[64, 32])
    latent_dims = parse_int_list(args.ae_latent_dims, fallback=[args.ae_latent_dim])
    return AutoencoderConfig(
        latent_dim=int(args.ae_latent_dim),
        latent_dims=tuple(latent_dims),
        hidden_dims=tuple(hidden_dims),
        activation=str(args.ae_activation),
        lr=float(args.ae_lr),
        weight_decay=float(args.ae_weight_decay),
        epochs=int(args.ae_epochs),
        patience=int(args.ae_patience),
        batch_size=int(args.ae_batch_size or batch_size),
        seed=int(seed),
    )


def parse_seeds(value: str | None, *, fallback: list[int]) -> list[int]:
    """Parse a comma-separated list of seeds with a fallback."""
    if value is None:
        return list(fallback)
    items = [v.strip() for v in value.split(",") if v.strip()]
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid seed: {item}") from exc
    return out or list(fallback)


def _resolve_batch_size(value: object, train_size: int, fallback: int) -> int:
    """Resolve batch size from numeric or 'full' tokens."""
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


def _top_trials_per_target(trials: list[dict[str, object]], top_k: int) -> dict[int, list[dict[str, object]]]:
    """Extract top trials per target based on validation loss."""
    by_target: dict[int, list[dict[str, object]]] = {0: [], 1: [], 2: [], 3: []}

    def _safe(value: object) -> float:
        """Convert metric value to float, treating NaN as infinity."""
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float("inf")
        if val != val:
            return float("inf")
        return val

    for trial in trials:
        seed_metrics = trial.get("seed_metrics", [])
        if not seed_metrics:
            continue
        per_target = seed_metrics[0].get("per_target", [])
        for entry in per_target:
            target_idx = entry.get("target_idx")
            if target_idx is None:
                continue
            idx = int(target_idx)
            by_target.setdefault(idx, [])
            # Store per-target metrics from search phase.
            by_target[idx].append(
                {
                    "trial": trial.get("trial"),
                    "best_epoch": trial.get("best_epoch"),
                    "params": trial.get("params", {}),
                    "search_metrics": {
                        "train": {
                            "loss": entry.get("train_loss"),
                            "mse": entry.get("train_mse"),
                            "mee": entry.get("train_mee"),
                        },
                        "val": {
                            "loss": entry.get("val_loss"),
                            "mse": entry.get("val_mse"),
                            "mee": entry.get("val_mee"),
                        },
                    },
                }
            )

    for idx, entries in by_target.items():
        entries.sort(key=lambda e: _safe(e["search_metrics"]["val"]["loss"]))
        by_target[idx] = entries[: max(int(top_k), 0)]

    return by_target


def _evaluate_top_targets(
    *,
    trials_by_target: dict[int, list[dict[str, object]]],
    args: argparse.Namespace,
    train_path: Path,
    test_path: Path,
    max_epochs: int,
) -> dict[int, list[dict[str, object]]]:
    """Retrain top per-target trials and evaluate on train/val/test splits."""
    results: dict[int, list[dict[str, object]]] = {}
    seed = SEEDS[0] if SEEDS else 0
    grad_clip_val = _resolve_grad_clip(args.grad_clip)

    for target_idx, entries in trials_by_target.items():
        results[target_idx] = []
        for entry in entries:
            params = _apply_lr_scale(dict(entry.get("params", {})), args.lr_scale)
            seed_everything(seed)

            raw_batch = params.get("batch_size", args.batch_size)
            init_batch = int(raw_batch) if not isinstance(raw_batch, str) else int(args.batch_size)
            data_module = CupDataModule(
                train_path=train_path,
                test_path=test_path,
                batch_size=init_batch,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                split_seed=int(args.split_seed + seed),
                seed=int(seed),
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                scale_inputs=args.scale_inputs,
            )
            data_module.setup("fit")
            train_size = len(data_module.train_dataset) if data_module.train_dataset is not None else init_batch
            resolved_batch = _resolve_batch_size(raw_batch, train_size, init_batch)
            autoencoder_metrics = None
            if args.use_autoencoder:
                # Autoencoder pretraining (optional).
                ae_config = _build_autoencoder_config(
                    args,
                    batch_size=resolved_batch,
                    seed=seed,
                )
                autoencoder_metrics = apply_autoencoder_to_datamodule(
                    data_module,
                    config=ae_config,
                    device=_autoencoder_device(args.accelerator),
                )

            train_loader = data_module.target_dataloader(
                "train",
                target_idx,
                shuffle=True,
                batch_size=resolved_batch,
            )
            train_eval_loader = data_module.target_dataloader(
                "train",
                target_idx,
                shuffle=False,
                batch_size=resolved_batch,
            )
            val_loader = data_module.target_dataloader(
                "val",
                target_idx,
                shuffle=False,
                batch_size=resolved_batch,
            )
            test_loader = data_module.target_dataloader(
                "test",
                target_idx,
                shuffle=False,
                batch_size=resolved_batch,
            )
            if train_loader is None or train_eval_loader is None:
                continue

            optimizer = str(params.get("optimizer", args.optimizer)).lower()
            momentum = params.get("momentum", args.momentum)
            if optimizer == "sgd":
                if momentum is None:
                    momentum = float(args.momentum)
            else:
                momentum = 0.0

            model = CupMLPModel(
                input_dim=int(data_module.input_dim or 0),
                output_dim=1,
                hidden_sizes=params.get("hidden_sizes", [64, 64]),
                activation=str(params.get("activation", args.activation)),
                dropout=float(params.get("dropout", args.dropout)),
                lr=float(params.get("lr", args.lr)),
                weight_decay=float(params.get("weight_decay", args.weight_decay)),
                optimizer=optimizer,
                momentum=float(momentum),
            )

            metrics_cb = BestMetricsTracker(monitor="val_loss", mode="min", track_state=True)
            callbacks: list[Callback] = [metrics_cb]
            if args.patience and args.patience > 0 and val_loader is not None:
                callbacks.append(EarlyStopping(monitor="val_mee", patience=args.patience, mode="min"))

            trainer = Trainer(
                max_epochs=int(params.get("epochs", max_epochs)),
                deterministic=True,
                logger=False,
                enable_checkpointing=False,
                callbacks=callbacks,
                accelerator=args.accelerator,
                precision="16-mixed" if args.accelerator == "gpu" else "32-true",
                enable_progress_bar=False,
                gradient_clip_val=grad_clip_val,
                gradient_clip_algorithm="norm",
            )

            if val_loader is not None:
                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            else:
                trainer.fit(model, train_dataloaders=train_loader)

            if metrics_cb.best_state is not None:
                # Restore best validation checkpoint before evaluation.
                model.load_state_dict(metrics_cb.best_state)

            train_metrics = evaluate_regression_metrics(model, train_eval_loader, l2_reg=0.0)
            val_metrics = evaluate_regression_metrics(model, val_loader, l2_reg=0.0)
            test_metrics = evaluate_regression_metrics(model, test_loader, l2_reg=0.0)

            results[target_idx].append(
                {
                    "trial": entry.get("trial"),
                    "best_epoch": metrics_cb.best_epoch,
                    "params": params,
                    "autoencoder": autoencoder_metrics,
                    "metrics": {
                        "train": train_metrics,
                        "val": val_metrics,
                        "test": test_metrics,
                    },
                    "search_metrics": entry.get("search_metrics"),
                }
            )

    return results


def _make_logger(
    args: argparse.Namespace,
    *,
    seed: int | None = None,
    target_idx: int | None = None,
) -> TensorBoardLogger | None:
    """Create a TensorBoard logger unless disabled."""
    if args.no_logger:
        return None
    name = "mlp_cup"
    if seed is not None:
        name = f"{name}_seed{seed}"
    if target_idx is not None:
        name = f"{name}_t{target_idx}"
    return TensorBoardLogger(save_dir=args.log_dir, name=name)


def _make_callbacks(
    args: argparse.Namespace,
    *,
    val_ratio: float,
    run_id: str | None = None,
    seed: int | None = None,
    target_idx: int | None = None,
) -> tuple[list, ModelCheckpoint | None]:
    """Build callbacks for checkpointing and early stopping."""
    callbacks: list = []
    ckpt_callback = None
    if val_ratio and val_ratio > 0:
        dirpath = Path(args.checkpoint_dir)
        if run_id is not None:
            dirpath = dirpath / run_id
        if seed is not None:
            dirpath = dirpath / f"seed{seed}"
        if target_idx is not None:
            dirpath = dirpath / f"target{target_idx}"
        filename = "mlp-cup-{epoch:02d}-{val_loss:.6f}"
        if target_idx is not None:
            filename = f"mlp-cup-t{target_idx}-" + filename
        # Save the best checkpoint based on validation loss.
        ckpt_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=str(dirpath),
            filename=filename,
        )
        callbacks.append(ckpt_callback)
        if args.patience and args.patience > 0:
            # Stop if validation loss does not improve.
            callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"))
    return callbacks, ckpt_callback


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training/search/prediction."""
    parser = argparse.ArgumentParser(description="MLP regression model for ML-CUP.")
    parser.add_argument("--train-path", type=str, default=None, help="Path to ML-CUP25-TR.csv")
    parser.add_argument("--test-path", type=str, default=None, help="Path to ML-CUP25-TS.csv")
    parser.add_argument("--output", type=str, default=None, help="Output predictions CSV path")
    parser.add_argument("--export-path", type=str, default=None, help="Optional model export path")

    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for train/val/test split.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for model init/shuffle.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--only-target", type=int, default=None, help="Train only one target idx (0..3) in --per-target mode.")

    parser.add_argument(
        "--scale",
        dest="scale_inputs",
        action="store_true",
        help="Enable input standardization.",
    )
    parser.add_argument(
        "--no-scale",
        dest="scale_inputs",
        action="store_false",
        help="Disable input standardization.",
    )
    parser.set_defaults(scale_inputs=True)

    parser.add_argument("--predict-ts", action="store_true", help="Generate predictions on TS.")
    parser.add_argument("--no-header", action="store_true", help="Do not write header lines in output.")
    parser.add_argument(
        "--per-target",
        action="store_true",
        help="Train 4 separate models (one per target) instead of a single multi-output model.",
    )
    parser.add_argument(
        "--per-target-top-k",
        type=int,
        default=2,
        help="Top-k per target to evaluate on test after per-target grid search (0 = skip).",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs per training run.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-scale", type=float, default=0.5, help="Scale learning rate values.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-sizes", type=str, default="64,64")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "leakyrelu", "leaky_relu"],
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm (0 disables).")
    parser.add_argument(
        "--final-seeds",
        type=str,
        default="0",
        help="Comma-separated seeds for final retraining/evaluation.",
    )

    parser.add_argument(
        "--use-autoencoder",
        action="store_true",
        help="Pre-train an autoencoder and use its latent features for the MLP.",
    )
    parser.add_argument("--ae-latent-dim", type=int, default=8)
    parser.add_argument("--ae-latent-dims", type=str, default="4,8,12,16,24,32")
    parser.add_argument("--ae-hidden-dims", type=str, default="64,32")
    parser.add_argument("--ae-activation", type=str, default="relu", choices=["relu", "tanh", "leakyrelu", "leaky_relu"])
    parser.add_argument("--ae-epochs", type=int, default=300)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--ae-weight-decay", type=float, default=0.0)
    parser.add_argument("--ae-patience", type=int, default=30)
    parser.add_argument("--ae-batch-size", type=int, default=None)

    parser.add_argument("--accelerator", type=str, default="gpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--no-logger", action="store_true")
    parser.add_argument("--log-dir", type=str, default="tb_logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    parser.add_argument("--no-search", action="store_true", help="Disable grid search.")
    parser.add_argument("--n-trials", type=int, default=0, help="Grid trials (0 = full grid).")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Unused (grid search runs sequentially).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch metrics.")
    parser.add_argument("--verbose-every", type=int, default=1, help="Epoch interval for verbose logs.")

    parser.add_argument("--results-dir", type=str, default=None, help="Directory for JSON results.")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory for saved models.")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (default: timestamp).")
    return parser.parse_args()


def train_and_predict(args: argparse.Namespace) -> Path | None:
    """Main training flow: grid search (optional), retrain, and TS prediction."""
    if args.accelerator == "gpu":
        # Improve matmul throughput on GPU when available.
        torch.set_float32_matmul_precision("medium")

    default_train, default_test = default_data_paths()
    train_path = Path(args.train_path) if args.train_path else default_train
    test_path = Path(args.test_path) if args.test_path else default_test
    run_id = args.run_id or now_run_id()

    results_dir = Path(args.results_dir) if args.results_dir else MODULE_DIR / "results"
    export_dir = Path(args.export_dir) if args.export_dir else MODULE_DIR / "exports"
    results_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    train_seed = args.seed if args.seed is not None else args.split_seed
    seed_everything(train_seed)
    max_epochs = int(args.epochs)
    grad_clip_val = _resolve_grad_clip(args.grad_clip)

    if args.no_search:
        # Use CLI parameters directly without grid search.
        params = {
            "hidden_sizes": parse_hidden_sizes(args.hidden_sizes),
            "activation": args.activation,
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "optimizer": args.optimizer,
            "momentum": float(args.momentum),
            "batch_size": int(args.batch_size),
            "epochs": int(max_epochs),
        }
        params = _apply_lr_scale(params, args.lr_scale)
        report_metrics = None
        best_epoch = None
        top_trials: list[dict[str, object]] = []
    else:
        autoencoder_params = None
        if args.use_autoencoder:
            # Autoencoder configuration for the search phase.
            autoencoder_params = {
                "latent_dim": int(args.ae_latent_dim),
                "latent_dims": parse_int_list(args.ae_latent_dims, fallback=[args.ae_latent_dim]),
                "hidden_dims": parse_hidden_dims(args.ae_hidden_dims, fallback=[64, 32]),
                "activation": str(args.ae_activation),
                "lr": float(args.ae_lr),
                "weight_decay": float(args.ae_weight_decay),
                "epochs": int(args.ae_epochs),
                "patience": int(args.ae_patience),
            }
            if args.ae_batch_size is not None:
                autoencoder_params["batch_size"] = int(args.ae_batch_size)
        search = run_grid_search(
            train_path=str(train_path),
            test_path=str(test_path),
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seeds=SEEDS,
            split_seed=args.split_seed,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            scale_inputs=args.scale_inputs,
            patience=args.patience,
            accelerator=args.accelerator,
            max_epochs=int(max_epochs),
            n_trials=args.n_trials,
            top_k=5,
            verbose=True,
            per_target=args.per_target,
            use_autoencoder=args.use_autoencoder,
            autoencoder_params=autoencoder_params,
            lr_scale=args.lr_scale,
            grad_clip=args.grad_clip,
        )
        params = search["best_params"]
        report_metrics = search["best_metrics"]
        best_epoch = search.get("best_epoch")
        top_trials = search.get("top_trials", [])
        params = _apply_lr_scale(params, args.lr_scale)

        if report_metrics:
            print(
                "Best metrics (mean over seeds, from model selection):",
                f"train_mee={report_metrics.get('train_mee_mean', float('nan')):.6f}",
                f"val_mee={report_metrics.get('val_mee_mean', float('nan')):.6f}",
                f"test_mee={report_metrics.get('test_mee_mean', float('nan')):.6f}",
            )
        if top_trials:
            print("Top trials (by val_loss, val_mse, val_mee):")
            for idx, entry in enumerate(top_trials, start=1):
                metrics = entry.get("metrics", {})
                params_row = entry.get("params", {})
                print(
                    f"  {idx:02d}) trial={entry.get('trial')} "
                    f"val_loss={metrics.get('val_loss_mean', float('nan')):.6f} "
                    f"val_mse={metrics.get('val_mse_mean', float('nan')):.6f} "
                    f"val_mee={metrics.get('val_mee_mean', float('nan')):.6f} "
                    f"params={params_row}"
                )

        summary_path = results_dir / f"search_summary_{run_id}.json"
        per_target_top_models = None
        if args.per_target and args.per_target_top_k > 0:
            top_trials_by_target = _top_trials_per_target(search.get("trials", []), args.per_target_top_k)
            per_target_top_models = _evaluate_top_targets(
                trials_by_target=top_trials_by_target,
                args=args,
                train_path=train_path,
                test_path=test_path,
                max_epochs=max_epochs,
            )

        summary_payload = {
            "run_id": run_id,
            "train_path": str(train_path),
            "test_path": str(test_path),
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seeds": SEEDS,
            "split_seed": args.split_seed,
            "scale_inputs": args.scale_inputs,
            "search_method": "grid",
            "autoencoder_enabled": bool(args.use_autoencoder),
            "autoencoder": autoencoder_params,
            "per_target": args.per_target,
            "per_target_top_k": args.per_target_top_k,
            "per_target_top_models": per_target_top_models,
            "n_trials": search.get("n_trials"),
            "best_params": params,
            "best_metrics": report_metrics,
            "best_epoch": best_epoch,
            "top_trials": top_trials,
            "trials": search.get("trials", []),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    hidden_sizes = params.get("hidden_sizes", [64, 64])
    if isinstance(hidden_sizes, str):
        hidden_sizes = parse_hidden_sizes(hidden_sizes)
    else:
        hidden_sizes = [int(x) for x in hidden_sizes]

    max_epochs = int(best_epoch or params.get("epochs", args.epochs))
    final_seeds = parse_seeds(args.final_seeds, fallback=FINAL_SEEDS)

    def _aggregate(values: list[float]) -> tuple[float, float]:
        """Return (mean, std) ignoring NaNs for a metric list."""
        arr = np.asarray(values, dtype=float)
        if np.isnan(arr).all():
            return float("nan"), float("nan")
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    seed_summaries: list[dict[str, object]] = []
    curve_paths_per_seed: dict[int, dict[str, str]] = {}
    per_target_curve_paths: dict[int, dict[int, dict[str, str]]] = {}
    best_seed = None
    best_val_loss = float("inf")
    best_model = None
    best_models_by_target: list[CupMLPModel] | None = None
    best_data_module = None
    best_trainer = None
    best_ckpt_path = None
    best_checkpoint_path = None
    best_export_path = None
    best_target_summaries: list[dict[str, object]] | None = None

    if args.per_target:
        # Train one model per target and aggregate results by seed.
        for seed in final_seeds:
            seed_everything(seed)

            raw_batch = params.get("batch_size", args.batch_size)
            init_batch = int(raw_batch) if not isinstance(raw_batch, str) else int(args.batch_size)
            data_module = CupDataModule(
                train_path=train_path,
                test_path=test_path,
                batch_size=init_batch,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                split_seed=int(args.split_seed + seed),
                seed=int(seed),
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                scale_inputs=args.scale_inputs,
            )
            data_module.setup("fit")
            train_size = len(data_module.train_dataset) if data_module.train_dataset is not None else init_batch
            resolved_batch = _resolve_batch_size(raw_batch, train_size, init_batch)
            data_module.batch_size = resolved_batch
            data_module.predict_batch_size = resolved_batch
            autoencoder_metrics = None
            if args.use_autoencoder:
                # Pre-train autoencoder and replace inputs with latent codes.
                ae_config = _build_autoencoder_config(
                    args,
                    batch_size=resolved_batch,
                    seed=seed,
                )
                autoencoder_metrics = apply_autoencoder_to_datamodule(
                    data_module,
                    config=ae_config,
                    device=_autoencoder_device(args.accelerator),
                )
            assert data_module.input_dim is not None
            assert data_module.target_dim is not None

            optimizer = str(params.get("optimizer", args.optimizer)).lower()
            momentum = params.get("momentum", args.momentum)
            if optimizer == "sgd":
                if momentum is None:
                    momentum = float(args.momentum)
            else:
                momentum = 0.0

            target_summaries: list[dict[str, object]] = []
            target_curve_paths: dict[int, dict[str, str]] = {}
            seed_models: list[CupMLPModel] = []
            trained_epochs_list: list[int] = []

            targets = list(range(int(data_module.target_dim)))
            if args.only_target is not None:
                targets = [int(args.only_target)]

            for target_idx in targets:
                # Build single-target dataloaders for current output index.
                train_loader = data_module.target_dataloader(
                    "train",
                    target_idx,
                    shuffle=True,
                    batch_size=resolved_batch,
                )
                train_eval_loader = data_module.target_dataloader(
                    "train",
                    target_idx,
                    shuffle=False,
                    batch_size=resolved_batch,
                )
                val_loader = data_module.target_dataloader(
                    "val",
                    target_idx,
                    shuffle=False,
                    batch_size=resolved_batch,
                )
                test_loader = data_module.target_dataloader(
                    "test",
                    target_idx,
                    shuffle=False,
                    batch_size=resolved_batch,
                )

                model = CupMLPModel(
                    input_dim=data_module.input_dim,
                    output_dim=1,
                    hidden_sizes=hidden_sizes,
                    activation=str(params.get("activation", args.activation)),
                    dropout=float(params.get("dropout", args.dropout)),
                    lr=float(params.get("lr", args.lr)),
                    weight_decay=float(params.get("weight_decay", args.weight_decay)),
                    optimizer=optimizer,
                    momentum=float(momentum),
                )

                callbacks, ckpt_callback = _make_callbacks(
                    args,
                    val_ratio=args.val_ratio,
                    run_id=run_id,
                    seed=seed,
                    target_idx=target_idx,
                )
                curve_tracker = CurveTracker(
                    train_loader=train_eval_loader,
                    test_loader=test_loader,
                    l2_reg=0.0,
                )
                callbacks.append(curve_tracker)
                if args.verbose:
                    callbacks.append(VerboseEpochLogger(every_n=args.verbose_every))

                trainer = Trainer(
                    max_epochs=max_epochs,
                    deterministic=True,
                    logger=_make_logger(args, seed=seed, target_idx=target_idx),
                    callbacks=callbacks,
                    enable_checkpointing=bool(ckpt_callback),
                    accelerator=args.accelerator,
                    precision="16-mixed" if args.accelerator == "gpu" else "32-true",
                    gradient_clip_val=grad_clip_val,
                    gradient_clip_algorithm="norm",
                )

                if val_loader is not None:
                    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                else:
                    trainer.fit(model, train_dataloaders=train_loader)

                trained_epochs = trainer.current_epoch + 1
                trained_epochs_list.append(trained_epochs)

                seed_best_model = model
                seed_best_ckpt = None
                if ckpt_callback is not None and ckpt_callback.best_model_path:
                    # Reload best checkpoint for evaluation/export.
                    seed_best_ckpt = ckpt_callback.best_model_path
                    seed_best_model = CupMLPModel.load_from_checkpoint(seed_best_ckpt)

                train_metrics = evaluate_regression_metrics(
                    seed_best_model,
                    train_eval_loader,
                    l2_reg=0.0,
                )
                val_metrics = evaluate_regression_metrics(
                    seed_best_model,
                    val_loader,
                    l2_reg=0.0,
                )
                test_metrics = evaluate_regression_metrics(
                    seed_best_model,
                    test_loader,
                    l2_reg=0.0,
                )

                curves_dir = MODULE_DIR / "curves" / run_id / f"seed{seed}" / f"target{target_idx}"
                mse_curve_path = save_mse_curve(
                    history=curve_tracker.history,
                    output_dir=curves_dir,
                    run_name=f"mlp_cup_seed{seed}_t{target_idx}",
                    title=f"MLP CUP - MSE Curve (seed {seed}, target {target_idx})",
                )
                mee_curve_path = save_mee_curve(
                    history=curve_tracker.history,
                    output_dir=curves_dir,
                    run_name=f"mlp_cup_seed{seed}_t{target_idx}",
                    title=f"MLP CUP - MEE Curve (seed {seed}, target {target_idx})",
                )
                target_curve_paths[int(target_idx)] = {
                    "mse": str(mse_curve_path),
                    "mee": str(mee_curve_path),
                }

                checkpoint_path = export_dir / f"mlp_cup_{run_id}_seed{seed}_target{target_idx}.ckpt"
                trainer.save_checkpoint(checkpoint_path, weights_only=False)

                export_path = export_dir / f"mlp_cup_{run_id}_seed{seed}_target{target_idx}.pt"
                export_params = dict(params)
                export_params["target_idx"] = int(target_idx)
                export_params["per_target"] = True
                # Save model weights and preprocessing artifacts for reuse.
                export_artifacts(
                    export_path,
                    model=seed_best_model,
                    input_dim=data_module.input_dim,
                    output_dim=1,
                    scaler=data_module.scaler,
                    params=export_params,
                )

                target_summaries.append(
                    {
                        "target_idx": int(target_idx),
                        "trained_epochs": int(trained_epochs),
                        "best_checkpoint": str(seed_best_ckpt) if seed_best_ckpt else None,
                        "checkpoint": str(checkpoint_path),
                        "exported_model": str(export_path),
                        "metrics": {
                            "train": train_metrics,
                            "val": val_metrics,
                            "test": test_metrics,
                        },
                        "curve_paths": {
                            "mse": str(mse_curve_path),
                            "mee": str(mee_curve_path),
                        },
                    }
                )
                seed_models.append(seed_best_model)

            per_target_curve_paths[int(seed)] = target_curve_paths

            def _aggregate_targets(split: str) -> dict[str, float]:
                """Aggregate metrics across targets for a given split."""
                loss_vals = [t["metrics"][split]["loss"] for t in target_summaries]
                mse_vals = [t["metrics"][split]["mse"] for t in target_summaries]
                mee_vals = [t["metrics"][split]["mee"] for t in target_summaries]
                loss_mean, _ = _aggregate(loss_vals)
                mse_mean, _ = _aggregate(mse_vals)
                mee_mean, _ = _aggregate(mee_vals)
                return {"loss": loss_mean, "mse": mse_mean, "mee": mee_mean}

            train_metrics = _aggregate_targets("train")
            val_metrics = _aggregate_targets("val")
            test_metrics = _aggregate_targets("test")

            trained_epochs = (
                int(round(sum(trained_epochs_list) / len(trained_epochs_list)))
                if trained_epochs_list
                else 0
            )

            seed_summaries.append(
                {
                    "seed": int(seed),
                    "split_seed": int(args.split_seed + seed),
                    "batch_size_used": int(resolved_batch),
                    "trained_epochs": int(trained_epochs),
                    "autoencoder": autoencoder_metrics,
                    "metrics": {
                        "train": train_metrics,
                        "val": val_metrics,
                        "test": test_metrics,
                    },
                    "targets": target_summaries,
                }
            )

            val_loss = float(val_metrics.get("loss", float("nan")))
            if val_loss != val_loss:
                val_loss = float("inf")
            # Keep the best seed based on validation loss.
            if best_seed is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_seed = int(seed)
                best_models_by_target = seed_models
                best_data_module = data_module
                best_target_summaries = target_summaries
    else:
        for seed in final_seeds:
            seed_everything(seed)

            raw_batch = params.get("batch_size", args.batch_size)
            init_batch = int(raw_batch) if not isinstance(raw_batch, str) else int(args.batch_size)
            data_module = CupDataModule(
                train_path=train_path,
                test_path=test_path,
                batch_size=init_batch,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                split_seed=int(args.split_seed + seed),
                seed=int(seed),
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                scale_inputs=args.scale_inputs,
            )
            data_module.setup("fit")
            train_size = len(data_module.train_dataset) if data_module.train_dataset is not None else init_batch
            resolved_batch = _resolve_batch_size(raw_batch, train_size, init_batch)
            data_module.batch_size = resolved_batch
            data_module.predict_batch_size = resolved_batch
            autoencoder_metrics = None
            if args.use_autoencoder:
                # Pre-train autoencoder and replace inputs with latent codes.
                ae_config = _build_autoencoder_config(
                    args,
                    batch_size=resolved_batch,
                    seed=seed,
                )
                autoencoder_metrics = apply_autoencoder_to_datamodule(
                    data_module,
                    config=ae_config,
                    device=_autoencoder_device(args.accelerator),
                )
            assert data_module.input_dim is not None
            assert data_module.target_dim is not None

            optimizer = str(params.get("optimizer", args.optimizer)).lower()
            momentum = params.get("momentum", args.momentum)
            if optimizer == "sgd":
                if momentum is None:
                    momentum = float(args.momentum)
            else:
                momentum = 0.0

            model = CupMLPModel(
                input_dim=data_module.input_dim,
                output_dim=data_module.target_dim,
                hidden_sizes=hidden_sizes,
                activation=str(params.get("activation", args.activation)),
                dropout=float(params.get("dropout", args.dropout)),
                lr=float(params.get("lr", args.lr)),
                weight_decay=float(params.get("weight_decay", args.weight_decay)),
                optimizer=optimizer,
                momentum=float(momentum),
            )

            callbacks, ckpt_callback = _make_callbacks(
                args,
                val_ratio=args.val_ratio,
                run_id=run_id,
                seed=seed,
            )
            curve_tracker = CurveTracker(
                train_loader=data_module.train_eval_dataloader(),
                test_loader=data_module.test_dataloader(),
                l2_reg=0.0,
            )
            callbacks.append(curve_tracker)
            if args.verbose:
                callbacks.append(VerboseEpochLogger(every_n=args.verbose_every))

            trainer = Trainer(
                max_epochs=max_epochs,
                deterministic=True,
                logger=_make_logger(args, seed=seed),
                callbacks=callbacks,
                enable_checkpointing=bool(ckpt_callback),
                accelerator=args.accelerator,
                precision="16-mixed" if args.accelerator == "gpu" else "32-true",
                gradient_clip_val=grad_clip_val,
                gradient_clip_algorithm="norm",
            )

            trainer.fit(model, datamodule=data_module)
            trained_epochs = trainer.current_epoch + 1

            seed_best_model = model
            seed_best_ckpt = None
            if ckpt_callback is not None and ckpt_callback.best_model_path:
                # Reload best checkpoint for metrics and export.
                seed_best_ckpt = ckpt_callback.best_model_path
                seed_best_model = CupMLPModel.load_from_checkpoint(seed_best_ckpt)

            train_metrics = evaluate_regression_metrics(
                seed_best_model,
                data_module.train_eval_dataloader(),
                l2_reg=0.0,
            )
            val_metrics = evaluate_regression_metrics(
                seed_best_model,
                data_module.val_dataloader(),
                l2_reg=0.0,
            )
            test_metrics = evaluate_regression_metrics(
                seed_best_model,
                data_module.test_dataloader(),
                l2_reg=0.0,
            )

            curves_dir = MODULE_DIR / "curves" / run_id / f"seed{seed}"
            mse_curve_path = save_mse_curve(
                history=curve_tracker.history,
                output_dir=curves_dir,
                run_name=f"mlp_cup_seed{seed}",
                title=f"MLP CUP - MSE Curve (seed {seed})",
            )
            mee_curve_path = save_mee_curve(
                history=curve_tracker.history,
                output_dir=curves_dir,
                run_name=f"mlp_cup_seed{seed}",
                title=f"MLP CUP - MEE Curve (seed {seed})",
            )
            curve_paths_per_seed[seed] = {"mse": str(mse_curve_path), "mee": str(mee_curve_path)}

            checkpoint_path = export_dir / f"mlp_cup_{run_id}_seed{seed}.ckpt"
            trainer.save_checkpoint(checkpoint_path, weights_only=False)

            export_path = export_dir / f"mlp_cup_{run_id}_seed{seed}.pt"
            # Save model weights and preprocessing artifacts for reuse.
            export_artifacts(
                export_path,
                model=seed_best_model,
                input_dim=data_module.input_dim,
                output_dim=data_module.target_dim,
                scaler=data_module.scaler,
                params=params,
            )

            seed_summaries.append(
                {
                    "seed": int(seed),
                    "split_seed": int(args.split_seed + seed),
                    "batch_size_used": int(resolved_batch),
                    "trained_epochs": int(trained_epochs),
                    "autoencoder": autoencoder_metrics,
                    "best_checkpoint": str(seed_best_ckpt) if seed_best_ckpt else None,
                    "checkpoint": str(checkpoint_path),
                    "exported_model": str(export_path),
                    "metrics": {
                        "train": train_metrics,
                        "val": val_metrics,
                        "test": test_metrics,
                    },
                }
            )

            val_loss = float(val_metrics.get("loss", float("nan")))
            if val_loss != val_loss:
                val_loss = float("inf")
            # Keep the best seed based on validation loss.
            if best_seed is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_seed = int(seed)
                best_model = seed_best_model
                best_data_module = data_module
                best_trainer = trainer
                best_ckpt_path = seed_best_ckpt
                best_checkpoint_path = checkpoint_path
                best_export_path = export_path

    def _aggregate_split(split: str) -> tuple[dict[str, float], dict[str, float]]:
        """Aggregate mean/std metrics across seeds for a split."""
        loss_vals = [s["metrics"][split]["loss"] for s in seed_summaries]
        mse_vals = [s["metrics"][split]["mse"] for s in seed_summaries]
        mee_vals = [s["metrics"][split]["mee"] for s in seed_summaries]
        loss_mean, loss_std = _aggregate(loss_vals)
        mse_mean, mse_std = _aggregate(mse_vals)
        mee_mean, mee_std = _aggregate(mee_vals)
        return (
            {"loss": loss_mean, "mse": mse_mean, "mee": mee_mean},
            {"loss": loss_std, "mse": mse_std, "mee": mee_std},
        )

    train_mean, train_std = _aggregate_split("train")
    val_mean, val_std = _aggregate_split("val")
    test_mean, test_std = _aggregate_split("test")

    out_path = None
    if args.predict_ts:
        if args.per_target:
            if best_models_by_target is None or best_data_module is None:
                raise RuntimeError("Best models not available for TS prediction.")
            device = "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
            preds_by_target: list[np.ndarray] = []
            for model in best_models_by_target:
                model.eval()
                model.to(device)
                outputs: list[torch.Tensor] = []
                for batch in best_data_module.predict_dataloader():
                    x = batch[0].to(device)
                    with torch.no_grad():
                        y_hat = model(x)
                    outputs.append(y_hat.detach().cpu())
                # Each model outputs a single target column.
                pred = torch.cat(outputs, dim=0).numpy().reshape(-1)
                preds_by_target.append(pred)
            preds = np.stack(preds_by_target, axis=1)
            ids = best_data_module.test_ids
            if ids is None:
                raise RuntimeError("Missing test ids in datamodule.")
            output_path = Path(args.output) if args.output else MODULE_DIR / "predictions" / "mlp_cup.csv"
            header = [] if args.no_header else DEFAULT_HEADER
            out_path = write_predictions_csv(output_path, ids, preds, header_lines=header)
        else:
            if best_trainer is None or best_model is None or best_data_module is None:
                raise RuntimeError("Best model not available for TS prediction.")
            # Multi-output prediction in one pass.
            pred_batches = best_trainer.predict(best_model, datamodule=best_data_module)
            preds = torch.cat([p.detach().cpu() for p in pred_batches], dim=0).numpy()
            ids = best_data_module.test_ids
            if ids is None:
                raise RuntimeError("Missing test ids in datamodule.")
            output_path = Path(args.output) if args.output else MODULE_DIR / "predictions" / "mlp_cup.csv"
            header = [] if args.no_header else DEFAULT_HEADER
            out_path = write_predictions_csv(output_path, ids, preds, header_lines=header)

    final_summary_path = results_dir / f"final_summary_{run_id}.json"
    final_payload = {
        "run_id": run_id,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "predict_ts": args.predict_ts,
        "params": params,
        "lr_scale": float(args.lr_scale),
        "grad_clip": float(args.grad_clip),
        "per_target": args.per_target,
        "autoencoder_enabled": bool(args.use_autoencoder),
        "autoencoder": (
            {
                "latent_dim": int(args.ae_latent_dim),
                "latent_dims": parse_int_list(args.ae_latent_dims, fallback=[args.ae_latent_dim]),
                "hidden_dims": parse_hidden_dims(args.ae_hidden_dims, fallback=[64, 32]),
                "activation": str(args.ae_activation),
                "lr": float(args.ae_lr),
                "weight_decay": float(args.ae_weight_decay),
                "epochs": int(args.ae_epochs),
                "patience": int(args.ae_patience),
                "batch_size": int(args.ae_batch_size) if args.ae_batch_size is not None else None,
            }
            if args.use_autoencoder
            else None
        ),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "seed": int(train_seed),
        "final_seeds": final_seeds,
        "scale_inputs": args.scale_inputs,
        "batch_size_used": None if not seed_summaries else int(seed_summaries[0]["batch_size_used"]),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "max_epochs": int(max_epochs),
        "report_metrics": report_metrics,
        "report_note": (
            "Report metrics (MEE train/val/test) are taken from model selection. "
            "Final metrics below are computed after training the selected model."
        ),
        "final_metrics": {
            "train": train_mean,
            "val": val_mean,
            "test": test_mean,
        },
        "final_metrics_std": {
            "train": train_std,
            "val": val_std,
            "test": test_std,
        },
        "final_metrics_per_seed": seed_summaries,
        "best_seed": best_seed,
        "curve_paths": (
            None
            if args.per_target
            else {
                "mse": curve_paths_per_seed.get(best_seed or -1, {}).get("mse"),
                "mee": curve_paths_per_seed.get(best_seed or -1, {}).get("mee"),
            }
        ),
        "curve_paths_per_seed": {} if args.per_target else curve_paths_per_seed,
        "per_target_curve_paths": per_target_curve_paths if args.per_target else None,
        "per_target_best_targets": best_target_summaries if args.per_target else None,
        "checkpoint": None if args.per_target else str(best_checkpoint_path) if best_checkpoint_path else None,
        "best_checkpoint": None if args.per_target else str(best_ckpt_path) if best_ckpt_path else None,
        "exported_model": None if args.per_target else str(best_export_path) if best_export_path else None,
        "predictions_path": str(out_path) if out_path is not None else None,
    }
    final_summary_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    return out_path


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    out_path = train_and_predict(args)
    if out_path is not None:
        print(f"Predictions saved to: {out_path}")
    else:
        print("Training complete. Use --predict-ts to generate the blind test predictions.")


if __name__ == "__main__":
    main()
