from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for name in ("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning"):
    logging.getLogger(name).setLevel(logging.ERROR)
logging.disable(logging.WARNING)

from cup_data import CupDataModule
from cup_io import DEFAULT_HEADER, export_artifacts, write_predictions_csv
from cup_metrics import evaluate_regression_metrics
from cup_model import CupLinearModel
from cup_search import SEEDS, run_optuna_search
from cup_callbacks import CurveTracker, average_histories, save_curve_plots


def default_data_paths() -> tuple[Path, Path]:
    """Return default train/test paths under the project data folder."""
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / "data" / "ML-CUP25-TR.csv"
    test_path = project_root / "data" / "ML-CUP25-TS.csv"
    return train_path, test_path


def now_run_id() -> str:
    """Generate a timestamped run identifier."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training, search, and prediction."""
    parser = argparse.ArgumentParser(description="Linear model (Lightning) for ML-CUP regression.")
    parser.add_argument("--train-path", type=str, default=None, help="Path to ML-CUP25-TR.csv")
    parser.add_argument("--test-path", type=str, default=None, help="Path to ML-CUP25-TS.csv")
    parser.add_argument("--output", type=str, default=None, help="Output predictions CSV path")
    parser.add_argument("--export-path", type=str, default=None, help="Optional model export path")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument(
        "--feature-map",
        type=str,
        default="identity",
        choices=["identity", "poly2"],
        help="Feature map for LBE (identity or poly2).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for train/val/test split.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
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
    parser.set_defaults(scale_inputs=False)
    parser.add_argument(
        "--predict-ts",
        action="store_true",
        help="Generate predictions on the blind test set (TS).",
    )

    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--no-logger", action="store_true")
    parser.add_argument("--log-dir", type=str, default="tb_logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-header", action="store_true", help="Do not write header lines in output.")
    parser.add_argument("--no-search", action="store_true", help="Disable Optuna adaptive grid search.")
    parser.add_argument("--n-trials", type=int, default=0, help="Optuna trials (0 = full grid).")
    parser.add_argument("--n-jobs", type=int, default=1, help="Optuna parallel jobs.")
    parser.add_argument("--optuna-seed", type=int, default=0, help="Seed for Optuna sampler.")
    parser.add_argument("--tb-search", action="store_true", help="Log Optuna trials to TensorBoard.")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory for JSON results.")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory for saved models.")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (default: timestamp).")
    parser.add_argument(
        "--per-target",
        action="store_true",
        help="Run Optuna search and retraining separately for each target (0-3).",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Run Optuna search and retraining for a single target index (0-3).",
    )
    return parser.parse_args()


def _make_logger(
    args: argparse.Namespace, *, run_id: str, seed: int | None = None
) -> TensorBoardLogger | None:
    """Create a TensorBoard logger unless disabled."""
    if args.no_logger:
        return None
    version = run_id if seed is None else f"{run_id}_seed{seed}"
    return TensorBoardLogger(save_dir=args.log_dir, name="linear_cup", version=version)


def _resolve_search_settings(args: argparse.Namespace) -> tuple[str, int, int]:
    """Adjust search settings for stability when using multiple jobs."""
    accelerator = args.accelerator
    num_workers = args.num_workers
    n_jobs = args.n_jobs
    if args.n_jobs > 1:
        # Guardrails: Optuna parallelism + GPU/DataLoader workers can be unstable.
        use_gpu = accelerator == "gpu" or (accelerator == "auto" and torch.cuda.is_available())
        if use_gpu:
            print("WARNING: n_jobs>1 with GPU can crash; forcing n_jobs=1 to keep GPU.")
            n_jobs = 1
        if num_workers > 0:
            print("WARNING: n_jobs>1 with DataLoader workers can be unstable; using num_workers=0.")
            num_workers = 0
    return accelerator, num_workers, n_jobs


def _aggregate_seed_metrics(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute mean/std metrics across seeds."""
    keys = [
        "train_loss",
        "train_mse",
        "train_mee",
        "test_loss",
        "test_mse",
        "test_mee",
    ]
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in keys:
        values = np.asarray([m.get(key, float("nan")) for m in per_seed], dtype=float)
        mean[key] = float(np.nanmean(values))
        std[key] = float(np.nanstd(values))
    return {"mean": mean, "std": std}


def _select_best_seed(per_seed: list[dict[str, float]]) -> int:
    """Select the best seed index using test metrics when available."""
    def _best_index(key: str) -> int | None:
        """Return index of the lowest metric value for the given key."""
        values = [(idx, metrics.get(key, float("nan"))) for idx, metrics in enumerate(per_seed)]
        values = [(idx, val) for idx, val in values if val == val]
        if not values:
            return None
        return min(values, key=lambda item: item[1])[0]

    # Prefer test metrics, then fall back to training metrics.
    for metric_key in ("test_mee", "test_mse", "train_mee", "train_mse"):
        idx = _best_index(metric_key)
        if idx is not None:
            return idx
    return 0


def _resolve_target_indices(args: argparse.Namespace) -> list[int | None]:
    """Resolve per-target selection from CLI flags."""
    if args.per_target and args.target_index is not None:
        raise ValueError("Use either --per-target or --target-index, not both.")
    if args.per_target:
        return [0, 1, 2, 3]
    if args.target_index is not None:
        if args.target_index < 0 or args.target_index > 3:
            raise ValueError("target-index must be in [0, 3].")
        return [args.target_index]
    return [None]


def _run_single_target(
    args: argparse.Namespace,
    *,
    train_path: Path,
    test_path: Path,
    run_id: str,
    results_dir: Path,
    export_dir: Path,
    target_index: int | None,
    write_predictions: bool,
) -> dict[str, object]:
    """Run search (optional), retrain, and prediction for one target mode."""
    label = f"Target {target_index + 1}" if target_index is not None else "All targets"
    prefix = f"[{label}] "

    if args.test_ratio <= 0:
        print(f"{prefix}WARNING: test_ratio=0, test curves and metrics will be empty.")

    best_epoch = None
    if args.no_search:
        # Use provided hyperparameters directly.
        params = {
            "lr": args.lr,
            "l2": args.l2,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "feature_map": args.feature_map,
        }
        best_metrics = None
        top_trials = []
        all_trials = []
    else:
        tb_search_dir = None
        if args.tb_search:
            tb_search_dir = str(Path(args.log_dir) / "optuna")
        search_accelerator, search_num_workers, search_n_jobs = _resolve_search_settings(args)
        search = run_optuna_search(
            train_path=str(train_path),
            test_path=None,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seeds=SEEDS,
            num_workers=search_num_workers,
            pin_memory=args.pin_memory,
            scale_inputs=args.scale_inputs,
            split_seed=args.split_seed,
            patience=args.patience,
            accelerator=search_accelerator,
            n_trials=args.n_trials,
            n_jobs=search_n_jobs,
            optuna_seed=args.optuna_seed,
            top_k=5,
            tb_log_dir=tb_search_dir,
            target_index=target_index,
        )
        params = search["best_params"]
        best_metrics = search["best_metrics"]
        best_epoch = search.get("best_epoch")
        top_trials = search.get("top_trials", [])
        all_trials = search.get("trials", [])

        if best_metrics:
            print(
                f"{prefix}Best metrics (mean over seeds, from model selection):",
                f"train_mee={best_metrics.get('train_mee_mean', float('nan')):.6f}",
                f"val_mee={best_metrics.get('val_mee_mean', float('nan')):.6f}",
                f"test_mee={best_metrics.get('test_mee_mean', float('nan')):.6f}",
            )
        if top_trials:
            print(f"{prefix}Top trials (by val_mee, val_mse, val_loss):")
            for idx, entry in enumerate(top_trials, start=1):
                metrics = entry.get("metrics", {})
                trial_params = entry.get("params", {})
                trial_epoch = entry.get("best_epoch")
                print(
                    f"  {idx:02d}) trial={entry.get('trial')} epoch={trial_epoch} "
                    f"val_mee={metrics.get('val_mee_mean', float('nan')):.6f} "
                    f"val_mse={metrics.get('val_mse_mean', float('nan')):.6f} "
                    f"val_loss={metrics.get('val_loss_mean', float('nan')):.6f} "
                    f"params={trial_params}"
                )

        summary_path = results_dir / f"search_summary_{run_id}.json"
        summary_payload = {
            "run_id": run_id,
            "target_index": target_index,
            "train_path": str(train_path),
            "test_path": str(test_path),
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seeds": SEEDS,
            "split_seed": args.split_seed,
            "scale_inputs": args.scale_inputs,
            "n_trials": search.get("n_trials"),
            "best_params": params,
            "best_metrics": best_metrics,
            "best_epoch": best_epoch,
            "top_trials": top_trials,
            "trials": all_trials,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    train_epochs = int(best_epoch or params["epochs"]) if not args.no_search else int(params["epochs"])
    print(
        f"{prefix}Final training: epochs={train_epochs} batch_size={params['batch_size']} "
        f"lr={params['lr']} l2={params['l2']} feature_map={params.get('feature_map', 'identity')} "
        f"seeds={SEEDS}"
    )
    per_seed_metrics: list[dict[str, float]] = []
    histories: list[dict[str, list[float]]] = []
    export_paths: list[str] = []
    checkpoint_paths: list[str] = []
    pred_list: list[np.ndarray] = []
    test_ids = None

    for seed in SEEDS:
        seed_everything(seed)
        print(f"{prefix}[SEED {seed}] training...")

        # Train on the full training set; only test split is used for metrics.
        data_module = CupDataModule(
            train_path=train_path,
            test_path=test_path if args.predict_ts else None,
            batch_size=int(params["batch_size"]),
            val_ratio=0.0,
            test_ratio=args.test_ratio,
            seed=seed,
            split_seed=args.split_seed,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            scale_inputs=args.scale_inputs,
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

        curve_cb = CurveTracker()
        trainer = Trainer(
            max_epochs=train_epochs,
            deterministic=True,
            logger=_make_logger(args, run_id=run_id, seed=seed),
            callbacks=[curve_cb],
            enable_checkpointing=False,
            accelerator=args.accelerator,
            limit_val_batches=0,
            enable_progress_bar=False,
        )
        trainer.fit(model, datamodule=data_module)

        # Evaluate on train/test splits to report final metrics.
        train_metrics = evaluate_regression_metrics(
            model,
            data_module.train_dataloader(),
            l2_reg=float(params["l2"]),
        )
        test_metrics = evaluate_regression_metrics(
            model,
            data_module.test_dataloader(),
            l2_reg=float(params["l2"]),
        )
        print(
            f"{prefix}[SEED {seed}] train_mse={train_metrics['mse']:.6f} "
            f"train_mee={train_metrics['mee']:.6f} "
            f"test_mse={test_metrics['mse']:.6f} "
            f"test_mee={test_metrics['mee']:.6f}"
        )

        per_seed_metrics.append(
            {
                "seed": seed,
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_mee": train_metrics["mee"],
                "test_loss": test_metrics["loss"],
                "test_mse": test_metrics["mse"],
                "test_mee": test_metrics["mee"],
            }
        )
        histories.append(curve_cb.history)

        # Save both Lightning checkpoint and lightweight artifact bundle.
        ckpt_path = export_dir / f"linear_cup_{run_id}_seed{seed}.ckpt"
        trainer.save_checkpoint(ckpt_path)
        checkpoint_paths.append(str(ckpt_path))

        export_path = export_dir / f"linear_cup_{run_id}_seed{seed}.pt"
        export_artifacts(
            export_path,
            model=model,
            input_dim=data_module.input_dim,
            output_dim=data_module.target_dim,
            scaler=data_module.scaler,
            feature_map=data_module.feature_map,
            feature_transformer=data_module.feature_transformer,
        )
        export_paths.append(str(export_path))

        if args.predict_ts:
            # Store predictions to later average across seeds.
            pred_batches = trainer.predict(model, datamodule=data_module)
            preds = torch.cat([p.detach().cpu() for p in pred_batches], dim=0).numpy()
            pred_list.append(preds)
            if test_ids is None:
                test_ids = data_module.test_ids

    curves_dir = MODULE_DIR / "curves" / run_id
    mean_curves = average_histories(histories)
    curve_paths = save_curve_plots(
        output_dir=curves_dir,
        run_name="linear_cup_mean",
        curves=mean_curves,
        title_prefix=f"Linear CUP - {label}",
    )

    metrics_summary = _aggregate_seed_metrics(per_seed_metrics)
    print(
        f"{prefix}Mean metrics:",
        f"train_mse={metrics_summary['mean']['train_mse']:.6f}",
        f"train_mee={metrics_summary['mean']['train_mee']:.6f}",
        f"test_mse={metrics_summary['mean']['test_mse']:.6f}",
        f"test_mee={metrics_summary['mean']['test_mee']:.6f}",
    )
    best_idx = _select_best_seed(per_seed_metrics)
    best_seed = per_seed_metrics[best_idx]["seed"]
    best_model_path = export_paths[best_idx]
    best_checkpoint_path = checkpoint_paths[best_idx]

    # Copy the best seed artifacts to stable filenames.
    best_model_copy = export_dir / f"linear_cup_{run_id}_best.pt"
    if Path(best_model_path).exists():
        shutil.copy2(best_model_path, best_model_copy)
    best_ckpt_copy = export_dir / f"linear_cup_{run_id}_best.ckpt"
    if Path(best_checkpoint_path).exists():
        shutil.copy2(best_checkpoint_path, best_ckpt_copy)

    out_path = None
    avg_preds = None
    if args.predict_ts:
        if not pred_list:
            raise RuntimeError("Predictions requested but no predictions were generated.")
        # Average predictions across seeds to reduce variance.
        avg_preds = np.mean(np.stack(pred_list, axis=0), axis=0)
        if test_ids is None:
            raise RuntimeError("Missing test ids in datamodule.")
        if write_predictions:
            output_path = (
                Path(args.output) if args.output else MODULE_DIR / "predictions" / "linear_cup.csv"
            )
            header = [] if args.no_header else DEFAULT_HEADER
            out_path = write_predictions_csv(output_path, test_ids, avg_preds, header_lines=header)

    summary_path = results_dir / f"final_summary_{run_id}.json"
    report_metrics = None
    if not args.no_search:
        report_metrics = best_metrics

    summary_payload = {
        "run_id": run_id,
        "target_index": target_index,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "predict_ts": args.predict_ts,
        "params": params,
        "epochs": train_epochs,
        "val_ratio": 0.0,
        "test_ratio": args.test_ratio,
        "seeds": SEEDS,
        "split_seed": args.split_seed,
        "scale_inputs": args.scale_inputs,
        "report_metrics": report_metrics,
        "report_note": (
            "Report metrics (MEE train/val/test) are taken from model selection. "
            "Retrain metrics below are for curves/inference only."
        ),
        "per_seed_metrics": per_seed_metrics,
        "metrics_mean": metrics_summary["mean"],
        "metrics_std": metrics_summary["std"],
        "curve_paths": curve_paths,
        "exported_models": export_paths,
        "checkpoints": checkpoint_paths,
        "best_seed": int(best_seed),
        "best_seed_metrics": per_seed_metrics[best_idx],
        "best_model_path": str(best_model_copy),
        "best_checkpoint_path": str(best_ckpt_copy),
        "predictions_path": str(out_path) if out_path is not None else None,
        "search_metrics": best_metrics if not args.no_search else None,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "summary_path": summary_path,
        "summary": summary_payload,
        "predictions": avg_preds,
        "prediction_ids": test_ids,
        "predictions_path": out_path,
    }


def train_and_predict(args: argparse.Namespace) -> Path | None:
    """Entry point for training (and optional prediction) from CLI args."""
    seed_everything(args.seed)

    default_train, default_test = default_data_paths()
    train_path = Path(args.train_path) if args.train_path else default_train
    test_path = Path(args.test_path) if args.test_path else default_test
    run_id = args.run_id or now_run_id()
    if args.predict_ts and not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    results_dir = Path(args.results_dir) if args.results_dir else MODULE_DIR / "results"
    export_dir = Path(args.export_dir) if args.export_dir else MODULE_DIR / "exports"
    results_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    target_indices = _resolve_target_indices(args)
    if target_indices == [None]:
        result = _run_single_target(
            args,
            train_path=train_path,
            test_path=test_path,
            run_id=run_id,
            results_dir=results_dir,
            export_dir=export_dir,
            target_index=None,
            write_predictions=True,
        )
        return result.get("predictions_path")

    per_target_results: list[dict[str, object]] = []
    per_target_preds: list[np.ndarray] = []
    pred_ids = None
    for idx in target_indices:
        target_run_id = f"{run_id}_t{idx + 1}"
        result = _run_single_target(
            args,
            train_path=train_path,
            test_path=test_path,
            run_id=target_run_id,
            results_dir=results_dir,
            export_dir=export_dir,
            target_index=idx,
            write_predictions=False,
        )
        per_target_results.append(
            {
                "target_index": idx,
                "summary_path": str(result["summary_path"]),
                "params": result["summary"].get("params"),
                "report_metrics": result["summary"].get("report_metrics"),
            }
        )
        if args.predict_ts:
            preds = result.get("predictions")
            if preds is None:
                raise RuntimeError("Predictions requested but missing per-target predictions.")
            per_target_preds.append(preds)
            if pred_ids is None:
                pred_ids = result.get("prediction_ids")

    out_path = None
    if args.predict_ts:
        if pred_ids is None:
            raise RuntimeError("Missing test ids for predictions.")
        columns: list[np.ndarray] = []
        for preds in per_target_preds:
            preds = np.asarray(preds)
            if preds.ndim == 2 and preds.shape[1] == 1:
                columns.append(preds[:, 0])
            elif preds.ndim == 1:
                columns.append(preds)
            else:
                raise RuntimeError(f"Unexpected per-target predictions shape: {preds.shape}")
        # Merge per-target outputs into a single (N, 4) matrix.
        combined = np.column_stack(columns)
        output_path = Path(args.output) if args.output else MODULE_DIR / "predictions" / "linear_cup.csv"
        header = [] if args.no_header else DEFAULT_HEADER
        out_path = write_predictions_csv(output_path, pred_ids, combined, header_lines=header)

    # Store a compact summary for multi-target runs.
    per_target_summary_path = results_dir / f"final_summary_{run_id}_per_target.json"
    per_target_payload = {
        "run_id": run_id,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "predict_ts": args.predict_ts,
        "per_target": True,
        "targets": per_target_results,
        "predictions_path": str(out_path) if out_path is not None else None,
    }
    per_target_summary_path.write_text(json.dumps(per_target_payload, indent=2), encoding="utf-8")

    return out_path


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    out_path = train_and_predict(args)
    if out_path is not None:
        print(f"Predictions saved to: {out_path}")
    else:
        print("Training complete. Use --predict-ts to generate the blind test predictions.")


if __name__ == "__main__":
    main()
 
