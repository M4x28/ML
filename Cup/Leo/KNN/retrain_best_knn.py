from __future__ import annotations

"""Retrain the best multi-target KNN model found in a summary."""

import argparse
import json
from pathlib import Path

import numpy as np

from cup_data import load_cup_train
from cup_metrics import mean_euclidean_error, mee_per_target, mse_per_instance, evaluate_metrics
from cup_io import export_artifacts
from knn_model import build_knn_pipeline


def _find_repo_root(start: Path) -> Path:
    """Locate the repo root by searching for the shared data folder."""
    for candidate in (start, *start.parents):
        if (candidate / "data" / "ML-CUP25-TR.csv").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)


def _load_best_params(summary_path: Path) -> tuple[bool, list[dict[str, object]] | dict[str, object], dict]:
    """Extract best params (per-target or global) from a summary JSON."""
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    per_target_params = payload.get("per_target_params")
    if isinstance(per_target_params, list) and per_target_params:
        # Summary already exposes per-target params explicitly.
        return True, list(per_target_params), payload

    params = payload.get("params") or payload.get("best_params")
    if isinstance(params, dict) and params.get("per_target") and params.get("targets"):
        # Fallback: params dict may contain a per-target structure.
        return True, list(params["targets"]), payload

    if isinstance(params, dict):
        # Standard single-model params.
        return False, dict(params), payload

    raise ValueError(f"Unable to read best params from {summary_path}")


def _metrics_from_predictions(pred: np.ndarray, target: np.ndarray) -> dict[str, object]:
    """Compute MSE/MEE metrics for multi-target predictions."""
    return {
        "mse": mse_per_instance(pred, target),
        "mee": mean_euclidean_error(pred, target),
        "mee_per_target": mee_per_target(pred, target),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for retraining."""
    parser = argparse.ArgumentParser(description="Retrain best multi-target KNN model from a summary JSON.")
    parser.add_argument(
        "--summary",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "final_summary_knn_fullgrid_global_01.json"),
        help="Path to a final_summary/search_summary JSON with best params.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(REPO_ROOT / "data" / "ML-CUP25-TR.csv"),
        help="Path to ML-CUP25-TR.csv.",
    )
    parser.add_argument("--knn-jobs", type=int, default=1, help="Parallel jobs for KNN.")
    parser.add_argument(
        "--export-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "exports" / "knn_cup_best_retrained.joblib"),
        help="Output joblib path for the retrained model.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip writing the joblib export (for dry runs).",
    )
    parser.add_argument(
        "--allow-per-target",
        action="store_true",
        help="Allow per-target retraining if the summary is per-target.",
    )
    return parser.parse_args()


def main() -> None:
    """Retrain the selected best model on the full training set."""
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    train_path = Path(args.train_path).resolve()

    per_target, params, payload = _load_best_params(summary_path)
    scale_inputs = bool(payload.get("scale_inputs", True))

    _ids, X, y = load_cup_train(train_path)
    if y.ndim == 1:
        # Ensure y is always 2D for multi-target logic.
        y = y.reshape(-1, 1)

    export_path = Path(args.export_path).resolve()

    if per_target and not args.allow_per_target:
        # Enforce multi-target retraining by default.
        raise ValueError(
            "Per-target parameters found in the summary. "
            "Use a global summary file or pass --allow-per-target."
        )

    if per_target:
        models = []
        for idx, target_params in enumerate(params):
            # Train each target independently on the full dataset.
            model = build_knn_pipeline(
                params=target_params,
                scale_inputs=scale_inputs,
                n_jobs=int(args.knn_jobs),
            )
            model.fit(X, y[:, idx])
            models.append(model)

        preds = np.column_stack([model.predict(X) for model in models])
        train_metrics = _metrics_from_predictions(preds, y)
        # Track configuration used to reproduce the retrain.
        export_params = {
            "per_target": True,
            "targets": list(params),
            "scale_inputs": scale_inputs,
            "source_summary": str(summary_path),
        }
        export_model = models
    else:
        model = build_knn_pipeline(
            params=params,
            scale_inputs=scale_inputs,
            n_jobs=int(args.knn_jobs),
        )
        model.fit(X, y)
        train_metrics = evaluate_metrics(model, X, y)
        # Store metadata alongside the single-model export.
        export_params = {
            **params,
            "scale_inputs": scale_inputs,
            "source_summary": str(summary_path),
        }
        export_model = model

    print("Retrain metrics on full training data:")
    print(f"  train_mse={train_metrics['mse']:.6f}")
    print(f"  train_mee={train_metrics['mee']:.6f}")
    if train_metrics.get("mee_per_target"):
        print("  train_mee_per_target:", train_metrics["mee_per_target"])

    if args.no_export:
        print("Skipping export (--no-export).")
        return

    export_artifacts(
        export_path,
        model=export_model,
        input_dim=int(X.shape[1]),
        output_dim=int(y.shape[1]),
        params=export_params,
    )
    print(f"Retrained model saved to: {export_path}")


if __name__ == "__main__":
    main()
