from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from Cup.Leo.common.cup_data import load_cup_test, load_cup_train, split_train_val_test
from Cup.Leo.common.cup_io import DEFAULT_HEADER, export_artifacts, write_predictions_csv
from Cup.Leo.common.cup_metrics import evaluate_metrics, mean_euclidean_error, mee_per_target, mse_per_instance
from svm_curves import build_curves, build_curves_per_target, save_mee_curve, save_mse_curve
from svm_model import build_svm_pipeline
from svm_search import SEEDS, run_search


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def default_data_paths() -> tuple[Path, Path]:
    """Return default train/test dataset paths."""
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "ML-CUP25-TR.csv"
    test_path = project_root / "data" / "ML-CUP25-TS.csv"
    return train_path, test_path


def now_run_id() -> str:
    """Return a timestamped run identifier."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_curve_iters(value: str | None) -> list[int]:
    """Parse comma-separated max_iter values for curve plots."""
    if value is None:
        return [10, 50, 100, 200, 500, 1000]
    items = [v.strip() for v in value.split(",") if v.strip()]
    out = []
    for item in items:
        try:
            num = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid curve iter: {item}") from exc
        if num > 0:
            out.append(num)
    if not out:
        raise ValueError("curve iters cannot be empty")
    return out


def parse_pca_components(value: str | None) -> float | int:
    """Parse PCA components as float (variance) or int (n_components)."""
    if value is None:
        return 0.95
    text = value.strip()
    if not text:
        raise ValueError("pca-components cannot be empty")
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid pca-components: {value}") from exc
    if numeric <= 0:
        raise ValueError("pca-components must be positive")
    if numeric <= 1.0:
        return float(numeric)
    if float(int(numeric)) != numeric:
        raise ValueError("pca-components > 1 must be an integer")
    return int(numeric)


def predict_per_target(models: list, X: np.ndarray) -> np.ndarray:
    """Predict with one model per target and stack outputs."""
    preds = [model.predict(X) for model in models]
    return np.column_stack(preds)


def metrics_from_predictions(pred: np.ndarray, target: np.ndarray) -> dict[str, object]:
    """Compute metrics from precomputed predictions."""
    return {
        "mse": mse_per_instance(pred, target),
        "mee": mean_euclidean_error(pred, target),
        "mee_per_target": mee_per_target(pred, target),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for search, training, and prediction."""
    parser = argparse.ArgumentParser(description="SVM (SVR) model for ML-CUP regression.")
    parser.add_argument("--train-path", type=str, default=None, help="Path to ML-CUP25-TR.csv")
    parser.add_argument("--test-path", type=str, default=None, help="Path to ML-CUP25-TS.csv")
    parser.add_argument("--output", type=str, default=None, help="Output predictions CSV path")
    parser.add_argument("--export-path", type=str, default=None, help="Optional model export path")

    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for train/val/test split.")
    parser.add_argument("--svm-jobs", type=int, default=1, help="Parallel jobs for multioutput SVR.")

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

    parser.add_argument(
        "--pca",
        dest="use_pca",
        action="store_true",
        help="Enable PCA after feature mapping.",
    )
    parser.add_argument(
        "--no-pca",
        dest="use_pca",
        action="store_false",
        help="Disable PCA after feature mapping.",
    )
    parser.set_defaults(use_pca=False)
    parser.add_argument(
        "--pca-components",
        type=parse_pca_components,
        default=0.95,
        help="PCA components: float in (0,1] for variance, or int > 1 for components.",
    )
    parser.add_argument(
        "--per-target",
        action="store_true",
        help="Optimize and train one SVR per target.",
    )

    parser.add_argument("--predict-ts", action="store_true", help="Generate predictions on TS.")
    parser.add_argument("--no-header", action="store_true", help="Do not write header lines in output.")

    parser.add_argument("--no-search", action="store_true", help="Disable Optuna adaptive grid search.")
    parser.add_argument("--n-trials", type=int, default=0, help="Optuna trials (0 = full grid).")
    parser.add_argument("--n-jobs", type=int, default=1, help="Optuna parallel jobs.")
    parser.add_argument("--optuna-seed", type=int, default=0, help="Seed for Optuna sampler.")
    parser.add_argument(
        "--search-method",
        type=str,
        default="tpe",
        choices=["tpe", "cmaes", "smac"],
        help="Search method for hyperparameters.",
    )
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name.")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL or sqlite file path (enables resume).",
    )
    parser.add_argument(
        "--cmaes-center-from",
        type=str,
        default=None,
        help="Path to a previous search_summary JSON to warm-start CMA-ES/SMAC.",
    )
    parser.add_argument(
        "--cmaes-log-span",
        type=float,
        default=1.0,
        help="Log10 span around the center for CMA-ES (e.g., 1.0 = one order).",
    )
    parser.add_argument("--results-dir", type=str, default=None, help="Directory for JSON results.")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory for saved models.")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (default: timestamp).")
    parser.add_argument("--curve-iters", type=str, default=None, help="Comma-separated max_iter values.")

    parser.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "poly", "linear"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--coef0", type=float, default=0.0)
    parser.add_argument("--feature-map", type=str, default="identity", choices=["identity", "poly2"])
    return parser.parse_args()


def train_and_predict(args: argparse.Namespace) -> Path | None:
    """Run search (optional), retrain, and optionally predict TS."""
    default_train, default_test = default_data_paths()
    train_path = Path(args.train_path) if args.train_path else default_train
    test_path = Path(args.test_path) if args.test_path else default_test
    run_id = args.run_id or now_run_id()
    if args.predict_ts and not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parent / "results"
    export_dir = Path(args.export_dir) if args.export_dir else Path(__file__).resolve().parent / "exports"
    results_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    num_targets = None
    if args.per_target:
        # Detect output dimensionality from training data.
        _, _, y_preview = load_cup_train(train_path)
        num_targets = int(y_preview.shape[1]) if y_preview.ndim > 1 else 1

    params: dict[str, object]
    report_metrics = None
    per_target_params: list[dict[str, object]] | None = None
    per_target_metrics: list[dict[str, object]] | None = None
    per_target_top_trials: list[list[dict[str, object]]] | None = None
    per_target_trials: list[list[dict[str, object]]] | None = None
    if args.no_search:
        # Use CLI params directly without search.
        gamma = args.gamma
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            gamma = float(gamma)
        params = {
            "kernel": args.kernel,
            "C": args.C,
            "epsilon": args.epsilon,
            "gamma": gamma,
            "degree": args.degree,
            "coef0": args.coef0,
            "feature_map": args.feature_map,
        }
        top_trials = []
        all_trials = []
        if args.per_target:
            if num_targets is None:
                raise RuntimeError("per-target mode requires target count.")
            per_target_params = [dict(params) for _ in range(num_targets)]
            params = {"per_target": True, "targets": per_target_params}
    else:
        storage = None
        if args.storage:
            if "://" in args.storage:
                storage = args.storage
            else:
                storage_path = Path(args.storage).resolve()
                storage = f"sqlite:///{storage_path.as_posix()}"
            if args.n_trials <= 0:
                raise ValueError("When using --storage, set --n-trials to the additional trials.")

        search_kwargs: dict[str, object] = {
            "train_path": str(train_path),
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seeds": SEEDS,
            "split_seed": args.split_seed,
            "scale_inputs": args.scale_inputs,
            "use_pca": args.use_pca,
            "pca_components": args.pca_components,
            "n_jobs": int(args.svm_jobs),
            "n_trials": args.n_trials,
            "optuna_seed": args.optuna_seed,
            "top_k": 5,
            "storage": storage,
            "study_name": args.study_name,
        }

        if args.search_method in ("cmaes", "smac"):
            kernel = args.kernel
            feature_map = args.feature_map
            degree = args.degree
            coef0 = args.coef0
            center_C = args.C
            center_gamma: float | None
            center_epsilon = args.epsilon

            if args.cmaes_center_from:
                # Warm-start CMA-ES/SMAC from a prior summary.
                center_path = Path(args.cmaes_center_from)
                if not center_path.exists():
                    raise FileNotFoundError(f"CMA-ES center file not found: {center_path}")
                payload = json.loads(center_path.read_text(encoding="utf-8"))
                best_params = payload.get("best_params", payload.get("params", {}))
                kernel = best_params.get("kernel", kernel)
                feature_map = best_params.get("feature_map", feature_map)
                degree = int(best_params.get("degree", degree))
                coef0 = float(best_params.get("coef0", coef0))
                center_C = float(best_params.get("C", center_C))
                center_epsilon = float(best_params.get("epsilon", center_epsilon))
                center_gamma = best_params.get("gamma", None)
            else:
                center_gamma = args.gamma

            if isinstance(center_gamma, str):
                if center_gamma in ("scale", "auto"):
                    center_gamma = None
                else:
                    center_gamma = float(center_gamma)

            search_kwargs.update(
                {
                    "kernel": kernel,
                    "feature_map": feature_map,
                    "degree": degree,
                    "coef0": coef0,
                    "center_C": float(center_C),
                    "center_gamma": center_gamma,
                    "center_epsilon": float(center_epsilon),
                    "log_span": float(args.cmaes_log_span),
                }
            )

        if args.per_target:
            if num_targets is None:
                raise RuntimeError("per-target mode requires target count.")
            per_target_params = []
            per_target_metrics = []
            per_target_top_trials = []
            per_target_trials = []
            for target_idx in range(num_targets):
                # Run a separate search per target.
                search = run_search(method=args.search_method, target_idx=target_idx, **search_kwargs)
                target_params = dict(search["best_params"])
                target_metrics = dict(search["best_metrics"])
                per_target_params.append(target_params)
                per_target_metrics.append(target_metrics)
                per_target_top_trials.append(search.get("top_trials", []))
                per_target_trials.append(search.get("trials", []))
                print(
                    f"Target {target_idx + 1} best metrics (mean over seeds):",
                    f"train_mee={target_metrics.get('train_mee_mean', float('nan')):.6f}",
                    f"val_mee={target_metrics.get('val_mee_mean', float('nan')):.6f}",
                    f"test_mee={target_metrics.get('test_mee_mean', float('nan')):.6f}",
                )
            params = {"per_target": True, "targets": per_target_params}
            report_metrics = {"per_target": per_target_metrics}
            top_trials = []
            all_trials = []
        else:
            search = run_search(method=args.search_method, **search_kwargs)
            params = search["best_params"]
            report_metrics = search["best_metrics"]
            top_trials = search.get("top_trials", [])
            all_trials = search.get("trials", [])

            if report_metrics:
                print(
                    "Best metrics (mean over seeds, from model selection):",
                    f"train_mee={report_metrics.get('train_mee_mean', float('nan')):.6f}",
                    f"val_mee={report_metrics.get('val_mee_mean', float('nan')):.6f}",
                    f"test_mee={report_metrics.get('test_mee_mean', float('nan')):.6f}",
                )
            if top_trials:
                print("Top trials (by val_mee, val_mse):")
                for idx, entry in enumerate(top_trials, start=1):
                    metrics = entry.get("metrics", {})
                    params_row = entry.get("params", {})
                    print(
                        f"  {idx:02d}) trial={entry.get('trial')} "
                        f"val_mee={metrics.get('val_mee_mean', float('nan')):.6f} "
                        f"val_mse={metrics.get('val_mse_mean', float('nan')):.6f} "
                        f"params={params_row}"
                    )

        summary_path = results_dir / f"search_summary_{run_id}.json"
        summary_payload = {
            "run_id": run_id,
            "train_path": str(train_path),
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seeds": SEEDS,
            "split_seed": args.split_seed,
            "scale_inputs": args.scale_inputs,
            "pca_enabled": args.use_pca,
            "pca_components": args.pca_components if args.use_pca else None,
            "per_target": args.per_target,
            "search_method": args.search_method,
            "study_name": args.study_name,
            "storage": storage,
            "n_trials": search.get("n_trials"),
            "best_params": params,
            "best_metrics": report_metrics,
            "selection_note": search.get("selection_note"),
            "top_trials": top_trials,
            "trials": all_trials,
            "best_params_per_target": per_target_params,
            "best_metrics_per_target": per_target_metrics,
            "top_trials_per_target": per_target_top_trials,
            "trials_per_target": per_target_trials,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    params = dict(params)
    params["pca_enabled"] = bool(args.use_pca)
    if args.use_pca and "pca_components" not in params:
        params["pca_components"] = args.pca_components
    if args.per_target and per_target_params is not None:
        for target_params in per_target_params:
            if args.use_pca and "pca_components" not in target_params:
                target_params["pca_components"] = args.pca_components

    if args.per_target and per_target_params is not None:
        for idx, target_params in enumerate(per_target_params):
            if target_params.get("feature_map") == "poly2" and target_params.get("kernel") == "poly":
                raise ValueError(
                    f"Invalid configuration for target {idx + 1}: feature_map=poly2 with kernel=poly."
                )
    elif params.get("feature_map") == "poly2" and params.get("kernel") == "poly":
        raise ValueError("Invalid configuration: feature_map=poly2 with kernel=poly.")

    ids, X, y = load_cup_train(train_path)
    splits = split_train_val_test(
        X,
        y,
        ids,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    X_tr, y_tr = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_te, y_te = splits["test"]["X"], splits["test"]["y"]

    if X_val is None or y_val is None:
        raise RuntimeError("Validation split is empty; set val_ratio > 0.")
    if X_te is None or y_te is None:
        raise RuntimeError("Test split is empty; set test_ratio > 0.")

    X_trainval = np.vstack([X_tr, X_val])
    y_trainval = np.vstack([y_tr, y_val])

    model = None
    models = None
    if args.per_target:
        if per_target_params is None:
            raise RuntimeError("per-target mode requires per-target parameters.")
        models = []
        for idx, target_params in enumerate(per_target_params):
            # Train one SVR per target on the combined train+val split.
            target_model = build_svm_pipeline(
                params=target_params,
                scale_inputs=args.scale_inputs,
                n_jobs=int(args.svm_jobs),
                use_pca=args.use_pca,
                multi_output=False,
            )
            target_model.fit(X_trainval, y_trainval[:, idx])
            models.append(target_model)
        train_pred = predict_per_target(models, X_trainval)
        test_pred = predict_per_target(models, X_te)
        retrain_train_metrics = metrics_from_predictions(train_pred, y_trainval)
        retrain_test_metrics = metrics_from_predictions(test_pred, y_te)
    else:
        model = build_svm_pipeline(
            params=params,
            scale_inputs=args.scale_inputs,
            n_jobs=int(args.svm_jobs),
            use_pca=args.use_pca,
        )
        model.fit(X_trainval, y_trainval)

        retrain_train_metrics = evaluate_metrics(model, X_trainval, y_trainval)
        retrain_test_metrics = evaluate_metrics(model, X_te, y_te)
    print(
        "Retrain metrics (for curves/inference only):",
        f"train_mse={retrain_train_metrics['mse']:.6f}",
        f"train_mee={retrain_train_metrics['mee']:.6f}",
        f"test_mse={retrain_test_metrics['mse']:.6f}",
        f"test_mee={retrain_test_metrics['mee']:.6f}",
    )
    if retrain_train_metrics.get("mee_per_target"):
        print("Retrain MEE per target (train):", retrain_train_metrics["mee_per_target"])
    if retrain_test_metrics.get("mee_per_target"):
        print("Retrain MEE per target (test):", retrain_test_metrics["mee_per_target"])

    curve_iters = parse_curve_iters(args.curve_iters)
    if args.per_target:
        if per_target_params is None:
            raise RuntimeError("Per-target parameters are missing; cannot build curves.")
        curves = build_curves_per_target(
            X_train=X_trainval,
            y_train=y_trainval,
            X_test=X_te,
            y_test=y_te,
            params_per_target=per_target_params,
            scale_inputs=args.scale_inputs,
            use_pca=args.use_pca,
            n_jobs=int(args.svm_jobs),
            curve_iters=curve_iters,
        )
    else:
        curves = build_curves(
            X_train=X_trainval,
            y_train=y_trainval,
            X_test=X_te,
            y_test=y_te,
            params=params,
            scale_inputs=args.scale_inputs,
            use_pca=args.use_pca,
            n_jobs=int(args.svm_jobs),
            curve_iters=curve_iters,
        )
    curves_dir = Path(__file__).resolve().parent / "curves" / run_id
    curve_path = save_mse_curve(
        curves=curves,
        output_dir=curves_dir,
        run_name="svm_cup",
        title="SVM CUP - MSE Curve (retrain)",
    )
    mee_curve_path = save_mee_curve(
        curves=curves,
        output_dir=curves_dir,
        run_name="svm_cup",
        title="SVM CUP - MEE Curve (retrain)",
    )

    export_path = (
        Path(args.export_path)
        if args.export_path
        else export_dir / f"svm_cup_{run_id}_best.joblib"
    )
    export_model = models if args.per_target else model
    if export_model is None:
        raise RuntimeError("Model export is unavailable; training did not produce a model.")
    export_artifacts(
        export_path,
        model=export_model,
        input_dim=int(X.shape[1]),
        output_dim=int(y.shape[1]),
        params=params,
    )

    out_path = None
    if args.predict_ts:
        ts_ids, X_ts = load_cup_test(test_path)
        if args.per_target:
            if models is None:
                raise RuntimeError("Per-target models are missing; cannot predict TS.")
            preds = predict_per_target(models, X_ts)
        else:
            if model is None:
                raise RuntimeError("Model is missing; cannot predict TS.")
            preds = model.predict(X_ts)
        output_path = Path(args.output) if args.output else Path(__file__).resolve().parent / "predictions" / "svm_cup.csv"
        header = [] if args.no_header else DEFAULT_HEADER
        out_path = write_predictions_csv(output_path, ts_ids, preds, header_lines=header)

    final_summary_path = results_dir / f"final_summary_{run_id}.json"
    final_payload = {
        "run_id": run_id,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "predict_ts": args.predict_ts,
        "params": params,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "scale_inputs": args.scale_inputs,
        "pca_enabled": args.use_pca,
        "pca_components": args.pca_components if args.use_pca else None,
        "per_target": args.per_target,
        "per_target_params": per_target_params,
        "report_metrics": report_metrics,
        "report_metrics_per_target": per_target_metrics,
        "report_note": (
            "Report metrics (MEE train/val/test) are taken from model selection. "
            "Retrain metrics below are for curves/inference only."
        ),
        "retrain_metrics": {
            "train": retrain_train_metrics,
            "test": retrain_test_metrics,
        },
        "curve_path": str(curve_path),
        "mee_curve_path": str(mee_curve_path),
        "exported_model": str(export_path),
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
