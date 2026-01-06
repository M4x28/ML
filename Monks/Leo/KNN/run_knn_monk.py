from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from knn_curves import build_mse_curves, save_accuracy_curve, save_mse_curve
from knn_search import run_grid_search
from monk_data import load_monk_task


OUTPUT_ROOT = Path(__file__).resolve().parent


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_k_values(value: str | None) -> list[int]:
    if value is None:
        return [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51]
    items = [v.strip() for v in value.split(",") if v.strip()]
    out: list[int] = []
    for item in items:
        try:
            num = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid k value: {item}") from exc
        if num > 0:
            out.append(num)
    if not out:
        raise ValueError("curve ks cannot be empty")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KNN grid search on MONK datasets.")
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which MONK tasks to run.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--export-dir", type=str, default=None)
    parser.add_argument("--curves-dir", type=str, default=None)
    parser.add_argument("--knn-jobs", type=int, default=1, help="Parallel jobs for KNN.")
    parser.add_argument("--curve-ks", type=str, default=None, help="Comma-separated K values for curves.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or now_run_id()

    results_root = Path(args.results_dir) if args.results_dir else OUTPUT_ROOT / "results"
    export_root = Path(args.export_dir) if args.export_dir else OUTPUT_ROOT / "exports"
    curves_root = Path(args.curves_dir) if args.curves_dir else OUTPUT_ROOT / "curves"
    results_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    curves_root.mkdir(parents=True, exist_ok=True)

    k_values = parse_k_values(args.curve_ks)

    for task_id in args.tasks:
        print(f"\n[MONK {task_id}] KNN grid search run_id={run_id}")
        summary = run_grid_search(
            task_id=task_id,
            run_id=run_id,
            results_root=results_root,
            export_root=export_root,
            n_jobs=int(args.knn_jobs),
        )

        best_params = dict(summary["best_params"])
        X_train, y_train = load_monk_task(task_id, split="train")
        X_test, y_test = load_monk_task(task_id, split="test")

        curves = build_mse_curves(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=best_params,
            n_jobs=int(args.knn_jobs),
            k_values=k_values,
        )

        curve_dir = curves_root / f"monk{task_id}"
        curve_path = curve_dir / f"knn_monk{task_id}_{run_id}_mse_curve.png"
        save_mse_curve(
            curves=curves,
            output_path=curve_path,
            title=f"MONK {task_id} - KNN MSE vs K",
        )
        acc_curve_path = curve_dir / f"knn_monk{task_id}_{run_id}_accuracy_curve.png"
        save_accuracy_curve(
            curves=curves,
            output_path=acc_curve_path,
            title=f"MONK {task_id} - KNN Accuracy vs K",
        )

        curve_json_path = curve_path.with_suffix(".json")
        curve_payload = {
            "task_id": task_id,
            "run_id": run_id,
            "k_values": curves["steps"],
            "train_mse": curves["train_mse"],
            "test_mse": curves["test_mse"],
            "train_acc": curves["train_acc"],
            "test_acc": curves["test_acc"],
        }
        curve_json_path.write_text(json.dumps(curve_payload, indent=2), encoding="utf-8")

        acc_json_path = acc_curve_path.with_suffix(".json")
        acc_payload = {
            "task_id": task_id,
            "run_id": run_id,
            "k_values": curves["steps"],
            "train_acc": curves["train_acc"],
            "test_acc": curves["test_acc"],
        }
        acc_json_path.write_text(json.dumps(acc_payload, indent=2), encoding="utf-8")

        summary["curve_path"] = str(curve_path)
        summary["curve_json"] = str(curve_json_path)
        summary["accuracy_curve_path"] = str(acc_curve_path)
        summary["accuracy_curve_json"] = str(acc_json_path)
        summary["curve_k_values"] = list(curves["steps"])
        summary_path = Path(summary["summary_json"])
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(
            "  best params:", summary["best_params"],
            " | final train MSE/acc:",
            f"{summary['final']['train']['mse']:.6f}/{summary['final']['train']['acc']:.4f}",
            " | final test MSE/acc:",
            f"{summary['final']['test']['mse']:.6f}/{summary['final']['test']['acc']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
