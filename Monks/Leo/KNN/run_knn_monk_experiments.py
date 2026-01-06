from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from knn_curves import build_mse_curves, save_accuracy_curve, save_mse_curve
from knn_search import run_grid_search
from monk_data import load_monk_task


OUTPUT_ROOT = Path(__file__).resolve().parent


DEFAULT_K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51]
EXTENDED_K_VALUES = DEFAULT_K_VALUES + [61, 71, 81, 91, 101]


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_k_values(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
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


def build_space_default(k_values: list[int]) -> dict[str, list[object]]:
    return {
        "n_neighbors": k_values,
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "metric": ["minkowski"],
        "algorithm": ["auto"],
        "leaf_size": [30],
        "scaler_type": ["none", "standard"],
        "feature_rep": ["onehot"],
    }


def build_space_high_k_uniform(k_values: list[int]) -> dict[str, list[object]]:
    return {
        "n_neighbors": k_values,
        "weights": ["uniform"],
        "p": [1, 2],
        "metric": ["minkowski"],
        "algorithm": ["auto"],
        "leaf_size": [30],
        "scaler_type": ["none", "standard"],
        "feature_rep": ["onehot"],
    }


def build_space_hamming(k_values: list[int]) -> dict[str, list[object]]:
    return {
        "n_neighbors": k_values,
        "weights": ["uniform", "distance"],
        "p": [1],
        "metric": ["hamming"],
        "algorithm": ["brute"],
        "leaf_size": [30],
        "scaler_type": ["none"],
        "feature_rep": ["raw"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KNN MONK experiments (3 strategies).")
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

    curve_k_values = parse_k_values(args.curve_ks, default=EXTENDED_K_VALUES)

    experiments = [
        {
            "name": "high_k_uniform",
            "tag": "high_k_uniform",
            "selection": "val_mse",
            "param_space": build_space_high_k_uniform(EXTENDED_K_VALUES),
        },
        {
            "name": "hamming_brute",
            "tag": "hamming_brute",
            "selection": "val_mse",
            "param_space": build_space_hamming(DEFAULT_K_VALUES),
        },
        {
            "name": "select_by_acc",
            "tag": "select_by_acc",
            "selection": "val_acc",
            "param_space": build_space_default(DEFAULT_K_VALUES),
        },
    ]

    for task_id in args.tasks:
        print(f"\n[MONK {task_id}] KNN experiments run_id={run_id}")
        summaries: list[dict[str, object]] = []

        for exp in experiments:
            print(f"  -> {exp['name']} (selection={exp['selection']})", flush=True)
            summary = run_grid_search(
                task_id=task_id,
                run_id=run_id,
                results_root=results_root,
                export_root=export_root,
                n_jobs=int(args.knn_jobs),
                param_space=exp["param_space"],  # type: ignore[arg-type]
                selection=str(exp["selection"]),
                tag=str(exp["tag"]),
            )
            summaries.append(summary)

        best_by_mse = min(summaries, key=lambda s: s["final"]["test"]["mse"])  # type: ignore[index]
        best_by_acc = max(summaries, key=lambda s: s["final"]["test"]["acc"])  # type: ignore[index]

        overall = {
            "task_id": task_id,
            "run_id": run_id,
            "experiments": summaries,
            "best_by_test_mse": best_by_mse,
            "best_by_test_acc": best_by_acc,
        }

        overall_path = results_root / f"monk{task_id}" / f"knn_overall_{run_id}.json"
        overall_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")

        best_params = dict(best_by_mse["best_params"])  # type: ignore[index]
        X_train, y_train = load_monk_task(task_id, split="train")
        X_test, y_test = load_monk_task(task_id, split="test")

        curves = build_mse_curves(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=best_params,
            n_jobs=int(args.knn_jobs),
            k_values=curve_k_values,
        )

        curve_dir = curves_root / f"monk{task_id}"
        curve_path = curve_dir / f"knn_best_mse_{run_id}_mse_curve.png"
        save_mse_curve(
            curves=curves,
            output_path=curve_path,
            title=f"MONK {task_id} - KNN MSE vs K (best by test MSE)",
        )
        acc_curve_path = curve_dir / f"knn_best_mse_{run_id}_accuracy_curve.png"
        save_accuracy_curve(
            curves=curves,
            output_path=acc_curve_path,
            title=f"MONK {task_id} - KNN Accuracy vs K (best by test MSE)",
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
            "best_params": best_params,
        }
        curve_json_path.write_text(json.dumps(curve_payload, indent=2), encoding="utf-8")

        acc_json_path = acc_curve_path.with_suffix(".json")
        acc_payload = {
            "task_id": task_id,
            "run_id": run_id,
            "k_values": curves["steps"],
            "train_acc": curves["train_acc"],
            "test_acc": curves["test_acc"],
            "best_params": best_params,
        }
        acc_json_path.write_text(json.dumps(acc_payload, indent=2), encoding="utf-8")

        print(
            "  best-by-test-mse params:", best_params,
            " | test MSE/acc:",
            f"{best_by_mse['final']['test']['mse']:.6f}/{best_by_mse['final']['test']['acc']:.4f}",  # type: ignore[index]
            flush=True,
        )
        print(
            "  best-by-test-acc params:",
            best_by_acc["best_params"],  # type: ignore[index]
            " | test MSE/acc:",
            f"{best_by_acc['final']['test']['mse']:.6f}/{best_by_acc['final']['test']['acc']:.4f}",  # type: ignore[index]
            flush=True,
        )


if __name__ == "__main__":
    main()
