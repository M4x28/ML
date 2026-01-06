from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import ParameterGrid

from knn_model import build_knn_pipeline
from metrics import accuracy, mse
from monk_data import load_monk_task, make_holdout_split

SEEDS = [0, 1, 2, 3, 4]


@dataclass(frozen=True)
class SeedMetrics:
    seed: int
    train_mse: float
    train_acc: float
    val_mse: float
    val_acc: float


def get_param_space() -> dict[str, list[Any]]:
    return {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "metric": ["minkowski"],
        "algorithm": ["auto"],
        "leaf_size": [30],
        "scaler_type": ["none", "standard"],
        "feature_rep": ["onehot"],
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _aggregate_metrics(per_seed: list[SeedMetrics]) -> dict[str, float]:
    keys = ["train_mse", "train_acc", "val_mse", "val_acc"]
    out: dict[str, float] = {}
    for key in keys:
        values = [getattr(entry, key) for entry in per_seed]
        mean, std = _mean_std(values)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return out


def _selection_key(agg: dict[str, float], selection: str) -> tuple[float, float]:
    if selection == "val_mse":
        return (float(agg["val_mse_mean"]), -float(agg["val_acc_mean"]))
    if selection == "val_acc":
        return (-float(agg["val_acc_mean"]), float(agg["val_mse_mean"]))
    raise ValueError(f"Unknown selection mode: {selection}")


def _predict_prob_pos(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        return proba.astype(float)
    if proba.shape[1] == 1:
        cls = int(model.classes_[0])
        return np.ones(len(X), dtype=float) if cls == 1 else np.zeros(len(X), dtype=float)
    classes = list(model.classes_)
    if 1 in classes:
        idx = classes.index(1)
        return proba[:, idx].astype(float)
    return proba[:, -1].astype(float)


def _eval_metrics(model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    y_prob = _predict_prob_pos(model, X)
    y_pred = (y_prob >= 0.5).astype(int)
    return mse(y, y_prob), accuracy(y, y_pred)


def run_grid_search(
    *,
    task_id: int,
    run_id: str,
    results_root: Path,
    export_root: Path,
    n_jobs: int,
    param_space: dict[str, list[Any]] | None = None,
    selection: str = "val_mse",
    tag: str | None = None,
) -> dict[str, Any]:
    X_full, y_full = load_monk_task(task_id, split="train")
    X_test, y_test = load_monk_task(task_id, split="test")

    results_dir = results_root / f"monk{task_id}"
    export_dir = export_root / f"monk{task_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    suffix = run_id if tag is None else f"{run_id}_{tag}"
    results_path = results_dir / f"knn_grid_{suffix}.jsonl"
    summary_path = results_dir / f"knn_summary_{suffix}.json"
    model_path = export_dir / f"knn_monk{task_id}_{suffix}.joblib"

    best_key: tuple[float, float] | None = None
    best_meta: dict[str, Any] | None = None

    space = {k: list(v) for k, v in (param_space or get_param_space()).items()}
    if "n_neighbors" in space:
        max_k = min(len(make_holdout_split(X_full, y_full, seed=seed)[0]) for seed in SEEDS)
        space["n_neighbors"] = [k for k in space["n_neighbors"] if int(k) <= max_k]
        if not space["n_neighbors"]:
            raise ValueError(f"No valid n_neighbors <= {max_k} for task {task_id}.")
    grid = list(ParameterGrid(space))

    with results_path.open("w", encoding="utf-8") as out_f:
        for idx, params in enumerate(grid, start=1):
            per_seed: list[SeedMetrics] = []
            for seed in SEEDS:
                X_train, X_val, y_train, y_val = make_holdout_split(X_full, y_full, seed=seed)
                model = build_knn_pipeline(params=params, n_jobs=n_jobs)
                model.fit(X_train, y_train)

                train_mse, train_acc = _eval_metrics(model, X_train, y_train)
                val_mse, val_acc = _eval_metrics(model, X_val, y_val)
                per_seed.append(
                    SeedMetrics(
                        seed=seed,
                        train_mse=float(train_mse),
                        train_acc=float(train_acc),
                        val_mse=float(val_mse),
                        val_acc=float(val_acc),
                    )
                )

            agg = _aggregate_metrics(per_seed)
            record = {
                "task_id": task_id,
                "combo_idx": idx,
                "params": params,
                "seeds": SEEDS,
                "per_seed": [asdict(m) for m in per_seed],
                "agg": agg,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            key = _selection_key(agg, selection)
            if best_key is None or key < best_key:
                best_key = key
                best_meta = {
                    "task_id": task_id,
                    "run_id": run_id,
                    "tag": tag,
                    "selection_mode": selection,
                    "best_params": params,
                    "best_combo_idx": idx,
                    "selection": {
                        "criterion": (
                            "min val_mse_mean, then max val_acc_mean"
                            if selection == "val_mse"
                            else "max val_acc_mean, then min val_mse_mean"
                        ),
                        "metrics": agg,
                        "per_seed": [asdict(m) for m in per_seed],
                    },
                }

    if best_meta is None:
        raise RuntimeError("No configurations evaluated; check parameter grid.")

    best_params = dict(best_meta["best_params"])
    final_model = build_knn_pipeline(params=best_params, n_jobs=n_jobs)
    final_model.fit(X_full, y_full)

    train_mse, train_acc = _eval_metrics(final_model, X_full, y_full)
    test_mse, test_acc = _eval_metrics(final_model, X_test, y_test)

    best_meta["final"] = {
        "train": {"mse": float(train_mse), "acc": float(train_acc)},
        "test": {"mse": float(test_mse), "acc": float(test_acc)},
    }
    best_meta["param_space"] = space

    joblib.dump(final_model, model_path)
    best_meta["exported_model"] = str(model_path)
    best_meta["results_jsonl"] = str(results_path)

    summary_path.write_text(json.dumps(best_meta, indent=2), encoding="utf-8")
    best_meta["summary_json"] = str(summary_path)
    return best_meta
