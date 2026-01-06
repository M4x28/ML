from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from knn_model import build_knn_pipeline
from metrics import accuracy, mse


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


def build_mse_curves(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
    n_jobs: int,
    k_values: list[int],
) -> dict[str, list[float]]:
    train_mse: list[float] = []
    test_mse: list[float] = []
    train_acc: list[float] = []
    test_acc: list[float] = []
    steps: list[int] = []

    for k in k_values:
        params_k = dict(params)
        params_k["n_neighbors"] = int(k)
        model = build_knn_pipeline(params=params_k, n_jobs=n_jobs)
        model.fit(X_train, y_train)

        train_prob = _predict_prob_pos(model, X_train)
        test_prob = _predict_prob_pos(model, X_test)
        train_pred = (train_prob >= 0.5).astype(int)
        test_pred = (test_prob >= 0.5).astype(int)

        train_mse.append(mse(y_train, train_prob))
        test_mse.append(mse(y_test, test_prob))
        train_acc.append(accuracy(y_train, train_pred))
        test_acc.append(accuracy(y_test, test_pred))
        steps.append(int(k))

    return {
        "steps": steps,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


def save_mse_curve(
    *,
    curves: dict[str, list[float]],
    output_path: str | Path,
    title: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = curves["steps"]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, curves["train_mse"], label="train_mse")
    plt.plot(steps, curves["test_mse"], label="test_mse")
    plt.xlabel("K")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_accuracy_curve(
    *,
    curves: dict[str, list[float]],
    output_path: str | Path,
    title: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = curves["steps"]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, curves["train_acc"], label="train_acc")
    plt.plot(steps, curves["test_acc"], label="test_acc")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
