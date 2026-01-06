from __future__ import annotations

import numpy as np


def mse_per_instance(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error averaged over instances (sum over outputs per sample)."""
    diff = pred - target
    if diff.ndim == 1:
        per_sample = diff ** 2
    else:
        # Sum over output dimension to get per-sample squared error.
        per_sample = np.sum(diff ** 2, axis=1)
    return float(np.mean(per_sample))


def mean_euclidean_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Euclidean Error (MEE) across output dimensions."""
    diff = pred - target
    if diff.ndim == 1:
        return float(np.mean(np.abs(diff)))
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def mee_per_target(pred: np.ndarray, target: np.ndarray) -> list[float]:
    """Compute mean absolute error per target dimension."""
    diff = pred - target
    if diff.ndim == 1:
        return [float(np.mean(np.abs(diff)))]
    return [float(np.mean(np.abs(diff[:, idx]))) for idx in range(diff.shape[1])]


def evaluate_metrics(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute MSE/MEE metrics for a fitted model."""
    pred = model.predict(X)
    mse = mse_per_instance(pred, y)
    mee = mean_euclidean_error(pred, y)
    return {"mse": mse, "mee": mee, "mee_per_target": mee_per_target(pred, y)}
