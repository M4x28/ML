from __future__ import annotations

"""Evaluation metrics used in ML-CUP experiments."""

import numpy as np


def mse_per_instance(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error computed per sample."""
    diff = pred - target
    if diff.ndim == 1:
        # Single target: squared error per sample.
        per_sample = diff ** 2
    else:
        # Multi-target: sum squared errors across targets per sample.
        per_sample = np.sum(diff ** 2, axis=1)
    return float(np.mean(per_sample))


def mean_euclidean_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Euclidean error (or MAE for a single target)."""
    diff = pred - target
    if diff.ndim == 1:
        # Single target: mean absolute error.
        return float(np.mean(np.abs(diff)))
    # Multi-target: mean L2 norm per sample.
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def mee_per_target(pred: np.ndarray, target: np.ndarray) -> list[float]:
    """Per-target MAE values."""
    diff = pred - target
    if diff.ndim == 1:
        # Single target: return a single-element list.
        return [float(np.mean(np.abs(diff)))]
    # Multi-target: average absolute error per output dimension.
    return [float(np.mean(np.abs(diff[:, idx]))) for idx in range(diff.shape[1])]


def evaluate_metrics(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute MSE/MEE metrics for a model on a dataset."""
    pred = model.predict(X)
    # Reuse the helper functions for consistent metrics.
    mse = mse_per_instance(pred, y)
    mee = mean_euclidean_error(pred, y)
    return {"mse": mse, "mee": mee, "mee_per_target": mee_per_target(pred, y)}
