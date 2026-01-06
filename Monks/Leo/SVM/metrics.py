"""
Metriche usate negli esperimenti MONK.

Nota:
- "MSE (LMS)" qui è inteso come mean squared error tra output continuo e target.
- "MEE" (Mean Euclidean Error) è la media della norma L2 dell'errore per pattern.
  Nel caso scalare coincide con la Mean Absolute Error.
"""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true_arr - y_pred_arr) ** 2))


def mee(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Euclidean Error (media norma L2 dell'errore).

    Per output scalare equivale alla Mean Absolute Error.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}")

    per_sample_norm = np.linalg.norm(y_true_arr - y_pred_arr, axis=1)
    return float(np.mean(per_sample_norm))


def accuracy_from_labels(y_true: np.ndarray, y_pred_labels: np.ndarray) -> float:
    """
    Accuracy su label discrete (0/1).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred_labels).astype(int)
    return float(np.mean(y_true_arr == y_pred_arr))

