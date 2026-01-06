"""
Basic metrics for MONK experiments.
"""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error on scalar outputs.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true_arr - y_pred_arr) ** 2))


def accuracy(y_true: np.ndarray, y_pred_labels: np.ndarray) -> float:
    """
    Accuracy on discrete labels (0/1).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred_labels).astype(int)
    return float(np.mean(y_true_arr == y_pred_arr))
