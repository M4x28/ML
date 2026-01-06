from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _read_cup_csv(path: str | Path) -> pd.DataFrame:
    """Read a CUP CSV file without header, skipping comment lines."""
    return pd.read_csv(path, comment="#", header=None)


def load_cup_train(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the training CSV and return ids, inputs, and targets.
    """
    df = _read_cup_csv(path)
    if df.shape[1] < 6:
        raise ValueError(f"Unexpected column count in train file: {df.shape[1]}")
    ids = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:-4].to_numpy(dtype=np.float32)
    y = df.iloc[:, -4:].to_numpy(dtype=np.float32)
    return ids, X, y


def load_cup_test(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the blind test CSV and return ids and inputs (no targets).
    """
    df = _read_cup_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected column count in test file: {df.shape[1]}")
    ids = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return ids, X


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Split data into train/val/test using a single holdout split."""
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio + test_ratio must be in [0, 1).")

    # Default: use full dataset when no holdout is requested.
    X_train, y_train, ids_train = X, y, ids
    X_val, y_val, ids_val = None, None, None
    X_test, y_test, ids_test = None, None, None

    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio > 0:
        # Split once into train and holdout, then split holdout into val/test.
        X_train, X_holdout, y_train, y_holdout, ids_train, ids_holdout = train_test_split(
            X,
            y,
            ids,
            test_size=holdout_ratio,
            shuffle=True,
            random_state=seed,
        )
        if test_ratio > 0:
            if val_ratio > 0:
                test_size = test_ratio / holdout_ratio
                X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
                    X_holdout,
                    y_holdout,
                    ids_holdout,
                    test_size=test_size,
                    shuffle=True,
                    random_state=seed,
                )
            else:
                X_test, y_test, ids_test = X_holdout, y_holdout, ids_holdout
        else:
            X_val, y_val, ids_val = X_holdout, y_holdout, ids_holdout

    return {
        "train": {"X": X_train, "y": y_train, "ids": ids_train},
        "val": {"X": X_val, "y": y_val, "ids": ids_val},
        "test": {"X": X_test, "y": y_test, "ids": ids_test},
    }
