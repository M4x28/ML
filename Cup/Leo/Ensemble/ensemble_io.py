from __future__ import annotations

"""I/O helpers for ML-CUP ensemble scripts."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Reuse common I/O helpers to avoid duplicated code
from common.cup_io import DEFAULT_HEADER, ensure_parent_dir, write_predictions_csv


def _read_cup_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", header=None)


def read_cup_train(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _read_cup_csv(path)
    if df.shape[1] < 6:
        raise ValueError(f"Unexpected column count in train file: {df.shape[1]}")
    ids = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:-4].to_numpy(dtype=np.float32)
    y = df.iloc[:, -4:].to_numpy(dtype=np.float32)
    return ids, X, y


def read_cup_test(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
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
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio + test_ratio must be in [0, 1).")

    X_train, y_train, ids_train = X, y, ids
    X_val, y_val, ids_val = None, None, None
    X_test, y_test, ids_test = None, None, None

    # First split into train/holdout, then split holdout into val/test.
    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio > 0:
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


# `ensure_parent_dir` and `write_predictions_csv` are imported from
# `common.cup_io` to keep a single canonical implementation.
