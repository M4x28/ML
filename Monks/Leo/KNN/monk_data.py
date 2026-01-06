"""
Utilities to load the MONK datasets.

Expected files in data/monk:
- monks-1.train / monks-1.test
- monks-2.train / monks-2.test
- monks-3.train / monks-3.test

Format (monks-*.train / monks-*.test):
    label  a1 a2 a3 a4 a5 a6  sample_id
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "monk"


def load_monk(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a MONK file and return (X, y) with categorical integer features.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    y = df.iloc[:, 0].astype(int).to_numpy()
    X = df.iloc[:, 1:7].astype(int).to_numpy()
    return X, y


def load_monk_task(
    task_id: int,
    split: Literal["train", "test"] = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load MONK-{task_id} from data/monk/monks-{task_id}.{split}.
    """
    if task_id not in (1, 2, 3):
        raise ValueError("task_id must be 1, 2, or 3.")
    file_path = DATA_DIR / f"monks-{task_id}.{split}"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return load_monk(file_path)


def make_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    80/20 stratified split for train/validation.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_val, y_train, y_val
