"""
Utility per caricare i dataset MONK e creare lo split hold-out 80/20.

I file sono attesi in `data/monk/` (relativo alla root del progetto):
- `monks-1.train` / `monks-1.test`
- `monks-2.train` / `monks-2.test`
- `monks-3.train` / `monks-3.test`

Formato atteso (file monks-*.train / monks-*.test):
    label  a1 a2 a3 a4 a5 a6  sample_id

Dove:
- label è binaria {0, 1}
- a1..a6 sono feature categoriali (interi)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "monk"


def load_monk(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica un file MONK.

    Returns:
        X: ndarray (n_samples, 6) con feature categoriali intere.
        y: ndarray (n_samples,) con label binarie 0/1.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")  # gestisce eventuali colonne vuote finali

    y = df[0].astype(int).to_numpy()
    X = df.iloc[:, 1:7].astype(int).to_numpy()
    return X, y


def load_monk_task(
    task_id: int,
    split: Literal["train", "test"] = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica MONK-{task_id} dal path standard `data/monk/monks-{task_id}.{split}`.
    """
    if task_id not in (1, 2, 3):
        raise ValueError("task_id deve essere 1, 2 oppure 3.")
    file_path = DATA_DIR / f"monks-{task_id}.{split}"
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato: {file_path}")
    return load_monk(file_path)


def make_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split hold-out 80/20 con shuffle e stratificazione.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
