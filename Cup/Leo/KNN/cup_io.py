from __future__ import annotations

"""File I/O utilities for ML-CUP KNN experiments."""

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np


# Submission header template filled with team metadata.
DEFAULT_HEADER = [
    "# Bertucci Samuele, Birardi Leonardo",
    "# Team Name: It Depends",
    "# ML-CUP25",
    "# Submission Date: 07/01/2026",
]


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_predictions_csv(
    path: str | Path,
    ids: Iterable[int],
    preds: np.ndarray,
    *,
    header_lines: Iterable[str] | None = None,
    float_format: str = "{:.10f}",
) -> Path:
    """Write ML-CUP predictions with an optional header block."""
    path = ensure_parent_dir(path)
    preds = np.asarray(preds, dtype=float)
    if preds.ndim != 2 or preds.shape[1] != 4:
        raise ValueError(f"Predictions must be (N, 4). Got {preds.shape}.")

    # Build the output lines, starting with optional header entries.
    lines: list[str] = []
    if header_lines:
        for line in header_lines:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("#"):
                line = f"# {line}"
            lines.append(line)

    # Append one row per sample in the required CSV format.
    ids_list = list(ids)
    if len(ids_list) != preds.shape[0]:
        raise ValueError("Number of ids does not match number of predictions.")
    for sample_id, row in zip(ids_list, preds):
        row_fmt = ",".join(float_format.format(v) for v in row)
        lines.append(f"{int(sample_id)},{row_fmt}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_artifacts(
    path: str | Path,
    *,
    model,
    input_dim: int,
    output_dim: int,
    params: dict,
) -> Path:
    """Serialize a trained model and its metadata to a joblib file."""
    path = ensure_parent_dir(path)
    payload = {
        "model": model,
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "params": dict(params),
    }
    joblib.dump(payload, path)
    return path
