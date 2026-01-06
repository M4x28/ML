from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


DEFAULT_HEADER = [
    "# Name1 Surname1, Name2 Surname2, Name3 Surname3",
    "# Team Name",
    "# ML-CUP25",
    "# Submission Date (e.g. 20/01/2026)",
]


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory exists and return the Path."""
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
    """
    Write predictions in the ML-CUP format: ID, y1, y2, y3, y4.
    """
    path = ensure_parent_dir(path)
    preds = np.asarray(preds, dtype=float)
    if preds.ndim != 2 or preds.shape[1] != 4:
        raise ValueError(f"Predictions must be (N, 4). Got {preds.shape}.")

    lines: list[str] = []
    if header_lines:
        # Normalize header lines to start with '#'.
        for line in header_lines:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("#"):
                line = f"# {line}"
            lines.append(line)

    ids_list = list(ids)
    if len(ids_list) != preds.shape[0]:
        raise ValueError("Number of ids does not match number of predictions.")
    for sample_id, row in zip(ids_list, preds):
        # Preserve sample ID in the first column.
        row_fmt = ",".join(float_format.format(v) for v in row)
        lines.append(f"{int(sample_id)},{row_fmt}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_artifacts(
    path: str | Path,
    *,
    model: torch.nn.Module,
    input_dim: int,
    output_dim: int,
    scaler: object | None,
    params: dict[str, object],
) -> Path:
    """
    Save model weights and preprocessing artifacts.
    """
    path = ensure_parent_dir(path)
    # Store tensors on CPU for portability.
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "scaler": scaler,
        "params": dict(params),
    }
    torch.save(payload, path)
    return path
