from __future__ import annotations

"""Predict on the blind test set using a retrained multi-target KNN model."""

import argparse
from pathlib import Path

import joblib
import numpy as np

from cup_data import load_cup_test
from cup_io import DEFAULT_HEADER, write_predictions_csv


def _find_repo_root(start: Path) -> Path:
    """Locate the repo root by searching for the shared data folder."""
    for candidate in (start, *start.parents):
        if (candidate / "data" / "ML-CUP25-TR.csv").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)


def _load_template_header(path: Path) -> list[str]:
    """Read the first four header lines from the template file, if present."""
    if not path.exists():
        return []
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(4):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return lines


def _load_model(path: Path):
    """Load a joblib model payload, falling back to the raw object."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def _predict(model, X: np.ndarray) -> np.ndarray:
    """Predict with a single model or a list of per-target models."""
    if isinstance(model, (list, tuple)):
        # Per-target models return one column each.
        preds = [m.predict(X) for m in model]
        return np.column_stack(preds)
    return model.predict(X)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for blind test prediction."""
    parser = argparse.ArgumentParser(
        description="Predict ML-CUP blind test outputs with a multi-target KNN."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "exports" / "knn_cup_best_retrained.joblib"),
        help="Path to a retrained joblib model.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(REPO_ROOT / "data" / "ML-CUP25-TS.csv"),
        help="Path to ML-CUP25-TS.csv.",
    )
    parser.add_argument(
        "--template-path",
        type=str,
        default=str(REPO_ROOT / "data" / "template-example-with-random-outputs_ML-CUP25-TS.csv"),
        help="Template CSV used to validate header format.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "predictions" / "knn_cup_blind.csv"),
        help="Output predictions CSV path.",
    )
    parser.add_argument("--no-header", action="store_true", help="Skip header lines.")
    parser.add_argument("--no-output", action="store_true", help="Skip writing the CSV.")
    parser.add_argument(
        "--allow-per-target",
        action="store_true",
        help="Allow per-target models (lists) instead of a multi-target model.",
    )
    return parser.parse_args()


def main() -> None:
    """Run prediction on the blind test set and write the submission CSV."""
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    test_path = Path(args.test_path).resolve()

    template_path = Path(args.template_path).resolve()
    template_lines = _load_template_header(template_path)
    if template_path.exists() and len(template_lines) < 4:
        # Ensure we can mimic the expected 4-line header format.
        raise ValueError("Template header is incomplete; expected 4 lines.")

    model = _load_model(model_path)
    if isinstance(model, (list, tuple)) and not args.allow_per_target:
        # Guard against using per-target models when multi-target is required.
        raise ValueError(
            "Per-target models detected. Use a multi-target model or pass --allow-per-target."
        )
    ids, X = load_cup_test(test_path)
    preds = _predict(model, X)

    if args.no_output:
        print("Prediction run complete (--no-output).")
        return

    output_path = Path(args.output).resolve()
    header = [] if args.no_header else DEFAULT_HEADER
    write_predictions_csv(output_path, ids, preds, header_lines=header)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
