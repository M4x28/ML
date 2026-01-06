from __future__ import annotations

"""Select and export the best MLP models per target."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shutil

LEO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    target_idx: int
    train_mee: float | None
    val_mee: float | None
    test_mee: float | None
    exported_model: Path | None
    source: str


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _candidate_from_metrics(
    *,
    target_idx: int,
    metrics: dict,
    exported_model: str | None,
    source: str,
) -> Candidate:
    return Candidate(
        target_idx=target_idx,
        train_mee=_as_float(metrics.get("train", {}).get("mee")),
        val_mee=_as_float(metrics.get("val", {}).get("mee")),
        test_mee=_as_float(metrics.get("test", {}).get("mee")),
        exported_model=Path(exported_model) if exported_model else None,
        source=source,
    )


def _parse_final_summary(payload: dict, source: str) -> list[Candidate]:
    results: list[Candidate] = []
    targets = payload.get("per_target_best_targets")
    if not isinstance(targets, list):
        return results
    for entry in targets:
        if not isinstance(entry, dict):
            continue
        target_idx = entry.get("target_idx")
        if target_idx is None:
            continue
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        results.append(
            _candidate_from_metrics(
                target_idx=int(target_idx),
                metrics=metrics,
                exported_model=entry.get("exported_model"),
                source=source,
            )
        )
    return results


def _parse_target_runs(payload: dict, source: str) -> list[Candidate]:
    results: list[Candidate] = []
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        return results
    for target_key, models in targets.items():
        try:
            target_idx = int(target_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, dict):
                continue
            metrics = model.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            results.append(
                _candidate_from_metrics(
                    target_idx=target_idx,
                    metrics=metrics,
                    exported_model=model.get("exported_model"),
                    source=source,
                )
            )
    return results


def load_candidates(results_dir: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in results_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        candidates.extend(_parse_final_summary(payload, path.name))
        candidates.extend(_parse_target_runs(payload, path.name))
    return candidates


def select_best(
    candidates: list[Candidate], *, require_exported: bool
) -> dict[int, Candidate]:
    best: dict[int, Candidate] = {}
    for candidate in candidates:
        if candidate.test_mee is None:
            continue
        if require_exported and candidate.exported_model is None:
            continue
        # Keep the lowest test MEE per target.
        current = best.get(candidate.target_idx)
        if current is None or candidate.test_mee < (current.test_mee or float("inf")):
            best[candidate.target_idx] = candidate
    return best


def export_models(selected: dict[int, Candidate], output_dir: Path) -> dict[int, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[int, str | None] = {}
    for target_idx, candidate in selected.items():
        src = candidate.exported_model
        if src is None or not src.exists():
            outputs[target_idx] = None
            continue
        # Copy to a stable per-target filename for ensemble configs.
        dst = output_dir / f"mlp_target{target_idx}.pt"
        shutil.copy2(src, dst)
        outputs[target_idx] = str(dst)
    return outputs


def write_summary(
    path: Path,
    *,
    best_all: dict[int, Candidate],
    best_exportable: dict[int, Candidate],
    exported_paths: dict[int, str | None],
) -> None:
    payload = {
        "best_all": {
            str(idx): {
                "train_mee": cand.train_mee,
                "val_mee": cand.val_mee,
                "test_mee": cand.test_mee,
                "source": cand.source,
                "exported_model": str(cand.exported_model) if cand.exported_model else None,
            }
            for idx, cand in best_all.items()
        },
        "best_exportable": {
            str(idx): {
                "train_mee": cand.train_mee,
                "val_mee": cand.val_mee,
                "test_mee": cand.test_mee,
                "source": cand.source,
                "exported_model": str(cand.exported_model) if cand.exported_model else None,
                "exported_path": exported_paths.get(idx),
            }
            for idx, cand in best_exportable.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MLP models from results JSONs.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(LEO_ROOT / "MLP" / "results"),
        help="Directory with MLP results JSONs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(LEO_ROOT / "Ensemble" / "models" / "mlp"),
        help="Output directory for exported MLP models.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(LEO_ROOT / "Ensemble" / "models" / "mlp" / "selection_summary.json"),
        help="Path for the selection summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    candidates = load_candidates(results_dir)
    best_all = select_best(candidates, require_exported=False)
    best_exportable = select_best(candidates, require_exported=True)
    output_dir = Path(args.output_dir).resolve()
    exported_paths = export_models(best_exportable, output_dir)
    summary_path = Path(args.summary_path).resolve()
    write_summary(
        summary_path,
        best_all=best_all,
        best_exportable=best_exportable,
        exported_paths=exported_paths,
    )
    print(f"Best MLP (all): {sorted(best_all)}")
    print(f"Best MLP (exportable): {sorted(best_exportable)}")
    print(f"Export dir: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
