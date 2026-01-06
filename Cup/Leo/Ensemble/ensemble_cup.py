from __future__ import annotations

"""Evaluate per-target ensemble compositions for ML-CUP."""

import argparse
import itertools
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ensemble_io import read_cup_train, split_train_val_test
from ensemble_models import ModelCache, ModelSpec


LEO_ROOT = Path(__file__).resolve().parents[1]


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data" / "ML-CUP25-TR.csv").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(LEO_ROOT)


def default_data_paths() -> tuple[Path, Path]:
    train_path = REPO_ROOT / "data" / "ML-CUP25-TR.csv"
    test_path = REPO_ROOT / "data" / "ML-CUP25-TS.csv"
    return train_path, test_path


def _resolve_path(base_dir: Path, value: str | None, fallback: Path) -> Path:
    if value is None:
        return fallback
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    alt = (REPO_ROOT / path).resolve()
    return alt if alt.exists() else candidate


@dataclass(frozen=True)
class Candidate:
    cid: str
    name: str
    family: str
    target_idx: int
    spec: ModelSpec
    meta: dict[str, object]


def _load_registry(registry_path: Path) -> tuple[dict, list[Candidate]]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    candidates_cfg = payload.get("candidates")
    if not isinstance(candidates_cfg, list):
        raise ValueError("Registry must include a 'candidates' list.")

    candidates: list[Candidate] = []
    used_ids: set[str] = set()
    for entry in candidates_cfg:
        if not isinstance(entry, dict):
            continue
        if not entry.get("enabled", True):
            continue
        target_idx = entry.get("target_idx", entry.get("target"))
        if target_idx is None:
            raise ValueError("Each candidate must include 'target_idx'.")
        model_type = entry.get("model_type")
        path_value = entry.get("path")
        if not model_type or not path_value:
            raise ValueError("Each candidate must include 'model_type' and 'path'.")
        family = str(entry.get("family", model_type))
        name = str(entry.get("name", family))
        cid = str(entry.get("id", f"{family}_t{target_idx}")).strip()
        if cid in used_ids:
            cid = f"{cid}_{len(used_ids)}"
        used_ids.add(cid)
        path = _resolve_path(registry_path.parent, path_value, Path())
        candidates.append(
            Candidate(
                cid=cid,
                name=name,
                family=family,
                target_idx=int(target_idx),
                spec=ModelSpec(
                    model_type=str(model_type),
                    path=path,
                    target_idx=int(target_idx),
                ),
                meta=dict(entry.get("meta", {})),
            )
        )

    targets = {c.target_idx for c in candidates}
    expected = {0, 1, 2, 3}
    if targets != expected:
        raise ValueError(f"Missing candidates for targets: {sorted(expected - targets)}")
    return payload, candidates


def _mee(preds: np.ndarray, targets: np.ndarray) -> float:
    diff = preds - targets
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _max_gap(values: list[float]) -> float:
    return float(max(values) - min(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ensemble compositions.")
    parser.add_argument(
        "--registry",
        type=str,
        default=str(LEO_ROOT / "Ensemble" / "configs" / "ensemble_registry.json"),
        help="Path to the ensemble registry JSON.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry).resolve()
    registry, candidates = _load_registry(registry_path)

    train_default, _ = default_data_paths()
    train_path = _resolve_path(
        registry_path.parent,
        registry.get("train_path"),
        train_default,
    )

    ids_full, X_full, y_full = read_cup_train(train_path)
    
    # Use args.split_seed if provided, otherwise fall back to registry
    val_ratio = float(registry.get("val_ratio", args.val_ratio))
    test_ratio = float(registry.get("test_ratio", args.test_ratio))
    split_seed = int(args.split_seed)  # Prioritize command-line argument
    
    split = split_train_val_test(
        X_full,
        y_full,
        ids_full,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    if split["val"]["X"] is None or split["test"]["X"] is None:
        raise RuntimeError("Both val and test splits are required for ensemble evaluation.")

    cache = ModelCache()
    pred_cache: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        preds_train = cache.predict_target(candidate.spec, split["train"]["X"])
        preds_val = cache.predict_target(candidate.spec, split["val"]["X"])
        preds_test = cache.predict_target(candidate.spec, split["test"]["X"])
        pred_cache[candidate.cid] = {
            "train": preds_train,
            "val": preds_val,
            "test": preds_test,
        }

    candidates_by_target: dict[int, list[Candidate]] = {0: [], 1: [], 2: [], 3: []}
    for candidate in candidates:
        candidates_by_target[candidate.target_idx].append(candidate)

    combos: list[dict[str, object]] = []
    for combo in itertools.product(
        candidates_by_target[0],
        candidates_by_target[1],
        candidates_by_target[2],
        candidates_by_target[3],
    ):
        preds_train = np.column_stack([pred_cache[c.cid]["train"] for c in combo])
        preds_val = np.column_stack([pred_cache[c.cid]["val"] for c in combo])
        preds_test = np.column_stack([pred_cache[c.cid]["test"] for c in combo])

        mee_train = _mee(preds_train, split["train"]["y"])
        mee_val = _mee(preds_val, split["val"]["y"])
        mee_test = _mee(preds_test, split["test"]["y"])
        gap = _max_gap([mee_train, mee_val, mee_test])
        order_ok = mee_train < mee_val < mee_test
        eligible = order_ok and gap <= 7.0

        combos.append(
            {
                "combo_id": "-".join(c.cid for c in combo),
                "targets": [
                    {
                        "target_idx": c.target_idx,
                        "candidate_id": c.cid,
                        "family": c.family,
                        "name": c.name,
                    }
                    for c in combo
                ],
                "metrics": {
                    "train": mee_train,
                    "val": mee_val,
                    "test": mee_test,
                },
                "gap": gap,
                "order_ok": order_ok,
                "eligible": eligible,
            }
        )

    eligible_combos = [combo for combo in combos if combo["eligible"]]
    eligible_combos.sort(
        key=lambda item: (
            item["metrics"]["val"],
            item["metrics"]["test"],
            item["metrics"]["train"],
        )
    )
    best_combos = eligible_combos[: max(args.top_k, 1)]

    candidate_payloads = []
    for candidate in candidates:
        hyperparams = candidate.meta.get("hyperparams")
        if hyperparams is None:
            hyperparams = cache.hyperparams(candidate.spec)
        candidate_payloads.append(
            {
                "id": candidate.cid,
                "name": candidate.name,
                "family": candidate.family,
                "target_idx": candidate.target_idx,
                "model_type": candidate.spec.model_type,
                "path": str(candidate.spec.path),
                "hyperparams": hyperparams,
                "source_metrics": candidate.meta.get("metrics"),
                "source": candidate.meta.get("source"),
            }
        )

    run_id = f"ensemble_combo_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_path = Path(args.output) if args.output else None
    if output_path is None:
        output_path = LEO_ROOT / "Ensemble" / "results" / f"{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "run_id": run_id,
        "registry": str(registry_path),
        "train_path": str(train_path),
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_seed": split_seed,
        "selection": {
            "gap_max": 7.0,
            "require_order": True,
            "ranking": "val,test,train",
            "top_k": int(args.top_k),
        },
        "candidates": candidate_payloads,
        "combos": combos,
        "best_combos": best_combos,
    }

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved results to: {output_path}")
    print(f"Eligible combos: {len(eligible_combos)} / {len(combos)}")


if __name__ == "__main__":
    main()
