from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ensemble_io import read_cup_test, DEFAULT_HEADER, write_predictions_csv
from ensemble_models import ModelCache, ModelSpec


def _find_repo_root(start: Path) -> Path:
    """
    Repo root = the first parent that contains data/ML-CUP25-TS.csv (or TR).
    """
    for candidate in (start, *start.parents):
        if (candidate / "data" / "ML-CUP25-TS.csv").exists():
            return candidate
        if (candidate / "data" / "ML-CUP25-TR.csv").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())


# Use `write_predictions_csv` from `ensemble_io` (which reuses `common.cup_io`) to
# avoid duplicated I/O implementations.


def _resolve_path(base_dir: Path, value: str) -> Path:
    """
    Resolve a possibly-relative path:
    1) absolute stays absolute
    2) relative to base_dir (registry folder)
    3) relative to REPO_ROOT
    """
    p = Path(value)
    if p.is_absolute():
        return p
    cand = (base_dir / p).resolve()
    if cand.exists():
        return cand
    alt = (REPO_ROOT / p).resolve()
    return alt if alt.exists() else cand


def _resolve_input_path(user_value: str) -> Path:
    """
    Resolve an input path given on CLI (registry / ts-path):
    - if exists as given (relative to CWD), ok
    - else try relative to REPO_ROOT
    """
    p = Path(user_value)
    if p.exists():
        return p.resolve()
    alt = (REPO_ROOT / p).resolve()
    if alt.exists():
        return alt
    # last attempt: resolve anyway to show a meaningful absolute path in the error
    return p.resolve()


def _load_candidates(payload: dict) -> list[dict]:
    """
    Support both registry schemas:
      A) {"candidates":[{enabled,target_idx,model_type,path,...}, ...]}
      B) {"models":[{target_idx,model_type,path,id?...}, ...]}
    Returns a normalized list of dicts with keys:
      enabled, target_idx, model_type, path
    """
    if isinstance(payload.get("candidates"), list):
        out: list[dict] = []
        for c in payload["candidates"]:
            if not isinstance(c, dict):
                continue
            out.append(c)
        return out

    if isinstance(payload.get("models"), list):
        out = []
        for m in payload["models"]:
            if not isinstance(m, dict):
                continue
            if "target_idx" not in m or "model_type" not in m or "path" not in m:
                continue
            out.append(
                {
                    "enabled": True,
                    "target_idx": int(m["target_idx"]),
                    "model_type": str(m["model_type"]),
                    "path": str(m["path"]),
                    "id": str(m.get("id", "")),
                }
            )
        return out

    raise ValueError("Registry must contain either a list 'candidates' or a list 'models'.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=str, required=True)
    ap.add_argument("--ts-path", type=str, default=str(REPO_ROOT / "data" / "ML-CUP25-TS.csv"))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # --- registry ---
    registry_path = _resolve_input_path(args.registry)
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    reg_dir = registry_path.parent
    payload = json.loads(registry_path.read_text(encoding="utf-8"))

    candidates = _load_candidates(payload)

    # --- pick one per target ---
    by_target: dict[int, dict] = {}
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if not c.get("enabled", True):
            continue

        t = c.get("target_idx", c.get("target"))
        if t is None:
            continue
        t = int(t)

        model_type = c.get("model_type")
        path_value = c.get("path")
        if not model_type or not path_value:
            continue

        by_target[t] = c

    for t in [0, 1, 2, 3]:
        if t not in by_target:
            raise ValueError(f"Missing candidate for target {t} in registry.")

    # --- TS ---
    ts_path = _resolve_input_path(args.ts_path)
    if not ts_path.exists():
        raise FileNotFoundError(f"TS file not found: {ts_path}")

    ids, X = read_cup_test(ts_path)

    if args.debug:
        print(f"[DEBUG] REPO_ROOT   = {REPO_ROOT}")
        print(f"[DEBUG] registry   = {registry_path}")
        print(f"[DEBUG] ts_path    = {ts_path}")

    # --- predict (NO post-processing) ---
    cache = ModelCache()
    preds_cols: list[np.ndarray] = []

    for t in [0, 1, 2, 3]:
        c = by_target[t]
        spec = ModelSpec(
            model_type=str(c["model_type"]),
            path=_resolve_path(reg_dir, str(c["path"])),
            target_idx=t,
        )
        pred_t = cache.predict_target(spec, X).reshape(-1)
        preds_cols.append(pred_t)

        if args.debug:
            print(f"[DEBUG] t{t} {spec.model_type} -> {spec.path}")

    preds = np.column_stack(preds_cols).astype(float)

    header = None if args.no_header else DEFAULT_HEADER
    out_path = _write_predictions_csv_raw(args.out, ids, preds, header_lines=header)
    print(f"[OK] Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
