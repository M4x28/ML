"""
Generate accuracy-vs-epochs plots for best Linear_Model_Monk runs.
Reads summary JSON files in results/ and retrains the best config/seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lightning import Trainer, seed_everything

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from linearMonk import (  # noqa: E402
    AccuracyCurveCallback,
    MonkDataModule,
    MonkLinearModel,
    get_monk_paths,
)

OUTPUT_ROOT = THIS_DIR


def _build_run_name(config: dict, seed: int) -> str:
    return (
        f"randcfg_lr{config['lr']}_bs{config['batch_size']}_ep{config['epochs']}"
        f"_l2{config['l2_reg']}_seed{seed}"
    )


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_entries(summary: dict) -> list[tuple[str, dict]]:
    entries = [("best_overall", summary["best_overall"])]
    for key in ("best_no_reg", "best_with_reg"):
        if key in summary:
            entries.append((key, summary[key]))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate accuracy curves for best Linear_Model_Monk models."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=OUTPUT_ROOT / "results",
        help="Directory with monk*_summary_*.json files.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional single summary JSON to process.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        help="Optional task id filter (1, 2, 3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate curves even if output already exists.",
    )
    args = parser.parse_args()

    if args.summary:
        summary_paths = [args.summary]
    else:
        summary_paths = sorted(args.results_dir.glob("monk*_summary_*.json"))

    if not summary_paths:
        print(f"No summary files found in {args.results_dir}")
        return 1

    seen_runs: set[tuple[int, str]] = set()
    for summary_path in summary_paths:
        summary = _load_summary(summary_path)
        task_id = int(summary.get("task_id", -1))
        if args.task is not None and task_id != args.task:
            continue

        for label, entry in _iter_entries(summary):
            config = entry["config"]
            seed = int(entry["best_seed"]["seed"])
            run_name = _build_run_name(config, seed)
            run_key = (task_id, run_name)
            if run_key in seen_runs:
                continue
            seen_runs.add(run_key)

            out_dir = OUTPUT_ROOT / "accuracy_curves" / f"monk{task_id}"
            out_path = out_dir / f"{run_name}_accuracy_curve.png"
            if out_path.exists() and not args.force:
                print(f"[skip] {out_path}")
                continue

            print(f"[run] monk{task_id} {label} seed={seed} -> {out_path.name}")
            seed_everything(seed)

            train_path, test_path = get_monk_paths(task_id)
            data_module = MonkDataModule(
                train_path=train_path,
                test_path=test_path,
                batch_size=config["batch_size"],
                val_ratio=0.2,
                poly_degree=2,
            )
            data_module.setup("fit")

            model = MonkLinearModel(
                input_dim=data_module.input_dim,
                lr=config["lr"],
                l2_reg=config["l2_reg"],
            )

            trainer = Trainer(
                max_epochs=config["epochs"],
                deterministic=True,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                callbacks=[
                    AccuracyCurveCallback(
                        output_dir=str(out_dir),
                        run_name=run_name,
                    )
                ],
            )
            trainer.fit(model, datamodule=data_module)
            print(f"[done] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
