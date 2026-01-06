"""
Generate MSE-vs-Epoch plots for MLP_Model_Monk runs.
Targets are inferred from existing mse_curves filenames.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl

from MLP_lightning import (  # noqa: E402
    MonkDataModule,
    MLPClassifier,
    eval_loader_metrics,
    get_accelerator,
    load_monk_task,
)

OUTPUT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Target:
    task_id: int
    want_reg: bool
    out_path: Path


class MSECurveCallback(pl.Callback):
    def __init__(
        self,
        *,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        out_path: Path,
        smooth_window: int,
    ) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.out_path = out_path
        self.smooth_window = max(int(smooth_window), 1)
        self.train_mse: list[float] = []
        self.test_mse: list[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        device = next(pl_module.parameters()).device
        was_training = pl_module.training
        train_mse, _ = eval_loader_metrics(pl_module, self.train_loader, device)
        test_mse, _ = eval_loader_metrics(pl_module, self.test_loader, device)
        if was_training:
            pl_module.train()
        self.train_mse.append(float(train_mse))
        self.test_mse.append(float(test_mse))

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.train_mse or not self.test_mse:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        epochs = list(range(1, min(len(self.train_mse), len(self.test_mse)) + 1))
        train_mse = _smooth(self.train_mse, self.smooth_window)
        test_mse = _smooth(self.test_mse, self.smooth_window)
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, train_mse[: len(epochs)], label="train_mse")
        plt.plot(epochs, test_mse[: len(epochs)], label="test_mse")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("MSE vs Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_path)
        plt.close()


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    out: list[float] = []
    q: deque[float] = deque()
    total = 0.0
    for v in values:
        q.append(float(v))
        total += float(v)
        if len(q) > window:
            total -= q.popleft()
        out.append(total / len(q))
    return out


def _is_reg_name(name: str) -> bool:
    lower = name.lower()
    if "no_reg" in lower:
        return False
    return "reg" in lower


def _discover_targets(mse_root: Path) -> list[Target]:
    targets: list[Target] = []
    for task_dir in sorted(mse_root.glob("monk*")):
        task_name = task_dir.name.replace("monk", "")
        if not task_name.isdigit():
            continue
        task_id = int(task_name)
        for mse_file in sorted(task_dir.glob("*_mse_curve.png")):
            want_reg = _is_reg_name(mse_file.stem)
            out_dir = OUTPUT_ROOT / "mse_curves" / f"monk{task_id}"
            targets.append(Target(task_id=task_id, want_reg=want_reg, out_path=out_dir / mse_file.name))
    return targets


def _best_summary_for_task(task_id: int, want_reg: bool) -> tuple[Path, dict] | None:
    summaries = list((OUTPUT_ROOT / "exported_models" / f"monk{task_id}" / "mlp").glob("*/summary.json"))
    best: tuple[tuple[float, float], Path, dict] | None = None
    for summary_path in summaries:
        summary = _load_summary(summary_path)
        weight_decay = float(summary.get("best_params", {}).get("weight_decay", 0.0))
        if want_reg and weight_decay == 0.0:
            continue
        if not want_reg and weight_decay != 0.0:
            continue
        metrics = summary.get("selection", {}).get("metrics", {})
        val_mse = float(metrics.get("val_mse_mean", float("nan")))
        val_acc = float(metrics.get("val_acc_mean", float("nan")))
        if not math.isfinite(val_mse):
            continue
        key = (val_mse, -val_acc if math.isfinite(val_acc) else 0.0)
        if best is None or key < best[0]:
            best = (key, summary_path, summary)
    if best is None:
        return None
    return best[1], best[2]


def _best_seed(summary: dict) -> int:
    per_seed = summary.get("selection", {}).get("per_seed", [])
    best: tuple[tuple[float, float], int] | None = None
    for entry in per_seed:
        val_mse = entry.get("val_mse", float("nan"))
        val_acc = entry.get("val_acc", float("nan"))
        if not math.isfinite(float(val_mse)):
            continue
        key = (float(val_mse), -float(val_acc) if math.isfinite(float(val_acc)) else 0.0)
        seed = int(entry.get("seed", 0))
        if best is None or key < best[0]:
            best = (key, seed)
    if best is None and per_seed:
        return int(per_seed[0].get("seed", 0))
    if best is None:
        return 0
    return best[1]


def _run_target(
    target: Target,
    *,
    force: bool,
    epochs: int,
    smooth_window: int,
    batch_size: int,
) -> None:
    if target.out_path.exists() and not force:
        print(f"[skip] {target.out_path}")
        return

    best = _best_summary_for_task(target.task_id, target.want_reg)
    if best is None:
        print(f"[skip] monk{target.task_id} reg={target.want_reg} (no summary found)")
        return
    summary_path, summary = best
    seed = _best_seed(summary)
    params = summary.get("best_params", {})

    epochs = max(int(epochs), 1)
    batch_size = max(int(batch_size), 1)

    pl.seed_everything(seed, workers=True)
    X_train_full, y_train_full = load_monk_task(target.task_id, split="train")
    X_test, y_test = load_monk_task(target.task_id, split="test")
    data_full = MonkDataModule(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
        batch_size=batch_size,
        num_workers=0,
        use_val=False,
    )
    data_full.setup("fit")
    assert data_full.input_dim is not None

    model = MLPClassifier(
        input_dim=data_full.input_dim,
        hidden_dim=int(params["hidden_dim"]),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    callback = MSECurveCallback(
        train_loader=data_full.train_eval_dataloader(),
        test_loader=data_full.test_dataloader(),
        out_path=target.out_path,
        smooth_window=smooth_window,
    )

    accelerator = get_accelerator()
    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=epochs,
        min_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=[callback],
    )
    print(
        f"[run] monk{target.task_id} reg={target.want_reg} seed={seed} "
        f"epochs={epochs} bs={batch_size} -> {target.out_path.name}"
    )
    trainer.fit(model, datamodule=data_full)
    device = next(model.parameters()).device
    train_mse, train_acc = eval_loader_metrics(model, data_full.train_eval_dataloader(), device)
    test_mse, test_acc = eval_loader_metrics(model, data_full.test_dataloader(), device)
    metrics_path = target.out_path.with_suffix(".json")
    metrics_payload = {
        "task_id": target.task_id,
        "tag": "reg" if target.want_reg else "no_reg",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "params": params,
        "summary_path": str(summary_path),
        "train": {"acc": float(train_acc), "mse": float(train_mse)},
        "test": {"acc": float(test_acc), "mse": float(test_mse)},
        "smooth_window": smooth_window,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"[done] {target.out_path} metrics={metrics_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MSE curves for MLP_Model_Monk.")
    parser.add_argument(
        "--mse-root",
        type=Path,
        default=OUTPUT_ROOT / "mse_curves",
        help="Root directory with existing mse_curves.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate curves even if output already exists.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to use for curve generation.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Moving average window (in epochs) for smoothing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for curve generation.",
    )
    args = parser.parse_args()

    targets = _discover_targets(args.mse_root)
    if not targets:
        print(f"No targets found in {args.mse_root}")
        return 1

    for target in targets:
        _run_target(
            target,
            force=args.force,
            epochs=args.epochs,
            smooth_window=args.smooth_window,
            batch_size=args.batch_size,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
