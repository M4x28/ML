from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

from cup_metrics import evaluate_regression_metrics


class CurveTracker(Callback):
    """Collects loss/MSE/MEE curves at the end of each epoch."""

    def __init__(self) -> None:
        """Initialize empty metric histories for train/test."""
        super().__init__()
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_mse": [],
            "train_mee": [],
            "test_loss": [],
            "test_mse": [],
            "test_mee": [],
        }

    def _append_metrics(self, prefix: str, metrics: dict[str, float]) -> None:
        """Append metrics to the history using the given prefix."""
        self.history[f"{prefix}_loss"].append(float(metrics["loss"]))
        self.history[f"{prefix}_mse"].append(float(metrics["mse"]))
        self.history[f"{prefix}_mee"].append(float(metrics["mee"]))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Evaluate metrics on train/test loaders and store them."""
        data_module = trainer.datamodule
        # Pull l2_reg from hparams when available to match training loss.
        l2_reg = float(getattr(pl_module.hparams, "l2_reg", 0.0))

        # Full-train evaluation for stable curves.
        train_loader = data_module.train_dataloader()
        train_metrics = evaluate_regression_metrics(
            pl_module, train_loader, l2_reg=l2_reg
        )
        self._append_metrics("train", train_metrics)

        test_loader = data_module.test_dataloader()
        if test_loader is None:
            return
        # Test metrics are optional if no test split is configured.
        test_metrics = evaluate_regression_metrics(
            pl_module, test_loader, l2_reg=l2_reg
        )
        self._append_metrics("test", test_metrics)


def _mean_curve(values: list[list[float]]) -> list[float]:
    """Average multiple curves, truncating to the shortest non-empty length."""
    if not values:
        return []
    non_empty = [v for v in values if v]
    if not non_empty:
        return []
    min_len = min(len(v) for v in non_empty)
    if min_len <= 0:
        return []
    # Trim to align different epoch counts before averaging.
    arr = np.asarray([v[:min_len] for v in non_empty], dtype=float)
    return list(np.nanmean(arr, axis=0))


def average_histories(histories: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    """Compute mean curves for each metric across multiple runs."""
    if not histories:
        return {}
    keys = histories[0].keys()
    return {key: _mean_curve([h.get(key, []) for h in histories]) for key in keys}


def _plot_curve(
    path: Path,
    *,
    train_values: list[float],
    test_values: list[float],
    ylabel: str,
    title: str,
) -> None:
    """Save a single curve plot for train/test values."""
    if not train_values and not test_values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use the max length so both series share the same epoch axis.
    epochs = list(range(1, max(len(train_values), len(test_values)) + 1))

    plt.figure(figsize=(7, 4))
    if train_values:
        plt.plot(epochs[: len(train_values)], train_values, label="train")
    if test_values:
        plt.plot(epochs[: len(test_values)], test_values, label="test")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_curve_plots(
    *,
    output_dir: str | Path,
    run_name: str,
    curves: dict[str, list[float]],
    title_prefix: str = "Linear CUP",
) -> dict[str, str]:
    """Write MSE/MEE/Loss curve images and return their paths."""
    output_dir = Path(output_dir)

    # Emit three separate plots to keep reports readable.
    mse_path = output_dir / f"{run_name}_mse_curve.png"
    _plot_curve(
        mse_path,
        train_values=curves.get("train_mse", []),
        test_values=curves.get("test_mse", []),
        ylabel="MSE",
        title=f"{title_prefix} - MSE vs Epochs",
    )

    mee_path = output_dir / f"{run_name}_mee_curve.png"
    _plot_curve(
        mee_path,
        train_values=curves.get("train_mee", []),
        test_values=curves.get("test_mee", []),
        ylabel="MEE",
        title=f"{title_prefix} - MEE vs Epochs",
    )

    loss_path = output_dir / f"{run_name}_loss_curve.png"
    _plot_curve(
        loss_path,
        train_values=curves.get("train_loss", []),
        test_values=curves.get("test_loss", []),
        ylabel="Loss",
        title=f"{title_prefix} - Loss vs Epochs",
    )

    return {
        "mse": str(mse_path),
        "mee": str(mee_path),
        "loss": str(loss_path),
    }
