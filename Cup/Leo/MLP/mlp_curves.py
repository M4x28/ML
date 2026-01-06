from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

from cup_metrics import evaluate_regression_metrics

matplotlib.use("Agg")


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class CurveTracker(Callback):
    """
    Tracks train/test MSE and MEE at the end of each epoch.
    """

    def __init__(
        self,
        *,
        train_loader,
        test_loader,
        l2_reg: float = 0.0,
    ) -> None:
        """Initialize curve tracking with evaluation loaders."""
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.l2_reg = float(l2_reg)
        self.history: dict[str, list[float]] = {
            "epoch": [],
            "train_mse": [],
            "train_mee": [],
            "test_mse": [],
            "test_mee": [],
        }

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Compute metrics on train/test and store them."""
        epoch = trainer.current_epoch + 1
        train_metrics = evaluate_regression_metrics(
            pl_module,
            self.train_loader,
            l2_reg=self.l2_reg,
        )
        test_metrics = None
        if self.test_loader is not None:
            test_metrics = evaluate_regression_metrics(
                pl_module,
                self.test_loader,
                l2_reg=self.l2_reg,
            )

        # Keep history arrays aligned by epoch.
        self.history["epoch"].append(float(epoch))
        self.history["train_mse"].append(float(train_metrics.get("mse", float("nan"))))
        self.history["train_mee"].append(float(train_metrics.get("mee", float("nan"))))
        if test_metrics is not None:
            self.history["test_mse"].append(float(test_metrics.get("mse", float("nan"))))
            self.history["test_mee"].append(float(test_metrics.get("mee", float("nan"))))


def _plot_curve(
    *,
    epochs: Iterable[float],
    train_values: Iterable[float],
    test_values: Iterable[float] | None,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    """Plot train/test curves and save to disk."""
    plt.figure(figsize=(7.0, 4.0))
    epochs_list = list(epochs)
    train_list = list(train_values)
    test_list = list(test_values) if test_values is not None else []
    plt.plot(epochs_list, train_list, label="train")
    if test_list:
        plt.plot(epochs_list, test_list, label="test")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_mse_curve(
    *,
    history: dict[str, list[float]],
    output_dir: str | Path,
    run_name: str,
    title: str,
) -> Path:
    """Save the MSE curve as a PNG and return its path."""
    output_dir = ensure_dir(output_dir)
    epochs = history.get("epoch", [])
    train_values = history.get("train_mse", [])
    test_values = history.get("test_mse", [])
    return _plot_curve(
        epochs=epochs,
        train_values=train_values,
        test_values=test_values,
        title=title,
        ylabel="mse",
        output_path=output_dir / f"{run_name}_mse_curve.png",
    )


def save_mee_curve(
    *,
    history: dict[str, list[float]],
    output_dir: str | Path,
    run_name: str,
    title: str,
) -> Path:
    """Save the MEE curve as a PNG and return its path."""
    output_dir = ensure_dir(output_dir)
    epochs = history.get("epoch", [])
    train_values = history.get("train_mee", [])
    test_values = history.get("test_mee", [])
    return _plot_curve(
        epochs=epochs,
        train_values=train_values,
        test_values=test_values,
        title=title,
        ylabel="mee",
        output_path=output_dir / f"{run_name}_mee_curve.png",
    )
