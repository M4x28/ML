from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from lightning import LightningModule

from cup_metrics import mean_euclidean_error


def _activation_from_name(name: str) -> nn.Module:
    """Return an activation module from a string name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_sizes: Iterable[int],
    activation: str,
    dropout: float,
) -> nn.Module:
    """Construct an MLP with optional dropout between layers."""
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for size in hidden_sizes:
        size = int(size)
        layers.append(nn.Linear(prev_dim, size))
        layers.append(_activation_from_name(activation))
        if dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        prev_dim = size
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class CupMLPModel(LightningModule):
    """
    MLP regression model for ML-CUP with MSE loss and MEE metrics.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int = 4,
        hidden_sizes: Iterable[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        momentum: float = 0.9,
    ) -> None:
        """Initialize the MLP and its optimization hyperparameters."""
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = (64, 64)
        self.save_hyperparameters()
        self.net = _build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the MLP."""
        return self.net(x)

    def _compute_loss_and_metrics(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Compute loss/MSE/MEE and log metrics for the given stage."""
        x, y = batch
        y_hat = self.forward(x)
        mse = self.loss_fn(y_hat, y)
        mee = mean_euclidean_error(y_hat, y)
        loss = mse
        batch_size = x.size(0)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_mse", mse, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_mee", mee, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning hook for training step."""
        return self._compute_loss_and_metrics(batch, stage="train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Lightning hook for validation step."""
        self._compute_loss_and_metrics(batch, stage="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Lightning hook for test step."""
        self._compute_loss_and_metrics(batch, stage="test")

    def predict_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Lightning hook for prediction; returns raw outputs."""
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self.forward(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build the optimizer selected by hyperparameters."""
        optimizer = str(self.hparams.optimizer).lower()
        if optimizer == "sgd":
            return torch.optim.SGD(
                self.parameters(),
                lr=float(self.hparams.lr),
                momentum=float(self.hparams.momentum),
                weight_decay=float(self.hparams.weight_decay),
            )
        if optimizer == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=float(self.hparams.lr),
                weight_decay=float(self.hparams.weight_decay),
            )
        raise ValueError(f"Unsupported optimizer: {optimizer}")
