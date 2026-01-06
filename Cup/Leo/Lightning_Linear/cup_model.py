from __future__ import annotations

import torch
import torch.nn as nn
from lightning import LightningModule

from cup_metrics import mean_euclidean_error, mse_per_instance


class CupLinearModel(LightningModule):
    """
    Linear regression model for ML-CUP with optional L2/Tikhonov regularization.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int = 4,
        lr: float = 1e-3,
        l2_reg: float = 0.0,
    ) -> None:
        """Initialize the linear layer and store hyperparameters."""
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the linear layer."""
        return self.linear(x)

    def _compute_loss_and_metrics(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """Compute loss/MSE/MEE for a batch and log them."""
        x, y = batch
        y_hat = self.forward(x)

        base_loss = mse_per_instance(y_hat, y)
        # L2 penalty uses only weights (bias excluded).
        l2_penalty = self.hparams.l2_reg * torch.sum(self.linear.weight ** 2)
        loss = base_loss + l2_penalty

        mee = mean_euclidean_error(y_hat, y)
        batch_size = x.size(0)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        if stage in ("train", "val", "test"):
            # Log metrics that are tracked for model selection and curves.
            self.log(
                f"{stage}_mse",
                base_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_mee",
                mee,
                prog_bar=stage in ("val", "test"),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning hook for the training step."""
        return self._compute_loss_and_metrics(batch, stage="train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Lightning hook for the validation step."""
        self._compute_loss_and_metrics(batch, stage="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Lightning hook for the test step."""
        self._compute_loss_and_metrics(batch, stage="test")

    def predict_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Lightning hook for prediction; returns raw model outputs."""
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self.forward(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the optimizer used for training."""
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
