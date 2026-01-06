from __future__ import annotations

import torch


def mse_per_instance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error averaged over instances (sum over outputs per sample).
    """
    diff = pred - target
    if diff.ndim == 1:
        per_sample = diff ** 2
    else:
        # Sum over output dimension to get per-sample squared error.
        per_sample = torch.sum(diff ** 2, dim=1)
    return torch.mean(per_sample)


def mean_euclidean_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Euclidean Error (MEE) on the last dimension.
    """
    if pred.ndim == 1:
        return torch.mean(torch.abs(pred - target))
    return torch.mean(torch.norm(pred - target, dim=1))


def _l2_penalty(model: torch.nn.Module) -> torch.Tensor:
    """Compute L2 penalty over non-bias parameters."""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias"):
            continue
        # Add squared L2 norm for each weight tensor.
        total = total + torch.sum(param ** 2)
    return total


@torch.no_grad()
def evaluate_regression_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader | None,
    *,
    l2_reg: float = 0.0,
) -> dict[str, float]:
    """Evaluate MSE/MEE and a loss that includes optional L2 penalty."""
    if dataloader is None:
        return {
            "loss": float("nan"),
            "mse": float("nan"),
            "mee": float("nan"),
        }

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    mse_sum = 0.0
    mee_sum = 0.0
    sample_count = 0

    for batch in dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        diff = pred - y
        if diff.ndim == 1:
            per_sample = diff ** 2
        else:
            per_sample = torch.sum(diff ** 2, dim=1)
        mse_sum += float(torch.sum(per_sample).item())

        if diff.ndim == 1:
            errors = torch.abs(diff)
        else:
            errors = torch.norm(diff, dim=1)
        # Accumulate per-sample Euclidean errors.
        mee_sum += float(torch.sum(errors).item())
        sample_count += int(errors.numel())

    if was_training:
        model.train()

    mse = mse_sum / sample_count if sample_count > 0 else float("nan")
    mee = mee_sum / sample_count if sample_count > 0 else float("nan")
    l2_penalty = float(_l2_penalty(model).detach().cpu().item())
    # Keep loss consistent with training: MSE + L2 penalty.
    loss = mse + float(l2_reg) * l2_penalty

    return {"loss": loss, "mse": mse, "mee": mee}
