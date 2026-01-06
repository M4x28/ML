from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cup_data import CupDataModule


def _activation_from_name(name: str) -> nn.Module:
    """Return an activation module from a string name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported activation: {name}")


class CupAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Iterable[int] = (64, 32),
        activation: str = "relu",
    ) -> None:
        """Build a symmetric MLP autoencoder for feature compression."""
        super().__init__()
        hidden = [int(h) for h in hidden_dims]
        if not hidden:
            raise ValueError("hidden_dims cannot be empty")
        act = _activation_from_name(activation)
        # Encoder: input -> hidden layers -> latent vector.
        encoder_layers: list[nn.Module] = []
        prev = int(input_dim)
        for size in hidden:
            encoder_layers.append(nn.Linear(prev, size))
            encoder_layers.append(act)
            prev = size
        encoder_layers.append(nn.Linear(prev, int(latent_dim)))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: latent vector -> reversed hidden layers -> input size.
        decoder_layers: list[nn.Module] = []
        prev = int(latent_dim)
        for size in reversed(hidden):
            decoder_layers.append(nn.Linear(prev, size))
            decoder_layers.append(act)
            prev = size
        decoder_layers.append(nn.Linear(prev, int(input_dim)))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input by encoding then decoding."""
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(x)


@dataclass(frozen=True)
class AutoencoderConfig:
    """Configuration for autoencoder training and latent search."""
    latent_dim: int = 8
    latent_dims: tuple[int, ...] | None = None
    hidden_dims: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 700
    patience: int = 30
    batch_size: int = 500
    seed: int = 0


def _make_feature_loader(
    x: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a DataLoader for feature tensors with optional shuffling."""
    generator = None
    if shuffle:
        generator = torch.Generator()
        # Deterministic shuffling for repeatable results.
        generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def train_autoencoder(
    *,
    train_x: torch.Tensor,
    val_x: torch.Tensor | None,
    config: AutoencoderConfig,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
) -> tuple[CupAutoEncoder, dict[str, float]]:
    """Train an autoencoder with early stopping on validation loss."""
    model = CupAutoEncoder(
        input_dim=int(train_x.shape[1]),
        latent_dim=int(config.latent_dim),
        hidden_dims=config.hidden_dims,
        activation=config.activation,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = nn.MSELoss()

    # Build loaders from raw features only.
    train_loader = _make_feature_loader(
        train_x,
        batch_size=int(config.batch_size),
        shuffle=True,
        seed=int(config.seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_x is not None:
        val_loader = _make_feature_loader(
            val_x,
            batch_size=int(config.batch_size),
            shuffle=False,
            seed=int(config.seed),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(int(config.epochs)):
        model.train()
        total_loss = 0.0
        seen = 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch_x)
            loss = loss_fn(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.size(0)
            seen += int(batch_x.size(0))

        if val_loader is None:
            continue

        model.eval()
        with torch.no_grad():
            # Validation reconstruction loss for early stopping.
            val_loss = 0.0
            val_seen = 0
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon = model(batch_x)
                loss = loss_fn(recon, batch_x)
                val_loss += float(loss.item()) * batch_x.size(0)
                val_seen += int(batch_x.size(0))
            val_loss = val_loss / max(val_seen, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            # Stop when patience is exhausted.
            if int(config.patience) > 0 and wait >= int(config.patience):
                break

    if val_loader is None:
        best_val = float("nan")
        best_epoch = int(config.epochs)
    elif best_state is not None:
        # Restore the best validation checkpoint.
        model.load_state_dict(best_state)

    metrics = {
        "best_val_loss": float(best_val),
        "best_epoch": float(best_epoch),
    }
    return model, metrics


def _resolve_latent_dims(config: AutoencoderConfig) -> list[int]:
    """Resolve latent dimensions list from config."""
    if config.latent_dims:
        dims = [int(d) for d in config.latent_dims if int(d) > 0]
        return sorted(set(dims))
    return [int(config.latent_dim)]


def train_autoencoder_with_search(
    *,
    train_x: torch.Tensor,
    val_x: torch.Tensor | None,
    config: AutoencoderConfig,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
) -> tuple[CupAutoEncoder, dict[str, float]]:
    """Train autoencoders for multiple latent sizes and pick the best."""
    latent_dims = _resolve_latent_dims(config)
    if val_x is None and len(latent_dims) > 1:
        latent_dims = latent_dims[:1]

    best_model: CupAutoEncoder | None = None
    best_metrics: dict[str, float] | None = None
    candidates: list[dict[str, float]] = []

    for latent_dim in latent_dims:
        # Train one candidate model for this latent size.
        local_config = replace(config, latent_dim=int(latent_dim), latent_dims=None)
        model, metrics = train_autoencoder(
            train_x=train_x,
            val_x=val_x,
            config=local_config,
            device=device,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        candidate = {
            "latent_dim": float(latent_dim),
            "best_val_loss": float(metrics.get("best_val_loss", float("nan"))),
            "best_epoch": float(metrics.get("best_epoch", float("nan"))),
        }
        candidates.append(candidate)

        current = candidate["best_val_loss"]
        if best_metrics is None or current < float(best_metrics.get("best_val_loss", float("inf"))):
            best_metrics = metrics
            best_model = model

    if best_model is None:
        raise RuntimeError("Autoencoder search failed to produce a model.")

    best_latent = int(candidates[0]["latent_dim"]) if candidates else int(config.latent_dim)
    if best_metrics is not None:
        best_latent = int(
            min(candidates, key=lambda c: c.get("best_val_loss", float("inf"))).get("latent_dim", best_latent)
        )

    summary = {
        "best_val_loss": float(best_metrics.get("best_val_loss", float("nan")) if best_metrics else float("nan")),
        "best_epoch": float(best_metrics.get("best_epoch", float("nan")) if best_metrics else float("nan")),
        "best_latent_dim": float(best_latent),
        "candidates": candidates,
    }
    if val_x is None:
        summary["note"] = "val set missing; latent_dim search skipped"
    return best_model, summary


def _encode_tensor(
    encoder: CupAutoEncoder,
    x: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
) -> torch.Tensor:
    """Encode a tensor in batches using the provided encoder."""
    loader = _make_feature_loader(
        x,
        batch_size=int(batch_size),
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    encoder.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z = encoder.encode(batch_x).cpu()
            chunks.append(z)
    if not chunks:
        # Return an empty tensor with correct feature dimension.
        return torch.empty((0, int(encoder.encoder[-1].out_features)), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def _encode_dataset(
    dataset: TensorDataset | None,
    *,
    encoder: CupAutoEncoder,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
) -> TensorDataset | None:
    """Encode the first tensor of a dataset, preserving extra tensors."""
    if dataset is None:
        return None
    tensors = dataset.tensors
    x = tensors[0]
    z = _encode_tensor(
        encoder,
        x,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if len(tensors) == 1:
        return TensorDataset(z)
    # Keep targets/metadata tensors aligned with encoded features.
    return TensorDataset(z, *tensors[1:])


def apply_autoencoder_to_datamodule(
    data_module: CupDataModule,
    *,
    config: AutoencoderConfig,
    device: torch.device,
) -> dict[str, float]:
    """Train an autoencoder and replace datasets with latent features."""
    if data_module.train_dataset is None:
        raise RuntimeError("DataModule must be setup before applying autoencoder.")
    if data_module.input_dim is None:
        raise RuntimeError("DataModule input_dim is missing.")

    train_x = data_module.train_dataset.tensors[0]
    val_x = None
    if data_module.val_dataset is not None:
        val_x = data_module.val_dataset.tensors[0]

    latent_dims = _resolve_latent_dims(config)
    if len(latent_dims) > 1:
        # Search over multiple latent sizes when validation is available.
        ae, metrics = train_autoencoder_with_search(
            train_x=train_x,
            val_x=val_x,
            config=config,
            device=device,
            num_workers=int(data_module.num_workers),
            pin_memory=bool(data_module.pin_memory),
        )
        config = replace(
            config,
            latent_dim=int(metrics.get("best_latent_dim", config.latent_dim)),
            latent_dims=None,
        )
    else:
        ae, metrics = train_autoencoder(
            train_x=train_x,
            val_x=val_x,
            config=config,
            device=device,
            num_workers=int(data_module.num_workers),
            pin_memory=bool(data_module.pin_memory),
        )

    # Encode each split in-place to use latent representations.
    data_module.train_dataset = _encode_dataset(
        data_module.train_dataset,
        encoder=ae,
        batch_size=config.batch_size,
        device=device,
        num_workers=int(data_module.num_workers),
        pin_memory=bool(data_module.pin_memory),
    )
    data_module.val_dataset = _encode_dataset(
        data_module.val_dataset,
        encoder=ae,
        batch_size=config.batch_size,
        device=device,
        num_workers=int(data_module.num_workers),
        pin_memory=bool(data_module.pin_memory),
    )
    data_module.test_dataset = _encode_dataset(
        data_module.test_dataset,
        encoder=ae,
        batch_size=config.batch_size,
        device=device,
        num_workers=int(data_module.num_workers),
        pin_memory=bool(data_module.pin_memory),
    )
    data_module.predict_dataset = _encode_dataset(
        data_module.predict_dataset,
        encoder=ae,
        batch_size=config.batch_size,
        device=device,
        num_workers=int(data_module.num_workers),
        pin_memory=bool(data_module.pin_memory),
    )
    data_module.input_dim = int(config.latent_dim)

    return metrics
