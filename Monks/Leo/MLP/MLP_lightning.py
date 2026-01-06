
"""
Optuna + PyTorch Lightning per la ricerca iperparametri di un MLP su MONK.

Pipeline:
1) Carica `monks-{task}.train` e `monks-{task}.test` separati.
2) Split 80/20 sul training set (train/val) con seed fissi: 0,1,2,3,4.
3) Selezione iperparametri su validation (test mai usato in selezione).
4) Input: One-Hot (senza StandardScaler).
5) Metriche richieste:
   - training: acc, mse
   - validation: acc, mse
   - test: acc, mse
6) Ricerca iperparametri con Optuna (spazio = griglia originale).
7) Dopo la selezione: retraining su tutto il training set e test sul test set per ogni seed.
8) Salvataggi: metriche medie sui seed + summary JSON (e modelli per seed).
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import OneHotEncoder

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import optuna
from optuna.samplers import TPESampler

# Per ridurre rumore su Windows con TensorBoard/TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings(
    "ignore",
    message=r".*does not have many workers.*",
    category=UserWarning,
    module=r"lightning\.pytorch\.trainer\.connectors\.data_connector",
)

SEEDS = [0, 1, 2, 3, 4]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "monk"
OUTPUT_ROOT = Path(__file__).resolve().parent

MAX_EPOCHS = 500
MIN_EPOCHS = 20
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-5
BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
LOG_EVERY_N_STEPS = 1
SGD_MOMENTUM = 0.9


# ============================================================
# DATA LOADING + SPLIT + PREPROCESSING
# ============================================================


def load_monk(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica un file MONK.

    Formato (per riga):
        label  a1 a2 a3 a4 a5 a6  sample_id

    Returns:
        X: ndarray (n_samples, 6) con feature categoriali intere.
        y: ndarray (n_samples,) con label binarie 0/1.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")

    y = df[0].astype(int).to_numpy()
    X = df.iloc[:, 1:7].astype(int).to_numpy()
    return X, y


def load_monk_task(task_id: int, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """
    Carica MONK-{task_id} da `data/monk/monks-{task_id}.{split}`.
    """
    if task_id not in (1, 2, 3):
        raise ValueError("task_id deve essere 1, 2 oppure 3.")
    if split not in ("train", "test"):
        raise ValueError("split deve essere 'train' oppure 'test'.")

    file_path = DATA_DIR / f"monks-{task_id}.{split}"
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato: {file_path}")
    return load_monk(file_path)


def make_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split hold-out 80/20 con shuffle e stratificazione.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_val, y_train, y_val


def _make_onehot_encoder() -> OneHotEncoder:
    """
    Compatibilita' tra versioni scikit-learn:
    - >=1.2: sparse_output
    - <1.2: sparse
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocess_splits(
    X_train: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, OneHotEncoder, None]:
    """
    One-Hot puro (fit solo su X_train).
    """
    encoder = _make_onehot_encoder()
    X_train_enc = encoder.fit_transform(X_train)
    X_val_enc = encoder.transform(X_val) if X_val is not None else None
    X_test_enc = encoder.transform(X_test) if X_test is not None else None

    return X_train_enc, X_val_enc, X_test_enc, encoder, None

# ============================================================
# LIGHTNING MODULES
# ============================================================


class MonkDataModule(pl.LightningDataModule):
    """
    DataModule MONK con split train/val sul training set.
    """

    def __init__(
        self,
        *,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        batch_size: int,
        num_workers: int,
        use_val: bool = True,
    ) -> None:
        super().__init__()
        self.X_train_full = X_train_full
        self.y_train_full = y_train_full
        self.X_test = X_test
        self.y_test = y_test
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_val = use_val

        self.encoder: OneHotEncoder | None = None
        self.scaler: object | None = None
        self.input_dim: int | None = None

        self.train_dataset: torch.utils.data.Dataset | None = None
        self.val_dataset: torch.utils.data.Dataset | None = None
        self.test_dataset: torch.utils.data.Dataset | None = None
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if self._setup_done:
            return
        if self.use_val:
            X_train, X_val, y_train, y_val = make_holdout_split(
                self.X_train_full,
                self.y_train_full,
                seed=self.seed,
            )
        else:
            X_train, y_train = self.X_train_full, self.y_train_full
            X_val, y_val = None, None

        X_train_s, X_val_s, X_test_s, encoder, scaler = preprocess_splits(
            X_train,
            X_val=X_val,
            X_test=self.X_test,
        )
        assert X_train_s is not None
        assert X_test_s is not None

        self.encoder = encoder
        self.scaler = scaler
        self.input_dim = int(X_train_s.shape[1])

        self.train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        )
        if X_val_s is not None and y_val is not None:
            self.val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val_s, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
            )
        else:
            # Lightning richiede un val_dataloader: usiamo un dataset vuoto.
            self.val_dataset = torch.utils.data.TensorDataset(
                torch.empty((0, self.input_dim), dtype=torch.float32),
                torch.empty((0, 1), dtype=torch.float32),
            )
        self.test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test_s, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1),
        )

        self._setup_done = True

    def _make_loader(self, dataset: torch.utils.data.Dataset, *, shuffle: bool) -> torch.utils.data.DataLoader:
        g = torch.Generator()
        g.manual_seed(self.seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=g if shuffle else None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.train_dataset is not None
        return self._make_loader(self.train_dataset, shuffle=True)

    def train_eval_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.train_dataset is not None
        return self._make_loader(self.train_dataset, shuffle=False)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.val_dataset is not None
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.test_dataset is not None
        return self._make_loader(self.test_dataset, shuffle=False)


class MLPClassifier(pl.LightningModule):
    """
    MLP molto semplice per classificazione binaria:
        input_dim -> hidden_dim (Tanh) -> 1 (Sigmoid)

    Nota: usiamo MSE come loss.
    """

    def __init__(self, *, input_dim: int, hidden_dim: int, lr: float, weight_decay: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _compute_batch_metrics(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        y_hat = self(x)
        mse = F.mse_loss(y_hat, y)
        acc = (y_hat > 0.5).float().eq(y).float().mean()
        return y_hat, mse, acc

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _, mse, acc = self._compute_batch_metrics(batch)
        bs = batch[0].size(0)
        self.log("train/mse", mse, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/acc", acc, on_step=False, on_epoch=True, batch_size=bs)
        return mse

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        _, mse, acc = self._compute_batch_metrics(batch)
        bs = batch[0].size(0)
        self.log("val/mse", mse, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val/acc", acc, on_step=False, on_epoch=True, batch_size=bs)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        _, mse, acc = self._compute_batch_metrics(batch)
        bs = batch[0].size(0)
        self.log("test/mse", mse, on_step=False, on_epoch=True, batch_size=bs)
        self.log("test/acc", acc, on_step=False, on_epoch=True, batch_size=bs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=SGD_MOMENTUM,
            weight_decay=self.hparams.weight_decay,
        )

class BestMetricsTracker(Callback):
    """
    Tiene traccia dell'epoca migliore su val/mse e salva metriche train/val.
    """

    def __init__(self, *, monitor: str = "val/mse", mode: str = "min") -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_epoch: int | None = None
        self.best_metrics: dict[str, float] = {}
        if mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = float("-inf")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        current = float(metrics[self.monitor].detach().cpu())
        improved = current < self.best_score if self.mode == "min" else current > self.best_score
        if not improved:
            return
        self.best_score = current
        self.best_epoch = trainer.current_epoch + 1

        def _get(name: str) -> float:
            value = metrics.get(name)
            if value is None:
                return float("nan")
            return float(value.detach().cpu())

        self.best_metrics = {
            "train_acc": _get("train/acc"),
            "train_mse": _get("train/mse"),
            "val_acc": _get("val/acc"),
            "val_mse": _get("val/mse"),
        }


class HistoryTracker(Callback):
    """
    Salva lo storico delle metriche di training per epoca.
    Se viene passato un test_loader, calcola anche test/mse a fine epoca.
    """

    def __init__(
        self,
        keys: Iterable[str],
        *,
        train_loader: torch.utils.data.DataLoader | None = None,
        test_loader: torch.utils.data.DataLoader | None = None,
    ) -> None:
        super().__init__()
        self.history: dict[str, list[float]] = {k: [] for k in keys}
        self.train_loader = train_loader
        self.test_loader = test_loader

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        recorded: set[str] = set()
        device = next(pl_module.parameters()).device

        if self.train_loader is not None:
            was_training = pl_module.training
            pl_module.eval()
            with torch.no_grad():
                train_mse, train_acc = eval_loader_metrics(pl_module, self.train_loader, device)
            if was_training:
                pl_module.train()
            if "train/mse" in self.history:
                self.history["train/mse"].append(float(train_mse))
                recorded.add("train/mse")
            if "train/acc" in self.history:
                self.history["train/acc"].append(float(train_acc))
                recorded.add("train/acc")

        if self.test_loader is not None and "test/mse" in self.history:
            was_training = pl_module.training
            pl_module.eval()
            with torch.no_grad():
                test_mse, _ = eval_loader_metrics(pl_module, self.test_loader, device)
            if was_training:
                pl_module.train()
            self.history["test/mse"].append(float(test_mse))
            recorded.add("test/mse")

        for key in self.history:
            if key in recorded:
                continue
            if key in metrics:
                self.history[key].append(float(metrics[key].detach().cpu()))


class VerboseEpochLogger(Callback):
    """
    Stampa metriche per epoca quando --verbose e' attivo.
    """

    def __init__(self, every_n: int = 1) -> None:
        super().__init__()
        self.every_n = every_n

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n != 0:
            return
        metrics = trainer.callback_metrics
        train_mse = metrics.get("train/mse")
        train_acc = metrics.get("train/acc")
        val_mse = metrics.get("val/mse")
        val_acc = metrics.get("val/acc")
        if train_mse is None or val_mse is None:
            return
        print(
            f"      epoch {epoch:03d} | train_mse={float(train_mse):.6f} train_acc={float(train_acc):.4f} "
            f"| val_mse={float(val_mse):.6f} val_acc={float(val_acc):.4f}",
            flush=True,
        )


class OptunaPruningCallback(Callback):
    """
    Pruning Optuna: interrompe il trial se la metrica monitorata non migliora.
    """

    def __init__(self, trial: optuna.trial.Trial, *, monitor: str, step_offset: int = 0) -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.step_offset = step_offset

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu())
        else:
            value = float(value)
        step = self.step_offset + trainer.current_epoch
        self.trial.report(value, step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at step {step} ({self.monitor}={value:.6f})")

# ============================================================
# METRICS + AGGREGATION
# ============================================================


@dataclass(frozen=True)
class SeedMetrics:
    seed: int
    train_acc: float
    train_mse: float
    val_acc: float
    val_mse: float
    test_acc: float
    test_mse: float


@dataclass(frozen=True)
class AggregatedMetrics:
    train_acc_mean: float
    train_acc_std: float
    train_mse_mean: float
    train_mse_std: float
    val_acc_mean: float
    val_acc_std: float
    val_mse_mean: float
    val_mse_std: float
    test_acc_mean: float
    test_acc_std: float
    test_mse_mean: float
    test_mse_std: float


def aggregate_metrics(per_seed: list[SeedMetrics]) -> AggregatedMetrics:
    """
    Media e deviazione standard sulle repliche (seed 0..4).
    """

    def mean_std(values: Iterable[float]) -> tuple[float, float]:
        arr = np.asarray(list(values), dtype=float)
        if np.all(np.isnan(arr)):
            return float("nan"), float("nan")
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    train_acc_mean, train_acc_std = mean_std(m.train_acc for m in per_seed)
    train_mse_mean, train_mse_std = mean_std(m.train_mse for m in per_seed)

    val_acc_mean, val_acc_std = mean_std(m.val_acc for m in per_seed)
    val_mse_mean, val_mse_std = mean_std(m.val_mse for m in per_seed)

    test_acc_mean, test_acc_std = mean_std(m.test_acc for m in per_seed)
    test_mse_mean, test_mse_std = mean_std(m.test_mse for m in per_seed)

    return AggregatedMetrics(
        train_acc_mean=train_acc_mean,
        train_acc_std=train_acc_std,
        train_mse_mean=train_mse_mean,
        train_mse_std=train_mse_std,
        val_acc_mean=val_acc_mean,
        val_acc_std=val_acc_std,
        val_mse_mean=val_mse_mean,
        val_mse_std=val_mse_std,
        test_acc_mean=test_acc_mean,
        test_acc_std=test_acc_std,
        test_mse_mean=test_mse_mean,
        test_mse_std=test_mse_std,
    )


@torch.no_grad()
def eval_loader_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Valuta MSE e accuracy su un loader.
    """
    model.eval()
    sse = 0.0
    correct = 0.0
    n = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        sse += float(torch.sum((y_hat - y) ** 2).detach().cpu().item())
        correct += float(torch.sum(((y_hat > 0.5).float() == y).float()).detach().cpu().item())
        n += float(y.numel())

    mse = sse / max(n, 1.0)
    acc = correct / max(n, 1.0)
    return mse, acc

# ============================================================
# OPTUNA SEARCH
# ============================================================


def get_param_space(
    *,
    weight_decay_values: list[float] | None = None,
    lr_values: list[float] | None = None,
) -> dict[str, list[Any]]:
    """
    Spazio iperparametri per MLP (uguale alla griglia di partenza).
    """
    return {
        "hidden_dim": [1, 2, 3, 4],
        "lr": lr_values if lr_values is not None else [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        "weight_decay": weight_decay_values if weight_decay_values is not None else [0.0, 1e-4, 1e-3, 1e-2],
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _cuda_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch_list = torch.cuda.get_arch_list()
        if not arch_list:
            return False
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        return arch in arch_list
    except Exception:
        return False


def get_accelerator() -> str:
    if _cuda_supported():
        return "gpu"
    return "cpu"


def _trainer(
    *,
    accelerator: str,
    max_epochs: int,
    min_epochs: int,
    callbacks: list[Callback],
    logger: TensorBoardLogger | None,
    verbose: bool,
) -> pl.Trainer:
    return pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=verbose,
        deterministic=True,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )


def _train_val_seed(
    *,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    params: dict[str, Any],
    accelerator: str,
    verbose: bool,
    num_workers: int,
    batch_size: int,
    min_epochs: int,
    trial: optuna.trial.Trial | None = None,
    prune_step_offset: int = 0,
    prune_monitor: str = "val/mse",
) -> tuple[dict[str, float], int]:
    pl.seed_everything(seed, workers=True)

    data = MonkDataModule(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        use_val=True,
    )
    data.setup("fit")
    assert data.input_dim is not None

    model = MLPClassifier(
        input_dim=data.input_dim,
        hidden_dim=int(params["hidden_dim"]),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    tracker = BestMetricsTracker(monitor="val/mse", mode="min")
    callbacks: list[Callback] = [
        tracker,
        EarlyStopping(
            monitor="val/mse",
            patience=EARLY_STOP_PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            mode="min",
        ),
    ]
    if trial is not None:
        callbacks.append(
            OptunaPruningCallback(
                trial,
                monitor=prune_monitor,
                step_offset=prune_step_offset,
            )
        )
    if verbose:
        callbacks.append(VerboseEpochLogger(every_n=1))

    trainer = _trainer(
        accelerator=accelerator,
        max_epochs=MAX_EPOCHS,
        min_epochs=min_epochs,
        callbacks=callbacks,
        logger=None,
        verbose=verbose,
    )
    trainer.fit(model, datamodule=data)

    best_epoch = tracker.best_epoch or (trainer.current_epoch + 1)
    best_metrics = tracker.best_metrics
    if not best_metrics:
        metrics = trainer.callback_metrics
        best_metrics = {
            "train_acc": float(metrics.get("train/acc", float("nan"))),
            "train_mse": float(metrics.get("train/mse", float("nan"))),
            "val_acc": float(metrics.get("val/acc", float("nan"))),
            "val_mse": float(metrics.get("val/mse", float("nan"))),
        }

    return best_metrics, int(best_epoch)


def eval_params_on_validation(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any],
    *,
    accelerator: str,
    verbose: bool,
    num_workers: int,
    batch_size: int,
    min_epochs: int,
    trial: optuna.trial.Trial | None = None,
) -> tuple[list[SeedMetrics], AggregatedMetrics, float]:
    """
    Per ogni seed:
    - split 80/20 sul training set: train/val
    - training con early stopping su val/mse
    - metriche: train (acc, mse), val (acc, mse)
    """
    per_seed: list[SeedMetrics] = []
    t0 = time.time()

    for seed_idx, seed in enumerate(SEEDS):
        prune_offset = seed_idx * (MAX_EPOCHS + 1)
        metrics, best_epoch = _train_val_seed(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
            params=params,
            accelerator=accelerator,
            verbose=verbose,
            num_workers=num_workers,
            batch_size=batch_size,
            min_epochs=min_epochs,
            trial=trial,
            prune_step_offset=prune_offset,
        )

        print(
            f"    seed={seed} best_epoch={best_epoch:03d} "
            f"val_mse={metrics['val_mse']:.6f} val_acc={metrics['val_acc']:.4f}",
            flush=True,
        )

        per_seed.append(
            SeedMetrics(
                seed=seed,
                train_acc=float(metrics["train_acc"]),
                train_mse=float(metrics["train_mse"]),
                val_acc=float(metrics["val_acc"]),
                val_mse=float(metrics["val_mse"]),
                test_acc=float("nan"),
                test_mse=float("nan"),
            )
        )

    agg = aggregate_metrics(per_seed)
    seconds = float(time.time() - t0)
    return per_seed, agg, seconds

def select_best_params(
    *,
    task_id: int,
    run_name: str,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    accelerator: str,
    n_trials: int,
    n_jobs: int,
    optuna_seed: int,
    verbose: bool,
    num_workers: int,
    batch_size: int,
    min_epochs: int,
    param_space: dict[str, list[Any]] | None = None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    """
    Ricerca iperparametri con Optuna usando SOLO la validation.
    """
    model_kind = "mlp"

    results_dir = OUTPUT_ROOT / "results" / f"monk{task_id}" / model_kind
    ckpt_dir = OUTPUT_ROOT / "checkpoints" / f"monk{task_id}" / model_kind / run_name
    export_dir = OUTPUT_ROOT / "exported_models" / f"monk{task_id}" / model_kind / run_name
    for p in (results_dir, ckpt_dir, export_dir):
        ensure_dir(p)

    space = param_space if param_space is not None else get_param_space()
    results_path = results_dir / f"{run_name}.jsonl"
    best_meta_ckpt_path = ckpt_dir / "best_params_so_far.json"

    best_meta: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None  # (val_mse_mean, -val_acc_mean)
    lock = threading.Lock()

    sampler = TPESampler(seed=optuna_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {name: trial.suggest_categorical(name, values) for name, values in space.items()}
        print(f"  [trial {trial.number + 1}/{n_trials}] params={params}", flush=True)
        per_seed, agg, seconds = eval_params_on_validation(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            params,
            accelerator=accelerator,
            verbose=verbose,
            num_workers=num_workers,
            batch_size=batch_size,
            min_epochs=min_epochs,
            trial=trial,
        )

        print(
            "    mean val:  "
            f"acc={agg.val_acc_mean:.4f} mse={agg.val_mse_mean:.6f} "
            f"({seconds:.2f}s)",
            flush=True,
        )

        record = {
            "task_id": task_id,
            "model_kind": model_kind,
            "combo_idx": int(trial.number),
            "params": params,
            "eval_split": "val",
            "seeds": SEEDS,
            "per_seed": [asdict(m) for m in per_seed],
            "agg": asdict(agg),
            "seconds": seconds,
        }

        key = (agg.val_mse_mean, -agg.val_acc_mean)

        with lock:
            with open(results_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(record) + "\n")
            nonlocal best_meta, best_key
            if best_key is None or key < best_key:
                best_key = key
                print("    -> NEW BEST", flush=True)

                best_meta = {
                    "task_id": task_id,
                    "model_kind": model_kind,
                    "run_name": run_name,
                    "best_params": params,
                    "best_combo_idx": int(trial.number),
                    "selection": {
                        "eval_split": "val",
                        "criterion": {
                            "min_val_mse_mean": agg.val_mse_mean,
                            "max_val_acc_mean": agg.val_acc_mean,
                        },
                        "metrics": asdict(agg),
                        "per_seed": [asdict(m) for m in per_seed],
                    },
                }
                best_meta_ckpt_path.write_text(json.dumps(best_meta, indent=2), encoding="utf-8")

        return float(agg.val_mse_mean)

    print(
        f"  Optuna search: trials={n_trials} jobs={n_jobs} seeds={SEEDS}",
        flush=True,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    if best_meta is None:
        raise RuntimeError("Nessuna combinazione valutata: controlla la griglia.")

    return best_meta, results_path, ckpt_dir, export_dir

# ============================================================
# FINAL TRAINING + TEST
# ============================================================


def final_retrain_and_test(
    *,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: dict[str, Any],
    accelerator: str,
    verbose: bool,
    num_workers: int,
    batch_size: int,
    min_epochs: int,
) -> tuple[list[SeedMetrics], AggregatedMetrics, float, list[dict[str, Any]]]:
    """
    Dopo aver scelto gli iperparametri sul validation set:
    - per ogni seed: retraining su tutto il training set e test sul test set.
    - si calcola la media delle metriche sui seed.
    """
    per_seed: list[SeedMetrics] = []
    t0 = time.time()

    seed_artifacts: list[dict[str, Any]] = []

    for seed in SEEDS:
        # (1) Early stopping su split train/val per stimare best_epoch.
        metrics, best_epoch = _train_val_seed(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
            params=best_params,
            accelerator=accelerator,
            verbose=verbose,
            num_workers=num_workers,
            batch_size=batch_size,
            min_epochs=min_epochs,
        )

        # (2) Retraining su tutto il training set per best_epoch epoche, poi test.
        pl.seed_everything(seed, workers=True)
        data_full = MonkDataModule(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
            use_val=False,
        )
        data_full.setup("fit")
        assert data_full.input_dim is not None

        model = MLPClassifier(
            input_dim=data_full.input_dim,
            hidden_dim=int(best_params["hidden_dim"]),
            lr=float(best_params["lr"]),
            weight_decay=float(best_params["weight_decay"]),
        )

        history_cb = HistoryTracker(
            keys=["train/mse", "train/acc", "test/mse"],
            train_loader=data_full.train_eval_dataloader(),
            test_loader=data_full.test_dataloader(),
        )
        callbacks: list[Callback] = [history_cb]
        if verbose:
            callbacks.append(VerboseEpochLogger(every_n=1))

        retrain_epochs = max(int(best_epoch), min_epochs)
        trainer = _trainer(
            accelerator=accelerator,
            max_epochs=retrain_epochs,
            min_epochs=min_epochs,
            callbacks=callbacks,
            logger=None,
            verbose=verbose,
        )
        trainer.fit(model, datamodule=data_full)

        device = next(model.parameters()).device
        train_mse, train_acc = eval_loader_metrics(model, data_full.train_eval_dataloader(), device)
        test_mse, test_acc = eval_loader_metrics(model, data_full.test_dataloader(), device)

        metrics_seed = SeedMetrics(
            seed=seed,
            train_acc=float(train_acc),
            train_mse=float(train_mse),
            val_acc=float(metrics["val_acc"]),
            val_mse=float(metrics["val_mse"]),
            test_acc=float(test_acc),
            test_mse=float(test_mse),
        )
        per_seed.append(metrics_seed)

        print(
            f"    seed={seed} retrain_epochs={retrain_epochs:03d} "
            f"test_mse={metrics_seed.test_mse:.6f} "
            f"test_acc={metrics_seed.test_acc:.4f}",
            flush=True,
        )

        state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        seed_artifacts.append(
            {
                "seed": seed,
                "best_epoch": int(best_epoch),
                "params": dict(best_params),
                "input_dim": int(data_full.input_dim),
                "state_dict": state_dict_cpu,
                "encoder": data_full.encoder,
                "scaler": data_full.scaler,
                "history": history_cb.history,
            }
        )

    agg = aggregate_metrics(per_seed)
    seconds = float(time.time() - t0)
    return per_seed, agg, seconds, seed_artifacts

# ============================================================
# RUN PIPELINE
# ============================================================


def run_task(
    *,
    task_id: int,
    run_name: str,
    n_trials: int,
    n_jobs: int,
    optuna_seed: int,
    verbose: bool,
    accelerator: str,
    num_workers: int,
    batch_size: int,
    min_epochs: int,
    param_space: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """
    Esegue l'intera pipeline per un task MONK:
    - selezione iperparametri su validation con Optuna
    - retraining finale 5 volte (seed 0..4) e test sul test esterno
    - salvataggio delle metriche medie + summary JSON
    """
    if accelerator == "gpu":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"  device=cuda name={name} capability={cap}", flush=True)
    else:
        print("  device=cpu", flush=True)

    X_train_full, y_train_full = load_monk_task(task_id, split="train")
    X_test, y_test = load_monk_task(task_id, split="test")

    best_meta, results_path, _ckpt_dir, export_dir = select_best_params(
        task_id=task_id,
        run_name=run_name,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        accelerator=accelerator,
        n_trials=n_trials,
        n_jobs=n_jobs,
        optuna_seed=optuna_seed,
        verbose=verbose,
        num_workers=num_workers,
        batch_size=batch_size,
        min_epochs=min_epochs,
        param_space=param_space,
    )

    print("  Final retraining su training set e valutazione su test (mai visto)...", flush=True)
    best_params = dict(best_meta["best_params"])
    final_per_seed, final_agg, final_seconds, seed_artifacts = final_retrain_and_test(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        best_params=best_params,
        accelerator=accelerator,
        verbose=verbose,
        num_workers=num_workers,
        batch_size=batch_size,
        min_epochs=min_epochs,
    )

    best_meta["final"] = {
        "eval_split": "test",
        "metrics": asdict(final_agg),
        "per_seed": [asdict(m) for m in final_per_seed],
        "seconds": final_seconds,
    }
    best_meta["final"]["params_used"] = dict(best_params)

    summary_path = export_dir / "summary.json"

    ensure_dir(export_dir)
    model_paths: list[str] = []
    for artifact in seed_artifacts:
        seed = artifact["seed"]
        model_path = export_dir / f"model_seed{seed}.pt"
        torch.save(artifact, model_path)
        model_paths.append(str(model_path))
    best_meta["final"]["model_paths"] = model_paths
    summary_path.write_text(json.dumps(best_meta, indent=2), encoding="utf-8")

    sel = best_meta["selection"]["metrics"]
    fin = best_meta["final"]["metrics"]
    print(f"  params (from val): {best_meta['best_params']}", flush=True)
    print(
        "  selection mean train: "
        f"acc={float(sel['train_acc_mean']):.4f} "
        f"mse={float(sel['train_mse_mean']):.6f}",
        flush=True,
    )
    print(
        "  selection mean val:   "
        f"acc={float(sel['val_acc_mean']):.4f} "
        f"mse={float(sel['val_mse_mean']):.6f}",
        flush=True,
    )
    print(
        "  final mean train:     "
        f"acc={float(fin['train_acc_mean']):.4f} "
        f"mse={float(fin['train_mse_mean']):.6f}",
        flush=True,
    )
    print(
        "  final mean test:      "
        f"acc={float(fin['test_acc_mean']):.4f} "
        f"mse={float(fin['test_mse_mean']):.6f}",
        flush=True,
    )
    return {
        "task_id": task_id,
        "run_name": run_name,
        "results_jsonl": str(results_path),
        "summary_json": str(summary_path),
        "model_paths": model_paths,
        "summary": best_meta,
        "seed_artifacts": seed_artifacts,
    }


def log_summary_to_tensorboard(
    *,
    task_id: int,
    run_name: str,
    summary: dict[str, Any],
    seed_artifacts: list[dict[str, Any]],
) -> Path:
    """
    TensorBoard: logga metriche aggregate e per-seed per MONK-{task_id}.
    """
    tb_dir = OUTPUT_ROOT / "tb_logs" / f"monk{task_id}" / run_name
    ensure_dir(tb_dir)
    logger = TensorBoardLogger(save_dir=str(tb_dir), name="")
    writer = logger.experiment

    sel = summary["selection"]["metrics"]
    fin = summary["final"]["metrics"]

    writer.add_scalar("selection/train_acc_mean", float(sel["train_acc_mean"]), 0)
    writer.add_scalar("selection/train_mse_mean", float(sel["train_mse_mean"]), 0)
    writer.add_scalar("selection/val_acc_mean", float(sel["val_acc_mean"]), 0)
    writer.add_scalar("selection/val_mse_mean", float(sel["val_mse_mean"]), 0)

    writer.add_scalar("final/train_acc_mean", float(fin["train_acc_mean"]), 0)
    writer.add_scalar("final/train_mse_mean", float(fin["train_mse_mean"]), 0)
    writer.add_scalar("final/test_acc_mean", float(fin["test_acc_mean"]), 0)
    writer.add_scalar("final/test_mse_mean", float(fin["test_mse_mean"]), 0)

    writer.add_text("params", json.dumps(summary["best_params"]), 0)

    for metrics in summary["final"]["per_seed"]:
        seed = metrics["seed"]
        writer.add_scalar(f"final/seed_{seed}/test_mse", float(metrics["test_mse"]), 0)
        writer.add_scalar(f"final/seed_{seed}/test_acc", float(metrics["test_acc"]), 0)

    for artifact in seed_artifacts:
        seed = artifact["seed"]
        history = artifact.get("history", {})
        for i, v in enumerate(history.get("train/mse", []), start=1):
            writer.add_scalar(f"seed_{seed}/train_mse", float(v), i)
        for i, v in enumerate(history.get("train/acc", []), start=1):
            writer.add_scalar(f"seed_{seed}/train_acc", float(v), i)
        for i, v in enumerate(history.get("test/mse", []), start=1):
            writer.add_scalar(f"seed_{seed}/test_mse", float(v), i)
        writer.add_text(f"seed_{seed}/retrain_epochs", str(artifact.get("best_epoch", "?")), 0)

    writer.flush()
    return tb_dir


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna + Lightning MLP su MONK con multi-seed.")
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Quali task MONK eseguire (default: 1 2 3).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help="Numero di trial Optuna (default: tutte le combinazioni della griglia).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Numero di job paralleli per Optuna (default: max(1, cpu_count-1)).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Numero di worker DataLoader (default: 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=MIN_EPOCHS,
        help=f"Min epoche prima di early stopping (default: {MIN_EPOCHS}).",
    )
    parser.add_argument(
        "--weight-decay-values",
        type=float,
        nargs="+",
        default=None,
        help="Valori di weight_decay da usare (default: griglia completa).",
    )
    parser.add_argument(
        "--lr-values",
        type=float,
        nargs="+",
        default=None,
        help="Valori di learning rate da usare (default: griglia completa).",
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=0,
        help="Seed per il sampler Optuna (default: 0).",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disabilita logging TensorBoard.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stampa l'avanzamento del training (per epoca).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = now_run_id()

    space = get_param_space(weight_decay_values=args.weight_decay_values, lr_values=args.lr_values)
    default_trials = len(ParameterGrid(space))
    n_trials = args.n_trials if args.n_trials > 0 else default_trials
    n_jobs = args.n_jobs if args.n_jobs > 0 else max(1, (os.cpu_count() or 1) - 1)
    num_workers = max(int(args.num_workers), 0)
    batch_size = max(int(args.batch_size), 1)
    min_epochs = max(int(args.min_epochs), 1)
    accelerator = get_accelerator()
    if accelerator == "gpu" and n_jobs > 1:
        print("  WARNING: GPU + n_jobs>1 puo' causare errori CUDA; uso CPU per parallelizzare.", flush=True)
        accelerator = "cpu"
    if n_jobs > 1 and num_workers > 0:
        print("  WARNING: n_jobs>1 con DataLoader workers puo' causare crash; uso num_workers=0.", flush=True)
        num_workers = 0
    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    for task_id in args.tasks:
        print(f"\n[MONK {task_id}] mlp | run={run_name}", flush=True)
        info = run_task(
            task_id=task_id,
            run_name=run_name,
            n_trials=n_trials,
            n_jobs=n_jobs,
            optuna_seed=args.optuna_seed,
            verbose=args.verbose,
            accelerator=accelerator,
            num_workers=num_workers,
            batch_size=batch_size,
            min_epochs=min_epochs,
            param_space=space,
        )

        print(
            "  -> salvati:",
            f"results={info['results_jsonl']}",
            f"summary={info['summary_json']}",
            f"models={info['model_paths']}",
            sep="\n     ",
            flush=True,
        )

        if not args.no_tensorboard:
            tb_dir = log_summary_to_tensorboard(
                task_id=task_id,
                run_name=run_name,
                summary=info["summary"],
                seed_artifacts=info["seed_artifacts"],
            )
            print(f"  TensorBoard logs: {tb_dir}", flush=True)


if __name__ == "__main__":
    main()
