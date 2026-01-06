

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

from typing import List
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn



# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Metrics: MSE (for training) & MEE (for evaluation)
# -----------------------------
@torch.no_grad()
def mee_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Mean Euclidean Error (MEE) for 4D regression:
    mean over samples of ||y_pred - y_true||_2
    (metric used for CUP ranking; report in original scale). :contentReference[oaicite:1]{index=1}
    """
    # shape: (N, 4)
    diff = y_pred - y_true
    per_sample = torch.sqrt(torch.sum(diff * diff, dim=1))  # (N,)
    return per_sample.mean().item()


@torch.no_grad()
def mse_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Plain MSE as a metric (not RMSE)."""
    return torch.mean((y_pred - y_true) ** 2).item()


# -----------------------------
# Dataset
# -----------------------------
class CupDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2
        assert y.ndim == 2 and y.shape[1] == 4
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -----------------------------
# Config for data step
# -----------------------------
@dataclass
class DataConfig:
    csv_path: str
    test_size: float = 0.2        
    val_size: float = 0.2         
    seed: int = 42

    batch_size: int = 500
    num_workers: int = 0
    pin_memory: bool = True

    scale_X: bool = True          
    


def load_cup_tr(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads ML-CUP25-TR.csv:
    - col 0: id/name
    - next 12 cols: inputs
    - last 4 cols: targets
    """
    df = pd.read_csv(csv_path, comment="#", header=None)
    if df.shape[1] != 1 + 12 + 4:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]} (expected 17)")

    X = df.iloc[:, 1:13].to_numpy(dtype=np.float64)  # 12 inputs
    y = df.iloc[:, 13:17].to_numpy(dtype=np.float64) # 4 targets
    return X, y


def make_splits_and_loaders(cfg: DataConfig) -> Dict[str, object]:
    """
    Returns:
      - scalers
      - numpy splits
      - dataloaders
    """
    set_seed(cfg.seed)

    X, y = load_cup_tr(cfg.csv_path)

    # 1) split off INTERNAL TEST 
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, shuffle=True
    )

    # 2) split remaining into TRAIN and VAL
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=cfg.val_size, random_state=cfg.seed, shuffle=True
    )

    # 3) scale X using train statistics only
    x_scaler: Optional[StandardScaler] = None
    if cfg.scale_X:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)

    # 4) datasets & loaders
    train_ds = CupDataset(X_train, y_train)
    val_ds = CupDataset(X_val, y_val)
    test_ds = CupDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "x_scaler": x_scaler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }





# -----------------------------
# Activation factory
# -----------------------------
def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {name}")


# -----------------------------
# Model config (for grid search & JSON saving)
# -----------------------------
@dataclass
class MLPConfig:
    input_dim: int = 12
    output_dim: int = 4

    hidden_layers: List[int] = None   
    activation: str = "relu"          
    dropout: float = 0.0              

    def to_dict(self):
        return asdict(self)


# -----------------------------
# MLP Model
# -----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()

        if cfg.hidden_layers is None or len(cfg.hidden_layers) == 0:
            raise ValueError("hidden_layers must be a non-empty list")

        layers = []
        prev_dim = cfg.input_dim
        act = get_activation(cfg.activation)

        for h in cfg.hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act)

            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(p=cfg.dropout))

            prev_dim = h

        # Output layer (LINEAR)
        layers.append(nn.Linear(prev_dim, cfg.output_dim))

        self.model = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """
        He initialization for hidden layers (good with ReLU / LeakyReLU)
        Output layer left with default init.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -----------------------------
# Loss function
# -----------------------------
def get_loss():
    """
    Training loss: MSE
    (MEE is used ONLY as evaluation metric)
    """
    return nn.MSELoss()





# -----------------------------
# Training config
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 900
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"        # adam | adamw | sgd
    momentum: float = 0.9          # used only if optimizer == "sgd"
    patience: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"




# -----------------------------
# Run one epoch (train or eval)
# -----------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    loss_fn: nn.Module,
    device: str,
) -> Dict[str, float]:

    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0
    all_y = []
    all_yhat = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        yhat = model(X)
        loss = loss_fn(yhat, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        all_y.append(y.detach().cpu())
        all_yhat.append(yhat.detach().cpu())

    y_all = torch.cat(all_y, dim=0)
    yhat_all = torch.cat(all_yhat, dim=0)

    return {
        "mse": total_loss / n_batches,
        "mee": mee_metric(yhat_all, y_all),
    }




# -----------------------------
# Training loop with early stopping
# -----------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> Dict[str, object]:

    device = cfg.device
    model.to(device)

    loss_fn = get_loss()

    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )

    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer}")

    
    scheduler = None
    if cfg.optimizer == "sgd":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )



    history = {
        "train_mse": [],
        "train_mee": [],
        "val_mse": [],
        "val_mee": [],
    }

    best_val_mee = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_metrics = run_epoch(
            model, val_loader, None, loss_fn, device
        )

        if scheduler is not None:
            scheduler.step(val_metrics["mee"])


        history["train_mse"].append(train_metrics["mse"])
        history["train_mee"].append(train_metrics["mee"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_mee"].append(val_metrics["mee"])

        # Early stopping on VALIDATION MEE
        if val_metrics["mee"] < best_val_mee:
            best_val_mee = val_metrics["mee"]
            best_epoch = epoch
            best_state = {
                k: v.cpu() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:03d}] "
                f"Train MSE={train_metrics['mse']:.6f} "
                f"Train MEE={train_metrics['mee']:.6f} | "
                f"Val MSE={val_metrics['mse']:.6f} "
                f"Val MEE={val_metrics['mee']:.6f}"
            )

        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    model.load_state_dict(best_state)

    return {
        "best_epoch": best_epoch,
        "best_val_mee": best_val_mee,
        "history": history,
        "model_state": best_state,
    }



# -----------------------------
# Evaluate model on a loader
# -----------------------------
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:

    model.eval()
    model.to(device)

    all_y = []
    all_yhat = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        yhat = model(X)

        all_y.append(y.cpu())
        all_yhat.append(yhat.cpu())

    y_all = torch.cat(all_y, dim=0)
    yhat_all = torch.cat(all_yhat, dim=0)

    return {
        "mse": mse_metric(yhat_all, y_all),
        "mee": mee_metric(yhat_all, y_all),
    }



# -----------------------------
# Hyperparameter search
# -----------------------------
def hyperparameter_search(
    data: Dict[str, object],
    train_cfg: TrainConfig,
) -> Dict[str, object]:

    search_space = {
        "hidden_layers": [
            [448],
            [224, 224],
            [256, 128, 64],
            [112, 112, 112, 112],
            [90, 90, 90, 90, 90],
            [75, 75, 75, 75, 75, 75]
        ],
        "activation": ["relu", "leakyrelu"],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-4],
        "optimizer": ["adam"]
    }

    best_val_mee = float("inf")
    best_result = None
    run_id = 0

    for hidden_layers in search_space["hidden_layers"]:
        for activation in search_space["activation"]:
            for dropout in search_space["dropout"]:
                for lr in search_space["lr"]:
                    for wd in search_space["weight_decay"]:
                        for opt in search_space["optimizer"]:

                            run_id += 1
                            print(f"\n=== Run {run_id} ===")

                            model_cfg = MLPConfig(
                                hidden_layers=hidden_layers,
                                activation=activation,
                                dropout=dropout,
                            )

                            model = MLPRegressor(model_cfg)

                            local_train_cfg = TrainConfig(
                                epochs=train_cfg.epochs,
                                lr=lr,
                                weight_decay=wd,
                                optimizer=opt,
                                patience=train_cfg.patience,
                                device=train_cfg.device,
                            )

                            result = train_model(
                                model,
                                data["train_loader"],
                                data["val_loader"],
                                local_train_cfg,
                            )

                            val_mee = result["best_val_mee"]
                            print(f"Val MEE = {val_mee:.6f}")

                            if val_mee < best_val_mee:
                                print(">>> NEW BEST MODEL <<<")
                                best_val_mee = val_mee

                                best_result = {
                                    "model_cfg": model_cfg.to_dict(),
                                    "train_cfg": asdict(local_train_cfg),
                                    "val_mee": val_mee,
                                    "best_epoch": result["best_epoch"],
                                    "history": result["history"],
                                    "model_state": result["model_state"],
                                }

    return best_result





# -----------------------------
# Adaptive random hyperparameter search
# -----------------------------
import random

def adaptive_random_search(
    data: Dict[str, object],
    base_train_cfg: TrainConfig,
    n_trials: int = 40,
    top_k: int = 5,
    seed: int = 42,
) -> Dict[str, object]:

    # seed for random search
    random.seed(seed)

    # seeds used to mean the performance
    seeds = [0, 1, 2, 3, 4] 

    # Search space 
    space = {
        "hidden_layers": [
            [250],
            [125, 125],
            [62, 62, 62]
        ],
        "activation": ["relu", "leakyrelu"],
        "dropout": [0.0, 0.05, 0.1],
        "lr": [5e-5, 1e-5, 5e-6],
        "weight_decay": [0.0, 1e-6, 1e-5],
        "optimizer": ["sgd", "adam", "adamw"],
    }

    results = []

    for trial in range(1, n_trials + 1):
        print(f"\n=== RANDOM TRIAL {trial}/{n_trials} ===")

        # ---- sample hyperparameters
        model_cfg = MLPConfig(
            hidden_layers=random.choice(space["hidden_layers"]),
            activation=random.choice(space["activation"]),
            dropout=random.choice(space["dropout"]),
        )

        train_cfg = TrainConfig(
            epochs=base_train_cfg.epochs,
            lr=random.choice(space["lr"]),
            weight_decay=random.choice(space["weight_decay"]),
            optimizer=random.choice(space["optimizer"]),
            patience=base_train_cfg.patience,
            device=base_train_cfg.device,
        )

        val_mees = []
        last_result = None

        # ---- run over multiple seeds
        for seed_i in seeds:
            print(f"   Seed {seed_i}")

            set_seed(seed_i)

            model = MLPRegressor(model_cfg)

            result = train_model(
                model,
                data["train_loader"],
                data["val_loader"],
                train_cfg,
            )

            val_mees.append(result["best_val_mee"])
            last_result = result  # keep last to store state if selected

        mean_val_mee = float(np.mean(val_mees))
        std_val_mee = float(np.std(val_mees))

        print(
            f"Mean Val MEE = {mean_val_mee:.6f} "
            f"(std = {std_val_mee:.6f})"
        )

        results.append({
            "model_cfg": model_cfg.to_dict(),
            "train_cfg": asdict(train_cfg),
            "seed_val_mees": val_mees,
            "mean_val_mee": mean_val_mee,
            "std_val_mee": std_val_mee,
            "best_epoch": last_result["best_epoch"],
            "history": last_result["history"],
            "model_state": last_result["model_state"],
        })

    # ---- select top-k by mean validation MEE
    results.sort(key=lambda x: x["mean_val_mee"])
    top_results = results[:top_k]

    print("\n=== TOP CONFIGURATIONS (by mean Val MEE) ===")
    for i, r in enumerate(top_results, 1):
        print(
            f"{i}) Mean Val MEE = {r['mean_val_mee']:.6f} "
            f"(std = {r['std_val_mee']:.6f}) | "
            f"{r['model_cfg']}"
        )

    # return the BEST one (mean-based)
    return top_results[0]





# -----------------------------
# Merge train + val loaders
# -----------------------------
def make_trainval_loader(
    data: Dict[str, object],
    batch_size: int,
) -> DataLoader:

    X = np.vstack([data["X_train"], data["X_val"]])
    y = np.vstack([data["y_train"], data["y_val"]])

    ds = CupDataset(X, y)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
    )




# -----------------------------
# Final retraining + learning curves
# -----------------------------
def retrain_final_model(
    model_cfg: MLPConfig,
    train_cfg: TrainConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> Dict[str, object]:

    model = MLPRegressor(model_cfg)
    model.to(train_cfg.device)

    loss_fn = get_loss()

    # ---- Optimizer
    if train_cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        scheduler = None

    elif train_cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        scheduler = None

    elif train_cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_cfg.lr,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )
    else:
        raise ValueError(f"Unknown optimizer {train_cfg.optimizer}")

    train_mse_curve = []
    test_mse_curve = []

    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, loss_fn, train_cfg.device
        )
        test_metrics = run_epoch(
            model, test_loader, None, loss_fn, train_cfg.device
        )

        train_mse_curve.append(train_metrics["mse"])
        test_mse_curve.append(test_metrics["mse"])

        if scheduler is not None:
            scheduler.step(test_metrics["mse"])

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[Final {epoch:03d}] "
                f"Train MSE={train_metrics['mse']:.6f} | "
                f"Internal Test MSE={test_metrics['mse']:.6f}"
            )

    return {
        "model_state": model.state_dict(),
        "train_mse_curve": train_mse_curve,
        "test_mse_curve": test_mse_curve,
    }



# -----------------------------
# Plot learning curves
# -----------------------------
import matplotlib.pyplot as plt

def plot_learning_curves(train_mse, test_mse, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_mse, label="Train MSE")
    plt.plot(test_mse, label="Test MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()





if __name__ == "__main__":

    # ---- DATA
    data_cfg = DataConfig(csv_path="./Datasets/ML-CUP25-TR.csv")
    data = make_splits_and_loaders(data_cfg)

    # ---- TRAIN BASE CONFIG
    base_train_cfg = TrainConfig(
        epochs=900,
        patience=30,
    )

    # ---- SEARCH
    best = adaptive_random_search(
        data,
        base_train_cfg,
        n_trials=40,
        top_k=5,
    )

    # ---- EVALUATION ON TRAIN / VAL / TEST
    best_model = MLPRegressor(MLPConfig(**best["model_cfg"]))
    best_model.load_state_dict(best["model_state"])

    device = base_train_cfg.device

    train_metrics = evaluate_model(best_model, data["train_loader"], device)
    val_metrics = evaluate_model(best_model, data["val_loader"], device)
    test_metrics = evaluate_model(best_model, data["test_loader"], device)

    final_report = {
        "model_cfg": best["model_cfg"],
        "train_cfg": best["train_cfg"],
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "best_epoch": best["best_epoch"],
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/best_model.json", "w") as f:
        json.dump(final_report, f, indent=2)

    torch.save(best["model_state"], "artifacts/best_model.pt")

    print("\n=== FINAL RESULTS ===")
    print("Train MEE:", train_metrics["mee"])
    print("Val   MEE:", val_metrics["mee"])
    print("Test  MEE:", test_metrics["mee"])

    # ---- FINAL RETRAIN
    trainval_loader = make_trainval_loader(
        data,
        batch_size=data_cfg.batch_size,
    )

    final = retrain_final_model(
        MLPConfig(**best["model_cfg"]),
        TrainConfig(**best["train_cfg"]),
        trainval_loader,
        data["test_loader"],
    )

    plot_learning_curves(
        final["train_mse_curve"],
        final["test_mse_curve"],
        "artifacts/learning_curves.png",
    )

    torch.save(final["model_state"], "artifacts/final_model.pt")