from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data" / "ML-CUP25-TR.csv").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "ML-CUP25-TR.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_EXPORT_PATH = REPO_ROOT / "Cup" / "Leo" / "Ensemble" / "models" / "ae_nn" / "ae_nn_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoencoder + NN grid search.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to ML-CUP25-TR.csv.")
    parser.add_argument("--output-dir", type=str, default=None, help="Root dir for run outputs.")
    parser.add_argument("--export-path", type=str, default=None, help="Path for the exported model.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split and init.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda.")
    return parser.parse_args()


def load_cup_tr(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ML-CUP training data (X, y) from CSV."""
    df = pd.read_csv(csv_path, comment="#", header=None)
    if df.shape[1] != 17:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]} (expected 17)")
    X = df.iloc[:, 1:13].to_numpy(dtype=np.float32)
    y = df.iloc[:, 13:17].to_numpy(dtype=np.float32)
    return X, y


def mee(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean Error across output dimensions."""
    return torch.mean(torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=1)))


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 12, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class Regressor(nn.Module):
    def __init__(self, latent_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_autoencoder(
    model: AutoEncoder,
    X_tr: torch.Tensor,
    X_val: torch.Tensor,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 20,
) -> AutoEncoder:
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), X_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), X_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_regressor(
    model: Regressor,
    Z_tr: torch.Tensor,
    y_tr: torch.Tensor,
    Z_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 30,
    track_curves: bool = False,
) -> tuple[Regressor, float, list[float], list[float]]:
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_mee = float("inf")
    best_state = None
    wait = 0

    train_mee_curve: list[float] = []
    val_mee_curve: list[float] = []

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Z_tr)
        train_loss = loss_fn(pred, y_tr)
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            tr_mee = mee(model(Z_tr), y_tr).item()
            vl_mee = mee(model(Z_val), y_val).item()

        if track_curves:
            train_mee_curve.append(tr_mee)
            val_mee_curve.append(vl_mee)

        if vl_mee < best_val_mee:
            best_val_mee = vl_mee
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_mee, train_mee_curve, val_mee_curve


def _state_dict_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state.items()}


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    device = torch.device(args.device)

    np.random.seed(seed)
    torch.manual_seed(seed)

    data_path = Path(args.data_path).resolve() if args.data_path else DEFAULT_DATA_PATH
    run_id = f"ae_nn_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_root = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    export_path = Path(args.export_path).resolve() if args.export_path else DEFAULT_EXPORT_PATH
    export_path.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_cup_tr(data_path)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=seed
    )
    X_val, X_ts, y_val, y_ts = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_ts = scaler.transform(X_ts)

    X_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_ts = torch.tensor(X_ts, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    y_ts = torch.tensor(y_ts, dtype=torch.float32, device=device)

    latent_dims = [4, 8, 12, 16, 24, 32]
    dropout_rates = [0.0, 0.1, 0.2, 0.3]
    weight_decays = [0.0, 1e-5, 1e-4, 1e-3]

    results = []
    best_val_mee_overall = float("inf")
    best_config = None

    total_runs = len(latent_dims) * len(dropout_rates) * len(weight_decays)
    current_run = 0

    for d, dropout, wd in product(latent_dims, dropout_rates, weight_decays):
        current_run += 1
        print(f"[{current_run}/{total_runs}] latent_dim={d}, dropout={dropout}, weight_decay={wd}")

        ae = AutoEncoder(latent_dim=d).to(device)
        ae = train_autoencoder(ae, X_tr, X_val, epochs=300, lr=1e-3, patience=20)

        ae.eval()
        with torch.no_grad():
            Z_tr = ae.encoder(X_tr)
            Z_val = ae.encoder(X_val)
            Z_ts = ae.encoder(X_ts)

        reg = Regressor(latent_dim=d, dropout=dropout).to(device)
        reg, best_val_mee, _, _ = train_regressor(
            reg,
            Z_tr,
            y_tr,
            Z_val,
            y_val,
            epochs=500,
            lr=1e-3,
            weight_decay=wd,
            patience=30,
            track_curves=False,
        )

        reg.eval()
        with torch.no_grad():
            train_mee = mee(reg(Z_tr), y_tr).item()
            val_mee = mee(reg(Z_val), y_val).item()
            test_mee = mee(reg(Z_ts), y_ts).item()

        gap = val_mee - train_mee
        gap_pct = (gap / train_mee) * 100 if train_mee > 0 else float("inf")

        results.append(
            {
                "latent_dim": d,
                "dropout": dropout,
                "weight_decay": float(wd),
                "train_mee": train_mee,
                "val_mee": val_mee,
                "test_mee": test_mee,
                "gap_pct": gap_pct,
            }
        )

        if val_mee < best_val_mee_overall:
            best_val_mee_overall = val_mee
            best_config = {
                "latent_dim": d,
                "dropout": dropout,
                "weight_decay": float(wd),
                "val_mee": val_mee,
                "test_mee": test_mee,
            }

    df = pd.DataFrame(results).sort_values("val_mee")
    df.to_csv(run_dir / "grid_search_results.csv", index=False)

    best_per_latent = df.groupby("latent_dim")["val_mee"].min().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(best_per_latent["latent_dim"], best_per_latent["val_mee"], marker="o", linewidth=2)
    plt.xlabel("Latent Dimension Size")
    plt.ylabel("Best Validation MEE")
    plt.title("Model Selection: Best Validation MEE vs Latent Dimension")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "latent_dim_sweep.png", dpi=150)
    plt.close()

    if best_config is None:
        raise RuntimeError("No valid configuration found in the grid search.")

    ae_final = AutoEncoder(latent_dim=best_config["latent_dim"]).to(device)
    ae_final = train_autoencoder(ae_final, X_tr, X_val, epochs=300, lr=1e-3, patience=20)

    ae_final.eval()
    with torch.no_grad():
        Z_tr_f = ae_final.encoder(X_tr)
        Z_val_f = ae_final.encoder(X_val)
        Z_ts_f = ae_final.encoder(X_ts)

    reg_final = Regressor(
        latent_dim=best_config["latent_dim"],
        dropout=best_config["dropout"],
    ).to(device)

    reg_final, _, train_mee_curve, val_mee_curve = train_regressor(
        reg_final,
        Z_tr_f,
        y_tr,
        Z_val_f,
        y_val,
        epochs=500,
        lr=1e-3,
        weight_decay=best_config["weight_decay"],
        patience=30,
        track_curves=True,
    )

    reg_final.eval()
    with torch.no_grad():
        train_final_mee = mee(reg_final(Z_tr_f), y_tr).item()
        val_final_mee = mee(reg_final(Z_val_f), y_val).item()
        test_final_mee = mee(reg_final(Z_ts_f), y_ts).item()

    plt.figure(figsize=(10, 6))
    plt.plot(train_mee_curve, label="Train MEE (eval-mode)", linewidth=2)
    plt.plot(val_mee_curve, label="Validation MEE (eval-mode)", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MEE")
    plt.title(
        "Final Model Learning Curves\n"
        f"(latent_dim={best_config['latent_dim']}, dropout={best_config['dropout']}, L2={best_config['weight_decay']})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "final_model_learning_curve.png", dpi=150)
    plt.close()

    final_hyperparams = {
        "run_id": run_id,
        "seed": seed,
        "device": str(device),
        "model_type": "Autoencoder + Regressor (dropout + L2)",
        "selection_metric": "Validation MEE (eval-mode, dropout OFF)",
        "best_config": best_config,
        "final_performance": {
            "train_mee": train_final_mee,
            "val_mee": val_final_mee,
            "test_mee": test_final_mee,
        },
        "data_split": {
            "train": "60%",
            "validation": "20%",
            "test": "20%",
            "split_seed": seed,
        },
        "grid_search": {
            "latent_dims": latent_dims,
            "dropout_rates": dropout_rates,
            "weight_decays": [float(wd) for wd in weight_decays],
            "total_configurations": total_runs,
        },
        "autoencoder": {
            "architecture": {
                "encoder": [12, 64, 32, best_config["latent_dim"]],
                "decoder": [best_config["latent_dim"], 32, 64, 12],
            },
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "epochs": 300,
            "patience": 20,
            "loss": "MSE",
        },
        "regressor": {
            "architecture": [best_config["latent_dim"], 64, 64, 4],
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "weight_decay": best_config["weight_decay"],
            "dropout": best_config["dropout"],
            "epochs": 500,
            "patience": 30,
            "loss": "MSE (training)",
            "evaluation_metric": "MEE (eval-mode, dropout OFF)",
        },
        "dataset": str(data_path),
    }

    (run_dir / "final_model_hyperparams.json").write_text(
        json.dumps(final_hyperparams, indent=2),
        encoding="utf-8",
    )

    artifact_payload = {
        "run_id": run_id,
        "input_dim": int(X_tr.shape[1]),
        "output_dim": 4,
        "latent_dim": int(best_config["latent_dim"]),
        "dropout": float(best_config["dropout"]),
        "encoder_state_dict": _state_dict_cpu(ae_final.encoder.state_dict()),
        "regressor_state_dict": _state_dict_cpu(reg_final.state_dict()),
        "scaler": scaler,
        "params": {
            "latent_dim": int(best_config["latent_dim"]),
            "dropout": float(best_config["dropout"]),
            "weight_decay": float(best_config["weight_decay"]),
            "optimizer": "adam",
            "lr": 1e-3,
            "epochs": 500,
            "patience": 30,
        },
        "autoencoder_params": {
            "encoder_dims": [12, 64, 32, int(best_config["latent_dim"])],
            "decoder_dims": [int(best_config["latent_dim"]), 32, 64, 12],
            "optimizer": "adam",
            "lr": 1e-3,
            "epochs": 300,
            "patience": 20,
        },
        "metrics": {
            "train_mee": train_final_mee,
            "val_mee": val_final_mee,
            "test_mee": test_final_mee,
        },
        "split": {
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "seed": seed,
        },
        "data_path": str(data_path),
    }

    torch.save(artifact_payload, export_path)
    torch.save(artifact_payload, run_dir / "ae_nn_best.pt")

    summary = {
        "run_id": run_id,
        "output_dir": str(run_dir),
        "export_path": str(export_path),
        "metrics": artifact_payload["metrics"],
        "best_config": best_config,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Results saved to:", run_dir)
    print("Exported model:", export_path)


if __name__ == "__main__":
    main()
