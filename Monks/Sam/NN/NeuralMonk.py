import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import itertools
import json
import matplotlib.pyplot as plt


# ============================================
# 1. DATASET + ONE-HOT
# ============================================

def load_monk_pandas(path: str, columns=None):
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
    y = df[0].astype(int).values.astype(np.float32)
    X_cat = df.iloc[:, 1:7]
    X_oh = pd.get_dummies(X_cat, columns=X_cat.columns)

    if columns is None:
        return X_oh.values.astype(np.float32), y, X_oh.columns
    else:
        X_oh = X_oh.reindex(columns=columns, fill_value=0)
        return X_oh.values.astype(np.float32), y


# ============================================
# DATASET
# ============================================

class MonkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================
# MLP
# ============================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation="tanh"):
        super().__init__()

        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError("Unsupported activation")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================
# LOSS & METRICS
# ============================================

loss_fn = nn.MSELoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss, tot_correct, n_samples = 0.0, 0, 0

    for x, y in loader:
        out = model(x)
        loss = loss_fn(out, y)
        preds = (out >= 0.5).float()
        
        tot_loss += loss.item() * len(x)
        tot_correct += (preds == y).sum().item()
        n_samples += len(x)

    return tot_loss / n_samples, tot_correct / n_samples


def train_one_epoch(model, loader, optimizer):
    model.train()
    tot_loss, tot_correct, n_samples = 0.0, 0, 0

    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        preds = (out >= 0.5).float()
        tot_loss += loss.item() * len(x)
        tot_correct += (preds == y).sum().item()
        n_samples += len(x)

    return tot_loss / n_samples, tot_correct / n_samples


# ============================================
# SEED
# ============================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    train_path = "../../Datasets/monks-2.train"
    test_path  = "../../Datasets/monks-2.test"

    IS_MONK3 = False       
    seeds = [0, 1, 2, 3, 4]
    epochs = 200
    batch_size = 256

    # -------- LOAD --------
    X_full, y_full, oh_cols = load_monk_pandas(train_path)
    X_test, y_test = load_monk_pandas(test_path, oh_cols)

    Xtr, Xval, ytr, yval = train_test_split(
        X_full, y_full, test_size=0.2, shuffle=True, random_state=42
    )

    train_ds = MonkDataset(Xtr, ytr)
    val_ds   = MonkDataset(Xval, yval)
    test_ds  = MonkDataset(X_test, y_test)

    # -------- GRID SEARCH --------
    hidden_units = [2, 4]
    learning_rates = [0.001, 0.005, 0.01, 0.05]

    
    weight_decays = [1e-5, 1e-4, 1e-3] if IS_MONK3 else [0.0]


    best_cfg = None
    best_val_mse = float("inf")
    grid_log = []

    for h, lr, wd in itertools.product(hidden_units, learning_rates, weight_decays):
        val_mses = []

        for seed in seeds:
            set_seed(seed)

            model = MLP(Xtr.shape[1], h)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=wd
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)

            for _ in range(epochs):
                train_one_epoch(model, train_loader, optimizer)

            val_mse, _ = evaluate(model, val_loader)
            val_mses.append(val_mse)

        mean_mse = float(np.mean(val_mses))
        std_mse  = float(np.std(val_mses))

        print(f"h={h}, lr={lr}, wd={wd} → val MSE {mean_mse:.4f} ± {std_mse:.4f}")

        grid_log.append({
            "hidden_units": h,
            "learning_rate": lr,
            "weight_decay": wd,
            "mean_val_MSE": mean_mse,
            "std_val_MSE": std_mse
        })

        if mean_mse < best_val_mse:
            best_val_mse = mean_mse
            best_cfg = (h, lr, wd)

    with open("nn_monks_grid.json", "w") as f:
        json.dump(grid_log, f, indent=4)

    print("\nBEST CONFIG:", best_cfg)

    # ============================================
    # FINAL TRAINING + CURVES
    # ============================================

    best_h, best_lr, best_wd = best_cfg

    all_tr_losses, all_tr_accs = [], []
    all_ts_losses, all_ts_accs = [], []

    for seed in seeds:
        set_seed(seed)

        model = MLP(X_full.shape[1], best_h)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=best_lr,
            momentum=0.9,
            weight_decay=best_wd
        )

        train_loader = DataLoader(
            MonkDataset(X_full, y_full),
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=256,
            shuffle=False
        )

        tr_losses, tr_accs = [], []
        ts_losses, ts_accs = [], []

        for _ in range(epochs):
            tr_l, tr_a = train_one_epoch(model, train_loader, optimizer)
            ts_l, ts_a = evaluate(model, test_loader)

            tr_losses.append(tr_l)
            tr_accs.append(tr_a)
            ts_losses.append(ts_l)
            ts_accs.append(ts_a)

        all_tr_losses.append(tr_losses)
        all_tr_accs.append(tr_accs)
        all_ts_losses.append(ts_losses)
        all_ts_accs.append(ts_accs)

    mean_tr_losses = np.mean(all_tr_losses, axis=0)
    mean_tr_accs   = np.mean(all_tr_accs, axis=0)
    mean_ts_losses = np.mean(all_ts_losses, axis=0)
    mean_ts_accs   = np.mean(all_ts_accs, axis=0)

    # -------- PLOT --------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(mean_tr_losses, label="Train")
    plt.plot(mean_ts_losses, label="Test")
    plt.title("MSE (mean over seeds)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mean_tr_accs, label="Train")
    plt.plot(mean_ts_accs, label="Test")
    plt.title("Accuracy (mean over seeds)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("nn_monks_learning_curves.png")
    plt.close()

    # -------- SUMMARY --------
    summary = {
        "best_config": {
            "hidden_units": best_h,
            "learning_rate": best_lr,
            "weight_decay": best_wd
        },
        "final_train_MSE": float(mean_tr_losses[-1]),
        "final_train_accuracy": float(mean_tr_accs[-1]),
        "final_test_MSE": float(mean_ts_losses[-1]),
        "final_test_accuracy": float(mean_ts_accs[-1])
    }

    with open("nn_monks_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nFINAL RESULTS (MEAN OVER SEEDS)")
    print(f"Train MSE: {mean_tr_losses[-1]:.4f}")
    print(f"Train ACC: {mean_tr_accs[-1]:.4f}")
    print(f"Test  MSE: {mean_ts_losses[-1]:.4f}")
    print(f"Test  ACC: {mean_ts_accs[-1]:.4f}")