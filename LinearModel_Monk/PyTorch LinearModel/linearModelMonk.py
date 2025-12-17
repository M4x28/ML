import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import json


# ============================================
# 1. LETTURA DEL DATASET + ONE-HOT
# ============================================

def load_monk_pandas(path: str):
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
    y = df[0].astype(int).values.astype(np.float32)
    X_cat = df.iloc[:, 1:7]
    X_oh = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = X_oh.values.astype(np.float32)
    return X, y


# ============================================
# FUNZIONE PER FISSARE IL SEED
# ============================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================
# 2. DATASET PYTORCH
# ============================================

class MonkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================
# 3. MODELLO LINEARE
# ============================================

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# ============================================
# LINEAR BASIS EXPANSION -> Phi Quadratica
# ============================================

def polynomial_quadratic_expansion(X):
    N, d = X.shape
    features = [X]
    features.append(X ** 2)

    interactions = []
    for i in range(d):
        for j in range(i+1, d):
            interactions.append((X[:, i] * X[:, j]).reshape(N, 1))

    features.append(np.hstack(interactions))
    return np.hstack(features)


# ============================================
# 4. LOSS MEE
# ============================================

def mee(pred, target):
    return torch.mean(torch.abs(pred - target))


# ============================================
# 5. TRAINING DI UNA SINGOLA RUN
# ============================================

def train_one_run(model, train_loader, optimizer, epochs=100):
    model.train()
    epoch_losses = []
    for ep in range(epochs):
        batch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = mee(pred, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(float(np.mean(batch_losses)))
    return epoch_losses


# ============================================
# 6. VALIDATION
# ============================================

def evaluate_mee(model, val_loader):
    model.eval()
    total = 0
    count = 0

    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            total += torch.abs(pred - y).sum().item()
            count += len(y)

    return total / count


# ============================================
# 7. MAIN + GRID SEARCH MULTI-SEED + L2
# ============================================

if __name__ == "__main__":

    # ------------ CARICAMENTO ----------------
    train_path = "./Datasets/monks-3.train"
    test_path  = "./Datasets/monks-3.test"

    X_full, y_full = load_monk_pandas(train_path)
    #X_full = polynomial_quadratic_expansion(X_full)

    X_test, y_test = load_monk_pandas(test_path)
    #X_test = polynomial_quadratic_expansion(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, shuffle=True, random_state=42
    )

    train_ds = MonkDataset(X_train, y_train)
    val_ds   = MonkDataset(X_val,   y_val)
    test_ds  = MonkDataset(X_test,  y_test)

    input_dim = X_train.shape[1]

    # ------------ GRID SEARCH ---------------
    learning_rates = [1e-3, 5e-3, 1e-2, 5e-2]
    batch_sizes = [16, 32, 64]
    epoch_list = [100, 200, 400]
    l2_list = [0.0, 1e-4, 1e-3, 1e-2]

    best_config = None
    best_mee_score = float("inf")

    results_log = []   # JSON LOG

    for lr, bs, epochs, l2 in itertools.product(learning_rates, batch_sizes, epoch_list, l2_list):

        print(f"\nTesting lr={lr}, bs={bs}, epochs={epochs}, L2={l2}")

        val_scores = []
        run_losses = []

        for seed in [0, 1, 2, 3, 4]:
            set_seed(seed)

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

            model = LinearModel(input_dim)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)

            train_losses = train_one_run(model, train_loader, optimizer, epochs)
            val_error = evaluate_mee(model, val_loader)

            val_scores.append(val_error)
            run_losses.append(train_losses)

        mean_val = float(np.mean(val_scores))
        std_val  = float(np.std(val_scores))

        print(f"Validation MEE → mean={mean_val:.4f}, std={std_val:.4f}")

        results_log.append({
            "learning_rate": lr,
            "batch_size": bs,
            "epochs": epochs,
            "l2": l2,
            "mean_val_MEE": mean_val,
            "std_val_MEE": std_val,
            "individual_run_MEE": val_scores,
            "individual_training_losses": run_losses
        })

        if mean_val < best_mee_score:
            best_mee_score = mean_val
            best_config = (lr, bs, epochs, l2)

    # Salvataggio JSON GRID SEARCH COMPLETO
    with open("grid_search_results_M3.json", "w") as f:
        json.dump(results_log, f, indent=4)

    print("\n=======================================")
    print(" BEST CONFIG:", best_config)
    print(" BEST VAL MEE (MEDIA):", best_mee_score)
    print("=======================================\n")


    # ============================================
    # TRAINING FINALE + TEST E SALVATAGGIO RISULTATI
    # ============================================

    best_lr, best_bs, best_epochs, best_l2 = best_config

    print("Training final model on FULL TRAIN SET...")

    final_ds = MonkDataset(X_full, y_full)
    final_loader = DataLoader(final_ds, batch_size=best_bs, shuffle=True)

    final_model = LinearModel(input_dim)
    final_optimizer = torch.optim.SGD(final_model.parameters(), lr=best_lr, weight_decay=best_l2)

    final_losses = train_one_run(final_model, final_loader, final_optimizer, best_epochs)

    torch.save(final_model.state_dict(), "best_model_weights.pt")
    print("Salvati pesi finali in best_model_weights.pt")

    # ------------ TEST FINALE --------------------
    print("\nEvaluating on TEST set...\n")
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    test_mee = evaluate_mee(final_model, test_loader)

    with torch.no_grad():
        for x, y in test_loader:
            pred = final_model(x)
            preds = (pred >= 0.5).float()
            test_acc = (preds == y).float().mean().item()

    print("FINAL TEST MEE:", test_mee)
    print("FINAL TEST ACC:", test_acc)

    # ============================================
    # AGGIUNTA TEST MEE E ACCURACY AL JSON
    # ============================================

    summary = {
        "best_config": {
            "learning_rate": best_lr,
            "batch_size": best_bs,
            "epochs": best_epochs,
            "l2": best_l2
        },
        "best_val_MEE": best_mee_score,
        "final_test_MEE": float(test_mee),
        "final_test_accuracy": float(test_acc)
    }

    with open("grid_search_summary_M3.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nRisultati finali salvati in grid_search_summary.json\n")
