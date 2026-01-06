import numpy as np
import pandas as pd
import json
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# ============================================
# 1. DATASET + ONE-HOT
# ============================================

def load_monk_pandas(path: str, columns=None):
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
    y = df[0].astype(int).values
    X_cat = df.iloc[:, 1:7]
    X_oh = pd.get_dummies(X_cat, columns=X_cat.columns)

    if columns is None:
        return X_oh.values.astype(np.float32), y, X_oh.columns
    else:
        X_oh = X_oh.reindex(columns=columns, fill_value=0)
        return X_oh.values.astype(np.float32), y


# ============================================
# SEED
# ============================================

def set_seed(seed):
    np.random.seed(seed)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    # ===== CONFIG =====
    train_path = "./Datasets/monks-3.train"
    test_path  = "./Datasets/monks-3.test"

    seeds = [0, 1, 2, 3, 4]

    # ===== LOAD DATA =====
    X_full, y_full, oh_cols = load_monk_pandas(train_path)
    X_test, y_test = load_monk_pandas(test_path, oh_cols)

    Xtr, Xval, ytr, yval = train_test_split(
        X_full, y_full, test_size=0.2, shuffle=True, random_state=42
    )

    # ============================================
    # GRID SEARCH
    # ============================================

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    }

    best_cfg = None
    best_val_acc = -1.0
    grid_log = []

    for cfg in itertools.product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["subsample"]
    ):
        n_estimators, max_depth, lr, subsample = cfg
        val_accs = []

        for seed in seeds:
            set_seed(seed)

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=subsample,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=seed
            )

            model.fit(Xtr, ytr)
            preds = model.predict(Xval)
            acc = accuracy_score(yval, preds)
            val_accs.append(acc)

        mean_acc = float(np.mean(val_accs))
        std_acc  = float(np.std(val_accs))

        print(
            f"n_est={n_estimators}, depth={max_depth}, "
            f"lr={lr}, subs={subsample} → "
            f"val ACC {mean_acc:.4f} ± {std_acc:.4f}"
        )

        grid_log.append({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": lr,
            "subsample": subsample,
            "mean_val_accuracy": mean_acc,
            "std_val_accuracy": std_acc
        })

        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            best_cfg = (n_estimators, max_depth, lr, subsample)

    with open("xgboost_monks_grid.json", "w") as f:
        json.dump(grid_log, f, indent=4)

    print("\nBEST CONFIG:", best_cfg)

    # ============================================
    # FINAL TRAINING + LEARNING CURVES
    # ============================================

    best_n, best_d, best_lr, best_sub = best_cfg

    all_tr_losses, all_va_losses = [], []
    all_tr_accs,   all_va_accs   = [], []

    for seed in seeds:
        set_seed(seed)

        model = XGBClassifier(
            n_estimators=best_n,
            max_depth=best_d,
            learning_rate=best_lr,
            subsample=best_sub,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed
        )

        model.fit(
            X_full, y_full,
            eval_set=[(X_full, y_full), (X_test, y_test)],
            verbose=False
        )

        history = model.evals_result()

        tr_loss = history["validation_0"]["logloss"]
        ts_loss = history["validation_1"]["logloss"]

        all_tr_losses.append(tr_loss)
        all_va_losses.append(ts_loss)

        tr_acc, ts_acc = [], []

        for i in range(1, best_n + 1):
            model.set_params(n_estimators=i)
            model.fit(X_full, y_full, verbose=False)

            tr_pred = model.predict(X_full)
            ts_pred = model.predict(X_test)

            tr_acc.append(accuracy_score(y_full, tr_pred))
            ts_acc.append(accuracy_score(y_test, ts_pred))

        all_tr_accs.append(tr_acc)
        all_va_accs.append(ts_acc)

    # ===== MEAN CURVES =====
    mean_tr_loss = np.mean(all_tr_losses, axis=0)
    mean_ts_loss = np.mean(all_va_losses, axis=0)
    mean_tr_acc  = np.mean(all_tr_accs, axis=0)
    mean_ts_acc  = np.mean(all_va_accs, axis=0)

    # ============================================
    # PLOT
    # ============================================

    x = np.arange(1, best_n + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, mean_tr_loss, label="Train")
    plt.plot(x, mean_ts_loss, label="Test")
    plt.xlabel("Number of Trees")
    plt.ylabel("Log-loss")
    plt.title("XGBoost Learning Curve (Loss)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, mean_tr_acc, label="Train")
    plt.plot(x, mean_ts_acc, label="Test")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.title("XGBoost Learning Curve (Accuracy)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("xgboost_monks_learning_curves.png")
    plt.close()

    # ============================================
    # SUMMARY
    # ============================================

    summary = {
        "best_config": {
            "n_estimators": best_n,
            "max_depth": best_d,
            "learning_rate": best_lr,
            "subsample": best_sub
        },
        "final_train_accuracy": float(mean_tr_acc[-1]),
        "final_test_accuracy": float(mean_ts_acc[-1])
    }

    with open("xgboost_monks_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nFINAL RESULTS (MEAN OVER SEEDS)")
    print(f"Train ACC: {mean_tr_acc[-1]:.4f}")
    print(f"Test  ACC: {mean_ts_acc[-1]:.4f}")
