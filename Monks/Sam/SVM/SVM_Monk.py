import numpy as np
import pandas as pd
import json
import itertools

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================
# 1. DATASET + ONE-HOT
# ============================================

def load_monk_pandas(path: str):
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
    y = df[0].astype(int).values.astype(np.int32)
    X_cat = df.iloc[:, 1:7]
    X_oh = pd.get_dummies(X_cat, columns=X_cat.columns)
    return X_oh.values.astype(np.float32), y


# ============================================
# 2. CV ACCURACY
# ============================================

def cv_accuracy(X, y, model, cv_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    accs = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        accs.append(np.mean(pred == y_va))

    return float(np.mean(accs)), float(np.std(accs))


# ============================================
# 3. MAIN
# ============================================

if __name__ == "__main__":

    train_path = "./Datasets/monks-1.train"
    test_path  = "./Datasets/monks-1.test"

    X_full, y_full = load_monk_pandas(train_path)
    X_test, y_test = load_monk_pandas(test_path)

    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC())
    ])

    # ============================================
    # 4. GRID 
    # ============================================

    grids = []

    # LINEAR
    for C in [0.1, 1, 10]:
        grids.append({
            "kernel": "linear",
            "C": C
        })

    # RBF
    for C, gamma in itertools.product([0.1, 1, 10], [0.01, 0.1, 1]):
        grids.append({
            "kernel": "rbf",
            "C": C,
            "gamma": gamma
        })

    # POLY
    for C, deg, coef0 in itertools.product([0.1, 1, 10], [2, 3], [0.0, 1.0]):
        grids.append({
            "kernel": "poly",
            "C": C,
            "degree": deg,
            "coef0": coef0,
            "gamma": "scale"
        })

    # ============================================
    # 5. GRID SEARCH
    # ============================================

    best = None
    best_mean = -1.0
    results_log = []

    for cfg in grids:
        params = {
            "svm__kernel": cfg["kernel"],
            "svm__C": cfg["C"]
        }

        if cfg["kernel"] == "rbf":
            params["svm__gamma"] = cfg["gamma"]

        if cfg["kernel"] == "poly":
            params["svm__degree"] = cfg["degree"]
            params["svm__coef0"]  = cfg["coef0"]
            params["svm__gamma"]  = cfg["gamma"]

        model = base_pipe.set_params(**params)

        mean_acc, std_acc = cv_accuracy(X_full, y_full, model)

        results_log.append({
            **cfg,
            "cv_mean_accuracy": mean_acc,
            "cv_std_accuracy": std_acc
        })

        if mean_acc > best_mean:
            best_mean = mean_acc
            best = cfg

        print(f"{cfg} → {mean_acc:.4f} ± {std_acc:.4f}")

    with open("svm_monks_grid.json", "w") as f:
        json.dump(results_log, f, indent=4)

    print("\nBEST CONFIG:", best)

    # ============================================
    # 6. FINAL TRAIN + TEST
    # ============================================

    final_params = {
        "svm__kernel": best["kernel"],
        "svm__C": best["C"]
    }

    if best["kernel"] == "rbf":
        final_params["svm__gamma"] = best["gamma"]

    if best["kernel"] == "poly":
        final_params["svm__degree"] = best["degree"]
        final_params["svm__coef0"]  = best["coef0"]
        final_params["svm__gamma"]  = best["gamma"]

    final_model = base_pipe.set_params(**final_params)
    final_model.fit(X_full, y_full)

    test_pred = final_model.predict(X_test)
    test_acc = float(np.mean(test_pred == y_test))

    summary = {
        "best_config": best,
        "final_test_accuracy": test_acc
    }

    with open("svm_monks_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nFINAL TEST ACC:", test_acc)
