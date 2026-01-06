import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import itertools
import json


# =========================
# 1) DATA
# =========================

def load_cup_dataset(path: str):
    df = pd.read_csv(path, comment="#", header=None)
    X = df.iloc[:, 1:13].values.astype(np.float32)
    Y = df.iloc[:, 13:17].values.astype(np.float32)
    return X, Y


def split_cup_dataset(X, Y, seed=42):
    X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(
        X, Y, test_size=0.4, random_state=seed, shuffle=True
    )
    X_va, X_ts, Y_va, Y_ts = train_test_split(
        X_tmp, Y_tmp, test_size=0.5, random_state=seed, shuffle=True
    )
    return X_tr, Y_tr, X_va, Y_va, X_ts, Y_ts


def scale_cup_data(X_tr, X_va, X_ts):
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_tr),
        scaler.transform(X_va),
        scaler.transform(X_ts),
        scaler
    )


# =========================
# 2) MODEL
# =========================

def make_svr(C=1.0, epsilon=0.1, kernel="rbf", gamma=None, degree=None, coef0=None):
    if kernel == "linear":
        return SVR(C=C, epsilon=epsilon, kernel=kernel)
    if kernel == "rbf":
        return SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
    if kernel == "poly":
        return SVR(
            C=C, epsilon=epsilon, kernel=kernel,
            gamma=gamma, degree=degree, coef0=coef0
        )
    raise ValueError(f"Unsupported kernel {kernel}")


def train_multioutput_svr(X_tr, Y_tr, params):
    models = []
    for i in range(4):
        svr = make_svr(**params)
        svr.fit(X_tr, Y_tr[:, i])
        models.append(svr)
    return models


def train_multioutput_svr_per_output(X_tr, Y_tr, params_list):
    models = []
    for i in range(4):
        svr = make_svr(**params_list[i])
        svr.fit(X_tr, Y_tr[:, i])
        models.append(svr)
    return models


def predict_multioutput(models, X):
    return np.column_stack([m.predict(X) for m in models])


# =========================
# 3) METRICS
# =========================

def mee_4d(Y_true, Y_pred):
    return float(np.mean(np.linalg.norm(Y_pred - Y_true, axis=1)))


def mae_1d(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


# =========================
# 4) GRID SEARCH
# =========================

def grid_search_global(X_tr, Y_tr, X_va, Y_va, grid, save_path=None):
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        models = train_multioutput_svr(X_tr, Y_tr, params)
        Y_va_pred = predict_multioutput(models, X_va)
        mee = mee_4d(Y_va, Y_va_pred)
        results.append({"params": params, "mee_val": mee})
    results.sort(key=lambda x: x["mee_val"])
    if save_path:
        json.dump(results, open(save_path, "w"), indent=4)
    return results


def grid_search_per_output(X_tr, Y_tr, X_va, Y_va, grid, out_idx, save_path=None):
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        svr = make_svr(**params)
        svr.fit(X_tr, Y_tr[:, out_idx])
        pred = svr.predict(X_va)
        mae = mae_1d(Y_va[:, out_idx], pred)
        results.append({"params": params, "mae_val": mae})
    results.sort(key=lambda x: x["mae_val"])
    if save_path:
        json.dump(results, open(save_path, "w"), indent=4)
    return results


# =========================
# 5) BLIND I/O
# =========================

def load_cup_blind(path):
    df = pd.read_csv(path, comment="#", header=None)
    return df.iloc[:, 0].values, df.iloc[:, 1:13].values.astype(np.float32)


def save_cup_predictions(path, ids, Y_pred, team_name):
    with open(path, "w") as f:
        f.write(f"# team: {team_name}\n")
        f.write("# ML-CUP25\n")
        f.write("# format: id,o1,o2,o3,o4\n")
        for i, y in zip(ids, Y_pred):
            f.write(",".join([str(i)] + [f"{v:.6f}" for v in y]) + "\n")


# =========================
# 6) MAIN
# =========================

if __name__ == "__main__":

    RBF_GRID = {
        "kernel": ["rbf"],
        "C": [10, 50, 100, 300],
        "epsilon": [0.01, 0.1, 0.2],
        "gamma": [0.01, 0.05, 0.1]
    }

    POLY_GRID = {
        "kernel": ["poly"],
        "C": [10, 50, 100],
        "epsilon": [0.01, 0.1, 0.2],
        "gamma": [0.01, 0.05],
        "degree": [2, 3],
        "coef0": [0.0, 1.0]
    }

    # Load & split
    X, Y = load_cup_dataset("Datasets/ML-CUP25-TR.csv")
    X_tr, Y_tr, X_va, Y_va, X_ts, Y_ts = split_cup_dataset(X, Y)
    X_tr_s, X_va_s, X_ts_s, _ = scale_cup_data(X_tr, X_va, X_ts)

    # GLOBAL
    global_best = grid_search_global(
        X_tr_s, Y_tr, X_va_s, Y_va, RBF_GRID,
        save_path="grid_global_rbf.json"
    )[0]

    # PER-OUTPUT RBF + POLY
    best_rbf, best_poly = [], []
    mae_rbf, mae_poly = [], []

    for i in range(4):
        rbf_i = grid_search_per_output(
            X_tr_s, Y_tr, X_va_s, Y_va,
            RBF_GRID, i, f"grid_output_{i}_rbf.json"
        )[0]
        poly_i = grid_search_per_output(
            X_tr_s, Y_tr, X_va_s, Y_va,
            POLY_GRID, i, f"grid_output_{i}_poly.json"
        )[0]

        best_rbf.append(rbf_i["params"])
        best_poly.append(poly_i["params"])
        mae_rbf.append(rbf_i["mae_val"])
        mae_poly.append(poly_i["mae_val"])

    # FINAL PER-OUTPUT SELECTION (kernel as hyperparameter)
    final_params_per_output = []
    for i in range(4):
        if mae_poly[i] < mae_rbf[i]:
            final_params_per_output.append(best_poly[i])
        else:
            final_params_per_output.append(best_rbf[i])

    # Evaluation (MODEL SELECTION METRICS)
    final_models = train_multioutput_svr_per_output(X_tr_s, Y_tr, final_params_per_output)
    mee_train = mee_4d(Y_tr, predict_multioutput(final_models, X_tr_s))
    mee_val = mee_4d(Y_va, predict_multioutput(final_models, X_va_s))
    mee_test = mee_4d(Y_ts, predict_multioutput(final_models, X_ts_s))

    json.dump({
        "final_params_per_output": final_params_per_output,
        "mee_train": mee_train,
        "mee_val": mee_val,
        "mee_test": mee_test
    }, open("svr_model_selection_metrics_final.json", "w"), indent=4)

    # RETRAIN TRAIN+VAL FOR BLIND
    X_trva = np.vstack([X_tr, X_va])
    Y_trva = np.vstack([Y_tr, Y_va])
    scaler_final = StandardScaler()
    X_trva_s = scaler_final.fit_transform(X_trva)

    final_models = train_multioutput_svr_per_output(
        X_trva_s, Y_trva, final_params_per_output
    )

    ids, X_blind = load_cup_blind("Datasets/ML-CUP25-TS.csv")
    Y_blind = predict_multioutput(final_models, scaler_final.transform(X_blind))

    save_cup_predictions(
        "teamname_ML-CUP25-TS.csv",
        ids,
        Y_blind,
        "TEAMNAME"
    )
