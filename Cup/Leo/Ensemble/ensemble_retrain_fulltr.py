from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

# -----------------------------
# Path setup (fix import issues)
# -----------------------------
LEO_ROOT = Path(__file__).resolve().parents[1]


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data" / "ML-CUP25-TR.csv").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(LEO_ROOT)

# Make sibling folders importable (KNN, MLP, etc.)
for rel in ["", "KNN", "MLP", "Ensemble", "SVM", "SVM_Model_Cup", "MLP_Model_Cup"]:
    p = (LEO_ROOT / rel).resolve()
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Project imports
from ensemble_io import read_cup_train  # noqa: E402
from cup_metrics import mean_euclidean_error  # noqa: E402
from knn_model import build_knn_pipeline  # noqa: E402
from mlp_model import CupMLPModel  # noqa: E402

# -----------------------------
# Hyperparams (FINAL ensemble)
# -----------------------------
SVR_T0: dict[str, Any] = {
    "scaler": True,
    "feature_map": "poly2",
    "pca_components": 0.98,
    "svr": {
        "C": 100.0,
        "epsilon": 0.1,
        "gamma": 0.01,
        "kernel": "rbf",
        "degree": 3,
        "coef0": 0.0,
    },
}
NN_T1: dict[str, Any] = {
    "activation": "tanh",
    "batch_size": 64,
    "dropout": 0.0,
    "hidden_sizes": [256, 128, 64],
    "lr": 0.005,
    "momentum": None,
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "target_idx": 1,
    "per_target": True,
}
KNN_T2: dict[str, Any] = {
    "n_neighbors": 3,
    "weights": "distance",
    "p": 2,
    "metric": "minkowski",
    "algorithm": "auto",
    "leaf_size": 30,
    "feature_map": "poly2",
    "pca_components": 0.95,
    "scaler_type": "power",
}
KNN_T3: dict[str, Any] = {
    "n_neighbors": 3,
    "weights": "distance",
    "p": 2,
    "metric": "minkowski",
    "algorithm": "auto",
    "leaf_size": 30,
    "feature_map": "poly2",
    "pca_components": 0.95,
    "scaler_type": "power",
}


def default_data_paths() -> tuple[Path, Path]:
    train_path = REPO_ROOT / "data" / "ML-CUP25-TR.csv"
    test_path = REPO_ROOT / "data" / "ML-CUP25-TS.csv"
    return train_path, test_path


def build_svr_pipeline(params: dict[str, Any]) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if params.get("scaler", True):
        steps.append(("scaler", StandardScaler()))

    feature_map = params.get("feature_map", "identity")
    if feature_map == "poly2":
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    pca_value = params.get("pca_components")
    if pca_value is not None:
        steps.append(("pca", PCA(n_components=pca_value)))

    svr_params = dict(params.get("svr", {}))
    steps.append(("svr", SVR(**svr_params)))
    return Pipeline(steps)


class TrainCurveCallback(Callback):
    """Collect train_loss (=MSE) per epoch."""
    def __init__(self) -> None:
        super().__init__()
        self.epochs: list[int] = []
        self.mse: list[float] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        metrics = trainer.callback_metrics
        v = metrics.get("train_loss")  # train_loss == MSE in CupMLPModel
        if v is None:
            return
        if isinstance(v, torch.Tensor):
            v = float(v.detach().cpu())
        else:
            v = float(v)
        self.epochs.append(int(trainer.current_epoch) + 1)
        self.mse.append(v)


def save_learning_curve(out_png: Path, out_json: Path, epochs: list[int], mse: list[float]) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {"epoch": epochs, "mse": mse}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Plot MSE vs epoch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(epochs, mse)
    plt.xlabel("epoch")
    plt.ylabel("MSE (train_loss)")
    plt.title("NN target1 learning curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _round4(x: float) -> float:
    return float(f"{x:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlp-epochs", type=int, default=47)
    p.add_argument("--n-jobs", type=int, default=None, help="n_jobs for KNN (None keeps sklearn default).")
    p.add_argument(
        "--outdir",
        type=str,
        default=str(REPO_ROOT / "final_models" / "ensemble_retrained"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    seed_everything(seed, workers=True)

    train_path, _ = default_data_paths()
    ids, X, y = read_cup_train(train_path)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------
    # 1) SVR target 0
    # -----------------
    svr0 = build_svr_pipeline(SVR_T0)
    svr0.fit(X, y[:, 0])

    svr0_path = outdir / "svr_t0.joblib"
    joblib.dump({"model": svr0, "params": {"per_target": True, "target_idx": 0, **SVR_T0}}, svr0_path)

    # -----------------
    # 2) NN target 1
    # -----------------
    # Fit scaler on FULL TR and save it in the torch payload (so predict uses same preprocessing).
    nn_scaler = StandardScaler()
    X_nn = nn_scaler.fit_transform(X).astype(np.float32)

    X_tensor = torch.tensor(X_nn, dtype=torch.float32)
    y_tensor = torch.tensor(y[:, 1:2], dtype=torch.float32)  # (N,1)

    ds = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(NN_T1["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    curve_cb = TrainCurveCallback()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        max_epochs=int(args.mlp_epochs),
        accelerator=accelerator,
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        callbacks=[curve_cb],
    )

    nn = CupMLPModel(
        input_dim=int(X.shape[1]),
        output_dim=1,
        hidden_sizes=NN_T1["hidden_sizes"],
        activation=str(NN_T1["activation"]),
        dropout=float(NN_T1["dropout"]),
        lr=float(NN_T1["lr"]),
        weight_decay=float(NN_T1["weight_decay"]),
        optimizer=str(NN_T1["optimizer"]),
        momentum=0.9 if NN_T1.get("momentum") is None else float(NN_T1["momentum"]),
    )

    trainer.fit(nn, train_dataloaders=dl)

    nn_path = outdir / "nn_t1.pt"
    torch.save(
        {
            "input_dim": int(X.shape[1]),
            "output_dim": 1,
            "state_dict": nn.state_dict(),
            "scaler": nn_scaler,
            "params": {"per_target": True, "target_idx": 1, **NN_T1},
        },
        nn_path,
    )

    # learning curve artifacts
    lc_png = outdir / "nn_t1_learning_curve_mse.png"
    lc_json = outdir / "nn_t1_learning_curve_mse.json"
    save_learning_curve(lc_png, lc_json, curve_cb.epochs, curve_cb.mse)

    # -----------------
    # 3) KNN target 2
    # -----------------
    knn2 = build_knn_pipeline(params=KNN_T2, scale_inputs=True, n_jobs=args.n_jobs)
    knn2.fit(X, y[:, 2])

    knn2_path = outdir / "knn_t2.joblib"
    joblib.dump({"model": knn2, "params": {"per_target": True, "target_idx": 2, **KNN_T2}}, knn2_path)

    # -----------------
    # 4) KNN target 3
    # -----------------
    knn3 = build_knn_pipeline(params=KNN_T3, scale_inputs=True, n_jobs=args.n_jobs)
    knn3.fit(X, y[:, 3])

    knn3_path = outdir / "knn_t3.joblib"
    joblib.dump({"model": knn3, "params": {"per_target": True, "target_idx": 3, **KNN_T3}}, knn3_path)

    # -----------------
    # Registry JSON (for predict script)
    # -----------------
    registry = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "train_path": str(train_path),
        "models": [
            {"id": "svr_t0", "target_idx": 0, "model_type": "svr", "path": str(svr0_path)},
            {"id": "nn_t1", "target_idx": 1, "model_type": "mlp", "path": str(nn_path)},
            {"id": "knn_t2", "target_idx": 2, "model_type": "knn", "path": str(knn2_path)},
            {"id": "knn_t3", "target_idx": 3, "model_type": "knn", "path": str(knn3_path)},
        ],
        "artifacts": {
            "nn_t1_learning_curve_png": str(lc_png),
            "nn_t1_learning_curve_json": str(lc_json),
        },
    }

    registry_path = outdir / "ensemble_registry_retrained.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # -----------------
    # Quick train metrics (rounded to 4 decimals)
    # -----------------
    pred0 = svr0.predict(X).reshape(-1)
    with torch.no_grad():
        nn.eval()
        pred1 = nn(torch.tensor(X_nn, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    pred2 = knn2.predict(X).reshape(-1)
    pred3 = knn3.predict(X).reshape(-1)

    P_np = np.column_stack([pred0, pred1, pred2, pred3]).astype(np.float32)
    y_np = y.astype(np.float32)

    P_t = torch.from_numpy(P_np)
    y_t = torch.from_numpy(y_np)

    mee_full = mean_euclidean_error(P_t, y_t)
    print(f"[TR metrics] MEE(full TR) = {_round4(float(mee_full))}")
    print(f"[OK] Saved registry: {registry_path}")
    print(f"[OK] NN learning curve: {lc_png}")


if __name__ == "__main__":
    main()
