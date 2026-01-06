from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

# -----------------------------
# robust imports (Leo project)
# -----------------------------
LEO_ROOT = Path(__file__).resolve().parents[1]  # .../Cup/Leo
for rel in ["Ensemble", "KNN", "MLP", ""]:
    p = (LEO_ROOT / rel).resolve()
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ensemble_io import read_cup_train, split_train_val_test  # noqa: E402
from knn_model import build_knn_pipeline  # noqa: E402
from mlp_model import CupMLPModel  # noqa: E402


def mse_np(yhat: np.ndarray, y: np.ndarray) -> float:
    yhat = np.asarray(yhat).reshape(-1)
    y = np.asarray(y).reshape(-1)
    return float(np.mean((yhat - y) ** 2))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ema(xs, alpha=0.2):
    out = []
    for x in xs:
        x = float(x)
        out.append(x if not out else alpha * x + (1 - alpha) * out[-1])
    return out

# -----------------------------
# 1) NN curve: MSE vs epoch (train + val), max 47 epochs
# (coerente con ensemble_retrain_fulltr.py: StandardScaler on X)
# -----------------------------
def nn_mse_epoch_curve(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    epochs: int,                 # e.g. 200
    batch_size: int,
    hidden_sizes: List[int],
    activation: str,
    dropout: float,
    lr: float,
    weight_decay: float,
    optimizer: str,
    out_png: Path,
    device: str,
    plot_until: int | None = None,   # e.g. 47
    best_epoch: int | None = None,   # e.g. 47
    ema_alpha: float = 0.15,         # EMA smoothing (visualization only)
) -> None:
    def ema(xs: List[float], alpha: float) -> List[float]:
        out: List[float] = []
        for x in xs:
            x = float(x)
            out.append(x if not out else alpha * x + (1.0 - alpha) * out[-1])
        return out

    # preprocess (StandardScaler on X) - fit ONLY on train
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32)
    ytr_t = torch.tensor(ytr.reshape(-1, 1).astype(np.float32), dtype=torch.float32)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32)
    yva_t = torch.tensor(yva.reshape(-1, 1).astype(np.float32), dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    val_ds = torch.utils.data.TensorDataset(Xva_t, yva_t)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = CupMLPModel(
        input_dim=int(Xtr.shape[1]),
        output_dim=1,
        hidden_sizes=hidden_sizes,
        activation=activation,
        dropout=float(dropout),
        lr=float(lr),
        weight_decay=float(weight_decay),
        optimizer=optimizer,
        momentum=0.9,  # ignored for adam
    ).to(device)

    if optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    elif optimizer.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.9, weight_decay=float(weight_decay))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # MSE loss, per-batch mean; we will compute epoch mean weighted by batch size
    loss_fn = nn.MSELoss(reduction="mean")

    train_epoch_raw: List[float] = []
    val_epoch_raw: List[float] = []

    for _ in range(int(epochs)):
        # -------- train: mean loss over epoch (weighted) --------
        model.train()
        sum_loss = 0.0
        n_seen = 0

        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            bs = int(xb.size(0))
            sum_loss += float(loss.item()) * bs
            n_seen += bs

        train_epoch_raw.append(sum_loss / max(1, n_seen))

        # -------- val: mean loss over epoch (weighted) --------
        model.eval()
        sum_vloss = 0.0
        n_vseen = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                bs = int(xb.size(0))
                sum_vloss += float(loss.item()) * bs
                n_vseen += bs

        val_epoch_raw.append(sum_vloss / max(1, n_vseen))

    # EMA smoothing ONLY for visualization
    train_epoch_s = ema(train_epoch_raw, alpha=float(ema_alpha)) if ema_alpha and ema_alpha > 0 else train_epoch_raw
    val_epoch_s = ema(val_epoch_raw, alpha=float(ema_alpha)) if ema_alpha and ema_alpha > 0 else val_epoch_raw

    # --- plotting (truncate if requested) ---
    n_plot = int(epochs) if plot_until is None else min(int(plot_until), int(epochs))
    ep_axis = list(range(1, n_plot + 1))

    ensure_dir(out_png.parent)
    plt.figure()

    # raw (faint) + smooth (main)
    plt.plot(ep_axis, train_epoch_raw[:n_plot], alpha=0.25, label="train_epoch_mean (raw)")
    plt.plot(ep_axis, val_epoch_raw[:n_plot], alpha=0.25, label="val_epoch_mean (raw)")
    plt.plot(ep_axis, train_epoch_s[:n_plot], label=f"train_epoch_mean (EMA α={ema_alpha})")
    plt.plot(ep_axis, val_epoch_s[:n_plot], label=f"val_epoch_mean (EMA α={ema_alpha})")

    if best_epoch is not None and 1 <= int(best_epoch) <= n_plot:
        plt.axvline(int(best_epoch), linestyle="--", alpha=0.7, label=f"best epoch={best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"NN Target 1")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# 2) SVR curve: MSE vs C (gamma fixed), train + val + test
# (coerente: scaler + poly2 + pca 0.98 + SVR rbf)
# -----------------------------
def build_svr_pipeline(C: float, gamma: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("pca", PCA(n_components=0.98)),
            ("svr", SVR(C=float(C), epsilon=0.1, gamma=float(gamma), kernel="rbf", degree=3, coef0=0.0)),
        ]
    )


def svr_mse_vs_C_curve_train_val_test(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    *,
    gamma: float,
    C_values: List[float],
    out_png: Path,
) -> None:
    tr_mse, va_mse, te_mse = [], [], []

    for C in C_values:
        m = build_svr_pipeline(C=C, gamma=gamma)
        m.fit(Xtr, ytr)
        tr_mse.append(mse_np(m.predict(Xtr), ytr))
        va_mse.append(mse_np(m.predict(Xva), yva))
        te_mse.append(mse_np(m.predict(Xte), yte))

    ensure_dir(out_png.parent)
    plt.figure()
    #plt.semilogx(C_values, tr_mse, marker="o", label="train_mse")
    #plt.semilogx(C_values, va_mse, marker="o", label="val_mse")
    plt.semilogx(C_values, te_mse, marker="o", label="test_mse")
    plt.xlabel("C (log scale)")
    plt.ylabel("MSE")
    plt.title(f"SVR t0: MSE vs C (gamma={gamma})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# 3) KNN curve: val MSE vs K
# (coerente: weights=distance, poly2, pca 0.95, scaler power)
# -----------------------------
KNN_BASE_PARAMS = {
    "weights": "distance",
    "p": 2,
    "metric": "minkowski",
    "algorithm": "auto",
    "leaf_size": 30,
    "feature_map": "poly2",
    "pca_components": 0.95,
    "scaler_type": "power",
}


def knn_val_mse_vs_K_curve(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    target_name: str,
    K_values: List[int],
    out_png: Path,
    n_jobs: int,
) -> None:
    val_mse = []
    for k in K_values:
        params = dict(KNN_BASE_PARAMS)
        params["n_neighbors"] = int(k)
        model = build_knn_pipeline(params=params, scale_inputs=True, n_jobs=n_jobs)
        model.fit(Xtr, ytr)
        val_mse.append(mse_np(model.predict(Xva), yva))

    val_mse_arr = np.asarray(val_mse, dtype=float)
    best_idx = int(np.argmin(val_mse_arr))
    best_k = int(K_values[best_idx])
    best_mse = float(val_mse_arr[best_idx])

    ensure_dir(out_png.parent)
    plt.figure()
    plt.plot(K_values, val_mse, marker="o", label="val_mse")

    # mark best K
    plt.axvline(best_k, linestyle="--", alpha=0.7, label=f"best K={best_k}")
    plt.scatter([best_k], [best_mse])

    # "sull'asse X voglio il valore del min MSE"
    plt.xlabel(f"K (n_neighbors) | min val MSE={best_mse:.4f} (K={best_k})")
    plt.ylabel("MSE")
    plt.title(f"KNN {target_name}: val MSE vs K")

    # force xticks + best marker
    plt.xticks(K_values, [str(k) if k != best_k else f"{k}*" for k in K_values])

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--outdir", type=str, default="curves/ensemble_single_models")
    ap.add_argument("--knn-n-jobs", type=int, default=-1)

    # NN params (t1) - DEFAULT = tuoi iperparametri
    ap.add_argument("--nn-batch-size", type=int, default=150)
    ap.add_argument("--nn-hidden-sizes", type=str, default="256,128,64")
    ap.add_argument("--nn-activation", type=str, default="tanh")
    ap.add_argument("--nn-dropout", type=float, default=0.0)
    ap.add_argument("--nn-lr", type=float, default=0.005)
    ap.add_argument("--nn-weight-decay", type=float, default=0.0001)
    ap.add_argument("--nn-optimizer", type=str, default="adam")

    # SVR params (t0) - gamma fisso = tuo iperparametro
    ap.add_argument("--svr-gamma", type=float, default=0.01)
    # include C=100 (il tuo), più vicinato per la curva
    ap.add_argument("--svr-C-values", type=str, default="1,3,10,30,100,300,1000")

    # KNN sweep K (include 3 che è il tuo)
    ap.add_argument("--knn-K-values", type=str, default="1,2,3,5,7,9,11,15,21,31")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    _, X, y = read_cup_train(args.train_path)

    split = split_train_val_test(
        X, y, np.arange(X.shape[0]),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    Xtr, ytr = split["train"]["X"], split["train"]["y"]
    Xva, yva = split["val"]["X"], split["val"]["y"]
    Xte, yte = split["test"]["X"], split["test"]["y"]

    if Xva is None or yva is None:
        raise ValueError("val_ratio must be > 0 to generate validation curves.")
    if Xte is None or yte is None:
        raise ValueError("test_ratio must be > 0 to generate SVR train/val/test curves.")

    outdir = ensure_dir(Path(args.outdir))

    # NN: limit to 47
    hidden_sizes = [int(x) for x in args.nn_hidden_sizes.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nn_mse_epoch_curve(
        Xtr=Xtr,
        ytr=ytr[:, 1],
        Xva=Xva,
        yva=yva[:, 1],
        epochs=200,
        plot_until=100,
        best_epoch=47,
        batch_size=int(args.nn_batch_size),
        hidden_sizes=hidden_sizes,
        activation=str(args.nn_activation),
        dropout=float(args.nn_dropout),
        lr=float(args.nn_lr),
        weight_decay=float(args.nn_weight_decay),
        optimizer=str(args.nn_optimizer),
        out_png=outdir / "nn_t1_mse_vs_epoch.png",
        device=device,
    )

    # SVR: train + val + test vs C (gamma fixed)
    C_values = [float(x.strip()) for x in args.svr_C_values.split(",") if x.strip()]
    svr_mse_vs_C_curve_train_val_test(
        Xtr=Xtr, ytr=ytr[:, 0],
        Xva=Xva, yva=yva[:, 0],
        Xte=Xte, yte=yte[:, 0],
        gamma=float(args.svr_gamma),
        C_values=C_values,
        out_png=outdir / f"svr_t0_mse_vs_C_gamma{args.svr_gamma}.png",
    )

    # KNN: val MSE vs K (targets 2 and 3)
    K_values = [int(x.strip()) for x in args.knn_K_values.split(",") if x.strip()]
    knn_val_mse_vs_K_curve(
        Xtr=Xtr, ytr=ytr[:, 2],
        Xva=Xva, yva=yva[:, 2],
        target_name="t2",
        K_values=K_values,
        out_png=outdir / "knn_t2_val_mse_vs_K.png",
        n_jobs=int(args.knn_n_jobs),
    )
    knn_val_mse_vs_K_curve(
        Xtr=Xtr, ytr=ytr[:, 3],
        Xva=Xva, yva=yva[:, 3],
        target_name="t3",
        K_values=K_values,
        out_png=outdir / "knn_t3_val_mse_vs_K.png",
        n_jobs=int(args.knn_n_jobs),
    )

    print(f"[OK] Saved plots in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
