import os
import json
import math
import itertools
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0) SEED
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 1) LETTURA CUP
# ============================================================

CUP_TR_PATH = "Datasets/ML-CUP25-TR.csv"
CUP_TS_PATH = "Datasets/ML-CUP25-TS.csv"

FEATURE_COLS = [f"F_{i}" for i in range(12)]
TARGET_COLS = [f"TARGET_{i}" for i in range(1, 5)]

def load_cup_train(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ritorna:
      - X: (N, 12)
      - Y: (N, 4)
      - ids: (N,)
    Gestisce le prime righe commentate con '#'.
    """
    df = pd.read_csv(path, comment="#", header=None)
    ids = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:1+12].to_numpy(dtype=np.float32)
    Y = df.iloc[:, 1+12:1+12+4].to_numpy(dtype=np.float32)
    return X, Y, ids

def load_cup_test(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blind test (TS): id + 12 feat (no target)
    """
    df = pd.read_csv(path, comment="#", header=None)
    ids = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:1+12].to_numpy(dtype=np.float32)
    return X, ids


# ============================================================
# 2) STANDARDIZZAZIONE
# ============================================================

class StandardScalerNumpy:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================
# 3) SPLIT TRAIN/VAL/TEST (internal test)
# ============================================================

def split_train_val_test(
    X: np.ndarray,
    Y: np.ndarray,
    ids: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Split semplice e riproducibile:
      - prima separa test
      - poi separa val dal restante train
    """
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and (val_ratio + test_ratio) < 1

    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)

    n_test = int(round(n * test_ratio))
    test_idx = perm[:n_test]
    rest_idx = perm[n_test:]


    n_val = int(round(n * val_ratio))

    val_idx = rest_idx[:n_val]
    train_idx = rest_idx[n_val:]

    def pack(indexes):
        return {"X": X[indexes], "Y": Y[indexes], "ids": ids[indexes]}

    return {
        "train": pack(train_idx),
        "val": pack(val_idx),
        "test": pack(test_idx),
    }


# ============================================================
# 4) DATASET PYTORCH
# ============================================================

class CupDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.Y = None if Y is None else torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is None:
            return self.X[idx]
        return self.X[idx], self.Y[idx]


# ============================================================
# 5) MODELLO LINEARE MULTI-OUTPUT
# ============================================================

class Linear4Out(nn.Module):
    def __init__(self, input_dim: int = 12, output_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# ============================================================
# 6) METRICHE: MSE (train), MEE (selection/report CUP)
# ============================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)

@torch.no_grad()
def mee_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Euclidean Error:
      mean_p || o_p - t_p ||_2
    """
    return torch.mean(torch.linalg.norm(pred - target, ord=2, dim=1))


# ============================================================
# 7) TRAIN / EVAL
# ============================================================

def build_optimizer(model: nn.Module, lr: float, l2: float) -> torch.optim.Optimizer:
    """
    Tikhonov/L2 = weight decay.
    Variante: applichiamo L2 solo ai pesi e NON ai bias (scelta comune).
    """
    weight_params = []
    bias_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("bias"):
            bias_params.append(p)
        else:
            weight_params.append(p)

    return torch.optim.SGD(
        [
            {"params": weight_params, "weight_decay": l2},
            {"params": bias_params, "weight_decay": 0.0},
        ],
        lr=lr
    )

def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = "cpu",
) -> List[float]:
    model.to(device)
    model.train()
    hist = []
    for _ in range(epochs):
        tot_loss, n_samples = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse_loss(pred, yb)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * len(xb)
            n_samples += len(xb)
        hist.append(tot_loss / n_samples)
    return hist

@torch.no_grad()
def evaluate_mee(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.to(device)
    model.eval()
    tot_euclidean_dist = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        # Somma le distanze euclidee di tutti gli esempi nel batch
        tot_euclidean_dist += torch.sum(torch.linalg.norm(pred - yb, ord=2, dim=1)).item()
        n_samples += len(xb)
    return tot_euclidean_dist / n_samples

@torch.no_grad()
def evaluate_mse(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.to(device)
    model.eval()
    tot_sq_err = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        tot_sq_err += torch.sum((pred - yb) ** 2).item()
        n_samples += len(xb)  
    return tot_sq_err / n_samples



@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    model.to(device)
    model.eval()
    outs = []
    for xb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        outs.append(pred)
    return np.vstack(outs)


# ============================================================
# 8) GRID SEARCH 
# ============================================================

@dataclass
class Config:
    lr: float
    batch_size: int
    epochs: int
    l2: float

def grid_search_multiseed(
    Xtr: np.ndarray, Ytr: np.ndarray,
    Xval: np.ndarray, Yval: np.ndarray,
    seeds: List[int],
    configs: List[Config],
    device: str = "cpu",
) -> Tuple[Config, Dict]:
    train_ds = CupDataset(Xtr, Ytr)
    val_ds = CupDataset(Xval, Yval)

    results = []
    best_cfg = None
    best_mean = float("inf")

    for cfg in configs:
        val_scores = []
        run_losses = []

        for s in seeds:
            set_seed(s)
            model = Linear4Out(input_dim=Xtr.shape[1], output_dim=Ytr.shape[1])
            opt = build_optimizer(model, lr=cfg.lr, l2=cfg.l2)

            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

            losses = train_one_run(model, train_loader, opt, epochs=cfg.epochs, device=device)
            val_mee = evaluate_mee(model, val_loader, device=device)

            val_scores.append(val_mee)
            run_losses.append(losses)

        mean_val = float(np.mean(val_scores))
        std_val = float(np.std(val_scores))

        results.append({
            "config": asdict(cfg),
            "val_MEE_mean": mean_val,
            "val_MEE_std": std_val,
            "val_MEE_per_seed": val_scores,
            "train_MSE_curve_per_seed": run_losses,
        })

        if mean_val < best_mean:
            best_mean = mean_val
            best_cfg = cfg

        print(f"[GRID] cfg={cfg} -> val MEE mean={mean_val:.6f} std={std_val:.6f}")

    assert best_cfg is not None
    log = {
        "best_config": asdict(best_cfg),
        "best_val_MEE_mean": best_mean,
        "all_results": results,
    }
    return best_cfg, log

def make_coarse_configs() -> List[Config]:
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    bss = [16, 32, 64, 128]
    epochs = [200, 500, 1000]
    l2s = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    return [Config(lr, bs, ep, l2) for lr, bs, ep, l2 in itertools.product(lrs, bss, epochs, l2s)]

def make_refine_configs(best: Config) -> List[Config]:
    """
    Raffina attorno al best:
      - lr: x0.5, x1, x2 (clamp)
      - l2: x0.1, x1, x10 (clamp)
      - batch/epochs: piccoli intorni
    """
    def uniq(vals):
        out = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    lr_cands = uniq([best.lr * 0.5, best.lr, best.lr * 2.0])
    lr_cands = [float(min(max(v, 1e-5), 5e-1)) for v in lr_cands]

    l2_cands = uniq([best.l2 * 0.1, best.l2, best.l2 * 10.0])
    l2_cands = [float(min(max(v, 0.0), 1e-1)) for v in l2_cands]
    if best.l2 == 0.0:
        l2_cands = [0.0, 1e-6, 1e-5, 1e-4]

    bs_cands = uniq([best.batch_size])
    if best.batch_size >= 32:
        bs_cands += [best.batch_size // 2]
    if best.batch_size <= 128:
        bs_cands += [best.batch_size * 2]
    bs_cands = [int(v) for v in bs_cands if v in [16, 32, 64, 128, 256]]

    ep_cands = uniq([best.epochs])
    # piccolo refine sugli epoch (senza esplodere)
    ep_cands += [max(100, int(best.epochs * 0.7)), int(best.epochs * 1.3)]
    ep_cands = [int(v) for v in ep_cands if 50 <= v <= 3000]

    return [Config(lr, bs, ep, l2) for lr, bs, ep, l2 in itertools.product(lr_cands, bs_cands, ep_cands, l2_cands)]


# ============================================================
# 9) RETRAIN FINALE MULTI-SEED SU TRAIN+VAL
# ============================================================

def retrain_and_test_multiseed(
    Xtrainval: np.ndarray, Ytrainval: np.ndarray,
    Xte: np.ndarray, Yte: np.ndarray,
    cfg: Config,
    seeds: List[int],
    device: str = "cpu",
) -> Dict:

    trainval_ds = CupDataset(Xtrainval, Ytrainval)
    test_ds     = CupDataset(Xte, Yte)

    train_loader = DataLoader(
        trainval_ds,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=len(test_ds),
        shuffle=False
)


    metrics = {
        "trainval_MEE": [],
        "test_MEE": [],
        "test_MSE": [],
        "train_curve_MSE": [],
        "test_curve_MSE": []
    }


    models_state = []

    for s in seeds:
        set_seed(s)

        model = Linear4Out(
            input_dim=Xtrainval.shape[1],
            output_dim=Ytrainval.shape[1]
        )

        opt = build_optimizer(model, lr=cfg.lr, l2=cfg.l2)

        train_curve = []
        test_curve = []

        for _ in range(cfg.epochs):
            model.train()
            tot_loss, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = mse_loss(pred, yb)
                loss.backward()
                opt.step()
                tot_loss += loss.item() * len(xb)
                n += len(xb)
            train_curve.append(tot_loss / n)

            test_curve.append(evaluate_mse(model, test_loader, device))

        metrics["train_curve_MSE"].append(train_curve)
        metrics["test_curve_MSE"].append(test_curve)

        metrics["trainval_MEE"].append(
            evaluate_mee(model, train_loader, device)
        )
        metrics["test_MEE"].append(
            evaluate_mee(model, test_loader, device)
        )
        metrics["test_MSE"].append(
            evaluate_mse(model, test_loader, device)
        )

        models_state.append(model.state_dict())

    # media sui seed
    mean_metrics = {
        k + "_mean": float(np.mean(v)) for k, v in metrics.items() if "curve" not in k
    }

    # curva media (epoch-wise)
    mean_train_curve = np.mean(metrics["train_curve_MSE"], axis=0).tolist()
    mean_test_curve  = np.mean(metrics["test_curve_MSE"], axis=0).tolist()

    return {
        "config": asdict(cfg),
        "metrics_mean": mean_metrics,
        "learning_curve": {
            "loss": "MSE",
            "train": mean_train_curve,
            "test": mean_test_curve,
            "epochs": len(mean_train_curve)
        },
        "models_state": models_state  
    }


# ============================================================
# 10) EXPORT CUP FILE (TS predictions)
# ============================================================

def write_cup_ts_output(
    out_path: str,
    ids: np.ndarray,
    preds: np.ndarray,
    names_line: str,
    team_nick: str,
    date_str: str,
    dataset_name: str = "ML-CUP25 v1",
):
    assert preds.shape[0] == len(ids) and preds.shape[1] == 4

    with open(out_path, "w", newline="\n") as f:
        f.write(f"# {names_line}\n")
        f.write(f"# {team_nick}\n")
        f.write(f"# {dataset_name}\n")
        f.write(f"# {date_str}\n")
        for i, p in zip(ids, preds):
            f.write(f"{int(i)},{p[0]},{p[1]},{p[2]},{p[3]}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    device = "cpu" 

    # 1) Carica training set
    X, Y, ids = load_cup_train(CUP_TR_PATH)
    print("Loaded TR:", X.shape, Y.shape)

    # 2) Split (train/val/test interno)
    splits = split_train_val_test(X, Y, ids, val_ratio=0.2, test_ratio=0.2, seed=42)
    Xtr, Ytr = splits["train"]["X"], splits["train"]["Y"]
    Xval, Yval = splits["val"]["X"], splits["val"]["Y"]
    Xte, Yte = splits["test"]["X"], splits["test"]["Y"]
    print("Split sizes:", len(Xtr), len(Xval), len(Xte))

    # 3) Standardizza X (fit su train)
    scaler = StandardScalerNumpy()
    Xtr_s = scaler.fit_transform(Xtr)
    Xval_s = scaler.transform(Xval)
    Xte_s = scaler.transform(Xte)

    # 4) GRID SEARCH multi-seed (coarse)
    seeds_gs = [0, 1, 2, 3, 4]
    coarse_configs = make_coarse_configs()
    print(f"Coarse grid size = {len(coarse_configs)}")

    best_coarse, log_coarse = grid_search_multiseed(
        Xtr_s, Ytr, Xval_s, Yval,
        seeds=seeds_gs,
        configs=coarse_configs,
        device=device,
    )

    with open("cup_linear_grid_coarse.json", "w") as f:
        json.dump(log_coarse, f, indent=2)

    # 5) GRID SEARCH adattiva (refine)
    refine_configs = make_refine_configs(best_coarse)
    print(f"Refine grid size = {len(refine_configs)}")

    best_refine, log_refine = grid_search_multiseed(
        Xtr_s, Ytr, Xval_s, Yval,
        seeds=seeds_gs,
        configs=refine_configs,
        device=device,
    )

    with open("cup_linear_grid_refine.json", "w") as f:
        json.dump(log_refine, f, indent=2)

    print("\n===== BEST (REFINE) =====")
    print(best_refine)
    print("val MEE mean:", log_refine["best_val_MEE_mean"])

    # 6) Retrain finale multi-seed su train+val, report su test interno
    Xtrainval = np.vstack([Xtr_s, Xval_s])
    Ytrainval = np.vstack([Ytr, Yval])

    final = retrain_and_test_multiseed(
        Xtrainval, Ytrainval,
        Xte_s, Yte,
        cfg=best_refine,
        seeds=[10, 11, 12, 13, 14],
        device=device,
    )



    

    # salva summary json (senza pesi)
    summary = {
        "best_config": final["config"],
        "final_model_metrics": {
            "MEE_TR+VL_mean": final["metrics_mean"]["trainval_MEE_mean"],
            "MEE_TS_mean": final["metrics_mean"]["test_MEE_mean"]
        },

        "learning_curve": final["learning_curve"],
        "notes": (
            "Metrics are averaged over multiple random seeds. "
            "Training performed with MSE loss, model selection and assessment based on MEE."
        )
    }

    with open("cup_linear_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== FINAL REPORT =====")
    print(json.dumps(summary, indent=2))

    # 7) (Opzionale) Predizioni su blind TS e scrittura file CUP
    if os.path.exists(CUP_TS_PATH):
        Xts, ids_ts = load_cup_test(CUP_TS_PATH)
        Xts_s = scaler.transform(Xts)

        model = Linear4Out(input_dim=12, output_dim=4)
        #model.load_state_dict(torch.load("cup_linear_best_model.pt", map_location="cpu"))

        ts_loader = DataLoader(CupDataset(Xts_s, None), batch_size=256, shuffle=False)
        preds_ts = predict(model, ts_loader, device=device)

        out_name = "team-name_ML-CUP25-TS.csv"
        write_cup_ts_output(
            out_path=out_name,
            ids=ids_ts,
            preds=preds_ts,
            names_line="YOUR NAME 1, YOUR NAME 2",
            team_nick="TEAMNICK",
            date_str="20 Dec 2025",
            dataset_name="ML-CUP25 v1",
        )
        print(f"\nWrote blind test predictions to: {out_name}")
    else:
        print("\nNOTE: TS file not found; skipping blind test export.")