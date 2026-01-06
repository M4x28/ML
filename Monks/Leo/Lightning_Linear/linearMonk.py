"""
Pipeline Lightning per MONK con espansione quadratica (LBE), MSE come metrica
e repliche multi-seed.

Struttura:
- load_monk_pandas: lettura e one-hot dei dataset MONK.
- MonkDataset / MonkDataModule: gestione dataset, split opzionale val, LBE via PolynomialFeatures.
- MonkLinearModel: modello lineare con MSE e L2 opzionale (lambda) + accuracy.
- MSECurveCallback: salva il grafico MSE-epoche per train/test (test calcolato ogni epoca).
- AccuracyCurveCallback: salva il grafico Accuracy-epoche per train/test (test calcolato ogni epoca).
- run_grid_search_for_task: campiona N configurazioni random dalla griglia, esegue K seed,
  logga MSE su val/test, salva modelli ed aggrega media/dev std.
- main: esegue random search per i task richiesti (MONK-3 con e senza regolarizzazione).
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from copy import deepcopy


OUTPUT_ROOT = Path(__file__).resolve().parent


# -------------------------------------------------------------------
# Caricamento dati MONK
# -------------------------------------------------------------------
def load_monk_pandas(path: str):
    """
    Legge il file MONK, estrae y e one-hot sulle 6 colonne categoriali.
    Restituisce X float32 e y float32.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
    y = df[0].astype(int).values.astype(np.float32)

    X_cat = df.iloc[:, 1:7]
    X_oh = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = X_oh.values.astype(np.float32)
    return X, y


# -------------------------------------------------------------------
# Dataset + DataModule con espansione quadratica (kernel esplicito)
# -------------------------------------------------------------------
class MonkDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Dataset Torch semplice (features + label).
        """
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MonkDataModule(L.LightningDataModule):
    """
    DataModule con espansione quadratica (PolynomialFeatures) per simulare il kernel.
    Supporta split train/val opzionale (val_ratio=0 disabilita la val).
    """

    def __init__(
        self,
        train_path,
        test_path,
        batch_size=32,
        val_ratio=0.2,
        poly_degree=2,
        num_workers: int | None = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.poly_degree = poly_degree
        # Su Windows i DataLoader multiprocess possono causare errori; default a 0.
        self.num_workers = num_workers if num_workers is not None else 0
        self.pin_memory = pin_memory
        self.poly = None
        self.input_dim = None
        self.val_dataset = None

    def setup(self, stage=None):
        X_full, y_full = load_monk_pandas(self.train_path)
        self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        X_full = self.poly.fit_transform(X_full).astype(np.float32)
        self.input_dim = X_full.shape[1]

        if self.val_ratio and self.val_ratio > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_full,
                y_full,
                test_size=self.val_ratio,
                shuffle=True,
                random_state=42,
                stratify=y_full,
            )
            self.val_dataset = MonkDataset(X_val, y_val)
        else:
            X_train, y_train = X_full, y_full
            self.val_dataset = None

        X_test, y_test = load_monk_pandas(self.test_path)
        X_test = self.poly.transform(X_test).astype(np.float32)

        self.train_dataset = MonkDataset(X_train, y_train)
        self.test_dataset = MonkDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


# -------------------------------------------------------------------
# Modello lineare con MSE e regularizzazione opzionale
# -------------------------------------------------------------------
class MonkLinearModel(L.LightningModule):
    """
    Modello lineare per classificazione binaria con MSE (su output sigmoid).
    La regolarizzazione L2 e' opzionale (lambda).
    """

    def __init__(self, input_dim: int, lr: float = 1e-2, l2_reg: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(input_dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.linear(x).squeeze(1)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def _accuracy(preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred_labels = (preds >= 0.5).float()
        return pred_labels.eq(y).float().mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.predict_proba(x)
        mse = self.loss_fn(preds, y)
        acc = self._accuracy(preds, y)
        l2_penalty = self.hparams.l2_reg * torch.sum(self.linear.weight ** 2)
        loss = mse + l2_penalty
        self.log("train_mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.predict_proba(x)
        mse = self.loss_fn(preds, y)
        acc = self._accuracy(preds, y)
        self.log("val_mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=False, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.predict_proba(x)
        mse = self.loss_fn(preds, y)
        acc = self._accuracy(preds, y)
        self.log("test_mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


# -------------------------------------------------------------------
# Callback: curva MSE train/test per epoca
# -------------------------------------------------------------------
class MSECurveCallback(Callback):
    def __init__(self, *, output_dir: str, run_name: str):
        super().__init__()
        self.output_dir = output_dir
        self.run_name = run_name
        self.train_mse: list[float] = []
        self.test_mse: list[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_mse" in metrics:
            self.train_mse.append(float(metrics["train_mse"].detach().cpu()))

        dataloader = trainer.datamodule.test_dataloader()
        if dataloader is None:
            return

        device = pl_module.device
        pl_module.eval()
        total = 0.0
        count = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                preds = pl_module.predict_proba(x)
                total += float(torch.sum((preds - y) ** 2).detach().cpu().item())
                count += float(y.numel())
        mse = total / max(count, 1.0)
        self.test_mse.append(mse)

    def on_train_end(self, trainer, pl_module):
        if not self.train_mse or not self.test_mse:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        epochs = list(range(1, min(len(self.train_mse), len(self.test_mse)) + 1))

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, self.train_mse[: len(epochs)], label="train_mse")
        plt.plot(epochs, self.test_mse[: len(epochs)], label="test_mse")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("MSE vs Epochs")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, f"{self.run_name}_mse_curve.png")
        plt.savefig(out_path)
        plt.close()


# -------------------------------------------------------------------
# Callback: curva Accuracy train/test per epoca
# -------------------------------------------------------------------
class AccuracyCurveCallback(Callback):
    def __init__(self, *, output_dir: str, run_name: str, threshold: float = 0.5):
        super().__init__()
        self.output_dir = output_dir
        self.run_name = run_name
        self.threshold = threshold
        self.train_acc: list[float] = []
        self.test_acc: list[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_acc" in metrics:
            self.train_acc.append(float(metrics["train_acc"].detach().cpu()))

        dataloader = trainer.datamodule.test_dataloader()
        if dataloader is None:
            return

        device = pl_module.device
        pl_module.eval()
        correct = 0.0
        count = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                preds = pl_module.predict_proba(x)
                pred_labels = (preds >= self.threshold).float()
                correct += float((pred_labels == y).sum().detach().cpu().item())
                count += float(y.numel())
        acc = correct / max(count, 1.0)
        self.test_acc.append(acc)

    def on_train_end(self, trainer, pl_module):
        if not self.train_acc or not self.test_acc:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        epochs = list(range(1, min(len(self.train_acc), len(self.test_acc)) + 1))

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, self.train_acc[: len(epochs)], label="train_acc")
        plt.plot(epochs, self.test_acc[: len(epochs)], label="test_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Epochs")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, f"{self.run_name}_accuracy_curve.png")
        plt.savefig(out_path)
        plt.close()


# -------------------------------------------------------------------
# Utility grid search
# -------------------------------------------------------------------
def get_monk_paths(task_id: int):
    project_root = Path(__file__).resolve().parents[2]
    base_path = project_root / "data" / "monk"
    train_path = base_path / f"monks-{task_id}.train"
    test_path = base_path / f"monks-{task_id}.test"
    return str(train_path), str(test_path)


def run_grid_search_for_task(
    task_id: int,
    param_grid: dict,
    val_ratio: float = 0.2,
    poly_degree: int = 2,
    patience: int = 20,
    num_random_configs: int = 5,
    seeds: list[int] | None = None,
    config_sample_seed: int = 0,
):
    """
    Random search su num_random_configs campionate dalla griglia:
    - per ogni config esegue i run indicati in seeds,
    - logga hparams/metriche,
    - salva il modello e il trasformatore LBE,
    - aggrega media/dev std su MSE di val/test.
    """
    train_path, test_path = get_monk_paths(task_id)

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    all_configs = list(ParameterGrid(param_grid))
    rng = np.random.default_rng(config_sample_seed)
    sampled_idx = rng.choice(
        len(all_configs), size=min(num_random_configs, len(all_configs)), replace=False
    )
    sampled_configs = [all_configs[i] for i in sampled_idx]

    if task_id == 3 and any("l2_reg" in cfg for cfg in all_configs):
        has_zero = any(cfg["l2_reg"] == 0 for cfg in sampled_configs)
        has_nonzero = any(cfg["l2_reg"] != 0 for cfg in sampled_configs)
        if not has_zero:
            zero_cfgs = [cfg for cfg in all_configs if cfg["l2_reg"] == 0]
            if zero_cfgs:
                sampled_configs.append(zero_cfgs[0])
        if not has_nonzero:
            nonzero_cfgs = [cfg for cfg in all_configs if cfg["l2_reg"] != 0]
            if nonzero_cfgs:
                sampled_configs.append(nonzero_cfgs[0])

    config_results = []
    export_dir = OUTPUT_ROOT / "exported_models" / f"monk{task_id}"
    export_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = OUTPUT_ROOT / "tb_logs"
    ckpt_dir = OUTPUT_ROOT / "checkpoints" / f"monk{task_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for config in sampled_configs:
        lr = config["lr"]
        batch_size = config["batch_size"]
        epochs = config["epochs"]
        l2_reg = config["l2_reg"]

        print("\n" + "=" * 60)
        print(
            f"[MONK-{task_id}] lr={lr}, batch_size={batch_size}, epochs={epochs}, l2={l2_reg}"
        )

        per_seed_metrics = []
        for run_seed in seeds:
            seed_everything(run_seed)

            data_module = MonkDataModule(
                train_path=train_path,
                test_path=test_path,
                batch_size=batch_size,
                val_ratio=val_ratio,
                poly_degree=poly_degree,
            )
            data_module.setup("fit")

            model = MonkLinearModel(
                input_dim=data_module.input_dim,
                lr=lr,
                l2_reg=l2_reg,
            )

            version_str = (
                f"randcfg_lr{lr}_bs{batch_size}_ep{epochs}_l2{l2_reg}_seed{run_seed}"
            )
            logger = TensorBoardLogger(
                save_dir=str(tb_dir),
                name=f"monk{task_id}",
                version=version_str,
            )

            ckpt_callback = ModelCheckpoint(
                monitor="val_mse",
                mode="min",
                save_top_k=1,
                filename=f"monk{task_id}_{version_str}" + "-{epoch:02d}-{val_mse:.4f}",
                dirpath=str(ckpt_dir),
            )

            callbacks = [
                EarlyStopping(monitor="val_mse", patience=patience, mode="min"),
                ckpt_callback,
                MSECurveCallback(
                    output_dir=str(OUTPUT_ROOT / "mse_curves" / f"monk{task_id}"),
                    run_name=version_str,
                ),
            ]

            trainer = Trainer(
                max_epochs=epochs,
                deterministic=True,
                logger=logger,
                callbacks=callbacks,
                enable_checkpointing=True,
            )

            trainer.fit(model, datamodule=data_module)
            val_results = trainer.validate(model, datamodule=data_module, verbose=False)
            val_mse = float(val_results[0]["val_mse"])

            test_results = trainer.test(model, datamodule=data_module, verbose=False)
            test_mse = float(test_results[0]["test_mse"])

            export_path = export_dir / (
                f"monk{task_id}_lr{lr}_bs{batch_size}_ep{epochs}_l2{l2_reg}_seed{run_seed}.pt"
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": deepcopy(config),
                    "seed": run_seed,
                    "input_dim": data_module.input_dim,
                    "poly_transformer": data_module.poly,
                },
                str(export_path),
            )

            if hasattr(logger, "log_hyperparams"):
                logger.log_hyperparams(
                    params={
                        "task_id": task_id,
                        "lr": lr,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "l2_reg": l2_reg,
                        "poly_degree": poly_degree,
                        "val_ratio": val_ratio,
                        "seed": run_seed,
                    },
                    metrics={
                        "hp/val_mse": val_mse,
                        "hp/test_mse": test_mse,
                    },
                )

            per_seed_metrics.append(
                {
                    "seed": run_seed,
                    "val_mse": val_mse,
                    "test_mse": test_mse,
                    "export_path": str(export_path),
                }
            )

        vals_mse = [m["val_mse"] for m in per_seed_metrics]
        tests_mse = [m["test_mse"] for m in per_seed_metrics]

        config_results.append(
            {
                "config": deepcopy(config),
                "seeds": seeds,
                "seed_metrics": per_seed_metrics,
                "val_mse_mean": float(np.mean(vals_mse)),
                "val_mse_std": float(np.std(vals_mse)),
                "test_mse_mean": float(np.mean(tests_mse)),
                "test_mse_std": float(np.std(tests_mse)),
            }
        )

    # Ordina per mean val_mse asc
    config_results.sort(key=lambda x: x["val_mse_mean"])
    return config_results


# -------------------------------------------------------------------
# MAIN: grid search su tutti i task MONK
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Impostazione base: griglia, task e lancio random search.
    seed_everything(0)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = OUTPUT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = [1e-3, 5e-3, 1e-2, 5e-2]
    batch_sizes = [16, 32, 64]
    epoch_list = [100, 200, 500]
    lambdas_no_reg = [0]
    lambdas_with_reg = [0, 1e-5, 1e-4, 1e-3, 1e-2]

    tasks = [1, 2, 3]
    results = {}

    for task_id in tasks:
        print("\n" + "#" * 70)
        print(f"Avvio random search per MONK-{task_id}")
        lambdas = lambdas_with_reg if task_id == 3 else lambdas_no_reg
        param_grid = {
            "lr": learning_rates,
            "batch_size": batch_sizes,
            "epochs": epoch_list,
            "l2_reg": lambdas,
        }

        config_results = run_grid_search_for_task(
            task_id=task_id,
            param_grid=param_grid,
            val_ratio=0.2,
            poly_degree=2,  # kernel quadratico esplicito (LBE)
            patience=20,
            num_random_configs=5,
            seeds=[0, 1, 2, 3, 4],
            config_sample_seed=0,
        )
        results[task_id] = config_results

        best_entry = config_results[0]
        export_dir = OUTPUT_ROOT / "exported_models" / f"monk{task_id}"

        def _best_seed(entry: dict) -> dict:
            return min(entry["seed_metrics"], key=lambda m: m["val_mse"])

        def _copy_best(entry: dict, tag: str) -> str | None:
            best_seed = _best_seed(entry)
            src = Path(best_seed["export_path"])
            if not src.exists():
                return None
            dst = export_dir / f"best_{tag}_{run_id}.pt"
            shutil.copy2(src, dst)
            return str(dst)

        best_model_path = _copy_best(best_entry, "overall")
        summary_payload = {
            "run_id": run_id,
            "task_id": task_id,
            "best_overall": {
                "config": best_entry["config"],
                "val_mse_mean": best_entry["val_mse_mean"],
                "val_mse_std": best_entry["val_mse_std"],
                "test_mse_mean": best_entry["test_mse_mean"],
                "test_mse_std": best_entry["test_mse_std"],
                "best_seed": _best_seed(best_entry),
                "best_model_path": best_model_path,
            },
        }
        print(
            f"[MONK-{task_id}] Best mean val_mse={best_entry['val_mse_mean']:.4f} "
            f"(std={best_entry['val_mse_std']:.4f}); "
            f"test_mse_mean={best_entry['test_mse_mean']:.4f} "
            f"(std={best_entry['test_mse_std']:.4f})"
        )

        if task_id == 3:
            no_reg = [c for c in config_results if c["config"].get("l2_reg", 0.0) == 0]
            with_reg = [c for c in config_results if c["config"].get("l2_reg", 0.0) != 0]
            if no_reg:
                best_no_reg = min(no_reg, key=lambda x: x["val_mse_mean"])
                best_no_reg_path = _copy_best(best_no_reg, "no_reg")
                summary_payload["best_no_reg"] = {
                    "config": best_no_reg["config"],
                    "val_mse_mean": best_no_reg["val_mse_mean"],
                    "val_mse_std": best_no_reg["val_mse_std"],
                    "test_mse_mean": best_no_reg["test_mse_mean"],
                    "test_mse_std": best_no_reg["test_mse_std"],
                    "best_seed": _best_seed(best_no_reg),
                    "best_model_path": best_no_reg_path,
                }
            if with_reg:
                best_with_reg = min(with_reg, key=lambda x: x["val_mse_mean"])
                best_with_reg_path = _copy_best(best_with_reg, "with_reg")
                summary_payload["best_with_reg"] = {
                    "config": best_with_reg["config"],
                    "val_mse_mean": best_with_reg["val_mse_mean"],
                    "val_mse_std": best_with_reg["val_mse_std"],
                    "test_mse_mean": best_with_reg["test_mse_mean"],
                    "test_mse_std": best_with_reg["test_mse_std"],
                    "best_seed": _best_seed(best_with_reg),
                    "best_model_path": best_with_reg_path,
                }

        summary_path = results_dir / f"monk{task_id}_summary_{run_id}.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("Riepilogo migliori configurazioni per ogni task:")
    for task_id, config_results in results.items():
        best_entry = config_results[0]
        print(
            f"MONK-{task_id}: mean val_mse={best_entry['val_mse_mean']:.4f} (std={best_entry['val_mse_std']:.4f}), "
            f"test_mse_mean={best_entry['test_mse_mean']:.4f} (std={best_entry['test_mse_std']:.4f}); "
            f"config={best_entry['config']}"
        )
        if task_id == 3:
            no_reg = [c for c in config_results if c["config"].get("l2_reg", 0.0) == 0]
            with_reg = [c for c in config_results if c["config"].get("l2_reg", 0.0) != 0]
            if no_reg:
                best_no_reg = min(no_reg, key=lambda x: x["val_mse_mean"])
                print(
                    f"MONK-3 no_reg: mean val_mse={best_no_reg['val_mse_mean']:.4f} "
                    f"(std={best_no_reg['val_mse_std']:.4f}), "
                    f"test_mse_mean={best_no_reg['test_mse_mean']:.4f} "
                    f"(std={best_no_reg['test_mse_std']:.4f}); "
                    f"config={best_no_reg['config']}"
                )
            if with_reg:
                best_with_reg = min(with_reg, key=lambda x: x["val_mse_mean"])
                print(
                    f"MONK-3 with_reg: mean val_mse={best_with_reg['val_mse_mean']:.4f} "
                    f"(std={best_with_reg['val_mse_std']:.4f}), "
                    f"test_mse_mean={best_with_reg['test_mse_mean']:.4f} "
                    f"(std={best_with_reg['test_mse_std']:.4f}); "
                    f"config={best_with_reg['config']}"
                )
