"""
Pipeline Lightning per MONK con espansione quadratica (LBE), Tikhonov, random search,
repliche su seed multipli e export dei modelli finali.

Struttura:
- load_monk_pandas: lettura e one-hot dei dataset MONK.
- MonkDataset / MonkDataModule: gestione dataset, split opzionale val, LBE via PolynomialFeatures.
- MonkLinearModel: regressione logistica con L2, logging di base_loss e loss totale.
- run_grid_search_for_task: campiona N configurazioni random dalla griglia, esegue K seed,
  logga metriche, salva modelli ed aggrega media/dev std di val/test.
- main: esegue random search per i task richiesti.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from copy import deepcopy


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
# Modello lineare con regularizzazione Tikhonov
# -------------------------------------------------------------------
class MonkLinearModel(L.LightningModule):
    """
    Modello lineare per classificazione binaria con L2/Tikhonov e base quadratica.
    train_loss = BCE + l2_reg * ||w||^2, train_base_loss = BCE pura.
    """

    def __init__(self, input_dim: int, lr: float = 1e-2, l2_reg: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(input_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.linear(x).squeeze(1)

    def _compute_loss_and_metrics(self, batch, stage: str):
        x, y = batch
        logits = self.forward(x)

        base_loss = self.loss_fn(logits, y)
        l2_penalty = self.hparams.l2_reg * torch.sum(self.linear.weight ** 2)
        loss = base_loss + l2_penalty

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_base_loss", base_loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._compute_loss_and_metrics(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._compute_loss_and_metrics(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# -------------------------------------------------------------------
# Utility grid search
# -------------------------------------------------------------------
def get_monk_paths(task_id: int):
    base_path = "../../data/monk"
    train_path = f"{base_path}/monks-{task_id}.train"
    test_path = f"{base_path}/monks-{task_id}.test"
    return train_path, test_path


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
    - aggrega media/dev std su val/test.
    """
    train_path, test_path = get_monk_paths(task_id)

    if seeds is None:
        seeds = [104, 17, 28, 33, 42]

    all_configs = list(ParameterGrid(param_grid))
    rng = np.random.default_rng(config_sample_seed)
    sampled_idx = rng.choice(
        len(all_configs), size=min(num_random_configs, len(all_configs)), replace=False
    )
    sampled_configs = [all_configs[i] for i in sampled_idx]

    config_results = []
    export_dir = os.path.join("exported_models", f"monk{task_id}")
    os.makedirs(export_dir, exist_ok=True)

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
                save_dir="tb_logs",
                name=f"monk{task_id}",
                version=version_str,
            )

            ckpt_callback = ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename=f"monk{task_id}_{version_str}" + "-{epoch:02d}-{val_loss:.4f}",
                dirpath="checkpoints",
            )

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
                ckpt_callback,
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
            val_acc = float(val_results[0]["val_acc"])
            val_loss = float(val_results[0]["val_loss"])

            test_results = trainer.test(model, datamodule=data_module, verbose=False)
            test_acc = float(test_results[0]["test_acc"])
            test_loss = float(test_results[0]["test_loss"])

            export_path = os.path.join(
                export_dir,
                f"monk{task_id}_lr{lr}_bs{batch_size}_ep{epochs}_l2{l2_reg}_seed{run_seed}.pt",
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": deepcopy(config),
                    "seed": run_seed,
                    "input_dim": data_module.input_dim,
                    "poly_transformer": data_module.poly,
                },
                export_path,
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
                        "hp/val_acc": val_acc,
                        "hp/val_loss": val_loss,
                        "hp/test_acc": test_acc,
                        "hp/test_loss": test_loss,
                    },
                )

            per_seed_metrics.append(
                {
                    "seed": run_seed,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "export_path": export_path,
                }
            )

        vals_acc = [m["val_acc"] for m in per_seed_metrics]
        vals_loss = [m["val_loss"] for m in per_seed_metrics]
        tests_acc = [m["test_acc"] for m in per_seed_metrics]
        tests_loss = [m["test_loss"] for m in per_seed_metrics]

        config_results.append(
            {
                "config": deepcopy(config),
                "seeds": seeds,
                "seed_metrics": per_seed_metrics,
                "val_acc_mean": float(np.mean(vals_acc)),
                "val_acc_std": float(np.std(vals_acc)),
                "val_loss_mean": float(np.mean(vals_loss)),
                "val_loss_std": float(np.std(vals_loss)),
                "test_acc_mean": float(np.mean(tests_acc)),
                "test_acc_std": float(np.std(tests_acc)),
                "test_loss_mean": float(np.mean(tests_loss)),
                "test_loss_std": float(np.std(tests_loss)),
            }
        )

    # Ordina per mean val_acc desc e val_loss asc
    config_results.sort(key=lambda x: (-x["val_acc_mean"], x["val_loss_mean"]))
    return config_results


# -------------------------------------------------------------------
# MAIN: grid search su tutti i task MONK
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Impostazione base: griglia, task e lancio random search.
    seed_everything(0)

    learning_rates = [1e-3, 5e-3, 1e-2, 5e-2]
    batch_sizes = [16, 32, 64]
    epoch_list = [100, 200, 400]
    lambdas = [0, 1e-5, 1e-4, 1e-3, 1e-2]

    param_grid = {
        "lr": learning_rates,
        "batch_size": batch_sizes,
        "epochs": epoch_list,
        "l2_reg": lambdas,
    }

    tasks = [1, 2, 3]
    results = {}

    for task_id in tasks:
        print("\n" + "#" * 70)
        print(f"Avvio random search per MONK-{task_id}")
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
        print(
            f"[MONK-{task_id}] Best mean val_acc={best_entry['val_acc_mean']:.4f} "
            f"(std={best_entry['val_acc_std']:.4f}), val_loss_mean={best_entry['val_loss_mean']:.4f} "
            f"(std={best_entry['val_loss_std']:.4f}); "
            f"test_acc_mean={best_entry['test_acc_mean']:.4f} (std={best_entry['test_acc_std']:.4f}), "
            f"test_loss_mean={best_entry['test_loss_mean']:.4f} (std={best_entry['test_loss_std']:.4f})"
        )

    print("\n" + "=" * 70)
    print("Riepilogo migliori configurazioni per ogni task:")
    for task_id, config_results in results.items():
        best_entry = config_results[0]
        print(
            f"MONK-{task_id}: mean val_acc={best_entry['val_acc_mean']:.4f} (std={best_entry['val_acc_std']:.4f}), "
            f"val_loss_mean={best_entry['val_loss_mean']:.4f} (std={best_entry['val_loss_std']:.4f}); "
            f"test_acc_mean={best_entry['test_acc_mean']:.4f} (std={best_entry['test_acc_std']:.4f}), "
            f"test_loss_mean={best_entry['test_loss_mean']:.4f} (std={best_entry['test_loss_std']:.4f}); "
            f"config={best_entry['config']}"
        )
