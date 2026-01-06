from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def _read_cup_csv(path: str | Path) -> pd.DataFrame:
    """Read a CUP CSV file without header, skipping comment lines."""
    return pd.read_csv(path, comment="#", header=None)


def load_cup_train(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the training CSV and return ids, inputs, and targets.
    """
    df = _read_cup_csv(path)
    if df.shape[1] < 6:
        raise ValueError(f"Unexpected column count in train file: {df.shape[1]}")
    ids = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:-4].to_numpy(dtype=np.float32)
    y = df.iloc[:, -4:].to_numpy(dtype=np.float32)
    return ids, X, y


def load_cup_test(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the blind test CSV and return ids and inputs (no targets).
    """
    df = _read_cup_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected column count in test file: {df.shape[1]}")
    ids = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return ids, X


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Split data into train/val/test with a single holdout split."""
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio + test_ratio must be in [0, 1).")

    # Default: use full data if no holdout is requested.
    X_train, y_train, ids_train = X, y, ids
    X_val, y_val, ids_val = None, None, None
    X_test, y_test, ids_test = None, None, None

    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio > 0:
        # Split once into train and holdout, then split holdout into val/test.
        X_train, X_holdout, y_train, y_holdout, ids_train, ids_holdout = train_test_split(
            X,
            y,
            ids,
            test_size=holdout_ratio,
            shuffle=True,
            random_state=seed,
        )
        if test_ratio > 0:
            if val_ratio > 0:
                test_size = test_ratio / holdout_ratio
                X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
                    X_holdout,
                    y_holdout,
                    ids_holdout,
                    test_size=test_size,
                    shuffle=True,
                    random_state=seed,
                )
            else:
                X_test, y_test, ids_test = X_holdout, y_holdout, ids_holdout
        else:
            X_val, y_val, ids_val = X_holdout, y_holdout, ids_holdout

    return {
        "train": {"X": X_train, "y": y_train, "ids": ids_train},
        "val": {"X": X_val, "y": y_val, "ids": ids_val},
        "test": {"X": X_test, "y": y_test, "ids": ids_test},
    }


class CupDataModule(LightningDataModule):
    """
    DataModule for ML-CUP regression with train/val/test holdout from TR
    and optional StandardScaler on inputs.
    """

    def __init__(
        self,
        *,
        train_path: str | Path,
        test_path: str | Path,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        split_seed: int = 0,
        seed: int = 0,
        num_workers: int | None = 0,
        pin_memory: bool = False,
        scale_inputs: bool = True,
        predict_batch_size: int | None = None,
    ) -> None:
        """Store configuration for dataset loading and preprocessing."""
        super().__init__()
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size or batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.seed = seed
        self.num_workers = num_workers if num_workers is not None else 0
        self.pin_memory = pin_memory
        self.scale_inputs = scale_inputs

        self.scaler: StandardScaler | None = None
        self.input_dim: int | None = None
        self.target_dim: int | None = None
        self.test_ids: np.ndarray | None = None

        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None
        self.test_dataset: TensorDataset | None = None
        self.predict_dataset: TensorDataset | None = None
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        """Load CSVs, split the data, and build TensorDatasets."""
        if self._setup_done:
            return

        _train_ids, X_full, y_full = load_cup_train(self.train_path)
        test_ids, X_blind = load_cup_test(self.test_path)

        split = split_train_val_test(
            X_full,
            y_full,
            _train_ids,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.split_seed,
        )
        X_train, y_train = split["train"]["X"], split["train"]["y"]
        X_val, y_val = split["val"]["X"], split["val"]["y"]
        X_test, y_test = split["test"]["X"], split["test"]["y"]

        if X_train.shape[1] != X_blind.shape[1]:
            raise ValueError("Train/test input dimensions do not match.")

        if self.scale_inputs:
            # Fit scaler on train split only to avoid leakage.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = scaler.transform(X_val)
            if X_test is not None:
                X_test = scaler.transform(X_test)
            X_blind = scaler.transform(X_blind)
            self.scaler = scaler
        else:
            self.scaler = None

        X_train = X_train.astype(np.float32)
        if X_val is not None:
            X_val = X_val.astype(np.float32)
        if X_test is not None:
            X_test = X_test.astype(np.float32)
        X_blind = X_blind.astype(np.float32)

        self.input_dim = int(X_train.shape[1])
        self.target_dim = int(y_train.shape[1])
        self.test_ids = test_ids

        # Build TensorDatasets for Lightning.
        self.train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        if X_val is not None and y_val is not None:
            self.val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
        if X_test is not None and y_test is not None:
            self.test_dataset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
            )
        self.predict_dataset = TensorDataset(
            torch.tensor(X_blind, dtype=torch.float32),
        )

        self._setup_done = True

    def _make_loader(self, dataset: TensorDataset, *, shuffle: bool, batch_size: int) -> DataLoader:
        """Build a DataLoader with deterministic shuffling when requested."""
        generator = None
        if shuffle:
            generator = torch.Generator()
            # Fix seed for reproducible mini-batch order.
            generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def _make_target_dataset(self, dataset: TensorDataset, target_idx: int) -> TensorDataset:
        """Slice a single target column for per-target training."""
        x, y = dataset.tensors
        if y.ndim == 1:
            y_sel = y
        else:
            y_sel = y[:, int(target_idx) : int(target_idx) + 1]
        return TensorDataset(x, y_sel)

    def target_dataloader(
        self,
        split: str,
        target_idx: int,
        *,
        shuffle: bool = False,
        batch_size: int | None = None,
    ) -> DataLoader | None:
        """Return a DataLoader for a single target and split."""
        dataset = None
        split = split.lower()
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unsupported split for target_dataloader: {split}")
        if dataset is None:
            return None
        batch = int(batch_size or self.batch_size)
        return self._make_loader(
            self._make_target_dataset(dataset, target_idx),
            shuffle=shuffle,
            batch_size=batch,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for training."""
        assert self.train_dataset is not None
        return self._make_loader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def train_eval_dataloader(self) -> DataLoader:
        """Return the DataLoader for train evaluation (no shuffle)."""
        assert self.train_dataset is not None
        return self._make_loader(self.train_dataset, shuffle=False, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader | None:
        """Return the DataLoader for validation (empty if disabled)."""
        if self.val_dataset is None:
            input_dim = int(self.input_dim or 0)
            target_dim = int(self.target_dim or 0)
            # Lightning expects a dataloader even if validation is disabled.
            empty_dataset = TensorDataset(
                torch.empty((0, input_dim), dtype=torch.float32),
                torch.empty((0, target_dim), dtype=torch.float32),
            )
            return self._make_loader(empty_dataset, shuffle=False, batch_size=self.batch_size)
        return self._make_loader(self.val_dataset, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader | None:
        """Return the DataLoader for test split, if present."""
        if self.test_dataset is None:
            return None
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        """Return the DataLoader for blind test prediction."""
        assert self.predict_dataset is not None
        return self._make_loader(
            self.predict_dataset,
            shuffle=False,
            batch_size=self.predict_batch_size,
        )
