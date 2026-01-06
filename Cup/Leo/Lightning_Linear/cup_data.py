from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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


class CupDataModule(LightningDataModule):
    """
    DataModule for ML-CUP regression with optional holdout split, scaling, and LBE.
    """

    def __init__(
        self,
        *,
        train_path: str | Path,
        test_path: str | Path | None,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 0,
        split_seed: int | None = None,
        num_workers: int | None = 0,
        pin_memory: bool = False,
        scale_inputs: bool = False,
        feature_map: str = "identity",
        target_index: int | None = None,
        predict_batch_size: int | None = None,
    ) -> None:
        """Store configuration for data loading and preprocessing."""
        super().__init__()
        self.train_path = Path(train_path)
        self.test_path = Path(test_path) if test_path is not None else None
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size or batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.split_seed = seed if split_seed is None else split_seed
        self.num_workers = num_workers if num_workers is not None else 0
        self.pin_memory = pin_memory
        self.scale_inputs = scale_inputs
        self.feature_map = feature_map
        self.target_index = target_index

        self.scaler: object | None = None
        self.feature_transformer: PolynomialFeatures | None = None
        self.input_dim: int | None = None
        self.target_dim: int | None = None
        self.test_ids: np.ndarray | None = None

        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None
        self.test_dataset: TensorDataset | None = None
        self.predict_dataset: TensorDataset | None = None
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        """Prepare datasets, scalers, and feature maps once."""
        if self._setup_done:
            return

        _train_ids, X_full, y_full = load_cup_train(self.train_path)
        test_ids = None
        X_blind = None
        if self.test_path is not None:
            test_ids, X_blind = load_cup_test(self.test_path)

        if self.val_ratio < 0 or self.test_ratio < 0 or (self.val_ratio + self.test_ratio) >= 1:
            raise ValueError("val_ratio + test_ratio must be in [0, 1).")

        # Default: use full dataset if no holdout split is requested.
        X_train, y_train = X_full, y_full
        X_val, y_val = None, None
        X_test, y_test = None, None

        holdout_ratio = self.val_ratio + self.test_ratio
        if holdout_ratio > 0:
            # First split train vs holdout, then split holdout into val/test.
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X_full,
                y_full,
                test_size=holdout_ratio,
                shuffle=True,
                random_state=self.split_seed,
            )
            if self.test_ratio > 0:
                if self.val_ratio > 0:
                    test_size = self.test_ratio / holdout_ratio
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_holdout,
                        y_holdout,
                        test_size=test_size,
                        shuffle=True,
                        random_state=self.split_seed,
                    )
                else:
                    X_test, y_test = X_holdout, y_holdout
            else:
                X_val, y_val = X_holdout, y_holdout

        if X_blind is not None and X_train.shape[1] != X_blind.shape[1]:
            raise ValueError("Train/test input dimensions do not match.")

        if self.target_index is not None:
            # Optional single-target training for per-target experiments.
            if y_full.ndim != 2:
                raise ValueError("Targets must be 2D to select a target index.")
            if self.target_index < 0 or self.target_index >= y_full.shape[1]:
                raise ValueError(
                    f"target_index out of range: {self.target_index} (max={y_full.shape[1] - 1})"
                )

            def _slice_target(values: np.ndarray | None) -> np.ndarray | None:
                """Slice a single target column while preserving 2D shape."""
                if values is None:
                    return None
                return values[:, [self.target_index]]

            y_train = _slice_target(y_train)
            y_val = _slice_target(y_val)
            y_test = _slice_target(y_test)

        if self.scale_inputs:
            # Fit scaler on train split only to avoid leakage.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = scaler.transform(X_val)
            if X_test is not None:
                X_test = scaler.transform(X_test)
            if X_blind is not None:
                X_blind = scaler.transform(X_blind)
            self.scaler = scaler
        else:
            self.scaler = None

        feature_map_name = self.feature_map or "identity"
        if feature_map_name == "identity":
            self.feature_transformer = None
        elif feature_map_name == "poly2":
            # LBE: quadratic expansion without bias.
            self.feature_transformer = PolynomialFeatures(degree=2, include_bias=False)
            X_train = self.feature_transformer.fit_transform(X_train)
            if X_val is not None:
                X_val = self.feature_transformer.transform(X_val)
            if X_test is not None:
                X_test = self.feature_transformer.transform(X_test)
            if X_blind is not None:
                X_blind = self.feature_transformer.transform(X_blind)
        else:
            raise ValueError(f"Unknown feature_map: {feature_map_name}")
        self.feature_map = feature_map_name

        X_train = X_train.astype(np.float32)
        if X_val is not None:
            X_val = X_val.astype(np.float32)
        if X_test is not None:
            X_test = X_test.astype(np.float32)
        if X_blind is not None:
            X_blind = X_blind.astype(np.float32)

        self.input_dim = int(X_train.shape[1])
        self.target_dim = int(y_train.shape[1])
        self.test_ids = test_ids

        # Build TensorDatasets for Lightning DataLoaders.
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
        if X_blind is not None:
            self.predict_dataset = TensorDataset(
                torch.tensor(X_blind, dtype=torch.float32),
            )
        else:
            self.predict_dataset = None

        self._setup_done = True

    def _make_loader(self, dataset: TensorDataset, *, shuffle: bool, batch_size: int) -> DataLoader:
        """Build a DataLoader with reproducible shuffling when requested."""
        generator = None
        if shuffle:
            generator = torch.Generator()
            # Fix seed for deterministic shuffling across runs.
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

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training split."""
        assert self.train_dataset is not None
        return self._make_loader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader | None:
        """Return the DataLoader for validation (empty if no val split)."""
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

    def predict_dataloader(self) -> DataLoader:
        """Return the DataLoader for blind test prediction."""
        assert self.predict_dataset is not None
        return self._make_loader(
            self.predict_dataset, shuffle=False, batch_size=self.predict_batch_size
        )

    def test_dataloader(self) -> DataLoader | None:
        """Return the DataLoader for the test split, if present."""
        if self.test_dataset is None:
            return None
        return self._make_loader(self.test_dataset, shuffle=False, batch_size=self.batch_size)
