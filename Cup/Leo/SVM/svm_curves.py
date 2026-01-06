from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Cup.Leo.common.cup_metrics import mean_euclidean_error, mse_per_instance
from svm_model import build_svm_pipeline


def build_curves(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
    scale_inputs: bool,
    use_pca: bool,
    n_jobs: int,
    curve_iters: list[int],
) -> dict[str, list[float]]:
    """Train models for different max_iter values and collect metrics."""
    train_mse: list[float] = []
    test_mse: list[float] = []
    train_mee: list[float] = []
    test_mee: list[float] = []
    steps: list[int] = []

    for max_iter in curve_iters:
        # Refit the model for each iteration budget.
        model = build_svm_pipeline(
            params=params,
            scale_inputs=scale_inputs,
            use_pca=use_pca,
            n_jobs=n_jobs,
            max_iter=max_iter,
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mse.append(mse_per_instance(train_pred, y_train))
        test_mse.append(mse_per_instance(test_pred, y_test))
        train_mee.append(mean_euclidean_error(train_pred, y_train))
        test_mee.append(mean_euclidean_error(test_pred, y_test))
        steps.append(int(max_iter))

    return {
        "steps": steps,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_mee": train_mee,
        "test_mee": test_mee,
    }


def build_curves_per_target(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params_per_target: list[dict],
    scale_inputs: bool,
    use_pca: bool,
    n_jobs: int,
    curve_iters: list[int],
) -> dict[str, list[float]]:
    """Train one SVR per target and collect curves over max_iter."""
    train_mse: list[float] = []
    test_mse: list[float] = []
    train_mee: list[float] = []
    test_mee: list[float] = []
    steps: list[int] = []

    for max_iter in curve_iters:
        # Fit a single-output model for each target.
        train_preds: list[np.ndarray] = []
        test_preds: list[np.ndarray] = []
        for idx, params in enumerate(params_per_target):
            model = build_svm_pipeline(
                params=params,
                scale_inputs=scale_inputs,
                use_pca=use_pca,
                n_jobs=n_jobs,
                multi_output=False,
                max_iter=max_iter,
            )
            model.fit(X_train, y_train[:, idx])
            train_preds.append(model.predict(X_train))
            test_preds.append(model.predict(X_test))

        train_pred = np.column_stack(train_preds)
        test_pred = np.column_stack(test_preds)

        train_mse.append(mse_per_instance(train_pred, y_train))
        test_mse.append(mse_per_instance(test_pred, y_test))
        train_mee.append(mean_euclidean_error(train_pred, y_train))
        test_mee.append(mean_euclidean_error(test_pred, y_test))
        steps.append(int(max_iter))

    return {
        "steps": steps,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_mee": train_mee,
        "test_mee": test_mee,
    }


def save_mse_curve(
    *,
    curves: dict[str, list[float]],
    output_dir: str | Path,
    run_name: str,
    title: str,
) -> Path:
    """Save the MSE curve plot and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = curves["steps"]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, curves["train_mse"], label="train_mse")
    plt.plot(steps, curves["test_mse"], label="test_mse")
    plt.xlabel("Max Iter")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = output_dir / f"{run_name}_mse_curve.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def save_mee_curve(
    *,
    curves: dict[str, list[float]],
    output_dir: str | Path,
    run_name: str,
    title: str,
) -> Path:
    """Save the MEE curve plot and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = curves["steps"]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, curves["train_mee"], label="train_mee")
    plt.plot(steps, curves["test_mee"], label="test_mee")
    plt.xlabel("Max Iter")
    plt.ylabel("MEE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = output_dir / f"{run_name}_mee_curve.png"
    plt.savefig(out_path)
    plt.close()
    return out_path
