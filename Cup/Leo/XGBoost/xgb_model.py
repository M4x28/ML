from __future__ import annotations

from typing import Any

from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - handled at runtime
    XGBRegressor = None


def build_xgb_pipeline(
    *,
    params: dict[str, Any],
    scale_inputs: bool,
    n_jobs: int,
    multi_output: bool = True,
    n_estimators: int | None = None,
    random_state: int | None = None,
) -> Pipeline:
    """Build a preprocessing + XGBoost pipeline (optionally multi-output)."""
    if XGBRegressor is None:
        raise RuntimeError("XGBoost is not installed. Install with: python -m pip install xgboost")

    steps: list[tuple[str, Any]] = []
    if scale_inputs:
        # Optional scaling for numerical stability.
        steps.append(("scaler", StandardScaler()))

    xgb_params: dict[str, Any] = {
        "n_estimators": int(params.get("n_estimators", 300)),
        "max_depth": int(params.get("max_depth", 6)),
        "learning_rate": float(params.get("learning_rate", 0.1)),
        "subsample": float(params.get("subsample", 1.0)),
        "colsample_bytree": float(params.get("colsample_bytree", 1.0)),
        "min_child_weight": float(params.get("min_child_weight", 1.0)),
        "reg_alpha": float(params.get("reg_alpha", 0.0)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "gamma": float(params.get("gamma", 0.0)),
        "objective": "reg:squarederror",
        "n_jobs": int(n_jobs),
        "verbosity": 0,
    }
    if n_estimators is not None:
        xgb_params["n_estimators"] = int(n_estimators)
    if "random_state" in params:
        xgb_params["random_state"] = int(params["random_state"])
    if random_state is not None:
        xgb_params["random_state"] = int(random_state)

    estimator = XGBRegressor(**xgb_params)
    if multi_output:
        # Wrap the estimator to handle multi-output regression.
        model = MultiOutputRegressor(estimator, n_jobs=None)
    else:
        model = estimator
    steps.append(("xgb", model))
    return Pipeline(steps)
