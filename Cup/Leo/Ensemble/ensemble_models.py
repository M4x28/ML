from __future__ import annotations

"""Model loading utilities for the ensemble runner."""

from dataclasses import dataclass
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn

LEO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_mlp_dir(root: Path) -> Path:
    candidates = [
        root / "MLP",
        root / "MLP_Model_Cup",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_linear_dir(root: Path) -> Path:
    candidates = [
        root / "Lightning_Linear",
        root / "Lightning_Linear" / "Lightning",
        root / "Linear_Model_Cup" / "Lightning",
        root / "LinearModel_Cup" / "Lightning",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MLP_DIR = _resolve_mlp_dir(LEO_ROOT)
LINEAR_DIR = _resolve_linear_dir(LEO_ROOT)

# Ensure local model modules are importable without installing packages.
if str(MLP_DIR) not in sys.path:
    sys.path.insert(0, str(MLP_DIR))
if str(LINEAR_DIR) not in sys.path:
    sys.path.insert(0, str(LINEAR_DIR))

try:
    from mlp_model import CupMLPModel
except Exception:  # pragma: no cover - fallback for missing deps
    CupMLPModel = None

try:
    from cup_model import CupLinearModel
except Exception:  # pragma: no cover - fallback for missing deps
    CupLinearModel = None


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    path: Path
    target_idx: int | None = None


@dataclass
class LoadedModel:
    kind: str
    model: object
    scaler: object | None = None
    feature_transformer: object | None = None
    params: dict[str, object] | None = None
    feature_map: str | None = None


def normalize_model_type(name: str) -> str:
    name = name.lower().strip()
    if name in {"knn", "xgb", "xgboost", "svm", "svr", "sklearn", "joblib"}:
        return "sklearn"
    if name in {"mlp", "nn", "neural"}:
        return "mlp"
    if name in {"ae_nn", "ae+nn", "aenn", "autoencoder_nn", "autoencoder+nn"}:
        return "ae_nn"
    if name in {"linear", "linear_model"}:
        return "linear"
    raise ValueError(f"Unsupported model_type: {name}")


def _select_target(preds: np.ndarray, target_idx: int | None) -> np.ndarray:
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] > 1:
        if target_idx is None:
            raise ValueError("target_idx is required for multi-output predictions.")
        return preds[:, target_idx]
    return preds.reshape(-1)


def _float_or_default(value: object, default: float) -> float:
    return float(default if value is None else value)


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not state:
        return state
    if all(key.startswith(prefix) for key in state):
        return {key[len(prefix):]: value for key, value in state.items()}
    return state


class AENNModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.regressor(z)


class ModelCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], LoadedModel] = {}

    def load(self, spec: ModelSpec) -> LoadedModel:
        model_type = normalize_model_type(spec.model_type)
        key = (model_type, str(spec.path))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Load each model type with its corresponding serialization format.
        if model_type == "sklearn":
            payload = joblib.load(spec.path)
            model = payload.get("model") if isinstance(payload, dict) else payload
            params = payload.get("params") if isinstance(payload, dict) else None
            loaded = LoadedModel(kind="sklearn", model=model, params=params)
        elif model_type == "mlp":
            if CupMLPModel is None:
                raise RuntimeError("CupMLPModel is unavailable; check MLP imports.")
            payload = _torch_load(spec.path)
            params = payload.get("params", {})
            model = CupMLPModel(
                input_dim=int(payload["input_dim"]),
                output_dim=int(payload["output_dim"]),
                hidden_sizes=params.get("hidden_sizes"),
                activation=params.get("activation", "relu"),
                dropout=_float_or_default(params.get("dropout"), 0.0),
                lr=_float_or_default(params.get("lr"), 1e-3),
                weight_decay=_float_or_default(params.get("weight_decay"), 0.0),
                optimizer=str(params.get("optimizer", "adam")),
                momentum=_float_or_default(params.get("momentum"), 0.9),
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
            loaded = LoadedModel(
                kind="mlp",
                model=model,
                scaler=payload.get("scaler"),
                params=params,
            )
        elif model_type == "linear":
            if CupLinearModel is None:
                raise RuntimeError("CupLinearModel is unavailable; check linear model imports.")
            payload = _torch_load(spec.path)
            model = CupLinearModel(
                input_dim=int(payload["input_dim"]),
                output_dim=int(payload["output_dim"]),
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
            loaded = LoadedModel(
                kind="linear",
                model=model,
                scaler=payload.get("scaler"),
                feature_transformer=payload.get("feature_transformer"),
                feature_map=payload.get("feature_map"),
            )
        elif model_type == "ae_nn":
            payload = _torch_load(spec.path)
            model = AENNModel(
                input_dim=int(payload["input_dim"]),
                latent_dim=int(payload["latent_dim"]),
                dropout=float(payload.get("dropout", 0.0)),
            )
            encoder_state = payload.get("encoder_state_dict")
            regressor_state = payload.get("regressor_state_dict")
            if encoder_state is None or regressor_state is None:
                raise ValueError("AE+NN payload missing encoder/regressor state dicts.")
            model.encoder.load_state_dict(encoder_state)
            model.regressor.load_state_dict(_strip_prefix(regressor_state, "net."))
            model.eval()
            loaded = LoadedModel(
                kind="ae_nn",
                model=model,
                scaler=payload.get("scaler"),
                params=payload.get("params"),
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self._cache[key] = loaded
        return loaded

    def predict_target(self, spec: ModelSpec, X: np.ndarray) -> np.ndarray:
        loaded = self.load(spec)
        target_idx = spec.target_idx

        if loaded.kind == "sklearn":
            model = loaded.model
            if isinstance(model, (list, tuple)):
                if target_idx is None:
                    raise ValueError("target_idx is required for per-target sklearn models.")
                preds = model[target_idx].predict(X)
                return _select_target(preds, None)
            preds = model.predict(X)
            return _select_target(preds, target_idx)

        if loaded.kind in {"mlp", "linear"}:
            # Apply scalers/feature transforms saved with the model before inference.
            X_proc = np.asarray(X, dtype=np.float32)
            if loaded.scaler is not None:
                X_proc = loaded.scaler.transform(X_proc)
            if loaded.feature_transformer is not None:
                X_proc = loaded.feature_transformer.transform(X_proc)
            with torch.no_grad():
                tensor = torch.tensor(X_proc, dtype=torch.float32)
                preds = loaded.model(tensor).detach().cpu().numpy()
            return _select_target(preds, target_idx)

        if loaded.kind == "ae_nn":
            X_proc = np.asarray(X, dtype=np.float32)
            if loaded.scaler is not None:
                X_proc = loaded.scaler.transform(X_proc)
            with torch.no_grad():
                tensor = torch.tensor(X_proc, dtype=torch.float32)
                preds = loaded.model(tensor).detach().cpu().numpy()
            return _select_target(preds, target_idx)

        raise ValueError(f"Unsupported loaded model kind: {loaded.kind}")

    def hyperparams(self, spec: ModelSpec) -> dict[str, object]:
        loaded = self.load(spec)
        return extract_hyperparams(loaded, spec.target_idx)


def _to_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def _select_keys(params: dict[str, object], keys: list[str]) -> dict[str, object]:
    return {key: params.get(key) for key in keys if key in params}


def _pipeline_params(model: object) -> dict[str, object]:
    params: dict[str, object] = {}
    named_steps = getattr(model, "named_steps", None)
    if not isinstance(named_steps, dict):
        return params

    if "scaler" in named_steps:
        params["scaler"] = True
    if "poly" in named_steps:
        poly = named_steps["poly"]
        degree = getattr(poly, "degree", None)
        if degree is not None:
            params["feature_map"] = f"poly{degree}"
    if "pca" in named_steps:
        pca = named_steps["pca"]
        params["pca_components"] = getattr(pca, "n_components", None)

    if "svr" in named_steps:
        svr = named_steps["svr"]
        params["svr"] = _select_keys(
            svr.get_params(deep=False),
            ["C", "epsilon", "gamma", "kernel", "degree", "coef0"],
        )
    if "xgb" in named_steps:
        xgb = named_steps["xgb"]
        params["xgb"] = _select_keys(
            xgb.get_params(deep=False),
            [
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "reg_lambda",
                "reg_alpha",
                "gamma",
                "random_state",
            ],
        )
    return params


def extract_hyperparams(loaded: LoadedModel, target_idx: int | None) -> dict[str, object]:
    if loaded.kind in {"mlp", "ae_nn"}:
        return _to_jsonable(loaded.params or {})

    if loaded.kind == "linear":
        params: dict[str, object] = {}
        if loaded.feature_map is not None:
            params["feature_map"] = loaded.feature_map
        return _to_jsonable(params)

    if loaded.kind == "sklearn":
        payload_params: dict[str, object] = {}
        if isinstance(loaded.params, dict):
            per_target = loaded.params.get("per_target")
            targets = loaded.params.get("targets")
            if per_target and isinstance(targets, list) and target_idx is not None:
                if 0 <= target_idx < len(targets) and isinstance(targets[target_idx], dict):
                    payload_params = targets[target_idx]
            else:
                payload_params = loaded.params

        model = loaded.model
        target_model = model
        if isinstance(model, (list, tuple)):
            if target_idx is None:
                return _to_jsonable(payload_params)
            if 0 <= target_idx < len(model):
                target_model = model[target_idx]
            else:
                return _to_jsonable(payload_params)

        pipeline_params = _pipeline_params(target_model)
        if payload_params and set(payload_params.keys()) != {"selected_from"}:
            merged = dict(payload_params)
            if pipeline_params:
                merged["pipeline"] = pipeline_params
            return _to_jsonable(merged)

        if pipeline_params:
            if payload_params:
                pipeline_params["selected_from"] = payload_params.get("selected_from")
            return _to_jsonable(pipeline_params)

        return _to_jsonable(payload_params)

    return {}
