from __future__ import annotations

from typing import Any

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_scaler(scaler_type: str | None) -> StandardScaler | None:
    if scaler_type is None or scaler_type == "none":
        return None
    if scaler_type == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler_type: {scaler_type}")


def build_knn_pipeline(
    *,
    params: dict[str, Any],
    n_jobs: int | None,
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    feature_rep = str(params.get("feature_rep", "onehot"))
    if feature_rep == "onehot":
        steps.append(("onehot", _make_onehot()))
    elif feature_rep == "raw":
        pass
    else:
        raise ValueError(f"Unknown feature_rep: {feature_rep}")

    scaler_type = str(params.get("scaler_type", "none"))
    if feature_rep == "raw":
        scaler_type = "none"
    scaler = make_scaler(scaler_type)
    if scaler is not None:
        steps.append(("scaler", scaler))

    n_neighbors = int(params.get("n_neighbors", 5))
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")

    knn_params: dict[str, Any] = {
        "n_neighbors": n_neighbors,
        "weights": params.get("weights", "uniform"),
        "p": int(params.get("p", 2)),
        "metric": params.get("metric", "minkowski"),
        "algorithm": params.get("algorithm", "auto"),
        "leaf_size": int(params.get("leaf_size", 30)),
    }
    if n_jobs is not None:
        knn_params["n_jobs"] = int(n_jobs)

    model = KNeighborsClassifier(**knn_params)
    steps.append(("knn", model))
    return Pipeline(steps)
