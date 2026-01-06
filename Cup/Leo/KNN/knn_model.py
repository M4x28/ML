from __future__ import annotations

"""KNN pipeline builder with optional preprocessing."""

from typing import Any

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, RobustScaler, StandardScaler


def make_scaler(scaler_type: str | None) -> Any | None:
    """Return a scaler instance or None if scaling is disabled."""
    if scaler_type is None or scaler_type == "none":
        return None
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "robust":
        return RobustScaler()
    if scaler_type == "power":
        return PowerTransformer()
    raise ValueError(f"Unknown scaler_type: {scaler_type}")


def build_knn_pipeline(
    *,
    params: dict[str, Any],
    scale_inputs: bool,
    n_jobs: int | None,
) -> Pipeline:
    """Assemble preprocessing + KNN regressor into a single pipeline."""
    steps: list[tuple[str, Any]] = []
    if scale_inputs:
        # Scale inputs before any feature expansion.
        scaler_type = params.get("scaler_type", "standard")
        scaler = make_scaler(str(scaler_type) if scaler_type is not None else None)
        if scaler is not None:
            steps.append(("scaler", scaler))

    feature_map = params.get("feature_map", "identity")
    if feature_map == "poly2":
        # Polynomial feature expansion up to degree 2.
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    pca_value = params.get("pca_components")
    if pca_value is not None:
        # Keep either a fraction of variance or a fixed component count.
        steps.append(("pca", PCA(n_components=pca_value)))

    n_neighbors = int(params.get("n_neighbors", 5))
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")

    # Collect the estimator hyperparameters with defaults.
    knn_params: dict[str, Any] = {
        "n_neighbors": n_neighbors,
        "weights": params.get("weights", "uniform"),
        "p": int(params.get("p", 2)),
        "metric": params.get("metric", "minkowski"),
        "algorithm": params.get("algorithm", "auto"),
        "leaf_size": int(params.get("leaf_size", 30)),
    }
    if n_jobs is not None:
        # Only pass n_jobs when the caller requests parallelism.
        knn_params["n_jobs"] = int(n_jobs)

    # Final pipeline step is the KNN regressor itself.
    model = KNeighborsRegressor(**knn_params)
    steps.append(("knn", model))
    return Pipeline(steps)
