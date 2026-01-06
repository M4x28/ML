from __future__ import annotations

from typing import Any

from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


def build_svm_pipeline(
    *,
    params: dict[str, Any],
    scale_inputs: bool,
    n_jobs: int | None,
    multi_output: bool = True,
    max_iter: int | None = None,
    use_pca: bool = False,
    pca_components: float | int = 0.95,
) -> Pipeline:
    """Build a preprocessing + SVR pipeline (optionally multi-output)."""
    steps: list[tuple[str, Any]] = []
    if scale_inputs:
        steps.append(("scaler", StandardScaler()))

    feature_map = params.get("feature_map", "identity")
    kernel = params.get("kernel", "rbf")
    if feature_map == "poly2" and kernel == "poly":
        # Avoid double polynomial expansion (feature map + poly kernel).
        raise ValueError("Invalid configuration: feature_map=poly2 with kernel=poly.")
    if feature_map == "poly2":
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    if use_pca:
        pca_value = params.get("pca_components", pca_components)
        # PCA after scaling/feature mapping for dimensionality reduction.
        steps.append(("pca", PCA(n_components=pca_value)))

    svr_params: dict[str, Any] = {
        "kernel": kernel,
        "C": float(params.get("C", 1.0)),
        "epsilon": float(params.get("epsilon", 0.1)),
    }
    if max_iter is not None:
        svr_params["max_iter"] = int(max_iter)
    elif "max_iter" in params:
        svr_params["max_iter"] = int(params["max_iter"])

    if kernel in ("rbf", "poly"):
        gamma = params.get("gamma", "scale")
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            gamma = float(gamma)
        svr_params["gamma"] = gamma
    if kernel == "poly":
        svr_params["degree"] = int(params.get("degree", 3))
        svr_params["coef0"] = float(params.get("coef0", 0.0))

    estimator = SVR(**svr_params)
    if multi_output:
        # Wrap per-target regressors to handle multi-output regression.
        model = MultiOutputRegressor(estimator, n_jobs=n_jobs)
    else:
        model = estimator
    steps.append(("svr", model))
    return Pipeline(steps)
