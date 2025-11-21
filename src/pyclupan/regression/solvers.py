"""Solvers for least square problems"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn import linear_model


@dataclass
class CEmodel:
    coeffs: np.ndarray
    intercept: float
    cluster_ids: Optional[np.ndarray] = None
    rmse: Optional[float] = None


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute root mean square errors."""
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def solver_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1),
    verbose: bool = False,
):
    """Solver ridge."""
    if verbose:
        print("Use Ridge solver.", flush=True)

    reg = linear_model.RidgeCV(alphas=alphas, fit_intercept=True, store_cv_results=True)
    reg.fit(x, y)
    y_pred = reg.predict(x)
    error = rmse(y, y_pred)
    model = CEmodel(reg.coef_, reg.intercept_, rmse=error)
    return model


def solver_lasso(
    x: np.ndarray,
    y: np.ndarray,
    alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1),
    verbose: bool = False,
):
    """Solver Lasso."""
    if verbose:
        print("Use Lasso solver.", flush=True)

    reg = linear_model.LassoCV(
        cv=None, alphas=alphas, fit_intercept=True, max_iter=10000
    )
    reg.fit(x, y)
    y_pred = reg.predict(x)
    error = rmse(y, y_pred)
    model = CEmodel(reg.coef_, reg.intercept_, rmse=error)
    if verbose:
        print("RMSE:", error * 1000, "meV/cell", flush=True)
    return model
