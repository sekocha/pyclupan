"""Solvers for least square problems"""

import numpy as np
from sklearn import linear_model

from pyclupan.core.model import CEmodel


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute root mean square errors."""
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def solver_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alphas: tuple = (1e-5, 1e-4, 1e-3),
    verbose: bool = False,
):
    """Solver ridge."""
    reg = linear_model.RidgeCV(alphas=alphas, fit_intercept=True, store_cv_results=True)
    reg.fit(x, y)
    y_pred = reg.predict(x)
    error = rmse(y, y_pred)
    model = CEmodel(reg.coef_, reg.intercept_, rmse=error)
    if verbose:
        print("RMSE:", error * 1000, "meV/cell", flush=True)
    return model


def solver_lasso(
    x: np.ndarray,
    y: np.ndarray,
    alphas: tuple = (1e-5, 1e-4, 1e-3),
    verbose: bool = False,
):
    """Solver Lasso."""
    reg = linear_model.LassoCV(alphas=alphas, fit_intercept=True, max_iter=10000)
    reg.fit(x, y)
    y_pred = reg.predict(x)
    error = rmse(y, y_pred)
    model = CEmodel(reg.coef_, reg.intercept_, rmse=error)
    if verbose:
        print("RMSE:", error * 1000, "meV/cell", flush=True)
    return model
