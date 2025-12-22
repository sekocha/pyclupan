"""Solvers for least square problems"""

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

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
    ce_models = []
    for alp in alphas:
        reg = linear_model.Lasso(alpha=alp, fit_intercept=True, max_iter=10000)
        reg.fit(x, y)
        scores = cross_val_score(
            reg, x, y, cv=10, scoring="neg_root_mean_squared_error"
        )
        cv_score = np.average(-scores)
        y_pred = reg.predict(x)
        error = rmse(y, y_pred)
        coef, intercept = reg.coef_, reg.intercept_
        model = CEmodel(coef, intercept, cv_score=cv_score, rmse=error, alpha=alp)
        ce_models.append(model)
        if verbose:
            print("- alpha:     ", alp, flush=True)
            print("  n_nonzero: ", np.count_nonzero(np.abs(coef) > 1e-12), flush=True)
            print("  10-fold CV:", np.round(cv_score * 1000, 5), "meV/cell", flush=True)
            print("  RMSE:      ", np.round(error * 1000, 5), "meV/cell", flush=True)

    if verbose:
        idx, min_model = min(enumerate(ce_models), key=lambda s: s[1].cv_score)
        print("--------------------------------", flush=True)
        print("CE model with CV minimum:", flush=True)
        print("  model:     ", str(idx + 1).zfill(2), flush=True)
        print("  alpha:     ", min_model.alpha, flush=True)
        coef = min_model.coeffs
        cv_score, error = min_model.cv_score, min_model.rmse
        print("  n_nonzero: ", np.count_nonzero(np.abs(coef) > 1e-12), flush=True)
        print("  10-fold CV:", np.round(cv_score * 1000, 5), "meV/cell", flush=True)
        print("  RMSE:      ", np.round(error * 1000, 5), "meV/cell", flush=True)

    return ce_models
