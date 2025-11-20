"""Solvers for least square problems"""

import numpy as np
from sklearn import linear_model


def solver_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1),
):
    """Solver ridge."""
    reg = linear_model.RidgeCV(alphas=alphas, fit_intercept=True, store_cv_results=True)
    reg.fit(x, y)
    print(reg.score(x, y))
    # print("Res", reg.cv_results_)
    coeffs = reg.coef_
    intercept = reg.intercept_
    return coeffs, intercept


def solver_lasso(
    x: np.ndarray,
    y: np.ndarray,
    alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1),
):
    """Solver Lasso."""
    reg = linear_model.LassoCV(
        cv=None, alphas=alphas, fit_intercept=True, max_iter=10000
    )
    reg.fit(x, y)
    print(reg.score(x, y))
    print(reg.intercept_)

    coeffs = reg.coef_
    intercept = reg.intercept_
    return coeffs, intercept
