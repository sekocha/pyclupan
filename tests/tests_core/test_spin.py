"""Tests of functions related to spin."""

from pathlib import Path

import numpy as np

from pyclupan.core.spin import eval_cluster_functions

cwd = Path(__file__).parent


def test_eval_cluster_functions():
    """Test eval_cluster_functions."""

    coeffs = np.array([[0.0, 1.22474487, 0.0], [2.12132034, 0.0, -1.41421356]])
    spins_from_orbit = np.array(
        [
            [
                [0, -1],
                [0, 0],
                [0, -1],
                [0, 0],
                [-1, 0],
                [-1, 0],
                [0, -1],
                [-1, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [-1, 0],
                [-1, -1],
                [-1, 0],
                [-1, -1],
                [0, -1],
                [0, -1],
                [-1, 0],
                [0, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
            ],
            [
                [0, 1],
                [0, 0],
                [0, 1],
                [0, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 1],
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
            ],
        ]
    )
    cf = eval_cluster_functions(coeffs, spins_from_orbit)
    cf_true = [0.21650635, -0.21650635]
    np.testing.assert_allclose(cf, cf_true, atol=1e-6)

    spins_from_orbit = np.array(
        [
            [0, -1],
            [0, 0],
            [0, -1],
            [0, 0],
            [-1, 0],
            [-1, 0],
            [0, -1],
            [-1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [-1, 0],
            [-1, -1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [0, -1],
            [-1, 0],
            [0, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
        ]
    )
    cf = eval_cluster_functions(coeffs, spins_from_orbit)
    cf_true = 0.21650635
    np.testing.assert_allclose(cf, cf_true, atol=1e-6)
