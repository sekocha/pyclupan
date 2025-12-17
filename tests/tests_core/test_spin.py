"""Tests of functions related to spin."""

from pathlib import Path

import numpy as np

from pyclupan.core.spin import eval_cluster_functions, gram_schmidt

cwd = Path(__file__).parent


def test_gram_schmidt():
    """Test gram_schmidt."""
    spins = [1, 0, -1]
    cons = gram_schmidt(spins)
    mat = np.array([np.polyval(cons.T, s) for s in spins])
    metric = mat.T @ mat
    np.testing.assert_allclose(metric[0][1], 0.0, atol=1e-8)
    np.testing.assert_allclose(metric[0][2], 0.0, atol=1e-8)
    np.testing.assert_allclose(metric[1][2], 0.0, atol=1e-8)

    spins = [2, 1, 0, -1]
    cons = gram_schmidt(spins)
    mat = np.array([np.polyval(cons.T, s) for s in spins])
    metric = mat.T @ mat
    np.testing.assert_allclose(metric[0][1], 0.0, atol=1e-8)
    np.testing.assert_allclose(metric[0][2], 0.0, atol=1e-8)
    np.testing.assert_allclose(metric[0][3], 0.0, atol=1e-8)
    np.testing.assert_allclose(metric[1][2], 0.0, atol=1e-8)
    np.testing.assert_allclose(metric[1][3], 0.0, atol=1e-8)
    np.testing.assert_allclose(metric[2][3], 0.0, atol=1e-8)


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
