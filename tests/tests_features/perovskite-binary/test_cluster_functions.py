"""Tests of cluster function calculations."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_features import PyclupanFeatures
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


def test_eval_cluster_functions():
    """Test eval_cluster_functions."""
    element_labels = {"Sr": 0, "Ti": 1, "O": 2, "V": 3}
    features = PyclupanFeatures("pyclupan_cluster.yaml")
    features.load_poscars(
        ["derivative-1", "derivative-2"],
        element_string_labels=element_labels,
    )
    cluster_functions = features.eval_cluster_functions()
    cf_true = np.array(
        [
            0.33333333,
            1.0,
            1.0,
            1.0,
            1.0,
            -0.33333333,
            -0.33333333,
            0.33333333,
            0.33333333,
            0.33333333,
            0.33333333,
            0.33333333,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -0.33333333,
        ]
    )
    np.testing.assert_allclose(cluster_functions[0], cf_true, atol=1e-8)
    np.testing.assert_allclose(cluster_functions[1], cf_true, atol=1e-8)


def test_eval_cluster_functions_from_labelings():
    """Test eval_cluster_functions"""
    features = PyclupanFeatures("pyclupan_cluster.yaml")
    unitcell = Poscar("perovskite-unitcell").structure
    hnf = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    labelings = np.array(
        [
            [2, 2, 2, 3, 2, 2],
            [2, 3, 3, 3, 2, 3],
        ]
    )
    cluster_functions = features.eval_cluster_functions(
        unitcell=unitcell,
        supercell_matrix=hnf,
        labelings=labelings,
    )
    print(cluster_functions)
    cf_true = np.array(
        [
            [
                0.66666667,
                0.66666667,
                0.33333333,
                1.0,
                0.66666667,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.33333333,
                0.66666667,
                0.66666667,
                0.0,
                0.0,
                0.0,
                0.0,
                0.33333333,
                1.0,
                0.33333333,
            ],
            [
                -0.33333333,
                0.66666667,
                0.33333333,
                0.33333333,
                0.0,
                0.0,
                0.0,
                -0.33333333,
                -0.33333333,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.33333333,
                -0.33333333,
                0.0,
            ],
        ]
    )
    np.testing.assert_allclose(cluster_functions, cf_true, atol=1e-8)
    # cf_calc = cluster_functions[:, np.array([0, 4, 7, 10, 20, 30])]
    # cf_true = np.array(
    #     [
    #         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #         [0.5, 0.33333333, 0.0, -0.16666667, -0.33333333, -0.16666667],
    #         [0.0, 1.0, 0.0, 0.0, 1.0, -0.33333333],
    #         [0.0, -0.33333333, 0.0, 0.0, 1.0, 0.0],
    #         [-0.5, 0.33333333, 0.0, 0.16666667, -0.33333333, -0.16666667],
    #         [-1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
    #     ]
    # )
    # np.testing.assert_allclose(cf_calc, cf_true, atol=1e-8)
