"""Tests of cluster function calculations."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_features import PyclupanFeatures
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


def test_eval_cluster_functions():
    """Test eval_cluster_functions."""
    element_labels = {"Ag": 0, "Au": 1}
    features = PyclupanFeatures(cluster_yaml=str(cwd) + "/pyclupan_cluster.yaml")
    features.load_poscars([str(cwd) + "/derivative-1", str(cwd) + "/derivative-2"])
    features.element_string_labels = element_labels
    cluster_functions = features.eval_cluster_functions()

    cf_calc1 = cluster_functions[0, :10]
    cf_calc2 = cluster_functions[1, :10]
    cf_true1 = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cf_true2 = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(cf_calc1, cf_true1, atol=1e-8)
    np.testing.assert_allclose(cf_calc2, cf_true2, atol=1e-8)


def test_eval_cluster_functions_from_labelings():
    """Test eval_cluster_functions"""
    features = PyclupanFeatures(cluster_yaml=str(cwd) + "/pyclupan_cluster.yaml")

    unitcell = Poscar("fcc-primitive").structure
    hnf = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 4]])
    labelings = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    cluster_functions = features.eval_cluster_functions(
        unitcell=unitcell,
        supercell_matrix=hnf,
        labelings=labelings,
    )

    cf_calc = cluster_functions[:, np.array([0, 4, 7, 10, 20, 30])]
    cf_true = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.33333333, 0.0, -0.16666667, -0.33333333, -0.16666667],
            [0.0, 1.0, 0.0, 0.0, 1.0, -0.33333333],
            [0.0, -0.33333333, 0.0, 0.0, 1.0, 0.0],
            [-0.5, 0.33333333, 0.0, 0.16666667, -0.33333333, -0.16666667],
            [-1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(cf_calc, cf_true, atol=1e-8)
