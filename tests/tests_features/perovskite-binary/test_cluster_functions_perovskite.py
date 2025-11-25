"""Tests of cluster function calculations."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_calc import PyclupanCalc
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


def test_eval_cluster_functions_from_derivatives():
    """Test eval_cluster_functions."""
    pyclupan = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    pyclupan.load_derivatives_yaml(str(cwd) + "/pyclupan_derivatives.yaml")
    cluster_functions = pyclupan.eval_cluster_functions()

    cf_true = np.array(
        [
            1.0,
            1.0,
            1.0,
            0.333333,
            0.333333,
            1.0,
            0.333333,
            -0.333333,
            0.333333,
            0.333333,
            -0.333333,
        ]
    )
    np.testing.assert_allclose(cluster_functions[:, 3], cf_true, atol=1e-6)


def test_eval_cluster_functions_from_poscars():
    """Test eval_cluster_functions."""
    element_strings = ("Sr", "Ti", "O", "V")
    features = PyclupanCalc(str(cwd) + "/pyclupan_clusters.yaml")
    features.load_poscars(
        [str(cwd) + "/derivative-1", str(cwd) + "/derivative-2"],
        element_strings=element_strings,
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


def test_eval_cluster_functions_from_structures():
    """Test eval_cluster_functions."""
    element_strings = ("Sr", "Ti", "O", "V")
    features = PyclupanCalc(str(cwd) + "/pyclupan_clusters.yaml")
    st1 = Poscar(str(cwd) + "/derivative-1").structure
    st2 = Poscar(str(cwd) + "/derivative-2").structure
    features.structures = [st1, st2]
    features.element_strings = element_strings
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
    features = PyclupanCalc(str(cwd) + "/pyclupan_clusters.yaml")
    unitcell = Poscar(str(cwd) + "/perovskite-unitcell").structure
    hnf = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    labelings = np.array(
        [
            [2, 2, 2, 3, 2, 2],
            [2, 3, 3, 3, 2, 3],
        ]
    )
    features.set_labelings(
        unitcell=unitcell,
        supercell_matrix=hnf,
        active_labelings=labelings,
    )
    cluster_functions = features.eval_cluster_functions()
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
