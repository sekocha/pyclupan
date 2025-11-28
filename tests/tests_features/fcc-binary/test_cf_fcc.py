"""Tests of cluster function calculations."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_calc import PyclupanCalc
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


def test_eval_cluster_functions_from_derivatives():
    """Test eval_cluster_functions using files for derivative structures."""
    features = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    features.load_sample_attrs_yaml(str(cwd) + "/pyclupan_sample_attrs.yaml")
    cluster_functions = features.eval_cluster_functions()
    np.testing.assert_allclose(cluster_functions[0, 1], 0.3333333333333, atol=1e-8)
    np.testing.assert_allclose(cluster_functions[1, 1], -0.111111111111, atol=1e-8)

    features.clear_structures()
    features.load_derivatives_yaml(str(cwd) + "/pyclupan_derivatives_3.yaml")
    cluster_functions = features.eval_cluster_functions()
    np.testing.assert_allclose(cluster_functions[0, 1], 0.3333333333333, atol=1e-8)
    np.testing.assert_allclose(cluster_functions[1, 1], -0.111111111111, atol=1e-8)


def test_eval_cluster_functions_from_poscars():
    """Test eval_cluster_functions using POSCAR files."""
    element_strings = ("Ag", "Au")
    features = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    features.load_poscars([str(cwd) + "/derivative-1", str(cwd) + "/derivative-2"])
    features.element_strings = element_strings
    cluster_functions = features.eval_cluster_functions()

    cf_calc1 = cluster_functions[0, :10]
    cf_calc2 = cluster_functions[1, :10]
    cf_true1 = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cf_true2 = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(cf_calc1, cf_true1, atol=1e-8)
    np.testing.assert_allclose(cf_calc2, cf_true2, atol=1e-8)


def test_eval_cluster_functions_from_structures():
    """Test eval_cluster_functions using structures setter.."""
    element_strings = ("Ag", "Au")
    features = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    st1 = Poscar(str(cwd) + "/derivative-1").structure
    st2 = Poscar(str(cwd) + "/derivative-2").structure
    features.structures = [st1, st2]
    features.element_strings = element_strings
    cluster_functions = features.eval_cluster_functions()

    cf_calc1 = cluster_functions[0, :10]
    cf_calc2 = cluster_functions[1, :10]
    cf_true1 = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cf_true2 = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(cf_calc1, cf_true1, atol=1e-8)
    np.testing.assert_allclose(cf_calc2, cf_true2, atol=1e-8)


def test_eval_cluster_functions_from_labelings():
    """Test eval_cluster_functions using given labelings."""
    features = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")

    unitcell = Poscar(str(cwd) + "/fcc-primitive").structure
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
    features.set_labelings(
        unitcell=unitcell,
        supercell_matrix=hnf,
        active_labelings=labelings,
    )
    cluster_functions = features.eval_cluster_functions()

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
