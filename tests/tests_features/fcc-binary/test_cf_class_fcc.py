"""Tests of cluster function class."""

from pathlib import Path

import numpy as np

from pyclupan.core.pypolymlp_utils import Poscar
from pyclupan.derivative.derivative_utils import (
    load_derivatives_yaml,
    load_sample_attrs_yaml,
)
from pyclupan.features.cluster_functions import ClusterFunctions

cwd = Path(__file__).parent


def test_cf_class_from_derivatives():
    """Test ClusterFunctions class using files for derivative structures."""
    cf = ClusterFunctions(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    cf.derivatives = load_sample_attrs_yaml(str(cwd) + "/pyclupan_sample_attrs.yaml")
    cluster_functions = cf.eval()
    np.testing.assert_allclose(cluster_functions[0, 1], 0.3333333333333, atol=1e-8)
    np.testing.assert_allclose(cluster_functions[1, 1], -0.111111111111, atol=1e-8)

    cf.clear_structures()
    cf.derivatives = load_derivatives_yaml(str(cwd) + "/pyclupan_derivatives_3.yaml")
    cluster_functions = cf.eval()
    np.testing.assert_allclose(cluster_functions[0, 1], 0.3333333333333, atol=1e-8)
    np.testing.assert_allclose(cluster_functions[1, 1], -0.111111111111, atol=1e-8)

    assert isinstance(cf.clusters, list)
    assert isinstance(cf.spin_basis_clusters, list)
    np.testing.assert_equal(cf.n_atoms_array, np.array([[1, 2], [2, 1], [1, 2]]))


def test_eval_cluster_functions_from_structures():
    """Test eval_cluster_functions using structures setter.."""
    cf = ClusterFunctions(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    cf.element_strings = ("Ag", "Au")
    st1 = Poscar(str(cwd) + "/derivative-1").structure
    st2 = Poscar(str(cwd) + "/derivative-2").structure
    cf.structures = [st1, st2]
    cluster_functions = cf.eval()

    cf_calc1 = cluster_functions[0, :10]
    cf_calc2 = cluster_functions[1, :10]
    cf_true1 = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cf_true2 = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(cf_calc1, cf_true1, atol=1e-8)
    np.testing.assert_allclose(cf_calc2, cf_true2, atol=1e-8)

    assert isinstance(cf.clusters, list)
    assert isinstance(cf.spin_basis_clusters, list)
    np.testing.assert_equal(cf.n_atoms_array, np.array([[2, 2], [2, 2]]))


def test_eval_cluster_functions_from_labelings():
    """Test eval_cluster_functions using given labelings."""
    cf = ClusterFunctions(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")

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
    cf.set_labelings(
        unitcell=unitcell,
        supercell_matrix=hnf,
        active_labelings=labelings,
    )
    cluster_functions = cf.eval()

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

    assert isinstance(cf.clusters, list)
    assert isinstance(cf.spin_basis_clusters, list)
    np.testing.assert_equal(
        cf.n_atoms_array, np.array([[4, 0], [3, 1], [2, 2], [2, 2], [1, 3], [0, 4]])
    )
