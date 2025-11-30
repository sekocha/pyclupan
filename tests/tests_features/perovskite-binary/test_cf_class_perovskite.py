"""Tests of cluster function class."""

from pathlib import Path

import numpy as np

from pyclupan.core.pypolymlp_utils import Poscar
from pyclupan.derivative.derivative_utils import load_derivatives_yaml
from pyclupan.features.cluster_functions import ClusterFunctions

cwd = Path(__file__).parent


def test_cf_class_from_derivatives():
    """Test ClusterFunctions class."""
    cf = ClusterFunctions(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    cf.derivatives = load_derivatives_yaml(str(cwd) + "/pyclupan_derivatives.yaml")
    cluster_functions = cf.eval()

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

    assert isinstance(cf.clusters, list)
    assert isinstance(cf.spin_basis_clusters, list)

    np.testing.assert_equal(
        cf.n_atoms_array[:3],
        np.array([[2, 2, 5, 1], [2, 2, 4, 2], [2, 2, 4, 2]]),
    )


def test_cf_class_from_structures():
    """Test ClusterFunctions class."""
    cf = ClusterFunctions(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    cf.element_strings = ("Sr", "Ti", "O", "V")
    st1 = Poscar(str(cwd) + "/derivative-1").structure
    st2 = Poscar(str(cwd) + "/derivative-2").structure
    cf.structures = [st1, st2]
    cluster_functions = cf.eval()

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

    assert isinstance(cf.clusters, list)
    assert isinstance(cf.spin_basis_clusters, list)

    np.testing.assert_equal(
        cf.n_atoms_array,
        np.array([[1, 1, 2, 1], [1, 1, 2, 1]]),
    )


def test_cf_class_from_labelings():
    """Test ClusterFunctions class."""
    cf = ClusterFunctions(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    unitcell = Poscar(str(cwd) + "/perovskite-unitcell").structure
    hnf = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    labelings = np.array(
        [
            [2, 2, 2, 3, 2, 2],
            [2, 3, 3, 3, 2, 3],
        ]
    )
    cf.set_labelings(
        unitcell=unitcell,
        supercell_matrix=hnf,
        active_labelings=labelings,
    )
    cluster_functions = cf.eval()

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

    assert isinstance(cf.clusters, list)
    assert isinstance(cf.spin_basis_clusters, list)

    np.testing.assert_equal(cf.n_atoms_array, np.array([[2, 2, 5, 1], [2, 2, 2, 4]]))
