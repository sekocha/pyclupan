"""Tests of ClusterFunctions class."""

from pathlib import Path

import numpy as np

from pyclupan.core.pypolymlp_utils import Poscar
from pyclupan.features.cluster_functions import ClusterFunctions

cwd = Path(__file__).parent


def test_eval_cluster_functions_from_structures():
    """Test ClusterFunctions class using structures setter."""
    cf = ClusterFunctions(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    cf.element_strings = ("Ag", "Au", "Cu")
    cf.structures = [Poscar(str(cwd) + "/derivative-1").structure]
    cluster_functions = cf.eval()

    cf_calc1 = cluster_functions[0, :10]
    cf_true1 = [
        0.30618622,
        0.1767767,
        0.5625,
        0.32475953,
        0.3125,
        0.1875,
        -0.10825318,
        0.3125,
        0.09375,
        0.05412659,
    ]
    np.testing.assert_allclose(cf_calc1, cf_true1, atol=1e-8)

    assert isinstance(cf.clusters, list)
    assert isinstance(cf.spin_basis_clusters, list)
    np.testing.assert_equal(cf.n_atoms_array, np.array([[2, 1, 1]]))


def test_eval_cluster_functions_from_labelings():
    """Test ClusterFunctions class using given labelings."""
    cf = ClusterFunctions(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")

    unitcell = Poscar(str(cwd) + "/fcc-primitive").structure
    hnf = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 3]])
    labelings = np.array([[0, 0, 1], [0, 1, 2], [1, 2, 2]])
    cf.set_labelings(
        unitcell=unitcell,
        supercell_matrix=hnf,
        active_labelings=labelings,
    )
    cluster_functions = cf.eval()

    cf_calc = cluster_functions[:, np.array([0, 4, 7, 10, 20, 30])]
    cf_true = np.array(
        [
            [0.81649658, -0.25, -0.25, 0.25, 0.51031036, 0.51031036],
            [0.0, -0.25, -0.25, 0.25, 0.0, 0.0],
            [-0.81649658, -0.25, -0.25, 0.25, -0.51031036, -0.51031036],
        ]
    )
    np.testing.assert_allclose(cf_calc, cf_true, atol=1e-8)

    assert isinstance(cf.clusters, list)
    assert isinstance(cf.spin_basis_clusters, list)

    np.testing.assert_equal(
        cf.n_atoms_array,
        np.array([[2, 1, 0], [1, 1, 1], [0, 1, 2]]),
    )
