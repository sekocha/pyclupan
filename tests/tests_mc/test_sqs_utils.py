"""Tests of sqs_utils."""

from pathlib import Path

import numpy as np

from pyclupan.core.cell_utils import supercell_diagonal
from pyclupan.core.lattice import Lattice
from pyclupan.features.cluster_functions import ClusterFunctions
from pyclupan.mc.sqs_utils import calc_ideal_cluster_functions

cwd = Path(__file__).parent


def test_calc_ideal_cluster_functions_binary_fcc(fcc_primitive_cell):
    """Test calc_ideal_cluster_functions."""
    lattice_unitcell = Lattice(fcc_primitive_cell, elements=[[0, 1]])
    supercell = supercell_diagonal(fcc_primitive_cell, (2, 2, 2))
    lattice_supercell = lattice_unitcell.lattice_supercell(supercell)

    path_file = str(cwd) + "/../files/binary_fcc/"
    clusters_yaml = path_file + "/pyclupan_clusters.yaml"
    cf = ClusterFunctions(clusters_yaml=clusters_yaml)

    active_spins = np.array([1, -1, 1, 1, 1, -1, 1, 1])
    ideals = calc_ideal_cluster_functions(
        lattice_unitcell, lattice_supercell, cf, active_spins
    )
    assert ideals[0] == 0.5
    np.testing.assert_allclose(ideals[1:5], 1.0 / 4.0)
    np.testing.assert_allclose(ideals[5:17], 1.0 / 8.0)
    np.testing.assert_allclose(ideals[17:], 1.0 / 16.0)


def test_calc_ideal_cluster_functions_ternary_fcc(fcc_primitive_cell):
    """Test calc_ideal_cluster_functions."""
    lattice_unitcell = Lattice(fcc_primitive_cell, elements=[[0, 1, 2]])
    supercell = supercell_diagonal(fcc_primitive_cell, (2, 2, 2))
    lattice_supercell = lattice_unitcell.lattice_supercell(supercell)

    path_file = str(cwd) + "/../files/ternary_fcc/"
    clusters_yaml = path_file + "/pyclupan_clusters.yaml"
    cf = ClusterFunctions(clusters_yaml=clusters_yaml)

    active_spins = np.array([1, -1, -1, 0, 1, -1, 0, -1])
    ideals = calc_ideal_cluster_functions(
        lattice_unitcell, lattice_supercell, cf, active_spins
    )
    np.testing.assert_allclose(ideals[0:2], [-0.30618622, 0.1767767])
    np.testing.assert_allclose(
        ideals[2:14],
        np.tile([0.09375, -0.05412659, 0.03125], 4),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ideals[14:20],
        [-0.02870496, 0.01657282, 0.01657282, -0.00956832, -0.00956832, 0.00552427],
        atol=1e-6,
    )
