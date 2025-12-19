"""Tests of cluster function diff class."""

from pathlib import Path

import numpy as np

from pyclupan.core.cell_utils import supercell_general
from pyclupan.features.cluster_functions import ClusterFunctions
from pyclupan.features.cluster_functions_mc import ClusterFunctionsMC

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/binary_fcc/"
clusters_yaml = path_file + "/pyclupan_clusters.yaml"


def _init(fcc_primitive_cell, supercell_size: tuple, refine: bool = True):
    """Initialize ClusterFunctionsMC."""
    sup = supercell_general(
        fcc_primitive_cell, supercell_matrix=supercell_size, refine=refine
    )
    cf = ClusterFunctions(clusters_yaml=clusters_yaml)
    lattice_unitcell = cf.lattice_unitcell
    lattice_supercell = lattice_unitcell.lattice_supercell(sup)
    cf_mc = ClusterFunctionsMC(cf, lattice_supercell)
    return cf_mc


def _run_test_spin_swap(cf_mc, spins):
    """Run single test for spin_flip."""
    cf_calc1 = cf_mc.eval_from_spins(spins)

    cf_diff12 = cf_mc.eval_from_spin_swap(spins, [0, 7])
    spins[0], spins[7] = spins[7], spins[0]
    cf_calc2 = cf_mc.eval_from_spins(spins)
    np.testing.assert_allclose(cf_calc2, cf_calc1 + cf_diff12, atol=1e-8)

    cf_diff23 = cf_mc.eval_from_spin_swap(spins, [1, 6])
    spins[1], spins[6] = spins[6], spins[1]
    cf_calc3 = cf_mc.eval_from_spins(spins)
    np.testing.assert_allclose(cf_calc3, cf_calc2 + cf_diff23, atol=1e-8)

    cf_diff34 = cf_mc.eval_from_spin_swap(spins, [2, 3])
    cf_diff34_stable = cf_mc.eval_from_spin_swap_stable(spins, [2, 3])
    spins[2], spins[3] = spins[3], spins[2]
    cf_calc4 = cf_mc.eval_from_spins(spins)
    np.testing.assert_allclose(cf_calc4, cf_calc3 + cf_diff34, atol=1e-8)
    np.testing.assert_allclose(cf_calc4, cf_calc3 + cf_diff34_stable, atol=1e-8)


def test_eval_diff_fcc1(fcc_primitive_cell):
    """Test eval_from_spin_swap."""
    cf_mc = _init(fcc_primitive_cell, supercell_size=(1, 1, 2), refine=True)
    spins = np.array([1, 1, -1, 1, 1, -1, -1, -1])
    _run_test_spin_swap(cf_mc, spins)


def test_eval_diff_fcc2(fcc_primitive_cell):
    """Test eval_from_spin_swap."""
    cf_mc = _init(fcc_primitive_cell, supercell_size=(2, 2, 2), refine=False)
    spins = np.array([1, 1, -1, 1, 1, -1, -1, -1])
    _run_test_spin_swap(cf_mc, spins)


def _run_test_spin_flip(cf_mc, spins):
    """Run single test for spin_flip."""
    cf_calc1 = cf_mc.eval_from_spins(spins)

    spin_new = spins[5] * -1
    cf_diff12 = cf_mc.eval_from_spin_flip(spins, 5, spin_new)
    spins[5] = spin_new
    cf_calc2 = cf_mc.eval_from_spins(spins)
    np.testing.assert_allclose(cf_calc2, cf_calc1 + cf_diff12, atol=1e-8)

    spin_new2 = spins[6] * -1
    cf_diff23 = cf_mc.eval_from_spin_flip(spins, 6, spin_new2)
    spins[6] = spin_new2
    cf_calc3 = cf_mc.eval_from_spins(spins)
    np.testing.assert_allclose(cf_calc3, cf_calc2 + cf_diff23, atol=1e-8)

    spin_new3 = spins[2] * -1
    cf_diff34 = cf_mc.eval_from_spin_flip(spins, 2, spin_new3)
    spins[2] = spin_new3
    cf_calc4 = cf_mc.eval_from_spins(spins)
    np.testing.assert_allclose(cf_calc4, cf_calc3 + cf_diff34, atol=1e-8)


def test_eval_diff_fcc_spin_flip1(fcc_primitive_cell):
    """Test eval_from_spin_flip."""
    cf_mc = _init(fcc_primitive_cell, supercell_size=(2, 2, 2), refine=True)
    spins = np.tile([1, -1, -1, 1], 8)
    _run_test_spin_flip(cf_mc, spins)


def test_eval_diff_fcc_spin_flip2(fcc_primitive_cell):
    """Test eval_from_spin_flip."""
    cf_mc = _init(fcc_primitive_cell, supercell_size=(1, 2, 4), refine=False)
    spins = np.tile([1, -1, -1, 1], 2)
    _run_test_spin_flip(cf_mc, spins)
