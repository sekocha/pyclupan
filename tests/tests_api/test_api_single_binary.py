"""Tests of automated cluster expansion."""

import glob
import os
import shutil
from pathlib import Path

import pytest

from pyclupan.api.pyclupan import Pyclupan

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/"


def test_deriv_binary_fcc_sample():
    """Test enumerating binary derivative structures for FCC."""
    poscar = path_file + "/poscar-fcc-primitive"
    element_strings = ("Ag", "Au")
    elements = [[0, 1]]

    pyclupan = Pyclupan(verbose=False)
    pyclupan.set_lattice_and_elements(
        elements=elements,
        element_strings=element_strings,
        poscar=poscar,
    )
    pyclupan.enum_derivatives(min_supercell_size=2, max_supercell_size=4)
    pyclupan.enum_derivatives(
        min_supercell_size=5,
        max_supercell_size=5,
        n_samples=10,
    )
    ds_set = pyclupan.derivative_structures
    assert len(ds_set) == 17
    assert sum([len(ds) for ds in ds_set]) == 55
    assert len(ds_set.all_structure_indices) == 55
    assert len(ds_set.sampled_structure_indices) == 37

    for fname in glob.glob("pyclupan_derivative_*.yaml"):
        os.remove(fname)


def test_deriv_binary_fcc():
    """Test enumerating binary derivative structures for FCC."""
    poscar = path_file + "/poscar-fcc-primitive"
    element_strings = ("Ag", "Au")
    elements = [[0, 1]]
    pot = path_file + "/Ag-Au/polymlp.yaml"

    pyclupan = Pyclupan(verbose=False)
    pyclupan.set_lattice_and_elements(
        elements=elements,
        element_strings=element_strings,
        poscar=poscar,
    )
    pyclupan.enum_derivatives(min_supercell_size=2, max_supercell_size=4)
    pyclupan.enum_derivatives(
        min_supercell_size=5,
        max_supercell_size=5,
    )
    ds_set = pyclupan.derivative_structures
    assert len(ds_set) == 17
    assert sum([len(ds) for ds in ds_set]) == 55
    assert len(ds_set.all_structure_indices) == 55
    assert len(ds_set.sampled_structure_indices) == 55

    for fname in glob.glob("pyclupan_derivative_*.yaml"):
        os.remove(fname)

    pyclupan.eval_energies(pot=pot, geometry_optimization=False, gtol=1e-4)
    assert len(pyclupan._sampled_structures) == 55
    assert pyclupan._y[0] == pytest.approx(-2.738429067387743)

    pyclupan.enum_cluster(max_order=4, cutoffs=(6.0, 4.0, 3.0))
    pyclupan.eval_cluster_functions()
    assert pyclupan.cluster_functions.shape == (55, 8)

    os.remove("pyclupan_cluster.yaml")

    pyclupan.eval_ecis()
    assert len(pyclupan.best_model.cluster_ids) == 8

    os.remove("pyclupan_prediction.dat")
    for fname in glob.glob("pyclupan_ecis_*.yaml"):
        os.remove(fname)

    pyclupan.eval_ce_energies()
    assert len(pyclupan.energies) == 55
    assert pyclupan.energies[0] == pytest.approx(-2.73597948607030, rel=1e-3)
    assert pyclupan.energies[10] == pytest.approx(-2.84629872887410, rel=1e-3)
    assert pyclupan.energies[20] == pytest.approx(-2.63138172567667, rel=1e-3)
    assert pyclupan.energies[30] == pytest.approx(-2.77949477805880, rel=1e-3)
    assert pyclupan.energies[40] == pytest.approx(-2.69368607494454, rel=1e-3)
    assert pyclupan.energies[50] == pytest.approx(-2.87823339559496, rel=1e-3)
    os.remove("pyclupan_energies.hdf5")

    pyclupan.eval_ce_formation_energies()
    assert len(pyclupan.formation_energies) == 55
    assert pyclupan.formation_energies[0] == pytest.approx(-0.01740504226, rel=1e-3)
    assert pyclupan.formation_energies[10] == pytest.approx(-0.00717150829, rel=1e-3)
    assert pyclupan.formation_energies[20] == pytest.approx(-0.03336005863, rel=1e-3)
    assert pyclupan.formation_energies[30] == pytest.approx(-0.01269922354, rel=1e-3)
    assert pyclupan.formation_energies[40] == pytest.approx(-0.02333274184, rel=1e-3)
    assert pyclupan.formation_energies[50] == pytest.approx(-0.01499561966, rel=1e-3)

    assert len(pyclupan.compositions) == 55
    assert pyclupan.convex.shape == (5, 4)

    os.remove("pyclupan_formation_energies.hdf5")
    os.remove("pyclupan_convexhull.yaml")
    shutil.rmtree("poscars")
