"""Tests of mc_structure."""

import copy
from pathlib import Path

import numpy as np

from pyclupan.core.cell_utils import supercell_diagonal
from pyclupan.core.lattice import Lattice
from pyclupan.mc.mc_structure import (
    _random_labeling,
    n_atoms_from_compositions,
    reordered_labeling,
    spins_from_structure,
    spins_random,
)

cwd = Path(__file__).parent


def test_reordered_labeling(perovskite_unitcell):
    """Test reordered_labeling."""
    st = copy.deepcopy(perovskite_unitcell)
    st.types = [0, 1, 2, 2, 3]
    st.elements = ["Sr", "Ti", "O", "O", "V"]
    st.positions = perovskite_unitcell.positions[:, np.array([0, 1, 4, 2, 3])]

    element_strings = ("Sr", "Ti", "O", "V")
    labeling = reordered_labeling(st, perovskite_unitcell, element_strings)
    np.testing.assert_equal(labeling, [0, 1, 2, 3, 2])

    st.types = [0, 1, 2, 3, 4]
    st.elements = ["Sr", "Ti", "O", "V", "W"]
    st.positions = perovskite_unitcell.positions[:, np.array([0, 1, 3, 4, 2])]

    element_strings = ("Sr", "Ti", "O", "V", "W")
    labeling = reordered_labeling(st, perovskite_unitcell, element_strings)
    np.testing.assert_equal(labeling, [0, 1, 4, 2, 3])


def test_n_atoms_from_compositions(perovskite_unitcell):
    """Test n_atoms_from_compositions."""
    lattice = Lattice(perovskite_unitcell, elements=[[0], [1], [2, 3]])
    compositions = (0.66666, 0.33333)
    n_atoms = n_atoms_from_compositions(lattice, compositions)
    np.testing.assert_equal(n_atoms, [2, 1])

    lattice = Lattice(perovskite_unitcell, elements=[[0], [1], [2, 3, 4]])
    compositions = (0.33333, 0.33333, 0.33333)
    n_atoms = n_atoms_from_compositions(lattice, compositions)
    np.testing.assert_equal(n_atoms, [1, 1, 1])


def test_n_atoms_from_compositions2(wurtzite_primitive_cell):
    """Test n_atoms_from_compositions."""
    lattice = Lattice(wurtzite_primitive_cell, elements=[[0, 1], [2, 3]])
    compositions = (0.25, 0.25, 0.25, 0.25)
    n_atoms = n_atoms_from_compositions(lattice, compositions)
    np.testing.assert_equal(n_atoms, [1, 1, 1, 1])


def test_spins_from_structure(wurtzite_primitive_cell):
    """Test spins_from_structure."""
    cell = supercell_diagonal(wurtzite_primitive_cell, size=(1, 2, 2))
    lattice = Lattice(cell, elements=[[0, 1], [2, 3]])
    st = copy.deepcopy(wurtzite_primitive_cell)
    st.types = [0, 1, 3, 2]
    st.elements = ["Si", "Al", "N", "C"]
    element_strings = ("Si", "Al", "C", "N")
    spins = spins_from_structure(lattice, st, element_strings)
    np.testing.assert_equal(
        spins, [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
    )


def test_spins_random(wurtzite_primitive_cell):
    """Test spins_random."""
    cell = supercell_diagonal(wurtzite_primitive_cell, size=(1, 2, 2))
    lattice = Lattice(cell, elements=[[0, 1], [2, 3]])
    labeling = _random_labeling(lattice, [4, 4, 4, 4])
    assert np.count_nonzero(labeling[:8] == 0) == 4
    assert np.count_nonzero(labeling[:8] == 1) == 4
    assert np.count_nonzero(labeling[:8] == 2) == 0
    assert np.count_nonzero(labeling[:8] == 3) == 0
    assert np.count_nonzero(labeling[8:] == 0) == 0
    assert np.count_nonzero(labeling[8:] == 1) == 0
    assert np.count_nonzero(labeling[8:] == 2) == 4
    assert np.count_nonzero(labeling[8:] == 3) == 4

    compositions = (0.25, 0.25, 0.25, 0.25)
    spins = spins_random(lattice, compositions)
    assert np.count_nonzero(spins[:8] == 1) == 4
    assert np.count_nonzero(spins[:8] == -1) == 4
    assert np.count_nonzero(spins[8:] == 1) == 4
    assert np.count_nonzero(spins[8:] == -1) == 4
