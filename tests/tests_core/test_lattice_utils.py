"""Tests of functions in Lattice_utils."""

from pathlib import Path

import numpy as np

from pyclupan.core.lattice_utils import (
    get_complete_labelings,
    set_element_strings,
    set_elements_on_sublattices,
    set_labelings_endmembers,
)
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


def test_elements_on_sublattices():
    """Test set_elements_on_sublattices."""
    n_sites = [1]
    elements = [[0, 1]]
    elements_lattice = set_elements_on_sublattices(n_sites)
    assert elements == elements_lattice

    occupation = [[0], [0]]
    elements_lattice = set_elements_on_sublattices(n_sites, occupation=occupation)
    assert elements == elements_lattice

    n_sites = [1, 1, 3]
    elements = [[0], [1], [2, 3]]
    elements_lattice = set_elements_on_sublattices(n_sites, elements=elements)
    assert elements == elements_lattice

    elements = [[0], [1], [2, 3]]
    occupation = [[0], [1], [2], [2]]
    elements_lattice = set_elements_on_sublattices(n_sites, occupation=occupation)
    assert elements == elements_lattice

    elements = [[0, 1], [0, 1, 2, 3], [4, 5, 6]]
    occupation = [[0, 1], [0, 1], [1], [1], [2], [2], [2]]
    elements_lattice = set_elements_on_sublattices(n_sites, occupation=occupation)
    assert elements == elements_lattice


def test_set_labelings_endmembers():
    """Test set_labelings_endmembers."""
    elements_lattice = [[0, 1]]
    endmembers = set_labelings_endmembers(elements_lattice)
    np.testing.assert_equal(endmembers, [[0], [1]])

    elements_lattice = [[0, 1, 2]]
    endmembers = set_labelings_endmembers(elements_lattice)
    np.testing.assert_equal(endmembers, [[0], [1], [2]])


def test_set_element_strings():
    """Test set_element_strings."""
    unitcell = Poscar(str(cwd) + "/poscar-fcc").structure
    elements_lattice = [[0, 1]]
    element_strings = set_element_strings(unitcell, elements_lattice, n_elements=2)
    np.testing.assert_equal(element_strings, ["Bi0", "Bi1"])

    elements_lattice = [[0, 1, 2]]
    element_strings = set_element_strings(unitcell, elements_lattice, n_elements=3)
    np.testing.assert_equal(element_strings, ["Bi0", "Bi1", "Bi2"])

    unitcell = Poscar(str(cwd) + "/poscar-perovskite").structure
    elements_lattice = [[0], [1], [2, 3]]
    element_strings = set_element_strings(unitcell, elements_lattice, n_elements=4)
    np.testing.assert_equal(element_strings, ["Sr", "Ti", "O2", "O3"])


def test_get_complete_labelings():
    """Test get_complete_labelings."""
    active_labelings = np.array([[0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 2, 1]])
    inactive_labeling = [3, 3, 4, 4, 4]
    active_sites = [0, 1, 2, 5, 6, 7]
    inactive_sites = [3, 4, 8, 9, 10]
    labelings = get_complete_labelings(
        active_labelings, inactive_labeling, active_sites, inactive_sites
    )
    labelings_true = np.array(
        [[0, 0, 1, 3, 3, 1, 2, 2, 4, 4, 4], [0, 1, 2, 3, 3, 0, 2, 1, 4, 4, 4]]
    )
    np.testing.assert_equal(labelings, labelings_true)
