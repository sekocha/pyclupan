"""Tests of Lattice class."""

from pathlib import Path

import numpy as np

from pyclupan.core.cell_utils import supercell_reduced
from pyclupan.core.lattice import Lattice

cwd = Path(__file__).parent


def test_lattice_binary_fcc1(fcc_primitive_cell):
    """Test Lattice class."""
    elements = [[0, 1]]
    lattice = Lattice(fcc_primitive_cell, elements=elements)

    lattice.cell = lattice.cell
    _ = (lattice.axis, lattice.positions, lattice.types)
    assert lattice.elements_on_lattice == elements
    assert list(lattice.active_sites) == [0]
    assert list(lattice.inactive_sites) == []
    assert list(lattice.n_active_sites) == [1]
    assert list(lattice.inactive_labeling) == []

    active_labelings = np.array([[0], [1]])
    complete_labelings = lattice.complete_labelings(active_labelings)
    np.testing.assert_equal(active_labelings, complete_labelings)

    assert list(lattice.to_active_site_rep([0])) == [0]
    assert lattice.is_active_size(active_labelings)
    assert lattice.is_active_element(active_labelings)

    active_spins = lattice.to_spins(active_labelings)
    np.testing.assert_equal(active_labelings, lattice.to_labelings(active_spins))

    assert lattice.spins_on_lattice == [[1, -1]]
    assert lattice.basis_on_lattice == [[0]]
    np.testing.assert_allclose(lattice.spin_polynomials, [[1.0, 0.0]])

    np.testing.assert_equal(lattice.labelings_endmembers, active_labelings)
    assert list(lattice.element_strings) == ["Ag0", "Ag1"]


def test_lattice_binary_fcc2(fcc_primitive_cell):
    """Test Lattice class."""
    elements = [[0, 1]]
    lattice = Lattice(fcc_primitive_cell, elements=elements)
    hnf = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 3]])
    supercell = supercell_reduced(fcc_primitive_cell, supercell_matrix=hnf)
    lattice_supercell = lattice.lattice_supercell(supercell)

    axis_true = np.array([[2.0, -0.0, -4.0], [0.0, 4.0, -2.0], [2.0, -0.0, 2.0]])
    positions_true = np.array(
        [
            [0.0, 0.33333333, 0.66666667],
            [0.0, 0.33333333, 0.66666667],
            [0.0, 0.66666667, 0.33333333],
        ]
    )
    types_true = [0, 0, 0]

    np.testing.assert_allclose(lattice_supercell.axis, axis_true, atol=1e-8)
    np.testing.assert_allclose(lattice_supercell.positions, positions_true, atol=1e-8)
    assert list(lattice_supercell.types) == types_true

    reduced = lattice_supercell.reduced_cell
    np.testing.assert_allclose(reduced.axis, lattice_supercell.axis, atol=1e-8)
    np.testing.assert_allclose(
        reduced.positions,
        lattice_supercell.positions,
        atol=1e-8,
    )

    labelings_single = np.array([0, 0, 1])
    spins = lattice_supercell.to_spins(labelings_single)
    np.testing.assert_equal(spins, [1, 1, -1])
    labelings_converted = lattice_supercell.to_labelings(spins)
    np.testing.assert_equal(labelings_converted, labelings_single)

    labelings = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
    spins = lattice_supercell.to_spins(labelings)
    spins_true = np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]])
    np.testing.assert_equal(spins, spins_true)

    labelings_converted = lattice_supercell.to_labelings(spins)
    np.testing.assert_equal(labelings_converted, labelings)

    assert lattice_supercell.elements_on_lattice == elements
    assert lattice_supercell.n_elements == 2
    assert list(lattice_supercell.active_sites) == [0, 1, 2]
    assert list(lattice_supercell.inactive_sites) == []
    assert list(lattice_supercell.inactive_labeling) == []
    complete = list(lattice_supercell.complete_labelings(labelings[0:1])[0])
    assert complete == [0, 0, 0]

    assert list(lattice_supercell.map_full_to_active_rep) == [0, 1, 2]
    assert lattice_supercell.is_active_size(labelings) == True
    assert lattice_supercell.is_active_element(labelings) == True

    orbit = orbit_true = np.array([[0, 1], [0, 2], [1, 2]])
    np.testing.assert_equal(lattice_supercell.to_active_site_rep(orbit), orbit_true)

    assert lattice_supercell.basis_on_lattice == [[0]]
    spin_poly = lattice_supercell.get_spin_polynomials([0, 0])
    np.isclose(spin_poly[0][0], 1.0)
    np.isclose(spin_poly[1][0], 1.0)


def test_lattice_ternary_fcc1(fcc_primitive_cell):
    """Test Lattice class."""
    elements = [[0, 1, 2]]
    lattice = Lattice(fcc_primitive_cell, elements=elements)

    lattice.cell = lattice.cell
    _ = (lattice.axis, lattice.positions, lattice.types)
    assert lattice.elements_on_lattice == elements
    assert list(lattice.active_sites) == [0]
    assert list(lattice.inactive_sites) == []
    assert list(lattice.n_active_sites) == [1]
    assert list(lattice.inactive_labeling) == []

    active_labelings = np.array([[0], [1], [2]])
    complete_labelings = lattice.complete_labelings(active_labelings)
    np.testing.assert_equal(active_labelings, complete_labelings)

    assert list(lattice.to_active_site_rep([0])) == [0]
    assert lattice.is_active_size(active_labelings)
    assert lattice.is_active_element(active_labelings)

    active_spins = lattice.to_spins(active_labelings)
    np.testing.assert_equal(active_labelings, lattice.to_labelings(active_spins))

    assert lattice.spins_on_lattice == [[1, 0, -1]]
    assert lattice.basis_on_lattice == [[0, 1]]
    np.testing.assert_allclose(
        lattice.spin_polynomials,
        [[0.0, 1.224745, 0.0], [2.12132, 0.0, -1.414214]],
        atol=1e-6,
    )

    np.testing.assert_equal(lattice.labelings_endmembers, active_labelings)
    assert list(lattice.element_strings) == ["Ag0", "Ag1", "Ag2"]


def test_lattice_binary_perovskite(perovskite_unitcell):
    """Test Lattice class."""
    elements = [[0], [1], [2, 3]]
    lattice = Lattice(perovskite_unitcell, elements=elements)
    hnf = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])

    supercell = supercell_reduced(perovskite_unitcell, supercell_matrix=hnf)
    lattice_supercell = lattice.lattice_supercell(supercell)

    axis_true = np.array(
        [[2.6965, -0.0, -5.393], [0.0, 5.393, -2.6965], [2.6965, -0.0, 2.6965]]
    )
    positions_true = np.array(
        [
            [0.0, 0.33333333, 0.66666667],
            [0.0, 0.33333333, 0.66666667],
            [0.0, 0.66666667, 0.33333333],
        ]
    )
    axis_true = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]])
    positions_true = np.array(
        [
            [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.25, 0.75, 0.0, 0.5, 0.25, 0.75, 0.25, 0.75],
        ]
    )

    np.testing.assert_allclose(lattice_supercell.axis, axis_true, atol=1e-8)
    np.testing.assert_allclose(lattice_supercell.positions, positions_true, atol=1e-8)
    assert list(lattice_supercell.types) == [0, 0, 1, 1, 2, 2, 2, 2, 2, 2]

    reduced = lattice_supercell.reduced_cell
    np.testing.assert_allclose(reduced.axis, lattice_supercell.axis, atol=1e-8)
    np.testing.assert_allclose(
        reduced.positions,
        lattice_supercell.positions,
        atol=1e-8,
    )

    labelings_single = np.array([2, 3, 2, 3, 2, 3])
    spins = lattice_supercell.to_spins(labelings_single)
    np.testing.assert_equal(spins, [1, -1, 1, -1, 1, -1])

    labelings_converted = lattice_supercell.to_labelings(spins)
    np.testing.assert_equal(labelings_converted, labelings_single)

    labelings = np.array(
        [
            [2, 2, 2, 3, 2, 2],
            [2, 3, 3, 3, 2, 3],
            [2, 3, 2, 3, 2, 3],
            [2, 2, 3, 3, 2, 2],
            [2, 2, 2, 2, 2, 3],
            [3, 3, 3, 3, 2, 3],
        ]
    )
    spins = lattice_supercell.to_spins(labelings)
    spins_true = np.array(
        [
            [1, 1, 1, -1, 1, 1],
            [1, -1, -1, -1, 1, -1],
            [1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1],
            [1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, 1, -1],
        ]
    )
    np.testing.assert_equal(spins, spins_true)

    labelings_converted = lattice_supercell.to_labelings(spins)
    np.testing.assert_equal(labelings_converted, labelings)

    assert lattice_supercell.elements_on_lattice == elements
    assert lattice_supercell.n_elements == 4
    assert list(lattice_supercell.active_sites) == [4, 5, 6, 7, 8, 9]
    assert list(lattice_supercell.inactive_sites) == [0, 1, 2, 3]
    assert list(lattice_supercell.inactive_labeling) == [0, 0, 1, 1]
    assert list(lattice_supercell.n_active_sites) == [6]
    complete = list(lattice_supercell.complete_labelings(labelings[0:1])[0])
    assert complete == [0, 0, 1, 1, 2, 2, 2, 3, 2, 2]

    map_true = np.array([None, None, None, None, 0, 1, 2, 3, 4, 5])
    np.testing.assert_equal(lattice_supercell.map_full_to_active_rep, map_true)
    assert lattice_supercell.is_active_size(labelings) == True
    assert lattice_supercell.is_active_element(labelings) == True

    orbit = np.array([[4, 6], [8, 9], [5, 7]])
    orbit_active_site_rep = lattice_supercell.to_active_site_rep(orbit)
    orbit_true = np.array([[0, 2], [4, 5], [1, 3]])
    np.testing.assert_equal(orbit_active_site_rep, orbit_true)

    assert lattice_supercell.basis_on_lattice == [[], [], [0]]
    spin_poly = lattice_supercell.get_spin_polynomials([0, 0])
    np.isclose(spin_poly[0][0], 1.0)
    np.isclose(spin_poly[1][0], 1.0)
    assert list(lattice_supercell.element_strings) == ["Sr", "Ti", "O2", "O3"]


def test_lattice_2x3_perovskite(perovskite_unitcell):
    """Test Lattice class."""
    elements = [[0, 1], [2], [3, 4, 5]]
    lattice = Lattice(perovskite_unitcell, elements=elements)
    hnf = np.diag([1, 1, 2])
    supercell = supercell_reduced(perovskite_unitcell, supercell_matrix=hnf)
    lattice = lattice.lattice_supercell(supercell)

    assert lattice.elements_on_lattice == elements
    assert lattice.n_elements == 6
    assert list(lattice.active_sites) == [0, 1, 4, 5, 6, 7, 8, 9]
    assert list(lattice.inactive_sites) == [2, 3]
    assert list(lattice.n_active_sites) == [2, 6]
    assert list(lattice.inactive_labeling) == [2, 2]

    labelings_single = np.array([0, 1, 3, 5, 4, 4, 3, 5])
    spins = lattice.to_spins(labelings_single)
    np.testing.assert_equal(spins, [1, -1, 1, -1, 0, 0, 1, -1])

    labelings_converted = lattice.to_labelings(spins)
    np.testing.assert_equal(labelings_converted, labelings_single)

    labelings = np.array(
        [
            [0, 0, 3, 3, 5, 4, 4, 3],
            [1, 0, 5, 3, 4, 5, 4, 3],
            [1, 1, 4, 4, 4, 5, 4, 3],
        ]
    )
    spins = lattice.to_spins(labelings)
    spins_true = np.array(
        [
            [1, 1, 1, 1, -1, 0, 0, 1],
            [-1, 1, -1, 1, 0, -1, 0, 1],
            [-1, -1, 0, 0, 0, -1, 0, 1],
        ]
    )
    np.testing.assert_equal(spins, spins_true)

    labelings_converted = lattice.to_labelings(spins)
    np.testing.assert_equal(labelings_converted, labelings)

    complete = lattice.complete_labelings(labelings)
    complete_true = np.array(
        [
            [0, 0, 2, 2, 3, 3, 5, 4, 4, 3],
            [1, 0, 2, 2, 5, 3, 4, 5, 4, 3],
            [1, 1, 2, 2, 4, 4, 4, 5, 4, 3],
        ]
    )
    np.testing.assert_equal(complete, complete_true)

    map_true = np.array([0, 1, None, None, 2, 3, 4, 5, 6, 7])
    np.testing.assert_equal(lattice.map_full_to_active_rep, map_true)

    assert lattice.is_active_size(labelings)
    assert lattice.is_active_element(labelings)

    orbit = np.array([[4, 6], [8, 9], [5, 7]])
    orbit_active_site_rep = lattice.to_active_site_rep(orbit)
    orbit_true = np.array([[2, 4], [6, 7], [3, 5]])
    np.testing.assert_equal(orbit_active_site_rep, orbit_true)

    assert lattice.basis_on_lattice == [[0], [], [1, 2]]
    spin_poly = lattice.get_spin_polynomials([0, 0])
    np.isclose(spin_poly[0][0], 1.0)
    np.isclose(spin_poly[1][0], 1.0)
    assert list(lattice.element_strings) == ["Sr0", "Sr1", "Ti", "O3", "O4", "O5"]
