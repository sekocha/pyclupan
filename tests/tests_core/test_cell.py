"""Tests of cell_utils."""

from pathlib import Path

import numpy as np

from pyclupan.core.cell_positions_utils import (
    decompose_fraction,
    get_matching_positions,
)
from pyclupan.core.cell_utils import (
    get_unitcell_reps,
    supercell,
    supercell_diagonal,
    supercell_reduced,
    unitcell_reps_to_supercell_reps,
)
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


def test_supercell_diagonal():
    """Test supercell diagonal."""
    unitcell = Poscar(str(cwd) + "/poscar-fcc").structure
    sup = supercell_diagonal(unitcell, size=(1, 2, 2))

    axis_true = np.array(
        [[0.0, 5.393, 5.393], [2.6965, 0.0, 5.393], [2.6965, 5.393, 0.0]]
    )
    positions_true = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.5, 0.0, 0.5]]
    )
    types_true = [0, 0, 0, 0]

    np.testing.assert_allclose(sup.axis, axis_true, atol=1e-6)
    np.testing.assert_allclose(sup.positions, positions_true, atol=1e-6)
    assert list(sup.types) == types_true


def test_supercell_reduced():
    """Test reduced function."""
    unitcell = Poscar(str(cwd) + "/poscar-fcc").structure
    hnf = np.array([[1, 0, 0], [0, 1, 0], [2, 0, 3]])
    supercell = supercell_reduced(unitcell, supercell_matrix=hnf)

    axis_true = np.array(
        [[-2.6965, -2.6965, 2.6965], [-0.0, 0.0, 8.0895], [-2.6965, 2.6965, -0.0]]
    )
    positions_true = np.array(
        [
            [0.0, 0.666667, 0.333333],
            [0.0, 0.666667, 0.333333],
            [0.0, 0.333333, 0.666667],
        ]
    )
    types_true = [0, 0, 0]

    np.testing.assert_allclose(supercell.axis, axis_true, atol=1e-6)
    np.testing.assert_allclose(supercell.positions, positions_true, atol=1e-6)
    assert list(supercell.types) == types_true


def test_supercell_reduced2():
    """Test reduced function."""
    unitcell = Poscar(str(cwd) + "/poscar-perovskite").structure
    hnf = np.array([[1, 0, 0], [1, 1, 0], [1, 2, 1]])

    supercell = supercell_reduced(unitcell, supercell_matrix=hnf)

    axis_true = np.array([[4.0, 0.0, -0.0], [0.0, -4.0, -0.0], [0.0, 0.0, -4.0]])
    positions_true = np.array(
        [
            [0.0, 0.5, 0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.5],
        ]
    )
    np.testing.assert_allclose(supercell.axis, axis_true, atol=1e-8)
    np.testing.assert_allclose(supercell.positions, positions_true, atol=1e-8)
    assert list(supercell.types) == [0, 1, 2, 2, 2]


def test_decompose_fraction():
    """Test decompose_fraction."""
    positions = np.array(
        [
            [0.9999999, 2.5, -1e-10],
            [0.999999999999, -0.5, -1e-13],
            [1.0001000, -1.5, -0.000001],
        ]
    )
    cells, fracs = decompose_fraction(positions, tol=1e-10)
    cells_true = np.array([[0, 2, 0], [1, -1, 0], [1, -2, -1]])

    fracs_true = np.array(
        [
            [9.99999900e-01, 5.00000000e-01, -1.00000000e-10],
            [-9.99977878e-13, 5.00000000e-01, -1.00000000e-13],
            [1.00000000e-04, 5.00000000e-01, 9.99999000e-01],
        ]
    )
    np.testing.assert_equal(cells, cells_true)
    np.testing.assert_allclose(fracs, fracs_true, atol=1e-8)

    cells, fracs = decompose_fraction(positions, tol=1e-7)
    cells_true = np.array([[1, 2, 0], [1, -1, 0], [1, -2, -1]])
    fracs_true = np.array(
        [
            [-1.000000e-07, 5.000000e-01, -1.000000e-10],
            [-9.999779e-13, 5.000000e-01, -1.000000e-13],
            [1.000000e-04, 5.000000e-01, 9.999990e-01],
        ]
    )
    np.testing.assert_equal(cells, cells_true)
    np.testing.assert_allclose(fracs, fracs_true, atol=1e-8)


def test_get_matching_positions():
    """Test get_matching_positions."""
    positions_ref = np.array(
        [
            [0.0, 0.5, 0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.5],
        ]
    )
    order = np.array([1, 2, 3, 4, 0])
    positions = positions_ref[:, order]
    sites = get_matching_positions(positions, positions_ref)
    np.testing.assert_equal(sites, order)

    sites = get_matching_positions(positions_ref, positions)
    sites_true = np.zeros(order.shape, dtype=int)
    for i, val in enumerate(order):
        sites_true[val] = i
    np.testing.assert_equal(sites, sites_true)


def test_get_unitcell_reps():
    """Test reduced function."""
    unitcell = Poscar(str(cwd) + "/poscar-fcc").structure
    hnf = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 3]])
    sup = supercell(unitcell, supercell_matrix=hnf)
    map_utos = get_unitcell_reps(unitcell, sup)
    assert map_utos[(0, (0, 0, 0))] == 0
    assert map_utos[(0, (0, 0, 1))] == 1
    assert map_utos[(0, (0, 0, 2))] == 2

    sup = supercell_reduced(unitcell, supercell_matrix=hnf)
    map_utos = get_unitcell_reps(unitcell, sup)
    assert map_utos[(0, (0, 0, 0))] == 0
    assert map_utos[(0, (1, 0, -2))] == 1
    assert map_utos[(0, (0, 0, -1))] == 2

    positions = np.array([[1, 0, 0], [-1, 0, 1], [2, 2, -2], [4, 3, 2]]).T
    sites = unitcell_reps_to_supercell_reps(positions, sup, unitcell=unitcell)
    assert list(sites) == [0, 1, 1, 2]
