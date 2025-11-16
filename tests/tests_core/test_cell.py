"""Tests of cell_utils."""

from pathlib import Path

import numpy as np

from pyclupan.core.cell_utils import decompose_fraction, supercell_reduced
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


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
