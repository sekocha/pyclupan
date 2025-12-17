"""Tests of spglib_utils."""

from pathlib import Path

import numpy as np

from pyclupan.core.pypolymlp_utils import supercell
from pyclupan.core.spglib_utils import (
    apply_symmetry_operations,
    get_permutation,
    get_rotations,
    get_symmetry,
    refine_cell,
)

cwd = Path(__file__).parent


def test_refine_cell(fcc_primitive_cell):
    """Test refine cell."""
    ref_cell = refine_cell(fcc_primitive_cell)
    np.testing.assert_allclose(
        ref_cell.axis, [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    )
    np.testing.assert_allclose(
        ref_cell.positions,
        [[0.0, 0.0, 0.5, 0.5], [0.0, 0.5, 0.0, 0.5], [0.0, 0.5, 0.5, 0.0]],
    )


def test_symmetry(fcc_primitive_cell):
    """Test get_rotations and get_symmetry."""
    rotations = get_rotations(fcc_primitive_cell)
    rotations2, translations = get_symmetry(fcc_primitive_cell)
    np.testing.assert_allclose(rotations, rotations2, atol=1e-8)
    np.testing.assert_allclose(translations, 0.0, atol=1e-8)


def test_permutation(fcc_primitive_cell):
    """Test permutation."""
    hnf = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]])
    sup = supercell(fcc_primitive_cell, supercell_matrix=hnf)
    permutation = get_permutation(sup, superperiodic=False)
    permutation_true = np.array(
        [
            [0, 1, 2, 3],
            [0, 3, 2, 1],
            [0, 3, 2, 1],
            [0, 1, 2, 3],
            [0, 3, 2, 1],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 3, 2, 1],
            [0, 3, 2, 1],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 3, 2, 1],
            [3, 0, 1, 2],
            [3, 2, 1, 0],
            [3, 2, 1, 0],
            [3, 0, 1, 2],
            [3, 2, 1, 0],
            [3, 0, 1, 2],
            [3, 0, 1, 2],
            [3, 2, 1, 0],
            [3, 2, 1, 0],
            [3, 0, 1, 2],
            [3, 0, 1, 2],
            [3, 2, 1, 0],
            [2, 3, 0, 1],
            [2, 1, 0, 3],
            [2, 1, 0, 3],
            [2, 3, 0, 1],
            [2, 1, 0, 3],
            [2, 3, 0, 1],
            [2, 3, 0, 1],
            [2, 1, 0, 3],
            [2, 1, 0, 3],
            [2, 3, 0, 1],
            [2, 3, 0, 1],
            [2, 1, 0, 3],
            [1, 2, 3, 0],
            [1, 0, 3, 2],
            [1, 0, 3, 2],
            [1, 2, 3, 0],
            [1, 0, 3, 2],
            [1, 2, 3, 0],
            [1, 2, 3, 0],
            [1, 0, 3, 2],
            [1, 0, 3, 2],
            [1, 2, 3, 0],
            [1, 2, 3, 0],
            [1, 0, 3, 2],
        ]
    )
    np.testing.assert_equal(permutation, permutation_true)

    _, perm_lt = get_permutation(sup, superperiodic=True, hnf=hnf)
    perm_lt_true = np.array([[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0]])
    np.testing.assert_equal(perm_lt, perm_lt_true)


def test_apply_symmetry_operations(fcc_primitive_cell):
    """Test apply_symmetry_operations."""
    rotations, translations = get_symmetry(fcc_primitive_cell)

    positions = np.array([[0, 0, 1], [0, 1, 2]]).T
    fracs, cells = apply_symmetry_operations(
        rotations,
        translations,
        positions,
    )
    np.testing.assert_allclose(fracs, 0.0, atol=1e-8)
    cells_true1 = np.array([[0, 0], [0, 1], [1, 2]])
    cells_true2 = np.array([[0, 1], [-1, -3], [0, 0]])
    np.testing.assert_equal(cells[0], cells_true1)
    np.testing.assert_equal(cells[-1], cells_true2)
