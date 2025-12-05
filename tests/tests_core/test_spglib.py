"""Tests of spglib_utils."""

from pathlib import Path

import numpy as np

from pyclupan.core.pypolymlp_utils import Poscar, supercell
from pyclupan.core.spglib_utils import (
    apply_symmetry_operations,
    get_permutation,
    get_rotations,
    get_symmetry,
    refine_cell,
)

cwd = Path(__file__).parent


def test_refine_cell():
    """Test refine cell."""
    unitcell = Poscar(str(cwd) + "/poscar-fcc").structure
    ref_cell = refine_cell(unitcell)
    np.testing.assert_allclose(
        ref_cell.axis, [[5.393, 0.0, 0.0], [0.0, 5.393, 0.0], [0.0, 0.0, 5.393]]
    )
    np.testing.assert_allclose(
        ref_cell.positions,
        [[0.0, 0.0, 0.5, 0.5], [0.0, 0.5, 0.0, 0.5], [0.0, 0.5, 0.5, 0.0]],
    )


def test_symmetry():
    """Test get_rotations and get_symmetry."""
    unitcell = Poscar(str(cwd) + "/poscar-fcc").structure
    rotations = get_rotations(unitcell)
    rotations2, translations = get_symmetry(unitcell)
    np.testing.assert_allclose(rotations, rotations2, atol=1e-8)
    np.testing.assert_allclose(translations, 0.0, atol=1e-8)


def test_permutation():
    """Test permutation."""
    unitcell = Poscar(str(cwd) + "/poscar-fcc").structure
    hnf = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]])
    sup = supercell(unitcell, supercell_matrix=hnf)
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


def test_apply_symmetry_operations():
    """Test apply_symmetry_operations."""
    unitcell = Poscar(str(cwd) + "/poscar-fcc").structure
    rotations, translations = get_symmetry(unitcell)

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
