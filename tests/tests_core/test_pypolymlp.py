"""Tests of pypolymlp_utils."""

from pathlib import Path

import numpy as np

from pyclupan.core.pypolymlp_utils import (
    ReducedCell,
    load_cell,
    load_cells,
    save_cell,
    save_cells,
    supercell,
)

cwd = Path(__file__).parent


def test_pypolymlp_functions(fcc_primitive_cell):
    """Test pypolymlp functions."""
    hnf = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    sup = supercell(fcc_primitive_cell, supercell_matrix=hnf)

    reduced = ReducedCell(sup.axis, method="delaunay")
    reduced_axis = reduced.reduced_axis
    reduced_positions = reduced.transform_fr_coords(sup.positions)

    axis_true = np.array([[-2.0, 0.0, 4.0], [-0.0, 2.0, 2.0], [-2.0, 2.0, -2.0]])
    positions_true = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, 0.5]])
    tmat_true = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    np.testing.assert_allclose(reduced_axis, axis_true, atol=1e-8)
    np.testing.assert_allclose(reduced_positions, positions_true, atol=1e-8)
    np.testing.assert_allclose(reduced.transformation_matrix, tmat_true, atol=1e-8)

    _ = save_cell
    _ = save_cells
    _ = load_cell
    _ = load_cells
