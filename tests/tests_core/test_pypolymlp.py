"""Tests of pypolymlp_utils."""

from pathlib import Path

import numpy as np
import pytest

from pyclupan.core.pypolymlp_utils import (
    KbEV,
    Polymlp,
    PolymlpStructure,
    Poscar,
    ReducedCell,
    Vasprun,
    load_cell,
    load_cells,
    save_cell,
    save_cells,
    supercell,
    write_poscar_file,
)

cwd = Path(__file__).parent


def test_pypolymlp_reduced_cell(fcc_primitive_cell):
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


def test_pypolymlp_functions():
    """Test pypolymlp functions."""
    _ = save_cell
    _ = save_cells
    _ = load_cell
    _ = load_cells
    _ = PolymlpStructure
    _ = Vasprun
    _ = write_poscar_file
    assert KbEV == pytest.approx(8.617389435726849e-05)


def test_pypolymlp_calculations():
    """Test Polymlp."""
    path = str(cwd) + "/../files/Ag-Au/"
    polymlp = Polymlp(pot=path + "polymlp.yaml")
    st = Poscar(path + "POSCAR.L12.Ag3Au").structure
    polymlp.run_geometry_optimization(st)
    assert polymlp.energy == pytest.approx(-10.806689752751552)
    assert polymlp.structure.axis[0, 0] == pytest.approx(4.146121177258233)
    assert isinstance(polymlp.structure, PolymlpStructure)
