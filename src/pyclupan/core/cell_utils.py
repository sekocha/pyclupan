"""Utility functions for cells."""

from typing import Literal, Optional

import numpy as np

import pyclupan.core.pypolymlp_utils as pypolymlp_utils
from pyclupan.core.pypolymlp_utils import PolymlpStructure, ReducedCell

supercell = pypolymlp_utils.supercell


def supercell_reduced(
    st: PolymlpStructure,
    supercell_matrix: np.ndarray,
    method: Literal["niggli", "delaunay"] = "delaunay",
):
    """Construct supercell for a given supercell matrix."""

    st_sup = supercell(st, supercell_matrix)
    reduced = ReducedCell(st_sup.axis, method=method)
    st_sup.axis = reduced.reduced_axis
    st_sup.positions = reduced.transform_fr_coords(st_sup.positions)
    st_sup.supercell_matrix = supercell_matrix @ reduced.transformation_matrix
    return st_sup


def is_cell_equal(cell1: PolymlpStructure, cell2: PolymlpStructure):
    """Check if two cells are equal."""
    if not np.allclose(cell1.axis, cell2.axis):
        return False
    if not np.allclose(cell1.positions, cell2.positions):
        return False
    if not list(cell1.types) == list(cell2.types):
        return False
    return True


def decompose_fraction(positions: np.ndarray, tol: float = 1e-10):
    """Decompose fractional coordinates into cell and positions from 0 and 1."""
    cells = np.floor(positions + tol).astype(int)
    positions_frac = positions - cells
    return cells, positions_frac


def get_matching_positions(
    positions: np.ndarray, positions_ref: np.ndarray, tol: float = 1e-10
):
    """Calculate matching of two set of positions."""
    import scipy.spatial.distance as distance

    sites = np.where(distance.cdist(positions.T, positions_ref.T) < tol)[1]
    return sites


def get_unitcell_reps(unitcell: PolymlpStructure, supercell: PolymlpStructure):
    """Return supercell fractional coordinates in unitcell axis representation.

    Definition
    ----------
    x_unitcell = supercell_matrix @ x_supercell
    """

    supercell_matrix = np.linalg.inv(unitcell.axis) @ supercell.axis
    if not np.allclose(supercell_matrix, np.round(supercell_matrix)):
        raise RuntimeError("Supercell matrix is not integer matrix.")

    positions = supercell_matrix @ supercell.positions

    n_expand = int(round(np.linalg.det(supercell_matrix)))
    n_atom_unitcell = unitcell.positions.shape[1]
    quotient_sites = [i for i in range(n_atom_unitcell) for n in range(n_expand)]

    cells, positions_frac = decompose_fraction(positions)
    for i, pos in enumerate(positions_frac.T):
        if not np.allclose(pos, unitcell.positions[:, quotient_sites[i]]):
            raise RuntimeError(
                "Positions in supercell cannot be mapped onto positions in unitcell."
            )

    map_unitcell_to_supercell_site = dict()
    for i, (site, cell) in enumerate(zip(quotient_sites, cells.T)):
        map_unitcell_to_supercell_site[site, tuple(cell)] = i
    return map_unitcell_to_supercell_site


def unitcell_reps_to_supercell_reps(
    positions: np.ndarray,
    supercell: PolymlpStructure,
    supercell_matrix_inv: Optional[np.ndarray] = None,
    unitcell: Optional[PolymlpStructure] = None,
):
    """Transform positions in unitcell rep. to those in supercell rep."""
    if supercell_matrix_inv is None:
        positions_sup = np.linalg.inv(supercell.axis) @ unitcell.axis @ positions
    else:
        positions_sup = supercell_matrix_inv @ positions
    _, positions_sup = decompose_fraction(positions_sup)
    sites = get_matching_positions(positions_sup, supercell.positions)
    return sites
