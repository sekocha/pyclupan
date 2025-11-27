"""Utility functions for cells."""

import copy
import itertools
from typing import Literal, Optional

import numpy as np

import pyclupan.core.pypolymlp_utils as pypolymlp_utils
from pyclupan.core.pypolymlp_utils import PolymlpStructure, ReducedCell

supercell = pypolymlp_utils.supercell
# supercell_diagonal = pypolymlp_utils.supercell_diagonal


def supercell_diagonal(st: PolymlpStructure, size: tuple = (2, 2, 2)):
    """Construct supercell using a diagonal supercell matrix."""
    supercell_matrix = np.diag(size)
    n_expand = np.prod(size)

    sup = copy.deepcopy(st)
    sup.axis = st.axis @ supercell_matrix
    sup.n_atoms = np.array(st.n_atoms) * n_expand
    sup.types = np.repeat(st.types, n_expand)
    sup.elements = np.repeat(st.elements, n_expand)
    sup.volume = st.volume * n_expand
    sup.supercell_matrix = supercell_matrix

    trans_all = np.indices(size).reshape(3, -1).T
    positions_new = (st.positions.T[:, None] + trans_all[None, :]).reshape((-1, 3))
    sup.positions = (positions_new / size).T
    return sup


def reduced(
    st: PolymlpStructure,
    method: Literal["niggli", "delaunay"] = "delaunay",
    return_transformation: bool = False,
):
    """Return structure with reduced axis."""
    reduced = ReducedCell(st.axis, method=method)
    st_reduced = copy.deepcopy(st)
    st_reduced.axis = reduced.reduced_axis
    st_reduced.positions = reduced.transform_fr_coords(st.positions)
    if return_transformation:
        return st_reduced, reduced.transformation_matrix
    return st_reduced


def supercell_reduced(
    unitcell: PolymlpStructure,
    supercell_matrix: np.ndarray,
    method: Literal["niggli", "delaunay"] = "delaunay",
):
    """Construct supercell for a given supercell matrix."""

    st_sup = supercell(unitcell, supercell_matrix)
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
    if sites.shape[0] != positions.shape[1]:
        raise RuntimeError("Any positions are inconsitent with reference positions.")
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


def supercell_pyclupan(unitcell: PolymlpStructure, supercell_matrix: np.ndarray):
    """Construct supercell for a given supercell matrix.

    Algorithm
    ---------
    1. Calculate supercell basis, A @ H, where H is supercell_matrix.
    2. Calculate Smith normal form, S = U @ H @ V.
    3. Calculate fractional coordinates in another supercell basis A @ H @ V.
       Using the new basis, the lattice is isomorphic to
       group Z_S[0,0] + Z_S[1,1] + Z_S[2,2].

       Lattice points of unitcell are given by integers of r = S @ z.
          A @ H @ V @ z = A @ [U^(-1)] @ S @ z
          (z: fractional coordinates in basis A @ H @ V)
       Fractional coordinates in A @ H supercell are calculated as V @ S^(-1) @ r.
    """
    from pyclupan.core.linalg_utils import snf

    S, U, V = snf(supercell_matrix)
    axis = unitcell.axis @ supercell_matrix

    coord_int = itertools.product(*[range(S[0, 0]), range(S[1, 1]), range(S[2, 2])])
    plattice_H = [V @ (np.array(c) / np.diag(S)) for c in coord_int]

    positions_H = np.linalg.inv(supercell_matrix) @ unitcell.positions
    positions_new = []
    for pos, lattice in itertools.product(*[positions_H.T, plattice_H]):
        _, frac = decompose_fraction(pos + lattice)
        positions_new.append(frac)
    positions_new = np.array(positions_new).T

    n_expand = S[0, 0] * S[1, 1] * S[2, 2]
    n_atoms = [n * n_expand for n in unitcell.n_atoms]
    types = [i for i, n in enumerate(n_atoms) for _ in range(n)]
    elements = [unitcell.elements[i] for i in types]

    st_supercell = PolymlpStructure(
        axis=axis,
        positions=positions_new,
        n_atoms=n_atoms,
        types=types,
        elements=elements,
    )
    return st_supercell
