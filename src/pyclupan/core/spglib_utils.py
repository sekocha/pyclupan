"""Utility functions for spglib."""

from typing import Optional

import numpy as np
import spglib
from pypolymlp.core.data_format import PolymlpStructure
from symfc.utils.utils import compute_sg_permutations

# from scipy.spatial import distance


def _structure_to_cell(st: PolymlpStructure):
    """Transform structure to spglib cell."""
    cell = (st.axis.T, st.positions.T, np.array(st.types))
    return cell


def get_rotations(st: PolymlpStructure, symprec: float = 1e-5):
    """Calculate rotations."""
    cell = _structure_to_cell(st)
    symmetry = spglib.get_symmetry(cell, symprec=symprec)
    return symmetry["rotations"]


def get_symmetry(st: PolymlpStructure, symprec: float = 1e-5):
    """Calculate symmetry operations."""
    cell = _structure_to_cell(st)
    symmetry = spglib.get_symmetry(cell, symprec=symprec)
    return symmetry["rotations"], symmetry["translations"]


def get_permutation(
    st: PolymlpStructure,
    superperiodic: bool = False,
    hnf: Optional[np.ndarray] = None,
    symprec: float = 1e-5,
):
    """Calculate atomic permutations by symmetry operations."""
    rotations, translations = get_symmetry(st, symprec=symprec)
    permutation = compute_sg_permutations(
        st.positions.T,
        rotations,
        translations,
        st.axis.T,
        symprec=symprec,
    )
    if not superperiodic:
        return permutation

    if superperiodic:
        if hnf is None:
            raise RuntimeError("HNF required.")
        lt_ids = _get_lattice_translations(rotations, translations, hnf)
        return permutation, permutation[lt_ids]


def _get_lattice_translations(
    rotations: np.ndarray,
    translations: np.ndarray,
    hnf: np.ndarray,
):
    """Calculate operations of lattice translation for HNF."""
    lattice_translations_ids = []
    for i, (r, t) in enumerate(zip(rotations, translations)):
        if np.allclose(r, np.eye(3)):
            vec = hnf @ t
            if np.allclose(vec, np.round(vec)):
                lattice_translations_ids.append(i)
    return np.array(lattice_translations_ids)


# def _symmetry_to_permutation(rotations, translations, positions, tol=1e-10):
#
#     # permutation (slice notation)
#     positions = positions.astype(float)
#     permutation = set()
#     for rot, trans in zip(rotations, translations):
#         posrot = (np.dot(rot, positions).T + trans).T
#         cells = np.floor(posrot).astype(int)
#         rposrot = posrot - cells
#         rposrot[np.where(rposrot > 1 - tol)] -= 1.0
#         sites = np.where(distance.cdist(rposrot.T, positions.T) < tol)[1]
#         permutation.add(tuple(sites))
#
#     return np.array(list(permutation))
#

#
# def apply_symmetric_operations(rotations,
#                                translations,
#                                positions,
#                                positions_ref,
#                                tol=1e-10):
#
#     sites_all, cells_all = [], []
#     for rot, trans in zip(rotations, translations):
#         posrot = (np.dot(rot, positions).T + trans).T
#         cells = np.floor(posrot).astype(int)
#         rposrot = posrot - cells
#         sites = np.where(distance.cdist(rposrot.T, positions_ref.T) < tol)[1]
#         sites_all.append(sites)
#         cells_all.append(cells)
#
#     return sites_all, cells_all
#
