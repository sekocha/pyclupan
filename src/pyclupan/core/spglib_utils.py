"""Utility functions for spglib."""

import copy
from typing import Optional

import numpy as np
import spglib
from symfc.utils.utils import compute_sg_permutations

import pyclupan.core.pypolymlp_utils as pypolymlp_utils
from pyclupan.core.cell_positions_utils import (
    decompose_fraction,
    get_matching_positions,
)
from pyclupan.core.pypolymlp_utils import PolymlpStructure

ReducedCell = pypolymlp_utils.ReducedCell


def _structure_to_cell(st: PolymlpStructure):
    """Transform structure to spglib cell."""
    cell = (st.axis.T, st.positions.T, np.array(st.types))
    return cell


def refine_cell(st: PolymlpStructure, symprec: float = 1e-5):
    """Refine cell."""
    cell = _structure_to_cell(st)

    map_elements = dict()
    for e, t in zip(st.elements, st.types):
        map_elements[t] = e

    lattice1, position1, types1 = spglib.refine_cell(cell, symprec=symprec)
    elements1 = [map_elements[t] for t in types1]
    st_rev = copy.deepcopy(st)
    st_rev.axis = lattice1.T
    st_rev.positions = position1.T
    st_rev.types = types1
    st_rev.elements = elements1
    st_rev = st_rev.reorder()
    return st_rev


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
            raise RuntimeError("HNF required if superperiodic = True.")
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


def apply_symmetry_operations(
    rotations: np.ndarray,
    translations: np.ndarray,
    positions: np.ndarray,
    positions_ref: Optional[np.ndarray] = None,
    tol: float = 1e-10,
):
    """Apply symmetry operations to fractional coordinates."""

    rot_positions, rot_cells = [], []
    for rot, trans in zip(rotations, translations):
        posrot = ((rot @ positions).T + trans).T
        cells, rposrot = decompose_fraction(posrot, tol=tol)
        rot_positions.append(rposrot)
        rot_cells.append(cells)

    if positions_ref is not None:
        rot_sites = [
            get_matching_positions(pos, positions_ref, tol=tol) for pos in rot_positions
        ]
        return np.array(rot_sites), np.array(rot_cells)

    return np.array(rot_positions), np.array(rot_cells)
