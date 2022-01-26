#!/usr/bin/env python
import numpy as np
import spglib 
import time

from scipy.spatial import distance

from mlptools.common.structure import Structure
from pyclupan.common.function import round_frac

def structure_to_cell(st):

    lattice = st.get_axis()
    positions = st.get_positions()
    types = st.get_types()
    cell = np.array(lattice.T), np.array(positions.T), np.array(types)
    return cell

def get_rotations(st: Structure, symprec=1e-5):

    cell = structure_to_cell(st)
    symmetry = spglib.get_symmetry(cell, symprec=symprec)
    return symmetry['rotations']

def get_symmetry(st: Structure, symprec=1e-5, superperiodic=False, hnf=None):

    cell = structure_to_cell(st)
    symmetry = spglib.get_symmetry(cell, symprec=symprec)
    if superperiodic == False:
        return symmetry['rotations'], symmetry['translations']
    else:
        if hnf is None:
            raise ValueError\
                (' hnf is required in ddtools.symmetry.get_symmetry')
        rotations_lt, translations_lt = _get_lattice_translation(symmetry, hnf)
        return symmetry['rotations'], symmetry['translations'], \
                rotations_lt, translations_lt

def get_permutation(st: Structure,
                    symprec=1e-5,
                    superperiodic=False,
                    hnf=None):

    positions = st.get_positions()
    if superperiodic == False:
        rotations, translations = get_symmetry(st, symprec=symprec)
    else:
        rotations, translations, rotations_lt, translations_lt \
            = get_symmetry(st, symprec=symprec, superperiodic=True, hnf=hnf)

    permutation = _symmetry_to_permutation(rotations, translations, positions)
    if superperiodic == False:
        return permutation
    else:
        permutation_lt = _symmetry_to_permutation(rotations_lt, 
                                                  translations_lt, 
                                                  positions)
        return permutation, permutation_lt

def _get_lattice_translation(symmetry, hnf):

    rotations, translations = [], []
    for r, t in zip(symmetry['rotations'], symmetry['translations']):
        if np.all(np.abs(r - np.eye(3)) < 1e-10):
            vec = np.dot(hnf, t)
            if np.all(np.abs(vec - np.round(vec)) < 1e-10):
                rotations.append(r)
                translations.append(t)
    return rotations, translations

def _symmetry_to_permutation(rotations, translations, positions, tol=1e-10):

    # permutation (slice notation)
    positions = positions.astype(float)
    permutation = set()
    for rot, trans in zip(rotations,translations):
        posrot = (np.dot(rot, positions).T + trans).T
        cells = np.floor(posrot).astype(int)
        rposrot = posrot - cells
        rposrot[np.where(rposrot > 1-tol)] -= 1.0
        sites = np.where(distance.cdist(rposrot.T, positions.T) < tol)[1]
        permutation.add(tuple(sites))

    return np.array(list(permutation))

def apply_symmetric_operations(rotations, 
                               translations, 
                               positions, 
                               positions_ref,
                               tol=1e-10):

    sites_all, cells_all = [], []
    for rot, trans in zip(rotations, translations):
        posrot = (np.dot(rot, positions).T + trans).T
        cells = np.floor(posrot).astype(int)
        rposrot = posrot - cells
        sites = np.where(distance.cdist(rposrot.T, positions_ref.T) < tol)[1]
        sites_all.append(sites)
        cells_all.append(cells)

    return sites_all, cells_all
    
