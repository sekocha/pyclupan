#!/usr/bin/env python
import numpy as np
import sys
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
            print(' hnf is required in ddtools.symmetry.get_symmetry')
            sys.exit()
        rotations_lt, translations_lt = _get_lattice_translation(symmetry, hnf)
        return symmetry['rotations'], symmetry['translations'], \
            rotations_lt, translations_lt

def get_permutation(st: Structure,
                    symprec=1e-5,
                    origin=0,
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
        if origin == 0:
            return permutation
        else:
            return permutation[:,origin:] - origin
    else:
        permutation_lt = _symmetry_to_permutation\
            (rotations_lt, translations_lt, positions)
        if origin == 0:
            return permutation, permutation_lt
        else:
            return permutation[:,origin:] - origin, \
                permutation_lt[:,origin:] - origin

def _get_lattice_translation(symmetry, hnf):

    rotations, translations = [], []
    for r, t in zip(symmetry['rotations'], symmetry['translations']):
        if np.all(np.abs(r - np.eye(3)) < 1e-10):
            vec = np.dot(hnf, t)
            if np.all(np.abs(vec - np.round(vec)) < 1e-10):
                rotations.append(r)
                translations.append(t)
    return rotations, translations

def _symmetry_to_permutation(rotations, translations, positions):

    # permutation (slice notation)
    positions = positions.astype(float)
    n_rot, n_atom = len(rotations), positions.shape[1]
    permutation = set()
    for n, (rot, trans) in enumerate(zip(rotations,translations)):
        posrot = np.dot(rot, positions) + np.tile(trans,(n_atom,1)).T
        posrot = np.array([[round_frac(p) for p in pos1] for pos1 in posrot.T])
        permutation.add\
            (tuple(np.where(distance.cdist(positions.T, posrot) < 1e-10)[1]))

    return np.array(list(permutation))


