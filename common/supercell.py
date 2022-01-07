#!/usr/bin/env python
import numpy as np
import itertools
from math import *

from mlptools.common.structure import Structure
from mlptools.common.readvasp import Poscar
from pyclupan.common.normal_form import snf
from pyclupan.common.function import round_frac

def supercell(H, axis, positions, n_atoms):

    ## S = U * H * V, H = U^(-1) * S * V^(-1)
    S, U, V = snf(H)
    Hinv = np.linalg.inv(H)

    axis_new = np.dot(axis, H)
    n_expand = S[0,0] * S[1,1] * S[2,2]
    n_atoms_new = [n * n_expand for n in n_atoms]

    ###########################################################
    # 1. a basis for supercell: A * H
    # 2. another basis: A * H * V
    # 3. Using the new basis, the lattice is isomorphic to 
    #       Z_S[0,0] + Z_S[1,1] + Z_S[2,2]
    #   A * H * V * z = A * [U^(-1)] * S * z 
    #   (z: fractional coordinates in basis A * H * V)
    # 4. fractional coordinates in basis A * H: V * z
    ###########################################################

    coord_int = itertools.product(*[range(S[0,0]),range(S[1,1]),range(S[2,2])])
    plattice_H = [np.dot(V, np.array(c) / np.diag(S)) for c in coord_int]

    positions_H = np.dot(Hinv, positions)
    positions_new = []
    for pos, lattice in itertools.product(*[positions_H.T, plattice_H]):
        positions_new.append([round_frac(p) for p in (pos + lattice)])
    positions_new = np.array(positions_new).T

    return axis_new, positions_new, n_atoms_new

def supercell_from_structure(st, H, return_structure=False):

    axis_new, positions_new, n_atoms_new \
        = supercell(H,st.get_axis(),st.get_positions(),st.get_n_atoms())
    if return_structure == False:
        return axis_new, positions_new, n_atoms_new
    return Structure(axis_new, positions_new, n_atoms_new)

def supercell_from_poscar(poscar_name, H, return_structure=False):

    p = Poscar(poscar_name)
    axis, positions, n_atoms, _, _ = p.get_structure()

    axis_new, positions_new, n_atoms_new = supercell(H,axis,positions,n_atoms)
    if return_structure == False:
        return axis_new, positions_new, n_atoms_new
    return Structure(axis_new, positions_new, n_atoms_new)

if __name__ == '__main__':

    axis = np.array([[1,1,0],[1,0,1],[0,1,1]])
    positions = np.array([[0,0.75],[0,0.75],[0,0.75]])
    n_atoms = [1,1]

    hnf = np.array([[1,0,0],[0,2,0],[0,0,3]])
 
    axis1, positions1, n_atoms1 = supercell(hnf, axis, positions, n_atoms)
    st = Structure(axis1, positions1, n_atoms1)
    
