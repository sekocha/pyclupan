#!/usr/bin/env python
import numpy as np
import itertools
from math import *

from mlptools.common.structure import Structure
from mlptools.common.readvasp import Poscar

from pyclupan.common.normal_form import snf
from pyclupan.common.function import round_frac, round_frac_array

class SupercellSite:
    
    def __init__(self, idx, site_pl, cell_plrep, position_plrep):
        self.idx = idx
        self.site_pl = site_pl
        self.cell_plrep = cell_plrep
        self.position_plrep = position_plrep

class Supercell:
    
    def __init__(self, 
                 st_prim=None, 
                 poscar_name=None,
                 axis=None, 
                 positions=None, 
                 n_atoms=None,
                 hnf=None):

        if st_prim is not None:
            self.st_prim = st_prim
        elif poscar_name is not None:
            p = Poscar(poscar_name)
            self.st_prim = p.get_structure_class()
        else:
            self.st_prim = Structure(axis, positions, n_atoms)

        if hnf is None:
            raise ValueError('hnf is required in Supercell')

        self.hnf = hnf
        self.hnf_inv = np.linalg.inv(self.hnf)
        self.construct_supercell()

        self.plrep = None
        self.map_plrep = None

    def construct_supercell(self):

        axis = self.st_prim.axis
        positions = self.st_prim.positions
        n_atoms = self.st_prim.n_atoms
        H = self.hnf
        Hinv = self.hnf_inv
    
        ## S = U * H * V, H = U^(-1) * S * V^(-1)
        S, U, V = snf(H)
    
        axis_new = np.dot(axis, H)
        self.n_expand = S[0,0] * S[1,1] * S[2,2]
        n_atoms_new = [n * self.n_expand for n in n_atoms]
   
        ###########################################################
        # 1. a basis for supercell: A * H
        # 2. another basis: A * H * V
        # 3. Using the new basis, the lattice is isomorphic to 
        #       Z_S[0,0] + Z_S[1,1] + Z_S[2,2]
        #   A * H * V * z = A * [U^(-1)] * S * z 
        #   (z: fractional coordinates in basis A * H * V)
        # 4. fractional coordinates in basis A * H: V * z
        ###########################################################
    
        coord_int = itertools.product(*[range(S[0,0]),
                                        range(S[1,1]),
                                        range(S[2,2])])
        self.plattice_H = [np.dot(V, np.array(c) / np.diag(S)) 
                           for c in coord_int]
    
        positions_H = np.dot(Hinv, positions)
        positions_new = []
        for pos, lattice in itertools.product(*[positions_H.T, 
                                                self.plattice_H]):
            positions_new.append([round_frac(p) for p in (pos + lattice)])
        positions_new = np.array(positions_new).T
    
        self.st_supercell = Structure(axis_new, positions_new, n_atoms_new)
        self.primitive_sites = [i for i in range(positions.shape[1]) 
                                  for n in range(self.n_expand)]

    def get_supercell(self):
        return self.st_supercell

    def get_primitive_contraction(self, supercell_site_idx):
        p_site_idx = self.primitive_sites[supercell_site_idx]
        return p_site_idx, self.st_prim.positions[:,p_site_idx]

    def get_lattice_positions_primitive_lattice_representation(self):
        return np.dot(self.hnf, self.plattice_H.T)

    def set_primitive_lattice_representation(self):

        positions_plrep = np.dot(self.hnf, self.st_supercell.positions)

        self.plrep = []
        for i, pos_plrep in enumerate(positions_plrep.T):
            site_pl, pos_pl = self.get_primitive_contraction(i)
            cell_plrep = np.round(pos_plrep - pos_pl).astype(int)
            self.plrep.append(SupercellSite(i, site_pl, cell_plrep, pos_plrep))

        self.map_plrep = dict()
        for site in self.plrep:
            self.map_plrep[(site.site_pl,tuple(site.cell_plrep))] = site.idx

        return self.plrep, self.map_plrep

    def identify_site_idx(self, site_pl, cell_plrep):

#        # set_primitive_lattice_representation must be called in advance
#        if self.plrep is None:
#            self.set_primitive_lattice_representation()

        prim = self.st_prim
        attr = (site_pl, tuple(cell_plrep))
        if not attr in self.map_plrep:
            pos_plrep = prim.positions[:,site_pl] + np.array(cell_plrep)
            pos_slrep = round_frac_array(np.dot(self.hnf_inv, pos_plrep))
            pos_plrep_rev = np.dot(self.hnf, pos_slrep)
            cell_plrep_rev = pos_plrep_rev - prim.positions[:,site_pl]
            cell_plrep_rev = np.round(cell_plrep_rev).astype(int)
            attr_rev = (site_pl, tuple(cell_plrep_rev))
            self.map_plrep[attr] = self.map_plrep[attr_rev]

        return self.map_plrep[attr]

if __name__ == '__main__':

    axis = np.array([[1,1,0],[1,0,1],[0,1,1]])
    positions = np.array([[0,0.75],[0,0.75],[0,0.75]])
    n_atoms = [1,1]

    hnf = np.array([[1,0,0],[0,2,0],[0,0,3]])
 
    axis1, positions1, n_atoms1 = supercell(hnf, axis, positions, n_atoms)
    st = Structure(axis1, positions1, n_atoms1)
    
