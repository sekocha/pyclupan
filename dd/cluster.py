#!/usr/bin/env python
import numpy as np
import argparse
import itertools
import time

from mlptools.common.structure import Structure
from pyclupan.common.symmetry import get_permutation
from pyclupan.common.function import round_frac_array

# for test
from mlptools.common.readvasp import Poscar
from pyclupan.common.supercell import supercell
       
class Cluster:

    def __init__(self, 
                 idx=None, 
                 n_body=None, 
                 site_indices=None, 
                 cell_indices=None, 
                 ele_indices=None, 
                 primitive_lattice=None):

        self.idx = idx
        self.n_body = n_body
        self.site_indices = site_indices
        self.cell_indices = cell_indices
        self.ele_indices = ele_indices

        self.cl_positions = None

        if primitive_lattice is not None:
            self.set_primitive_lattice(primitive_lattice)
        else:
            self.prim = None

    def set_element_indices(self, ele_indices):
        self.ele_indices = ele_indices

    def set_primitive_lattice(self, primitive_lattice):
        self.prim = primitive_lattice
#        self.set_siteset_indices()
        self.set_positions()

#    def set_siteset_indices(self):
#        self.siteset_indices = []
#        for s in self.site_indices:
#            n_sum = 0
#            for idx, n in enumerate(self.prim.n_atoms):
#                n_sum += n;
#                if s < n_sum:
#                    self.siteset_indices.append(idx)
#                    break

    def set_positions(self):
        self.cl_positions = []
        for s, c in zip(self.site_indices, self.cell_indices):
            pos = self.prim.positions[:,s] + np.array(c)
            self.cl_positions.append(pos)
        self.cl_positions = np.array(self.cl_positions).T

    def compute_orbit(self, 
                      supercell_st: Structure,
                      supercell_mat: np.array=None,
                      permutations=None):

        if permutations is None:
            perm = get_permutation(supercell_st)
        else:
            perm = permutations

        sites = self.identify_cluster(supercell_st, supercell_mat)
        orbit = set([tuple(sorted(cl_perm)) 
                     for cl_perm in perm[:,np.array(sites)]])
        return sorted(orbit)

    def identify_cluster(self, 
                         supercell_st: Structure,
                         supercell_mat: np.array=None):

        if self.cl_positions is None:
            self.set_positions()

        if supercell_mat is None:
            sup_mat_inv = np.dot(np.linalg.inv(supercell_st.axis), 
                                 self.prim.axis)
        else:
            sup_mat_inv = np.linalg.inv(supercell_mat)

        cl_positions_sup = np.dot(sup_mat_inv, self.cl_positions)
        cl_positions_sup = round_frac_array(cl_positions_sup)

        site_indices = []
        for pos1 in cl_positions_sup.T:
            for idx, pos2 in enumerate(supercell_st.positions.T):
                if np.all(np.isclose(pos1, pos2)):
                    site_indices.append(idx)
        return site_indices

    def print(self):
        print(' cluster', self.idx, ':', end=' ')
        for site, cell in zip(self.site_indices, self.cell_indices):
            print(cell, site, end=' ')
        if self.ele_indices is not None:
            print(' elements =', self.ele_indices, end='')
        print('')

class ClusterSet:
    
    def __init__(self, clusters, primitive_lattice=None):

        self.clusters = clusters
        if primitive_lattice is None:
            self.prim = clusters[0].prim
        else:
            self.prim = primitive_lattice

    def print(self):
        for cl in self.clusters:
            cl.print()

#    def nonequiv_element_configs(self, elements_siteset=None):
#
#        perm = get_permutation(self.prim)
#
#        for cl in self.clusters:
#            sites = cl.site_indices
#            candidates = itertools.product\
#                (*[elements_siteset[s] for s in cl.siteset_indices])
#            perm_match = []
#            for perm in perm[:,np.array(sites)]:
#                if set(sites) == set(perm):
#                    perm_match.append(perm)
#            perm_match = np.array(perm_match)
#            print(perm_match)
#
#            for i, s in enumerate(sites):
#                perm_match == s
#
#                   

if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('-p',
                    '--poscar',
                    type=str,
                    default='POSCAR',
                    help='poscar file for primitive cell')
    args = ps.parse_args()
 
    prim = Poscar(args.poscar).get_structure_class()

#    # test FCC
#    site_indices = [0,0,0]
#    cell_indices = [[0,0,0],
#                    [1,0,0],
#                    [2,0,0]]
#    n_body = len(site_indices)
#    occ = [[0],[0],[0]]
#    elements_siteset = [[0,1,2]]

    # test perovkite
    # ***
    site_indices = [2,3,3]
    cell_indices = [[0,0,0],
                    [1,0,0],
                    [0,1,0]]
    n_body = len(site_indices)
    occ = [[2],[2],[0],[1]]
    elements_siteset = [[2],[3],[0,1]]


    cl = Cluster(0, n_body, site_indices, cell_indices, primitive_lattice=prim)
    
    H = [[3,0,0],
         [1,2,0],
         [1,0,1]]
    axis_s, positions_s, n_atoms_s = supercell(H, prim.axis, 
                                               prim.positions, 
                                               prim.n_atoms)
    sup = Structure(axis_s, positions_s, n_atoms_s)

    orbit = cl.compute_orbit(sup, H)
    print(orbit)

    clset = ClusterSet([cl])
    clset.print()
#    clset.nonequiv_element_configs(elements_siteset=elements_siteset)

