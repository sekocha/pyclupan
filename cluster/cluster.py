#!/usr/bin/env python
import numpy as np
import argparse
import time

from mlptools.common.structure import Structure
from pyclupan.common.symmetry import get_permutation
from pyclupan.common.function import round_frac_array

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
        self.set_positions()

    def set_positions(self):
        self.cl_positions = []
        for s, c in zip(self.site_indices, self.cell_indices):
            pos = self.prim.positions[:,s] + np.array(c)
            self.cl_positions.append(pos)
        self.cl_positions = np.array(self.cl_positions).T

    def compute_orbit(self, 
                      supercell_st: Structure,
                      supercell_mat: np.array=None,
                      permutations=None,
                      distinguish_element=False):

        if permutations is None:
            perm = get_permutation(supercell_st)
        else:
            perm = permutations

        sites = self.identify_cluster(supercell_st, supercell_mat)
        sites_perm = perm[:,np.array(sites)]

        if distinguish_element == False:
            orbit = set([tuple(sorted(s1)) for s1 in sites_perm])
            return sorted(orbit)
        else:
            orbit = set()
            for s1 in sites_perm:
                cmpnt = [tuple([s,e]) for s,e in zip(s1,self.ele_indices)]
                orbit.add(tuple(sorted(cmpnt)))

            s_all, e_all = [], []
            for cmpnt in sorted(orbit):
                s_array, e_array = [], []
                for s, e in cmpnt:
                    s_array.append(s)
                    e_array.append(e)
                s_all.append(s_array)
                e_all.append(e_array)

            s_all = np.array(s_all)
            e_all = np.array(e_all)
 
            return (s_all, e_all)

    def identify_cluster(self, 
                         supercell_st: Structure,
                         supercell_mat: np.array=None):

        if self.cl_positions is None:
            self.set_positions()

        if supercell_mat is None:
            sup_axis_inv = np.linalg.inv(supercell_st.axis)
            sup_mat_inv = np.dot(sup_axis_inv, self.prim.axis)
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

## must be faster
#def count_orbit_components(orbit, labeling:np.array):
#    sites, ele = orbit
#    count = np.count_nonzero(np.all(labeling[sites] == ele, axis=1))
#    return count
#
#
