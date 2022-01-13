#!/usr/bin/env python
import numpy as np
import argparse
import itertools
import time

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure

from pyclupan.common.supercell import supercell
from pyclupan.common.symmetry import get_permutation
from pyclupan.common.function import round_frac_array

from pyclupan.dd.cluster import ClusterAttr

class ClusterOrbit:

    def __init__(self, 
                 cluster: ClusterAttr,
                 primitive_lattice: Structure):

        self.cl = cluster
        self.prim = primitive_lattice

        self.cl_positions = []
        for s, c in zip(self.cl.site_indices, self.cl.cell_indices):
            pos = self.prim.positions[:,s] + np.array(c)
            self.cl_positions.append(pos)
        self.cl_positions = np.array(self.cl_positions).T

    def compute_orbit(self, 
                      supercell_st: Structure,
                      supercell_mat: np.array=None):

        perm = get_permutation(supercell_st)
        sites = self.identify_cluster(supercell_st, supercell_mat)

        orbit = set([tuple(sorted(cl_perm)) 
                     for cl_perm in perm[:,np.array(sites)]])
        return sorted(orbit)

    def identify_cluster(self, 
                         supercell_st: Structure,
                         supercell_mat: np.array=None):

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

if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('-p',
                    '--poscar',
                    type=str,
                    default='POSCAR',
                    help='poscar file for primitive cell')
    args = ps.parse_args()
 
    prim = Poscar(args.poscar).get_structure_class()

    # test FCC
    site_indices = [0,0,0]
    cell_indices = [[0,0,0],
                    [1,0,0],
                    [2,0,0]]
    n_body = len(site_indices)

    # test perovkite
    # ***

    cl = ClusterAttr(0, n_body, site_indices, cell_indices)
    
    H = [[3,0,0],
         [1,2,0],
         [1,0,1]]
    axis_s, positions_s, n_atoms_s = supercell(H, prim.axis, 
                                               prim.positions, 
                                               prim.n_atoms)
    sup = Structure(axis_s, positions_s, n_atoms_s)

    cl_orb = ClusterOrbit(cl, prim)
    orbit = cl_orb.compute_orbit(sup, H)

