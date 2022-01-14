#!/usr/bin/env python
import numpy as np
import argparse
import itertools
import time
import copy

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure

from pyclupan.common.supercell import supercell
from pyclupan.common.symmetry import get_permutation
from pyclupan.common.reduced_cell import NiggliReduced

from pyclupan.cluster.cluster import Cluster, ClusterSet
from pyclupan.io.yaml import Yaml

class NonequivalentClusters:

    def __init__(self, structure, lattice=None):

        self.st = structure
        if lattice is None:
            self.lattice = list(range(len(self.st.n_atoms)))
        else:
            self.lattice = lattice

    def set_sites(self, st):

        sites = []
        idx = 0
        for i, n in enumerate(st.n_atoms):
            if i in self.lattice:
                for j in range(n):
                    sites.append(idx)
                    idx += 1
            else:
                idx += n
        return sites

    def find_nonequivalent_sites(self):
        perm = get_permutation(self.st)
        sites = self.set_sites(self.st)
        nonequiv_sites = np.unique(np.min(perm[:,np.array(sites)], axis=0))
        print(' number of nonequivalent sites =', len(nonequiv_sites))

    def find_nonequivalent_clusters(self, 
                                    n_body_ub=2, 
                                    cutoff=[1.0],
                                    elements_lattice=None):

        niggli = NiggliReduced(self.st.axis)
        niggli_axis, niggli_tmat = niggli.niggli_axis, niggli.tmat
        niggli_positions = niggli.transform_fr_coords(self.st.positions)

        maxcut = max(cutoff) 
        niggli_norm = np.linalg.norm(niggli_axis, axis=0)
        H = np.diag(np.ceil(np.ones(3)*maxcut*2 / niggli_norm))

        axis_s, positions_s, n_atoms_s = supercell(H, 
                                                   niggli_axis,
                                                   niggli_positions,
                                                   self.st.n_atoms)
        st_s = Structure(axis_s, positions_s, n_atoms_s)

        print(' computing permutations ... ')
        perm = get_permutation(st_s)
        print('   finished.')
        sites = self.set_sites(st_s)
        nonequiv_sites = np.unique(np.min(perm[:,np.array(sites)], axis=0))
        print(' number of nonequivalent sites =', len(nonequiv_sites))

        self.distance_dict = dict()
        clusters_small = clusters = [tuple([s]) for s in nonequiv_sites]

        if n_body_ub > 1:
            for n_body in range(2,n_body_ub+1):
                print("  searching for "+str(n_body)+"-body clusters ...")
                cut = cutoff[n_body-2]
                clusters_cutoff = []
                for cl, s in itertools.product(clusters_small, sites):
                    if not s in cl:
                        cl_trial = sorted(list(cl) + [s])
                        is_cutoff, _ = self.check_cluster_cutoff(cl_trial, 
                                                                 axis_s, 
                                                                 positions_s,
                                                                 cut)
                        if is_cutoff:
                            clusters_cutoff.append(cl_trial)

                cl_rep = set()
                for cl in clusters_cutoff:
                    cl_perm = perm[:,np.array(cl)]
                    cl_rep.add(min([tuple(sorted(p)) for p in cl_perm]))

                clusters_small = sorted(cl_rep)
                clusters.extend(clusters_small)
                print("  ", len(clusters_small), 
                      str(n_body)+"-body clusters are found.")

        print(' number of nonequivalent clusters (< cutoff) =', len(clusters))

        if elements_lattice is not None:
            element_combs = self.find_nonequiv_element_combinations\
                                                            (clusters, 
                                                             perm, 
                                                             st_s.types,
                                                             elements_lattice)
 
        nonequiv_clusters = []
        T = np.dot(niggli_tmat, H)
        for cl_idx, cl in enumerate(clusters):
            n_body = len(cl)
            positions_cl = np.zeros((3,n_body))
            positions_cl[:,0] = positions_s[:,cl[0]]
            if n_body > 1:
                cut = cutoff[n_body-2]
                _, positions_j = self.check_cluster_cutoff(cl, 
                                                           axis_s, 
                                                           positions_s,
                                                           cut)
                positions_cl[:,1:] = positions_j
                
            site_indices, cell_indices = [], []
            for pos in positions_cl.T:
                pos_t = np.dot(T, pos)
                cell = np.floor(pos_t).astype(int)
                pos_prim = pos_t - cell
                for j, pos2 in enumerate(self.st.positions.T):
                    if np.all(np.isclose(pos_prim, pos2)):
                        idx = j
                        break
                site_indices.append(idx)
                cell_indices.append(cell)

            cl_attr = Cluster(cl_idx, 
                              len(site_indices), 
                              site_indices, 
                              cell_indices)
            nonequiv_clusters.append(cl_attr)

        cl_set = ClusterSet(nonequiv_clusters)

        if elements_lattice is None:
            return cl_set, None

        nonequiv_clusters_ele = []
        for cl, combs in zip(nonequiv_clusters, element_combs):
            for ele_indices in combs:
                cl1 = copy.copy(cl)
                cl1.set_element_indices(ele_indices)
                nonequiv_clusters_ele.append(cl1)

        cl_set_ele = ClusterSet(nonequiv_clusters_ele)
        return cl_set, cl_set_ele
          
    def compute_distance(self, axis, positions, i, j):

        diff1 = positions[:,j] - positions[:,i]
        position_j = positions[:,j] - np.round(diff1)
        if (i,j) not in self.distance_dict:
            diff = position_j - positions[:,i]
            dis = np.linalg.norm(np.dot(axis, diff))
            self.distance_dict[(i,j)] = dis

        return self.distance_dict[(i,j)], position_j

    def check_cluster_cutoff(self, cluster, axis, positions, cut, tol=1e-10):

        is_cutoff = True
        positions_nearest = []
        n_body = len(cluster)
        i = cluster[0]
        for j in cluster[1:]:
            dis, position_j = self.compute_distance(axis, positions, i, j)
            positions_nearest.append(position_j)
            if dis > cut + tol:
                is_cutoff = False
                break

        positions_nearest = np.array(positions_nearest).T
        if n_body > 2 and is_cutoff == True:
            for i, j in itertools.combinations(range(n_body-1),2):
                diff = positions_nearest[:,j] - positions_nearest[:,i]
                dis = np.linalg.norm(np.dot(axis, diff))
                if dis > cut + tol:
                    is_cutoff = False
                    break

        return is_cutoff, positions_nearest

    def find_nonequiv_element_combinations(self, 
                                           clusters, 
                                           perm, 
                                           types,
                                           elements_lattice):

        element_combinations = []
        for cl_sites in clusters:
            cl_sites = np.array(cl_sites)
            candidates = itertools.product\
                (*[elements_lattice[types[s]] for s in cl_sites])

            perm_match = set()
            for p in perm[:,cl_sites]:
                if set(cl_sites) == set(p):
                    perm_match.add(tuple(p))

            perm_match = np.array([np.array(p) for p in sorted(perm_match)])
            for i, site in enumerate(cl_sites):
                perm_match[np.where(perm_match == site)] = i

            element_combs = set()
            for c in candidates:
                c_rep = min([tuple(np.array(c)[p]) for p in perm_match])
                element_combs.add(c_rep)
            element_combinations.append(sorted(element_combs))

        return element_combinations

 
if __name__ == '__main__':

    # Examples
    # cluster_nonequiv.py -p structures/perovskite-unitcell 
    #                   --lattice 2 --n_body 4 --cutoff 9.0 6.0 6.0
    # cluster_nonequiv.py -p structures/fcc-primitive -c 2.1 -n 3
    # cluster_nonequiv.py -p structures/perovskite-unitcell 
    #                   -c 7.0 5.0 -n 3 -l 2 -e 0 -e 1 -e 2 3

    ps = argparse.ArgumentParser()
    ps.add_argument('-p',
                    '--poscar',
                    type=str,
                    default='POSCAR',
                    help='poscar file for primitive cell')
    ps.add_argument('-l',
                    '--lattice',
                    nargs='*',
                    type=int,
                    default=None,
                    help='lattice indices')
    ps.add_argument('-e',
                    '--elements',
                    nargs='*',
                    type=int,
                    action='append',
                    default=None,
                    help='elements on a lattice')
    ps.add_argument('-c',
                    '--cutoff',
                    nargs='*',
                    type=float,
                    default=None,
                    help='cutoff [2-body, 3-body, ...]')
    ps.add_argument('-n',
                    '--n_body',
                    type=int,
                    default=2,
                    help='Maximum number of sites')
    args = ps.parse_args()

    if args.cutoff is None:
        args.cutoff = [6.0 for n in range(args.n_body-1)]
    if len(args.cutoff) < args.n_body - 1:
        for i in range(args.n_body - 1 - len(args.cutoff)):
            args.cutoff.append(args.cutoff[-1])

    prim = Poscar(args.poscar).get_structure_class()

    if args.elements is not None:
        print(' elements on lattices =', args.elements)
        if len(args.elements) != len(prim.n_atoms):
            raise ValueError('-e options must be repeated n_lattice times')
        if args.lattice is None:
            args.lattice = [i for i, e in enumerate(args.elements) 
                              if len(e) > 1]
            print(' lattice =', args.lattice)

    clobj = NonequivalentClusters(prim, lattice=args.lattice)
    cl, cl_ele = clobj.find_nonequivalent_clusters\
                                     (n_body_ub=args.n_body, 
                                      cutoff=args.cutoff,
                                      elements_lattice=args.elements)

    yaml = Yaml()
    yaml.write_clusters_yaml(prim, args.cutoff, cl, args.elements, cl_ele)


