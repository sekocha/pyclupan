#!/usr/bin/env python
import numpy as np
import argparse
import itertools
import time

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure

from pyclupan.common.supercell import supercell
from pyclupan.common.symmetry import get_permutation
from pyclupan.common.reduced_cell import NiggliReduced

from pyclupan.dd.dd_supercell import DDSupercell
from pyclupan.dd.dd_enumeration import DDEnumeration

class ClusterAttr:

    def __init__(self, idx, n_body, site_indices, cell_indices):
        self.idx = idx
        self.n_body = n_body
        self.site_indices = site_indices
        self.cell_indices = cell_indices

    def print(self):
        print(' cluster', self.idx, ':', end=' ')
        for site, cell in zip(self.site_indices, self.cell_indices):
            print(cell, site, end=' ')
        print('')

class Cluster:

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

    def find_nonequivalent_clusters(self, n_body_ub=2, cutoff=[1.0]):

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

        print(' compute permtations ... ')
        perm = get_permutation(st_s)
        print('   finished.')
        sites = self.set_sites(st_s)
        nonequiv_sites = np.unique(np.min(perm[:,np.array(sites)], axis=0))
        print(' number of nonequivalent sites =', len(nonequiv_sites))

        self.distance_dict = dict()
        clusters_small = clusters = [tuple([s]) for s in nonequiv_sites]

        if n_body_ub > 1:
            for n_body in range(2,n_body_ub+1):
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

        print(' number of nonequivalent clusters (< cutoff) =', len(clusters))

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

            cl_attr = ClusterAttr(cl_idx, 
                                  len(site_indices), 
                                  site_indices, 
                                  cell_indices)
            nonequiv_clusters.append(cl_attr)

        for cl_attr in nonequiv_clusters:
            cl_attr.print()

        return nonequiv_clusters
           
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


class ClusterDD:

    def __init__(self, structure, lattice=None):

        self.st = structure

        n_atoms = self.st.n_atoms
        if lattice is None:
            lattice = list(range(len(n_atoms)))
            self.occ = [lattice, lattice]
        else:
            lattice_all = list(range(len(n_atoms)))
            lattice_excluded = sorted(set(lattice_all) - set(lattice))
            self.occ = [lattice, lattice]
            for l in lattice_excluded:
                self.occ.append([l])

    def find_nonequivalent_sites(self):
        H = np.eye(3)
        dd_sup = DDSupercell(self.st.axis, H, self.st.axis,
                             positions=self.st.positions,
                             n_sites=self.st.n_atoms,
                             n_elements=len(self.occ),
                             occupation=self.occ)
        dd_enum = DDEnumeration(dd_sup, structure=self.st)
        gs_all = dd_enum.nonequivalent_permutations(num_edges=1)
        print(' number of nonequivalent sites =', gs_all.len())

        labelings = dd_sup.convert_graphs_to_labelings(gs_all)
        self.nonequiv_sites = np.where(labelings == 0)[1]


    def find_nonequivalent_clusters(self, n_body_ub=2, cutoff=[1.0]):

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

        # parameters must be generalized
        dd_sup = DDSupercell(axis_s, H, self.st.axis,
                             positions=positions_s,
                             n_sites=n_atoms_s,
                             n_elements=len(self.occ),
                             occupation=self.occ)
        dd_enum = DDEnumeration(dd_sup, structure=st_s)

        n_edges = range(1,n_body_ub+1)
        gs_all = dd_enum.nonequivalent_permutations(num_edges=n_edges)

        print(' number of nonequivalent clusters =', gs_all.len())
        labelings = dd_sup.convert_graphs_to_labelings(gs_all)
        match_sites = np.where(labelings == 0)

        nonequiv_site_indices = dict()
        for i, j in zip(*match_sites):
            if i in nonequiv_site_indices:
                nonequiv_site_indices[i].append(j)
            else:
                nonequiv_site_indices[i] = [j]

        # another policy is possible to use
        clusters = self.find_clusters_within_cutoff(axis_s,
                                                    positions_s, 
                                                    nonequiv_site_indices,
                                                    cutoff)

        print(' number of nonequivalent clusters (< cutoff) =', len(clusters))

        nonequiv_clusters = []
        T = np.dot(niggli_tmat, H)
        for cl_idx, fr in enumerate(clusters):
            fr_t = np.dot(T, fr)
            origin_cell = np.floor(fr_t[:,0]).reshape(-1,1)
            positions_cl = fr_t - np.tile(origin_cell,(1,fr_t.shape[1]))

            site_indices ,cell_indices = [], []
            for pos in positions_cl.T:
                cell = np.floor(pos).astype(int)
                pos_prim = pos - cell
                for j, pos2 in enumerate(self.st.positions.T):
                    if np.all(np.isclose(pos_prim, pos2)):
                        idx = j
                        break
                cell_indices.append(cell)
                site_indices.append(idx)

            cl_attr = ClusterAttr(cl_idx, 
                                  len(site_indices), 
                                  site_indices, 
                                  cell_indices)
            nonequiv_clusters.append(cl_attr)

        nonequiv_clusters = sorted(nonequiv_clusters, key=lambda u: u.n_body)
        for idx, cl_attr in enumerate(nonequiv_clusters):
            cl_attr.idx = idx
            cl_attr.print()

        return nonequiv_clusters
            

    def find_clusters_within_cutoff(self, 
                                    axis_s, 
                                    positions_s, 
                                    clusters, 
                                    cutoff):

        distance_dict = dict()
        clusters_cutoff = []
        for k, v in clusters.items():
            n_body = len(v)
            positions_cl = positions_s[:,np.array(v)]
            if n_body > 1:
                cut = cutoff[n_body-2]
                append = True
                positions_nearest = []
                i = v[0]
                for j in v[1:]:
                    diff1 = positions_s[:,j] - positions_s[:,i]
                    position_j = positions_s[:,j] - np.round(diff1)
                    if (i,j) in distance_dict:
                        dis = distance_dict[(i,j)]
                    else:
                        diff = position_j - positions_s[:,i]
                        dis = np.linalg.norm(np.dot(axis_s, diff))
                        distance_dict[(i,j)] = dis

                    positions_nearest.append(position_j)
                    if dis > cut + 1e-10:
                        append = False
                        break

                positions_nearest = np.array(positions_nearest).T
                if n_body > 2 and append == True:
                    for i, j in itertools.combinations(range(n_body-1),2):
                        diff = positions_nearest[:,j] - positions_nearest[:,i]
                        dis = np.linalg.norm(np.dot(axis_s, diff))
                        if dis > cut + 1e-10:
                            append = False
                            break

                if append == True:
                    app = np.zeros_like(positions_cl)
                    app[:,0] = positions_cl[:,0]
                    app[:,1:] = positions_nearest
                    clusters_cutoff.append(app)
            else:
                clusters_cutoff.append(positions_cl)

        return clusters_cutoff


    def find_orbit(self, 
                   cluster=None, 
                   supercell_st=None,
                   supercell_hnf=None):
        pass

if __name__ == '__main__':

    # Examples
    # cluster.py -p structures/perovskite-unitcell 
    #           --lattice 2 --n_body 4 --cutoff 9.0 6.0 6.0
    # cluster.py -p structures/fcc-primitive -c 2.1 -n 3

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

    st_p = Poscar(args.poscar).get_structure_class()

    cl = Cluster(st_p, lattice=args.lattice)
#    cl.find_nonequivalent_sites()
    cl.find_nonequivalent_clusters(n_body_ub=args.n_body, 
                                   cutoff=args.cutoff)
 
    dd = False
    if dd == True:
        cl_dd = ClusterDD(st_p, lattice=args.lattice)
#        cl_dd.find_nonequivalent_sites()
        cl_dd.find_nonequivalent_clusters(n_body_ub=args.n_body, 
                                          cutoff=args.cutoff)
