#!/usr/bin/env python
import numpy as np
import collections
import time
from collections import defaultdict

from mlptools.common.structure import Structure
from pyclupan.common.symmetry import apply_symmetric_operations
from pyclupan.common.symmetry import get_symmetry

class OrbitAttr:

    def __init__(self):
        self.sites = []
        self.cells = []
        self.elements = []
        self.supercell_sites = []
        self.map_supercell_sites = dict()

    def append_cluster(self, cluster):
        self.sites.append(tuple([c[0] for c in cluster]))
        self.cells.append(np.array([c[1:4] for c in cluster]))
        self.elements.append(tuple([c[4] for c in cluster]))

    def optimize_type(self):
        self.cells = np.array(self.cells)

    def eliminate_duplicates(self):

        orbit = [(s,e) for s, e in zip(self.supercell_sites, self.elements)]
        count = collections.Counter(orbit)

        supercell_sites, elements = [], []
        for k, v in count.items():
            multiplicity = round(v/len(k[0]))
            for i in range(multiplicity):
                supercell_sites.append(k[0])
                elements.append(k[1])

        self.supercell_sites = supercell_sites
        self.elements = elements

    def eliminate_incomplete(self):

        supercell_sites, elements = [], []
        for sites, eles in zip(self.supercell_sites, self.elements):
            d = defaultdict(list)
            for s, e in zip(sites, eles):
                d[s].append(e)
            if np.all(np.array([len(set(v)) for v in d.values()]) == 1):
                supercell_sites.append(sites)
                elements.append(eles)

        self.supercell_sites = supercell_sites
        self.elements = elements

    def get_orbit_supercell(self):
        return np.array(self.supercell_sites), np.array(self.elements)

    def count(self):
        orbit = [(s,e) for s, e in zip(self.supercell_sites, self.elements)]
        return collections.Counter(orbit)

class Cluster:

    """
    for single cluster orbit
    orbit_attr = clusters.clusters[i].find_orbit_supercell
                                    (sup, rotations, translations)
                        
    """   
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

        # used for computing cluster orbit
        self.sites_sym = None
        self.cells_sym = None
        self.orbit_attr_prim = None
    
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

    def print(self):
        print(' cluster', self.idx, ':', end=' ')
        for site, cell in zip(self.site_indices, self.cell_indices):
            print(cell, site, end=' ')
        if self.ele_indices is not None:
            print(' elements =', self.ele_indices, end='')
        print('')

    def apply_sym_operations(self, rotations, translations):

        if self.sites_sym is None and self.cells_sym is None:
            res = apply_symmetric_operations(rotations,
                                             translations,
                                             self.cl_positions,
                                             self.prim.positions)
            self.sites_sym, self.cells_sym = res

        return self.sites_sym, self.cells_sym

    def coordination_number(self,
                            n_cells,
                            rotations=None,
                            translations=None):

        if self.orbit_attr_prim is None:
            self.find_orbit_primitive_cell(rotations=rotations,
                                           translations=translations)

        n_sites_prim = sum(self.prim.n_atoms)
        n_sites = n_sites_prim * n_cells
        coord_numbers = np.zeros(n_sites, dtype=int)
        for k, v in self.orbit_attr_prim.items():
            begin = k * n_cells
            z = len(v.sites)
            coord_numbers[begin:begin+n_cells] = z

        return coord_numbers

    def find_orbit_primitive_cell(self,
                                  rotations=None,
                                  translations=None,
                                  distinguish_element=False):

        if self.orbit_attr_prim is not None:
            return self.orbit_attr_prim

        if self.sites_sym is None and self.cells_sym is None:
            if rotations is None or translations is None:
                raise ValueError('find_orbit_primitive_cell: ' + \
                                 'rotations and translations are required')
            self.apply_sym_operations(rotations, translations)
    
        if distinguish_element == False:
            elements = [0] * 10
        else:
            elements = self.ele_indices
    
        t1 = time.time()
        # time consuming (part1)
        orbit_all = set()
        for sites, cells in zip(self.sites_sym, self.cells_sym):
            for origin in cells.T:
                cells_shift = cells.T - origin
                cand = tuple(sorted(zip(sites,
                                        cells_shift[:,0],
                                        cells_shift[:,1],
                                        cells_shift[:,2],
                                        elements)))
                orbit_all.add(cand)

        t2 = time.time()
        orbit_site = collections.defaultdict(list)
        for cl_cmpnt in orbit_all:
            for attr in cl_cmpnt:
                cell = attr[1:4]
                if any(cell) == False:
                    site = attr[0]
                    orbit_site[site].append(cl_cmpnt)
        t3 = time.time()
    
        self.orbit_attr_prim = dict()
        for site, orbit in orbit_site.items():
            orbit_attr = OrbitAttr()
            for cl_cmpnt in orbit:
                orbit_attr.append_cluster(cl_cmpnt)
            orbit_attr.optimize_type()
            self.orbit_attr_prim[site] = orbit_attr
        t4 = time.time()
       #print(t2-t1,t3-t2,t4-t3)

    def find_orbit_supercell(self, sup,
                             rotations=None,
                             translations=None,
                             orbit=None,
                             distinguish_element=False):

        if sup.plrep is None:
            sup.set_primitive_lattice_representation()

        t1 = time.time()
        self.find_orbit_primitive_cell(rotations=rotations,
                                       translations=translations,
                                       distinguish_element=distinguish_element)
        t2 = time.time()

        orbit_supercell = OrbitAttr()
        if orbit is None:
            for site_obj in sup.plrep:
                if site_obj.site_pl in self.orbit_attr_prim:
                    orbit_attr = self.orbit_attr_prim[site_obj.site_pl]
                    orbit_cells_plrep = orbit_attr.cells + site_obj.cell_plrep
                    #############
                    # time consuming part
                    for sites, cells in zip(orbit_attr.sites,
                                            orbit_cells_plrep):
                        sup_sites = tuple([sup.identify_site_idx(s, c)
                                          for s, c in zip(sites, cells)])
                        orbit_supercell.supercell_sites.append(sup_sites)
                        key = (sites, tuple(cells.ravel()))
                        orbit_supercell.map_supercell_sites[key] = sup_sites
                    #############
                    orbit_supercell.elements.extend(orbit_attr.elements)
        else:
            map_site = orbit.map_supercell_sites
            for site_obj in sup.plrep:
                if site_obj.site_pl in self.orbit_attr_prim:
                    orbit_attr = self.orbit_attr_prim[site_obj.site_pl]
                    orbit_cells_plrep = orbit_attr.cells + site_obj.cell_plrep
                    #############
                    # time consuming part
                    for sites, cells in zip(orbit_attr.sites, 
                                            orbit_cells_plrep):
                        key = (sites, tuple(cells.ravel()))
                        orbit_supercell.supercell_sites.append(map_site[key])
                    #############
                    orbit_supercell.elements.extend(orbit_attr.elements)

        orbit_supercell.eliminate_duplicates()
        orbit_supercell.eliminate_incomplete()

        t3 = time.time()
        #print(t2-t1, t3-t2)
        return orbit_supercell

class ClusterSet:
    
    def __init__(self, clusters, primitive_lattice=None):

        self.clusters = clusters
        if len(self.clusters) > 0 and primitive_lattice is None:
            self.prim = clusters[0].prim
        else:
            self.prim = primitive_lattice

    def get_num_clusters(self):
        return len(self.clusters)

    def print(self):
        for cl in self.clusters:
            cl.print()

    def apply_sym_operations(self, cluster_set=None):

        if cluster_set is None:
            rotations, translations = get_symmetry(self.prim)
            for cl in self.clusters:
                cl.apply_sym_operations(rotations, translations)
        else:
            for cl in self.clusters:
                cl.sites_sym = cluster_set.clusters[cl.idx].sites_sym
                cl.cells_sym = cluster_set.clusters[cl.idx].cells_sym

    # required for efficiently computing cluster orbits in supercell 
    def precompute_orbit_supercell(self, 
                                   cluster_set=None, 
                                   cluster_ids=None,
                                   distinguish_element=True):

        if cluster_ids is not None:
            target_clusters = [self.clusters[i] for i in cluster_ids]
        else:
            target_clusters = self.clusters

        self.apply_sym_operations(cluster_set=cluster_set)
        for cl in target_clusters:
            cl.find_orbit_primitive_cell\
                    (distinguish_element=distinguish_element)

    def compute_orbit_supercell(self, sup, ids=None, distinguish_element=True):

        if ids is not None:
            target_clusters = [self.clusters[i] for i in ids]
        else:
            target_clusters = self.clusters

        size = max([cl.idx for cl in self.clusters]) + 1
        orbit_set = [None] * size

        orbit_all = []
        for cl in target_clusters:
            orbit_pre = orbit_set[cl.idx]
            orbit_obj = cl.find_orbit_supercell\
                                (sup, orbit=orbit_pre,
                                 distinguish_element=distinguish_element)
            orbit_all.append(orbit_obj.get_orbit_supercell())
            if orbit_set[cl.idx] is None:
                orbit_set[cl.idx] = orbit_obj

        return orbit_all


