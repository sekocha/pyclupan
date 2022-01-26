#!/usr/bin/env python
import numpy as np
import collections
import time

from mlptools.common.structure import Structure
from pyclupan.common.symmetry import apply_symmetric_operations

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

    def get_orbit_supercell(self):
        return np.array(self.supercell_sites), np.array(self.elements)

    def count(self):
        orbit = [(s,e) for s, e in zip(self.supercell_sites, self.elements)]
        return collections.Counter(orbit)

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

    def print(self):
        for cl in self.clusters:
            cl.print()

###########################################################################
# obsolete functions
###    def compute_orbit(self, 
###                      supercell_st: Structure,
###                      supercell_mat: np.array=None,
###                      permutations=None,
###                      distinguish_element=False):
###
###        if permutations is None:
###            perm = get_permutation(supercell_st)
###        else:
###            perm = permutations
###
###        sites = self.identify_cluster(supercell_st, supercell_mat)
###        sites_perm = perm[:,np.array(sites)]
###
###        if distinguish_element == False:
###            orbit = set([tuple(sorted(s1)) for s1 in sites_perm])
###            return sorted(orbit)
###        else:
###            orbit = set()
###            for s1 in sites_perm:
###                cmpnt = [tuple([s,e]) for s,e in zip(s1,self.ele_indices)]
###                orbit.add(tuple(sorted(cmpnt)))
###
###            s_all, e_all = [], []
###            for cmpnt in sorted(orbit):
###                s_array, e_array = [], []
###                for s, e in cmpnt:
###                    s_array.append(s)
###                    e_array.append(e)
###                s_all.append(s_array)
###                e_all.append(e_array)
###
###            return (np.array(s_all), np.array(e_all))
###
###    def identify_cluster(self, 
###                         supercell_st: Structure,
###                         supercell_mat: np.array=None):
###
###        if self.cl_positions is None:
###            self.set_positions()
###
###        if supercell_mat is None:
###            sup_axis_inv = np.linalg.inv(supercell_st.axis)
###            sup_mat_inv = np.dot(sup_axis_inv, self.prim.axis)
###        else:
###            sup_mat_inv = np.linalg.inv(supercell_mat)
###
###        cl_positions_sup = np.dot(sup_mat_inv, self.cl_positions)
###        cl_positions_sup = round_frac_array(cl_positions_sup)
###
###        ########## time consuming part #############
###        site_indices = []
###        for pos1 in cl_positions_sup.T:
###            for idx, pos2 in enumerate(supercell_st.positions.T):
###                if np.linalg.norm(pos1 - pos2) < 1e-10:
###                    site_indices.append(idx)
###                    break
###        ############################################
###
###        return site_indices
###
###
