#!/usr/bin/env python
import numpy as np
import sys
import time
import itertools

from scipy.spatial import distance

from mlptools.common.structure import Structure

from pyclupan.common.supercell import Supercell
from pyclupan.common.symmetry import get_symmetry
from pyclupan.common.function import round_frac_array
from pyclupan.common.io.yaml import Yaml
from pyclupan.cluster.cluster import Cluster, ClusterSet

def find_orbit_primitive_cell(cl, rotations, translations):

    prim = cl.prim

    n_body = cl.cl_positions.shape[1]
    orbit_all = set()
    for rot, trans in zip(rotations, translations):
        posrot = np.dot(rot, cl.cl_positions) + np.tile(trans,(n_body,1)).T
        cells = np.floor(posrot).astype(int)
        rposrot = posrot - cells
        sites = np.where(distance.cdist(rposrot.T,prim.positions.T) < tol)[1]

        for origin in cells.T:
            cells_shift = cells.T - origin
            cand = sorted([(s,tuple(c)) for s,c in zip(sites, cells_shift)])
            orbit_all.add(tuple(cand))
    
    orbit_site = dict()
    for cl_cmpnt in orbit_all:
        for site, cell in cl_cmpnt:
            if cell == (0,0,0):
                if site in orbit_site:
                    orbit_site[site].append(cl_cmpnt)
                else:
                    orbit_site[site] = [cl_cmpnt]

#    print(len(orbit), 'nbody=', n_body)
#    for k, v in orbit_site.items():
#        print(k, len(v))
#
    return orbit_site

def find_orbit_supercell(cl, rotations,translations, sup):

    orbit_prim = find_orbit_primitive_cell(cl, rotations, translations)

    orbit = []
    for site_obj in sup.plrep:
        cell_plrep = site_obj.cell_plrep
        if site_obj.site_pl in orbit_prim:
            for cl_cmpnt in orbit_prim[site_obj.site_pl]:
                orbit_sites = []
                for site, cell in cl_cmpnt:
                    cell_plrep_cl = np.array(cell) + cell_plrep
                    idx = sup.identify_site_idx(site, cell_plrep_cl)
                    orbit_sites.append(idx)
                    
                orbit.append(sorted(orbit_sites))

    print(len(orbit))


if __name__ == '__main__':

    yaml = Yaml()
    clusters, clusters_ele = yaml.parse_clusters_yaml(filename='clusters.yaml')
    st_prim = yaml.get_primitive_cell()
    rotations, translations = get_symmetry(st_prim)

    H = [[1,0,0],
         [0,2,0],
         [0,2,5]]

    sup = Supercell(st_prim=st_prim, hnf=H)
    st_sup = sup.get_supercell()
    sup.set_primitive_lattice_representation()

#    plrep, map_plrep = sup.set_primitive_lattice_representation()

    #supercell_site_attr = sup.set_supercell_site_attr()
    #supercell_site_attr = sup.get_supercell_site_attr()
    Hinv = np.linalg.inv(H)
    

    t1 = time.time()
    tol = 1e-13
    for cl in clusters.clusters:
        find_orbit_supercell(cl, 
                             rotations, 
                             translations, 
                             sup)
                             #plrep,
                             #map_plrep,
                             #H,
                             #Hinv)
    t2 = time.time()
    print(t2-t1)


#####    test1, test2 = True, False
#####    if test1:
#####        site_indices = [2,2]
#####        cell_indices = [[0,0,0],
#####                        [1,0,0],
#####                        [0,1,0]]
#####
#####
#####        n_body = len(site_indices)
#####        cl = Cluster(0, n_body, 
#####                     site_indices, 
#####                     cell_indices, 
#####                     primitive_lattice=st_prim)
#####        orbit = cl.compute_orbit_with_weight(st_sup, 
#####                                             supercell_mat=H, 
#####                                             permutations=perm_sup)
######

#
#
#    test1, test2 = False, False
#    if test1:
#        site_indices = [2,2]
#        cell_indices = [[0,0,0],
#                        [1,0,0],
#                        [0,1,0]]
#
#        n_body = len(site_indices)
#        cl = Cluster(0, n_body, site_indices, cell_indices, 
#                     primitive_lattice=prim)
#        orbit = cl.compute_orbit(sup, H, permutations=perm_sup)
#        cl.print()
#        print(' cluster orbit in supercell')
#        print(orbit)
#
#        cl.set_element_indices([0,1])
#        orbit_ele = cl.compute_orbit(sup, H, 
#                                     permutations=perm_sup, 
#                                     distinguish_element=True)
#        print(' cluster orbit with element configurations in supercell')
#        print(orbit_ele)
#
#        n_count = count_orbit_components(orbit_ele, labeling)
#        print(' orbit components in labeling =', n_count)
#
#    if test2:
#        n_count_all = []
#        t1 = time.time()
#        for cl in clusters_ele.clusters:
#            orbit_ele = cl.compute_orbit(sup, H, 
#                                         permutations=perm_sup, 
#                                         distinguish_element=True)
#            n_count = count_orbit_components(orbit_ele, labeling)
#            n_count_all.append(n_count)
#            print(' idx:', cl.idx, ', elements =', cl.ele_indices)
#            print(' orbit components in labeling =', n_count)
#        print(n_count_all)
#        t2 = time.time()
#        print(t2-t1)
#
