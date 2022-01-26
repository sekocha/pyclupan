#!/usr/bin/env python
import numpy as np
import time

from mlptools.common.structure import Structure

from pyclupan.common.supercell import Supercell
from pyclupan.common.symmetry import get_symmetry
from pyclupan.common.io.yaml import Yaml
from pyclupan.cluster.cluster import Cluster, ClusterSet

if __name__ == '__main__':

    yaml = Yaml()
    clusters, clusters_ele = yaml.parse_clusters_yaml(filename='clusters.yaml')
    st_prim = yaml.get_primitive_cell()

    H = [[1,0,0],
         [0,2,0],
         [0,2,5]]
    n_cells = 10

    sup = Supercell(st_prim=st_prim, hnf=H)
    st_sup = sup.get_supercell()
    sup.set_primitive_lattice_representation()
    
    t1 = time.time()

    rotations, translations = get_symmetry(st_prim)
    for cl in clusters.clusters:
        cl.apply_sym_operations(rotations, translations)
    for cl in clusters_ele.clusters:
        cl.sites_sym = clusters.clusters[cl.idx].sites_sym
        cl.cells_sym = clusters.clusters[cl.idx].cells_sym

    t2 = time.time()

    for cl in clusters.clusters:
        cl.find_orbit_primitive_cell()

    t3 = time.time()

    orbit_set = []
    for cl in clusters.clusters:
        orbit = cl.find_orbit_supercell(sup)
        orbit_set.append(orbit)
        #CN = cl.coordination_number(n_cells)

    t4 = time.time()

    print('-----')
    for cl in clusters_ele.clusters:
        cl.find_orbit_primitive_cell(distinguish_element=True)

    t5 = time.time()

    if len(orbit_set) == 0:
        orbit_set = [None] * len(clusters.clusters)

    for cl in clusters_ele.clusters:
        orbit_pre = orbit_set[cl.idx]
        orbit = cl.find_orbit_supercell(sup,
                                        orbit=orbit_pre,
                                        distinguish_element=True)
        if orbit_set[cl.idx] is None:
            orbit_set[cl.idx] = orbit
        #print(np.array(orbit.supercell_sites).shape)
    t6 = time.time()
    print(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)


