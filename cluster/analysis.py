#!/usr/bin/env python
import numpy as np
import argparse
import time

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure
from pyclupan.common.supercell import supercell
from pyclupan.common.symmetry import get_permutation

from pyclupan.io.yaml import Yaml
from pyclupan.cluster.cluster import Cluster, ClusterSet
       
# must be faster
def count_orbit_components(orbit, labeling:np.array):
    sites, ele = orbit
    count = np.count_nonzero(np.all(labeling[sites] == ele, axis=1))
    return count

if __name__ == '__main__':

    yaml = Yaml()
    clusters, clusters_ele = yaml.parse_clusters_yaml()
    prim = yaml.get_primitive_cell()

    H = [[3,0,0],
         [1,2,0],
         [1,0,1]]
    axis_s, positions_s, n_atoms_s = supercell(H, 
                                               prim.axis, 
                                               prim.positions, 
                                               prim.n_atoms)
    sup = Structure(axis_s, positions_s, n_atoms_s)
    perm_sup = get_permutation(sup)

#    # test FCC
#    site_indices = [0,0,0]
#    cell_indices = [[0,0,0],
#                    [1,0,0],
#                    [2,0,0]]

    # test perovkite
    labeling = [2] * 6 + [3] * 6 + [0,1,0] * 6
    labeling = np.array(labeling)

    test1, test2 = False, True
    if test1:
        site_indices = [2,2]
        cell_indices = [[0,0,0],
                        [1,0,0],
                        [0,1,0]]

        n_body = len(site_indices)
        cl = Cluster(0, n_body, site_indices, cell_indices, 
                     primitive_lattice=prim)
        orbit = cl.compute_orbit(sup, H, permutations=perm_sup)
        cl.print()
        print(' cluster orbit in supercell')
        print(orbit)

        cl.set_element_indices([0,1])
        orbit_ele = cl.compute_orbit(sup, H, 
                                     permutations=perm_sup, 
                                     distinguish_element=True)
        print(' cluster orbit with element configurations in supercell')
        print(orbit_ele)

        n_count = count_orbit_components(orbit_ele, labeling)
        print(' orbit components in labeling =', n_count)

    if test2:
        n_count_all = []
        t1 = time.time()
        for cl in clusters_ele.clusters:
            orbit_ele = cl.compute_orbit(sup, H, 
                                         permutations=perm_sup, 
                                         distinguish_element=True)
            n_count = count_orbit_components(orbit_ele, labeling)
            n_count_all.append(n_count)
            print(' idx:', cl.idx, ', elements =', cl.ele_indices)
            print(' orbit components in labeling =', n_count)
        print(n_count_all)
        t2 = time.time()
        print(t2-t1)

