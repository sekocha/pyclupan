#!/usr/bin/env python
import numpy as np
import glob
import argparse
import joblib
import time

from mlptools.common.structure import Structure
from pyclupan.common.symmetry import get_permutation

from pyclupan.io.yaml import Yaml
from pyclupan.cluster.cluster import Cluster, ClusterSet
from pyclupan.derivative.derivative import DSSet, DSSample
       
def count_orbit_components(orbit, labeling: np.array):
    sites, ele = orbit
    count = np.count_nonzero(np.all(labeling[sites] == ele, axis=1))
    return count

if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('--derivative_pkl',
                    type=str,
                    default='derivative-all.pkl',
                    help='pkl file for DSSamp object')
    ps.add_argument('--clusters_yaml',
                    type=str,
                    default='clusters.yaml',
                    help='location of clusters.yaml')
    ps.add_argument('--poscars',
                    type=str,
                    nargs='*',
                    default='POSCAR',
                    help='POSCARs')
    args = ps.parse_args()

    yaml = Yaml()
    _, clusters_ele = yaml.parse_clusters_yaml(filename=args.clusters_yaml)
#    prim = yaml.get_primitive_cell()

    ds_samp = joblib.load(args.derivative_pkl)

    target_ids = []
    for string in args.poscars:
        ids_string = string.split('/')[-1].replace('POSCAR-','').split('-')
        target_ids.append(tuple([int(i) for i in ids_string]))
    target_ids = sorted(target_ids)

    n_counts = []
    n_cell_prev, s_id_prev = None, None
    for n_cell, s_id, l_id in target_ids:
        labeling = ds_samp.get_labeling(n_cell, s_id, l_id)

        t1 = time.time()
        if n_cell != n_cell_prev or s_id != s_id_prev:
            supercell = ds_samp.get_supercell(n_cell, s_id)
            hnf = ds_samp.get_hnf(n_cell, s_id)
            perm = get_permutation(supercell)

            orbit_array = []
            for cl in clusters_ele.clusters:
                orbit_ele = cl.compute_orbit(supercell, hnf,
                                             permutations=perm, 
                                             distinguish_element=True)
                orbit_array.append(orbit_ele)

        n_all = [count_orbit_components(orbit_ele, labeling)
                    for orbit_ele in orbit_array]
        n_counts.append(n_all)

        n_cell_prev = n_cell
        s_id_prev = s_id

        t2 = time.time()
   #     print(t2-t1)
    n_counts = np.array(n_counts)
    print(n_counts.shape)
    f = open('cluster_analysis.dat', 'w')
    for ids, n in zip(target_ids, n_counts):
        name = '-'.join([str(i) for i in ids])
        print('', name, list(n), file=f)
    f.close()

    test1, test2 = False, False
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

