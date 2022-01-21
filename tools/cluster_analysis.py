#!/usr/bin/env python
import numpy as np
import glob
import argparse
import joblib
import time

from mlptools.common.structure import Structure
from pyclupan.common.symmetry import get_permutation
from pyclupan.common.io.yaml import Yaml
from pyclupan.cluster.cluster import Cluster, ClusterSet
from pyclupan.derivative.derivative import DSSet, DSSample
       
def count_orbit_components(orbit, labeling: np.array):

    sites, ele = orbit
    count = np.count_nonzero(np.all(labeling[sites] == ele, axis=1))
    return count

# must be reconsidered
# How to consider duplicate clusters (How to compute coordination number) ?
def compute_orbits(ds_samp, 
                   n_cell,
                   s_id, 
                   clusters_set, 
                   distinguish_element=False):

    supercell = ds_samp.get_supercell(n_cell, s_id)
    hnf = ds_samp.get_hnf(n_cell, s_id)
    perm = get_permutation(supercell)

    orbit_array = []
    for cl in clusters_set.clusters:
        orbit_ele = cl.compute_orbit(supercell, hnf,
                                     permutations=perm, 
                                     distinguish_element=True)
        orbit_array.append(orbit_ele)

    return orbit_array

if __name__ == '__main__':

    # Examples:
    #
    #  cluster_analysis.py --poscars derivative_poscars/00001/POSCAR-*
    #
    # Examples for parsing results of cluster_analysis
    #
    #  _, ids, counts = joblib.load('cluster_analysis.pkl')
    #  _, ids, counts = yaml.parse_cluster_analysis_yaml()
    #

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

        if n_cell != n_cell_prev or s_id != s_id_prev:
            orbit_array = compute_orbits(ds_samp, 
                                         n_cell, 
                                         s_id, 
                                         clusters_ele, 
                                         distinguish_element=True)

        n_all = [count_orbit_components(orbit_ele, labeling)
                            for orbit_ele in orbit_array]
        n_counts.append(n_all)

        n_cell_prev, s_id_prev = n_cell, s_id

    n_counts = np.array(n_counts)
    print(' (n_structures, n_clusters) =', n_counts.shape)

    yaml.write_cluster_analysis_yaml(clusters_ele, target_ids, n_counts)
    joblib.dump((clusters_ele, target_ids, n_counts), 
                'cluster_analysis.pkl', 
                compress=3)


