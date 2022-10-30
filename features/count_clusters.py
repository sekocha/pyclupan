#!/usr/bin/env python 
import numpy as np
import argparse
import joblib
from joblib import Parallel,delayed
import time

from mlptools.common.structure import Structure

from pyclupan.common.supercell import Supercell
from pyclupan.common.io.yaml import Yaml
from pyclupan.cluster.cluster import Cluster
from pyclupan.cluster.cluster import ClusterSet
from pyclupan.derivative.derivative import DSSet
from pyclupan.derivative.derivative import DSSample

from pyclupan.features.features_common import parse_clusters_yaml
from pyclupan.features.features_common import parse_derivatives
from pyclupan.features.features_common import sample_from_ds
from pyclupan.features.features_common import compute_orbits
from pyclupan.features.features_common import Features

def count_orbit_components(orbit, labelings: np.array):
    sites, ele = orbit
    if len(sites) > 0:
        count = np.count_nonzero\
            (np.all(labelings[:,sites] == ele, axis=2), axis=1)
        return count
    return np.zeros(labelings.shape[0], dtype=int)

def function1(orbits, lbls):
    n_all = [count_orbit_components(orb, lbls) for orb in orbits]
    return n_all
 
if __name__ == '__main__':

    # Examples:
    #
    #  cluster_analysis.py --poscars derivative_poscars/00001/POSCAR-*
    #  cluster_analysis.py --n_cell_ub 6
    #

    ps = argparse.ArgumentParser()
    ps.add_argument('--derivative',
                    type=str,
                    nargs='*',
                    default=['derivative-all.pkl'],
                    help='Location of DSSet pkl files')
    ps.add_argument('--clusters_yaml',
                    type=str,
                    default='clusters.yaml',
                    help='Location of clusters.yaml')
    ps.add_argument('--poscars',
                    type=str,
                    nargs='*',
                    default=None,
                    help='POSCARs')
    ps.add_argument('--n_cell_ub',
                    type=int,
                    default=None,
                    help='Maximum number of cells')
    ps.add_argument('--yaml',
                    action='store_true',
                    help='generating yaml file')
    args = ps.parse_args()

    print(' parsing clusters.yaml ...')
    clusters, clusters_ele = parse_clusters_yaml(args.clusters_yaml)

    print(' parsing derivative-all.pkl ...')
    ds_samp = parse_derivatives(args.derivative)
    prim = ds_samp.get_primitive_cell()

    print(' building structure list ...')
    features_array = sample_from_ds(ds_samp, 
                                    poscars=args.poscars, 
                                    n_cell_ub=args.n_cell_ub)

    #################################################################
    # non-colored cluster orbits are pre-computed

    print(' initial setting for computing non-colored cluster orbits ...')
    clusters.apply_sym_operations()
    clusters_ele.find_orbits_primitive(noncolored_cluster_set=clusters,
                                       distinguish_element=True)

    #################################################################

    print(' computing cluster orbits ...')
    for f in features_array:
        orbits = compute_orbits(ds_samp, f.n_cell, f.s_id, clusters_ele)
        f.set_orbits(orbits)

    print(' computing number of clusters in structures (labelings) ...')
    n_total = sum([f.labelings.shape[0] for f in features_array])
    print('   - total number of structures =', n_total)

    if n_total > 100000:
        n_jobs = 8
    elif n_total > 20000:
        n_jobs = 3 
    else:
        n_jobs = 1

    t1 = time.time()
    n_all = Parallel(n_jobs=n_jobs)(delayed(function1)
                        (f.orbits, f.labelings) for f in features_array)
    for f, n in zip(features_array, n_all):
        f.set_features(np.array(n).T)
    t2 = time.time()


    n_counts = np.vstack([f.features for f in features_array])
    target_ids = [(f.n_cell, f.s_id, l_id) 
                  for f in features_array for l_id in f.labeling_ids]
    orbit_sizes = [((f.n_cell, f.s_id), f.orbit_sizes) for f in features_array]
    orbit_sizes = dict(orbit_sizes)

    print('  - elapsed time (counting) =', f'{t2-t1:.2f}', '(s)')
    print('  - (n_structures, n_clusters) =', n_counts.shape)
    
    # output
    print(' generating output files ...')
    joblib.dump((clusters_ele, orbit_sizes, target_ids, n_counts), 
                'count_clusters.pkl', compress=3)
    if args.yaml:
        yaml = Yaml()
        yaml.write_count_clusters_yaml(clusters_ele, 
                                       orbit_sizes, 
                                       target_ids, 
                                       n_counts)
#
#
