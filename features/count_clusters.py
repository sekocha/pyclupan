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

def compute_orbits(ds_samp, 
                   n_cell,
                   s_id, 
                   target_cluster_set, 
                   distinguish_element=True):

    prim = ds_samp.get_primitive_cell()
    supercell = ds_samp.get_supercell(n_cell, s_id)
    hnf = ds_samp.get_hnf(n_cell, s_id)

    sup = Supercell(st_prim=prim,
                    hnf=hnf,
                    st_supercell=supercell)
    sup.set_primitive_lattice_representation()

    orbit_all = target_cluster_set.compute_orbit_supercell\
                                (sup, distinguish_element=True)

    return orbit_all

def count_orbit_components(orbit, labelings: np.array):
    sites, ele = orbit
    count = np.count_nonzero(np.all(labelings[:,sites] == ele, axis=2), axis=1)
    return count

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
    print(args.n_cell_ub)
    print(args.poscars)
    labelings, target_ids = sample_from_ds(ds_samp, 
                                           poscars=args.poscars, 
                                           n_cell_ub=args.n_cell_ub)
    print(labelings)

#    if args.poscars is not None:
#        for string in args.poscars:
#            ids_string = string.split('/')[-1].replace('POSCAR-','').split('-')
#            target_ids.append(tuple([int(i) for i in ids_string]))
#        target_ids = sorted(target_ids)
#
#        labelings, labelings_ids = dict(), dict()
#        for n_cell, s_id, l_id in target_ids:
#            ids = (n_cell, s_id)
#            l = ds_samp.get_labeling(n_cell, s_id, l_id)
#            if ids in labelings:
#                labelings[ids].append(l)
#                labelings_ids[ids].append(l_id)
#            else:
#                labelings[ids] = [l]
#                labelings_ids[ids] = [l_id]
#
#        for ids in labelings_ids.keys():
#            labelings[ids] = np.array(labelings[ids])


    #################################################################
    # setting for computing cluster orbits efficiently

    print(' initial setting for computing cluster orbits ...')
    distinguish_element = True
    clusters.apply_sym_operations()
    clusters_ele.precompute_orbit_supercell\
                                (cluster_set=clusters,
                                 distinguish_element=distinguish_element)

    #################################################################

    print(' computing cluster orbits ...')
    orbit_all = dict()
    for ids in sorted(labelings_ids.keys()):
        n_cell, s_id = ids
        orbits = compute_orbits(ds_samp, n_cell, s_id, clusters_ele)
        orbit_all[ids] = orbits

    print(' computing number of clusters in structures (labelings) ...')
    n_total = len(target_ids)
    print('   - total number of structures =', n_total)
    if n_total > 100000:
        n_jobs = 8
    elif n_total > 20000:
        n_jobs = 3 
    else:
        n_jobs = 1

    t1 = time.time()
    n_all = Parallel(n_jobs=n_jobs)(delayed(function1)
                                (orbit_all[ids], labelings[ids])
                                 for ids in sorted(labelings_ids.keys()))
    n_counts = np.hstack(n_all).T
    t2 = time.time()
    print('  - elapsed time (counting) =', f'{t2-t1:.2f}', '(s)')
    print('  - (n_structures, n_clusters) =', n_counts.shape)

    print(' generating output files ...')
    joblib.dump((clusters_ele, target_ids, n_counts), 
                'cluster_analysis.pkl', 
                compress=3)
    if args.yaml:
       yaml.write_cluster_analysis_yaml(clusters_ele, target_ids, n_counts)


