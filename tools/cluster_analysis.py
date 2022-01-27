#!/usr/bin/env python
import numpy as np
import argparse
import joblib
from joblib import Parallel,delayed
import time

from mlptools.common.structure import Structure

from pyclupan.common.supercell import Supercell
from pyclupan.common.symmetry import get_symmetry
from pyclupan.common.io.yaml import Yaml
from pyclupan.cluster.cluster import Cluster, ClusterSet
from pyclupan.derivative.derivative import DSSet, DSSample

def compute_orbits(ds_samp, 
                   n_cell,
                   s_id, 
                   clusters, 
                   clusters_ele, 
                   distinguish_element=True):

    prim = ds_samp.get_primitive_cell()
    supercell = ds_samp.get_supercell(n_cell, s_id)
    hnf = ds_samp.get_hnf(n_cell, s_id)

    sup = Supercell(st_prim=prim,
                    hnf=hnf,
                    st_supercell=supercell)
    sup.set_primitive_lattice_representation()

    orbit_all = []
    if distinguish_element == False:
        for cl in clusters.clusters:
            orbit = cl.find_orbit_supercell(sup)
            # must be revised
            orbit_all.append(orbit)
    else:
        orbit_set = [None] * len(clusters.clusters)
        for cl in clusters_ele.clusters:
            orbit_pre = orbit_set[cl.idx]
            orbit_obj = cl.find_orbit_supercell(sup,
                                                orbit=orbit_pre,
                                                distinguish_element=True)
            orbit_all.append(orbit_obj.get_orbit_supercell())
            if orbit_set[cl.idx] is None:
                orbit_set[cl.idx] = orbit_obj

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
    # Examples for parsing results of cluster_analysis
    #
    #  cluster_set, ids, counts = joblib.load('cluster_analysis.pkl')
    #  cluster_set, ids, counts = yaml.parse_cluster_analysis_yaml()
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

    print(' parsing clusters.yaml and derivative-all.pkl ...')
    yaml = Yaml()
    clusters, clusters_ele = yaml.parse_clusters_yaml\
                                    (filename=args.clusters_yaml)
    ds_samp = joblib.load(args.derivative_pkl)
    prim = ds_samp.get_primitive_cell()

    print(' building structure list ...')
    target_ids = []
    if args.poscars is not None:
        for string in args.poscars:
            ids_string = string.split('/')[-1].replace('POSCAR-','').split('-')
            target_ids.append(tuple([int(i) for i in ids_string]))
        target_ids = sorted(target_ids)

        labelings, labelings_ids = dict(), dict()
        for n_cell, s_id, l_id in target_ids:
            ids = (n_cell, s_id)
            l = ds_samp.get_labeling(n_cell, s_id, l_id)
            if ids in labelings:
                labelings[ids].append(l)
                labelings_ids[ids].append(l_id)
            else:
                labelings[ids] = [l]
                labelings_ids[ids] = [l_id]

        for ids in labelings_ids.keys():
            labelings[ids] = np.array(labelings[ids])

    else:
        labelings, labelings_ids \
            = ds_samp.get_all_labelings(n_cell_ub=args.n_cell_ub)
        for ids in sorted(labelings_ids.keys()):
            n_cell, s_id = ids
            for l_id in labelings_ids[ids]:
                target_ids.append((n_cell, s_id, l_id))

    #################################################################
    # setting for computing cluster orbits efficiently

    print(' initial setting for computing cluster orbits ...')
    rotations, translations = get_symmetry(prim)
    for cl in clusters.clusters:
        cl.apply_sym_operations(rotations, translations)
    for cl in clusters_ele.clusters:
        cl.sites_sym = clusters.clusters[cl.idx].sites_sym
        cl.cells_sym = clusters.clusters[cl.idx].cells_sym

    distinguish_element = True
    if distinguish_element == False:
        for cl in clusters.clusters:
            cl.find_orbit_primitive_cell()
    else:
        for cl in clusters_ele.clusters:
            cl.find_orbit_primitive_cell(distinguish_element=True)

    #################################################################

    print(' computing cluster orbits ...')
    orbit_all = dict()
    for ids in sorted(labelings_ids.keys()):
        n_cell, s_id = ids
        orbits = compute_orbits(ds_samp, n_cell, s_id, clusters, clusters_ele)
        orbit_all[ids] = orbits

    print(' computing number of clusters in structures (labelings) ...')
    n_total = len(target_ids)
    print('   - total number of structures =', n_total)
    if n_total > 100000:
        n_jobs = 5 
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


# backup
#    n_counts = []
#    for ids in sorted(labelings_ids.keys()):
#        orbits = orbit_all[ids]
#        labelings[ids] = np.array(labelings[ids])
#        n_all = [count_orbit_components(orb, labelings[ids]) for orb in orbits]
#        n_counts.extend(np.array(n_all).T)
#
#
