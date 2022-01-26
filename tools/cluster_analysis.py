#!/usr/bin/env python
import numpy as np
import glob
import argparse
import joblib
import time

from mlptools.common.structure import Structure

from pyclupan.common.supercell import Supercell
from pyclupan.common.symmetry import get_symmetry
from pyclupan.common.io.yaml import Yaml
from pyclupan.cluster.cluster import Cluster, ClusterSet
from pyclupan.derivative.derivative import DSSet, DSSample
       
def count_orbit_components(orbit, labeling: np.array):

    sites, ele = orbit
    count = np.count_nonzero(np.all(labeling[sites] == ele, axis=1))
    return count

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
    args = ps.parse_args()

    print(' parsing clusters.yaml and derivative-all.pkl ...')
    yaml = Yaml()
    clusters, clusters_ele = yaml.parse_clusters_yaml\
                                    (filename=args.clusters_yaml)
    ds_samp = joblib.load(args.derivative_pkl)
    prim = ds_samp.get_primitive_cell()

    print(' building structure list ...')
    if args.poscars is not None:
        target_ids = []
        for string in args.poscars:
            ids_string = string.split('/')[-1].replace('POSCAR-','').split('-')
            target_ids.append(tuple([int(i) for i in ids_string]))
        target_ids = sorted(target_ids)

        labelings = [ds_samp.get_labeling(n_cell, s_id, l_id)
                        for n_cell, s_id, l_id in target_ids]
    else:
        labelings, target_ids \
            = ds_samp.get_all_labelings(n_cell_ub=args.n_cell_ub)

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

    print(' computing number of clusters in structures (labelings) ...')
    n_counts = []
    n_cell_prev, s_id_prev = None, None
    for labeling, (n_cell, s_id, l_id) in zip(labelings, target_ids):

        if n_cell != n_cell_prev or s_id != s_id_prev:
            orbit_all = compute_orbits(ds_samp, 
                                       n_cell, 
                                       s_id, 
                                       clusters,
                                       clusters_ele)

        n_all = [count_orbit_components(orbit, labeling)
                                  for orbit in orbit_all]
        n_counts.append(n_all)
        n_cell_prev, s_id_prev = n_cell, s_id

    n_counts = np.array(n_counts)
    print(' (n_structures, n_clusters) =', n_counts.shape)

    print(' generating output files ...')
    yaml.write_cluster_analysis_yaml(clusters_ele, target_ids, n_counts)
    joblib.dump((clusters_ele, target_ids, n_counts), 
                'cluster_analysis.pkl', 
                compress=3)

