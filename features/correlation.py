#!/usr/bin/env python 
import numpy as np
import argparse
import joblib
from joblib import Parallel,delayed
import time
import math

from mlptools.common.structure import Structure

from pyclupan.common.io.yaml import Yaml
from pyclupan.common.io.io_alias import parse_clusters_yaml
from pyclupan.common.io.io_alias import parse_derivatives
from pyclupan.common.supercell import Supercell

from pyclupan.cluster.cluster import Cluster
from pyclupan.cluster.cluster import ClusterSet
from pyclupan.derivative.derivative import DSSet
from pyclupan.derivative.derivative import DSSample

from pyclupan.features.features_common import sample_from_ds
from pyclupan.features.features_common import compute_orbits
from pyclupan.features.features_common import Features
from pyclupan.features.spin_polynomial import gram_schmidt

#def function1(orbits, lbls):
#    n_all = [count_orbit_components(orb, lbls) for orb in orbits]
#    return n_all

def set_spins(element_orbit):

    n_lattice = 0
    binary = True
    spins, cons = dict(), dict()
    eliminate_basis_id = set()
    for ele in element_orbit:
        if len(ele) == 1:
            spin_array = [-1000]
            #cons[ele[0]] = [0.0]
        elif len(ele) == 2:
            spin_array = [1,-1]
        elif len(ele) == 3:
            spin_array = [1,0,-1]
        elif len(ele) == 4:
            spin_array = [2,1,0,-1]
        elif len(ele) == 5:
            spin_array = [2,1,0,-1,2]

        for i, s in enumerate(spin_array):
            spins[ele[i]] = s

        if len(ele) > 1:
            n_lattice += 1
            for i, basis in enumerate(gram_schmidt(spin_array)):
                basis_id = ele[i]
                if np.allclose(basis[:-1], np.zeros(basis.shape[0]-1)) \
                    and math.isclose(basis[-1], 1.0):
                    eliminate_basis_id.add(basis_id)
                else:
                    cons[ele[i]] = basis

        if len(ele) > 2:
            binary = False

    if n_lattice == 1 and binary == True:
        normal = True
    else:
        normal = False

    return spins, normal, cons, eliminate_basis_id

def compute_binary(features_array, spins):

    for f in features_array:
        labelings = f.labelings
        for ele, s in spins.items():
            condition = f.labelings == ele
            labelings[condition] = s

        correlation = []
        for sites in f.orbits:
           spin_cl = labelings[:,sites]
           ave = np.average(np.prod(spin_cl, axis=2), axis=1)
           correlation.append(ave)
        correlation = np.array(correlation).T
        f.set_features(correlation)

    return features_array, correlation
 
if __name__ == '__main__':

    # Examples:
    #
    #  correlation.py --poscars derivative_poscars/00001/POSCAR-*
    #  correlation.py --n_cell_ub 6
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

    # common part in computing other features
    print(' parsing clusters.yaml ...')
    clusters, clusters_ele = parse_clusters_yaml(args.clusters_yaml)

    print(' parsing derivative-all.pkl ...')
    ds_samp = parse_derivatives(args.derivative)
    prim = ds_samp.get_primitive_cell()

    print(' building structure list ...')
    features_array = sample_from_ds(ds_samp, 
                                    poscars=args.poscars, 
                                    n_cell_ub=args.n_cell_ub)
    # common part (end)

    # setting spins and cluster functions
    spins, normal, cons, eliminate_basis_id = set_spins(ds_samp.element_orbit)
    print(cons)
    print(eliminate_basis_id)

    # temporarily
    normal = False

    if normal == True:
        print(' computing cluster orbits (in prim. cell) ...')
        clusters.find_orbits_primitive(distinguish_element=False)
        print(' computing cluster orbits (in supercells) ...')
        for f in features_array:
            orbits = compute_orbits(ds_samp, f.n_cell, f.s_id, clusters)
            orbits = np.array([sites for sites, _ in orbits])
            f.set_orbits(orbits)

        print(' computing correlation functions in structures (labelings) ...')
        n_total = sum([f.labelings.shape[0] for f in features_array])
        print('   - total number of structures =', n_total)

        t1 = time.time()
        features_array, correlation = compute_binary(features_array, spins)
        t2 = time.time()

    else: 
        print(clusters_ele.get_num_clusters())
        active = []
        for cl in clusters_ele.clusters:
            if not 1 in cl.ele_indices:
                active.append(cl)
        clusters_ele = ClusterSet(active)
        print(clusters_ele.get_num_clusters())

        print(' computing cluster orbits (in prim. cell) ...')
        clusters.apply_sym_operations()
        clusters_ele.find_orbits_primitive(noncolored_cluster_set=clusters,
                                           distinguish_element=True)
        print(' computing cluster orbits (in supercells) ...')
        for f in features_array:
            orbits = compute_orbits(ds_samp, f.n_cell, f.s_id, clusters_ele)
            f.set_orbits(orbits)

        print(' computing correlation functions in structures (labelings) ...')
        n_total = sum([f.labelings.shape[0] for f in features_array])
        print('   - total number of structures =', n_total)

        t1 = time.time()
        #features_array, correlation = compute_n_ary(features_array, 
        #                                             spins, 
        #                                             cons)
        t2 = time.time()

                   
    correlation_all = np.vstack([f.features for f in features_array])
    target_ids = [(f.n_cell, f.s_id, l_id) 
                  for f in features_array for l_id in f.labeling_ids]

    print('   - (n_structures, n_clusters) =', correlation_all.shape)
    print('   - elapsed time               =', f'{t2-t1:.2f}', '(s)')
    
    # output
    print(' generating output files ...')
    if normal == True:
        joblib.dump((clusters, target_ids, correlation_all), 
                    'correlations.pkl', compress=3)
        if args.yaml:
            yaml = Yaml()
            yaml.write_correlations_yaml(clusters, target_ids, correlation_all)
    else:
        joblib.dump((clusters_ele, target_ids, correlation_all), 
                    'correlations.pkl', compress=3)
        if args.yaml:
            yaml = Yaml()
            #yaml.write_cluster_functions_yaml(clusters_ele, 
            #                                   target_ids, 
            #                                   correlation_all)


# computing correlation functions (slow but correct)

        #t1 = time.time()
        #for i, f in enumerate(features_array):
        #    labelings = f.labelings
        #    for ele, s in spins.items():
        #        condition = f.labelings == ele
        #        labelings[condition] = s

        #    print('id =', i)
        #    for l in labelings:
        #        correlations = []
        #        for sites in f.orbits:
        #            spin_cl = l[sites]
        #            corr = np.average(np.prod(spin_cl, axis=1))
        #            correlations.append(corr)
        #        print(correlations)
        #t2 = time.time()
        #print(t2-t1)

#    if n_total > 100000:
#        n_jobs = 8
#    elif n_total > 20000:
#        n_jobs = 3 
#    else:
#        n_jobs = 1
##    n_all = Parallel(n_jobs=n_jobs)(delayed(function1)
#                        (f.orbits, f.labelings) for f in features_array)


