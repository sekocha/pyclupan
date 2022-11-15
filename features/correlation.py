#!/usr/bin/env python 
import numpy as np
import sys, os
import argparse
import joblib
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../c++/lib')
import pyclupancpp

def set_spins(element_orbit):

    n_lattice = 0
    binary = True
    spins, cons = dict(), dict()
    eliminate_basis_id = set()

    for ele in element_orbit:
        if len(ele) == 1:
            spin_array = [-1000]
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

    return features_array

def compute_n_ary(features_array, spins, cons_list):

    for i, f in enumerate(features_array):
        labelings = f.labelings
        for ele, s in spins.items():
            condition = f.labelings == ele
            labelings[condition] = s

        site_cls = [sites_cl for sites_cl, _ in f.orbits]
        cons_id_cls = [cons_id_cl for _, cons_id_cl in f.orbits]

        cfobj = pyclupancpp.ComputeCF(labelings, 
                                      site_cls, 
                                      cons_id_cls, 
                                      cons_list)
        correlations = cfobj.get_values()
        f.set_features(correlations)

    return features_array


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
    cons_list = [[] for i in range(max(cons.keys())+1)]
    for k, v in cons.items():
        cons_list[k] = v

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
        features_array = compute_binary(features_array, spins)
        t2 = time.time()

    else: 
        active = []
        for cl in clusters_ele.clusters:
            if len(eliminate_basis_id & set(cl.ele_indices)) == 0:
                active.append(cl)
        clusters_ele = ClusterSet(active)

        print(' computing cluster orbits (in prim. cell) ...')
        clusters.apply_sym_operations()
        clusters_ele.find_orbits_primitive(noncolored_cluster_set=clusters,
                                           distinguish_element=True)
        print(' computing cluster orbits (in supercells) ...')
        for f in features_array:
            orbits = compute_orbits(ds_samp, f.n_cell, f.s_id, clusters_ele)
            f.set_orbits(orbits)

        print(' computing products of cluster functions',
              'in structures (labelings) ...')
        n_total = sum([f.labelings.shape[0] for f in features_array])
        print('   - total number of structures =', n_total)

        t1 = time.time()
        features_array = compute_n_ary(features_array, spins, cons_list)
        t2 = time.time()
                   
    correlation_all = np.vstack([f.features for f in features_array])
    target_ids = [(f.n_cell, f.s_id, l_id) 
                  for f in features_array for l_id in f.labeling_ids]

    print('   - (n_structures, n_clusters) =', correlation_all.shape)
    print('   - elapsed time               =', f'{t2-t1:.2f}', '(s)')
    
    print(' generating output files ...')
    if normal == True:
        clusters_out = clusters
        cons_out = None
    else:
        clusters_out = clusters_ele
        cons_out = cons

    joblib.dump((clusters_out, target_ids, correlation_all), 
                'correlations.pkl', compress=3)
    if args.yaml:
        yaml = Yaml()
        yaml.write_correlations_yaml(clusters_out, 
                                     target_ids, 
                                     correlation_all,
                                     cons=cons_out)
 

# computing correlation functions (slow but correct)

#normal = True
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

#normal = False
#for i, f in enumerate(features_array):
#    labelings = f.labelings
#    for ele, s in spins.items():
#        condition = f.labelings == ele
#        labelings[condition] = s
#    print('id =', i)
#    correlations = []
#    for l in labelings:
#        correlations_l = []
#        for sites_cl, cons_cl in f.orbits:
#            spin_cl = l[sites_cl]
#            #cons_cl = np.array([i1 for i1 in basis_ids])
#            corr = 0.0
#            for c, s in zip(cons_cl, spin_cl):
#                coeffs = [cons[id1] for id1 in c]
#                corr += eval_basis_prod(coeffs, s)
#            corr /= spin_cl.shape[0]
#            correlations_l.append(corr)
#        correlations.append(correlations_l)
# 

