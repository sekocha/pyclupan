#!/usr/bin/env python 
import numpy as np
import argparse
import joblib
from joblib import Parallel,delayed
import time

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

#def count_orbit_components(orbit, labelings: np.array):
#    sites, ele = orbit
#    if len(sites) > 0:
#        count = np.count_nonzero\
#            (np.all(labelings[:,sites] == ele, axis=2), axis=1)
#        return count
#    return np.zeros(labelings.shape[0], dtype=int)
#
#def function1(orbits, lbls):
#    n_all = [count_orbit_components(orb, lbls) for orb in orbits]
#    return n_all

def set_spins(element_orbit):

    spins = dict()
    n_lattice = 0
    binary = True
    for ele in element_orbit:
        if len(ele) == 1:
            spins[ele[0]] = -1000
        elif len(ele) == 2:
            spins[ele[0]] = 1
            spins[ele[1]] = -1
            n_lattice += 1
        elif len(ele) == 3:
            spins[ele[0]] = 1
            spins[ele[1]] = 0
            spins[ele[2]] = -1
            n_lattice += 1
            binary = False
        elif len(ele) == 4:
            n_lattice += 1
            binary = False
            pass
        elif len(ele) == 5:
            n_lattice += 1
            binary = False
            pass

    if n_lattice == 1 and binary == True:
        normal = True
    else:
        normal = False

    return spins, normal


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
    clusters, _ = parse_clusters_yaml(args.clusters_yaml)

    print(' parsing derivative-all.pkl ...')
    ds_samp = parse_derivatives(args.derivative)
    prim = ds_samp.get_primitive_cell()

    print(' building structure list ...')
    features_array = sample_from_ds(ds_samp, 
                                    poscars=args.poscars, 
                                    n_cell_ub=args.n_cell_ub)
    # common part (end)

    # setting spins and cluster functions
    spins, normal = set_spins(ds_samp.element_orbit)

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

    if normal == True:
        t1 = time.time()
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
        t2 = time.time()
                    
#    t1 = time.time()
#    n_all = Parallel(n_jobs=n_jobs)(delayed(function1)
#                        (f.orbits, f.labelings) for f in features_array)
#    t2 = time.time()
#
#
    correlation_all = np.vstack([f.features for f in features_array])
    target_ids = [(f.n_cell, f.s_id, l_id) 
                  for f in features_array for l_id in f.labeling_ids]

    print('  - elapsed time (counting) =', f'{t2-t1:.2f}', '(s)')
    print('  - (n_structures, n_clusters) =', correlation_all.shape)
    
    # output
    print(' generating output files ...')
    joblib.dump((clusters, target_ids, correlation_all), 
                'correlations.pkl', compress=3)

    if args.yaml:
        yaml = Yaml()
        yaml.write_correlations_yaml(clusters, target_ids, correlation_all)


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
#
