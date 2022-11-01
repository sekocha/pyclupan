#!/usr/bin/env python
import numpy as np
#import argparse
#import os, sys
#import time
#import joblib
#import itertools

import signal
import warnings

from mlptools.common.structure import Structure

from pyclupan.common.supercell import Supercell
from pyclupan.common.symmetry import get_permutation

from pyclupan.dd.dd_node import DDNodeHandler
from pyclupan.dd.dd_constructor import DDConstructor
from pyclupan.cluster.cluster import ClusterSet

from pyclupan.derivative.preamble import preamble_ds_enum_from_cluster_yaml

#from fractions import Fraction
##from pyclupan.common.io.yaml import Yaml
##from pyclupan.derivative.derivative import DSSet
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../c++/lib')
#import pyclupancpp

signal.signal(signal.SIGINT, signal.SIG_DFL)
warnings.simplefilter('ignore')

clusters_yaml = '../3-clusters/clusters.yaml'
comp = [1, 2/3, None] # Sn3O4(vac2)
n_expand = 3

pre_data = preamble_ds_enum_from_cluster_yaml(clusters_yaml, n_expand)
dd_handler, st_prim, hnf_all, clusters_ele = pre_data

#cluster_ids = [114,136,165]
cluster_ids = [114]
#tholds = [0.15]

n_all, n_match = 0, 0
for hnf in hnf_all:
    dd_const = DDConstructor(dd_handler)
    sup = Supercell(st_prim=st_prim, hnf=hnf)
    st_sup = sup.get_supercell()
    
    gs = dd_const.one_of_k()
    gs &= dd_const.composition(comp)
    n_all += gs.len()
   
    print(' computing cluster orbits in supercell ...')
    orbits = clusters_ele.find_orbits_supercell(st_sup, hnf, ids=cluster_ids)
    
    for orb, cl_id in zip(orbits, cluster_ids):
        print('- zdd excluding cluster id:', cl_id)
        sites, ele = orb
        orbit_node_rep = []
        for s1, e1 in zip(sites, ele):
            nodes = [dd_handler.compose_node(s2,e2) for s2,e2 in zip(s1,e1)]
            orbit_node_rep.append(tuple(sorted(nodes)))
    
        gs = dd_const.excluding_clusters(orbit_node_rep, gs=gs)

    site_perm, site_perm_lt = get_permutation(st_sup, 
                                              superperiodic=True, 
                                              hnf=hnf)
    gs = dd_const.nonequivalent_permutations(site_perm, gs=gs)
    gs &= dd_const.no_endmembers(verbose=0)
    n_match += gs.len()

print(' n (st.) (all, constraint) =', n_all, n_match)


#ds_set_all = []
#for idx, H in enumerate(hnf_all):
#    res = dd_handler.convert_graphs_to_labelings(gs)
#    active_labelings, inactive_labeling, active_sites, inactive_sites = res
#
#    ds_set = DSSet(active_labelings=active_labelings,
#                   inactive_labeling=inactive_labeling,
#                   active_sites=active_sites,
#                   inactive_sites=inactive_sites,
#                   primitive_cell=st_prim,
#                   n_expand=args.n_expand,
#                   elements=dd_handler.elements,
#                   comp=comp,
#                   comp_lb=comp_lb,
#                   comp_ub=comp_ub,
#                   hnf=H,
#                   supercell=st_sup,
#                   supercell_id=idx)
#
#    if args.superperiodic == False:
#        obj = pyclupancpp.NonequivLBLSuperPeriodic(ds_set.all_labelings, 
#                                                   site_perm_lt)
#        all_labelings = obj.get_labelings()
#        ds_set.replace_labelings(all_labelings)
#
#        print(' number of structures (superperiodic eliminated) =', 
#                all_labelings.shape[0])
#
#    ds_set_all.append(ds_set)
#
#prefix = 'derivative-'+str(args.n_expand)
#yaml = Yaml()
#yaml.write_derivative_yaml(st_prim, ds_set_all, filename=prefix+'.yaml')
#
#if args.nodump == False:
#    joblib.dump(ds_set_all, prefix+'.pkl', compress=3)
#
#
