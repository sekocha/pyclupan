#!/usr/bin/env python
import numpy as np
import os, sys
import joblib
import signal
import warnings

from mlptools.common.structure import Structure

from pyclupan.common.io.yaml import Yaml
from pyclupan.common.supercell import Supercell
from pyclupan.common.symmetry import get_permutation

from pyclupan.dd.dd_node import DDNodeHandler
from pyclupan.dd.dd_constructor import DDConstructor

from pyclupan.cluster.cluster import ClusterSet

from pyclupan.derivative.derivative import DSSet
from pyclupan.derivative.preamble import preamble_ds_enum_from_cluster_yaml

signal.signal(signal.SIGINT, signal.SIG_DFL)
warnings.simplefilter('ignore')

clusters_yaml = '../3-clusters/clusters.yaml'
comp = [1, 2/3, None] # Sn3O4(vac2)
n_expand = 3

pre_data = preamble_ds_enum_from_cluster_yaml(clusters_yaml, n_expand)
dd_handler, st_prim, hnf_all, clusters_ele = pre_data

#cluster_ids = [114,136,165]
cluster_ids = [136,114,165,304,288,331,322,329]

ds_set_all = []
n_all = 0
for idx, hnf in enumerate(hnf_all):
    ## pre-computed 
    dd_const = DDConstructor(dd_handler)
    sup = Supercell(st_prim=st_prim, hnf=hnf)
    st_sup = sup.get_supercell()

    print(' computing permutations in supercell ...')
    site_perm, site_perm_lt = get_permutation(st_sup, 
                                              superperiodic=True, 
                                              hnf=hnf)
    print(' computing cluster orbits in supercell ...')
    orbits = clusters_ele.find_orbits_supercell(st_sup, hnf, ids=cluster_ids)
    #######

    print(' constructing zdd ...')
    gs = dd_const.one_of_k()
    gs &= dd_const.composition(comp)
    n_all += gs.len()

    for orb, cl_id in zip(orbits, cluster_ids):
        print('- zdd excluding cluster id:', cl_id)
        orbit_dd = dd_handler.convert_to_orbit_dd(orb)
        gs = dd_const.excluding_clusters(orbit_dd, gs=gs)

    gs = dd_const.nonequivalent_permutations(site_perm, gs=gs)
    gs &= dd_const.no_endmembers(verbose=0)

    res = dd_handler.convert_graphs_to_labelings(gs)
    ds_set = DSSet(labelings_info=res,
                   primitive_cell=st_prim,
                   n_expand=n_expand,
                   elements=dd_handler.elements,
                   comp=comp,
                   hnf=hnf,
                   supercell=st_sup,
                   supercell_id=idx)

    ds_set_all.append(ds_set)

n_match = sum([ds_set.get_size() for ds_set in ds_set_all])
print(' n (st.) (all, constraint) =', n_all, n_match)

prefix = 'derivative-screen-'+str(n_expand)
Yaml().write_derivative_yaml(st_prim, ds_set_all, filename=prefix+'.yaml')
joblib.dump(ds_set_all, prefix+'.pkl', compress=3)


#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../c++/lib')
#import pyclupancpp

#superperiodic == True
#    if superperiodic == False:
#        obj = pyclupancpp.NonequivLBLSuperPeriodic(ds_set.all_labelings, 
#                                                   site_perm_lt)
#        all_labelings = obj.get_labelings()
#        ds_set.replace_labelings(all_labelings)
#
#        print(' number of structures (superperiodic eliminated) =', 
#                all_labelings.shape[0])


