#!/usr/bin/env python
import numpy as np
import argparse
import os, sys
import time
import joblib

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure

from pyclupan.common.supercell import Supercell
from pyclupan.common.symmetry import get_permutation
from pyclupan.common.normal_form import get_nonequivalent_hnf
from pyclupan.common.io.yaml import Yaml

from pyclupan.dd.dd_node import DDNodeHandler
from pyclupan.dd.dd_constructor import DDConstructor
from pyclupan.derivative.derivative import DSSet
from pyclupan.cluster.cluster import Cluster, ClusterSet

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../c++/lib')
import pyclupancpp

def convert_orbit_to_node_pairs(orbit, dd_handler):
    for sites_all, elements_all in orbit:
        for sites, elements in zip(sites_all, elements_all):
            nodes = [dd_handler.compose_node(s_idx, e_idx) 
                        for s_idx, e_idx in zip(sites, elements)]
            print(sites, elements, nodes)

            
    

if __name__ == '__main__':

    occupation = [[0],[1],[2],[2]]
    n_elements = len(occupation)

    comp = [None, None, 2/3, 1/3]

    yaml = Yaml()
    clusters, clusters_ele = yaml.parse_clusters_yaml()
    st_prim = yaml.get_primitive_cell()

    cluster_ids = [3,5]

    clusters.apply_sym_operations()
    clusters_ele.precompute_orbit_supercell(cluster_set=clusters, 
                                            cluster_ids=cluster_ids, 
                                            distinguish_element=True)

    n_cell = 4
    hnf_all = get_nonequivalent_hnf(4, st_prim)

    n_sites = list(np.array(st_prim.n_atoms) * n_cell)
    dd_handler = DDNodeHandler(n_sites=n_sites,
                               occupation=occupation,
                               one_of_k_rep=False)

    ds_set_all = []
    for idx, H in enumerate(hnf_all):
        sup = Supercell(st_prim=st_prim, hnf=H)

        orbit = clusters_ele.compute_orbit_supercell(sup, ids=cluster_ids)
        convert_orbit_to_node_pairs(orbit, dd_handler)

        st_sup = sup.get_supercell()
        site_perm, site_perm_lt = get_permutation(st_sup, 
                                                  superperiodic=True, 
                                                  hnf=H)

        dd_const = DDConstructor(dd_handler)
        gs = dd_const.enumerate_nonequiv_configs(site_permutations=site_perm,
                                                 comp=comp)



        t1 = time.time()
        res = dd_handler.convert_graphs_to_labelings(gs)
        active_labelings, inactive_labeling, active_sites, inactive_sites = res
        t2 = time.time()
        print(' elapsed time (labeling)    =', t2-t1)

        ds_set = DSSet(active_labelings=active_labelings,
                       inactive_labeling=inactive_labeling,
                       active_sites=active_sites,
                       inactive_sites=inactive_sites,
                       primitive_cell=st_prim,
                       n_expand=args.n_expand,
                       elements=dd_handler.elements,
                       comp=comp,
                       comp_lb=comp_lb,
                       comp_ub=comp_ub,
                       hnf=H,
                       supercell=st_sup,
                       supercell_id=idx)

        # eliminate superlattces
#        obj = pyclupancpp.NonequivLBLSuperPeriodic(ds_set.all_labelings, 
#                                                   site_perm_lt)
#        all_labelings = obj.get_labelings()
#        ds_set.replace_labelings(all_labelings)
#
#        print(' number of structures (superperiodic eliminated) =', 
#                all_labelings.shape[0])
#        ##########################
#
#        ds_set_all.append(ds_set)
#
#    prefix = 'derivative-'+str(args.n_expand)
#    yaml = Yaml()
#    yaml.write_derivative_yaml(st_prim, ds_set_all, filename=prefix+'.yaml')
#
#    if args.nodump == False:
#        joblib.dump(ds_set_all, prefix+'.pkl', compress=3)
#
