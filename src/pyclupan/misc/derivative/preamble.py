#!/usr/bin/env python
import numpy as np

from pyclupan.common.io.io_alias import parse_clusters_yaml
from pyclupan.common.normal_form import get_nonequivalent_hnf
from pyclupan.dd.dd_node import DDNodeHandler
from pyclupan.cluster.cluster import Cluster
from pyclupan.cluster.cluster import ClusterSet

def preamble_ds_enum_from_cluster_yaml(clusters_yaml, 
                                       n_expand,
                                       hnf=None):

    print(' parsing clusters.yaml ...')
    clusters, clusters_ele = parse_clusters_yaml(clusters_yaml)
    st_prim = clusters_ele.prim
    elements = clusters_ele.elements_lattice # [[0],[1,2]]

    n_sites = list(np.array(st_prim.n_atoms) * n_expand)
    dd_handler = DDNodeHandler(n_sites=n_sites,
                               elements_lattice=elements,
                               one_of_k_rep=False)

    #hnf = np.array([[1,0,0],
#                    [0,1,0],
#                    [0,0,3]])
    hnf_all = get_nonequivalent_hnf(n_expand, st_prim)

    print(' initial setting for computing non-colored cluster orbits ...')
    clusters.apply_sym_operations()
    clusters_ele.find_orbits_primitive(noncolored_cluster_set=clusters,
                                       distinguish_element=True)

    return (dd_handler, st_prim, hnf_all, clusters_ele)


