#!/usr/bin/env python 
import numpy as np
import joblib

from pyclupan.common.io.yaml import Yaml
from pyclupan.cluster.cluster import ClusterSet
from pyclupan.derivative.derivative import DSSet
from pyclupan.derivative.derivative import DSSample

def parse_clusters_yaml(filename):

    yaml = Yaml()
    clusters, clusters_ele = yaml.parse_clusters_yaml(filename=filename)
    print('  - yaml file (cluster) =', filename)

    if clusters_ele.get_num_clusters() == 0:
        raise RuntimeError(
                'Unable to parse nonequiv_element_config in clusters.yaml')
    return clusters, clusters_ele

def parse_derivatives(filenames):

    if len(filenames) > 1 and 'derivative-all.pkl' in filenames:
        filenames.remove('derivative-all.pkl')
    print('  - pkl files (derivative st.) =', filenames)

    ds_set_all = []
    for f in filenames:
        ds_set_all.extend(joblib.load(f))

    return DSSample(ds_set_all)

def sample_from_ds(ds_samp: DSSample, poscars=None, n_cell_ub=None):

    target_ids = []
    if poscars is None:
        lbls = ds_samp.get_all_labelings(n_cell_ub=n_cell_ub)
        for ids in sorted(lbls.keys()):
            n_cell, s_id = ids
            for l_id in lbls[ids][1]:
                target_ids.append((n_cell, s_id, l_id))
    else:
        for string in poscars:
            ids_string = string.split('/')[-1].replace('POSCAR-','').split('-')
            target_ids.append(tuple([int(i) for i in ids_string]))
        target_ids = sorted(target_ids)

        lbls = dict()
        for n_cell, s_id, l_id in target_ids:
            ids = (n_cell, s_id)
            l = ds_samp.get_labeling(n_cell, s_id, l_id)
            if ids in lbls:
                lbls[ids][0].append(l)
                lbls[ids][1].append(l_id)
            else:
                lbls[ids] = [[l], [l_id]]

        for ids in lbls.keys():
            lbls[ids][0] = np.array(lbls[ids][0])
            lbls[ids][1] = np.array(lbls[ids][1])

    return lbls, target_ids
