#!/usr/bin/env python 
import numpy as np
import joblib

from pyclupan.common.io.yaml import Yaml
from pyclupan.common.supercell import Supercell
from pyclupan.cluster.cluster import ClusterSet
from pyclupan.derivative.derivative import DSSet
from pyclupan.derivative.derivative import DSSample

class Features:
    def __init__(self, 
                 n_cell=None, 
                 s_id=None, 
                 labeling_ids=None, 
                 labelings=None,
                 features=None):
        
        self.n_cell = n_cell
        self.s_id = s_id
        self.labeling_ids = labeling_ids
        self.labelings = labelings
        self.features = features

    def set_features(self, features):
        self.features = features

    def set_orbits(self, orbits):
        self.orbits = orbits

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

    features_array = []
    if poscars is None:
        lbls = ds_samp.get_all_labelings(n_cell_ub=n_cell_ub)
        for ids in sorted(lbls.keys()):
            n_cell, s_id = ids
            lbl_ids = list(range(lbls[ids].shape[0]))
            features = Features(n_cell=n_cell,
                                s_id=s_id,
                                labeling_ids=lbl_ids,
                                labelings=lbls[ids])
            features_array.append(features)
    else:
        for string in poscars:
            ids_string = string.split('/')[-1].replace('POSCAR-','').split('-')
            target_ids.append(tuple([int(i) for i in ids_string]))
        target_ids = sorted(target_ids)

        lbls = dict()
        for n_cell, s_id, l_id in sorted(target_ids):
            ids = (n_cell, s_id)
            l = ds_samp.get_labeling(n_cell, s_id, l_id)
            if ids in lbls:
                lbls[ids][0].append(l)
                lbls[ids][1].append(l_id)
            else:
                lbls[ids] = [[l], [l_id]]

        for ids in sorted(lbls.keys()):
            n_cell, s_id = ids
            lbl_ids = lbls[ids][1]
            features = Features(n_cell=n_cell,
                                s_id=s_id,
                                labeling_ids=lbl_ids,
                                labelings=np.array(lbls[ids][0]))
            features_array.append(features)
 
    return features_array

def compute_orbits(ds_samp,
                   n_cell,
                   s_id,
                   cluster_set,
                   distinguish_element=True):

    prim = ds_samp.get_primitive_cell()
    supercell = ds_samp.get_supercell(n_cell, s_id)
    hnf = ds_samp.get_hnf(n_cell, s_id)

    sup = Supercell(st_prim=prim, hnf=hnf, st_supercell=supercell)
    sup.set_primitive_lattice_representation()

    return cluster_set.find_orbits_supercell(sup, distinguish_element=True)


