#!/usr/bin/env python 
import numpy as np

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
        self.orbits = None
        self.orbit_sizes = None

    def set_features(self, features):
        self.features = features

    def set_orbits(self, orbits):

        """
        orbits = [[sites, ele], # cluster1 
                  [sites, ele], # cluster2
                  ...]
        """
        self.orbits = orbits  
        self.orbit_sizes = np.array([orb[0].shape[0] for orb in orbits])


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

def compute_orbits(ds_samp, n_cell, s_id, cluster_set):

    sup = ds_samp.get_supercell(n_cell, s_id)
    hnf = ds_samp.get_hnf(n_cell, s_id)
    return cluster_set.find_orbits_supercell(sup, hnf, distinguish_element=True)


