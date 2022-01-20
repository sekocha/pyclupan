#!/usr/bin/env python
import numpy as np
import os
import itertools
import random
import time

class DSSample:

    def __init__(self, ds_set_all):

        self.ds_set_all = ds_set_all
        self.n_cell_all = [ds_set.n_expand for ds_set in self.ds_set_all]

        self.serial_id_list = []
        for group_id, ds_set in enumerate(self.ds_set_all):
            for sup_id, lab_id in ds_set.get_id_list():
                self.serial_id_list.append((group_id, sup_id, lab_id))

        self.map_to_gid = dict()
        for g_id, ds_set in enumerate(self.ds_set_all):
            n_cell = self.n_cell_all[g_id]
            for sup_id in ds_set.supercell_idset:
                self.map_to_gid[(n_cell, sup_id)] = g_id
                

    def sample_random(self, k=5, element_symbols=None):

        st_attr_all, id_all = [], []
        samples = random.sample(self.serial_id_list, k=k)
        for g_id, s_id, l_id in sorted(samples):
            n_cell = self.n_cell_all[g_id]
            attr = self.ds_set_all[g_id].sample(s_id, l_id)
            st_attr_all.append(attr)
            id_all.append((n_cell, g_id, s_id, l_id))
        return st_attr_all, id_all

    def sample_all(self, n_cell_ub=None, element_symbols=None):

        if n_cell_ub is None:
            n_cell_ub = np.inf

        st_attr_all, id_all = [], []
        for g_id, (ds_set, n_cell) in enumerate(zip(self.ds_set_all, 
                                                    self.n_cell_all)):
            if n_cell <= n_cell_ub:
                attr = list(ds_set.sample_all())
                n_labelings = ds_set.n_labelings
                for s_id in ds_set.supercell_idset:
                    ids = [(n_cell,g_id,s_id,l_id) 
                            for l_id in range(n_labelings)]
                    st_attr_all.extend(attr)
                    id_all.extend(ids)
        return st_attr_all, id_all

    def get_labeling(self, n_cell, s_id, l_id):
        g_id = self.map_to_gid[(n_cell, s_id)]
        return self.ds_set_all[g_id].all_labelings[l_id]

    def get_supercell(self, n_cell, s_id):
        g_id = self.map_to_gid[(n_cell, s_id)]
        return self.ds_set_all[g_id].get_supercell_from_id(s_id)

    def get_hnf(self, n_cell, s_id):
        g_id = self.map_to_gid[(n_cell, s_id)]
        return self.ds_set_all[g_id].get_hnf_from_id(s_id)

class DSSet:

    def __init__(self, 
                 active_labelings=None,
                 inactive_labeling=None,
                 active_sites=None,
                 inactive_sites=None,
                 primitive_cell=None,
                 elements=None,
                 n_expand=None,
                 comp=None,
                 comp_lb=None,
                 comp_ub=None,
                 hnf=None,
                 supercell=None,
                 supercell_id=None,
                 hnf_set=None,
                 supercell_set=None,
                 supercell_idset=None):

        if self.is_none(hnf, hnf_set): 
            raise ValueError('hnf and hnf_set are None')
        if self.is_none(supercell, supercell_set): 
            raise ValueError('supercell and supercell_set are None')
        if self.is_none(hnf, supercell_set):
            raise ValueError('use (hnf,supercell) or (hnf_set,supercell_set)')

        self.prim = primitive_cell
        self.comp = comp
        self.comp_lb = comp_lb
        self.comp_ub = comp_ub

        self.active_labelings = active_labelings
        self.inactive_labeling = inactive_labeling
        self.active_sites = np.array(active_sites)
        self.inactive_sites = np.array(inactive_sites)
        self.n_sites = max(max(self.active_sites), max(inactive_sites)) + 1 

        self.elements = elements
        self.n_labelings = self.active_labelings.shape[0]

        # single supercell
        if hnf is not None:
            self.hnf_set = [hnf]
            self.supercell_set = [supercell]
            if supercell_id is None:
                self.supercell_idset = [0]
            else:
                self.supercell_idset = [supercell_id]
        else:
            # multiple supercells with common permutations
            if len(supercell_set) != len(hnf_set):
                raise ValueError('len(supercell_set) != len(hnf_set)')
            elif len(supercell_set) != len(supercell_idset):
                raise ValueError('len(supercell_set) != len(supercell_idset)')

            self.hnf_set = hnf_set
            self.supercell_set = supercell_set
            if supercell_idset is None:
                self.supercell_idset = list(range(supercell_set))
            else:
                self.supercell_idset = supercell_idset

        self.supercell_map = dict()
        for i, idx in enumerate(self.supercell_idset):
            self.supercell_map[idx] = i

        self.n_expand = n_expand
        if n_expand is None:
            if self.hnf is not None:
                self.n_expand = round(np.linalg.det(self.hnf))
            else:
                self.n_expand = round(np.linalg.det(self.hnf_set[0]))

        self.all_labelings = self.combine_all_labelings()

    def is_none(self, a, b):
        if a is None and b is None:
            return True
        return False

    def get_id_list(self):
        n_labelings = self.active_labelings.shape[0]
        return itertools.product(self.supercell_idset, range(n_labelings))

    def get_supercell_from_id(self, supercell_id):
        idx = self.supercell_map[supercell_id]
        return self.supercell_set[idx]

    def get_hnf_from_id(self, supercell_id):
        idx = self.supercell_map[supercell_id]
        return self.hnf_set[idx]

    def sample(self, supercell_id, labeling_id, return_labeling=False):
        labelings = self.all_labelings[labeling_id]
        if return_labeling == True:
            return self.get_supercell_from_id(supercell_id), labeling

        order = np.argsort(labeling)
        n_atoms = [np.sum(labeling == l) for l in self.elements]
        return (order, n_atoms)

    def sample_all(self, return_labeling=False):
        labelings = self.all_labelings
        if return_labeling == True:
            return self.supercell_set, self.supercell_idset, labelings

        order_all = np.argsort(labelings, axis=1)
        n_atoms_all = np.array([np.sum(labelings == l, axis=1) 
                                for l in self.elements]).T
        return zip(order_all, n_atoms_all)
 
    def combine_labeling(self, labeling_idx):
        labeling = np.zeros(self.n_sites, dtype=int)
        labeling[self.active_sites] = self.active_labelings[labeling_idx]
        labeling[self.inactive_sites] = self.inactive_labeling
        return labeling

    def combine_all_labelings(self):
        n_labelings = self.active_labelings.shape[0]
        labelings = np.zeros((n_labelings, self.n_sites), dtype=int)
        labelings[:,self.active_sites] = self.active_labelings
        labelings[:,self.inactive_sites] = self.inactive_labeling
        return labelings

