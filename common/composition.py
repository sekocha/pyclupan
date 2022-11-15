#!/usr/bin/env python
import numpy as np

class Composition:

    def __init__(self, n_atoms_end, e_end=None):

        #n_atoms_end = np.array((n_types, n_ends), dtype=int)
        #n_types: number of atom types 
        #n_ends: number of end members

        self.n_atoms_end_rec = np.linalg.pinv(n_atoms_end)
        self.e_end = e_end

    def get_comp(self, n_atoms):

        partition = np.dot(self.n_atoms_end_rec, n_atoms)
        comp = partition / sum(partition)
        return comp, partition

    def compute_formation_energy(self, e, partition=None, n_atoms=None):

        if partition is None and n_atoms is not None:
            _, partition = self.get_comp(n_atoms)
        return (e - np.dot(self.e_end, partition)) / sum(partition)


