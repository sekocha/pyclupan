#!/usr/bin/env python
import numpy as np
import spglib

from pyclupan.common.function import round_frac_array


class NiggliReduced:

    def __init__(self, axis):
        self.axis = axis
        self.tmat = None
        self.niggli_axis = None
        self.niggli_metric = None

        self.axis_to_niggli_reduce()

    # M = A^{-1} A' --> A' = AM
    def transformation_matrix(self, a_old, a_new):
        return np.dot(np.linalg.inv(a_old), a_new)

    def axis_to_metric(self, a: np.array):
        return np.dot(a.T, a)

    def axis_to_niggli_reduce(self):
        lattice = self.axis.T
        self.niggli_axis = spglib.niggli_reduce(lattice, eps=1e-15).T
        self.tmat = self.transformation_matrix(self.axis, self.niggli_axis)
        self.niggli_metric = self.axis_to_metric(self.niggli_axis)
        return self.niggli_axis, self.tmat, self.niggli_metric

    def transform_fr_coords(self, positions):
        pos = np.dot(np.linalg.inv(self.tmat), positions)
        return round_frac_array(pos)

    def get_niggli_axis(self):
        return self.niggli_axis

    def get_niggli_metric(self):
        return self.niggli_metric
