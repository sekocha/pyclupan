"""Utility functions for labelings from zdd."""

import numpy as np


def get_nonequivalent_labelings(labelings: np.array, perms: np.array):
    """Calculate non-equivalent labelings."""
    reps = np.array([np.unique(l, axis=0)[0] for l in labelings[:, perms]])
    nonequivs = np.unique(reps, axis=0)
    return nonequivs


def eliminate_superperiodic_labelings(labelings: np.array, perms_lt: np.array):
    """Eliminate superperiodic labelings."""
    labelings_perm = labelings[:, perms_lt]
    uniq_ids = [
        i
        for i, l in enumerate(labelings_perm)
        if np.unique(l, axis=0).shape[0] == l.shape[0]
    ]
    return labelings[np.array(uniq_ids)]
