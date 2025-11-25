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
    if len(uniq_ids) == 0:
        return None
    return labelings[np.array(uniq_ids)]


def get_complete_labelings(
    active_labelings: np.ndarray,
    inactive_labeling: np.ndarray,
    active_sites: np.ndarray,
    inactive_sites: np.ndarray,
):
    """Return complete labelings from both active and inactive labelings."""
    n_site = active_labelings.shape[1] + len(inactive_labeling)
    n_labelings = active_labelings.shape[0]
    labelings = np.zeros((n_labelings, n_site), dtype=np.uint8)
    labelings[:, active_sites] = active_labelings
    if len(inactive_sites) > 0:
        labelings[:, inactive_sites] = inactive_labeling
    return labelings
