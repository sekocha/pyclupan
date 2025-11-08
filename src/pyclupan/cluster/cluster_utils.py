"""Utility functions for cluster search."""

import numpy as np

# from pyclupan.core.pypolymlp_utils import PolymlpStructure


def calc_distance_pairs(
    axis: np.ndarray,
    positions_i: np.ndarray,
    positions_j: np.ndarray,
):
    """Calculate distances between two sites."""
    if positions_i.shape != positions_j.shape:
        raise RuntimeError("Inconsistent shape of positions.")

    diff = positions_j - positions_i
    diff -= np.round(diff)
    distance = np.linalg.norm(axis @ diff, axis=0)
    return distance
