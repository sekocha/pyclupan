"""Utility functions for cell positions."""

import numpy as np


def decompose_fraction(positions: np.ndarray, tol: float = 1e-10):
    """Decompose fractional coordinates into cell and positions from 0 and 1."""
    cells = np.floor(positions + tol).astype(int)
    positions_frac = positions - cells
    return cells, positions_frac


def get_matching_positions(
    positions: np.ndarray, positions_ref: np.ndarray, tol: float = 1e-10
):
    """Calculate matching of two set of positions."""
    import scipy.spatial.distance as distance

    sites = np.where(distance.cdist(positions.T, positions_ref.T) < tol)[1]
    if sites.shape[0] != positions.shape[1]:
        raise RuntimeError("Any positions are inconsitent with reference positions.")
    return sites
