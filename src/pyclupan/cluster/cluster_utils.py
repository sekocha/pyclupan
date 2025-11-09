"""Utility functions for cluster search."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ClusterAttr:
    """Dataclass for cluster attributes.

    Parameters
    ----------
    sites_unitcell: Site IDs in unitcell.
    cells_unitcell: Fractional coordinates of cells in unitcell representation.
    sites_supercell: Site IDs in reduced supercell.
    positions_supercell: Fractional coordinates of sites in reduced supercell.
    elements: Elements on sites.
    elements_combinations: Set of elements on sites.
    """

    sites_unitcell: Optional[tuple] = None
    cells_unitcell: Optional[np.ndarray] = None

    sites_supercell: Optional[tuple] = None
    positions_supercell: Optional[np.array] = None

    elements: Optional[tuple] = None
    elements_combinations: Optional[tuple] = None


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
