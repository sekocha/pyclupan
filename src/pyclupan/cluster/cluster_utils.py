"""Utility functions for cluster search."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, supercell


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
    spin_basis: Spin polynomial basis on sites.
    spin_basis_combinations: Set of spin polynomial basis on sites.
    cluster_id: ID for distinguishing lattice site cluster.
    colored_cluster_id: ID for distinguishing cluster with species.
    """

    sites_unitcell: Optional[tuple] = None
    cells_unitcell: Optional[np.ndarray] = None

    sites_supercell: Optional[tuple] = None
    positions_supercell: Optional[np.array] = None

    elements: Optional[tuple] = None
    elements_combinations: Optional[tuple] = None

    spin_basis: Optional[tuple] = None
    spin_basis_combinations: Optional[tuple] = None

    cluster_id: Optional[int] = None
    colored_cluster_id: Optional[int] = None

    def positions(self, unitcell: PolymlpStructure):
        """Return fractional coordinates of cluster."""
        cl_positions = unitcell.positions[:, np.array(self.sites_unitcell)]
        cl_positions += self.cells_unitcell
        return cl_positions


def find_supercell(unitcell: PolymlpStructure, max_cut: float):
    """Find supercell expansion used for searching clusters."""
    norm = np.linalg.norm(unitcell.axis, axis=0)
    supercell_matrix = np.diag(np.ceil(np.ones(3) * max_cut * 2 / norm))
    return supercell(unitcell, supercell_matrix=supercell_matrix)


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
