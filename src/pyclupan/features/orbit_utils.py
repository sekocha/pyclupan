"""Class for calculating cluster orbits."""

# import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pyclupan.cluster.cluster_utils import ClusterAttr
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.core.spglib_utils import apply_symmetry_operations


@dataclass
class OrbitAttr:
    """Class for cluster orbit attributes."""

    sites: np.ndarray
    cells: np.ndarray


def find_orbit_unitcell(
    cluster: ClusterAttr,
    unitcell: PolymlpStructure,
    rotations: np.ndarray,
    translations: np.ndarray,
):
    """Find orbit of a cluster."""
    sites_sym, cells_sym = apply_symmetry_operations(
        rotations,
        translations,
        cluster.positions(unitcell),
        positions_ref=unitcell.positions,
    )

    # t1 = time.time()
    orbit = set()
    for sites, cells in zip(sites_sym, cells_sym):
        for origin in cells.T:
            cells_shifted = (cells.T - origin).T
            if np.any(np.linalg.norm(cells_shifted, axis=0) < 1e-15):
                to_sort = [(s, tuple(c)) for s, c in zip(sites, cells_shifted.T)]
                orbit.add(tuple(sorted(to_sort)))

    # t2 = time.time()
    orbit_site = defaultdict(list)
    for cl_info in orbit:
        for site, cell in cl_info:
            if np.linalg.norm(cell) < 1e-15:
                sites = np.array([s for s, c in cl_info])
                cells = np.array([c for s, c in cl_info]).T
                attr = OrbitAttr(sites, cells)
                orbit_site[site].append(attr)
    # t3 = time.time()
    # print(t2-t1, t3-t2, t3-t1, len(orbit_site[0]))
    return orbit_site
