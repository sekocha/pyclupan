"""Class for calculating cluster orbits."""

# import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pyclupan.cluster.cluster_utils import ClusterAttr
from pyclupan.core.cell_utils import unitcell_reps_to_supercell_reps

# from pyclupan.core.cell_utils import get_unitcell_reps
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.core.spglib_utils import apply_symmetry_operations


@dataclass
class OrbitAttr:
    """Class for cluster orbit attributes."""

    sites: np.ndarray
    cells: np.ndarray


@dataclass
class Orbit:
    """Class for cluster orbit."""

    attrs: list[OrbitAttr]


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


def get_orbit_supercell(
    unitcell: PolymlpStructure,
    supercell: PolymlpStructure,
    orbit_unitcell: dict,
    map_unit_to_sup: dict,
):
    """Extend orbit for unitcell to orbit for supercell."""
    orbit_site = defaultdict(list)
    for key, site_sup in map_unit_to_sup.items():
        site_unit, cell = key
        for attr in orbit_unitcell[site_unit]:
            unitcell_frac = unitcell.positions[:, attr.sites]
            unitcell_frac += attr.cells
            unitcell_frac = (unitcell_frac.T + cell).T
            sites = unitcell_reps_to_supercell_reps(unitcell_frac, unitcell, supercell)
            orbit_site[site_sup].append(sites)
    return orbit_site
