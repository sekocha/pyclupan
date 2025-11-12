"""Class for calculating cluster orbits."""

from collections import defaultdict

import numpy as np

from pyclupan.cluster.cluster_utils import ClusterAttr
from pyclupan.core.cell_utils import unitcell_reps_to_supercell_reps
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.core.spglib_utils import apply_symmetry_operations


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
    orbit_sites = defaultdict(list)
    orbit_positions = defaultdict(list)
    for cl_info in orbit:
        for site, cell in cl_info:
            if np.linalg.norm(cell) < 1e-15:
                sites = np.array([s for s, c in cl_info])
                cells = np.array([c for s, c in cl_info]).T
                fracs = unitcell.positions[:, sites] + cells
                orbit_sites[site].append(sites)
                orbit_positions[site].append(fracs)
    # t3 = time.time()
    # print(t2-t1, t3-t2, t3-t1, len(orbit_site[0]))
    return orbit_sites, orbit_positions


def find_orbit_supercell(
    unitcell: PolymlpStructure,
    supercell: PolymlpStructure,
    orbit_positions_unitcell: dict,
    map_unit_to_sup: dict,
):
    """Extend orbit for unitcell to orbit for supercell."""
    orbit_sites = defaultdict(list)
    supercell_matrix_inv = np.linalg.inv(supercell.axis) @ unitcell.axis
    for (site_unit, cell), site_sup in map_unit_to_sup.items():
        for unitcell_frac in orbit_positions_unitcell[site_unit]:
            frac = (unitcell_frac.T + cell).T
            sites = unitcell_reps_to_supercell_reps(
                frac,
                supercell,
                supercell_matrix_inv=supercell_matrix_inv,
            )
            orbit_sites[site_sup].append(sites)

    return orbit_sites


def find_orbit(
    cluster: ClusterAttr,
    unitcell: PolymlpStructure,
    supercell: PolymlpStructure,
    rotations_unitcell: np.ndarray,
    translations_unitcell: np.ndarray,
    map_unit_to_sup: dict,
):
    """Find orbit for supercell using orbit for unitcell."""
    _, orbit_fracs = find_orbit_unitcell(
        cluster, unitcell, rotations_unitcell, translations_unitcell
    )
    orbit_sites = find_orbit_supercell(
        unitcell, supercell, orbit_fracs, map_unit_to_sup
    )
    return orbit_sites
