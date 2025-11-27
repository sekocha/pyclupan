"""Class for calculating cluster orbits."""

from collections import defaultdict
from typing import Optional

import numpy as np
import scipy.spatial.distance as distance

from pyclupan.cluster.cluster_utils import ClusterAttr
from pyclupan.core.cell_positions_utils import decompose_fraction
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

    orbit = set()
    for sites, cells in zip(sites_sym, cells_sym):
        for origin in cells.T:
            cells_shifted = (cells.T - origin).T
            if np.any(np.linalg.norm(cells_shifted, axis=0) < 1e-15):
                to_sort = [(s, tuple(c)) for s, c in zip(sites, cells_shifted.T)]
                orbit.add(tuple(sorted(to_sort)))

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
    return orbit_sites, orbit_positions


def _get_matching_positions(
    multiple_positions: np.ndarray,
    positions_ref: np.ndarray,
    tol: float = 1e-8,
):
    """Return matching of two sets of positions."""
    size = multiple_positions.shape[2]
    multiple_positions_rev = multiple_positions.transpose((0, 2, 1)).reshape((-1, 3))
    sites = np.where(distance.cdist(multiple_positions_rev, positions_ref.T) < tol)[1]
    sites = sites.reshape((-1, size))
    return sites


def _get_matching_positions_usemap(
    multiple_positions: np.ndarray,
    map_supercell_positions: dict,
    decimals: int = 5,
):
    """Return matching of two sets of positions."""
    size = multiple_positions.shape[2]
    scale = 10**decimals
    multiple_positions_rev = (
        ((np.round(multiple_positions, decimals) * scale).astype(np.int64))
        .transpose((0, 2, 1))
        .reshape((-1, 3))
    )
    sites = [map_supercell_positions[pos.tobytes()] for pos in multiple_positions_rev]
    sites = np.array(sites).reshape((-1, size))
    return np.array(sites)


def find_orbit_supercell_nomap(
    unitcell: PolymlpStructure,
    supercell: PolymlpStructure,
    orbit_positions_unitcell: dict,
    map_unit_to_sup: dict,
    return_array: bool = False,
):
    """Extend orbit for unitcell to orbit for supercell."""
    orbit_sites = [] if return_array else defaultdict(list)

    supercell_matrix_inv = np.linalg.inv(supercell.axis) @ unitcell.axis
    for (site_unit, cell), site_sup in map_unit_to_sup.items():
        unitcell_fracs = np.array(orbit_positions_unitcell[site_unit])
        if unitcell_fracs.shape[0] > 0:
            fracs = unitcell_fracs + np.array(cell)[None, :, None]
            _, positions_sup = decompose_fraction(supercell_matrix_inv @ fracs)
            sites = _get_matching_positions(positions_sup, supercell.positions)
            if return_array:
                orbit_sites.extend(sites)
            else:
                orbit_sites[site_sup].extend(sites)

    return orbit_sites


def get_map_positions(cell: PolymlpStructure, decimals: int = 5):
    """Return mapping from position to site ID."""
    map_positions = dict()
    scale = 10**decimals
    for i, pos in enumerate(cell.positions.T):
        key = (np.round(pos, decimals) * scale).astype(np.int64).tobytes()
        map_positions[key] = i
    return map_positions


def find_orbit_supercell_usemap(
    unitcell: PolymlpStructure,
    supercell: PolymlpStructure,
    orbit_positions_unitcell: dict,
    map_unit_to_sup: dict,
    map_supercell_positions: Optional[dict] = None,
    return_array: bool = False,
    decimals: int = 5,
):
    """Extend orbit for unitcell to orbit for supercell."""
    orbit_sites = [] if return_array else defaultdict(list)

    if map_supercell_positions is None:
        map_supercell_positions = get_map_positions(supercell, decimals=decimals)

    supercell_matrix_inv = np.linalg.inv(supercell.axis) @ unitcell.axis
    for (site_unit, cell), site_sup in map_unit_to_sup.items():
        unitcell_fracs = np.array(orbit_positions_unitcell[site_unit])
        if unitcell_fracs.shape[0] > 0:
            fracs = unitcell_fracs + np.array(cell)[None, :, None]
            _, fracs_sup = decompose_fraction(supercell_matrix_inv @ fracs)
            sites = _get_matching_positions_usemap(fracs_sup, map_supercell_positions)
            if return_array:
                orbit_sites.extend(sites)
            else:
                orbit_sites[site_sup].extend(sites)
    return orbit_sites


def find_orbit_supercell(
    unitcell: PolymlpStructure,
    supercell: PolymlpStructure,
    orbit_positions_unitcell: dict,
    map_unit_to_sup: dict,
    map_supercell_positions: Optional[dict] = None,
    return_array: bool = False,
):
    """Extend orbit for unitcell to orbit for supercell."""
    if map_supercell_positions is None or len(supercell.types) < 100:
        return find_orbit_supercell_nomap(
            unitcell,
            supercell,
            orbit_positions_unitcell,
            map_unit_to_sup,
            return_array=return_array,
        )
    else:
        return find_orbit_supercell_usemap(
            unitcell,
            supercell,
            orbit_positions_unitcell,
            map_unit_to_sup,
            map_supercell_positions=map_supercell_positions,
            return_array=return_array,
        )


def find_orbit(
    cluster: ClusterAttr,
    unitcell: PolymlpStructure,
    supercell: PolymlpStructure,
    rotations_unitcell: np.ndarray,
    translations_unitcell: np.ndarray,
    map_unit_to_sup: dict,
    return_array: bool = False,
):
    """Find orbit for supercell using orbit for unitcell."""
    _, orbit_fracs = find_orbit_unitcell(
        cluster, unitcell, rotations_unitcell, translations_unitcell
    )
    orbit_sites = find_orbit_supercell(
        unitcell,
        supercell,
        orbit_fracs,
        map_unit_to_sup,
        return_array=return_array,
    )
    return orbit_sites


def find_orbit_supercell_working(
    unitcell: PolymlpStructure,
    supercell: PolymlpStructure,
    orbit_positions_unitcell: dict,
    map_unit_to_sup: dict,
    return_array: bool = False,
):
    """Extend orbit for unitcell to orbit for supercell."""
    orbit_sites = [] if return_array else defaultdict(list)

    supercell_matrix_inv = np.linalg.inv(supercell.axis) @ unitcell.axis
    for (site_unit, cell), site_sup in map_unit_to_sup.items():
        for unitcell_frac in orbit_positions_unitcell[site_unit]:
            frac = (unitcell_frac.T + cell).T
            sites = unitcell_reps_to_supercell_reps(
                frac,
                supercell,
                supercell_matrix_inv=supercell_matrix_inv,
            )
            if return_array:
                orbit_sites.append(sites)
            else:
                orbit_sites[site_sup].append(sites)
    return orbit_sites
