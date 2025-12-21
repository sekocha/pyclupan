"""Class for calculating cluster orbits."""

from typing import Optional

import numpy as np

from pyclupan.cluster.cluster_utils import ClusterAttr
from pyclupan.core.cell_utils import get_unitcell_reps
from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.core.spglib_utils import get_symmetry
from pyclupan.features.orbit_utils import (
    find_orbit_supercell,
    find_orbit_unitcell,
    get_map_positions,
)


def get_orbit_unitcell(clusters: list[ClusterAttr], unitcell: PolymlpStructure):
    """Find orbits of clusters in unitcell."""
    rotations, translations = get_symmetry(unitcell)

    orbit_fracs_all = []
    for cl in clusters:
        _, orbit_fracs = find_orbit_unitcell(cl, unitcell, rotations, translations)
        orbit_fracs_all.append(orbit_fracs)
    return orbit_fracs_all


def get_orbit_supercell(
    lattice_unitcell: Lattice,
    lattice_supercell: Lattice,
    orbit_fracs_unitcell: list,
    mask_clusters: Optional[np.ndarray] = None,
    decimals: int = 5,
    return_array: bool = True,
    verbose: bool = False,
):
    """Find orbits of clusters in supercell from unitcell orbits."""
    unitcell, supercell = lattice_unitcell.cell, lattice_supercell.cell
    map_unit_to_sup = get_unitcell_reps(unitcell, supercell)
    map_supercell_positions = get_map_positions(supercell, decimals=decimals)

    if mask_clusters is None:
        mask_clusters = np.ones(len(orbit_fracs_unitcell), dtype=bool)

    orbit_sites_supercell = [None for _ in orbit_fracs_unitcell]
    for i, (orbit_f, mask) in enumerate(zip(orbit_fracs_unitcell, mask_clusters)):
        if not mask:
            continue

        if verbose:
            print("Calculating orbits for cluster", i, flush=True)

        orbit = find_orbit_supercell(
            unitcell,
            supercell,
            orbit_f,
            map_unit_to_sup,
            map_supercell_positions=map_supercell_positions,
            return_array=return_array,
        )
        orbit_sites_supercell[i] = lattice_supercell.to_active_site_rep(orbit)
    return orbit_sites_supercell


# def find_orbit_supercell(
#     unitcell: PolymlpStructure,
#     supercell: PolymlpStructure,
#     orbit_positions_unitcell: dict,
#     map_unit_to_sup: dict,
#     map_supercell_positions: Optional[dict] = None,
#     return_array: bool = False,
# ):
#     """Extend orbit for unitcell to orbit for supercell."""
#     if map_supercell_positions is None or len(supercell.types) < 100:
#         return find_orbit_supercell_nomap(
#             unitcell,
#             supercell,
#             orbit_positions_unitcell,
#             map_unit_to_sup,
#             return_array=return_array,
#         )
#     else:
#         return find_orbit_supercell_usemap(
#             unitcell,
#             supercell,
#             orbit_positions_unitcell,
#             map_unit_to_sup,
#             map_supercell_positions=map_supercell_positions,
#             return_array=return_array,
#         )
