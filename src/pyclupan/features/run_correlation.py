"""Class for calculating cluster functions."""

import time

import numpy as np

from pyclupan.cluster.cluster_io import load_cluster_yaml
from pyclupan.core.cell_utils import get_unitcell_reps, is_cell_equal, supercell_reduced
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.core.spglib_utils import get_symmetry
from pyclupan.features.orbit_utils import find_orbit


def run_correlation(
    unitcell: PolymlpStructure,
    supercell_matrix: np.ndarray,
    labelings: np.ndarray,
    cluster_yaml: str = "pyclupan_cluster.yaml",
    verbose: bool = False,
):
    """Search nonequivalent clusters.

    Parameters
    ----------
    unitcell: Unitcell.
    supercell_matrix: Supercell matrix.
    labelings: Element labelings in supercell.
    cluster_yaml: Name of output file for cluster search results.
    """
    lattice, clusters, _, spin_basis_clusters = load_cluster_yaml(cluster_yaml)
    if not is_cell_equal(unitcell, lattice.cell):
        raise RuntimeError("Unitcell in cluster.yaml is not equal to given unitcell.")

    supercell = supercell_reduced(unitcell, supercell_matrix=supercell_matrix)
    lattice.cell = supercell
    supercell_matrix = supercell.supercell_matrix

    map_unit_to_sup = get_unitcell_reps(unitcell, supercell)
    rotations, translations = get_symmetry(unitcell)
    t1 = time.time()
    orbit_all = []
    for cl in clusters:
        orbit = find_orbit(
            cl,
            unitcell,
            supercell,
            rotations,
            translations,
            map_unit_to_sup,
        )
        orbit_all.append(orbit)
    t2 = time.time()
    print(t2 - t1)

    print(labelings)
    spins = lattice.to_spins(labelings)
    print(spins)

    for cl in spin_basis_clusters:
        orbit = orbit_all[cl.cluster_id]
    print(lattice.spin_polynomials)

    return None
