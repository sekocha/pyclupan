"""Class for calculating cluster functions."""

import time

import numpy as np

from pyclupan.cluster.cluster_io import load_cluster_yaml
from pyclupan.core.cell_utils import get_unitcell_reps, is_cell_equal, supercell_reduced
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.core.spglib_utils import get_symmetry
from pyclupan.core.spin import eval_cluster_functions
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

    Return
    ------
    cluster_functions: Cluster functions for labelings.
        shape: (n_labeling, n_features)
    """
    lattice, clusters, _, spin_basis_clusters = load_cluster_yaml(cluster_yaml)
    if not is_cell_equal(unitcell, lattice.cell):
        raise RuntimeError("Unitcell in cluster.yaml is not equal to given unitcell.")

    supercell = supercell_reduced(unitcell, supercell_matrix=supercell_matrix)
    lattice.cell = supercell
    supercell_matrix = supercell.supercell_matrix

    map_unit_to_sup = get_unitcell_reps(unitcell, supercell)
    rotations, translations = get_symmetry(unitcell)

    spins = lattice.to_spins(labelings)

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
            return_array=True,
        )
        orbit_all.append(orbit)
    t2 = time.time()

    cluster_functions = []
    for cl in spin_basis_clusters:
        orbit = np.array(orbit_all[cl.cluster_id])
        coeffs = lattice.get_spin_polynomials(cl.spin_basis)
        print(coeffs)
        print(spins[:, orbit])
        cf = eval_cluster_functions(coeffs, spins[:, orbit])
        cluster_functions.append(cf)
    cluster_functions = np.array(cluster_functions).T

    t3 = time.time()
    print(t2 - t1, t3 - t2)
    print(cluster_functions)
    return cluster_functions
