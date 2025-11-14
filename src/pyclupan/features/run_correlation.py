"""Class for calculating cluster functions."""

# import time

import numpy as np

from pyclupan.cluster.cluster_io import load_cluster_yaml
from pyclupan.core.cell_utils import (
    get_unitcell_reps,
    is_cell_equal,
    reduced,
    supercell_reduced,
)
from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.core.spglib_utils import get_symmetry
from pyclupan.core.spin import eval_cluster_functions
from pyclupan.features.features_utils import (
    element_strings_to_labeling,
    structure_to_lattice,
)
from pyclupan.features.orbit_utils import find_orbit


def calc_correlation(
    lattice_unitcell: Lattice,
    lattice_supercell: Lattice,
    labelings: np.ndarray,
    clusters: list,
    spin_basis_clusters: list,
    verbose: bool = False,
):
    """Calculate cluster functions without loading cluster yaml file."""
    unitcell = lattice_unitcell.cell
    supercell = lattice_supercell.cell
    map_unit_to_sup = get_unitcell_reps(unitcell, supercell)
    rotations, translations = get_symmetry(unitcell)

    spins = lattice_supercell.to_spins(labelings)

    # t1 = time.time()
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
    # t2 = time.time()

    cluster_functions = []
    for cl in spin_basis_clusters:
        orbit = np.array(orbit_all[cl.cluster_id])
        coeffs = lattice_supercell.get_spin_polynomials(cl.spin_basis)
        cf = eval_cluster_functions(coeffs, spins[:, orbit])
        cluster_functions.append(cf)
    cluster_functions = np.array(cluster_functions).T

    # t3 = time.time()
    # print(t2 - t1, t3-t2)
    return cluster_functions


def run_correlation_from_structures(
    structures: list[PolymlpStructure],
    element_labels: dict,
    cluster_yaml: str = "pyclupan_cluster.yaml",
    verbose: bool = False,
):
    """Calculate cluster functions from derivative structure."""
    lattice, clusters, _, spin_basis_clusters = load_cluster_yaml(cluster_yaml)

    cluster_functions = []
    for st in structures:
        supercell_matrix = np.linalg.inv(lattice.axis) @ st.axis
        if not np.allclose(supercell_matrix - np.round(supercell_matrix), 0.0):
            raise RuntimeError("Axis of given structure not consistent with lattice.")

        supercell, tmat = reduced(st, return_transformation=True)
        supercell.supercell_matrix = supercell_matrix @ tmat
        lattice_supercell, labelings_order = structure_to_lattice(supercell, lattice)

        labeling = element_strings_to_labeling(supercell.elements, element_labels)
        labelings = np.array([labeling])[:, labelings_order]

        cf = calc_correlation(
            lattice,
            lattice_supercell,
            labelings,
            clusters,
            spin_basis_clusters,
            verbose=verbose,
        )
        cluster_functions.extend(cf)
    return np.array(cluster_functions)


def run_correlation(
    unitcell: PolymlpStructure,
    supercell_matrix: np.ndarray,
    labelings: np.ndarray,
    cluster_yaml: str = "pyclupan_cluster.yaml",
    verbose: bool = False,
):
    """Calculate cluster functions.

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
    lattice_supercell = lattice.lattice_supercell(supercell)

    cluster_functions = calc_correlation(
        lattice,
        lattice_supercell,
        labelings,
        clusters,
        spin_basis_clusters,
        verbose=verbose,
    )
    return cluster_functions
