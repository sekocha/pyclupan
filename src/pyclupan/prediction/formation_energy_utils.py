"""Utility functions for formation energy."""

import copy
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

from pyclupan.core.composition import Composition
from pyclupan.core.model import CEmodel
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.features.cluster_functions import ClusterFunctions
from pyclupan.features.features_utils import get_chemical_compositions


def _calc_cluster_functions_endmembers(
    cf: ClusterFunctions,
    structures: Optional[list[PolymlpStructure]] = None,
    element_strings: Optional[tuple] = None,
    labelings: Optional[np.ndarray] = None,
    supercell_matrices: Optional[np.ndarray] = None,
):
    """Calculate cluster functions for endmembers."""
    if structures is None and labelings is None:
        raise RuntimeError("structures or labelings required.")

    unitcell = cf.lattice_unitcell.cell
    cf_end = copy.deepcopy(cf)
    if structures is not None:
        if element_strings is None:
            raise RuntimeError("Element strings required.")
        cf_end.structures = structures
        cf_end.element_strings = element_strings
        cluster_functions = cf_end.eval()
    elif labelings is not None:
        if supercell_matrices is None:
            supercell_matrices = [np.eye(3) for _ in labelings]
        else:
            if len(labelings) != len(supercell_matrices):
                raise RuntimeError("Sizes of labelings and matrices are inconsistent.")

        cluster_functions = []
        for single_labeling, supercell_matrix in zip(labelings, supercell_matrices):
            single_labeling = np.array([single_labeling])
            cf_end.set_labelings(unitcell, supercell_matrix, single_labeling)
            cluster_functions.append(cf_end.eval()[0])
        cluster_functions = np.array(cluster_functions)
    return cluster_functions


def get_formation_energies(
    energies: np.ndarray,
    model: CEmodel,
    cf: ClusterFunctions,
    structures_endmembers: Optional[list[PolymlpStructure]] = None,
    element_strings: Optional[tuple] = None,
    labelings_endmembers: Optional[np.ndarray] = None,
    supercell_matrices_endmembers: Optional[np.ndarray] = None,
    verbose: bool = False,
):
    """Evaluate formation energies.

    Parameters
    ----------
    energies: Energies per unitcell for structure set.
    model: CEmodel instance used for calculating formation energies.
    """
    if structures_endmembers is None and labelings_endmembers is None:
        labelings_endmembers = cf.lattice_unitcell.labelings_endmembers

    cluster_functions = _calc_cluster_functions_endmembers(
        cf,
        structures=structures_endmembers,
        element_strings=element_strings,
        labelings=labelings_endmembers,
        supercell_matrices=supercell_matrices_endmembers,
    )

    unitcell = cf.lattice_unitcell.cell
    n_atoms_array = cf.n_atoms_array
    if n_atoms_array is None:
        raise RuntimeError("Number of atoms not found.")
    if len(energies) != len(n_atoms_array):
        raise RuntimeError("Sizes of energies and n_atoms are not consistent.")

    chemical_comps_end_members = get_chemical_compositions(
        structures=structures_endmembers,
        element_strings=element_strings,
        labelings=labelings_endmembers,
        n_elements=cf.lattice_unitcell.n_elements,
    )

    comp = Composition(chemical_comps_end_members)
    comp.energies_end_members = model.eval(cluster_functions)

    n_cells = np.sum(n_atoms_array, axis=1) / np.sum(unitcell.n_atoms)
    formation_energies = comp.compute_formation_energies(
        energies * n_cells, n_atoms_array
    )
    compositions = comp.compositions
    return formation_energies, compositions


def append_formation_energies_endmembers(
    compositions: np.ndarray,
    formation_energies: np.ndarray,
    struture_ids: list,
):
    """Append formation energies for endmembers."""
    n_type = compositions.shape[1]
    compositions = np.vstack([compositions, np.eye(n_type)])
    formation_energies = np.concatenate([formation_energies, np.zeros(n_type)])
    for i in range(n_type):
        struture_ids.append("Endmember-" + str(i + 1))
    return compositions, formation_energies, struture_ids


def find_convex_hull(
    compositions: np.ndarray,
    formation_energies: np.ndarray,
    struture_ids: list,
):
    """Find convex hull of formation energies."""
    if compositions.shape[0] != formation_energies.shape[0]:
        raise RuntimeError("Inconsistent sizes of compositions and energies.")

    d_target = np.hstack([compositions[:, 1:], formation_energies.reshape((-1, 1))])
    ch = ConvexHull(d_target)
    v_convex = np.unique(ch.simplices)

    compositions_convex = compositions[v_convex]
    formation_energies_convex = formation_energies[v_convex]
    struture_ids_convex = np.array(struture_ids)[v_convex]

    lower_convex = formation_energies_convex < 1e-10
    convex_data = np.hstack(
        [
            compositions_convex[lower_convex],
            formation_energies_convex[lower_convex].reshape((-1, 1)),
            struture_ids_convex[lower_convex].reshape((-1, 1)),
        ]
    )
    index = np.lexsort(convex_data.T[:-2])
    convex_data = convex_data[index]
    return convex_data
