"""Utility functions for formation energy."""

from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

from pyclupan.core.composition import Composition
from pyclupan.core.lattice import Lattice
from pyclupan.core.model import CEmodel
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.features.run_correlation import (
    run_correlation,
    run_correlation_from_structures,
)


def get_chemical_compositions(
    structures: Optional[list[PolymlpStructure]] = None,
    element_strings: Optional[tuple] = None,
    labelings: Optional[np.ndarray] = None,
):
    """Return chemical compositions of endmembers."""
    if structures is None and labelings is None:
        raise RuntimeError("structures or labelings required.")

    chemical_comps = []
    if structures is not None:
        if element_strings is None:
            raise RuntimeError("Element strings required.")
        for st in structures:
            elements = np.array(st.elements)
            chem = [np.sum(elements == ele) for ele in element_strings]
            chemical_comps.append(chem)
    elif labelings is not None:
        uniq_types = np.unique(labelings)
        for single_labeling in labelings:
            chem = [np.sum(single_labeling == t) for t in uniq_types]
            chemical_comps.append(chem)
    chemical_comps = np.array(chemical_comps)
    return chemical_comps


def get_formation_energies(
    energies: np.ndarray,
    n_atoms_array: np.ndarray,
    model: CEmodel,
    lattice: Lattice,
    clusters: list,
    spin_basis_clusters: list,
    structures: Optional[list[PolymlpStructure]] = None,
    element_strings: Optional[tuple] = None,
    labelings: Optional[np.ndarray] = None,
    supercell_matrices: Optional[np.ndarray] = None,
    verbose: bool = False,
):
    """Evaluate formation energies.

    Parameters
    ----------
    energies: Energies per unitcell for structure set.
    n_atoms_array: Numbers of atoms for structure set.
    model: CEmodel instance used for calculating formation energies.
    lattice: Lattice in unitcell representation.
    """
    if structures is None and labelings is None:
        raise RuntimeError("structures or labelings required.")

    if structures is not None:
        if element_strings is None:
            raise RuntimeError("Element strings required.")
        cluster_functions = run_correlation_from_structures(
            structures,
            element_strings,
            lattice=lattice,
            clusters=clusters,
            spin_basis_clusters=spin_basis_clusters,
            verbose=verbose,
        )
    elif labelings is not None:
        labelings = np.array(labelings)
        if supercell_matrices is None:
            supercell_matrices = [np.eye(3) for _ in labelings]
        else:
            if len(labelings) != len(supercell_matrices):
                raise RuntimeError("Sizes of labelings and matrices are inconsistent.")

        cluster_functions = []
        for single_labeling, supercell_matrix in zip(labelings, supercell_matrices):
            single_labeling = np.array([single_labeling])
            cf = run_correlation(
                unitcell=lattice.cell,
                supercell_matrix=supercell_matrix,
                labelings=single_labeling,
                lattice=lattice,
                clusters=clusters,
                spin_basis_clusters=spin_basis_clusters,
                verbose=verbose,
            )
            cluster_functions.append(cf[0])
        cluster_functions = np.array(cluster_functions)

    chemical_comps_end_members = get_chemical_compositions(
        structures=structures,
        element_strings=element_strings,
        labelings=labelings,
    )

    comp = Composition(chemical_comps_end_members)
    comp.energies_end_members = model.eval(cluster_functions)

    n_cells = np.sum(n_atoms_array, axis=1) / np.sum(lattice.cell.n_atoms)
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
