"""Utility for initial structures in MC simulation."""

import copy
from typing import Optional

import numpy as np

from pyclupan.core.cell_utils import get_matching_positions, supercell_general
from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.features.features_utils import element_strings_to_labeling


def reordered_labeling(
    structure: PolymlpStructure,
    structure_ref: PolymlpStructure,
    element_strings: tuple,
):
    """Return labeling reordered with respect to reference structure."""
    labeling = element_strings_to_labeling(structure.elements, element_strings)
    order = get_matching_positions(structure_ref.positions, structure.positions)
    labeling = labeling[order]
    return labeling


def spins_from_structure(
    lattice_supercell: Lattice,
    structure: PolymlpStructure,
    element_strings: tuple,
    verbose: bool = False,
):
    """Set spins from structure file."""
    supercell = lattice_supercell.cell
    supercell_matrix = np.linalg.inv(structure.axis) @ supercell.axis
    if not np.allclose(supercell_matrix - np.round(supercell_matrix), 0.0):
        raise RuntimeError("Axis of given structure not consistent with MC supercell.")

    if verbose:
        print("Constructing supercell of given structure.", flush=True)
    st_sup = supercell_general(structure, supercell_matrix, refine=False)

    labeling = reordered_labeling(st_sup, supercell, element_strings)
    active_labelings = labeling[lattice_supercell.active_sites]
    active_spins = lattice_supercell.to_spins(active_labelings)
    return active_spins


def n_atoms_from_compositions(
    lattice_supercell: Lattice,
    compositions: tuple,
    decimals: int = 4,
    verbose: bool = False,
):
    """Return number of atoms from compositions."""
    if not np.isclose(np.sum(compositions), 1.0):
        raise RuntimeError("Sum of given compositions is not one.")

    elements = [e2 for ele in lattice_supercell.active_elements for e2 in ele]
    if len(elements) != len(compositions):
        raise RuntimeError("Size of given compositions is not consistent.")

    n_total_sites = len(lattice_supercell.active_sites)
    n_atoms = np.array([c * n_total_sites for ele, c in zip(elements, compositions)])
    if not np.allclose(n_atoms - np.rint(n_atoms), 0.0, atol=10 ** (-decimals)):
        raise RuntimeError("Given supercell cannot express compositions.")

    n_atoms = np.rint(n_atoms).astype(int)
    if verbose:
        print("Structure is generated from compositions.", flush=True)
        print("- Number of active sites:", n_total_sites, flush=True)
        for e, n in zip(elements, n_atoms):
            print("- Active element", e, ":", n, flush=True)

    return n_atoms


def _random_labeling(lattice_supercell: Lattice, n_atoms: np.ndarray):
    """Set random labeling."""
    n_total_sites = len(lattice_supercell.active_sites)
    active_labelings = np.ones(n_total_sites, dtype=int) * -1

    n_active_sites = lattice_supercell.n_active_sites
    elements = lattice_supercell.active_elements

    begin, element_id = 0, 0
    for ele, n_sites_sub in zip(elements, n_active_sites):
        perm = np.random.permutation(n_sites_sub) + begin
        begin2 = 0
        for e in ele:
            perm_slice = perm[begin2 : begin2 + n_atoms[element_id]]
            active_labelings[perm_slice] = e
            begin2 += n_atoms[element_id]
            element_id += 1
        begin += begin2
    assert np.all(active_labelings != -1)
    return active_labelings


def spins_random(
    lattice_supercell: Lattice,
    compositions: tuple,
    decimals: int = 4,
    verbose: bool = False,
):
    """Set spins randomly.

    Parameters
    ----------
    compositions: Compositions for active elements.
        Array indices correspond to element IDs.
        The compositions are defined as
        (number of atoms) / (number of active sites).
    """
    n_atoms = n_atoms_from_compositions(lattice_supercell, compositions, decimals)
    active_labelings = _random_labeling(lattice_supercell, n_atoms)
    active_spins = lattice_supercell.to_spins(active_labelings)
    return active_spins


def spins_initial(
    lattice_supercell: Lattice,
    structure: Optional[PolymlpStructure] = None,
    element_strings: Optional[tuple] = None,
    compositions: Optional[tuple] = None,
    verbose: bool = False,
):
    """Return initial active spins."""
    if structure is None and compositions is None:
        raise RuntimeError("Structure or compositions required.")

    if structure is not None:
        if element_strings is None:
            raise RuntimeError("Element strings required.")
        active_spins = spins_from_structure(
            lattice_supercell,
            structure,
            element_strings,
            verbose=verbose,
        )
    elif compositions is not None:
        active_spins = spins_random(lattice_supercell, compositions, verbose=verbose)
    return active_spins


def structure_from_spins(
    lattice_supercell: Lattice, active_spins: np.ndarray, element_strings: tuple
):
    """Return structure corresponding to spins."""
    active_labeling = lattice_supercell.to_labelings(active_spins)
    active_labeling = np.array([active_labeling])
    labeling = lattice_supercell.complete_labelings(active_labeling)[0]

    st = copy.deepcopy(lattice_supercell.cell)
    st.types = labeling
    if element_strings is None:
        element_strings = lattice_supercell.element_strings

    st.elements = [element_strings[t] for t in labeling]
    return st.reorder()
