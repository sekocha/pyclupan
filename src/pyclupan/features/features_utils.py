"""Utility functions for calculating features."""

from typing import Optional

import h5py
import numpy as np

from pyclupan.core.cell_utils import get_matching_positions
from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import PolymlpStructure, supercell


def element_strings_to_labeling(elements: list, element_strings: tuple):
    """Convert elements in structure to labeling."""
    labeling = np.zeros(len(elements), dtype=int)
    for label, ele in enumerate(element_strings):
        labeling[np.array(elements) == ele] = label
    return labeling


def get_chemical_compositions(
    structures: Optional[list[PolymlpStructure]] = None,
    element_strings: Optional[tuple] = None,
    labelings: Optional[np.ndarray] = None,
    n_elements: Optional[int] = None,
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
        if n_elements is None:
            raise RuntimeError("Number of elements required.")
        uniq_labels = np.arange(n_elements, dtype=int)
        for single_labeling in labelings:
            chem = [np.sum(single_labeling == t) for t in uniq_labels]
            chemical_comps.append(chem)
    chemical_comps = np.array(chemical_comps)
    return chemical_comps


def structure_to_lattice(st: PolymlpStructure, lattice_unitcell: Lattice):
    """Find lattice attributes in derivative supercell structure."""
    if st.supercell_matrix is None:
        raise RuntimeError("Supercell matrix attribute is required.")

    sup = supercell(lattice_unitcell.cell, st.supercell_matrix)
    lattice_supercell = lattice_unitcell.lattice_supercell(sup)
    labelings_order = get_matching_positions(sup.positions, st.positions)
    return lattice_supercell, labelings_order


def save_cluster_functions_hdf5(
    cluster_functions: np.ndarray,
    ids: list,
    n_atoms_array: np.array,
    filename: str = "pyclupan_features.hdf5",
):
    """Save cluster functions in HDF5 format."""
    with h5py.File(filename, "w") as hdf5_file:
        hdf5_file.create_dataset("cluster_functions", data=cluster_functions)
        hdf5_file.create_dataset("ids", data=ids)
        hdf5_file.create_dataset("n_atoms", data=n_atoms_array)


def load_cluster_functions_hdf5(filename: str = "pyclupan_features.hdf5"):
    """Load cluster functions in HDF5 format."""
    with h5py.File(filename, "r") as hdf5_file:
        cluster_functions = hdf5_file["cluster_functions"][:]
        ids = hdf5_file["ids"][:]
        try:
            n_atoms_array = hdf5_file["n_atoms"][:]
        except:
            n_atoms_array = None
    ids = [i.decode("utf-8") for i in ids]
    return cluster_functions, ids, n_atoms_array
