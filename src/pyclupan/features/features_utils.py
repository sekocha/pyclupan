"""Utility functions for calculating features."""

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


def structure_to_lattice(
    st: PolymlpStructure,
    lattice_unitcell: Lattice,
    only_active: bool = True,
):
    """Find lattice attributes in derivative supercell structure."""
    if st.supercell_matrix is None:
        raise RuntimeError("Supercell matrix attribute is required.")

    sup = supercell(lattice_unitcell.cell, st.supercell_matrix)
    lattice_supercell = lattice_unitcell.lattice_supercell(sup)
    labelings_order = get_matching_positions(sup.positions, st.positions)

    if only_active:
        labelings_order = labelings_order[lattice_supercell.active_sites]
    return lattice_supercell, labelings_order


def save_cluster_functions_hdf5(
    cluster_functions: np.ndarray,
    ids: list,
    filename: str = "pyclupan_features.hdf5",
):
    """Save cluster functions in HDF5 format."""
    with h5py.File(filename, "w") as hdf5_file:
        hdf5_file.create_dataset("cluster_functions", data=cluster_functions)
        hdf5_file.create_dataset("ids", data=ids)


def load_cluster_functions_hdf5(filename: str = "pyclupan_features.hdf5"):
    """Load cluster functions in HDF5 format."""
    with h5py.File(filename, "r") as hdf5_file:
        cluster_functions = hdf5_file["cluster_functions"][:]
        ids = hdf5_file["ids"][:]
    return cluster_functions, ids
