"""Utility functions for input/output of properties."""

import h5py
import numpy as np


def save_energies_hdf5(
    energies: np.ndarray,
    ids: list,
    n_atoms_array: np.ndarray,
    filename: str = "pyclupan_energies.hdf5",
):
    """Save energies in HDF5 format."""
    with h5py.File(filename, "w") as hdf5_file:
        hdf5_file.create_dataset("energies", data=energies)
        hdf5_file.create_dataset("ids", data=ids)
        hdf5_file.create_dataset("n_atoms", data=n_atoms_array)


def load_energies_hdf5(filename: str = "pyclupan_energies.hdf5"):
    """Load energies in HDF5 format."""
    with h5py.File(filename, "r") as hdf5_file:
        energies = hdf5_file["energies"][:]
        ids = hdf5_file["ids"][:]
        n_atoms_array = hdf5_file["n_atoms"][:]
    ids = [i.decode("utf-8") for i in ids]
    return energies, ids, n_atoms_array


def save_formation_energies_hdf5(
    energies: np.ndarray,
    compositions: np.ndarray,
    ids: list,
    filename: str = "pyclupan_formation_energies.hdf5",
):
    """Save formation energies in HDF5 format."""
    with h5py.File(filename, "w") as hdf5_file:
        hdf5_file.create_dataset("energies", data=energies)
        hdf5_file.create_dataset("compositions", data=compositions)
        hdf5_file.create_dataset("ids", data=ids)


def load_formation_energies_hdf5(filename: str = "pyclupan_formation_energies.hdf5"):
    """Load formation energies in HDF5 format."""
    with h5py.File(filename, "r") as hdf5_file:
        energies = hdf5_file["energies"][:]
        compositions = hdf5_file["compositions"][:]
        ids = hdf5_file["ids"][:]
    ids = [i.decode("utf-8") for i in ids]
    return energies, compositions, ids


def save_convex_yaml(
    convex_data: np.ndarray,
    filename: str = "pyclupan_convexhull.yaml",
):
    """Save convex hull."""
    with open(filename, "w") as f:
        print("convexhull:", file=f)
        for c in convex_data:
            print("- composition:     ", list(c[:-2].astype(float)), file=f)
            print("  formation_energy:", c[-2], file=f)
            print("  structure_id:    ", c[-1], file=f)
