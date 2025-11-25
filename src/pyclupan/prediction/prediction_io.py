"""Utility functions for input/output of properties."""

import h5py
import numpy as np


def save_energies_hdf5(
    energies: np.ndarray,
    ids: list,
    filename: str = "pyclupan_energies.hdf5",
):
    """Save energies in HDF5 format."""
    with h5py.File(filename, "w") as hdf5_file:
        hdf5_file.create_dataset("energies", data=energies)
        hdf5_file.create_dataset("ids", data=ids)


def load_energies_hdf5(filename: str = "pyclupan_energies.hdf5"):
    """Load energies in HDF5 format."""
    with h5py.File(filename, "r") as hdf5_file:
        energies = hdf5_file["energies"][:]
        ids = hdf5_file["ids"][:]
    ids = [i.decode("utf-8") for i in ids]
    return energies, ids
