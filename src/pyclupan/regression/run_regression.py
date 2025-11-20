"""Class for calculating effective cluster interactions."""

from typing import Optional

import numpy as np

from pyclupan.features.features_utils import load_cluster_functions_hdf5


def find_matching_ids(ids: str, ids_ref: str):
    """Find matching of two ID sets."""
    order = []
    for i in ids_ref:
        match = np.where(ids == i)[0]
        if len(match) == 0:
            order.append(None)
        else:
            order.append(match[0])
    return np.array(order)


def load_energy_dat(energy_dat: str):
    """Load energy in text format."""
    data = np.loadtxt(energy_dat, dtype=str)
    ids = data[:, 0]
    energies = data[:, 1].astype(float)
    return ids, energies


def load_regression_data(
    features_hdf5: str = "pyclupan_features.hdf5",
    energy_yaml: Optional[str] = None,
    energy_dat: str = None,
):
    """Load regression data."""
    cluster_functions, ids_x = load_cluster_functions_hdf5(features_hdf5)
    if energy_yaml is not None:
        pass
    elif energy_dat is not None:
        ids_y, energies = load_energy_dat(energy_dat)

    order_y = find_matching_ids(ids_y, ids_x)
    X = np.array(
        [cluster_functions[ix] for ix, iy in enumerate(order_y) if iy is not None]
    )
    y = np.array([energies[iy] for iy in order_y if iy is not None])
    return X, y


def run_regression(
    features_hdf5: str = "pyclupan_features.hdf5",
    energy_yaml: Optional[str] = None,
    energy_dat: str = None,
):
    """Run regression for estimating interactions."""
    X, y = load_regression_data(
        features_hdf5,
        energy_yaml=energy_yaml,
        energy_dat=energy_dat,
    )
    print(X, y)
