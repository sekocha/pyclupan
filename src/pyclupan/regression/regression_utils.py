"""Utility functions for regression."""

import numpy as np


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


def check_data(
    features: np.ndarray,
    energies: np.ndarray,
    structure_ids_x: list[str],
    structure_ids_y: list[str],
):
    """Check matching of data entries."""
    order_y = find_matching_ids(structure_ids_y, structure_ids_x)
    X = np.array([features[ix] for ix, iy in enumerate(order_y) if iy is not None])
    y = np.array([energies[iy] for iy in order_y if iy is not None])
    return X, y


def save_eci(
    coeffs: np.array,
    intercept: float,
    filename: str = "pyclupan_eci.yaml",
    tol: float = 1e-12,
):
    """Save interactions to file."""
    with open(filename, "w") as f:
        print("intercept:", intercept, file=f)
        print("eci:", file=f)
        for i, c in enumerate(coeffs):
            if np.abs(c) > tol:
                print("- id   :", i, file=f)
                print("  coeff:", c, file=f)
