"""Utility functions for regression."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml


@dataclass
class CEmodel:
    coeffs: np.ndarray
    intercept: float
    cluster_ids: Optional[np.ndarray] = None
    rmse: Optional[float] = None


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


def check_data(
    features: np.ndarray,
    energies: np.ndarray,
    structure_ids_x: list[str],
    structure_ids_y: list[str],
):
    """Check matching of data entries."""
    order_y = find_matching_ids(structure_ids_y, structure_ids_x)
    x = np.array([features[ix] for ix, iy in enumerate(order_y) if iy is not None])
    y = np.array([energies[iy] for iy in order_y if iy is not None])
    return x, y


def load_energy_dat(energy_dat: str):
    """Load energy in text format."""
    data = np.loadtxt(energy_dat, dtype=str)
    ids = data[:, 0]
    energies = data[:, 1].astype(float)
    return ids, energies


def save_ecis(
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


def load_ecis(filename: str = "pyclupan_ecis.yaml"):
    """Load interaction."""
    data = yaml.safe_load(open(filename))
    intercept = data["intercept"]
    cluster_ids = [d["id"] for d in data["eci"]]
    coeffs = [d["coeff"] for d in data["eci"]]
    model = CEmodel(coeffs, intercept, cluster_ids=cluster_ids)
    return model
