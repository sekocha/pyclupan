"""Functions for CE model."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CEmodel:
    """Dataclass for cluster expansion model."""

    coeffs: np.ndarray
    intercept: float
    cluster_ids: Optional[np.ndarray] = None
    rmse: Optional[float] = None

    def __post_init__(self):
        """Post init method."""
        self.coeffs = np.array(self.coeffs)

    def eval(self, cluster_functions: np.ndarray):
        """Evaluate energies."""
        if cluster_functions.ndim == 1:
            if cluster_functions.shape[0] != len(self.coeffs):
                raise RuntimeError("Inconsistent dim. of cluster functions and ECIs.")
        elif cluster_functions.ndim == 2:
            if cluster_functions.shape[1] != len(self.coeffs):
                raise RuntimeError("Inconsistent dim. of cluster functions and ECIs.")

        energies = cluster_functions @ self.coeffs
        energies += self.intercept
        return energies

    def nonzero_spin_basis(self, spin_basis: list):
        """Extract spin basis clusters with nonzero ECIs."""
        return [spin_basis[i] for i in self.cluster_ids]

    def supercell(self, n_expand: int):
        """Transform unit of eV/unitcell to eV/supercell."""
        self.coeffs *= n_expand
        self.intercept *= n_expand
