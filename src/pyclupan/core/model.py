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

    def eval(self, cluster_functions: np.ndarray):
        """Evaluate energies."""
        if cluster_functions.shape[1] != self.coeffs.shape[0]:
            raise RuntimeError("Inconsistent dimension of cluster functions and ECIs.")

        energies = cluster_functions @ self.coeffs
        energies += self.intercept
        return energies
