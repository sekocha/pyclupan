"""Utility class and functions for MC simulation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MCAttr:
    """Dataclass for attributes in MC simulation."""

    active_spins: Optional[np.ndarray] = None
    cluster_functions: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None

    @property
    def n_sites(self):
        """Return number of active sites."""
        return len(self.active_spins)
