"""Utility class and functions for MC simulation."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class MCAttr:
    """Dataclass for attributes in MC simulation.

    Parameters
    ----------
    TODO: Add document.
    """

    active_spins: Optional[np.ndarray] = None
    energy: Optional[float] = None
    cluster_functions: Optional[np.ndarray] = None

    active_element_species: Optional[np.ndarray] = None
    spin_species: Optional[np.ndarray] = None

    average_energy: Optional[float] = None
    average_cluster_functions: Optional[np.ndarray] = None

    @property
    def n_sites(self):
        """Return number of active sites."""
        if self.active_spins is None:
            return None
        return len(self.active_spins)

    def print_attrs(self):
        """Print attributes."""
        print("----------------------------------------------", flush=True)
        print("Lattices:", flush=True)
        print("  Number of active sites:", self.n_sites, flush=True)
        print("  Elements:              ", self.active_element_species, flush=True)
        print("  Spins:                 ", self.spin_species, flush=True)
        print("----------------------------------------------", flush=True)
        print("Properties:", flush=True)
        print("  Cluster functions:", flush=True)
        print(self.cluster_functions, flush=True)
        print("  Energy:", self.energy, flush=True)
        print("----------------------------------------------", flush=True)


@dataclass
class MCParams:
    """Dataclass for parameters in MC simulation.

    Parameters
    ----------
    TODO: Add document.
    """

    n_steps_init: int = 100
    n_steps_eq: int = 1000
    temperature: float = 1000
    ensemble: Literal["canonical", "semi_grand_canonical"] = "canonical"
    mu: Optional[float] = None

    temperatures: Optional[np.ndarray] = None

    def __post_init__(self):
        """Post init method."""
        if self.temperatures is None:
            self.temperatures = np.array([self.temperature])

        if self.ensemble == "semi_grand_canonical" and self.mu is None:
            raise RuntimeError("Chemical potential for SGCMC not found.")

    def set_temperature_range(
        self, temp_init: float, temp_final: float, temp_step: float
    ):
        """Set temperatures."""
        t_step = -temp_step if temp_final < temp_init else temp_step
        tol = -1e-8 if temp_final < temp_init else 1e-8
        self.temperatures = np.arange(temp_init, temp_final + tol, t_step)
        return self

    def print_parameters(self):
        """Print parameters."""
        print("----------------------------------------------", flush=True)
        print("MC parameters:", flush=True)
        print("  Ensemble:                     ", self.ensemble, flush=True)
        print("  n_steps (initialization):     ", self.n_steps_init, flush=True)
        print("  n_steps (average calculation):", self.n_steps_eq, flush=True)
        print("  Temperatures:", flush=True)
        for temp in self.temperatures:
            print("  -", temp, flush=True)
        print("----------------------------------------------", flush=True)
