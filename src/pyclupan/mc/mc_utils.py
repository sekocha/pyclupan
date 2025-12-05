"""Utility class and functions for MC simulation."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from pyclupan.core.model import CEmodel
from pyclupan.core.pypolymlp_utils import PolymlpStructure


@dataclass
class MCAttr:
    """Dataclass for attributes in MC simulation.

    Parameters
    ----------
    active_spins: Active spins on MC lattice.
    energy: Energy value of spin configuration.
    cluster_functions: Cluster functions of spin_configuration.

    active_element_species: Element IDs activated in MC.
    spin_species: Spin values activated in MC.

    average_energy: Average energy from MC simulation.
    average_cluster_functions: Average cluster functions from MC simulation.
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
        print("  Number of sites:", self.n_sites, flush=True)
        print("  Elements:       ", list(self.active_element_species), flush=True)
        print("  Spins:          ", list(self.spin_species), flush=True)
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
    n_steps_init: Number of steps for initialization.
    n_steps_eq: Number of steps for taking avarages.
    temperature: Single temperature.
    ensemble: Ensemble. Set "canonical" or "semi_grand_canonical".
    mu: Chemical potential difference.
    temperatures: Multiple temperatures.
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


def save_mc_yaml(
    model: CEmodel,
    mc_attr: MCAttr,
    mc_params: MCParams,
    average_energies: np.array,
    average_cluster_functions: np.array,
    supercell: PolymlpStructure,
    filename: str = "pyclupan_mc.yaml",
):
    """Save properties from MC."""
    np.set_printoptions(suppress=True)
    with open(filename, "w") as f:
        print("mc_parameters:", file=f)
        n_sites = mc_attr.n_sites
        supercell_matrix = supercell.supercell_matrix.astype(int)
        print("  supercell_matrix:", file=f)
        print("  -", list(supercell_matrix[0]), file=f)
        print("  -", list(supercell_matrix[1]), file=f)
        print("  -", list(supercell_matrix[2]), file=f)
        print("  n_sites:        ", n_sites, file=f)

        print("  elements:       ", list(mc_attr.active_element_species), file=f)
        print("  spins:          ", list(mc_attr.spin_species), file=f)
        print("  ensemble:       ", mc_params.ensemble, file=f)
        if mc_params.ensemble == "canonical":
            print("  n_elements:     ", list(supercell.n_atoms), file=f)
        elif mc_params.ensemble == "semi_grand_canonical":
            print("  mu:             ", mc_params.mu, file=f)

        print("  n_steps_init:   ", mc_params.n_steps_init * n_sites, file=f)
        print("  n_steps_average:", mc_params.n_steps_eq * n_sites, file=f)
        print(file=f)

        print("mc_results:", file=f)
        for temp, e, cfs in zip(
            mc_params.temperatures, average_energies, average_cluster_functions
        ):
            print("- temperature:", temp, file=f)
            print("  energy:     ", e, file=f)
            print("  cluster_functions:", file=f)
            for cluster_id, cf in zip(model.cluster_ids, cfs):
                print("  - id:   ", cluster_id, file=f)
                print("    value:", np.round(cf, 7), file=f)


def set_temperatures_sa(
    temperature_init: float,
    temperature_final: float,
    temperature_step: float,
):
    """Set temperatures for simulated annealing."""
