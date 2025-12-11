"""API class for performing Monte Carlo simulations."""

from typing import Literal, Optional

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar, write_poscar_file
from pyclupan.mc.mc import MC


class PyclupanMC:
    """API class for performing Monte Carlo simulations."""

    def __init__(
        self,
        clusters_yaml: str = "pyclupan_clusters.yaml",
        ecis_yaml: str = "pyclupan_ecis.yaml",
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        clusters_yaml: File of cluster attributes from cluster search.
        ecis_yaml: File of ECIs from regression.
        """
        self._verbose = verbose
        self._mc = MC(
            clusters_yaml=clusters_yaml,
            ecis_yaml=ecis_yaml,
            verbose=verbose,
        )
        np.set_printoptions(legacy="1.21")

    def set_supercell(self, supercell_matrix: np.ndarray, refine: bool = False):
        """Set supercell.

        Parameters
        ----------
        supercell_matrix: Supercell matrix.
            If three elements are given, a diagonal supercell matrix of these
            elements will be used.
        refine: Refine unitcell before applying supercell matrix. Default: False.
            If True, a supercell is constructed by the expansion of given supercell
            matrix for the refined cell.
        """
        self._mc.set_supercell(supercell_matrix=supercell_matrix, refine=refine)
        return self

    def set_parameters(
        self,
        n_steps_init: int = 100,
        n_steps_eq: int = 1000,
        temperature: float = 1000,
        temperatures: Optional[np.ndarray] = None,
        temperature_init: Optional[float] = None,
        temperature_final: Optional[float] = None,
        temperature_step: Optional[float] = None,
        ensemble: Literal["canonical", "semi_grand_canonical"] = "canonical",
        mu: Optional[tuple] = None,
    ):
        """Set parameters.

        Parameters
        ----------
        n_steps_init: Number of steps for initialization.
        n_steps_eq: Number of steps for taking avarages.
        temperature: Single temperature.
        temperatures: Multiple temperatures.
        temperature_init: Initial temperature to set temperatures automatically.
        temperature_final: Final temperature to set temperatures automatically.
        temperature_step: Temperature step to set temperatures automatically.
        ensemble: Ensemble. Set "canonical" or "semi_grand_canonical".
        simulated_annealing: Perform simulated annealing.
            Three variables of temperature_init, temperature_final,
            and temperature_step are needed.
        mu: Chemical potential differences.
        """
        self._mc.set_parameters(
            n_steps_init=n_steps_init,
            n_steps_eq=n_steps_eq,
            temperature=temperature,
            temperatures=temperatures,
            temperature_init=temperature_init,
            temperature_final=temperature_final,
            temperature_step=temperature_step,
            ensemble=ensemble,
            mu=mu,
        )
        return self

    def set_parameters_simulated_annealing(
        self,
        n_steps_init: int = 100,
        n_steps_eq: int = 1000,
        temperature_init: float = 1000.0,
        temperature_final: float = 1.0,
        n_temperatures: int = 10,
    ):
        """Set parameters for performing simulated annealing.

        Parameters
        ----------
        n_steps_init: Number of steps for initialization.
        n_steps_eq: Number of steps for taking avarages.
        temperature_init: Initial temperature to set temperatures automatically.
        temperature_final: Final temperature to set temperatures automatically.
        n_temperatures: Number of temperatures.
        """
        if temperature_init < temperature_final:
            raise RuntimeError(
                "Final temperature must be smaller than initial temperature."
            )
        init, final = np.log10(temperature_init), np.log10(temperature_final)
        temperatures = np.logspace(init, final, num=n_temperatures)
        self._mc.set_parameters(
            n_steps_init=n_steps_init,
            n_steps_eq=n_steps_eq,
            temperatures=temperatures,
            ensemble="canonical",
        )
        return self

    def set_init(
        self,
        structure: Optional[PolymlpStructure] = None,
        poscar: Optional[str] = None,
        element_strings: Optional[tuple] = None,
        compositions: Optional[tuple] = None,
    ):
        """Set initial conditions.

        Parameters
        ----------
        structure: Initial structure for MC simulation.
        poscar: POSCAR file of initial structure for MC simulation.
        element_strings: Element strings to define element IDs.
        compositions: Compositions.
        """
        if poscar is not None:
            structure = Poscar(poscar).structure

        self._mc.set_init(
            structure=structure,
            element_strings=element_strings,
            compositions=compositions,
        )
        return self

    def run(self):
        """Run MC simulation."""
        self._mc.run()
        return self

    @property
    def structure(self):
        """Return final structure."""
        return self._mc.structure

    @property
    def temperatures(self):
        """Return simulation temperatures."""
        return self._mc.temperatures

    @property
    def average_energies(self):
        """Return average energies for simulation temperatures."""
        return self._mc.average_energies

    @property
    def average_cluster_functions(self):
        """Return average cluster functions for simulation temperatures."""
        return self._mc.average_cluster_functions

    def save_structure(self, filename: str = "POSCAR", header: str = "MC by clupan"):
        """Save structure to POSCAR file."""
        write_poscar_file(self.structure, filename=filename, header=header)

    def save_mc_yaml(self, filename: str = "pyclupan_mc.yaml"):
        """Save properties from MC."""
        self._mc.save_mc_yaml(filename=filename)
        return self
