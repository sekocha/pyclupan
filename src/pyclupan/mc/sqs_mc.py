"""Class for performing simulated annealing to search SQS."""

from typing import Optional

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.features.cluster_functions import ClusterFunctions
from pyclupan.mc.mc_structure import spins_initial, structure_from_spins
from pyclupan.mc.mc_utils import MCAttr, MCParams, set_supercell
from pyclupan.mc.sqs_mc_runs import cmc
from pyclupan.mc.sqs_utils import calc_ideal_cluster_functions


class SqsMC:
    """Class for performing simulated annealing to search SQS."""

    def __init__(
        self,
        clusters_yaml: str = "pyclupan_clusters.yaml",
        cluster_ids: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        clusters_yaml: File of cluster attributes from cluster search.
        """
        self._verbose = verbose
        self._cf = ClusterFunctions(clusters_yaml=clusters_yaml, verbose=verbose)
        if cluster_ids is not None:
            self._cf.spin_basis_clusters = [
                self._cf.spin_basis_clusters[i] for i in cluster_ids
            ]
        self._cf_mc = None

        self._lattice_unitcell = self._cf.lattice_unitcell
        self._lattice_supercell = None
        self._element_strings = None

        self._mc_params = None
        self._mc_attr = MCAttr(
            active_element_species=self._lattice_unitcell._active_elements,
            spin_species=self._lattice_unitcell._active_spins,
        )
        self._ideal_cluster_functions = None

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
        if self._verbose:
            print("Constructing supercell.", flush=True)

        self._cf_mc, self._lattice_supercell = set_supercell(
            self._cf, supercell_matrix, refine=refine, verbose=self._verbose
        )
        return self

    def set_init(
        self,
        structure: Optional[PolymlpStructure] = None,
        element_strings: Optional[tuple] = None,
        compositions: Optional[tuple] = None,
    ):
        """Set initial conditions.

        Parameters
        ----------
        compositions: Compositions for active elements.
            Array indices correspond to element IDs.
            The compositions are defined as
            (number of atoms) / (number of active sites).
        """
        if self._lattice_supercell is None:
            raise RuntimeError("Set supercell first.")

        self._element_strings = element_strings
        active_spins = spins_initial(
            self._lattice_supercell,
            structure=structure,
            element_strings=element_strings,
            compositions=compositions,
            verbose=self._verbose,
        )
        print(active_spins)

        if self._verbose:
            print("Calculating cluster functions of initial structure.", flush=True)
        cluster_functions = self._cf_mc.eval_from_spins(active_spins)

        self._mc_attr.active_spins = active_spins
        self._mc_attr.n_active_sites = self._lattice_supercell.n_active_sites
        self._mc_attr.cluster_functions = cluster_functions

        self._ideal_cluster_functions = calc_ideal_cluster_functions(
            self._lattice_unitcell,
            self._lattice_supercell,
            self._cf,
            active_spins,
        )
        if self._verbose:
            print("Ideal cluster functions.", flush=True)
            print(self._ideal_cluster_functions)
        return self

    def set_parameters(
        self,
        n_steps: int = 100,
        temperature_init: float = 1000.0,
        temperature_final: float = 0.1,
        n_temperatures: int = 20,
    ):
        """Set parameters.

        Parameters
        ----------
        n_steps: Number of steps for simulated annealing at each temperature.
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
        self._mc_params = MCParams(n_steps_eq=n_steps, temperatures=temperatures)
        return self

    def run(self):
        """Run MC simulation."""
        if self._cf_mc is None:
            raise RuntimeError("Supercell and calculator not found.")
        if self._mc_attr.active_spins is None:
            raise RuntimeError("Initial configuration not found.")
        if self._mc_params is None:
            raise RuntimeError("Parameters not found.")
        if self._ideal_cluster_functions is None:
            raise RuntimeError("Ideal cluster functions not found.")

        if self._verbose:
            print("Run MC simulation.", flush=True)
            self._mc_params.print_parameters()
            self._mc_attr.print_attrs()

        for temp in self._mc_params.temperatures:
            if self._verbose:
                print("--- Temperature:", temp, "---", flush=True)

            self._mc_attr, score = cmc(
                temp,
                self._mc_attr,
                self._mc_params,
                self._ideal_cluster_functions,
                self._cf_mc,
                verbose=self._verbose,
            )
            if np.isclose(score, 0.0):
                break

        return self

    @property
    def unitcell(self):
        """Return unitcell."""
        return self._lattice_unitcell.cell

    @property
    def supercell(self):
        """Return supercell."""
        return self._lattice_supercell.cell

    @property
    def mc_attr(self):
        """Return attributes from MC."""
        return self._mc_attr

    @property
    def mc_params(self):
        """Return parameters used for MC."""
        return self._mc_params

    @property
    def temperatures(self):
        """Return simulation temperatures."""
        return self._mc_params.temperatures

    @temperatures.setter
    def temperatures(self, temperatures: tuple):
        """Setter of simulation temperatures."""
        self._mc_params.temperatures = temperatures

    @property
    def structure(self):
        """Return final structure."""
        st = structure_from_spins(
            self._lattice_supercell,
            self._mc_attr.active_spins,
            self._element_strings,
        )
        return st
