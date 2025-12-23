"""Class for performing Monte Carlo simulations."""

import copy
from typing import Literal, Optional

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.features.cluster_functions import ClusterFunctions
from pyclupan.mc.mc_runs import cmc, sgcmc
from pyclupan.mc.mc_structure import spins_from_structure, spins_random
from pyclupan.mc.mc_utils import MCAttr, MCParams, save_mc_yaml, set_supercell
from pyclupan.regression.regression_utils import load_ecis


class MC:
    """Class for performing Monte Carlo simulations."""

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
        self._cf = ClusterFunctions(clusters_yaml=clusters_yaml, verbose=verbose)
        self._model = load_ecis(ecis_yaml)
        self._cf.spin_basis_clusters = self._model.nonzero_spin_basis(
            self._cf.spin_basis_clusters
        )
        self._cf_mc = None

        self._lattice_unitcell = self._cf.lattice_unitcell
        self._lattice_supercell = None
        self._element_strings = None

        self._mc_params = None
        self._mc_attr = MCAttr(
            active_element_species=self._lattice_unitcell._active_elements,
            spin_species=self._lattice_unitcell._active_spins,
        )

        self._average_energies = None
        self._average_cfs = None
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
        n_expand = np.linalg.det(self._lattice_supercell.cell.supercell_matrix)
        self._model.supercell(n_expand)
        return self

    def _set_init_structure(
        self,
        structure: Optional[PolymlpStructure] = None,
        element_strings: Optional[tuple] = None,
        compositions: Optional[tuple] = None,
    ):
        """Set initial structure."""
        if structure is None and compositions is None:
            raise RuntimeError("Structure or compositions required.")
        if self._lattice_supercell is None:
            raise RuntimeError("Supercell not found.")

        if structure is not None:
            if element_strings is None:
                raise RuntimeError("Element strings required.")
            active_spins = spins_from_structure(
                self._lattice_supercell,
                structure,
                element_strings,
                verbose=self._verbose,
            )
        elif compositions is not None:
            active_spins = spins_random(
                self._lattice_supercell,
                compositions,
                verbose=self._verbose,
            )
        self._mc_attr.active_spins = active_spins
        self._mc_attr.n_active_sites = self._lattice_supercell.n_active_sites
        return self

    def _set_init_properties(self):
        """Set initial properties."""
        if self._verbose:
            print("Calculating cluster functions of initial structure.", flush=True)

        cluster_functions = self._cf_mc.eval_from_spins(self._mc_attr.active_spins)
        self._mc_attr.energy = self._model.eval(cluster_functions)
        self._mc_attr.cluster_functions = cluster_functions
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
        self._set_init_structure(
            structure=structure,
            element_strings=element_strings,
            compositions=compositions,
        )
        self._set_init_properties()
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
        """Set parameters."""
        elements = [e2 for ele in self._mc_attr.active_element_species for e2 in ele]
        if mu is not None and len(mu) != len(elements) - 1:
            raise RuntimeError("Size of chemical potentials is not consistent.")

        self._mc_params = MCParams(
            n_steps_init=n_steps_init,
            n_steps_eq=n_steps_eq,
            temperature=temperature,
            temperatures=temperatures,
            ensemble=ensemble,
            mu=mu,
        )
        if temperature_init is not None:
            if temperature_final is None:
                raise RuntimeError("Final temperature not found.")
            if temperature_step is None:
                raise RuntimeError("Temperature step not found.")
            self._mc_params.set_temperature_range(
                temperature_init, temperature_final, temperature_step
            )
        return self

    def run(self):
        """Run MC simulation."""
        if self._cf_mc is None:
            raise RuntimeError("Supercell and calculator not found.")
        if self._mc_attr.active_spins is None:
            raise RuntimeError("Initial configuration not found.")
        if self._mc_params is None:
            raise RuntimeError("Parameters not found.")

        if self._verbose:
            print("Run MC simulation.", flush=True)
            self._mc_params.print_parameters()
            self._mc_attr.print_attrs()

        if self._mc_params.ensemble == "canonical":
            self._run_cmc()
        elif self._mc_params.ensemble == "semi_grand_canonical":
            self._run_sgcmc()

        return self

    def _run_cmc(self):
        """Run canoncial MC simulation."""
        self._average_energies, self._average_cfs = [], []
        for temp in self._mc_params.temperatures:
            if self._verbose:
                print("--- Temperature:", temp, "---", flush=True)

            self._mc_attr = cmc(
                temp,
                self._mc_attr,
                self._mc_params,
                self._cf_mc,
                self._model,
                verbose=self._verbose,
            )
            self._average_energies.append(self._mc_attr.average_energy)
            self._average_cfs.append(self._mc_attr.average_cluster_functions)
        return self

    def _run_sgcmc(self):
        """Run semi-grand canoncial MC simulation."""
        self._average_energies, self._average_cfs = [], []
        for temp in self._mc_params.temperatures:
            if self._verbose:
                print("--- Temperature:", temp, "---", flush=True)

            self._mc_attr = sgcmc(
                temp,
                self._mc_attr,
                self._mc_params,
                self._cf_mc,
                self._model,
                verbose=self._verbose,
            )
            self._average_energies.append(self._mc_attr.average_energy)
            self._average_cfs.append(self._mc_attr.average_cluster_functions)
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
    def average_energies(self):
        """Return average energies."""
        return np.array(self._average_energies)

    @property
    def average_cluster_functions(self):
        """Return average cluster functions."""
        return np.array(self._average_cfs)

    @property
    def structure(self):
        """Return final structure."""
        active_spins = self._mc_attr.active_spins
        active_labeling = self._lattice_supercell.to_labelings(active_spins)
        active_labeling = np.array([active_labeling])
        labeling = self._lattice_supercell.complete_labelings(active_labeling)[0]

        st = copy.deepcopy(self.supercell)
        st.types = labeling
        if self._element_strings is None:
            self._element_strings = self._lattice_supercell.element_strings

        st.elements = [self._element_strings[t] for t in labeling]
        return st.reorder()

    def save_mc_yaml(self, filename: str = "pyclupan_mc.yaml"):
        """Save properties from MC."""
        save_mc_yaml(
            self._model,
            self._mc_attr,
            self._mc_params,
            self._average_energies,
            self._average_cfs,
            self.supercell,
            filename=filename,
        )
