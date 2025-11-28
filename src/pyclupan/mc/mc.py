"""Class for performing Monte Carlo simulations."""

from typing import Literal, Optional

import numpy as np

from pyclupan.core.cell_utils import supercell_general
from pyclupan.features.cluster_functions_utils import ClusterFunctionsMC
from pyclupan.features.run_correlation import ClusterFunctions
from pyclupan.mc.mc_runs import cmc, sgcmc
from pyclupan.mc.mc_utils import MCAttr, MCParams
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

        self._mc_params = None
        self._mc_attr = MCAttr(
            active_element_species=self._lattice_unitcell._active_elements,
            spin_species=self._lattice_unitcell._active_spins,
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
        if self._verbose:
            print("Constructing supercell.", flush=True)

        sup = supercell_general(
            self._lattice_unitcell.cell,
            supercell_matrix,
            refine=refine,
            verbose=self._verbose,
        )
        self._lattice_supercell = self._lattice_unitcell.lattice_supercell(sup)
        self._cf_mc = ClusterFunctionsMC(
            self._cf,
            self._lattice_supercell,
            verbose=self._verbose,
        )
        return self

    def _set_init_structure_random(self, compositions: tuple):
        """Set initial structure randomly.

        Parameters
        ----------
        compositions: Compositions for active elements.
            Array indices correspond to element IDs.
            The compositions are defined as
            (number of atoms) / (number of active sites).
        """
        if not np.isclose(np.sum(compositions), 1.0):
            raise RuntimeError("Sum of given compositions is not one.")

        n_sites = len(self._lattice_supercell.active_sites)
        elements = self._mc_attr.active_element_species
        n_atoms = np.array([compositions[ele] * n_sites for ele in elements])
        if not np.allclose(n_atoms - np.round(n_atoms), 0.0):
            raise RuntimeError("Given supercell cannot express compositions.")

        n_atoms = np.round(n_atoms).astype(int)
        if self._verbose:
            print("Initial structure:", flush=True)
            print("- Number of active sites:", n_sites, flush=True)
            for e, n in zip(elements, n_atoms):
                print("- Active element", e, ":", n, flush=True)

        perm = np.random.permutation(n_sites)
        active_labelings = np.ones(n_sites, dtype=int) * -1
        begin = 0
        for ele, n in zip(elements, n_atoms):
            active_labelings[perm[begin : begin + n]] = ele
            begin += n
        assert np.all(active_labelings != -1)

        active_spins = self._lattice_supercell.to_spins(np.array([active_labelings]))[0]
        return active_spins

    def set_init_structure(self, compositions: Optional[tuple] = None):
        """Set initial structure."""
        if self._lattice_supercell is None:
            raise RuntimeError("Supercell not found.")

        if compositions is not None:
            active_spins = self._set_init_structure_random(compositions)
        else:
            pass

        self._mc_attr.active_spins = active_spins
        return self

    def set_init_properties(self):
        """Set initial properties."""
        if self._verbose:
            print("Calculating cluster functions of initial structure.", flush=True)

        cluster_functions = self._cf_mc.eval_from_spins(self._mc_attr.active_spins)
        self._mc_attr.energy = self._model.eval(cluster_functions)
        self._mc_attr.cluster_functions = cluster_functions
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
    ):
        """Set parameters."""
        self._mc_params = MCParams(
            n_steps_init=n_steps_init,
            n_steps_eq=n_steps_eq,
            temperature=temperature,
            temperatures=temperatures,
            ensemble=ensemble,
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

    def set_init(self, compositions: Optional[tuple] = None):
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

        self.set_init_structure(compositions=compositions)
        self.set_init_properties()
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
            self.run_cmc()
        elif self._mc_params.ensemble == "semi_grand_canonical":
            self.run_sgcmc()
        return self

    def run_cmc(self):
        """Run canoncial MC simulation."""
        for temp in self._mc_params.temperatures:
            cmc(
                temp,
                self._mc_attr,
                self._mc_params,
                self._cf_mc,
                self._model,
                verbose=self._verbose,
            )

    def run_sgcmc(self):
        """Run semi-grand canoncial MC simulation."""
        for temp in self._mc_params.temperatures:
            sgcmc()
            # sgcmc(self._mc_attr, self._mc_params, temp)

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


#     @property
#     def structures(self):
#         """Return structures."""
#         return self._structures
