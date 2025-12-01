"""API class for performing Monte Carlo simulations."""

from typing import Literal, Optional

import numpy as np

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
        mu: Optional[float] = None,
    ):
        """Set parameters.

        Parameters
        ----------
        TODO: Add docstrings.
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

    def set_init(self, compositions: tuple):
        """Set initial conditions.

        Parameters
        ----------
        TODO: Add docstrings.
        """
        self._mc.set_init(compositions)
        return self

    def run(self):
        """Run MC simulation."""
        self._mc.run()
        return self


#     @property
#     def structures(self):
#         """Return structures."""
#         return self._structures
#
#     @structures.setter
#     def structures(self, strs: list[PolymlpStructure]):
#         """Set structures to be calculated.
#
#         Parameter
#         ---------
#         str: List of structures to be calculated.
#         """
#         self._structures = self._cf.structures = strs
#         self._structure_ids = ["str-" + str(i) for i in range(len(strs))]
#
#     @property
#     def element_strings(self):
#         """Return element strings."""
#         return self._cf.element_strings
#
#     @element_strings.setter
#     def element_strings(self, element_strings: tuple):
#         """Set element strings.
#
#         Parameter
#         ---------
#         element_strings: Element strings.
#             The location index corresponds to label integer.
#             For example, element_strings are ("Ag", "Au"),
#             labels 0 and 1 indicate elements Ag and Au, respectively.
#         """
#         self._cf.element_strings = element_strings
#
#     @property
#     def cluster_functions(self):
#         """Return cluster functions."""
#         return self._cluster_functions
#
#     @property
#     def energies(self):
#         """Return energies (per unitcell)."""
#         return self._energies
#
#     @property
#     def formation_energies(self):
#         """Return formation energies (per unitcell)."""
#         return self._formation_energies
#
#     @property
#     def convexhull(self):
#         """Return convex hull of formation energies."""
#         return self._convex
