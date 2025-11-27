"""Class for performing Monte Carlo simulations."""

# from typing import Optional

import numpy as np

from pyclupan.core.cell_utils import supercell, supercell_diagonal
from pyclupan.core.spglib_utils import refine_cell
from pyclupan.features.run_correlation import ClusterFunctions
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
        self._lattice_unitcell = self._cf.lattice_unitcell

        self._model = load_ecis(ecis_yaml)
        self._cf.spin_basis_clusters = self._model.nonzero_spin_basis(
            self._cf.spin_basis_clusters
        )

        self._lattice_supercell = None
        self._cluster_functions = None
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

        unitcell = self._lattice_unitcell.cell
        if refine:
            unitcell_rev = refine_cell(unitcell)
            if self._verbose:
                if not np.allclose(unitcell_rev.axis - unitcell.axis, 0.0):
                    print("Unitcell has been refined.", flush=True)
        else:
            unitcell_rev = unitcell

        if np.array(supercell_matrix).size == 9:
            if self._verbose:
                print("Supercell matrix:", flush=True)
                print(supercell_matrix, flush=True)
            sup = supercell(unitcell_rev, supercell_matrix=supercell_matrix)
        elif np.array(supercell_matrix).size == 3:
            if self._verbose:
                print("Diagonal supercell:", supercell_matrix, flush=True)
            sup = supercell_diagonal(unitcell_rev, size=supercell_matrix)

        sup.supercell_matrix = np.linalg.inv(unitcell.axis) @ sup.axis
        sup.supercell_matrix = np.rint(sup.supercell_matrix).astype(int)
        self._lattice_supercell = self._lattice_unitcell.lattice_supercell(sup)
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
