"""API class for calculating energies."""

from typing import Optional

import numpy as np

from pyclupan.api.pyclupan_calc_cf import PyclupanCalcFeatures
from pyclupan.core.model import CEmodel
from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar
from pyclupan.prediction.formation_energy_utils import (
    append_formation_energies_endmembers,
    find_convex_hull,
    get_formation_energies,
)
from pyclupan.prediction.prediction_io import (
    load_energies_hdf5,
    load_formation_energies_hdf5,
    save_convex_yaml,
    save_energies_hdf5,
    save_formation_energies_hdf5,
)
from pyclupan.regression.regression_utils import load_ecis


class PyclupanCalcModel(PyclupanCalcFeatures):
    """API class for calculating features and energies."""

    def __init__(
        self,
        clusters_yaml: str = "pyclupan_clusters.yaml",
        verbose: bool = False,
    ):
        """Init method.

        Parameter
        ---------
        clusters_yaml: Pyclupan result file for cluster attributes.
        """
        super().__init__(clusters_yaml=clusters_yaml, verbose=verbose)

        self._model = None

        self._energies = None
        self._formation_energies = None
        self._compositions = None
        self._convex = None

        np.set_printoptions(legacy="1.21")

    def load_ecis(self, filename: str = "pyclupan_ecis.yaml"):
        """Load effective cluster interactions.

        Parameter
        ---------
        filename: File of ECIs from regression.
        """
        self._model = load_ecis(filename)
        self._cf.spin_basis_clusters = self._model.nonzero_spin_basis(
            self._cf.spin_basis_clusters
        )
        return self

    @property
    def model(self):
        """Return CE model."""
        return self._model

    @model.setter
    def model(self, model: CEmodel):
        """Setter of CE model."""
        self._model = model
        self._cf.spin_basis_clusters = self._model.eliminate_zeros(
            self._cf.spin_basis_clusters
        )

    def eval_energies(self):
        """Evaluate energies."""
        if self._model is None:
            raise RuntimeError("CE model not found.")
        if self._cluster_functions is None:
            raise RuntimeError("Cluster functions not found.")

        self._energies = self._model.eval(self._cluster_functions)
        return self._energies

    def save_energies(self, filename: str = "pyclupan_energies.hdf5"):
        """Save energies.

        Parameter
        ---------
        filename: HDF5 file for outputing energies.
        """
        if self._energies is None:
            raise RuntimeError("Energies not found.")

        save_energies_hdf5(
            self._energies,
            self._structure_ids,
            self._cf.n_atoms_array,
            filename=filename,
        )
        return self

    def load_energies(self, filename: str = "pyclupan_energies.hdf5"):
        """Load energies.

        Parameter
        ---------
        filename: HDF5 file for energies.
        """
        res = load_energies_hdf5(filename=filename)
        self._energies, self._structure_ids, n_atoms_array = res
        self._cf.n_atoms_array = n_atoms_array
        return self

    def eval_formation_energies(
        self,
        structures_endmembers: Optional[list[PolymlpStructure]] = None,
        poscars_endmembers: Optional[list[PolymlpStructure]] = None,
        element_strings: Optional[tuple] = None,
        labelings_endmembers: Optional[np.ndarray] = None,
        supercell_matrices_endmembers: Optional[np.ndarray] = None,
    ):
        """Evaluate formation energies.

        Structures or labelings are needed to specity endmembers.
        Their formation energies are calculated using CE model.

        Parameters
        ----------
        structures_endmembers: Structures of endmembers.
        poscars_endmembers: POSCAR files of endmembers.
        element_strings: Element strings.
            The location index corresponds to label integer.
            For example, element_strings are ("Ag", "Au"),
            labels 0 and 1 indicate elements Ag and Au, respectively.
        labelings_endmembers: Labelings of endmembers.
        supercell_matrices_endmembers:
            Supercell matrice corresponding to each labeling.
        """
        if self._energies is None:
            raise RuntimeError("Energies not found.")
        if self._model is None:
            raise RuntimeError("CE model not found.")
        if self._cf.n_atoms_array is None:
            raise RuntimeError("Number of atoms not found.")

        if poscars_endmembers is not None:
            structures_endmembers = [Poscar(p).structure for p in poscars_endmembers]

        self._formation_energies, self._compositions = get_formation_energies(
            self._energies,
            self._model,
            self._cf,
            structures_endmembers=structures_endmembers,
            element_strings=element_strings,
            labelings_endmembers=labelings_endmembers,
            supercell_matrices_endmembers=supercell_matrices_endmembers,
            verbose=self._verbose,
        )
        print(self._formation_energies)
        print(self._compositions)
        res = append_formation_energies_endmembers(
            self._compositions,
            self._formation_energies,
            self._structure_ids,
        )
        self._convex = find_convex_hull(*res)
        return self._formation_energies, self._compositions, self._convex

    def save_formation_energies(
        self,
        filename: str = "pyclupan_formation_energies.hdf5",
    ):
        """Save formation energies.

        Parameter
        ---------
        filename: HDF5 file for outputing formation energies.
        """
        if self._formation_energies is None:
            raise RuntimeError("Formation energies not found.")

        save_formation_energies_hdf5(
            self._formation_energies,
            self._compositions,
            self._structure_ids,
            filename=filename,
        )
        return self

    def load_formation_energies(
        self,
        filename: str = "pyclupan_formation_energies.hdf5",
    ):
        """Load formation energies.

        Parameter
        ---------
        filename: HDF5 file for formation energies.
        """
        res = load_formation_energies_hdf5(filename=filename)
        self._formation_energies, self._compositions, self._structure_ids = res
        return self

    def save_convex_hull_yaml(self, filename: str = "pyclupan_convexhull.yaml"):
        """Save convex hull of formation energies.

        Parameter
        ---------
        filename: Yaml file for outputing convex hull of formation energies.
        """
        if self._convex is None:
            raise RuntimeError("Convex hull not found.")

        save_convex_yaml(self._convex, filename=filename)
        return self


#     def save_convex_hull_poscars_from_derivatives(self, element_strings: tuple):
#         """Save derivative structures on convex hull.
#
#         Parameter
#         ---------
#         element_strings: Element strings.
#             The location index corresponds to label integer.
#             For example, element_strings are ("Ag", "Au"),
#             labels 0 and 1 indicate elements Ag and Au, respectively.
#         """
#         if self._convex is None:
#             raise RuntimeError("Convex hull not found.")
#         if self._derivatives is None:
#             raise RuntimeError("Derivative structures not found.")
#
#         ids = [i for i in self._convex[:, -1] if "End" not in i]
#         keys = [i.split("-") for i in ids]
#         keys = [tuple([int(k2) for k2 in k]) for k in keys]
#         run_sampling_derivatives(
#             ds_set=self._derivatives,
#             element_strings=element_strings,
#             keys=keys,
#             save_poscars=True,
#         )
#         return self
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
#
#
