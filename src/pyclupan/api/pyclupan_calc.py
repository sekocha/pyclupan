"""API class for calculating features and energies."""

from typing import Optional, Union

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar
from pyclupan.derivative.derivative_utils import (
    DerivativesSet,
    load_derivatives_yaml,
    load_sample_attrs_yaml,
)
from pyclupan.features.features_utils import (
    load_cluster_functions_hdf5,
    save_cluster_functions_hdf5,
)
from pyclupan.features.run_correlation import ClusterFunctions
from pyclupan.prediction.formation_energy_utils import (
    append_formation_energies_endmembers,
    find_convex_hull,
    get_chemical_compositions,
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


class PyclupanCalc:
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
        self._verbose = verbose

        self._cf = ClusterFunctions(clusters_yaml=clusters_yaml, verbose=verbose)
        self._spin_clusters = None
        self._cluster_functions = None

        self._structures = None
        self._derivative_set = DerivativesSet([])
        self._structure_ids = None
        self._n_atoms_array = None

        self._model = None

        self._energies = None
        self._formation_energies = None
        self._compositions = None
        self._convex = None

        np.set_printoptions(legacy="1.21")

    def clear_structures(self):
        """Clear structures and labelings to be evaluated."""

        return self

    def load_ecis(self, filename: str = "pyclupan_ecis.yaml"):
        """Load effective cluster interactions.

        Parameter
        ---------
        filename: File of ECIs from regression.
        """
        self._model = load_ecis(filename)
        self._cf.spin_basis_clusters = [
            self._spin_clusters[i] for i in self._model.cluster_ids
        ]
        return self

    def load_poscars(
        self,
        poscars: Union[str, list, tuple, np.ndarray],
        element_strings: Optional[tuple] = None,
    ):
        """Load POSCAR files used for evaluating features.

        Parameters
        ----------
        poscars: POSCAR file or List of POSCAR files.
        element_strings: Element strings.
            The location index corresponds to label integer.
            For example, element_strings are ("Ag", "Au"),
            labels 0 and 1 indicate elements Ag and Au, respectively.
        """
        if isinstance(poscars, str):
            self.structures = [Poscar(poscars).structure]
        elif isinstance(poscars, (list, tuple, np.ndarray)):
            self.structures = [Poscar(p).structure for p in poscars]
        else:
            raise RuntimeError(
                "Parameter in load_poscars must be string or array-like."
            )

        self._structure_ids = poscars
        if element_strings is not None:
            self.element_strings = element_strings

        return self

    def load_sample_attrs_yaml(self, filename: str = "pyclupan_samples_attrs.yaml"):
        """Load pyclupan_samples_attrs.yaml file.

        Parameter
        ---------
        filename: Name of pyclupan_samples_attrs.yaml file.
            If other files are alreadly loaded, the file will be appended
            to the existing dataset.
        """
        self._derivative_set.append(load_sample_attrs_yaml(filename))
        self._cf.derivatives = self._derivative_set
        return self

    def load_derivatives_yaml(self, filename: str = "pyclupan_derivatives.yaml"):
        """Load pyclupan_derivatives.yaml file.

        Parameter
        ---------
        filename: Name of pyclupan_derivatives.yaml file.
            If other files are alreadly loaded, the file will be appended
            to the existing dataset.
        """
        self._derivative_set.append(load_derivatives_yaml(filename))
        self._cf.derivatives = self._derivative_set
        return self

    def set_labelings(
        self,
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
        active_labelings: np.ndarray,
    ):
        """Set labelings.

        Parameters
        ----------
        unitcell: Unitcell.
        supercell_matrix: Supercell matrix.
        active_labelings: Labelings only for active sites.
        """
        self._cf.set_labelings(unitcell, supercell_matrix, active_labelings)
        self._structure_ids = [str(i).zfill(5) for i, l in enumerate(active_labelings)]
        # TODO: Use complete labelings
        self._n_atoms_array = get_chemical_compositions(labelings=active_labelings)
        return self

    def _eval_cluster_functions_from_derivatives(self):
        """Evaluate cluster functions from derivative structure set."""
        self._structure_ids = []
        self._n_atoms_array = []
        for d in self._derivative_set:
            self._structure_ids.extend(d.structure_ids)
            self._n_atoms_array.extend(
                get_chemical_compositions(labelings=d.get_complete_labelings())
            )
        return self

    def eval_cluster_functions(self):
        """Evaluate cluster functions."""
        if len(self._derivative_set) > 0:
            self._eval_cluster_functions_from_derivatives()

        self._cluster_functions = self._cf.eval()
        return self._cluster_functions

    def save_features(self, filename: str = "pyclupan_features.hdf5"):
        """Save features in HDF5 format.

        Parameter
        ---------
        filename: HDF5 file for outputing features.
        """
        if self._cluster_functions is None:
            raise RuntimeError("Cluster functions not found.")

        save_cluster_functions_hdf5(
            self._cluster_functions,
            ids=self._structure_ids,
            filename=filename,
        )
        return self

    def load_features(self, filename: str = "pyclupan_features.hdf5"):
        """Load features in HDF5 format.

        Parameter
        ---------
        filename: HDF5 file for features.
        """

        self._cluster_functions, self._structure_ids = load_cluster_functions_hdf5(
            filename=filename,
        )
        return self

    def eval_energies(self):
        """Evaluate energies."""
        if self._model is None:
            raise RuntimeError("CE model not found.")

        if self._cluster_functions is None:
            self.eval_cluster_functions()

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

        save_energies_hdf5(self._energies, self._structure_ids, filename=filename)
        return self

    def load_energies(self, filename: str = "pyclupan_energies.hdf5"):
        """Load energies.

        Parameter
        ---------
        filename: HDF5 file for energies.
        """
        self._energies, self._structure_ids = load_energies_hdf5(filename=filename)
        return self

    def eval_formation_energies(
        self,
        structures: Optional[list[PolymlpStructure]] = None,
        element_strings: Optional[tuple] = None,
        labelings: Optional[np.ndarray] = None,
        supercell_matrices: Optional[np.ndarray] = None,
    ):
        """Evaluate formation energies.

        Parameters
        ----------
        TODO: Add parameters.
        """
        if self._energies is None:
            raise RuntimeError("Energies not found.")
        if self._model is None:
            raise RuntimeError("CE model not found.")
        if self._n_atoms_array is None:
            raise RuntimeError("Number of atoms not found.")

        self._formation_energies, self._compositions = get_formation_energies(
            self._energies,
            self._n_atoms_array,
            self._model,
            self._lattice,
            self._clusters,
            self._spin_clusters,
            structures=structures,
            element_strings=element_strings,
            labelings=labelings,
            supercell_matrices=supercell_matrices,
            verbose=self._verbose,
        )
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

    def save_convex_hull_yaml(self, filename: str = "pyclupan_convexhull.yaml"):
        """Save convex hull of formation energies.

        Parameter
        ---------
        filename: Yaml file for outputing convex hull of formation energies.
        """
        if self._convex is None:
            raise RuntimeError("Convex hull not found.")

        save_convex_yaml(self._convex, filename=filename)
        # TODO: POSCAR files.
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

    @property
    def structures(self):
        """Return structures."""
        return self._structures

    @structures.setter
    def structures(self, strs: list[PolymlpStructure]):
        """Set structures to be calculated.

        Parameter
        ---------
        str: List of structures to be calculated.
        """
        # TODO: n_atoms_array
        # TODO: Use complete labelings
        # self._n_atoms_array = get_chemical_compositions(labelings=active_labelings)

        self._structures = self._cf.structures = strs
        self._structure_ids = ["str-" + str(i) for i in range(len(strs))]

    @property
    def element_strings(self):
        """Return element strings."""
        return self._cf.element_strings

    @element_strings.setter
    def element_strings(self, element_strings: tuple):
        """Set element strings.

        Parameter
        ---------
        element_strings: Element strings.
            The location index corresponds to label integer.
            For example, element_strings are ("Ag", "Au"),
            labels 0 and 1 indicate elements Ag and Au, respectively.
        """
        self._cf.element_strings = element_strings

    @property
    def cluster_functions(self):
        """Return cluster functions."""
        return self._cluster_functions

    @property
    def energies(self):
        """Return energies (per unitcell)."""
        return self._energies

    @property
    def formation_energies(self):
        """Return formation energies (per unitcell)."""
        return self._formation_energies

    @property
    def convexhull(self):
        """Return convex hull of formation energies."""
        return self._convex
