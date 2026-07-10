"""API class for calculating features and energies."""

from typing import Optional, Union

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar
from pyclupan.derivative.derivative_utils import (
    DerivativesSet,
    load_derivatives_yaml,
    load_sample_attrs_yaml,
)

# from pyclupan.derivative.run_sample import run_sampling_derivatives
from pyclupan.features.cluster_functions import ClusterFunctions
from pyclupan.features.features_utils import (
    load_cluster_functions_hdf5,
    save_cluster_functions_hdf5,
)


class PyclupanCalcFeatures:
    """API class for calculating features."""

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
        self._cluster_functions = None
        self.clear_structures()

        np.set_printoptions(legacy="1.21")

    def clear_structures(self):
        """Clear structure data."""
        self._derivatives = DerivativesSet([])
        self._structures = None
        self._structure_ids = None
        self._cf.clear_structures()
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

    def _set_structure_ids(self):
        """Set structure IDs for derivative structures."""
        self._structure_ids = [
            "-".join([str(i) for i in ids])
            for ds in self._derivatives
            for ids in ds.structure_ids
        ]

    @property
    def derivatives(self):
        """Return derivative structures."""
        return self._derivatives

    @derivatives.setter
    def derivatives(self, derivatives: DerivativesSet):
        """Setter of derivative structures."""
        self._derivatives = derivatives
        self._cf.derivatives = self._derivatives
        self._set_structure_ids()

    def append_derivatives(self, derivatives: DerivativesSet):
        """Append derivative structures."""
        self._derivatives.append(derivatives)
        self._cf.derivatives = self._derivatives
        self._set_structure_ids()

    def append_sample_attrs_yaml(self, filename: str = "pyclupan_samples_attrs.yaml"):
        """Load and append pyclupan_samples_attrs.yaml file.

        Parameter
        ---------
        filename: Name of pyclupan_samples_attrs.yaml file.
            If other files are alreadly loaded, the file will be appended
            to the existing dataset.
        """
        derivs = load_sample_attrs_yaml(filename)
        self.append_derivatives(derivs)
        return self

    def append_derivatives_yaml(self, filename: str = "pyclupan_derivatives.yaml"):
        """Load and append pyclupan_derivatives.yaml file.

        Parameter
        ---------
        filename: Name of pyclupan_derivatives.yaml file.
            If other files are alreadly loaded, the file will be appended
            to the existing dataset.
        """
        derivs = load_derivatives_yaml(filename)
        self.append_derivatives(derivs)
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
        return self

    def eval_cluster_functions(self):
        """Evaluate cluster functions."""
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
            self._structure_ids,
            self._cf.n_atoms_array,
            filename=filename,
        )
        return self

    def load_features(self, filename: str = "pyclupan_features.hdf5"):
        """Load features in HDF5 format.

        Parameter
        ---------
        filename: HDF5 file for features.
        """
        res = load_cluster_functions_hdf5(filename=filename)
        self._cluster_functions, self._structure_ids, n_atoms_array = res
        self._cf.n_atoms_array = n_atoms_array
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
    def structure_indices(self):
        """Return structure indices."""
        return self._structure_ids
