"""API class for calculating features and energies."""

from typing import Optional, Union

import numpy as np

from pyclupan.cluster.cluster_io import load_clusters_yaml
from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar
from pyclupan.derivative.derivative_utils import (
    DerivativesSet,
    load_derivatives_yaml,
    load_sample_attrs_yaml,
)
from pyclupan.features.features_utils import save_cluster_functions_hdf5
from pyclupan.features.run_correlation import (
    run_correlation,
    run_correlation_from_structures,
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

        cluster_res = load_clusters_yaml(clusters_yaml)
        self._lattice, self._clusters, _, self._spin_clusters = cluster_res
        self.clear_structures()

        self._cluster_functions = None
        self._structure_ids = None

        self._model = None

    def clear_structures(self):
        """Clear structures and labelings to be evaluated."""
        self._structures = None
        self._derivative_set = DerivativesSet([])

        self._unitcell = None
        self._supercell_matrix = None
        self._labelings = None
        return self

    def load_ecis(self, filename: str = "pyclupan_ecis.yaml"):
        """Load effective cluster interactions."""
        self._model = load_ecis(filename)
        self._spin_clusters = [self._spin_clusters[i] for i in self._model.cluster_ids]
        for cl in self._spin_clusters:
            print(cl)

        return self

    def load_poscars(
        self,
        poscars: Union[str, list, tuple, np.ndarray],
        element_strings: Optional[tuple] = None,
    ):
        """Load POSCAR files used for evaluating features.

        Parameter
        ---------
        poscars: POSCAR file or List of POSCAR files.
        element_strings: Element strings.
            The location index corresponds to label integer.
            For example, element_strings are ("Ag", "Au"),
            labels 0 and 1 indicate elements Ag and Au, respectively.
        """
        if isinstance(poscars, str):
            self._structures = [Poscar(poscars).structure]
        elif isinstance(poscars, (list, tuple, np.ndarray)):
            self._structures = [Poscar(p).structure for p in poscars]
        else:
            raise RuntimeError(
                "Parameter in load_poscars must be string or array-like."
            )

        self._structure_ids = poscars
        if element_strings is not None:
            self._element_strings = element_strings

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
        return self

    def set_labelings(
        self,
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
        labelings: np.ndarray,
    ):
        """Set labelings.

        Parameter
        ---------
        unitcell: Unitcell.
        supercell_matrix: Supercell matrix.
        labelings: Labelings for active sites.
        """
        self._unitcell = unitcell
        self._supercell_matrix = supercell_matrix
        self._labelings = labelings
        return self

    def _eval_cluster_functions_from_labelings(self):
        """Evaluate cluster functions from labelings."""
        if self._unitcell is None:
            raise RuntimeError("Unitcell is required.")
        if self._supercell_matrix is None:
            raise RuntimeError("Supercell matrix is required.")

        self._cluster_functions = run_correlation(
            unitcell=self._unitcell,
            supercell_matrix=self._supercell_matrix,
            labelings=self._labelings,
            lattice=self._lattice,
            clusters=self._clusters,
            spin_basis_clusters=self._spin_clusters,
        )
        self._structure_ids = [str(i).zfill(5) for i, l in enumerate(self._labelings)]
        return self._cluster_functions

    def _eval_cluster_functions_from_derivatives(self):
        """Evaluate cluster functions from derivative structure set."""
        self._cluster_functions = []
        self._structure_ids = []
        for d in self._derivative_set:
            cf = run_correlation(
                unitcell=d.unitcell,
                supercell_matrix=d.supercell_matrix,
                labelings=d.active_labelings,
                lattice=self._lattice,
                clusters=self._clusters,
                spin_basis_clusters=self._spin_clusters,
            )
            self._cluster_functions.extend(cf)
            self._structure_ids.extend(d.structure_ids)
        self._cluster_functions = np.array(self._cluster_functions)
        return self._cluster_functions

    def _eval_cluster_functions_from_structures(self):
        """Evaluate cluster functions from structures."""
        if self._structures is None:
            raise RuntimeError("Structures are required.")
        if self._element_strings is None:
            raise RuntimeError("Labels for element strings are required.")

        self._cluster_functions = run_correlation_from_structures(
            structures=self._structures,
            element_strings=self._element_strings,
            lattice=self._lattice,
            clusters=self._clusters,
            spin_basis_clusters=self._spin_clusters,
        )
        return self._cluster_functions

    def eval_cluster_functions(self):
        """Evaluate cluster functions."""
        if self._labelings is not None:
            if self._verbose:
                print("Evaluating cluster functions from labelings.", flush=True)
            return self._eval_cluster_functions_from_labelings()
        elif len(self._derivative_set) > 0:
            if self._verbose:
                print("Evaluating cluster functions from derivatives.", flush=True)
            return self._eval_cluster_functions_from_derivatives()

        if self._verbose:
            print("Evaluating cluster functions from structures.", flush=True)
        return self._eval_cluster_functions_from_structures()

    def save(self, filename: str = "pyclupan_features.hdf5"):
        """Save features in HDF5 format."""
        if self._cluster_functions is None:
            raise RuntimeError("Cluster functions not found.")

        save_cluster_functions_hdf5(
            self._cluster_functions,
            ids=self._structure_ids,
            filename=filename,
        )

    @property
    def element_strings(self):
        """Return element strings."""
        return self._element_strings

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
        self._element_strings = element_strings

    @property
    def cluster_functions(self):
        """Return cluster functions."""
        return self._cluster_functions
