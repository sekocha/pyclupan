"""API class for calculating features."""

from typing import Optional, Union

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar
from pyclupan.derivative.sample_utils import (
    DerivativesSet,
    load_derivative_yaml,
    load_sample_attrs_yaml,
)
from pyclupan.features.run_correlation import (
    run_correlation,
    run_correlation_from_structures,
)


class PyclupanFeatures:
    """API class for calculating features."""

    def __init__(self, cluster_yaml: str = "pyclupan_cluster.yaml"):
        """Init method."""
        self._cluster_yaml = cluster_yaml
        self._structures = None
        self._derivative_set = DerivativesSet([])

    def load_poscars(
        self,
        poscars: Union[str, list, tuple, np.ndarray],
        element_string_labels: Optional[dict] = None,
    ):
        """Load POSCAR files used for evaluating features.

        Parameter
        ---------
        poscars: POSCAR file or List of POSCAR files.
        element_string_labels: Dictionary of element string and label integer.
            (e. g.) element_labels = {"Ag": 0, "Au": 1}
        """
        if isinstance(poscars, str):
            self._structures = [Poscar(poscars).structure]
        elif isinstance(poscars, (list, tuple, np.ndarray)):
            self._structures = [Poscar(p).structure for p in poscars]
        else:
            raise RuntimeError(
                "Parameter in load_poscars must be string or array-like."
            )

        if element_string_labels is not None:
            self._element_labels = element_string_labels

        return self

    def load_sample_attrs_yaml(self, filename: str = "pyclupan_samples_attrs.yaml"):
        """Load pyclupan_samples_attrs.yaml file.

        Parameter
        ---------
        filename: Name of pyclupan_samples_attrs.yaml file.
        """
        self._derivative_set.append(load_sample_attrs_yaml(filename))
        return self

    def load_derivative_yaml(self, filename: str = "pyclupan_derivatives.yaml"):
        """Load pyclupan_derivatives.yaml file.

        Parameter
        ---------
        filename: Name of pyclupan_derivatives.yaml file.
        """
        self._derivative_set.append(load_derivative_yaml(filename))
        return self

    def eval_cluster_functions(
        self,
        unitcell: Optional[PolymlpStructure] = None,
        supercell_matrix: Optional[np.ndarray] = None,
        labelings: Optional[np.ndarray] = None,
    ):
        """Evaluate cluster functions from structures."""
        if labelings is not None:
            if unitcell is None:
                raise RuntimeError("Unitcell is required.")
            if supercell_matrix is None:
                raise RuntimeError("Supercell matrix is required.")

            cluster_functions = run_correlation(
                unitcell=unitcell,
                supercell_matrix=supercell_matrix,
                labelings=labelings,
                cluster_yaml=self._cluster_yaml,
            )
            return cluster_functions

        elif len(self._derivative_set) > 0:
            cluster_functions = []
            for d in self._derivative_set:
                cf = run_correlation(
                    unitcell=d.unitcell,
                    supercell_matrix=d.supercell_matrix,
                    labelings=d.active_labelings,
                    cluster_yaml=self._cluster_yaml,
                )
                cluster_functions.extend(cf)
            cluster_functions = np.array(cluster_functions)
            return cluster_functions

        if self._structures is None:
            raise RuntimeError("Structures are required.")
        if self._element_labels is None:
            raise RuntimeError("Labels for element strings are required.")

        cluster_functions = run_correlation_from_structures(
            structures=self._structures,
            element_labels=self._element_labels,
            cluster_yaml=self._cluster_yaml,
        )
        return cluster_functions

    @property
    def element_string_labels(self):
        """Return labels for element strings."""
        return self._element_labels

    @element_string_labels.setter
    def element_string_labels(self, element_labels: dict):
        """Set labels for element strings.

        Parameter
        ---------
        element_string_labels: Dictionary of element string and label integer.
            (e. g.) element_labels = {"Ag": 0, "Au": 1}
        """
        self._element_labels = element_labels
