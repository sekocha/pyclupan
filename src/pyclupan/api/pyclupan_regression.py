"""API Class for regression."""

from typing import Optional

from pyclupan.features.features_utils import load_cluster_functions_hdf5
from pyclupan.regression.regression_utils import check_data, load_energy_dat
from pyclupan.regression.solvers import solver_lasso, solver_ridge


class PyclupanRegression:
    """API Class for regression."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._features = None
        self._energies = None

        self._structure_ids_x = None
        self._structure_ids_y = None

        self._x = None
        self._y = None

        self._feature_ids = None
        self._coeffs = None
        self._intercept = None

    def load_features(self, features_hdf5: str = "pyclupan_features.hdf5"):
        """Load feature data used as predictors.

        Parameter
        ---------
        features_hdf5: HDF5 file containing features.
        """
        res = load_cluster_functions_hdf5(features_hdf5)
        self._features, self._structure_ids_x = res
        return self

    def load_energies(
        self,
        energy_yaml: Optional[str] = None,
        energy_dat: Optional[str] = None,
    ):
        """Load energy data used as observations.

        Parameters
        ----------
        energy_yaml: TODO
        energy_dat: Text file for structure IDs and energies.
        """
        if energy_yaml is not None:
            pass
        elif energy_dat is not None:
            self._structure_ids_y, self._energies = load_energy_dat(energy_dat)
        else:
            raise RuntimeError("Energy data not found.")
        return self

    def _check_data(self):
        """Check matching of data entries."""
        if self._features is None:
            raise RuntimeError("Feature data not found.")
        if self._energies is None:
            raise RuntimeError("Energy data not found.")

        self._x, self._y = check_data(
            self._features,
            self._energies,
            self._structure_ids_x,
            self._structure_ids_y,
        )
        return self

    def _check_regression_data(self):
        """Check if regression data exist."""
        if self._features is not None and self._energies is not None:
            self._check_data()

        if self._x is None:
            raise RuntimeError("Predictor data not found.")
        if self._y is None:
            raise RuntimeError("Observation data not found.")
        return self

    def run_ridge(self):
        """Run Ridge solver."""
        self._check_regression_data()
        coeffs, intercept = solver_ridge(
            x=self._x,
            y=self._y,
            alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
        )

    def run_lasso(self):
        """Run Lasso solver."""
        self._check_regression_data()
        solver_lasso(x=self._x, y=self._y)

    @property
    def coeffs(self):
        """Return coefficents."""
        return self._coeffs

    @property
    def intercept(self):
        """Return intercept."""
        return self._intercept

    @property
    def feature_ids(self):
        """Return feature IDs corresponding to coefficients."""
        return self._feature_ids
