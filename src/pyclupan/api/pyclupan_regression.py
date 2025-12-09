"""API Class for regression."""

from typing import Optional

import numpy as np

from pyclupan.features.features_utils import load_cluster_functions_hdf5
from pyclupan.regression.regression_utils import check_data, load_energy_dat, save_ecis
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
        self._structure_ids = None

        self._x = None
        self._y = None

        self._model = None
        np.set_printoptions(legacy="1.21")

    def load_features(self, features_hdf5: str = "pyclupan_features.hdf5"):
        """Load feature data used as predictors.

        Parameter
        ---------
        features_hdf5: HDF5 file containing features.
        """
        res = load_cluster_functions_hdf5(features_hdf5)
        self._features, self._structure_ids_x, _ = res
        if self._verbose:
            print("Load features:", self._features.shape, flush=True)
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

        if self._verbose:
            print("Load energies:", self._energies.shape, flush=True)
        return self

    def _check_data(self):
        """Check matching of data entries."""
        if self._features is None:
            raise RuntimeError("Feature data not found.")
        if self._energies is None:
            raise RuntimeError("Energy data not found.")

        self._x, self._y, self._structure_ids = check_data(
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

    def run_ridge(self, alphas: tuple = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1)):
        """Run Ridge solver.

        Parameter
        ---------
        alphas: Regularization parameters.
        """
        if self._verbose:
            print("Use Ridge solver.", flush=True)

        self._check_regression_data()
        self._model = solver_ridge(
            x=self._x,
            y=self._y,
            alphas=alphas,
            verbose=self._verbose,
        )
        return self

    def run_lasso(self, alphas: tuple = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1)):
        """Run Lasso solver.

        Parameter
        ---------
        alphas: Regularization parameters.
        """
        if self._verbose:
            print("Use Lasso solver.", flush=True)

        self._check_regression_data()
        self._model = solver_lasso(
            x=self._x,
            y=self._y,
            alphas=alphas,
            verbose=self._verbose,
        )
        return self

    def save_predictions(self, filename: str = "pyclupan_prediction.dat"):
        """Save predicted values for dataset."""
        if self._model is None:
            raise RuntimeError("CE model not found.")

        pred = self._model.eval(self._x)
        with open(filename, "w") as f:
            print("# DFT (eV/cell), CE (eV/cell), Error (meV/cell)", file=f)
            for y1, y2, idx in zip(self._y, pred, self._structure_ids):
                diff = (y2 - y1) * 1000
                print(idx, np.round(y1, 8), np.round(y2, 8), np.round(diff, 5), file=f)

    def save(self, filename: str = "pyclupan_ecis.yaml"):
        """Save coefficients and intercept.

        Parameter
        ---------
        filename: Filename to save ECIs.
        """
        save_ecis(self.coeffs, self.intercept, filename=filename)
        return self

    @property
    def model(self):
        """Return CE model."""
        return self._model

    @property
    def coeffs(self):
        """Return coefficents."""
        return self._model.coeffs

    @property
    def intercept(self):
        """Return intercept."""
        return self._model.intercept
