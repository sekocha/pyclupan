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
        self._models = None
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

    def run_ridge(self, alphas: tuple = np.logspace(-7, -3, 20)):
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

    def run_lasso(self, alphas: tuple = np.logspace(-7, -3, 20)):
        """Run Lasso solver.

        Parameter
        ---------
        alphas: Regularization parameters.
        """
        if self._verbose:
            print("Use Lasso solver.", flush=True)

        self._check_regression_data()
        self._models = solver_lasso(
            x=self._x,
            y=self._y,
            alphas=alphas,
            verbose=self._verbose,
        )
        return self

    @property
    def best_model(self):
        """Return best CE model."""
        if self._model is not None:
            return self._model
        return min(self._models, key=lambda s: s.cv_score)

    def save_predictions(self, filename: str = "pyclupan_prediction.dat"):
        """Save predicted values for dataset."""
        if self._model is None and self._models is None:
            raise RuntimeError("CE model not found.")

        model = self.best_model
        pred = model.eval(self._x)
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
        if self._model is None and self._models is None:
            raise RuntimeError("CE model not found.")

        if self._model is not None:
            save_ecis(self._model, filename=filename)
        elif self._models is not None:
            for i, model in enumerate(self._models):
                idx = str(i + 1).zfill(3)
                name = filename.replace(".yaml", "") + "_" + idx + ".yaml"
                save_ecis(model, filename=name)
        return self

    @property
    def x(self):
        """Return x."""
        return self._x

    @x.setter
    def x(self, x: np.ndarray):
        """Setter of x."""
        self._x = x

    @property
    def y(self):
        """Return y."""
        return self._y

    @y.setter
    def y(self, y: np.ndarray):
        """Setter of y."""
        self._y = y

    @property
    def structure_ids(self):
        """Return structure_ids."""
        return self._structure_ids

    @structure_ids.setter
    def structure_ids(self, structure_ids: np.ndarray):
        """Setter of structure_ids."""
        self._structure_ids = structure_ids

    @property
    def model(self):
        """Return CE model."""
        return self._model

    @property
    def models(self):
        """Return CE models."""
        return self._models

    @property
    def coeffs(self):
        """Return coefficents."""
        return self._model.coeffs

    @property
    def intercept(self):
        """Return intercept."""
        return self._model.intercept
