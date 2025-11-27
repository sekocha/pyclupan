"""Tests of regression."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_regression import PyclupanRegression

cwd = Path(__file__).parent


def test_ridge():
    """Test ridge."""
    pyclupan = PyclupanRegression(verbose=False)
    features_hdf5 = str(cwd) + "/pyclupan_features.hdf5"
    pyclupan.load_features(features_hdf5)
    pyclupan.load_energies(energy_dat=str(cwd) + "/energy.dat")
    pyclupan.run_ridge(alphas=(1e-5, 1e-4, 1e-3))
    model = pyclupan.model

    np.testing.assert_allclose(model.intercept, -3.0144481263541745, atol=1e-6)
    np.testing.assert_allclose(model.rmse, 0.00023249746401957284, atol=1e-6)


def test_lasso():
    """Test Lasso."""
    pyclupan = PyclupanRegression(verbose=False)
    features_hdf5 = str(cwd) + "/pyclupan_features.hdf5"
    pyclupan.load_features(features_hdf5)
    pyclupan.load_energies(energy_dat=str(cwd) + "/energy.dat")
    pyclupan.run_lasso(alphas=(1e-5, 1e-4, 1e-3))
    model = pyclupan.model

    nonzero = np.where(np.abs(model.coeffs) > 1e-12)[0]
    assert len(nonzero) == 24
    np.testing.assert_allclose(model.intercept, -3.0153654473861815, atol=1e-6)
    np.testing.assert_allclose(model.rmse, 0.0003688802553296099, atol=1e-6)
