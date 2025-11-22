"""Tests of regression."""

from pathlib import Path

import numpy as np

from pyclupan.regression.regression_utils import (
    find_matching_ids,
    load_ecis,
    load_energy_dat,
)

cwd = Path(__file__).parent


def test_load_energy_dat():
    """Test load_energy_dat."""
    ids, energies = load_energy_dat(str(cwd) + "/energy.dat")
    ids_true = ["2-0-0", "2-1-0", "3-0-0", "3-0-1", "3-1-0"]
    energies_true = [-3.01327349, -3.03075283, -2.91765828, -3.08803944, -2.93614183]
    np.testing.assert_equal(ids[:5], ids_true)
    np.testing.assert_allclose(energies[:5], energies_true, atol=1e-6)


def test_load_ecis():
    """Test load_ecis."""
    model = load_ecis(str(cwd) + "/pyclupan_ecis.yaml")
    assert len(model.cluster_ids) == 24
    coeffs_true = [
        2.50741407e-01,
        4.11624568e-02,
        4.65679200e-04,
        3.89110990e-03,
        8.05275479e-05,
    ]
    np.testing.assert_allclose(model.intercept, -3.0153654473861815, atol=1e-6)
    np.testing.assert_allclose(model.coeffs[:5], coeffs_true, atol=1e-6)

    cfs = np.ones((3, 24)) * 0.01
    energies = model.eval(cfs)
    np.testing.assert_allclose(energies, -3.01238762, atol=1e-6)


def test_find_matching_ids():
    """Test find_matching_ids."""
    ids = ["3-1-0", "2-0-0", "3-0-0", "3-0-1", "2-1-0"]
    ids_ref = ["2-0-0", "2-1-0", "3-0-0", "3-0-1", "3-1-0"]
    order = find_matching_ids(ids, ids_ref)
    assert list(order) == [1, 4, 2, 3, 0]
    assert ids_ref == [ids[i] for i in order]
