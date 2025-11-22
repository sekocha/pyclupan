"""Tests of classes for derivative structures."""

from pathlib import Path

import numpy as np

from pyclupan.derivative.derivative_utils import (
    DerivativesSet,
    get_complete_labelings,
    load_derivatives_yaml,
)

cwd = Path(__file__).parent


def test_get_complete_labelings():
    """Test get_complete_labelings."""
    active_labelings = np.array([[0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 2, 1]])
    inactive_labeling = [3, 3, 4, 4, 4]
    active_sites = [0, 1, 2, 5, 6, 7]
    inactive_sites = [3, 4, 8, 9, 10]
    labelings = get_complete_labelings(
        active_labelings, inactive_labeling, active_sites, inactive_sites
    )
    labelings_true = np.array(
        [[0, 0, 1, 3, 3, 1, 2, 2, 4, 4, 4], [0, 1, 2, 3, 3, 0, 2, 1, 4, 4, 4]]
    )
    np.testing.assert_equal(labelings, labelings_true)


def test_derivatives_classes():
    """Test Derivatives and DerivativesSet."""
    ds_set = load_derivatives_yaml(str(cwd) + "/pyclupan_derivatives_3.yaml")
    assert len(ds_set) == 3

    files = [
        str(cwd) + "/pyclupan_derivatives_3.yaml",
        str(cwd) + "/pyclupan_derivatives_4.yaml",
    ]
    ds_set = DerivativesSet([])
    for f in files:
        ds = load_derivatives_yaml(f)
        ds_set.append(ds)
    assert len(ds_set) == 10

    for d in ds_set[:3]:
        np.testing.assert_equal(d.active_sites, [0, 1, 2])
        np.testing.assert_equal(d.inactive_sites, [])
        assert d.supercell_size == 3
        assert d.n_labelings == 2
    for d in ds_set[3:]:
        np.testing.assert_equal(d.active_sites, [0, 1, 2, 3])
        np.testing.assert_equal(d.inactive_sites, [])
        assert d.supercell_size == 4
    np.testing.assert_equal(ds_set[0].active_sites, [0, 1, 2])
    np.testing.assert_equal(ds_set.n_labelings, [2, 2, 2, 3, 3, 3, 3, 3, 2, 2])

    axis_true = np.array([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
    np.testing.assert_allclose(ds_set[0].unitcell.axis, axis_true)
    np.testing.assert_allclose(ds_set.unitcell.axis, axis_true)
    axis_true = np.array([[-2.0, 0.0, -4.0], [2.0, -2.0, -4.0], [-0.0, -2.0, 4.0]])
    np.testing.assert_allclose(ds_set[0].supercell.axis, axis_true)

    ds_set.all()
    assert list(ds_set[5].sample) == [0, 1, 2]
    assert list(ds_set[7].sample) == [0, 1, 2]
    assert list(ds_set[9].sample) == [0, 1]
