"""Tests of classes for derivative structures."""

import copy
import shutil
from pathlib import Path

import numpy as np

from pyclupan.derivative.derivative_utils import DerivativesSet, load_derivatives_yaml

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/binary_fcc/"


files = [
    path_file + "/pyclupan_derivatives_3.yaml",
    path_file + "/pyclupan_derivatives_4.yaml",
]
ds_set_parsed = DerivativesSet([])
for f in files:
    ds = load_derivatives_yaml(f)
    ds_set_parsed.append(ds)


def test_derivatives_classes():
    """Test Derivatives and DerivativesSet."""
    ds_set = copy.deepcopy(ds_set_parsed)
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


def test_derivatives_classes_sample():
    """Test samples in Derivatives and DerivativesSet."""
    ds_set = copy.deepcopy(ds_set_parsed)
    ds_set.all()
    assert list(ds_set[5].sample) == [0, 1, 2]
    assert list(ds_set[7].sample) == [0, 1, 2]
    assert list(ds_set[9].sample) == [0, 1]
    ds_set.clear_samples()
    assert len(ds_set[5].sample) == 0

    ds_set.uniform(n_samples=10)
    assert sum([len(ds.sample) for ds in ds_set]) == 10

    ds_set.clear_samples()
    ds_set.uniform(n_samples=1000)
    assert sum([len(ds.sample) for ds in ds_set]) == 25

    ds_set.clear_samples()
    ds_set.random(n_samples=10)
    assert sum([len(ds.sample) for ds in ds_set]) == 10

    ds_set.clear_samples()
    ds_set.select((4, 2, 1))
    ds_set.select((4, 3, 2))
    assert list(ds_set[5].sample) == [1]
    assert list(ds_set[6].sample) == [2]
    assert sum([len(ds.sample) for ds in ds_set]) == 2

    ds_set.save(("Ag", "Au"), path="tmp")
    shutil.rmtree("tmp")

    strs = ds_set.get_sampled_structures(("Ag", "Au"))
    assert len(strs) == 2

    all_indices = ds_set.all_structure_indices
    assert len(all_indices) == 25
    assert all_indices[0] == (3, 0, 0)
    assert all_indices[-1] == (4, 6, 1)

    sampled_indices = ds_set.sampled_structure_indices
    assert len(sampled_indices) == 2
    assert sampled_indices[0] == (4, 2, 1)
    assert sampled_indices[1] == (4, 3, 2)
