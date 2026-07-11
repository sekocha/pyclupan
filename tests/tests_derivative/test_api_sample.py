"""Tests of API for sampling derivatives"""

from pathlib import Path

from pyclupan.api.pyclupan_derivatives import PyclupanDerivatives

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/binary_fcc/"

files = [
    path_file + "/pyclupan_derivatives_3.yaml",
    path_file + "/pyclupan_derivatives_4.yaml",
]
element_strings = ("Ag", "Au")


def test_sampling_derivatives_all():
    """Test run_sampling_derivatives."""
    pyclupan = PyclupanDerivatives()
    pyclupan.load_derivatives(files)
    pyclupan.sample_derivatives(method="all", save_poscars=False)
    ds_set = pyclupan.derivative_structures

    assert list(ds_set[0].sample) == [0, 1]
    assert list(ds_set[1].sample) == [0, 1]
    assert list(ds_set[2].sample) == [0, 1]
    assert list(ds_set[3].sample) == [0, 1, 2]
    assert list(ds_set[4].sample) == [0, 1, 2]
    assert list(ds_set[5].sample) == [0, 1, 2]
    assert list(ds_set[6].sample) == [0, 1, 2]
    assert list(ds_set[7].sample) == [0, 1, 2]
    assert list(ds_set[8].sample) == [0, 1]
    assert list(ds_set[9].sample) == [0, 1]

    assert list(ds_set[2].sample_ids) == [(3, 2, 0), (3, 2, 1)]
    assert list(ds_set[9].sample_ids) == [(4, 6, 0), (4, 6, 1)]

    strs = pyclupan.get_sampled_structures(element_strings)
    assert len(strs) == 25


def test_sampling_derivatives_uniform():
    """Test run_sampling_derivatives."""
    pyclupan = PyclupanDerivatives()
    pyclupan.load_derivatives(files)
    pyclupan.sample_derivatives(method="uniform", n_samples=5, save_poscars=False)
    strs = pyclupan.get_sampled_structures(element_strings)
    assert len(strs) == 5


def test_sampling_derivatives_random():
    """Test run_sampling_derivatives."""
    pyclupan = PyclupanDerivatives()
    pyclupan.load_derivatives(files)
    pyclupan.sample_derivatives(method="random", n_samples=5, save_poscars=False)
    strs = pyclupan.get_sampled_structures(element_strings)
    assert len(strs) == 5


def test_sampling_derivatives_from_keys():
    """Test sampling_derivatives."""
    pyclupan = PyclupanDerivatives()
    pyclupan.load_derivatives(files)

    keys = [(3, 0, 0), (4, 1, 1), (4, 2, 2)]
    pyclupan.sample_derivatives_from_keys(
        keys, element_strings=element_strings, save_poscars=False
    )
    ds_set = pyclupan.derivative_structures

    assert list(ds_set[0].sample) == [0]
    assert len(ds_set[1].sample) == 0
    assert len(ds_set[2].sample) == 0
    assert len(ds_set[3].sample) == 0
    assert list(ds_set[4].sample) == [1]
    assert list(ds_set[5].sample) == [2]
    assert len(ds_set[6].sample) == 0
    assert len(ds_set[7].sample) == 0

    assert list(ds_set[0].sample_ids) == [(3, 0, 0)]
    assert list(ds_set[4].sample_ids) == [(4, 1, 1)]
    assert list(ds_set[5].sample_ids) == [(4, 2, 2)]

    assert list(ds_set[0].sampled_active_labelings[0]) == [1, 0, 0]
    assert list(ds_set[0].sampled_complete_labelings[0]) == [1, 0, 0]

    strs = pyclupan.get_sampled_structures(element_strings)
    assert len(strs) == 3
