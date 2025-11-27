"""Tests of sampling derivatives"""

from pathlib import Path

from pyclupan.derivative.run_sample import run_sampling_derivatives

cwd = Path(__file__).parent


def test_run_sampling_derivatives():
    """Test run_sampling_derivatives."""
    files = [
        str(cwd) + "/pyclupan_derivatives_3.yaml",
        str(cwd) + "/pyclupan_derivatives_4.yaml",
    ]

    element_strings = ("Ag", "Au")
    ds_set = run_sampling_derivatives(
        files=files, element_strings=element_strings, method="all", save_poscars=False
    )
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

    keys = [(3, 0, 0), (4, 1, 1), (4, 2, 2)]
    ds_set = run_sampling_derivatives(
        files=files, element_strings=element_strings, keys=keys, save_poscars=False
    )
    assert list(ds_set[0].sample) == [0]
    assert ds_set[1].sample is None
    assert ds_set[2].sample is None
    assert ds_set[3].sample is None
    assert list(ds_set[4].sample) == [1]
    assert list(ds_set[5].sample) == [2]
    assert ds_set[6].sample is None
    assert ds_set[7].sample is None

    assert list(ds_set[0].sample_ids) == [(3, 0, 0)]
    assert list(ds_set[4].sample_ids) == [(4, 1, 1)]
    assert list(ds_set[5].sample_ids) == [(4, 2, 2)]

    assert list(ds_set[0].sampled_active_labelings[0]) == [1, 0, 0]
    assert list(ds_set[0].sampled_complete_labelings[0]) == [1, 0, 0]
