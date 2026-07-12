"""Tests of classes for derivative structures."""

from pathlib import Path

from pyclupan.derivative.derivative_utils import (
    load_derivatives_yaml,
    load_sample_attrs_yaml,
)

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/binary_fcc/"


def test_derivatives_files():
    """Test load and save files."""
    ds_set = load_derivatives_yaml(path_file + "/pyclupan_derivatives_3.yaml")
    assert len(ds_set) == 3
    ds_set = load_sample_attrs_yaml(path_file + "/pyclupan_sample_attrs.yaml")
    assert len(ds_set) == 2
