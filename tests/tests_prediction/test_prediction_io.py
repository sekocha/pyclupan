"""Tests of prediction_io."""

from pathlib import Path

from pyclupan.prediction.prediction_io import (
    load_energies_hdf5,
    load_formation_energies_hdf5,
)

cwd = Path(__file__).parent


def test_load_energies():
    """Test load_energies and load_formation_energies."""
    energies, ids, _ = load_energies_hdf5(str(cwd) + "/pyclupan_energies.hdf5")
    assert energies.shape[0] == 27
    energies, comps, ids = load_formation_energies_hdf5(
        str(cwd) + "/pyclupan_formation_energies.hdf5"
    )
    assert energies.shape[0] == 135
    assert comps.shape == (135, 2)
