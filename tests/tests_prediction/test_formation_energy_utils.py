"""Tests of formation_energy_utils."""

from pathlib import Path

import numpy as np

from pyclupan.prediction.formation_energy_utils import (
    append_formation_energies_endmembers,
    find_convex_hull,
)

cwd = Path(__file__).parent


def test_get_formation_energies():
    """Test get_formation_energies."""
    pass


def test_find_convex_hull():
    """Test find_convex_hull and append_formation_energies_endmembers."""

    comps = np.array(
        [
            [0.25, 0.5, 0.25],
            [0.5, 0.25, 0.25],
            [0.75, 0.125, 0.125],
            [0.5, 0.35, 0.15],
            [0.15, 0.35, 0.5],
        ]
    )
    energies = np.array([-0.35, -0.5, -0.2, -0.1, -0.2])
    ids = ["str-1", "str-2", "str-3", "str-4", "str-5"]
    comps, energies, ids = append_formation_energies_endmembers(comps, energies, ids)

    assert comps.shape == (8, 3)
    assert energies.shape[0] == 8
    assert len(ids) == 8
    np.testing.assert_allclose(comps[5:], np.eye(3), atol=1e-8)
    np.testing.assert_allclose(energies[5:], np.zeros(3), atol=1e-8)

    convex = find_convex_hull(comps, energies, ids)
    pred = convex[:, :4].astype(float)
    true = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.25, 0.25, -0.5],
            [0.25, 0.5, 0.25, -0.35],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(pred, true, atol=1e-8)
