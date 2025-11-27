"""Tests of Composition class."""

from pathlib import Path

import numpy as np

from pyclupan.core.composition import Composition

cwd = Path(__file__).parent


def test_composition():
    """Test Composition class."""
    n_atoms_end1 = [1, 1]
    n_atoms_end2 = [1, 2]
    chemical_comps_end_members = np.array([n_atoms_end1, n_atoms_end2])
    comp = Composition(chemical_comps_end_members=chemical_comps_end_members)

    val, _ = comp.get_composition([3, 4])
    np.testing.assert_allclose(val, [2 / 3, 1 / 3], atol=1e-8)
    val, _ = comp.get_compositions([[3, 4], [2, 3], [4, 5]])
    np.testing.assert_allclose(
        val, [[2 / 3, 1 / 3], [1 / 2, 1 / 2], [3 / 4, 1 / 4]], atol=1e-8
    )

    comp.energies_end_members = [0.1, 0.2]
    e_f = comp.compute_formation_energy(0.3, [3, 4])
    np.testing.assert_allclose(e_f, -1 / 30, atol=1e-8)

    energies = [0.3, -0.4, 0.5]
    n_atoms_array = [[3, 4], [2, 3], [4, 5]]
    e_f = comp.compute_formation_energies(energies, n_atoms_array)
    np.testing.assert_allclose(e_f, [-1 / 30, -0.35, 0.0], atol=1e-8)
