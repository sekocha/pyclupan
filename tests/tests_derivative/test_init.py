"""Tests of initializing parameters for enumerating derivative structures"""

from pathlib import Path

import numpy as np

from pyclupan.derivative.init_utils import set_charges, set_compositions

cwd = Path(__file__).parent


def test_compositions_single_lattice():
    """Test composition settings."""
    elements_lattice = [[0, 1, 2]]
    n_sites_supercell = [6]

    comp = [(0, 1.0), (1, 2.0), (2, 3.0)]
    comp, _, _ = set_compositions(
        elements_lattice=elements_lattice,
        n_sites_supercell=n_sites_supercell,
        comp=comp,
    )
    comp_true = [1 / 6, 1 / 3, 1 / 2]
    np.testing.assert_allclose(comp, comp_true, atol=1e-6)


def test_compositions_multiple_sublattices1():
    """Test composition settings."""
    elements_lattice = [[0], [1], [2, 3]]
    n_sites_supercell = [4, 4, 12]

    comp = [(2, 2.5), (3, 0.5)]
    comp, _, _ = set_compositions(
        elements_lattice=elements_lattice,
        n_sites_supercell=n_sites_supercell,
        comp=comp,
    )
    comp_true = [5 / 6, 1 / 6]
    np.testing.assert_allclose(comp[2:], comp_true, atol=1e-6)
    assert comp[0] is None
    assert comp[1] is None


def test_compositions_multiple_sublattices2():
    """Test composition settings."""
    elements_lattice = [[0, 1], [0, 1, 2], [3, 4]]
    n_sites_supercell = [4, 4, 12]

    comp = [(0, 2.0), (1, 1.0), (2, 1.0), (3, 5.0), (4, 1.0)]
    comp, _, _ = set_compositions(
        elements_lattice=elements_lattice,
        n_sites_supercell=n_sites_supercell,
        comp=comp,
    )
    comp_true = [1 / 2, 1 / 4, 1 / 2, 5 / 6, 1 / 6]
    np.testing.assert_allclose(comp, comp_true, atol=1e-6)

    comp = [(0, 2.0), (1, 1.0), (2, 1.0)]
    comp, _, _ = set_compositions(
        elements_lattice=elements_lattice,
        n_sites_supercell=n_sites_supercell,
        comp=comp,
    )
    comp_true = [1 / 2, 1 / 4, 1 / 2]
    np.testing.assert_allclose(comp[:3], comp_true, atol=1e-6)
    assert comp[3] is None
    assert comp[4] is None


def test_charges():
    """Test set_charges."""
    elements_lattice = [[0, 1], [2]]
    charges_in = [(0, "1.0"), (1, "3.0"), (2, "-2.0")]
    charges = set_charges(charges_in, elements_lattice)
    assert charges == [1.0, 3.0, -2.0]
