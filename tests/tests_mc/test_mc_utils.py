"""Tests of mc_utils."""

from pathlib import Path

import numpy as np

from pyclupan.mc.mc_utils import MCAttr, MCParams

cwd = Path(__file__).parent


def test_mc_attr():
    """Test MCAttr."""
    active_spins = np.array([1, -1, 1, -1, 1, -1])
    mc_attr = MCAttr(active_spins=active_spins)
    assert mc_attr.n_total_sites == 6


def _test_sample_two(mc_attr):
    """Test sampling two sites."""
    active_spins = mc_attr.active_spins
    i, j = mc_attr.select_two_sites(active_spins)
    if i in {0, 1, 2}:
        assert j in {0, 1, 2}
    else:
        assert j in {3, 4, 5}


def _test_sample_one(mc_attr):
    """Test sampling one site."""
    active_spins = mc_attr.active_spins
    i, spin_new = mc_attr.select_one_site(active_spins)
    if i == 0:
        assert spin_new == -1
    elif i == 1:
        assert spin_new == 1
    elif i == 2:
        assert spin_new == -1
    elif i == 3:
        assert spin_new in {0, 1}
    elif i == 4:
        assert spin_new in {0, -1}
    else:
        assert spin_new in {1, -1}


def test_mc_attr_sample():
    """Test sample methods in MCAttr."""
    active_spins = np.array([1, -1, 1, -1, 1, 0])
    n_active_sites = np.array([3, 3])
    elements = [[0, 1], [2, 3, 4]]
    spins = [np.array([1, -1]), np.array([1, 0, -1])]
    mc_attr = MCAttr(
        active_spins=active_spins,
        active_element_species=elements,
        spin_species=spins,
        n_active_sites=n_active_sites,
    )
    mc_attr.print_attrs()

    for i in range(100):
        mc_attr.select_sublattice() in {0, 1}
        _test_sample_two(mc_attr)
        _test_sample_one(mc_attr)


def test_mc_params():
    """Test MCParams."""
    params = MCParams()

    tinit, tfinal, tstep = 2000, 1000, 300
    params.set_temperature_range(tinit, tfinal, tstep)
    np.testing.assert_allclose(params.temperatures, np.arange(tinit, tfinal, -tstep))

    tinit, tfinal, tstep = 2000, 2000, 100
    params.set_temperature_range(tinit, tfinal, tstep)
    np.testing.assert_allclose(params.temperatures, [2000])
