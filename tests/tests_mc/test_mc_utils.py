"""Tests of mc_utils."""

from pathlib import Path

import numpy as np

from pyclupan.mc.mc_utils import MCAttr, MCParams

cwd = Path(__file__).parent


def test_mc_attr():
    """Test MCAttr."""
    active_spins = np.array([1, -1, 1, -1, 1, -1])
    mc_attr = MCAttr(active_spins=active_spins)
    assert mc_attr.n_sites == 6


def test_mc_params():
    """Test MCParams."""
    params = MCParams()

    tinit, tfinal, tstep = 2000, 1000, 300
    params.set_temperature_range(tinit, tfinal, tstep)
    np.testing.assert_allclose(params.temperatures, np.arange(tinit, tfinal, -tstep))

    tinit, tfinal, tstep = 2000, 2000, 100
    params.set_temperature_range(tinit, tfinal, tstep)
    np.testing.assert_allclose(params.temperatures, [2000])
