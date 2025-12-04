"""Tests of MC simulations."""

from pathlib import Path

from pyclupan.api.pyclupan_mc import PyclupanMC

# import numpy as np


cwd = Path(__file__).parent


def test_cmc_fcc_binary():
    """Test canonical MC."""
    pyclupan = PyclupanMC(
        clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml",
        ecis_yaml=str(cwd) + "/pyclupan_ecis.yaml",
        verbose=True,
    )
    size = (1, 1, 2)
    pyclupan.set_supercell(supercell_matrix=size, refine=True)
    pyclupan.set_init(compositions=(0.5, 0.5))
    pyclupan.set_parameters(
        n_steps_init=1,
        n_steps_eq=2,
        temperature=100,
        ensemble="canonical",
    )
    pyclupan.run()
    assert 0 == 1
