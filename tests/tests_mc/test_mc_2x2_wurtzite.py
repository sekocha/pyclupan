"""Tests of MC simulations."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_mc import PyclupanMC
from pyclupan.core.pypolymlp_utils import PolymlpStructure

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/2x2_wurtzite"


def test_cmc_2x2_wurtzite():
    """Test canonical MC."""
    pyclupan = PyclupanMC(
        clusters_yaml=path_file + "/pyclupan_clusters.yaml",
        ecis_yaml=path_file + "/pyclupan_ecis.yaml",
        verbose=False,
    )
    size = (1, 1, 2)
    pyclupan.set_supercell(supercell_matrix=size, refine=True)
    pyclupan.set_init(compositions=(0.25, 0.25, 0.25, 0.25))
    pyclupan.set_parameters(
        n_steps_init=1,
        n_steps_eq=2,
        temperature=100,
        ensemble="canonical",
    )
    np.testing.assert_allclose(pyclupan.temperatures, [100])
    pyclupan.run()

    pyclupan.set_parameters(
        n_steps_init=1,
        n_steps_eq=2,
        temperature_init=200,
        temperature_final=100,
        temperature_step=50,
        ensemble="canonical",
    )
    pyclupan.run()
    np.testing.assert_allclose(pyclupan.temperatures, [200, 150, 100])
    assert isinstance(pyclupan.structure, PolymlpStructure)
    assert pyclupan.average_energies.shape[0] == 3
    assert pyclupan.average_cluster_functions.shape == (3, 160)
