"""Tests of SQS search."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_sqs import PyclupanSQS
from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar

cwd = Path(__file__).parent


def test_sqs_binary_fcc():
    """Test sqs_search."""
    path_file = str(cwd) + "/../files/binary_fcc"
    pyclupan = PyclupanSQS(
        clusters_yaml=path_file + "/pyclupan_clusters.yaml",
        verbose=False,
    )
    size = (2, 2, 2)
    pyclupan.set_supercell(supercell_matrix=size, refine=True)
    pyclupan.set_init(compositions=(0.5, 0.5))
    pyclupan.set_parameters(
        n_steps=2,
        temperature_init=10,
        temperature_final=1,
        n_temperatures=3,
    )
    pyclupan.run()
    assert isinstance(pyclupan.structure, PolymlpStructure)

    st = Poscar(str(cwd) + "/POSCAR_init_fcc").structure
    pyclupan.set_init(structure=st, element_strings=("Ag", "Au"))
    pyclupan.set_parameters(
        n_steps=2,
        temperature_init=10,
        temperature_final=1,
        n_temperatures=3,
    )
    pyclupan.run()
    assert isinstance(pyclupan.structure, PolymlpStructure)


def test_sqs_ternary_fcc():
    """Test sqs_search."""
    path_file = str(cwd) + "/../files/ternary_fcc"
    pyclupan = PyclupanSQS(
        clusters_yaml=path_file + "/pyclupan_clusters.yaml",
        cluster_ids=np.arange(10),
        verbose=False,
    )
    size = (2, 2, 2)
    pyclupan.set_supercell(supercell_matrix=size, refine=True)
    pyclupan.set_init(compositions=(0.25, 0.25, 0.5))
    pyclupan.set_parameters(
        n_steps=1,
        temperature_init=10,
        temperature_final=1,
        n_temperatures=3,
    )
    pyclupan.run()
    assert isinstance(pyclupan.structure, PolymlpStructure)
