"""Tests of VASP interface."""

from pathlib import Path

import numpy as np

from pyclupan.core.interface_vasp import load_vasp_results

cwd = Path(__file__).parent


def test_load_vasp():
    """Test load_vasp_results."""
    structure_ids, energies, structures = load_vasp_results(
        [str(cwd) + "/vasprun.xml.2-0-0", str(cwd) + "/vasprun.xml.2-1-0"]
    )
    np.testing.assert_allclose(energies, [-6.02654699, -6.06150566], atol=1e-8)
