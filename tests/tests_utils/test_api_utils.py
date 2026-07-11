"""Tests of utility functions."""

import os
from pathlib import Path

from pyclupan.api.api_utils import save_energy_dat
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


def test_save_energy_dat():
    """Test save_energy_dat."""
    poscar = str(cwd) + "/../files/binary_fcc/poscar-end1"
    unitcell = Poscar(poscar).structure
    vaspruns = [
        str(cwd) + "/vasprun.xml.2-0-0",
        str(cwd) + "/vasprun.xml.2-1-0",
    ]
    save_energy_dat(vaspruns, unitcell, filename="tmp.dat")
    os.remove("tmp.dat")
