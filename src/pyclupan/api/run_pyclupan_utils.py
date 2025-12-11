"""Command lines for running pyclupan functions."""

import argparse
import signal

import numpy as np

from pyclupan.api.api_utils import print_credit
from pyclupan.api.pyclupan_utils import save_energy_dat
from pyclupan.core.pypolymlp_utils import Poscar


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        default="POSCAR",
        help="Unitcell (Primitive) of POSCAR file",
    )
    parser.add_argument(
        "-v",
        "--vaspruns",
        nargs="*",
        type=str,
        default="./*-*-*/vasprun.xml",
        help="vasprun.xml files",
    )
    args = parser.parse_args()

    print_credit()
    np.set_printoptions(legacy="1.21")

    unitcell = Poscar(args.poscar).structure
    vaspruns = sorted(args.vaspruns)
    save_energy_dat(vaspruns, unitcell, filename="pyclupan_energy.dat")
