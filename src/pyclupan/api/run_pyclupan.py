"""Command lines for running pyclupan functions."""

import argparse
import signal

import numpy as np

from pyclupan.api.api_utils import print_credit
from pyclupan.api.pyclupan import Pyclupan


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        default=None,
        help="Primitive cell in POSCAR format.",
    )
    parser.add_argument(
        "-e",
        "--elements",
        nargs="*",
        type=int,
        action="append",
        default=None,
        help="Element IDs on a lattice",
    )
    parser.add_argument(
        "-o",
        "--occupation",
        nargs="*",
        type=int,
        action="append",
        default=None,
        help="Lattice IDs occupied by element",
    )
    parser.add_argument(
        "-c",
        "--comp",
        nargs=2,
        type=str,
        action="append",
        default=None,
        help="Composition (n_elements / n_sites)",
    )
    parser.add_argument(
        "--comp_lb",
        nargs=2,
        type=str,
        action="append",
        default=None,
        help="Lower bound of composition (n_elements / n_sites)",
    )
    parser.add_argument(
        "--comp_ub",
        nargs=2,
        type=str,
        action="append",
        default=None,
        help="Upper bound of composition (n_elements / n_sites)",
    )
    parser.add_argument(
        "--hnf",
        type=int,
        nargs=9,
        default=None,
        help="Hermite normal form",
    )
    parser.add_argument(
        "--supercell_size",
        type=int,
        default=None,
        help="Determinant of Hermite normal form",
    )
    parser.add_argument(
        "--charge",
        type=str,
        nargs=2,
        action="append",
        default=None,
        help="Charge of element (element id, charge).",
    )
    parser.add_argument(
        "--superperiodic",
        action="store_true",
        help="Include superperiodic structures.",
    )

    parser.add_argument(
        "--yaml",
        type=str,
        default="derivatives.yaml",
        help="Yaml file for derivative structures.",
    )
    parser.add_argument(
        "--element_strings",
        nargs="*",
        type=str,
        default=None,
        help="Element strings for structure files.",
    )

    args = parser.parse_args()

    print_credit()
    np.set_printoptions(legacy="1.21")
    clupan = Pyclupan(verbose=True)

    if args.hnf is not None:
        args.hnf = np.array(args.hnf).reshape((3, 3))
    if args.poscar:
        clupan.load_poscar(args.poscar)
        clupan.run_derivative(
            occupation=args.occupation,
            elements=args.elements,
            comp=args.comp,
            comp_lb=args.comp_lb,
            comp_ub=args.comp_ub,
            supercell_size=args.supercell_size,
            hnf=args.hnf,
            charges=args.charge,
            superperiodic=args.superperiodic,
        )
        clupan.save_derivatives(filename="pyclupan_derivatives.yaml")
    elif args.yaml:
        if args.element_strings is None:
            raise RuntimeError("Element string must be given.")

        clupan.load_derivatives(args.yaml)
        clupan.sample_derivatives(
            method="uniform",
            n_samples=20,
            elements=args.element_strings,
        )
