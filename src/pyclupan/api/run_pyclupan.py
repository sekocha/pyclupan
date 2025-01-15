"""Command lines for running pyclupan functions."""

import argparse
import signal

import numpy as np

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
        nargs="*",
        type=str,
        action="append",
        default=None,
        help="Composition (n_elements / n_sites)",
    )
    parser.add_argument(
        "--comp_lb",
        nargs="*",
        type=str,
        action="append",
        default=None,
        help="Lower bound of composition (n_elements / n_sites)",
    )
    parser.add_argument(
        "--comp_ub",
        nargs="*",
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
        "--charges",
        type=float,
        nargs="*",
        default=None,
        help="Charges of elements",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default="derivatives.yaml",
        help="Yaml file.",
    )
    args = parser.parse_args()

    np.set_printoptions(legacy="1.21")
    clupan = Pyclupan(verbose=True)
    if args.poscar:
        clupan.load_poscar(args.poscar)
        clupan.run(
            occupation=args.occupation,
            elements=args.elements,
            comp=args.comp,
            comp_lb=args.comp_lb,
            comp_ub=args.comp_ub,
            supercell_size=args.supercell_size,
            hnf=args.hnf,
            charges=args.charges,
        )
        clupan.save_derivatives(filename="derivatives.yaml")
    elif args.yaml:
        clupan.load_derivatives(args.yaml)
        clupan.sample_derivatives(method="uniform", n_samples=30)
