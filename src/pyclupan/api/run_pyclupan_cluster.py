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
        "--order",
        type=int,
        default=4,
        help="Maximum order of clusters",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="*",
        type=float,
        default=(6.0, 6.0, 6.0),
        help="Cutoff radius for each order of cluster",
    )

    args = parser.parse_args()

    print_credit()
    np.set_printoptions(legacy="1.21")
    clupan = Pyclupan(verbose=True)

    clupan.load_poscar(args.poscar)
    clupan.run_cluster(
        occupation=args.occupation,
        elements=args.elements,
        max_order=args.order,
        cutoffs=args.cutoffs,
        filename="pyclupan_clusters.yaml",
    )
