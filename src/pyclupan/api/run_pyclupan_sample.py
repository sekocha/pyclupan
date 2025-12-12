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
        "--yaml",
        nargs="*",
        type=str,
        required=True,
        help="Yaml files for derivative structures.",
    )
    parser.add_argument(
        "--method",
        choices=["all", "uniform", "random"],
        default="uniform",
        help="Sampling method.",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=20,
        help="Number of sample structures.",
    )
    parser.add_argument(
        "--element_strings",
        nargs="*",
        type=str,
        required=True,
        help="Element strings for structure files.",
    )

    args = parser.parse_args()

    print_credit()
    np.set_printoptions(legacy="1.21")
    clupan = Pyclupan(verbose=True)

    clupan.load_derivatives(args.yaml)
    clupan.sample_derivatives(
        method=args.method,
        n_samples=args.n_samples,
        elements=args.element_strings,
        path="poscars",
    )
