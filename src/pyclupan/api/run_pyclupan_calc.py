"""Command lines for running pyclupan functions."""

import argparse
import signal

import numpy as np

from pyclupan.api.api_utils import print_credit
from pyclupan.api.pyclupan_calc import PyclupanCalc


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cluster",
        type=str,
        required=True,
        default="pyclupan_cluster.yaml",
        help="Cluster search result file.",
    )
    parser.add_argument(
        "-p",
        "--poscars",
        nargs="*",
        type=str,
        default=None,
        help="POSCAR files.",
    )
    parser.add_argument(
        "--element_strings",
        nargs="*",
        type=str,
        default=None,
        help="Element strings.",
    )
    parser.add_argument(
        "--derivatives",
        type=str,
        nargs="*",
        default=None,
        help="Yaml files for derivative structures.",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Yaml file for derivative structures sampled.",
    )

    args = parser.parse_args()

    if args.poscars is None and args.derivatives is None and args.samples is None:
        raise RuntimeError("Structure files are required.")

    print_credit()
    np.set_printoptions(legacy="1.21")
    clupan = PyclupanCalc(args.cluster, verbose=True)
    if args.derivatives:
        for d in args.derivatives:
            clupan.load_derivatives_yaml(d)
    elif args.samples:
        clupan.load_sample_attrs_yaml(args.samples)
    elif args.poscars:
        if args.element_strings is None:
            raise RuntimeError("Element strings are required.")
        clupan.load_poscars(args.poscars, element_strings=args.element_strings)

    clupan.eval_cluster_functions()
    clupan.save_features(filename="pyclupan_features.hdf5")
