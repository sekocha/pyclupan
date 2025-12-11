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
        "--clusters",
        type=str,
        required=True,
        default="pyclupan_clusters.yaml",
        help="Cluster search result file.",
    )
    parser.add_argument(
        "-e",
        "--ecis",
        type=str,
        default=None,
        help="ECIs obtained from regression.",
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
    parser.add_argument(
        "--end_poscars",
        nargs="*",
        type=str,
        default=None,
        help="POSCAR files for end members.",
    )

    args = parser.parse_args()

    if args.poscars is None and args.derivatives is None and args.samples is None:
        raise RuntimeError("Structure files are required.")

    print_credit()
    np.set_printoptions(legacy="1.21")
    clupan = PyclupanCalc(args.clusters, verbose=True)
    if args.derivatives:
        for d in args.derivatives:
            clupan.load_derivatives_yaml(d)
    elif args.samples:
        clupan.load_sample_attrs_yaml(args.samples)
    elif args.poscars:
        if args.element_strings is None:
            raise RuntimeError("Element strings are required.")
        clupan.load_poscars(args.poscars, element_strings=args.element_strings)

    calc_energy = True
    try:
        clupan.load_ecis(args.ecis)
    except:
        calc_energy = False

    clupan.eval_cluster_functions()

    if not calc_energy:
        clupan.save_features()
    else:
        clupan.eval_energies()
        clupan.save_energies()
        clupan.eval_formation_energies(
            poscars_endmembers=args.end_poscars,
            element_strings=args.element_strings,
        )
        clupan.save_formation_energies()
        clupan.save_convex_hull_yaml()
        clupan.load_formation_energies()
        clupan.save_convex_hull_poscars_from_derivatives(
            element_strings=args.element_strings
        )
