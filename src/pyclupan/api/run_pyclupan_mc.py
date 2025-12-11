"""Command lines for running MC simulations."""

import argparse
import signal

import numpy as np

from pyclupan.api.api_utils import print_credit
from pyclupan.api.pyclupan_mc import PyclupanMC


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        default=None,
        help="Initial structure.",
    )
    parser.add_argument(
        "--element_strings",
        nargs="*",
        type=str,
        default=None,
        help="Element strings for structure files.",
    )
    parser.add_argument(
        "-c",
        "--comp",
        nargs="*",
        type=float,
        default=None,
        help="Composition (n_elements / n_sites).",
    )

    parser.add_argument(
        "--clusters",
        type=str,
        default="pyclupan_clusters.yaml",
        help="File from cluster search.",
    )
    parser.add_argument(
        "--ecis",
        type=str,
        default="pyclupan_ecis.yaml",
        help="File from ECI estimation.",
    )

    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        required=True,
        help="Diagonal supercell size.",
    )
    parser.add_argument(
        "--n_steps",
        nargs=2,
        type=int,
        default=(100, 1000),
        help="Numbers of steps for (initialization, average).",
    )
    parser.add_argument(
        "-t",
        "--temps",
        nargs="*",
        type=float,
        default=None,
        help="Temperature (K).",
    )
    parser.add_argument(
        "--temp_init",
        type=float,
        default=None,
        help="Initial temperature (K).",
    )
    parser.add_argument(
        "--temp_final",
        type=float,
        default=None,
        help="Final temperature (K).",
    )
    parser.add_argument(
        "--n_temps",
        type=int,
        default=None,
        help="Number of temperatures.",
    )
    parser.add_argument(
        "--simulated_annealing",
        action="store_true",
        help="Perform simulated annealing.",
    )
    parser.add_argument(
        "--mu",
        nargs="*",
        type=float,
        default=None,
        help="Chemical potentials (eV).",
    )

    args = parser.parse_args()

    print_credit()
    np.set_printoptions(legacy="1.21")

    pyclupan = PyclupanMC(
        clusters_yaml=args.clusters,
        ecis_yaml=args.ecis,
        verbose=True,
    )

    pyclupan.set_supercell(supercell_matrix=args.supercell, refine=True)
    pyclupan.set_init(
        poscar=args.poscar,
        element_strings=args.element_strings,
        compositions=args.comp,
    )

    if args.simulated_annealing:
        pyclupan.set_parameters_simulated_annealing(
            n_steps_init=args.n_steps[0],
            n_steps_eq=args.n_steps[1],
            temperature_init=args.temp_init,
            temperature_final=args.temp_final,
            n_temperatures=args.n_temps,
        )
    else:
        ensemble = "semi_grand_canonical" if args.mu is not None else "canonical"
        if args.temps is None and args.temp_init is not None:
            args.temperatures = np.linspace(
                args.temp_init,
                args.temp_final,
                num=args.n_temps,
            )

        pyclupan.set_parameters(
            n_steps_init=args.n_steps[0],
            n_steps_eq=args.n_steps[1],
            temperatures=args.temps,
            ensemble=ensemble,
            mu=args.mu,
        )

    pyclupan.run()
    pyclupan.save_structure()
    pyclupan.save_mc_yaml()
