"""Command lines for running regressions."""

import argparse
import signal

import numpy as np

from pyclupan.api.api_utils import print_credit
from pyclupan.api.pyclupan_regression import PyclupanRegression


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        required=True,
        default="pyclupan_features.hdf5",
        help="HDF5 file for features.",
    )
    parser.add_argument(
        "-e",
        "--energy",
        type=str,
        required=True,
        default="energy.dat",
        help="Data file for energy.",
    )

    args = parser.parse_args()

    print_credit()
    np.set_printoptions(legacy="1.21")

    pyclupan = PyclupanRegression(verbose=True)
    pyclupan.load_features(args.features)
    pyclupan.load_energies(energy_dat=args.energy)

    pyclupan.run_lasso()
    pyclupan.save_predictions()
    pyclupan.save()
