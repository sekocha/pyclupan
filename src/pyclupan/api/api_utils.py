"""Utility functions for pyclupan API."""

from pyclupan._version import __version__


def print_credit():
    """Print credit of pyclupan."""
    print("Pyclupan", "version", __version__, flush=True)
    # print("  polynomial machine learning potential:", flush=True)
    # print("  A. Seko, J. Appl. Phys. 133, 011101 (2023)", flush=True)
