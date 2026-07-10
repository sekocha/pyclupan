"""Utility functions for pyclupan API."""

from pyclupan._version import __version__


def print_credit():
    """Print credit of pyclupan."""
    print("Pyclupan", "version", __version__, flush=True)
    print("  A. Seko et al., Phys. Rev. B 80, 165122 (2009)", flush=True)
