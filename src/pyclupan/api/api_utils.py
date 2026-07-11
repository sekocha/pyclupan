"""Utility functions for pyclupan API."""

import pyclupan.core.interface_vasp as interface_vasp
from pyclupan._version import __version__

save_energy_dat = interface_vasp.save_energy_dat


def print_credit():
    """Print credit of pyclupan."""
    print("Pyclupan", "version", __version__, flush=True)
    print("  A. Seko et al., Phys. Rev. B 80, 165122 (2009)", flush=True)
