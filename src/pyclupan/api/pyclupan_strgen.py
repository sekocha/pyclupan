"""API Class for generating substitutional configs with displacements."""

# from typing import Optional
#
# import numpy as np

from .pyclupan_base import PyclupanBase


class PyclupanStructureGenerator(PyclupanBase):
    """API Class for generating substitutional configs with displacements."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        super().__init__(verbose=verbose)
