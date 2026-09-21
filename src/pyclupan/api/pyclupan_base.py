"""API Base Class for pyclupan."""

# from abc import ABC, abstractmethod

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar


class PyclupanBase:
    """API Base Class for pyclupan."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose
        self._unitcell = None
        np.set_printoptions(legacy="1.21")

    def load_poscar(self, poscar: str = "POSCAR") -> PolymlpStructure:
        """Parse POSCAR files to define lattice.

        Parameter
        ---------
        poscar: Name of POSCAR file.

        Returns
        -------
        structure: Structure in PolymlpStructure format.
        """
        self._unitcell = Poscar(poscar).structure
        return self._unitcell

    @property
    def unitcell(self):
        """Return unitcell to define lattice."""
        return self._unitcell

    @unitcell.setter
    def unitcell(self, cell: PolymlpStructure):
        """Setter of unitcell to define lattice."""
        self._unitcell = cell
