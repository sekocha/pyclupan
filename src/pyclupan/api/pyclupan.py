"""API Class for pyclupan."""

# from typing import Literal, Optional, Union
#
# import numpy as np
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar


class Pyclupan:
    """API Class for pyclupan."""

    def __init__(
        self,
        verbose: bool = False,
    ):
        self._structure = None

    def load_poscar(self, poscar: str = "POSCAR") -> PolymlpStructure:
        """Parse POSCAR files.

        Returns
        -------
        structure: Structure in PolymlpStructure format.
        """
        self._structure = Poscar(poscar).structure
        return self._structure
