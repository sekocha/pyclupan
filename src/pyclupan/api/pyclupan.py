"""API Class for pyclupan."""

from typing import Optional

import numpy as np
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar

from pyclupan.derivative.run_derivative import run_derivatives


class Pyclupan:
    """API Class for pyclupan."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._unitcell = None
        self._verbose = verbose

    def load_poscar(self, poscar: str = "POSCAR") -> PolymlpStructure:
        """Parse POSCAR files.

        Returns
        -------
        structure: Structure in PolymlpStructure format.
        """
        self._unitcell = Poscar(poscar).structure
        return self._unitcell

    def run(
        self,
        occupation: Optional[list] = None,
        elements: Optional[list] = None,
        comp: Optional[list] = None,
        comp_lb: Optional[list] = None,
        comp_ub: Optional[list] = None,
        supercell_size: Optional[int] = None,
        hnf: Optional[np.ndarray] = None,
        one_of_k_rep: bool = False,
    ):
        """Enumerate derivative structures.

        Parameters
        ----------
        occupation: Lattice IDs occupied by elements.
                    Example: [[0], [1], [2], [2]].
        elements: Element IDs on lattices.
                  Example: [[0],[1],[2, 3]].
        comp: Compositions for sublattices (n_elements / n_sites).
              Compositions are not needed to be normalized.
              Format: [(element ID, composition), (element ID, compositions),...]
        comp_lb: Lower bounds of compositions for sublattices.
              Format: [(element ID, composition), (element ID, compositions),...]
        comp_ub: Upper bounds of compositions for sublattices.
              Format: [(element ID, composition), (element ID, compositions),...]
        supercell_size: Determinant of supercell matrices.
                    Derivative structures for all nonequivalent HNFs are enumerated.
        hnf: Supercell matrix in Hermite normal form.
        """
        run_derivatives(
            self._unitcell,
            occupation=occupation,
            elements=elements,
            comp=comp,
            comp_lb=comp_lb,
            comp_ub=comp_ub,
            supercell_size=supercell_size,
            hnf=hnf,
            one_of_k_rep=one_of_k_rep,
            verbose=self._verbose,
        )
