"""API Class for pyclupan."""

from typing import Optional

import numpy as np
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar

from pyclupan.core.normal_form import get_nonequivalent_hnf
from pyclupan.derivative.derivative_utils import (
    set_compositions,
    set_elements_on_sublattices,
)
from pyclupan.zdd.zdd_base import ZddLattice


class Pyclupan:
    """API Class for pyclupan."""

    def __init__(
        self,
        verbose: bool = False,
    ):
        self._unitcell = None
        self._zdd = None
        self._verbose = verbose

    def load_poscar(self, poscar: str = "POSCAR") -> PolymlpStructure:
        """Parse POSCAR files.

        Returns
        -------
        structure: Structure in PolymlpStructure format.
        """
        self._unitcell = Poscar(poscar).structure
        return self._unitcell

    def set_derivative_params(
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
        """Set parameters for enumerating derivative structures.

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
        """

        n_sites = self._unitcell.n_atoms
        elements_lattice = set_elements_on_sublattices(
            n_sites=n_sites,
            occupation=occupation,
            elements=elements,
        )
        comp, comp_lb, comp_ub = set_compositions(
            elements_lattice=elements_lattice,
            comp=comp,
            comp_lb=comp_lb,
            comp_ub=comp_ub,
        )

        if supercell_size is None and hnf is None:
            raise RuntimeError("supercell_size or hnf required.")

        if hnf is None:
            hnf_all = get_nonequivalent_hnf(supercell_size, self._unitcell)
        else:
            hnf_all = [hnf]
        self._hnf_all = hnf_all

        self._zdd = ZddLattice(
            n_sites=n_sites,
            elements_lattice=elements_lattice,
            one_of_k_rep=one_of_k_rep,
            verbose=self._verbose,
        )
