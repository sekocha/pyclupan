"""API Class for pyclupan."""

from typing import Literal, Optional

import numpy as np
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar

from pyclupan.derivative.derivative_io import (
    load_derivative_yaml,
    write_derivative_yaml,
)
from pyclupan.derivative.run_derivative import run_derivatives


class Pyclupan:
    """API Class for pyclupan."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._unitcell = None
        self._verbose = verbose

        self._derivs_set = None

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
        superperiodic: bool = False,
        end_members: bool = False,
        charges: Optional[list] = None,
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
        superperiodic: Include superperiodic derivative structures.
        end_members: Include structures of end members.
        charges: Charges of elements.
        """
        self._derivs_set = run_derivatives(
            self._unitcell,
            occupation=occupation,
            elements=elements,
            comp=comp,
            comp_lb=comp_lb,
            comp_ub=comp_ub,
            supercell_size=supercell_size,
            hnf=hnf,
            one_of_k_rep=one_of_k_rep,
            superperiodic=superperiodic,
            end_members=end_members,
            charges=charges,
            verbose=self._verbose,
        )
        return self

    def save_derivatives(self, filename: str = "derivatives.yaml"):
        """Save derivative structures."""
        write_derivative_yaml(self._derivs_set, filename=filename)
        return self

    def load_derivatives(self, filename: str = "derivatives.yaml"):
        """Parse derivatives.yaml.

        Returns
        -------
        TODO: ***
        """
        self._derivs_set = load_derivative_yaml(filename=filename)
        return self

    def sample_derivatives(
        self,
        method: Literal["all", "uniform", "random"] = "uniform",
        n_samples: int = 100,
        path: str = "poscars",
        elements: tuple = ("Al", "Cu"),
    ):
        """Parse derivatives.yaml.

        Returns
        -------
        TODO: ***
        """
        if method == "all":
            self._derivs_set.all()
        elif method == "uniform":
            self._derivs_set.uniform(n_samples=n_samples)
        elif method == "random":
            self._derivs_set.random(n_samples=n_samples)
        self._derivs_set.save(path=path, elements=elements)
