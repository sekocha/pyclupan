"""API Class for enumerating derivative structures."""

from typing import Literal, Optional

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar
from pyclupan.derivative.derivative_utils import DerivativesSet
from pyclupan.derivative.run_sample import run_sampling_derivatives


class PyclupanDerivatives:
    """API Class for enumerating derivative structures."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._unitcell = None
        self._derivs_set = None
        self._zdd = None

        np.set_printoptions(legacy="1.21")

    def load_poscar(self, poscar: str = "POSCAR") -> PolymlpStructure:
        """Parse POSCAR files.

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
        """Return unitcell."""
        return self._unitcell

    @unitcell.setter
    def unitcell(self, cell: PolymlpStructure):
        """Setter of unitcell."""
        self._unitcell = cell

    def run_derivative(
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
                  Example: [[0], [1], [2, 3]].
        comp: Compositions for sublattices (n_elements / n_sites).
              Compositions are not needed to be normalized.
              Format: [(element ID, composition), (element ID, composition),...]
        comp_lb: Lower bounds of compositions for sublattices.
              Format: [(element ID, composition), (element ID, composition),...]
        comp_ub: Upper bounds of compositions for sublattices.
              Format: [(element ID, composition), (element ID, composition),...]
        supercell_size: Determinant of supercell matrices.
                    Derivative structures for all nonequivalent HNFs are enumerated.
        hnf: Supercell matrix in Hermite normal form.
        superperiodic: Include superperiodic derivative structures.
        end_members: Include structures of end members.
        charges: Charges of elements.
              Format: [(element ID, charge), (element ID, charge),...]
        """
        from pyclupan.derivative.run_derivative import run_derivatives

        if self._unitcell is None:
            raise RuntimeError("Unitcell not found.")

        self._derivs_set, self._zdd = run_derivatives(
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

    def save_derivatives(self, filename: str = "pyclupan_derivatives.yaml"):
        """Save derivative structures.

        Parameter
        ---------
        filename: YAML file for saving derivative structures.
        """
        from pyclupan.derivative.derivative_utils import write_derivatives_yaml

        if self._derivs_set is None:
            raise RuntimeError("Derivative structures not found.")

        fname_output = write_derivatives_yaml(
            self._derivs_set,
            self._zdd,
            filename=filename,
        )
        if self._verbose:
            if fname_output is None:
                print("No result file is not generated.", flush=True)
            else:
                print(fname_output, "is generated.", flush=True)

        return self

    def load_derivatives(self, filename: str = "pyclupan_derivatives.yaml"):
        """Load derivatives.yaml.

        Parameter
        ---------
        filename: Single YAML file or multiple YAML files for derivative structures.
        """
        from pyclupan.derivative.derivative_utils import load_derivatives_yaml

        if isinstance(filename, str):
            self._derivs_set = load_derivatives_yaml(filename)
        elif isinstance(filename, (list, tuple, np.ndarray)):
            self._derivs_set = DerivativesSet([])
            for f in filename:
                ds = load_derivatives_yaml(f)
                self._derivs_set.append(ds)

        return self

    def sample_derivatives(
        self,
        method: Literal["all", "uniform", "random"] = "uniform",
        n_samples: int = 100,
        path: str = "poscars",
        element_strings: tuple = ("Al", "Cu"),
        save_poscars: bool = True,
    ):
        """Parse derivatives.yaml.

        Parameters
        ----------
        method: Sampling method.
                Methods of "all", "uniform", and "random" are available.
        n_samples: Number of sample structures.
        save_poscars: Save poscar files.
        path: Directory path for saving structure files if save_poscars = True.
        elements: Element strings used to save structure files if save_poscars = True.
        """
        if self._derivs_set is None:
            raise RuntimeError("Derivative structures not found.")

        run_sampling_derivatives(
            ds_set=self._derivs_set,
            n_samples=n_samples,
            method=method,
            path_poscars=path,
            element_strings=element_strings,
            save_poscars=save_poscars,
        )
        return self

    def sample_derivatives_from_keys(
        self,
        keys: list,
        path: str = "poscars",
        element_strings: tuple = ("Al", "Cu"),
        save_poscars: bool = True,
    ):
        """Sample derivative structures from keys.

        Parameters
        ----------
        keys: List of keys for derivative structures,
              [(supercell_size, supercell_id, structure_id), ...].
        save_poscars: Save poscar files.
        path: Directory path for saving structure files if save_poscars = True.
        elements: Element strings used to save structure files if save_poscars = True.
        """
        if self._derivs_set is None:
            raise RuntimeError("Derivative structures not found.")

        run_sampling_derivatives(
            ds_set=self._derivs_set,
            keys=keys,
            path_poscars=path,
            element_strings=element_strings,
            save_poscars=save_poscars,
        )
        return self

    def get_sampled_structures(self, element_strings: tuple):
        """Return sampled structures."""
        if self._derivs_set is None:
            raise RuntimeError("Derivative structures not found.")
        return self._derivs_set.get_sampled_structures(element_strings)

    @property
    def derivative_structures(self):
        """Return derivative structures.

        Return
        ------
        deriv_set: Instance of DerivativeSet class.
        """
        return self._derivs_set

    @derivative_structures.setter
    def derivative_structures(self, derivs: DerivativesSet):
        """Setter of derivative structures."""
        self._derivs_set = derivs
