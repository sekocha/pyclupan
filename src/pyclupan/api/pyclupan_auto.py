"""API Class for generating CE model using polymlp."""

from typing import Optional

import numpy as np

from pyclupan.api.pyclupan import Pyclupan

# from pyclupan.api.pyclupan_calc import PyclupanCalc
from pyclupan.core.pypolymlp_utils import Polymlp
from pyclupan.derivative.derivative_utils import DerivativesSet


class PyclupanCE:
    """API Class for generating CE model using polymlp."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._pyclupan = Pyclupan(verbose=verbose)
        self._pyclupan_calc = None

        self._unitcell = None
        self._elements = None
        self._element_strings = None
        np.set_printoptions(legacy="1.21")

        self._sampled_structures = []
        self._ds_set = DerivativesSet([])

        self._energies = None

    def set_lattice_and_elements(
        self,
        elements: list,
        element_strings: list,
        poscar: str = "POSCAR",
    ):
        """Set lattice and elements.

        Parameter
        ---------
        poscar: Name of POSCAR file.
        elements: Element integers on sublattice.
        element_strings: Element strings corresponding to element integers.
        """
        self._elements = elements
        self._element_strings = element_strings
        self._unitcell = self._pyclupan.load_poscar(poscar)
        return self

    def enum_derivatives(
        self,
        min_supercell_size: int = 1,
        max_supercell_size: int = 8,
        n_samples: Optional[int] = None,
        end_members: bool = True,
        superperiodic: bool = False,
    ):
        """Enumerate derivative structures."""
        for size in range(min_supercell_size, max_supercell_size + 1):
            self._pyclupan.run_derivative(
                elements=self._elements,
                supercell_size=size,
                end_members=end_members,
                superperiodic=superperiodic,
            )
            filename = "pyclupan_derivative_" + str(size) + ".yaml"
            self._pyclupan.save_derivatives(filename=filename)
            self._ds_set.append(self._pyclupan.derivative_structures)

            # for ds in ds_set:
            #     ds.structure_ids = ds.all_ids

            method = "all" if n_samples is None else "uniform"
            self._pyclupan.sample_derivatives(
                method=method,
                n_samples=n_samples,
                save_poscars=False,
            )
            ds_set = self._pyclupan.derivative_structures
            structures = ds_set.get_sampled_structures(
                element_strings=self._element_strings
            )
            self._sampled_structures.extend(structures)
        return self

    def eval_energies(self, pot: Optional[str] = "polymlp.yaml"):
        """Evaluate energies using polymp."""
        if self._sampled_structures is None:
            raise RuntimeError("Sampled structures not found.")

        polymlp = Polymlp(pot=pot)
        success = np.ones(len(self._sampled_structures), dtype=bool)
        n_atom_unitcell = len(self._unitcell.elements)
        energies = []
        for i, st in enumerate(self._sampled_structures):
            suc = polymlp.run_geometry_optimization(st, gtol=1e-4)
            if not suc:
                success[i] = False
                continue

            n_atom = len(polymlp.structure.elements)
            n_unitcell = n_atom / n_atom_unitcell
            energies.append(polymlp.energy / n_unitcell)
        self._energies = np.array(energies)
        return self._energies


#     def run_cluster(
#         self,
#         occupation: Optional[list] = None,
#         elements: Optional[list] = None,
#         max_order: int = 4,
#         cutoffs: tuple[float] = (6.0, 6.0, 6.0),
#         filename: str = "pyclupan_cluster.yaml",
#     ):
#         """Search nonequivalent clusters.
#         Parameters
#         ----------
#         occupation: Lattice IDs occupied by elements.
#                     Example: [[0], [1], [2], [2]].
#         elements: Element IDs on lattices.
#                   Example: [[0], [1], [2, 3]].
#         max_order: Maximum order of clusters.
#         cutoffs: Cutoff distances for orders >= 2.
#                 (two-body, three-body, four-body, ...)
#                 Size of cutoffs must be equal to max_order - 1.
#                 Cutoffs must be smaller or equal to those for smaller orders.
#         filename: Name of output file for cluster search results.
#                   If None, no file will be generated.
#         """
#         if self._unitcell is None:
#             raise RuntimeError("Unitcell not found.")
#
#         self._clusters = run_cluster(
#             unitcell=self._unitcell,
#             occupation=occupation,
#             elements=elements,
#             max_order=max_order,
#             cutoffs=cutoffs,
#             filename=filename,
#             verbose=self._verbose,
#         )
#         return self
#
#     def run_derivative(
#         self,
#         occupation: Optional[list] = None,
#         elements: Optional[list] = None,
#         comp: Optional[list] = None,
#         comp_lb: Optional[list] = None,
#         comp_ub: Optional[list] = None,
#         supercell_size: Optional[int] = None,
#         hnf: Optional[np.ndarray] = None,
#         one_of_k_rep: bool = False,
#         superperiodic: bool = False,
#         end_members: bool = False,
#         charges: Optional[list] = None,
#     ):
#         """Enumerate derivative structures.
#
#         Parameters
#         ----------
#         occupation: Lattice IDs occupied by elements.
#                     Example: [[0], [1], [2], [2]].
#         elements: Element IDs on lattices.
#                   Example: [[0], [1], [2, 3]].
#         comp: Compositions for sublattices (n_elements / n_sites).
#               Compositions are not needed to be normalized.
#               Format: [(element ID, composition), (element ID, composition),...]
#         comp_lb: Lower bounds of compositions for sublattices.
#               Format: [(element ID, composition), (element ID, composition),...]
#         comp_ub: Upper bounds of compositions for sublattices.
#               Format: [(element ID, composition), (element ID, composition),...]
#         supercell_size: Determinant of supercell matrices.
#                     Derivative structures for all nonequivalent HNFs are enumerated.
#         hnf: Supercell matrix in Hermite normal form.
#         superperiodic: Include superperiodic derivative structures.
#         end_members: Include structures of end members.
#         charges: Charges of elements.
#               Format: [(element ID, charge), (element ID, charge),...]
#         """
#         from pyclupan.derivative.run_derivative import run_derivatives
#
#         if self._unitcell is None:
#             raise RuntimeError("Unitcell not found.")
#
#         self._derivs_set, self._zdd = run_derivatives(
#             self._unitcell,
#             occupation=occupation,
#             elements=elements,
#             comp=comp,
#             comp_lb=comp_lb,
#             comp_ub=comp_ub,
#             supercell_size=supercell_size,
#             hnf=hnf,
#             one_of_k_rep=one_of_k_rep,
#             superperiodic=superperiodic,
#             end_members=end_members,
#             charges=charges,
#             verbose=self._verbose,
#         )
#         return self
#
#     def save_derivatives(self, filename: str = "pyclupan_derivatives.yaml"):
#         """Save derivative structures.
#
#         Parameter
#         ---------
#         filename: YAML file for saving derivative structures.
#         """
#         from pyclupan.derivative.derivative_utils import write_derivatives_yaml
#
#         if self._derivs_set is None:
#             raise RuntimeError("Derivative structures not found.")
#
#         fname_output = write_derivatives_yaml(
#             self._derivs_set,
#             self._zdd,
#             filename=filename,
#         )
#         if self._verbose:
#             if fname_output is None:
#                 print("No result file is not generated.", flush=True)
#             else:
#                 print(fname_output, "is generated.", flush=True)
#
#         return self
#
#     def load_derivatives(self, filename: str = "pyclupan_derivatives.yaml"):
#         """Load derivatives.yaml.
#
#         Parameter
#         ---------
#         filename: Single YAML file or multiple YAML files for derivative structures.
#         """
#         from pyclupan.derivative.derivative_utils import load_derivatives_yaml
#
#         if isinstance(filename, str):
#             self._derivs_set = load_derivatives_yaml(filename)
#         elif isinstance(filename, (list, tuple, np.ndarray)):
#             self._derivs_set = DerivativesSet([])
#             for f in filename:
#                 ds = load_derivatives_yaml(f)
#                 self._derivs_set.append(ds)
#
#         return self
#
#     def sample_derivatives(
#         self,
#         method: Literal["all", "uniform", "random"] = "uniform",
#         n_samples: int = 100,
#         path: str = "poscars",
#         elements: tuple = ("Al", "Cu"),
#     ):
#         """Parse derivatives.yaml.
#
#         Parameters
#         ----------
#         method: Sampling method.
#                 Methods of "all", "uniform", and "random" are available.
#         n_samples: Number of sample structures.
#         path: Directory path for saving structure files.
#         elements: Element strings used to save structure files.
#         """
#         if self._derivs_set is None:
#             raise RuntimeError("Derivative structures not found.")
#
#         run_sampling_derivatives(
#             ds_set=self._derivs_set,
#             n_samples=n_samples,
#             method=method,
#             path_poscars=path,
#             element_strings=elements,
#             save_poscars=True,
#         )
#
#     @property
#     def derivative_structures(self):
#         """Return derivative structures.
#
#         Return
#         ------
#         deriv_set: Instance of DerivativeSet class.
#         """
#         return self._derivs_set
#
#     @property
#     def clusters(self):
#         """Return nonequivalent clusters.
#
#         Return
#         ------
#         clusters: Nonequivalent clusters, dict[list[ClusterAttr]].
#                   Dictionary keys are cluster orders.
#         """
#         return self._clusters
