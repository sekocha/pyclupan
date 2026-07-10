"""API Class for generating CE model using polymlp."""

from typing import Optional

import numpy as np

from pyclupan.api.pyclupan import Pyclupan
from pyclupan.api.pyclupan_calc_cf import PyclupanCalcFeatures
from pyclupan.api.pyclupan_calc_model import PyclupanCalcModel

# from pyclupan.api.pyclupan_calc import PyclupanCalc
from pyclupan.api.pyclupan_regression import PyclupanRegression
from pyclupan.core.pypolymlp_utils import Polymlp
from pyclupan.derivative.derivative_utils import DerivativesSet


class PyclupanCE:
    """API Class for generating CE model using polymlp."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._pyclupan = Pyclupan(verbose=verbose)
        self._pyclupan_features = None
        self._pyclupan_model = None
        self._pyclupan_reg = None

        self._unitcell = None
        self._elements = None
        self._element_strings = None
        np.set_printoptions(legacy="1.21")

        self._sampled_structures = []
        self._ds_set = DerivativesSet([])

        self._X = None
        self._y = None
        self._structure_ids = None
        self._success_go = None

        self._model = None
        self._models = None

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
        elements: Element IDs on lattices.
                  Example: [[0], [1], [2, 3]].
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
        comp: Optional[list] = None,
        comp_lb: Optional[list] = None,
        comp_ub: Optional[list] = None,
        end_members: bool = True,
        superperiodic: bool = False,
    ):
        """Enumerate derivative structures.

        Parameters
        ----------
        comp: Compositions for sublattices (n_elements / n_sites).
              Compositions are not needed to be normalized.
              Format: [(element ID, composition), (element ID, composition),...]
        comp_lb: Lower bounds of compositions for sublattices.
              Format: [(element ID, composition), (element ID, composition),...]
        comp_ub: Upper bounds of compositions for sublattices.
              Format: [(element ID, composition), (element ID, composition),...]
        end_members: Include structures of end members.
        superperiodic: Include superperiodic derivative structures.
        """
        for size in range(min_supercell_size, max_supercell_size + 1):
            self._pyclupan.run_derivative(
                elements=self._elements,
                supercell_size=size,
                end_members=end_members,
                superperiodic=superperiodic,
                comp=comp,
                comp_lb=comp_lb,
                comp_ub=comp_ub,
            )
            filename = "pyclupan_derivative_" + str(size) + ".yaml"
            self._pyclupan.save_derivatives(filename=filename)
            self._ds_set.append(self._pyclupan.derivative_structures)

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
            # self._ds_set_sample.append(ds_set)
            self._sampled_structures.extend(structures)
        return self

    def eval_energies(self, pot: Optional[str] = "polymlp.yaml"):
        """Evaluate energies using polymlp."""
        if self._sampled_structures is None:
            raise RuntimeError("Sampled structures not found.")

        self._success_go = np.ones(len(self._sampled_structures), dtype=bool)
        n_atom_unitcell = len(self._unitcell.elements)

        polymlp = Polymlp(pot=pot)
        energies = []
        for i, st in enumerate(self._sampled_structures):
            suc = polymlp.run_geometry_optimization(st, gtol=1e-4)
            if not suc:
                self._success_go[i] = False
                continue

            n_atom = len(polymlp.structure.elements)
            n_unitcell = n_atom / n_atom_unitcell
            energies.append(polymlp.energy / n_unitcell)

        self._y = np.array(energies)
        return self._y

    def enum_cluster(
        self,
        max_order: int = 4,
        cutoffs: tuple[float] = (6.0, 6.0, 6.0),
        filename: str = "pyclupan_cluster.yaml",
    ):
        """Enumerate nonequivalent clusters.
        Parameters
        ----------
        max_order: Maximum order of clusters.
        cutoffs: Cutoff distances for orders >= 2.
                (two-body, three-body, four-body, ...)
                Size of cutoffs must be equal to max_order - 1.
                Cutoffs must be smaller or equal to those for smaller orders.
        filename: Name of output file for cluster search results.
        """
        if self._elements is None:
            raise RuntimeError("Elements not given.")

        self._pyclupan.run_cluster(
            elements=self._elements,
            max_order=max_order,
            cutoffs=cutoffs,
            filename=filename,
        )

        self._pyclupan_features = PyclupanCalcFeatures(
            clusters_yaml=filename,
            verbose=False,
        )
        self._pyclupan_model = PyclupanCalcModel(
            clusters_yaml=filename,
            verbose=False,
        )
        return self

    def eval_cluster_functions(self):
        """Evaluate cluster functions for enumerated derivative structures."""
        if self._pyclupan_features is None:
            raise RuntimeError("Feature class not found.")

        self._pyclupan_features.derivatives = self._ds_set
        cluster_functions = self._pyclupan_features.eval_cluster_functions()
        self._structure_ids = self._pyclupan_features.structure_indices
        # self._structure_ids = [
        #     "-".join([str(i) for i in ids])
        #     for ds in self._ds_set_sample
        #     for ids in ds.all_ids
        # ]
        return cluster_functions

    def eval_predictor_matrix(self):
        """Evaluate predictor matrix for enumerated derivative structures."""
        if self._success_go is None:
            raise RuntimeError("Results from geometry optimization not found.")

        cluster_functions = self.eval_cluster_functions()
        self._X = cluster_functions[self._success_go]
        self._structure_ids = np.array(self._structure_ids)[self._success_go]
        return self

    def eval_ecis(self):
        """Evaluate effective cluster interactions."""
        if self._X is None:
            raise RuntimeError("Matrix X not calculated.")
        if self._y is None:
            raise RuntimeError("Vector y not calculated.")

        self._pyclupan_reg = PyclupanRegression(verbose=self._verbose)
        self._pyclupan_reg.x = self._X
        self._pyclupan_reg.y = self._y
        self._pyclupan_reg.structure_ids = self._structure_ids

        self._pyclupan_reg.run_lasso()
        self._pyclupan_reg.save_predictions()
        self._pyclupan_reg.save()

        self._model = self._pyclupan_reg.best_model
        self._models = self._pyclupan_reg.models
        self._pyclupan_model.model = self._model
        return self

    def eval_ce_energies(self):
        """Evaluate energies of CE model."""
        if self._pyclupan_model is None:
            raise RuntimeError("CE calculation model class not provided.")
        if self._model is None:
            raise RuntimeError("CE calculation model not provided.")

        self._pyclupan_model.derivatives = self._ds_set
        self._pyclupan_model.eval_cluster_functions()
        self._energies = self._pyclupan_model.eval_energies()
        self._structure_ids = self._pyclupan_model.structure_indices
        return self._energies

    def eval_ce_formation_energies(self):
        """Evaluate formation energies of CE model."""
        if self._pyclupan_model is None:
            raise RuntimeError("CE calculation model class not provided.")
        if self._model is None:
            raise RuntimeError("CE calculation model not provided.")

        self._pyclupan_model.derivatives = self._ds_set
        self._pyclupan_model.eval_cluster_functions()
        self._formation_energies = self._pyclupan_model.eval_formation_energies()
        self._structure_ids = self._pyclupan_model.structure_indices


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
