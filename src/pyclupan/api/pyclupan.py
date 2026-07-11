"""API Class for generating CE model using polymlp."""

from typing import Optional

import numpy as np

from pyclupan.api.pyclupan_calc_model import PyclupanCalcModel
from pyclupan.api.pyclupan_cluster import PyclupanCluster
from pyclupan.api.pyclupan_derivatives import PyclupanDerivatives
from pyclupan.api.pyclupan_features import PyclupanCalcFeatures
from pyclupan.api.pyclupan_regression import PyclupanRegression
from pyclupan.core.pypolymlp_utils import Polymlp
from pyclupan.derivative.derivative_utils import DerivativesSet


# TODO: Derivative structure enumeration using uniform and random.
class Pyclupan:
    """API Class for generating CE model using polymlp."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._pyclupan_deriv = PyclupanDerivatives(verbose=verbose)
        self._pyclupan_cluster = PyclupanCluster(verbose=verbose)
        self._pyclupan_features = None
        self._pyclupan_model = None
        self._pyclupan_reg = None

        self._unitcell = None
        self._elements = None
        self._element_strings = None

        self._sampled_structures = []
        self._ds_set = DerivativesSet([])

        self._X = None
        self._y = None
        self._structure_ids = None
        self._success_go = None

        self._model = None
        self._models = None

        self._energies = None
        self._formation_energies = None
        self._compositions = None
        self._convex = None

        np.set_printoptions(legacy="1.21")

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
        self._unitcell = self._pyclupan_deriv.load_poscar(poscar)
        self._pyclupan_cluster.unitcell = self._unitcell
        return self

    def enum_derivatives(
        self,
        min_supercell_size: int = 1,
        max_supercell_size: int = 8,
        n_samples: Optional[int] = None,
        comp: Optional[list] = None,
        comp_lb: Optional[list] = None,
        comp_ub: Optional[list] = None,
        end_members: bool = False,
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
        if self._elements is None:
            raise RuntimeError("Elements not given.")

        for size in range(min_supercell_size, max_supercell_size + 1):
            self._pyclupan_deriv.run_derivative(
                elements=self._elements,
                supercell_size=size,
                end_members=end_members,
                superperiodic=superperiodic,
                comp=comp,
                comp_lb=comp_lb,
                comp_ub=comp_ub,
            )
            filename = "pyclupan_derivative_" + str(size) + ".yaml"
            self._pyclupan_deriv.save_derivatives(filename=filename)
            self._ds_set.append(self._pyclupan_deriv.derivative_structures)

            method = "all" if n_samples is None else "uniform"
            self._pyclupan_deriv.sample_derivatives(
                method=method,
                n_samples=n_samples,
                save_poscars=False,
            )
            structures = self._pyclupan_deriv.get_sampled_structures(
                self._element_strings
            )
            self._sampled_structures.extend(structures)
        return self

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

        self._pyclupan_cluster.run_cluster(
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
        if len(self._ds_set) == 0:
            raise RuntimeError("Derivative structures not found.")

        self._pyclupan_features.derivatives = self._ds_set
        cluster_functions = self._pyclupan_features.eval_cluster_functions()
        self._structure_ids = self._pyclupan_features.structure_indices
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

        self._pyclupan_model.save_energies()
        return self._energies

    def eval_ce_formation_energies(self):
        """Evaluate formation energies of CE model."""
        if self._pyclupan_model is None:
            raise RuntimeError("CE calculation model class not provided.")
        if self._model is None:
            raise RuntimeError("CE calculation model not provided.")
        if self._energies is None:
            raise RuntimeError("CE energy not calculated.")

        res = self._pyclupan_model.eval_formation_energies()
        (self._formation_energies, self._compositions, self._convex) = res
        self._structure_ids = self._pyclupan_model.structure_indices

        self._pyclupan_model.save_formation_energies()
        self._pyclupan_model.save_convex_hull_yaml()
        self._pyclupan_model.save_convex_hull_poscars(self._element_strings)
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

    @property
    def unitcell(self):
        """Return unit cell."""
        return self._unitcell

    @property
    def best_model(self):
        """Return CE best model."""
        return self._model

    @property
    def models(self):
        """Return CE models."""
        return self._models

    @property
    def energies(self):
        """Return CE energies."""
        return self._energies

    @property
    def formation_energies(self):
        """Return CE formation energies."""
        return self._formation_energies

    @property
    def compositions(self):
        """Return compositions of structures."""
        return self._compositions

    @property
    def convex(self):
        """Return convex hull of CE formation energies."""
        return self._convex

    @property
    def structure_indices(self):
        """Return structure indices."""
        return self._ds_set.all_structure_indices
