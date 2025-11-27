"""Class for performing Monte Carlo simulations."""

from typing import Optional

import numpy as np

from pyclupan.core.cell_utils import supercell, supercell_diagonal
from pyclupan.core.spglib_utils import refine_cell
from pyclupan.features.run_correlation import ClusterFunctions
from pyclupan.mc.mc_utils import MCAttr
from pyclupan.regression.regression_utils import load_ecis


class MC:
    """Class for performing Monte Carlo simulations."""

    def __init__(
        self,
        clusters_yaml: str = "pyclupan_clusters.yaml",
        ecis_yaml: str = "pyclupan_ecis.yaml",
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        clusters_yaml: File of cluster attributes from cluster search.
        ecis_yaml: File of ECIs from regression.
        """
        self._verbose = verbose
        self._cf = ClusterFunctions(clusters_yaml=clusters_yaml, verbose=verbose)
        self._model = load_ecis(ecis_yaml)
        self._cf.spin_basis_clusters = self._model.nonzero_spin_basis(
            self._cf.spin_basis_clusters
        )

        self._lattice_unitcell = self._cf.lattice_unitcell
        self._lattice_supercell = None
        self._mc_attr = MCAttr()
        np.set_printoptions(legacy="1.21")

    def _set_init_structure_random(self, compositions: tuple):
        """Set initial structure randomly."""
        if not np.isclose(np.sum(compositions), 1.0):
            raise RuntimeError("Sum of given compositions is not one.")

        n_sites = len(self._lattice_supercell.active_sites)
        elements = self._lattice_supercell._active_elements
        n_atoms = np.array([compositions[ele] * n_sites for ele in elements])
        if not np.allclose(n_atoms - np.round(n_atoms), 0.0):
            raise RuntimeError("Given supercell cannot express compositions.")

        n_atoms = np.round(n_atoms).astype(int)
        if self._verbose:
            print("Number of active sites:", n_sites, flush=True)
            for e, n in zip(elements, n_atoms):
                print("- Active element", e, ":", n, flush=True)

        perm = np.random.permutation(n_sites)
        active_labelings = np.ones(n_sites, dtype=int) * -1
        begin = 0
        for ele, n in zip(elements, n_atoms):
            active_labelings[perm[begin : begin + n]] = ele
            begin += n
        assert np.all(active_labelings != -1)

        active_spins = self._lattice_supercell.to_spins(np.array([active_labelings]))[0]
        return active_spins

    def set_init_structure(self, compositions: Optional[tuple] = None):
        """Set initial structure."""
        if self._lattice_supercell is None:
            raise RuntimeError("Set supercell first.")

        if compositions is not None:
            active_spins = self._set_init_structure_random(compositions)
        else:
            pass

        self._mc_attr.active_spins = active_spins
        return self

    def set_init(self, compositions: Optional[tuple] = None):
        """Set initial conditions.

        Parameters
        ----------
        compositions: Compositions for active elements.
            Array indices correspond to element IDs.
            The compositions are defined as
            (number of atoms) / (number of active sites).
        """
        if self._lattice_supercell is None:
            raise RuntimeError("Set supercell first.")

        self.set_init_structure(compositions=compositions)

        if self._verbose:
            print("Constructing cluster orbits in supercell.", flush=True)
        self._cf.get_orbit_supercell(self._lattice_supercell)

        if self._verbose:
            print("Calculating cluster functions of initial structure.", flush=True)
        cluster_functions = self._cf.eval_from_labelings(
            self._lattice_supercell,
            active_spins=np.array([self._mc_attr.active_spins]),
        )[0]
        energy = self._model.eval(cluster_functions)
        if self._verbose:
            print("Initial structures:", flush=True)
            print("- Cluster functions:", flush=True)
            print(cluster_functions, flush=True)
            print("- Energy:", energy, flush=True)

        self._mc_attr.cluster_functions = cluster_functions
        self._mc_attr.energy = energy
        return self

    def set_supercell(self, supercell_matrix: np.ndarray, refine: bool = False):
        """Set supercell.

        Parameters
        ----------
        supercell_matrix: Supercell matrix.
            If three elements are given, a diagonal supercell matrix of these
            elements will be used.
        refine: Refine unitcell before applying supercell matrix. Default: False.
            If True, a supercell is constructed by the expansion of given supercell
            matrix for the refined cell.
        """
        if self._verbose:
            print("Constructing supercell.", flush=True)

        unitcell = self._lattice_unitcell.cell
        if refine:
            unitcell_rev = refine_cell(unitcell)
            if self._verbose:
                if not np.allclose(unitcell_rev.axis - unitcell.axis, 0.0):
                    print("Unitcell has been refined.", flush=True)
        else:
            unitcell_rev = unitcell

        if np.array(supercell_matrix).size == 9:
            if self._verbose:
                print("Supercell matrix:", flush=True)
                print(supercell_matrix, flush=True)
            sup = supercell(unitcell_rev, supercell_matrix=supercell_matrix)

        elif np.array(supercell_matrix).size == 3:
            if self._verbose:
                print("Diagonal supercell:", supercell_matrix, flush=True)
            sup = supercell_diagonal(unitcell_rev, size=supercell_matrix)

        sup.supercell_matrix = np.linalg.inv(unitcell.axis) @ sup.axis
        self._lattice_supercell = self._lattice_unitcell.lattice_supercell(sup)
        return self

    @property
    def unitcell(self):
        """Return unitcell."""
        return self._lattice_unitcell.cell

    @property
    def supercell(self):
        """Return supercell."""
        return self._lattice_supercell.cell

    @property
    def mc_attr(self):
        """Return attributes from MC."""
        return self._mc_attr


#     @property
#     def structures(self):
#         """Return structures."""
#         return self._structures
