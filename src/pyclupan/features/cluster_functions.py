"""Class for calculating cluster functions."""

import numpy as np

from pyclupan.cluster.cluster_io import load_clusters_yaml
from pyclupan.core.cell_utils import (
    get_unitcell_reps,
    is_cell_equal,
    reduced,
    supercell_reduced,
)
from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.core.spglib_utils import get_symmetry
from pyclupan.core.spin import eval_cluster_functions
from pyclupan.derivative.derivative_utils import DerivativesSet
from pyclupan.features.features_utils import (
    element_strings_to_labeling,
    get_chemical_compositions,
    structure_to_lattice,
)
from pyclupan.features.orbit_utils import (
    find_orbit_supercell,
    find_orbit_unitcell,
    get_map_positions,
)


def calc_correlation(
    lattice_unitcell: Lattice,
    lattice_supercell: Lattice,
    labelings: np.ndarray,
    orbit_fracs_unitcell: list,
    spin_basis_clusters: list,
    verbose: bool = False,
):
    """Calculate cluster functions without loading cluster yaml file."""
    if not lattice_supercell.is_active_size(labelings):
        raise RuntimeError("Size of given labelings not consistent with lattice.")
    if not lattice_supercell.is_active_element(labelings):
        raise RuntimeError("Some of given labelings are not on active lattices.")

    unitcell = lattice_unitcell.cell
    supercell = lattice_supercell.cell
    map_unit_to_sup = get_unitcell_reps(unitcell, supercell)
    map_supercell_positions = get_map_positions(supercell, decimals=5)

    orbit_all = []
    for orbit_f in orbit_fracs_unitcell:
        orbit = find_orbit_supercell(
            unitcell,
            supercell,
            orbit_f,
            map_unit_to_sup,
            map_supercell_positions=map_supercell_positions,
            return_array=True,
        )
        orbit = lattice_supercell.to_active_site_rep(orbit)
        orbit_all.append(orbit)

    spins = lattice_supercell.to_spins(labelings)
    cluster_functions = []
    for cl in spin_basis_clusters:
        orbit = orbit_all[cl.cluster_id]
        coeffs = lattice_supercell.get_spin_polynomials(cl.spin_basis)
        cf = eval_cluster_functions(coeffs, spins[:, orbit])
        cluster_functions.append(cf)
    cluster_functions = np.array(cluster_functions).T
    return cluster_functions


class ClusterFunctions:
    """Class for calculating cluster functions."""

    def __init__(
        self,
        clusters_yaml: str = "pyclupan_clusters.yaml",
        verbose: bool = False,
    ):
        """Init method.

        Parameter
        ---------
        clusters_yaml: Name of output file for cluster search results.
        """
        cluster_attrs = load_clusters_yaml(clusters_yaml)
        self._lattice_unitcell = cluster_attrs[0]
        self._clusters = cluster_attrs[1]
        self._spin_basis_clusters = cluster_attrs[3]
        self._mask_clusters = np.ones(len(self._clusters), dtype=bool)
        self._verbose = verbose

        self._cluster_functions = None
        self._n_atoms_array = None

        self._lattice_supercell = None
        self._active_labelings = None
        self._structures = None
        self._element_strings = None
        self._derivatives = None

        self._orbit_frac_unitcell = None
        self._orbit_sites_supercell = None

        self._eval_unitcell_attrs()

    def _eval_unitcell_attrs(self):
        """Evaluate required attributes for unitcell.."""
        unitcell = self._lattice_unitcell.cell
        rotations, translations = get_symmetry(unitcell)
        self._orbit_fracs_unitcell = []
        for cl in self._clusters:
            _, orbit_fracs = find_orbit_unitcell(cl, unitcell, rotations, translations)
            self._orbit_fracs_unitcell.append(orbit_fracs)
        return self

    def clear_structures(self):
        """Clear structures."""
        self._structures = None
        self._derivatives = None
        self._active_labelings = None
        return self

    def set_labelings(
        self,
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
        active_labelings: np.ndarray,
    ):
        """Calculate cluster functions.

        Parameters
        ----------
        unitcell: Unitcell.
        supercell_matrix: Supercell matrix.
        active_labelings: Element labelings in supercell.
                          Only active labelings should be given.
        """
        if not is_cell_equal(unitcell, self._lattice_unitcell.cell):
            raise RuntimeError(
                "Unitcell in cluster.yaml is not equal to given unitcell."
            )
        if not self._lattice_unitcell.is_active_element(active_labelings):
            raise RuntimeError("Some labels are not active.")

        supercell = supercell_reduced(unitcell, supercell_matrix=supercell_matrix)
        self._lattice_supercell = self._lattice_unitcell.lattice_supercell(supercell)
        self._active_labelings = active_labelings

        if not self._lattice_supercell.is_active_size(active_labelings):
            raise RuntimeError("Size of active labelings is not size of active sites.")

        return self

    def _eval_from_structures(self):
        """Evaluate cluster functions from structures."""
        if self._structures is None:
            raise RuntimeError("Structures are required.")
        if self._element_strings is None:
            raise RuntimeError("Labels for element strings are required.")

        self._cluster_functions = []
        self._n_atoms_array = []
        for st in self._structures:
            supercell_matrix = np.linalg.inv(self._lattice_unitcell.axis) @ st.axis
            if not np.allclose(supercell_matrix - np.round(supercell_matrix), 0.0):
                raise RuntimeError(
                    "Axis of given structure not consistent with lattice."
                )

            supercell, tmat = reduced(st, return_transformation=True)
            supercell.supercell_matrix = supercell_matrix @ tmat
            labeling = element_strings_to_labeling(
                supercell.elements, self._element_strings
            )

            lattice_supercell, labelings_order = structure_to_lattice(
                supercell,
                self._lattice_unitcell,
            )
            complete_labelings = np.array([labeling])[:, labelings_order]
            active_labelings = complete_labelings[:, lattice_supercell.active_sites]

            cf = calc_correlation(
                self._lattice_unitcell,
                lattice_supercell,
                active_labelings,
                self._orbit_fracs_unitcell,
                self._spin_basis_clusters,
                verbose=self._verbose,
            )
            self._cluster_functions.extend(cf)
            n_atoms = get_chemical_compositions(
                labelings=complete_labelings,
                n_elements=lattice_supercell.n_elements,
            )[0]
            self._n_atoms_array.append(n_atoms)

        self._cluster_functions = np.array(self._cluster_functions)
        return self._cluster_functions

    def _eval_from_labelings(self):
        """Evaluate cluster functions from labelings."""
        self._cluster_functions = calc_correlation(
            self._lattice_unitcell,
            self._lattice_supercell,
            self._active_labelings,
            self._orbit_fracs_unitcell,
            self._spin_basis_clusters,
            verbose=self._verbose,
        )
        complete_labelings = self._lattice_supercell.complete_labelings(
            self._active_labelings
        )
        self._n_atoms_array = get_chemical_compositions(
            labelings=complete_labelings,
            n_elements=self._lattice_supercell.n_elements,
        )
        return self._cluster_functions

    def _eval_from_derivatives(self):
        """Evaluate cluster functions from derivative structures."""
        self._cluster_functions = []
        self._n_atoms_array = []
        for d in self._derivatives:
            if self._verbose:
                print("Supercell Size:", d.supercell_size, flush=True)
                print("- Supercell:", flush=True)
                print(d.supercell_matrix, flush=True)
                print("- n_labelings:", d.active_labelings.shape[0], flush=True)

            supercell = supercell_reduced(
                d.unitcell, supercell_matrix=d.supercell_matrix
            )
            lattice_supercell = self._lattice_unitcell.lattice_supercell(supercell)

            cf = calc_correlation(
                self._lattice_unitcell,
                lattice_supercell,
                d.active_labelings,
                self._orbit_fracs_unitcell,
                self._spin_basis_clusters,
                verbose=self._verbose,
            )
            self._cluster_functions.extend(cf)
            n_atoms = get_chemical_compositions(
                labelings=d.get_complete_labelings(),
                n_elements=lattice_supercell.n_elements,
            )
            self._n_atoms_array.extend(n_atoms)
        self._cluster_functions = np.array(self._cluster_functions)

        if self._verbose:
            print("Cluster Function Size:", self._cluster_functions.shape, flush=True)
        return self._cluster_functions

    def eval(self):
        """Evaluate cluster functions."""
        if self._active_labelings is not None:
            if self._verbose:
                print("Evaluating cluster functions from labelings.", flush=True)
            self._cluster_functions = self._eval_from_labelings()
        elif self._structures is not None:
            if self._verbose:
                print("Evaluating cluster functions from structures.", flush=True)
            self._cluster_functions = self._eval_from_structures()
        elif len(self._derivatives) > 0:
            if self._verbose:
                print("Evaluating cluster functions from derivatives.", flush=True)
            self._cluster_functions = self._eval_from_derivatives()
        else:
            raise RuntimeError("Structures or labelings not found.")
        return self._cluster_functions

    @property
    def element_strings(self):
        """Return element strings."""
        return self._element_strings

    @element_strings.setter
    def element_strings(self, element_strings: tuple):
        """Set element strings.

        Parameter
        ---------
        element_strings: Element strings.
            The location index corresponds to label integer.
            For example, element_strings are ("Ag", "Au"),
            labels 0 and 1 indicate elements Ag and Au, respectively.
        """
        self._element_strings = element_strings

    @property
    def structures(self):
        """Return structures."""
        return self._structures

    @structures.setter
    def structures(self, st: list[PolymlpStructure]):
        """Setter of structures."""
        self._structures = st

    @property
    def derivatives(self):
        """Return set of derivative class instance."""
        return self._derivatives

    @derivatives.setter
    def derivatives(self, derivs: DerivativesSet):
        """Setter of derivative class instance."""
        self._derivatives = derivs

    @property
    def clusters(self):
        """Return cluster attributes."""
        return self._clusters

    @property
    def spin_basis_clusters(self):
        """Return spin basis clusters."""
        return self._spin_basis_clusters

    @spin_basis_clusters.setter
    def spin_basis_clusters(self, sp_clusters: list):
        """Setter of spin basis clusters."""
        self._spin_basis_clusters = sp_clusters
        self._mask_clusters = np.zeros(len(self._clusters), dtype=bool)
        for cl in sp_clusters:
            self._mask_clusters[cl.cluster_id] = True

    @property
    def n_atoms_array(self):
        """Return numbers of atoms in structures."""
        return np.array(self._n_atoms_array)

    @n_atoms_array.setter
    def n_atoms_array(self, n_atoms_array: np.ndarray):
        """Setter of numbers of atoms in structures."""
        self._n_atoms_array = n_atoms_array

    @property
    def lattice_unitcell(self):
        """Return lattice in unitcell representation."""
        return self._lattice_unitcell

    @property
    def cluster_functions(self):
        """Return cluster functions."""
        return self._cluster_functions
