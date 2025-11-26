"""Class for calculating cluster functions."""

import numpy as np

from pyclupan.cluster.cluster_io import load_clusters_yaml
from pyclupan.core.cell_utils import (
    get_unitcell_reps,
    is_cell_equal,
    reduced,
    supercell_reduced,
)
from pyclupan.core.labelings_utils import get_complete_labelings
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
from pyclupan.features.orbit_utils import find_orbit


def calc_correlation(
    lattice_unitcell: Lattice,
    lattice_supercell: Lattice,
    labelings: np.ndarray,
    clusters: list,
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
    rotations, translations = get_symmetry(unitcell)

    # import time
    # t1 = time.time()
    orbit_all = []
    for cl in clusters:
        orbit = find_orbit(
            cl,
            unitcell,
            supercell,
            rotations,
            translations,
            map_unit_to_sup,
            return_array=True,
        )
        orbit = lattice_supercell.to_active_site_rep(orbit)
        orbit_all.append(orbit)
    # t2 = time.time()

    spins = lattice_supercell.to_spins(labelings)
    cluster_functions = []
    for cl in spin_basis_clusters:
        orbit = orbit_all[cl.cluster_id]
        coeffs = lattice_supercell.get_spin_polynomials(cl.spin_basis)
        cf = eval_cluster_functions(coeffs, spins[:, orbit])
        cluster_functions.append(cf)
    cluster_functions = np.array(cluster_functions).T

    # t3 = time.time()
    # print(t2 - t1, t3-t2)
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
        self._verbose = verbose

        self._cluster_functions = None
        self._n_atoms_array = None

        self._lattice_supercell = None
        self._active_labelings = None
        self._structures = None
        self._element_strings = None
        self._derivatives = None

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
        supercell = supercell_reduced(unitcell, supercell_matrix=supercell_matrix)
        self._lattice_supercell = self._lattice_unitcell.lattice_supercell(supercell)
        self._active_labelings = active_labelings

        inactive_labeling = []
        for i, n in enumerate(self._lattice_supercell.cell.n_atoms):
            elements = self._lattice_supercell.elements_on_lattice[i]
            if len(elements) == 1:
                inactive_labeling.extend([elements[0] for j in range(n)])

        complete_labelings = get_complete_labelings(
            active_labelings,
            inactive_labeling,
            self._lattice_supercell.active_sites,
            self._lattice_supercell.inactive_sites,
        )
        self._n_atoms_array = get_chemical_compositions(
            labelings=complete_labelings,
            n_elements=self._lattice_supercell.n_elements,
        )
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
            labelings = np.array([labeling])[:, labelings_order]
            active_labelings = labelings[:, lattice_supercell.active_sites]
            n_atoms = get_chemical_compositions(
                labelings=labelings,
                n_elements=lattice_supercell.n_elements,
            )[0]
            self._n_atoms_array.append(n_atoms)

            cf = calc_correlation(
                self._lattice_unitcell,
                lattice_supercell,
                active_labelings,
                self._clusters,
                self._spin_basis_clusters,
                verbose=self._verbose,
            )
            self._cluster_functions.extend(cf)
        self._cluster_functions = np.array(self._cluster_functions)
        return self._cluster_functions

    def _eval_from_labelings(self):
        """Evaluate cluster functions from labelings."""
        self._cluster_functions = calc_correlation(
            self._lattice_unitcell,
            self._lattice_supercell,
            self._active_labelings,
            self._clusters,
            self._spin_basis_clusters,
            verbose=self._verbose,
        )
        return self._cluster_functions

    def _eval_from_derivatives(self):
        """Evaluate cluster functions from derivative structures."""
        self._cluster_functions = []
        self._n_atoms_array = []
        for d in self._derivatives:
            supercell = supercell_reduced(
                d.unitcell, supercell_matrix=d.supercell_matrix
            )
            lattice_supercell = self._lattice_unitcell.lattice_supercell(supercell)
            cf = calc_correlation(
                self._lattice_unitcell,
                lattice_supercell,
                d.active_labelings,
                self._clusters,
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
        return self._cluster_functions

    @property
    def structures(self):
        """Return structures."""
        return self._structures

    @structures.setter
    def structures(self, st: list[PolymlpStructure]):
        """Setter of structures."""
        self._structures = st

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
    def derivatives(self):
        """Return set of derivative class instance."""
        return self._derivatives

    @derivatives.setter
    def derivatives(self, derivs: DerivativesSet):
        """Setter of derivative class instance."""
        self._derivatives = derivs

    @property
    def cluster_functions(self):
        """Return cluster functions."""
        return self._cluster_functions

    @property
    def active_labelings(self):
        """Return active labelings."""
        return self._active_labelings

    @property
    def complete_labelings(self):
        """Return complete labelings."""
        # TODO: inactive_labeling
        inactive_labeling = 1
        return get_complete_labelings(
            self._active_labelings,
            inactive_labeling,
            self._lattice_supercell.active_sites,
            self._lattice_supercell.inactive_sites,
        )

    @property
    def lattice_unitcell(self):
        """Return lattice in unitcell representation."""
        return self._lattice_unitcell

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

    @property
    def n_atoms_array(self):
        """Return numbers of atoms in structures."""
        return self._n_atoms_array


# def _check_cluster_attrs(
#    clusters_yaml: str = "pyclupan_cluster.yaml",
#    lattice: Optional[Lattice] = None,
#    clusters: Optional[list] = None,
#    spin_basis_clusters: Optional[list] = None,
# ):
#    """Check cluster attributes."""
#    if lattice is None:
#        lattice, clusters, _, spin_basis_clusters = load_clusters_yaml(clusters_yaml)
#        return lattice, clusters, spin_basis_clusters
#
#    if clusters is None:
#        raise RuntimeError("Cluster attributes required.")
#    if spin_basis_clusters is None:
#        raise RuntimeError("Spin-basis cluster attributes required.")
#    return lattice, clusters, spin_basis_clusters
#
#
# def run_correlation_from_structures(
#    structures: list[PolymlpStructure],
#    element_strings: tuple,
#    clusters_yaml: str = "pyclupan_cluster.yaml",
#    lattice: Optional[Lattice] = None,
#    clusters: Optional[list] = None,
#    spin_basis_clusters: Optional[list] = None,
#    verbose: bool = False,
# ):
#    """Calculate cluster functions from derivative structure."""
#    lattice, clusters, spin_basis_clusters = _check_cluster_attrs(
#        clusters_yaml,
#        lattice=lattice,
#        clusters=clusters,
#        spin_basis_clusters=spin_basis_clusters,
#    )
#
#    cluster_functions = []
#    for st in structures:
#        supercell_matrix = np.linalg.inv(lattice.axis) @ st.axis
#        if not np.allclose(supercell_matrix - np.round(supercell_matrix), 0.0):
#            raise RuntimeError("Axis of given structure not consistent with lattice.")
#
#        supercell, tmat = reduced(st, return_transformation=True)
#        supercell.supercell_matrix = supercell_matrix @ tmat
#        labeling = element_strings_to_labeling(supercell.elements, element_strings)
#
#        lattice_supercell, labelings_order = structure_to_lattice(
#            supercell,
#            lattice,
#            only_active=True,
#        )
#        labelings = np.array([labeling])[:, labelings_order]
#
#        cf = calc_correlation(
#            lattice,
#            lattice_supercell,
#            labelings,
#            clusters,
#            spin_basis_clusters,
#            verbose=verbose,
#        )
#        cluster_functions.extend(cf)
#    return np.array(cluster_functions)
#
#
# def run_correlation(
#    unitcell: PolymlpStructure,
#    supercell_matrix: np.ndarray,
#    labelings: np.ndarray,
#    clusters_yaml: str = "pyclupan_cluster.yaml",
#    lattice: Optional[Lattice] = None,
#    clusters: Optional[list] = None,
#    spin_basis_clusters: Optional[list] = None,
#    verbose: bool = False,
# ):
#    """Calculate cluster functions.
#
#    Parameters
#    ----------
#    unitcell: Unitcell.
#    supercell_matrix: Supercell matrix.
#    labelings: Element labelings in supercell. Only active labelings should be given.
#    clusters_yaml: Name of output file for cluster search results.
#
#    Return
#    ------
#    cluster_functions: Cluster functions for labelings.
#        shape: (n_labeling, n_features)
#    """
#    lattice, clusters, spin_basis_clusters = _check_cluster_attrs(
#        clusters_yaml,
#        lattice=lattice,
#        clusters=clusters,
#        spin_basis_clusters=spin_basis_clusters,
#    )
#
#    if not is_cell_equal(unitcell, lattice.cell):
#        raise RuntimeError("Unitcell in cluster.yaml is not equal to given unitcell.")
#
#    supercell = supercell_reduced(unitcell, supercell_matrix=supercell_matrix)
#    lattice_supercell = lattice.lattice_supercell(supercell)
#
#    cluster_functions = calc_correlation(
#        lattice,
#        lattice_supercell,
#        labelings,
#        clusters,
#        spin_basis_clusters,
#        verbose=verbose,
#    )
#    return cluster_functions
#
