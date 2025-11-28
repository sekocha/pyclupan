"""Class for calculating cluster functions."""

# from typing import Optional

import numpy as np

from pyclupan.core.cell_utils import get_unitcell_reps
from pyclupan.core.lattice import Lattice
from pyclupan.core.spin import eval_cluster_functions
from pyclupan.features.orbit_utils import find_orbit_supercell, get_map_positions
from pyclupan.features.run_correlation import ClusterFunctions

# from pyclupan.cluster.cluster_io import load_clusters_yaml
# from pyclupan.core.cell_utils import (
#     get_unitcell_reps,
#     is_cell_equal,
#     reduced,
#     supercell_reduced,
# )
# from pyclupan.core.pypolymlp_utils import PolymlpStructure
# from pyclupan.core.spglib_utils import get_symmetry
# from pyclupan.core.spin import eval_cluster_functions
# from pyclupan.derivative.derivative_utils import DerivativesSet
# from pyclupan.features.features_utils import (
#     element_strings_to_labeling,
#     get_chemical_compositions,
#     structure_to_lattice,
# )


class ClusterFunctionsMC:
    """Class for calculating cluster functions in MC."""

    def __init__(
        self,
        cf: ClusterFunctions,
        lattice_supercell: Lattice,
        verbose: bool = False,
    ):
        """Init method."""
        self._spin_basis_clusters = cf.spin_basis_clusters
        self._lattice_unitcell = cf.lattice_unitcell
        self._lattice_supercell = lattice_supercell
        self._verbose = verbose

        if cf._orbit_fracs_unitcell is None:
            raise RuntimeError("Orbit for unitcell not found.")

        self._orbit_unitcell = cf._orbit_fracs_unitcell
        self._mask_clusters = cf._mask_clusters

        self._orbit_sites_supercell = None
        self._get_orbit_supercell()

        self._cluster_functions = None

    def _get_orbit_supercell(self, decimals: int = 5):
        """Return orbit for supercell."""
        unitcell = self._lattice_unitcell.cell
        supercell = self._lattice_supercell.cell
        map_unit_to_sup = get_unitcell_reps(unitcell, supercell)
        map_supercell_positions = get_map_positions(supercell, decimals=decimals)

        orbit_unitcell = self._orbit_unitcell
        self._orbit_sites_supercell = [None for _ in orbit_unitcell]
        for i, (orbit_f, mask) in enumerate(zip(orbit_unitcell, self._mask_clusters)):
            if mask:
                if self._verbose:
                    print("Calculating orbits for cluster", i, flush=True)

                orbit = find_orbit_supercell(
                    self._lattice_unitcell,
                    self._lattice_supercell,
                    orbit_f,
                    map_unit_to_sup,
                    map_supercell_positions=map_supercell_positions,
                    return_array=True,
                )
                orbit = self._lattice_supercell.to_active_site_rep(orbit)
                self._orbit_sites_supercell[i] = orbit
        return self

    def eval_from_spins(self, active_spins: np.ndarray):
        """Evaluate cluster functions from active labelings."""
        self._cluster_functions = []
        for cl in self._spin_basis_clusters:
            orbit = self._orbit_sites_supercell[cl.cluster_id]
            coeffs = self._lattice_supercell.get_spin_polynomials(cl.spin_basis)
            cf = eval_cluster_functions(coeffs, active_spins[orbit])
            self._cluster_functions.append(cf)
        self._cluster_functions = np.array(self._cluster_functions)
        return self._cluster_functions


#     def eval_from_spin_swap(
#         self,
#         lattice_supercell: Lattice,
#         sites: np.array,
#         spins: np.array,
#     ):
#         """Evaluate cluster function changes from spin swap."""
#         self._cluster_functions = []
#         for cl in self._spin_basis_clusters:
#             orbit = self._orbit_sites_supercell[cl.cluster_id]
#             coeffs = lattice_supercell.get_spin_polynomials(cl.spin_basis)
#             cf = eval_cluster_functions(coeffs, active_spins[:, orbit])
#             self._cluster_functions.append(cf)
#         self._cluster_functions = np.array(self._cluster_functions).T
#         return self._cluster_functions
#
#
