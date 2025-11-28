"""Class for calculating cluster functions."""

# from typing import Optional

import numpy as np

from pyclupan.core.cell_utils import get_unitcell_reps
from pyclupan.core.lattice import Lattice
from pyclupan.core.spin import eval_cluster_functions
from pyclupan.features.orbit_utils import find_orbit_supercell, get_map_positions
from pyclupan.features.run_correlation import ClusterFunctions


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

        self._orbit_sizes = None
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
                    return_array=False,
                )
                # TODO: Check results when active sites and entire sites are different.
                for k, v in orbit.items():
                    orbit[k] = self._lattice_supercell.to_active_site_rep(v)
                self._orbit_sites_supercell[i] = orbit
        return self

    def eval_from_spins(self, active_spins: np.ndarray):
        """Evaluate cluster functions from active labelings."""
        self._cluster_functions = []
        self._orbit_sizes = []
        for cl in self._spin_basis_clusters:
            coeffs = self._lattice_supercell.get_spin_polynomials(cl.spin_basis)
            orbit = self._orbit_sites_supercell[cl.cluster_id]
            if isinstance(orbit, dict):
                orbit_array = []
                for v in orbit.values():
                    orbit_array.extend(v)
                orbit = np.array(orbit_array)
            self._orbit_sizes.append(orbit.shape[0])

            cf = eval_cluster_functions(coeffs, active_spins[orbit])
            self._cluster_functions.append(cf)
        self._cluster_functions = np.array(self._cluster_functions)
        return self._cluster_functions

    def eval_from_spin_swap(
        self,
        active_spins: np.array,
        sites: np.array,
    ):
        """Evaluate cluster function changes from spin swap."""
        if len(sites) != 2:
            raise RuntimeError("Size of sites must be two.")

        diff_cluster_functions = []
        for spin_cl_id, cl in enumerate(self._spin_basis_clusters):
            coeffs = self._lattice_supercell.get_spin_polynomials(cl.spin_basis)
            cluster_size = coeffs.shape[0]
            orbit = self._orbit_sites_supercell[cl.cluster_id]
            i, j = sites[0], sites[1]

            cf_new, cf_old = 0.0, 0.0
            for s in sites:
                cf_old += eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[s]],
                    return_sum=True,
                )
            active_spins[i], active_spins[j] = active_spins[j], active_spins[i]
            for s in sites:
                cf_new += eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[s]],
                    return_sum=True,
                )
            active_spins[i], active_spins[j] = active_spins[j], active_spins[i]
            # TODO: Check when more than binary.
            diff_cf = (cf_new - cf_old) / self._orbit_sizes[spin_cl_id] * cluster_size
            diff_cluster_functions.append(diff_cf)
        diff_cluster_functions = np.array(diff_cluster_functions)
        return diff_cluster_functions
