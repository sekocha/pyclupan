"""Class for calculating cluster functions."""

# from typing import Optional

import copy

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
        self._orbit_sizes = None
        self._duplicate_n_sites = None
        self._cluster_functions = None

        self._get_orbit_supercell()

    def _get_orbit_supercell(self, decimals: int = 5):
        """Return orbit for supercell."""
        unitcell = self._lattice_unitcell.cell
        supercell = self._lattice_supercell.cell
        map_unit_to_sup = get_unitcell_reps(unitcell, supercell)
        map_supercell_positions = get_map_positions(supercell, decimals=decimals)

        orbit_unitcell = self._orbit_unitcell
        self._orbit_sites_supercell = [None for _ in orbit_unitcell]
        self._duplicate_n_sites = dict()
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

                for k, v in orbit.items():
                    # TODO: Check results when active sites
                    #       and entire sites are different.
                    orbit[k] = np.array(self._lattice_supercell.to_active_site_rep(v))
                    n_duplicate = np.sum(orbit[k] == k, axis=1).astype(float)
                    self._duplicate_n_sites[(i, k)] = n_duplicate

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
        # TODO: Check when more than binary.

        i, j = sites[0], sites[1]

        diff_cluster_functions = []
        for spin_cl_id, cl in enumerate(self._spin_basis_clusters):
            coeffs = self._lattice_supercell.get_spin_polynomials(cl.spin_basis)
            cluster_size = coeffs.shape[0]
            orbit = self._orbit_sites_supercell[cl.cluster_id]

            duplicate_i = np.sum(orbit[i] == j, axis=1).astype(float)
            duplicate_i += self._duplicate_n_sites[(cl.cluster_id, i)]
            weight_i = np.reciprocal(duplicate_i) * cluster_size

            duplicate_j = np.sum(orbit[j] == i, axis=1).astype(float)
            duplicate_j += self._duplicate_n_sites[(cl.cluster_id, j)]
            weight_j = np.reciprocal(duplicate_j) * cluster_size

            sum_i = np.sum(duplicate_i)
            sum_j = np.sum(duplicate_j)
            if sum_i == len(duplicate_i) and sum_j == len(duplicate_j):
                spin_i = copy.deepcopy(active_spins[i])
                spin_j = copy.deepcopy(active_spins[j])

                dspin = active_spins[j] - active_spins[i]
                active_spins[i] = dspin
                prods = eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[i]],
                    return_sum=True,
                )
                diff_cf = prods @ weight_i

                active_spins[j] = -dspin
                prods = eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[j]],
                    return_sum=True,
                )
                diff_cf += prods @ weight_j
                active_spins[i], active_spins[j] = spin_i, spin_j
                diff_cf /= self._orbit_sizes[spin_cl_id]
            else:
                cf_new, cf_old = 0.0, 0.0
                prods = eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[i]],
                    return_sum=True,
                )
                cf_old += prods @ weight_i
                prods = eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[j]],
                    return_sum=True,
                )
                cf_old += prods @ weight_j

                active_spins[i], active_spins[j] = active_spins[j], active_spins[i]
                prods = eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[i]],
                    return_sum=True,
                )
                cf_new += prods @ weight_i
                prods = eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[j]],
                    return_sum=True,
                )
                cf_new += prods @ weight_j
                active_spins[i], active_spins[j] = active_spins[j], active_spins[i]

                diff_cf = (cf_new - cf_old) / self._orbit_sizes[spin_cl_id]
            diff_cluster_functions.append(diff_cf)
        diff_cluster_functions = np.array(diff_cluster_functions)
        return diff_cluster_functions

    def eval_from_spin_swap_stable(
        self,
        active_spins: np.array,
        sites: np.array,
    ):
        """Evaluate cluster function changes from spin swap."""
        if len(sites) != 2:
            raise RuntimeError("Size of sites must be two.")
        # TODO: Check when more than binary.

        i, j = sites[0], sites[1]
        diff_cluster_functions = []
        for spin_cl_id, cl in enumerate(self._spin_basis_clusters):
            coeffs = self._lattice_supercell.get_spin_polynomials(cl.spin_basis)
            cluster_size = coeffs.shape[0]
            orbit = self._orbit_sites_supercell[cl.cluster_id]

            cf_new, cf_old = 0.0, 0.0
            for s in sites:
                prods = eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[s]],
                    return_sum=True,
                )
                sum1 = np.sum(orbit[s] == i, axis=1).astype(float)
                sum1 += np.sum(orbit[s] == j, axis=1).astype(float)
                weight = np.reciprocal(sum1) * cluster_size
                cf_old += prods @ weight

            active_spins[i], active_spins[j] = active_spins[j], active_spins[i]
            for s in sites:
                prods = eval_cluster_functions(
                    coeffs,
                    active_spins[orbit[s]],
                    return_sum=True,
                )
                sum1 = np.sum(orbit[s] == i, axis=1).astype(float)
                sum1 += np.sum(orbit[s] == j, axis=1).astype(float)
                weight = np.reciprocal(sum1) * cluster_size
                cf_new += prods @ weight

            active_spins[i], active_spins[j] = active_spins[j], active_spins[i]
            diff_cf = (cf_new - cf_old) / self._orbit_sizes[spin_cl_id]
            diff_cluster_functions.append(diff_cf)
        diff_cluster_functions = np.array(diff_cluster_functions)
        return diff_cluster_functions
