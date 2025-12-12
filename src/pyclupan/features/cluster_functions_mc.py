"""Class for calculating cluster functions."""

import numpy as np

from pyclupan.core.cell_utils import get_unitcell_reps
from pyclupan.core.lattice import Lattice
from pyclupan.core.spin import eval_cluster_functions
from pyclupan.features.cluster_functions import ClusterFunctions
from pyclupan.features.orbit_utils import find_orbit_supercell, get_map_positions


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
        self._binary = self._is_binary()
        self._independent = self._check_neighbors()
        self._poly_coeffs = self._set_polynomial_coeffs()

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

                orbit_active_site_rep = dict()
                for k, v in orbit.items():
                    site = self._lattice_supercell.to_active_site_rep([k])[0]
                    orbit_active_site_rep[site] = np.array(
                        self._lattice_supercell.to_active_site_rep(v)
                    )
                    n_duplicate = np.sum(
                        orbit_active_site_rep[site] == site, axis=1
                    ).astype(float)
                    self._duplicate_n_sites[(i, site)] = n_duplicate

                self._orbit_sites_supercell[i] = orbit_active_site_rep

        return self

    def _is_binary(self):
        """Check if configuration is binary."""
        self._binary = True
        for cl in self._spin_basis_clusters:
            coeffs = self._lattice_supercell.get_spin_polynomials(cl.spin_basis)
            if not np.allclose(coeffs[:, 0], 1.0) and not np.allclose(
                coeffs[:, 1], 0.0
            ):
                self._binary = False
                break
        return self._binary

    def _check_neighbors(self):
        """Check if every two sites are connected."""
        if self._verbose:
            print("Check if every two sites are connected.", flush=True)

        n_sites = len(self._lattice_supercell.active_sites)
        self._independent = np.ones((n_sites, n_sites), dtype=bool)
        for cl in self._spin_basis_clusters:
            orbit = self._orbit_sites_supercell[cl.cluster_id]
            for i in range(n_sites):
                self._independent[i, orbit[i].reshape(-1)] = False
        return self._independent

    def _set_polynomial_coeffs(self):
        """Set polynomial coefficients of clusters."""
        self._poly_coeffs = [
            self._lattice_supercell.get_spin_polynomials(cl.spin_basis)
            for cl in self._spin_basis_clusters
        ]
        return self._poly_coeffs

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

    def _calc_products(
        self,
        active_spins: np.ndarray,
        orbit_per_site: np.ndarray,
        coeffs: np.ndarray,
    ):
        """Calculate products of cluster functions."""
        if self._binary:
            prods = np.multiply.reduce(active_spins[orbit_per_site], axis=1)
        else:
            prods = eval_cluster_functions(
                coeffs,
                active_spins[orbit_per_site],
                return_array=True,
            )
        return prods

    def _calc_sum_products(
        self,
        active_spins: np.ndarray,
        orbit_i: np.ndarray,
        orbit_j: np.ndarray,
        coeffs: np.ndarray,
    ):
        """Calculate products of cluster functions."""
        if self._binary:
            cf_sum_i = np.sum(np.multiply.reduce(active_spins[orbit_i], axis=1))
            cf_sum_j = np.sum(np.multiply.reduce(active_spins[orbit_j], axis=1))
            return cf_sum_i + cf_sum_j

        cf_sum_i = np.sum(
            eval_cluster_functions(coeffs, active_spins[orbit_i], return_array=True)
        )
        cf_sum_j = np.sum(
            eval_cluster_functions(coeffs, active_spins[orbit_j], return_array=True)
        )
        return cf_sum_i + cf_sum_j

    def eval_from_spin_swap(
        self,
        active_spins: np.array,
        sites: np.array,
    ):
        """Evaluate cluster function changes from spin swap."""

        if len(sites) != 2:
            raise RuntimeError("Size of sites must be two.")

        i, j = sites[0], sites[1]
        spin_i, spin_j = active_spins[i], active_spins[j]
        dspin = spin_j - spin_i

        diff_cluster_functions = []
        for spin_cl_id, cl in enumerate(self._spin_basis_clusters):
            coeffs = self._poly_coeffs[spin_cl_id]
            cluster_size = coeffs.shape[0]
            orbit = self._orbit_sites_supercell[cl.cluster_id]
            orbit_size = self._orbit_sizes[spin_cl_id]

            if not self._independent[i, j]:
                duplicate_i = np.sum(orbit[i] == j, axis=1).astype(float)
                duplicate_i += self._duplicate_n_sites[(cl.cluster_id, i)]
                duplicate_j = np.sum(orbit[j] == i, axis=1).astype(float)
                duplicate_j += self._duplicate_n_sites[(cl.cluster_id, j)]
                sum_i, sum_j = np.sum(duplicate_i), np.sum(duplicate_j)
                independent = (
                    sum_i == duplicate_i.shape[0] and sum_j == duplicate_j.shape[0]
                )

            if self._independent[i, j] or independent:
                if self._binary:
                    active_spins[i], active_spins[j] = dspin, -dspin
                    val = self._calc_sum_products(
                        active_spins, orbit[i], orbit[j], coeffs
                    )
                else:
                    # TODO: Implement efficient algorithm
                    val1 = self._calc_sum_products(
                        active_spins, orbit[i], orbit[j], coeffs
                    )
                    active_spins[i], active_spins[j] = spin_j, spin_i
                    val2 = self._calc_sum_products(
                        active_spins, orbit[i], orbit[j], coeffs
                    )
                    val = val2 - val1
                active_spins[i], active_spins[j] = spin_i, spin_j
                diff_cf = val * cluster_size / orbit_size
            else:
                weight_i = cluster_size / duplicate_i
                weight_j = cluster_size / duplicate_j

                prods_i = self._calc_products(active_spins, orbit[i], coeffs)
                prods_j = self._calc_products(active_spins, orbit[j], coeffs)
                cf_old = prods_i @ weight_i + prods_j @ weight_j

                active_spins[i], active_spins[j] = active_spins[j], active_spins[i]

                prods_i = self._calc_products(active_spins, orbit[i], coeffs)
                prods_j = self._calc_products(active_spins, orbit[j], coeffs)
                cf_new = prods_i @ weight_i + prods_j @ weight_j

                active_spins[i], active_spins[j] = active_spins[j], active_spins[i]

                diff_cf = (cf_new - cf_old) / orbit_size
            diff_cluster_functions.append(diff_cf)
        return np.array(diff_cluster_functions)

    def eval_from_spin_swap_stable(
        self,
        active_spins: np.array,
        sites: np.array,
    ):
        """Evaluate cluster function changes from spin swap."""
        if len(sites) != 2:
            raise RuntimeError("Size of sites must be two.")

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
                    return_array=True,
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
                    return_array=True,
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

    def eval_from_spin_flip(
        self,
        active_spins: np.array,
        site: int,
        spin_new: int,
    ):
        """Evaluate cluster function changes from spin swap."""
        i = site
        spin_i = active_spins[i]
        dspin = spin_new - spin_i

        diff_cluster_functions = []
        for spin_cl_id, cl in enumerate(self._spin_basis_clusters):
            coeffs = self._poly_coeffs[spin_cl_id]
            cluster_size = coeffs.shape[0]
            orbit = self._orbit_sites_supercell[cl.cluster_id]
            orbit_size = self._orbit_sizes[spin_cl_id]

            if self._binary:
                active_spins[i] = dspin
                val = np.sum(np.multiply.reduce(active_spins[orbit[i]], axis=1))
            else:
                val1 = np.sum(
                    eval_cluster_functions(
                        coeffs, active_spins[orbit[i]], return_array=True
                    )
                )
                active_spins[i] = spin_new
                val2 = np.sum(
                    eval_cluster_functions(
                        coeffs, active_spins[orbit[i]], return_array=True
                    )
                )
                val = val2 - val1
            active_spins[i] = spin_i
            diff_cf = val * cluster_size / orbit_size
            diff_cluster_functions.append(diff_cf)
        return np.array(diff_cluster_functions)
