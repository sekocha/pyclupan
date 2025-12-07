"""Class for searching nonequivalent clusters."""

import copy
import itertools
from typing import Optional

import numpy as np

from pyclupan.cluster.cluster_io import save_cluster_yaml
from pyclupan.cluster.cluster_utils import (
    ClusterAttr,
    calc_distance_pairs,
    find_supercell,
)
from pyclupan.core.cell_positions_utils import decompose_fraction
from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.core.spglib_utils import get_permutation


class ClusterSearch:
    """Class for performing cluster search."""

    def __init__(self, lattice: Lattice, verbose: bool = False):
        """Init method."""
        self._lattice = lattice
        self._verbose = verbose

        self._elements_lattice = lattice.elements_on_lattice

        self._lattice_supercell = None
        self._active_sites = None
        self._supercell = None
        self._permutation = None

        self._nonequiv_sites = None
        self._distances = dict()
        self._enum_clusters = dict()

        self._cutoffs = None

    def _find_supercell(self, max_cut: float):
        """Find supercell expansion for searching clusters."""
        self._supercell = find_supercell(self._lattice.reduced_cell, max_cut)
        self._lattice_supercell = copy.deepcopy(self._lattice)
        self._lattice_supercell.cell = self._supercell
        self._active_sites = self._lattice_supercell.active_sites
        self._permutation = get_permutation(self._supercell)
        return self

    def _find_nonequivalent_sites(self, max_cut: float):
        """Find nonequivalent sites."""
        if self._permutation is None:
            raise RuntimeError("Permutation not found.")

        rep = np.min(self._permutation[:, self._active_sites], axis=0)
        self._nonequiv_sites = np.unique(rep)

        point_clusters = []
        for s in self._nonequiv_sites:
            cl_attr = ClusterAttr(
                sites_supercell=[s],
                positions_supercell=np.array([self._supercell.positions[:, s]]).T,
            )
            point_clusters.append(cl_attr)

        self._update_active_sites(max_cut)
        return point_clusters

    def _update_active_sites(self, max_cut: float, tol: float = 1e-10):
        """Update active sites.

        Sites with distances from non-equivalent sites
        larger than max_cut are eliminated.
        """
        if self._nonequiv_sites is None:
            raise RuntimeError("Run find_nonequivalent_sites in advance.")

        axis, positions = self._supercell.axis, self._supercell.positions
        active_sites_updated = []
        for i in self._nonequiv_sites:
            positions_i = np.tile(positions[:, i], (len(self._active_sites), 1)).T
            positions_j = positions[:, self._active_sites]
            distances = calc_distance_pairs(axis, positions_i, positions_j)
            for j, dis in zip(self._active_sites, distances):
                key = tuple(sorted([i, j]))
                self._distances[key] = dis
                if dis <= max_cut + tol:
                    active_sites_updated.append(j)
            active_sites_updated.append(i)
        self._active_sites = np.unique(active_sites_updated)
        return self

    def _extend_cluster_order(self, clusters_prev: list[ClusterAttr]):
        """Increase site to clusters from enumerated smaller clusters."""
        sites_reps = set()
        for cl, s in itertools.product(clusters_prev, self._active_sites):
            if s not in cl.sites_supercell:
                sites_trial = np.array(list(cl.sites_supercell) + [s])
                sites_perm = self._permutation[:, sites_trial]
                sites_perm = np.sort(sites_perm, axis=1)
                sites_min = np.unique(sites_perm, axis=0)[0]
                sites_reps.add(tuple(sites_min))

        return sorted(sites_reps)

    def _define_cluster_origin(self, sites: list):
        """Pick up a nonequivalent site from cluster sites."""
        neighbors = []
        for i, s1 in enumerate(sites):
            if s1 in self._nonequiv_sites:
                origin = s1
                neighbors.extend(sites[i + 1 :])
                break
            else:
                neighbors.append(s1)
        return origin, neighbors

    def _is_within_cutoff(self, sites: list, cut: float, tol: float = 1e-10):
        """Check if all pairs of cluster are below cutoff distance."""

        axis = self._supercell.axis
        positions = self._supercell.positions

        positions_nearest = []
        i, neighbors = self._define_cluster_origin(sites)
        for j in neighbors:
            key = tuple(sorted([i, j]))
            if self._distances[key] > cut + tol:
                return False, None, None

            diff1 = positions[:, j] - positions[:, i]
            position_j = positions[:, j] - np.round(diff1)
            positions_nearest.append(position_j)

        if len(sites) > 2:
            for pos_i, pos_j in itertools.combinations(positions_nearest, 2):
                if np.linalg.norm(axis @ (pos_j - pos_i)) > cut + tol:
                    return False, None, None

        cluster_reordered = tuple([i] + neighbors)
        cl_positions = np.array([positions[:, i]] + positions_nearest).T
        return True, cluster_reordered, cl_positions

    def _print_log(self, cutoffs: tuple):
        """Print logs about initial parameters."""
        print("Finding nonequivalent clusters.", flush=True)
        print("Sublattices:", flush=True)
        for i, elements in enumerate(self._elements_lattice):
            print("- Lattice:   ", i, flush=True)
            print("  Elements:  ", elements, flush=True)
            print("  active:    ", len(elements) > 1, flush=True)
        print("Parameters:", flush=True)
        print("  Max order: ", len(cutoffs) + 1, flush=True)
        for i, cut in enumerate(cutoffs):
            order = i + 2
            print("  Cutoff:", flush=True)
            print("  - Order:   ", order, flush=True)
            print("    Distance:", cut, flush=True)

        return self

    def search(self, max_order: int = 4, cutoffs: tuple[float] = (6.0, 6.0, 6.0)):
        """Search nonequivalent clusters."""
        if len(cutoffs) != max_order - 1:
            raise RuntimeError("Cutoff size must be equal to max_order - 1.")

        if not np.allclose(cutoffs, list(reversed(sorted(cutoffs)))):
            raise RuntimeError(
                "Cutoffs must be smaller or equal to those for smaller orders."
            )

        if self._verbose:
            self._print_log(cutoffs)

        self._cutoffs = cutoffs
        max_cut = max(cutoffs)

        self._find_supercell(max_cut)
        self._enum_clusters[1] = self._find_nonequivalent_sites(max_cut)

        if self._verbose:
            print("Nonequivalent sites:", self._nonequiv_sites)

        for order in range(2, max_order + 1):
            if self._verbose:
                print("Searching for clusters (order", str(order) + ")", flush=True)
            cut = cutoffs[order - 2]
            sites_cand = self._extend_cluster_order(self._enum_clusters[order - 1])

            clusters = []
            for sites_trial in sites_cand:
                is_cutoff, sites, fracs = self._is_within_cutoff(sites_trial, cut)
                if is_cutoff:
                    cl = ClusterAttr(sites_supercell=sites, positions_supercell=fracs)
                    clusters.append(cl)
            sorted_clusters = sorted(clusters, key=lambda p: p.sites_supercell)
            self._enum_clusters[order] = sorted_clusters

            if self._verbose:
                prefix = "Number of clusters (order " + str(order) + "):"
                print(prefix, len(self._enum_clusters[order]), flush=True)

        if self._verbose:
            n_total_clusters = sum(len(v) for v in self._enum_clusters.values())
            print("Total number of clusters:", n_total_clusters, flush=True)
        return self

    def _extract_cluster_sites_permutation(self, cl_sites: np.ndarray):
        """Get permutations for cluster sites."""
        perm_cluster = self._permutation[:, cl_sites]
        ids = np.where(perm_cluster == cl_sites[0])[0]
        for s in cl_sites[1:]:
            ids = np.intersect1d(ids, np.where(perm_cluster == s)[0])
        perm_cluster = np.unique(perm_cluster[ids], axis=0)

        perm = np.zeros(perm_cluster.shape, dtype=int)
        for i, site in enumerate(cl_sites):
            perm[np.where(perm_cluster == site)] = i
        return perm

    def _find_combinations(self, cluster_sites: tuple):
        """Find possible element and spin basis combinations."""
        lattice_types = self._lattice_supercell.types
        cl_sites = np.array(cluster_sites)
        perm = self._extract_cluster_sites_permutation(cl_sites)

        element_combs = set()
        elements = [self._elements_lattice[lattice_types[s]] for s in cl_sites]
        candidates = itertools.product(*elements)
        for c in candidates:
            c_rep = np.unique(np.array(c)[perm], axis=0)[0]
            element_combs.add(tuple(c_rep))

        basis_lattice = self._lattice_supercell.basis_on_lattice
        basis_combs = set()
        basis = [basis_lattice[lattice_types[s]] for s in cl_sites]
        candidates = itertools.product(*basis)
        for c in candidates:
            c_rep = np.unique(np.array(c)[perm], axis=0)[0]
            basis_combs.add(tuple(c_rep))
        return sorted(element_combs), sorted(basis_combs)

    def search_with_colors(self):
        """Search clusters with distinguishing elements and spin basis."""
        if self._enum_clusters is None:
            raise RuntimeError("Run search() first.")

        for order, clusters in self._enum_clusters.items():
            for cl in clusters:
                sites = cl.sites_supercell
                res = self._find_combinations(sites)
                cl.elements_combinations, cl.spin_basis_combinations = res
        return self

    def represent_in_unitcell(self, tol: float = 1e-14):
        """Calculate representation of cluster positions in unitcell.

        Definition
        ----------
        A_(sup,reduce) = A_(reduce) @ H = A_(unit) @ T @ H
        x = A_(unit)^-1 @ A_(sup,reduce) @ x_(sup,reduce)
          = T @ H @ x_(sup,reduce)
        """
        if self._enum_clusters is None:
            raise RuntimeError("Run search() first.")

        unitcell, supercell = self._lattice.cell, self._supercell
        mat = np.linalg.inv(unitcell.axis) @ supercell.axis

        for order, clusters in self._enum_clusters.items():
            for cl in clusters:
                positions = mat @ cl.positions_supercell
                cells, positions_frac = decompose_fraction(positions, tol=tol)

                sites = []
                for pos1 in positions_frac.T:
                    for j, pos2 in enumerate(unitcell.positions.T):
                        if np.allclose(pos1, pos2):
                            sites.append(j)
                            break
                cl.sites_unitcell = sites
                cl.cells_unitcell = cells
        return self

    def run(self, max_order: int = 4, cutoffs: tuple = (6.0, 6.0, 6.0)):
        """Search clusters and get required attributes."""
        self.search(max_order=max_order, cutoffs=cutoffs)
        self.search_with_colors()
        self.represent_in_unitcell()
        return self

    def save(self, filename: str = "pyclupan_cluster.yaml"):
        """Save results of cluster search."""
        save_cluster_yaml(
            self._enum_clusters,
            self._lattice,
            self._cutoffs,
            filename=filename,
        )
        return self

    @property
    def clusters(self):
        """Return enumerated clusters."""
        return self._enum_clusters


def run_cluster(
    unitcell: PolymlpStructure,
    occupation: Optional[list] = None,
    elements: Optional[list] = None,
    max_order: int = 4,
    cutoffs: tuple = (6.0, 6.0, 6.0),
    filename: Optional[str] = None,
    verbose: bool = False,
):
    """Search nonequivalent clusters.

    Parameters
    ----------
    occupation: Lattice IDs occupied by elements.
                Example: [[0], [1], [2], [2]].
    elements: Element IDs on lattices.
              Example: [[0], [1], [2, 3]].
    max_order: Maximum order of clusters.
    cutoffs: Cutoff distances for orders >= 2.
            (two-body, three-body, four-body, ...)
            Size of cutoffs must be equal to max_order - 1.
            Cutoffs must be smaller or equal to those for smaller orders.
    filename: Name of output file for cluster search results.
              If None, no file will be generated.
    """
    lattice = Lattice(
        cell=unitcell,
        occupation=occupation,
        elements=elements,
    )
    cs = ClusterSearch(lattice, verbose=verbose)
    cs.run(max_order=max_order, cutoffs=cutoffs)
    if filename is not None:
        cs.save(filename=filename)
    return cs.clusters
