"""Class for searching nonequivalent clusters."""

# from dataclasses import dataclass
import copy
import itertools
import time
from typing import Optional

import numpy as np

from pyclupan.cluster.cluster_utils import calc_distance_pairs
from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import PolymlpStructure, supercell
from pyclupan.core.spglib_utils import get_permutation


class ClusterSearch:
    """Class for performing cluster search."""

    def __init__(self, lattice: Lattice, verbose: bool = False):
        """Init method."""
        self._lattice = lattice
        self._verbose = verbose

        self._lattice_supercell = None
        self._supercell = None
        self._permutation = None
        self._active_sites = None
        self._nonequiv_sites = None
        self._distances = dict()
        self._enum_clusters = dict()

        # elements_lattice = lattice.elements_on_lattice

    def _find_supercell(self, max_cut: float):
        """Find supercell expansion for searching clusters."""
        reduced_cell = self._lattice.reduced_cell

        reduced_norm = np.linalg.norm(reduced_cell.axis, axis=0)
        supercell_matrix = np.diag(np.ceil(np.ones(3) * max_cut * 2 / reduced_norm))
        self._supercell = supercell(reduced_cell, supercell_matrix=supercell_matrix)

        self._permutation = get_permutation(self._supercell)
        self._lattice_supercell = copy.deepcopy(self._lattice)
        self._lattice_supercell.cell = self._supercell
        self._active_sites = self._lattice_supercell.active_sites
        return self

    def _find_nonequivalent_sites(self, max_cut: float):
        """Find nonequivalent sites."""
        if self._permutation is None:
            raise RuntimeError("Permutation not found.")

        rep = np.min(self._permutation[:, self._active_sites], axis=0)
        self._nonequiv_sites = np.unique(rep)
        if self._verbose:
            print("Nonequivalent sites:", self._nonequiv_sites)
        return self._nonequiv_sites

    def _update_active_sites(self, max_cut: float, tol: float = 1e-10):
        """Update active sites.

        Sites with distances from non-equivalent sites
        larger than max_cut are eliminated.
        """
        axis = self._supercell.axis
        positions = self._supercell.positions
        self._active_sites_updated = []
        for i in self._nonequiv_sites:
            positions_i = np.tile(positions[:, i], (positions.shape[1], 1)).T
            positions_j = positions[:, self._active_sites]
            distances = calc_distance_pairs(axis, positions_i, positions_j)
            for j, dis in zip(self._active_sites, distances):
                key = tuple(sorted([i, j]))
                self._distances[key] = dis
                if dis <= max_cut + tol:
                    self._active_sites_updated.append(j)
            self._active_sites_updated.append(i)
        self._active_sites = np.array(self._active_sites_updated)
        return self

    def _extend_cluster_order(self, clusters_prev: list):
        """Increase site to clusters from enumerated smaller clusters."""
        t1 = time.time()
        cl_reps = set()
        for cl, s in itertools.product(clusters_prev, self._active_sites):
            if s not in cl:
                cl_trial = np.array(sorted(list(cl) + [s]))
                cl_perm = self._permutation[:, cl_trial]
                cl_min = min([tuple(sorted(cl)) for cl in cl_perm])
                cl_reps.add(cl_min)
        t2 = time.time()
        print(t2 - t1)
        return sorted(cl_reps)

    def search(self, max_order: int = 4, cutoffs: tuple[float] = (6.0, 6.0, 6.0)):
        """Search clusters."""
        if len(cutoffs) != max_order - 1:
            raise RuntimeError("Cutoff size must be equal to max_order - 1.")

        max_cut = max(cutoffs)
        self._find_supercell(max_cut)
        self._nonequiv_sites = self._find_nonequivalent_sites(max_cut)
        self._update_active_sites(max_cut)

        self._enum_clusters[1] = [[s] for s in self._nonequiv_sites]
        for order in range(2, max_order + 1):
            if self._verbose:
                print("Searching for clusters (order =", str(order) + ")", flush=True)
            cut = cutoffs[order - 2]
            clusters_cand = self._extend_cluster_order(self._enum_clusters[order - 1])
            clusters = []
            for cl_trial in clusters_cand:
                is_cutoff, _ = self._is_within_cutoff(cl_trial, cut)
                if is_cutoff:
                    clusters.append(cl_trial)
            self._enum_clusters[order] = sorted(clusters)

            if self._verbose:
                prefix = "Number of clusters (order = " + str(order) + "):"
                print(prefix, len(self._enum_clusters[order]), flush=True)

        if self._verbose:
            n_total_clusters = sum(len(v) for v in self._enum_clusters.values())
            print("Total number of clusters:", n_total_clusters, flush=True)
        return self

    def _define_cluster_origin(self, cluster: list):
        """Pick up a nonequivalent site from cluster."""
        neighbors = []
        for i, s1 in enumerate(cluster):
            if s1 in self._nonequiv_sites:
                origin = s1
                neighbors.extend(cluster[i + 1 :])
                break
            else:
                neighbors.append(s1)
        return origin, neighbors

    def _compute_distance(self, i, j):

        axis = self._supercell.axis
        positions = self._supercell.positions

        diff1 = positions[:, j] - positions[:, i]
        position_j = positions[:, j] - np.round(diff1)

        key = tuple(sorted([i, j]))
        if key not in self._distances:
            diff = position_j - positions[:, i]
            dis = np.linalg.norm(axis @ diff)
            self._distances[key] = dis

        return self._distances[key], position_j

    def _is_within_cutoff(self, cluster: tuple, cut: float, tol: float = 1e-10):
        """Check if all pairs of cluster are below cutoff distance."""

        axis = self._lattice_supercell.cell.axis
        positions = self._lattice_supercell.cell.positions

        is_cutoff = True
        positions_nearest = []
        i, neighbors = self._define_cluster_origin(cluster)
        for j in neighbors:
            key = tuple(sorted([i, j]))
            dis = self._distances[key]
            if dis > cut + tol:
                is_cutoff = False
                return is_cutoff, None

            diff1 = positions[:, j] - positions[:, i]
            position_j = positions[:, j] - np.round(diff1)
            positions_nearest.append(position_j)

        positions_nearest = np.array(positions_nearest).T
        order = len(cluster)
        if order > 2:
            for i, j in itertools.combinations(range(order - 1), 2):
                diff = positions_nearest[:, j] - positions_nearest[:, i]
                dis = np.linalg.norm(axis @ diff)
                if dis > cut + tol:
                    is_cutoff = False
                    return is_cutoff, None

        return is_cutoff, positions_nearest


def run_cluster(
    unitcell: PolymlpStructure,
    occupation: Optional[list] = None,
    elements: Optional[list] = None,
    cutoff: float = 6.0,
    max_order: int = 4,
    verbose: bool = False,
):
    """Search nonequivalent clusters."""

    lattice = Lattice(
        cell=unitcell,
        occupation=occupation,
        elements=elements,
        verbose=verbose,
    )
    cs = ClusterSearch(lattice, verbose=verbose)
    cs.search()
