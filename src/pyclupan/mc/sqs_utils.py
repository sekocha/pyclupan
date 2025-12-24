"""Utility functions for SQS calculation."""

import numpy as np

from pyclupan.core.lattice import Lattice
from pyclupan.features.cluster_functions_mc import ClusterFunctions


def calc_ideal_cluster_functions(
    lattice_unitcell: Lattice,
    lattice_supercell: Lattice,
    cf: ClusterFunctions,
    active_spins: np.ndarray,
):
    """Calculate ideal cluster functions from compositions."""
    lattice = lattice_supercell
    begin = 0
    average_spins = []
    for n, spin_sub in zip(lattice.n_active_sites, lattice.active_spins):
        spins = active_spins[begin : begin + n]
        n_atoms = np.array([np.count_nonzero(spins == s) for s in spin_sub])
        average = (n_atoms @ spin_sub) / np.sum(n_atoms)
        average_spins.append(average)
        begin += n
    average_spins = np.array(average_spins)

    lattice = lattice_unitcell
    ideal_cluster_functions = []
    for cl in cf.spin_basis_clusters:
        coeffs = lattice.get_spin_polynomials(cl.spin_basis)
        cluster = cf.clusters[cl.cluster_id]
        sites = cluster.sites_unitcell
        sublattices = lattice.sublattice_id[np.array(sites)]
        spins = average_spins[sublattices]
        val = np.prod([np.polyval(c, s) for c, s in zip(coeffs, spins)])
        ideal_cluster_functions.append(val)
    ideal_cluster_functions = np.array(ideal_cluster_functions)
    return ideal_cluster_functions
