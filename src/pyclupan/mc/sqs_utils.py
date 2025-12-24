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
    compositions = []
    for n, spin_sub in zip(lattice.n_active_sites, lattice.active_spins):
        spins = active_spins[begin : begin + n]
        n_atoms = np.array([np.count_nonzero(spins == s) for s in spin_sub])
        comp = n_atoms / np.sum(n_atoms)
        compositions.append(comp)
        begin += n

    lattice = lattice_unitcell
    ideal_cluster_functions = []
    for cl in cf.spin_basis_clusters:
        coeffs = lattice.get_spin_polynomials(cl.spin_basis)
        sites = cf.clusters[cl.cluster_id].sites_unitcell
        sublattices = lattice.sublattice_id[np.array(sites)]

        ave_point_cfs = []
        for sub, c in zip(sublattices, coeffs):
            comp = compositions[sub]
            spins = lattice.active_spins[sub]
            ave = np.polyval(c, spins) @ comp
            ave_point_cfs.append(ave)
        val = np.prod(ave_point_cfs)
        ideal_cluster_functions.append(val)
    ideal_cluster_functions = np.array(ideal_cluster_functions)
    return ideal_cluster_functions
