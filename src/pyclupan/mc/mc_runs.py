"""Functions for running single MC simulation."""

# import time

import numpy as np

from pyclupan.core.model import CEmodel
from pyclupan.core.pypolymlp_utils import KbEV
from pyclupan.features.cluster_functions_mc import ClusterFunctionsMC
from pyclupan.mc.mc_utils import MCAttr, MCParams

# from typing import Optional


def _select_two_sites(spins: np.ndarray, spin_species: np.ndarray):
    """Select two sites with different spins."""
    spin_vals = np.random.choice(spin_species, size=2, replace=False)
    return [np.random.choice(np.where(spins == v)[0]) for v in spin_vals]


def _select_one_site(spins: np.ndarray, spin_species: np.ndarray):
    """Select two sites with different spins."""
    i = np.random.choice(len(spins))
    spin_candidates = spin_species[spin_species != spins[i]]
    spin_new = np.random.choice(spin_candidates)
    return i, spin_new


def cmc(
    temp: float,
    mc_attr: MCAttr,
    mc_params: MCParams,
    cf: ClusterFunctionsMC,
    model: CEmodel,
    assert_direct: bool = False,
    # assert_direct: bool = True,
    verbose_interval: int = 10000,
    verbose: bool = False,
):
    """Run canonical MC."""
    n_sites = mc_attr.n_sites
    spins = mc_attr.active_spins.astype(np.int32)
    energy = mc_attr.energy
    cfs = mc_attr.cluster_functions
    beta = 1.0 / (KbEV * temp)

    for n_steps in [mc_params.n_steps_init * n_sites, mc_params.n_steps_eq * n_sites]:
        for mc_iter in range(n_steps):
            # t1 = time.time()
            i, j = _select_two_sites(spins, mc_attr.spin_species)
            # t2 = time.time()

            diff_cfs = cf.eval_from_spin_swap(spins, [i, j])
            cfs_new = cfs + diff_cfs
            # t3 = time.time()
            energy_new = model.eval(cfs_new)

            if assert_direct:
                spins[i], spins[j] = spins[j], spins[i]
                cfs_new_direct = cf.eval_from_spins(spins)
                energy_new_direct = model.eval(cfs_new_direct)
                spins[i], spins[j] = spins[j], spins[i]
                print("DIRECT:  ")
                print(cfs_new_direct)
                print("DIFF:    ")
                print(cfs_new)
                print("Energy:", energy_new_direct, energy_new)
                np.testing.assert_allclose(cfs_new, cfs_new_direct, atol=1e-8)

            delta_e = energy_new - energy
            # TODO: Use supercell energy unit
            threshold = np.exp(-beta * delta_e * n_sites)
            if np.random.rand() < threshold:
                energy = energy_new
                cfs = cfs_new
                spins[i], spins[j] = spins[j], spins[i]
            # t4 = time.time()
            # print(t3 - t2)

            if verbose and (mc_iter + 1) % verbose_interval == 0:
                print("Iter", mc_iter + 1, ":", energy, flush=True)

    mc_attr.active_spins = spins
    mc_attr.energy = energy
    mc_attr.cluster_functions = cfs
    return mc_attr


def sgcmc(
    temp: float,
    mc_attr: MCAttr,
    mc_params: MCParams,
    cf: ClusterFunctionsMC,
    model: CEmodel,
    assert_direct=False,
    verbose_interval: int = 10000,
    verbose: bool = False,
):
    """Run semi-grand canonical MC."""
    n_sites = mc_attr.n_sites
    spins = mc_attr.active_spins.astype(np.int32)
    energy = mc_attr.energy
    cfs = mc_attr.cluster_functions
    beta = 1.0 / (KbEV * temp)

    # TODO: Define chemical potential for multicomponent systems
    mu = mc_params.mu

    for n_steps in [mc_params.n_steps_init * n_sites, mc_params.n_steps_eq * n_sites]:
        for mc_iter in range(n_steps):
            # t1 = time.time()
            i, spin_new = _select_one_site(spins, mc_attr.spin_species)
            # t2 = time.time()

            diff_cfs = cf.eval_from_spin_flip(spins, i, spin_new)
            cfs_new = cfs + diff_cfs
            # t3 = time.time()
            energy_new = model.eval(cfs_new)

            if assert_direct:
                spin_old = spins[i]
                spins[i] = spin_new
                cfs_new_direct = cf.eval_from_spins(spins)
                energy_new_direct = model.eval(cfs_new_direct)
                spins[i] = spin_old
                print("DIRECT:  ")
                print(cfs_new_direct)
                print("DIFF:    ")
                print(cfs_new)
                print("Energy:", energy_new_direct, energy_new)
                np.testing.assert_allclose(cfs_new, cfs_new_direct, atol=1e-8)

            # TODO: Define chemical potential for multicomponent systems
            delta_mu = mu if spin_new == -1 else -mu

            delta_e = energy_new - energy
            # TODO: Use supercell energy unit
            threshold = np.exp(-beta * (delta_e * n_sites - delta_mu))
            if np.random.rand() < threshold:
                energy = energy_new
                cfs = cfs_new
                spins[i] = spin_new
            # t4 = time.time()
            # print(t3 - t2)

            if verbose and (mc_iter + 1) % verbose_interval == 0:
                print("Iter", mc_iter + 1, ":", energy, flush=True)
                print(np.count_nonzero(spins == 1))
                print(np.count_nonzero(spins == -1))

    mc_attr.active_spins = spins
    mc_attr.energy = energy
    mc_attr.cluster_functions = cfs
    return mc_attr
