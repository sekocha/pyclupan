"""Functions for running single MC simulation."""

# import time

import numpy as np

from pyclupan.core.model import CEmodel
from pyclupan.core.pypolymlp_utils import KbEV
from pyclupan.features.cluster_functions_mc import ClusterFunctionsMC
from pyclupan.mc.mc_utils import MCAttr, MCParams


def _select_one_site(spins: np.ndarray, spin_species: np.ndarray):
    """Select two sites with different spins."""
    i = np.random.choice(len(spins))
    spin_candidates = spin_species[spin_species != spins[i]]
    spin_new = np.random.choice(spin_candidates)
    return i, spin_new


def _select_two_sites(spins: np.ndarray, spin_species: np.ndarray):
    """Select two sites with different spins."""
    spin_vals = np.random.choice(spin_species, size=2, replace=False)
    return [np.random.choice(np.where(spins == v)[0]) for v in spin_vals]


def _print_iteration(
    mc_iter: int, energy: float, average_energy: float, average_cfs: np.ndarray
):
    """Print properties at an iteration."""
    print("Iteration:", mc_iter + 1, flush=True)
    print("- Energy:        ", energy, flush=True)
    print("- Average energy:", average_energy / (mc_iter + 1), flush=True)
    print("- Average cluster functions:", flush=True)
    print(average_cfs / (mc_iter + 1), flush=True)


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
    if verbose:
        np.set_printoptions(suppress=True)

    n_sites = mc_attr.n_sites
    spins = mc_attr.active_spins.astype(np.int32)
    energy = mc_attr.energy
    cfs = mc_attr.cluster_functions
    beta = 1.0 / (KbEV * temp)

    for n_steps in [mc_params.n_steps_init * n_sites, mc_params.n_steps_eq * n_sites]:
        average_energy = 0.0
        average_cfs = np.zeros(len(cfs))
        for mc_iter in range(n_steps):
            # t1 = time.time()
            i, j = _select_two_sites(spins, mc_attr.spin_species)
            # t2 = time.time()

            cfs_new = cfs + cf.eval_from_spin_swap(spins, [i, j])
            energy_new = model.eval(cfs_new)
            # t3 = time.time()

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
            threshold = np.exp(-beta * delta_e)
            if np.random.rand() < threshold:
                energy = energy_new
                cfs = cfs_new
                spins[i], spins[j] = spins[j], spins[i]
            # t4 = time.time()
            # print(t3 - t2)

            average_energy += energy
            average_cfs += cfs
            if verbose and (mc_iter + 1) % verbose_interval == 0:
                _print_iteration(mc_iter, energy, average_energy, average_cfs)

        average_energy /= n_steps
        average_cfs /= n_steps

    mc_attr.active_spins = spins
    mc_attr.energy = energy
    mc_attr.average_energy = average_energy
    mc_attr.cluster_functions = cfs
    mc_attr.average_cluster_functions = average_cfs
    return mc_attr


def sgcmc(
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
    """Run semi-grand canonical MC."""
    if verbose:
        np.set_printoptions(suppress=True)

    n_sites = mc_attr.n_sites
    spins = mc_attr.active_spins.astype(np.int32)
    energy = mc_attr.energy
    cfs = mc_attr.cluster_functions
    beta = 1.0 / (KbEV * temp)

    mu = np.array([0.0] + list(mc_params.mu))
    spin_species = np.array(mc_attr.spin_species)

    delta_mu_dict = dict()
    for spin1 in spin_species:
        mu1 = mu[np.where(spin_species == spin1)[0][0]]
        for spin2 in spin_species:
            mu2 = mu[np.where(spin_species == spin2)[0][0]]
            delta_mu_dict[(spin1, spin2)] = mu2 - mu1

    for n_steps in [mc_params.n_steps_init * n_sites, mc_params.n_steps_eq * n_sites]:
        average_energy = 0.0
        average_cfs = np.zeros(len(cfs))
        for mc_iter in range(n_steps):
            i, spin_new = _select_one_site(spins, spin_species)
            spin_old = spins[i]
            cfs_new = cfs + cf.eval_from_spin_flip(spins, i, spin_new)
            energy_new = model.eval(cfs_new)

            if assert_direct:
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

            delta_mu = delta_mu_dict[(spin_old, spin_new)]
            delta_e = energy_new - energy
            threshold = np.exp(-beta * (delta_e - delta_mu))
            if np.random.rand() < threshold:
                energy = energy_new
                cfs = cfs_new
                spins[i] = spin_new

            average_energy += energy
            average_cfs += cfs
            if verbose and (mc_iter + 1) % verbose_interval == 0:
                _print_iteration(mc_iter, energy, average_energy, average_cfs)

        average_energy /= n_steps
        average_cfs /= n_steps

    mc_attr.active_spins = spins
    mc_attr.energy = energy
    mc_attr.average_energy = average_energy
    mc_attr.cluster_functions = cfs
    mc_attr.average_cluster_functions = average_cfs
    return mc_attr
