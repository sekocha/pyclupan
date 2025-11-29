"""Functions for running single MC simulation."""

import time

import numpy as np

from pyclupan.core.model import CEmodel
from pyclupan.core.pypolymlp_utils import KbEV
from pyclupan.features.cluster_functions_utils import ClusterFunctionsMC
from pyclupan.mc.mc_utils import MCAttr, MCParams

# from typing import Optional


def _select_two_sites(spins: np.ndarray, spin_species: np.ndarray):
    """Select two sites with different spins."""
    spin_vals = np.random.choice(spin_species, size=2, replace=False)
    return [np.random.choice(np.where(spins == v)[0]) for v in spin_vals]

    # indices = np.random.choice(len(spins), size=2, replace=False)
    # spins[indices[0]
    # np.random.choice(mc_attr.spin_species, size=2, replace=False)
    # indices = np.where(arr == v)[0]
    # idx = np.random.choice(indices)


def cmc(
    temp: float,
    mc_attr: MCAttr,
    mc_params: MCParams,
    cf: ClusterFunctionsMC,
    model: CEmodel,
    assert_direct: bool = False,
    # assert_direct: bool = True,
    verbose: bool = False,
):
    """Run canonical MC."""
    n_sites = mc_attr.n_sites
    n_steps_array = [mc_params.n_steps_init * n_sites, mc_params.n_steps_eq * n_sites]

    spins = mc_attr.active_spins
    energy = mc_attr.energy
    cfs = mc_attr.cluster_functions
    beta = 1.0 / (KbEV * temp)
    for n_steps in n_steps_array:
        for mc_iter in range(n_steps):
            t1 = time.time()
            i, j = _select_two_sites(spins, mc_attr.spin_species)
            t2 = time.time()

            # TODO: Time consuming part.
            diff_cfs = cf.eval_from_spin_swap(spins, [i, j])
            cfs_new = cfs + diff_cfs
            t3 = time.time()
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
            t4 = time.time()
            print(t2 - t1, t3 - t2, t4 - t3)

            if verbose and (mc_iter + 1) % 1000 == 0:
                print("Iter", mc_iter + 1, ":", energy, flush=True)

    mc_attr.active_spins = spins
    mc_attr.energy = energy
    mc_attr.cluster_functions = cfs
    return mc_attr


def sgcmc():
    pass


# def sgcmc(mc_attr: MCAttr, mc_params: MCParams, cf: ClusterFunctions, temp: float):
#     """Run semi-grand canonical MC."""
#     pass
