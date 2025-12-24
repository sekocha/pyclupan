"""Functions for running single MC simulation."""

import numpy as np

from pyclupan.core.pypolymlp_utils import KbEV
from pyclupan.features.cluster_functions_mc import ClusterFunctionsMC
from pyclupan.mc.mc_utils import MCAttr, MCParams


def _print_iteration(mc_iter: int, score: float, cfs: np.ndarray):
    """Print properties at an iteration."""
    print("Iteration:", mc_iter + 1, flush=True)
    print("- Score:", score, flush=True)
    print("- Cluster functions:", flush=True)
    print(cfs, flush=True)


def _score(cfs: np.ndarray, ideal_cfs: np.ndarray):
    """Define score between cluster functions and ideal ones."""
    return np.linalg.norm(cfs - ideal_cfs)


def cmc(
    temp: float,
    mc_attr: MCAttr,
    mc_params: MCParams,
    ideal_cfs: np.ndarray,
    cf: ClusterFunctionsMC,
    verbose_interval: int = 10000,
    verbose: bool = False,
):
    """Run canonical MC."""
    if verbose:
        np.set_printoptions(suppress=True)

    n_sites = mc_attr.n_total_sites
    spins = mc_attr.active_spins.astype(np.int32)
    cfs = mc_attr.cluster_functions
    beta = 1.0 / (KbEV * temp)

    score = _score(cfs, ideal_cfs)
    n_steps = mc_params.n_steps_eq * n_sites
    for mc_iter in range(n_steps):
        i, j = mc_attr.select_two_sites(spins)
        cfs_new = cfs + cf.eval_from_spin_swap(spins, [i, j])
        score_new = _score(cfs_new, ideal_cfs)
        delta = score_new - score
        threshold = np.exp(-beta * delta)
        if np.random.rand() < threshold:
            cfs = cfs_new
            score = score_new
            spins[i], spins[j] = spins[j], spins[i]

        if verbose and (mc_iter + 1) % verbose_interval == 0:
            _print_iteration(mc_iter, score, cfs)

        if np.isclose(score, 0.0):
            if verbose:
                print("SQS with zero error is found.", flush=True)
                print("Cluster functions:", flush=True)
                print(cfs)
            break

    mc_attr.active_spins = spins
    mc_attr.cluster_functions = cfs
    return (mc_attr, score)
