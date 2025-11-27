"""Functions of calculating spin polynomials."""

import copy

import numpy as np


def _polyval(coeffs: np.ndarray, spins: np.ndarray, orders: np.ndarray):
    """Evaluate values of polynomial function."""
    values = np.zeros(len(orders))
    for c, order in zip(coeffs, orders):
        values += c * np.power(spins, order)
    return values


def _inner_prod(
    coeffs1: np.ndarray,
    coeffs2: np.ndarray,
    spins: np.ndarray,
    orders: list,
):
    """Calculate inner products between two polynomials."""
    return np.mean(_polyval(coeffs1, spins, orders) * _polyval(coeffs2, spins, orders))


def _normalize(coeffs: np.ndarray, spins: np.ndarray, orders: list):
    """Normalize polynomial coefficients."""
    prod = np.mean(np.square(_polyval(coeffs, spins, orders)))
    coeffs_normalized = coeffs / np.sqrt(prod)
    return np.array(coeffs_normalized)


def gram_schmidt(spins: np.ndarray, orders: list):
    """Construct orthogonal point functions from spin values using Gram-Schmidt.

    Return
    ------
    cons: Coefficients of complete orthonomal system.
          Binary point function: cons[0] * spin + cons[1]
          Ternary point function:  cons[0] * spin**2 + cons[1] * spin + cons[2]
          k-ary point function: cons[0] * spin**(k-1) + ... + cons[-1]
    """
    n_type = len(spins)
    start = np.eye(n_type)[::-1]

    cons = []
    for i, w in enumerate(start):
        update = copy.deepcopy(w)
        for j in range(i):
            update -= _inner_prod(cons[j], w, spins, orders) * cons[j]
        update = _normalize(update, spins, orders)
        cons.append(update)
    return np.array(cons)


def define_spins(n_type: int):
    """Define spins."""
    if n_type == 1:
        spin_array = [-1000]
    elif n_type == 2:
        spin_array = [1, -1]
    elif n_type == 3:
        spin_array = [1, 0, -1]
    elif n_type == 4:
        spin_array = [2, 1, 0, -1]
    elif n_type == 5:
        spin_array = [2, 1, 0, -1, 2]
    elif n_type == 6:
        spin_array = [3, 2, 1, 0, -1, 2]
    else:
        raise RuntimeError("Spin values not defined.")

    return spin_array


# def set_spins(element_lattice: list):
#     """Define spin values."""
#     # active_elements = [e2 for e1 in element_lattice if len(e1) > 1 for e2 in e1]
#     # basis_size = len(active_elements)
#
#     spins_lattice, basis_set, basis_lattice = [], [], []
#     basis_id, begin_order = 0, 0
#     for ele in element_lattice:
#         spins_sublattice = define_spins(len(ele))
#         spins_lattice.append(spins_sublattice)
#
#         ids = []
#         if len(ele) > 1:
#             end_order = begin_order + len(ele)
#             orders = list(range(begin_order, end_order))
#             basis_local = np.zeros((len(ele), basis_size))
#             basis_local[:, begin_order:end_order] = gram_schmidt(
#                 spins_sublattice, orders=orders
#             )
#             for basis in basis_local:
#                 basis_set.append(basis)
#                 ids.append(basis_id)
#                 basis_id += 1
#             begin_order = end_order
#         basis_lattice.append(ids)
#     basis_set = np.array(basis_set)
#
#     return spins_lattice, basis_lattice, basis_set
#


def eval_cluster_functions(coeffs: np.ndarray, spins_from_orbit: np.ndarray):
    """Evaluate cluster functions.

    Parameters
    ----------
    coeffs: Polynomial coefficients of spin polynomials for atom sites in cluster.
            shape: (n_sites, cluster_order)
    spins_from_orbit: Spin values of clusters in structures.
            shape: (n_structure, orbit_size, cluster_order)

    Return
    ------
    Cluster functions.
        shape: (n_structure)
    """
    vals = np.zeros(spins_from_orbit.shape)
    for i, c in enumerate(coeffs):
        vals[:, :, i] = np.polyval(c, spins_from_orbit[:, :, i])
    cf = np.average(np.prod(vals, axis=2), axis=1)
    return cf
