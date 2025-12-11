"""Functions of calculating spin polynomials."""

import copy

import numpy as np


def _inner_prod(coeffs1: np.ndarray, coeffs2: np.ndarray, spins: np.ndarray):
    """Calculate inner products between two polynomials."""
    return np.mean(np.polyval(coeffs1, spins) * np.polyval(coeffs2, spins))


def _normalize(coeffs: np.ndarray, spins: np.ndarray):
    """Normalize polynomial coefficients."""
    prod = np.mean(np.square(np.polyval(coeffs, spins)))
    coeffs_normalized = coeffs / np.sqrt(prod)
    return np.array(coeffs_normalized)


def gram_schmidt(spins: np.ndarray):
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
            update -= _inner_prod(cons[j], w, spins) * cons[j]
        update = _normalize(update, spins)
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


def set_spins(element_lattice: list):
    """Define spin values."""
    spins_lattice, basis_set, basis_lattice = [], [], []
    basis_id = 0
    for ele in element_lattice:
        spins_sublattice = define_spins(len(ele))
        spins_lattice.append(spins_sublattice)

        ids = []
        if len(ele) > 1:
            for basis in gram_schmidt(spins_sublattice):
                if not np.allclose(basis[:-1], 0.0) or not np.isclose(basis[-1], 1.0):
                    basis_set.append(basis)
                    ids.append(basis_id)
                    basis_id += 1
        basis_lattice.append(ids)

    return spins_lattice, basis_lattice, basis_set


#
#
# def set_spins(element_lattice: list):
#     """Define spin values."""
#     active_elements = [e2 for e1 in element_lattice if len(e1) > 1 for e2 in e1]
#     inactive_elements = [e1[0] for e1 in element_lattice if len(e1) == 1]
#     active_spins = define_spins(len(active_elements))
#     map_spin = dict()
#     for e, s in zip(active_elements, active_spins):
#         map_spin[e] = s
#     for e in inactive_elements:
#         map_spin[e] = -1000
#
#     spin_lattice = [[map_spin[e] for e in ele] for ele in element_lattice]
#     print(spin_lattice)
#
#     for ele, spin in zip(element_lattice, spin_lattice):
#         basis_set = gram_schmidt(spin)
#         print(basis_set)
#
#
#     spins_lattice, basis_set, basis_lattice = [], [], []
#     basis_id = 0
#     for ele in element_lattice:
#         spins_sublattice = define_spins(len(ele))
#         spins_lattice.append(spins_sublattice)
#
#         ids = []
#         if len(ele) > 1:
#             for basis in gram_schmidt(spins_sublattice):
#                 if not np.allclose(basis[:-1], 0.0) or not np.isclose(basis[-1], 1.0):
#                     basis_set.append(basis)
#                     ids.append(basis_id)
#                     basis_id += 1
#         basis_lattice.append(ids)
#
#     return spins_lattice, basis_lattice, basis_set
#


def eval_cluster_functions(
    coeffs: np.ndarray,
    spins_from_orbit: np.ndarray,
    return_array: bool = False,
):
    """Evaluate cluster functions.

    Parameters
    ----------
    coeffs: Polynomial coefficients of spin polynomials for atom sites in cluster.
            shape: (n_sites, cluster_order)
    spins_from_orbit: Spin values of clusters in structures.
            shape: (n_structure, orbit_size, cluster_order)
                    or (orbit_size, cluster_order)

    Return
    ------
    Cluster functions.
        shape: (n_structure) or float
    """
    vals = np.zeros(spins_from_orbit.shape)
    if spins_from_orbit.ndim == 3:
        for i, c in enumerate(coeffs):
            vals[:, :, i] = np.polyval(c, spins_from_orbit[:, :, i])
        if return_array:
            # cf = np.prod(vals, axis=2)
            cf = np.multiply.reduce(vals, axis=2)
        else:
            # cf = np.average(np.prod(vals, axis=2), axis=1)
            cf = np.average(np.multiply.reduce(vals, axis=2), axis=1)
    elif spins_from_orbit.ndim == 2:
        for i, c in enumerate(coeffs):
            vals[:, i] = np.polyval(c, spins_from_orbit[:, i])
        if return_array:
            cf = np.multiply.reduce(vals, axis=1)
        else:
            cf = np.average(np.multiply.reduce(vals, axis=1))
    return cf
