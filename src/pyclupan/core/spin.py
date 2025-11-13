"""Functions of calculating spin polynomials."""

import copy

import numpy as np

# @dataclass
# class ClusterFunction:
#    """Class for cluster function."""
#    cons: np.ndarray
#    spins: np.ndarray
#
#    def eval(self, spins_input: np.ndarray):
#        """Evaluate cluster function value for spin."""
#        return np.polyval(coeff, spin)
#


# def _eval_basis(coeffs: np.ndarray, spin: int):
#    """Evaluate cluster function values."""
#    return np.polyval(coeffs, spin)
#
#
# def eval_basis_prod(coeffs_cl, spin_cl):
#    """Evaluate cluster function values."""
#    return np.prod([_eval_basis(c, s) for c, s in zip(coeffs_cl, spin_cl)])
#


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
