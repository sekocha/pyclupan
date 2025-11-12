"""Functions of calculating spin polynomials."""

import copy

import numpy as np


def _eval_basis(coeff, spin):
    return np.polyval(coeff, spin)


def eval_basis_prod(coeffs_cl, spin_cl):
    return np.prod([_eval_basis(c, s) for c, s in zip(coeffs_cl, spin_cl)])


def _inner_prod(coeff1, coeff2, spins):
    return np.mean(np.polyval(coeff1, spins) * np.polyval(coeff2, spins))


def _normalize(coeff: np.ndarray, spins: np.ndarray):
    """Normalize polynomial coefficients."""
    prod = np.mean(np.square(np.polyval(coeff, spins)))
    coeff_normilized = coeff / np.sqrt(prod)
    return np.array(coeff_normilized)


def gram_schmidt(spins: np.ndarray):
    """Calculate orthogonal cluster functions from spin values using Gram-Schmidt.

    Return
    ------
    cons: Coefficients of complete orthonomal system.
          Binary cluster functions: cons[0] * spin + cons[1]
          Ternary cluster function:  cons[0] * spin**2 + cons[1] * spin + cons[2]
          k-ary cluster function: cons[0] * spin**(k-1) + ... + cons[-1]
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


if __name__ == "__main__":

    spins = [1, 0, -1]
    cons = gram_schmidt(spins)
    print(cons)
    spins = [2, 1, 0, -1]
    cons = gram_schmidt(spins)
    print(cons)
