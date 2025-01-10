"""Functions for obtaining normal forms."""

import itertools

import numpy as np
from pypolymlp.core.data_format import PolymlpStructure

from pyclupan.core.spglib_utils import get_rotations


def _factorization(n: int):
    """Calculate factors for given integer."""
    factors = []
    i = 2
    while i <= n:
        if n % i == 0:
            factors.append(i)
            n //= i
        else:
            i += 1
    return factors


def enumerate_hnf(n: int):
    """Enumerate the entire set of Hermite normal forms.

    Parameters
    ----------
    n: Determinant of HNF.
    """
    factors = _factorization(n)
    diag_all = set()
    for index in itertools.product(*[range(3) for i in range(len(factors))]):
        diag = [1, 1, 1]
        for i, f in zip(index, factors):
            diag[i] *= f
        diag_all.add(tuple(diag))

    hnf_array = []
    for diag in sorted(diag_all):
        prod = itertools.product(*[range(diag[1]), range(diag[2]), range(diag[2])])
        for i, j, k in prod:
            hnf = np.diag(diag)
            hnf[1, 0], hnf[2, 0], hnf[2, 1] = i, j, k
            hnf_array.append(hnf)
    return np.array(hnf_array)


def _is_unimodular(M: np.ndarray, tol: float = 1e-10):
    """Check whether matrix is unimodular."""
    if np.isclose(np.linalg.det(M), 1.0):
        diff = M - np.round(M)
        if np.linalg.norm(diff) < tol:
            return True
    return False


def get_nonequivalent_hnf(
    n: int,
    st: PolymlpStructure,
    symprec: float = 1e-5,
    tol: float = 1e-10,
):
    """Enumerate a complete set of non-equivalent Hermite normal forms.

    Consider two HNfs H and G, they are equivalent if
    there exist a rotation R such that G^(-1) @ R @ H is unimodular.
    """
    hnf_array = enumerate_hnf(n)
    hnfinv_array = np.array([np.linalg.inv(hnf) for hnf in hnf_array])

    rotations = get_rotations(st, symprec=symprec)
    dots = np.array([rotations @ hnf for hnf in hnf_array])

    n_hnf = len(hnf_array)
    reps = np.arange(n_hnf, dtype=int)
    # time consuming part
    for i1, i2 in itertools.combinations(range(n_hnf), 2):
        if reps[i1] == i1 and reps[i2] == i2:
            for d in dots[i2]:
                if _is_unimodular(hnfinv_array[i1] @ d):
                    reps[i2] = i1
                    break

    nonequiv_hnfs = hnf_array[np.unique(reps)]
    return nonequiv_hnfs


def snf(mat: np.ndarray):
    """Calculating Smith normal form.

    S = U * matrix * V
    matrix = U^(-1) * S * V^(-1)
    """
    from smithnormalform import matrix, snfproblem, z

    size1, size2 = np.array(mat).shape
    mat1 = matrix.Matrix(size1, size2, [z.Z(int(h2)) for h1 in mat for h2 in h1])
    prob = snfproblem.SNFProblem(mat1)
    prob.computeSNF()
    S = [[int(prob.J.get(i, j).a) for j in range(3)] for i in range(3)]
    U = [[int(prob.S.get(i, j).a) for j in range(3)] for i in range(3)]
    V = [[int(prob.T.get(i, j).a) for j in range(3)] for i in range(3)]
    trans = np.eye(3)
    for i in range(3):
        if S[i][i] < 0:
            trans[i, i] = -1
            S[i][i] *= -1
    return np.array(S), np.dot(trans, np.array(U)), np.array(V)
