"""Functions for linear algebra."""

import itertools

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure
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
    det_hnfinv = np.linalg.det(hnfinv_array)

    rotations = get_rotations(st, symprec=symprec)
    RHs = np.array([rotations @ hnf for hnf in hnf_array])
    det_RHs = np.linalg.det(RHs)

    n_hnf = len(hnf_array)
    reps = np.arange(n_hnf, dtype=int)
    for i1, i2 in itertools.combinations(range(n_hnf), 2):
        if reps[i1] == i1 and reps[i2] == i2:
            det1 = np.isclose(det_hnfinv[i1] * det_RHs[i2], 1.0)
            dots = (hnfinv_array[i1] @ RHs[i2, det1]).reshape((-1, 9))
            norm_diff = np.linalg.norm(dots - np.round(dots), axis=1)
            if np.any(norm_diff < tol):
                reps[i2] = i1

    reps = np.array([i for i, j in enumerate(reps) if i == j])
    nonequiv_hnfs = hnf_array[reps]
    return nonequiv_hnfs


def snf(mat: np.ndarray):
    """Calculate Smith normal form.

    SNF = U @ matrix @ V
    matrix = U^(-1) @ SNF @ V^(-1)
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
    return np.array(S), trans @ np.array(U), np.array(V)
