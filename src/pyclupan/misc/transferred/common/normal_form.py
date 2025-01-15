#!/usr/bin/env python
# for test
import argparse
import itertools
import time

import numpy as np
from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure
from smithnormalform import matrix, snfproblem, z

from pyclupan.common.symmetry import get_rotations

# def snf_sage(mat):
#
#    import sage.all
#    from sage.matrix.constructor import Matrix
#
#    ## S = U * matrix * V
#    ## matrix = U^(-1) * S * V^(-1)
#    #mat1 = sage.matrix.constructor.Matrix(mat)
#    mat1 = Matrix(mat)
#    S, U, V = mat1.smith_form()
#    return np.array(S), np.array(U), np.array(V)


def snf(mat):

    ## S = U * matrix * V
    ## matrix = U^(-1) * S * V^(-1)
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


def enumerate_hnf(n):

    factors = factorization(n)
    diag_all = set()
    for index in itertools.product(*[range(3) for i in range(len(factors))]):
        diag = [1, 1, 1]
        for i, f in zip(index, factors):
            diag[i] *= f
        diag_all.add(tuple(diag))

    hnf_array = []
    for diag in sorted(diag_all):
        for i, j, k in itertools.product(
            *[range(diag[1]), range(diag[2]), range(diag[2])]
        ):
            hnf = np.diag(diag)
            hnf[1, 0], hnf[2, 0], hnf[2, 1] = i, j, k
            hnf_array.append(hnf)

    return hnf_array


def factorization(n):
    factors = []
    i = 2
    while i <= n:
        if n % i == 0:
            factors.append(i)
            n //= i
        else:
            i += 1
    return factors


def get_nonequivalent_hnf(n, st: Structure, symprec=1e-5, tol=1e-10):

    rotations = get_rotations(st, symprec=symprec)
    hnf_array = enumerate_hnf(n)
    hnfinv_array = [np.linalg.inv(hnf) for hnf in hnf_array]
    dots = [[np.dot(rot, hnf) for rot in rotations] for hnf in hnf_array]

    snf_array = [tuple(sorted(np.diag(snf(h)[0]))) for h in hnf_array]

    ############# time consuming part #############
    n_hnf = len(hnf_array)
    reps = dict(zip(range(n_hnf), range(n_hnf)))
    for i1, i2 in itertools.combinations(range(n_hnf), 2):
        if reps[i1] == i1 and reps[i2] == i2 and snf_array[i1] == snf_array[i2]:
            hnf1inv = hnfinv_array[i1]
            # whether H^(-1) (rot * H') is unimodular
            for d in dots[i2]:
                mat1 = np.dot(hnf1inv, d)
                diff = mat1 - np.trunc(mat1)
                if np.linalg.norm(diff) < tol:
                    reps[i2] = i1
                    break
    ############# time consuming part end #############

    return [hnf_array[i] for i in set(reps.values())]


if __name__ == "__main__":

    #    ps = argparse.ArgumentParser()
    #    ps.add_argument('-p',
    #                    '--poscar',
    #                    type=str,
    #                    default='POSCAR',
    #                    help='poscar file for primitive cell')
    #    args = ps.parse_args()
    #
    #    prim = Poscar(args.poscar).get_structure_class()
    #    hnf_array = get_nonequivalent_hnf(8, prim)
    #    print(' number of Hermite normal form =', len(hnf_array))
    #
    mat1 = np.array([[2, 0, 0], [1, 2, 0], [1, 0, 3]])
    S, U, V = snf(mat1)
    print(S)
    print(U)
    print(V)


# def get_nonequivalent_hnf(n, st: Structure, symprec=1e-5):
#
#    t0 = time.time()
#    rotations = get_rotations(st, symprec=symprec)
#    hnf_array = enumerate_hnf(n)
#    hnfinv_array = [np.linalg.inv(hnf) for hnf in hnf_array]
#    dots = [[np.dot(rot, hnf) for rot in rotations] for hnf in hnf_array]
#
#    t1 = time.time()
#    ############# slow part #############
#    n_hnf = len(hnf_array)
#    reps = dict(zip(range(len(hnf_array)), range(len(hnf_array))))
#    for i1, i2 in itertools.combinations(range(len(hnf_array)),2):
#        if reps[i1] == i1 and reps[i2] == i2:
#            hnf1inv = hnfinv_array[i1]
#            # whether H^(-1) (rot * H') is unimodular
#            for d in dots[i2]:
#                mat1 = np.dot(hnf1inv, d)
#                if np.linalg.norm(np.round(mat1)-mat1) < 1e-10:
#    #                and abs(np.linalg.det(mat1)) - 1 < 1e-10):
#                    reps[i2] = i1
#                    break
#    ############# slow part end #############
#    t2 = time.time()
#    print(t1-t0, t2-t1)
#
#    return [hnf_array[i] for i in set(reps.values())]
#
