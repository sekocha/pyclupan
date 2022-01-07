#!/usr/bin/env python
import numpy as np
import itertools
import time

from mlptools.common.structure import Structure
from pyclupan.common.symmetry import get_rotations

from smithnormalform import matrix, snfproblem, z
def snf(mat):

    ## S = U * matrix * V
    ## matrix = U^(-1) * S * V^(-1)
    size1, size2 = np.array(mat).shape
    mat1 = matrix.Matrix(size1,size2,[z.Z(int(h2)) for h1 in mat for h2 in h1])
    prob = snfproblem.SNFProblem(mat1)
    prob.computeSNF()
    S = [[int(prob.J.get(i,j).a) for j in range(3)] for i in range(3)]
    U = [[int(prob.S.get(i,j).a) for j in range(3)] for i in range(3)]
    V = [[int(prob.T.get(i,j).a) for j in range(3)] for i in range(3)]
    trans = np.eye(3)
    for i in range(3):
        if S[i][i] < 0: 
            trans[i,i] = -1
            S[i][i] *= -1
    return np.array(S), np.dot(trans, np.array(U)), np.array(V)

import sympy
from desr.matrix_normal_forms import smf, is_smf
def snf2(mat):

    ## S = U * matrix * V
    ## matrix = U^(-1) * S * V^(-1)
    mat1 = [[int(m2) for m2 in m1] for m1 in mat]
    S, U, V = smf(sympy.Matrix(np.array(mat1)))
    S = sympy.matrix2numpy(S)
    U = sympy.matrix2numpy(U).astype(np.float)
    V = sympy.matrix2numpy(V).astype(np.float)
    return S, U, V

def enumerate_hnf(n):

    factors = factorization(n)
    diag_all = set()
    for index in itertools.product(*[range(3) for i in range(len(factors))]):
        diag = [1,1,1]
        for i, f in zip(index, factors):
            diag[i] *= f
        diag_all.add(tuple(diag))

    hnf_array = []
    for diag in sorted(diag_all):
        for i,j,k in itertools.product\
            (*[range(diag[1]),range(diag[2]),range(diag[2])]):
            hnf = np.diag(diag)
            hnf[1,0], hnf[2,0], hnf[2,1] = i, j, k
            hnf_array.append(hnf)

    return hnf_array

def factorization(n):
    factors = []
    i = 2
    while i <= n:
        if n%i == 0:
            factors.append(i)
            n //= i
        else:
            i += 1
    return factors

def get_nonequivalent_hnf(n, st: Structure, symprec=1e-5):

    rotations = get_rotations(st, symprec=symprec)
    hnf_array = enumerate_hnf(n)
    hnfinv_array = [np.linalg.inv(hnf) for hnf in hnf_array]
    dots = [[np.dot(rot, hnf) for rot in rotations] for hnf in hnf_array]

    t1 = time.time()
    ############# slow part #############
    reps = dict(zip(range(len(hnf_array)), range(len(hnf_array))))
    for i1, i2 in itertools.combinations(range(len(hnf_array)),2):
        if reps[i1] == i1 and reps[i2] == i2:
            hnf1inv = hnfinv_array[i1]
            # whether H^(-1) (rot * H') is unimodular
            for d in dots[i2]:
                mat1 = np.dot(hnf1inv, d)
                if np.linalg.norm(np.round(mat1)-mat1) < 1e-10:
    #                and abs(np.linalg.det(mat1)) - 1 < 1e-10):
                    reps[i2] = i1
                    break
    ############# slow part end #############
    t2 = time.time()

    return [hnf_array[i] for i in set(reps.values())]
