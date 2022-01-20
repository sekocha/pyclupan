#!/usr/bin/env python
import numpy as np
import itertools
import time

from smithnormalform import matrix, snfproblem, z
def snf2(mat):

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
def snf3(mat):

    ## S = U * matrix * V
    ## matrix = U^(-1) * S * V^(-1)
    mat1 = [[int(m2) for m2 in m1] for m1 in mat]
    S, U, V = smf(sympy.Matrix(np.array(mat1)))
    S = sympy.matrix2numpy(S)
    U = sympy.matrix2numpy(U).astype(np.float)
    V = sympy.matrix2numpy(V).astype(np.float)
    return S, U, V

