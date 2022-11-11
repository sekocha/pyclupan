#!/usr/bin/env python 
import numpy as np
import copy
from math import sqrt

def inner_prod(coeff1, coeff2, spins):
    return np.mean(np.polyval(coeff1, spins) * np.polyval(coeff2, spins))

def normalize(coeff, spins):
    prod = np.mean(np.square(np.polyval(coeff, spins)))
    coeff_normilized = coeff / sqrt(prod)
    return np.array(coeff_normilized)

def gram_schmidt(spins):

    n_type = len(spins)
    start = np.eye(n_type)[::-1]

    cons = []
    for i, w in enumerate(start):
        update = copy.deepcopy(w)
        for j in range(i):
            update -= inner_prod(cons[j], w, spins) * cons[j]
        update = normalize(update, spins)
        cons.append(update)
    return np.array(cons)
   
if __name__ == '__main__':

    spins = [1,0,-1]
    cons = gram_schmidt(spins)
    print(cons)


