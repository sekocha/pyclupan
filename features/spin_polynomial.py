#!/usr/bin/env python 
import numpy as np
import copy
from math import *
#from numpy.polynomial import Polynomial

#def eval(coeff, spin):

#coeff1 = [1,0,2]
##coeff2 = [2,0,1]
#coeff1 = [1,0,-2/3]
#coeff2 = [1,0,-2/3]
#print(np.polyval(coeff1, spins))


def inner_prod(coeff1, coeff2, spins):
    return np.mean(np.polyval(coeff1, spins) * np.polyval(coeff2, spins))

def normalize(coeff, spins):
    prod = np.mean(np.square(np.polyval(coeff, spins)))
    coeff_normilized = coeff / sqrt(prod)
    return np.array(coeff_normilized)

def gram_schmidt(spins):

    n_type = len(spins)
    start = np.eye(n_type)[::-1]
    print(start)

    cons = []
    for i, w in enumerate(start):
        if i == 0:
            update = normalize(w, spins)
        else:
            update = copy.deepcopy(w)
            for j in range(i):
                update -= inner_prod(cons[j], w, spins) * cons[j]
            update = normalize(update, spins)
        cons.append(update)
    return np.array(cons)
   

#print(inner_prod(coeff1, coeff2, spins))

spins = [1,0,-1]
cons = gram_schmidt(spins)
print(cons)
