#!/usr/bin/env python
import numpy as np
from sympy.combinatorics.permutations import Permutation
import pynauty

def construct_graph(perm):

    n_sites = perm.shape[1]
   
    # must be modified using the cyclic feature of permutation
    g = pynauty.Graph(n_sites)
    for j in range(n_sites):
        #connect_vertex = sorted(set(perm[:,j]))
        connect_vertex = list(perm[:,j])
        print(connect_vertex)
        if connect_vertex[0] != -1:
            g.connect_vertex(j, connect_vertex)
    return g

def permutation_isomorphism(perm1, perm2):

    if perm1.shape != perm2.shape:
        return False, []

    n_sites = perm1.shape[1]

    g1 = construct_graph(perm1)
    g2 = construct_graph(perm2)
    if pynauty.isomorphic(g1, g2):
        print(pynauty.canon_label(g1))
        print(pynauty.canon_label(g2))

if __name__ == '__main__':

    a1 = Permutation([0, 1, 2, 3])
    a2 = Permutation([1, 2, 3, 0])
    a3 = Permutation([2, 3, 0, 1])
    a4 = Permutation([3, 0, 1, 2])
    print(a1, a1.cyclic_form)
    print(a2, a2.cyclic_form)
    print(a3, a3.cyclic_form)
    print(a4, a4.cyclic_form)
    b1 = Permutation([1, 0, 2, 3])
    b2 = Permutation([0, 2, 3, 1])
    b3 = Permutation([2, 3, 1, 0])
    b4 = Permutation([3, 1, 0, 2])
    print(b1, b1.cyclic_form)
    print(b2, b2.cyclic_form)
    print(b3, b3.cyclic_form)
    print(b4, b4.cyclic_form)

#    a = Permutation([2, 0, 3, 1, 5, 4])
#    b = Permutation([3, 1, 2, 5, 4, 0])
#    print(a, a.cyclic_form)
#    print(b, b.cyclic_form)

#    g = Graph(n_sites)
#    for j in range(n_sites):
#        connected_vertex = sorted(set(perm1[:,j]))
#        if connected_vertex[0] != -1:
#            g.connect_vertex(j, connect_vertex)
#
#
        
