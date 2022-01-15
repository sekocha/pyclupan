#!/usr/bin/env python
import numpy as np
import os, sys
import argparse
import time

import signal
import warnings

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure
from pyclupan.common.supercell import supercell_from_structure
from pyclupan.common.symmetry import get_permutation

from pyclupan.dd.dd_node import DDNodeHandler
from pyclupan.dd.dd_constructor import DDConstructor

if __name__ == '__main__':

    # Examples
    #  graph.py -p structures/perovskite-unitcell 
    #           -o 0 -o 1 -o 2 -o 2
    #                = occupation: [[0],[1],[2],[2]],
    #
    #  graph.py -p structures/perovskite-unitcell 
    #           -e 0 -e 1 -e 2 3
    #                = elements_lattice: [[0],[1],[2,3]]

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    warnings.simplefilter('ignore')

    ps = argparse.ArgumentParser()
    ps.add_argument('-p',
                    '--poscar',
                    type=str,
                    default='POSCAR',
                    help='poscar file for primitive cell')
    ps.add_argument('-e',
                    '--elements',
                    nargs='*',
                    type=int,
                    action='append',
                    default=None,
                    help='elements on a lattice')
    ps.add_argument('-o',
                    '--occupation',
                    nargs='*',
                    type=int,
                    action='append',
                    default=None,
                    help='elements on a lattice')
    args = ps.parse_args()

#    ps.add_argument('-c','--comp',type=float,nargs='*',default=None,\
#        help='composition [comp (type 0): c[0], comp (type 1): 1 - c[0]]')
#    ps.add_argument('--incomplete',action='store_true',\
#        help='Including incomplete structures (valid if comp == None)')
#    ps.add_argument('--swap',action='store_true',\
#        help='Including swapped structures (valid if comp == None)')
#    ps.add_argument('--hnf',type=int, nargs=9, default=None,
#        help='Hermite normal form')
#    ps.add_argument('--index',type=int, default=None,
#        help='Determinant of Hermite normal form (valid if hnf == None)')
#    ps.add_argument('--output_structure',action='store_true',\
#        help='Generating poscars and structure.pkl')

    st_prim = Poscar(args.poscar).get_structure_class()

    comp = [None, None, 2/3, 1/3]
    comp_lb = [None]
    comp_ub = [None]
#    comp = [None, None, None, None]
#    comp_lb = [None, None, 4/8, 3/8]
#    comp_ub = [None, None, 5/8, 4/8]

    H = np.array([[1,0,0], [0,1,0], [0,0,8]])
    st_sup = supercell_from_structure(st_prim, H, return_structure=True)

    print('computing permutations')
    site_perm = get_permutation(st_sup)
    print('computing permutations (finished)')

    dd_handler = DDNodeHandler(n_sites=st_sup.n_atoms,
                               occupation=args.occupation,
                               elements_lattice=args.elements,
                               one_of_k_rep=False)

    dd_const = DDConstructor(dd_handler)
    gs_all = dd_const.enumerate_nonequiv_configs(site_permutations=site_perm,
                                                 comp=comp,
                                                 comp_lb=comp_lb,
                                                 comp_ub=comp_ub)
    t1 = time.time()
    labelings = dd_handler.convert_graphs_to_labelings(gs_all)
    t2 = time.time()
    print(' elapsed time (labeling)    =', t2-t1)




###def get_nonequiv_permutation(hnf_array, supercell_array, size):
###
###    n_hnf = len(hnf_array)
###    permutation_array, permutation_lt_array = [], []
###    for hnf, st in zip(hnf_array, supercell_array):
###        permutation, permutation_lt \
###            = get_permutation(st,superperiodic=True,hnf=hnf)
###        permutation, permutation_lt \
###            = permutation[:,:size], permutation_lt[:,:size]
###        permutation = np.array(sorted([tuple(p) for p in permutation]))
###
###        permutation_array.append(permutation)
###        permutation_lt_array.append(permutation_lt)
###
###    hnfmap = dict(zip(range(n_hnf), range(n_hnf)))
###    for i, j in itertools.combinations(range(n_hnf),2):
###        if hnfmap[i] == i and hnfmap[j] == j:
###            perm1, perm2 = permutation_array[i], permutation_array[j]
###            if perm1.shape == perm2.shape and np.all(perm1 - perm2 == 0):
###                hnfmap[j] = i
###
###    uniq_map = list(set(hnfmap.values()))
###    nonequiv_permutation = [permutation_array[i] for i in uniq_map]
###    nonequiv_permutation_lt = [permutation_lt_array[i] for i in uniq_map]
###
###    permutation_map = [uniq_map.index(v) for k, v in sorted(hnfmap.items())]
###    return nonequiv_permutation, nonequiv_permutation_lt, permutation_map
###
###           
###
####    if hnf is not None:
####        hnf_array = [hnf]
####    else:
####        hnf_array = get_nonequivalent_hnf(supercell_size, st_p)
####
####    supercell_array = [supercell_from_structure\
####        (st_p, hnf, return_structure=True) for hnf in hnf_array]
####    print(' number of HNFs =', len(hnf_array))
####
####    print(' computing permutation isomorphism')
####    nonequiv_permutation, nonequiv_permutation_lt, permutation_map \
####        = get_nonequiv_permutation(hnf_array, supercell_array, size)
####    print(' number of nonequivalent permutations =', len(nonequiv_permutation))
###
