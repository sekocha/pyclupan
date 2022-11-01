#!/usr/bin/env python
import numpy as np
import os, sys
import argparse
import time

import signal
import warnings

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure

from pyclupan.common.supercell import supercell
from pyclupan.dd.dd_supercell import DDSupercell
from pyclupan.dd.dd_enumeration import DDEnumeration

#from pyclupan.common.normal_form import get_nonequivalent_hnf

#from ddtools.labeling import nonequivalent_labelings, graphs_to_labelings


if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    warnings.simplefilter('ignore')

    ps = argparse.ArgumentParser()
    ps.add_argument('-p',
                    '--poscar',
                    type=str,
                    default='POSCAR',
                    help='poscar file for primitive cell')
    args = ps.parse_args()

#    ps.add_argument('-l','--lattice',type=int,nargs='*',default=[0],\
#        help='sublattice index for substitution')
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

    axis, positions, n_atoms, _, _ = Poscar(args.poscar).get_structure()

    H = np.array([[1,0,0], [0,1,0], [0,0,8]])
    axis_s, positions_s, n_atoms_s = supercell(H, axis, positions, n_atoms)
    st_s = Structure(axis_s, positions_s, n_atoms_s)

    dd_sup = DDSupercell(axis_s, H, axis, 
                         positions=positions_s,
                         n_sites=n_atoms_s,
                         n_elements=5,
                         occupation=[[0],[1],[1],[2],[2]],
                         one_of_k_rep=True,
                         inactive_elements=[4])

    comp = [None, None, None, 5/6, 1/6]
    comp_lb = [None for i in range(n_elements)]
    comp_ub = [None for i in range(n_elements)]
    charge = [2,2,3,-2,0]

    dd_enum = DDEnumeration(dd_sup, structure=st_s)
    gs_all = dd_enum.one_of_k()
    print(' number of structures (one-of-k)        =', gs_all.len())

    gs_charge = dd_enum.charge_balance(charge, comp=comp)
    gs_all &= gs_charge
    print(' number of structures (charge balanced) =', gs_all.len())

    if comp.count(None) != len(comp):
        gs_comp = dd_enum.composition(comp)
        gs_all &= gs_comp
        print(' number of structures (composition)     =', gs_all.len())

    if comp_lb.count(None) != len(comp_lb) \
        or comp_ub.count(None) != len(comp_ub):
        gs_comp = dd_enum.composition_range(comp_lb, comp_ub)
        gs_all &= gs_comp
        print(' number of structures (composition)     =', gs_all.len())

    t1 = time.time()
    gs_noneq = dd_enum.nonequivalent_permutations()
    gs_all &= gs_noneq
    t2 = time.time()
    print(' number of structures (nonequiv.)       =', gs_all.len())
    print(' elapsed time (nonequiv.)       =', t2-t1)

    labelings = dd_sup.convert_graphs_to_labelings(gs_all)
    t3 = time.time()
    print(' elapsed time (labeling)        =', t3-t2)


###def labelings_to_structures(uniq, st):
###
###    axis, positions, n_atoms \
###        = st.get_axis(), st.get_positions(), st.get_n_atoms()
###
###    st_array = []
###    for l in uniq:
###        i0, i1 = list(np.where(l == 0)[0]), list(np.where(l == 1)[0])
###        index = i0 + i1 + list(range(n_atoms[0], sum(n_atoms)))
###        n_atoms1 = [len(i0), len(i1)] + n_atoms[1:]
###        positions1 = positions[:,index]
###        st_array.append(Structure(axis, positions1, n_atoms1))
###
###    return st_array
###
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
####
####    uniq_array = []
####    for i, (perm, perm_lt) in enumerate\
####        (zip(nonequiv_permutation, nonequiv_permutation_lt)):
####        print(' # permutation', i + 1)
####        uniq = nonequivalent_labelings(labelings,perm,perm_lt,swap=args.swap)
####        uniq_array.append(uniq)
####        print(uniq)
####
####    ########################
####    # structure output
####    ########################
####    if args.output_structure == True:
####        print(' generating structure files')
####        st_array = []
####        for i, uniq in enumerate(uniq_array):
####            for j in np.where(np.array(permutation_map) == i)[0]:
####                st = supercell_array[j]
####                st_array.extend(labelings_to_structures(uniq, st))
####        print(' number of all nonequivalent subgraphs =', len(st_array))
####
####        joblib.dump(st_array, 'structure.pkl')
####        os.makedirs('poscars', exist_ok=True)
####        for i, st in enumerate(st_array):
####            name = 'poscars/poscar-'+str(i+1).zfill(4)
####            st.print_poscar_tofile(filename=name)
####    else:
####        n_sum = 0
####        for i, uniq in enumerate(uniq_array):
####            n_sum += uniq.shape[0] \
####                * len(np.where(np.array(permutation_map) == i)[0])
####        print(' number of all nonequivalent subgraphs =', n_sum)
####
####
