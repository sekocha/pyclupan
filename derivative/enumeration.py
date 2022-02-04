#!/usr/bin/env python
import numpy as np
import argparse
import os, sys
import time
import joblib
import itertools
from fractions import Fraction

import signal
import warnings

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure

from pyclupan.common.supercell import Supercell
from pyclupan.common.symmetry import get_permutation
from pyclupan.common.normal_form import get_nonequivalent_hnf
from pyclupan.common.io.yaml import Yaml

from pyclupan.dd.dd_node import DDNodeHandler
from pyclupan.dd.dd_constructor import DDConstructor
from pyclupan.derivative.derivative import DSSet

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../c++/lib')
import pyclupancpp

def set_compositions(comp_in, n_elements):

    comp = [None for i in range(n_elements)]
    if comp_in is not None:
        for i in range(round(len(comp_in)/2)):
            ele, c = int(comp_in[2*i]), Fraction(comp_in[2*i+1])
            comp[ele] = c
    return comp

#from pyclupan.common.isomorphism import permutation_isomorphism
def get_nonequiv_permutation(primitive_cell, size, inactive_sites=None):

    hnf_all = get_nonequivalent_hnf(size, primitive_cell)

    sup_cell_all, site_perm_all = [], []
    for hnf in hnf_all:
        sup = Supercell(st_prim=primitive_cell, hnf=hnf)
        sup_cell = sup.get_supercell()

        site_perm = get_permutation(sup_cell)
        site_perm[:,np.array(inactive_sites)] = -1
        normalized_site_perm \
            = np.array(sorted(set([tuple(p) for p in site_perm])))

        sup_cell_all.append(sup_cell)
        site_perm_all.append(normalized_site_perm)

    t1 = time.time()
    n_hnf = len(hnf_all)
    group = dict(zip(range(n_hnf), range(n_hnf)))
    for i, j in itertools.combinations(range(n_hnf),2):
        if group[i] == i and group[j] == j:
            perm1, perm2 = site_perm_all[i], site_perm_all[j]
            iso, map_sites = permutation_isomorphism(perm1, perm2)
            if iso:
                group[j] = i
    t2 = time.time()
    print(t2-t1)
    print(group)

#
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
#
if __name__ == '__main__':

    # Examples
    #  enumeration.py -p structures/perovskite-unitcell 
    #                   -o 0 -o 1 -o 2 -o 2
    #       = occupation: [[0],[1],[2],[2]],
    #
    #  enumeration.py -p structures/perovskite-unitcell 
    #                   -e 0 -e 1 -e 2 3
    #       = elements_lattice: [[0],[1],[2,3]]
    #
    #  enumeration.py -p structures/perovskite-unitcell 
    #                   -e 0 -e 1 -e 2 3
    #                   -c 2 2/3 3 1/3
    #

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
    ps.add_argument('-c',
                    '--comp',
                    nargs='*',
                    type=str,
                    default=None,
                    help='composition (n_elements / n_sites)')
    ps.add_argument('--comp_lb',
                    nargs='*',
                    type=str,
                    default=None,
                    help='Lower bound of composition (n_elements / n_sites)')
    ps.add_argument('--comp_ub',
                    nargs='*',
                    type=str,
                    default=None,
                    help='Upper bound of composition (n_elements / n_sites)')
    ps.add_argument('--hnf',
                    type=int, 
                    nargs=9, 
                    default=None,
                    help='Hermite normal form')
    ps.add_argument('-n',
                    '--n_expand',
                    type=int, 
                    default=None,
                    help='Determinant of Hermite normal form')
    ps.add_argument('--nodump',
                    action='store_true',
                    help='No dump file of DSSet object')
    args = ps.parse_args()

    if args.occupation is not None:
        n_elements = len(args.occupation)
    elif args.elements is not None:
        n_elements = max([e2 for e1 in args.elements for e2 in e1]) + 1
    else: 
        n_elements = 2

    comp = set_compositions(args.comp, n_elements)
    comp_lb = set_compositions(args.comp_lb, n_elements)
    comp_ub = set_compositions(args.comp_ub, n_elements)

    st_prim = Poscar(args.poscar).get_structure_class()

    if args.hnf is not None:
        hnf_all = [args.hnf]
    else:
        hnf_all = get_nonequivalent_hnf(args.n_expand, st_prim)

    n_sites = list(np.array(st_prim.n_atoms) * args.n_expand)
    dd_handler = DDNodeHandler(n_sites=n_sites,
                               occupation=args.occupation,
                               elements_lattice=args.elements,
                               one_of_k_rep=False)

#    get_nonequiv_permutation(st_prim, 
#                             n_expand, 
#                             inactive_sites=dd_handler.inactive_sites)

    ds_set_all = []
    for idx, H in enumerate(hnf_all):
        sup = Supercell(st_prim=st_prim, hnf=H)
        st_sup = sup.get_supercell()

        site_perm, site_perm_lt = get_permutation(st_sup, 
                                                  superperiodic=True, 
                                                  hnf=H)
#        site_perm = get_permutation(st_sup)

        dd_const = DDConstructor(dd_handler)
        gs = dd_const.enumerate_nonequiv_configs(site_permutations=site_perm,
                                                 comp=comp,
                                                 comp_lb=comp_lb,
                                                 comp_ub=comp_ub)
        t1 = time.time()
        res = dd_handler.convert_graphs_to_labelings(gs)
        active_labelings, inactive_labeling, active_sites, inactive_sites = res
        t2 = time.time()
        print(' elapsed time (labeling)    =', t2-t1)

        ds_set = DSSet(active_labelings=active_labelings,
                       inactive_labeling=inactive_labeling,
                       active_sites=active_sites,
                       inactive_sites=inactive_sites,
                       primitive_cell=st_prim,
                       n_expand=args.n_expand,
                       elements=dd_handler.elements,
                       comp=comp,
                       comp_lb=comp_lb,
                       comp_ub=comp_ub,
                       hnf=H,
                       supercell=st_sup,
                       supercell_id=idx)

        # eliminate superlattces
        obj = pyclupancpp.NonequivLBLSuperPeriodic(ds_set.all_labelings, 
                                                   site_perm_lt)
        all_labelings = obj.get_labelings()
        ds_set.replace_labelings(all_labelings)

        print(' number of structures (superperiodic eliminated) =', 
                all_labelings.shape[0])
        ##########################

        ds_set_all.append(ds_set)

    prefix = 'derivative-'+str(args.n_expand)
    yaml = Yaml()
    yaml.write_derivative_yaml(st_prim, ds_set_all, filename=prefix+'.yaml')

    if args.nodump == False:
        joblib.dump(ds_set_all, prefix+'.pkl', compress=3)

