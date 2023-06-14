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
        for ele, c in comp_in:
            ele, c = int(ele), Fraction(c)
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
    #


    signal.signal(signal.SIGINT, signal.SIG_DFL)
    warnings.simplefilter('ignore')

    ps = argparse.ArgumentParser()
    ps.add_argument('-p',
                    '--poscar',
                    type=str,
                    default='POSCAR',
                    help='POSCAR file for primitive cell')
    ps.add_argument('-e',
                    '--elements',
                    nargs='*',
                    type=int,
                    action='append',
                    default=None,
                    help='Elements on a lattice')
    ps.add_argument('-o',
                    '--occupation',
                    nargs='*',
                    type=int,
                    action='append',
                    default=None,
                    help='Lattice IDs that are occupied by an element')
    ps.add_argument('-c',
                    '--comp',
                    nargs='*',
                    type=str,
                    action='append',
                    default=None,
                    help='Composition (n_elements / n_sites)')
    ps.add_argument('--comp_lb',
                    nargs='*',
                    type=str,
                    action='append',
                    default=None,
                    help='Lower bound of composition (n_elements / n_sites)')
    ps.add_argument('--comp_ub',
                    nargs='*',
                    type=str,
                    action='append',
                    default=None,
                    help='Upper bound of composition (n_elements / n_sites)')
    ps.add_argument('--charge',
                    type=float, 
                    nargs='*', 
                    default=None,
                    help='Charges for elements')
 
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
    ps.add_argument('--superperiodic',
                    action='store_true',
                    help='Including superperiodic structures')
    ps.add_argument('--endmember',
                    action='store_true',
                    help='Including endmembers (incomplete structures)')
    ps.add_argument('--nodump',
                    action='store_true',
                    help='No dump file of DSSet object')
    ps.add_argument('--no_labelings',
                    action='store_true',
                    help='Only counting number of structures')
    args = ps.parse_args()

    if args.occupation is None and args.elements is None:
        raise KeyError(' occupation or elements is required.')

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

    if args.n_expand is None and args.hnf is None:
        raise KeyError(' n_expand or hnf is required.')

    print(' charge =', args.charge)
    if args.charge is not None:
        one_of_k_rep=True
    else:
        one_of_k_rep=False

    if args.hnf is not None:
        hnf_all = [args.hnf]
    else:
        hnf_all = get_nonequivalent_hnf(args.n_expand, st_prim)

    print(' number of HNFs =', len(hnf_all))

    n_sites = list(np.array(st_prim.n_atoms) * args.n_expand)
    dd_handler = DDNodeHandler(n_sites=n_sites,
                               occupation=args.occupation,
                               elements_lattice=args.elements,
                               one_of_k_rep=one_of_k_rep)

    n_total = 0
    ds_set_all = []
    for idx, H in enumerate(hnf_all):
        sup = Supercell(st_prim=st_prim, hnf=H)
        st_sup = sup.get_supercell()

        site_perm, site_perm_lt = get_permutation(st_sup, 
                                                  superperiodic=True, 
                                                  hnf=H)


        dd_const = DDConstructor(dd_handler)
        gs = dd_const.enumerate_nonequiv_configs(site_permutations=site_perm,
                                                 comp=comp,
                                                 comp_lb=comp_lb,
                                                 comp_ub=comp_ub)
        if args.endmember == False:
            gs &= dd_const.no_endmembers()
            print(' number of structures (end members eliminated) =', gs.len())

        if args.charge is not None:
            gs = dd_const.charge_balance(args.charge, gs=gs)
            print(' number of structures (charge_balance) =', gs.len())

        if args.no_labelings == False:
            t1 = time.time()
            res = dd_handler.convert_graphs_to_labelings(gs)
            t2 = time.time()
            print(' elapsed time (labeling)    =', t2-t1)

            ds_set = DSSet(labelings_info=res,
                           primitive_cell=st_prim,
                           n_expand=args.n_expand,
                           elements=dd_handler.elements,
                           element_orbit=dd_handler.get_element_orbit(),
                           comp=comp,
                           comp_lb=comp_lb,
                           comp_ub=comp_ub,
                           hnf=H,
                           supercell=st_sup,
                           supercell_id=idx)

            if args.superperiodic == False:
                t2 = time.time()
                obj = pyclupancpp.NonequivLBLSuperPeriodic(ds_set.all_labelings, 
                                                           site_perm_lt)
                all_labelings = obj.get_labelings()
                ds_set.replace_labelings(all_labelings)
                t3 = time.time()

                print(' number of structures (superperiodic eliminated) =', 
                        all_labelings.shape[0])
                print(' elapsed time (superperiodic)    =', t3-t2)

            ds_set_all.append(ds_set)
            n_total += ds_set.all_labelings.shape[0]
        else:
            n_total += gs.len()

    if args.no_labelings == False:
        prefix = 'derivative-'+str(args.n_expand)
        yaml = Yaml()
        yaml.write_derivative_yaml(st_prim, ds_set_all, filename=prefix+'.yaml')

        if args.nodump == False:
            joblib.dump(ds_set_all, prefix+'.pkl', compress=3)

    print(' -- summary --')
    print(' number of HNFs =', len(hnf_all))
    print(' number of total structures =', n_total)


