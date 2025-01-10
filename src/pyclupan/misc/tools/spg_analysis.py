#!/usr/bin/env python
import numpy as np
import argparse
import joblib
import glob, os
import time

from mlptools.common.structure import Structure
from mlptools.tools.spg import SymCell

from pyclupan.derivative.derivative import DSSet, DSSample

def get_structure(supercell, 
                  n_atoms, 
                  order, 
                  remove_indices=[]):

    # remove_indices must be sorted
    if len(remove_indices) > 0:
        positions = supercell.positions[:,order]
        for idx in remove_indices:
            begin = int(np.sum(n_atoms[:idx]))
            end = begin + n_atoms[idx]
            positions = np.delete(positions,
                                  range(begin,end), 
                                  axis=1)
            n_atoms = np.delete(n_atoms, idx)
    else:
        positions = supercell.positions[:,order]

    return Structure(supercell.axis, positions, n_atoms)

if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('-e',
                    '--elements',
                    nargs='*',
                    type=str,
                    default=None,
                    help='elements')
    args = ps.parse_args()

    files = glob.glob('derivative*.pkl')
    if 'derivative-all.pkl' in files:
        files.remove('derivative-all.pkl')
    print(' files =', files)

    ds_set_all = []
    for f in files:
        ds_set_all.extend(joblib.load(f))

    ds_samp = DSSample(ds_set_all)
    st_attr, indices = ds_samp.sample_all()
    n_samples = len(st_attr)
    print(' total number of sampled structures =', n_samples)

    if args.elements is None:
        strings = 'ABCDEFGHIJK'
        elements = [strings[e] for e in ds_set_all[0].elements]
    else:
        elements = args.elements

    remove_indices = [i for i, e in enumerate(elements) 
                        if e == 'vac' or e == 'Vac' or e == 'VAC']
    remove_indices = sorted(remove_indices, reverse=True)
    for idx in remove_indices:
        del elements[idx]

    res = []
    for i in range(len(st_attr)):
        order, n_atoms = st_attr[i]
        n_cell, g_id, s_id, l_id = indices[i]

        cell = ds_set_all[g_id].get_supercell_from_id(s_id)
        st = get_structure(cell, 
                           n_atoms, 
                           order, 
                           remove_indices=remove_indices)

        sc = SymCell(structure=st, symprec=1e-3)
        spg = sc.get_spacegroup()
        fname = str(n_cell) + '-' + str(s_id) + '-' + str(l_id)
        res.append([fname, spg])

    np.savetxt('space_group.dat', res, fmt='%s')

   
