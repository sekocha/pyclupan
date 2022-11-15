#!/usr/bin/env python
import numpy as np
import argparse
import glob

from mlptools.common.readvasp import Vasprun

def get_info(vasprun_xml):

    vasp = Vasprun(vasprun_xml)
    e = vasp.get_energy()
    _, _, n_atoms, _, _, _ = vasp.get_structure()
    return e, np.array(n_atoms)


if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('--vaspruns',
                    nargs='*',
                    type=str,
                    required=True,
                    help='vasprun files')

    # --end 01/vasprun.xml 2.0 0 1 --end 02/vasprun.xml 2.0 2 3 
    # (vasprun file, unit, elements)
    ps.add_argument('--end',
                    nargs='*',
                    type=str,
                    action='append',
                    default=None,
                    help='vasprun files (end members)')
 
    args = ps.parse_args()

    dft_data = []
    if args.end is None:
        g = open('summary_dft.dat', 'w')
        for f in args.vaspruns:
            e, n_atoms = get_info(f)
            print(f, e/sum(n_atoms), file=g)
        g.close()
    else:
        elements = [int(e) for attr in args.end for e in attr[2:]]
        n_type = len(elements)

        n_ends = len(args.end)
        e_ends = np.zeros(n_ends)
        n_atoms_end = np.zeros((n_type,n_ends), dtype=int)
        for i, attr in enumerate(args.end):
            f, unit, ele = attr[0], float(attr[1]), [int(j) for j in attr[2:]]
            e, n_atoms = get_info(f)
            n_atoms_end[np.array(ele),i] = n_atoms / unit
            e_ends[i] = e / unit

            comp = np.zeros(len(args.end))
            comp[i] = 1.0
            data = (f, comp, 0.0)
            dft_data.append(data)

        n_atoms_end_rec = np.linalg.pinv(n_atoms_end)
        for f in args.vaspruns:
            e, n_atoms = get_info(f)
            if len(n_atoms) == n_atoms_end_rec.shape[1]:
                ratio = np.dot(n_atoms_end_rec, n_atoms)
                comp = ratio / sum(ratio)
                e_form = (e - np.dot(e_ends, ratio)) / sum(ratio)
                data = (f, comp, e_form)
                dft_data.append(data)

        g = open('summary_dft.dat', 'w')
        print('# filename, compositions, e_formation', file=g)
        for d in dft_data:
            print(d[0], end=' ', file=g)
            for c in d[1]:
                print('{:.15f}'.format(c), end=' ', file=g)
            print('{:.15f}'.format(d[2]), file=g)
        g.close()

            
        print(dft_data)
   
