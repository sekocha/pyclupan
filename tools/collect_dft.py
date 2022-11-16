#!/usr/bin/env python
import numpy as np
import os
import argparse

from mlptools.common.readvasp import Vasprun
from pyclupan.common.composition import Composition
from pyclupan.common.io.yaml import Yaml

#    ~/git/pyclupan/tools/collect_dft.py --end finished/1-0-0/vasprun.xml_to_mlip 2.0 0 2 --end finished/1-0-3/vasprun.xml_to_mlip 2.0 1 3 --vaspruns finished/*/vasprun.xml_to_mlip


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

        abspath = [os.path.abspath(end[0]) for end in args.end]
        comp_obj = Composition(n_atoms_end, e_end=e_ends, path_end=abspath)

        for f in args.vaspruns:
            e, n_atoms = get_info(f)
            if len(n_atoms) == n_type:
                comp, partition = comp_obj.get_comp(n_atoms)
                e_form = comp_obj.compute_formation_energy(e, partition)
                data = (f, comp, e_form)
                dft_data.append(data)

        Yaml().write_dft_yaml(dft_data, comp_obj=comp_obj)

  
