"""Interface for vasp."""

import numpy as np

from pyclupan.core.pypolymlp_utils import Vasprun


def load_vasp_results(vaspruns: list):
    """Load results from vasp calculations."""
    structure_ids, energies, structures = [], [], []
    for vasp_file in vaspruns:
        path = "/".join(vasp_file.split("/")[:-1])
        vasp = Vasprun(vasp_file)
        try:
            f = open(path + "/POSCAR")
            first_line = f.readline()
            f.close()
            if "pyclupan" in first_line:
                structure_id = first_line.split()[-1]
            else:
                structure_id = path
        except:
            structure_id = path

        structure_ids.append(structure_id)
        energies.append(vasp.energy)
        structures.append(vasp.structure)

    return structure_ids, np.array(energies), structures
