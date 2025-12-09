"""Interface for vasp."""

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, Vasprun


def load_vasp_results(vaspruns: list):
    """Load results from vasp calculations."""
    structure_ids, energies, structures = [], [], []
    for vasp_file in vaspruns:
        path = "/".join(vasp_file.split("/")[:-1])
        try:
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
        except:
            pass

    return structure_ids, np.array(energies), structures


def save_energy_dat(
    vaspruns: list,
    unitcell: PolymlpStructure,
    filename: str = "energy.dat",
):
    """Save energy.dat used for regression."""
    n_atom_unitcell = np.sum(unitcell.n_atoms)
    ids, energies, structures = load_vasp_results(vaspruns)

    n_cell = np.array([np.sum(st.n_atoms) / n_atom_unitcell for st in structures])
    energies = energies / n_cell

    with open(filename, "w") as f:
        for i, e in zip(ids, energies):
            print(i, e, file=f)
