"""Functions for saving and loading yaml files for derivative structures."""

import numpy as np
import yaml

from pyclupan.derivative.derivative_utils import Derivatives
from pyclupan.utils.yaml_utils import load_cell, save_cell
from pyclupan.zdd.pyclupan_zdd import PyclupanZdd


def _write_list_no_space(a: list, file):
    """Write list without spaces.."""
    print("[", end="", file=file)
    print(*list(a), sep=",", end="]\n", file=file)


def write_derivative_yaml(
    derivs_all: list[Derivatives],
    filename: str = "derivatives.yaml",
):
    """Save labelings of derivative structures to yaml file."""
    with open(filename, "w") as f:
        derivs = derivs_all[0]
        save_cell(derivs.unitcell, tag="unitcell", file=f)
        print("zdd:", file=f)
        print("  n_cells: ", derivs.supercell_size, file=f)
        print("  one_of_k:", derivs.zdd_lattice.one_of_k_rep, file=f)
        print("  element_sets:", file=f)
        for i, ele in enumerate(derivs.element_orbit):
            print("  - id:         ", i, file=f)
            print("    elements:   ", ele, file=f)
        print("", file=f)

        print("compositions:", file=f)
        print("  comp:   ", list(derivs.comp), file=f)
        print("  comp_lb:", list(derivs.comp_lb), file=f)
        print("  comp_ub:", list(derivs.comp_ub), file=f)
        print("", file=f)

        print("derivative_structures:", file=f)
        for i, derivs in enumerate(derivs_all):
            print("- id:", i, file=f)
            print("  HNF:", file=f)
            print("  -", list(derivs.hnf[0]), file=f)
            print("  -", list(derivs.hnf[1]), file=f)
            print("  -", list(derivs.hnf[2]), file=f)
            # save_cell(derivs.supercell, tag="supercell", file=f)

            print("  inactive_sites:   ", end=" ", file=f)
            _write_list_no_space(derivs.inactive_sites, file=f)
            print("  inactive_labeling:", end=" ", file=f)
            _write_list_no_space(derivs.inactive_labeling, file=f)
            print("  active_sites:     ", end=" ", file=f)
            _write_list_no_space(derivs.active_sites, file=f)
            print("  n_labelings:      ", derivs.active_labelings.shape[0], file=f)
            print("", file=f)
            print("  active_labelings:", file=f)
            for l in derivs.active_labelings:
                print("  - ", end="", file=f)
                _write_list_no_space(l, file=f)
            print("", file=f)


def load_derivative_yaml(filename: str = "derivative.yaml", verbose: bool = False):
    """Load labelings of derivative structures."""
    data = yaml.safe_load(open(filename))
    unitcell = load_cell(yaml_data=data, tag="unitcell")
    n_cells = data["zdd"]["n_cells"]
    one_of_k_rep = data["zdd"]["one_of_k"]
    element_orbit = [d["elements"] for d in data["zdd"]["element_sets"]]
    elements_lattice = [e[0] for e in element_orbit]

    zdd = PyclupanZdd(verbose=verbose)
    zdd.unitcell = unitcell
    zdd.initialize_zdd(
        supercell_size=n_cells,
        elements_lattice=elements_lattice,
        one_of_k_rep=one_of_k_rep,
    )

    derivs_all = []
    for d in data["derivative_structures"]:
        derivs = Derivatives(
            zdd_lattice=zdd.zdd_lattice,
            unitcell=unitcell,
            hnf=np.array(d["HNF"]),
            active_labelings=np.array(d["active_labelings"]),
            inactive_labeling=d["inactive_labeling"],
            supercell_id=int(d["id"]),
        )
        derivs_all.append(derivs)

    return derivs_all
