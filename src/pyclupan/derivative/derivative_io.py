"""Functions for saving and loading yaml files for derivative structures."""

# import numpy as np
# import yaml

from pyclupan.derivative.derivative_utils import Derivatives
from pyclupan.utils.yaml_utils import save_cell


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
        save_cell(derivs_all[0].unitcell, tag="unitcell", file=f)
        print("element_sets:", file=f)
        for i, ele in enumerate(derivs_all[0].element_orbit):
            print("- id:         ", i, file=f)
            print("  elements:   ", ele, file=f)
        print("", file=f)

        print("derivative_structures:", file=f)
        for i, derivs in enumerate(derivs_all):
            print("- group:", i, file=f)
            print("  n_cell:", derivs.supercell_size, file=f)
            print("  HNF:", file=f)
            print("  -", derivs.hnf[0], file=f)
            print("  -", derivs.hnf[1], file=f)
            print("  -", derivs.hnf[2], file=f)
            if derivs.comp is not None:
                print("  comp:   ", list(derivs.comp), file=f)
            if derivs.comp_lb is not None:
                print("  comp_lb:", list(derivs.comp_lb), file=f)
            if derivs.comp_ub is not None:
                print("  comp_ub:", list(derivs.comp_ub), file=f)
            # save_cell(derivs.supercell, tag="supercell", file=f)

            print("  inactive_sites:   ", end="", file=f)
            _write_list_no_space(derivs.inactive_sites, file=f)
            print("  inactive_labeling:", end="", file=f)
            _write_list_no_space(derivs.inactive_labeling, file=f)
            print("  active_sites:     ", end="", file=f)
            _write_list_no_space(derivs.active_sites, file=f)
            print("  n_labelings:      ", derivs.active_labelings.shape[0], file=f)
            print("", file=f)
            print("  active_labelings:", file=f)
            for l in derivs.active_labelings:
                print("  - ", end="", file=f)
                _write_list_no_space(l, file=f)
            print("", file=f)


# def load_derivative_yaml(filename: str = "derivative.yaml"):
#     """Load labelings of derivative structures."""
#     data = yaml.safe_load(open(filename))
#     unitcell = load_cell(data, tag="unitcell")
#     element_orbit = [d["elements"] for d in data["element_sets"]]
#     elements = sorted([e for ele in element_orbit for e in ele])
#
#     derivs_all = []
#     for d in data["derivative_structures"]:
#         n_cell = int(d["n_cell"])
#         hnf = np.array(d["HNF"])
#         supercell_id = int(d["group"])
#         supercell = load_cell(d, tag="supercell")
#
#         active_sites = d["active_sites"]
#         inactive_sites = d["inactive_sites"]
#         inactive_labeling = d["inactive_labeling"]
#         active_labelings = np.array(d["active_labelings"])
#
#         derivs = Derivative(
#             active_labelings=active_labelings,
#             inactive_labeling=inactive_labeling,
#             active_sites=active_sites,
#             inactive_sites=inactive_sites,
#             primitive_cell=prim,
#             n_expand=n_cell,
#             hnf=hnf,
#             elements=elements,
#             element_orbit=element_orbit,
#             supercell_set=supercell_set,
#             supercell_idset=supercell_idset,
#         )
#         derivs_all.append(derivs)
#
#     return derivs_all
