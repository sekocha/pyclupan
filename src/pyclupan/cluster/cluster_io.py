"""Functions for input/output files from cluster search."""

import io
from typing import Optional

import numpy as np
import yaml

from pyclupan.cluster.cluster_utils import ClusterAttr
from pyclupan.core.io_utils import write_list_no_space
from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import load_cell


def _write_clusters(clusters: dict, file: Optional[str] = None):
    """Save cluster attributes."""
    if isinstance(file, str):
        f = open(file, "w")
    elif isinstance(file, io.IOBase):
        f = file
    else:
        raise RuntimeError("file is not str or io.IOBase")

    seq_id = 0
    print("clusters:", file=f)
    for order, clusters_list in clusters.items():
        for cl in clusters_list:
            print("- id:   ", seq_id, file=f)
            print("  sites: ", end="", file=f)
            write_list_no_space(list(cl.sites_unitcell), file=f)
            print("  cells:", file=f)
            for cell in cl.cells_unitcell.T:
                print("  - ", end="", file=f)
                write_list_no_space(list(cell), file=f)
            print(file=f)
            seq_id += 1

    print("element_clusters:", file=f)
    seq_id, seq_cl_id = 0, 0
    for order, clusters_list in clusters.items():
        for cl in clusters_list:
            for elements in cl.elements_combinations:
                print("- serial_id: ", seq_id, file=f)
                print("  cluster_id:", seq_cl_id, file=f)
                print("  elements:   ", end="", file=f)
                write_list_no_space(list(elements), file=f)
                print(file=f)
                seq_id += 1
            seq_cl_id += 1

    print("spin_basis_clusters:", file=f)
    seq_id, seq_cl_id = 0, 0
    for order, clusters_list in clusters.items():
        for cl in clusters_list:
            for basis in cl.spin_basis_combinations:
                print("- serial_id: ", seq_id, file=f)
                print("  cluster_id:", seq_cl_id, file=f)
                print("  basis:      ", end="", file=f)
                write_list_no_space(list(basis), file=f)
                print(file=f)
                seq_id += 1
            seq_cl_id += 1

    return clusters


def save_cluster_yaml(
    clusters: dict,
    lattice: Lattice,
    cutoffs: tuple,
    filename: str = "pyclupan_cluster.yaml",
):
    """Save clusters to yaml file."""
    np.set_printoptions(legacy="1.21")
    f = open(filename, "w")
    lattice.save(file=f)
    print("parameters:", file=f)
    print("  cutoff:", file=f)
    for i, c in enumerate(cutoffs):
        print("  - order:   ", str(i + 2), file=f)
        print("    distance:", c, file=f)
    print(file=f)

    _write_clusters(clusters, file=f)
    f.close()


def load_clusters_yaml(filename: str = "pyclupan_clusters.yaml"):
    """Load pyclupan_clusters.yaml."""
    yaml_data = yaml.safe_load(open(filename))
    unitcell = load_cell(yaml_data=yaml_data, tag="lattice_cell")
    elements_lattice = yaml_data["elements_on_lattice"]

    lattice = Lattice(cell=unitcell, elements=elements_lattice)

    cluster_attrs = []
    for cl in yaml_data["clusters"]:
        sites = tuple(cl["sites"])
        cells = np.array(cl["cells"]).T
        attr = ClusterAttr(
            sites_unitcell=sites,
            cells_unitcell=cells,
            cluster_id=cl["id"],
        )
        cluster_attrs.append(attr)

    element_cluster_attrs = []
    for cl in yaml_data["element_clusters"]:
        attr = ClusterAttr(
            elements=tuple(cl["elements"]),
            cluster_id=cl["cluster_id"],
            colored_cluster_id=cl["serial_id"],
        )
        element_cluster_attrs.append(attr)

    spin_cluster_attrs = []
    for cl in yaml_data["spin_basis_clusters"]:
        attr = ClusterAttr(
            spin_basis=tuple(cl["basis"]),
            cluster_id=cl["cluster_id"],
            colored_cluster_id=cl["serial_id"],
        )
        spin_cluster_attrs.append(attr)

    return (lattice, cluster_attrs, element_cluster_attrs, spin_cluster_attrs)
