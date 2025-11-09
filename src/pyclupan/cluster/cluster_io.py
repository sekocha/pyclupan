"""Functions for input/output files from cluster search."""

import io
from typing import Optional

import numpy as np
import yaml

from pyclupan.core.pypolymlp_utils import PolymlpStructure, load_cell, save_cell


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
            print("- serial_id: ", seq_id, file=f)
            print("  lattice_sites:   ", file=f)
            for site, cell in zip(cl.sites_unitcell, cl.cells_unitcell.T):
                print("  - site:    ", site, file=f)
                print("    cell:    ", list(cell), file=f)
            print(file=f)

            seq_id += 1

    print("element_clusters:", file=f)
    seq_id, seq_cl_id = 0, 0
    for order, clusters_list in clusters.items():
        for cl in clusters_list:
            for elements in cl.elements_combinations:
                print("- serial_id: ", seq_id, file=f)
                print("  cluster_id:", seq_cl_id, file=f)
                print("  lattice_sites:", file=f)
                print("    sites:   ", list(cl.sites_unitcell), file=f)
                print("    elements:", list(elements), file=f)
                print("    cells:   ", file=f)
                for c in cl.cells_unitcell.T:
                    print("    -", list(c), file=f)
                print(file=f)
                seq_id += 1
            seq_cl_id += 1
    return clusters


def save_cluster_yaml(
    clusters: dict,
    unitcell: PolymlpStructure,
    cutoffs: tuple,
    filename: str = "pyclupan_cluster.yaml",
):
    """Save clusters to yaml file."""
    np.set_printoptions(legacy="1.21")
    f = open(filename, "w")
    save_cell(unitcell, tag="unitcell", file=f)
    print("parameters:", file=f)
    print("  cutoff:", file=f)
    for i, c in enumerate(cutoffs):
        print("  - order:   ", str(i + 2), file=f)
        print("    distance:", c, file=f)
    print(file=f)

    _write_clusters(clusters, file=f)
    f.close()


def load_cluster_yaml(filename: str = "pyclupan_cluster.yaml"):
    """Load cluster.yaml."""
    yaml_data = yaml.safe_load(open(filename))
    unitcell = load_cell(yaml_data=yaml_data)
    clusters = yaml_data["clusters"]
    element_clusters = yaml_data["element_clusters"]
    return unitcell, clusters, element_clusters
