"""Functions for input/output files from cluster search."""

import io
from typing import Optional

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, save_cell


def _write_clusters(
    clusters: dict,
    #    element_tag="element",
    file: Optional[str] = None,
):
    """Save cluster attributes."""
    if isinstance(file, str):
        f = open(file, "w")
    elif isinstance(file, io.IOBase):
        f = file
    else:
        raise RuntimeError("file is not str or io.IOBase")

    seq_id = 0
    for order, clusters in clusters.items():
        for cl in clusters:
            print("- serial_id: ", seq_id, file=f)
            # print("  id:        ", cl.idx, file=f)
            print("  lattice_sites:   ", file=f)
            for site, cell in zip(cl.sites_unitcell, cl.cells_unitcell.T):
                print("  - site:    ", site, file=f)
                print("    cell:    ", list(cell), file=f)
            print(file=f)

            # if cl.ele_indices is None:
            # else:
            #     for site, cell, ele in zip(
            #         cl.site_indices, cl.cell_indices, cl.ele_indices
            #     ):
            #         print("  - site:    ", site, file=f)
            #         print("    cell:    ", list(cell), file=f)
            #         print("    " + element_tag + ": ", ele, file=f)
            #         print("", file=f)
            seq_id += 1


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
    print("  cutoff_distances:", file=f)
    for i, c in enumerate(cutoffs):
        print("  - order: ", str(i + 2), file=f)
        print("    cutoff:", c, file=f)
    print(file=f)

    print("clusters:", file=f)
    _write_clusters(clusters, file=f)

    #        if elements_lattice is not None:
    #            print("element_configs:", file=f)
    #            for i, elements in enumerate(elements_lattice):
    #                print("- lattice:    ", i, file=f)
    #                if len(elements) > 0:
    #                    print("  elements:   ", elements, file=f)
    #                else:
    #                    print("  elements:   ", [], file=f)
    #                print("", file=f)

    #        if cluster_set_element is not None:
    #            print("nonequiv_element_configs:", file=f)
    #            self._write_clusters(cluster_set_element, f)
    f.close()
