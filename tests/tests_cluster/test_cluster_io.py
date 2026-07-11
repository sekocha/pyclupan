"""Tests of IO for cluster search."""

from pathlib import Path

import numpy as np

from pyclupan.cluster.cluster_io import load_clusters_yaml

cwd = Path(__file__).parent


def test_load_clusters():
    """Test load clusters.yaml"""
    filename = str(cwd) + "/../files/binary_fcc/pyclupan_clusters.yaml"
    unitcell, clusters, el_clusters, _ = load_clusters_yaml(filename)
    assert len(clusters) == 52
    assert len(el_clusters) == 467

    cl = clusters[-1]
    assert cl.sites_unitcell == (0, 0, 0, 0)
    cell_true = np.array([[0, -2, -1, -1], [0, 1, 1, 2], [0, 0, 1, -1]])
    np.testing.assert_equal(cl.cells_unitcell, cell_true)

    cl = el_clusters[-2]
    assert cl.cluster_id == 51
    assert cl.colored_cluster_id == 465
    assert cl.elements == (0, 1, 1, 1)
