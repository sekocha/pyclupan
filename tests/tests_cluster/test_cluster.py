"""Tests of cluster search."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan import Pyclupan
from pyclupan.cluster.cluster_io import load_clusters_yaml

cwd = Path(__file__).parent


def _calc_num_ele_combs(clusters: dict):
    """Calculate number of element combinations."""
    n_combs_all = []
    for order, clusters_list in clusters.items():
        n_combs = sum([len(cl.elements_combinations) for cl in clusters_list])
        n_combs_all.append(n_combs)
    return n_combs_all


def test_perovskite():
    """Test cluster search in perovskite."""
    clupan = Pyclupan(verbose=False)
    clupan.load_poscar(str(cwd) + "/poscar-perovskite")
    elements = [[0, 1], [0, 1], [2]]

    clupan.run_cluster(
        elements=elements,
        max_order=4,
        cutoffs=(6.0, 6.0, 6.0),
        filename=None,
    )
    clusters = clupan.clusters
    assert len(clusters[1]) == 2
    assert len(clusters[2]) == 5
    assert len(clusters[3]) == 8
    assert len(clusters[4]) == 13

    n_combs_all = _calc_num_ele_combs(clusters)
    assert n_combs_all == [4, 16, 44, 105]

    elements = [[0, 1, 2], [2, 3], [4, 5]]
    clupan.run_cluster(
        elements=elements,
        max_order=4,
        cutoffs=(6.0, 5.0, 4.0),
        filename=None,
    )
    clusters = clupan.clusters
    assert len(clusters[1]) == 3
    assert len(clusters[2]) == 17
    assert len(clusters[3]) == 39
    assert len(clusters[4]) == 13

    n_combs_all = _calc_num_ele_combs(clusters)
    assert n_combs_all == [7, 70, 335, 185]


def test_fcc():
    """Test cluster search in fcc."""
    clupan = Pyclupan(verbose=False)
    clupan.load_poscar(str(cwd) + "/poscar-fcc")
    elements = [[0, 1, 2]]

    clupan.run_cluster(
        elements=elements,
        max_order=4,
        cutoffs=(6.0, 5.0, 4.0),
        filename=None,
    )
    clusters = clupan.clusters
    assert len(clusters[1]) == 1
    assert len(clusters[2]) == 4
    assert len(clusters[3]) == 7
    assert len(clusters[4]) == 3

    n_combs_all = _calc_num_ele_combs(clusters)
    assert n_combs_all == [3, 24, 119, 72]


def test_load_clusters():
    """Test load clusters.yaml"""
    filename = str(cwd) + "/pyclupan_clusters.yaml"
    unitcell, clusters, el_clusters, _ = load_clusters_yaml(filename)
    assert len(clusters) == 52
    assert len(el_clusters) == 467

    cl = el_clusters[-2]
    assert cl.cluster_id == 51
    assert cl.colored_cluster_id == 465
    assert cl.sites_unitcell == (0, 0, 0, 0)
    assert cl.elements == (0, 1, 1, 1)
    cell_true = np.array([[0, -2, -1, -1], [0, 1, 1, 2], [0, 0, 1, -1]])
    np.testing.assert_equal(cl.cells_unitcell, cell_true)
