"""Tests of cluster search."""

from pathlib import Path

from pyclupan.api.pyclupan_cluster import PyclupanCluster

cwd = Path(__file__).parent


def _calc_num_ele_combs(clusters: dict):
    """Calculate number of element combinations."""
    n_combs_all = []
    for order, clusters_list in clusters.items():
        n_combs = sum([len(cl.elements_combinations) for cl in clusters_list])
        n_combs_all.append(n_combs)
    return n_combs_all


def test_wurtzite():
    """Test cluster search in 2x2 wurtzite."""
    clupan = PyclupanCluster(verbose=False)
    clupan.load_poscar(str(cwd) + "/../files/poscar-wurtzite-primitive")
    elements = [[0, 1], [2, 3]]

    clupan.run_cluster(
        elements=elements,
        max_order=4,
        cutoffs=(6.0, 4.0, 4.0),
        filename=None,
    )
    clusters = clupan.clusters
    assert len(clusters[1]) == 2
    assert len(clusters[2]) == 26
    assert len(clusters[3]) == 26
    assert len(clusters[4]) == 39

    n_combs_all = _calc_num_ele_combs(clusters)
    assert n_combs_all == [4, 102, 172, 500]


def test_perovskite():
    """Test cluster search in perovskite."""
    clupan = PyclupanCluster(verbose=False)
    clupan.load_poscar(str(cwd) + "/../files/poscar-perovskite-unitcell")
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
    clupan = PyclupanCluster(verbose=False)
    clupan.load_poscar(str(cwd) + "/../files/poscar-fcc-primitive")
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
