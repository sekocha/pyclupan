"""Tests of cluster search."""

from pathlib import Path

from pyclupan.api.pyclupan import Pyclupan

cwd = Path(__file__).parent


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

    n_combs_all = []
    for order, clusters_list in clusters.items():
        n_combs = sum([len(cl.elements_combinations) for cl in clusters_list])
        n_combs_all.append(n_combs)
    assert n_combs_all == [3, 24, 119, 72]
