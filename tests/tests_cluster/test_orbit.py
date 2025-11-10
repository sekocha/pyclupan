"""Tests of cluster orbit search."""

from pathlib import Path

from pyclupan.cluster.cluster_io import load_cluster_yaml
from pyclupan.core.spglib_utils import get_symmetry
from pyclupan.features.orbit_utils import find_orbit_unitcell

cwd = Path(__file__).parent


def test_fcc():
    """Test cluster orbit search in fcc."""
    filename = str(cwd) + "/pyclupan_cluster.yaml"
    unitcell, clusters, _ = load_cluster_yaml(filename)
    rotations, translations = get_symmetry(unitcell)

    n_orbits = []
    for cl in clusters:
        orbit_sites = find_orbit_unitcell(cl, unitcell, rotations, translations)
        n_orbits.append(len(orbit_sites[0]))
    assert n_orbits == [
        1,
        12,
        12,
        24,
        6,
        18,
        72,
        24,
        144,
        36,
        72,
        72,
        24,
        36,
        36,
        24,
        72,
        192,
        96,
        48,
        48,
        48,
        96,
        96,
        32,
        96,
        96,
        192,
        192,
        192,
        192,
        8,
        96,
        32,
        48,
        48,
        96,
        192,
        96,
        192,
        96,
        96,
        12,
        96,
        24,
        192,
        48,
        8,
        32,
        96,
        12,
        24,
    ]

    orbit_sites = find_orbit_unitcell(clusters[31], unitcell, rotations, translations)
    for orbit in orbit_sites[0]:
        assert list(orbit.sites) == [0, 0, 0, 0]

    assert list(orbit_sites[0][0].cells[0]) == [-1, 0, 0, 0]
    assert list(orbit_sites[0][0].cells[1]) == [1, 0, 1, 1]
    assert list(orbit_sites[0][0].cells[2]) == [0, 0, -1, 0]

    assert list(orbit_sites[0][1].cells[0]) == [0, 1, 1, 1]
    assert list(orbit_sites[0][1].cells[1]) == [0, -1, 0, 0]
    assert list(orbit_sites[0][1].cells[2]) == [0, 0, -1, 0]

    assert list(orbit_sites[0][2].cells[0]) == [-1, 0, 0, 0]
    assert list(orbit_sites[0][2].cells[1]) == [0, -1, 0, 0]
    assert list(orbit_sites[0][2].cells[2]) == [0, 0, -1, 0]

    assert list(orbit_sites[0][3].cells[0]) == [0, 0, 0, 1]
    assert list(orbit_sites[0][3].cells[1]) == [-1, -1, 0, -1]
    assert list(orbit_sites[0][3].cells[2]) == [0, 1, 0, 0]

    assert list(orbit_sites[0][4].cells[0]) == [-1, -1, -1, 0]
    assert list(orbit_sites[0][4].cells[1]) == [0, 0, 1, 0]
    assert list(orbit_sites[0][4].cells[2]) == [0, 1, 0, 0]

    assert list(orbit_sites[0][5].cells[0]) == [-1, 0, 0, 0]
    assert list(orbit_sites[0][5].cells[1]) == [0, -1, 0, 0]
    assert list(orbit_sites[0][5].cells[2]) == [1, 1, 0, 1]

    assert list(orbit_sites[0][6].cells[0]) == [0, 0, 0, 1]
    assert list(orbit_sites[0][6].cells[1]) == [0, 0, 1, 0]
    assert list(orbit_sites[0][6].cells[2]) == [-1, 0, -1, -1]

    assert list(orbit_sites[0][7].cells[0]) == [0, 0, 0, 1]
    assert list(orbit_sites[0][7].cells[1]) == [0, 0, 1, 0]
    assert list(orbit_sites[0][7].cells[2]) == [0, 1, 0, 0]
