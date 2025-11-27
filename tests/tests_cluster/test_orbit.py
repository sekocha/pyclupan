"""Tests of cluster orbit search."""

from pathlib import Path

from pyclupan.cluster.cluster_io import load_clusters_yaml
from pyclupan.core.spglib_utils import get_symmetry
from pyclupan.features.orbit_utils import find_orbit_unitcell

cwd = Path(__file__).parent


def test_fcc():
    """Test cluster orbit search in fcc."""
    filename = str(cwd) + "/pyclupan_clusters.yaml"
    unitcell, clusters, _, _ = load_clusters_yaml(filename)
    rotations, translations = get_symmetry(unitcell)

    n_orbits = []
    for cl in clusters:
        orbit_sites, _ = find_orbit_unitcell(cl, unitcell, rotations, translations)
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

    orbit_sites, orbit_positions = find_orbit_unitcell(
        clusters[31], unitcell, rotations, translations
    )
    for orbit in orbit_sites[0]:
        assert list(orbit) == [0, 0, 0, 0]

    positions = orbit_positions[0]
    assert list(positions[0][0]) == [-1, 0, 0, 0]
    assert list(positions[0][1]) == [1, 0, 1, 1]
    assert list(positions[0][2]) == [0, 0, -1, 0]

    assert list(positions[1][0]) == [0, 1, 1, 1]
    assert list(positions[1][1]) == [0, -1, 0, 0]
    assert list(positions[1][2]) == [0, 0, -1, 0]

    assert list(positions[2][0]) == [-1, 0, 0, 0]
    assert list(positions[2][1]) == [0, -1, 0, 0]
    assert list(positions[2][2]) == [0, 0, -1, 0]

    assert list(positions[3][0]) == [0, 0, 0, 1]
    assert list(positions[3][1]) == [-1, -1, 0, -1]
    assert list(positions[3][2]) == [0, 1, 0, 0]

    assert list(positions[4][0]) == [-1, -1, -1, 0]
    assert list(positions[4][1]) == [0, 0, 1, 0]
    assert list(positions[4][2]) == [0, 1, 0, 0]

    assert list(positions[5][0]) == [-1, 0, 0, 0]
    assert list(positions[5][1]) == [0, -1, 0, 0]
    assert list(positions[5][2]) == [1, 1, 0, 1]

    assert list(positions[6][0]) == [0, 0, 0, 1]
    assert list(positions[6][1]) == [0, 0, 1, 0]
    assert list(positions[6][2]) == [-1, 0, -1, -1]

    assert list(positions[7][0]) == [0, 0, 0, 1]
    assert list(positions[7][1]) == [0, 0, 1, 0]
    assert list(positions[7][2]) == [0, 1, 0, 0]


# TODO: orbit in supercell
