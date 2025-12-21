"""Tests of cluster orbit search."""

from pathlib import Path

import numpy as np

from pyclupan.core.cell_utils import get_unitcell_reps, supercell_general
from pyclupan.core.spglib_utils import get_symmetry
from pyclupan.features.orbit_utils import (
    find_orbit_supercell,
    find_orbit_unitcell,
    get_map_positions,
)

cwd = Path(__file__).parent


def test_fcc(fcc_binary_clusters):
    """Test cluster orbit search in fcc."""
    lattice_unitcell, clusters, _, _ = fcc_binary_clusters
    unitcell = lattice_unitcell.cell
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


def test_perovskite_orbit_unitcell(perovskite_binary_clusters):
    """Test orbits in perovskite."""
    lattice_unitcell, clusters, _, _ = perovskite_binary_clusters
    unitcell = lattice_unitcell.cell
    rotations, translations = get_symmetry(unitcell)

    n_orbits = []
    for cl in clusters:
        orbit_sites, _ = find_orbit_unitcell(cl, unitcell, rotations, translations)
        n_orbits.append(len(orbit_sites[2]))

    n_orbits_true = [
        1,
        4,
        4,
        2,
        8,
        8,
        16,
        12,
        24,
        12,
        24,
        48,
        24,
        48,
        12,
        24,
        12,
        24,
        48,
        8,
        24,
        24,
        8,
        4,
        32,
        64,
        64,
        8,
        32,
        64,
        64,
        64,
        32,
        4,
        32,
        64,
        64,
        8,
        64,
        64,
        16,
        32,
        32,
        32,
        32,
        8,
        64,
        64,
        16,
        32,
        32,
        32,
        32,
        4,
        32,
        16,
        32,
        8,
        16,
        64,
        64,
        16,
    ]
    assert n_orbits == n_orbits_true


def test_perovskite_orbit_supercell(perovskite_binary_clusters):
    """Test orbits in perovskite."""
    lattice_unitcell, clusters, _, _ = perovskite_binary_clusters
    unitcell = lattice_unitcell.cell
    rotations, translations = get_symmetry(unitcell)

    hnf = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    supercell = supercell_general(unitcell, hnf)
    lattice_supercell = lattice_unitcell.lattice_supercell(supercell)

    map_unit_to_sup = get_unitcell_reps(unitcell, supercell)
    map_supercell_positions = get_map_positions(supercell, decimals=5)

    cl = clusters[20]
    _, orbit_fracs = find_orbit_unitcell(cl, unitcell, rotations, translations)
    orbit = find_orbit_supercell(
        unitcell,
        supercell,
        orbit_fracs,
        map_unit_to_sup,
        map_supercell_positions=map_supercell_positions,
        return_array=False,
    )
    assert list(orbit.keys()) == [4, 5, 6, 7, 8, 9]
    orbit_true = np.array(
        [
            [4, 7, 8],
            [4, 7, 9],
            [4, 7, 9],
            [4, 7, 9],
            [4, 6, 9],
            [4, 7, 8],
            [4, 6, 8],
            [4, 7, 8],
            [4, 7, 8],
            [4, 7, 8],
            [4, 7, 9],
            [4, 7, 9],
            [4, 6, 8],
            [4, 6, 8],
            [4, 6, 9],
            [4, 6, 8],
            [4, 7, 8],
            [4, 6, 9],
            [4, 6, 9],
            [4, 6, 8],
            [4, 6, 8],
            [4, 6, 9],
            [4, 7, 9],
            [4, 6, 9],
        ]
    )
    np.testing.assert_equal(np.array(orbit[4]), orbit_true)

    orbit_active_rep = lattice_supercell.to_active_site_rep(orbit)
    assert list(orbit_active_rep.keys()) == [0, 1, 2, 3, 4, 5]
    np.testing.assert_equal(np.array(orbit_active_rep[0]), orbit_true - 4)
