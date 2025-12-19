"""Pytest conftest.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyclupan.cluster.cluster_io import load_clusters_yaml
from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def pytest_addoption(parser):
    """Add command option to pytest."""
    parser.addoption(
        "--runbig", action="store_true", default=False, help="run big tests"
    )


def pytest_configure(config):
    """Set up marker big."""
    config.addinivalue_line("markers", "big: mark test as big to run")


def pytest_collection_modifyitems(config, items):
    """Add mechanism to run with --runbig."""
    if config.getoption("--runbig"):
        # --runbig given in cli: do not skip slow tests
        return
    skip_big = pytest.mark.skip(reason="need --runbig option to run")
    for item in items:
        if "big" in item.keywords:
            item.add_marker(skip_big)


@pytest.fixture(scope="session")
def fcc_primitive_cell() -> PolymlpStructure:
    """Return unitcell of FCC."""
    return Poscar(path_file + "/poscar-fcc-primitive").structure


@pytest.fixture(scope="session")
def perovskite_unitcell() -> PolymlpStructure:
    """Return unitcell of perovskite."""
    return Poscar(path_file + "/poscar-perovskite-unitcell").structure


@pytest.fixture(scope="session")
def wurtzite_primitive_cell() -> PolymlpStructure:
    """Return primitive cell of wurtzite."""
    return Poscar(path_file + "/poscar-wurtzite-primitive").structure


@pytest.fixture(scope="session")
def sc_primitive_cell() -> PolymlpStructure:
    """Return unitcell of simple cubic."""
    return Poscar(path_file + "/poscar-sc-primitive").structure


@pytest.fixture(scope="session")
def tetra_primitive_cell() -> PolymlpStructure:
    """Return unitcell of simple tetragonal."""
    return Poscar(path_file + "/poscar-tetra-primitive").structure


@pytest.fixture(scope="session")
def fcc_binary_clusters():
    """Return cluster attributes of fcc binary system."""
    filename = path_file + "/binary_fcc/pyclupan_clusters.yaml"
    unitcell, clusters, ele_clusters, spin_clusters = load_clusters_yaml(filename)
    return (unitcell, clusters, ele_clusters, spin_clusters)
