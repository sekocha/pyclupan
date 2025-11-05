"""Tests of polynomial MLP development"""

from pathlib import Path

import numpy as np
from pypolymlp.core.interface_vasp import Poscar

from pyclupan.core.normal_form import enumerate_hnf, get_nonequivalent_hnf

cwd = Path(__file__).parent


def test_entire_hnfs():
    n_hnfs = [len(enumerate_hnf(n)) for n in range(2, 11)]
    n_refs = [7, 13, 35, 31, 91, 57, 155, 130, 217]
    np.testing.assert_equal(n_hnfs, n_refs)


def test_fcc_hnfs():
    """Test number of non-equivalent HNFs for FCC."""
    st = Poscar(str(cwd) + "/poscar-fcc").structure
    n_hnfs = [len(get_nonequivalent_hnf(n, st)) for n in range(2, 11)]
    n_refs = [2, 3, 7, 5, 10, 7, 20, 14, 18]
    np.testing.assert_equal(n_hnfs, n_refs)


def test_sc_hnfs():
    """Test number of non-equivalent HNFs for SC."""
    st = Poscar(str(cwd) + "/poscar-sc").structure
    n_hnfs = [len(get_nonequivalent_hnf(n, st)) for n in range(2, 11)]
    n_refs = [3, 3, 9, 5, 13, 7, 24, 14, 23]
    np.testing.assert_equal(n_hnfs, n_refs)


def test_tetra_hnfs():
    """Test number of non-equivalent HNFs for a tetragonal structure."""
    st = Poscar(str(cwd) + "/poscar-tetra").structure
    n_hnfs = [len(get_nonequivalent_hnf(n, st)) for n in range(2, 11)]
    n_refs = [5, 5, 17, 9, 29, 13, 51, 28, 53]
    np.testing.assert_equal(n_hnfs, n_refs)
