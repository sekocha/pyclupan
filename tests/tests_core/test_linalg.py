"""Tests of polynomial MLP development"""

from pathlib import Path

import numpy as np

from pyclupan.core.linalg_utils import enumerate_hnf, get_nonequivalent_hnf

cwd = Path(__file__).parent


def test_entire_hnfs():
    """Test number of HNFs."""
    n_hnfs = [len(enumerate_hnf(n)) for n in range(2, 17)]
    n_refs = [7, 13, 35, 31, 91, 57, 155, 130, 217, 133, 455, 183, 399, 403, 651]
    np.testing.assert_equal(n_hnfs, n_refs)


def test_fcc_hnfs(fcc_primitive_cell):
    """Test number of non-equivalent HNFs for FCC."""
    n_hnfs = [len(get_nonequivalent_hnf(n, fcc_primitive_cell)) for n in range(2, 11)]
    n_refs = [2, 3, 7, 5, 10, 7, 20, 14, 18]
    np.testing.assert_equal(n_hnfs, n_refs)


def test_sc_hnfs(sc_primitive_cell):
    """Test number of non-equivalent HNFs for SC."""
    n_hnfs = [len(get_nonequivalent_hnf(n, sc_primitive_cell)) for n in range(2, 11)]
    n_refs = [3, 3, 9, 5, 13, 7, 24, 14, 23]
    np.testing.assert_equal(n_hnfs, n_refs)


def test_tetra_hnfs(tetra_primitive_cell):
    """Test number of non-equivalent HNFs for a tetragonal structure."""
    n_hnfs = [len(get_nonequivalent_hnf(n, tetra_primitive_cell)) for n in range(2, 11)]
    n_refs = [5, 5, 17, 9, 29, 13, 51, 28, 53]
    np.testing.assert_equal(n_hnfs, n_refs)


def test_perovskite_hnfs(perovskite_unitcell):
    """Test number of non-equivalent HNFs for cubic perovskite."""
    n_hnfs = [len(get_nonequivalent_hnf(n, perovskite_unitcell)) for n in range(2, 11)]
    n_refs = [3, 3, 9, 5, 13, 7, 24, 14, 23]
    np.testing.assert_equal(n_hnfs, n_refs)
