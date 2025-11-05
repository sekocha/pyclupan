"""Tests of enumerating derivative structures."""

from pathlib import Path

from pyclupan.api.pyclupan import Pyclupan

cwd = Path(__file__).parent


def test_fcc_hnfs():
    """Test number of non-equivalent HNFs for FCC."""
    clupan = Pyclupan(verbose=False)
    clupan.load_poscar(str(cwd) + "/poscar-fcc")
    elements = [[0, 1]]
    clupan.run(elements=elements, supercell_size=3)
    ds = clupan.derivative_structures
    assert ds.n_structures == 6

    clupan.run(elements=elements, supercell_size=6)
    ds = clupan.derivative_structures
    assert ds.n_structures == 80


# def test_sc_hnfs():
#    """Test number of non-equivalent HNFs for SC."""
#    st = Poscar(str(cwd) + "/poscar-sc").structure
#    n_hnfs = [len(get_nonequivalent_hnf(n, st)) for n in range(2, 11)]
#    n_refs = [3, 3, 9, 5, 13, 7, 24, 14, 23]
#    np.testing.assert_equal(n_hnfs, n_refs)
#
#
# def test_tetra_hnfs():
#    """Test number of non-equivalent HNFs for a tetragonal structure."""
#    st = Poscar(str(cwd) + "/poscar-tetra").structure
#    n_hnfs = [len(get_nonequivalent_hnf(n, st)) for n in range(2, 11)]
#    n_refs = [5, 5, 17, 9, 29, 13, 51, 28, 53]
#    np.testing.assert_equal(n_hnfs, n_refs)
