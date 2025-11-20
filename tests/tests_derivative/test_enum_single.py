"""Tests of enumerating derivative structures."""

from pathlib import Path

from pyclupan.api.pyclupan import Pyclupan

cwd = Path(__file__).parent


def test_deriv_binary_fcc():
    """Test enumerating binary derivative structures for FCC."""
    clupan = Pyclupan(verbose=False)
    clupan.load_poscar(str(cwd) + "/poscar-fcc")
    elements = [[0, 1]]

    clupan.run_derivative(elements=elements, supercell_size=3)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [2, 2, 2]

    clupan.run_derivative(elements=elements, supercell_size=4)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [3, 3, 3, 3, 3, 2, 2]

    clupan.run_derivative(elements=elements, supercell_size=5)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [6, 6, 6, 6, 4]

    clupan.run_derivative(elements=elements, hnf=[[1, 0, 0], [0, 1, 0], [1, 0, 5]])
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [6]


def test_deriv_ternary_fcc():
    """Test enumerating ternary derivative structures for FCC."""
    clupan = Pyclupan(verbose=False)
    clupan.load_poscar(str(cwd) + "/poscar-fcc")
    elements = [[0, 1, 2]]

    clupan.run_derivative(elements=elements, supercell_size=3)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [7, 7, 7]

    clupan.run_derivative(elements=elements, supercell_size=4)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [15, 15, 15, 15, 15, 12, 9]

    clupan.run_derivative(elements=elements, supercell_size=5)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [36, 36, 36, 36, 21]

    clupan.run_derivative(elements=elements, hnf=[[1, 0, 0], [0, 1, 0], [1, 0, 5]])
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [36]


def test_deriv_ternary_tetra():
    """Test enumerating ternary derivative structures for a tetragonal structure."""
    clupan = Pyclupan(verbose=False)
    clupan.load_poscar(str(cwd) + "/poscar-tetra")
    elements = [[0, 1, 2]]

    clupan.run_derivative(elements=elements, supercell_size=3)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [7, 7, 7, 7, 7]

    clupan.run_derivative(elements=elements, supercell_size=4)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 12, 15, 15, 15, 12]
