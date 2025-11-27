"""Tests of enumerating derivative structures."""

from pathlib import Path

from pyclupan.api.pyclupan import Pyclupan

cwd = Path(__file__).parent


def test_deriv_perovskite_single_sublattice():
    """Test enumerating derivative structures for perovskite."""
    clupan = Pyclupan(verbose=False)
    clupan.load_poscar(str(cwd) + "/poscar-perovskite")
    elements = [[0], [1], [2, 3]]

    clupan.run_derivative(elements=elements, supercell_size=4)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [333, 333, 408, 145, 243, 183, 230, 193, 71]

    clupan.run_derivative(elements=elements, supercell_size=4, comp_lb=[(2, 0.66)])
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [69, 69, 85, 32, 53, 41, 48, 42, 16]

    clupan.run_derivative(
        elements=elements,
        supercell_size=4,
        comp_lb=[(2, 0.66)],
        comp_ub=[(2, 0.8)],
    )
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [60, 60, 74, 27, 45, 34, 43, 37, 14]

    clupan.run_derivative(
        elements=elements, supercell_size=4, comp=[(2, 2 / 3), (3, 1 / 3)]
    )
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [40, 40, 47, 18, 29, 21, 27, 23, 8]


def test_deriv_perovskite_multiple_sublattices1():
    """Test enumerating derivative structures for perovskite."""
    clupan = Pyclupan(verbose=False)
    clupan.load_poscar(str(cwd) + "/poscar-perovskite")
    elements = [[0, 1], [0, 1, 2], [3, 4]]

    clupan.run_derivative(elements=elements, supercell_size=2)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [474, 357, 162]


def test_deriv_perovskite_multiple_sublattices2():
    """Test enumerating derivative structures for perovskite."""
    clupan = Pyclupan(verbose=False)
    clupan.load_poscar(str(cwd) + "/poscar-perovskite")
    elements = [[0, 1], [2, 3], [4]]

    charges = [(0, 2.0), (1, 3.0), (2, 4.0), (3, 3.0), (4, -2.0)]
    clupan.run_derivative(elements=elements, supercell_size=4, charges=charges)
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [10, 11, 7, 10, 8, 7, 5, 8, 3]
    assert sum(n_str) == 69

    clupan.run_derivative(
        elements=elements, supercell_size=4, charges=charges, superperiodic=True
    )
    ds = clupan.derivative_structures
    n_str = [ds1.n_labelings for ds1 in ds]
    assert n_str == [13, 15, 10, 13, 12, 10, 9, 13, 6]
    assert sum(n_str) == 101
