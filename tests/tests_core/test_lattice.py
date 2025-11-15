"""Tests of Lattice class."""

from pathlib import Path

import numpy as np

from pyclupan.core.cell_utils import supercell_reduced
from pyclupan.core.lattice import Lattice, set_elements_on_sublattices
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


def test_elements_on_sublattices():
    """Test set_elements_on_sublattices."""
    n_sites = [1]
    elements = [[0, 1]]
    elements_lattice = set_elements_on_sublattices(n_sites)
    assert elements == elements_lattice

    occupation = [[0], [0]]
    elements_lattice = set_elements_on_sublattices(n_sites, occupation=occupation)
    assert elements == elements_lattice

    n_sites = [1, 1, 3]
    elements = [[0], [1], [2, 3]]
    elements_lattice = set_elements_on_sublattices(n_sites, elements=elements)
    assert elements == elements_lattice

    elements = [[0], [1], [2, 3]]
    occupation = [[0], [1], [2], [2]]
    elements_lattice = set_elements_on_sublattices(n_sites, occupation=occupation)
    assert elements == elements_lattice

    elements = [[0, 1], [0, 1, 2, 3], [4, 5, 6]]
    occupation = [[0, 1], [0, 1], [1], [1], [2], [2], [2]]
    elements_lattice = set_elements_on_sublattices(n_sites, occupation=occupation)
    assert elements == elements_lattice


def test_lattice():
    """Test Lattice class."""
    unitcell = Poscar("poscar-fcc").structure
    elements = [[0, 1]]
    lattice = Lattice(unitcell, elements=elements)
    hnf = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 3]])

    supercell = supercell_reduced(unitcell, supercell_matrix=hnf)
    lattice_supercell = lattice.lattice_supercell(supercell)

    axis_true = np.array(
        [[2.6965, -0.0, -5.393], [0.0, 5.393, -2.6965], [2.6965, -0.0, 2.6965]]
    )

    np.testing.assert_allclose(lattice_supercell.axis, axis_true)

    print(lattice_supercell.positions)
    print(lattice_supercell.types)
    assert 1 == 0


#     labelings = np.array(
#         [
#             [0, 0, 0, 1],
#             [0, 1, 1, 1],
#             [0, 1, 0, 1],
#             [0, 0, 1, 1],
#             [0, 0, 0, 0],
#             [1, 1, 1, 1],
#         ]
#     )
