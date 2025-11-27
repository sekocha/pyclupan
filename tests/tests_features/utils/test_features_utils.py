"""Tests of utility functions for calculating features."""

from pathlib import Path

import numpy as np

from pyclupan.core.pypolymlp_utils import Poscar
from pyclupan.features.features_utils import (
    element_strings_to_labeling,
    get_chemical_compositions,
    load_cluster_functions_hdf5,
)

cwd = Path(__file__).parent


def test_element_strings_to_labeling():
    """Test element_strings_to_labeling."""
    elements = ["Ag", "Ag", "Ag", "Au", "Au"]
    labeling = element_strings_to_labeling(elements, element_strings=("Ag", "Au"))
    np.testing.assert_equal(labeling, [0, 0, 0, 1, 1])


def test_load_cluster_functions_hdf5():
    """Test load_cluster_functions_hdf5."""
    cfs, ids, n_atoms = load_cluster_functions_hdf5(
        str(cwd) + "/pyclupan_features.hdf5"
    )
    np.testing.assert_allclose(cfs[2][3], 0.0, atol=1e-8)
    np.testing.assert_allclose(cfs[12][16], -1 / 3, atol=1e-8)
    assert ids[0] == "2-0-0"
    assert ids[1] == "2-1-0"
    assert ids[2] == "3-0-0"


def test_get_chemical_compositions():
    """Test get_chemical_compositions."""
    st1 = Poscar(str(cwd) + "/derivative-1").structure
    st2 = Poscar(str(cwd) + "/derivative-2").structure
    chems = get_chemical_compositions(
        structures=[st1, st2],
        element_strings=("Ag", "Au"),
    )
    np.testing.assert_equal(chems, [[2, 2], [1, 3]])

    labelings = [
        [0, 0, 0, 0, 2, 2, 3, 3, 3, 3],
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 3],
        [0, 0, 1, 1, 2, 2, 3, 3, 3, 3],
        [0, 1, 1, 1, 2, 2, 3, 3, 3, 3],
        [1, 1, 1, 1, 2, 2, 3, 3, 3, 3],
    ]
    chems = get_chemical_compositions(labelings=labelings, n_elements=4)
    chems_true = np.array(
        [[4, 0, 2, 4], [3, 1, 2, 4], [2, 2, 2, 4], [1, 3, 2, 4], [0, 4, 2, 4]]
    )
    np.testing.assert_equal(chems, chems_true)
