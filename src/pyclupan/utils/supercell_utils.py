"""Classes for constructing supercells."""

from typing import Literal

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import supercell
from pypolymlp.utils.spglib_utils import ReducedCell


def supercell_reduced(
    st: PolymlpStructure,
    supercell_matrix: np.ndarray,
    method: Literal["niggli", "delaunay"] = "delaunay",
):
    """Construct supercell for a given supercell matrix."""
    st_sup = supercell(st, supercell_matrix)
    reduced = ReducedCell(st_sup.axis, method=method)
    st_sup.axis = reduced.reduced_axis
    st_sup.positions = reduced.transform_fr_coords(st_sup.positions)
    return st_sup
