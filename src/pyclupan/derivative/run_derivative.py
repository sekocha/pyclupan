"""Class and functions for enumerating derivative structures."""

# from typing import Optional

import numpy as np
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import supercell

from pyclupan.core.spglib_utils import get_permutation


def enum_derivatives(unitcell: PolymlpStructure, hnf: np.array):
    """Enumerate derivative structures for given HNF."""
    sup = supercell(unitcell, hnf)
    site_perm, site_perm_lt = get_permutation(sup, superperiodic=True, hnf=hnf)
    print(site_perm)
