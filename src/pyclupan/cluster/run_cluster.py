"""Functions for searchning nonequivalent clusters."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import PolymlpStructure, supercell
from pyclupan.core.spglib_utils import get_permutation


@dataclass
class LatticeCS:
    """Class for defining lattice used in cluster search."""

    supercell: PolymlpStructure
    permutation: Optional[np.ndarray] = None


class ClusterSearch:
    """Class for performing cluster search."""

    def __init__(self, lattice: Lattice, verbose: bool = False):
        """Init method."""
        self._lattice = lattice
        self._verbose = verbose
        self._reduced_cell = lattice.reduced_cell

        self._supercell = None
        self._permutation = None

        self._enum_clusters = dict()

        # elements_lattice = lattice.elements_on_lattice

    def _find_supercell(self, cutoffs: tuple):
        """Find supercell expansion for searching clusters."""
        max_cut = max(cutoffs)
        reduced_norm = np.linalg.norm(self._reduced_cell.axis, axis=0)

        supercell_matrix = np.diag(np.ceil(np.ones(3) * max_cut * 2 / reduced_norm))
        self._supercell = supercell(
            self._lattice.reduced_cell,
            supercell_matrix=supercell_matrix,
        )
        self._permutation = get_permutation(self._supercell)
        return self

    def search(self, max_order: int = 4, cutoffs: tuple[float] = (6.0, 6.0, 6.0)):
        """Search clusters."""
        if len(cutoffs) != max_order - 1:
            raise RuntimeError("Cutoff size must be equal to max_order - 1.")

        self._find_supercell(cutoffs)
        rep = np.min(self._permutation[:, self._lattice.active_sites], axis=0)
        nonequiv_sites = np.unique(rep)
        if self._verbose:
            print("Nonequivalent sites:", nonequiv_sites)

        self._enum_clusters[1] = nonequiv_sites
        for order in range(2, max_order + 1):
            pass
            # cut = cutoffs[order - 2]


def run_cluster(
    unitcell: PolymlpStructure,
    occupation: Optional[list] = None,
    elements: Optional[list] = None,
    cutoff: float = 6.0,
    max_order: int = 4,
    verbose: bool = False,
):
    """Search nonequivalent clusters."""

    lattice = Lattice(
        unitcell=unitcell,
        occupation=occupation,
        elements=elements,
        verbose=verbose,
    )
    cs = ClusterSearch(lattice, verbose=verbose)
    cs.search()
