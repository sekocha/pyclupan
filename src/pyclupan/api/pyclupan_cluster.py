"""API Class for enumerating nonequivalent clusters."""

from typing import Optional

import numpy as np

from pyclupan.cluster.run_cluster import run_cluster
from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar


class PyclupanCluster:
    """API Class for enumerating clusters."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._unitcell = None
        self._clusters = None
        np.set_printoptions(legacy="1.21")

    def load_poscar(self, poscar: str = "POSCAR") -> PolymlpStructure:
        """Parse POSCAR files.

        Parameter
        ---------
        poscar: Name of POSCAR file.

        Returns
        -------
        structure: Structure in PolymlpStructure format.
        """
        self._unitcell = Poscar(poscar).structure
        return self._unitcell

    @property
    def unitcell(self):
        """Return unitcell."""
        return self._unitcell

    @unitcell.setter
    def unitcell(self, cell: PolymlpStructure):
        """Setter of unitcell."""
        self._unitcell = cell

    def run_cluster(
        self,
        occupation: Optional[list] = None,
        elements: Optional[list] = None,
        max_order: int = 4,
        cutoffs: tuple[float] = (6.0, 6.0, 6.0),
        filename: str = "pyclupan_cluster.yaml",
    ):
        """Search nonequivalent clusters.
        Parameters
        ----------
        occupation: Lattice IDs occupied by elements.
                    Example: [[0], [1], [2], [2]].
        elements: Element IDs on lattices.
                  Example: [[0], [1], [2, 3]].
        max_order: Maximum order of clusters.
        cutoffs: Cutoff distances for orders >= 2.
                (two-body, three-body, four-body, ...)
                Size of cutoffs must be equal to max_order - 1.
                Cutoffs must be smaller or equal to those for smaller orders.
        filename: Name of output file for cluster search results.
                  If None, no file will be generated.
        """
        if self._unitcell is None:
            raise RuntimeError("Unitcell not found.")

        self._clusters = run_cluster(
            unitcell=self._unitcell,
            occupation=occupation,
            elements=elements,
            max_order=max_order,
            cutoffs=cutoffs,
            filename=filename,
            verbose=self._verbose,
        )
        return self

    @property
    def clusters(self):
        """Return nonequivalent clusters.

        Return
        ------
        clusters: Nonequivalent clusters, dict[list[ClusterAttr]].
                  Dictionary keys are cluster orders.
        """
        return self._clusters
