"""API class for constructing ZDD."""

from typing import Optional

import numpy as np
from graphillion import GraphSet
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar

# from pyclupan.zdd.zdd import ZddCore
# from pyclupan.zdd.zdd_base import ZddLattice


class PyclupanZdd:
    """API class for constructing ZDD."""

    def __init__(self, verbose: bool = False):
        """Init method."""

        self._unitcell = None
        self._hnf = None

        self._zdd_lattice = None
        self._zdd = None

        self._verbose = verbose

    def load_poscar(self, poscar: str = "POSCAR") -> PolymlpStructure:
        """Parse POSCAR files.

        Returns
        -------
        structure: Structure in PolymlpStructure format.
        """
        self._unitcell = Poscar(poscar).structure
        return self._unitcell

    def initialize_zdd(self, elements: list, hnfsupercell_size: int):
        """Initialize ZDD.

        Parameters
        ----------
        elements: Element IDs on lattices.
                Example: [[0],[1],[2, 3]].
        """
        # n_sites = np.array(unitcell.n_atoms) * supercell_size
        # zdd_lattice = ZddLattice(
        #     n_sites=n_sites,
        #     elements_lattice=elements_lattice,
        #     one_of_k_rep=one_of_k_rep,
        #     verbose=self._verbose,
        # )

    def all(self):
        """Return graph for all combinations."""
        return self._zdd.all()

    def empty(self):
        """Return empty graph."""
        return self._zdd.empty()

    def one_of_k(self):
        """Apply one-of-k representations."""
        return self._zdd.one_of_k()

    def composition(self, comp: tuple, tol: float = 1e-3):
        """Apply composition."""
        return self._zdd.composition(comp, tol=tol)

    def composition_range(self, comp_lb: tuple, comp_ub: tuple):
        """Apply composition lower and upper bounds."""
        # TODO: a test is required
        return self._zdd.composition_range(comp_lb, comp_ub)

    def no_endmembers(self):
        """Return graph with no endmember structures."""
        return self._zdd.no_endmembers()

    def nonequivalent_permutations(
        self,
        site_permutations: np.ndarray,
        num_edges: Optional[int] = None,
        gs: Optional[GraphSet] = None,
    ):
        """Return ZDD of non-equivalent configurations."""
        return self._zdd.nonequivalent_permutations(
            site_permutations, num_edges=num_edges, gs=gs
        )

    def including(self, node_idx: int):
        """Return graph including a node."""
        return self._zdd.including(node_idx)

    def including_single_cluster(self, nodes: list, gs: Optional[GraphSet] = None):
        """Return graph including a single cluster with nodes."""
        return self._zdd.including_single_cluster(nodes, gs=gs)

    def excluding_single_cluster(self, nodes: list, gs: Optional[GraphSet] = None):
        """Return graph excluding a single cluster with nodes."""
        return self._zdd.excluding_single_cluster(nodes, gs=gs)

    def excluding_clusters(self, nodes_list: list, gs: Optional[GraphSet] = None):
        """Return graph excluding clusters with node lists."""
        return self._zdd.excluding_clusters(nodes_list, gs=gs)

    def charge_balance(
        self, charge: list, gs: Optional[GraphSet] = None, eps: float = 1e-5
    ):
        """Return graph for charge-balanced strucures."""
        # TODO: Test is needed.
        return self._zdd.charge_balance(charge, gs=gs, eps=eps)
