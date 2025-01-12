"""API class for constructing ZDD."""

from typing import Optional

import numpy as np
from graphillion import GraphSet
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar

from pyclupan.zdd.zdd import ZddCore
from pyclupan.zdd.zdd_base import ZddLattice


class PyclupanZdd:
    """API class for constructing ZDD."""

    def __init__(self, verbose: bool = False):
        """Init method."""

        self._unitcell = None
        self._elements_lattice = None
        self._one_of_k_rep = None
        self._hnf = None
        self._site_perm = None
        self._site_perm_lt = None

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

    def initialize_zdd(
        self,
        supercell_size: int,
        elements_lattice: list,
        one_of_k_rep: bool = False,
    ):
        """Initialize ZDD.

        Parameters
        ----------
        supercell_size: Supercell size.
        elements_lattice: Element IDs on lattices. Example: [[0],[1],[2, 3]].
        one_of_k_rep: Use one-of-k representation.
        """
        self._elements_lattice = elements_lattice
        self._one_of_k_rep = one_of_k_rep

        n_sites = np.array(self._unitcell.n_atoms) * supercell_size
        self._zdd_lattice = ZddLattice(
            n_sites=n_sites,
            elements_lattice=elements_lattice,
            one_of_k_rep=one_of_k_rep,
            verbose=self._verbose,
        )
        self._zdd = ZddCore(self._zdd_lattice, verbose=self._verbose)
        return self

    def reset_zdd(self):
        """Reset zdd graph."""
        if self._zdd_lattice is None:
            raise RuntimeError("Initialize zdd in advance.")

        self._zdd = ZddCore(self._zdd_lattice, verbose=self._verbose)
        return self

    def set_permutations(self, supercell_matrix: np.ndarray):
        """Set atomic permutations."""
        from pypolymlp.utils.structure_utils import supercell

        from pyclupan.core.spglib_utils import get_permutation

        self._hnf = supercell_matrix
        sup = supercell(self._unitcell, supercell_matrix)
        self._site_perm, self._site_perm_lt = get_permutation(
            sup, superperiodic=True, hnf=supercell_matrix
        )
        return self

    @property
    def unitcell(self):
        """Return unit cell."""
        return self._unitcell

    @unitcell.setter
    def unitcell(self, cell):
        """Set unit cell."""
        self._unitcell = cell

    def all(self):
        """Return graph for all combinations."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.all()

    def empty(self):
        """Return empty graph."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.empty()

    def one_of_k(self):
        """Apply one-of-k representations."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.one_of_k()

    def composition(self, comp: tuple, tol: float = 1e-3):
        """Apply composition."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.composition(comp, tol=tol)

    def composition_range(self, comp_lb: tuple, comp_ub: tuple):
        """Apply composition lower and upper bounds."""
        # TODO: a test is required
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.composition_range(comp_lb, comp_ub)

    def no_endmembers(self):
        """Return graph with no endmember structures."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.no_endmembers()

    def nonequivalent_permutations(
        self,
        num_edges: Optional[int] = None,
        gs: Optional[GraphSet] = None,
    ):
        """Return ZDD of non-equivalent configurations."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        if self._site_perm is None:
            raise RuntimeError("Set permutations in advance.")

        return self._zdd.nonequivalent_permutations(
            self._site_perm, num_edges=num_edges, gs=gs
        )

    def including(self, node_idx: int):
        """Return graph including a node."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.including(node_idx)

    def including_single_cluster(self, nodes: list, gs: Optional[GraphSet] = None):
        """Return graph including a single cluster with nodes."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.including_single_cluster(nodes, gs=gs)

    def excluding_single_cluster(self, nodes: list, gs: Optional[GraphSet] = None):
        """Return graph excluding a single cluster with nodes."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.excluding_single_cluster(nodes, gs=gs)

    def excluding_clusters(self, nodes_list: list, gs: Optional[GraphSet] = None):
        """Return graph excluding clusters with node lists."""
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.excluding_clusters(nodes_list, gs=gs)

    def charge_balance(
        self, charge: list, gs: Optional[GraphSet] = None, eps: float = 1e-5
    ):
        """Return graph for charge-balanced strucures."""
        # TODO: Test is needed.
        if self._zdd is None:
            raise RuntimeError("Initialize zdd in advance.")
        return self._zdd.charge_balance(charge, gs=gs, eps=eps)
