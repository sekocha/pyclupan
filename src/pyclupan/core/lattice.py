"""Functions for handling lattice."""

import io
from typing import Optional

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, save_cell
from pyclupan.core.spin import set_spins


def _check_elements(elements_lattice: list, n_sublattice: int):
    """Check element IDs."""
    uniq_elements = np.unique([e2 for e1 in elements_lattice for e2 in e1])
    if len(uniq_elements) != np.max(uniq_elements) + 1:
        raise RuntimeError("Element IDs are not sequential.")

    if n_sublattice != len(elements_lattice):
        raise RuntimeError(
            "Number of sublattices is not equal to the size of elements."
        )
    return True


def set_elements_on_sublattices(
    n_sites: list,
    occupation: Optional[list] = None,
    elements: Optional[list] = None,
):
    """Initialize elements on sublattices.

    n_sites: Number of lattice sites for primitive cell.
    occupation: Lattice IDs occupied by each element.
            Example: [[0], [1], [2], [2]].
    elements: Element IDs on each lattices.
            Example: [[0], [1], [2, 3]].
    """
    if occupation is None and elements is None:
        elements_lattice = [[0, 1] for n in n_sites]
    elif elements is not None:
        elements_lattice = elements
    elif occupation is not None:
        max_lattice_id = max([oc2 for oc1 in occupation for oc2 in oc1])
        elements_lattice = [[] for i in range(max_lattice_id + 1)]
        for e, oc1 in enumerate(occupation):
            for oc2 in oc1:
                elements_lattice[oc2].append(e)
        elements_lattice = [sorted(e1) for e1 in elements_lattice]

    _check_elements(elements_lattice, len(n_sites))
    return elements_lattice


class Lattice:
    """Class for defining lattice."""

    def __init__(
        self,
        cell: PolymlpStructure,
        occupation: Optional[list] = None,
        elements: Optional[list] = None,
    ):
        """Init method.

        Parameter
        ---------
        cell: Unit cell of lattice.
        occupation: Lattice IDs occupied by each element.
                Example: [[0], [1], [2], [2]].
        elements: Element IDs on each lattices.
                Example: [[0], [1], [2, 3]].
        """
        self._elements_on_lattice = set_elements_on_sublattices(
            n_sites=cell.n_atoms,
            occupation=occupation,
            elements=elements,
        )
        self._cell = cell
        self._active_sites = None
        self._reduced_cell = None

        self._active_lattice = [
            i for i, e in enumerate(self._elements_on_lattice) if len(e) > 1
        ]
        self._set_spins()

    def _set_spins(self):
        """Set spin values and point cluster functions."""
        spin_info = set_spins(self._elements_on_lattice)
        self._spins_on_lattice, self._basis_on_lattice, self._spin_poly = spin_info

    @property
    def cell(self):
        """Return cell of lattice."""
        return self._cell

    @cell.setter
    def cell(self, c: PolymlpStructure):
        """Set cell."""
        self._cell = c

    @property
    def axis(self):
        """Return lattice axis."""
        return self._cell.axis

    @property
    def positions(self):
        """Return lattice sites in fractional coordinates."""
        return self._cell.positions

    @property
    def types(self):
        """Return lattice types for each site."""
        return self._cell.types

    @property
    def elements_on_lattice(self):
        """Return elements on sublattices."""
        return self._elements_on_lattice

    @property
    def active_sites(self):
        """Return active sites."""
        if self._active_sites is not None:
            return self._active_sites

        n_sites = self._cell.n_atoms
        self._active_sites = []
        for lattice_id in self._active_lattice:
            begin = sum(n_sites[:lattice_id])
            self._active_sites.extend(list(range(begin, begin + n_sites[lattice_id])))
        self._active_sites = np.array(self._active_sites)
        return self._active_sites

    @property
    def reduced_cell(self):
        """Return reduced cell."""
        if self._reduced_cell is not None:
            return self._reduced_cell

        from pyclupan.core.pypolymlp_utils import ReducedCell

        reduced = ReducedCell(self._cell.axis, method="delaunay")
        self._reduced_cell = reduced.reduce_structure(self._cell)
        return self._reduced_cell

    def save(self, file: Optional[str] = None):
        """Save lattice."""
        if isinstance(file, str):
            f = open(file, "w")
        elif isinstance(file, io.IOBase):
            f = file
        else:
            raise RuntimeError("file is not str or io.IOBase")

        save_cell(self._cell, tag="lattice_cell", file=f)
        print("elements_on_lattice:", file=f)
        for ele in self._elements_on_lattice:
            print("-", list(ele), file=f)
        print(file=f)

        if isinstance(file, str):
            f.close()
        return self

    def to_spins(self, elements: np.ndarray):
        """Convert elements (labelings) to spins."""
        if elements.shape[1] != sum(self._cell.n_atoms):
            raise RuntimeError("Size of given elements not consistent with lattice.")

        # TODO: Test
        spins_assigned = np.zeros(elements.shape, dtype=int)
        i = 0
        for ele, spins in zip(self._elements_on_lattice, self._spins_on_lattice):
            begin = sum(self._cell.n_atoms[:i])
            end = begin + self._cell.n_atoms[i]
            for e, s in zip(ele, spins):
                spins_assigned[:, begin:end][elements[:, begin:end] == e] = s
            i += 1
        return spins_assigned

    @property
    def basis_on_lattice(self):
        """Return basis IDs on lattice."""
        return self._basis_on_lattice

    @property
    def spin_polynomials(self):
        """Return spin polynomial functions."""
        return self._spin_poly
