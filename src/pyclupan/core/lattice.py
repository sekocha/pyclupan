"""Functions for handling lattice."""

from typing import Optional

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure


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
        unitcell: PolymlpStructure,
        occupation: Optional[list] = None,
        elements: Optional[list] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameter
        ---------
        unitcell: Unit cell of lattice.
        occupation: Lattice IDs occupied by each element.
                Example: [[0], [1], [2], [2]].
        elements: Element IDs on each lattices.
                Example: [[0], [1], [2, 3]].
        """
        self._elements_on_lattice = set_elements_on_sublattices(
            n_sites=unitcell.n_atoms,
            occupation=occupation,
            elements=elements,
        )

        self._unitcell = unitcell
        self._verbose = verbose

        self._active_lattice = [
            i for i, e in enumerate(self._elements_on_lattice) if len(e) > 1
        ]
        self._active_sites = None
        self._reduced_cell = None

    @property
    def unitcell(self):
        """Return unitcell of lattice."""
        return self._unitcell

    @property
    def elements_on_lattice(self):
        """Return elements on sublattices."""
        return self._elements_on_lattice

    @property
    def active_sites(self):
        """Return active sites."""
        if self._active_sites is not None:
            return self._active_sites

        n_sites = self._unitcell.n_atoms
        self._active_sites = []
        for lattice_id in self._active_lattice:
            begin = sum(n_sites[:lattice_id])
            self._active_sites.extend(list(range(begin, begin + n_sites[lattice_id])))
        self._active_sites = np.array(self._active_sites)
        return self._active_sites

    @property
    def reduced_cell(self):
        """Return reduced unitcell."""
        if self._reduced_cell is not None:
            return self._reduced_cell

        from pyclupan.core.pypolymlp_utils import ReducedCell

        reduced = ReducedCell(self._unitcell.axis, method="delaunay")
        return reduced.reduce_structure(self._unitcell)
