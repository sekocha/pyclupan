"""Functions for handling lattice."""

import copy
import io
from typing import Optional

import numpy as np

from pyclupan.core.lattice_utils import (
    extract_sites,
    get_complete_labelings,
    get_inactive_labeling,
    is_active_size,
    map_active_array,
    set_element_strings,
    set_elements_on_sublattices,
    set_labelings_endmembers,
)
from pyclupan.core.pypolymlp_utils import PolymlpStructure, save_cell
from pyclupan.core.spin import set_spins


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
        self._set_spins()

        self._cell = cell
        self._reduced_cell = None
        self._active_sites = None
        self._inactive_sites = None
        self._inactive_labeling = None
        self._map_full_to_active_rep = None

        elements = self._elements_on_lattice
        spins = self._spins_on_lattice

        self._n_elements = max([e2 for e in elements for e2 in e]) + 1
        self._active_lattice = [i for i, e in enumerate(elements) if len(e) > 1]
        self._inactive_lattice = [i for i, e in enumerate(elements) if len(e) == 1]
        self._active_elements = [e2 for e in elements if len(e) > 1 for e2 in e]
        self._inactive_elements = [e2 for e in elements if len(e) == 1 for e2 in e]
        self._active_spins = np.array([s2 for s in spins if len(s) > 1 for s2 in s])

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
    def n_elements(self):
        """Return number of elements on the entire lattice."""
        return self._n_elements

    @property
    def active_sites(self):
        """Return active sites."""
        if self._active_sites is not None:
            return self._active_sites

        self._active_sites = extract_sites(self._cell, self._active_lattice)
        return self._active_sites

    @property
    def inactive_sites(self):
        """Return inactive sites."""
        if self._inactive_sites is not None:
            return self._inactive_sites

        self._inactive_sites = extract_sites(self._cell, self._inactive_lattice)
        return self._inactive_sites

    @property
    def inactive_labeling(self):
        """Return inactive labeling."""
        if self._inactive_labeling is not None:
            return self._inactive_labeling

        self._inactive_labeling = get_inactive_labeling(
            self._cell, self._elements_on_lattice, self._inactive_lattice
        )
        return self._inactive_labeling

    def complete_labelings(self, active_labelings: np.ndarray):
        """Return complete labeling from active labelings."""
        return get_complete_labelings(
            active_labelings,
            self.inactive_labeling,
            self.active_sites,
            self.inactive_sites,
        )

    @property
    def map_full_to_active_rep(self):
        """Return mapping full site ID to active site ID."""
        if self._map_full_to_active_rep is not None:
            return self._map_full_to_active_rep

        self._map_full_to_active_rep = np.array([None for _ in self._cell.types])
        for i, s in enumerate(self.active_sites):
            self._map_full_to_active_rep[s] = i
        return self._map_full_to_active_rep

    def to_active_site_rep(self, sites: np.ndarray):
        """Convert full sites to active site representation."""
        return self.map_full_to_active_rep[sites].astype(int)

    def is_active_size(self, active_labelings: np.ndarray):
        """Check if size of given elements, labelings, and spin is appropriate."""
        return is_active_size(active_labelings, self.active_sites)

    def is_active_element(self, labelings: np.ndarray):
        """Check if labelings are composed of active elements."""
        return np.all(np.isin(labelings, self._active_elements))

    def lattice_supercell(self, supercell: PolymlpStructure):
        """Return Lattice instance for supercell representation."""
        lattice_supercell = copy.deepcopy(self)
        lattice_supercell.cell = supercell
        lattice_supercell._reduced_cell = None
        lattice_supercell._active_sites = None
        lattice_supercell._inactive_sites = None
        lattice_supercell._inactive_labeling = None
        lattice_supercell._map_full_to_active_rep = None

        if len(self._cell.n_atoms) != len(supercell.n_atoms):
            raise RuntimeError("Number of sublattices in supercell is not consistent.")

        return lattice_supercell

    def to_spins(self, active_labelings: np.ndarray):
        """Convert active elements (labelings) to spins.

        active_labelings: Active elements (labelings) converted to spins.
        """
        return map_active_array(
            active_labelings,
            self.active_sites,
            self._cell,
            self._elements_on_lattice,
            self._spins_on_lattice,
        )

    def to_labelings(self, active_spins: np.ndarray):
        """Convert active spins to active labelings.

        active_spins: Active spins converted to labelings.
        """
        return map_active_array(
            active_spins,
            self.active_sites,
            self._cell,
            self._spins_on_lattice,
            self._elements_on_lattice,
        )

    @property
    def spins_on_lattice(self):
        """Return spins on sublattices."""
        return self._spins_on_lattice

    @property
    def basis_on_lattice(self):
        """Return basis IDs on lattice."""
        return self._basis_on_lattice

    @property
    def spin_polynomials(self):
        """Return spin polynomial functions."""
        return self._spin_poly

    def get_spin_polynomials(self, basis_ids: np.ndarray):
        """Return spin polynomial coefficients for given basis IDs."""
        return np.array([self._spin_poly[i] for i in basis_ids])

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

    @property
    def labelings_endmembers(self):
        """Return labelings of endmembers."""
        return set_labelings_endmembers(self._elements_on_lattice)

    @property
    def element_strings(self):
        """Return element strings using lattice attributes."""
        return set_element_strings(
            self._cell, self._elements_on_lattice, self._n_elements
        )
