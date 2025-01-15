"""Utility functions for pyclupan."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

import numpy as np
from pypolymlp.core.data_format import PolymlpStructure

from pyclupan.zdd.zdd_base import ZddLattice


def set_elements_on_sublattices(
    n_sites: list,
    occupation: Optional[list] = None,
    elements: Optional[list] = None,
):
    """Initialize elements on sublattices.

    n_sites: Number of lattice sites for primitive cell.
    occupation: Lattice IDs occupied by elements.
                Example: [[0], [1], [2], [2]].
    elements: Element IDs on lattices.
              Example: [[0],[1],[2, 3]].
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

    if len(n_sites) != len(elements_lattice):
        raise RuntimeError(
            "Inconsistent numbers of sublattices in n_sites and elements_lattice."
        )
    return elements_lattice


def set_compositions(
    elements_lattice: Optional[list] = None,
    comp: Optional[list] = None,
    comp_lb: Optional[list] = None,
    comp_ub: Optional[list] = None,
):
    """Set compositions from input parameters.

    Parameters
    ----------
    elements_lattice : Element IDs on lattices.
                       Example: [[0],[1],[2, 3]].
    comp: Compositions for sublattices (n_elements / n_sites).
          Compositions are not needed to be normalized.
    comp_lb: Lower bounds of compositions for sublattices.
    comp_ub: Upper bounds of compositions for sublattices.
    """
    n_elements = max([e2 for e1 in elements_lattice for e2 in e1]) + 1
    comp = normalize_compositions(comp, n_elements, elements_lattice)
    comp_lb = normalize_compositions(comp_lb, n_elements, elements_lattice)
    comp_ub = normalize_compositions(comp_ub, n_elements, elements_lattice)
    return (comp, comp_lb, comp_ub)


def normalize_compositions(comp_in: list, n_elements: int, elements_lattice: list):
    """Normalize compositions."""
    if comp_in is None:
        comp = [None for i in range(n_elements)]
        return comp

    for comp_pair in comp_in:
        if len(comp_pair) != 2:
            raise RuntimeError(
                "Composition must be given as (element ID, composition)."
            )

    comp = [None for i in range(n_elements)]
    for ele, c in comp_in:
        ele, c = int(ele), float(Fraction(c))
        comp[ele] = c

    comp = np.array(comp)
    for elements in elements_lattice:
        target_comp = comp[elements]
        if list(target_comp).count(None) != len(target_comp):
            for c, ele in zip(target_comp, elements):
                if c is None:
                    raise RuntimeError("Composition not found for element", ele)
            total = sum(target_comp)
            comp[elements] = target_comp / total
    return list(comp)


@dataclass
class Derivatives:
    """Dataclass for derivative structures.

    Parameters
    ----------
    TODO: Add docs on parameters.
    """

    zdd_lattice: ZddLattice
    unitcell: PolymlpStructure
    hnf: np.ndarray
    active_labelings: np.ndarray
    inactive_labeling: Optional[np.ndarray] = None
    comp: Optional[tuple] = None
    comp_lb: Optional[tuple] = None
    comp_ub: Optional[tuple] = None
    supercell_id: int = 0

    active_sites: Optional[list] = None
    inactive_sites: Optional[list] = None
    elements: Optional[list] = None
    element_orbit: Optional[list] = None
    n_total_sites: Optional[int] = None
    n_labelings: Optional[int] = None
    supercell_size: Optional[int] = None

    def __post_init__(self):
        """Post init method."""
        site_set = self.zdd_lattice.site_attrs_set
        self.active_sites = np.array(site_set.active_sites)
        self.inactive_sites = np.array(site_set.inactive_sites)
        self.elements = site_set.elements
        self.element_orbit = self.zdd_lattice._element_orbit
        self.n_total_sites = self.zdd_lattice._n_total_sites

        self.n_labelings = self.active_labelings.shape[0]
        self.supercell_size = round(np.linalg.det(self.hnf))

    @property
    def supercell(self):
        """Return supercell."""
        from pypolymlp.utils.structure_utils import supercell

        return supercell(self.unitcell, self.hnf)

    @property
    def complete_labelings(self):
        """Return complete labelings for both active and inactive sites."""
        labelings = np.zeros((self.n_labelings, self.n_total_sites), dtype=np.uint8)
        labelings[:, self.active_sites] = self.active_labelings
        if len(self.inactive_sites) > 0:
            labelings[:, self.inactive_sites] = self.inactive_labeling
        return labelings

    @complete_labelings.setter
    def complete_labelings(self, labelings: np.array):
        """Set complete labelings for both active and inactive sites."""
        self.active_labelings = labelings[:, self.active_sites]
        self.n_labelings = self.active_labelings.shape[0]

    def get_complete_labeling(self, idx: int):
        """Return a single complete labeling for given id."""
        labeling = np.zeros(self.n_total_sites, dtype=np.uint8)
        labeling[self.active_sites] = self.active_labelings[idx]
        if len(self.inactive_sites) > 0:
            labeling[self.inactive_sites] = self.inactive_labeling
        return labeling

    def all(self):
        """Sample all derivative structures."""
        self._samples = np.arange(self.n_labelings, dtype=int)
        return self

    def random(self, n_samples: int = 100):
        """Sample derivative structures randomly."""
        candidates = np.arange(self.n_labelings, dtype=int)
        self._samples = np.random.choice(candidates, size=n_samples, replace=False)
        return self


@dataclass
class DerivativesSet:
    """Dataclass for set of derivative structures.

    Parameters
    ----------
    TODO: Add docs on parameters.
    """

    derivatives_set: list[Derivatives]

    def __post_init__(self):
        """Post init method."""
        self._cnt = np.array([derivs.n_labelings for derivs in self.derivatives_set])

    def _choose_cell(self, n_samples: int = 100, prob: np.ndarray = None):
        """Choose HNF ids randomly."""
        candidates = np.arange(len(self.derivatives_set), dtype=int)
        icells = np.random.choice(candidates, size=n_samples, replace=True, p=prob)
        key, cnt = np.unique(icells, return_counts=True)

        n_samples_all = np.zeros(len(self.derivatives_set), dtype=int)
        n_samples_all[key] = cnt
        while np.any(n_samples_all > self._cnt):
            ids_full = np.where(n_samples_all > self._cnt)[0]
            ids = np.where(n_samples_all < self._cnt)[0]
            n_samples_all[ids_full] = self._cnt[ids_full]
            n_resamples = n_samples - np.sum(n_samples_all)

            p = None if prob is None else prob[ids]
            icells = np.random.choice(ids, size=n_resamples, replace=True, p=p)
            key, cnt = np.unique(icells, return_counts=True)
            n_samples_all[key] += cnt
        return n_samples_all

    def all(self):
        """Sample all derivative structures for all HNFs."""
        for derivs in self.derivatives_set:
            derivs.all()

    def uniform(self, n_samples: int = 100):
        """Sample derivative structures randomly from uniformly-sampled HNFs."""
        if n_samples >= np.sum(self._cnt):
            self.all()
            return self

        n_samples_cell = self._choose_cell(n_samples=n_samples)
        for n, derivs in zip(n_samples_cell, self.derivatives_set):
            derivs.random(n_samples=n)
        return self

    def random(self, n_samples: int = 100):
        """Sample derivative structures randomly."""
        if n_samples >= np.sum(self._cnt):
            self.all()
            return self

        n_samples_cell = self._choose_cell(n_samples=n_samples, p=self._cnt)
        for n, derivs in zip(n_samples_cell, self.derivatives_set):
            derivs.random(n_samples=n)
        return self
