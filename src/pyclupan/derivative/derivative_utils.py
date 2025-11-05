"""Utility functions for pyclupan."""

import copy
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.vasp_utils import write_poscar_file

from pyclupan.zdd.zdd_base import ZddLattice


def reorder_positions(st: PolymlpStructure):
    """Reorder positions, types, and elements."""

    map_elements = dict()
    for t, e in zip(st.types, st.elements):
        map_elements[t] = e

    n_atoms, positions_reorder, types_reorder = [], [], []
    for i in sorted(set(st.types)):
        ids = np.array(st.types) == i
        n_atoms.append(np.count_nonzero(ids))
        positions_reorder.extend(st.positions.T[ids])
        types_reorder.extend(np.array(st.types)[ids])

    st.positions = np.array(positions_reorder).T
    st.n_atoms = n_atoms
    st.types = types_reorder
    st.elements = [map_elements[t] for t in types_reorder]
    return st


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

        self._samples = None

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

    def save(self, path: str = "poscars", elements: tuple = ("Al", "Cu")):
        """Save derivative structures sampled."""
        if self._samples is None:
            raise RuntimeError("Sampled structures are not found.")

        if len(self.elements) != len(elements):
            raise RuntimeError("Number of element strings is not compatible.")

        os.makedirs(path, exist_ok=True)
        sup = self.supercell
        prefix = "POSCAR-" + str(self.supercell_size) + "-" + str(self.supercell_id + 1)
        elements = np.array(elements)
        for i, sample_id in enumerate(self._samples):
            filename = prefix + "-" + str(i + 1).zfill(5)
            sup_copy = copy.deepcopy(sup)
            sup_copy.types = self.get_complete_labeling(sample_id)
            sup_copy.elements = elements[sup_copy.types]
            sup_copy = reorder_positions(sup_copy)
            write_poscar_file(sup_copy, filename=path + "/" + filename)
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

    def save(self, path: str = "poscars", elements: tuple = ("Al", "Cu")):
        """Save derivative structures sampled."""
        os.makedirs(path, exist_ok=True)
        for derivs in self.derivatives_set:
            if derivs._samples is not None:
                derivs.save(path=path, elements=elements)
