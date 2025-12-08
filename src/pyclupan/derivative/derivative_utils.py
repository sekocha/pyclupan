"""Utility functions for sampling derivative structures."""

from __future__ import annotations

import copy
import io
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import yaml

from pyclupan.core.cell_utils import supercell_reduced
from pyclupan.core.lattice import Lattice
from pyclupan.core.pypolymlp_utils import load_cell, save_cell, write_poscar_file


def get_structure_id(supercell_size: int, supercell_id: int, structure_id: int):
    """Return structure ID."""
    return "-".join((str(supercell_size), str(supercell_id), str(structure_id)))


@dataclass
class Derivatives:
    """Dataclass for derivative structures.

    Parameters
    ----------
    TODO: Add docs on parameters.
    """

    lattice_unitcell: Lattice
    supercell_matrix: Optional[np.ndarray] = None
    supercell_id: int = 0
    active_labelings: Optional[np.ndarray] = None
    inactive_labeling: Optional[np.ndarray] = None
    structure_ids: Optional[list] = None

    comp: Optional[list] = None
    comp_lb: Optional[list] = None
    comp_ub: Optional[list] = None

    lattice_supercell: Optional[Lattice] = None

    def __post_init__(self):
        """Post init method."""
        sup_red = supercell_reduced(self.lattice_unitcell.cell, self.supercell_matrix)
        self.lattice_supercell = self.lattice_unitcell.lattice_supercell(sup_red)
        self._sample = None

    @property
    def active_sites(self):
        """Return active sites."""
        return self.lattice_supercell.active_sites

    @property
    def inactive_sites(self):
        """Return inactive sites."""
        return self.lattice_supercell.inactive_sites

    @property
    def n_labelings(self):
        """Return number of labelings."""
        return self.active_labelings.shape[0]

    @property
    def supercell_size(self):
        """Return supercell size."""
        return round(np.linalg.det(self.supercell_matrix))

    @property
    def unitcell(self):
        """Return unitcell."""
        return self.lattice_unitcell.cell

    @property
    def supercell(self):
        """Return supercell."""
        return self.lattice_supercell.cell

    def all(self):
        """Sample all labelings."""
        self._sample = np.arange(self.n_labelings, dtype=int)
        return self._sample

    def random(self, n_samples: int = 100):
        """Sample labelings randomly."""
        candidates = np.arange(self.n_labelings, dtype=int)
        self._sample = np.random.choice(candidates, size=n_samples, replace=False)
        self._sample = np.sort(self._sample)
        return self._sample

    def select(self, key: int):
        """Sample single labeling."""
        if self._sample is None:
            self._sample = []
        self._sample.append(key)
        return self

    @property
    def sample(self):
        """Return IDs of sampled labelings."""
        return self._sample

    @sample.setter
    def sample(self, s: np.ndarray):
        """Setter IDs of sampled labelings."""
        self._sample = np.array(s)

    @property
    def sample_ids(self):
        """Return complete labelings from both active and inactive labelings."""
        if self._sample is None:
            raise RuntimeError("Sampled labelings not found.")
        return [(self.supercell_size, self.supercell_id, i) for i in self._sample]

    def get_complete_labelings(self, active_labelings: Optional[np.ndarray] = None):
        """Return complete labelings from both active and inactive labelings."""
        if active_labelings is None:
            active_labelings = self.active_labelings
        return self.lattice_supercell.complete_labelings(active_labelings)

    @property
    def sampled_active_labelings(self):
        """Return active labelings."""
        if self._sample is None:
            raise RuntimeError("Sampled labelings not found.")
        return self.active_labelings[self._sample]

    @property
    def sampled_complete_labelings(self):
        """Return complete labelings from both active and inactive labelings."""
        if self._sample is None:
            raise RuntimeError("Sampled labelings not found.")
        return self.get_complete_labelings(self.sampled_active_labelings)

    def save(self, element_strings: tuple, path: str = "poscars"):
        """Save derivative structures sampled."""
        if self._sample is not None and len(self._sample) > 0:
            os.makedirs(path, exist_ok=True)
            for ids, labeling in zip(self.sample_ids, self.sampled_complete_labelings):
                structure_id = get_structure_id(*ids)
                filename = "poscar-" + structure_id
                sup = copy.deepcopy(self.supercell)
                sup.types = labeling
                sup.elements = list(np.array(element_strings)[sup.types])
                sup = sup.reorder()

                header = "pyclupan: " + structure_id
                write_poscar_file(sup, filename=path + "/" + filename, header=header)
        return self

    def write_attrs(self, file: Union[str, io.IOBase]):
        """Save attributes of derivative structures sampled."""
        if isinstance(file, str):
            f = open(file, "w")
        elif isinstance(file, io.IOBase):
            f = file
        else:
            raise RuntimeError("file is not str or io.IOBase")

        if self._sample is not None and len(self._sample) > 0:
            print("- supercell_matrix:", file=f)
            print("  -", list(self.supercell_matrix[0]), file=f)
            print("  -", list(self.supercell_matrix[1]), file=f)
            print("  -", list(self.supercell_matrix[2]), file=f)
            print("  inactive_labeling:", list(self.inactive_labeling), file=f)
            print("  active_labelings:", file=f)
            for labeling in self.sampled_active_labelings:
                print("  -", list(labeling), file=f)
            print("  id:", file=f)
            for ids in self.sample_ids:
                print("  -", get_structure_id(*ids), file=f)
            print(file=f)
        return self


@dataclass
class DerivativesSet:
    """Dataclass for derivative structure set.

    Parameters
    ----------
    TODO: Add docs on parameters.
    """

    derivatives_set: list[Derivatives]

    def __post_init__(self):
        """Post init method."""
        self._sample = None
        self._set_map_supercell_ids()

    def __iter__(self):
        """Iter method."""
        return iter(self.derivatives_set)

    def __getitem__(self, index: int):
        """Getitem method."""
        return self.derivatives_set[index]

    def __setitem__(self, index: int, value: Derivatives):
        """Setitem method."""
        self.derivatives_set[index] = value

    def __len__(self):
        """Len method."""
        return len(self.derivatives_set)

    def _set_map_supercell_ids(self):
        """Set mapping from supercell IDs to array id."""
        self._map_supercell_ids = dict()
        for i, d in enumerate(self):
            key = (d.supercell_size, d.supercell_id)
            self._map_supercell_ids[key] = i
        return self

    def append(self, ds: Union[list, tuple, Derivatives, DerivativesSet]):
        """Append data of derivative structures."""
        if isinstance(ds, Derivatives):
            self.derivatives_set.append(ds)
        elif isinstance(ds, DerivativesSet):
            self.derivatives_set.extend(ds.derivatives_set)
        elif isinstance(ds, (list, tuple)):
            self.derivatives_set.extend(ds)
        return self

    @property
    def unitcell(self):
        """Return unitcell."""
        return self[0].lattice_unitcell.cell

    @property
    def n_labelings(self):
        """Return list of numbers of labelings."""
        return np.array([d.n_labelings for d in self])

    def all(self):
        """Sample all labelings for each HNF."""
        self._sample = [d.all() for d in self]
        return self._sample

    def uniform(self, n_samples: int = 100):
        """Sample labelings randomly from uniformly-sampled HNFs."""
        if n_samples >= np.sum(self.n_labelings):
            self._samples = self.all()
            return self._sample

        n_samples_cell = self._choose_cell(n_samples=n_samples)
        self._sample = [d.random(n_samples=n) for d, n in zip(self, n_samples_cell)]
        return self._sample

    def random(self, n_samples: int = 100):
        """Sample labeling randomly."""
        if n_samples >= np.sum(self.n_labelings):
            self._samples = self.all()
            return self._sample

        n_samples_cell = self._choose_cell(n_samples=n_samples, prob=self.n_labelings)
        self._sample = [d.random(n_samples=n) for d, n in zip(self, n_samples_cell)]
        return self._sample

    def select(self, key: tuple):
        """Sample single labeling from IDs.

        Parameter
        ---------
        key: Tuple of IDs, (supercell_size, supercell_id, labeling_id).
        """
        if len(self._map_supercell_ids) != len(self.derivatives_set):
            self._set_map_supercell_ids()

        supercell_size, supercell_id, labeling_id = key
        iset = self._map_supercell_ids[(supercell_size, supercell_id)]
        self[iset].select(labeling_id)
        return self

    def _choose_cell(self, n_samples: int = 100, prob: np.ndarray = None):
        """Choose supercell id randomly."""
        n_labelings = self.n_labelings
        n_samples_all = np.zeros(len(self), dtype=int)
        n_resamples = n_samples
        while n_resamples > 0:
            ids_full = np.where(n_samples_all >= n_labelings)[0]
            n_samples_all[ids_full] = n_labelings[ids_full]

            n_resamples = n_samples - np.sum(n_samples_all)
            ids = np.where(n_samples_all < n_labelings)[0]
            p = None if prob is None else prob[ids] / np.sum(prob[ids])
            icells = np.random.choice(ids, size=n_resamples, replace=True, p=p)
            key, cnt = np.unique(icells, return_counts=True)
            n_samples_all[key] += cnt
        return n_samples_all

    def save(self, element_strings: tuple, path: str = "poscars"):
        """Save derivative structures sampled."""
        if element_strings is None:
            element_strings = self[0].lattice_unitcell.element_strings

        os.makedirs(path, exist_ok=True)
        filename = "pyclupan_sample_attrs.yaml"
        with open(path + "/" + filename, "w") as f:
            d_rep = self[0]
            save_cell(d_rep.unitcell, tag="unitcell", file=f)

            print("element_strings:", list(element_strings), file=f)
            print("elements:", file=f)
            for ele in d_rep.lattice_unitcell.elements_on_lattice:
                print("-", ele, file=f)
            print(file=f)

            print("sampled_labelings:", file=f)
            for d in self:
                d.write_attrs(file=f)
                d.save(element_strings, path=path)
        return self


def _write_list_no_space(a: list, file):
    """Write list without spaces.."""
    print("[", end="", file=file)
    print(*list(a), sep=",", end="]\n", file=file)


def write_derivatives_yaml(
    derivs_set: DerivativesSet,
    zdd: Any,
    filename: str = "derivatives.yaml",
):
    """Save labelings of derivative structures to yaml file."""
    if len(derivs_set) == 0:
        return None

    with open(filename, "w") as f:
        derivs = derivs_set[0]
        save_cell(derivs.unitcell, tag="unitcell", file=f)
        print("zdd:", file=f)
        print("  n_cells: ", derivs.supercell_size, file=f)
        print("  one_of_k:", zdd.zdd_lattice.one_of_k_rep, file=f)
        print("  element_sets:", file=f)
        for i, ele in enumerate(zdd.zdd_lattice.get_element_orbit(dd=True)):
            print("  - id:         ", i, file=f)
            print("    elements:   ", ele[0], file=f)
            print("    elements_dd:", ele[1], file=f)
        print("", file=f)

        print("compositions:", file=f)
        print("  comp:   ", list(derivs.comp), file=f)
        print("  comp_lb:", list(derivs.comp_lb), file=f)
        print("  comp_ub:", list(derivs.comp_ub), file=f)
        print("", file=f)

        print("derivative_structures:", file=f)
        for i, derivs in enumerate(derivs_set):
            print("- id:", i, file=f)
            print("  HNF:", file=f)
            print("  -", list(derivs.supercell_matrix[0]), file=f)
            print("  -", list(derivs.supercell_matrix[1]), file=f)
            print("  -", list(derivs.supercell_matrix[2]), file=f)

            print("  inactive_sites:   ", end=" ", file=f)
            _write_list_no_space(derivs.inactive_sites, file=f)
            print("  inactive_labeling:", end=" ", file=f)
            _write_list_no_space(derivs.inactive_labeling, file=f)
            print("  active_sites:     ", end=" ", file=f)
            _write_list_no_space(derivs.active_sites, file=f)
            print("  n_labelings:      ", derivs.active_labelings.shape[0], file=f)
            print("", file=f)
            print("  active_labelings:", file=f)
            for l in derivs.active_labelings:
                print("  - ", end="", file=f)
                _write_list_no_space(l, file=f)
            print("", file=f)
    return filename


def load_derivatives_yaml(filename: str = "pyclupan_derivatives.yaml"):
    """Load labelings of derivative structures."""
    data = yaml.safe_load(open(filename))
    unitcell = load_cell(yaml_data=data, tag="unitcell")
    elements_lattice = [d["elements"] for d in data["zdd"]["element_sets"]]
    lattice = Lattice(unitcell, elements=elements_lattice)

    derivs_all = []
    for d in data["derivative_structures"]:
        ids = [
            get_structure_id(data["zdd"]["n_cells"], d["id"], i)
            for i in range(d["n_labelings"])
        ]
        derivs = Derivatives(
            lattice_unitcell=lattice,
            supercell_matrix=np.array(d["HNF"]),
            supercell_id=int(d["id"]),
            active_labelings=np.array(d["active_labelings"]),
            inactive_labeling=np.array(d["inactive_labeling"]),
            structure_ids=ids,
        )
        derivs_all.append(derivs)

    return DerivativesSet(derivs_all)


def load_sample_attrs_yaml(filename: str = "pyclupan_sample_attrs.yaml"):
    """Load sampled labelings of derivative structures."""
    data = yaml.safe_load(open(filename))
    unitcell = load_cell(yaml_data=data, tag="unitcell")
    elements_lattice = data["elements"]
    lattice = Lattice(unitcell, elements=elements_lattice)

    derivs_all = []
    for d in data["sampled_labelings"]:
        derivs = Derivatives(
            lattice_unitcell=lattice,
            supercell_matrix=np.array(d["supercell_matrix"]),
            active_labelings=np.array(d["active_labelings"]),
            inactive_labeling=np.array(d["inactive_labeling"]),
            structure_ids=d["id"],
        )
        derivs_all.append(derivs)

    return DerivativesSet(derivs_all)
