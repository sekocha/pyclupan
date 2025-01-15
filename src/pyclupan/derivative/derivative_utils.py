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


#    def get_id_list(self):
#        n_labelings = self.active_labelings.shape[0]
#        return itertools.product(self.supercell_idset, range(n_labelings))
#
#    def get_supercell_from_id(self, supercell_id):
#        idx = self.supercell_map[supercell_id]
#        return self.supercell_set[idx]
#
#    def get_hnf_from_id(self, supercell_id):
#        idx = self.supercell_map[supercell_id]
#        return self.hnf_set[idx]
#
#    def sample(self, supercell_id, labeling_id, return_labeling=False):
#        labeling = self.all_labelings[labeling_id]
#        if return_labeling == True:
#            return self.get_supercell_from_id(supercell_id), labeling
#
#        order = np.argsort(labeling)
#        n_atoms = [np.sum(labeling == l) for l in self.elements]
#        return (order, n_atoms)
#
#    def sample_all(self, return_labeling=False):
#        labelings = self.all_labelings
#        if return_labeling == True:
#            return self.supercell_set, self.supercell_idset, labelings
#
#        order_all = np.argsort(labelings, axis=1)
#        n_atoms_all = np.array(
#            [np.sum(labelings == l, axis=1) for l in self.elements]
#        ).T
#        return zip(order_all, n_atoms_all)


# class DSSample:
#
#     def __init__(self, ds_set_all):
#
#         self.ds_set_all = ds_set_all
#         self.n_cell_all = [ds_set.n_expand for ds_set in self.ds_set_all]
#         self.element_orbit = ds_set_all[0].element_orbit
#
#         self.serial_id_list = []
#         for group_id, ds_set in enumerate(self.ds_set_all):
#             for sup_id, lab_id in ds_set.get_id_list():
#                 self.serial_id_list.append((group_id, sup_id, lab_id))
#
#         self.map_to_gid = dict()
#         for g_id, ds_set in enumerate(self.ds_set_all):
#             n_cell = self.n_cell_all[g_id]
#             for sup_id in ds_set.supercell_idset:
#                 self.map_to_gid[(n_cell, sup_id)] = g_id
#
#     def sample_single(self, g_id, s_id, l_id):
#
#         n_cell = self.n_cell_all[g_id]
#         attr = self.ds_set_all[g_id].sample(s_id, l_id)
#         st_attr_all = [attr]
#         id_all = [(n_cell, g_id, s_id, l_id)]
#         return st_attr_all, id_all
#
#     def sample_random(self, k=5, element_symbols=None):
#
#         st_attr_all, id_all = [], []
#         samples = random.sample(self.serial_id_list, k=k)
#         for g_id, s_id, l_id in sorted(samples):
#             n_cell = self.n_cell_all[g_id]
#             attr = self.ds_set_all[g_id].sample(s_id, l_id)
#             st_attr_all.append(attr)
#             id_all.append((n_cell, g_id, s_id, l_id))
#         return st_attr_all, id_all
#
#     def sample_all(self, n_cell_ub=None, element_symbols=None):
#
#         if n_cell_ub is None:
#             n_cell_ub = np.inf
#
#         st_attr_all, id_all = [], []
#         for g_id, (ds_set, n_cell) in enumerate(zip(self.ds_set_all, self.n_cell_all)):
#             if n_cell <= n_cell_ub:
#                 attr = list(ds_set.sample_all())
#                 n_labelings = ds_set.n_labelings
#                 for s_id in ds_set.supercell_idset:
#                     ids = [(n_cell, g_id, s_id, l_id) for l_id in range(n_labelings)]
#                     st_attr_all.extend(attr)
#                     id_all.extend(ids)
#         return st_attr_all, id_all
#
#     def get_all_labelings(self, n_cell_ub=None):
#
#         if n_cell_ub is None:
#             n_cell_ub = np.inf
#
#         labelings_all = dict()
#         for g_id, (ds_set, n_cell) in enumerate(zip(self.ds_set_all, self.n_cell_all)):
#             if n_cell <= n_cell_ub:
#                 labelings = ds_set.all_labelings
#                 n_labelings = ds_set.n_labelings
#                 for s_id in ds_set.supercell_idset:
#                     labelings_all[(n_cell, s_id)] = labelings
#
#         return labelings_all
#
#     def get_labeling(self, n_cell, s_id, l_id):
#         g_id = self.map_to_gid[(n_cell, s_id)]
#         return self.ds_set_all[g_id].all_labelings[l_id]
#
#     def get_hnf(self, n_cell, s_id):
#         g_id = self.map_to_gid[(n_cell, s_id)]
#         return self.ds_set_all[g_id].get_hnf_from_id(s_id)
#
#     def get_supercell(self, n_cell, s_id):
#         g_id = self.map_to_gid[(n_cell, s_id)]
#         return self.ds_set_all[g_id].get_supercell_from_id(s_id)
#
#     def get_primitive_cell(self):
#         return self.ds_set_all[0].prim
#
#
