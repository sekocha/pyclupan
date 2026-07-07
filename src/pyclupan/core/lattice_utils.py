"""Utility functions for lattice."""

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


def set_labelings_endmembers(elements_on_lattice: list):
    """Set active labelings of endmembers automatically."""
    active_elements_on_lattice = [ele for ele in elements_on_lattice if len(ele) > 1]
    if len(active_elements_on_lattice) > 1:
        raise RuntimeError("Automatic setting of endmembers has been failed.")

    labelings_endmembers = [[e] for e in active_elements_on_lattice[0]]
    return labelings_endmembers


def set_element_strings(
    cell: PolymlpStructure,
    elements_on_lattice: list,
    n_elements: int,
):
    """Set element strings from lattice."""

    idx = 0
    elements = []
    for n in cell.n_atoms:
        elements.append(cell.elements[idx])
        idx += n

    element_strings = [None for i in range(n_elements)]
    for ele, labels in zip(elements, elements_on_lattice):
        if len(labels) > 1:
            for l in labels:
                element_strings[l] = ele + str(l)
        else:
            element_strings[labels[0]] = ele
    return element_strings


def get_complete_labelings(
    active_labelings: np.ndarray,
    inactive_labeling: np.ndarray,
    active_sites: np.ndarray,
    inactive_sites: np.ndarray,
):
    """Return complete labelings from both active and inactive labelings."""
    if active_labelings.shape[1] != len(active_sites):
        raise RuntimeError("Given shape of active_labelings is not consistent.")

    n_site = active_labelings.shape[1] + len(inactive_labeling)
    n_labelings = active_labelings.shape[0]
    labelings = np.zeros((n_labelings, n_site), dtype=np.uint8)
    labelings[:, active_sites] = active_labelings
    if len(inactive_sites) > 0:
        labelings[:, inactive_sites] = inactive_labeling
    return labelings


def extract_sites(cell: PolymlpStructure, lattice_ids: np.ndarray):
    """Extract sites from the entire sites."""
    n_sites = cell.n_atoms
    extsites = []
    for lattice_id in lattice_ids:
        begin = sum(n_sites[:lattice_id])
        extsites.extend(list(range(begin, begin + n_sites[lattice_id])))
    return np.array(extsites)


def get_inactive_labeling(
    cell: PolymlpStructure,
    elements_on_lattice: list,
    inactive_lattice: np.ndarray,
):
    """Return inactive labeling."""
    n_sites = cell.n_atoms
    inactive_labeling = []
    for lattice_id in inactive_lattice:
        ele = elements_on_lattice[lattice_id][0]
        inactive_labeling.extend([ele for j in range(n_sites[lattice_id])])
    return np.array(inactive_labeling)


def is_active_size(labelings: np.ndarray, active_sites: np.ndarray):
    """Check if size of labelings (elements, spin) is consistent with active sites."""
    labelings = np.array(labelings)
    if labelings.ndim == 2:
        if labelings.shape[1] != active_sites.shape[0]:
            return False
        return True

    if labelings.shape[0] != active_sites.shape[0]:
        return False
    return True


def map_active_array(
    values: np.ndarray,
    active_sites: np.ndarray,
    cell: PolymlpStructure,
    source: list,
    target: list,
):
    """Convert labelings to spins or spins to labelings."""
    if not is_active_size(values, active_sites):
        raise RuntimeError("Size of given values not consistent with lattice.")

    values = np.array(values)
    assigned = np.zeros(values.shape, dtype=int)
    if values.ndim == 2:
        begin = 0
        for src, tar, n in zip(source, target, cell.n_atoms):
            if len(tar) > 1:
                end = begin + n
                for s, t in zip(src, tar):
                    assigned[:, begin:end][values[:, begin:end] == s] = t
                begin = end
    elif values.ndim == 1:
        begin = 0
        for src, tar, n in zip(source, target, cell.n_atoms):
            if len(tar) > 1:
                end = begin + n
                for s, t in zip(src, tar):
                    assigned[begin:end][values[begin:end] == s] = t
                begin = end
    return assigned
