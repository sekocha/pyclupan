"""Utility functions for pyclupan."""

from fractions import Fraction
from typing import Optional

import numpy as np


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
