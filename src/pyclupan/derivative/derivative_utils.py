"""Utility functions for pyclupan."""

from fractions import Fraction
from typing import Optional


def set_compositions(
    occupation: Optional[list] = None,
    elements: Optional[list] = None,
    comp: Optional[list] = None,
    comp_lb: Optional[list] = None,
    comp_ub: Optional[list] = None,
):
    """Set compositions from input parameters.

    Parameters
    ----------
    occupation: Lattice IDs occupied by elements.
                Example: [[0], [1], [2], [2]].
    elements: Element IDs on lattices.
              Example: [[0],[1],[2, 3]].
    comp: Compositions for sublattices (n_elements / n_sites).
          Compositions are not needed to be normalized.
    comp_lb: Lower bounds of compositions for sublattices.
    comp_ub: Upper bounds of compositions for sublattices.
    """
    if occupation is None and elements is None:
        raise RuntimeError("occupation or elements required.")

    if occupation is not None:
        n_elements = len(occupation)
    elif elements is not None:
        n_elements = max([e2 for e1 in elements for e2 in e1]) + 1
    else:
        n_elements = 2

    comp = normalize_compositions(comp, n_elements)
    comp_lb = normalize_compositions(comp_lb, n_elements)
    comp_ub = normalize_compositions(comp_ub, n_elements)
    return (comp, comp_lb, comp_ub)


def normalize_compositions(comp_in: list, n_elements: int):
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
    return comp
