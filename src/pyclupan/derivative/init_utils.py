"""Functions for initializing derivative structure enumeration."""

from collections import defaultdict
from fractions import Fraction
from typing import Optional

import numpy as np
from scipy.sparse.csgraph import connected_components


def _check_composition_bounds(comps: list):
    """Check if composition bounds are included between 0 and 1."""
    for c in comps:
        if c is not None and (c > 1.0 + 1e-8 or c < -1e-8):
            raise RuntimeError("0 <= comp (bound) <= 1 must be satisfied.")


def set_compositions(
    elements_lattice: Optional[list] = None,
    n_sites_supercell: Optional[list] = None,
    comp: Optional[list] = None,
    comp_lb: Optional[list] = None,
    comp_ub: Optional[list] = None,
):
    """Set compositions from input parameters.

    Parameters
    ----------
    elements_lattice : Element IDs on lattices.
                       Example: [[0],[1],[2, 3]].
    n_sites_supercell: Number of sites for each sublattice in supercell.
    comp: Compositions for sublattices (n_elements / n_sites).
          Compositions are not needed to be normalized.
    comp_lb: Lower bounds of compositions for sublattices.
    comp_ub: Upper bounds of compositions for sublattices.
    """
    n_elements = max([e2 for e1 in elements_lattice for e2 in e1]) + 1
    comp = normalize_compositions(comp, n_elements, elements_lattice, n_sites_supercell)
    comp_lb = normalize_compositions(
        comp_lb, n_elements, elements_lattice, n_sites_supercell, for_bound=True
    )
    comp_ub = normalize_compositions(
        comp_ub, n_elements, elements_lattice, n_sites_supercell, for_bound=True
    )
    return (comp, comp_lb, comp_ub)


def _find_element_group(n_elements: int, elements_lattice: list):
    """Find groups of elements sharing sublattices."""
    adj = np.zeros((n_elements, n_elements), dtype=int)
    for elements in elements_lattice:
        for e1 in elements:
            for e2 in elements:
                adj[e1, e2] = 1
    _, labels = connected_components(adj, directed=False, return_labels=True)
    ele_group = defaultdict(list)
    for e, l in enumerate(labels):
        ele_group[l].append(e)
    ele_group = [g for g in ele_group.values()]

    lattices_group = []
    for elements in ele_group:
        lattices = set()
        for e in elements:
            for lattice_id, elements2 in enumerate(elements_lattice):
                if e in elements2:
                    lattices.add(lattice_id)
        lattices_group.append(list(lattices))

    return ele_group, lattices_group


def normalize_compositions(
    comp_in: list,
    n_elements: int,
    elements_lattice: list,
    n_sites_supercell: list,
    for_bound: bool = False,
):
    """Normalize compositions."""
    comp = [None for i in range(n_elements)]
    if comp_in is None:
        return comp

    for comp_pair in comp_in:
        if len(comp_pair) != 2:
            raise RuntimeError(
                "Composition must be given in the format (element ID, composition)."
            )

    for ele, c in comp_in:
        if int(ele) >= n_elements:
            raise RuntimeError("Element type must satisfy 0 <= ID < n_element.")
        ele, c = int(ele), float(Fraction(c))
        comp[ele] = c

    if for_bound:
        _check_composition_bounds(comp)
        return comp

    n_cand_sites = np.zeros(n_elements, dtype=int)
    for i, elements in enumerate(elements_lattice):
        for e in elements:
            n_cand_sites[e] += n_sites_supercell[i]

    ele_group, lattices_group = _find_element_group(n_elements, elements_lattice)
    comp_sites = np.array(comp)
    for elements, lattices in zip(ele_group, lattices_group):
        target_comp = comp_sites[elements]
        if list(target_comp).count(None) != len(target_comp):
            for c, ele in zip(target_comp, elements):
                if c is None:
                    raise RuntimeError("Composition not found for element", ele)
            target_comp = target_comp / np.sum(target_comp)
            n_sites_group = np.sum([n_sites_supercell[i] for i in lattices])
            n_sites = (target_comp * n_sites_group).astype(np.float64)
            if not np.allclose(np.rint(n_sites) - n_sites, 0.0):
                raise RuntimeError("Compositions are compatible with lattice sites.")

            comp_sites[elements] = n_sites / n_cand_sites[elements]

    return list(comp_sites)


def set_charges(charges_in: list, elements_lattice: list):
    """Set charges."""
    if charges_in is None:
        return None

    for charge_pair in charges_in:
        if len(charge_pair) != 2:
            raise RuntimeError(
                "Charge must be given in the format (element ID, charge)."
            )

    n_elements = max([e2 for e1 in elements_lattice for e2 in e1]) + 1
    charges = [None for i in range(n_elements)]
    for ele, c in charges_in:
        if int(ele) >= n_elements:
            raise RuntimeError("Element type must satisfy 0 <= ID < n_element.")
        ele, c = int(ele), float(Fraction(c))
        charges[ele] = c

    for i, c in enumerate(charges):
        if c is None:
            raise RuntimeError("Charge for element", i, "not found.")

    return charges
