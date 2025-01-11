"""Class and functions for enumerating derivative structures."""

from typing import Optional

import numpy as np
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import supercell

from pyclupan.core.normal_form import get_nonequivalent_hnf
from pyclupan.core.spglib_utils import get_permutation
from pyclupan.derivative.derivative_utils import (
    set_compositions,
    set_elements_on_sublattices,
)
from pyclupan.zdd.zdd import Zdd
from pyclupan.zdd.zdd_base import ZddLattice


def run_derivatives(
    unitcell: PolymlpStructure,
    occupation: Optional[list] = None,
    elements: Optional[list] = None,
    comp: Optional[list] = None,
    comp_lb: Optional[list] = None,
    comp_ub: Optional[list] = None,
    supercell_size: Optional[int] = None,
    hnf: Optional[np.ndarray] = None,
    one_of_k_rep: bool = False,
    verbose: bool = False,
):
    """Enumerate derivative structures.

    Parameters
    ----------
    unitcell: Lattice for unit cell.
    occupation: Lattice IDs occupied by elements.
                Example: [[0], [1], [2], [2]].
    elements: Element IDs on lattices.
              Example: [[0],[1],[2, 3]].
    comp: Compositions for sublattices (n_elements / n_sites).
          Compositions are not needed to be normalized.
          Format: [(element ID, composition), (element ID, compositions),...]
    comp_lb: Lower bounds of compositions for sublattices.
          Format: [(element ID, composition), (element ID, compositions),...]
    comp_ub: Upper bounds of compositions for sublattices.
          Format: [(element ID, composition), (element ID, compositions),...]
    supercell_size: Determinant of supercell matrices.
                Derivative structures for all nonequivalent HNFs are enumerated.
    hnf: Supercell matrix in Hermite normal form.
    """
    if supercell_size is None and hnf is None:
        raise RuntimeError("supercell_size or hnf required.")

    elements_lattice = set_elements_on_sublattices(
        n_sites=unitcell.n_atoms,
        occupation=occupation,
        elements=elements,
    )
    comp, comp_lb, comp_ub = set_compositions(
        elements_lattice=elements_lattice,
        comp=comp,
        comp_lb=comp_lb,
        comp_ub=comp_ub,
    )

    if hnf is None:
        hnf_all = get_nonequivalent_hnf(supercell_size, unitcell)
        n_sites = np.array(unitcell.n_atoms) * supercell_size
    else:
        hnf_all = [hnf]
        n_sites = np.array(unitcell.n_atoms) * round(np.linalg.det(hnf))

    if verbose:
        size = round(np.linalg.det(hnf_all[0]))
        print("Supercell size       :", size, flush=True)
        print("Number of unique HNFs:", len(hnf_all), flush=True)

    zdd_lattice = ZddLattice(
        n_sites=n_sites,
        elements_lattice=elements_lattice,
        one_of_k_rep=one_of_k_rep,
        verbose=verbose,
    )
    for hnf in hnf_all:
        enum_derivatives(
            zdd_lattice=zdd_lattice,
            unitcell=unitcell,
            hnf=hnf,
            comp=comp,
            comp_lb=comp_lb,
            comp_ub=comp_ub,
            verbose=verbose,
        )


def enum_derivatives(
    zdd_lattice: ZddLattice,
    unitcell: PolymlpStructure,
    hnf: np.array,
    comp: Optional[list] = None,
    comp_lb: Optional[list] = None,
    comp_ub: Optional[list] = None,
    verbose: bool = False,
):
    """Enumerate derivative structures for given HNF."""
    sup = supercell(unitcell, hnf)
    site_perm, site_perm_lt = get_permutation(sup, superperiodic=True, hnf=hnf)
    zdd = Zdd(zdd_lattice, verbose=verbose)

    _ = zdd.enumerate_nonequiv_configs(
        site_permutations=site_perm, comp=comp, comp_lb=comp_lb, comp_ub=comp_ub
    )
