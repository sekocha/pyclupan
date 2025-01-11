"""Class and functions for enumerating derivative structures."""

import time
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
from pyclupan.zdd.zdd import ZddCore
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

    if verbose:
        print("Constructing ZDD for derivative structures", flush=True)

    n_derivs = 0
    for hnf in hnf_all:
        if verbose:
            print("HNF:", hnf[0], flush=True)
            print("    ", hnf[1], flush=True)
            print("    ", hnf[2], flush=True)

        gs = enum_derivatives(
            zdd_lattice=zdd_lattice,
            unitcell=unitcell,
            hnf=hnf,
            comp=comp,
            comp_lb=comp_lb,
            comp_ub=comp_ub,
            verbose=verbose,
        )
        n_derivs += gs.len()

    if verbose:
        print("Number of derivative structures:", n_derivs, flush=True)


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
    zdd = ZddCore(zdd_lattice, verbose=verbose)
    gs = enumerate_nonequiv_configs(
        zdd=zdd,
        site_permutations=site_perm,
        comp=comp,
        comp_lb=comp_lb,
        comp_ub=comp_ub,
        verbose=verbose,
    )
    return gs


def enumerate_nonequiv_configs(
    zdd: ZddCore,
    site_permutations: np.ndarray,
    comp: tuple = (None),
    comp_lb: tuple = (None),
    comp_ub: tuple = (None),
    verbose: bool = False,
):
    """Return ZDD of non-equivalent configurations."""
    gs = zdd.one_of_k()
    if verbose:
        print("n_str (one-of-k)           :", gs.len(), flush=True)

    if comp.count(None) != len(comp):
        gs &= zdd.composition(comp)
        if verbose:
            print("Composition:                ", comp, flush=True)
            print("n_str (composition)        :", gs.len(), flush=True)

    if comp_lb.count(None) != len(comp_lb) or comp_ub.count(None) != len(comp_ub):
        gs &= zdd.composition_range(comp_lb, comp_ub)
        if verbose:
            print("n_str (composition lb & ub):", gs.len(), flush=True)

    if site_permutations is not None:
        t1 = time.time()
        try:
            gs = zdd.nonequivalent_permutations(site_permutations, gs=gs)
        except:
            # TODO: Case of no automorphism option in graphillion.
            pass
        t2 = time.time()

        if verbose:
            print("n_str (non-equivalent)     :", gs.len(), flush=True)
            print(
                "Elapsed_time (non-equiv.)  :",
                np.round(t2 - t1, 3),
                "(s)",
                flush=True,
            )
    # gs &= zdd.no_endmembers()
    return gs
