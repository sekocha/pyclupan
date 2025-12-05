"""Class and functions for enumerating derivative structures."""

import time
from typing import Optional

import numpy as np

from pyclupan.core.lattice import Lattice, get_complete_labelings
from pyclupan.core.linalg_utils import get_nonequivalent_hnf
from pyclupan.core.pypolymlp_utils import PolymlpStructure
from pyclupan.derivative.derivative_utils import Derivatives, DerivativesSet
from pyclupan.derivative.init_utils import set_charges, set_compositions
from pyclupan.derivative.labelings_utils import (
    eliminate_superperiodic_labelings,
    get_nonequivalent_labelings,
)
from pyclupan.zdd.pyclupan_zdd import PyclupanZdd


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
    superperiodic: bool = False,
    end_members: bool = False,
    charges: Optional[list] = None,
    filename: str = "derivatives.yaml",
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
    comp: Compositions for sublattices.
          Compositions are not needed to be normalized.
          Format: [(element ID, composition), (element ID, composition),...]
    comp_lb: Lower bounds of compositions for sublattices.
          Format: [(element ID, composition), (element ID, composition),...]
    comp_ub: Upper bounds of compositions for sublattices.
          Format: [(element ID, composition), (element ID, composition),...]
    supercell_size: Determinant of supercell matrices.
                Derivative structures for all nonequivalent HNFs are enumerated.
    hnf: Supercell matrix in Hermite normal form.
    superperiodic: Include superperiodic derivative structures.
    end_members: Include structures of end members.
    charges: Charges of elements.
          Format: [(element ID, charge), (element ID, charge),...]
    """
    if supercell_size is None and hnf is None:
        raise RuntimeError("supercell_size or hnf required.")
    if hnf is not None:
        supercell_size = round(np.linalg.det(hnf))

    lattice_unitcell = Lattice(unitcell, occupation=occupation, elements=elements)
    elements_lattice = lattice_unitcell.elements_on_lattice

    n_sites_supercell = [n * supercell_size for n in unitcell.n_atoms]
    comp, comp_lb, comp_ub = set_compositions(
        elements_lattice=elements_lattice,
        n_sites_supercell=n_sites_supercell,
        comp=comp,
        comp_lb=comp_lb,
        comp_ub=comp_ub,
    )

    charges = set_charges(charges, elements_lattice)
    if charges is not None:
        one_of_k_rep = True

    if hnf is None:
        hnf_all = get_nonequivalent_hnf(supercell_size, unitcell)
    else:
        hnf_all = [hnf]
    supercell_size = round(np.linalg.det(hnf_all[0]))
    if verbose:
        print("Supercell size       :", supercell_size, flush=True)
        print("Number of unique HNFs:", len(hnf_all), flush=True)

    zdd = PyclupanZdd(verbose=verbose)
    zdd.unitcell = unitcell
    zdd.initialize_zdd(
        supercell_size=supercell_size,
        elements_lattice=elements_lattice,
        one_of_k_rep=one_of_k_rep,
    )

    if verbose:
        print("Constructing ZDD for derivative structures", flush=True)

    derivs_all = []
    n_derivs = 0
    for supercell_id, hnf in enumerate(hnf_all):
        if verbose:
            print(
                "---------- supercell:", supercell_id + 1, "--------------", flush=True
            )
            print("HNF:", hnf[0], flush=True)
            print("    ", hnf[1], flush=True)
            print("    ", hnf[2], flush=True)
        labelings, inactive_labeling = enum_derivatives(
            zdd=zdd,
            hnf=hnf,
            comp=comp,
            comp_lb=comp_lb,
            comp_ub=comp_ub,
            end_members=end_members,
            charges=charges,
            verbose=verbose,
        )
        if labelings.shape[0] > 0:
            # Eliminate superperiodic labelings.
            if not superperiodic:
                # TODO: Refactor ZDD classes.
                site_perm_lt = zdd.site_permutations_lattice_translations
                active_sites = zdd.zdd_lattice.site_attrs_set.active_sites
                inactive_sites = zdd.zdd_lattice.site_attrs_set.inactive_sites
                complete_labelings = get_complete_labelings(
                    labelings,
                    inactive_labeling,
                    active_sites,
                    inactive_sites,
                )
                complete_labelings = eliminate_superperiodic_labelings(
                    complete_labelings,
                    site_perm_lt,
                )
                labelings = complete_labelings[:, active_sites]
                if verbose:
                    prefix = "n_str (superperiodic)     :"
                    print(prefix, complete_labelings.shape[0], flush=True)

            derivs = Derivatives(
                lattice_unitcell=lattice_unitcell,
                supercell_matrix=hnf,
                supercell_id=supercell_id,
                active_labelings=labelings,
                inactive_labeling=inactive_labeling,
                comp=comp,
                comp_lb=comp_lb,
                comp_ub=comp_ub,
            )
            n_derivs += derivs.n_labelings
            derivs_all.append(derivs)

    if verbose:
        print("Number of derivative structures:", n_derivs, flush=True)

    return DerivativesSet(derivs_all), zdd


def enum_derivatives(
    zdd: PyclupanZdd,
    hnf: np.array,
    comp: tuple = (None),
    comp_lb: tuple = (None),
    comp_ub: tuple = (None),
    end_members: bool = False,
    charges: Optional[list] = None,
    verbose: bool = False,
):
    """Return ZDD of non-equivalent configurations.

    Parameters
    ----------
    zdd: Initialized PyclupanZdd instance.
    hnf: Supercell matrix in Hermite normal form.
    comp: Compositions for sublattices (n_elements / n_sites).
          Compositions are not needed to be normalized.
          Format: [(element ID, composition), (element ID, compositions),...]
    comp_lb: Lower bounds of compositions for sublattices.
          Format: [(element ID, composition), (element ID, compositions),...]
    comp_ub: Upper bounds of compositions for sublattices.
          Format: [(element ID, composition), (element ID, compositions),...]
    charges: Charges for elements.
          Format: [(element ID, charge), (element ID, charge),...]
    """
    zdd.reset_zdd()
    zdd.set_permutations(hnf)

    gs = zdd.one_of_k()
    if verbose:
        print("n_str (one-of-k):           ", gs.len(), flush=True)

    # Apply compositions
    if comp.count(None) != len(comp):
        gs &= zdd.composition(comp)
        if verbose:
            print("Composition:                ", comp, flush=True)
            print("  Definition: n_element / n_possible_sites", flush=True)
            print("n_str (composition):        ", gs.len(), flush=True)

    # Apply composition ranges
    if comp_lb.count(None) != len(comp_lb) or comp_ub.count(None) != len(comp_ub):
        gs &= zdd.composition_range(comp_lb, comp_ub)
        if verbose:
            print("n_str (composition lb & ub):", gs.len(), flush=True)

    # Apply charge balance rule
    if charges is not None:
        gs &= zdd.charge_balance(charges, gs=gs)
        if verbose:
            print("Charges:                    ", charges, flush=True)
            print("n_str (charge balanced):    ", gs.len(), flush=True)

    # Apply symmetry operations
    try:
        if verbose:
            print("Using graphillion for enumerating nonequiv. structures", flush=True)
        t1 = time.time()
        gs = zdd.nonequivalent_permutations(gs=gs)
        t2 = time.time()
        # Eliminate end members
        # if not end_members:
        #     gs &= zdd.no_endmembers()
        labelings, inactive_labeling = zdd.to_labelings(gs)
    except:
        if verbose:
            print("Using labelings for enumerating nonequiv. structures", flush=True)
        t1 = time.time()
        labelings, inactive_labeling = zdd.to_labelings(gs)
        labelings = get_nonequivalent_labelings(labelings, zdd.site_permutations)
        t2 = time.time()

    if verbose:
        print("n_str (non-equivalent)     :", labelings.shape[0], flush=True)
        t_round = np.round(t2 - t1, 3)
        print("Elapsed_time (non-equiv.)  :", t_round, "(s)", flush=True)
    return labelings, inactive_labeling
