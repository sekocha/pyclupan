"""Class for handling ZDD."""

import itertools
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


def _compose_node(site_idx: int, element_idx: int):
    """Return node ID from site and element IDs."""
    return int(element_idx * 1000 + site_idx)


def _decompose_node(node_idx: int):
    """Return site and element IDs from node ID."""
    site_idx = _decompose_node_to_site(node_idx)
    element_idx = _decompose_node_to_element(node_idx)
    return site_idx, element_idx


def _decompose_node_to_site(node_idx: int):
    """Return site ID from node ID."""
    site_idx = node_idx % 1000
    return site_idx


def _decompose_node_to_element(node_idx: int):
    """Return element ID from node ID."""
    element_idx = node_idx // 1000
    return element_idx


@dataclass
class ZDDSite:
    """Dataclass for site attributes.

    Parameters
    ----------
    idx:
    ele:
    ele_dd:
    one_of_k: Use one-of-k representation.
    """

    site_idx: int
    ele: list[int]
    ele_dd: list[int]
    one_of_k: bool = False
    active: bool = False

    def __post_init__(self):
        """Post init method."""
        if len(self.ele) == len(self.ele_dd):
            self.one_of_k = True
        if len(self.ele_dd) > 0:
            self.active = True


@dataclass
class ZDDSiteSet:
    """Dataclass for a set of site attributes.

    Parameters
    ----------
    site_attrs: List of ZDDSite for the entire lattice.
    """

    n_sites: list
    elements_lattice: list
    one_of_k_rep: bool = False
    inactive_elements: Optional[list] = None

    site_attrs: Optional[list[ZDDSite]] = None
    active_site_attrs: Optional[list[ZDDSite]] = None

    nodes: Optional[list] = None
    active_nodes: Optional[list] = None
    inactive_nodes: Optional[list] = None
    sites: Optional[list] = None
    active_sites: Optional[list] = None
    elements: Optional[list] = None
    active_elements: Optional[list] = None
    active_elements_dd: Optional[list] = None

    def __post_init__(self):
        """Post init method."""
        self.n_total_sites = sum(self.n_sites)
        self.nodes = self._set_all_node_ids()
        self.site_attrs = self._set_site_attrs()
        self._set_propeties()

    def _set_all_node_ids(self):
        """Initialize zdd nodes using element-lattice mapping."""
        nodes = []
        begin = 0
        for l, elements in enumerate(self.elements_lattice):
            end = begin + self.n_sites[l]
            for ele_idx in elements:
                for site_idx in range(begin, end):
                    nodes.append(_compose_node(site_idx, ele_idx))
            begin = end
        nodes = sorted(nodes)
        return nodes

    def _set_site_attrs(self):
        """Initialize site attributes."""
        if not self.one_of_k_rep:
            elements_dd_exclude = self._set_excluding_elements_dd(self.elements_lattice)

        site_attr = []
        for s in range(self.n_total_sites):
            nodes = [i for i in self.nodes if _decompose_node_to_site(i) == s]
            ele = [_decompose_node_to_element(n) for n in nodes]

            ele_dd = sorted(set(ele) - set(self.inactive_elements))
            if not self.one_of_k_rep:
                ele_dd = sorted(set(ele) - set(elements_dd_exclude))

            site_attr.append(ZDDSite(s, ele, ele_dd))
        return site_attr

    def _set_excluding_elements_dd(self, elements_lattice):
        """Exclude ids of unnecessary elements when one_of_k = False."""
        elements_dd_exclude = []
        for ele1 in self.elements_lattice:
            common = False
            for ele2 in self.elements_lattice:
                if tuple(ele1) != tuple(ele2) and len(set(ele1) & set(ele2)) > 0:
                    common = True
                    break
            if not common:
                elements_dd_exclude.append(ele1[-1])
        return elements_dd_exclude

    def _set_propeties(self):
        """Initialize active nodes, sites, and elements."""
        self.active_site_attrs = [s for s in self.site_attrs if s.active]
        self.active_nodes = sorted(
            [_compose_node(s.site_idx, e) for s in self.site_attrs for e in s.ele_dd]
        )
        self.inactive_nodes = sorted(set(self.nodes) - set(self.active_nodes))

        self.sites = [s.site_idx for s in self.site_attrs]
        self.active_sites = sorted([s.site_idx for s in self.site_attrs if s.active])

        self.elements = sorted(set([e for s in self.site_attrs for e in s.ele]))
        self.active_elements = [e for s in self.site_attrs if s.active for e in s.ele]
        self.active_elements = sorted(set(self.active_elements))
        self.active_elements_dd = [e for s in self.site_attrs for e in s.ele_dd]
        self.active_elements_dd = sorted(set(self.active_elements_dd))
        return self

    def print_lattice(self):
        """Print lattice attributes."""
        print("Sites:         ", self.sites, flush=True)
        print("Elements:      ", self.elements, flush=True)
        print("Nodes:         ", self.nodes, flush=True)
        print("Lattice:", flush=True)
        for s, attr in enumerate(self.site_attrs):
            print("- site:        ", s, flush=True)
            print("  elements:    ", attr.ele, flush=True)
            print("  elements_zdd:", attr.ele_dd, flush=True)

        print("Active:", flush=True)
        print("- sites:       ", self.active_sites, flush=True)
        print("- elements:    ", self.active_elements, flush=True)
        print("- elements_zdd:", self.active_elements_dd, flush=True)
        print("- nodes:       ", self.active_nodes, flush=True)

        return self


class ZddLattice:
    """Class for handling ZDD."""

    def __init__(
        self,
        n_sites: list,
        elements_lattice: Optional[list] = None,
        min_n_elements: int = 2,
        one_of_k_rep: bool = False,
        inactive_elements: Optional[list] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        n_sites: Number of lattice sites for primitive cell.
        elements_lattice: Element IDs on lattices.
                          Example: [[0],[1],[2, 3]].

        TODO: other parameters
        """
        self._verbose = verbose
        self._n_sites = n_sites
        self._n_total_sites = sum(n_sites)

        self._min_n_elements = min_n_elements
        self._one_of_k_rep = one_of_k_rep
        # if self._min_n_elements == 1:
        #     self._one_of_k_rep = True

        if inactive_elements is None:
            inactive_elements = []

        self._site_set = ZDDSiteSet(
            n_sites=n_sites,
            elements_lattice=elements_lattice,
            one_of_k_rep=one_of_k_rep,
            inactive_elements=inactive_elements,
        )
        if self._verbose:
            self._site_set.print_lattice()

        self._n_elements = len(self._site_set.elements)
        self._element_orbit = self._set_element_orbit()

    def _set_element_orbit(self):
        """Initialize orbits of elements."""
        uniq_ele = set()
        for site in self._site_set.site_attrs:
            uniq_ele.add((tuple(site.ele), tuple(site.ele_dd)))

        element_orbit = self._find_element_orbit(uniq_ele)
        return element_orbit

    def _find_element_orbit(self, uniq_ele: set):
        """Find element orbit from unique elements."""
        uniq_ele = sorted(uniq_ele)
        orbit_id = list(range(len(uniq_ele)))
        for i, j in itertools.combinations(orbit_id, 2):
            ele1, _ = uniq_ele[i]
            ele2, _ = uniq_ele[j]
            intersect = set(ele1) & set(ele2)
            if len(intersect) > 0:
                min_id = min(orbit_id[i], orbit_id[j])
                orbit_id[i] = min_id
                orbit_id[j] = min_id

        element_orbit = []
        for i in sorted(set(orbit_id)):
            ele, ele_dd = set(), set()
            for idx in np.where(np.array(orbit_id) == i)[0]:
                ele |= set(uniq_ele[idx][0])
                ele_dd |= set(uniq_ele[idx][1])
            element_orbit.append([sorted(ele), sorted(ele_dd)])

        return element_orbit

    def get_sites(self, active: bool = True):
        """Return sites."""
        if active:
            return self._site_set.active_sites
        return self._site_set.sites

    def get_elements(self, active: bool = True, dd: bool = True):
        """Return elements."""
        if active and dd:
            return self._site_set.active_elements_dd
        elif active and not dd:
            return self._site_set.active_elements
        return self._site_set.elements

    def get_element_orbit(self, dd: bool = False):
        """Return orbits of elements."""
        if dd:
            return self._element_orbit
        return [ele for ele, _ in self._element_orbit]

    def get_edge_rep(self, node_idx: int):
        """Return edge representation of given node."""
        return (node_idx, node_idx)

    def get_nodes(
        self,
        edge_rep: bool = True,
        active: bool = False,
        inactive: bool = False,
        element: Optional[Union[list, int]] = None,
        site: Optional[Union[list, int]] = None,
    ):
        """Return node IDs."""
        if active:
            nodes_match = self._site_set.active_nodes
        elif inactive:
            nodes_match = self._site_set.inactive_nodes
        else:
            nodes_match = self._site_set.nodes

        if element is not None:
            if isinstance(element, list):
                nodes_match = [
                    i for i in nodes_match if _decompose_node_to_element(i) in element
                ]
            elif isinstance(element, int):
                nodes_match = [
                    i for i in nodes_match if _decompose_node_to_element(i) == element
                ]
            else:
                raise RuntimeError("element must be int or list")

        if site is not None:
            if isinstance(site, list):
                nodes_match = [
                    i for i in nodes_match if _decompose_node_to_site(i) in site
                ]
            elif isinstance(site, int):
                nodes_match = [
                    i for i in nodes_match if _decompose_node_to_site(i) == site
                ]
            else:
                raise RuntimeError("site must be int or list")

        if edge_rep:
            return [(i, i) for i in nodes_match]
        return nodes_match

    def convert_graphs_to_entire_labelings(self, graphs):

        labelings = np.zeros((len(graphs), self.n_total_sites), dtype=int)
        for n_idx in self.inactive_nodes:
            s_idx, e_idx = self.decompose_node(n_idx)
            labelings[:, s_idx] = e_idx

        smap, emap = dict(), dict()
        for n_idx in self.active_nodes:
            smap[n_idx], emap[n_idx] = self.decompose_node(n_idx)

        for i, graph in enumerate(graphs):
            for n_idx, _ in graph:
                labelings[i, smap[n_idx]] = emap[n_idx]

        return labelings

    def convert_graphs_to_labelings(self, graphs):

        inactive, active, active_no_dd = [], [], []
        inactive_labeling = []
        for site in self.site_attr:
            if len(site.ele) == 1:
                inactive.append(site.idx)
                inactive_labeling.append(site.ele[0])
            else:
                active.append(site.idx)
                if len(site.ele) != len(site.ele_dd):
                    e_idx = list(set(site.ele) - set(site.ele_dd))[0]
                    active_no_dd.append((site.idx, e_idx))

        smap, emap = dict(), dict()
        for n_idx in self.active_nodes:
            smap[n_idx], emap[n_idx] = self.decompose_node(n_idx)

        map_active_sites = dict()
        for i, s_idx in enumerate(active):
            map_active_sites[s_idx] = i

        labelings = np.zeros((len(graphs), len(active)), dtype=int)
        for s_idx, e_idx in active_no_dd:
            a_idx = map_active_sites[s_idx]
            labelings[:, a_idx] = e_idx

        for i, graph in enumerate(graphs):
            for n_idx, _ in graph:
                a_idx = map_active_sites[smap[n_idx]]
                labelings[i, a_idx] = emap[n_idx]

        return (labelings, inactive_labeling, active, inactive)

    def convert_to_orbit_dd(self, orbit):
        sites, ele = orbit
        orbit_node_rep = []
        for s1, e1 in zip(sites, ele):
            nodes = [self.compose_node(s2, e2) for s2, e2 in zip(s1, e1)]
            orbit_node_rep.append(tuple(sorted(nodes)))
        return orbit_node_rep

    def compose_node(self, site_idx: int, element_idx: int):
        """Return node ID from site and element IDs."""
        return _compose_node(site_idx, element_idx)

    def decompose_node(self, node_idx: int):
        """Return site and element IDs from node ID."""
        return _decompose_node(node_idx)

    def decompose_node_to_site(self, node_idx: int):
        """Return site ID from node ID."""
        return _decompose_node_to_site(node_idx)

    def decompose_node_to_element(self, node_idx: int):
        """Return element ID from node ID."""
        return _decompose_node_to_element(node_idx)

    @property
    def site_attrs(self):
        """Return site attributes of lattice."""
        return self._site_set.site_attrs

    @property
    def active_site_attrs(self):
        """Return active site attributes of lattice."""
        return self._site_set.active_site_attrs

    @property
    def n_sites(self):
        """Return number of sites on lattice."""
        return self._n_sites
