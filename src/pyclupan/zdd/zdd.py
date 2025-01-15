"""Class for constructing ZDD satisfying various constraints."""

# import collections
from typing import Optional

import numpy as np
from graphillion import GraphSet

from pyclupan.zdd.zdd_base import ZddLattice

# from pyclupan.dd.dd_combinations import DDCombinations


class ZddCore:
    """Class for constructing ZDD satisfying various constraints."""

    def __init__(self, zdd_lattice: ZddLattice, verbose: bool = False):
        """Init method."""

        self._zdd_lattice = zdd_lattice
        self._verbose = verbose

        self._nodes = zdd_lattice.get_nodes(active=True)
        GraphSet.set_universe(self._nodes)

        self._site_attrs = zdd_lattice.active_site_attrs
        self._site_attrs_dict = dict()
        for attr in self._site_attrs:
            self._site_attrs_dict[attr.site_idx] = attr

        self._nodes_single_rep = [i for i, j in self._nodes]
        self._elements_dd = zdd_lattice.get_elements(active=True, dd=True)
        self._elements = zdd_lattice.get_elements(active=True, dd=False)
        self._element_orbit = zdd_lattice.get_element_orbit(dd=True)

    def all(self):
        """Return graph for all combinations."""
        gs = GraphSet({}).graphs()
        return gs

    def empty(self):
        """Return empty graph."""
        gs = GraphSet({"exclude": set(self._nodes)})
        return gs

    def one_of_k(self):
        """Apply one-of-k representations."""
        gs = self.empty()
        for site in self._site_attrs:
            tnodes = self._zdd_lattice.get_nodes(site=site.site_idx, active=True)
            gs1 = GraphSet({"exclude": set(self._nodes) - set(tnodes)})
            if site.one_of_k:
                gs1 = gs1.graphs(num_edges=1)
            else:
                gs1 = gs1.graphs().smaller(2)
            gs = gs.join(gs1)
        return gs

    def composition(self, comp: tuple, tol: float = 1e-3):
        """Apply composition."""
        gs = self.empty()
        for ele in self._elements_dd:
            tnodes = self._zdd_lattice.get_nodes(element=ele, active=True)
            gs1 = GraphSet({"exclude": set(self._nodes) - set(tnodes)})
            if comp[ele] is not None:
                val = len(tnodes) * comp[ele]
                if abs(round(val) - val) < tol:
                    n_edges = round(val)
                else:
                    n_edges = 10**7
                gs1 = gs1.graphs(num_edges=n_edges)
            else:
                gs1 = gs1.graphs()
            gs = gs.join(gs1)
        return gs

    def composition_range(self, comp_lb: tuple, comp_ub: tuple):
        """Apply composition lower and upper bounds."""
        # TODO: a test is required
        if self._verbose:
            print(
                "Warning: composition_range in dd.constructor.py",
                "is being developed. Results must be carefully examined.",
                flush=True,
            )

        gs = self.empty()
        for ele in self._elements_dd:
            tnodes = self._zdd_lattice.get_nodes(element=ele, active=True)
            gs1 = GraphSet({"exclude": set(self._nodes) - set(tnodes)}).graphs()
            if comp_lb[ele] is not None:
                lb = np.ceil(len(tnodes) * comp_lb[ele])
                gs1 = gs1.larger(lb - 1)
            if comp_ub[ele] is not None:
                ub = np.floor(len(tnodes) * comp_ub[ele])
                gs1 = gs1.smaller(ub + 1)
            gs = gs.join(gs1)
        return gs

    def no_endmembers(self):
        """Eliminate endmember structures."""
        if self._verbose:
            print("Orbits of elements used for eliminating end members:", flush=True)
            print(self._element_orbit)

        gs = self.empty()
        for ele, ele_dd in self._element_orbit:
            gs1_all = GraphSet({"exclude": set(self._nodes)})
            for e in ele_dd:
                tnodes = self._zdd_lattice.get_nodes(element=e, active=True)
                gs1 = GraphSet({"exclude": set(self._nodes) - set(tnodes)})
                gs1 = gs1.larger(0)
                gs1 = gs1.smaller(len(tnodes))
                gs1_all = gs1_all.join(gs1)

            n_hidden_ele = len(ele) - len(ele_dd)
            if n_hidden_ele > 0:
                sites = set()
                for e in ele_dd:
                    tnodes = self._zdd_lattice.get_nodes(element=e, active=True)
                    for n in tnodes:
                        sites.add(self._zdd_lattice.decompose_node_to_site(n[0]))
                n_sites = len(sites)
                gs1_all = gs1_all.smaller(n_sites + 1 - n_hidden_ele)

            gs = gs.join(gs1_all)
        return gs

    def nonequivalent_permutations(
        self,
        site_permutations: np.ndarray,
        num_edges: Optional[int] = None,
        gs: Optional[GraphSet] = None,
    ):
        """Return ZDD of non-equivalent configurations."""
        automorphism = []
        for p in site_permutations:
            auto1 = []
            for n_idx, _ in self._nodes:
                s_idx, e_idx = self._zdd_lattice.decompose_node(n_idx)
                n_idx_perm = self._zdd_lattice.compose_node(p[s_idx], e_idx)
                auto1.append(((n_idx, n_idx), (n_idx_perm, n_idx_perm)))
            automorphism.append(auto1)

        if gs is None:
            gs = GraphSet.graphs(permutations=automorphism, num_edges=num_edges)
        else:
            gs = gs.graphs(permutations=automorphism, num_edges=num_edges)

        return gs

    def define_functions(self, node_idx: int):
        """Get function type of inclusion or exclusion for a given node ID."""
        if node_idx in self.nodes_single_rep:
            f_include = True
            return ([node_idx], f_include)

        f_include = False
        site = self._zdd_lattice.decompose_node_to_site(node_idx)
        ids = [
            self._zdd_lattice.compose_node(site, e)
            for e in self._site_attrs_dict[site].ele_dd
        ]
        return (ids, f_include)

    def including(self, node_idx: int):
        """Return graph including a node."""
        gs = self.all()
        gs = gs.including(node_idx)
        return gs

    def including_single_cluster(self, nodes: list, gs: Optional[GraphSet] = None):
        """Return graph including a single cluster with nodes."""
        if gs is None:
            gs = self.one_of_k()

        for n in nodes:
            ids, f_include = self.define_functions(n)
            if f_include:
                for i in ids:
                    gs &= gs.including(i)
            else:
                for i in ids:
                    gs = gs.excluding(i)
        return gs

    def excluding_single_cluster(self, nodes: list, gs: Optional[GraphSet] = None):
        """Return graph excluding a single cluster with nodes."""
        if gs is None:
            gs = self.one_of_k()
        gs -= self.including_single_cluster(nodes, gs=self.all())
        return gs

    def excluding_clusters(self, nodes_list: list, gs: Optional[GraphSet] = None):
        """Return graph excluding clusters with node lists."""
        if gs is None:
            gs = self.one_of_k()

        for nodes in nodes_list:
            gs &= self.excluding_single_cluster(nodes, gs=self.all())
        return gs

    def charge_balance(
        self, charge: list, gs: Optional[GraphSet] = None, eps: float = 1e-5
    ):
        """Return graph for charge-balanced strucures."""
        if gs is None:
            gs = self.one_of_k()

        charge_sum = 0.0
        inactive_nodes = self._zdd_lattice.get_nodes(inactive=True, edge_rep=False)
        for node_idx in inactive_nodes:
            ele = self._zdd_lattice.decompose_node_to_element(node_idx)
            charge_sum -= charge[ele]

        weight = []
        nodes_weight = sorted(self._nodes)
        for node_idx, _ in nodes_weight:
            ele = self._zdd_lattice.decompose_node_to_element(node_idx)
            weight.append((node_idx, node_idx, charge[ele]))
        lconst = [(weight, (charge_sum - eps, charge_sum + eps))]

        gs = gs.graphs(linear_constraints=lconst)
        return gs

    # def num_clusters_smaller(self, gs, cluster_nodes, n_clusters=1):
    #     """Retrun graph with number of clusters smaller than given threshold."""
    #     # slow ?
    #     # a test is required
    #     if n_clusters < 1:
    #         gs = GraphSet().graphs()  # empty graphs
    #     elif n_clusters == 1:
    #         gs = self.excluding_cluster(gs, cluster_nodes)
    #     else:
    #         count = collections.Counter([tuple(n) for n in cluster_nodes])

    #         nodes, weight = [], []
    #         for k, v in count.items():
    #             nodes.append(k)
    #             weight.append(v)
    #         n_total_clusters = sum(weight)

    #         components = list(range(len(nodes)))
    #         comb_obj = DDCombinations(components, weight=weight)
    #         lb = n_total_clusters - n_clusters + 1
    #         combs = comb_obj.sum_weight(lb=lb)

    #         gs_array = []
    #         for k in count.keys():
    #             edges = [self.handler.get_edge_rep(n) for n in k]
    #             gs_array.append(gs.including(edges))

    #         gs0 = GraphSet().graphs()
    #         for comb in combs:
    #             gs1 = gs_array[comb[0]].copy()
    #             for c in comb[1:]:
    #                 gs1 |= gs_array[c]
    #             gs0 |= gs - gs1
    #         gs = gs0

    #     return gs

    #    # bak
    #    def charge_balance(self, charge, comp=None, eps=1e-5):
    #
    #        gs = self.empty()
    #
    #        charge_sum = 0.0
    #        inactive_nodes = self.handler.get_nodes(inactive=True,edge_rep=False)
    #        for n_idx in inactive_nodes:
    #            ele = self.handler.get_element(n_idx)
    #            charge_sum -= charge[ele]
    #
    #        nodes_noweight = []
    #        if comp is not None:
    #            for ele in self.elements_dd:
    #                if comp[ele] is not None:
    #                    tnodes = self.handler.get_nodes(element=ele, active=True)
    #                    sites = [self.handler.get_site(n_idx)
    #                                for n_idx, _ in tnodes]
    #                    if len(sites) == len(set(sites)):
    #                        charge_sum -= charge[ele] * len(sites) * comp[ele]
    #                        nodes_noweight.extend([n for n in tnodes])
    #
    #                        gs1 = GraphSet({'exclude':set(self.nodes)-set(tnodes)})
    #                        val = len(tnodes) * comp[ele]
    #                        if abs(round(val) - val) < 1e-10:
    #                            n_edges = round(val)
    #                        else:
    #                            n_edges = 100000
    #                        gs1 = gs1.graphs(num_edges=n_edges)
    #                        gs = gs.join(gs1)
    #
    #        weight = []
    #        nodes_weight = sorted(set(self.nodes) - set(nodes_noweight))
    #        for n_idx, _ in nodes_weight:
    #            ele = self.handler.get_element(n_idx)
    #            weight.append((n_idx,n_idx,charge[ele]))
    #        lconst = [(weight, (charge_sum-eps, charge_sum+eps))]
    #
    #        gs1 = GraphSet({'exclude':nodes_noweight})
    #        gs1 = gs1.graphs(linear_constraints=lconst)
    #        gs = gs.join(gs1)
    #
    #        return gs
