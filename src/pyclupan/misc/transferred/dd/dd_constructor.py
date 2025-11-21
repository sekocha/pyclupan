#!/usr/bin/env python
import collections
import time
from math import *

import numpy as np
from graphillion import GraphSet

from pyclupan.dd.dd_combinations import DDCombinations
from pyclupan.dd.dd_node import DDNodeHandler


class DDConstructor:

    def __init__(self, handler: DDNodeHandler):

        self.handler = handler

        self.nodes = handler.get_nodes(active=True)
        self.nodes_single_rep = [i for i, j in self.nodes]

        self.elements_dd = handler.get_elements(active=True, dd=True)
        self.elements = handler.get_elements(active=True, dd=False)
        self.element_orbit = handler.get_element_orbit(dd=True)

        self.site_attr = handler.active_site_attr
        self.site_attr_dict = dict()
        for attr in self.site_attr:
            self.site_attr_dict[attr.idx] = attr

        GraphSet.set_universe(self.nodes)

    def all(self):
        gs = GraphSet({}).graphs()
        return gs

    def empty(self):
        gs = GraphSet({"exclude": set(self.nodes)})
        return gs

    def one_of_k(self):

        gs = self.empty()
        for site in self.site_attr:
            tnodes = self.handler.get_nodes(site=site.idx, active=True)
            gs1 = GraphSet({"exclude": set(self.nodes) - set(tnodes)})
            if site.one_of_k == True:
                gs1 = gs1.graphs(num_edges=1)
            else:
                gs1 = gs1.graphs().smaller(2)
            gs = gs.join(gs1)

        return gs

    def composition(self, comp, tol=1e-3):

        gs = self.empty()
        for ele in self.elements_dd:
            tnodes = self.handler.get_nodes(element=ele, active=True)
            gs1 = GraphSet({"exclude": set(self.nodes) - set(tnodes)})
            if comp[ele] is not None:
                val = len(tnodes) * comp[ele]
                if abs(round(val) - val) < tol:
                    n_edges = round(val)
                else:
                    n_edges = 100000
                gs1 = gs1.graphs(num_edges=n_edges)
            else:
                gs1 = gs1.graphs()
            gs = gs.join(gs1)

        return gs

    # a test is required
    def composition_range(self, comp_lb, comp_ub):

        print(
            " Warning: composition_range in dd.constructor.py"
            + " is being developed. Results must be carefully examined."
        )

        gs = self.empty()
        for ele in self.elements_dd:
            tnodes = self.handler.get_nodes(element=ele, active=True)
            gs1 = GraphSet({"exclude": set(self.nodes) - set(tnodes)}).graphs()
            if comp_lb[ele] is not None:
                lb = ceil(len(tnodes) * comp_lb[ele])
                gs1 = gs1.larger(lb - 1)
            if comp_ub[ele] is not None:
                ub = floor(len(tnodes) * comp_ub[ele])
                gs1 = gs1.smaller(ub + 1)
            gs = gs.join(gs1)

        return gs

    def no_endmembers(self, verbose=1):

        if verbose > 0:
            print(" element orbit used for eliminating end members")
            print("  =", self.element_orbit)

        gs = self.empty()
        for ele, ele_dd in self.element_orbit:
            gs1_all = GraphSet({"exclude": set(self.nodes)})
            for e in ele_dd:
                tnodes = self.handler.get_nodes(element=e, active=True)
                gs1 = GraphSet({"exclude": set(self.nodes) - set(tnodes)})
                gs1 = gs1.larger(0)
                gs1 = gs1.smaller(len(tnodes))
                gs1_all = gs1_all.join(gs1)

            n_hidden_ele = len(ele) - len(ele_dd)
            if n_hidden_ele > 0:
                sites = set()
                for e in ele_dd:
                    tnodes = self.handler.get_nodes(element=e, active=True)
                    for n in tnodes:
                        sites.add(self.handler.get_site(n[0]))
                n_sites = len(sites)
                gs1_all = gs1_all.smaller(n_sites + 1 - n_hidden_ele)

            gs = gs.join(gs1_all)

        return gs

    def nonequivalent_permutations(self, site_permutations, num_edges=None, gs=None):

        automorphism = []
        for p in site_permutations:
            auto1 = []
            for n_idx, _ in self.nodes:
                s_idx, e_idx = self.handler.decompose_node(n_idx)
                n_idx_perm = self.handler.compose_node(p[s_idx], e_idx)
                auto1.append(((n_idx, n_idx), (n_idx_perm, n_idx_perm)))
            automorphism.append(auto1)

        if gs is None:
            gs = GraphSet.graphs(permutations=automorphism, num_edges=num_edges)
        else:
            gs = gs.graphs(permutations=automorphism, num_edges=num_edges)

        return gs

    def enumerate_nonequiv_configs(
        self, site_permutations=None, comp=[None], comp_lb=[None], comp_ub=[None]
    ):

        gs = self.one_of_k()
        print(" number of structures (one-of-k)    =", gs.len())

        if comp.count(None) != len(comp):
            gs &= self.composition(comp)
            print(" number of structures (composition) =", gs.len())

        if comp_lb.count(None) != len(comp_lb) or comp_ub.count(None) != len(comp_ub):
            gs &= self.composition_range(comp_lb, comp_ub)
            print(" number of structures (composition) =", gs.len())

        if site_permutations is not None:
            t1 = time.time()
            gs = self.nonequivalent_permutations(site_permutations, gs=gs)
            t2 = time.time()
            print(" number of structures (nonequiv.)   =", gs.len())
            print(" elapsed time (nonequiv.)   =", t2 - t1)

        return gs

    def define_functions(self, node_idx):

        if node_idx in self.nodes_single_rep:
            f_include = True
            return ([node_idx], f_include)

        f_include = False
        site = self.handler.get_site(node_idx)
        ids = [
            self.handler.compose_node(site, e) for e in self.site_attr_dict[site].ele_dd
        ]
        return (ids, f_include)

    def including(self, node_idx):
        gs = self.all()
        gs = gs.including(node_idx)
        return gs

    def including_single_cluster(self, nodes, gs=None):

        if gs is None:
            gs = self.one_of_k()

        for n in nodes:
            ids, f_include = self.define_functions(n)
            if f_include == True:
                for i in ids:
                    gs &= gs.including(i)
            else:
                for i in ids:
                    gs = gs.excluding(i)
        return gs

    def excluding_single_cluster(self, nodes, gs=None):

        if gs is None:
            gs = self.one_of_k()

        gs -= self.including_single_cluster(nodes, gs=self.all())
        return gs

    def excluding_clusters(self, nodes_list, gs=None):

        if gs is None:
            gs = self.one_of_k()

        for nodes in nodes_list:
            gs &= self.excluding_single_cluster(nodes, gs=self.all())

        return gs

    def charge_balance(self, charge, gs=None, eps=1e-5):

        if gs is None:
            gs = self.one_of_k()

        charge_sum = 0.0
        inactive_nodes = self.handler.get_nodes(inactive=True, edge_rep=False)
        for node_idx in inactive_nodes:
            ele = self.handler.get_element(node_idx)
            charge_sum -= charge[ele]

        weight = []
        nodes_weight = sorted(self.nodes)
        for node_idx, _ in nodes_weight:
            ele = self.handler.get_element(node_idx)
            weight.append((node_idx, node_idx, charge[ele]))
        lconst = [(weight, (charge_sum - eps, charge_sum + eps))]

        gs = gs.graphs(linear_constraints=lconst)

        return gs

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

    #    # a test is required
    # slow ?
    def num_clusters_smaller(self, gs, cluster_nodes, n_clusters=1):

        if n_clusters < 1:
            gs = GraphSet().graphs()  # empty graphs
        elif n_clusters == 1:
            gs = self.excluding_cluster(gs, cluster_nodes)
        else:
            count = collections.Counter([tuple(n) for n in cluster_nodes])

            nodes, weight = [], []
            for k, v in count.items():
                nodes.append(k)
                weight.append(v)
            n_total_clusters = sum(weight)

            components = list(range(len(nodes)))
            comb_obj = DDCombinations(components, weight=weight)
            lb = n_total_clusters - n_clusters + 1
            combs = comb_obj.sum_weight(lb=lb)

            gs_array = []
            for k in count.keys():
                edges = [self.handler.get_edge_rep(n) for n in k]
                gs_array.append(gs.including(edges))

            gs0 = GraphSet().graphs()
            for comb in combs:
                gs1 = gs_array[comb[0]].copy()
                for c in comb[1:]:
                    gs1 |= gs_array[c]
                gs0 |= gs - gs1
            gs = gs0

        return gs
