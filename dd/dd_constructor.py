#!/usr/bin/env python
import numpy as np
import time
from math import *

from pyclupan.dd.dd_node import DDNodeHandler
from graphillion import GraphSet

class DDConstructor:

    def __init__(self, handler:DDNodeHandler):

        self.handler = handler

        self.nodes = handler.get_nodes(active=True)
        self.elements = handler.get_elements(active=True)
        self.site_attr = handler.active_site_attr

        GraphSet.set_universe(self.nodes)

    def all(self):
        gs = GraphSet({}).graphs()
        return gs

    def empty(self):
        gs = GraphSet({'exclude': set(self.nodes)})
        return gs

    def including(self, node_idx):
        gs = GraphSet({}).graphs().including(node_idx)
        return gs

    def one_of_k(self):

        gs = GraphSet({'exclude': set(self.nodes)})
        for site in self.site_attr:
            tnodes = self.handler.get_nodes(site=site.idx, active=True)
            gs1 = GraphSet({'exclude':set(self.nodes)-set(tnodes)})
            if site.one_of_k == True:
                gs1 = gs1.graphs(num_edges=1)
            else:
                gs1 = gs1.graphs().smaller(2)
            gs = gs.join(gs1)

        return gs

    def composition(self, comp, tol=1e-3):

        gs = GraphSet({'exclude': set(self.nodes)})
        for ele in self.elements:
            tnodes = self.handler.get_nodes(element=ele, active=True)
            gs1 = GraphSet({'exclude':set(self.nodes)-set(tnodes)})
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
    
    def composition_range(self, comp_lb, comp_ub):
    
        gs = GraphSet({'exclude': set(self.nodes)})
        for ele in self.elements:
            tnodes = self.handler.get_nodes(element=ele, active=True)
            gs1 = GraphSet({'exclude':set(self.nodes)-set(tnodes)}).graphs()
            if comp_lb[ele] is not None:
                lb = ceil(len(tnodes) * comp_lb[ele])
                gs1 = gs1.larger(lb-1)
            if comp_ub[ele] is not None:
                ub = floor(len(tnodes) * comp_ub[ele])
                gs1 = gs1.smaller(ub+1)
            gs = gs.join(gs1)
    
        return gs

    def charge_balance(self, charge, comp=None, eps=1e-5):

        gs = GraphSet({'exclude': set(self.nodes)})

        charge_sum = 0.0
        inactive_nodes = self.handler.get_nodes(inactive=True,edge_rep=False)
        for n_idx in inactive_nodes:
            ele = self.handler.get_element(n_idx)
            charge_sum -= charge[ele]

        nodes_noweight = []
        if comp is not None:
            for ele in self.elements:
                if comp[ele] is not None:
                    tnodes = self.handler.get_nodes(element=ele, active=True)
                    sites = [self.handler.get_site(n_idx) 
                                for n_idx, _ in tnodes]
                    if len(sites) == len(set(sites)):
                        charge_sum -= charge[ele] * len(sites) * comp[ele]
                        nodes_noweight.extend([n for n in tnodes])

                        gs1 = GraphSet({'exclude':set(self.nodes)-set(tnodes)})
                        val = len(tnodes) * comp[ele]
                        if abs(round(val) - val) < 1e-10:
                            n_edges = round(val)
                        else:
                            n_edges = 100000
                        gs1 = gs1.graphs(num_edges=n_edges)
                        gs = gs.join(gs1)
 
        weight = []
        nodes_weight = sorted(set(self.nodes) - set(nodes_noweight))
        for n_idx, _ in nodes_weight:
            ele = self.handler.get_element(n_idx)
            weight.append((n_idx,n_idx,charge[ele]))
        lconst = [(weight, (charge_sum-eps, charge_sum+eps))]

        gs1 = GraphSet({'exclude':nodes_noweight})
        gs1 = gs1.graphs(linear_constraints=lconst)
        gs = gs.join(gs1)

        return gs
 
    def nonequivalent_permutations(self, 
                                   site_permutations,
                                   num_edges=None):
        
        automorphism = []
        for p in site_permutations:
            auto1 = []    
            for n_idx, _ in self.nodes:
                s_idx, e_idx = self.handler.decompose_node(n_idx)
                n_idx_perm = self.handler.compose_node(p[s_idx], e_idx) 
                auto1.append(((n_idx, n_idx), (n_idx_perm, n_idx_perm)))
            automorphism.append(auto1)

        gs = GraphSet.graphs(permutations=automorphism, 
                             num_edges=num_edges)

        return gs

    def enumerate_nonequiv_configs(self, 
                                   site_permutations=None,
                                   comp=[None], 
                                   comp_lb=[None], 
                                   comp_ub=[None]):

        gs = self.one_of_k()
        print(' number of structures (one-of-k)    =', gs.len())

        if comp.count(None) != len(comp):
            gs &= self.composition(comp)
            print(' number of structures (composition) =', gs.len())

        if comp_lb.count(None) != len(comp_lb) \
            or comp_ub.count(None) != len(comp_ub):
            gs &= self.composition_range(comp_lb, comp_ub)
            print(' number of structures (composition) =', gs.len())

        if site_permutations is not None:
            t1 = time.time()
            gs &= self.nonequivalent_permutations(site_permutations)
            t2 = time.time()
            print(' number of structures (nonequiv.)   =', gs.len())
            print(' elapsed time (nonequiv.)   =', t2-t1)

        return gs


