#!/usr/bin/env python
import numpy as np
from math import *

from mlptools.common.structure import Structure
from pyclupan.common.symmetry import get_permutation
from pyclupan.dd.dd_supercell import DDSupercell

from graphillion import GraphSet

class DDEnumeration:

    def __init__(self, 
                 dd_sup:DDSupercell,
                 structure:Structure=None,
                 active=True):

        self.dd_sup = dd_sup
        self.st = structure
        self.one_of_k_rep = dd_sup.one_of_k_rep
        self.active = active

        nodes = dd_sup.get_nodes(active=active)
        GraphSet.set_universe(nodes)

    def one_of_k(self):

        nodes = self.dd_sup.get_nodes(active=self.active)
        sites = self.dd_sup.get_sites(active=self.active)

        if self.one_of_k_rep == True:
            gs = GraphSet({'exclude': set(nodes)})
            for s in sites:
                tnodes = self.dd_sup.get_nodes(site=s, active=self.active)
                gs1 = GraphSet({'exclude':set(nodes)-set(tnodes)})\
                               .graphs(num_edges=1)
                gs = gs.join(gs1)
        else:
            gs = GraphSet({'exclude': set(nodes)})
            for s in sites:
                tnodes = self.dd_sup.get_nodes(site=s, active=self.active)
                gs1 = GraphSet({'exclude':set(nodes)-set(tnodes)})
                gs1 = gs1.graphs().smaller(2)
                gs = gs.join(gs1)

        return gs

    def composition(self, comp):

        nodes = self.dd_sup.get_nodes(active=self.active)
        active_elments = self.dd_sup.get_elements(active=self.active)
    
        gs = GraphSet({'exclude': set(nodes)})
        for ele in active_elments:
            tnodes = self.dd_sup.get_nodes(element=ele, active=self.active)
            if comp[ele] is not None:
                val = len(tnodes) * comp[ele]
                if abs(round(val) - val) < 1e-10:
                    n_edges = round(val)
                else:
                    n_edges = 100000
                gs1 = GraphSet({'exclude':set(nodes)-set(tnodes)})\
                                .graphs(num_edges=n_edges)
            else:
                gs1 = GraphSet({'exclude':set(nodes)-set(tnodes)}).graphs()
            gs = gs.join(gs1)
    
        return gs
    
    def composition_range(self, comp_lb, comp_ub):
    
        nodes = self.dd_sup.get_nodes(active=self.active)
        active_elments = self.dd_sup.get_elements(active=self.active)
    
        gs = GraphSet({'exclude': set(nodes)})
        for ele in active_elments:
            tnodes = self.dd_sup.get_nodes(element=ele, active=self.active)
    
            if comp_lb[ele] is not None:
                lb = ceil(len(tnodes) * comp_lb[ele])
                gs1 = GraphSet({'exclude':set(nodes)-set(tnodes)})
                gs1 = gs1.larger(lb-1)
            else:
                gs1 = GraphSet({'exclude':set(nodes)-set(tnodes)}).graphs()
    
            if comp_ub[ele] is not None:
                ub = floor(len(tnodes) * comp_ub[ele])
                gs2 = GraphSet({'exclude':set(nodes)-set(tnodes)})
                gs2 = gs2.smaller(ub+1)
            else:
                gs2 = GraphSet({'exclude':set(nodes)-set(tnodes)}).graphs()
    
            gs1 &= gs2
            gs = gs.join(gs1)
    
        return gs

    def nonequivalent_permutations(self, structure=None):
        
        if structure is not None:
            self.st = structure

        nodes = self.dd_sup.get_nodes(active=self.active)
        perm = get_permutation(self.st)

        automorphism = []
        for p in perm:
            auto1 = []    
            for n_idx, _ in nodes:
                s_idx, e_idx = self.dd_sup.decompose_node(n_idx)
                n_idx_perm = self.dd_sup.compose_node(p[s_idx], e_idx) 
                auto1.append(((n_idx, n_idx), (n_idx_perm, n_idx_perm)))
            automorphism.append(auto1)

        gs = GraphSet.graphs(permutations=automorphism)
        return gs

