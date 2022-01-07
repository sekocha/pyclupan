#!/usr/bin/env python
import numpy as np

#class Site:
#
#    def __init__(self, substit_idx=None, position=None):
#
#        self.substit_idx = substit_idx
#        self.position = position

class DDSupercell:

    def __init__(self, 
                 axis=None, 
                 hnf=None, 
                 primitive_cell=None,
                 positions=None, 
                 n_sites=None, 
                 n_elements=2,
                 one_of_k_rep=False,
                 occupation=[[0],[0]]):

        self.axis = axis
        self.hnf = hnf
        self.primitive_cell = primitive_cell

        self.n_elements = n_elements
        self.n_total_sites = sum(n_sites)
        self.active_nodes = []

        self.one_of_k_rep = one_of_k_rep

        ############################################################### 
        #  initialization of self.nodes

        if len(occupation) != n_elements:
            raise ValueError(
                "length of occupation is not equal to n_elements")

        self.nodes = []
        for ele_idx, occ1 in enumerate(occupation):
            for occ2 in occ1:
                begin = sum(n_sites[:occ2])
                for site_idx in range(begin, begin+n_sites[occ2]):
                    self.nodes.append(self.compose_node(site_idx, ele_idx))

        ###############################################################

#        self.sites = []
#        pos_idx = 0
#        for n, idx in zip(n_sites, substit_indices):
#            for _ in range(n):
#                s = Site(substit_idx=idx, position=positions.T[pos_idx])
#                self.sites.append(s)
#                pos_idx += 1

    def compose_node(self, site_idx, element_idx):
        return int(element_idx * 1000 + site_idx)

    def decompose_node(self, node_idx):
        site_idx = self.get_site(node_idx)
        element_idx = self.get_element(node_idx)
        return site_idx, element_idx

    def get_element(self, node_idx):
        element_idx = int(node_idx / 1000)
        return element_idx

    def get_elements(self, active=True):
        if active:
            if len(self.active_nodes) == 0:
                self.set_active_nodes(self.nodes)
            return sorted(set([self.get_element(i) for i in self.active_nodes]))
        return list(range(self.n_elements))

    def get_site(self, node_idx):
        site_idx = int(node_idx % 1000)
        return site_idx

    def get_sites(self, active=True):
        if active:
            if len(self.active_nodes) == 0:
                self.set_active_nodes(self.nodes)
            return sorted(set([self.get_site(i) for i in self.active_nodes]))
        return sorted(set([self.get_site(i) for i in self.nodes]))

    def set_active_nodes(self, nodes_all):
        for s in range(self.n_total_sites):
            nodes = [i for i in nodes_all if self.get_site(i) == s]
            if len(nodes) > 1:
                if self.one_of_k_rep == False:
                    self.active_nodes.extend(nodes[:-1])
                else:
                    self.active_nodes.extend(nodes)
        self.active_nodes = sorted(self.active_nodes)
 
    def get_nodes(self, 
                  edge_rep=True, 
                  active=False,
                  element=None, 
                  site=None):

        if active: 
            if len(self.active_nodes) == 0:
                self.set_active_nodes(self.nodes)
            nodes_match = self.active_nodes
        else:
            nodes_match = self.nodes

        if element is not None:
            if isinstance(element, list):
                nodes_match = [i for i in nodes_match
                                 if self.get_element(i) in element]
            elif isinstance(element, int):
                nodes_match = [i for i in nodes_match
                                 if self.get_element(i) == element]
            else:
                raise ValueError("type(element) is not int or list")

        if site is not None:
            if isinstance(site, list):
                nodes_match = [i for i in nodes_match 
                                 if self.get_site(i) in site]
            elif isinstance(site, int):
                nodes_match = [i for i in nodes_match 
                                 if self.get_site(i) == site]
            else:
                raise ValueError("type(site) is not int or list")

        if edge_rep == True:
            return [(i, i) for i in nodes_match]
        return nodes_match

    def print_settings(self):
        for s in range(self.n_total_sites):
            nodes = [i for i in self.nodes if self.get_site(i) == s]
            print(' site', s, ': elements =', 
                   [self.get_element(n) for n in nodes])


