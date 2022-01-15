#!/usr/bin/env python
import numpy as np
import time

class Site:

    def __init__(self, idx, ele, ele_dd):

        self.idx = idx
        self.ele = ele
        self.ele_dd = ele_dd
        if len(ele) == len(ele_dd):
            self.one_of_k = True
        else:
            self.one_of_k = False

class DDNodeHandler:

    def __init__(self, 
                 n_sites=None, 
                 occupation=None,
                 elements_lattice=None,
                 min_n_elements=2,
                 one_of_k_rep=False,
                 inactive_elements=[]):

        self.n_total_sites = sum(n_sites)

        self.min_n_elements = min_n_elements
        self.one_of_k_rep = one_of_k_rep
        self.inactive_elements = inactive_elements
        if self.min_n_elements == 1:
            self.one_of_k_rep = True

        ############################################################### 
        #  initialization of self.nodes and related attributes

        if occupation is None and elements_lattice is None:
            self.n_elements = 2
            occupation = [[0],[0]]

        self.nodes = []
        if occupation is not None:
            self.n_elements = len(occupation)
            self.elements = list(range(self.n_elements))

            for ele_idx, occ1 in enumerate(occupation):
                for occ2 in occ1:
                    begin = sum(n_sites[:occ2])
                    end = begin + n_sites[occ2]
                    for site_idx in range(begin, end):
                        self.nodes.append(self.compose_node(site_idx, ele_idx))
                            
        elif elements_lattice is not None:
            if len(n_sites) != len(elements_lattice):
                raise ValueError\
                    ("len(elements_lattice) is not equal to len(n_sites)")

            print('Warning: elements_lattice implementation is in development')
            print('         setting comp and labeling (e.g. -e 0 -e 1 -e 5 4)')

            self.elements = sorted([e for elements in elements_lattice 
                                      for e in elements])
            self.n_elements = max(self.elements) + 1

            for l, elements in enumerate(elements_lattice):
                begin = sum(n_sites[:l])
                end = begin + n_sites[l]
                for ele_idx in elements:
                    for site_idx in range(begin, end):
                        self.nodes.append(self.compose_node(site_idx, ele_idx))

        self.nodes = sorted(self.nodes)

        self.site_attr, self.active_nodes = self.set_site_attr()
        self.active_site_attr = [site for site in self.site_attr 
                                      if len(site.ele_dd) > 0]
        self.inactive_nodes = sorted(set(self.nodes) - set(self.active_nodes))  

        self.sites = list(range(self.n_total_sites))
        self.active_sites = [s.idx for s in self.site_attr if len(s.ele_dd) > 0]
        self.active_elements = [e for s in self.site_attr for e in s.ele_dd]
        self.active_elements = sorted(set(self.active_elements))

        ###############################################################

    def set_site_attr(self):

        site_attr = []
        for s in range(self.n_total_sites):
            nodes = [i for i in self.nodes if self.get_site(i) == s]
            ele = [self.get_element(n) for n in nodes]

            if len(ele) >= self.min_n_elements:
                ele_dd = sorted(set(ele) - set(self.inactive_elements))
                if self.one_of_k_rep == False:
                    if len(ele_dd) == len(ele):
                        ele_dd = ele[:-1]
            else:
                ele_dd = []
            site = Site(s, ele, ele_dd)
            site_attr.append(site)
            print(' site', s, ': elements =', ele, ': elements(dd) =', ele_dd)

        active_nodes = []
        for site in site_attr:
            for e in site.ele_dd:
                node = self.compose_node(site.idx, e)
                active_nodes.append(node)

        return site_attr, sorted(active_nodes)

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
            return self.active_elements
        return self.elements

    def get_site(self, node_idx):
        site_idx = int(node_idx % 1000)
        return site_idx

    def get_sites(self, active=True):
        if active:
            return self.active_sites
        return self.sites

    def get_nodes(self, 
                  edge_rep=True, 
                  active=False,
                  inactive=False,
                  element=None, 
                  site=None):

        if active: 
            nodes_match = self.active_nodes
        elif inactive:
            nodes_match = self.inactive_nodes
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

    def convert_graphs_to_labelings(self, graphs):
        
        labelings = np.zeros((len(graphs), self.n_total_sites), dtype=int)
        for n_idx in self.inactive_nodes:
            s_idx, e_idx = self.decompose_node(n_idx)
            labelings[:,s_idx] = e_idx

        smap, emap = dict(), dict()
        for n_idx in self.active_nodes:
            smap[n_idx], emap[n_idx] = self.decompose_node(n_idx)

        for i, graph in enumerate(graphs):
            for n_idx, _ in graph:
                labelings[i,smap[n_idx]] = emap[n_idx]

        return labelings

