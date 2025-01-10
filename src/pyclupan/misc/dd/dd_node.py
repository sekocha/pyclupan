#!/usr/bin/env python
import numpy as np
import itertools

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
        self.inactive_elements = inactive_elements

        self.min_n_elements = min_n_elements
        self.one_of_k_rep = one_of_k_rep
        if self.min_n_elements == 1:
            self.one_of_k_rep = True

        if occupation is None and elements_lattice is None:
            elements_lattice = [[0, 1]]   # occupation = [[0],[0]]
        elif elements_lattice is not None:
            if len(n_sites) != len(elements_lattice):
                raise ValueError("len(elements_lattice) != len(n_sites)")
        elif occupation is not None:
            elements_lattice = self.convert_occupation_to_element(occupation)

        self.elements = sorted(set([e2 for e1 in elements_lattice 
                                    for e2 in e1]))
        self.n_elements = max(self.elements) + 1

        #  initialization of self.nodes and related attributes
        self.nodes = []
        for l, elements in enumerate(elements_lattice):
            begin = sum(n_sites[:l])
            end = begin + n_sites[l]
            for ele_idx in elements:
                for site_idx in range(begin, end):
                    self.nodes.append(self.compose_node(site_idx, ele_idx))
        self.nodes = sorted(self.nodes)

        self.site_attr, self.active_nodes, self.element_orbit \
                                = self.set_site_attr(elements_lattice)

        self.active_site_attr = [site for site in self.site_attr 
                                      if len(site.ele_dd) > 0]
        self.inactive_nodes = sorted(set(self.nodes) - set(self.active_nodes))  

        self.sites = list(range(self.n_total_sites))
        self.active_sites = [s.idx for s in self.site_attr if len(s.ele_dd) > 0]
        self.active_sites = sorted(self.active_sites)
        self.inactive_sites = sorted(set(self.sites) - set(self.active_sites))  
        self.active_elements_dd = [e for s in self.site_attr for e in s.ele_dd]
        self.active_elements_dd = sorted(set(self.active_elements_dd))
        self.active_elements = [e for s in self.site_attr 
                                if len(s.ele_dd) > 0 for e in s.ele]
        self.active_elements = sorted(set(self.active_elements))

    def convert_occupation_to_element(self, occupation):

        max_lattice_id = max([oc2 for oc1 in occupation for oc2 in oc1])
        elements_lattice = [[] for i in range(max_lattice_id + 1)]
        for e, oc1 in enumerate(occupation):
            for oc2 in oc1:
                elements_lattice[oc2].append(e)
        elements_lattice = [sorted(e1) for e1 in elements_lattice]

        return elements_lattice

    def set_excluding_elements_dd(self, elements_lattice):

        elements_dd_exclude = []
        for ele1 in elements_lattice:
            common = False
            for ele2 in elements_lattice:
                if tuple(ele1) != tuple(ele2) \
                    and len(set(ele1) & set(ele2)) > 0:
                    common = True
                    break
            if common == False:
                elements_dd_exclude.append(ele1[-1])
                
        return elements_dd_exclude

    def set_site_attr(self, elements_lattice=None):

        if self.one_of_k_rep == False:
            elements_dd_exclude \
                = self.set_excluding_elements_dd(elements_lattice)

        site_attr, active_nodes = [], []
        uniq_ele = set()
        for s in range(self.n_total_sites):
            nodes = [i for i in self.nodes if self.get_site(i) == s]
            ele = [self.get_element(n) for n in nodes]
            if self.one_of_k_rep == False:
                ele_dd = sorted(set(ele) - set(elements_dd_exclude))
            else:
                ele_dd = sorted(set(ele) - set(self.inactive_elements))

            site = Site(s, ele, ele_dd)
            site_attr.append(site)

            for e in ele_dd:
                node = self.compose_node(site.idx, e)
                active_nodes.append(node)

            uniq_ele.add((tuple(ele),tuple(ele_dd)))
            print(' site', s, ': elements =', ele, ': elements(dd) =', ele_dd)

        element_orbit = self.find_element_orbit(uniq_ele)

        return site_attr, sorted(active_nodes), element_orbit

    def find_element_orbit(self, uniq_ele):

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

    def compose_node(self, site_idx, element_idx):
        return int(element_idx * 1000 + site_idx)

    def decompose_node(self, node_idx):
        site_idx = self.get_site(node_idx)
        element_idx = self.get_element(node_idx)
        return site_idx, element_idx

    def get_element(self, node_idx):
        element_idx = int(node_idx / 1000)
        return element_idx

    def get_elements(self, active=True, dd=True):
        if active and dd:
            return self.active_elements_dd
        elif active and dd == False:
            return self.active_elements
        return self.elements

    def get_element_orbit(self, dd=False):
        if dd == True:
            return self.element_orbit
        return [ele for ele, _ in self.element_orbit]

    def get_site(self, node_idx):
        site_idx = int(node_idx % 1000)
        return site_idx

    def get_sites(self, active=True):
        if active:
            return self.active_sites
        return self.sites

    def get_edge_rep(self, node_idx):
        return (node_idx, node_idx)

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

    def convert_graphs_to_entire_labelings(self, graphs):
        
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

        labelings = np.zeros((len(graphs),len(active)), dtype=int)
        for s_idx, e_idx in active_no_dd:
            a_idx = map_active_sites[s_idx]
            labelings[:,a_idx] = e_idx

        for i, graph in enumerate(graphs):
            for n_idx, _ in graph:
                a_idx = map_active_sites[smap[n_idx]]
                labelings[i,a_idx] = emap[n_idx]

        return (labelings, inactive_labeling, active, inactive)

    def convert_to_orbit_dd(self, orbit):
        sites, ele = orbit
        orbit_node_rep = []
        for s1, e1 in zip(sites, ele):
            nodes = [self.compose_node(s2,e2) for s2,e2 in zip(s1,e1)]
            orbit_node_rep.append(tuple(sorted(nodes)))
        return orbit_node_rep

        
