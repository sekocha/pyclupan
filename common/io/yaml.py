#!/usr/bin/env python
import numpy as np
import yaml

from mlptools.common.structure import Structure
from pyclupan.cluster.cluster import Cluster, ClusterSet
from pyclupan.derivative.derivative import DSSet

class Yaml:

    def __init__(self):
        pass

    def write_derivative_yaml(self, 
                              primitive_cell,
                              ds_set_all,
                              filename='derivative.yaml'):

        f = open(filename, 'w')

        print('#  Nonequivalent derivative structures (pyclupan)', file=f)
        print('', file=f)

        self._write_structure(primitive_cell, f, tag='primitive_cell')
        self._write_element_orbit(ds_set_all, f)
        self._write_ds(ds_set_all, f)

        f.close()

    def parse_derivative_yaml(self, filename='derivative.yaml'):

        data = yaml.safe_load(open(filename))
        prim = self._parse_structure(data, tag='primitive_cell')

        element_orbit = [d['elements'] for d in data['element_sets']]
        elements = sorted([e for ele in element_orbit for e in ele])

        ds_set_all = []
        for d in data['derivative_structures']:
            n_cell = d['n_cell']
            hnf_set, supercell_set, supercell_idset = [], [], []
            for cell in d['supercells']:
                hnf_set.append(np.array(cell['HNF']))
                sup = self._parse_structure(cell, tag='structure')
                supercell_set.append(sup)
                supercell_idset.append(cell['id'])

            active_sites = d['active_sites']
            inactive_sites = d['inactive_sites']
            inactive_labeling = d['inactive_labeling']
            active_labelings = np.array(d['active_labelings'])

            ds_set = DSSet(active_labelings=active_labelings,
                           inactive_labeling=inactive_labeling,
                           active_sites=active_sites,
                           inactive_sites=inactive_sites,
                           primitive_cell=prim,
                           n_expand=n_cell,
                           hnf_set=hnf_set,
                           elements=elements,
                           element_orbit=element_orbit,
                           supercell_set=supercell_set,
                           supercell_idset=supercell_idset)
            ds_set_all.append(ds_set)

        return ds_set_all

    def _write_element_orbit(self, ds_set_all, stream):

        element_orbit = ds_set_all[0].element_orbit
        if element_orbit is not None:
            print('element_sets:', file=stream)
            for i, ele in enumerate(element_orbit):
                print('- id:         ', i, file=stream)
                print('  elements:   ', ele, file=stream)
                print('', file=stream)

    def _write_ds(self, ds_set_all, stream):

        print('derivative_structures:', file=stream)
        for i, ds_set in enumerate(ds_set_all):
            print('- group:', i, file=stream)
            print('', file=stream)
            print('  n_cell:', ds_set.n_expand, file=stream)
            print('', file=stream)

            if ds_set.comp is not None:
                print('  comp:', list(ds_set.comp), file=stream)
                print('', file=stream)
            if ds_set.comp_lb is not None:
                print('  comp_lb:', list(ds_set.comp_lb), file=stream)
                print('', file=stream)
            if ds_set.comp_ub is not None:
                print('  comp_ub:', list(ds_set.comp_ub), file=stream)
                print('', file=stream)

            print('  supercells:', file=stream)
            for hnf, supercell, supercell_id in zip(ds_set.hnf_set,
                                                    ds_set.supercell_set,
                                                    ds_set.supercell_idset):
                self._write_ds_supercell(supercell_id, 
                                         hnf, 
                                         stream)
                self._write_structure(supercell, 
                                      stream, 
                                      tag='structure',
                                      indent=4)

            print('  inactive_sites:  ', end='', file=stream)
            self._write_list_no_space(ds_set.inactive_sites, stream)
            print('', file=stream)
            print('  inactive_labeling:  ', end='', file=stream)
            self._write_list_no_space(ds_set.inactive_labeling, stream)
            print('', file=stream)
            print('  active_sites:  ', end='', file=stream)
            self._write_list_no_space(ds_set.active_sites, stream)
            print('', file=stream)
            print('  n_labelings:', ds_set.active_labelings.shape[0], 
                  file=stream)
            print('', file=stream)
            print('  active_labelings:', file=stream)
            for l in ds_set.active_labelings:
                print('    -  ', end='', file=stream)
                self._write_list_no_space(l, stream)
            print('', file=stream)

    def _write_ds_supercell(self, idx, hnf, stream, indent=2):

        addspace = ' ' * 2
        print(addspace + '- id:    ', idx, file=stream)
        print(addspace + '  HNF:', file=stream)
        for row in hnf:
            print(addspace + '    -    ', list(row), file=stream)
        print('', file=stream)

    def write_clusters_yaml(self, 
                            primitive_cell,
                            cutoff,
                            cluster_set: ClusterSet, 
                            elements_lattice=None,
                            cluster_set_element: ClusterSet=None, 
                            filename='clusters.yaml'):

        f = open(filename, 'w')

        print('#  Nonequivalent clusters (pyclupan)', file=f)
        print('', file=f)

        print('cutoff:', file=f)
        for i, c in enumerate(cutoff):
            print('- ' + str(i+2) + '-body:   ', c, file=f)
            print('', file=f)

        if elements_lattice is not None:
            print('element_configs:', file=f)
            for i, elements in enumerate(elements_lattice):
                print('- lattice:    ', i, file=f)
                if len(elements) > 0:
                    print('  elements:   ', elements, file=f)
                else:
                    print('  elements:   ', [], file=f)
                print('', file=f)

        self._write_structure(primitive_cell, f, tag='primitive_cell')

        print('nonequiv_clusters:', file=f)
        self._write_clusters(cluster_set, f)

        if cluster_set_element is not None:
            print('nonequiv_element_configs:', file=f)
            self._write_clusters(cluster_set_element, f)
        f.close()

    def parse_clusters_yaml(self, filename='clusters.yaml'):

        data = yaml.safe_load(open(filename))
        self.prim = self._parse_structure(data, tag='primitive_cell')

        clusters = self._parse_clusters(data, 
                                        tag='nonequiv_clusters',
                                        prim=self.prim)
        cluster_set = ClusterSet(clusters)
        clusters_ele = self._parse_clusters(data, 
                                            tag='nonequiv_element_configs',
                                            prim=self.prim)

        if len(data['element_configs']) > 0:
            elements_lattice = []
            for d in data['element_configs']:
                elements_lattice.append(d['elements'])
        else:
            elements_lattice = None

        cluster_set_ele = ClusterSet(clusters_ele, 
                                     elements_lattice=elements_lattice)
        return cluster_set, cluster_set_ele

    def _write_structure(self, st, stream, tag='primitive_cell', indent=0):

        axis = st.axis
        positions = st.positions
        n_atoms = st.n_atoms
        lattice = st.types

        addspace = ' ' * indent

        print(addspace + tag + ':', file=stream)
        print(addspace + '  axis:   ', file=stream)
        for i in range(3):
            print(addspace + '    -   ', list(axis[:,i]), file=stream)
        print('', file=stream)
        print(addspace + '  sites:   ', file=stream)
        for i in range(positions.shape[1]):
            print(addspace + '    - id:            ', i, file=stream)
            print(addspace + '      coordinates:   ', 
                  list(positions[:,i]), file=stream)
            print(addspace + '      lattice:       ', lattice[i], file=stream)
            print('', file=stream)
        print(addspace + '  number_of_sites:   ', list(n_atoms), file=stream)
        print('', file=stream)

    def _parse_structure(self, data, tag='primitive_cell'):

        axis = np.array(data[tag]['axis']).T
        positions = [d['coordinates'] for d in data[tag]['sites']]
        positions = np.array(positions).T
        n_atoms = data[tag]['number_of_sites']

        return Structure(axis, positions, n_atoms)

    def _write_clusters(self, set_obj, stream):

        for i, cl in enumerate(set_obj.clusters):
            print('- serial_id: ', i, file=stream)
            print('  id:        ', cl.idx, file=stream)
            print('  lattice_sites:   ', file=stream)
            if cl.ele_indices is None:
                for site, cell in zip(cl.site_indices, cl.cell_indices):
                    print('  - site:    ', site, file=stream)
                    print('    cell:    ', list(cell), file=stream)
                    print('',file=stream)
            else:
                for site, cell, ele in zip(cl.site_indices, 
                                           cl.cell_indices,
                                           cl.ele_indices):
                    print('  - site:    ', site, file=stream)
                    print('    cell:    ', list(cell), file=stream)
                    print('    element: ', ele, file=stream)
                    print('', file=stream)

    def _parse_clusters(self, data, prim=None, tag='nonequiv_clusters'):

        if not tag in data:
            return []

        clusters = []
        for d in data[tag]:
            sites, cells, elements = [], [], []
            for d1 in d['lattice_sites']:
                sites.append(d1['site'])
                cells.append(d1['cell'])
                if 'element' in d1:
                    elements.append(d1['element'])
            if len(elements) == 0:
                elements = None
            cl = Cluster(idx=d['id'],
                         n_body=len(sites),
                         site_indices=sites,
                         cell_indices=cells,
                         ele_indices=elements,
                         primitive_lattice=prim)
            clusters.append(cl)
        return clusters

    def get_primitive_cell(self):
        return self.prim
 
    def _write_list_no_space(self, a, stream):
        print('[', end='', file=stream)
        print(*list(a), sep=',', end=']\n', file=stream)

    def write_count_clusters_yaml(self, 
                                  cluster_set_element: ClusterSet, 
                                  orbit_sizes,
                                  structure_indices,
                                  n_clusters: np.array,
                                  filename='count_clusters.yaml'):

        f = open(filename, 'w')
        print('nonequiv_element_configs:', file=f)
        self._write_clusters(cluster_set_element, f)

        print('orbit_sizes:', file=f)
        for ids, sizes in sorted(orbit_sizes.items()):
            n_cell, s_id = ids
            print('- n_cell:        ', n_cell, file=f)
            print('  supercell_id:  ', s_id, file=f)
            print('  orbit_sizes:    ', end='', file=f)
            self._write_list_no_space(sizes, f)
            print('', file=f)

        print('number_of_clusters:', file=f)
        for indices, n_all in zip(structure_indices, n_clusters):
            n_cell, s_id, l_id = indices
            print('- n_cell:        ', n_cell, file=f)
            print('  supercell_id:  ', s_id, file=f)
            print('  labeling_id:   ', l_id, file=f)
            print('  n_clusters:    ', end='', file=f)
            self._write_list_no_space(n_all, f)
            print('', file=f)

        f.close()

    def parse_count_clusters_yaml(self, filename='count_clusters.yaml'):

        data = yaml.safe_load(open(filename))
        cluster_set = self._parse_clusters(data,tag='nonequiv_element_configs')

        orbit_sizes, structure_indices, n_clusters = dict(), [], []
        for d in data['orbit_sizes']:
            ids = (d['n_cell'], d['supercell_id'])
            orbit_sizes[ids] = np.array(d['orbit_sizes'])

        for d in data['number_of_clusters']:
            st_id = (d['n_cell'], d['supercell_id'], d['labeling_id'])
            structure_indices.append(st_id)
            n_clusters.append(d['n_clusters'])

        return cluster_set, orbit_sizes, structure_indices, np.array(n_clusters)

    def write_correlations_yaml(self, 
                                cluster_set: ClusterSet, 
                                structure_indices,
                                correlations: np.array,
                                filename='correlations.yaml'):

        f = open(filename, 'w')
        print('nonequiv_clusters:', file=f)
        self._write_clusters(cluster_set, f)

        print('correlation_functions:', file=f)
        for indices, c in zip(structure_indices, correlations):
            n_cell, s_id, l_id = indices
            print('- n_cell:        ', n_cell, file=f)
            print('  supercell_id:  ', s_id, file=f)
            print('  labeling_id:   ', l_id, file=f)
            print('  correlations:    ', end='', file=f)
            self._write_list_no_space(c, f)
            print('', file=f)

        f.close()

    def parse_correlations_yaml(self, filename='correlations.yaml'):

        data = yaml.safe_load(open(filename))
        cluster_set = self._parse_clusters(data,tag='nonequiv_clusters')

        structure_indices, correlations = [], []
        for d in data['correlation_functions']:
            st_id = (d['n_cell'], d['supercell_id'], d['labeling_id'])
            structure_indices.append(st_id)
            n_clusters.append(d['correlations'])

        return cluster_set, structure_indices, np.array(correlations)


