#!/usr/bin/env python
import numpy as np
import sys
import yaml

from mlptools.common.structure import Structure
from pyclupan.dd.cluster import Cluster, ClusterSet

class Yaml:

    def __init__(self):
        pass

    def write_clusters_yaml(self, 
                            primitive_cell,
                            cutoff,
                            elements_lattice,
                            cluster_set: ClusterSet, 
                            cluster_set_element: ClusterSet=None, 
                            filename='clusters.yaml'):

        f = open(filename, 'w')

        print('#  Nonequivalent clusters (pyclupan)', file=f)
        print('', file=f)

        print('cutoff:', file=f)
        for i, c in enumerate(cutoff):
            print('- ' + str(i+2) + '-body:   ', c, file=f)
            print('', file=f)
        print('', file=f)

        print('element_configs:', file=f)
        for i, elements in enumerate(elements_lattice):
            print('- lattice:    ', i, file=f)
            if len(elements) > 0:
                print('  elements:   ', elements, file=f)
            else:
                print('  elements:   ', [], file=f)
            print('', file=f)
        print('', file=f)

        self.write_primitive_cell(primitive_cell, f)

        print('nonequiv_clusters:', file=f)
        self.write_clusters(cluster_set, f)

        if cluster_set_element is not None:
            print('nonequiv_element_configs:', file=f)
            self.write_clusters(cluster_set_element, f)
        f.close()

    def write_primitive_cell(self, primitive_cell, stream):

        axis = primitive_cell.axis
        positions = primitive_cell.positions
        n_atoms = primitive_cell.n_atoms
        lattice = primitive_cell.types

        print('primitive_cell:', file=stream)
        print('  axis:   ', file=stream)
        for i in range(3):
            print('    -   ', list(axis[:,i]), file=stream)
        print('', file=stream)
        print('  sites:   ', file=stream)
        for i in range(positions.shape[1]):
            print('    - coordinates:   ', list(positions[:,i]), file=stream)
            print('      lattice:       ', lattice[i], file=stream)
            print('', file=stream)
        print('  number_of_sites:   ', list(n_atoms), file=stream)
        print('', file=stream)


    def write_clusters(self, set_obj, stream):

        for cl in set_obj.clusters:
            print('- id:   ', cl.idx, file=stream)
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

    def parse_primitive_cell(self, data):

        axis = np.array(data['primitive_cell']['axis']).T
        n_atoms = data['primitive_cell']['number_of_sites']

        positions = [d['coordinates'] for d in data['primitive_cell']['sites']]
        positions = np.array(positions).T
        print(positions)

        return Structure(axis, positions, n_atoms)

    def parse_clusters_yaml(self, filename='clusters.yaml'):

        data = yaml.safe_load(open(filename))

        prim = self.parse_primitive_cell(data)

        for d in data['nonequiv_clusters']:
            print(d)

        # clusters = []
        #
        # cluster_set = ClusterSet(clusters)
        # return cluster_set
            


