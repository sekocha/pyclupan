#!/usr/bin/env python 
import numpy as np
import joblib

from pyclupan.common.io.yaml import Yaml


if __name__ == '__main__':

    #yaml = Yaml()
    #cluster_set, orbit_sizes, target_ids, n_counts \
    #    = yaml.parse_count_clusters_yaml('count_clusters.yaml')

    cluster_set, orbit_sizes, target_ids, n_counts \
        = joblib.load('count_clusters.pkl')

    print(orbit_sizes)
    print(n_counts)
