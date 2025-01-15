#!/usr/bin/env python
import joblib
import numpy as np

from pyclupan.cluster.cluster import Cluster, ClusterSet
from pyclupan.common.io.yaml import Yaml
from pyclupan.derivative.derivative import DSSample, DSSet


def parse_clusters_yaml(filename):

    yaml = Yaml()
    clusters, clusters_ele = yaml.parse_clusters_yaml(filename=filename)
    print("  - yaml file (cluster) =", filename)

    if clusters_ele.get_num_clusters() == 0:
        raise RuntimeError("Unable to parse nonequiv_element_config in clusters.yaml")
    return clusters, clusters_ele


def parse_derivatives(filenames):

    if len(filenames) > 1 and "derivative-all.pkl" in filenames:
        filenames.remove("derivative-all.pkl")
    print("  - pkl files (derivative st.) =", filenames)

    ds_set_all = []
    for f in filenames:
        ds_set_all.extend(joblib.load(f))

    return DSSample(ds_set_all)
