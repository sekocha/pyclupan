#!/usr/bin/env python 
import numpy as np
import argparse
import joblib
from joblib import Parallel,delayed
import time

from mlptools.common.structure import Structure

from pyclupan.common.supercell import Supercell
from pyclupan.common.io.yaml import Yaml
from pyclupan.cluster.cluster import Cluster
from pyclupan.cluster.cluster import ClusterSet
from pyclupan.derivative.derivative import DSSet
from pyclupan.derivative.derivative import DSSample


if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('--derivative_pkl',
                    type=str,
                    nargs='*',
                    default=['derivative-all.pkl'],
                    help='DSSet pkl files')
    ps.add_argument('--clusters_yaml',
                    type=str,
                    default='clusters.yaml',
                    help='location of clusters.yaml')
    args = ps.parse_args()

    if len(args.derivative_pkl) > 1:
        if 'derivative-all.pkl' in args.derivative_pkl:
            args.derivative_pkl.remove('derivative-all.pkl')
    print(' derivative_pkl files =', args.derivative_pkl)

    ds_set_all = []
    for f in args.derivative_pkl:
        ds_set_all.extend(joblib.load(f))
    ds_samp = DSSample(ds_set_all)
    prim = ds_samp.get_primitive_cell()
    print(prim)



