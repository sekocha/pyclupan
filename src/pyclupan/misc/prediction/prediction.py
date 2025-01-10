#!/usr/bin/env python
import numpy as np
import argparse
import joblib

from pyclupan.common.io.yaml import Yaml

"""
~/git/pyclupan/prediction/prediction.py -x ../4-correlation/correlations.pkl --coeffs ../5-reg/regression.yaml
"""

if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('-x',
                    type=str,
                    default='correlations.pkl',
                    help='location of correlations.pkl')

    ps.add_argument('--coeffs',
                    type=str,
                    default='regression.yaml',
                    help='location of yaml file for model coefficients')

    args = ps.parse_args()

    cluster_set, target_ids, correlations, n_atoms_all = joblib.load(args.x)

    yaml = Yaml()
    coeffs, cluster_ids, intercept, comp_obj \
                    = yaml.parse_regression_yaml(args.coeffs)

    y_pred = np.dot(correlations[:,cluster_ids], coeffs) + intercept

    comp_all, partition = comp_obj.get_comp_multiple(n_atoms_all)

    f = open('prediction.dat', 'w')
    print(' # st_id, compositions, ..., energy', file=f)
    for st_id, yp1, comp in zip(target_ids, y_pred, comp_all):
        print(st_id, end=' ', file=f)
        for c in comp:
            print('{:.8f}'.format(c), end=' ', file=f)
        print('{:.15f}'.format(yp1), file=f)
    f.close()

