#!/usr/bin/env python
import numpy as np
import argparse
import joblib

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge

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

    cluster_set, target_ids, correlations = joblib.load(args.x)

    yaml = Yaml()
    coeffs, intercept = yaml.parse_regression_yaml(args.coeffs)
    y_pred = np.dot(correlations, coeffs) + intercept
    
    f = open('prediction.dat', 'w')
    for st_id, yp1 in zip(target_ids, y_pred):
        print(st_id, '{:.15f}'.format(yp1), file=f)
    f.close()

