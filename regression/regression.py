#!/usr/bin/env python
import numpy as np
import argparse
import joblib

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

from pyclupan.common.io.yaml import Yaml

#~/git/pyclupan/regression/regression.py -x ../4-correlation/correlations.pkl -y ../2-dft/summary_dft.dat --upper_bound 0.5

if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('-x',
                    type=str,
                    default='correlations.pkl',
                    help='location of correlations.pkl')

    ps.add_argument('-y',
                    type=str,
                    default='summary_dft.dat',
                    help='location of dft data')

    ps.add_argument('-m', '--method',
                    type=str,
                    choices=['lasso','ridge'],
                    default='lasso',
                    help='regression method (lasso or ridge')

    ps.add_argument('--upper_bound',
                    type=float,
                    default=np.inf,
                    help='excluding data with e > e_ub')
 
    args = ps.parse_args()

    cluster_set, target_ids, correlations, _ = joblib.load(args.x)
    correlations_dict = dict()
    for st_id, corr in zip(target_ids, correlations):
        correlations_dict[st_id] = corr

    yaml = Yaml()
    comp, y, fname, comp_obj = yaml.parse_dft_yaml(args.y)

    ids = np.where(y < args.upper_bound)[0]
    fname, y = fname[ids], y[ids]

    X, st_ids = [], []
    for f in fname:
        items = f.split('/')
        is_found = False
        for item in items:
            if item.count('-') and len(item.split('-')) == 3:
                st_id = tuple([int(o) for o in item.split('-')])
                is_found = True
                break
        if is_found == False:
            print(' No data of correlation funcntions. st_id =', st_id)
            raise KeyError
        X.append(correlations_dict[st_id])
        st_ids.append(st_id)
    X = np.array(X)

    print(' reg. method  =', args.method)
    if args.method == 'lasso':
        clf = LassoCV().fit(X, y)
        coeffs = clf.coef_
        intercept = clf.intercept_
        y_pred = clf.predict(X)

        print(' best alpha   =', clf.alpha_)
        print(' score        =', clf.score(X, y))

    elif args.method == 'ridge':
        clf = RidgeCV(alphas=[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0], 
                      store_cv_values=True)
        clf.fit(X, y)
        coeffs = clf.coef_
        intercept = clf.intercept_
        y_pred = clf.predict(X)

        print(' best alpha   =', clf.alpha_)
        print(' score        =', clf.score(X, y))

    rmse = np.sqrt(np.mean(np.square(y - y_pred)))

    f = open('prediction.dat', 'w')
    for id1, yt1, yp1 in zip(st_ids, y, y_pred):
        print(id1, '{:.15f}'.format(yt1),'{:.15f}'.format(yp1), file=f)
    f.close()

    yaml.write_regression_yaml(cluster_set, coeffs, intercept, rmse, 
                               comp_obj=comp_obj)
    

