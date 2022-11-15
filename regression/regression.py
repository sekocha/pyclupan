#!/usr/bin/env python
import numpy as np
import argparse
import joblib

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge

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

    ps.add_argument('--upper_bound',
                    type=float,
                    default=np.inf,
                    help='excluding data with e > e_ub')
 
    args = ps.parse_args()

    _, target_ids, correlations = joblib.load(args.x)
    correlations_dict = dict()
    for st_id, corr in zip(target_ids, correlations):
        correlations_dict[st_id] = corr

    dft = np.loadtxt(args.y, dtype=str)
    ids = np.where(dft[:,-1].astype(float) < args.upper_bound)[0]
    fname = dft[ids,0]
    y = dft[ids,-1].astype(float)

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

    #method = 'lasso'
    method = 'ridge'
    if method == 'lasso':
        #alpha_array = [1e-2,1e-3,1e-4,1e-5,1e-6]
        #alpha_array = [1e-5]
        #for alpha in alpha_array:
        #    clf = Lasso(alpha=alpha)
        #    clf.fit(X, y)

        #    print(clf.coef_)
        #    print(clf.intercept_)

        reg = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X, y)
        print(reg.score(X, y))
        for y_pred, y_true in zip(reg.predict(X), y):
            print(y_true, y_pred)

    elif method == 'ridge':
        clf = Ridge(alpha=1e-8, fit_intercept=True)
        clf.fit(X, y)
        #print(clf.coef_)
        y_pred = clf.predict(X)

    print(' rmse =', np.sqrt(np.mean(np.square(y - y_pred))))
#    y_data = np.vstack([y, y_pred]).T

    f = open('prediction.dat', 'w')
    for id1, yt1, yp1 in zip(st_ids, y, y_pred):
        print(id1, '{:.15f}'.format(yt1),'{:.15f}'.format(yp1), file=f)
    f.close()

#    np.savetxt('prediction.dat', y_data, fmt='%.15f')
    


