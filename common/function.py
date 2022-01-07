#!/usr/bin/env python
from math import *

def round_frac(x, tol=1e-13):
    return_x = x - floor(x)
    if return_x > 1 - tol:
        return_x -= 1
    return return_x

#def round_frac(x, tol=1e-13):
#    return_x = x
#    if x > 1 or x < 0:
#        return_x = x - floor(x)
#    if return_x > 1-tol:
#        return_x -= 1
#    return return_x
#
