#!/usr/bin/env python
from math import *

import numpy as np


def round_frac(x, tol=1e-13):
    return_x = x - floor(x)
    if return_x > 1 - tol:
        return return_x - 1
    return return_x


# faster than round_frac ?
def round_frac_array(pos, tol=1e-13):
    pos1 = pos - np.floor(pos)
    pos1[np.where(pos1 > 1 - tol)] -= 1.0
    return pos1
