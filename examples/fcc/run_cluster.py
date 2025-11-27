# import numpy as np

from pyclupan.cluster.run_cluster import run_cluster
from pyclupan.core.pypolymlp_utils import Poscar

st = Poscar("fcc-primitive").structure

elements = [[0, 1]]
run_cluster(unitcell=st, elements=elements, verbose=True)
