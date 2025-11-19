"""Functions for sampling derivative structures."""

# import time
from typing import Union

#
import numpy as np

from pyclupan.derivative.sample_utils import DerivativesSet, load_derivative_yaml

# from pypolymlp.core.data_format import PolymlpStructure


def run_sampling_derivatives(
    files: Union[str, list],
    verbose: bool = False,
):
    """Enumerate derivative structures.

    Parameters
    ----------
    """
    pass
    if isinstance(files, str):
        ds_set = load_derivative_yaml(files)
        print(ds_set)
    elif isinstance(files, (list, tuple, np.ndarray)):
        ds_set = DerivativesSet([])
        for f in files:
            ds = load_derivative_yaml(f)
            ds_set.append(ds)
        print(ds_set)

    print(len(ds_set))
    for d in ds_set:
        print(d.inactive_sites)
        print(d.active_sites)
    print(ds_set[0])
