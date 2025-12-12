"""Functions for sampling derivative structures."""

from typing import Literal, Optional, Union

import numpy as np

from pyclupan.derivative.derivative_utils import DerivativesSet, load_derivatives_yaml


def run_sampling_derivatives(
    files: Optional[Union[str, list]] = None,
    ds_set: Optional[DerivativesSet] = None,
    keys: Optional[list] = None,
    method: Literal["all", "uniform", "random"] = "uniform",
    n_samples: int = 100,
    element_strings: Optional[tuple] = None,
    save_poscars: bool = True,
    path_poscars: str = "poscars",
):
    """Enumerate derivative structures.

    Parameters
    ----------
    TODO: Make docstrings.
    """
    if files is not None and ds_set is None:
        if isinstance(files, str):
            ds_set = load_derivatives_yaml(files)
        elif isinstance(files, (list, tuple, np.ndarray)):
            ds_set = DerivativesSet([])
            for f in files:
                ds = load_derivatives_yaml(f)
                ds_set.append(ds)

    if keys is not None:
        for k in keys:
            ds_set.select(k)
    else:
        if method == "all":
            _ = ds_set.all()
        elif method == "uniform":
            _ = ds_set.uniform(n_samples=n_samples)
        elif method == "random":
            _ = ds_set.random(n_samples=n_samples)

    if save_poscars:
        ds_set.save(element_strings, path=path_poscars)
    return ds_set
