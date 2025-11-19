"""Functions for sampling derivative structures."""

from typing import Literal, Optional, Union

import numpy as np

from pyclupan.derivative.sample_utils import DerivativesSet, load_derivative_yaml


def run_sampling_derivatives(
    files: Union[str, list],
    element_strings: Optional[tuple] = None,
    n_samples: int = 100,
    method: Literal["all", "uniform", "random"] = "uniform",
):
    """Enumerate derivative structures.

    Parameters
    ----------
    TODO: Make docstrings.
    """
    if isinstance(files, str):
        ds_set = load_derivative_yaml(files)
    elif isinstance(files, (list, tuple, np.ndarray)):
        ds_set = DerivativesSet([])
        for f in files:
            ds = load_derivative_yaml(f)
            ds_set.append(ds)

    if method == "all":
        _ = ds_set.all()
    elif method == "uniform":
        _ = ds_set.uniform(n_samples=n_samples)
    elif method == "random":
        _ = ds_set.random(n_samples=n_samples)

    ds_set.save(element_strings)
