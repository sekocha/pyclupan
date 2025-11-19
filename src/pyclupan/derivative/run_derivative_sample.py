"""Functions for sampling derivative structures."""

from typing import Optional, Union

import numpy as np

from pyclupan.derivative.sample_utils import DerivativesSet, load_derivative_yaml


def run_sampling_derivatives(
    files: Union[str, list],
    element_strings: Optional[tuple] = None,
    verbose: bool = False,
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

    ds_set.all()
    ds_set.save(element_strings)

    # ds_set.random()
    # ds_set.labeling_ids
