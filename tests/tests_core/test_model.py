"""Tests of CE model."""

from pathlib import Path

import numpy as np

from pyclupan.core.model import CEmodel

cwd = Path(__file__).parent


def test_CEmodel():
    """Test CEmodel."""
    coeffs = [0.5, 1 / 3, 2.1]
    intercept = 0.234
    cluster_ids = [0, 2, 3]
    rmse = 0.001

    model = CEmodel(coeffs, intercept, cluster_ids=cluster_ids, rmse=rmse)
    cfs = np.array(
        [
            [0.3333, 0.666667, -0.33333],
            [0.5, 0.25, 0.25],
            [-0.5, -0.666666, 0.16666666],
        ]
    )
    energies = model.eval(cfs)
    np.testing.assert_allclose(energies, [-0.077121, 1.092333, 0.111778], atol=1e-6)

    model.supercell(3)
    energies = model.eval(cfs)
    np.testing.assert_allclose(energies, [-0.231362, 3.277, 0.335334], atol=1e-6)
