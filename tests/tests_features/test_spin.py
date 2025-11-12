"""Tests of spin polynomials."""

from pathlib import Path

import numpy as np

from pyclupan.features.spin_polynomial import gram_schmidt

cwd = Path(__file__).parent


def test_spin_polynomial():
    """Test spin polynomials."""
    spins = [1, 0, -1]
    cons = gram_schmidt(spins)
    cons_true = np.array([[0, 0, 1], [0, 1.22474487, 0], [2.12132034, 0, -1.41421356]])
    np.testing.assert_allclose(cons, cons_true)

    spins = [2, 1, 0, -1]
    cons = gram_schmidt(spins)
    cons_true = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.89442719, -0.4472136],
            [0.0, 1.0, -1.0, -1.0],
            [1.49071198, -2.23606798, -1.93792558, 1.34164079],
        ]
    )
    np.testing.assert_allclose(cons, cons_true)
