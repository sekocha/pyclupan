"""Tests of labelings"""

from pathlib import Path

import numpy as np

from pyclupan.derivative.labelings_utils import (
    eliminate_superperiodic_labelings,
    get_nonequivalent_labelings,
)

cwd = Path(__file__).parent


def test_get_nonequivalent_labelings():
    """Test get_nonequivalent_labelings."""

    labelings = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    perms = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]

    labelings_perm = get_nonequivalent_labelings(labelings, perms)
    labelings_perm_true = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])

    np.testing.assert_allclose(labelings_perm, labelings_perm_true, atol=1e-6)


def test_eliminate_superperiodic_labelings():
    """Test eliminate_superperiodic_labelings."""
    labelings = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    perms_lt = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
    labelings_perm_lt = eliminate_superperiodic_labelings(labelings, perms_lt)

    labelings_perm_lt_true = np.array([[0, 0, 1, 1]])
    np.testing.assert_allclose(labelings_perm_lt, labelings_perm_lt_true, atol=1e-6)
