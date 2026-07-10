"""Tests of energy predictions."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_calc_model import PyclupanCalcModel

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/2x2_wurtzite/"


def test_prediction_from_labelings(wurtzite_primitive_cell):
    """Test energy prediction using structures."""
    pyclupan = PyclupanCalcModel(clusters_yaml=path_file + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(path_file + "/pyclupan_ecis.yaml")

    hnf = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    labelings = np.array(
        [
            [0, 0, 0, 0, 2, 2, 2, 2],
            [0, 0, 0, 1, 2, 2, 3, 2],
            [0, 1, 0, 1, 3, 2, 3, 2],
            [0, 0, 1, 1, 3, 3, 2, 2],
            [0, 1, 1, 1, 3, 2, 3, 3],
            [1, 1, 1, 1, 3, 3, 3, 3],
        ]
    )
    pyclupan.set_labelings(
        unitcell=wurtzite_primitive_cell,
        supercell_matrix=hnf,
        active_labelings=labelings,
    )
    pyclupan.eval_energies()
    labelings_end = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
    _ = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)

    energies_true = np.array(
        [-30.083181, -28.964802, -28.86106, -29.782275, -28.812842, -29.810974]
    )
    f_energies_true = np.array([0.0, 1.050328, 1.086018, 0.1648023, 1.066184, 0.0])
    np.testing.assert_allclose(pyclupan.energies, energies_true, atol=1e-6)
    np.testing.assert_allclose(pyclupan.formation_energies, f_energies_true, atol=1e-6)

    ch_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)
