"""Tests of energy predictions."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_calc import PyclupanCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/ternary_fcc/"


def test_prediction_from_poscars():
    """Test energy prediction using poscars."""
    pyclupan = PyclupanCalc(clusters_yaml=path_file + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(path_file + "/pyclupan_ecis.yaml")

    poscars = [path_file + "/derivative-1"]
    pyclupan.load_poscars(poscars)
    pyclupan.element_strings = ("Cu", "Ag", "Au")
    pyclupan.eval_cluster_functions()
    pyclupan.eval_energies()
    labelings_end = np.array([[0], [1], [2]])
    _ = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)

    energies_true = np.array([-3.072257])
    f_energies_true = np.array([0.015309])
    np.testing.assert_allclose(pyclupan.energies, energies_true, atol=1e-6)
    np.testing.assert_allclose(pyclupan.formation_energies, f_energies_true, atol=1e-6)

    ch_true = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)


def test_prediction_from_labelings(fcc_primitive_cell):
    """Test energy prediction using structures."""
    pyclupan = PyclupanCalc(clusters_yaml=path_file + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(path_file + "/pyclupan_ecis.yaml")

    hnf = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 4]])
    labelings = np.array(
        [
            [0, 2, 0, 0],
            [0, 0, 2, 1],
            [0, 1, 2, 1],
            [0, 2, 1, 1],
            [0, 1, 2, 1],
            [1, 1, 2, 1],
        ]
    )
    pyclupan.set_labelings(
        unitcell=fcc_primitive_cell,
        supercell_matrix=hnf,
        active_labelings=labelings,
    )
    pyclupan.eval_energies()
    labelings_end = np.array([[0], [1], [2]])
    _ = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)

    energies_true = np.array(
        [-3.615976, -3.313697, -3.067989, -3.075452, -3.067989, -2.874264]
    )
    f_energies_true = np.array(
        [-0.036874, 0.019637, 0.019577, 0.012114, 0.019577, -0.032467]
    )
    np.testing.assert_allclose(pyclupan.energies, energies_true, atol=1e-6)
    np.testing.assert_allclose(pyclupan.formation_energies, f_energies_true, atol=1e-6)

    ch_true = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.75, 0.0, 0.25, -0.03687395],
            [0.0, 0.75, 0.25, -0.03246657],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)


def test_prediction_from_derivatives():
    """Test energy prediction using derivatives."""
    pyclupan = PyclupanCalc(clusters_yaml=path_file + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(path_file + "/pyclupan_ecis.yaml")
    pyclupan.load_derivatives_yaml(path_file + "/pyclupan_derivatives_1.yaml")
    pyclupan.load_derivatives_yaml(path_file + "/pyclupan_derivatives_2.yaml")
    pyclupan.load_derivatives_yaml(path_file + "/pyclupan_derivatives_3.yaml")

    pyclupan.eval_energies()
    labelings_end = np.array([[0], [1], [2]])
    _ = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)
    ch_true = np.array(
        [
            [1.00000000e00, 0.00000000e00, 0.00000000e00, 4.44089210e-16],
            [0.00000000e00, 1.00000000e00, 0.00000000e00, 0.00000000e00],
            [6.66666667e-01, 0.00000000e00, 3.33333333e-01, -5.69142985e-02],
            [0.00000000e00, 6.66666667e-01, 3.33333333e-01, -4.95900128e-02],
            [5.00000000e-01, 0.00000000e00, 5.00000000e-01, -7.21695345e-02],
            [0.00000000e00, 5.00000000e-01, 5.00000000e-01, -6.43968598e-02],
            [3.33333333e-01, 0.00000000e00, 6.66666667e-01, -5.81009299e-02],
            [0.00000000e00, 3.33333333e-01, 6.66666667e-01, -4.94959313e-02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, -4.44089210e-16],
        ]
    )
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)
