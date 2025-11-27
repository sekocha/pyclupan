"""Tests of energy predictions."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_calc import PyclupanCalc
from pyclupan.core.pypolymlp_utils import Poscar

cwd = Path(__file__).parent


def test_prediction_from_poscars():
    """Test energy prediction using poscars."""
    pyclupan = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(str(cwd) + "/pyclupan_ecis.yaml")

    poscars = [str(cwd) + "/derivative-1", str(cwd) + "/derivative-2"]
    pyclupan.load_poscars(poscars)
    pyclupan.element_strings = ("Ag", "Au")
    pyclupan.eval_cluster_functions()
    pyclupan.eval_energies()
    labelings_end = np.array([[0], [1]])
    _ = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)

    energies_true = np.array([-2.99605414, -3.11993823])
    f_energies_true = np.array([-0.02735838, -0.02568571])
    np.testing.assert_allclose(pyclupan.energies, energies_true, atol=1e-6)
    np.testing.assert_allclose(pyclupan.formation_energies, f_energies_true, atol=1e-6)

    ch_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, -0.027358375542346547],
            [0.25, 0.75, -0.025685706225164306],
            [0.0, 1.0, 0.0],
        ]
    )
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)


def test_prediction_from_structures():
    """Test energy prediction using structures."""
    pyclupan = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(str(cwd) + "/pyclupan_ecis.yaml")

    st1 = Poscar(str(cwd) + "/derivative-1").structure
    st2 = Poscar(str(cwd) + "/derivative-2").structure
    pyclupan.structures = [st1, st2]
    pyclupan.element_strings = ("Ag", "Au")
    pyclupan.eval_energies()
    labelings_end = np.array([[0], [1]])
    _ = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)

    energies_true = np.array([-2.99605414, -3.11993823])
    f_energies_true = np.array([-0.02735838, -0.02568571])
    np.testing.assert_allclose(pyclupan.energies, energies_true, atol=1e-6)
    np.testing.assert_allclose(pyclupan.formation_energies, f_energies_true, atol=1e-6)

    ch_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, -0.027358375542346547],
            [0.25, 0.75, -0.025685706225164306],
            [0.0, 1.0, 0.0],
        ]
    )
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)


def test_prediction_from_labelings():
    """Test energy prediction using structures."""
    pyclupan = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(str(cwd) + "/pyclupan_ecis.yaml")

    unitcell = Poscar(str(cwd) + "/fcc-primitive").structure
    hnf = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 4]])
    labelings = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    pyclupan.set_labelings(
        unitcell=unitcell,
        supercell_matrix=hnf,
        active_labelings=labelings,
    )
    pyclupan.eval_energies()
    labelings_end = np.array([[0], [1]])
    _ = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)

    energies_true = np.array(
        [-2.717582, -2.88247, -3.030677, -3.016268, -3.133167, -3.219809]
    )
    f_energies_true = np.array([0.0, -0.039331, -0.061982, -0.047572, -0.038915, 0.0])
    np.testing.assert_allclose(pyclupan.energies, energies_true, atol=1e-6)
    np.testing.assert_allclose(pyclupan.formation_energies, f_energies_true, atol=1e-6)

    ch_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.75, 0.25, -0.039331],
            [0.5, 0.5, -0.061982],
            [0.25, 0.75, -0.03891461],
            [0.0, 1.0, 0.0],
        ]
    )
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)


def test_prediction_from_derivatives():
    """Test energy prediction using derivatives."""
    pyclupan = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(str(cwd) + "/pyclupan_ecis.yaml")
    pyclupan.load_derivatives_yaml(str(cwd) + "/pyclupan_derivatives_2.yaml")
    pyclupan.load_derivatives_yaml(str(cwd) + "/pyclupan_derivatives_3.yaml")
    pyclupan.load_derivatives_yaml(str(cwd) + "/pyclupan_derivatives_4.yaml")

    energies_true = np.array(
        [
            -3.01414175,
            -3.03067727,
            -2.91751881,
            -3.08664788,
            -2.93588289,
            -3.10201068,
            -2.92704007,
            -3.09532142,
            -2.86753467,
            -2.99605414,
            -3.11993823,
            -2.88247015,
            -3.01626789,
            -3.13316713,
            -2.88186463,
            -3.02254074,
            -3.13218195,
            -2.87467561,
            -3.00143248,
            -3.12644339,
            -2.88833075,
            -3.02723509,
            -3.13795694,
            -2.88132276,
            -3.13227422,
            -2.88912044,
            -3.14029247,
        ]
    )
    formation_energies_true = np.array(
        [
            -0.04544599,
            -0.06198151,
            -0.03252755,
            -0.03424761,
            -0.05089163,
            -0.04961041,
            -0.04204881,
            -0.04292115,
            -0.02439566,
            -0.02735838,
            -0.02568571,
            -0.03933114,
            -0.04757212,
            -0.03891461,
            -0.03872562,
            -0.05384497,
            -0.03792942,
            -0.03153661,
            -0.03273672,
            -0.03219086,
            -0.04519174,
            -0.05853932,
            -0.04370441,
            -0.03818375,
            -0.03802169,
            -0.04598144,
            -0.04603994,
        ]
    )

    pyclupan.eval_energies()
    labelings_end = np.array([[0], [1]])
    res = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)
    np.testing.assert_allclose(res[0], formation_energies_true, atol=1e-6)

    st_end1 = Poscar(str(cwd) + "/poscar-end1").structure
    st_end2 = Poscar(str(cwd) + "/poscar-end2").structure
    res = pyclupan.eval_formation_energies(
        structures_endmembers=[st_end1, st_end2],
        element_strings=("Ag", "Au"),
    )
    np.testing.assert_allclose(res[0], formation_energies_true, atol=1e-6)
    np.testing.assert_allclose(
        pyclupan.formation_energies, formation_energies_true, atol=1e-6
    )
    np.testing.assert_allclose(pyclupan.energies, energies_true, atol=1e-6)
    ch_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.75, 0.25, -0.045981436763470995],
            [0.5, 0.5, -0.06198150625768761],
            [0.25, 0.75, -0.04603994362635033],
            [0.0, 1.0, 0.0],
        ]
    )
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)


def test_prediction_from_cluster_functions():
    """Test energy prediction using cluster_functions."""
    pyclupan = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(str(cwd) + "/pyclupan_ecis.yaml")
    pyclupan.load_features(str(cwd) + "/pyclupan_features.hdf5")

    formation_energies_true = np.array(
        [
            -0.04544599,
            -0.06198151,
            -0.03252755,
            -0.03424761,
            -0.05089163,
            -0.04961041,
            -0.04204881,
            -0.04292115,
            -0.02439566,
            -0.02735838,
            -0.02568571,
            -0.03933114,
            -0.04757212,
            -0.03891461,
            -0.03872562,
            -0.05384497,
            -0.03792942,
            -0.03153661,
            -0.03273672,
            -0.03219086,
            -0.04519174,
            -0.05853932,
            -0.04370441,
            -0.03818375,
            -0.03802169,
            -0.04598144,
            -0.04603994,
        ]
    )

    pyclupan.eval_energies()
    labelings_end = np.array([[0], [1]])
    res = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)
    np.testing.assert_allclose(res[0], formation_energies_true, atol=1e-6)

    ch_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.75, 0.25, -0.045981436763470995],
            [0.5, 0.5, -0.06198150625768761],
            [0.25, 0.75, -0.04603994362635033],
            [0.0, 1.0, 0.0],
        ]
    )
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)


def test_prediction_from_energies():
    """Test energy prediction using energies."""
    pyclupan = PyclupanCalc(clusters_yaml=str(cwd) + "/pyclupan_clusters.yaml")
    pyclupan.load_ecis(str(cwd) + "/pyclupan_ecis.yaml")
    pyclupan.load_energies(str(cwd) + "/pyclupan_energies.hdf5")

    formation_energies_true = np.array(
        [
            -0.04544599,
            -0.06198151,
            -0.03252755,
            -0.03424761,
            -0.05089163,
            -0.04961041,
            -0.04204881,
            -0.04292115,
            -0.02439566,
            -0.02735838,
            -0.02568571,
            -0.03933114,
            -0.04757212,
            -0.03891461,
            -0.03872562,
            -0.05384497,
            -0.03792942,
            -0.03153661,
            -0.03273672,
            -0.03219086,
            -0.04519174,
            -0.05853932,
            -0.04370441,
            -0.03818375,
            -0.03802169,
            -0.04598144,
            -0.04603994,
        ]
    )

    labelings_end = np.array([[0], [1]])
    res = pyclupan.eval_formation_energies(labelings_endmembers=labelings_end)
    np.testing.assert_allclose(res[0], formation_energies_true, atol=1e-6)

    ch_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.75, 0.25, -0.045981436763470995],
            [0.5, 0.5, -0.06198150625768761],
            [0.25, 0.75, -0.04603994362635033],
            [0.0, 1.0, 0.0],
        ]
    )
    ch_pred = pyclupan.convexhull[:, :-1].astype(float)
    np.testing.assert_allclose(ch_pred, ch_true, atol=1e-6)
