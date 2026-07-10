"""Tests of cluster function calculations."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_features import PyclupanCalcFeatures

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/2x2_wurtzite/"


def _init_calc():
    features = PyclupanCalcFeatures(clusters_yaml=path_file + "/pyclupan_clusters.yaml")
    return features


def test_eval_cluster_functions_from_derivatives():
    """Test eval_cluster_functions using files for derivative structures."""
    features = _init_calc()
    features.append_derivatives_yaml(path_file + "/pyclupan_derivatives_1.yaml")
    cluster_functions = features.eval_cluster_functions()

    cf_calc1 = cluster_functions[0, :10]
    cf_calc2 = cluster_functions[1, :10]
    cf_calc3 = cluster_functions[2, :10]
    cf_calc4 = cluster_functions[3, :10]
    cf_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cf_2 = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    cf_3 = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
    cf_4 = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(cf_calc1, cf_1, atol=1e-8)
    np.testing.assert_allclose(cf_calc2, cf_2, atol=1e-8)
    np.testing.assert_allclose(cf_calc3, cf_3, atol=1e-8)
    np.testing.assert_allclose(cf_calc4, cf_4, atol=1e-8)
