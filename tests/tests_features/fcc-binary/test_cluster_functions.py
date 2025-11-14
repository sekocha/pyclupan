"""Tests of cluster function calculations."""

from pathlib import Path

import numpy as np

from pyclupan.api.pyclupan_features import PyclupanFeatures

cwd = Path(__file__).parent


def test_eval_cluster_functions():
    """Test eval_cluster_functions."""
    element_labels = {"Ag": 0, "Au": 1}
    features = PyclupanFeatures(cluster_yaml=str(cwd) + "/pyclupan_cluster.yaml")
    features.load_poscars([str(cwd) + "/derivative-1", str(cwd) + "/derivative-2"])
    features.element_string_labels = element_labels
    cluster_functions = features.eval_cluster_functions()

    cf_calc1 = cluster_functions[0, :10]
    cf_calc2 = cluster_functions[1, :10]
    cf_true1 = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cf_true2 = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(cf_calc1, cf_true1, atol=1e-8)
    np.testing.assert_allclose(cf_calc2, cf_true2, atol=1e-8)
