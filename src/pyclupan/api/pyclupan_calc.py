"""API Class for calculations."""

from pyclupan.cluster.cluster_io import load_cluster_yaml
from pyclupan.regression.regression_utils import load_eci


class PyclupanCalc:
    """API Class for calculations."""

    def __init__(
        self,
        file_cluster: str = "pyclupan_clusters.yaml",
        file_eci: str = "pyclupan_eci.yaml",
        verbose: bool = False,
    ):
        """Init method."""
        self._verbose = verbose

        self._model = load_eci(file_eci)
        self._unitcell, clusters, _, spin_clusters = load_cluster_yaml(file_cluster)
        self._spin_clusters = [spin_clusters[i] for i in self._model.cluster_ids]

        self._coeffs = self._model.coeffs
        self._intercept = self._model.intercept
        for cl in self._spin_clusters:
            print(cl)

    @property
    def model(self):
        """Return CE model."""
        return self._model
