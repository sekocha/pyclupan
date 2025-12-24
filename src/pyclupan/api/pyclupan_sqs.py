"""API class for performing SQS calculations."""

from typing import Optional

import numpy as np

from pyclupan.core.pypolymlp_utils import PolymlpStructure, Poscar, write_poscar_file
from pyclupan.mc.sqs_mc import SqsMC


class PyclupanSQS:
    """API class for performing SQS calculations."""

    def __init__(
        self,
        clusters_yaml: str = "pyclupan_clusters.yaml",
        cluster_ids: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        clusters_yaml: File of cluster attributes from cluster search.
        """
        self._verbose = verbose
        self._mc = SqsMC(
            clusters_yaml=clusters_yaml,
            cluster_ids=cluster_ids,
            verbose=verbose,
        )
        np.set_printoptions(legacy="1.21")

    def set_supercell(self, supercell_matrix: np.ndarray, refine: bool = False):
        """Set supercell.

        Parameters
        ----------
        supercell_matrix: Supercell matrix.
            If three elements are given, a diagonal supercell matrix of these
            elements will be used.
        refine: Refine unitcell before applying supercell matrix. Default: False.
            If True, a supercell is constructed by the expansion of given supercell
            matrix for the refined cell.
        """
        self._mc.set_supercell(supercell_matrix=supercell_matrix, refine=refine)
        return self

    def set_parameters(
        self,
        n_steps: int = 100,
        temperature_init: float = 1000.0,
        temperature_final: float = 0.1,
        n_temperatures: int = 20,
    ):
        """Set parameters.

        Parameters
        ----------
        n_steps: Number of steps for simulated annealing at each temperature.
        temperature_init: Initial temperature to set temperatures automatically.
        temperature_final: Final temperature to set temperatures automatically.
        n_temperatures: Number of temperatures.
        """
        self._mc.set_parameters(
            n_steps=n_steps,
            temperature_init=temperature_init,
            temperature_final=temperature_final,
            n_temperatures=n_temperatures,
        )
        return self

    def set_init(
        self,
        structure: Optional[PolymlpStructure] = None,
        poscar: Optional[str] = None,
        element_strings: Optional[tuple] = None,
        compositions: Optional[tuple] = None,
    ):
        """Set initial conditions.

        Parameters
        ----------
        structure: Initial structure for MC simulation.
        poscar: POSCAR file of initial structure for MC simulation.
        element_strings: Element strings to define element IDs.
        compositions: Compositions.
        """
        if poscar is not None:
            structure = Poscar(poscar).structure

        self._mc.set_init(
            structure=structure,
            element_strings=element_strings,
            compositions=compositions,
        )
        return self

    def run(self):
        """Run MC simulation."""
        self._mc.run()
        return self

    @property
    def structure(self):
        """Return final structure."""
        return self._mc.structure

    @property
    def temperatures(self):
        """Return simulation temperatures."""
        return self._mc.temperatures

    def save_structure(self, filename: str = "POSCAR-SQS", header: str = "pyclupan"):
        """Save structure to POSCAR file."""
        write_poscar_file(self.structure, filename=filename, header=header)
