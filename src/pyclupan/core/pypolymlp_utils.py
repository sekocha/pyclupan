"""Functions from pypolymlp."""

import os
from typing import Union

import pypolymlp.core.data_format as data_format
import pypolymlp.core.interface_vasp as interface_vasp
import pypolymlp.core.units as units
import pypolymlp.utils.vasp_utils as vasp_utils
import pypolymlp.utils.yaml_utils as yaml_utils
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.api.pypolymlp_str import PypolymlpStructureGenerator

PolymlpStructure = data_format.PolymlpStructure

Poscar = interface_vasp.Poscar
Vasprun = interface_vasp.Vasprun
write_poscar_file = vasp_utils.write_poscar_file

save_cell = yaml_utils.save_cell
save_cells = yaml_utils.save_cells
load_cell = yaml_utils.load_cell
load_cells = yaml_utils.load_cells

KbEV = units.KbEV

try:
    import pypolymlp.utils.structure_utils as supercell_utils

    supercell = supercell_utils.supercell
    supercell_diagonal = supercell_utils.supercell_diagonal
except:
    pass

try:
    import pypolymlp.utils.spglib_utils as spglib_utils

    ReducedCell = spglib_utils.ReducedCell
    SpglibCell = spglib_utils.SymCell
except:
    pass


class Polymlp:
    """API class for using pypolymlp."""

    def __init__(self, pot: str = "polymlp.yaml"):
        """Init method."""
        self._polymlp = PypolymlpCalc(pot=pot)
        self._energy = None
        self._structure = None

    def eval(self, st: PolymlpStructure | list):
        """Evaluate properties."""
        return self._polymlp.eval(st)

    def run_geometry_optimization(
        self,
        st: PolymlpStructure,
        gtol: float = 1e-4,
    ):
        """Run geometry optmization."""
        self._polymlp.init_geometry_optimization(
            init_str=st,
            with_sym=True,
            relax_cell=True,
            relax_volume=True,
            relax_positions=True,
        )
        try:
            self._polymlp.run_geometry_optimization(gtol=gtol)
            self._structure = self._polymlp.converged_structure
            self._energy = self._polymlp.go_data[0]
            success = True
        except:
            success = False

        return success

    @property
    def energy(self):
        """Return energy."""
        return self._energy

    @property
    def structure(self):
        """Return converged structure."""
        return self._structure


class PolymlpStructureGenerator:
    """API class for using pypolymlp structure generator."""

    def __init__(
        self,
        base_structures: Union[PolymlpStructure, list[PolymlpStructure]],
    ):
        """Init method."""
        self._polymlp = PypolymlpStructureGenerator(base_structures=base_structures)

    def run_const_displacements(self, n_samples: int = 100, distance: float = 0.03):
        """Generate random structures with constant magnitude of displacements.

        Parameters
        ----------
        n_samples: Number of structures generated for each supercell structure.
        distance: Magnitude of atomic displacements.
        """
        return self._polymlp.run_const_displacements(
            n_samples=n_samples, distance=distance
        )

    def run_standard_algorithm(self, n_samples: int = 100, max_distance: float = 1.5):
        """Generate random structures from base structures using a standard algorithm.

        In the standard algorithm, displacements in i-th structure are given by
            disp = max([(i + 1) / n_samples]**3 * max_distance, 0.01).

        Parameters
        ----------
        n_samples: Number of structures generated from a single POSCAR file
                   using a standard algorithm.
        max_distance: Maximum distance of displacement distributions.
        """
        self._polymlp.build_supercells_auto(max_natom=1)
        return self._polymlp.run_standard_algorithm(
            n_samples=n_samples, max_distance=max_distance
        )

    @property
    def sample_structures(self) -> list[PolymlpStructure]:
        """Return sample structures."""
        return self._polymlp.sample_structures

    def save_structures(self, path: str = "poscars"):
        """Save structures in poscar format."""
        if self._polymlp.sample_structures is None:
            raise RuntimeError("Sampled structures not found.")

        os.makedirs(path, exist_ok=True)
        for i, st in enumerate(self._polymlp.sample_structures, start=1):
            write_poscar_file(
                st,
                filename=path + "/poscar-" + str(i).zfill(5),
                header="pyclupan: disp-" + str(i).zfill(5),
            )
