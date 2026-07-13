"""Functions from pypolymlp."""

import pypolymlp.core.data_format as data_format
import pypolymlp.core.interface_vasp as interface_vasp
import pypolymlp.core.units as units
import pypolymlp.utils.vasp_utils as vasp_utils
import pypolymlp.utils.yaml_utils as yaml_utils
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

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
