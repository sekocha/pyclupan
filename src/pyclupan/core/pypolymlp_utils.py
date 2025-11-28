"""Functions from pypolymlp."""

import pypolymlp.core.data_format as data_format
import pypolymlp.core.interface_vasp as interface_vasp
import pypolymlp.core.units as units
import pypolymlp.utils.vasp_utils as vasp_utils
import pypolymlp.utils.yaml_utils as yaml_utils

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
