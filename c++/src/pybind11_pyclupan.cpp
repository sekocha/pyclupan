/***************************************************************************

        Copyright (C) 2021 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to
        the Free Software Foundation, Inc., 51 Franklin Street,
        Fifth Floor, Boston, MA 02110-1301, USA, or see
        http://www.gnu.org/copyleft/gpl.txt

*****************************************************************************/

#include "pybind11_pyclupan.h"

PYBIND11_MODULE(pyclupancpp, m) {

    py::class_<NoneqLBL>(m, "NonequivLBL")
        .def(py::init<const py::array_t<short>&, 
                      const py::array_t<short>&>())
        .def("get_labelings", &NoneqLBL::get_labelings, 
                py::return_value_policy::reference_internal)
        ;

    py::class_<NoneqLBLSPeriodic>(m, "NonequivLBLSuperPeriodic")
        .def(py::init<const py::array_t<short>&, 
                      const py::array_t<short>&>())
        .def("get_labelings", &NoneqLBLSPeriodic::get_labelings, 
                py::return_value_policy::reference_internal)
        ;

    py::class_<ComputeClusterFunction>(m, "ComputeCF")
        .def(py::init<const vector3d&,
                      const vector2i&>())
        .def("get_value", &ComputeClusterFunction::get_value, 
                py::return_value_policy::reference_internal)
        ;

//    py::class_<Pauling>(m, "Pauling")
//        .def(py::init<const py::array_t<short>&, 
//                      const vector1i&,
//                      const vector2i&,
//                      const vector2i&,
//                      const py::dict&,
//                      const int,
//                      const int,
//                      const int,
//                      const int>())
//        .def("get_scores", &Pauling::get_scores, 
//                py::return_value_policy::reference_internal)
//        ;
//
}
