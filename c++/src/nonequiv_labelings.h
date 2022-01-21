/****************************************************************************

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

****************************************************************************/

#ifndef __NONEQUIV_LABELINGS
#define __NONEQUIV_LABELINGS

#include "pyclupan.h"

class NoneqLBL{

    Eigen::MatrixXi labelings_output;

    public: 

    NoneqLBL(const py::array_t<short>& labeling,
             const py::array_t<short>& permutation);
    
    ~NoneqLBL();
    const Eigen::MatrixXi& get_labelings() const;

};

class NoneqLBLSPeriodic{

    Eigen::MatrixXi labelings_output;

    public: 

    NoneqLBLSPeriodic(const py::array_t<short>& labeling,
                      const py::array_t<short>& permutation);
    ~NoneqLBLSPeriodic();
    const Eigen::MatrixXi& get_labelings() const;

};

#endif
