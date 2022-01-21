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

#ifndef __FUNCTION
#define __FUNCTION

#include "ddcpp.h"

#include <mutex>
#include <thread>
#include <Eigen/Core>

class Pauling{

    Eigen::VectorXd scores;

    double compute_std(const vector1d& data);

    public: 

    Pauling
        (const py::array_t<short>& labeling_v,
         const vector1i& n_atoms,
         const vector2i& neighbors_AB,
         const vector2i& neighbors_O,
         const py::dict& valence_dict,
         const int center_label,
         const int begin_O,
         const int min_ZA,
         const int min_ZB);
    
    ~Pauling();

    Eigen::VectorXd& get_scores();

};

#endif
