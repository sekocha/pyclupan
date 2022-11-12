/****************************************************************************

        Copyright (C) 2022 Atsuto Seko
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

#ifndef __COMPUTE_CLUSTER_FUNCTION
#define __COMPUTE_CLUSTER_FUNCTION

#include <mutex>
#include <thread>
#include <Eigen/Core>

#include <pyclupan.h>

class ComputeClusterFunction{

    double value;
    Eigen::VectorXd values;
    double eval_poly(const vector1d& coeff, const int spin);

    public: 

    ComputeClusterFunction(const vector3d& cons_cl,
                           const vector2i& spin_cl);
   
    ~ComputeClusterFunction();

    double& get_value();
    Eigen::VectorXd& get_values();

};

#endif
