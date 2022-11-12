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

*****************************************************************************/

#include "correlation.h"

ComputeClusterFunction::ComputeClusterFunction(const vector3d& cons_cl,
                                               const vector2i& spin_cl){

//    values = Eigen::VectorXd(spin_cl.size());
    value = 0.0;
    for (int i = 0; i < spin_cl.size(); ++i){
        double prod(1.0);
        for (int j = 0; j < spin_cl[i].size(); ++j){
            const auto& coeffs = cons_cl[i][j];
            const int spin = spin_cl[i][j];
            prod *= eval_poly(coeffs, spin);
        }
        value += prod;
    }
    value /= double(spin_cl.size());

//    #ifdef _OPENMP
//    #pragma omp parallel for schedule(guided,1) if(shape[0] >= 10000)
//    #endif
}

ComputeClusterFunction::~ComputeClusterFunction(){}

double ComputeClusterFunction::eval_poly(const vector1d& coeffs, 
                                         const int spin){

    const int n = coeffs.size() - 1;
    double val = coeffs[n];
    for (int i = 0; i < n; ++i) val += coeffs[i] * pow(spin, n - i);
    return val;
}

double& ComputeClusterFunction::get_value(){ return value; }
Eigen::VectorXd& ComputeClusterFunction::get_values(){ return values; }


