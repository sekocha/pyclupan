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

ComputeClusterFunction::ComputeClusterFunction
(const py::array_t<short>& labeling_spinrep,
 const vector3i& site_cls,
 const vector3i& cons_id_cls,
 const vector2d& cons){

    const auto &buff_info = labeling_spinrep.request();
    const auto &shape = buff_info.shape;

    const int n_features = site_cls.size();
    values = Eigen::MatrixXd(shape[0], n_features);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided,1) if(shape[0] >= 50000)
    #endif
    for (int i = 0; i < shape[0]; ++i){
        for (int j = 0; j < n_features; ++j){
            const auto& site_cl = site_cls[j];
            const auto& cons_id_cl = cons_id_cls[j];
            double value1 = 0.0;
            for (int k = 0; k < site_cl.size(); ++k){
                double prod = 1.0;
                for (int l = 0; l < site_cl[k].size(); ++l){
                    const auto& coeffs = cons[cons_id_cl[k][l]];
                    const short spin = *labeling_spinrep.data(i, site_cl[k][l]);
                    prod *= eval_poly(coeffs, spin);
                }
                value1 += prod;
            }
            values(i,j) = value1 / double(site_cl.size());
        }
    }
}

ComputeClusterFunction::ComputeClusterFunction(const vector3d& cons_cl,
                                               const vector2i& spin_cl){

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
Eigen::MatrixXd& ComputeClusterFunction::get_values(){ return values; }
