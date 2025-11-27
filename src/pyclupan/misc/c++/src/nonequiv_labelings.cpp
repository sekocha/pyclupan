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

*****************************************************************************/

#include "nonequiv_labelings.h"

NoneqLBL::NoneqLBL(const py::array_t<short>& labeling,
                   const py::array_t<short>& permutation){

    const auto &buff_info = labeling.request();
    const auto &shape = buff_info.shape;
    const auto &buff_info2 = permutation.request();
    const auto &shape2 = buff_info2.shape;

    vector2s nonequiv_labelings_vec(shape[0]);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided, 1) if(shape[0] >= 1000)\
        shared(nonequiv_labelings_vec)
    #endif
    for (int i = 0; i < shape[0]; ++i){
        vector2s orbit(shape2[0], vector1s(shape2[1]));
        for (int j = 0; j < shape2[0]; ++j){
            for (int k = 0; k < shape2[1]; ++k){
                short p = *permutation.data(j,k);
                orbit[j][k] = *labeling.data(i,p);
            }
        }
        nonequiv_labelings_vec[i] = *std::min_element(orbit.begin(),
                                                      orbit.end());
    }

    std::set<vector1s> nonequiv_labelings(nonequiv_labelings_vec.begin(),
                                          nonequiv_labelings_vec.end());

    labelings_output = Eigen::MatrixXi(nonequiv_labelings.size(), shape[1]);
    int i = 0;
    for (const auto& x1: nonequiv_labelings){
        for (int j = 0; j < x1.size(); ++j) labelings_output(i,j) = x1[j];
        ++i;
    }
}

NoneqLBL::~NoneqLBL(){}

const Eigen::MatrixXi& NoneqLBL::get_labelings() const{
    return labelings_output;
}

NoneqLBLSPeriodic::NoneqLBLSPeriodic(const py::array_t<short>& labeling,
                                     const py::array_t<short>& permutation){

    const auto &buff_info = labeling.request();
    const auto &shape = buff_info.shape;
    const auto &buff_info2 = permutation.request();
    const auto &shape2 = buff_info2.shape;

    std::vector<bool> superperiodic(shape[0], false);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided, 1) if(shape[0] >= 10000)
    #endif
    for (int i = 0; i < shape[0]; ++i){
        std::set<vector1s> orbit;
        for (int j = 0; j < shape2[0]; ++j){
            vector1s l_perm(shape2[1]);
            for (int k = 0; k < shape2[1]; ++k){
                short p = *permutation.data(j,k);
                l_perm[k] = *labeling.data(i,p);
            }
            orbit.insert(l_perm);
            if (orbit.size() != j + 1) {
                superperiodic[i] = true;
                break;
            }
        }
    }

    int size1 = std::count(superperiodic.begin(), superperiodic.end(), false);

    labelings_output = Eigen::MatrixXi(size1, shape[1]);
    int count = 0;
    for (int i = 0; i < shape[0]; ++i){
        if (superperiodic[i] == false){
            for (int j = 0; j < shape[1]; ++j)
                labelings_output(count,j) = *labeling.data(i,j);
            ++count;
        }
    }
}

NoneqLBLSPeriodic::~NoneqLBLSPeriodic(){}

const Eigen::MatrixXi& NoneqLBLSPeriodic::get_labelings() const{
    return labelings_output;
}
