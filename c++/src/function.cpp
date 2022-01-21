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

#include <set>
#include <algorithm>
#include "function.h"

Pauling::Pauling
(const py::array_t<short>& labeling_v,
 const vector1i& n_atoms,
 const vector2i& neighbors_AB,
 const vector2i& neighbors_O,
 const py::dict& valence_dict,
 const int center_label,
 const int begin_O,
 const int min_ZA,
 const int min_ZB){

    const auto &buff_info = labeling_v.request();
    const auto &shape = buff_info.shape;

    std::map<int, int> valence;
    for (std::pair<py::handle, py::handle> item : valence_dict){
        auto key = item.first.cast<int>();
        auto value = item.second.cast<int>();
        valence[key] = value;
    }

    scores = Eigen::VectorXd(shape[0]);
    int split = int(neighbors_AB.size()/2);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided,1) if(shape[0] >= 10000)
    #endif
    for (int i = 0; i < shape[0]; ++i){
        vector1i Z, ZA, ZB;
        for (const auto& n1: neighbors_AB){
            vector1i l_nn;
            for (const auto& n2: n1){ 
                auto v = *labeling_v.data(i, n2);
                l_nn.emplace_back(v);
            }
            Z.emplace_back(std::count(l_nn.begin(), l_nn.end(), center_label));
        }
        for (int j = 0; j < split; ++j){
            ZA.emplace_back(Z[j]);
            ZB.emplace_back(Z[j+split]);
        }

        if (*std::min_element(ZA.begin(), ZA.end()) >= min_ZA
            and *std::min_element(ZB.begin(), ZB.end()) >= min_ZB){

            vector1d Zv(Z.size());
            for (int j = 0; j < Z.size(); ++j) {
                auto label = *labeling_v.data(i,j);
                Zv[j] = valence[label] / double(Z[j]);
            }
            vector1i index_O;
            for (int j = Z.size(); j < shape[1]; ++j){
                auto label = *labeling_v.data(i,j);
                if (label == center_label){
                    index_O.emplace_back(j - begin_O);
                }
            }
            vector1d v_sum_all;
            for (auto j: index_O){
                double vsum = 0.0;
                for (auto k: neighbors_O[j]){
                    vsum += Zv[k];
                }
                v_sum_all.emplace_back(vsum);
            }
            scores[i] = compute_std(v_sum_all);
        }
        else {
            scores[i] = 1e10;
        }
    }
}

Pauling::~Pauling(){}

double Pauling::compute_std(const vector1d& data){

    const int size = data.size();
    const auto ave = std::accumulate(data.begin(), data.end(), 0.0) / size;
    const auto var = std::inner_product
        (data.begin(), data.end(), data.begin(), 0.0) / size - ave * ave;
    return sqrt(var);
}

Eigen::VectorXd& Pauling::get_scores(){ return scores; }


