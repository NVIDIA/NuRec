/*
 * BSD License
 *
 * For PyTorch3D software
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name Meta nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <queue>
#include <torch/extension.h>
#include <tuple>

namespace {

// Distance between p1[n][i1] and p2[n][i2] under the requested norm.
template <typename FloatAccessor>
float point_distance(
    const FloatAccessor& p1_a,
    const FloatAccessor& p2_a,
    const int n,
    const int64_t i1,
    const int64_t i2,
    const int D,
    const int norm) {
    float dist = 0;
    for (int d = 0; d < D; ++d) {
        float diff = p1_a[n][i1][d] - p2_a[n][i2][d];
        if (norm == 1) {
            dist += abs(diff);
        } else { // norm is 2 (default)
            dist += diff * diff;
        }
    }
    return dist;
}

// Max-heap holding the K nearest neighbours of p1[n][i1] among p2[n][0, length2),
// as (distance, index) pairs.
template <typename FloatAccessor>
std::priority_queue<std::tuple<float, int>> nearest_neighbors(
    const FloatAccessor& p1_a,
    const FloatAccessor& p2_a,
    const int n,
    const int64_t i1,
    const int64_t length2,
    const int D,
    const int norm,
    const int K) {
    std::priority_queue<std::tuple<float, int>> q;
    for (int64_t i2 = 0; i2 < length2; ++i2) {
        float dist = point_distance(p1_a, p2_a, n, i1, i2, D, norm);
        int size   = static_cast<int>(q.size());
        if (size < K || dist < std::get<0>(q.top())) {
            q.emplace(dist, i2);
            if (size >= K) {
                q.pop();
            }
        }
    }
    return q;
}

// Accumulate the gradients contributed by the neighbours of p1[n][i1].
template <typename FloatAccessor, typename IndexAccessor>
void accumulate_neighbor_gradients(
    const FloatAccessor& p1_a,
    const FloatAccessor& p2_a,
    const IndexAccessor& idxs_a,
    const FloatAccessor& grad_dists_a,
    FloatAccessor& grad_p1_a,
    FloatAccessor& grad_p2_a,
    const int n,
    const int64_t i1,
    const int64_t num_neighbors,
    const int D,
    const int norm) {
    for (int64_t k = 0; k < num_neighbors; ++k) {
        const int64_t i2 = idxs_a[n][i1][k];
        // If the index is the pad value of -1 then ignore it
        if (i2 == -1) {
            continue;
        }
        for (int64_t d = 0; d < D; ++d) {
            float diff = 0.0;
            if (norm == 1) {
                float sign = (p1_a[n][i1][d] > p2_a[n][i2][d]) ? 1.0 : -1.0;
                diff       = grad_dists_a[n][i1][k] * sign;
            } else { // norm is 2 (default)
                diff = 2.0f * grad_dists_a[n][i1][k] *
                       (p1_a[n][i1][d] - p2_a[n][i2][d]);
            }
            grad_p1_a[n][i1][d] += diff;
            grad_p2_a[n][i2][d] += -1.0f * diff;
        }
    }
}

} // namespace

std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int norm,
    const int K) {
    const int N  = p1.size(0);
    const int P1 = p1.size(1);
    const int D  = p1.size(2);

    auto long_opts      = lengths1.options().dtype(torch::kInt64);
    torch::Tensor idxs  = torch::full({N, P1, K}, 0, long_opts);
    torch::Tensor dists = torch::full({N, P1, K}, 0, p1.options());

    auto p1_a       = p1.accessor<float, 3>();
    auto p2_a       = p2.accessor<float, 3>();
    auto lengths1_a = lengths1.accessor<int64_t, 1>();
    auto lengths2_a = lengths2.accessor<int64_t, 1>();
    auto idxs_a     = idxs.accessor<int64_t, 3>();
    auto dists_a    = dists.accessor<float, 3>();

    for (int n = 0; n < N; ++n) {
        const int64_t length1 = lengths1_a[n];
        const int64_t length2 = lengths2_a[n];
        for (int64_t i1 = 0; i1 < length1; ++i1) {
            // Use a priority queue to store (distance, index) tuples.
            auto q = nearest_neighbors(p1_a, p2_a, n, i1, length2, D, norm, K);
            while (!q.empty()) {
                auto t = q.top();
                q.pop();
                const int k       = q.size();
                dists_a[n][i1][k] = std::get<0>(t);
                idxs_a[n][i1][k]  = std::get<1>(t);
            }
        }
    }
    return std::make_tuple(idxs, dists);
}

// ------------------------------------------------------------- //
//                   Backward Operators                          //
// ------------------------------------------------------------- //

std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackwardCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const int norm,
    const at::Tensor& grad_dists) {
    const int N  = p1.size(0);
    const int P1 = p1.size(1);
    const int D  = p1.size(2);
    const int P2 = p2.size(1);
    const int K  = idxs.size(2);

    torch::Tensor grad_p1 = torch::full({N, P1, D}, 0, p1.options());
    torch::Tensor grad_p2 = torch::full({N, P2, D}, 0, p2.options());

    auto p1_a         = p1.accessor<float, 3>();
    auto p2_a         = p2.accessor<float, 3>();
    auto lengths1_a   = lengths1.accessor<int64_t, 1>();
    auto lengths2_a   = lengths2.accessor<int64_t, 1>();
    auto idxs_a       = idxs.accessor<int64_t, 3>();
    auto grad_dists_a = grad_dists.accessor<float, 3>();
    auto grad_p1_a    = grad_p1.accessor<float, 3>();
    auto grad_p2_a    = grad_p2.accessor<float, 3>();

    for (int n = 0; n < N; ++n) {
        const int64_t length1 = lengths1_a[n];
        int64_t length2       = lengths2_a[n];
        length2               = (length2 < K) ? length2 : K;
        for (int64_t i1 = 0; i1 < length1; ++i1) {
            accumulate_neighbor_gradients(
                p1_a, p2_a, idxs_a, grad_dists_a, grad_p1_a, grad_p2_a, n, i1, length2, D, norm);
        }
    }
    return std::make_tuple(grad_p1, grad_p2);
}
