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

#pragma once

#include <torch/extension.h>

#include <tuple>

#include <ku/common_host.h>

// Compute indices of K nearest neighbors in pointcloud p2 to points
// in pointcloud p1.
//
// Args:
//    p1: FloatTensor of shape (N, P1, D) giving a batch of pointclouds each
//        containing P1 points of dimension D.
//    p2: FloatTensor of shape (N, P2, D) giving a batch of pointclouds each
//        containing P2 points of dimension D.
//    lengths1: LongTensor, shape (N,), giving actual length of each P1 cloud.
//    lengths2: LongTensor, shape (N,), giving actual length of each P2 cloud.
//    norm: int specifying the norm for the distance (1 for L1, 2 for L2)
//    K: int giving the number of nearest points to return.
//    version: Integer telling which implementation to use.
//
// Returns:
//    p1_neighbor_idx: LongTensor of shape (N, P1, K), where
//        p1_neighbor_idx[n, i, k] = j means that the kth nearest
//        neighbor to p1[n, i] in the cloud p2[n] is p2[n, j].
//        It is padded with zeros so that it can be used easily in a later
//        gather() operation.
//
//    p1_neighbor_dists: FloatTensor of shape (N, P1, K) containing the squared
//        distance from each point p1[n, p, :] to its K neighbors
//        p2[n, p1_neighbor_idx[n, p, k], :].

// CPU implementation.
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int norm,
    const int K);

// CUDA implementation
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int norm,
    const int K,
    const int version);

// Implementation which is exposed.
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdx(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int norm,
    const int K,
    const int version) {
    if (p1.is_cuda() || p2.is_cuda()) {
        CHECK_CUDA(p1);
        CHECK_CUDA(p2);
        return KNearestNeighborIdxCuda(
            p1, p2, lengths1, lengths2, norm, K, version);
    }
    return KNearestNeighborIdxCpu(p1, p2, lengths1, lengths2, norm, K);
}

// Compute gradients with respect to p1 and p2
//
// Args:
//    p1: FloatTensor of shape (N, P1, D) giving a batch of pointclouds each
//        containing P1 points of dimension D.
//    p2: FloatTensor of shape (N, P2, D) giving a batch of pointclouds each
//        containing P2 points of dimension D.
//    lengths1: LongTensor, shape (N,), giving actual length of each P1 cloud.
//    lengths2: LongTensor, shape (N,), giving actual length of each P2 cloud.
//    p1_neighbor_idx: LongTensor of shape (N, P1, K), where
//        p1_neighbor_idx[n, i, k] = j means that the kth nearest
//        neighbor to p1[n, i] in the cloud p2[n] is p2[n, j].
//        It is padded with zeros so that it can be used easily in a later
//        gather() operation. This is computed from the forward pass.
//    norm: int specifying the norm for the distance (1 for L1, 2 for L2)
//    grad_dists: FLoatTensor of shape (N, P1, K) which contains the input
//        gradients.
//
// Returns:
//    grad_p1: FloatTensor of shape (N, P1, D) containing the output gradients
//        wrt p1.
//    grad_p2: FloatTensor of shape (N, P2, D) containing the output gradients
//        wrt p2.

// CPU implementation.
std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackwardCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const int norm,
    const at::Tensor& grad_dists);

// CUDA implementation
std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackwardCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const int norm,
    const at::Tensor& grad_dists);

// Implementation which is exposed.
std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackward(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const int norm,
    const at::Tensor& grad_dists) {
    if (p1.is_cuda() || p2.is_cuda()) {
        CHECK_CUDA(p1);
        CHECK_CUDA(p2);
        return KNearestNeighborBackwardCuda(
            p1, p2, lengths1, lengths2, idxs, norm, grad_dists);
    }
    return KNearestNeighborBackwardCpu(
        p1, p2, lengths1, lengths2, idxs, norm, grad_dists);
}

// Utility to check whether a KNN version can be used.
//
// Args:
//    version: Integer in the range 0 <= version <= 3 indicating one of our
//        KNN implementations.
//    D: Number of dimensions for the input and query point clouds
//    K: Number of neighbors to be found
//
// Returns:
//    Whether the indicated KNN version can be used.
bool KnnCheckVersion(int version, const int64_t D, const int64_t K);
