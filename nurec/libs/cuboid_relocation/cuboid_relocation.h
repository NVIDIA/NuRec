// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace cuboid_relocation {
// Validate host-visible tensor metadata only; no GPU-to-CPU synchronization.
inline void check_vector(const torch::Tensor& value, const torch::Tensor& reference,
                         at::ScalarType dtype, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == reference.device(),
                name, " must be on the reference CUDA device");
    TORCH_CHECK(value.layout() == torch::kStrided && value.dim() == 1,
                name, " must be a strided vector");
    TORCH_CHECK(value.scalar_type() == dtype, name, " has an unexpected dtype");
    TORCH_CHECK(value.numel() < (int64_t{1} << 31), name, " must have fewer than 2^31 elements");
}

} // namespace cuboid_relocation

/** Compute per-track masks and metadata, overwriting all outputs.
 * Inputs are same-device CUDA tensors: order int64[N] permutation, offsets
 * int64[G+1] positive partitions of N, weights float32[N], dead bool[N].
 * Outputs: sorted_dead bool[N], metadata int64[G,2], group int64[N].
 * All element strides are supported. N < 2^31. Caller guarantees valid
 * permutation/partitions, nonnegative weights and nonaliasing outputs.
 * All-dead tracks have no live donor and retain their input dead mask.
 * No gradients; launch uses the input device's current CUDA stream.
 */
void prepare_tracks(const torch::Tensor& order, const torch::Tensor& offsets,
                    const torch::Tensor& weights, const torch::Tensor& dead,
                    const torch::Tensor& sorted_dead, const torch::Tensor& metadata,
                    const torch::Tensor& group);

/** Overwrite track-local inclusive double CDFs.
 * weights float32[C], offsets int64[G+1], cdf float64[C], same CUDA device;
 * strided tensors supported, C < 2^31. Caller guarantees offsets partition C
 * (empty tracks allowed), finite nonnegative weights, nonaliasing output.
 * Uses a 256-thread inclusive scan with double-precision accumulation.
 */
void track_cdf(const torch::Tensor& weights, const torch::Tensor& offsets,
               const torch::Tensor& cdf);

/** Overwrite S sampled donor IDs using supplied uniforms; no RNG is consumed.
 * Same-device CUDA tensors: rows int64[C], cdf float64[C], offsets int64[G+1],
 * draw_offsets int64[G+1], uniforms float64[S], sources int64[S]. Strides are
 * honored, C,S < 2^31. Caller guarantees partitions of C/S, track-local CDFs,
 * uniforms in [0,1), positive CDF totals for tracks with draws, and nonaliasing
 * output. Empty tracks must request zero draws. Uses the current CUDA stream.
 */
void sample_tracks(const torch::Tensor& rows, const torch::Tensor& cdf,
                   const torch::Tensor& offsets, const torch::Tensor& draw_offsets,
                   const torch::Tensor& uniforms, const torch::Tensor& sources);

/** Count global donor IDs, build keep, then apply gsplat relocation correction.
 * Same-device strided CUDA inputs follow prepare_tracks/sample_tracks contracts.
 * scales float32[N,3], binoms float32[B,B], B > 0. Caller guarantees valid row
 * IDs, groups, finite activated values and binomial coefficients, positive scales,
 * and zero-initialized occurrences int32[N]. Outputs must not alias inputs:
 * keep bool[N], revised_opacities float32[S], revised_scales float32[S,3].
 * No gradients or host readbacks; both launches use the current CUDA stream.
 */
void correct_relocation(const torch::Tensor& order, const torch::Tensor& sorted_dead,
                        const torch::Tensor& group, const torch::Tensor& metadata,
                        const torch::Tensor& sources, const torch::Tensor& opacities,
                        const torch::Tensor& scales, const torch::Tensor& binoms,
                        double min_opacity, const torch::Tensor& occurrences,
                        const torch::Tensor& keep, const torch::Tensor& revised_opacities,
                        const torch::Tensor& revised_scales);

/** Reject outward cuboid-local proposal components and update positions in place.
 * All tensors are strided CUDA tensors on the same device. Positions/proposals
 * float32[N,3], IDs int64[N], sorted unique tracks int64[G], centers/dimensions
 * float32[G,3], rotations float32[G,3,3]. Caller validates finite geometry,
 * positive dimensions, and that every row ID belongs to tracks. No gradients,
 * allocation or host readbacks; current CUDA stream. Proposal must not alias positions.
 */
void constrain_perturbation(const torch::Tensor& positions, const torch::Tensor& proposal,
                            const torch::Tensor& ids, const torch::Tensor& tracks,
                            const torch::Tensor& centers, const torch::Tensor& rotations,
                            const torch::Tensor& dimensions, double expansion);
