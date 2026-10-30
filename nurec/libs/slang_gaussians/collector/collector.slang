// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Gaussian Parameter Collection Kernels
//
// This module implements GPU kernels for collecting, transforming, and concatenating
// Gaussian parameters from multiple layers into unified output tensors. The kernels
// support various layer types with different activation functions and transformations.
//
// The module uses a task-based architecture where each operation (e.g., rotation
// normalization, scale exponentiation) is implemented as an ITask. Multiple tasks
// are fused into a single kernel via the run_tasks() template function, enabling
// efficient fused execution with minimal kernel launch overhead.

module collector;

#include <slang_gaussians/common/ray_intersect.slang>
#include <slang_gaussians/collector/timestamp_cache.slang>
#include <slang_gaussians/common/sigmoid.slang>

namespace utils
{

// Get the 2D index for a tensor with rows of N components.
[ForceInline]
uint2 get_index<let N : int>(no_diff uint index)
{
    const uint index_i = index / N;
    const uint index_j = index - index_i * N;
    return uint2(index_i, index_j);
}

// Dense launches traverse one block's rows before moving to another tile.
// Using the active width also gives a partial final block exactly one writer
// per component.
//
// Requires a 1-D grid and 1-D blocks, with gridDim.x == div_up(count, blockDim.x)
// (one block per tile, no oversubscription or grid-stride loop). The launcher in
// collector_slang.py supplies this geometry. tid must equal
// cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x and satisfy tid < count;
// run_tasks_dense enforces this before dispatching tasks. Thus block_start < count
// whenever this helper is reached, preventing unsigned count - block_start
// underflow. N must be positive and component must be in [0, N).
// Returns the (row, column) index within the count-by-N tensor for this lane
// and component iteration.
[ForceInline]
uint2 get_block_index<let N : int>(no_diff uint tid, no_diff uint count, no_diff uint component)
{
    const uint block_start = cudaBlockIdx().x * cudaBlockDim().x;
    const uint active_width = min(cudaBlockDim().x, count - block_start);
    return get_index<N>(block_start * N + component * active_width + tid - block_start);
}

// This function passes a value through, but only optionally allows the
// gradient to pass through.
[ForwardDerivative(__pass_forward)]
[BackwardDerivative(__pass_backward)]
[Differentiable]
[ForceInline]
__generic<T: IDifferentiable>
T pass(T x, no_diff bool pass_gradient) {
    return x;
}

[ForceInline]
__generic<T: IDifferentiable>
DifferentialPair<T> __pass_forward(
    DifferentialPair<T> x, no_diff bool pass_gradient
    ) {
    if (pass_gradient) {
        return x;
    }
    else {
        return diffPair(x.p, T.dzero());
    }
}

[ForceInline]
__generic<T: IDifferentiable>
void __pass_backward(
    inout DifferentialPair<T> x, no_diff bool pass_gradient, T.Differential dOut
    ) {
    if (!pass_gradient) {
        dOut = T.dzero();
    }
    x = diffPair(x.p, dOut);
}

}; // namespace utils.


// Which layer-local row a segmented launch slot works on.
//
// `segment_starts[s]` is the first row of window s and `window_size_prefix[s]` the first
// slot it owns (exclusive prefix of the window sizes, with
// `window_size_prefix[n_segments] == active slots`). Dense kernels do not construct or
// receive this type.
public struct LayerSegments
{
    public TensorView<int32_t> segment_starts;
    public TensorView<int32_t> window_size_prefix;
    public uint n_segments;

    uint row(uint slot)
    {
        // find_surrounding_indices performs the endpoint checks required by the search
        // strategy's strict in-range contract. Since prefix[0] == 0 and slot is less
        // than prefix[n_segments], its lower result is the owning window both at an
        // exact window boundary and inside a window.
        typedef search::BinarySearch<int32_t, int32_t> SearchStrategy;
        let prefix = ImmutablePtr<int32_t>(window_size_prefix.data_ptr_at(0));
        const var surrounding = search::find_surrounding_indices<SearchStrategy>(
            int32_t(slot), prefix, int32_t(n_segments) + 1);
        const uint window = uint(surrounding._0);
        return uint(segment_starts.load(window)) + (slot - uint(prefix[window]));
    }
};

// Layout is specialized at compilation. Sparse tasks receive an already mapped row;
// dense tasks keep the original coalesced component traversal.
public interface ICollectRows
{
    uint row(uint tid);
    uint2 component<let N : int>(uint i, uint tid, uint count);
};

public struct DenseRows : ICollectRows
{
    uint row(uint tid) { return tid; }
    uint2 component<let N : int>(uint i, uint tid, uint count)
    {
        return utils::get_block_index<N>(tid, count, i);
    }
};

public struct SparseRows : ICollectRows
{
    uint row_index;
    [ForceInline]
    uint row(uint tid) { return row_index; }
    uint2 component<let N : int>(uint i, uint tid, uint count)
    {
        return uint2(row_index, i);
    }
};

// Tasks share one implementation; only row/component addressing varies.
public interface ITask
{
    [Differentiable]
    void run<Rows : ICollectRows>(
        no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows);
};

// Keep the generic method call outside the variadic expansion.
[Differentiable]
void run_task<Rows : ICollectRows, T : ITask>(
    no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows, T task)
{
    task.run(tid, count, out_offset, rows);
}

[Differentiable]
public void run_tasks_dense<each T>(
    no_diff uint out_offset, no_diff uint count, expand each T args) where T : ITask
{
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }
    expand run_task(tid, count, out_offset, DenseRows(), each args);
}

public void run_tasks_segmented<each T>(
    no_diff uint out_offset, no_diff uint count, no_diff LayerSegments segments,
    expand each T args) where T : ITask
{
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }
    SparseRows rows;
    rows.row_index = segments.row(tid);
    expand run_task(tid, count, out_offset, rows, each args);
}

//
// Implementations for base layer.
//

// A couple of helpers for simple things like copy with single function applied to each value.
interface IUnaryFunction
{
    [Differentiable]
    static float apply(float x);
};

struct CopyFunction : IUnaryFunction
{
    [Differentiable]
    static float apply(float x) { return x; }
};

struct ExpFunction : IUnaryFunction
{
    [Differentiable]
    static float apply(float x) { return exp(x); }
};

struct SigmoidFunction : IUnaryFunction
{
    [Differentiable]
    static float apply(float x) { return sigmoid::standard(x); }
};

// Dense launches visit input rows [block_start, block_start + active_width) in
// block-local order. Sparse launches keep the segment's mapped row/component
// pair. Physical addresses still follow each tensor's strides.
//
// OPTME: In the case of the pure copy, we could do vectorized float4 load.
// Manual backward for apply_to_buffer to avoid Slang's autodiff unrolling
// which causes register pressure blowup.
// OffsetDim: column offset added to the destination index (0 for simple copy).
[BackwardDerivative(apply_to_buffer_bwd)]
void apply_to_buffer<UnaryFunction, let N : int, let OffsetDim : int, A : IDiffTensorWrapper, Rows : ICollectRows>(
    no_diff DiffTensorView<float, A> output, no_diff DiffTensorView<float, A> input,
    no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows
    ) where UnaryFunction : IUnaryFunction
{
    [ForceUnroll]
    for (int i = 0; i < N; ++i) {
        const uint2 index = rows.component<N>(uint(i), tid, count);
        const uint2 out_index = uint2(index.x + out_offset, uint(OffsetDim) + index.y);
        output.primal.store(out_index, UnaryFunction.apply(input.primal.load(index)));
    }
}

void apply_to_buffer_bwd<UnaryFunction, let N : int, let OffsetDim : int, A : IDiffTensorWrapper, Rows : ICollectRows>(
    no_diff DiffTensorView<float, A> output, no_diff DiffTensorView<float, A> input,
    no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows
    ) where UnaryFunction : IUnaryFunction
{
    // collect() permits backward only for dense launches.
    [ForceUnroll]
    for (int i = 0; i < N; ++i) {
        const uint2 index = utils::get_block_index<N>(tid, count, i);
        const uint2 out_index = uint2(index.x + out_offset, uint(OffsetDim) + index.y);
        float d_out = output.diff.storeOnce_backward<float, 2>(out_index);
        // Chain rule: use Slang's autodiff for the unary function derivative.
        var dp = diffPair(input.primal.load(index));
        bwd_diff(UnaryFunction.apply)(dp, d_out);
        input.diff.loadOnce_backward<float, 2>(index, dp.d);
    }
};

// It would be nice to be able to define a generic implementation of ITask
// which would take a UnaryFunction and N as a parameter, and simply typedef to it,
// but SlangTorch does not export the parameters to it.
public struct Collector_Copy<let N : int> : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        apply_to_buffer<CopyFunction, N, 0>(output, input, tid, count, out_offset, rows);
    }
};

// Positions.
// Use Collector_Copy

// Rotations.
public struct RotationsCollector_Normalize : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        const uint row = rows.row(tid);
        const uint out_index = out_offset + row;

        // OPTME: We could do vectorized float4 load.
        var rot = input.loadVecOnce<4>(uint2(row,0));
        rot = quaternion::normalizeSafe(rot);
        output.storeVecOnce<4>(uint2(out_index,0), rot);
    }
};

// Scales.
public struct ScalesCollector_Exp : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        apply_to_buffer<ExpFunction, 3, 0>(output, input, tid, count, out_offset, rows);
    }
};

// Densities.
public struct DensitiesCollector_Sigmoid : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        apply_to_buffer<SigmoidFunction, 1, 0>(output, input, tid, count, out_offset, rows);
    }
};

// Extra signal.
// Use Collector_Copy

// Camera extra signal.
// Use Collector_Copy

// Lidar extra signal.
// Use Collector_Copy


// Output view for the SH buffer. Lets one generic helper write either fp32 or fp16 by
// dispatching store() through a conformance, so we don't duplicate the helper per dtype.
interface IOutputView
{
    [Differentiable]
    void store(no_diff uint2 index, float value);
}

// fp32 conformance
extension DiffTensorView : IOutputView
{
    [Differentiable]
    void store(no_diff uint2 index, float value)
    {
        this.storeOnce(index, value);
    }
}

// fp16 conformance. Two plain TensorView<half>s with a hand-written [BackwardDerivative]
// sidestep slangtorch's DiffTensorView<half> binding bug. Store is non-atomic: each
// (gaussian, component) cell has a unique writer per launch.
struct HalfDiffView : IOutputView
{
    TensorView<half> primal;
    TensorView<half> diff;

    [BackwardDerivative(store_bwd)]
    void store(no_diff uint2 index, float value)
    {
        primal.store(index, half(value));
    }

    void store_bwd(no_diff uint2 index, inout DifferentialPair<float> dpValue)
    {
        float g = float(diff.load(index));
        dpValue = diffPair(dpValue.p, g);
    }
}


//
// Implementations for spherical harmonics layer.
//

// Marks the following function with PreferRecompute, instructing the Slang autodiff system
// to recompute its body from scratch during the backward pass, rather than storing any
// intermediate outputs or branch decisions in the checkpoint state. This reduces memory
// usage at the cost of additional computation. SideEffectBehavior.Allow indicates that
// recomputation is permitted even if the function may have side effects.
[PreferRecompute(SideEffectBehavior.Allow)]
[Differentiable]
float load_spherical_feature<let AlbedoDim : int>(
    DiffTensorView albedo,
    DiffTensorView specular,
    no_diff uint2 index)
{
    if (index.y < AlbedoDim) {
        return albedo.loadOnce(index);
    }
    else {
        return specular.loadOnce(uint2(index.x, index.y - AlbedoDim));
    }
}

[Differentiable]
void compute_spherical_features_copy<let AlbedoDim : int, let SpecularDim : int, OutputView : IOutputView, Rows : ICollectRows>(
    no_diff uint tid,
    no_diff uint count,
    no_diff uint out_offset,
    no_diff Rows rows,
    DiffTensorView albedo,
    DiffTensorView specular,
    OutputView output)
{
    static const var nb_components = AlbedoDim + SpecularDim;

    [ForceUnroll]
    for (int i = 0; i < nb_components; ++i) {
        const var index = rows.component<nb_components>(uint(i), tid, count);
        const var out_index = uint2(index.x + out_offset, index.y);
        output.store(out_index, load_spherical_feature<AlbedoDim>(albedo, specular, index));
    }
}

public struct SphericalFeaturesCollector_Copy<let AlbedoDim : int, let SpecularDim : int> : ITask
{
    DiffTensorView albedo;
    DiffTensorView specular;
    DiffTensorView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        compute_spherical_features_copy<AlbedoDim, SpecularDim>(
            tid, count, out_offset, rows, albedo, specular, output);
    }
};

// fp16 output variant
public struct SphericalFeaturesCollector_Copy_FP16<let AlbedoDim : int, let SpecularDim : int> : ITask
{
    DiffTensorView albedo;
    DiffTensorView specular;
    HalfDiffView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        compute_spherical_features_copy<AlbedoDim, SpecularDim>(
            tid, count, out_offset, rows, albedo, specular, output);
    }
};

// Interface for time embedding strategies.
// 
// Converts integer timestamps (microseconds) to normalized floating-point
// values suitable for Fourier features or hash grid inputs.
public interface ITimeEmbedding
{
    [Differentiable]
    public float embed(no_diff uint tid, no_diff int64_t timestamp);
};

public struct IndividualRemapTimeInputEmbedding : ITimeEmbedding
{
    no_diff TensorView<int32_t> instance_idx;
    no_diff TensorView<int64_t> timestamps_ranges;
    no_diff float remap_min;
    no_diff float remap_max;

    [Differentiable]
    [NoDiffThis]
    public float embed(no_diff uint tid, no_diff int64_t timestamp)
    {
        var instance_id = instance_idx[tid];
        var range_min = timestamps_ranges[uint2(instance_id, 0)];
        var range_max = timestamps_ranges[uint2(instance_id, 1)];
        // This could be done with a single fma if the ranges were recomputed differently.
        const float ratio = (timestamp - range_min) / float(range_max - range_min);
        const float time_embed = clamp(ratio, 0, 1) * (remap_max - remap_min) + remap_min;
        return time_embed;
    }
};

public struct HolisticRemapTimeInputEmbedding : ITimeEmbedding
{
    no_diff int64_t timestamps_us_min;
    no_diff int64_t timestamps_us_max;
    no_diff float remap_min;
    no_diff float remap_max;

    [Differentiable]
    [NoDiffThis]
    public float embed(no_diff uint tid, no_diff int64_t timestamp)
    {
        // This could be done with a single fma if the ranges were recomputed differently.
        const float ratio = (timestamp - timestamps_us_min) / float(timestamps_us_max - timestamps_us_min);
        const float time_embed = ratio * (remap_max - remap_min) + remap_min;
        return time_embed;
    }
};


// Individual step-function time embedding with per-instance ranges (n_dims=1 only)
// Based on NSC paper (https://arxiv.org/abs/2306.07970)
//
// Parameters:
//   u: [n_tracks, 1, n_steps] - learnable step positions
//   timestamps_ranges: [n_tracks, 2] - per-track [min, max] timestamps in microseconds
//   beta: stepness parameter (in microseconds, gets scaled by time range)
//
// Output: single float per Gaussian (mean over n_steps)
public struct IndividualStepTimeInputEmbedding<let NumSteps : int> : ITimeEmbedding
{
    no_diff TensorView<int32_t> instance_idx;       // [num_gaussians]
    no_diff TensorView<int64_t> timestamps_ranges;  // [num_tracks, 2]
    DiffTensorView u;                                // [num_tracks, 1, n_steps]
    no_diff TensorView<float> beta;                 // stepness in microseconds

    [Differentiable]
    public float embed(no_diff uint tid, no_diff int64_t timestamp)
    {
        let instance_id = instance_idx[tid];
        let range_min = timestamps_ranges[uint2(instance_id, 0)];
        let range_max = timestamps_ranges[uint2(instance_id, 1)];
        
        // Compute ratio
        let time_range = float(range_max - range_min);
        let ratio = float(timestamp - range_min) / time_range;
        
        // Compute beta_scaled = beta / time_range (matching Python exactly)
        // Then clamp |beta_scaled| to avoid division by zero
        let beta_scaled = beta[0] / time_range;
        let abs_beta = max(abs(beta_scaled), 1e-3f);
        
        // Compute mean over steps for n_dims=1
        // Python: output = ratio[:, None, None] - self.u[instance_idx]
        //         time_emb = output.mean(dim=-1)  # shape [N, n_dims]
        float sum = 0.0f;
        
        [ForceUnroll]
        for (int s = 0; s < NumSteps; ++s)
        {
            // Load u[instance_id, 0, s]
            // Using loadEx because multiple Gaussians share the same instance_id,
            // so gradients need to be accumulated atomically during backprop
            let u_val = u.loadEx(uint3(instance_id, 0, s));
            
            let diff = ratio - u_val;
            
            // Step function:
            //   if diff <= 0: 0.5 * exp(diff / abs_beta)
            //   else: 1 - 0.5 * exp(-diff / abs_beta)
            float step_val;
            if (diff <= 0.0f) {
                step_val = 0.5f * exp(diff / abs_beta);
            } else {
                step_val = 1.0f - 0.5f * exp(-diff / abs_beta);
            }
            sum += step_val;
        }
        
        // Mean over steps
        return sum / float(NumSteps);
    }
}

// Interface for fetching timestamps based on different query strategies.
public interface ITimestampsFetcher {
    int64_t fetch(no_diff uint track_id);
}

public struct TimestampsGlobalFetcher : ITimestampsFetcher {
    // Declare this member public so constructor can be generated automatically.
    public no_diff int64_t timestamp;

    int64_t fetch(no_diff uint track_id)
    {
        return timestamp;
    }
};

public struct TimestampsPerTrackFetcher : ITimestampsFetcher {
    TensorView<int64_t> timestamps;

    int64_t fetch(no_diff uint track_id)
    {
        return timestamps.load(track_id);
    }
};

// Instance embedding
public struct WeightedInstanceInputEmbedding<let EmbeddingDim : int>
{
    no_diff TensorView<int32_t> instance_idx;    // [N]
    DiffTensorView embedding_weights;            // [num_instances, embedding_dim]
    DiffTensorView output;                       // Output tensor
    
    // Embed to a specific output row starting at given column
    // col_start specifies where the embedding begins (e.g., 3 if after xyz)
    [Differentiable]
    public void embed_to_row(no_diff uint tid, no_diff uint output_row, no_diff uint col_start)
    {
        let instance_id = instance_idx[tid];
        [ForceUnroll]
        for (int i = 0; i < EmbeddingDim; ++i) {
            // Use loadEx() instead of loadOnce() because multiple samples may have the same
            // instance_id, so gradients need to be accumulated atomically during backprop.
            let emb_value = embedding_weights.loadEx(uint2(instance_id, i));
            
            // Use storeOnce() because each thread writes to a unique output location (unique output_row).
            output.storeOnce(uint2(output_row, col_start + i), emb_value);
        }
    }
};

[Differentiable]
void compute_spherical_features_fourier<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    TimestampsFetcher : ITimestampsFetcher,
    TimeEmbedding : ITimeEmbedding,
    OutputView: IOutputView,
    Rows : ICollectRows,
    >(
    no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows,
    TimestampsFetcher timestamps_fetcher,
    TimeEmbedding time_embedding,
    DiffTensorView albedo,
    DiffTensorView specular,
    OutputView output,
    )
{
    // We do the Fourier transform in double-precision since it's sensitive.
    // We can trade-off quality for speed later if we prefer.
    typedef double scalar_t;

    const uint row = rows.row(tid);
    const int64_t timestamp = timestamps_fetcher.fetch(row);
    const scalar_t time_embed = (scalar_t)time_embedding.embed(row, timestamp);

    scalar_t feature_albedo[AlbedoDim];
    [ForceUnroll]
    for (int i = 0; i < AlbedoDim; ++i) {
        feature_albedo[i] = 0;
    }

    // On some hardware, directly evaluating sin/cos(PI * time_embed * j) may be
    // faster than recurrence for small FourierDim; A40 was faster for dims 2..8.
    // j=0 is cos(0), so its Fourier weight is exactly 1.
    [ForceUnroll]
    for (int i = 0; i < AlbedoDim; ++i) {
        feature_albedo[i] += (scalar_t)albedo.loadOnce(uint3(row, 0, i));
    }

    // Fourier band-0 theta must stay in genuine fp64. Slang types unsuffixed
    // float literals (and constants::pi<scalar_t>()) as fp32 (HLSL heritage),
    // which silently truncated pi to fp32 precision even when scalar_t is double.
    // Spell out a custom full-precision fp64 pi with the `lf` suffix instead.
    const scalar_t PI_FP64 = 3.141592653589793lf;
    const scalar_t theta = 2.0lf * PI_FP64 * time_embed;
    const scalar_t sin_theta = sin(theta);
    const scalar_t cos_theta = cos(theta);
    scalar_t sin_k = 0.0;
    scalar_t cos_k = 1.0;

    [ForceUnroll]
    for (int k = 1; k <= FourierDim / 2; ++k) {
        const scalar_t next_sin = sin_k * cos_theta + cos_k * sin_theta;
        const scalar_t next_cos = cos_k * cos_theta - sin_k * sin_theta;
        sin_k = next_sin;
        cos_k = next_cos;

        const int sin_j = 2 * k - 1;
        [ForceUnroll]
        for (int i = 0; i < AlbedoDim; ++i) {
            feature_albedo[i] += sin_k * (scalar_t)albedo.loadOnce(uint3(row, sin_j, i));
        }

        const int cos_j = 2 * k;
        if (cos_j < FourierDim) {
            [ForceUnroll]
            for (int i = 0; i < AlbedoDim; ++i) {
                feature_albedo[i] += cos_k * (scalar_t)albedo.loadOnce(uint3(row, cos_j, i));
            }
        }
    }

    const uint out_index = out_offset + row;
    [ForceUnroll]
    for (int i = 0; i < AlbedoDim; ++i) {
        output.store(uint2(out_index, i), (float)feature_albedo[i]);
    }

    [ForceUnroll]
    for (int i = 0; i < SpecularDim; ++i) {
        const uint2 spec_index = rows.component<SpecularDim>(uint(i), tid, count);
        const uint2 out_idx = uint2(spec_index.x + out_offset, uint(AlbedoDim) + spec_index.y);
        output.store(out_idx, specular.loadOnce(spec_index));
    }
}

// slangtorch has problems with kernel parameters with generic parts.  If
// the same struct is instantiated multiple times with a different concrete
// type in the same compile unit, slangtorch will generate more than one
// functions with the same name.  To workaround this, we create 2 separate
// types, one per concrete embedding.
public struct SphericalFeaturesCollector_Fourier_Individual<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    > : ITask
{
    IndividualRemapTimeInputEmbedding time_embedding;
    no_diff int64_t timestamp;
    DiffTensorView albedo;
    DiffTensorView specular;
    DiffTensorView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        let timestamps_fetcher = TimestampsGlobalFetcher(timestamp);
        compute_spherical_features_fourier<AlbedoDim, SpecularDim, FourierDim>(
            tid, count, out_offset, rows, timestamps_fetcher, time_embedding, albedo, specular, output,
        );
    }
};

public struct SphericalFeaturesCollector_Fourier_Holistic<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    > : ITask
{
    HolisticRemapTimeInputEmbedding time_embedding;
    no_diff int64_t timestamp;
    DiffTensorView albedo;
    DiffTensorView specular;
    DiffTensorView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        let timestamps_fetcher = TimestampsGlobalFetcher(timestamp);
        compute_spherical_features_fourier<AlbedoDim, SpecularDim, FourierDim>(
            tid, count, out_offset, rows, timestamps_fetcher, time_embedding, albedo, specular, output,
        );
    }
};

public struct SphericalFeaturesCollector_Fourier_IndividualStep<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    let NumSteps : int,
    > : ITask
{
    IndividualStepTimeInputEmbedding<NumSteps> time_embedding;
    no_diff int64_t timestamp;
    DiffTensorView albedo;
    DiffTensorView specular;
    DiffTensorView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        let timestamps_fetcher = TimestampsGlobalFetcher(timestamp);
        compute_spherical_features_fourier<AlbedoDim, SpecularDim, FourierDim>(
            tid, count, out_offset, rows, timestamps_fetcher, time_embedding, albedo, specular, output,
        );
    }
};

// fp16 output variants of the three Fourier task structs.

public struct SphericalFeaturesCollector_Fourier_Individual_FP16<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    > : ITask
{
    IndividualRemapTimeInputEmbedding time_embedding;
    no_diff int64_t timestamp;
    DiffTensorView albedo;
    DiffTensorView specular;
    HalfDiffView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        let timestamps_fetcher = TimestampsGlobalFetcher(timestamp);
        compute_spherical_features_fourier<AlbedoDim, SpecularDim, FourierDim>(
            tid, count, out_offset, rows, timestamps_fetcher, time_embedding, albedo, specular, output,
        );
    }
};

public struct SphericalFeaturesCollector_Fourier_Holistic_FP16<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    > : ITask
{
    HolisticRemapTimeInputEmbedding time_embedding;
    no_diff int64_t timestamp;
    DiffTensorView albedo;
    DiffTensorView specular;
    HalfDiffView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        let timestamps_fetcher = TimestampsGlobalFetcher(timestamp);
        compute_spherical_features_fourier<AlbedoDim, SpecularDim, FourierDim>(
            tid, count, out_offset, rows, timestamps_fetcher, time_embedding, albedo, specular, output,
        );
    }
};

public struct SphericalFeaturesCollector_Fourier_IndividualStep_FP16<
    let AlbedoDim : int,
    let SpecularDim : int,
    let FourierDim : int,
    let NumSteps : int,
    > : ITask
{
    IndividualStepTimeInputEmbedding<NumSteps> time_embedding;
    no_diff int64_t timestamp;
    DiffTensorView albedo;
    DiffTensorView specular;
    HalfDiffView output;

    [Differentiable]
    public void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        let timestamps_fetcher = TimestampsGlobalFetcher(timestamp);
        compute_spherical_features_fourier<AlbedoDim, SpecularDim, FourierDim>(
            tid, count, out_offset, rows, timestamps_fetcher, time_embedding, albedo, specular, output,
        );
    }
};

//
// Implementations for dynamic rigid layer.
//

#include <slang_gaussians/collector/tracks_calib/interface.slang>
#include <slang_gaussians/collector/tracks_calib/direct.slang>
#include <slang_gaussians/collector/tracks_calib/elevation_aware_unicycle.slang>

// Track-culling pre-pass.
//
// Per-track thread decides whether the track is potentially visible to
// the active sensor in this frame.  Active tracks atomically claim a
// slot in `output_active_tracks_indices` via `InterlockedAdd` on
// `output_active_count[0]`, and write their own track index there.
// Culled tracks write nothing.  After the kernel runs,
// `output_active_count[0]` holds the compacted active-track count and
// `output_active_tracks_indices[0..count)` holds those indices in
// arbitrary (atomic-add-determined) order.
//
// There are two entry points so the time-only path does not accept a
// dummy frustum ray tensor:
//   (1) `build_active_tracks_indices_time_only`: time-range overlap only.
//       This is sufficient for a 360-degree/lidar bundle when all lidar
//       tiles may render, since any track in time-range may be hit by
//       some lidar ray.
//   (2) `build_active_tracks_indices_time_frustum`: time-range overlap plus
//       a 4-plane frustum test on a conservative swept bounding sphere.
//       Each track thread re-derives the frustum planes from the 4 FOV-corner
//       rays, keeping the pre-pass to a single kernel launch.
//
// These kernels are not differentiable.
[ForceInline]
float3 load_interpolated_track_center(
    TensorView<float> tracks_poses,
    ImmutablePtr<int64_t> track_timestamps_ptr,
    int32_t track_start,
    Tuple<int32_t, int32_t, bool> indices,
    int64_t timestamp,
) {
    if (indices._0 == indices._1) {
        return tracks_poses.loadVec<3>(uint2(track_start + indices._0, 0));
    }

    let pose_left = transform::SE3.from_vec(
        tracks_poses.loadVec<3>(uint2(track_start + indices._0, 0)),
        tracks_poses.loadVec<4>(uint2(track_start + indices._0, 3)),
    );
    let pose_right = transform::SE3.from_vec(
        tracks_poses.loadVec<3>(uint2(track_start + indices._1, 0)),
        tracks_poses.loadVec<4>(uint2(track_start + indices._1, 3)),
    );
    let time_left = track_timestamps_ptr[indices._0];
    let time_right = track_timestamps_ptr[indices._1];
    let alpha = (timestamp - time_left) / float(time_right - time_left);
    let pose = transform::interpolate(pose_left, pose_right, alpha);
    let pose_vec = pose.to_vec();
    return pose_vec._0;
}

[CUDAKernel]
[AutoPyBindCUDA]
void build_active_tracks_indices_time_only(
    TensorView<int32_t> output_active_tracks_indices,
    TensorView<int32_t> output_active_count,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    no_diff int64_t frame_start_timestamp,
    no_diff int64_t frame_end_timestamp,
) {
    let nb_tracks = tracks_packinfo.size(0);
    let tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= nb_tracks) {
        return;
    }

    let track_start = tracks_packinfo.load(uint2(tid, 0));
    let track_length = tracks_packinfo.load(uint2(tid, 1));
    if (track_length <= 0) {
        return;
    }

    let track_timestamps_ptr = ImmutablePtr<int64_t>(tracks_timestamps.data_ptr_at(track_start));
    let track_t0 = track_timestamps_ptr[0];
    let track_tN = track_timestamps_ptr[track_length - 1];

    if (track_t0 > frame_end_timestamp || track_tN < frame_start_timestamp) {
        return;
    }

    int32_t my_idx;
    output_active_count.InterlockedAdd(uint(0), int32_t(1), my_idx);
    output_active_tracks_indices[my_idx] = int32_t(tid);
}

[CUDAKernel]
[AutoPyBindCUDA]
// Used by the camera timestamp path, and by the lidar timestamp path when
// active tile rendering supplies camera frustum rays for culling.
void build_active_tracks_indices_time_frustum(
    TensorView<int32_t> output_active_tracks_indices,
    TensorView<int32_t> output_active_count,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    TensorView<float> frustum_rays,
    no_diff int64_t frame_start_timestamp,
    no_diff int64_t frame_end_timestamp,
    no_diff int32_t frustum_rays_height,
    no_diff int32_t frustum_rays_width,
    no_diff float frustum_margin_rad,
) {
    let nb_tracks = tracks_packinfo.size(0);
    let tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= nb_tracks) {
        return;
    }

    let track_start = tracks_packinfo.load(uint2(tid, 0));
    let track_length = tracks_packinfo.load(uint2(tid, 1));
    if (track_length <= 0) {
        return;
    }

    let track_timestamps_ptr = ImmutablePtr<int64_t>(tracks_timestamps.data_ptr_at(track_start));
    let track_t0 = track_timestamps_ptr[0];
    let track_tN = track_timestamps_ptr[track_length - 1];

    // Stage 1: time-range overlap with the frame exposure window.
    if (track_t0 > frame_end_timestamp || track_tN < frame_start_timestamp) {
        return;
    }

    // Stage 2: conservative spatial cull. Use a sphere with radius equal
    // to the cuboid half diagonal, swept piecewise between pose centers.
    //
    //   pose0 ---- start ---- pose1 ---- pose2 ---- end ---- pose3
    //                |----------|----------|---------|
    //                capsule 0  capsule 1  capsule 2
    //
    // Frame endpoints are interpolated; crossed keyframes split the sweep.
    // A track is culled when every segment is fully outside the same plane.
    typedef search::BinarySearch<int64_t, int32_t> SearchStrategy;
    let sweep_start_indices = search::find_surrounding_indices<SearchStrategy>(
        frame_start_timestamp, track_timestamps_ptr, track_length);
    let sweep_end_indices = search::find_surrounding_indices<SearchStrategy>(
        frame_end_timestamp, track_timestamps_ptr, track_length);

    let dim = cuboids_dimensions.loadVec<3>(uint2(tid, 0));
    let half = dim * 0.5f;
    let sphere_radius = length(half);
    let sphere_start_center_world = load_interpolated_track_center(
        tracks_poses, track_timestamps_ptr, track_start, sweep_start_indices, frame_start_timestamp);
    let sphere_end_center_world = load_interpolated_track_center(
        tracks_poses, track_timestamps_ptr, track_start, sweep_end_indices, frame_end_timestamp);

    int32_t first_interior_pose_idx;
    if (sweep_start_indices._0 == sweep_start_indices._1) {
        first_interior_pose_idx = sweep_start_indices._0 + 1;
    }
    else {
        first_interior_pose_idx = sweep_start_indices._1;
    }

    int32_t end_interior_pose_idx_exclusive;
    if (sweep_end_indices._0 == sweep_end_indices._1) {
        end_interior_pose_idx_exclusive = sweep_end_indices._0;
    }
    else {
        end_interior_pose_idx_exclusive = sweep_end_indices._0 + 1;
    }

    // Derive the frustum apex + 4 corner rays from the FOV corners of
    // the camera image grid.  frustum_rays is laid out as the flattening of an
    // [H, W, 6] tensor (B=1).  The 4 corners are the first/last column
    // of the first/last row.
    let idx_TL = uint(0);
    let idx_TR = uint(frustum_rays_width - 1);
    let idx_BR = uint(frustum_rays_height * frustum_rays_width - 1);
    let idx_BL = uint((frustum_rays_height - 1) * frustum_rays_width);
    let o_TL = frustum_rays.loadVec<3>(uint2(idx_TL, 0));
    let o_TR = frustum_rays.loadVec<3>(uint2(idx_TR, 0));
    let o_BR = frustum_rays.loadVec<3>(uint2(idx_BR, 0));
    let o_BL = frustum_rays.loadVec<3>(uint2(idx_BL, 0));
    let d_TL = normalize(frustum_rays.loadVec<3>(uint2(idx_TL, 3)));
    let d_TR = normalize(frustum_rays.loadVec<3>(uint2(idx_TR, 3)));
    let d_BR = normalize(frustum_rays.loadVec<3>(uint2(idx_BR, 3)));
    let d_BL = normalize(frustum_rays.loadVec<3>(uint2(idx_BL, 3)));

    let apex = (o_TL + o_TR + o_BR + o_BL) * 0.25f;
    let forward = normalize(d_TL + d_TR + d_BR + d_BL);

    // 4 plane normals (outward), built from cyclic cross-products of
    // adjacent corner directions.  Order: top (TL to TR), right (TR to BR),
    // bottom (BR to BL), left (BL to TL).
    float3 plane_dirs[5] = { d_TL, d_TR, d_BR, d_BL, d_TL };

    let cos_m = cos(frustum_margin_rad);
    let sin_m = sin(frustum_margin_rad);
    for (int p = 0; p < 4; ++p) {
        let c_a = plane_dirs[p];
        let c_b = plane_dirs[p + 1];
        let opposite = plane_dirs[(p + 2) % 4]; // a corner not on this plane
        var n = normalize(cross(c_b, c_a));
        // Outward: a corner on the OPPOSITE side of the cone should be
        // inside (dot(n, opposite) <= 0).  Flip if not.
        if (dot(n, opposite) > 0.0f) {
            n = -n;
        }
        // Widen by `frustum_margin_rad`: rotate n outward (away from
        // forward) in the (n, forward) plane.  perp is the in-plane
        // unit vector perpendicular to n pointing toward forward; we
        // subtract it to tilt n away from forward.
        let perp_raw = forward - dot(forward, n) * n;
        let perp_len = length(perp_raw);
        if (perp_len > 1e-6f) {
            let perp = perp_raw / perp_len;
            n = cos_m * n - sin_m * perp;
        }

        // Cull by this plane only if the whole swept capsule chain is outside.
        // For a plane, this means every start/interior/end center is farther
        // than `sphere_radius` outside the plane.
        // Tangency counts as intersection for conservative culling.
        var all_centers_outside = dot(sphere_start_center_world - apex, n) > sphere_radius;
        for (int32_t pose_idx = first_interior_pose_idx;
             pose_idx < end_interior_pose_idx_exclusive && all_centers_outside;
             ++pose_idx) {
            let current_center_world = tracks_poses.loadVec<3>(uint2(track_start + pose_idx, 0));
            all_centers_outside = dot(current_center_world - apex, n) > sphere_radius;
        }

        if (all_centers_outside) {
            all_centers_outside = dot(sphere_end_center_world - apex, n) > sphere_radius;
        }

        if (all_centers_outside) {
            return;
        }
    }

    // Survived all 4 plane tests. Claim a slot in the compact list.
    int32_t my_idx;
    output_active_count.InterlockedAdd(uint(0), int32_t(1), my_idx);
    output_active_tracks_indices[my_idx] = int32_t(tid);
}

// We choose the intersection strategy which matches our tracks interpolation method.
typedef ray_intersect::LieIntersectionStrategy TimestampEstimationIntersectionStrategy;

// Intersection handler performance depends on how many concurrent threads (rays) are
// intersecting with the same track.
// We choose the intersection handler which has a slightly worst performance in
// the low contention case, but scales the better in high contention to have
// stable performance regardless of the contention level.
// If this becomes relevant we can optimize later.
typedef ray_intersect::timestamps_estimation::WaveReduceSharedAccumulatorsIntersectionHandler TimestampEstimationIntersectionHandler;

// We create a kernel entry point to be able to call the tracks timestamps estimation.
//
// This kernel is not differentiable.
[CUDAKernel]
[AutoPyBindCUDA]
void estimate_tracks_timestamps(
    TensorView<int64_t> output_tracks_timestamps_sum_and_count,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    no_diff int64_t frame_timestamp,
) {
    ray_intersect::intersect_rays_with_timed_cuboids<TimestampEstimationIntersectionStrategy, TimestampEstimationIntersectionHandler>(
        rays,
        rays_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        cuboids_dimensions,
        TimestampEstimationIntersectionHandler(output_tracks_timestamps_sum_and_count, frame_timestamp),
    );
}

// Variant of `estimate_tracks_timestamps` that processes only an explicit subset
// of tracks (the "active" tracks). A bounded grid iterates over active
// track/ray-block pairs using the device-side count. This lets the caller pre-cull tracks
// (e.g. with a frustum test) and skip the expensive ray/cuboid intersection
// for tracks that cannot possibly be hit.
//
// The output buffer is still indexed by the real track index, so culled tracks
// keep whatever value the caller initialized them to (typically zero).  When
// consumed by `TimestampsEstimationFetcher`, a (sum=0, count=0) entry falls
// back to `default_timestamp`, so the downstream interpolation kernel does not
// need to know about the culling.
[CUDAKernel]
[AutoPyBindCUDA]
void estimate_tracks_timestamps_masked(
    TensorView<int64_t> output_tracks_timestamps_sum_and_count,
    TensorView<float> rays,
    TensorView<int64_t> rays_timestamps,
    TensorView<float> tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TensorView<float> cuboids_dimensions,
    TensorView<int32_t> active_tracks_indices,
    TensorView<int32_t> active_count,
    no_diff int64_t frame_timestamp,
) {
    typedef ray_intersect::timestamps_estimation::SparseWaveReduceIntersectionHandler MaskedTimestampEstimationIntersectionHandler;
    ray_intersect::intersect_rays_with_active_timed_cuboids<TimestampEstimationIntersectionStrategy, MaskedTimestampEstimationIntersectionHandler>(
        rays,
        rays_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        cuboids_dimensions,
        active_tracks_indices,
        active_count,
        MaskedTimestampEstimationIntersectionHandler(output_tracks_timestamps_sum_and_count, frame_timestamp),
    );
}

// Track interpolation.
interface IPoseFetcher {
    [Differentiable]
    [ForceInline]
    static transform::SE3 fetch(
        DiffTensorView tracks_poses,
        int32_t index_left, int32_t index_right,
        int64_t time_left, int64_t time_right, int64_t timestamp
        );
}

// Note that when we are doing track interpolation, we can use loadOnce()
// because each track will read a pose (or two) and no 2 tracks can read
// the same pose.  However, we still need to zero the gradients buffer
// because not all poses will be read (only one or two per track), therefore
// many poses will not get a gradient assigned by the kernel.
struct PoseInterpolator : IPoseFetcher {
    [Differentiable]
    [ForceInline]
    static transform::SE3 fetch(
        DiffTensorView tracks_poses,
        int32_t index_left, int32_t index_right,
        int64_t time_left, int64_t time_right, int64_t timestamp
        ) {
        let pose_left = transform.SE3.from_vec(
            tracks_poses.loadVecOnce<3>(uint2(index_left, 0)),
            tracks_poses.loadVecOnce<4>(uint2(index_left, 3)),
        );
        let pose_right = transform.SE3.from_vec(
            tracks_poses.loadVecOnce<3>(uint2(index_right, 0)),
            tracks_poses.loadVecOnce<4>(uint2(index_right, 3)),
        );

        let alpha = (timestamp - time_left) / float(time_right - time_left);

        let pose_final = transform::interpolate(pose_left, pose_right, alpha);

        return pose_final;
    }
};

struct PoseNearestNeighbor : IPoseFetcher {
    [Differentiable]
    [ForceInline]
    static transform::SE3 fetch(
        DiffTensorView tracks_poses,
        int32_t index_left, int32_t index_right,
        int64_t time_left, int64_t time_right, int64_t timestamp
        ) {
        let time_diff_left = abs(timestamp - time_left);
        let time_diff_right = abs(timestamp - time_right);
        int32_t index_nearest;
        if (time_diff_right < time_diff_left) {
            index_nearest = index_right;
        }
        else {
            index_nearest = index_left;
        }

        let pose_final = transform::SE3.from_vec(
            tracks_poses.loadVecOnce<3>(uint2(index_nearest, 0)),
            tracks_poses.loadVecOnce<4>(uint2(index_nearest, 3)),
        );

        return pose_final;
    }
};

public struct TimestampsEstimationFetcher : ITimestampsFetcher {
    TensorView<int64_t> tracks_timestamps_sum_and_count;
    int64_t default_timestamp;

    int64_t fetch(no_diff uint track_idx)
    {
        let count = tracks_timestamps_sum_and_count.load(uint2(track_idx, 1));
        if (count > 0) {
            return default_timestamp + tracks_timestamps_sum_and_count.load(uint2(track_idx, 0)) / count;
        }
        else {
            return default_timestamp;
        }
    }
}


// To avoid compiler errors in auto differentiation with pointer operations,
// we extract the search in a non differentiable function.
Tuple<int32_t, int32_t, bool> search_in_tracks(
    TensorView<int64_t> tracks_timestamps, int64_t timestamp, int32_t start_index, int32_t length
) {
    typedef search::BinarySearch<int64_t, int32_t> SearchStrategy;

    let data = ImmutablePtr<int64_t>(tracks_timestamps.data_ptr_at(start_index));

    return search::find_surrounding_indices<SearchStrategy>(timestamp, data, length);
}

[Differentiable]
void fetch_tracks<TimestampsFetcher: ITimestampsFetcher, PoseFetcher: IPoseFetcher>(
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    no_diff TensorView<int64_t> output_resolved_timestamps,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsFetcher timestamps_fetcher,
    no_diff TensorView<int64_t> max_positive_timestamp_gap_us,
    no_diff int64_t frame_exposure_us,
    uint track_id,
) {
    let start_index = tracks_packinfo.load(uint2(track_id,0));
    let length = tracks_packinfo.load(uint2(track_id,1));

    let timestamp = timestamps_fetcher.fetch(track_id);

    let indices = search_in_tracks(tracks_timestamps, timestamp, start_index, length);

    // Store whether the timestamp was inside the track or close enough to use the endpoint pose.
    bool inside = indices._2;
    int64_t resolved_timestamp = timestamp;
    let clamp_outside_track_tolerance_us =
        max_positive_timestamp_gap_us.load(track_id) + frame_exposure_us;
    if (!inside && clamp_outside_track_tolerance_us > 0 && length > 0) {
        let endpoint_timestamp = tracks_timestamps.load(start_index + indices._0);
        let timestamp_delta = timestamp < endpoint_timestamp
            ? endpoint_timestamp - timestamp
            : timestamp - endpoint_timestamp;
        inside = timestamp_delta <= clamp_outside_track_tolerance_us;
        if (inside) {
            // The pose is clamped to this endpoint, so use the same timestamp
            // to condition non-rigid deformation. Rejected queries retain the
            // raw query timestamp, but their output is masked inactive.
            resolved_timestamp = endpoint_timestamp;
        }
    }
    output_inside.store(track_id, inside);
    output_resolved_timestamps.store(track_id, resolved_timestamp);

    // Store the final pose.
    transform::SE3 pose;
    if (indices._0 < 0) {
        // Empty track, return identity.
        pose = transform::SE3.identity();
    }
    else if (indices._0 == indices._1) {
        // Matching exactly a pose timestamp, return this pose.
        let index = start_index + indices._0;
        pose = transform::SE3.from_vec(
            tracks_poses.loadVecOnce<3>(uint2(index, 0)),
            tracks_poses.loadVecOnce<4>(uint2(index, 3)),
        );
    }
    else {
        // Between two poses.
        let index_left = start_index + indices._0;
        let index_right = start_index + indices._1;
        let time_left = tracks_timestamps.load(index_left);
        let time_right = tracks_timestamps.load(index_right);
        pose = PoseFetcher.fetch(tracks_poses, index_left, index_right, time_left, time_right, timestamp);
    }

    var pose_vec = pose.to_vec();
    output_poses.storeVecOnce<3>(uint2(track_id, 0), pose_vec._0);
    output_poses.storeVecOnce<4>(uint2(track_id, 3), pose_vec._1);
}

// We create kernel entry points to be able to call the interpolation.
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void interpolate_tracks_poses_global(
    no_diff uint count,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    no_diff TensorView<int64_t> output_resolved_timestamps,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsGlobalFetcher timestamps,
    no_diff TensorView<int64_t> max_positive_timestamp_gap_us,
    no_diff int64_t frame_exposure_us,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    fetch_tracks<TimestampsGlobalFetcher, PoseInterpolator>(
        output_poses,
        output_inside,
        output_resolved_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        max_positive_timestamp_gap_us,
        frame_exposure_us,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void interpolate_tracks_poses_per_track(
    no_diff uint count,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    no_diff TensorView<int64_t> output_resolved_timestamps,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsPerTrackFetcher timestamps,
    no_diff TensorView<int64_t> max_positive_timestamp_gap_us,
    no_diff int64_t frame_exposure_us,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    fetch_tracks<TimestampsPerTrackFetcher, PoseInterpolator>(
        output_poses,
        output_inside,
        output_resolved_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        max_positive_timestamp_gap_us,
        frame_exposure_us,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void interpolate_tracks_poses_with_estimation(
    no_diff uint count,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    no_diff TensorView<int64_t> output_resolved_timestamps,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsEstimationFetcher timestamps,
    no_diff TensorView<int64_t> max_positive_timestamp_gap_us,
    no_diff int64_t frame_exposure_us,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    fetch_tracks<TimestampsEstimationFetcher, PoseInterpolator>(
        output_poses,
        output_inside,
        output_resolved_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        max_positive_timestamp_gap_us,
        frame_exposure_us,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void nearest_neighbor_tracks_poses_global(
    no_diff uint nb_tracks,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    no_diff TensorView<int64_t> output_resolved_timestamps,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsGlobalFetcher timestamps,
    no_diff TensorView<int64_t> max_positive_timestamp_gap_us,
    no_diff int64_t frame_exposure_us,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= nb_tracks) {
        return;
    }

    fetch_tracks<TimestampsGlobalFetcher, PoseNearestNeighbor>(
        output_poses,
        output_inside,
        output_resolved_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        max_positive_timestamp_gap_us,
        frame_exposure_us,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void nearest_neighbor_tracks_poses_per_track(
    no_diff uint nb_tracks,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    no_diff TensorView<int64_t> output_resolved_timestamps,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsPerTrackFetcher timestamps,
    no_diff TensorView<int64_t> max_positive_timestamp_gap_us,
    no_diff int64_t frame_exposure_us,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= nb_tracks) {
        return;
    }

    fetch_tracks<TimestampsPerTrackFetcher, PoseNearestNeighbor>(
        output_poses,
        output_inside,
        output_resolved_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        max_positive_timestamp_gap_us,
        frame_exposure_us,
        tid,
    );
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void nearest_neighbor_tracks_poses_with_estimation(
    no_diff uint nb_tracks,
    DiffTensorView output_poses,
    TensorView<bool> output_inside,
    no_diff TensorView<int64_t> output_resolved_timestamps,
    DiffTensorView tracks_poses,
    TensorView<int64_t> tracks_timestamps,
    TensorView<int32_t> tracks_packinfo,
    TimestampsEstimationFetcher timestamps,
    no_diff TensorView<int64_t> max_positive_timestamp_gap_us,
    no_diff int64_t frame_exposure_us,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= nb_tracks) {
        return;
    }

    fetch_tracks<TimestampsEstimationFetcher, PoseNearestNeighbor>(
        output_poses,
        output_inside,
        output_resolved_timestamps,
        tracks_poses,
        tracks_timestamps,
        tracks_packinfo,
        timestamps,
        max_positive_timestamp_gap_us,
        frame_exposure_us,
        tid,
    );
}


public struct SceneContractor
{
    static const float NORM_COMPARISON_EPS = 1e-7;

    no_diff TensorView<float> aabb_blb;
    no_diff TensorView<float> aabb_trf;
    no_diff float degree;
    no_diff bool is_merf;
    no_diff bool has_contraction;

    
    [ForceInline]
    float3 get_aabb_extent(no_diff uint instance_id)
    {
        float3 blb = this.aabb_blb.loadVec<3>(uint2(instance_id, 0));
        float3 trf = this.aabb_trf.loadVec<3>(uint2(instance_id, 0));
        return trf - blb;
    }
    
    [Differentiable]
    [ForceInline]
    float3 points_to_unit_cube(float3 point, no_diff uint instance_id)
    {
        float3 blb = this.aabb_blb.loadVec<3>(uint2(instance_id, 0));
        float3 extent = this.get_aabb_extent(instance_id);
        return (point - blb) / extent;
    }
    
    [Differentiable]
    [ForceInline]
    static float compute_norm(float3 point, no_diff float degree)
    {
        if (isinf(degree)) {
            // L-infinity norm: max(|x|, |y|, |z|)
            return max(max(abs(point.x), abs(point.y)), abs(point.z));
        }
        else {
            return pow(
                pow(abs(point.x), degree) + 
                pow(abs(point.y), degree) + 
                pow(abs(point.z), degree),
                1.0 / degree
            );
        }
    }
    
    [Differentiable]
    [ForceInline]
    static float3 merf_contract(float3 points, float norm)
    {
        float scale = max(norm, 1.0);
        bool3 is_largest_coord = (abs(points) >= norm - NORM_COMPARISON_EPS);
        
        float3 scaled_points = points / scale;
        float3 contracted = select(is_largest_coord, scaled_points * (2.0 - 1.0 / scale), scaled_points);

        return contracted;
    }
    
    [Differentiable]
    [ForceInline]
    static float3 standard_contract(float3 points, float norm)
    {
        if (norm < 1.0) {
            return points;
        }
        else {
            return (2.0 - 1.0 / norm) * (points / norm);
        }
    }
    
    [Differentiable]
    [ForceInline]
    [NoDiffThis]
    float3 to_contracted_space(float3 world_point, no_diff uint instance_id)
    {
        float3 points = this.points_to_unit_cube(world_point, instance_id);
        
        if (!this.has_contraction) {
            return points;
        }
        
        points = 2.0 * points - 1.0;
        
        float norm = SceneContractor.compute_norm(points, this.degree);
        
        float3 contracted_points;
        if (this.is_merf) {
            contracted_points = SceneContractor.merf_contract(points, norm);
        }
        else {
            contracted_points = SceneContractor.standard_contract(points, norm);
        }
        
        return contracted_points * 0.25 + 0.5;
    }
}

// Input embedding computation for hash grid inputs.
// Computes concatenated embeddings: [xyz (3) + instance_emb (InstanceEmbeddingDim) + time_emb (1)]
// When NbTimestamps > 1, outputs multiple full embeddings with different time values stacked along batch dimension:
//   Row 0 to N-1:     [xyz, instance, time=base_timestamp + timestamp_offsets[0]]
//   Row N to 2N-1:    [xyz, instance, time=base_timestamp + timestamp_offsets[1]]
//   Row 2N to 3N-1:   [xyz, instance, time=base_timestamp + timestamp_offsets[2]]
//   ...
//   Output shape: [NbTimestamps*N, base_dim] (stacked along batch dimension, NOT feature dimension)
//   Column layout: [xyz(0-2), instance(3..3+InstanceEmbeddingDim), time(3+InstanceEmbeddingDim)]
// Common helper function to avoid code duplication
[Differentiable]
void compute_input_embedding_impl<
    let InstanceEmbeddingDim : int,
    let NbTimestamps : int,
    TimestampsFetcher : ITimestampsFetcher
>(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<InstanceEmbeddingDim> instance_embedding,
    TimestampsFetcher timestamps_fetcher,
    no_diff int64_t timestamp_offsets[NbTimestamps],
    IndividualRemapTimeInputEmbedding time_embedding,
    DiffTensorView output,
    no_diff uint tid
)
{
    no_diff var instance_id = instance_idx[tid];
    const int64_t timestamp = timestamps_fetcher.fetch(uint(instance_id));
    float3 xyz = xyzs.loadVecOnce<3>(uint2(tid, 0));
    var contracted_xyz = scene_contractor.to_contracted_space(xyz, instance_id);
    contracted_xyz = clamp(contracted_xyz, float3(0.0), float3(1.0));
    
    // Column positions: xyz at 0-2, instance at 3, time at 3+InstanceEmbeddingDim
    const int instance_col = 3;  // After xyz
    const int time_col = instance_col + InstanceEmbeddingDim;
    
    // Loop bound is compile-time constant, fully unrolled
    [ForceUnroll]
    for (int variant = 0; variant < NbTimestamps; ++variant) {
        const int64_t variant_timestamp = timestamp + timestamp_offsets[variant];
        // Each variant writes to a different row: tid, tid+count, tid+2*count, ...
        const uint output_row = tid + variant * count;
        
        // xyz at columns 0-2
        [ForceUnroll]
        for (int i = 0; i < 3; ++i) {
            output.storeOnce(uint2(output_row, i), contracted_xyz[i]);
        }
        
        // instance embedding starting at column 3
        instance_embedding.embed_to_row(tid, output_row, instance_col);
        
        // time embedding after instance embedding
        var time_emb = time_embedding.embed(tid, variant_timestamp);
        output.storeOnce(uint2(output_row, time_col), time_emb);
    }
}

// Generic timestamp-dispatch helpers keep the embedding computation identical
// for global and per-track timestamps. Only timestamps_delta is per sample.
[Differentiable]
void prepare_input_embeddings_impl<TimestampsFetcher : ITimestampsFetcher>(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<1> instance_embedding,
    TimestampsFetcher timestamps_fetcher,
    IndividualRemapTimeInputEmbedding time_embedding,
    DiffTensorView output,
    no_diff uint sample_idx
) {
    int64_t timestamp_offsets[1] = { 0 };
    compute_input_embedding_impl</*InstanceEmbeddingDim=*/1, /*NbTimestamps=*/1, TimestampsFetcher>(
        count,
        xyzs,
        instance_idx,
        scene_contractor,
        instance_embedding,
        timestamps_fetcher,
        timestamp_offsets,
        time_embedding,
        output,
        sample_idx,
    );
}

[Differentiable]
void prepare_input_embeddings_with_timestamps_delta_impl<TimestampsFetcher : ITimestampsFetcher>(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<1> instance_embedding,
    TimestampsFetcher timestamps_fetcher,
    IndividualRemapTimeInputEmbedding time_embedding,
    no_diff TensorView<int64_t> timestamps_delta,
    DiffTensorView output,
    no_diff uint sample_idx
) {
    const int64_t delta = timestamps_delta[sample_idx];
    int64_t timestamp_offsets[3] = { 0, -delta, delta };
    compute_input_embedding_impl</*InstanceEmbeddingDim=*/1, /*NbTimestamps=*/3, TimestampsFetcher>(
        count,
        xyzs,
        instance_idx,
        scene_contractor,
        instance_embedding,
        timestamps_fetcher,
        timestamp_offsets,
        time_embedding,
        output,
        sample_idx,
    );
}

// Kernel entry point for input embeddings
// Output shape: [count, base_dim] where base_dim = 3 + instance_dim + time_dim
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void prepare_input_embeddings(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<1> instance_embedding,
    TimestampsGlobalFetcher timestamps_fetcher,
    IndividualRemapTimeInputEmbedding time_embedding,
    DiffTensorView output,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }
    
    prepare_input_embeddings_impl<TimestampsGlobalFetcher>(
        count, xyzs, instance_idx, scene_contractor, instance_embedding,
        timestamps_fetcher, time_embedding, output, tid);
}

// Compact per-track timestamp variant used by deformable Gaussians whose
// time input follows the rolling-shutter timestamp resolved for their track.
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void prepare_input_embeddings_per_track(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<1> instance_embedding,
    TimestampsPerTrackFetcher timestamps_fetcher,
    IndividualRemapTimeInputEmbedding time_embedding,
    DiffTensorView output,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    prepare_input_embeddings_impl<TimestampsPerTrackFetcher>(
        count, xyzs, instance_idx, scene_contractor, instance_embedding,
        timestamps_fetcher, time_embedding, output, tid);
}

// Kernel entry point for input embeddings - With timestamps delta version
// Output shape: [count * 3, base_dim] where:
//   Rows 0 to count-1:         embeddings at time t
//   Rows count to 2*count-1:   embeddings at time t-delta
//   Rows 2*count to 3*count-1: embeddings at time t+delta
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void prepare_input_embeddings_with_timestamps_delta(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<1> instance_embedding,
    TimestampsGlobalFetcher timestamps_fetcher,
    IndividualRemapTimeInputEmbedding time_embedding,
    no_diff TensorView<int64_t> timestamps_delta,
    DiffTensorView output,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }
    
    prepare_input_embeddings_with_timestamps_delta_impl<TimestampsGlobalFetcher>(
        count, xyzs, instance_idx, scene_contractor, instance_embedding,
        timestamps_fetcher, time_embedding, timestamps_delta, output, tid);
}

// Per-track timestamp counterpart of the smoothness-sampling kernel above.
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void prepare_input_embeddings_per_track_with_timestamps_delta(
    no_diff uint count,
    DiffTensorView xyzs,
    no_diff TensorView<int32_t> instance_idx,
    SceneContractor scene_contractor,
    WeightedInstanceInputEmbedding<1> instance_embedding,
    TimestampsPerTrackFetcher timestamps_fetcher,
    IndividualRemapTimeInputEmbedding time_embedding,
    no_diff TensorView<int64_t> timestamps_delta,
    DiffTensorView output,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    prepare_input_embeddings_with_timestamps_delta_impl<TimestampsPerTrackFetcher>(
        count, xyzs, instance_idx, scene_contractor, instance_embedding,
        timestamps_fetcher, time_embedding, timestamps_delta, output, tid);
}


interface IModifier<let ModifierDim : int> : IDifferentiable {
    [Differentiable]
    vector<float, ModifierDim> apply(vector<float, ModifierDim> value);
};

struct Modifier_Copy<let ModifierDim : int> : IModifier<ModifierDim> {
    [Differentiable]
    [ForceInline]
    vector<float, ModifierDim> apply(vector<float, ModifierDim> value)
    {
        return value;
    }
};

struct Modifier_Add<let ModifierDim : int> : IModifier<ModifierDim> {
    vector<float, ModifierDim> add_value;

    [Differentiable]
    [ForceInline]
    vector<float, ModifierDim> apply(vector<float, ModifierDim> value)
    {
        return value + add_value;
    }
};

struct Modifier_Rotate : IModifier<4> {
    vector<float, 4> rotation;

    [Differentiable]
    [ForceInline]
    vector<float, 4> apply(vector<float, 4> value)
    {
        return quaternion::multiply(quaternion::normalizeSafe(rotation), value);
    }
};

// Positions, rotations and densities.
//
// We do positions and rotations together to do fewer reads to the poses to apply.
// We do densities too just to avoid re-reading the track id.
//
// It could be fewer GPU operations to compute the rotation matrices
// for each track during interpolation and use those.
// However, it would use more memory (9 floats instead of 4) and since
// we are typically memory bound, we trade the loads for a few more operations
// per gaussian.
// That being said, since there aren't that many tracks, maybe the loads would
// not be an issue so we could consider this other approach.
[Differentiable]
void collect_positions_rotations_densities_for_tracks<
    PositionsModifier : IModifier<3>,
    RotationsModifier : IModifier<4>,
>(
    no_diff uint tid, no_diff uint out_offset,
    DiffTensorView input_positions,
    DiffTensorView input_rotations,
    DiffTensorView input_densities,
    DiffTensorView output_positions,
    DiffTensorView output_rotations,
    DiffTensorView output_densities,
    DiffTensorView tracks_poses,
    no_diff TensorView<int32_t> tracks_ids,
    no_diff TensorView<bool> keep_mask,
    PositionsModifier positions_modifier,
    RotationsModifier rotations_modifier,
) {
    let track_idx = tracks_ids.load(tid);
    let keep = keep_mask.load(track_idx);

    // Read the pose.
    const var pose = transform.SE3.from_vec(
        utils::pass(tracks_poses.loadVecEx<3>(uint2(track_idx, 0)), keep),
        utils::pass(tracks_poses.loadVecEx<4>(uint2(track_idx, 3)), keep),
    );

    // Read the position to transform.
    float3 position = input_positions.loadVecOnce<3>(uint2(tid, 0));
    position = positions_modifier.apply(position);

    // Read the rotation to transform.
    // It is stored in wxyz, but the current library expects xyzw.
    // wxyz -> xyzw => (0,1,2,3) -> (1,2,3,0) => We read yzwx
    float4 rotation = input_rotations.loadVecOnce<4>(uint2(tid, 0)).yzwx;
    rotation = quaternion::normalizeSafe(rotation);
    rotation = rotations_modifier.apply(rotation);

    position = (pose * transform::Point(position)).p;
    rotation = (pose.so3 * transform::SO3.from_vec(rotation)).to_vec();

    const uint out_index = out_offset + tid;

    output_positions.storeVecOnce<3>(uint2(out_index, 0), position);

    // We need to store the rotation in wxyz.
    // xyzw -> wxyz => (0,1,2,3) -> (3,0,1,2) => We write wxyz
    output_rotations.storeVecOnce<4>(uint2(out_index, 0), rotation.wxyz);

    // Densities.
    float density;
    if (keep) {
        density = input_densities.loadOnce(uint2(tid, 0));
        density = SigmoidFunction.apply(density);
    }
    else {
        // No gradient will be written back for this index.
        density = 0.0f;
    }
    output_densities.storeOnce(uint2(out_index, 0), density);
}

public struct PositionsRotationsDensitiesCollector_Tracks : ITask
{
    DiffTensorView input_positions;
    DiffTensorView input_rotations;
    DiffTensorView input_densities;
    DiffTensorView output_positions;
    DiffTensorView output_rotations;
    DiffTensorView output_densities;
    DiffTensorView tracks_poses;
    no_diff TensorView<int32_t> tracks_ids;
    no_diff TensorView<bool> keep_mask;

    [Differentiable]
    void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        collect_positions_rotations_densities_for_tracks(
            rows.row(tid), out_offset,
            input_positions, input_rotations, input_densities,
            output_positions, output_rotations, output_densities,
            tracks_poses, tracks_ids, keep_mask,
            Modifier_Copy<3>(), Modifier_Copy<4>(),
        );
    }
};

// These types are not really used by the rigid layer, but they are still created
// temporarily before they are replaced by the PositionsRotationsDensitiesCollector_Tracks.
// So we make sure they are exported by slangtorch.
[CUDAKernel]
[AutoPyBindCUDA]
void trigger_type_export(
    Collector_Copy<3> positions_collector,
    RotationsCollector_Normalize rotations_collector,
    DensitiesCollector_Sigmoid densities_collector,
    PositionsRotationsDensitiesCollector_Tracks tracks_collector,
) {
}

// Zero one (Gaussian, channel) of a flattened [N, D] grad-buffer view. Python
// flattens all trailing dimensions into D; adjacent threads share a Gaussian
// and take consecutive channels so a warp writes a contiguous span.
void zero_at_coord(
    no_diff uint gaussian_idx,
    no_diff uint channel_idx,
    no_diff TensorView<float> buffer,
) {
    if (channel_idx < buffer.size(1)) {
        buffer.store(uint2(gaussian_idx, channel_idx), 0.0f);
    }
}

// 2D launch: thread.x = channel, thread.y = Gaussian. Flattened views also
// cover trailing channels that the old scalar loop missed for rank-3+ grads;
// the thread mapping makes these stores coalesce.
[CUDAKernel]
[AutoPyBindCUDA]
void zero_gradients_for_disabled_tracks_2d(
    TensorView<int32_t> tracks_ids,
    TensorView<bool> keep_mask,
    TensorView<float> positions_grad,
    TensorView<float> rotations_grad,
    TensorView<float> scales_grad,
    TensorView<float> densities_grad,
    TensorView<float> extra_signal_grad,
    TensorView<float> camera_extra_signal_grad,
    TensorView<float> lidar_extra_signal_grad,
    TensorView<float> features_albedo_grad,
    TensorView<float> features_specular_grad,
) {
    const uint channel_idx = cudaBlockIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    const uint gaussian_idx = cudaBlockIdx().x * cudaBlockDim().y + cudaThreadIdx().y;
    if (gaussian_idx >= tracks_ids.size(0)) {
        return;
    }

    uint gaussian_is_kept = 0;
    if (cudaThreadIdx().x == 0) {
        let track_id = tracks_ids.load(gaussian_idx);
        gaussian_is_kept = keep_mask.load(track_id) ? 1 : 0;
    }
    if (WaveReadLaneFirst(gaussian_is_kept) != 0) {
        return;
    }

    zero_at_coord(gaussian_idx, channel_idx, positions_grad);
    zero_at_coord(gaussian_idx, channel_idx, rotations_grad);
    zero_at_coord(gaussian_idx, channel_idx, scales_grad);
    zero_at_coord(gaussian_idx, channel_idx, densities_grad);
    zero_at_coord(gaussian_idx, channel_idx, extra_signal_grad);
    zero_at_coord(gaussian_idx, channel_idx, camera_extra_signal_grad);
    zero_at_coord(gaussian_idx, channel_idx, lidar_extra_signal_grad);
    zero_at_coord(gaussian_idx, channel_idx, features_albedo_grad);
    zero_at_coord(gaussian_idx, channel_idx, features_specular_grad);
}


//
// Implementations for dynamic rigid layer.
//

public struct PositionsRotationsDensitiesCollector_Deformable : ITask
{
    DiffTensorView input_positions;
    DiffTensorView input_rotations;
    DiffTensorView input_densities;
    DiffTensorView output_positions;
    DiffTensorView output_rotations;
    DiffTensorView output_densities;
    DiffTensorView tracks_poses;
    no_diff TensorView<int32_t> tracks_ids;
    no_diff TensorView<bool> keep_mask;
    DiffTensorView deform_positions;
    DiffTensorView deform_rotations;

    [Differentiable]
    void run<Rows : ICollectRows>(no_diff uint tid, no_diff uint count, no_diff uint out_offset, no_diff Rows rows)
    {
        const uint row = rows.row(tid);
        const float3 deform_position = deform_positions.loadVecOnce<3>(uint2(row, 0));
        float4 deform_rotation = deform_rotations.loadVecOnce<4>(uint2(row, 0));
        // The deformation rotations are offsets from the identity rotation,
        // so we add 1 to the w component to get the full rotation offset.
        deform_rotation.w += 1.0f;

        collect_positions_rotations_densities_for_tracks(
            row, out_offset,
            input_positions, input_rotations, input_densities,
            output_positions, output_rotations, output_densities,
            tracks_poses, tracks_ids, keep_mask,
            Modifier_Add<3>(deform_position), Modifier_Rotate(deform_rotation),
        );
    }
};
