// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Lerp.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>

namespace {
constexpr int kClasses           = 20;
constexpr int kReductionWidth    = 4;
constexpr int kFloatKeyBits      = std::numeric_limits<uint32_t>::digits;
constexpr uint32_t kFloatSignBit = uint32_t{1} << (kFloatKeyBits - 1);

// The deployed Mask2Former has twenty classes. Keep probabilities in registers
// and reproduce the pinned PyTorch spatial-softmax and four-accumulator sum ordering.
__global__ void categorical_entropy_kernel(float const* logits, float* output, int64_t pixels) {
    int64_t const pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixels)
        return;
    float values[kClasses];
    float maximum = -FLT_MAX;
#pragma unroll
    for (int c = 0; c < kClasses; ++c) {
        values[c] = logits[static_cast<int64_t>(c) * pixels + pixel];
        maximum   = isnan(values[c]) ? values[c] : fmaxf(maximum, values[c]);
    }
    float denominator = 0.0f;
#pragma unroll
    for (int c = 0; c < kClasses; ++c) {
        values[c]   = expf(values[c] - maximum);
        denominator = __fadd_rn(denominator, values[c]);
    }
#pragma unroll
    for (int c = 0; c < kClasses; ++c) {
        float const p = values[c] / denominator;
        values[c]     = isnan(p) ? p : p > 0.0f ? __fmul_rn(-p, logf(p))
                                                : 0.0f;
    }
    float sums[kReductionWidth] = {};
#pragma unroll
    for (int c = 0; c < kClasses; ++c) {
        sums[c % kReductionWidth] = __fadd_rn(sums[c % kReductionWidth], values[c]);
    }
    // At large spatial extents all block lanes map to outputs, not classes.
    output[pixel] = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
}
} // namespace

// Inference-only, contiguous FP32 [20,H,W] -> preallocated FP32 [H,W].
// The specialization matches PyTorch's vectorized four-way reduction domain.
void categorical_entropy_out(at::Tensor const& logits, at::Tensor const& output) {
    constexpr int kMinVectorizedPixels = 128;
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 3 &&
                    logits.size(0) == kClasses && logits.is_contiguous(),
                "entropy requires contiguous CUDA FP32 [20,H,W]");
    auto const pixels = logits.size(1) * logits.size(2);
    TORCH_CHECK(pixels >= kMinVectorizedPixels && pixels % kReductionWidth == 0, "entropy requires a vectorized spatial reduction");
    TORCH_CHECK(output.device() == logits.device() && output.scalar_type() == at::kFloat &&
                    output.is_contiguous() && output.sizes() == logits.sizes().slice(1),
                "invalid entropy output");
    TORCH_CHECK(!logits.requires_grad(), "entropy native kernel supports inference only");
    c10::cuda::CUDAGuard guard(logits.device());
    constexpr int threads = 256;
    auto stream           = at::cuda::getCurrentCUDAStream(logits.get_device());
    categorical_entropy_kernel<<<(pixels + threads - 1) / threads, threads, 0, stream>>>(
        logits.const_data_ptr<float>(), output.mutable_data_ptr<float>(), pixels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
// Refine two order-statistic ranks through 11 + 11 + 10 float-key bits.
// Histogram counts are 64-bit; no full-size sorting or index buffers are needed.
constexpr int quantile_threads = 256;
using QuantileCount            = unsigned long long;
struct QuantileState {
    uint32_t prefix[2];
    QuantileCount rank[2];
    uint32_t common, any_nan;
    float weight;
};
// Map FP32 values to increasing unsigned keys, reserving the maximum for NaNs.
__device__ uint32_t ordered_float_key(float x) {
    if (isnan(x))
        return std::numeric_limits<uint32_t>::max();
    uint32_t bits = __float_as_uint(x);
    return bits & kFloatSignBit ? ~bits : bits ^ kFloatSignBit;
}
// Invert an ordered key; the reserved NaN key reconstructs a quiet NaN.
__device__ float float_from_key(uint32_t key) {
    return __uint_as_float(key & kFloatSignBit ? key ^ kFloatSignBit : ~key);
}
// One thread initializes zero-based neighboring ranks and the FP32 interpolation weight.
__global__ void initialize_quantile(QuantileState* state, QuantileCount lower, QuantileCount upper, float weight) {
    state->prefix[0] = state->prefix[1] = 0;
    state->rank[0]                      = lower;
    state->rank[1]                      = upper;
    state->common                       = 1;
    state->any_nan                      = 0;
    state->weight                       = weight;
}
// Count the next Bits of input keys matching state prefixes; Shift counts the
// unprocessed low bits. Each CTA visits pixels with a grid stride, accumulates
// in shared memory, then merges into the caller-zeroed global_hist.
template <int Bits, int Shift>
__global__ void quantile_histogram(const float* input, int64_t pixels, QuantileState* state, QuantileCount* global_hist) {
    constexpr int bins = 1 << Bits;
    extern __shared__ QuantileCount local_hist[];
    uint32_t const prefix0 = state->prefix[0], prefix1 = state->prefix[1];
    bool const common = prefix0 == prefix1;
    int const slots   = common ? bins : 2 * bins;
    // Freeze this choice before the two selector CTAs update their prefixes.
    if (blockIdx.x == 0 && threadIdx.x == 0)
        state->common = common;
    for (int i = threadIdx.x; i < slots; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();
    for (int64_t pixel = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; pixel < pixels;
         pixel += int64_t(blockDim.x) * gridDim.x) {
        uint32_t const key = ordered_float_key(input[pixel]);
        int slot           = -1;
        if constexpr (Bits + Shift == kFloatKeyBits) {
            slot = int((key >> Shift) & (bins - 1));
        } else {
            uint32_t const previous = key >> (Bits + Shift);
            if (previous == prefix0)
                slot = int((key >> Shift) & (bins - 1));
            else if (!common && previous == prefix1)
                slot = bins + int((key >> Shift) & (bins - 1));
        }
        unsigned const peers = __match_any_sync(__activemask(), static_cast<unsigned>(slot));
        if (slot >= 0 && int(threadIdx.x % warpSize) == __ffs(peers) - 1)
            atomicAdd(local_hist + slot, static_cast<QuantileCount>(__popc(peers)));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < slots; i += blockDim.x) {
        QuantileCount const count = local_hist[i];
        if (count)
            atomicAdd(global_hist + i, count);
    }
}
// One CTA per neighboring rank scans hist and refines its state prefix/rank.
// First also records NaN presence; hist contains completed prefix counts.
template <int Bits, bool First>
__global__ void select_quantile_bins(const QuantileCount* hist, QuantileState* state) {
    constexpr int bins = 1 << Bits, items = bins / quantile_threads;
    using Scan = cub::BlockScan<QuantileCount, quantile_threads>;
    __shared__ typename Scan::TempStorage storage;
    __shared__ QuantileCount rank;
    __shared__ uint32_t prefix;
    int const target = blockIdx.x;
    int const offset = state->common ? 0 : target * bins;
    if (threadIdx.x == 0) {
        rank   = state->rank[target];
        prefix = state->prefix[target];
        if constexpr (First)
            if (target == 0)
                state->any_nan = hist[bins - 1] != 0;
    }
    __syncthreads();
    QuantileCount counts[items], sums[items];
#pragma unroll
    for (int i = 0; i < items; ++i)
        counts[i] = hist[offset + threadIdx.x * items + i];
    Scan(storage).InclusiveSum(counts, sums);
#pragma unroll
    for (int i = 0; i < items; ++i) {
        QuantileCount const before = sums[i] - counts[i];
        if (before <= rank && rank < sums[i]) {
            state->prefix[target] = (prefix << Bits) | (threadIdx.x * items + i);
            state->rank[target]   = rank - before;
        }
    }
}
// Interpolate selected values with pinned Torch rounding at every input size.
__device__ float quantile_value(const QuantileState& state) {
    float const lower = float_from_key(state.prefix[0]), upper = float_from_key(state.prefix[1]);
    return state.any_nan ? CUDART_NAN_F : at::native::lerp(lower, upper, state.weight);
}

// Return device-resident selection state for the 90th percentile of entropy.
// Enqueue reductions on the current stream using bounded temporary storage.
at::Tensor histogram_quantile_state(at::Tensor const& entropy) {
    int64_t const pixels = entropy.numel();
    // Three passes cover every ordered FP32 key bit with <=2048 bins per rank.
    constexpr int radix_bits = 11;
    constexpr int tail_bits  = kFloatKeyBits - 2 * radix_bits;
    constexpr int bins       = 1 << radix_bits;
    // Extend Torch's FP32 rank rule beyond its 2^24-element API limit.
    constexpr float quantile  = 0.90f;
    float const index         = quantile * static_cast<float>(pixels - 1);
    QuantileCount const lower = static_cast<QuantileCount>(index);
    QuantileCount const upper = static_cast<QuantileCount>(ceilf(index));
    auto state_tensor         = at::empty({static_cast<int64_t>(sizeof(QuantileState))}, entropy.options().dtype(at::kByte));
    auto counts               = at::empty({2 * bins}, entropy.options().dtype(at::kLong));
    auto* state               = reinterpret_cast<QuantileState*>(state_tensor.mutable_data_ptr<uint8_t>());
    auto* hist                = reinterpret_cast<QuantileCount*>(counts.mutable_data_ptr<int64_t>());
    auto stream               = at::cuda::getCurrentCUDAStream();
    initialize_quantile<<<1, 1, 0, stream>>>(state, lower, upper, static_cast<float>(index - lower));
    // Bound histogram contention and schedule enough work for large images.
    constexpr int pixels_per_block = 4096;
    constexpr int max_blocks       = 1024;
    int const blocks               = static_cast<int>(std::min<int64_t>((pixels + pixels_per_block - 1) / pixels_per_block, max_blocks));
    C10_CUDA_CHECK(cudaMemsetAsync(hist, 0, counts.nbytes(), stream));
    quantile_histogram<radix_bits, radix_bits + tail_bits><<<blocks, quantile_threads, bins * sizeof(QuantileCount), stream>>>(entropy.const_data_ptr<float>(), pixels, state, hist);
    select_quantile_bins<radix_bits, true><<<2, quantile_threads, 0, stream>>>(hist, state);
    C10_CUDA_CHECK(cudaMemsetAsync(hist, 0, counts.nbytes(), stream));
    quantile_histogram<radix_bits, tail_bits><<<blocks, quantile_threads, 2 * bins * sizeof(QuantileCount), stream>>>(entropy.const_data_ptr<float>(), pixels, state, hist);
    select_quantile_bins<radix_bits, false><<<2, quantile_threads, 0, stream>>>(hist, state);
    C10_CUDA_CHECK(cudaMemsetAsync(hist, 0, counts.nbytes(), stream));
    quantile_histogram<tail_bits, 0><<<blocks, quantile_threads, bins * sizeof(QuantileCount), stream>>>(entropy.const_data_ptr<float>(), pixels, state, hist);
    select_quantile_bins<tail_bits, false><<<2, quantile_threads, 0, stream>>>(hist, state);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return state_tensor;
}

// The threshold is device-resident. Fuse mask, label cast and both masked
// logits assignments, avoiding boolean indexing and its dynamic index buffers.
// Each thread owns one of pixels labels/mask values and the corresponding logits
// column; predictions and entropy are read-only, and update_logits gates writes.
template <typename label_t, bool update_logits>
__global__ void uncertainty_filter_kernel(float* logits, label_t const* predictions, float const* entropy,
                                          QuantileState const* quantile,
                                          uint8_t building_id, uint8_t* labels, bool* mask, int64_t pixels) {
    int64_t const pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixels)
        return;
    float const threshold = quantile_value(*quantile);
    bool const uncertain  = entropy[pixel] > threshold;
    mask[pixel]           = uncertain;
    labels[pixel]         = uncertain ? building_id : static_cast<uint8_t>(predictions[pixel]);
    if constexpr (update_logits) {
        if (uncertain) {
#pragma unroll
            for (int c = 0; c < kClasses; ++c)
                logits[static_cast<int64_t>(c) * pixels + pixel] = c == building_id ? 1.0f : 0.0f;
        }
    }
}
} // namespace

// Full inference postprocessing on the caller's stream. Select the two ranks
// with refining histograms, then fuse interpolation with uncertainty filtering.
// logits is CUDA FP32 [20,H,W]; pred_labels is integral [1,H,W]. Write entropy
// FP32, labels uint8, and mask bool [H,W] on that device. Only estimate_logits
// overwrites uncertain logits columns with the building_id one-hot vector.
void mask2former_postprocess_out(at::Tensor const& logits, at::Tensor const& pred_labels, int64_t building_id,
                                 bool estimate_logits, at::Tensor const& entropy, at::Tensor const& labels,
                                 at::Tensor const& mask) {
    TORCH_CHECK(pred_labels.device() == logits.device() && pred_labels.is_contiguous() &&
                    pred_labels.dim() == 3 && pred_labels.size(0) == 1 &&
                    pred_labels.sizes().slice(1) == logits.sizes().slice(1) &&
                    at::isIntegralType(pred_labels.scalar_type(), false),
                "invalid predicted labels");
    TORCH_CHECK(building_id >= 0 && building_id < kClasses, "invalid building class");
    TORCH_CHECK(labels.device() == logits.device() && labels.scalar_type() == at::kByte &&
                    labels.is_contiguous() && labels.sizes() == logits.sizes().slice(1),
                "invalid labels output");
    TORCH_CHECK(mask.device() == logits.device() && mask.scalar_type() == at::kBool &&
                    mask.is_contiguous() && mask.sizes() == labels.sizes(),
                "invalid mask output");
    c10::cuda::CUDAGuard guard(logits.device());
    categorical_entropy_out(logits, entropy);
    int64_t const pixels      = entropy.numel();
    auto const quantile_state = histogram_quantile_state(entropy);
    constexpr int threads     = 256;
    auto const blocks         = (pixels + threads - 1) / threads;
    auto stream               = at::cuda::getCurrentCUDAStream(logits.get_device());
    AT_DISPATCH_INTEGRAL_TYPES(pred_labels.scalar_type(), "mask2former_uncertainty", [&] {
        if (estimate_logits) {
            uncertainty_filter_kernel<scalar_t, true><<<blocks, threads, 0, stream>>>(
                logits.mutable_data_ptr<float>(), pred_labels.const_data_ptr<scalar_t>(), entropy.const_data_ptr<float>(),
                reinterpret_cast<QuantileState const*>(quantile_state.const_data_ptr<uint8_t>()), static_cast<uint8_t>(building_id),
                labels.mutable_data_ptr<uint8_t>(), mask.mutable_data_ptr<bool>(), pixels);
        } else {
            uncertainty_filter_kernel<scalar_t, false><<<blocks, threads, 0, stream>>>(
                logits.mutable_data_ptr<float>(), pred_labels.const_data_ptr<scalar_t>(), entropy.const_data_ptr<float>(),
                reinterpret_cast<QuantileState const*>(quantile_state.const_data_ptr<uint8_t>()), static_cast<uint8_t>(building_id),
                labels.mutable_data_ptr<uint8_t>(), mask.mutable_data_ptr<bool>(), pixels);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
