// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "finalize_weighted_losses.h"

#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cub/block/block_reduce.cuh>

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace nurec::loss_aggregation_ops {
namespace {

constexpr int kThreadsPerBlock = 256;
using BlockReduce              = cub::BlockReduce<float, kThreadsPerBlock>;

enum class FinalizeScalarType : int32_t {
    Float32,
    Float16,
    BFloat16,
};

struct FinalizeDescriptor {
    void* data;
    float weight;
    FinalizeScalarType scalar_type;
};

static_assert(sizeof(FinalizeDescriptor) == 16, "unexpected finalizer descriptor layout");
static_assert(std::is_standard_layout_v<FinalizeDescriptor>, "finalizer descriptors must have stable layout");
static_assert(
    std::is_trivially_copyable_v<FinalizeDescriptor>,
    "finalizer descriptors must support bytewise upload");

template <int Capacity>
struct InlineFinalizeDescriptors {
    FinalizeDescriptor values[Capacity];

    const __device__ __forceinline__ FinalizeDescriptor& operator[](const int64_t index) const {
        return values[index];
    }
};

static_assert(
    sizeof(InlineFinalizeDescriptors<16>) == 16 * sizeof(FinalizeDescriptor),
    "unexpected inline finalizer descriptor layout");

template <typename Launch>
void dispatch_inline_descriptor_capacity(const int64_t n_descriptors, Launch&& launch) {
    if (n_descriptors <= 1) {
        std::forward<Launch>(launch)(std::integral_constant<int, 1>{});
    } else if (n_descriptors <= 2) {
        std::forward<Launch>(launch)(std::integral_constant<int, 2>{});
    } else if (n_descriptors <= 4) {
        std::forward<Launch>(launch)(std::integral_constant<int, 4>{});
    } else if (n_descriptors <= 8) {
        std::forward<Launch>(launch)(std::integral_constant<int, 8>{});
    } else {
        std::forward<Launch>(launch)(std::integral_constant<int, 16>{});
    }
}

FinalizeScalarType finalize_scalar_type(const at::ScalarType dtype) {
    switch (dtype) {
    case at::kFloat:
        return FinalizeScalarType::Float32;
    case at::kHalf:
        return FinalizeScalarType::Float16;
    case at::kBFloat16:
        return FinalizeScalarType::BFloat16;
    default:
        TORCH_CHECK(false, "unsupported finalizer scalar dtype ", dtype);
    }
}

__device__ __forceinline__ float load_scalar(const FinalizeDescriptor& descriptor) {
    switch (descriptor.scalar_type) {
    case FinalizeScalarType::Float32:
        return *static_cast<const float*>(descriptor.data);
    case FinalizeScalarType::Float16:
        return static_cast<float>(*static_cast<const c10::Half*>(descriptor.data));
    case FinalizeScalarType::BFloat16:
        return static_cast<float>(*static_cast<const c10::BFloat16*>(descriptor.data));
    }
    return 0.0f;
}

__device__ __forceinline__ void store_scalar(const FinalizeDescriptor& descriptor, const float value) {
    switch (descriptor.scalar_type) {
    case FinalizeScalarType::Float32:
        *static_cast<float*>(descriptor.data) = value;
        break;
    case FinalizeScalarType::Float16:
        *static_cast<c10::Half*>(descriptor.data) = c10::Half(value);
        break;
    case FinalizeScalarType::BFloat16:
        *static_cast<c10::BFloat16*>(descriptor.data) = c10::BFloat16(value);
        break;
    }
}

at::Tensor upload_descriptors(const std::vector<FinalizeDescriptor>& descriptors, const at::Device device) {
    TORCH_CHECK(
        c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None,
        "finalize_accumulated_losses: CUDA graph capture supports at most 16 descriptors; got ",
        descriptors.size() + 1);
    const int64_t bytes = static_cast<int64_t>(descriptors.size() * sizeof(FinalizeDescriptor));
    at::Tensor host =
        at::empty({bytes}, at::TensorOptions().device(at::kCPU).dtype(at::kByte).pinned_memory(true));
    std::memcpy(host.data_ptr(), descriptors.data(), static_cast<size_t>(bytes));
    at::Tensor uploaded = at::empty({bytes}, at::TensorOptions().device(device).dtype(at::kByte));
    uploaded.copy_(host, true);
    return uploaded;
}

template <typename Descriptors>
__global__ void finalize_accumulated_losses_fwd_kernel(
    const float* __restrict__ weighted_vector,
    const int64_t n_slots,
    const __grid_constant__ Descriptors descriptors,
    const int64_t n_scalars,
    float* __restrict__ total,
    float* __restrict__ weighted_values) {
    float local_sum = 0.0f;
    for (int64_t index = threadIdx.x; index < n_slots; index += blockDim.x) {
        const float value      = weighted_vector[index];
        weighted_values[index] = value;
        local_sum += value;
    }
    for (int64_t index = 0; index < n_scalars; ++index) {
        if (threadIdx.x == index % blockDim.x) {
            const FinalizeDescriptor descriptor = descriptors[index];
            const float weighted =
                descriptor.weight == 0.0f ? 0.0f : descriptor.weight * load_scalar(descriptor);
            weighted_values[n_slots + index] = weighted;
            local_sum += weighted;
        }
    }

    __shared__ typename BlockReduce::TempStorage temp_storage;
    const float block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (threadIdx.x == 0) {
        total[0] = block_sum;
    }
}

template <typename Descriptors>
__global__ void finalize_accumulated_losses_bwd_kernel(
    float* __restrict__ v_weighted_vector,
    const int64_t n_slots,
    const __grid_constant__ Descriptors descriptors,
    const int64_t n_scalars,
    const float* __restrict__ v_total,
    const float* __restrict__ v_weighted_values) {
    const float total_upstream = v_total != nullptr ? v_total[0] : 0.0f;
    for (int64_t index = threadIdx.x; index < n_slots; index += blockDim.x) {
        v_weighted_vector[index] =
            v_weighted_values != nullptr ? total_upstream + v_weighted_values[index] : total_upstream;
    }
    for (int64_t index = 0; index < n_scalars; ++index) {
        if (threadIdx.x == index % blockDim.x) {
            const FinalizeDescriptor descriptor = descriptors[index];
            if (descriptor.weight == 0.0f) {
                store_scalar(descriptor, 0.0f);
            } else {
                const float scalar_upstream =
                    v_weighted_values != nullptr ? v_weighted_values[n_slots + index] : 0.0f;
                store_scalar(descriptor, descriptor.weight * (total_upstream + scalar_upstream));
            }
        }
    }
}

FinalizeDescriptor make_descriptor(
    const int64_t index,
    at::TensorList scalar_values,
    const std::vector<double>& scalar_weights) {
    return {
        scalar_values[index].data_ptr(),
        static_cast<float>(scalar_weights[index]),
        finalize_scalar_type(scalar_values[index].scalar_type())};
}

std::vector<FinalizeDescriptor> make_descriptors(
    at::TensorList scalar_values,
    const std::vector<double>& scalar_weights) {
    std::vector<FinalizeDescriptor> descriptors;
    descriptors.reserve(scalar_values.size());
    for (int64_t index = 0; index < static_cast<int64_t>(scalar_values.size()); ++index) {
        descriptors.push_back(make_descriptor(index, scalar_values, scalar_weights));
    }
    return descriptors;
}

template <int Capacity>
void launch_finalize_accumulated_losses_fwd_inline(
    const at::Tensor& weighted_vector,
    at::TensorList scalar_values,
    const std::vector<double>& scalar_weights,
    float* total,
    float* weighted_values,
    const cudaStream_t stream) {
    InlineFinalizeDescriptors<Capacity> descriptors{};
    const int64_t n_scalars = static_cast<int64_t>(scalar_values.size());
    for (int64_t index = 0; index < n_scalars; ++index) {
        descriptors.values[index] = make_descriptor(index, scalar_values, scalar_weights);
    }
    finalize_accumulated_losses_fwd_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
        weighted_vector.data_ptr<float>(), weighted_vector.numel(), descriptors, n_scalars, total, weighted_values);
}

template <int Capacity>
void launch_finalize_accumulated_losses_bwd_inline(
    at::Tensor& v_weighted_vector,
    at::TensorList v_scalar_values,
    const std::vector<double>& scalar_weights,
    const float* v_total,
    const float* v_weighted_values,
    const cudaStream_t stream) {
    InlineFinalizeDescriptors<Capacity> descriptors{};
    const int64_t n_scalars = static_cast<int64_t>(v_scalar_values.size());
    for (int64_t index = 0; index < n_scalars; ++index) {
        descriptors.values[index] = make_descriptor(index, v_scalar_values, scalar_weights);
    }
    finalize_accumulated_losses_bwd_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
        v_weighted_vector.data_ptr<float>(), v_weighted_vector.numel(), descriptors, n_scalars, v_total, v_weighted_values);
}

void record_tensor(const at::Tensor& tensor, const c10::cuda::CUDAStream& stream) {
    if (tensor.numel() > 0) {
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
}

} // namespace

void launch_finalize_accumulated_losses_fwd(
    const at::Tensor& weighted_vector,
    at::TensorList scalar_values,
    const std::vector<double>& scalar_weights,
    at::Tensor& total,
    at::Tensor& weighted_values) {
    const c10::cuda::CUDAGuard device_guard(total.device());
    const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(total.get_device());
    record_tensor(weighted_vector, stream);
    for (const at::Tensor& tensor : scalar_values) {
        record_tensor(tensor, stream);
    }
    record_tensor(total, stream);
    record_tensor(weighted_values, stream);

    const int64_t n_scalars = static_cast<int64_t>(scalar_values.size());
    // Count the vector as one descriptor to preserve the existing graph limit.
    const int64_t n_descriptors = 1 + n_scalars;
    if (n_descriptors <= 16) {
        dispatch_inline_descriptor_capacity(
            n_descriptors,
            [&](auto capacity) {
                launch_finalize_accumulated_losses_fwd_inline<decltype(capacity)::value>(
                    weighted_vector, scalar_values, scalar_weights, total.data_ptr<float>(), weighted_values.data_ptr<float>(), stream);
            });
    } else {
        const std::vector<FinalizeDescriptor> host_descriptors = make_descriptors(scalar_values, scalar_weights);
        const at::Tensor device_descriptors                    = upload_descriptors(host_descriptors, total.device());
        finalize_accumulated_losses_fwd_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            weighted_vector.data_ptr<float>(),
            weighted_vector.numel(),
            reinterpret_cast<const FinalizeDescriptor*>(device_descriptors.data_ptr()),
            n_scalars,
            total.data_ptr<float>(),
            weighted_values.data_ptr<float>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_finalize_accumulated_losses_bwd(
    const at::Tensor* v_total,
    const at::Tensor* v_weighted_values,
    const std::vector<double>& scalar_weights,
    at::Tensor& v_weighted_vector,
    at::TensorList v_scalar_values) {
    const c10::cuda::CUDAGuard device_guard(v_weighted_vector.device());
    const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(v_weighted_vector.get_device());
    if (v_total != nullptr) {
        record_tensor(*v_total, stream);
    }
    if (v_weighted_values != nullptr) {
        record_tensor(*v_weighted_values, stream);
    }
    record_tensor(v_weighted_vector, stream);
    for (const at::Tensor& tensor : v_scalar_values) {
        record_tensor(tensor, stream);
    }

    const int64_t n_scalars             = static_cast<int64_t>(v_scalar_values.size());
    const int64_t n_descriptors         = 1 + n_scalars;
    const float* v_total_data           = v_total != nullptr ? v_total->data_ptr<float>() : nullptr;
    const float* v_weighted_values_data = v_weighted_values != nullptr ? v_weighted_values->data_ptr<float>() : nullptr;
    if (n_descriptors <= 16) {
        dispatch_inline_descriptor_capacity(
            n_descriptors,
            [&](auto capacity) {
                launch_finalize_accumulated_losses_bwd_inline<decltype(capacity)::value>(
                    v_weighted_vector, v_scalar_values, scalar_weights, v_total_data, v_weighted_values_data, stream);
            });
    } else {
        const std::vector<FinalizeDescriptor> host_descriptors = make_descriptors(v_scalar_values, scalar_weights);
        const at::Tensor device_descriptors                    = upload_descriptors(host_descriptors, v_weighted_vector.device());
        finalize_accumulated_losses_bwd_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            v_weighted_vector.data_ptr<float>(),
            v_weighted_vector.numel(),
            reinterpret_cast<const FinalizeDescriptor*>(device_descriptors.data_ptr()),
            n_scalars,
            v_total_data,
            v_weighted_values_data);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace nurec::loss_aggregation_ops
