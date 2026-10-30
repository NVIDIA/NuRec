// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#include <cstddef>

namespace py = pybind11;

namespace {

constexpr const char* GPUMEMORY_ALLOCATE_CAPSULE_NAME   = "tinycudann.GPUMemoryAllocator.allocate";
constexpr const char* GPUMEMORY_DEALLOCATE_CAPSULE_NAME = "tinycudann.GPUMemoryAllocator.deallocate";

using AllocateFn   = void* (*)(size_t n_bytes, void* context);
using DeallocateFn = void (*)(void* ptr, void* context);

// The two signatures below are fixed by the tinycudann GPUMemoryAllocator C ABI
// (see AllocateFn / DeallocateFn above): they are cast to those exact function
// pointer types before being handed to tinycudann, so the untyped pointers cannot
// be replaced with typed ones without breaking the contract.
void* torch_cuda_allocate(size_t n_bytes, void*) { // NOSONAR - signature fixed by the tinycudann C ABI
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(n_bytes, stream);
}

void torch_cuda_deallocate(void* ptr, void*) { // NOSONAR - signature fixed by the tinycudann C ABI
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

py::capsule torch_cuda_allocate_capsule() {
    return py::capsule(reinterpret_cast<void*>(static_cast<AllocateFn>(&torch_cuda_allocate)), GPUMEMORY_ALLOCATE_CAPSULE_NAME);
}

py::capsule torch_cuda_deallocate_capsule() {
    return py::capsule(reinterpret_cast<void*>(static_cast<DeallocateFn>(&torch_cuda_deallocate)), GPUMEMORY_DEALLOCATE_CAPSULE_NAME);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_cuda_allocate", &torch_cuda_allocate_capsule);
    m.def("torch_cuda_deallocate", &torch_cuda_deallocate_capsule);
}
