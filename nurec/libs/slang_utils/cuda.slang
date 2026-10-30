// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

namespace cuda_utils {

// Assert on CUDA devices; a no-op on other targets.
[ForceInline]
void cuda_device_assert(bool condition)
{
    __target_switch
    {
    case cuda:
        __requirePrelude("#include <assert.h>");
        __intrinsic_asm "assert($0)";
    default:
        return;
    }
}

// Slang exposes CUDA block indices, but not grid dimensions. Read the launch
// dimensions directly so callers can tune the bounded grid independently.
[ForceInline]
uint cuda_linear_block_idx()
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)";
    default:
        return 0;
    }
}

// Return the total number of blocks in the CUDA launch, including all grid axes.
[ForceInline]
uint cuda_block_count()
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "gridDim.x * gridDim.y * gridDim.z";
    default:
        return 1;
    }
}

}
