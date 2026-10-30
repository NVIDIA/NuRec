// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

inline constexpr int TRACK_FRAME_SLOTS_THREADS_PER_BLOCK = 128;

/** Write track-relative nearest-frame indices, preferring the earlier neighbor on ties.
 * @param target Query timestamp in microseconds.
 * @param timestamps [N] source timestamps, sorted within each track.
 * @param packinfo [T, 2] start/count pairs; empty/singleton tracks produce zero.
 * @param slots Writable [T] output, not aliasing any input.
 * All tensors must be int64 on one CUDA device and may be strided.
 * Packed ranges must fit timestamps and timestamp differences must fit int64.
 * The output is written on the device's current stream without synchronization.
 * Invalid tensor metadata raises an exception; data invariants are caller-owned.
 */
void nearest_track_frame_slots_cu(
    int64_t target, const torch::Tensor& timestamps, const torch::Tensor& packinfo, const torch::Tensor& slots);
