// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

struct RollingShutterParameters {
    std::array<float, 7 * 2> T_world_sensors; // represents two tquat [t,q] poses
    std::array<uint64_t, 2> timestamps_us;
};
