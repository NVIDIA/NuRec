// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Common interfaces and structs shared between camera and LiDAR sensors.

// ============================================================================
// Common Output Structs
// ============================================================================

// World ray output for sensor ray generation (used by both cameras and LiDARs)
struct WorldRay {
    float3 origin;
    float3 direction;
};
