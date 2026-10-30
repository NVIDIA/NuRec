// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A track calibrator produces one world-space pose for each packed keyframe.
interface TracksCalib
{
    [Differentiable]
    transform::SE3 calib(no_diff uint pose_idx);
};
