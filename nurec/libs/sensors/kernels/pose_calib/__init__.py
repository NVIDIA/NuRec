# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose calibration kernels package - Layer 0 GPU operations for pose calibration."""

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch  # noqa: F401

from nurec.libs.sensors.kernels.pose_calib.bindings import compute_poses_and_timestamps


def __getattr__(name: str):
    """Lazy-load CUDA and Slang pose_calib modules."""
    if name == "pose_calib_cuda":
        import nurec.libs.sensors.libpose_calib_cuda_cc as pose_calib_cuda  # type: ignore

        globals()["pose_calib_cuda"] = pose_calib_cuda
        return pose_calib_cuda
    if name == "pose_calib_slang":
        import nurec.libs.sensors.libpose_calib_slang_cc as pose_calib_slang  # type: ignore

        globals()["pose_calib_slang"] = pose_calib_slang
        return pose_calib_slang
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # CUDA module (production, lazy-loaded)
    "pose_calib_cuda",
    # Kernel functions
    "compute_poses_and_timestamps",
]
