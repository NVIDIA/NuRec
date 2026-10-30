# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Legacy PPISP (Slang-based) implementation.

This module retains the Slang-kernel PPISP implementation for rendering
backwards compatibility with old checkpoints. Training is not supported.
"""

import torch
import torch.nn as nn

from torch import Tensor

from nurec.models.post_processings.constants import DEFAULT_SOURCE_CHROMS_VALUES
from nurec.models.post_processings.ppisp_legacy.slang import PPISPLegacyFunction


@torch.no_grad()
def softplus_inverse(x: float, epsilon: float = 1e-5) -> float:
    """Compute the inverse of softplus function for initialization."""
    clamped_value = max(epsilon, float(x))
    tensor_value = torch.tensor(clamped_value)
    return float((tensor_value.exp() - 1).log())


class PPISPLegacy(nn.Module):
    """
    Legacy Physically Plausible Image Signal Processing (PPISP) module using Slang kernels.

    This implementation is kept for rendering backwards compatibility with old
    checkpoints only. Training (optimizer creation, loss computation) is not
    supported. The Slang kernel applies:
    1. Exposure compensation (per-frame)
    2. Vignetting correction (per-camera, per-channel)
    3. Color correction via homography (per-frame)
    4. Camera Response Function via piecewise power function (per-camera, per-channel)
    """

    _default_source_chroms: Tensor

    def __init__(
        self, device: torch.device | str, n_frames_per_camera: list[int], num_cameras: int, num_frames: int
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_frames_per_camera = n_frames_per_camera
        self.num_cameras = num_cameras
        self.num_frames = num_frames

        self.exposure_params = nn.Parameter(torch.zeros(num_frames, device=self.device))

        self.vignetting_params = nn.Parameter(torch.zeros(num_cameras, 3, 5, device=self.device))

        # Initialize homography as identity matrix [1,0,0,0,1,0,0,0] for each frame
        identity_homography = torch.zeros(num_frames, 8, device=self.device)
        identity_homography[:, 0] = 1.0  # H[0,0] = 1
        identity_homography[:, 4] = 1.0  # H[1,1] = 1
        self.color_params = nn.Parameter(identity_homography)

        # Initialize CRF parameters: 7 raw params per camera per channel
        # [x0_offset, y0, y1_fract, toe_length, shoulder_length, shoulder_overshoot, gamma]
        crf_raw = torch.zeros(7, device=self.device)
        crf_raw[0] = softplus_inverse(0.166 / (1.0 + 1.0))  # x0_offset
        crf_raw[1] = float(torch.log(torch.tensor(0.25) / (1.0 - 0.25)))  # y0 (sigmoid_inverse)
        crf_raw[2] = float(
            torch.log(torch.tensor((0.9 - 0.25) / (1.0 - 0.25)) / (1.0 - (0.9 - 0.25) / (1.0 - 0.25)))
        )  # y1_fract
        crf_raw[3] = softplus_inverse(1.0)  # toe_length
        crf_raw[4] = softplus_inverse(1.0)  # shoulder_length
        crf_raw[5] = softplus_inverse(0.1)  # shoulder_overshoot
        crf_raw[6] = softplus_inverse(1.0 / 2.2)  # gamma
        self.crf_params = nn.Parameter(crf_raw.unsqueeze(0).unsqueeze(0).expand(num_cameras, 3, 7).contiguous())

        # Register default source chromaticities as a buffer to avoid CUDA sync on each access
        self.register_buffer(
            "_default_source_chroms",
            torch.tensor(DEFAULT_SOURCE_CHROMS_VALUES, device=self.device),
        )

        # Makes sure parameters are on the correct device, even when launched out of the main pipeline
        self.to(self.device)

    def forward(
        self,
        rgb: torch.Tensor,
        coords_xy: torch.Tensor,
        unique_camera_idcs: torch.Tensor,
        frame_idcs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the legacy PPISP pipeline via Slang kernel.

        Args:
            rgb: Input RGB values of shape (B, 3)
            coords_xy: Normalized image coordinates of shape (B, 2)
            unique_camera_idcs: Camera indices of shape (B,), dtype int16
            frame_idcs: Frame indices of shape (B,), dtype int32

        Returns:
            Processed RGB values of same shape as input.
        """
        assert rgb.shape[0] == coords_xy.shape[0], "Batch sizes must match"
        assert rgb.shape[0] == unique_camera_idcs.shape[0], "Batch sizes must match"
        assert rgb.shape[0] == frame_idcs.shape[0], "Batch sizes must match"
        assert rgb.shape[1] == 3, f"Expected rgb to have 3 channels, got {rgb.shape[1]}"
        assert coords_xy.shape[1] == 2, f"Expected coords_xy to have 2 dimensions (x,y), got {coords_xy.shape[1]}"

        return PPISPLegacyFunction.apply(
            rgb.shape[0],
            self.num_cameras,
            self.num_frames,
            self.exposure_params,
            self.vignetting_params,
            self.color_params,
            self.crf_params,
            rgb,
            coords_xy,
            unique_camera_idcs,
            frame_idcs,
        )
