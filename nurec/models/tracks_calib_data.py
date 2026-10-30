# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch


@dataclass(slots=True, frozen=True)
class TracksCalibData:
    """Base class for calibration state consumed by collector backends."""


@dataclass(slots=True)
class TracksCalibResiduals:
    """Named residual families produced by a structured track calibrator."""

    consistency: torch.Tensor
    smoothness: torch.Tensor
    pose_regularization: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the residual tensors in their canonical family order."""
        return self.consistency, self.smoothness, self.pose_regularization


@dataclass(slots=True, frozen=True)
class ElevationAwareUnicycleTracksCalibData(TracksCalibData):
    """Immutable view of elevation-aware unicycle calibration state.

    Pose channels are indexed by packed keyframe. Motion channels are indexed
    by adjacent keyframe pair. Ground transforms are indexed by track.

    Field bindings cannot be replaced. Their tensors remain live references to
    the calibrator's parameters and buffers rather than copied values.
    """

    xy: torch.Tensor
    height: torch.Tensor
    theta: torch.Tensor
    pitch: torch.Tensor
    velocity: torch.Tensor
    theta_velocity: torch.Tensor
    pitch_velocity: torch.Tensor
    pose_gradient_mask: torch.Tensor
    tracks_idx: torch.Tensor
    pose_pair_start_indices: torch.Tensor
    motion_triplet_start_indices: torch.Tensor
    time_delta_s: torch.Tensor
    time_delta_v_s: torch.Tensor
    initial_pose_xy: torch.Tensor
    initial_pose_height: torch.Tensor
    ground_t: torch.Tensor
    ground_q: torch.Tensor
    Tclc_tquat: torch.Tensor
    pitch_slope_clamp_rad: float
