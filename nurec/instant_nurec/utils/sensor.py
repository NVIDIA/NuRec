# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from typing import Literal

import numpy as np
import torch

from ncore.impl.data.types import (
    ConcreteCameraModelParametersUnion,
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from ncore.impl.sensors.camera import camera_model_from_parameters


log = logging.getLogger(__name__)

# Tracks (resolution, principal_point) signatures already warned about in
# to_simple_pinhole_model_parameters, so the FOV-clamp warning is emitted once
# per camera rather than once per frame/chunk.
_FOV_CLAMP_WARNED_SIGNATURES: set[tuple] = set()


def to_simple_pinhole_model_parameters(
    camera_model_parameters: ConcreteCameraModelParametersUnion,
    method: Literal["corner", "horizontal", "vertical"],
    reduce: Literal["min", "max", "mean"],
    percentile: float = 1.0,
) -> OpenCVPinholeCameraModelParameters:
    """
    Converts any camera model parameters to simple pinhole model parameters (equal focal lengths, principal point at image center).
    The method works by computing the rays either at the corners or the horizontal/vertical directions of the image (specified by `method`),
      and then reduce the fov of the rays (specified by `reduce`). The reduced fov value is then used to determine the fov value at `method` direction
      in the final simple pinhole model parameters.

    Args:
        camera_model_parameters: The camera model parameters to convert.
        method: The method to use to compute the rays.
        reduce: The method to use to reduce the fov of the rays.
        percentile: The percentile of the pixels to determine the mapping (use a small value to approximate 1st-order gradient as implemented in apps.viewer.viewpoint).
    Returns:
        The simple pinhole model parameters.
    """
    camera_model = camera_model_from_parameters(camera_model_parameters, device="cpu")
    original_resolution = camera_model_parameters.resolution.astype(np.int64)
    original_principal_point = camera_model_parameters.principal_point
    pinhole_principal_point = camera_model_parameters.resolution.astype(np.float32) / 2.0

    match method:
        case "corner":
            image_points = torch.tensor(
                [
                    [0, 0],
                    [0, original_resolution[1]],
                    [original_resolution[0], 0],
                    [original_resolution[0], original_resolution[1]],
                ]
            )
            pinhole_pixel_distance = np.linalg.norm(pinhole_principal_point).item()
        case "horizontal":
            image_points = torch.tensor(
                [
                    [0, original_principal_point[1]],
                    [original_resolution[0], original_principal_point[1]],
                ]
            )
            pinhole_pixel_distance = pinhole_principal_point[0].item()
        case "vertical":
            image_points = torch.tensor(
                [
                    [original_principal_point[0], 0],
                    [original_principal_point[0], original_resolution[1]],
                ]
            )
            pinhole_pixel_distance = pinhole_principal_point[1].item()
        case _:
            raise ValueError(f"Invalid method: {method}")

    if percentile != 1.0:
        center_torch = torch.from_numpy(original_principal_point)
        image_points = (image_points.float() - center_torch) * percentile + center_torch
        pinhole_pixel_distance *= percentile

    camera_rays = camera_model.image_points_to_camera_rays(image_points.float())
    camera_rays = torch.nn.functional.normalize(camera_rays, dim=-1)
    angles = torch.arccos(camera_rays[:, 2])
    match reduce:
        case "min":
            fov = torch.min(angles).item()
        case "max":
            fov = torch.max(angles).item()
        case "mean":
            fov = torch.mean(angles).item()
        case _:
            raise ValueError(f"Invalid reduce: {reduce}")

    # Border pixels can fall outside a lens's valid FOV, where image_points_to_camera_rays extrapolates to a
    # bogus (often >90 deg) ray; a pinhole also can't represent a half-FOV >= 90 deg. Bound the probed half-FOV
    # by the camera's valid FOV (`max_angle` for FTheta/fisheye, >0 by construction) so the focal stays positive.
    max_half_fov_rad = float(np.deg2rad(89.0))
    match camera_model_parameters:
        case FThetaCameraModelParameters() | OpenCVFisheyeCameraModelParameters():
            max_half_fov_rad = min(max_half_fov_rad, float(camera_model_parameters.max_angle))
        case OpenCVPinholeCameraModelParameters():
            pass  # no distortion to extrapolate; < 90 deg by construction

    if not np.isfinite(fov) or fov > max_half_fov_rad:
        signature = (
            tuple(np.asarray(camera_model_parameters.resolution).tolist()),
            tuple(np.round(np.asarray(camera_model_parameters.principal_point, dtype=np.float64), 2).tolist()),
        )
        if signature not in _FOV_CLAMP_WARNED_SIGNATURES:
            _FOV_CLAMP_WARNED_SIGNATURES.add(signature)
            log.warning(
                "to_simple_pinhole_model_parameters: probed half-FOV %.1f deg exceeds valid/pinhole limit "
                "%.1f deg; bounding it (camera resolution=%s, principal_point=%s).",
                np.rad2deg(fov),
                np.rad2deg(max_half_fov_rad),
                signature[0],
                signature[1],
            )
        fov = max_half_fov_rad

    focal = pinhole_pixel_distance / np.tan(fov)
    return OpenCVPinholeCameraModelParameters(
        resolution=np.copy(camera_model_parameters.resolution),
        shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        principal_point=pinhole_principal_point,
        focal_length=np.array([focal, focal], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )


def rectify_image(
    source_image: torch.Tensor,
    source_camera_model_parameters: ConcreteCameraModelParametersUnion,
    target_camera_model_parameters: ConcreteCameraModelParametersUnion,
) -> torch.Tensor:
    """
    Rectifies the source image to the target camera model parameters.

    Args:
        source_image: The source image to rectify (H, W, 3).
        source_camera_model_parameters: The source camera model parameters.
        target_camera_model_parameters: The target camera model parameters.

    Returns:
        The rectified image.
    """
    device = source_image.device
    source_camera_model = camera_model_from_parameters(source_camera_model_parameters, device=device)
    target_camera_model = camera_model_from_parameters(target_camera_model_parameters, device=device)

    target_image_points = torch.stack(
        torch.meshgrid(
            torch.arange(target_camera_model_parameters.resolution[0], device=device, dtype=torch.float32),
            torch.arange(target_camera_model_parameters.resolution[1], device=device, dtype=torch.float32),
            indexing="xy",
        ),
        dim=-1,
    )
    target_camera_rays = target_camera_model.image_points_to_camera_rays(target_image_points.reshape(-1, 2))
    source_image_points = source_camera_model.camera_rays_to_image_points(target_camera_rays).image_points

    # Normalize to -1 to 1 (align_corners=False since it's image points)
    source_image_points = (
        source_image_points / torch.from_numpy(source_camera_model_parameters.resolution).to(device) * 2 - 1
    )
    source_image_points = source_image_points.reshape(
        1, target_camera_model_parameters.resolution[1], target_camera_model_parameters.resolution[0], 2
    )
    target_image = torch.nn.functional.grid_sample(
        source_image.moveaxis(-1, 0)[None], source_image_points, align_corners=False
    )
    return target_image.squeeze(0).moveaxis(0, -1)
