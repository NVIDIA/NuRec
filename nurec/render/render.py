# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import os

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Self, Tuple, cast

import lietorch
import torch

from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter

import nurec.compression.usdz as compact_usdz

from ncore.data import ConcreteCameraModelParametersUnion
from ncore.impl.common.transformations import transform_point_cloud
from nurec.artifact import Artifact
from nurec.config.model import ModelConfig, RendererConfigType
from nurec.config.trainer import TrainerConfig
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.track_split import TrackSplitResolver
from nurec.datasets.tracks import CuboidTracks
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.instant_nurec.primitives.celsius_primitive import CelsiusInstantNuRecPrimitive
from nurec.instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from nurec.libs.sensors.kernels.cameras.bindings import generate_image_points
from nurec.libs.vren.interface import (  # type: ignore
    image_points_to_world_rays_shutter_pose,
)
from nurec.libs.vren.lidars import (  # type: ignore
    elements_to_world_rays_shutter_pose,
)
from nurec.models.calib import FreePoseCalib
from nurec.models.composite import CompositeModel
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.renderers import BaseGaussianRenderer
from nurec.models.gaussians.utils import Asset, make_ppisp_crf_inverse_rgb_post_processing
from nurec.render.actors import ActorsSnapshot, ActorTracks
from nurec.render.lidar_firing_pattern import LidarFiringPattern, build_unstructured_tiling
from nurec.render.utils import frame_transform_poses, transform_intrinsics_to_resolution
from nurec.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    LidarFrameLabels,
    RenderingBatch,
    RenderingData,
)
from nurec.utils.geometry import interpolate_se3_poses, quat_slerp, se3_matrix_inverse, tquat_to_se3_matrix
from nurec.utils.lidar_model import LidarModelBundle
from nurec.utils.lidar_post_processing import angle_gradient_filter, distance_based_filter
from nurec.utils.misc import map_optional, to_float_device, tree_map, unpack_optional
from nurec.utils.profiling import ScopedTimer
from nurec.utils.types import (
    Checkpoint,
    FrameConversion,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    RigTrajectories,
    TrackFlags,
)
from nurec.utils.upgrade import upgrade_config, upgrade_model


log = logging.getLogger(__name__)


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


class MissingRenderOutputError(ValueError):
    """A requested render output (distance/opacity/semantic) was not produced because the model or
    render path lacks the corresponding head/capability.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep working; callers that
    distinguish model capability (e.g. the gRPC server) can catch this specifically and surface an
    actionable ``FAILED_PRECONDITION`` instead of a generic ``INVALID_ARGUMENT``.
    """


@dataclass
class CameraRenderOptions:
    """Select fields of :class:`CameraFrame` to populate during rendering.

    Used by :meth:`RenderableModel.render_camera_frame` and
    :meth:`RenderableModel.render_camera_frame_from_ray_bundle`.

    Note: ``distance_image=True``, ``opacity_image=True``, and ``semantic_image=True`` raise
    :class:`MissingRenderOutputError` if the model or render path does not produce the requested
    output.
    """

    color_image: bool = True
    distance_image: bool = False
    opacity_image: bool = False
    semantic_image: bool = False

    def __post_init__(self) -> None:
        if not (self.color_image or self.distance_image or self.opacity_image or self.semantic_image):
            raise ValueError("At least one CameraRenderOptions flag must be True")


@dataclass
class CameraFrame:
    """Rendered framebuffers of shape (height, width, channels) returned from rendering functions.

    Fields are optional since the fields to be populated can be configured in the render calls.

    Fields:
      color_image: RGB image of shape (height, width, 3) as torch.float32 with values in [0, 1]
      distance_image: Distance image of shape (height, width) and type torch.float32 (ray distance)
      opacity_image: Opacity image of shape (height, width) and type torch.float32
      semantic_image: Per-pixel class index of shape (height, width), torch.int64 (from semantic_logits argmax)
    """

    color_image: Optional[torch.Tensor] = None
    distance_image: Optional[torch.Tensor] = None
    opacity_image: Optional[torch.Tensor] = None
    semantic_image: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.color_image is not None:
            assert self.color_image.dim() == 3, (
                f"Color image tensor does not have 3 dimensions, got {self.color_image.dim()}"
            )
            assert self.color_image.shape[2] == 3, (
                f"Color image does not have 3 channels, got shape {tuple(self.color_image.shape)}"
            )

        if self.distance_image is not None:
            assert self.distance_image.dim() == 2, (
                f"Depth image tensor does not have 2 dimensions, got {self.distance_image.dim()}"
            )

        if self.opacity_image is not None:
            assert self.opacity_image.dim() == 2, (
                f"Opacity image tensor does not have 2 dimensions, got {self.opacity_image.dim()}"
            )

        if self.semantic_image is not None:
            assert self.semantic_image.dim() == 2, (
                f"Semantic class image must have 2 dimensions (h, w), got {self.semantic_image.dim()}"
            )
            assert self.semantic_image.dtype in (torch.int32, torch.int64), (
                f"Semantic class image must be int32 or int64, got {self.semantic_image.dtype}"
            )


@dataclass
class LidarFrame:
    """A simulated LiDAR "frame", containing a point cloud typically from a single revolution
    (revolving LiDAR) or a single shot (solid-state LiDAR)

    Fields:
      point_positions: float32 tensor of shape (n, 3), position of each scanned point
      point_intensities: optional float32 tensor of shape (n,), intensity of each point
    """

    point_positions: torch.Tensor
    point_intensities: Optional[torch.Tensor] = None

    def __post_init__(self):
        assert len(self.point_positions.shape) == 2, "Point positions tensor does not have 2 dimensions"
        assert self.point_positions.shape[1] == 3, "Point positions tensor does not have 3 dimensions"
        if self.point_intensities is not None:
            assert len(self.point_intensities.shape) == 1, "Intensity tensor does not have 1 dimension"
            assert self.point_intensities.shape[0] == self.point_positions.shape[0], (
                "Intensity tensor does not have the same number of points as point positions tensor"
            )


@dataclass
class LidarRenderingSettings:
    """Settings for lidar rendering."""

    lidar_raydrop_threshold: float = 0.0
    lidar_opacity_threshold: float = 0.8
    lidar_post_filter_threshold: float = 0.02
    enable_lidar_post_filter: bool = False
    # Server-config default neighbour source for the floating-point filter: "rangeview" (structured
    # 2D grid) or "angle" (sensor-agnostic; for model-free / unstructured lidar). A render request
    # may override it per call; if it does not, this default is used. Angle-path resolution params
    # below.
    lidar_post_filter_neighbor_source: str = "rangeview"
    lidar_post_filter_elevation_resolution_deg: Optional[float] = None
    lidar_post_filter_azimuth_max_gap_deg: Optional[float] = None

    def __post_init__(self):
        assert self.lidar_raydrop_threshold >= 0.0 and self.lidar_raydrop_threshold <= 1.0
        assert self.lidar_opacity_threshold >= 0.0 and self.lidar_opacity_threshold <= 1.0
        assert self.lidar_post_filter_threshold >= 0.0 and self.lidar_post_filter_threshold <= 1.0
        assert self.lidar_post_filter_neighbor_source in ("rangeview", "angle")


@dataclass
class PoseRange:
    start_pose_tquat_sensor_world: torch.Tensor  # 7D translation-quaternion Tensor of shape (7,)
    end_pose_tquat_sensor_world: torch.Tensor  # 7D translation-quaternion Tensor of shape (7,)
    start_timestamp_us: int  # Frame-start timestamp
    end_timestamp_us: int  # Frame-end timestamp

    def __post_init__(self):
        assert self.start_pose_tquat_sensor_world.shape == (7,)
        assert self.end_pose_tquat_sensor_world.shape == (7,)
        assert self.start_timestamp_us <= self.end_timestamp_us


@dataclass
class SensorTrajectory:
    """Frame acquisition start/end timestamps and sensor-to-world poses for a sequence of frames from a sensor"""

    # TODO: Turn into tquat (n_frames, 2, 7) for consistency with PoseRange and to restrict the degrees of freedom.
    poses_startend_sensor_world: torch.Tensor  # (n_frames, 2, 4, 4)
    timestamps_startend_us: torch.Tensor  # (n_frames, 2)

    def __post_init__(self):
        assert self.timestamps_startend_us.ndim == 2
        assert self.poses_startend_sensor_world.ndim == 4
        n_frames = self.timestamps_startend_us.shape[0]
        assert self.poses_startend_sensor_world.shape == (n_frames, 2, 4, 4)
        assert self.timestamps_startend_us.shape == (n_frames, 2)

    def __len__(self):
        return self.timestamps_startend_us.shape[0]


@dataclass
class RayBundle:
    """Rays sampled over a raster from a single camera frame."""

    rendering_data: RenderingData
    frame_meta: FrameMeta

    @classmethod
    @torch.autocast(device_type="cuda", enabled=False)
    def build(
        cls,
        camera_model_parameters: ConcreteCameraModelParametersUnion,
        camera_to_world: PoseRange,
        world_to_nurec: FrameConversion,
        unique_sensor_idx: Optional[int] = None,
        unique_frame_idx: Optional[int] = None,
        training_frame_idx: Optional[int] = None,
    ) -> RayBundle:
        """Generate a ray bundle in NuRec model space from a single camera view defined by its camera intrinsics and pose
        with optional rolling shutter support.

        Args:
          - camera_model: Lens model of the camera view.
          - camera_to_world: Frame start and end poses (camera-to-world transforms) and corresponding timestamps for
            the camera view with rolling shutter (simply feed the same start/end poses/timestamps for global shutter).
          - world_to_nurec: Transformation from world space to NuRec model space.
          - unique_sensor_idx: Optional unique index of a camera used during training if the intrinsic parameters of
              the view to generate rays is from such a camera (accessible via SceneInfo.get_camera()).
          - unique_frame_idx: Identity of the frame in the all-frame source dataset (training + held-out frames).
              Used by trajectory lookup, calibration, metrics, and other consumers that need frame identity.
              Pass None for free / novel views unrelated to any recorded frame.
          - training_frame_idx: Index into per-frame parameter tensors of training-only post-processings
              (PPISP, bilateral grids). Pass None for novel views or held-out frames that have no training-time
              parameter slot (stored as -1 in FrameMeta). When no holdout sampler is used, this equals
              unique_frame_idx.
        """
        with ScopedTimer("RayBundle.build"):
            device = torch.device("cuda")
            width, height = camera_model_parameters.resolution.tolist()

            # Generate a grid in 2D (float) pixel coordinates such that grid nodes fall in the center of the output pixels.
            with ScopedTimer("constructing_image_points"):
                image_points = generate_image_points(resolution=(width, height), device=device)

            # 4x4 transformation matrices from camera to world frame at the start- and end-time of the frame.
            with ScopedTimer("preparing_T_sensor_worlds"):
                poses_tquat_camera_world = torch.stack(
                    [camera_to_world.start_pose_tquat_sensor_world, camera_to_world.end_pose_tquat_sensor_world],
                    dim=0,
                )
                if poses_tquat_camera_world.device.type == "cpu":
                    poses_tquat_camera_world = poses_tquat_camera_world.pin_memory().to(
                        device=device, non_blocking=True
                    )
                else:
                    poses_tquat_camera_world = poses_tquat_camera_world.to(device=device)
                assert poses_tquat_camera_world.shape == (2, 7)

                # Apply extra transformation from the metric world to NuRec model space to both start and end poses.
                poses_tquat_camera_nurec = frame_transform_poses(
                    world_to_nurec, poses_tquat_camera_world, is_tquat=True
                )

            # Generate rays in the NuRec model space through the sampled grid points in the image.
            with ScopedTimer("image_points_to_world_rays_shutter_pose"):
                timestamps_us = torch.tensor(
                    [camera_to_world.start_timestamp_us, camera_to_world.end_timestamp_us], device=device
                )
                world_rays = image_points_to_world_rays_shutter_pose(
                    camera_model_parameters,
                    image_points,
                    T_sensor_worlds=poses_tquat_camera_nurec,
                    timestamps_us=timestamps_us,
                )

            with ScopedTimer("RenderingData::__init__"):
                # Rendering from any viewpoint should not require referring to a training sensor/view/trajectory.
                # Allow for missing camera / trajectory / frame index (i.e. training information) in general.
                # These are missing when we have a free (e.g. interactive) camera, completely independent of training views.
                # Workaround: selects a default training sensor, trajectory, frame, even for an independent/free viewpoint,
                # but this may undesirable side effects.
                # Keep GPU version for GPU operations, create CPU copy to avoid .item() calls
                timestamps_us_gpu = timestamps_us.unsqueeze(0)
                timestamps_us_cpu = torch.tensor(
                    [camera_to_world.start_timestamp_us, camera_to_world.end_timestamp_us],
                    device="cpu",
                    dtype=torch.int64,
                ).unsqueeze(0)
                rendering_data = RenderingData(
                    rays=world_rays.world_rays.reshape(1, height, width, 6),
                    sensor_model_parameters=[camera_model_parameters],
                    poses_tquat_startend=poses_tquat_camera_nurec.unsqueeze(0),  # [1, 2, 7]
                    timestamps_startend_us=timestamps_us_gpu,  # [1, 2] - on GPU
                    rays_timestamps_us=map_optional(world_rays.timestamps_us, lambda x: x.reshape(1, height, width, 1)),
                    timestamps_startend_us_cpu=timestamps_us_cpu,  # [1, 2] - on CPU
                )
                frame_meta = FrameMeta(
                    unique_sensor_idx=unique_sensor_idx if unique_sensor_idx is not None else -1,
                    unique_frame_idx=unique_frame_idx if unique_frame_idx is not None else -1,
                    training_frame_idx=training_frame_idx if training_frame_idx is not None else -1,
                )

            return cls(rendering_data=rendering_data, frame_meta=frame_meta)


def _crop_to_sweep(lidar_to_world: PoseRange, sweep_us: int) -> PoseRange:
    """The caller's frame, narrowed to the part of it the unit sweeps.

    A unit that idles between sweeps fires over only part of the frame, while the tiling's
    relative times run across the sweep. Both must mean the same window. Re-parameterising is
    exact: translation is linear and slerp holds a constant angular rate.
    """
    requested_us = lidar_to_world.end_timestamp_us - lidar_to_world.start_timestamp_us
    start = lidar_to_world.start_pose_tquat_sensor_world.reshape(7)
    end = lidar_to_world.end_pose_tquat_sensor_world.reshape(7)
    elapsed = torch.tensor([sweep_us / requested_us], dtype=start.dtype, device=start.device)
    sweep_end = torch.cat(
        [
            torch.lerp(start[:3], end[:3], elapsed),
            quat_slerp(start[3:].unsqueeze(0), end[3:].unsqueeze(0), elapsed)[0],
        ]
    )
    return PoseRange(
        start_pose_tquat_sensor_world=start,
        end_pose_tquat_sensor_world=sweep_end,
        start_timestamp_us=lidar_to_world.start_timestamp_us,
        end_timestamp_us=lidar_to_world.start_timestamp_us + sweep_us,
    )


@dataclass
class LidarRayBundle:
    """Rays sampled for a single lidar frame."""

    rendering_data: RenderingData
    frame_meta: FrameMeta

    # TODO: remove it in favor of rendering_data.poses_tquat_startend
    T_nurec_sensor_end: torch.Tensor
    model_elements: torch.Tensor

    @classmethod
    def build_model_free(
        cls,
        firing_pattern: LidarFiringPattern,
        lidar_to_world: PoseRange,
        world_to_nurec: FrameConversion,
        unique_sensor_idx: Optional[int] = None,
        unique_frame_idx: Optional[int] = None,
        training_frame_idx: Optional[int] = None,
    ) -> LidarRayBundle:
        """Build a bundle for a lidar with no sensor model, from an observed firing pattern.

        Where :meth:`build` synthesises an ``(n_rows, n_columns)`` grid from the model, the
        rays here are the pattern's measured directions as an ``(N, 1)`` list, each placed at
        its own firing instant so the sweep stays rolling-shutter correct.
        """
        # The render is CUDA-only, so the rays are built where they will be consumed.
        device = torch.device("cuda")
        firing_pattern.check_window(lidar_to_world.end_timestamp_us - lidar_to_world.start_timestamp_us)
        lidar_to_world = _crop_to_sweep(lidar_to_world, firing_pattern.span_us)
        frame_start_us = lidar_to_world.start_timestamp_us
        frame_end_us = lidar_to_world.end_timestamp_us

        tquat_sensor_nurec = frame_transform_poses(
            world_to_nurec,
            torch.stack(
                [
                    lidar_to_world.start_pose_tquat_sensor_world.reshape(7),
                    lidar_to_world.end_pose_tquat_sensor_world.reshape(7),
                ]
            ).to(device=device),
            is_tquat=True,
        )  # (2, 7)
        T_sensor_nurec = tquat_to_se3_matrix(tquat_sensor_nurec)  # (2, 4, 4)
        T_nurec_sensor_end = se3_matrix_inverse(T_sensor_nurec[1])

        ray_timestamps_us = torch.from_numpy(firing_pattern.ray_timestamps_us(frame_start_us)).to(device=device)
        n_rays = firing_pattern.n_rays

        # Fraction of the frame elapsed at each firing, which places that ray's origin along
        # the sensor's path through the frame.
        duration_us = max(frame_end_us - frame_start_us, 1)
        alpha = ((ray_timestamps_us - frame_start_us).to(torch.float32) / float(duration_us)).clamp(0.0, 1.0)

        directions_sensor = torch.from_numpy(firing_pattern.directions_sensor).to(device=device)
        # Use the shared rigid-pose interpolation: linear translation plus quaternion SLERP.
        # This preserves a valid rotation at every firing, including large rotations.
        interpolated_poses = interpolate_se3_poses(T_sensor_nurec[0], T_sensor_nurec[1], alpha)
        directions_nurec = torch.bmm(interpolated_poses[:, :3, :3], directions_sensor.unsqueeze(-1)).squeeze(-1)
        origins = interpolated_poses[:, :3, 3]
        rays = torch.cat([origins, directions_nurec], dim=1)  # (N, 6)

        tiling = build_unstructured_tiling(firing_pattern, frame_start_us, device)

        timestamps_startend_us = (frame_start_us, frame_end_us)
        rendering_data = RenderingData(
            rays=rays.reshape(1, n_rays, 1, 6),
            sensor_model_parameters=[None],  # model-free: routes to the unstructured render
            poses_tquat_startend=tquat_sensor_nurec.unsqueeze(0),  # (1, 2, 7)
            timestamps_startend_us=torch.tensor(timestamps_startend_us, dtype=torch.int64, device=device).view(1, 2),
            rays_timestamps_us=ray_timestamps_us.reshape(1, n_rays, 1, 1),
            timestamps_startend_us_cpu=torch.tensor(timestamps_startend_us, dtype=torch.int64).view(1, 2),
            unstructured_lidar_tiling=tiling,
        )
        frame_meta = FrameMeta(
            unique_sensor_idx=unique_sensor_idx if unique_sensor_idx is not None else -1,
            unique_frame_idx=unique_frame_idx if unique_frame_idx is not None else -1,
            training_frame_idx=training_frame_idx if training_frame_idx is not None else -1,
        )
        return cls(
            rendering_data=rendering_data,
            frame_meta=frame_meta,
            T_nurec_sensor_end=T_nurec_sensor_end,
            # Model elements index a sensor grid, which a model-free frame has none of. Their
            # only consumer is the range-view floating-point filter, which refuses these
            # frames for that reason; the angle source needs no grid and reads none of this.
            model_elements=torch.zeros((n_rays, 2), dtype=torch.int32, device=device),
        )

    @classmethod
    def build(
        cls,
        lidar_model: LidarModelBundle,
        lidar_to_world: PoseRange,
        world_to_nurec: FrameConversion,
        unique_sensor_idx: Optional[int] = None,
        unique_frame_idx: Optional[int] = None,
        training_frame_idx: Optional[int] = None,
    ) -> LidarRayBundle:
        device = torch.device("cuda")
        start_pose_tquat_sensor_world = lidar_to_world.start_pose_tquat_sensor_world
        if start_pose_tquat_sensor_world.device.type == "cpu":
            start_pose_tquat_sensor_world = start_pose_tquat_sensor_world.pin_memory().to(
                device=device, non_blocking=True
            )
        else:
            start_pose_tquat_sensor_world = start_pose_tquat_sensor_world.to(device=device)
        end_pose_tquat_sensor_world = lidar_to_world.end_pose_tquat_sensor_world
        if end_pose_tquat_sensor_world.device.type == "cpu":
            end_pose_tquat_sensor_world = end_pose_tquat_sensor_world.pin_memory().to(device=device, non_blocking=True)
        else:
            end_pose_tquat_sensor_world = end_pose_tquat_sensor_world.to(device=device)
        tquat_sensor_nurec_start = frame_transform_poses(
            world_to_nurec,
            start_pose_tquat_sensor_world.reshape(-1, 7),
            is_tquat=True,
        )
        tquat_sensor_nurec_end = frame_transform_poses(
            world_to_nurec,
            end_pose_tquat_sensor_world.reshape(-1, 7),
            is_tquat=True,
        )
        T_nurec_sensor_end = se3_matrix_inverse(tquat_to_se3_matrix(tquat_sensor_nurec_end))
        virtual_model_elements = lidar_model.elements

        height = cast(int, lidar_model.lidar_model.n_rows)
        width = cast(int, lidar_model.lidar_model.n_columns)

        nurec_rays = elements_to_world_rays_shutter_pose(
            vren_lidar_model_parameters=lidar_model.vren_lidar.parameters,
            element=torch.tensor(virtual_model_elements, dtype=torch.int32),
            T_sensor_worlds=torch.cat([tquat_sensor_nurec_start, tquat_sensor_nurec_end], dim=0),
            timestamps_us=torch.tensor(
                [lidar_to_world.start_timestamp_us, lidar_to_world.end_timestamp_us], dtype=torch.int64
            ),
        )

        rays = nurec_rays.world_rays
        timestamps_us = nurec_rays.timestamps_us

        # TODO: Make indices optional.
        if unique_sensor_idx is None:
            unique_sensor_idx = -1

        if unique_frame_idx is None:
            unique_frame_idx = -1

        if training_frame_idx is None:
            training_frame_idx = -1

        frame_timestamps_startend_us = (lidar_to_world.start_timestamp_us, lidar_to_world.end_timestamp_us)

        # Keep GPU version for GPU operations, create CPU copy to avoid .item() calls
        device = rays.device
        timestamps_tensor_gpu = torch.tensor(frame_timestamps_startend_us, dtype=torch.int64, device=device).view(1, 2)
        timestamps_tensor_cpu = torch.tensor(frame_timestamps_startend_us, dtype=torch.int64, device="cpu").view(1, 2)
        rendering_data = RenderingData(
            rays=rays.reshape(1, height, width, 6),
            sensor_model_parameters=[lidar_model.lidar_parameters],
            poses_tquat_startend=torch.cat([tquat_sensor_nurec_start, tquat_sensor_nurec_end], dim=0).unsqueeze(
                0
            ),  # (1, 2, 7)
            timestamps_startend_us=timestamps_tensor_gpu,  # on GPU
            rays_timestamps_us=map_optional(timestamps_us, lambda x: x.reshape(1, height, width, 1)),
            timestamps_startend_us_cpu=timestamps_tensor_cpu,  # on CPU
        )

        frame_meta = FrameMeta(
            unique_sensor_idx=unique_sensor_idx,
            unique_frame_idx=unique_frame_idx,
            training_frame_idx=training_frame_idx,
        )

        return cls(
            rendering_data=rendering_data,
            frame_meta=frame_meta,
            T_nurec_sensor_end=T_nurec_sensor_end,
            model_elements=torch.from_numpy(lidar_model.elements).to(T_nurec_sensor_end.device),
        )


@dataclass
class RenderableModel:
    """High-level API unified across model types for rendering camera frames off a model."""

    _model: GaussiansComposite | BaseInstantNuRecPrimitive
    _autocast: Optional[torch.autocast] = None

    # The subset of methods that require it will raise if not provided.
    _world_to_nurec: Optional[FrameConversion] = None

    _lidar_rendering_settings: LidarRenderingSettings = field(default_factory=LidarRenderingSettings)
    _track_split_resolver: TrackSplitResolver = field(default_factory=TrackSplitResolver)
    _compact_lidar_extra_signal_tensors_missing: Tuple[str, ...] = ()
    _compact_camera_extra_signals_missing: bool = False

    # Default camera inference emits only RGB; semantic and distance renders
    # opt back in via config_overrides (later entries win the merge). Lidar
    # gates keep their defaults because lidar inference reads intensity,
    # raydrop, and distance. See nurec/render/README.md.
    _INFERENCE_DEFAULT_OVERRIDES: ClassVar[Tuple[str, ...]] = (
        "model.renderer.outputs.camera.enable_extended_features=False",
        "model.renderer.outputs.camera.enable_sensor_features=False",
        "model.renderer.outputs.camera.enable_distance=False",
    )
    _RENDERER_NAME_KEY: ClassVar[str] = "model.renderer.name"
    _LIDAR_OUTPUT_SIGNAL_GATES: ClassVar[Tuple[Tuple[str, str], ...]] = (
        ("model.renderer.outputs.lidar.enable_extended_features", "extra_signal"),
        ("model.renderer.outputs.lidar.enable_sensor_features", "lidar_extra_signal"),
    )

    # Track-bearing layers that the per-track timestamp estimation runs on.
    _TRACK_LAYER_NAMES: ClassVar[Tuple[str, ...]] = ("dynamic_rigids", "dynamic_deformables")

    @classmethod
    def _ordered_config_overrides(
        cls, stride_overrides: Tuple[str, ...], caller_overrides: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        """Assemble the override dotlist so precedence is explicit.

        Precedence, weakest first: inference defaults < ray-stride < caller. `OmegaConf.from_dotlist`
        applies entries left-to-right and the last assignment to a key wins, so ordering *is* the
        precedence rule. The ray-stride has to sit ahead of the caller for the same reason the
        inference defaults do: an explicit
        `model.layers.*.tracks_timestamp_estimation_ray_stride=N` passed by the caller is a
        deliberate request and must outrank the CLI-level default, which is applied unconditionally.
        """
        return cls._INFERENCE_DEFAULT_OVERRIDES + tuple(stride_overrides) + tuple(caller_overrides)

    @classmethod
    def tracks_timestamp_ray_stride_overrides(cls, stride: int, present_layers: list[str]) -> Tuple[str, ...]:
        """Inference-time ray-stride for per-track rolling-shutter timestamp estimation.

        `stride=1` (default) keeps full estimation and adds nothing. `stride>1`
        subsamples the `estimate_track_timestamps` ray grid ~N-fold on the dynamic track
        layers (a 2D grid decimation ~sqrt(N) per sensor-grid axis, applied in
        RigidGaussianModel), cutting the kernel's per-track ray work ~N-fold. The per-track
        timestamp is a smooth aggregate over its rays, so uniform subsampling at the
        validated default (stride 16) is quality-neutral on both camera and lidar; larger
        strides stay camera-neutral but measurably degrade lidar.

        Only emits overrides for track layers actually present in `present_layers`, so a
        static or single-dynamic-layer artifact never gets a phantom partial layer (a
        `{tracks_timestamp_estimation_ray_stride: N}` entry with no discriminator) injected
        by the dotlist merge. Applied through `load_from_artifact`, so this is an
        inference/serve-time knob only -- the training path never sees it.
        """
        if stride < 1:
            raise ValueError(f"tracks_timestamp_ray_stride must be >= 1, got {stride}")
        if stride == 1:
            return ()
        present = set(present_layers)
        return tuple(
            f"model.layers.{layer}.tracks_timestamp_estimation_ray_stride={stride}"
            for layer in cls._TRACK_LAYER_NAMES
            if layer in present
        )

    @classmethod
    def load_from_artifact(
        cls,
        artifact: Artifact,
        config_overrides: Tuple[str, ...] = (),
        tracks_timestamp_ray_stride: int = 1,
        *,
        before_model_load: Callable[[], None] | None = None,
    ) -> Self:
        """Load a model from an artifact with the requested configuration overrides.

        If supplied, ``before_model_load`` runs after configuration upgrades, overrides,
        and CUDA environment setup, immediately before model loading.
        """
        orig_untyped_config = OmegaConf.create(artifact.parsed_config)

        # An explicit renderer override (`--renderer gsplat`, or the gRPC GSplat backend) picks the
        # renderer the artifact is actually rendered with, so it has to be visible to the upgraders:
        # they refuse to migrate configs whose trained renderer has no automatic migration path, and
        # naming a supported renderer is the documented manual recovery for exactly that case.
        untyped_config = upgrade_config(cls._with_renderer_override(orig_untyped_config, config_overrides))

        # Ray-stride is applied here (not by the caller) so it can be scoped to the track
        # layers actually present in this artifact -- emitting it for an absent layer would
        # inject a phantom partial layer and fail typed-layer validation.
        present_layers = list((OmegaConf.select(untyped_config, "model.layers", default={}) or {}).keys())
        stride_overrides = cls.tracks_timestamp_ray_stride_overrides(tracks_timestamp_ray_stride, present_layers)
        if tracks_timestamp_ray_stride > 1 and not stride_overrides:
            log.warning(
                "tracks_timestamp_ray_stride=%d requested but artifact has no %s track layer; "
                "ray-stride has no effect.",
                tracks_timestamp_ray_stride,
                "/".join(cls._TRACK_LAYER_NAMES),
            )

        config_overrides = cls._ordered_config_overrides(stride_overrides, config_overrides)

        # Apply Hydra-style config overrides (e.g., model.renderer.name=gsplat)
        # NOTE: We're forced to apply the overrides here because load_from_artifact is actually doing two things:
        # loading the artifact from disk, and creating the RenderableModel instance. We can't apply the override
        # after the instance is created, not before the artifact is loaded. A better design would split these two steps,
        # allowing the override to be applied by the caller in between these operations
        if config_overrides:
            log.info(f"Applying {len(config_overrides)} Hydra-style override(s)")
            for override in config_overrides:
                log.info(f"  Override: {override}")

            # OmegaConf.from_dotlist handles format validation, parsing, type conversion, and nested key creation
            untyped_config = cast(
                DictConfig,
                OmegaConf.unsafe_merge(
                    untyped_config, cast(DictConfig, OmegaConf.from_dotlist(list(config_overrides)))
                ),
            )

        lidar_raydrop_threshold = OmegaConf.select(untyped_config, "system.test.lidar.raydrop_threshold", default=0.0)
        lidar_opacity_threshold = OmegaConf.select(untyped_config, "system.test.lidar.opacity_threshold", default=0.8)
        lidar_post_filter_threshold = OmegaConf.select(
            untyped_config, "system.test.lidar.floating_point_filter.threshold", default=0.02
        )
        enable_lidar_post_filter = OmegaConf.select(
            untyped_config, "system.test.lidar.floating_point_filter.enabled", default=False
        )
        lidar_post_filter_neighbor_source = OmegaConf.select(
            untyped_config, "system.test.lidar.floating_point_filter.neighbor_source", default="rangeview"
        )
        lidar_post_filter_elevation_resolution_deg = OmegaConf.select(
            untyped_config, "system.test.lidar.floating_point_filter.elevation_resolution_deg", default=None
        )
        lidar_post_filter_azimuth_max_gap_deg = OmegaConf.select(
            untyped_config, "system.test.lidar.floating_point_filter.azimuth_max_gap_deg", default=None
        )

        OmegaConf.resolve(untyped_config)
        compact_preserved_extra_signal_tensors = (
            compact_usdz.compact_preserved_extra_signal_tensors(artifact.source) if artifact.is_compact else frozenset()
        )
        compact_lidar_extra_signal_tensors_missing = cls._compact_lidar_output_signal_tensors_missing(
            artifact, untyped_config, compact_preserved_extra_signal_tensors
        )
        compact_camera_extra_signals_missing = cls._compact_signal_tensor_missing(
            artifact, compact_preserved_extra_signal_tensors, "camera_extra_signal"
        )

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # If set, configure torch's default allocation block size - appending to any existing key:value configs.
        if (
            "max_split_size_mb" in untyped_config.system
            and (split_size := untyped_config.system.max_split_size_mb) is not None
        ):
            new_entry = f"max_split_size_mb:{split_size}"
            conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_entry if conf is None else ",".join([conf, new_entry])

        if before_model_load is not None:
            before_model_load()

        model: GaussiansComposite | BaseInstantNuRecPrimitive = cls._load_model_and_upgrade(
            untyped_config, artifact, orig_untyped_config
        )

        world_to_nurec = RigTrajectories.from_dict(artifact.rig_trajectories).world_to_nurec
        track_split_resolver = TrackSplitResolver(artifact.track_split_metadata)

        return cls(
            model,
            torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16),
            world_to_nurec,
            LidarRenderingSettings(
                lidar_raydrop_threshold,
                lidar_opacity_threshold,
                lidar_post_filter_threshold,
                enable_lidar_post_filter,
                lidar_post_filter_neighbor_source,
                lidar_post_filter_elevation_resolution_deg,
                lidar_post_filter_azimuth_max_gap_deg,
            ),
            track_split_resolver,
            compact_lidar_extra_signal_tensors_missing,
            compact_camera_extra_signals_missing,
        )

    @classmethod
    def _with_renderer_override(cls, untyped_config: DictConfig, config_overrides: Tuple[str, ...]) -> DictConfig:
        """Return `untyped_config` with any `model.renderer.name` override from `config_overrides` applied.

        The input config is left untouched; it stays the record of what the artifact was trained
        with, which the state-dict upgraders key off. Only an already-present renderer name is
        rewritten, so this never fabricates a renderer section for a config that has none.
        """
        prefix = f"{cls._RENDERER_NAME_KEY}="
        # Later dotlist entries win the merge below, so the last override is the effective one.
        renderer_name = next(
            (override[len(prefix) :] for override in reversed(config_overrides) if override.startswith(prefix)),
            None,
        )
        if renderer_name is None:
            return untyped_config
        if OmegaConf.select(untyped_config, cls._RENDERER_NAME_KEY, default=None) is None:
            return untyped_config
        overridden = copy.deepcopy(untyped_config)
        OmegaConf.update(overridden, cls._RENDERER_NAME_KEY, renderer_name)
        return overridden

    @classmethod
    def _requested_lidar_output_signal_tensors(cls, untyped_config: DictConfig) -> Tuple[str, ...]:
        """Return extra-signal tensors requested by LiDAR output gates."""
        return tuple(
            tensor
            for gate, tensor in cls._LIDAR_OUTPUT_SIGNAL_GATES
            if bool(OmegaConf.select(untyped_config, gate, default=True))
        )

    @classmethod
    def _compact_lidar_output_signal_tensors_missing(
        cls,
        artifact: Artifact,
        untyped_config: DictConfig,
        preserved_signal_tensors: frozenset[str],
    ) -> Tuple[str, ...]:
        """Return requested LiDAR extra-signal tensors missing from a compact artifact."""
        if not artifact.is_compact:
            return ()
        return tuple(
            tensor
            for tensor in cls._requested_lidar_output_signal_tensors(untyped_config)
            if cls._compact_signal_tensor_missing(artifact, preserved_signal_tensors, tensor)
        )

    @staticmethod
    def _compact_signal_tensor_missing(
        artifact: Artifact, preserved_signal_tensors: frozenset[str], signal_tensor: str
    ) -> bool:
        """Return true when a compact artifact omitted one extra-signal tensor."""
        return bool(artifact.is_compact and signal_tensor not in preserved_signal_tensors)

    @classmethod
    def _load_model_and_upgrade(
        cls, untyped_config: DictConfig, artifact: Artifact, orig_untyped_config: DictConfig
    ) -> GaussiansComposite | BaseInstantNuRecPrimitive:
        match untyped_config.model.name:
            case "gaussians_primitive" | "gaussians-composite":
                return cls._load_model_gaussians_composite_and_upgrade(untyped_config, artifact, orig_untyped_config)
            case "celsius" | "kelvin":
                return cls._load_model_instant_nurec_and_upgrade(untyped_config, artifact, orig_untyped_config)
            case _:
                raise TypeError(f"Unsupported model name '{untyped_config.model.name}'")

    @classmethod
    def _load_model_gaussians_composite_and_upgrade(
        cls, untyped_config: DictConfig, artifact: Artifact, orig_untyped_config: DictConfig
    ) -> GaussiansComposite:
        model_config = ModelConfig.model_validate(untyped_config.model)

        # Log which renderer will be used
        renderer_name = model_config.renderer.name
        log.info(f"Using renderer: {renderer_name}")

        datasource_summary = DataSourceSummary.from_dict(
            artifact.datasource_summary,
            infer_missing=True,  # For better backward-compatibility.
            track_split_metadata=artifact.track_split_metadata,
        )
        trainer_config = TrainerConfig.model_validate(untyped_config.trainer)
        model = GaussiansComposite(
            model_config, trainer_config, datasource=datasource_summary, init_from_datasource=False
        )

        checkpoint = cls._load_gaussian_checkpoint(artifact)
        model_state_dict = {
            k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
        }
        # Upgrade the state_dict to the current version of the software, should the artifact be from an earlier version.
        upgraded_model_state_dict = upgrade_model(
            model_state_dict, orig_untyped_config, datasource_summary=datasource_summary
        )

        model.load_state_dict(upgraded_model_state_dict, assign=not checkpoint.get("load_in_place", True))
        model.to("cuda")
        model.eval()

        return model

    @staticmethod
    def _load_gaussian_checkpoint(artifact: Artifact) -> dict[str, Any]:
        """Load a Gaussian checkpoint from normal or compact USDZ."""
        if artifact.is_compact:
            if not torch.cuda.is_available():
                raise RuntimeError("compact USDZ rendering requires CUDA")
            # The Gaussian render path materializes models on CUDA below; unpack
            # compact tensors there directly instead of staging dense CPU tensors.
            return compact_usdz.load_compact_checkpoint(artifact.source, device="cuda")
        return torch.load(artifact.checkpoint, weights_only=False)

    @classmethod
    def _load_model_instant_nurec_and_upgrade(
        cls, untyped_config: DictConfig, artifact: Artifact, orig_untyped_config: DictConfig
    ) -> BaseInstantNuRecPrimitive:
        model_name = untyped_config.model.name
        if artifact.is_compact:
            raise ValueError("compact USDZ artifacts support Gaussian models only")
        checkpoint: Checkpoint = torch.load(artifact.checkpoint, weights_only=False)
        primitive: CelsiusInstantNuRecPrimitive | KelvinInstantNuRecPrimitive
        if model_name == "celsius":
            primitive = CelsiusInstantNuRecPrimitive.load_from_checkpoint(checkpoint)
        elif model_name == "kelvin":
            primitive = KelvinInstantNuRecPrimitive.load_from_checkpoint(checkpoint)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        renderer_config: RendererConfigType = TypeAdapter(RendererConfigType).validate_python(
            untyped_config.model.renderer
        )
        gaussians_renderer = BaseGaussianRenderer.factory(
            renderer_config.name,
            renderer_config,
            primitive,  # type: ignore
        )
        primitive.gaussians_renderer = gaussians_renderer
        return primitive

    @classmethod
    def load_from_instant_nurec_primitive(
        cls, instant_nurec_primitive: BaseInstantNuRecPrimitive, world_to_nurec: FrameConversion
    ) -> Self:
        return cls(instant_nurec_primitive, None, world_to_nurec)

    @property
    def device(self) -> torch.device:
        if isinstance(self._model, BaseInstantNuRecPrimitive):
            return self._model.device()
        return next(iter(self._model.parameters())).device

    @property
    def autocast(self):
        return self._autocast if self._autocast is not None else nullcontext()

    def get_all_gaussian_positions(self) -> torch.Tensor | None:
        """
        Get all Gaussian center positions from the underlying model.

        Returns:
            Optional[torch.Tensor]: Positions of all Gaussians in NuRec frame, shape (N, 3),
                                   or None if not available
        """
        return self._model.get_all_gaussian_positions()

    def get_camera_trajectories(self, calibrated: bool = False) -> Dict[str, SensorTrajectory]:
        """Get all camera training view start/end poses and timestamps from the calib module and return them per camera.

        Args:
            calibrated: returns the calibrated poses if available when True, or the original input poses otherwise.

        Returns:
            A dictionary that maps unique camera ids (in the form "camera_id@sequence_id") to the training trajectory
            of the camera. Each trajectory consists of a frame start/end pose, as well as start/end timestamps
            for each training view of the camera, in increasing order of frame-end timestamps.
            Each pose is a 4x4 SE(3) matrix transforming points from sensor space to world space.
        """

        if not isinstance(self._model, GaussiansComposite):
            raise TypeError("Only GaussiansComposite is supported for now")
        # FreePoseCalib is always instantiated internally currently, even if calib is skipped or disabled.
        if not isinstance(self._model.calib, FreePoseCalib):
            raise TypeError("Only FreePoseCalib is supported for now")
        if self._model.calib.camera_view_geometry is None:
            raise ValueError("Camera view geometry is not enabled")
        if self._world_to_nurec is None:
            raise ValueError("World-to-NuRec transformation is required but not initialized")

        with torch.inference_mode():
            # Get the start/end timestamps and poses for all training frames/views vectorized over all cameras.
            timestamps_startend_us_allviews = self._model.calib.camera_view_geometry.get_timestamps()
            assert timestamps_startend_us_allviews.ndim == 2
            assert timestamps_startend_us_allviews.shape[1] == 2
            T_sensor_world_startend_allviews = self._model.calib.camera_view_geometry.get_poses(
                skip_calib=not calibrated
            )
            assert T_sensor_world_startend_allviews.ndim == 4
            assert T_sensor_world_startend_allviews.shape[1:] == (2, 4, 4)
            assert T_sensor_world_startend_allviews.shape[0] == timestamps_startend_us_allviews.shape[0]

        # Poses from the calib module are camera-to-nurec space transforms, convert them to camera-to-world transforms.
        # The shape is (N, 2, 4, 4) but the transformation is only applicable to (M, 4, 4) tensors.
        num_frames = T_sensor_world_startend_allviews.shape[0]
        T_reshaped = T_sensor_world_startend_allviews.reshape(2 * num_frames, 4, 4)
        T_transformed = frame_transform_poses(self._world_to_nurec.inverse(), T_reshaped, is_tquat=False)
        T_sensor_world_startend_allviews = T_transformed.reshape(num_frames, 2, 4, 4)

        # Extract the trajectory of each camera from the vectorized poses and timestamps.
        camera_trajectories: Dict[str, SensorTrajectory] = {}
        frame_ranges = self._model.calib.camera_view_geometry.get_frame_ranges_per_sensor()
        for unique_sensor_id, frame_range in frame_ranges.items():
            camera_trajectories[unique_sensor_id] = SensorTrajectory(
                poses_startend_sensor_world=T_sensor_world_startend_allviews[frame_range],
                timestamps_startend_us=timestamps_startend_us_allviews[frame_range],
            )

        return camera_trajectories

    def model_class_name(self) -> str:
        """Return the underlying model's class name for diagnostics and error messages."""
        return type(self._model).__name__

    def supports_edit_actors(self) -> bool:
        return (
            isinstance(self._model, CompositeModel)
            and isinstance(self._model.cuboid_tracks, CuboidTracks)
            and self._world_to_nurec is not None
        )

    def _to_parent_facing_cuboid_tracks(self, cuboid_tracks: CuboidTracks) -> CuboidTracks:
        if not self._track_split_resolver.has_splits:
            return cuboid_tracks
        return CuboidTracks.Ops.merge_split_children(cuboid_tracks, self._track_split_resolver.metadata)

    def _resolve_actors_snapshot_to_internal_tracks(
        self,
        actors_snapshot: ActorsSnapshot,
        frame_start_us: int,
        frame_end_us: int,
    ) -> ActorsSnapshot:
        if not self._track_split_resolver.has_splits or actors_snapshot.num_actors() == 0:
            return actors_snapshot

        actor_ids: List[str] = []
        actor_poses: List[torch.Tensor] = []
        cuboid_dims: List[torch.Tensor] = []
        for actor_idx, actor_id in enumerate(actors_snapshot.actor_ids):
            for child_id in self._track_split_resolver.child_ids_overlapping_interval(
                actor_id,
                frame_start_us,
                frame_end_us,
            ):
                actor_ids.append(child_id)
                actor_poses.append(actors_snapshot.actor_poses[actor_idx])
                if actors_snapshot.cuboid_dims is not None:
                    cuboid_dims.append(actors_snapshot.cuboid_dims[actor_idx])

        if not actor_ids:
            return ActorsSnapshot.empty()

        return ActorsSnapshot(
            actor_ids=actor_ids,
            actor_poses=torch.stack(actor_poses, dim=0),
            cuboid_dims=torch.stack(cuboid_dims, dim=0) if cuboid_dims else None,
        )

    def get_actor_tracks(self) -> ActorTracks:
        """
        Returns the actor tracks in the world frame after applying the learned track calibration (delta poses)
        if the tracks_calib model was enabled during training.
        """
        if self._world_to_nurec is None:
            raise ValueError("World-to-NuRec transformation is required but not initialized")

        if not self.supports_edit_actors():
            raise ValueError(f"Actor tracks not supported by models of type {self._model.__class__.__name__}")

        assert isinstance(self._model, CompositeModel)
        cuboid_tracks = self._model.get_updated_cuboid_tracks()
        cuboid_tracks = CuboidTracks.Ops.transform_with_frame_conversion(
            cuboid_tracks, self._world_to_nurec.inverse(), None
        )
        cuboid_tracks.tracks_data.tracks_flags |= TrackFlags.CONTROLLABLE
        cuboid_tracks = self._to_parent_facing_cuboid_tracks(cuboid_tracks)
        return ActorTracks._from_cuboid_tracks(cuboid_tracks)

    def save_training_parameters(self) -> None:
        """
        Save the original pre-edit state of self._model that was loaded from the usdz.
        This includes both Gaussian parameters and structural track state required to undo grpc asset edits.
        """
        if isinstance(self._model, GaussiansComposite):
            self._model.save_training_parameters()
        else:
            raise NotImplementedError(f"Save training parameters not supported for model type {type(self._model)}")

    def restore_training_parameters(self) -> None:
        """
        Restore the original pre-edit state of self._model that was loaded from the usdz.
        This undoes both Gaussian parameter edits and structural grpc asset edits.
        """
        if isinstance(self._model, GaussiansComposite):
            self._model.restore_training_parameters()
        else:
            raise NotImplementedError(f"Restore training parameters not supported for model type {type(self._model)}")

    def replace_asset(self, track_id: str, asset: Asset, dims_offset: torch.Tensor) -> bool:
        """
        Replace the gaussians of a target track_id with gaussians from an Asset
        The Asset is the same as PlyGaussianLoader(nurec/models/gaussians/utils.py) class

        Args:
            track_id: The track ID to override
            asset: Asset to use (preferred)
        """
        # Hard coded transform, we are replacing only the gaussians of an asset, not applying modifications to its pose
        transform = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cuda"
        )

        if isinstance(self._model, GaussiansComposite):
            try:
                override_results = [
                    self._model.track_ply_override(
                        child_track_id,
                        asset=asset,
                        transform=transform,
                        dims_offset=dims_offset,
                    )
                    for child_track_id in self._track_split_resolver.child_ids_for_parent(track_id)
                ]
                return all(override_results)
            except Exception:
                log.exception(f"Failed to apply track PLY override for {track_id}")
                raise
        else:
            raise NotImplementedError(f"Track PLY override not supported for model type {type(self._model)}")

    def insert_asset(self, asset: Asset, cuboid_tracks: CuboidTracks, dims_offset: torch.Tensor) -> bool:
        """
        Insert track gaussians with associated CuboidTracks.

        Args:
            asset: Asset containing gaussian parameters
            cuboid_tracks: CuboidTracks data for the track (should be in NuRec coordinates)
        """
        if not self.supports_edit_actors():
            raise RuntimeError("Model does not support actor updates")

        if not isinstance(self._model, GaussiansComposite):
            raise RuntimeError("Track insertion only supported for GaussiansComposite models")

        assert cuboid_tracks.n_tracks == 1, "Expected a single cuboid track for inserting a single asset"
        rgb_post_processing = None
        # [TODO] (shsolanki): remove this env toggle and always apply the CRF inverse by default.
        if _env_flag_enabled("NUREC_PPISP_ASSET_CRF_INVERSE"):
            # The (num_cameras, 3, 4) shape distinguishes the new ppisp CRF
            # ([toe, shoulder, gamma, center] per channel) from the legacy (*, 3, 7) CRF.
            for post_processing in getattr(self._model, "post_processings", []):
                crf_params = getattr(getattr(post_processing, "ppisp", None), "crf_params", None)
                if torch.is_tensor(crf_params) and crf_params.dim() == 3 and crf_params.shape[1:] == (3, 4):
                    rgb_post_processing = make_ppisp_crf_inverse_rgb_post_processing(crf_params, camera_index=0)
                    break

        # Delegate to underlying model - cuboid_tracks should already be in NuRec frame.
        # The inserted asset is a single set of Gaussians shared across all render cameras, so one
        # reference CRF must be chosen; camera 0 matches ppisp_export reference_selection's default.
        return self._model.insert_asset(
            asset,
            cuboid_tracks,
            dims_offset,
            rgb_post_processing=rgb_post_processing,
        )

    @ScopedTimer()
    def _edit_actors(
        self,
        actors_snapshot: ActorsSnapshot,
        frame_start_us: int,
        frame_end_us: int,
    ) -> Optional[CuboidTracks]:
        """
        Updates the cuboid poses of the controllable actors in the scene's model.

        Effectively this works by `freezing` the dynamic objects at the requested poses for all times.

        Args:
            actors_snapshot: start and end poses of all actors present at the start and end time of the frame capture.
            frame_start_us: start timestamp (in microseconds) of the frame captured.
            frame_end_us: end timestamp (in microseconds) of the frame captured.

        Returns:
            edited_cuboid_tracks, or None if there are no actors to edit.
        """
        if not self.supports_edit_actors():
            raise NotImplementedError("Actor track updates not supported by model")

        if actors_snapshot.num_actors() == 0:
            return None

        actors_snapshot = self._resolve_actors_snapshot_to_internal_tracks(
            actors_snapshot,
            frame_start_us,
            frame_end_us,
        )
        if actors_snapshot.num_actors() == 0:
            return None

        # Transform the poses to the NuRec frame
        # Supports_edit_actors() called above must handle this case.
        if self._world_to_nurec is None:
            raise ValueError("World-to-NuRec transformation is required but not initialized")

        track_poses_nurec = frame_transform_poses(
            self._world_to_nurec, actors_snapshot.actor_poses.to(self.device).reshape(-1, 7), is_tquat=True
        ).reshape(-1, 2, 7)  # Reshape needed in case track_poses is empty

        assert isinstance(self._model, CompositeModel)  # Type must have a cuboid_tracks member used below.

        cuboid_tracks = self._model.cuboid_tracks
        cuboid_tracks = CuboidTracks.Ops.subset_from_tracks_id(
            cuboid_tracks=cuboid_tracks, tracks_id=actors_snapshot.actor_ids
        )
        cuboid_tracks = CuboidTracks.Ops.freeze(
            cuboid_tracks=cuboid_tracks,
            min_timestamps_us=torch.full((cuboid_tracks.n_tracks,), frame_start_us, dtype=torch.int64).cuda(),
            max_timestamps_us=torch.full((cuboid_tracks.n_tracks,), frame_end_us, dtype=torch.int64).cuda(),
            tracks_poses_start=lietorch.SE3(data=track_poses_nurec[:, 0]),
            tracks_poses_end=lietorch.SE3(data=track_poses_nurec[:, 1]),
        )
        return cuboid_tracks

    @ScopedTimer()
    def _render_volume_from_ray_bundle(
        self,
        ray_bundle: RayBundle,
        actors_snapshot: Optional[ActorsSnapshot] = None,
        frame_start_us: int = 0,
        frame_end_us: int = 0,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> GaussiansRenderReturn:
        """Render a model given a ray bundle generated from a single camera view and return a GaussiansRenderReturn.

        Use RayBundle.build() to generate rays from a camera view.

        Args:
            ray_bundle: rays in NuRec model space with metadata frame indices
            actors_snapshot: Optional ActorsSnapshot containing the actors ids and poses
            frame_start_us: start timestamp (in microseconds) of the frame captured
            frame_end_us: end timestamp (in microseconds) of the frame captured
            edited_cuboid_tracks: Optional edited tracks to use for stateless actor edits

        Returns:
            GaussiansRenderReturn containing rendered outputs
        """

        # This function is not supposed to be used directly.
        # TODO: Merge this function into render_camera_frame_from_ray_bundle() once all use-cases are eliminated.

        # NOTE: _edit_actors creates tensors through its use of frame_transform_poses.
        # If it is placed below in the autocast section, the input tensors will end up
        # being converted to the autocasted dtype (e.g. float16) and we have asserts in
        # interpolate_tracks_poses that expect tracks_interpolation_data.tracks_poses.dtype
        # (originating from the cuboid_tracks dtype in _edit_actors) to be float32. So
        # it is crucial to place it here before the autocast section.
        if actors_snapshot is not None:
            actor_edited_cuboid_tracks = self._edit_actors(
                actors_snapshot=actors_snapshot, frame_start_us=frame_start_us, frame_end_us=frame_end_us
            )
            if edited_cuboid_tracks is None:
                edited_cuboid_tracks = actor_edited_cuboid_tracks

        with self.autocast, torch.no_grad(), ScopedTimer("RenderableModel._render_volume_from_ray_bundle"):
            # Copy all tensors with the ray bundle to GPU
            ray_bundle = cast(RayBundle, tree_map(ray_bundle, to_float_device(self.device)))

            # Call the model. Different models behave slightly differently.
            # TODO: replace model type branching with an abstract base class like CameraRenderable
            model_return: GaussiansCompositeReturn
            model = self._model

            if isinstance(model, GaussiansComposite):
                # TODO: directly passing in RenderingBatch instead of having to create a DataAndRenderingBatch
                batch = DataAndRenderingBatch(
                    data=DataBatch(
                        camera=DataBatch.Camera(
                            meta=[ray_bundle.frame_meta],
                            labels=CameraFrameLabels(),  # empty labels
                        ),
                    ),
                    rendering=RenderingBatch(camera=ray_bundle.rendering_data),
                )
                model_return = model(
                    batch=batch,
                    enable_calib=False,
                    enable_pp=True,
                    edited_cuboid_tracks=edited_cuboid_tracks,
                )
            elif isinstance(model, BaseInstantNuRecPrimitive):
                model_return = model.forward(
                    rendering_cam_data=ray_bundle.rendering_data, frames_cam_meta=[ray_bundle.frame_meta]
                )
            else:
                raise TypeError(f"Unsupported model type {type(model)}.")

            return unpack_optional(model_return.rendered_cam)

    def render_camera_frame_from_ray_bundle(
        self,
        ray_bundle: RayBundle,
        render_options: Optional[CameraRenderOptions] = None,
        actors_snapshot: Optional[ActorsSnapshot] = None,
        frame_start_us: int = 0,
        frame_end_us: int = 0,
    ) -> CameraFrame:
        """Render frame buffers of a model given a ray bundle generated from a camera view.

        Use RayBundle.build() to generate rays from a camera view.

        Args:
            ray_bundle: Ray bundle containing rays and metadata for rendering
            render_options: CameraRenderOptions object specifying which fields of CameraFrame to populate
            actors_snapshot: Optional ActorsSnapshot containing the actors ids and poses
            frame_start_us: start timestamp (in microseconds) of the frame captured
            frame_end_us: end timestamp (in microseconds) of the frame captured

        Returns:
            CameraFrame with color_image as (h, w, 3) float32 tensor with values in [0, 1]
        """
        if render_options is None:
            render_options = CameraRenderOptions()
        if render_options.semantic_image and self._compact_camera_extra_signals_missing:
            raise ValueError(
                "compact USDZ artifacts omit camera_extra_signal; "
                "semantic renders require recompressing with "
                "--preserve-extra-signal-tensor camera_extra_signal"
            )
        rendered = self._render_volume_from_ray_bundle(
            ray_bundle=ray_bundle,
            actors_snapshot=actors_snapshot,
            frame_start_us=frame_start_us,
            frame_end_us=frame_end_us,
        )

        raster_width = ray_bundle.rendering_data.w
        raster_height = ray_bundle.rendering_data.h

        rgb_image_f32: Optional[torch.Tensor] = None
        distance_image_f32: Optional[torch.Tensor] = None
        opacity_image_f32: Optional[torch.Tensor] = None
        semantic_image_int64: Optional[torch.Tensor] = None

        if render_options.color_image:
            rgb_image_f32 = unpack_optional(rendered.rgb)  # [(hw), c]
            rgb_image_f32 = rearrange(rgb_image_f32, "(h w) d -> h w d", h=raster_height, w=raster_width).float()

        if render_options.distance_image:
            if rendered.distance is None:
                raise MissingRenderOutputError(
                    "Distance image requested, but the model or render path did not produce distance output. "
                    "This model does not support the distance product."
                )
            distance_image_f32 = rendered.distance  # [(hw),]
            distance_image_f32 = rearrange(distance_image_f32, "(h w) -> h w", h=raster_height, w=raster_width)

        if render_options.opacity_image:
            if rendered.opacity is None:
                raise MissingRenderOutputError(
                    "Opacity image requested, but the model or render path did not produce opacity output."
                )
            opacity_image_f32 = rendered.opacity  # [(hw),]
            opacity_image_f32 = rearrange(opacity_image_f32, "(h w) -> h w", h=raster_height, w=raster_width)

        if render_options.semantic_image:
            if rendered.extra_ray_signals is None or rendered.extra_ray_signals.semantic_logits is None:
                raise MissingRenderOutputError(
                    "Semantic image requested, but the model or render path did not produce semantic_logits. "
                    "Semantics output requires a model trained with semantic heads."
                )
            semantic_logits = rendered.extra_ray_signals.semantic_logits
            n_classes = semantic_logits.shape[-1]
            assert semantic_logits.numel() == raster_height * raster_width * n_classes, (
                f"semantic_logits shape {tuple(semantic_logits.shape)} not ({raster_height}*{raster_width}, n_classes)"
            )
            # (hw, n_classes) -> argmax -> (hw,) -> (h, w)
            class_ids = semantic_logits.argmax(dim=-1).to(torch.int64)
            semantic_image_int64 = rearrange(class_ids, "(h w) -> h w", h=raster_height, w=raster_width)

        return CameraFrame(
            color_image=rgb_image_f32,
            distance_image=distance_image_f32,
            opacity_image=opacity_image_f32,
            semantic_image=semantic_image_int64,
        )

    def render_camera_frame(
        self,
        camera_intrinsics: ConcreteCameraModelParametersUnion,
        camera_to_world: PoseRange,
        resolution: Tuple[int, int],  # (width, height)
        unique_sensor_idx: Optional[int] = None,
        unique_frame_idx: Optional[int] = None,
        training_frame_idx: Optional[int] = None,
        render_options: Optional[CameraRenderOptions] = None,
        actors_snapshot: Optional[ActorsSnapshot] = None,
        frame_start_us: int = 0,
        frame_end_us: int = 0,
    ) -> CameraFrame:
        """Render a model at a specified resolution from a camera view defined by its camera intrinsics and pose,
        with the option to provide training camera and trajectory information, should the model require it.

        Allows to render the model in the following use-cases:
        (a) view that was part of the training data (training camera intrinsics and pose), or
        (b) view with intrinsics from a camera available during training but an arbitrary camera pose, or
        (c) view with intrinsics obtained by modifying those of a training camera, and an arbitrary pose, or
        (d) free viewpoint, a novel view with intrinsics and pose completely independent from training cameras or poses.

        Some stages in inference may not be independent of viewpoint, but specific to a training frame
        (e.g. post-processing learned per frame), or training camera (e.g. post-processing learned per camera).
        If such stages are enabled in the model, a frame index and/or camera and trajectory index must be provided.
        As a rule of thumb, whenever the provided intrinsics originate from a training camera on a training trajectory
        make sure that the camera and trajectory index are provided.

        Args:
          - camera_intrinsics: Intrinsic parameters of camera view to render from.
          - camera_to_world: Frame start and end poses (camera-to-world transforms) and corresponding timestamps for
            the camera view with rolling shutter (simply feed the same start/end poses/timestamps for global shutter).
          - resolution: The desired width and height (width, height) of the frame to be rendered.
          - unique_sensor_idx: Optional unique index of a camera used during training if the intrinsic parameters of
              the view to generate rays is from such a camera (obtain via SceneInfo.get_camera()).
          - unique_frame_idx: Identity of the frame in the all-frame source dataset (training + held-out frames).
              Used by trajectory lookup, calibration, metrics, and other consumers that need frame identity.
              Pass None for free / novel views unrelated to any recorded frame.
          - training_frame_idx: Index into per-frame parameter tensors of training-only post-processings
              (PPISP, bilateral grids). Pass None for novel views or held-out frames that have no training-time
              parameter slot. When no holdout sampler is used, this equals unique_frame_idx.
          - render_options: CameraRenderOptions object specifying which fields of CameraFrame to populate.
          - actors_snapshot: Optional ActorsSnapshot containing the actors ids and poses.
          - frame_start_us: start timestamp (in microseconds) of the frame captured.
          - frame_end_us: end timestamp (in microseconds) of the frame captured.

        Returns:
          - CameraFrame.color_image: Predicted RGB image as f32 torch tensor of shape (height, width, channels).
        """

        # TODO: Disable inference stages that require any of the optional inputs above that are not provided.

        if self._world_to_nurec is None:
            raise ValueError("World-to-NuRec transformation is required but not initialized")

        camera_intrinsics = transform_intrinsics_to_resolution(camera_intrinsics, resolution)

        ray_bundle = RayBundle.build(
            camera_intrinsics,
            camera_to_world,
            self._world_to_nurec,
            unique_sensor_idx,
            unique_frame_idx,
            training_frame_idx,
        )

        return self.render_camera_frame_from_ray_bundle(
            ray_bundle,
            render_options=render_options,
            actors_snapshot=actors_snapshot,
            frame_start_us=frame_start_us,
            frame_end_us=frame_end_us,
        )

    def _lidar_rays_inference(
        self,
        ray_bundle: LidarRayBundle,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
        lidar_stream: Optional[torch.cuda.Stream] = None,
    ) -> GaussiansCompositeReturn:
        # TODO: This function is too low-level to be used from client code (e.g. gRPC service).
        #       Expose a high-level function instead that computes the tensors provided here from LiDAR intrinsics
        #       and pose information, similar to RenderableModel.render_camera_frame().
        if isinstance(self._model, GaussiansComposite):
            with self.autocast, torch.no_grad():
                batch = DataAndRenderingBatch(
                    data=DataBatch(
                        lidar=DataBatch.Lidar(
                            meta=[ray_bundle.frame_meta],
                            labels=LidarFrameLabels(),  # empty labels
                        ),
                    ),
                    rendering=RenderingBatch(lidar=ray_bundle.rendering_data),
                )
                return self._model(
                    batch=batch,
                    enable_calib=False,
                    enable_pp=True,
                    edited_cuboid_tracks=edited_cuboid_tracks,
                    lidar_stream=lidar_stream,
                )
        else:
            raise TypeError(f"Lidar rendering does not support models of type {type(self._model)}.")

    def default_lidar_neighbor_source(self) -> str:
        """Server-config default floating-point-filter neighbour source ("rangeview" or "angle").

        Exposed so callers (e.g. the gRPC service) can seed a concrete per-request value from the
        artifact config instead of passing a sentinel.
        """
        return self._lidar_rendering_settings.lidar_post_filter_neighbor_source

    def render_lidar_frame_from_ray_bundle(
        self,
        ray_bundle: LidarRayBundle,
        raydrop_threshold: Optional[float] = None,
        opacity_threshold: Optional[float] = None,
        enable_distance_filter: Optional[bool] = None,
        distance_filter_threshold: Optional[float] = None,
        neighbor_source: str = "rangeview",
        elevation_resolution_deg: Optional[float] = None,
        azimuth_max_gap_deg: Optional[float] = None,
        actors_snapshot: Optional[ActorsSnapshot] = None,
        frame_start_us: int = 0,
        frame_end_us: int = 0,
        lidar_stream: Optional[torch.cuda.Stream] = None,
    ) -> LidarFrame:
        """Render a model given a ray bundle generated from a single lidar view and return a LidarFrame.

        Args:
            ray_bundle: The lidar ray bundle to render.
            raydrop_threshold: Override for raydrop probability threshold [0-1].
                Rays with raydrop > threshold are dropped. If None, uses artifact config default.
            opacity_threshold: Override for opacity threshold [0-1].
                Rays with opacity <= threshold are dropped. Set to 0.0 to disable (matching validation).
                If None, uses artifact config default.
            enable_distance_filter: Override to enable/disable distance-based edge filtering.
                If None, uses artifact config default.
            distance_filter_threshold: Override for distance filter threshold [0-1].
                Higher = fewer points filtered. If None, uses artifact config default.
            neighbor_source: Floating-point filter neighbour source: "rangeview" (structured 2D
                grid) or "angle" (sensor-agnostic; for model-free / unstructured lidar). Not
                Optional -- callers pass a concrete value (the gRPC service seeds it from the
                artifact-config default via `default_lidar_neighbor_source()`); falls back to
                "rangeview" only if omitted.
            elevation_resolution_deg: Angle source only. Elevation row spacing in degrees;
                None = auto-estimate (falls back to the artifact config default).
            azimuth_max_gap_deg: Angle source only. Max azimuth neighbour gap in degrees;
                None = auto (falls back to the artifact config default).
            actors_snapshot: Optional ActorsSnapshot containing the actors ids and poses
            frame_start_us: start timestamp (in microseconds) of the frame captured
            frame_end_us: end timestamp (in microseconds) of the frame captured
            lidar_stream: Optional CUDA stream to run the lidar collect+render on. When
                provided (e.g. a per-request stream owned by the serving Backend), the
                composite uses it instead of its own; when None it falls back to the
                composite-owned stream.

        Returns:
            LidarFrame containing the rendered point cloud.
        """
        if self._compact_lidar_extra_signal_tensors_missing:
            names = ", ".join(self._compact_lidar_extra_signal_tensors_missing)
            raise ValueError(
                f"compact USDZ artifacts omit requested LiDAR extra-signal tensors: {names}; "
                "disable lidar extra-signal outputs for geometry-only lidar rendering "
                "or recompress with --preserve-extra-signal-tensor for each required tensor"
            )

        # Use provided overrides or fall back to artifact config defaults
        effective_raydrop_threshold = (
            raydrop_threshold
            if raydrop_threshold is not None
            else self._lidar_rendering_settings.lidar_raydrop_threshold
        )
        effective_opacity_threshold = (
            opacity_threshold
            if opacity_threshold is not None
            else self._lidar_rendering_settings.lidar_opacity_threshold
        )
        effective_enable_distance_filter = (
            enable_distance_filter
            if enable_distance_filter is not None
            else self._lidar_rendering_settings.enable_lidar_post_filter
        )
        effective_distance_filter_threshold = (
            distance_filter_threshold
            if distance_filter_threshold is not None
            else self._lidar_rendering_settings.lidar_post_filter_threshold
        )
        assert neighbor_source in ("rangeview", "angle"), (
            f"neighbor_source must be 'rangeview' or 'angle', got {neighbor_source!r}"
        )
        effective_neighbor_source = neighbor_source
        effective_elevation_resolution_deg = (
            elevation_resolution_deg
            if elevation_resolution_deg is not None
            else self._lidar_rendering_settings.lidar_post_filter_elevation_resolution_deg
        )
        effective_azimuth_max_gap_deg = (
            azimuth_max_gap_deg
            if azimuth_max_gap_deg is not None
            else self._lidar_rendering_settings.lidar_post_filter_azimuth_max_gap_deg
        )

        edited_cuboid_tracks: Optional[CuboidTracks] = None
        if actors_snapshot is not None:
            edited_cuboid_tracks = self._edit_actors(
                actors_snapshot=actors_snapshot, frame_start_us=frame_start_us, frame_end_us=frame_end_us
            )

        # Copy all tensors with the ray bundle to GPU
        ray_bundle = cast(LidarRayBundle, tree_map(ray_bundle, to_float_device(self.device)))
        model_return = self._lidar_rays_inference(
            ray_bundle, edited_cuboid_tracks=edited_cuboid_tracks, lidar_stream=lidar_stream
        )

        rendered = unpack_optional(model_return.rendered_lidar)
        dist = unpack_optional(rendered.distance)
        rays = ray_bundle.rendering_data.rays.reshape(-1, 6)
        xyz_end = torch.addcmul(rays[:, 0:3], rays[:, 3:6], dist.unsqueeze(1))

        if rendered.extra_ray_signals is not None and rendered.extra_ray_signals.intensity is not None:
            intensity = rendered.extra_ray_signals.intensity.squeeze()
        else:
            intensity = None

        # filter by opacity mask
        opacity = unpack_optional(rendered.opacity)
        keep_mask = opacity > effective_opacity_threshold

        if rendered.extra_ray_signals is not None and rendered.extra_ray_signals.raydrop is not None:
            # filter by predicted raydrop mask
            raydrop = rendered.extra_ray_signals.raydrop.flatten()
            did_return_pred = raydrop < effective_raydrop_threshold
            keep_mask &= did_return_pred
        else:
            did_return_pred = keep_mask.clone()

        # floating-point ("flying pixel") filter. neighbor_source selects how ray neighbours are
        # found: "rangeview" (structured 2D grid) or "angle" (sensor-agnostic; works for model-free /
        # unstructured lidar, which has no range-view grid).
        if effective_enable_distance_filter:
            if effective_neighbor_source == "angle":
                # Sensor-frame ray directions (NuRec-frame directions rotated by T_nurec_sensor_end),
                # matching the validation path's `ray_direction_sensor`.
                ray_direction_sensor = (ray_bundle.T_nurec_sensor_end[:3, :3] @ rays[:, 3:6].T).T
                filter_mask = angle_gradient_filter(
                    rendered,
                    ray_direction_sensor,
                    did_return_pred,
                    effective_distance_filter_threshold,
                    effective_elevation_resolution_deg,
                    effective_azimuth_max_gap_deg,
                )
            elif ray_bundle.rendering_data.sensor_model_parameters[0] is None:
                # A model-free unit has no (rows, columns) grid for rangeview to walk. The angle
                # source above is the one that works here, so say which knob to turn.
                raise ValueError(
                    "the range-view floating-point filter needs the sensor's (rows, columns) grid, "
                    "which a model-free lidar does not have. Request neighbor_source='angle', or "
                    "disable the filter for this unit."
                )
            else:
                height = ray_bundle.rendering_data.rays.shape[1]
                width = ray_bundle.rendering_data.rays.shape[2]
                model_elements = ray_bundle.model_elements.to(dtype=torch.int32)
                # remove points if filter_mask equal to True
                filter_mask = distance_based_filter(
                    rendered, model_elements, did_return_pred, effective_distance_filter_threshold, height, width
                )
            keep_mask &= ~filter_mask

        xyz_end = xyz_end[keep_mask]
        if intensity is not None:
            intensity = intensity[keep_mask]
            intensity = intensity.contiguous()

        pc_nurec = xyz_end.contiguous().unsqueeze(0)  # (1, n, 3)

        # Transform the points from NuRec space to Lidar space (the end timestamp)
        T_nurec_sensor_end = ray_bundle.T_nurec_sensor_end.unsqueeze(0)  # (1, 4, 4)
        pc_sensor = transform_point_cloud(pc_nurec, T_nurec_sensor_end).squeeze(0)  # (n, 3)

        return LidarFrame(point_positions=pc_sensor, point_intensities=intensity)
