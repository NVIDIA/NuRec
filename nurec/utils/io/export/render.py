# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import sys
import time
import zipfile

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, OrderedDict, Tuple

import click
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import torch

from imageio_ffmpeg import get_codec_config
from omegaconf import OmegaConf
from PIL import Image

import ncore.impl.common.transformations as ncore_transformations
import nurec.utils.cli as cli

from nurec.artifact import Artifact
from nurec.asset_edits.runtime import (
    AspectOverrideRuntime,
    AspectScheduleRegistry,
    load_manifest_from_path,
    load_manifest_from_usdz,
    load_schedule_from_path,
)
from nurec.datasets.samplers.base import BaseFrameSampler
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.render import (
    ActorsSnapshot,
    ActorTracks,
    CameraFrame,
    CameraRenderOptions,
    PoseRange,
    RenderableModel,
    SceneInfo,
    SensorTrajectory,
    TrackLabeler,
    overlay_bounding_boxes,
    transform_intrinsics_to_resolution,
)
from nurec.utils.cli import SettingsCollector
from nurec.utils.cudaeventtimer import CudaEventTimer
from nurec.utils.geometry import pose_offsets_to_se3, se3_matrix_to_tquat
from nurec.utils.io.export.render_helpers import (
    CameraOutputSubdir,
    parse_frame_index,
    select_frame_indices,
    validate_sensor_output_names,
)
from nurec.utils.io.export.render_lidar import render_lidar_frames
from nurec.utils.io.export.render_nsight import NSIGHT_OUTPUT_DIR_ENV, NsightCapture, run_nsight_profile
from nurec.utils.metrics import MetricsCollector
from nurec.utils.types import RigTrajectories
from nurec.utils.visualize import (
    SEMANTICS_COLOR_LUT_DEFAULT_SIZE,
    build_semantics_class_colors,
    create_semantics_legend,
    scalar2img,
    sem2img,
)


log = logging.getLogger(__name__)


# Overlay box color for --label-tracks / --list-tracks (the option value is the
# Gaussian layer id whose cuboid tracks are labeled).
_TRACK_LABEL_COLOR: Tuple[int, int, int] = (0, 255, 255)


def _centroid_distance(point: Tuple[float, float, float], ref: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(point, dtype=np.float64) - ref))


def _render_start_position(rig_trajectories: RigTrajectories) -> Optional[np.ndarray]:
    """World-frame position of the render start pose, used to distance-sort tracks.

    Returns ``None`` if the trajectory exposes no usable rig pose (distance sort is then skipped).
    The rig poses share the world frame of the labeled tracks (both feed the rendered
    ``camera_to_world``), so Euclidean distances are directly comparable.
    """
    try:
        traj = rig_trajectories.rig_trajectories[0]
        mat = traj.T_rig_worlds[0]
        mat_np = mat.detach().cpu().numpy() if hasattr(mat, "detach") else np.asarray(mat)
        return mat_np[:3, 3].astype(np.float64)
    except (AttributeError, IndexError, TypeError) as exc:
        log.debug("no usable rig start pose (%s); track listing will be unordered.", exc)
        return None


def build_training_frame_mapping(
    parsed_config: dict,
    rig_trajectory: RigTrajectories.RigTrajectory,
) -> npt.NDArray[np.int32]:
    """Build the all-frame -> training-only camera frame index mapping.

    Reconstructs the same mapping that ``NCOREDataSource.setup_training_frame_mapping``
    produces during training. The effective training-frame set is the union of the frames
    reachable by the configured training samplers: a frame is considered a training frame
    iff at least one sampler group's frame sampler can return it. This is computed directly
    from the sampler groups stored in the artifact's parsed config together with the
    per-camera frame counts from the rig trajectory (it is NOT inferred from validation
    holdout parameters).

    Args:
        parsed_config: The artifact's parsed training configuration (dict form of the OmegaConf config).
        rig_trajectory: The rig trajectory from the artifact, providing per-camera frame counts.

    Returns:
        An int32 array of length ``total_frames`` (sum of frames across all cameras in the
        dataset). It is indexed by ``unique_frame_idx`` (0-based linear index over all
        camera frames, training and non-training combined). Each entry is the contiguous
        0-based training frame index over all training camera frames, or -1 for a frame
        that is unreachable by any configured training sampler (i.e. not a training frame).
        The training index space has no gaps where non-training frames were excluded, which
        lets PPISP size its per-frame parameter tensors tightly to the number of training
        frames.
    """
    cfg = OmegaConf.create(parsed_config)

    samplers = OmegaConf.to_container(OmegaConf.select(cfg, "dataset.samplers", default={})) or {}
    assert isinstance(samplers, dict), "dataset.samplers must be a dict"
    groups = list(samplers.values())

    # Determine frame counts per camera from the rig trajectory timestamps.
    camera_frame_counts: dict[str, int] = {
        unique_camera_id: timestamps.shape[0]
        for unique_camera_id, timestamps in rig_trajectory.cameras_frame_timestamps_us.items()
    }
    total_frames = sum(camera_frame_counts.values())

    # Reconstruct the effective training-frame masks exactly like
    # NCOREDataSource.setup_training_frame_mapping: a frame is a training
    # frame iff some sampler group's frame sampler can return it (NOT
    # inferred from the validation holdout configuration).
    camera_masks: dict[str, npt.NDArray[np.bool_]] = {}
    for unique_camera_id, n_frames in camera_frame_counts.items():
        mask = np.zeros(n_frames, dtype=np.bool_)
        for group in groups:
            mask |= BaseFrameSampler.group_training_frame_mask(group, "camera", n_frames)
        camera_masks[unique_camera_id] = mask

    if (
        not groups
        or all(bool(m.all()) for m in camera_masks.values())
        or not any(bool(m.any()) for m in camera_masks.values())
    ):
        # Identity mapping (all frames are training frames) — mirrors the
        # datasource's fallback for unrestricted or absent samplers.
        return np.arange(total_frames, dtype=np.int32)

    cameras_start_frame_indices = rig_trajectory.cameras_linear_start_frame_indices
    assert cameras_start_frame_indices is not None, (
        "cameras_linear_start_frame_indices is required to build the training frame mapping"
    )

    global_to_training = np.full(total_frames, -1, dtype=np.int32)
    training_counter = 0
    for unique_camera_id in sorted(cameras_start_frame_indices.keys()):
        n_frames = camera_frame_counts[unique_camera_id]
        linear_start = cameras_start_frame_indices[unique_camera_id]
        mask = camera_masks[unique_camera_id]
        for local_idx in range(n_frames):
            if mask[local_idx]:
                global_to_training[linear_start + local_idx] = training_counter
                training_counter += 1

    return global_to_training


def export_video(
    image_dir: Path,
    video_path: Path,
    fps: float = 30.0,
    crf: int = 20,
    max_keyframe_interval_seconds: float = 0.5,
    image_extension: str = "png",
) -> bool:
    """Convert an image file sequence stored in a directory to an H264 MP4 video.

    Args:
        image_dir: Directory containing a sequence of images.
        video_path: Path to the output MP4 file.
        fps: Frame rate of the video to generate.
        crf: Video quality (Constant Rate Factor). Valid range is 0-51. Typical range is 18-28.
          Lower means higher quality.
        max_keyframe_interval_seconds: Maximum interval between keyframes in seconds.
        image_extension: Extension (without the dot) of the image files to find in the directory.

    Returns:
        True if successful, False if image files were not found in the directory.

    Raises if arguments are out of range or if video writer failed.
    """

    if fps <= 0.0:
        raise ValueError(f"fps ({fps}) is negative")
    if crf < 0 or crf > 51:
        raise ValueError(f"crf ({crf}) is outside of the allowed range (0, 51)")
    if max_keyframe_interval_seconds <= 0.0:
        raise ValueError(f"max_keyframe_interval_seconds ({max_keyframe_interval_seconds}) is negative")

    image_files = sorted(image_dir.glob("*." + image_extension))
    if not image_files:
        log.warning(f"No image files found in {image_dir}")
        return False

    log.info(f"Exporting {len(image_files)} frames to {video_path}")

    # Loads all frames into memory - poses a memory risk for long sequences and/or high resolution frames.
    frames = []
    for image_file in image_files:
        frame = iio.imread(image_file)
        frames.append(frame)

    max_keyframe_interval_frames = max(round(max_keyframe_interval_seconds * fps), 1)

    # Codec resolver: NVENC for .mp4 if GPU available and it accepts the frame
    # size, else libsvtav1 (CPU AV1) fallback. CRF translates to "-cq" for NVENC,
    # "-crf" for SVT-AV1. See imageio_ffmpeg.get_codec_config (bundled in the wheel).
    codec, codec_params = get_codec_config(video_path, crf=crf, size=(frames[0].shape[1], frames[0].shape[0]))
    iio.imwrite(
        video_path,
        frames,
        fps=fps,
        codec=codec,
        pixelformat="yuv420p",
        # yuv420p needs even dimensions; the default of 16 resizes 1080-pixel frames to 1088.
        macro_block_size=2,
        quality=None,  # Set to None to use the codec-specific quality flag from codec_params
        # Encoder-specific quality params (from get_codec_config) +
        # container-level params common to both encoders.
        output_params=[
            *codec_params,
            # Guarantees a keyframe at least every second for an improved seeking experience.
            "-g",
            str(max_keyframe_interval_frames),
            # Web-friendly optimization, see: https://trac.ffmpeg.org/wiki/Encode/H.264#faststartforwebvideo
            "-movflags",
            "+faststart",
        ],
    )

    return True


def export_videos_and_json(
    output_dir: str,
    camera_ids: List[str],
    fps: float = 30.0,
    crf: int = 20,
    image_extension: str = "png",
    frames_subdir: CameraOutputSubdir = CameraOutputSubdir.COLOR,
) -> None:
    """Export frames from each camera subdirectory to an MP4 video and create a JSON index.

    Each video is written inside its own camera directory, beside the frames it was made from,
    which is where ``eval-rendering-metrics`` looks for it: its ``--render-dir`` contract is
    ``<render-dir>/<image-seq>/video.mp4``. The JSON index stays in the output-type subdirectory
    so that multi-output-type renders do not collide, and records paths relative to it.

    Args:
        output_dir: Base output directory (frames live under ``<frames_subdir>/<camera_id>/``).
        camera_ids: List of camera IDs that were rendered
        fps: Frames per second for the output videos
        crf: Video quality (Constant Rate Factor). Typical range is 18-28. Lower means higher quality.
        image_extension: Extension (without the dot) of the image files to find in the directory.
        frames_subdir: Which output type subdirectory to export (see :class:`CameraOutputSubdir`).
    """
    validate_sensor_output_names(camera_ids, sensor_type="camera")
    output_path = Path(output_dir)
    subdir_path = output_path / frames_subdir
    video_paths: Dict[str, str] = {}

    for camera_id in camera_ids:
        camera_dir = subdir_path / camera_id
        if not camera_dir.exists():
            log.warning(f"Camera directory not found: {camera_dir}")
            continue

        video_relative_path = f"{camera_id}/video.mp4"
        video_path = subdir_path / video_relative_path
        if export_video(camera_dir, video_path, fps=fps, crf=crf, image_extension=image_extension):
            video_paths[camera_id] = video_relative_path

    if not video_paths:
        return

    # Save JSON with relative paths next to the videos so each output type has its own index.
    json_path = subdir_path / "render_cli_videos.json"
    with open(json_path, "w") as f:
        json.dump(video_paths, f, indent=2)

    log.info(f"Exported {len(video_paths)} videos under {subdir_path}, JSON saved to {json_path}")


def _save_rendered_frame_to_disk(
    numpy_array: np.ndarray,
    image_path: str,
) -> None:
    """Convert a rendered frame to a PIL image and save it to disk.

    Intended to be run in a thread pool to avoid blocking the main rendering loop on disk I/O.

    Args:
        numpy_array: HxWx3 uint8 RGB image array.
        image_path: Full path to write the image file.
    """
    rgb_image = Image.fromarray(numpy_array)

    if image_path.lower().endswith((".jpg", ".jpeg")):
        rgb_image.save(image_path, quality=93)
    else:
        rgb_image.save(image_path)


def _save_semantics_legend(
    output_dir: str,
    semantics_class_colors: np.ndarray,
    semantic_class_names: Optional[Dict[int, str]],
) -> None:
    """Write ``semantics legend`` under ``output_dir`` from the class color LUT.

    ``semantic_class_names`` may be sparse (e.g. ``{0: "bg", 1: "car", 5: "ped"}``); any ids that
    fall within the color LUT but have no name fall back to numeric labels in the legend.
    """
    n_leg = semantics_class_colors.shape[0]
    legend_names: Optional[Dict[int, str]] = None
    if semantic_class_names:
        max_named_id = max(semantic_class_names)
        if max_named_id >= n_leg:
            raise ValueError(
                f"semantic_class_names contains class id {max_named_id} but the color LUT only has "
                f"{n_leg} rows; the color LUT must cover every named class id."
            )
        legend_names = semantic_class_names
    legend_img = create_semantics_legend(semantics_class_colors, class_names=legend_names)
    seg_root = os.path.join(output_dir, CameraOutputSubdir.SEMANTICS_COLORIZED)
    os.makedirs(seg_root, exist_ok=True)
    legend_path = os.path.join(seg_root, "semantic_classes.png")
    Image.fromarray(legend_img).save(legend_path)
    log.info("Wrote %s", legend_path)


def get_unique_frame_index(
    sensor_id: str,
    frame_idx: int,
    sensors_start_frame_indices: Optional[Dict[str, int]] = None,
    sequence_id: Optional[str] = None,
) -> Optional[int]:
    """Calculate the unique frame index for a camera frame used during training.

    Args:
        sensor_id: The sensor logical id, e.g. camera_front_wide_120fov.
        frame_idx: The 0-based index of the training frame from the selected sensor.
        sensors_start_frame_indices: Optional mapping of unique sensor identifiers to the linear unique frame index of
            the first training frame provided by the sensor.
        sequence_id: The sequence ID used to construct unique sensor identifiers.

    Returns:
        The unique frame index of the selected frame (unique within the sensor type),
        or None if cameras_start_frame_indices is not provided.
    """
    if sensors_start_frame_indices is None:
        return None  # Logged a warning in __init__() to avoid repeated logging per frame.
    unique_sensor_id = f"{sensor_id}@{sequence_id}"
    if unique_sensor_id not in sensors_start_frame_indices:
        raise ValueError(f"Sensor {unique_sensor_id} not found in sensors_start_frame_indices")
    return sensors_start_frame_indices[unique_sensor_id] + frame_idx


def manipulate_actor_poses(actors_snapshot: ActorsSnapshot) -> ActorsSnapshot:
    """Turns an ActorsSnapshot into another ActorsSnapshot with actors lifted from the ground along the world Z-axis."""
    poses = actors_snapshot.actor_poses  # [M,2,7], poses[i,j,:] ~ [tx,ty,tz,qx,qy,qz,qw] where j=0,1 (start,end pose)
    assert poses.shape == (len(actors_snapshot.actor_ids), 2, 7)  # Just info for the reader
    edited_poses = poses.clone()
    edited_poses[:, :, 2] += 1.0  # Translates all actors in world space along the Z axis
    return ActorsSnapshot(
        actor_ids=actors_snapshot.actor_ids,
        actor_poses=edited_poses,
        cuboid_dims=actors_snapshot.cuboid_dims,
    )


def split_unique_sensor_ids(unique_sensor_ids: Iterable[str]) -> Tuple[List[str], Optional[str]]:
    """Split each unique sensor ID in a list into its constituents <sensor_id>@<sequence_id> and
    make sure that the sequence_id part is identical for all, otherwise raises an error.
    """
    sensor_ids = []
    sequence_ids = []
    for unique_sensor_id in unique_sensor_ids:
        parts = unique_sensor_id.split("@")
        sensor_ids.append(parts[0])
        sequence_ids.append(parts[1] if len(parts) > 1 else None)
    # Make sure scene_id is identical for all sensors
    unique_sequence_ids = set(sequence_ids)
    if len(unique_sequence_ids) > 1:
        raise ValueError(f"Sequence IDs are not identical for all provided sensors: {unique_sensor_ids}")
    return sensor_ids, unique_sequence_ids.pop() if len(unique_sequence_ids) > 0 else None


def fix_frame_timestamps(
    frame_timestamps: np.ndarray,
    min_timestamp: int,
    max_timestamp: int,
    rolling_shutter_duration: Optional[int] = None,
):
    """Validate a (n_frames, 2) array of frame start and end timestamps, fill in missing start timestamps or
    set the start timestamps to meet a fixed rolling shutter duration if specified.

    - Raise an error if frame end timestamps are invalid (e.g. -1).
    - Frame start timestamps can be missing (set to -1), this is not an error.
    - If rolling shutter duration is specified, frame start timestamps are overwritten to respect the duration,
      and then clamped to within the range (min_timestamp, max_timestamp).
    - If no rolling shutter duration is specified, but frame start timestamps are set to -1,
      they are clamped to the frame end timestamps, effectively disabling rolling shutter.
    - Frame start and end timestamps must be within the (min_timestamp, max_timestamp) range.

    Args:
        frame_timestamps: (n_frames, 2) array of frame start and end timestamps
        min_timestamp: Min allowed timestamp value
        max_timestamp: Max allowed timestamp value
        rolling_shutter_duration: Optional rolling shutter duration

    Returns: (n_frames, 2) array of clamped frame start and end timestamps.
    """

    num_frames = frame_timestamps.shape[0]
    assert frame_timestamps.shape == (num_frames, 2), "frame timestamps expected as an array of shape (n_frames, 2)"
    assert frame_timestamps.dtype == np.int64, "frame timestamps expected as an int64 array"

    frame_start_timestamps = frame_timestamps[:, 0]
    frame_end_timestamps = frame_timestamps[:, 1]

    # Make sure no frame-end timestamp is missing.
    if (count := np.sum(frame_end_timestamps < 0)) > 0:
        raise ValueError(f"Frame-end timestamps are required but {count} out of {num_frames} are missing")
    # Make sure all frame-end timestamps are within the trajectory timestamp range.
    if (count := np.sum((frame_end_timestamps < min_timestamp) | (frame_end_timestamps > max_timestamp))) > 0:
        raise ValueError(f"{count} of {num_frames} frame-end timestamps are out of bounds")

    if rolling_shutter_duration is not None:
        # Set frame start timestamps such that
        # frame_start_timestamps_us + rolling_shutter_duration = frame_end_timestamps_us,
        # and clamp within the overall time range of the trajectory.
        if min_timestamp + rolling_shutter_duration > max_timestamp:
            raise ValueError(
                f"The specified frame time ({rolling_shutter_duration} us) exceeds the overall trajectory length "
                f"({max_timestamp - min_timestamp} us)."
            )
        log.warning(
            "Overriding frame start timestamps to meet the specified "
            f"rolling shutter duration ({rolling_shutter_duration} us)"
        )
        frame_start_timestamps = (
            np.maximum(min_timestamp + rolling_shutter_duration, frame_end_timestamps) - rolling_shutter_duration
        )

        if (num_clamped := np.sum(frame_start_timestamps + rolling_shutter_duration < frame_end_timestamps)) > 0:
            log.warning(f"{num_clamped} start timestamps clamped within the trajectory timestamp range")

    elif (count := np.sum(frame_start_timestamps < 0)) > 0:
        # Handle case when frame start timestamps stored inside the artifact are -1.
        # Happened when saving rig_trajectories with obsolete datasource.get_rig_trajectories(end_frame_timestamps_only=True) argument.
        log.warning(
            f"Rolling shutter effect disabled due to {count} out of {num_frames} missing frame-start timestamps"
            "(consider the --rolling-shutter-duration option if you would like to simulate rolling shutter anyway)"
        )
        frame_start_timestamps = frame_end_timestamps

    fixed_timestamps = np.stack([frame_start_timestamps, frame_end_timestamps], axis=1)

    # Make sure all timestamps are within the allowed range.
    if (count := np.sum((fixed_timestamps < min_timestamp) | (fixed_timestamps > max_timestamp))) > 0:
        raise ValueError(f"{count} of {num_frames} frame start/end timestamps are out of bounds")

    assert fixed_timestamps.shape == (num_frames, 2)
    return fixed_timestamps


def get_trajectory_time_range(rig_trajectories: RigTrajectories) -> Tuple[int, int]:
    """Get the (start,end) timestamp of the rig trajectory in microseconds."""
    rig_trajectory = rig_trajectories.rig_trajectories[0]  # Assuming only one trajectory per clip.
    clip_start_us = int(rig_trajectory.T_rig_world_timestamps_us[0].item())
    clip_end_us = int(rig_trajectory.T_rig_world_timestamps_us[-1].item())
    assert clip_start_us >= 0
    assert clip_end_us >= 0
    return clip_start_us, clip_end_us


def _compute_bbox_overlay(
    *,
    render_bboxes: str,
    overlay_actors_snapshot: Optional[ActorsSnapshot],
    camera_model_parameters: Any,
    camera_to_world: PoseRange,
    image_width: int,
    image_height: int,
    bbox_color: Tuple[int, int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Project cuboid edges once; returns (drawn_lines_on_black, pixel_mask) or None if no overlay.

    Separating projection from drawing lets callers apply the same overlay to every
    :class:`CameraOutputSubdir` product without re-running the cuboid projection.
    """
    if render_bboxes == "none" or overlay_actors_snapshot is None or overlay_actors_snapshot.num_actors() == 0:
        return None
    cam_params = transform_intrinsics_to_resolution(
        camera_model_parameters,
        (image_width, image_height),
    )
    camera_to_world_tquat = torch.stack(
        [
            camera_to_world.start_pose_tquat_sensor_world,
            camera_to_world.end_pose_tquat_sensor_world,
        ],
        dim=0,
    )
    blank = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    drawn = overlay_bounding_boxes(
        blank, cam_params, camera_to_world_tquat, overlay_actors_snapshot, box_color_rgb=bbox_color
    )
    # mask is True where any channel is non-zero (i.e. where lines were drawn).
    # Note: black bbox_color (0,0,0) produces an invisible overlay and mask will be all False.
    mask = drawn.any(axis=-1)
    return drawn, mask


def _apply_precomputed_bbox_overlay(
    image_rgb_u8: np.ndarray,
    overlay: Optional[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Apply a precomputed bbox overlay (from _compute_bbox_overlay) to an image."""
    result = image_rgb_u8.copy()
    if overlay is None:
        return result
    drawn, mask = overlay
    result[mask] = drawn[mask]
    return result


class SensorCalibProvider(ABC):
    """Base class for providing sensor intrinsics and poses in order to abstract away the calibration data source

    The data source may be all training view intrinsics and poses or a batch of novel view intrinsics and poses.
    """

    _camera_ids: List[str]
    _sequence_id: str | None

    def __init__(self, unique_sensor_ids: Iterable[str]):
        self._camera_ids, self._sequence_id = split_unique_sensor_ids(unique_sensor_ids)

    def get_unique_sensor_id(self, camera_id: str) -> str:
        if self._sequence_id is not None:
            return f"{camera_id}@{self._sequence_id}"
        else:
            return f"{camera_id}"

    @abstractmethod
    def get_available_camera_ids(self) -> List[str]:
        """Get the available camera ids that can be used to query poses"""
        pass

    @abstractmethod
    def get_num_camera_frames(self, camera_id: str) -> int:
        """Get the number of frames available to query poses for a camera"""
        pass

    @abstractmethod
    def get_camera_view_pose(self, camera_id: str, frame_idx: int) -> PoseRange:
        """Get the start/end poses and timestamps for a frame"""
        pass

    @abstractmethod
    def get_camera_calibration(self, camera_id: str) -> RigTrajectories.CameraCalibration:
        """Get the camera intrinsics and extrinsics (sensor-to-rig pose) for a selected camera."""
        pass

    def get_unique_frame_index(self, camera_id: str, frame_idx: int) -> Optional[int]:
        """Get the unique frame index for a given camera and frame index.

        Returns a unique frame index that can be passed to rendering call, e.g. to apply the ISP of a training view,
        or None if the object type can not provide this index.
        """
        log.warning(f"get_unique_frame_index() not supported by {self.__class__.__name__}")
        return None


class SensorCalibProviderFromRigTrajectories(SensorCalibProvider):
    """Provides sensor intrinsics and poses extracted from a RigTrajectories instance"""

    def __init__(
        self,
        rig_trajectories: RigTrajectories,
        rig_translation_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rig_rotation_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        interpolate_poses: bool = False,
        rolling_shutter_duration: Optional[int] = None,
    ):
        super().__init__(unique_sensor_ids=rig_trajectories.camera_calibrations.keys())

        self._T_rig_world_base = rig_trajectories.T_world_base
        self._camera_calibrations = rig_trajectories.camera_calibrations
        self._rig_trajectory = rig_trajectories.rig_trajectories[0]  # Assuming only one trajectory per clip
        self._rig_pose_interpolator: Optional[ncore_transformations.PoseInterpolator] = None

        if self._rig_trajectory.cameras_linear_start_frame_indices is None:
            # get_unique_frame_index() will return None. Warn here to avoid repeated warnings per frame.
            log.warning("Unique frame indices for training frames are not available")

        # Get the timestamp range of the rig trajectory for some checks below.
        trajectory_start_us = int(self._rig_trajectory.T_rig_world_timestamps_us[0].item())
        trajectory_end_us = int(self._rig_trajectory.T_rig_world_timestamps_us[-1].item())

        # Check and fill in missing camera timestamps, and support a manually specified rolling shutter duration.
        # Maps each camera_id to a (num_frames, 2) array of frame start and end timestamps in microseconds.
        self._camera_frame_timestamps_us: dict[str, np.ndarray] = {}
        for unique_sensor_id, frame_timestamps_us in self._rig_trajectory.cameras_frame_timestamps_us.items():
            self._camera_frame_timestamps_us[unique_sensor_id] = fix_frame_timestamps(
                frame_timestamps_us.cpu().numpy(), trajectory_start_us, trajectory_end_us, rolling_shutter_duration
            )

        # Precompute the rig pose offset transformation from its parameters.
        self._rig_offset_se3 = pose_offsets_to_se3(rig_translation_offset, rig_rotation_offset)

        if interpolate_poses:
            log.info(f"{self.__class__.__name__}: Pose interpolation ENABLED")
            self._rig_pose_interpolator = ncore_transformations.PoseInterpolator(
                poses=self._rig_trajectory.T_rig_worlds.cpu(),
                timestamps=self._rig_trajectory.T_rig_world_timestamps_us.cpu(),
            )
        else:
            log.info(f"{self.__class__.__name__}: Pose interpolation DISABLED")
            if self._rig_trajectory.cameras_frame_T_rig_worlds is None:
                raise ValueError(
                    "Per-frame rig-to-world poses are missing from input trajectory data but is "
                    "required when pose interpolation is disabled."
                )

    def get_available_camera_ids(self) -> List[str]:
        return list(self._camera_ids)

    def get_num_camera_frames(self, camera_id: str) -> int:
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        return len(self._rig_trajectory.cameras_frame_timestamps_us[unique_sensor_id])

    def get_camera_calibration(self, camera_id: str) -> RigTrajectories.CameraCalibration:
        """Get the camera intrinsics and extrinsics for a selected camera."""
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        return self._camera_calibrations[unique_sensor_id]

    def get_camera_view_pose(self, camera_id: str, frame_idx: int) -> PoseRange:
        """Get the start/end poses and timestamps for a frame"""
        # Access the frame start and end timestamps for the given camera and frame index.
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        camera_timestamps_us = self._camera_frame_timestamps_us[unique_sensor_id]
        assert camera_timestamps_us.dtype == np.int64
        frame_start_us = camera_timestamps_us[frame_idx, 0]
        frame_end_us = camera_timestamps_us[frame_idx, 1]

        # Get the camera-to-rig pose from the camera calibration.
        camera_to_rig = self.get_camera_calibration(camera_id).T_sensor_rig.cpu().numpy()

        # Modify sensor extrinsics by the rig offset. This effectively shifts the cameras in the rig frame.
        # rig_offset_se3 should be identity when pose offsets are not specified (they should default to zero).
        camera_to_rig = self._rig_offset_se3 @ camera_to_rig

        if self._rig_pose_interpolator is not None:
            # Interpolate the test trajectory at the frame (start, end) timestamps to obtain frame start and end
            # rig-to-world poses for the frame to be rendered.
            rig_to_world = self._rig_pose_interpolator.interpolate_to_timestamps([frame_start_us, frame_end_us]).astype(
                np.float32
            )
        else:
            # Directly fetch the per-frame pose from the trajectory data instead of interpolating.
            # Downstream code expects float32 dtype.
            assert self._rig_trajectory.cameras_frame_T_rig_worlds is not None
            rig_to_world = (
                self._rig_trajectory.cameras_frame_T_rig_worlds[unique_sensor_id][frame_idx].numpy().astype(np.float32)
            )

        assert rig_to_world.shape == (2, 4, 4)
        assert rig_to_world.dtype == np.float32

        camera_to_world = PoseRange(
            start_pose_tquat_sensor_world=se3_matrix_to_tquat(rig_to_world[0] @ camera_to_rig),
            end_pose_tquat_sensor_world=se3_matrix_to_tquat(rig_to_world[1] @ camera_to_rig),
            start_timestamp_us=int(frame_start_us),
            end_timestamp_us=int(frame_end_us),
        )
        return camera_to_world

    def get_unique_frame_index(self, camera_id: str, frame_idx: int) -> Optional[int]:
        """Get the unique frame index for a given camera and frame index."""
        return get_unique_frame_index(
            camera_id, frame_idx, self._rig_trajectory.cameras_linear_start_frame_indices, self._sequence_id
        )


class SensorCalibProviderFromSensorTrajectories(SensorCalibProvider):
    """Provides sensor poses from the output of a RenderableModel.get_camera_trajectories() call
    and intrinsics from a RigTrajectories.CameraCalibration object, passable to a render call.
    """

    def __init__(
        self,
        camera_calibrations: OrderedDict[str, RigTrajectories.CameraCalibration],
        camera_trajectories: Dict[str, SensorTrajectory],
        cameras_start_frame_indices: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            camera_calibrations: A dictionary of camera calibrations (unique_camera_id -> CameraCalibration).
            camera_trajectories: A dictionary of camera trajectories (unique_camera_id -> SensorTrajectory).
            cameras_start_frame_indices: An optional dictionary of linear start frame indices (unique_camera_id -> int),
                needed by get_unique_frame_index(). If not provided, get_unique_frame_index() will return None.
        """
        # Camera IDs in the camera_calibrations and camera_trajectories should be the same.
        # But being tolerant in anticipation of differences of (currently unknown) edge cases.
        unique_camera_ids_calib = set(camera_calibrations.keys())
        unique_camera_ids_traj = set(camera_trajectories.keys())
        if unique_camera_ids_calib != unique_camera_ids_traj:
            log.warning(
                f"{self.__class__.__name__}: "
                f"Camera IDs in camera_calibrations and those in camera_trajectories do not match: "
                f"{list(camera_calibrations.keys())} vs. {list(camera_trajectories.keys())}"
            )

        unique_camera_ids = unique_camera_ids_calib.intersection(unique_camera_ids_traj)

        if cameras_start_frame_indices is None:
            # get_unique_frame_index() will return None. Warn here to avoid repeated warnings per frame.
            log.warning("Unique frame indices for training frames are not available")

        # No overlap between keys means that this class is completely disfunctional, so raise an error.
        if len(unique_camera_ids) == 0:
            raise ValueError("Camera IDs in camera calibrations and those in camera_trajectories do not overlap")

        self._camera_calibrations = OrderedDict((k, camera_calibrations[k]) for k in unique_camera_ids)
        self._camera_trajectories = {k: camera_trajectories[k] for k in unique_camera_ids}
        self._cameras_start_frame_indices = cameras_start_frame_indices

        super().__init__(unique_sensor_ids=unique_camera_ids)

    def get_available_camera_ids(self) -> List[str]:
        """Get the available camera ids that can be used to query poses"""
        return self._camera_ids

    def get_num_camera_frames(self, camera_id: str) -> int:
        """Get the number of frames available to query poses for a camera"""
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        return len(self._camera_trajectories[unique_sensor_id])

    def get_camera_view_pose(self, camera_id: str, frame_idx: int) -> PoseRange:
        """Get the start/end poses and timestamps for a frame"""
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        camera_trajectory = self._camera_trajectories[unique_sensor_id]
        poses = camera_trajectory.poses_startend_sensor_world[frame_idx].cpu().numpy()
        timestamps = camera_trajectory.timestamps_startend_us[frame_idx].cpu().numpy()
        assert poses.shape == (2, 4, 4)
        assert timestamps.shape == (2,)
        return PoseRange(
            start_pose_tquat_sensor_world=se3_matrix_to_tquat(poses[0]),
            end_pose_tquat_sensor_world=se3_matrix_to_tquat(poses[1]),
            start_timestamp_us=int(timestamps[0]),
            end_timestamp_us=int(timestamps[1]),
        )

    def get_camera_calibration(self, camera_id: str) -> RigTrajectories.CameraCalibration:
        """Get the camera intrinsics and extrinsics for a selected camera."""
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        return self._camera_calibrations[unique_sensor_id]

    def get_unique_frame_index(self, camera_id: str, frame_idx: int) -> Optional[int]:
        """Get the unique frame index for a given camera and frame index."""
        return get_unique_frame_index(camera_id, frame_idx, self._cameras_start_frame_indices, self._sequence_id)


@dataclass
class CameraRenderSample:
    """A measured output and the frame metadata needed to save it later."""

    frame_index: int
    iteration: int
    images: dict[CameraOutputSubdir, np.ndarray]
    camera_to_world: PoseRange
    cuda_ms: float
    valid: bool = True


def validate_camera_output(
    camera_frame: CameraFrame, resolution: tuple[int, int], render_options: CameraRenderOptions
) -> None:
    """Check requested CUDA buffers after timing, without reading or downloading pixels."""
    width, height = resolution
    for name in ("color_image", "distance_image", "semantic_image"):
        if not getattr(render_options, name):
            continue
        image = getattr(camera_frame, name)
        shape = (height, width, 3) if name == "color_image" else (height, width)
        if image is None or image.device.type != "cuda" or tuple(image.shape) != shape:
            raise ValueError(f"Renderer returned invalid {name}; expected a CUDA buffer of shape {shape}")


@click.command("render")
@click.option(
    "--artifact-path",
    type=str,
    help="Path to an USDZ artifact file exported from training, e.g. last.usdz",
    default=None,
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Directory for metrics, settings, and sensor output folders. No run or trajectory subfolder is added.",
    required=True,
    default=None,
)
@click.option(
    "--camera-id",
    "camera_ids",  # Tuple[str]
    multiple=True,
    type=str,
    help="Selection of a camera used during training by id to render from. "
    "Specify multiple times to render from multiple cameras, e.g. --camera-id cam1 --camera-id cam2. "
    "Omit to render all available cameras.",
    required=False,
)
@click.option(
    "--height",  # Keeping this option as is for CLI backward-compatibility.
    type=int,
    help="Height of the rendered frames in pixels for all selected cameras."
    " Cannot be combined with --image-scale. Omit both to use each camera's native resolution.",
    required=False,
    default=None,
)
@click.option(
    "--image-scale",  # Allows for easy scaling of the output resolution over multiple cameras.
    type=float,
    help="Specify the output image resolution in proportion of the resolution of each camera, in the range (0.0, 1.0]. "
    "Requested dimensions are floor(input_image_resolution * image_scale), then adjusted by the renderer "
    "to preserve aspect ratio. The minimum requested resolution is 2x2 pixels."
    " Cannot be combined with --height. Omit both to use each camera's native resolution.",
    required=False,
    default=None,
)
@click.option(
    "--frame-index",
    default="all",
    show_default=True,
    metavar="INDEX|START-END|all",
    help="Render one zero-based frame, an inclusive range, or all frames for each selected sensor.",
)
@click.option(
    "--frame-step",
    type=click.IntRange(min=1),
    help="Frame step within the selection, starting at its first frame.",
    default=1,
)
@click.option(
    "--warmup-frames",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Warm-up renders of each sensor's first selected frame, excluded from metrics. No frames are skipped.",
)
@click.option(
    "--iterations",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Renders per selected frame. Every iteration contributes to the metrics unless --no-report is set.",
)
@click.option(
    "--report/--no-report",
    default=True,
    show_default=True,
    help="Report timing metrics and per-iteration timing logs. --no-report skips wall-clock measurements "
    "and metrics output while retaining CUDA events and synchronization. Combine with --save none "
    "to also discard timing samples like warm-up renders.",
)
@click.option(
    "--lidar",
    is_flag=True,
    help="Render the first available LiDAR in the first trajectory at its native ray grid; save PLY point clouds. "
    "Omit this flag to render camera outputs instead.",
)
@click.option(
    "--nsight",
    is_flag=True,
    help="Capture one continuous Nsight report per camera or LiDAR sensor, including all selected frames and iterations. "
    "Each capture includes warm-up; the first also includes model initialization, all excluded from metrics. "
    "Reports are exported between sensors, never between frames; captured timings include tracing overhead. "
    "Mutually exclusive with --list-tracks.",
)
@click.option(
    "--save",
    metavar="MODES",
    default=None,
    help="Save additional outputs in color-extra/lidar-extra (or the selected image product's -extra folder). "
    "Comma-separated modes: min, max (fastest/slowest CUDA renders per sensor), "
    "last-iteration, or all-iterations. By default, ordinary outputs contain the last iteration of every frame. "
    "Use none alone to disable all image/point-cloud saving and output transfers while retaining metrics and Nsight reports. "
    "Warm-up renders are never saved.",
)
@click.option(
    "--rolling-shutter-duration",
    type=int,
    help=(
        "Elapsed time between the start and end of each frame in integer microseconds. "
        "If set, it overrides the frame start timestamps in the USDZ input to the frame end timestamps "
        "minus the specified frame time. The timestamps are then clamped within the trajectory timestamp range."
    ),
    required=False,
    default=None,
)
@click.option(
    "--image-format",
    type=click.Choice(["png", "jpg", "jpeg"]),
    help="png or jpeg",
    default="png",
)
@click.option(
    "--export-video",
    is_flag=True,
    default=False,
    help="Export rendered frames as an MP4 H.264 video per camera after rendering completes."
    " All rendered frames are loaded back to memory per camera (requires sufficient memory).",
)
@click.option(
    "--video-fps",
    type=float,
    default=30.0,
    help="Frames per second for exported videos. Must be a positive float. Only used when --export-video is set.",
)
@click.option(
    "--video-crf",
    type=int,
    default=20,
    help="Desired quality (Constant Rate Factor) of the exported videos for --export-video. "
    "Valid range is 0-51. Typical range is 18-28. Lower means higher quality.",
)
@click.option(
    "--renderer",
    type=click.Choice(["default", "gsplat"], case_sensitive=False),
    default="default",
    help="Renderer backend: 'default' uses the artifact's trained renderer (PyTorch forward pass), "
    "'gsplat' forces GSplatRenderer.",
)
@click.option(
    "--enable-editing-actors",
    is_flag=True,
    help="Enable the actor editing API to allow modifications to dynamic actor poses.",
    default=False,
)
@click.option(
    "--demo-actor-transform",
    is_flag=True,
    help="Demo: apply a precomputed transformation to actor poses. Requires --enable-editing-actors.",
    default=False,
)
@click.option(
    "--rig-translation-offset",
    nargs=3,
    type=float,
    help="Translation offsets (tx,ty,tz) in meters in rig space to be applied to the rig prior to rendering.",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--rig-rotation-offset",
    nargs=3,
    type=float,
    help="Rotation offsets (yaw, -roll, -pitch) in degrees in rig space to be applied to the rig prior to rendering.",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--replicate-training-views/--no-replicate-training-views",
    help="If set, replicate the exact training views (intrinsics, poses, and post-processing) for rendering."
    " Overrides --calib-source to 'training-sensor-poses-calib'.",
    default=True,
)
@click.option(
    "--calib-source",
    type=click.Choice(
        [
            "training-rig-poses",
            "training-rig-poses-per-frame",
            "training-sensor-poses-nocalib",
            "training-sensor-poses-calib",
        ]
    ),
    help=(
        "Specifies the input source and to obtain the sensor-to-world pose for each frame to be rendered:\n"
        "'training-rig-poses' = Interpolate the training rig trajectory from the USDZ to obtain the rig-to-world "
        "poses at the frame start/end timestamps, then compose the camera-to-rig pose on top. \n"
        "'training-rig-poses-per-frame' - Use the already interpolated per-frame rig-to-world poses from the USDZ "
        "and compose the camera-to-rig pose on top (should give the same results as 'training-rig-poses').\n"
        "'training-sensor-poses-nocalib' - Get the unoptimized sensor-to-world training poses from the calib module "
        "that is initialized from the USDZ as part of the model.\n"
        "'training-sensor-poses-calib' - Get the optimized sensor-to-world training poses from the calib module if "
        "calib was enabled during training, otherwise fall back to the uncalibrated sensor poses."
        "If the required data is missing from the USDZ, or if the required model does not support the calib module, "
        "an error is raised, respectively.\n"
    ),
    default="training-rig-poses",
)
@click.option(
    "--frame-naming",
    type=click.Choice(["frame-end-timestamp", "contiguous-output-index"]),
    help="Naming for ordinary outputs: absolute frame-end timestamp or contiguous saved-output index. "
    "Additional --save outputs always include the resolution, source frame index, and 1-based iteration, "
    "with min/max suffixes for extrema.",
    default="contiguous-output-index",
)
@click.option(
    "--custom-rig-trajectory",
    type=str,
    help="Optional path to a custom rig trajectory JSON file. "
    "If set, its first trajectory is used for the selected rendering mode instead of the training rig trajectory. "
    "LiDAR rendering requires LiDAR calibration and frame timestamps in the JSON. "
    "Each camera in the provided JSON should have its unique sensor index set to a valid training sensor index "
    "for ISP post-processing to pick up the correct trained parameters per sensor.",
    default=None,
)
@click.option(
    "--render-bboxes",
    type=click.Choice(["none", "model", "artifact"]),
    default="none",
    help=(
        "Cuboid wireframe overlay on rendered images: "
        "none = off; "
        "model = controllable tracks from the model checkpoint (requires composite model); "
        "artifact = all annotation tracks from sequence_tracks.json in the USDZ."
    ),
)
@click.option(
    "--bbox-color",
    type=(click.IntRange(0, 255), click.IntRange(0, 255), click.IntRange(0, 255)),
    default=(0, 255, 0),
    help="RGB color (0-255) for bounding box edges when --render-bboxes is model or artifact. Default: 0 255 0 (green).",
)
@click.option(
    "--label-tracks",
    metavar="LAYER_ID",
    default=None,
    help=(
        "Overlay a labeled box around every cuboid track of the given Gaussian layer "
        "(e.g. 'dynamic_rigids'), annotated with its nurec_track_id. Watch the rendered video "
        "with this on (or feed it to an AI tool) to read off which on-screen object maps to "
        "which id, then bind those ids in an asset_edits.yaml. Drawn on EVERY product named "
        "by --output-type, not just color. Requires a GaussiansComposite model."
    ),
)
@click.option(
    "--list-tracks",
    metavar="LAYER_ID",
    default=None,
    help=(
        "Write the cuboid tracks of the given Gaussian layer (nurec_track_id, class, world "
        "centroid, distance from the render start pose; nearest first) to "
        "'layer_tracks.json' in the output dir and log them, then exit WITHOUT rendering. "
        "Use this to query a container for available ids. Mutually exclusive with --nsight."
    ),
)
@click.option(
    "--asset-edits-schedule",
    "asset_edits_schedule",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    default=None,
    help=(
        "Optional path to an aspect schedule JSON (timed aspect switches, absolute "
        "microseconds). Replays the baked asset-edit clusters at render time. The "
        "geometry manifest is read from the baked USDZ (or override it with "
        "--asset-manifest). This is the render-time SCHEDULE; the authoring YAML is a "
        "different format and goes to `nurec compile-asset-edits --asset-edits-yaml`."
    ),
)
@click.option(
    "--asset-manifest",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    default=None,
    help=(
        "Override path to the baked asset-edits geometry manifest. Defaults to the "
        "'asset_edits_manifest.json' embedded inside the baked USDZ artifact."
    ),
)
@click.option(
    "--max-pending-save-tasks",
    type=int,
    help="Max number of save tasks allowed to be pending; render loop blocks until below this.",
    default=64,
)
@click.option(
    "--save-workers",
    type=int,
    help="Number of workers to use for saving frames to disk.",
    default=16,
)
@click.option(
    "--output-type",
    "output_types",
    type=click.Choice(["color", "distance-colorized", "semantics-colorized"]),
    multiple=True,
    default=("color",),
    help=(
        "Output product(s) to render and save. Repeatable. "
        f"color: rendered color frames under {CameraOutputSubdir.COLOR.value}/<camera_id>/; "
        f"distance-colorized: per-pixel ray distance turbo colormap "
        f"under {CameraOutputSubdir.DISTANCE_COLORIZED.value}/<camera_id>/; "
        f"semantics-colorized: semantic class colors under "
        f"{CameraOutputSubdir.SEMANTICS_COLORIZED.value}/<camera_id>/. "
        "(per-pixel depth, not orthographic depth) "
        "If the loaded checkpoint + active renderer cannot produce a requested signal "
        "(e.g. semantic_logits), the run aborts; re-run without that --output-type."
    ),
)
@click.option(
    "--semantics-legend/--no-semantics-legend",
    default=True,
    help="When semantics-colorized is in --output-type, write semantics legend image (color swatch + class ID/name). "
    "Per-frame images omit the legend. Uses semantics class names and colors from the artifact when present.",
)
@click.option(
    "--colorization-max-distance",
    type=float,
    help="When distance-colorized is in --output-type, max distance (in world units) for distance colormap is set. Farther pixels are clamped.",
    default=80.0,
)
@cli.tracks_timestamp_ray_stride_cli_option()
@cli.scopedtimer_cli_options(print_func=log.info)
@click.argument("hydra-overrides", nargs=-1)
@click.pass_context
def render(
    ctx: click.Context,
    artifact_path: str,
    output_dir: str,
    height: Optional[int],
    image_scale: Optional[float],
    camera_ids: Optional[Tuple[str]],
    frame_step: int,
    frame_index: str,
    warmup_frames: int,
    iterations: int,
    report: bool,
    nsight: bool,
    lidar: bool,
    save: Optional[str],
    rolling_shutter_duration: Optional[int],
    image_format: str,
    export_video: bool,
    video_fps: float,
    video_crf: int,
    renderer: str,
    enable_editing_actors: bool,
    demo_actor_transform: bool,
    rig_translation_offset: Tuple[float, float, float],
    rig_rotation_offset: Tuple[float, float, float],
    replicate_training_views: bool,
    calib_source: str,
    frame_naming: str,
    custom_rig_trajectory: Optional[str],
    render_bboxes: str,
    bbox_color: Tuple[int, int, int],
    label_tracks: Optional[str],
    list_tracks: Optional[str],
    asset_edits_schedule: Optional[str],
    asset_manifest: Optional[str],
    max_pending_save_tasks: int,
    save_workers: int,
    output_types: Tuple[str, ...],
    semantics_legend: bool,
    colorization_max_distance: float,
    tracks_timestamp_ray_stride: int,
    hydra_overrides: Tuple[str, ...],
) -> None:
    """Render cameras or LiDAR from training poses or a custom rig trajectory.

    Rig-pose camera modes and LiDAR use the first rig trajectory; sensor-pose camera modes use stored training poses.
    Camera rendering supports optional actor editing.
    """

    try:
        parse_frame_index(frame_index)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint="--frame-index") from error

    if nsight and list_tracks is not None:
        raise click.UsageError(
            "--nsight and --list-tracks are mutually exclusive: --list-tracks exits without rendering."
        )

    # Render cameras by default; --lidar selects the LiDAR pipeline instead.
    color = not lidar
    if not color:
        # Defaults are harmless; explicitly requested camera/image controls would be ignored.
        rgb_options = (
            "camera_ids",
            "height",
            "image_scale",
            "rolling_shutter_duration",
            "image_format",
            "export_video",
            "video_fps",
            "video_crf",
            "rig_translation_offset",
            "rig_rotation_offset",
            "replicate_training_views",
            "calib_source",
            "enable_editing_actors",
            "demo_actor_transform",
            "render_bboxes",
            "bbox_color",
            "label_tracks",
            "list_tracks",
            "output_types",
            "semantics_legend",
            "colorization_max_distance",
            "asset_edits_schedule",
            "asset_manifest",
        )
        for name in rgb_options:
            if ctx.get_parameter_source(name) is click.core.ParameterSource.COMMANDLINE:
                flag = {"camera_ids": "--camera-id", "output_types": "--output-type"}.get(
                    name, "--" + name.replace("_", "-")
                )
                raise click.UsageError(f"{flag} is invalid with --lidar; omit --lidar to render camera outputs")

    if (
        color
        and custom_rig_trajectory not in [None, "None", ""]
        and calib_source in ("training-sensor-poses-calib", "training-sensor-poses-nocalib")
    ):
        raise click.UsageError(
            f"--custom-rig-trajectory cannot be used with --calib-source {calib_source}, which uses training sensor poses. "
            "Use --calib-source training-rig-poses or training-rig-poses-per-frame to render the custom trajectory."
        )

    save_modes = {mode.strip() for mode in save.split(",")} if save is not None else set()
    if not save_modes <= {"none", "min", "max", "last-iteration", "all-iterations"}:
        raise click.BadParameter(
            "Choose none alone, or comma-separated modes from min, max, last-iteration, or all-iterations.",
            param_hint="--save",
        )
    if "none" in save_modes and len(save_modes) > 1:
        raise click.BadParameter("none cannot be combined with other save modes.", param_hint="--save")
    save_outputs = "none" not in save_modes
    if not save_outputs and export_video:
        raise click.UsageError("--save none cannot be combined with --export-video because no frames are saved.")
    extrema_modes = save_modes & {"min", "max"}

    if video_fps <= 0.0:
        raise ValueError(f"--video-fps {video_fps} is not positive")
    if video_crf < 0 or video_crf > 51:
        raise ValueError(f"--video-crf {video_crf} is outside of the allowed range (0, 51)")
    if video_crf < 18 or video_crf > 28:
        log.warning(f"--video-crf {video_crf} is outside of the recommended range (18, 28).")

    if max_pending_save_tasks <= 0:
        raise ValueError(f"--max-pending-save-tasks {max_pending_save_tasks} is not greater than 0")

    if save_workers <= 0:
        raise ValueError(f"--save-workers {save_workers} is not greater than 0")

    # CLI input is authoritative: every render call uses these same options. If the loaded
    # checkpoint + active renderer can't produce a requested signal (e.g. semantic_logits),
    # render_camera_frame raises ValueError and the run aborts — the caller should re-run
    # without that --output-type instead of getting silent partial output.
    render_options = CameraRenderOptions(
        color_image="color" in output_types,
        distance_image="distance-colorized" in output_types,
        semantic_image="semantics-colorized" in output_types,
    )

    # The Nsight child and ordinary rendering both write directly into the requested directory.
    capture_output_dir = os.environ.get(NSIGHT_OUTPUT_DIR_ENV) if nsight else None
    output_dir = str(Path(capture_output_dir or output_dir).expanduser().resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    enabled_outputs = ("lidar",) if lidar else output_types
    log.info("Run output: %s. Enabled output types: %s", output_dir, ", ".join(enabled_outputs))

    if nsight and not capture_output_dir:
        # Relaunch the same CLI once under nsys, before loading a model in this process.
        try:
            run_nsight_profile([sys.executable, *sys.argv], Path(output_dir), expect_metrics=report)
        except (OSError, RuntimeError) as error:
            raise click.ClickException(str(error)) from error
        log.info("Rendering complete. Output: %s", output_dir)
        return

    # Capture, log, and save CLI settings
    collector = SettingsCollector.from_click_context(ctx, "render")
    collector.log_settings(log)
    collector.save_json(Path(output_dir) / "render_cli_args.json")

    if not Path(artifact_path).is_file():
        raise FileNotFoundError(artifact_path)

    if image_scale is not None and height is not None:
        raise click.UsageError("--height and --image-scale cannot be used together.")

    if image_scale is not None and (image_scale <= 0.0 or image_scale > 1.0):
        raise ValueError(f"Image scale {image_scale} is outside of the allowed range (0.0, 1.0]")

    if height is not None and (height <= 0):
        raise ValueError(f"Height {height} is not positive")

    # Check if translation or rotation offsets are set
    has_pose_offset = any(x != 0.0 for x in rig_translation_offset) or any(x != 0.0 for x in rig_rotation_offset)

    if replicate_training_views and color:
        # Make sure that translation and rotation offsets are zero
        if has_pose_offset:
            raise ValueError(
                "--replicate-training-views is set but --rig-translation-offset or --rig-rotation-offset are non-zero. "
                "No pose offset is allowed when replicating training views."
            )
        # Check rolling shutter duration is not overridden
        if rolling_shutter_duration is not None:
            raise ValueError(
                "--replicate-training-views is set, but --rolling-shutter-duration is overridden. "
                "No override for rolling shutter is allowed when replicating training views."
            )
        # Check actors are not edited
        if enable_editing_actors or demo_actor_transform:
            raise ValueError(
                "--replicate-training-views is set, but --enable-editing-actors or --demo-actor-transform is also set. "
                "No actor editing is allowed when replicating training views."
            )
        # Check custom trajectory is not used
        if custom_rig_trajectory not in [None, "None", ""]:
            raise ValueError(
                "--replicate-training-views is set, but --custom-rig-trajectory is also set. "
                "Custom trajectory is not allowed when replicating training views."
            )

        # Training views can only be replicated with calibrated sensor poses if calibration was enabled during training,
        # otherwise rendered frames may not necessarily align with training frames
        # (depending on the amount of pose correction achieved during training).
        # If calib was disabled during training, this will fall back to uncalibrated sensor poses.
        calib_source = "training-sensor-poses-calib"
        log.warning(f"Overriding --calib-source to {calib_source} because --replicate-training-views is set.")

    if demo_actor_transform and not enable_editing_actors:
        raise ValueError("--demo-actor-transform requires --enable-editing-actors to be set")

    # The native loader starts capture after applying CUDA device and allocator settings.
    nsight_capture = NsightCapture(Path(output_dir) if nsight else None)
    # Click closes callbacks in reverse order, including when rendering is interrupted.
    ctx.call_on_close(nsight_capture.write_manifest)
    ctx.call_on_close(nsight_capture.stop)

    log.info(f"Loading artifact from {artifact_path}")
    artifact = Artifact(Path(artifact_path))
    semantic_class_names: Optional[Dict[int, str]] = artifact.semantic_class_names
    semantic_class_colors = artifact.semantic_class_colors

    class_color_lut: Optional[np.ndarray] = None  # (lut_size, 3) uint8 visualization LUT
    semantics_lut_size = 0
    if render_options.semantic_image:
        if semantic_class_names:
            # IDs may be sparse — size the LUT to cover the largest class id (+1), not the count.
            semantics_lut_size = max(semantic_class_names) + 1
            if semantic_class_colors is not None and len(semantic_class_colors) > 0:
                if len(semantic_class_colors) < semantics_lut_size:
                    raise ValueError(
                        f"Artifact has semantic_class_names with max id {semantics_lut_size - 1} but "
                        f"only {len(semantic_class_colors)} semantic_class_colors; the color LUT must "
                        f"cover every named class id."
                    )
                log.info(
                    "Semantic class colors: %d-entry LUT from artifact (covers %d named class id(s); "
                    "unnamed ids fall back to numeric labels).",
                    semantics_lut_size,
                    len(semantic_class_names),
                )
            else:
                log.info(
                    "Semantic class colors: %d-entry LUT (no color mapping in artifact; "
                    "default hues generated for each class).",
                    semantics_lut_size,
                )
        else:
            semantics_lut_size = SEMANTICS_COLOR_LUT_DEFAULT_SIZE
        class_color_lut = build_semantics_class_colors(semantics_lut_size, semantic_class_colors)

    try:
        # Get the rig trajectories of the training clip.
        training_clip_rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)

        # Load custom rig trajectories if provided and set rendering rig trajectories accordingly
        if custom_rig_trajectory not in [None, "None", ""]:
            assert custom_rig_trajectory is not None  # Type narrowing for type checker
            log.info(f"Loading custom rig trajectories from {custom_rig_trajectory}")
            artifact.load_custom_rig_trajectories(custom_rig_trajectory)
            log.info("Using custom rig trajectories for rendering")
            rendering_rig_trajectories = RigTrajectories.from_dict(artifact.custom_rig_trajectories)
        else:
            log.info("Using training rig trajectories for rendering")
            rendering_rig_trajectories = training_clip_rig_trajectories
        if not rendering_rig_trajectories.rig_trajectories:
            raise ValueError("No rig trajectories are available for rendering.")
    except (ValueError, AssertionError, KeyError, TypeError) as error:
        detail = str(error) or "inconsistent sensor offsets, timestamps, or poses"
        raise click.ClickException(f"Invalid rig trajectory metadata: {detail}") from error
    log.info(f"Constructing a renderable model for scene id '{artifact.scene_id}' with renderer='{renderer}'")
    config_overrides_list = list(hydra_overrides)
    if renderer.lower() == "gsplat":
        config_overrides_list.append("model.renderer.name=3dgut-gsplat")
    # Probed before the (expensive) model load, and independently of --asset-edits-schedule:
    # an edit-baked artifact rendered without a schedule shows every controllable cluster in
    # its baked state with nothing driving it, which is not a useful render.
    embedded_manifest = None
    embedded_manifest_clusters = 0
    try:
        embedded_manifest = load_manifest_from_usdz(Path(artifact_path))
        embedded_manifest_clusters = 0 if embedded_manifest is None else len(embedded_manifest.clusters)
    except (OSError, zipfile.BadZipFile):
        embedded_manifest_clusters = 0  # not a readable zip: the artifact loader below gives the real error
    except (ValueError, KeyError) as exc:
        # The zip HAS a manifest member but it does not parse: rendering would silently
        # show a broken bake, so refuse before the model load.
        raise click.ClickException(
            f"{Path(artifact_path).name} embeds a corrupt asset-edits manifest: {exc}. Re-bake."
        ) from exc

    # --list-tracks/--label-tracks are the id-DISCOVERY tools for authoring a schedule
    # (for world-bind props the ids exist only in the baked artifact), so the no-schedule
    # gate must not lock them out: listing never renders, and a labeled baked-state render
    # is exactly what schedule authoring needs.
    if embedded_manifest_clusters and asset_edits_schedule is None and not (list_tracks or label_tracks):
        raise click.ClickException(
            f"{Path(artifact_path).name} embeds an asset-edits manifest with "
            f"{embedded_manifest_clusters} controllable cluster(s), but no "
            "--asset-edits-schedule was given, so every controllable cluster would render "
            "in its baked state with nothing driving it. Pass a schedule JSON, render an "
            "artifact without baked edits, or use --list-tracks/--label-tracks (exempt: "
            "they exist to discover the ids a schedule needs)."
        )
    if asset_manifest is not None and asset_edits_schedule is None:
        # The manifest names geometry but carries no timing, so on its own it cannot
        # change a single frame. Checked before the (expensive) model load.
        raise click.ClickException(
            "--asset-manifest only takes effect together with --asset-edits-schedule <schedule.json>."
        )
    # semantics-colorized reads `semantic_logits`, which is declared with sensor_type=camera
    # in configs/model/gaussians/extra_signal/semantic_logits.yaml, so it lives in the gsplat
    # camera_extra_signal channels — gated by `enable_sensor_features`. Other extra signals
    # may be configured as common (gated by `enable_extended_features`). RenderableModel
    # disables both by default for inference (CDIM=4 perf win); re-enable both on the camera
    # path when the caller wants semantic output. Caller overrides win the dotlist merge.
    if render_options.semantic_image:
        config_overrides_list.append("model.renderer.outputs.camera.enable_extended_features=True")
        config_overrides_list.append("model.renderer.outputs.camera.enable_sensor_features=True")
    # RenderableModel disables the camera depth channel by default for inference (CDIM 4 -> 3).
    # Re-enable it on the camera path when the caller asks for distance output. Caller overrides
    # win the dotlist merge.
    if render_options.distance_image:
        config_overrides_list.append("model.renderer.outputs.camera.enable_distance=True")
    renderable = RenderableModel.load_from_artifact(
        artifact,
        config_overrides=tuple(config_overrides_list),
        tracks_timestamp_ray_stride=tracks_timestamp_ray_stride,
        before_model_load=nsight_capture.start,
    )

    # Install the asset-edits runtime if requested. The resolved schedule drives
    # which baked aspects are visible per track at each frame's timestamp;
    # the geometry manifest (which Gaussian rows belong to which cluster) is produced by
    # `nurec compile-asset-edits` and embedded in the baked USDZ.
    if asset_edits_schedule is not None:
        if not isinstance(renderable._model, GaussiansComposite):
            raise click.ClickException(
                f"--asset-edits-schedule requires a GaussiansComposite model (got {renderable.model_class_name()})."
            )
        manifest = load_manifest_from_path(Path(asset_manifest)) if asset_manifest is not None else embedded_manifest
        if manifest is None:
            raise click.ClickException(
                "No asset-edits geometry manifest found. Either pass --asset-manifest, "
                "or compile the USDZ with `nurec compile-asset-edits` so "
                "'asset_edits_manifest.json' is embedded inside the artifact."
            )
        log.info(
            "Loaded asset-edits manifest with %d cluster(s) across layer(s) %s",
            len(manifest.clusters),
            manifest.layer_ids(),
        )
        # Bind the manifest to THIS checkpoint before anything can mutate a frame. The
        # same gate the gRPC server applies at install (serve._install_edit_overrides):
        # --asset-manifest will happily pair any manifest with any artifact, and every
        # mismatch writes baked opacities into unrelated gaussians rather than erroring.
        try:
            manifest.validate_against_layers(dict(renderable._model.gaussians_nodes))
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        schedule = load_schedule_from_path(Path(asset_edits_schedule))
        try:
            manifest.validate_schedule(schedule)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        log.info(
            "Loaded schedule for %d track(s) from %s",
            len(schedule.tracks),
            asset_edits_schedule,
        )
        registry = AspectScheduleRegistry()
        registry.set_schedule(schedule)
        runtime = AspectOverrideRuntime(manifest=manifest, registry=registry)
        renderable._model.set_edit_overrides(runtime)

    def render_trajectory() -> None:
        """Execute camera rendering, LiDAR rendering, or track-only listing."""
        if save_outputs and render_options.semantic_image and semantics_legend and class_color_lut is not None:
            _save_semantics_legend(
                output_dir,
                class_color_lut,
                semantic_class_names,
            )

        metrics_collector = None
        if report:
            metrics_collector = MetricsCollector(
                mode="render",
                run_id=Path(output_dir).name,
                frame_metrics=True,
                nsight=nsight,
            )
            for metric, method in (("render_time_ms", "wall_clock"), ("cuda_ms", "cuda_event")):
                metrics_collector.set_metric_metadata(
                    metric,
                    timing_method=method,
                    model=renderable.model_class_name(),
                    renderer=renderer,
                    artifact=str(Path(artifact_path).resolve()),
                    trajectory_id=0,
                    sequence_id=rendering_rig_trajectories.rig_trajectories[0].sequence_id,
                    frame_index=frame_index,
                    warmup_frames=warmup_frames,
                    iterations=iterations,
                )

        if lidar:
            try:
                render_lidar_frames(
                    renderable,
                    rendering_rig_trajectories,
                    output_dir=Path(output_dir),
                    frame_step=frame_step,
                    frame_index=frame_index,
                    warmup_frames=warmup_frames,
                    iterations=iterations,
                    save_modes=save_modes,
                    frame_naming=frame_naming,
                    metrics_collector=metrics_collector,
                    nsight_capture=nsight_capture,
                )
            except ValueError as error:
                raise click.ClickException(str(error)) from error
            if metrics_collector is not None:
                metrics_collector.write_metrics_to_yaml(output_dir)
            return

        # Build track labelers if the user wants labeled boxes on the render or just a
        # listing of tracks by id. The option value is a Gaussian layer of cuboid tracks; the
        # label drawn is the layer-local nurec_track_id that asset_edits.yaml edits bind to.
        # Each mode fails LOUD on a layer without cuboid tracks: a listing must never fall
        # through to a full (expensive) render, and a labeling render without labels is not
        # what was asked for.
        def _labeler_for(flag: str, layer_id: str) -> TrackLabeler:
            if not isinstance(renderable._model, GaussiansComposite):
                raise click.ClickException(
                    f"{flag} requires a GaussiansComposite model (got {renderable.model_class_name()})."
                )
            labeler = TrackLabeler.from_model(
                renderable._model, world_to_nurec=renderable._world_to_nurec, layer_id=layer_id
            )
            if labeler is None:
                raise click.ClickException(f"{flag}: layer {layer_id!r} has no cuboid tracks.")
            return labeler

        label_labeler = _labeler_for("--label-tracks", label_tracks) if label_tracks else None
        if list_tracks:
            labeler = (
                label_labeler
                if list_tracks == label_tracks and label_labeler
                else _labeler_for("--list-tracks", list_tracks)
            )
            # Distance reference: render start pose (same world frame as the labeled tracks).
            ego_ref = _render_start_position(rendering_rig_trajectories)
            tracks = labeler.list_tracks()
            if ego_ref is not None:
                tracks = sorted(tracks, key=lambda t: _centroid_distance(t.world_centroid, ego_ref))
            layer_json = {
                "layer_id": labeler.layer_id,
                "count": len(tracks),
                "tracks": [
                    {
                        "nurec_track_id": t.nurec_track_id,
                        "track_id_str": t.track_id_str,
                        "label_class": t.label_class,
                        "world_centroid": list(t.world_centroid),
                        "dims": list(t.dims),
                        "distance_m": (
                            None if ego_ref is None else round(_centroid_distance(t.world_centroid, ego_ref), 3)
                        ),
                    }
                    for t in tracks
                ],
            }
            order_note = "nearest first" if ego_ref is not None else "unordered; no rig start pose"
            log.info("%d track(s) in layer '%s' (%s):", len(tracks), labeler.layer_id, order_note)
            for t in tracks:
                cx, cy, cz = t.world_centroid
                dist_str = "  n/a" if ego_ref is None else f"{_centroid_distance(t.world_centroid, ego_ref):6.1f}"
                log.info(
                    "  nurec_track_id=%-3d dist=%sm class=%-24s world_centroid=(%.2f, %.2f, %.2f)",
                    t.nurec_track_id,
                    dist_str,
                    t.label_class,
                    cx,
                    cy,
                    cz,
                )
            tracks_json_path = Path(output_dir) / "layer_tracks.json"
            with open(tracks_json_path, "w") as f:
                json.dump({"layers": {labeler.layer_id: layer_json}}, f, indent=2)
            log.info("--list-tracks set: wrote %s; skipping render.", tracks_json_path)
            return

        # Get the look-up table of unique frame indices that can be used to identify training frames.
        training_clip_cameras_start_frame_indices = training_clip_rig_trajectories.rig_trajectories[
            0
        ].cameras_linear_start_frame_indices

        # Build the unique_frame_idx -> training_frame_idx mapping for per-frame post-processing (e.g. PPISP).
        # This reconstructs the same mapping used during training from the artifact's parsed config.
        training_frame_mapping: npt.NDArray[np.int32] | None = None
        if replicate_training_views:
            training_frame_mapping = build_training_frame_mapping(
                artifact.parsed_config, training_clip_rig_trajectories.rig_trajectories[0]
            )
            # Sanity check: verify the number of training frames matches the model's post-processing expectations.
            if isinstance(renderable._model, GaussiansComposite):
                pp_num_frames = next(
                    (
                        getattr(pp, "num_frames", None)
                        for pp in renderable._model.post_processings
                        if getattr(pp, "num_frames", None) is not None
                    ),
                    None,
                )
                if pp_num_frames is not None:
                    num_training_frames_from_mapping = int((training_frame_mapping >= 0).sum())
                    assert num_training_frames_from_mapping == pp_num_frames, (
                        f"Training frame mapping mismatch: reconstructed {num_training_frames_from_mapping} "
                        f"training frames from the artifact config, but the model's post-processing expects "
                        f"{pp_num_frames} training frames. This may indicate a holdout configuration mismatch."
                    )

        # Get the timestamp range of the training clip (always needed for actor tracks).
        training_clip_start_us, training_clip_end_us = get_trajectory_time_range(training_clip_rig_trajectories)
        assert training_clip_start_us <= training_clip_end_us

        # Get the timestamp range for rendering (may differ from training if using custom trajectory).
        rendering_clip_start_us, rendering_clip_end_us = get_trajectory_time_range(rendering_rig_trajectories)
        assert rendering_clip_start_us <= rendering_clip_end_us

        # Instantiate a provider of sensor intrinsics and poses for rendering individual views.
        calib_provider: SensorCalibProvider
        if calib_source == "training-rig-poses" or calib_source == "training-rig-poses-per-frame":
            interpolate_poses = calib_source == "training-rig-poses"
            calib_provider = SensorCalibProviderFromRigTrajectories(
                rendering_rig_trajectories,
                rig_translation_offset,
                rig_rotation_offset,
                interpolate_poses,
                rolling_shutter_duration,
            )
        elif calib_source == "training-sensor-poses-nocalib" or calib_source == "training-sensor-poses-calib":
            calibrated = calib_source == "training-sensor-poses-calib"
            if has_pose_offset:
                # Do not allow rig pose offsets in this mode.
                # Pose offsets are supposed to shift all sensors consistently in the rig frame but the notion of a rigid
                # rig is lost when using sensor-to-world poses directly from the calib module because
                # camera views are optimized independently without any rigidity constraint ("free poses") currently.
                raise ValueError(
                    "The rig pose offset set via --rig-translation-offset or --rig-rotation-offset is non-zero but "
                    f"the selected calib source '{calib_source}' does not support it."
                )
            calib_provider = SensorCalibProviderFromSensorTrajectories(
                camera_calibrations=training_clip_rig_trajectories.camera_calibrations,
                camera_trajectories=renderable.get_camera_trajectories(calibrated=calibrated),
                cameras_start_frame_indices=training_clip_cameras_start_frame_indices,
            )
        else:
            raise ValueError(f"Unsupported pose input type '{calib_source}'")

        # Query cameras from the scene and select an existing trajectory and camera.
        available_camera_ids = calib_provider.get_available_camera_ids()
        if len(available_camera_ids) == 0:
            raise ValueError("No cameras found in the artifact")
        log.info("Available cameras:\n    " + "\n    ".join(available_camera_ids))

        if not camera_ids:
            render_camera_ids: List[str] = available_camera_ids
            log.info(f"No --camera-id specified, rendering all {len(render_camera_ids)} available cameras")
        else:
            render_camera_ids = list(camera_ids)

        # Validate the full selection before any camera writes frames, extras, or videos.
        validate_sensor_output_names(render_camera_ids, sensor_type="camera")

        # Make sure the requested cameras exist.
        for camera_id in render_camera_ids:
            if camera_id not in available_camera_ids:
                raise ValueError(
                    f"Could not find the requested camera: '{camera_id}'. Only found {available_camera_ids}"
                )

        actor_tracks: Optional[ActorTracks] = None
        actor_tracks_to_overlay: Optional[ActorTracks] = None
        if enable_editing_actors:
            if not renderable.supports_edit_actors():
                raise ValueError(
                    "--enable-editing-actors requires a model with cuboid tracks (e.g. GaussiansComposite). "
                    f"This model ({renderable.model_class_name()}) does not support actor editing."
                )
            edit_tracks = renderable.get_actor_tracks()
            actor_tracks = edit_tracks
            log.info(f"Actor editing uses {edit_tracks.num_tracks()} track(s) from the loaded model")

        if render_bboxes == "artifact":
            log.info("Loading actor tracks from artifact for bbox overlay")
            actor_tracks_to_overlay = SceneInfo(artifact).get_actor_tracks()
            if actor_tracks_to_overlay.num_tracks() == 0:
                log.warning(
                    "Bounding box overlay with --render-bboxes artifact: no tracks in the artifact "
                    "(missing or empty sequence_tracks); no boxes will be drawn."
                )
        elif render_bboxes == "model":
            if not renderable.supports_edit_actors():
                raise ValueError(
                    "--render-bboxes model requires a model with cuboid tracks (e.g. GaussiansComposite). "
                    f"This model ({renderable.model_class_name()}) does not support actor tracks."
                )
            actor_tracks_to_overlay = actor_tracks if actor_tracks is not None else renderable.get_actor_tracks()
            log.info("Actor tracks for bbox overlay: loaded model (checkpoint) tracks")
            if actor_tracks_to_overlay.num_tracks() == 0:
                log.warning(
                    "Bounding box overlay with --render-bboxes model: no tracks in the loaded model; no boxes will be drawn."
                )

        # Reuse NuRec's metric lists and aggregates, adding frame/iteration timing summaries.
        metrics_sequence_id = None
        if report:
            _, metrics_sequence_id = split_unique_sensor_ids(
                [calib_provider.get_unique_sensor_id(cid) for cid in render_camera_ids]
            )
        cuda_timer = CudaEventTimer("nurec.render.forward")

        camera_resolutions: dict[str, list[int]] = {}
        # Each selected camera owns its warm-up, iterations and output snapshots.
        for camera_id in render_camera_ids:
            semantics_lut_clip_warned = False
            log.info(f"Selecting camera {camera_id}")
            if camera_id not in available_camera_ids:
                raise ValueError(f"Could not find the requested camera {camera_id}")

            requested_camera = calib_provider.get_camera_calibration(camera_id)

            camera_width, camera_height = requested_camera.camera_model_parameters.resolution
            if camera_width <= 0 or camera_height <= 0:
                raise ValueError(f"Camera {camera_id!r} has invalid native resolution")
            log.info(f"Camera resolution: {camera_width}x{camera_height}")
            if height is not None:
                requested_resolution = (1, height)
            elif image_scale is not None:
                requested_resolution = (int(camera_width * image_scale), int(camera_height * image_scale))
            else:
                requested_resolution = (int(camera_width), int(camera_height))
            if min(requested_resolution) <= 0 or (image_scale is not None and min(requested_resolution) < 2):
                raise ValueError(f"Camera {camera_id!r} has invalid render resolution {requested_resolution}")
            # Native calibration may adjust rounded dimensions. Preserve the original
            # request while validating and reporting the size the renderer will produce.
            output_intrinsics = transform_intrinsics_to_resolution(
                requested_camera.camera_model_parameters, requested_resolution
            )
            output_width, output_height = output_intrinsics.resolution
            output_resolution = (int(output_width), int(output_height))
            if metrics_collector is not None:
                camera_resolutions[camera_id] = list(output_resolution)
                for metric in ("render_time_ms", "cuda_ms"):
                    metrics_collector.set_metric_metadata(metric, resolutions=camera_resolutions)

            num_frames = calib_provider.get_num_camera_frames(camera_id)
            try:
                frame_indices = select_frame_indices(frame_index, num_frames, frame_step)
            except ValueError as error:
                raise click.ClickException(f"Camera {camera_id}: {error}") from error
            log.info("Camera %s: %d selected frames of %d available", camera_id, len(frame_indices), num_frames)

            output_subdirs = [
                subdir
                for subdir, enabled in (
                    (CameraOutputSubdir.COLOR, render_options.color_image),
                    (CameraOutputSubdir.DISTANCE_COLORIZED, render_options.distance_image),
                    (CameraOutputSubdir.SEMANTICS_COLORIZED, render_options.semantic_image),
                )
                if enabled
            ]
            output_dirs = {
                subdir: os.path.join(output_dir, subdir, camera_id) for subdir in output_subdirs if save_outputs
            }
            extra_output_dirs = {
                subdir: os.path.join(output_dir, f"{subdir.value}-extra", camera_id)
                for subdir in output_dirs
                if save_modes
            }
            for directory in (*output_dirs.values(), *extra_output_dirs.values()):
                os.makedirs(directory, exist_ok=True)

            # Render frames along the test trajectory.
            # Post-render (PIL + disk I/O) runs in a thread pool to unblock the next render.
            save_futures: List[Any] = []
            output_frame_timestamps: List[Dict[str, Any]] = []
            extra_frame_timestamps: List[Dict[str, Any]] = []
            best_samples: Dict[str, CameraRenderSample] = {}
            # Reuse transfer storage; prepared NumPy images own their data independently.
            rgb_download_buffer: torch.Tensor | None = None
            with (
                torch.cuda.device(renderable.device),
                nsight_capture.sensor(camera_id, sensor_dir=Path(output_dir) / "nsight" / camera_id),
                ThreadPoolExecutor(max_workers=save_workers) as executor,
            ):

                def prepare_images(
                    camera_frame: CameraFrame,
                    camera_to_world: PoseRange,
                    overlay_actors_snapshot: Optional[ActorsSnapshot],
                    copy_label: str,
                ) -> dict[CameraOutputSubdir, np.ndarray]:
                    """Prepare selected images once, retaining independent CPU arrays after download."""
                    nonlocal semantics_lut_clip_warned, rgb_download_buffer
                    images: dict[CameraOutputSubdir, np.ndarray] = {}
                    frame_start_us = camera_to_world.start_timestamp_us
                    frame_end_us = camera_to_world.end_timestamp_us
                    image = next(
                        image
                        for image in (
                            camera_frame.color_image,
                            camera_frame.distance_image,
                            camera_frame.semantic_image,
                        )
                        if image is not None
                    )
                    result_height, result_width = image.shape[:2]

                    # Compute one bbox overlay per prepared render; reuse it across enabled output types.
                    bbox_overlay = _compute_bbox_overlay(
                        render_bboxes=render_bboxes,
                        overlay_actors_snapshot=overlay_actors_snapshot,
                        camera_model_parameters=requested_camera.camera_model_parameters,
                        camera_to_world=camera_to_world,
                        image_width=result_width,
                        image_height=result_height,
                        bbox_color=bbox_color,
                    )

                    # Track labels go on EVERY selected output product, not just color:
                    # --output-type is repeatable and a caller who asks for only
                    # distance-colorized still asked for labels. The labeler returns a new
                    # array, so it composes identically on each product.
                    def _label(image_rgb: np.ndarray) -> np.ndarray:
                        """Draw the requested layer's labeled boxes onto one output product."""
                        if label_labeler is None:
                            return image_rgb
                        camera_to_world_tquat = torch.stack(
                            [
                                camera_to_world.start_pose_tquat_sensor_world,
                                camera_to_world.end_pose_tquat_sensor_world,
                            ],
                            dim=0,
                        )
                        return label_labeler.overlay(
                            image=image_rgb,
                            camera_model_parameters=requested_camera.camera_model_parameters,
                            camera_to_world=camera_to_world_tquat,
                            frame_start_us=frame_start_us,
                            frame_end_us=frame_end_us,
                            box_color_rgb=_TRACK_LABEL_COLOR,
                        )

                    if render_options.color_image and camera_frame.color_image is not None:
                        # Identify conversion/download work separately from the measured forward pass.
                        with torch.cuda.nvtx.range(copy_label):
                            color_image_u8 = (camera_frame.color_image * 255).clamp(0, 255).to(torch.uint8)
                            if rgb_download_buffer is None or rgb_download_buffer.shape != color_image_u8.shape:
                                rgb_download_buffer = torch.empty_like(color_image_u8, device="cpu", pin_memory=True)
                            # Complete the download before CPU use; keep the native independent image copy.
                            rgb_download_buffer.copy_(color_image_u8)
                            numpy_array = rgb_download_buffer.numpy().copy()
                        numpy_array = _apply_precomputed_bbox_overlay(numpy_array, bbox_overlay)
                        numpy_array = _label(numpy_array)

                        images[CameraOutputSubdir.COLOR] = numpy_array

                    # Optionally save distance (colormapped) and/or semantics (colormapped)
                    if render_options.distance_image and camera_frame.distance_image is not None:
                        with torch.cuda.nvtx.range(copy_label):
                            distance_np = camera_frame.distance_image.cpu().numpy()
                        distance_rgb = scalar2img(distance_np, vmin=0.0, vmax=colorization_max_distance, cmap="turbo")
                        distance_rgb = _apply_precomputed_bbox_overlay(distance_rgb, bbox_overlay)
                        distance_rgb = _label(distance_rgb)
                        images[CameraOutputSubdir.DISTANCE_COLORIZED] = distance_rgb
                    if render_options.semantic_image and camera_frame.semantic_image is not None:
                        assert class_color_lut is not None
                        with torch.cuda.nvtx.range(copy_label):
                            sem_np = camera_frame.semantic_image.cpu().numpy().astype(np.int64)
                        if sem_np.size and int(sem_np.max()) >= 0:
                            if int(sem_np.max()) >= semantics_lut_size and not semantics_lut_clip_warned:
                                log.warning(
                                    "Semantic class id(s) >= %d; clipping to LUT for visualization. "
                                    "Check semantic_class_names.json / semantic_class_colors.json length or raise "
                                    "SEMANTICS_COLOR_LUT_DEFAULT_SIZE when names are absent.",
                                    semantics_lut_size,
                                )
                                semantics_lut_clip_warned = True
                            # Negative ids (e.g. -1 ray-miss / unmapped sentinels) get colored black to
                            # distinguish them from class 0 — np.clip alone would collapse them onto class 0.
                            neg_mask = sem_np < 0
                            sem_safe = np.clip(sem_np, 0, semantics_lut_size - 1)
                            seg_rgb = sem2img(sem_safe, class_color_lut)
                            seg_rgb[neg_mask] = 0
                        else:
                            seg_rgb = np.zeros((*sem_np.shape, 3), dtype=np.uint8)
                        seg_rgb = _apply_precomputed_bbox_overlay(seg_rgb, bbox_overlay)
                        seg_rgb = _label(seg_rgb)
                        images[CameraOutputSubdir.SEMANTICS_COLORIZED] = seg_rgb

                    return images

                def save_frame(sample: CameraRenderSample, selection: str = "", *, extra: bool = False) -> None:
                    """Write already prepared images without converting or downloading them again."""
                    frame_start_us = sample.camera_to_world.start_timestamp_us
                    frame_end_us = sample.camera_to_world.end_timestamp_us
                    result_height, result_width = next(iter(sample.images.values())).shape[:2]

                    # Pop the oldest future if the max number of pending save tasks is reached.
                    # Ignore uneven latency for simplicity.
                    while len(save_futures) >= max_pending_save_tasks:
                        save_futures.pop(0).result()

                    timestamps = extra_frame_timestamps if extra else output_frame_timestamps
                    output_index = len(timestamps)
                    if extra:
                        suffix = f"-{selection}" if selection else ""
                        stem = (
                            f"render-{result_width}x{result_height}-frame-{sample.frame_index}"
                            f"-iteration-{sample.iteration}{suffix}"
                        )
                    else:
                        match frame_naming:
                            case "frame-end-timestamp":
                                stem = f"{frame_end_us:020d}"
                            case "contiguous-output-index":
                                stem = f"{output_index:06d}"
                            case _:
                                raise ValueError(f"Invalid frame naming scheme: {frame_naming}")
                    file_name = f"{stem}.{image_format}"

                    timestamps.append(
                        {
                            "file_name": file_name,
                            "render_frame_idx": output_index,
                            "source_frame_idx": sample.frame_index,
                            "iteration": sample.iteration,
                            "cuda_ms": sample.cuda_ms,
                            "valid": sample.valid,
                            "frame_start_timestamp_us": frame_start_us,
                            "frame_end_timestamp_us": frame_end_us,
                        }
                    )

                    directories = extra_output_dirs if extra else output_dirs
                    for subdir, image in sample.images.items():
                        image_path = os.path.join(directories[subdir], file_name)
                        log.info(f"Saving image to {image_path}")
                        save_futures.append(executor.submit(_save_rendered_frame_to_disk, image, image_path))

                for frame_idx in frame_indices:
                    # Get the camera-to-world pose of the current frame.
                    camera_to_world = calib_provider.get_camera_view_pose(camera_id, frame_idx)

                    frame_start_us = camera_to_world.start_timestamp_us
                    frame_end_us = camera_to_world.end_timestamp_us
                    assert frame_start_us >= 0 and frame_end_us >= 0

                    log.info(
                        f"Frame {frame_idx} at timestamp {frame_start_us} with {frame_end_us - frame_start_us} us exposure"
                    )

                    # Test whether the provided frame capture timestamps are within the timestamp range of the rendering clip.
                    # Actors do not render correctly beyond this timestamp range.
                    if frame_start_us < rendering_clip_start_us or frame_end_us > rendering_clip_end_us:
                        raise ValueError(
                            f"Frame capture timestamp range"
                            f"({frame_start_us}, {frame_end_us}) is outside of rendering clip"
                            f"timestamp range ({rendering_clip_start_us}, {rendering_clip_end_us})."
                        )

                    # Only update actors if enable_editing_actors is enabled
                    # Otherwise, skip to ensure all objects are rendered without modification.
                    actors_snapshot: Optional[ActorsSnapshot] = None
                    if enable_editing_actors:
                        assert actor_tracks is not None
                        actors_snapshot = actor_tracks.get_snapshot_at_frame(frame_start_us, frame_end_us)
                        log.info(f"{actors_snapshot.num_actors()} active actors")
                        if demo_actor_transform:
                            actors_snapshot = manipulate_actor_poses(actors_snapshot)

                    overlay_actors_snapshot: Optional[ActorsSnapshot] = None
                    if actor_tracks_to_overlay is not None:
                        overlay_actors_snapshot = actor_tracks_to_overlay.get_snapshot_at_frame(
                            frame_start_us, frame_end_us
                        )
                        if overlay_actors_snapshot.num_actors() == 0 and actor_tracks_to_overlay.num_tracks() > 0:
                            log.warning(
                                f"Frame {frame_idx}: --render-bboxes {render_bboxes}: "
                                f"{actor_tracks_to_overlay.num_tracks()} track(s) loaded but none overlap "
                                f"frame timestamp [{frame_start_us}, {frame_end_us}]; no boxes drawn. "
                                "Check that the render trajectory timestamps overlap the track timestamps."
                            )

                    unique_frame_idx = (
                        calib_provider.get_unique_frame_index(camera_id, frame_idx)
                        if replicate_training_views
                        else None
                    )

                    # Resolve training_frame_idx for per-frame post-processing (e.g. PPISP).
                    # The mapping array uses -1 for held-out frames; normalize to None at the API boundary.
                    training_frame_idx: int | None = None
                    if unique_frame_idx is not None and training_frame_mapping is not None:
                        if not 0 <= unique_frame_idx < len(training_frame_mapping):
                            raise ValueError(
                                f"Unique frame index {unique_frame_idx} is outside the training-frame mapping"
                            )
                        mapped = int(training_frame_mapping[unique_frame_idx])
                        if mapped >= 0:
                            training_frame_idx = mapped

                    render_frame = partial(
                        renderable.render_camera_frame,
                        camera_intrinsics=requested_camera.camera_model_parameters,
                        camera_to_world=camera_to_world,
                        resolution=requested_resolution,
                        unique_sensor_idx=requested_camera.unique_sensor_idx,
                        unique_frame_idx=unique_frame_idx,
                        training_frame_idx=training_frame_idx,
                        render_options=render_options,
                        actors_snapshot=actors_snapshot,
                        frame_start_us=frame_start_us,
                        frame_end_us=frame_end_us,
                    )

                    with (
                        torch.cuda.device(renderable.device),
                        torch.inference_mode(),
                        nsight_capture.frame(camera_id, frame_idx),
                    ):
                        # Warm the first selected frame, then measure it normally; never skip input frames.
                        if frame_idx == frame_indices.start:
                            for warmup_iteration in range(1, warmup_frames + 1):
                                # Match measured event boundaries without publishing warmup samples.
                                with cuda_timer(
                                    renderable.device,
                                    label=f"nurec.render.warmup(frame={frame_idx}, iteration={warmup_iteration})",
                                    record_sample=False,
                                ):
                                    warmup = render_frame()
                                validate_camera_output(warmup, output_resolution, render_options)
                                del warmup

                        # Group measured iterations after all warm-up GPU work has completed.
                        with torch.cuda.nvtx.range(f"nurec.render.frame(camera={camera_id}, frame={frame_idx})"):
                            for iteration in range(1, iterations + 1):
                                # Include native per-frame uploads; exports happen after the stop event.
                                if report:
                                    with cuda_timer(renderable.device):
                                        start_time = time.perf_counter()
                                        camera_frame = render_frame()
                                        elapsed_time = (time.perf_counter() - start_time) * 1000
                                else:
                                    with cuda_timer(renderable.device, record_sample=save_outputs):
                                        camera_frame = render_frame()
                                validate_camera_output(camera_frame, output_resolution, render_options)
                                if not report and not save_outputs:
                                    # Use the same timing and output lifetime as warm-up renders.
                                    del camera_frame
                                    continue
                                cuda_ms = cuda_timer.milliseconds[-1]
                                cuda_valid = cuda_timer.valid[-1]

                                if metrics_collector is not None:
                                    result_width, result_height = output_resolution
                                    log.info(
                                        f"Frame {frame_idx}, iteration {iteration}/{iterations}: {result_width}x{result_height} rendered in "
                                        f"{elapsed_time:.3f} ms (wall clock), cuda_ms={cuda_ms:.3f}"
                                    )
                                    # Both timings describe this same measured render; warm-up never reaches this point.
                                    metrics_collector.record_frame(
                                        sequence_id=metrics_sequence_id or "",
                                        sensor_id=camera_id,
                                        frame=frame_idx,
                                        iteration=iteration,
                                        render_time_ms=elapsed_time,
                                        cuda_ms=cuda_ms,
                                        metric_validity={"cuda_ms": cuda_valid},
                                    )

                                if not save_outputs:
                                    # Match warmup output lifetime and skip all export conversion/download work.
                                    del camera_frame
                                    continue

                                save_primary = iteration == iterations
                                save_extra = "all-iterations" in save_modes or (
                                    "last-iteration" in save_modes and iteration == iterations
                                )
                                improved = [
                                    mode
                                    for mode in ("min", "max")
                                    if cuda_valid
                                    and mode in extrema_modes
                                    and (
                                        mode not in best_samples
                                        or (mode == "min" and cuda_ms < best_samples[mode].cuda_ms)
                                        or (mode == "max" and cuda_ms > best_samples[mode].cuda_ms)
                                    )
                                ]
                                if not save_primary and not save_extra and not improved:
                                    continue
                                # A render selected by several save modes is converted/downloaded only once.
                                selection = ",".join((["index"] if save_primary or save_extra else []) + improved)
                                images = prepare_images(
                                    camera_frame,
                                    camera_to_world,
                                    overlay_actors_snapshot,
                                    f"nurec.render.image_copy(selection={selection}, "
                                    f"frame={frame_idx}, iteration={iteration})",
                                )
                                if not images:
                                    continue
                                sample = CameraRenderSample(
                                    frame_idx, iteration, images, camera_to_world, cuda_ms, valid=cuda_valid
                                )
                                if save_primary:
                                    save_frame(sample)
                                if save_extra:
                                    save_frame(sample, extra=True)
                                for mode in improved:
                                    best_samples[mode] = sample

                for mode in ("min", "max"):
                    if mode in best_samples:
                        save_frame(best_samples[mode], mode, extra=True)

            # Wait for all frames to be saved to disk.
            for future in save_futures:
                future.result()

            # Save the per-frame timestamps next to the frame directories that were written.
            # Same content for every enabled output type — they share a frame schedule.
            for directories, timestamps in (
                (output_dirs, output_frame_timestamps),
                (extra_output_dirs, extra_frame_timestamps),
            ):
                for ts_dir in directories.values():
                    with open(os.path.join(ts_dir, "timestamps.json"), "w") as f:
                        json.dump(timestamps, f, indent=2)

        if export_video:
            log.info("Exporting videos...")
            video_subdirs = [
                (CameraOutputSubdir.COLOR, render_options.color_image),
                (CameraOutputSubdir.DISTANCE_COLORIZED, render_options.distance_image),
                (CameraOutputSubdir.SEMANTICS_COLORIZED, render_options.semantic_image),
            ]
            for subdir, enabled in video_subdirs:
                if not enabled:
                    continue
                export_videos_and_json(
                    output_dir,
                    render_camera_ids,
                    fps=video_fps,
                    crf=video_crf,
                    image_extension=image_format.lower(),
                    frames_subdir=subdir,
                )

        if metrics_collector is not None:
            metrics_collector.write_metrics_to_yaml(output_dir)

    try:
        render_trajectory()
    except (ValueError, RuntimeError, IndexError, KeyError) as error:
        raise click.ClickException(f"First trajectory: {error}") from error
    log.info("Rendering complete. Output: %s", output_dir)
