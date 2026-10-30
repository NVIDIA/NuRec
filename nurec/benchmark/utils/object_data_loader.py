# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object data loader for object-level metrics evaluation.

This module provides the ObjectDataLoader class for loading and managing
data required for object-level metrics computation.
"""

import glob
import json
import logging
import os

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from nurec.benchmark.utils.bbox_projector import BBoxProjector
from nurec.benchmark.utils.occlusion_utils import (
    DEFAULT_PROXY_CONFIG,
    OcclusionStatistics,
    ProxyConfig,
    compute_occlusion_ratios_for_frame,
    should_skip_object,
)
from nurec.benchmark.utils.shard_data_manager import ShardDataManager


log = logging.getLogger(__name__)


def _valid_corner_counts(valid_indices, num_objects: int, corners_per_obj: int) -> np.ndarray:
    """Count how many projected corners are valid (in front + in image) per object.

    ``valid_indices`` are flat indices into the ``num_objects * corners_per_obj``
    stacked corner array, as returned by the camera model with
    ``return_valid_indices=True``. Returns an int array of length ``num_objects``
    with the per-object count of valid corners.
    """
    if isinstance(valid_indices, torch.Tensor):
        valid_indices = valid_indices.detach().cpu().numpy()
    valid_indices = np.asarray(valid_indices).astype(np.int64).ravel()
    if valid_indices.size == 0:
        return np.zeros(num_objects, dtype=np.int64)
    return np.bincount(valid_indices // corners_per_obj, minlength=num_objects)[:num_objects]


def _corners_in_front_counts(all_corners_np: List[np.ndarray], t_world_sensor: np.ndarray) -> np.ndarray:
    """Count how many of each object's corners lie in front of the camera.

    Unlike :func:`_valid_corner_counts` (which requires corners to also land
    inside the image), this is a pure depth test: a corner is "in front" when its
    camera-frame Z is positive, matching the camera model's projection validity
    (``z > 0``). Image bounds are intentionally ignored, so large/near boxes that
    overflow the frame -- and boxes that enclose the camera (which always have a
    corner in front) -- are still counted. It is used to cull only occluders that
    are fully behind the camera, which the ray test would reject anyway.

    Args:
        all_corners_np: Per-object corner arrays, each shape (8, 3), in world
            coordinates.
        t_world_sensor: 4x4 world->sensor transform (the same matrix whose inverse
            yields the camera origin used for occlusion).

    Returns:
        Int array of length ``len(all_corners_np)`` with the per-object count of
        corners in front of the camera.
    """
    if not all_corners_np:
        return np.zeros(0, dtype=np.int64)
    # sensor->world: translation is the camera origin, third column is the camera
    # +Z (forward) axis in world. dot(corner - origin, forward) equals the corner's
    # camera-frame Z, so > 0 means in front.
    sensor_to_world = np.linalg.inv(np.asarray(t_world_sensor, dtype=np.float64))
    cam_origin = sensor_to_world[:3, 3]
    forward = sensor_to_world[:3, 2]
    counts = []
    for corners in all_corners_np:
        depths = (np.asarray(corners, dtype=np.float64) - cam_origin) @ forward
        counts.append(int(np.count_nonzero(depths > 0)))
    return np.array(counts, dtype=np.int64)


def index_label_tracks(
    labels_3d: Dict[int, Dict[str, dict]],
    label_timestamps_us: np.ndarray,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, Optional[float]]], Optional[float]]:
    """Index label frames per track.

    Returns, per track, its sorted observation timestamps, their label-frame indices and its
    sampling interval (its median gap capped at the clip interval; None with a single observation);
    and the clip label interval, the median of all tracks' gaps pooled (None when no track has two
    observations).
    """
    times: Dict[str, List[int]] = {}
    frames: Dict[str, List[int]] = {}
    for frame_idx in sorted(labels_3d):
        for track_id in labels_3d[frame_idx]:
            times.setdefault(track_id, []).append(int(label_timestamps_us[frame_idx]))
            frames.setdefault(track_id, []).append(frame_idx)
    arrays = {track_id: np.asarray(ts, dtype=np.int64) for track_id, ts in times.items()}
    medians = {track_id: float(np.median(np.diff(ts))) for track_id, ts in arrays.items() if ts.size >= 2}
    gaps = [np.diff(ts) for ts in arrays.values() if ts.size >= 2]
    clip_interval_us = float(np.median(np.concatenate(gaps))) if gaps else None
    # A track's own gaps overstate its cadence when it misses frames, so its interval is capped at the
    # clip's: a missing observation never widens the window a track is accepted in.
    tracks = {
        track_id: (
            ts,
            np.asarray(frames[track_id], dtype=np.int64),
            min(medians[track_id], clip_interval_us) if track_id in medians and clip_interval_us else None,
        )
        for track_id, ts in arrays.items()
    }
    return tracks, clip_interval_us


def select_track_labels(
    tracks: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[float]]],
    labels_3d: Dict[int, Dict[str, dict]],
    query_timestamp_us: int,
    fallback_interval_us: float,
) -> Dict[str, dict]:
    """Each track's observation standing for ``query_timestamp_us``, keyed by track id.

    Labels may carry one reference time per frame or one per observation, so tracks are matched
    individually: each track's nearest observation is used when it lies within half the track's
    interval, or when the query falls between two consecutive observations (at most 1.5 intervals
    apart, so jitter is tolerated but a missed sample is not bridged). Single-observation tracks
    use ``fallback_interval_us``.
    """
    boxes = {}
    for track_id, (ts, frame_idxs, interval_us) in tracks.items():
        interval_us = fallback_interval_us if interval_us is None else interval_us
        j = int(np.searchsorted(ts, query_timestamp_us))
        if 0 < j < ts.size and ts[j] != query_timestamp_us:
            i = j if ts[j] - query_timestamp_us <= query_timestamp_us - ts[j - 1] else j - 1
            consecutive = ts[j] - ts[j - 1] <= 1.5 * interval_us
        else:
            i, consecutive = min(j, ts.size - 1), False
        if not consecutive and abs(int(ts[i]) - query_timestamp_us) > interval_us / 2:
            continue
        boxes[track_id] = labels_3d[int(frame_idxs[i])][track_id]
    return boxes


def load_timestamps_file(filepath: str) -> List[int]:
    """Load timestamps from file (supports 'frame_idx timestamp' or 'timestamp' format).

    Args:
        filepath: Path to timestamps file.

    Returns:
        List of timestamps.
    """
    timestamps: List[int] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts:
                    try:
                        timestamps.append(int(parts[-1]))
                    except ValueError:
                        log.warning("Skipping invalid timestamp: %s", line.strip())
    except OSError as e:
        log.warning("Could not read timestamps file %s: %s", filepath, e)
    return timestamps


def normalize_track_id(track_id: Any) -> str:
    """Normalize track ID to a consistent string representation.

    Handles multiple formats:
    - Numeric: 35382.0 -> "35382", 35382 -> "35382"
    - String floats: "35382.0" -> "35382"
    - Autolabel format: "255@scene:obstacles:..." -> "255"
    - Other strings: kept as-is

    Args:
        track_id: Track ID in any format (float, int, str).

    Returns:
        Normalized string representation (integer if possible).
    """
    # Handle numeric types
    if isinstance(track_id, float) and track_id.is_integer():
        return str(int(track_id))
    if isinstance(track_id, (int, float)):
        return str(track_id)

    # Convert to string for pattern matching
    str_id = str(track_id)

    # Handle string floats like "35518.0" -> "35518"
    if str_id.endswith(".0"):
        try:
            return str(int(float(str_id)))
        except ValueError:
            pass

    # Handle "255@scene:obstacles:autolabels:v2:" -> "255"
    if "@" in str_id:
        prefix = str_id.split("@")[0]
        if prefix.isdigit():
            return prefix

    return str_id


def load_camera_offset_json(filepath: str) -> Tuple[float, float, float]:
    """Load camera offset from NDAS JSON file.

    Args:
        filepath: Path to JSON file with keys tx_m, ty_m, tz_m.

    Returns:
        Tuple of (x_offset, y_offset, z_offset) in meters.

    Raises:
        ValueError: On file/parse/key errors.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support nested structure (e.g., data["offset"]) or flat
        if "offset" in data:
            data = data["offset"]
        # Require keys to exist
        return (float(data["tx_m"]), float(data["ty_m"]), float(data["tz_m"]))
    except (OSError, json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to load camera offset from {filepath}: {e}") from e


class ObjectDataLoader:
    """Loads and manages data for object-level metrics evaluation.

    This class is responsible for:
    1. Loading shard data (3D labels, camera/LiDAR sensors)
    2. Loading rendered video frames
    3. Pre-loading GT frames from shards
    4. Extracting object crops from frames
    5. Providing data access methods for metric computation

    It does NOT compute metrics - that's handled by MetricManager.
    """

    def __init__(
        self,
        shard_pattern: str,
        camera_id: str,
        device: str = "cuda",
        crop_padding_ratio: float = 0.0,
        min_bbox_size: int = 50,
        occlusion_filter_threshold: Optional[float] = None,
        proxy_config: ProxyConfig = DEFAULT_PROXY_CONFIG,
    ):
        """Initialize object data loader.

        Per-object occlusion ratios (class-aware shape proxy) are always computed
        from the GT and novel camera viewpoints and attached to each object's crop
        record.

        Args:
            shard_pattern: Glob pattern for V3 shard files, or path to a NCore dataset
                meta `.json` file (V4)
            camera_id: Camera ID to process
            device: Compute device ('cuda' or 'cpu')
            crop_padding_ratio: Padding ratio around bounding boxes
            min_bbox_size: Minimum bbox size in pixels
            occlusion_filter_threshold: Optional filtering threshold. When set,
            objects whose occlusion ratio exceeds this value in either view are
            flagged as filtered so they are excluded from per-track/per-class
            aggregation (they are still recorded in the detailed metrics). When
            None, occlusion is annotated but never filtered.
            proxy_config: Tunable geometry parameters for the class-aware
            occlusion proxies.
        """
        self.device = device
        self.crop_padding_ratio = crop_padding_ratio
        self.min_bbox_size = min_bbox_size
        self.camera_id = camera_id
        self.occlusion_filter_threshold = occlusion_filter_threshold
        self.proxy_config = proxy_config

        # Data containers
        self.shard_manager: Optional[ShardDataManager] = None
        self.labels_3d: Optional[Dict] = None
        self._label_tracks: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[float]]] = {}
        self._label_interval_us: Optional[float] = None
        self._rendered_video_path: Optional[str] = None
        self._num_frames: int = 0
        self._bbox_projector: Optional[BBoxProjector] = None
        self._x_offset: float = 0.0
        self._y_offset: float = 0.0
        self._z_offset: float = 0.0
        self._max_frames: Optional[int] = None
        self._gt_subsample_frames: int = 1
        self._rendered_timestamps: Optional[List[int]] = None
        self._video_cap: Optional[cv2.VideoCapture] = None
        self._occlusion_stats: OcclusionStatistics = OcclusionStatistics()

        # Load shard data
        log.info("Initializing ObjectDataLoader")
        log.info("  Camera: %s", camera_id)
        log.info("  Crop padding ratio: %f", crop_padding_ratio)
        log.info("  Min bbox size: %d", min_bbox_size)
        log.info("  Occlusion filter threshold: %s", occlusion_filter_threshold)

        self._load_shard(shard_pattern)

    @property
    def occlusion_stats(self) -> OcclusionStatistics:
        """Aggregate occlusion statistics collected across processed frames."""
        return self._occlusion_stats

    def _load_shard(self, shard_pattern: str) -> None:
        """Load shard data and compile 3D labels.

        Args:
            shard_pattern: Glob pattern for V3 shard files, or path to a NCore dataset
                meta `.json` file (V4).
        """
        log.info("Loading shard data...")
        log.info("  Shard source: %s", shard_pattern)

        if not shard_pattern.endswith(".json"):
            shard_files = glob.glob(shard_pattern)
            if not shard_files:
                raise ValueError(f"No shard files found: {shard_pattern}")

        # Initialize shard manager
        self.shard_manager = ShardDataManager(shard_pattern, self.camera_id)

        # Compile 3D labels
        log.info("Compiling 3D labels...")
        self.labels_3d = self.shard_manager.compile_3d_labels()
        log.info("  Compiled labels for %d frames", len(self.labels_3d))
        self._label_tracks, self._label_interval_us = index_label_tracks(
            self.labels_3d, np.asarray(self.shard_manager.label_timestamps_us, dtype=np.int64)
        )

        # Create bbox projector
        self._bbox_projector = BBoxProjector(
            self.shard_manager.camera_model,
            self.shard_manager.camera_sensor,
        )

        log.info("Shard data loaded successfully")

    def load_video(
        self,
        rendered_video_path: str,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        max_frames: Optional[int] = None,
        gt_subsample_frames: int = 1,
        rendered_timestamps_file: Optional[str] = None,
    ) -> None:
        """Load rendered video and GT frames.

        Args:
            rendered_video_path: Path to rendered video file.
            x_offset: Camera X offset in meters.
            y_offset: Camera Y offset in meters.
            z_offset: Camera Z offset in meters.
            max_frames: Maximum frames to process (None = all).
            gt_subsample_frames: GT frame stride for alignment. Ignored if
                rendered_timestamps_file is provided.
            rendered_timestamps_file: Optional path to timestamps file containing frame indices and timestamps.
                When provided, enables timestamp-based frame matching instead of gt_subsample_frames.
        """
        if not rendered_video_path or not os.path.isfile(rendered_video_path):
            raise ValueError(f"Invalid video path: {rendered_video_path}")

        log.info("Loading video metadata...")
        log.info("  Video: %s", rendered_video_path)

        # Open video to get metadata (don't load frames yet)
        cap = cv2.VideoCapture(rendered_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {rendered_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Determine number of frames to process
        self._num_frames = min(total_frames, max_frames) if max_frames else total_frames

        log.info("  Video info: %d frames @ %.1f FPS, %dx%d", total_frames, fps, width, height)
        log.info("  Will process: %d frames", self._num_frames)

        # Store parameters (no pre-loading!)
        self._rendered_video_path = rendered_video_path
        self._x_offset = x_offset
        self._y_offset = y_offset
        self._z_offset = z_offset
        self._max_frames = max_frames
        self._gt_subsample_frames = gt_subsample_frames

        # Load timestamps from file if provided
        if rendered_timestamps_file is not None:
            loaded_timestamps = load_timestamps_file(rendered_timestamps_file)

            if loaded_timestamps:
                self._rendered_timestamps = loaded_timestamps
                log.info("  Matching rendered frames to GT using %d timestamps", len(loaded_timestamps))
            else:
                self._rendered_timestamps = None
                log.warning(
                    "  Timestamps file empty/invalid, using GT subsample matching (stride=%d)", gt_subsample_frames
                )
        else:
            self._rendered_timestamps = None
            log.info("  Matching rendered frames to GT using subsample (stride=%d)", gt_subsample_frames)

        log.info("Video metadata loaded (frames will be loaded on-demand)")
        log.info("  Offsets: X=%.2fm, Y=%.2fm, Z=%.2fm", x_offset, y_offset, z_offset)

    def _load_rendered_frame(self, frame_idx: int) -> np.ndarray:
        """Load a single rendered frame on-demand.

        Args:
            frame_idx: Frame index to load

        Returns:
            Frame as numpy array (RGB, uint8)
        """
        if not self._rendered_video_path:
            raise ValueError("No video path set")

        if frame_idx >= self._num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range (0-{self._num_frames - 1})")

        # Reuse VideoCapture for efficiency
        if self._video_cap is None or not self._video_cap.isOpened():
            self._video_cap = cv2.VideoCapture(self._rendered_video_path)

        assert self._video_cap is not None  # For type checker
        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._video_cap.read()

        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        """Release video capture resources."""
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

    @property
    def num_frames(self) -> int:
        """Get number of rendered frames."""
        return self._num_frames

    def get_frame_crops_with_metadata(self, rendered_video_idx: int) -> Optional[Dict[str, Any]]:
        """Extract object crops from a frame with full metadata.

        Args:
            rendered_video_idx: Rendered video frame index.

        Returns:
            Dictionary containing crops and metadata, or None if no valid crops.
        """
        if not self._rendered_video_path or self._bbox_projector is None:
            raise ValueError("Data not loaded. Call load_video() first.")

        assert self.shard_manager is not None
        assert self.labels_3d is not None

        camera_timestamps = self._bbox_projector.camera_sensor.get_frames_timestamps_us()
        num_camera_frames = len(self._bbox_projector.camera_sensor.get_frame_index_range())

        # GT camera frames are bounded to the clip subsection declared by the dataset meta
        # (mirrors the label-side bound in ShardDataManager): index-based matching starts at
        # the first in-clip frame, and timestamp-based matching only considers in-clip frames.
        camera_range_start, camera_range_stop = self.shard_manager.camera_frame_index_range
        camera_range_stop = min(camera_range_stop, num_camera_frames)
        camera_ts_in_range = np.asarray(camera_timestamps[camera_range_start:camera_range_stop], dtype=np.int64)

        # Determine GT camera frame index based on matching mode
        if self._rendered_timestamps is not None:
            # Timestamp-based matching: find closest GT camera frame
            if rendered_video_idx >= len(self._rendered_timestamps):
                log.warning(
                    "Frame %d: No timestamp available (only %d)", rendered_video_idx, len(self._rendered_timestamps)
                )
                return None

            rendered_timestamp_us = self._rendered_timestamps[rendered_video_idx]

            # Find closest camera frame by timestamp using vectorized numpy ops
            if camera_ts_in_range.size == 0:
                gt_camera_idx = None
            else:
                # int64: sensor timestamps are uint64; unsigned arithmetic would wrap for
                # GT frames earlier than the rendered timestamp and break the argmin.
                camera_ts_array = camera_ts_in_range
                gt_camera_idx = camera_range_start + int(np.argmin(np.abs(camera_ts_array - rendered_timestamp_us)))

                # argmin always snaps to *some* frame, even when the rendered
                # timestamp falls well outside the GT range (clock-domain bug,
                # off-by-N, render extends past GT). Drop matches whose gap
                # exceeds one camera inter-frame interval.
                if camera_ts_array.size >= 2:
                    median_dt_us = float(np.median(np.diff(camera_ts_array)))
                    if median_dt_us <= 0:
                        raise ValueError(
                            f"GT camera timestamps are degenerate (median inter-frame interval "
                            f"{median_dt_us:.1f} us <= 0); timestamp-based matching is unreliable."
                        )
                    time_diff_us = abs(int(camera_ts_array[gt_camera_idx - camera_range_start]) - rendered_timestamp_us)
                    if time_diff_us > median_dt_us:
                        log.info(
                            "Frame %d: skipping - timestamp gap %.1fms exceeds %.1fms (one inter-frame interval)",
                            rendered_video_idx,
                            time_diff_us / 1000.0,
                            median_dt_us / 1000.0,
                        )
                        return None

            if gt_camera_idx is None:
                log.warning(
                    "Frame %d: No matching GT frame for timestamp %d", rendered_video_idx, rendered_timestamp_us
                )
                return None
        else:
            # Subsample-based matching (original logic), offset to the first in-clip frame
            gt_camera_idx = camera_range_start + rendered_video_idx * self._gt_subsample_frames

            if gt_camera_idx >= camera_range_stop:
                log.info(
                    "Frame %d: GT frame %d out of clip range [%d, %d)",
                    rendered_video_idx,
                    gt_camera_idx,
                    camera_range_start,
                    camera_range_stop,
                )
                return None

        # Get GT camera timestamp for label matching
        gt_camera_timestamp_us = camera_timestamps[gt_camera_idx]

        # A track with a single observation, in a clip where no track has two, is bounded by one
        # camera interval on either side.
        camera_median_dt_us = int(np.median(np.diff(camera_ts_in_range))) if camera_ts_in_range.size >= 2 else 0
        fallback_us = self._label_interval_us if self._label_interval_us is not None else 2.0 * camera_median_dt_us
        frame_boxes = select_track_labels(self._label_tracks, self.labels_3d, int(gt_camera_timestamp_us), fallback_us)
        if not frame_boxes:
            log.info("Frame %d: No labelled track within its label interval", rendered_video_idx)
            return None

        # Use the GT camera frame index we calculated
        camera_frame_idx = gt_camera_idx

        # Load frames on-demand (memory efficient!)
        try:
            rendered_frame = self._load_rendered_frame(rendered_video_idx)
            # Load GT frame directly from camera sensor
            gt_frame = self._bbox_projector.camera_sensor.get_frame_image_array(gt_camera_idx)
        except RuntimeError as e:
            log.warning("Frame %d: Could not load frames: %s", rendered_video_idx, e)
            return None

        # Get resolution info
        video_height, video_width = rendered_frame.shape[:2]
        gt_height, gt_width = gt_frame.shape[:2]
        scale_x = video_width / gt_width
        scale_y = video_height / gt_height

        # Get rendered timestamp if available (for detailed metrics)
        rendered_timestamp: int | None = None
        if self._rendered_timestamps is not None:
            rendered_timestamp = self._rendered_timestamps[rendered_video_idx]

        # Extract crops
        crops_data: Dict[str, Any] = {
            "pred_crops": [],
            "gt_crops": [],
            "track_ids": [],
            "class_names": [],
            "bboxes_gt": [],
            "bboxes_rendered": [],
            "occlusion_gt": [],
            "occlusion_ren": [],
            "camera_inside": [],
            "occlusion_filtered": [],
            "camera_frame_idx": camera_frame_idx,
            "rendered_video_idx": rendered_video_idx,
            "rendered_timestamp": rendered_timestamp,
        }

        # Batch projection: collect all object corners for vectorized processing
        track_ids = []
        class_names = []
        all_corners = []

        for track_id, box_data in frame_boxes.items():
            # Normalize track_id to consistent string format
            track_ids.append(normalize_track_id(track_id))
            class_names.append(box_data.get("class", "unknown"))
            corners = box_data["vertices"]
            if isinstance(corners, np.ndarray):
                corners = torch.from_numpy(corners).float()
            all_corners.append(corners)

        if not all_corners:
            return None

        # Stack all corners into single tensor [num_objects * 8, 3]
        all_corners_tensor = torch.cat(all_corners, dim=0)

        # Get camera transformations (same for all objects).
        # With V4 NCore the poses are resolved via interpolation with strict
        # bounds. At a clip boundary the first/last in-clip frame's shutter
        # START/END timepoint can fall just outside the ego-pose spline domain,
        # which ncore surfaces as a ValueError from scipy interp1d. Skip that
        # single frame instead of letting the exception abort the whole clip.
        try:
            t_world_sensor_original_start, t_world_sensor_original_end = (
                self._bbox_projector.get_shifted_camera_transformations(
                    camera_frame_idx,
                    0.0,  # No offset for GT
                    0.0,
                    0.0,
                )
            )

            t_world_sensor_shifted_start, t_world_sensor_shifted_end = (
                self._bbox_projector.get_shifted_camera_transformations(
                    camera_frame_idx,
                    self._x_offset,
                    self._y_offset,
                    self._z_offset,
                )
            )
        except ValueError as e:
            log.info(
                "Frame %d: skipping - camera pose out of interpolation range (GT frame %d): %s",
                rendered_video_idx,
                camera_frame_idx,
                e,
            )
            return None

        # Batch project all corners at once (2 calls vs 2*N calls). Done before
        # occlusion so the projection visibility can drive frustum culling below.
        proj_original_batch = self._bbox_projector.camera_model.world_points_to_image_points_shutter_pose(
            all_corners_tensor,
            t_world_sensor_original_start,
            t_world_sensor_original_end,
            return_valid_indices=True,
            return_all_projections=True,
        )

        proj_shifted_batch = self._bbox_projector.camera_model.world_points_to_image_points_shutter_pose(
            all_corners_tensor,
            t_world_sensor_shifted_start,
            t_world_sensor_shifted_end,
            return_valid_indices=True,
            return_all_projections=True,
        )

        num_corners_per_obj = 8

        # Compute per-object occlusion ratios for the whole frame once
        # (class-aware shape proxy). Ratios are aligned by object index with
        # track_ids/class_names/all_corners.
        # Occlusion is always computed for object-level metrics.
        camera_origin_gt = np.linalg.inv(t_world_sensor_original_start.numpy())[:3, 3]
        camera_origin_novel = np.linalg.inv(t_world_sensor_shifted_start.numpy())[:3, 3]
        all_corners_np = [c.detach().cpu().numpy().astype(np.float64) for c in all_corners]

        # Occlusion cost grows with the object count, and much of a frame's labels
        # sit behind or beside the camera. Two masks keep that cost off objects
        # that cannot matter.
        # target_mask must repeat the visibility gate the crop loop applies below:
        # an object this mask skips but the loop keeps is recorded with a
        # fabricated zero. occluder_mask only drops objects fully behind the
        # camera, which the ray test rejects anyway, so it cannot change a ratio
        # and stays looser on purpose.
        orig_counts = _valid_corner_counts(proj_original_batch.valid_indices, len(all_corners_np), num_corners_per_obj)
        shifted_counts = _valid_corner_counts(
            proj_shifted_batch.valid_indices, len(all_corners_np), num_corners_per_obj
        )
        target_mask = ((orig_counts >= 4) & (shifted_counts >= 4)).tolist()
        front_orig = _corners_in_front_counts(all_corners_np, t_world_sensor_original_start.numpy())
        front_shifted = _corners_in_front_counts(all_corners_np, t_world_sensor_shifted_start.numpy())
        occluder_mask = ((front_orig >= 1) | (front_shifted >= 1)).tolist()

        occlusion_gt_list, occlusion_ren_list, camera_inside_list = compute_occlusion_ratios_for_frame(
            all_corners_np,
            camera_origin_gt,
            camera_origin_novel,
            classes=class_names,
            config=self.proxy_config,
            target_mask=target_mask,
            occluder_mask=occluder_mask,
        )

        # Extract crops for each object using batched projection results

        for obj_idx, (track_id, class_name) in enumerate(zip(track_ids, class_names)):
            try:
                # Extract this object's corners from batch
                start_idx = obj_idx * num_corners_per_obj
                end_idx = start_idx + num_corners_per_obj

                proj_original_points = proj_original_batch.image_points[start_idx:end_idx]
                proj_original_valid = proj_original_batch.valid_indices
                orig_valid_mask = (proj_original_valid >= start_idx) & (proj_original_valid < end_idx)
                orig_sum = orig_valid_mask.sum().item()

                proj_shifted_points = proj_shifted_batch.image_points[start_idx:end_idx]
                proj_shifted_valid = proj_shifted_batch.valid_indices
                shifted_valid_mask = (proj_shifted_valid >= start_idx) & (proj_shifted_valid < end_idx)
                shifted_sum = shifted_valid_mask.sum().item()

                # Require at least 4 visible corners in both views
                if orig_sum < 4 or shifted_sum < 4:
                    continue

                # Get 2D bboxes from projected corners
                proj_original_points_np = proj_original_points.numpy()
                bbox_original = self._bbox_projector.get_2d_bbox_from_projection(proj_original_points_np)

                proj_shifted_points_np = proj_shifted_points.numpy()

                # Map projections into the rendered frame, whose resolution may differ from the camera's.
                proj_shifted_points_scaled = proj_shifted_points_np.copy()
                proj_shifted_points_scaled[:, 0] *= scale_x
                proj_shifted_points_scaled[:, 1] *= scale_y
                bbox_shifted = self._bbox_projector.get_2d_bbox_from_projection(proj_shifted_points_scaled)

                # Extract object crops
                gt_crop = self._bbox_projector.crop_object(
                    gt_frame,
                    bbox_original,
                    min_size=self.min_bbox_size,
                )

                if gt_crop is None or gt_crop.size == 0:
                    continue

                # Scale min_size for rendered video
                scaled_min_size = self._bbox_projector.compute_scaled_min_size(
                    reference_shape=gt_frame.shape,
                    target_shape=rendered_frame.shape,
                    reference_min_size=self.min_bbox_size,
                )

                # Crop rendered object using bbox_shifted
                rendered_crop = self._bbox_projector.crop_object(
                    rendered_frame,
                    bbox_shifted,
                    min_size=scaled_min_size,
                )

                if rendered_crop is None or rendered_crop.size == 0:
                    continue

                # Occlusion ratios for this object, always computed for object-level metrics.
                occ_gt = float(occlusion_gt_list[obj_idx])
                occ_novel = float(occlusion_ren_list[obj_idx])
                camera_inside = bool(camera_inside_list[obj_idx])
                occlusion_filtered = False

                # Filtering is applied only when an occlusion threshold is
                # explicitly set. Objects exceeding it in either view are
                # flagged as filtered: they are still recorded (with occlusion
                # + camera-inside info) but excluded from per-track/per-class
                # aggregation downstream. The crop is NOT dropped here. When no
                # threshold is set, occlusion is annotated but never filtered.
                if self.occlusion_filter_threshold is not None:
                    skip, reason = should_skip_object(
                        occ_gt,
                        occ_novel,
                        occlusion_threshold=self.occlusion_filter_threshold,
                    )
                    occlusion_filtered = skip
                    self._occlusion_stats.record_object(occ_gt, occ_novel, skip, reason)
                    if skip:
                        log.debug(
                            "Filtering track %s at frame %d (%s): occ_gt=%.3f occ_novel=%.3f",
                            track_id,
                            rendered_video_idx,
                            reason,
                            occ_gt,
                            occ_novel,
                        )
                else:
                    self._occlusion_stats.record_object(occ_gt, occ_novel, False, "")

                # Store crops and metadata
                crops_data["pred_crops"].append(rendered_crop)
                crops_data["gt_crops"].append(gt_crop)
                crops_data["track_ids"].append(track_id)
                crops_data["class_names"].append(class_name)
                crops_data["bboxes_gt"].append(bbox_original)  # bbox from GT projection
                crops_data["bboxes_rendered"].append(bbox_shifted)  # bbox from rendered projection
                crops_data["occlusion_gt"].append(occ_gt)
                crops_data["occlusion_ren"].append(occ_novel)
                crops_data["camera_inside"].append(camera_inside)
                crops_data["occlusion_filtered"].append(occlusion_filtered)

            except (ValueError, RuntimeError) as e:
                log.debug("Failed to process box for track %s at frame %d: %s", track_id, rendered_video_idx, e)
                continue

        if not crops_data["pred_crops"]:
            return None

        return crops_data
