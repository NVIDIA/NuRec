# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Generator, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image as PILImage
import point_cloud_utils as pcu
import torch

from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError, cKDTree
from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore[import-not-found, import-untyped]

from nurec.config.ground_mesh import GroundMeshExtensionToCuboidsConfig
from nurec.datasets.ncore import NCOREDataSource
from nurec.datasets.tracks import CuboidTracks, Tracks
from nurec.utils.cuboid import cuboid_base_grid_local
from nurec.utils.geometry import Plane, PlaneDetectorRansac, tquat_to_se3_matrix
from nurec.utils.io.ply import save_ply
from nurec.utils.misc import unpack_optional
from nurec.utils.types import FrameConversion, PointCloud, RayFlags


logger = logging.getLogger(__name__)

GROUND_MESH_RANDOM_SEED = 123  # Seed used for deterministic results throughout the module
_NEUTRAL_ROAD_COLOR = np.array([128, 128, 128], dtype=np.uint8)


def _fill_invalid_road_colors(
    points: NDArray[np.float32],
    colors: NDArray[np.uint8],
    color_valid_mask: NDArray[np.bool_],
    road_mask: NDArray[np.bool_],
    *,
    neighbor_count: int = 8,
    max_distance_m: float = 2.0,
    max_height_delta_m: float = 0.35,
) -> NDArray[np.uint8]:
    """Fill colorless road returns without treating masked RGB as evidence.

    Road geometry is selected independently from camera color validity. For a
    road return hidden by an ego/camera mask, use the robust median of nearby
    directly colored road returns. Three-dimensional distance plus a height
    gate prevents colors crossing onto a nearby overpass or other surface. A
    directly observed road-color medoid is the conservative fallback for a
    donor-free local patch. If the entire road is masked, use an explicit
    neutral mid-gray fallback and warn rather than treating zero placeholders
    as black color evidence.
    """

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    color_valid_mask = np.asarray(color_valid_mask, dtype=bool)
    road_mask = np.asarray(road_mask, dtype=bool)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected matching (N, 3) points/colors, got {points.shape} and {colors.shape}")
    if color_valid_mask.shape != (len(points),) or road_mask.shape != (len(points),):
        raise ValueError("color_valid_mask and road_mask must have shape (N,)")
    if neighbor_count <= 0 or max_distance_m <= 0.0 or max_height_delta_m <= 0.0:
        raise ValueError("Color-fill neighbor count and distance limits must be positive")

    missing_mask = road_mask & ~color_valid_mask
    if not np.any(missing_mask):
        return colors.copy()

    donor_mask = road_mask & color_valid_mask
    if not np.any(donor_mask):
        logger.warning(
            "Cannot fill %d masked road colors because no directly observed road colors are available; "
            "using neutral mid-gray",
            int(np.sum(missing_mask)),
        )
        result = colors.copy()
        result[missing_mask] = _NEUTRAL_ROAD_COLOR
        return result

    donor_points = points[donor_mask]
    donor_colors = colors[donor_mask]
    missing_points = points[missing_mask]
    k = min(neighbor_count, len(donor_points))
    distances, neighbor_indices = cKDTree(donor_points).query(
        missing_points,
        k=k,
        distance_upper_bound=max_distance_m,
        workers=-1,
    )
    if k == 1:
        distances = distances[:, None]
        neighbor_indices = neighbor_indices[:, None]

    neighbor_exists = np.isfinite(distances) & (neighbor_indices < len(donor_points))
    safe_indices = np.minimum(neighbor_indices, len(donor_points) - 1)
    candidate_points = donor_points[safe_indices]
    neighbor_exists &= np.abs(candidate_points[:, :, 2] - missing_points[:, None, 2]) <= max_height_delta_m

    median_color = np.median(donor_colors.astype(np.float32), axis=0)
    medoid_index = int(np.argmin(np.sum((donor_colors.astype(np.float32) - median_color[None, :]) ** 2, axis=1)))
    global_road_medoid = donor_colors[medoid_index].astype(np.float32)
    candidate_colors = donor_colors[safe_indices].astype(np.float32)
    no_local_donor = ~np.any(neighbor_exists, axis=1)
    if np.any(no_local_donor):
        candidate_colors[no_local_donor, 0] = global_road_medoid
        neighbor_exists[no_local_donor, 0] = True
    candidate_colors[~neighbor_exists] = np.nan
    filled_colors = np.nanmedian(candidate_colors, axis=1)

    result = colors.copy()
    result[missing_mask] = np.clip(np.rint(filled_colors), 0, 255).astype(np.uint8)
    logger.info(
        "Road color validity: %d direct, %d local fills, %d observed-medoid fallbacks",
        int(np.sum(donor_mask)),
        int(np.sum(missing_mask) - np.sum(no_local_donor)),
        int(np.sum(no_local_donor)),
    )
    if np.any(no_local_donor):
        logger.warning(
            "%d masked road returns had no directly colored donor within %.2f m and %.2f m height; "
            "using the observed road-color medoid",
            int(np.sum(no_local_donor)),
            max_distance_m,
            max_height_delta_m,
        )
    return result


_ground_mesh_extension_cache: dict[str, Optional[Any]] = {}


class GroundMeshReconstructionSettings(BaseModel):
    """Pydantic parameters that determine a reconstructed ground mesh."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_frame: int = Field(default=1, gt=0)
    lidar_ids: Optional[tuple[str, ...]] = None
    camera_ids: Optional[tuple[str, ...]] = None
    valid_points_only: bool = False
    non_dynamic_points_only: bool = True
    min_ray_length: float = Field(default=0.1, gt=0.0)
    ground_compat_max_distance: float = Field(default=0.5, gt=0.0)
    ground_compat_max_angle_deg: float = Field(default=60.0, gt=0.0, le=90.0)
    z_offset: float = 0.0
    num_plane_hypotheses: int = Field(default=100, gt=0)
    plane_max_distance: float = Field(default=0.3, gt=0.0)
    plane_max_angle_deg: float = Field(default=30.0, gt=0.0, le=90.0)
    plane_min_eval_points: int = Field(default=1000, gt=0)
    plane_max_eval_points: int = Field(default=10000, gt=0)
    plane_min_inliers: int = Field(default=1000, gt=0)
    enable_plane_extension: bool = False
    voxel_size: float = Field(default=0.1, gt=0.0)
    min_points_per_voxel: int = Field(default=1, gt=0)
    grid_hole_fill_min_neighbors: int = Field(default=6, ge=0, le=8)
    smoothing_passes: int = Field(default=10, gt=0)
    mesh_topology: Literal["delaunay", "grid"] = "delaunay"
    smoothing_method: Literal["neighbor_average", "local_plane"] = "neighbor_average"
    height_statistic: Literal["mean", "median"] = "mean"
    color_statistic: Literal["mean", "median"] = "mean"
    color_type: Literal["camera-rgb", "camera-rgb-frame-valid-pixels"] = "camera-rgb"
    outlier_height_threshold: Optional[float] = Field(default=None, gt=0.0)
    outlier_neighbor_radius: int = Field(default=1, ge=0)
    outlier_min_neighbors: int = Field(default=4, gt=0)
    max_triangle_edge_length: Optional[float] = Field(default=None, gt=0.0)
    max_triangle_normal_angle_deg: Optional[float] = Field(default=None, gt=0.0, le=90.0)
    spin_mesh_device: Literal["auto", "cpu", "cuda"] = "auto"
    # ``-1`` (default) auto-picks a worker count for the per-spin segmenter
    # loop; ``0`` or ``1`` keeps the legacy serial path; ``N > 1`` uses
    # exactly N workers. Auto-resolution happens at the call site via
    # ``_resolve_num_workers``.
    num_workers: int = Field(default=-1)

    extend_to_cuboids: GroundMeshExtensionToCuboidsConfig = Field(
        default_factory=GroundMeshExtensionToCuboidsConfig,
        description="Settings for extending the ground mesh to cover cuboid track positions.",
    )

    @field_validator("lidar_ids", "camera_ids", mode="before")
    @classmethod
    def _coerce_optional_str_tuple(cls, value: Any) -> Optional[tuple[str, ...]]:
        if value is None:
            return None
        if isinstance(value, str):
            # A bare string is almost never what callers mean; treat as a single id.
            return (value,)
        if isinstance(value, Iterable):
            return tuple(str(v) for v in value)
        raise TypeError(f"Expected None or an iterable of ids, got {type(value).__name__}")

    @classmethod
    def from_mapping(cls, source: Any, /, **overrides: Any) -> "GroundMeshReconstructionSettings":
        """Build parameters from any mapping or attribute-bearing config object."""

        def _extract(name: str) -> Any:
            if hasattr(source, name):
                return getattr(source, name)
            if isinstance(source, dict) or hasattr(source, "__getitem__"):
                try:
                    return source[name]
                except (KeyError, TypeError):
                    return ...
            return ...

        data: dict[str, Any] = {}
        for field_name in cls.model_fields:
            value = _extract(field_name)
            if value is not ...:
                data[field_name] = value
        data.update(overrides)
        return cls.model_validate(data)


@dataclass(frozen=True)
class _PerSpinConfig:
    """Read-only config snapshot shipped to each worker thread.

    Frozen so each worker's view of the per-segment config is unambiguous.
    """

    min_ray_length: float
    ground_control_point: NDArray[np.float32]
    ground_compat_max_distance: float
    ground_compat_max_angle_deg: float
    num_plane_hypotheses: int
    plane_max_distance: float
    plane_max_angle_deg: float
    plane_min_eval_points: int
    plane_max_eval_points: int
    plane_min_inliers: int
    enable_plane_extension: bool
    spin_mesh_device: str
    transformation: Optional["FrameConversion"]


@dataclass
class _PerSpinResult:
    """What each worker returns for one LiDAR spin."""

    idx: int
    points: NDArray[np.float32]  # (N, 3); transformed if a FrameConversion was applied
    normals: NDArray[np.float32]  # (N, 3)
    road_prior_mask: NDArray[np.bool_]  # (N,)
    road_mask: NDArray[np.bool_]  # (N,); dominant-plane inliers, all False if skipped
    frame_colors: Optional[NDArray[np.uint8]]  # (N, 3) or None
    skip_reason: Optional[str]  # 'no_ground_compat' / 'no_eval_pts' / 'no_inliers' or None
    skip_count: Optional[int]  # observed count for the skip warning, if skipped


@dataclass(frozen=True)
class _PreparedSpinPayload:
    """Transformed per-spin arrays shared by CPU workers and CUDA precompute."""

    idx: int
    points: NDArray[np.float32]
    centered_points: NDArray[np.float32]
    mask_valid_points: NDArray[np.bool_]
    road_prior_mask: NDArray[np.bool_]
    frame_colors: Optional[NDArray[np.uint8]]


def _prepare_spin_payload(
    idx: int,
    ray_starts: NDArray[np.float32],
    ray_ends: NDArray[np.float32],
    flags_road_prior: NDArray[np.bool_],
    frame_colors: Optional[NDArray[np.uint8]],
    config: _PerSpinConfig,
) -> _PreparedSpinPayload:
    points = ray_ends
    if config.transformation is not None:
        ray_starts = config.transformation.transform_points(ray_starts)
        points = config.transformation.transform_points(points)

    ray_lengths = np.sqrt(np.sum((points - ray_starts) ** 2, axis=1))
    mask_valid_points = ray_lengths >= config.min_ray_length
    return _PreparedSpinPayload(
        idx=idx,
        points=points,
        centered_points=points - ray_starts,
        mask_valid_points=mask_valid_points,
        road_prior_mask=flags_road_prior.astype(bool),
        frame_colors=frame_colors,
    )


def _compute_spin_normals_for_prepared_payload(
    payload: _PreparedSpinPayload,
    config: _PerSpinConfig,
) -> NDArray[np.float32]:
    subset_normals, _ = compute_mesh_normals_in_lidar_spin(
        payload.centered_points[payload.mask_valid_points],
        verbose=False,
        device=config.spin_mesh_device,
    )
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (len(payload.points), 1))
    normals[payload.mask_valid_points] = subset_normals
    return normals


def _process_single_spin_worker(
    idx: int,
    ray_starts: NDArray[np.float32],
    ray_ends: NDArray[np.float32],
    flags_road_prior: NDArray[np.bool_],
    frame_colors: Optional[NDArray[np.uint8]],
    config: _PerSpinConfig,
) -> _PerSpinResult:
    """Pure function executing the per-spin body of DominantPlaneRoadSegmenter.segment_road.

    Numerically identical to the inline serial loop body.

    Per-frame diagnostic file writes (``export_per_frame_diagnostics=True``)
    are intentionally not produced here — the segmenter forces ``num_workers``
    back to 1 when diagnostics are requested so the serial code path emits
    them in spin order.
    """
    payload = _prepare_spin_payload(idx, ray_starts, ray_ends, flags_road_prior, frame_colors, config)
    return _process_prepared_spin_worker(payload, config)


def _process_prepared_spin_worker(
    payload: _PreparedSpinPayload,
    config: _PerSpinConfig,
) -> _PerSpinResult:
    points = payload.points
    centered_points = payload.centered_points
    road_prior_mask = payload.road_prior_mask
    road_prior_indices = np.where(road_prior_mask)[0]

    normals = _compute_spin_normals_for_prepared_payload(payload, config)

    road_mask = np.zeros((len(points),), dtype=bool)

    up_vector = np.array([0, 0, 1], dtype=np.float32)
    gc_mask = find_ground_compatible_points(
        centered_points[road_prior_indices],
        normals[road_prior_indices],
        config.ground_control_point,
        ref_normal=up_vector,
        max_distance=config.ground_compat_max_distance,
        max_angle_deg=config.ground_compat_max_angle_deg,
    )

    ground_compatible_indices = road_prior_indices[gc_mask]

    if len(ground_compatible_indices) < config.plane_min_eval_points:
        return _PerSpinResult(
            idx=payload.idx,
            points=points,
            normals=normals,
            road_prior_mask=road_prior_mask,
            road_mask=road_mask,
            frame_colors=payload.frame_colors,
            skip_reason="no_ground_compat",
            skip_count=len(ground_compatible_indices),
        )

    eval_point_sub_indices = subsample(len(ground_compatible_indices), max_samples=config.plane_max_eval_points)
    eval_point_indices = ground_compatible_indices[eval_point_sub_indices]

    if len(eval_point_indices) < config.plane_min_eval_points:
        return _PerSpinResult(
            idx=payload.idx,
            points=points,
            normals=normals,
            road_prior_mask=road_prior_mask,
            road_mask=road_mask,
            frame_colors=payload.frame_colors,
            skip_reason="no_eval_pts",
            skip_count=len(eval_point_indices),
        )

    plane, inlier_mask, ransac = detect_dominant_plane(
        points[ground_compatible_indices],
        normals[ground_compatible_indices],
        distance_threshold=config.plane_max_distance,
        angle_threshold_deg=config.plane_max_angle_deg,
        num_hypotheses=config.num_plane_hypotheses,
        eval_point_indices=eval_point_sub_indices,
    )
    inlier_indices = ground_compatible_indices[inlier_mask]
    inlier_mask = np.zeros((len(points),), dtype=bool)
    inlier_mask[inlier_indices] = True
    if config.enable_plane_extension:
        inlier_mask = ransac.find_inliers(plane, points, normals)
        inlier_indices = np.where(inlier_mask)[0]

    if len(inlier_indices) < config.plane_min_inliers:
        return _PerSpinResult(
            idx=payload.idx,
            points=points,
            normals=normals,
            road_prior_mask=road_prior_mask,
            road_mask=road_mask,
            frame_colors=payload.frame_colors,
            skip_reason="no_inliers",
            skip_count=len(inlier_indices),
        )

    return _PerSpinResult(
        idx=payload.idx,
        points=points,
        normals=normals,
        road_prior_mask=road_prior_mask,
        road_mask=inlier_mask,
        frame_colors=payload.frame_colors,
        skip_reason=None,
        skip_count=None,
    )


def _log_pooled_skip_warning(idx: int, result: _PerSpinResult, config: _PerSpinConfig) -> None:
    assert result.skip_reason is not None
    assert result.skip_count is not None
    if result.skip_reason == "no_ground_compat":
        logger.warning(
            f"Frame {idx} contains too few ground-compatible points ({result.skip_count} but "
            f"{config.plane_min_eval_points} required), skipping segmentation"
        )
    elif result.skip_reason == "no_eval_pts":
        logger.warning(
            f"Frame {idx} has too few plane evaluation samples ({result.skip_count}, "
            f"{config.plane_min_eval_points} required), skipping segmentation"
        )
    elif result.skip_reason == "no_inliers":
        logger.warning(
            f"Frame {idx} has too few inliers ({result.skip_count}) to trust the dominant plane, "
            f"{config.plane_min_inliers} required, skipping segmentation"
        )
    else:
        raise ValueError(f"Unknown pooled road segmentation skip reason: {result.skip_reason}")


def _assert_detected_threadpools_single_threaded() -> None:
    multithreaded_pools = [
        pool
        for pool in threadpool_info()
        if pool.get("user_api") in {"blas", "openmp"} and pool.get("num_threads", 1) > 1
    ]
    assert not multithreaded_pools, f"Expected single-threaded BLAS/OpenMP pools, got {multithreaded_pools}"


@contextmanager
def _single_threaded_blas_context() -> Generator[None, None, None]:
    """Pin BLAS/OpenMP libraries to a single thread while workers run.

    The per-spin work calls into pcu mesh-normal estimation and numpy
    reductions, both of which spin OpenMP teams under the hood. With N
    threads each launching an OpenMP team of size all-cores the box thrashes.
    The env vars cover libraries initialized after this point. ``threadpoolctl``
    also limits already-loaded BLAS/OpenMP runtimes for the duration of the
    worker pool.
    """
    variable_names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    previous_values = {name: os.environ.get(name) for name in variable_names}
    try:
        for name in variable_names:
            os.environ[name] = "1"
        with threadpool_limits(limits=1, user_api=None):
            _assert_detected_threadpools_single_threaded()
            yield
    finally:
        for name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


@dataclass
class DominantPlaneRoadSegmenter:
    """Algorithm to segment the road in Lidar spins around an ego vehicle, based on dominant plane detection"""

    # Point cloud filtering parameters
    min_ray_length: float

    # Parameters controlling geometric constraints for plane hypotheses.
    ground_control_point: NDArray[np.float32]
    ground_compat_max_distance: float
    ground_compat_max_angle_deg: float

    # Parameters for RANSAC plane fitting.
    num_plane_hypotheses: int
    plane_max_distance: float
    plane_max_angle_deg: float
    plane_min_eval_points: int  # Min points to evaluate plane hypotheses on, skipping frame if not reached
    plane_max_eval_points: int  # Max points to evaluate plane hypotheses on, subsampling if exceeded
    plane_min_inliers: int  # Min number of inliers for accepting the dominant plane, skipping frame if not reached
    # Whether to extend the found dominant plane to all inlier points inside a full Lidar spin (True) or
    # only use the subset initially marked road (False).
    enable_plane_extension: bool = False

    # If true, camera colors are considered valid only where RGB_LABEL is set.
    # This is opt-in so existing callers that provide colored PointClouds
    # without RGB flags retain their legacy all-colors-valid behavior.
    respect_rgb_label_validity: bool = False

    # Parameters to control the output.
    export_per_frame_diagnostics: bool = False
    output_path: Path = Path(".")
    verbose: bool = False

    # Per-spin work is independent across spins. ``num_workers > 1`` runs the
    # per-spin loop body in a ``ThreadPoolExecutor``. Threads (not
    # multiprocessing) are used because the heavy per-spin work (the scanline
    # triangulator's numpy reductions and pcu mesh-normal estimation) releases
    # the GIL during compute, so threading parallelizes the dominant cost
    # without the bootstrap and CUDA-fork issues a multiprocessing pool would
    # face inside an init-time bazel-wrapped entry point.
    #
    #   num_workers <= 1: serial code path (default).
    #   num_workers > 1:  threaded path with that many worker threads.
    #
    # ``__post_init__`` forces ``num_workers=1`` if
    # ``export_per_frame_diagnostics=True``, since per-frame PLY writes assume
    # in-order, single-thread execution.
    num_workers: int = 1

    # Device for the per-spin spherical-mesh + vertex-normals computation.
    # ``"auto"`` picks CUDA when available, otherwise CPU. Explicit ``"cuda"``
    # / ``"cpu"`` skip auto-detection.
    spin_mesh_device: Literal["auto", "cpu", "cuda"] = "auto"

    def __post_init__(self):
        assert self.ground_control_point.size == 3
        assert self.ground_compat_max_distance > 0.0
        assert self.ground_compat_max_angle_deg > 0.0
        assert self.num_plane_hypotheses > 0
        assert self.plane_max_distance > 0.0
        assert self.plane_max_angle_deg > 0.0
        assert self.plane_min_eval_points > 0
        assert self.plane_max_eval_points > 0
        assert self.plane_min_eval_points < self.plane_max_eval_points
        assert self.plane_min_inliers > 0
        if self.spin_mesh_device not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"Unsupported spin_mesh_device={self.spin_mesh_device!r}; expected 'auto', 'cpu', or 'cuda'"
            )
        if self.export_per_frame_diagnostics and self.num_workers > 1:
            logger.warning(
                "DominantPlaneRoadSegmenter: export_per_frame_diagnostics=True forces num_workers=1 "
                "(per-frame PLY writes are emitted only by the serial path). "
                f"Overriding num_workers from {self.num_workers} to 1."
            )
            self.num_workers = 1

        if self.spin_mesh_device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = self.spin_mesh_device
        self._spin_mesh_device = resolved

    def segment_road(
        self,
        point_cloud_generator: Generator[PointCloud, None, None],
        transformation: Optional[FrameConversion] = None,
    ) -> Tuple[
        NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], Optional[NDArray[np.uint8]]
    ]:
        # When num_workers > 1, run the per-spin loop body in a thread pool.
        # The pooled path is numerically equivalent to the serial path; it
        # forgoes the debug-only per-frame diagnostic PLYs (which
        # __post_init__ forces back to num_workers=1 when both are set).
        if self.num_workers > 1:
            return self._segment_road_pooled(
                point_cloud_generator=point_cloud_generator,
                transformation=transformation,
            )

        # The generator yields one Lidar spin at a time. The input already carries
        # a semantic road segmentation that is not yet geometrically verified.
        collected_points: List[NDArray[np.float32]] = []
        collected_normals: List[NDArray[np.float32]] = []
        collected_road_masks: List[NDArray[np.bool_]] = []
        collected_road_prior_masks: List[NDArray[np.bool_]] = []
        collected_colors: List[NDArray[np.uint8]] = []
        collected_color_valid_masks: List[NDArray[np.bool_]] = []
        has_colors: Optional[bool] = None  # Determined from first frame

        highlight_color = np.array([255, 255, 50], dtype=np.uint8)

        start_time_loop = perf_counter()

        logger.info("Road segmentation per lidar frame")
        for idx, point_cloud in enumerate(point_cloud_generator):
            # Output path to be used in frame export functions
            frame_output_path = self.output_path / f"{idx:06d}"

            ray_starts = point_cloud.xyz_start.cpu().numpy()
            points = point_cloud.xyz_end.cpu().numpy()
            # color_type is set uniformly across the generator, so all frames typically either
            # have or lack colors. This guard is defensive against generator contract changes.
            frame_colors = point_cloud.color.cpu().numpy() if point_cloud.color is not None else None
            frame_color_valid_mask = (
                torch.bitwise_and(unpack_optional(point_cloud.flags), RayFlags.RGB_LABEL.value)
                .cpu()
                .numpy()
                .astype(bool)
                if self.respect_rgb_label_validity
                else np.ones(len(points), dtype=bool)
            )
            if has_colors is None:
                has_colors = frame_colors is not None
            elif has_colors and frame_colors is None:
                logger.warning("Frame has no colors despite previous frames having colors, disabling color collection")
                has_colors = False

            if self.verbose:
                logger.info(f"Frame {idx} contains {len(points)} rays")
            if len(points) == 0:  # Just in case
                logger.warning(f"Frame {idx} contains no rays, skipping frame")
                continue

            if transformation is not None:
                ray_starts = transformation.transform_points(ray_starts)
                points = transformation.transform_points(points)

            # Drop LiDAR returns closer to the sensor than the near threshold.
            # Stray near-sensor returns have been observed on some datasets (H8.1) and can
            # tear apart the LiDAR spin mesh.
            ray_lengths = np.sqrt(np.sum((points - ray_starts) ** 2, axis=1))
            mask_invalid_points = ray_lengths < self.min_ray_length
            mask_valid_points = ~mask_invalid_points
            if self.verbose:
                logger.info(
                    f"{np.sum(mask_invalid_points)} near points (d<{self.min_ray_length}) detected "
                    f"({100.0 * np.sum(mask_invalid_points) / max(len(points), 1):.1f}%)"
                )

            if self.export_per_frame_diagnostics:
                colors = np.zeros((len(points), 3), dtype=np.uint8)
                colors[mask_invalid_points] = highlight_color  # No failure in case of empty index list, TODO: test
                save_ply(str(frame_output_path) + "_input_near_points_marked.ply", points, colors=colors, logger=logger)

            # The input points carry a road segmentation produced by projecting camera-image
            # semantic labels onto the cloud. This prior contains outliers (points marked
            # road that are far from the road plane) that we filter out before meshing.
            road_prior_mask = (
                torch.bitwise_and(unpack_optional(point_cloud.flags), RayFlags.ROAD_SEMANTIC.value)
                .cpu()
                .numpy()
                .astype(bool)
            )
            road_prior_indices = np.where(road_prior_mask)[0]
            if self.verbose:
                logger.info(f"{len(road_prior_indices)} input points initially marked road")

            if self.export_per_frame_diagnostics:
                colors = np.zeros((len(points), 3), dtype=np.uint8)
                colors[road_prior_indices] = highlight_color  # No failure in case of empty index list
                save_ply(str(frame_output_path) + "_input_points.ply", points, colors=colors, logger=logger)

            # Estimate per-point normals via a "lidar spin mesh" reconstruction in
            # azimuth-elevation space. This is robust to ray drops and to the sparse
            # elevation sampling typical of spinning mobile lidars, and avoids k-NN
            # normal estimation which is unstable within a single spin and degrades
            # with distance (close neighbors tend to lie on the same scanline).
            # Centering rays at the origin approximates the original Lidar-frame
            # geometry by discarding the sensor-to-world rotation and per-ray motion
            # compensation. The mesh topology is practically invariant to those
            # transformations as long as rig motion within a spin is mostly
            # translational, which holds well enough in practice even during turns.
            # TODO: if artifacts ever appear, transform rays back to the sensor
            # frame instead of centering.
            centered_points = points - ray_starts
            subset_normals, subset_triangles = compute_mesh_normals_in_lidar_spin(
                centered_points[mask_valid_points],
                verbose=self.verbose,
                device=self._spin_mesh_device,
            )

            # LiDAR spin mesh was computed from only valid points.
            # Maps normals and triangles back to the original points.
            index_remap = np.arange(len(points), dtype=np.int32)[mask_valid_points]
            triangles = index_remap[subset_triangles]
            normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (len(points), 1))
            normals[mask_valid_points] = subset_normals

            if self.export_per_frame_diagnostics:
                save_ply(
                    str(frame_output_path) + "_spin_mesh.ply",
                    points,
                    triangles,
                    normals,
                    logger=logger,
                )

            # Collect relevant data from each frame.
            collected_points.append(points)
            collected_normals.append(normals)
            collected_road_prior_masks.append(road_prior_mask)
            if has_colors and frame_colors is not None:
                collected_colors.append(frame_colors)
                collected_color_valid_masks.append(frame_color_valid_mask)
            # Pre-append an empty mask so the road-inlier detection steps below can
            # bail out at any point and leave this frame with zero detected road points.
            collected_road_masks.append(np.zeros((len(points),), dtype=bool))

            # Cull points using simple "ground-compatibility" geometric priors so that
            # the subsequent plane-hypothesis generation is not wasted on obvious
            # non-ground points:
            # 1. Point and normal define a plane that passes near the nominal ground
            #    point under the sensor.
            # 2. Point normal is close enough to the up direction. The tolerance must
            #    accommodate road inclination (extreme San Francisco streets reach
            #    ~22 deg inclination at 40% grade), vehicle pitch/roll (under 2 deg
            #    at normal driving, up to 5 deg aggressive, higher for SUVs and
            #    trucks), and normal estimation noise.
            if self.verbose:
                logger.info("Finding ground-compatible points")
            up_vector = np.array([0, 0, 1], dtype=np.float32)
            gc_mask = find_ground_compatible_points(
                centered_points[road_prior_indices],
                normals[road_prior_indices],
                self.ground_control_point,
                ref_normal=up_vector,
                max_distance=self.ground_compat_max_distance,
                max_angle_deg=self.ground_compat_max_angle_deg,
            )

            # Discard points marked road that are not ground-compatible.
            ground_compatible_indices = road_prior_indices[gc_mask]
            del gc_mask  # Submask only, prevent further use.
            if self.verbose:
                logger.info(
                    f"{len(ground_compatible_indices)} ground-compatible points "
                    f"({100.0 * len(ground_compatible_indices) / max(len(road_prior_indices), 1):.1f}%)"
                )
            if self.export_per_frame_diagnostics:
                colors = np.zeros((len(points), 3), dtype=np.uint8)
                colors[ground_compatible_indices] = highlight_color
                save_ply(
                    str(frame_output_path) + "_ground_compatible_points.ply",
                    points,
                    normals=normals,
                    colors=colors,
                    logger=logger,
                )
            if len(ground_compatible_indices) < self.plane_min_eval_points:
                logger.warning(
                    f"Frame {idx} contains too few ground-compatible points ({len(ground_compatible_indices)} but "
                    f"{self.plane_min_eval_points} required), skipping segmentation"
                )
                continue

            # Randomly select a subset of points to evaluate plane hypotheses on (reduces the computational burden)
            eval_point_sub_indices = subsample(len(ground_compatible_indices), max_samples=self.plane_max_eval_points)
            eval_point_indices = ground_compatible_indices[eval_point_sub_indices]
            if self.verbose:
                logger.info(
                    f"{len(eval_point_indices)} of {len(ground_compatible_indices)} points "
                    f"sampled for evaluating plane hypotheses"
                )
            if self.export_per_frame_diagnostics:
                save_ply(
                    str(frame_output_path) + "_plane_eval_sample.ply",
                    points[eval_point_indices],
                    normals=normals[eval_point_indices],
                    logger=logger,
                )
            if len(eval_point_indices) < self.plane_min_eval_points:
                logger.warning(
                    f"Frame {idx} has too few plane evaluation samples ({len(eval_point_indices)}, "
                    f"{self.plane_min_eval_points} required), skipping segmentation"
                )
                continue

            # Robust plane fitting on points presegmented as road that are also
            # ground-compatible. Conservative by design: high precision over recall,
            # so it may miss some road points but rarely admits false positives.
            start_time = perf_counter()
            plane, inlier_mask, ransac = detect_dominant_plane(
                points[ground_compatible_indices],
                normals[ground_compatible_indices],
                distance_threshold=self.plane_max_distance,
                angle_threshold_deg=self.plane_max_angle_deg,
                num_hypotheses=self.num_plane_hypotheses,
                eval_point_indices=eval_point_sub_indices,
            )
            inlier_indices = ground_compatible_indices[inlier_mask]
            inlier_mask = np.zeros((len(points),), dtype=bool)
            inlier_mask[inlier_indices] = True
            if self.enable_plane_extension:
                # Find inliers in the full point cloud. Less conservative: the
                # plane extends globally so points far from the road can
                # accidentally pass the test and re-introduce false positives.
                inlier_mask = ransac.find_inliers(plane, points, normals)
                inlier_indices = np.where(inlier_mask)[0]

            elapsed_time = perf_counter() - start_time
            if self.verbose:
                logger.info(
                    f"{self.num_plane_hypotheses} plane hypotheses evaluated against "
                    f"{len(eval_point_sub_indices)} points and winner has {len(inlier_indices)} inliers "
                    f"out of {len(ground_compatible_indices)} points "
                    f"({100.0 * len(inlier_indices) / max(len(ground_compatible_indices), 1):.1f}%), "
                    f"took {elapsed_time:.3f} s"
                )
            if self.export_per_frame_diagnostics:
                colors = np.zeros((len(points), 3), dtype=np.uint8)
                colors[inlier_indices] = highlight_color
                save_ply(
                    str(frame_output_path) + "_plane_segmentation.ply",
                    points,
                    normals=normals,
                    colors=colors,
                    logger=logger,
                )
                save_ply(
                    str(frame_output_path) + "_plane_inliers.ply",
                    points[inlier_indices],
                    normals=normals[inlier_indices],
                    logger=logger,
                )

                image = DominantPlaneRoadSegmenter._render_points(
                    centered_points, colors, camera_position=np.array([-10, 0, 20])
                )
                _save_image(str(frame_output_path) + "_plane_segmentation.png", image)

            if len(inlier_indices) < self.plane_min_inliers:
                logger.warning(
                    f"Frame {idx} has too few inliers ({len(inlier_indices)}) to trust the dominant plane, "
                    f"{self.plane_min_inliers} required, skipping segmentation"
                )
                continue

            # Accept segmentation result by storing it.
            # Replaces the default no-inliers mask already stored upfront.
            collected_road_masks[-1] = inlier_mask

        elapsed_time = perf_counter() - start_time_loop
        logger.info(f"Road segmentation for {len(collected_points)} lidar frames took {elapsed_time:.3f} s")

        if len(collected_points) == 0 or collected_points[0].size == 0:
            raise ValueError("No points in any of the lidar frames")

        logger.info(f"Merging {len(collected_points)} lidar point clouds")
        start_time = perf_counter()
        points = np.concatenate(collected_points)  # Raises ValueError for [] input
        normals = np.concatenate(collected_normals)
        road_mask = np.concatenate(collected_road_masks)
        road_prior_mask = np.concatenate(collected_road_prior_masks)
        merged_colors = np.concatenate(collected_colors) if has_colors and collected_colors else None
        if merged_colors is not None and self.respect_rgb_label_validity:
            merged_color_valid_mask = np.concatenate(collected_color_valid_masks)
            merged_colors = _fill_invalid_road_colors(
                points,
                merged_colors,
                merged_color_valid_mask,
                road_mask,
            )
        elapsed_time = perf_counter() - start_time
        logger.info(f"Merged {len(points)} points from {len(collected_points)} spins in {elapsed_time:.3f} s")

        return points, normals, road_mask, road_prior_mask, merged_colors

    def _segment_road_pooled(
        self,
        point_cloud_generator: Generator[PointCloud, None, None],
        transformation: Optional[FrameConversion],
    ) -> Tuple[
        NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], Optional[NDArray[np.uint8]]
    ]:
        """Multi-threaded equivalent of segment_road's per-spin loop.

        Three steps:

        1. Drain the generator on the calling thread (it is backed by a single
           PyTorch dataloader, so concurrent iteration would just contend for
           the same I/O). Each yielded ``PointCloud`` is unpacked into plain
           numpy arrays.
        2. Dispatch the independent per-spin normal and segmentation work via
           ``ThreadPoolExecutor``. Keeping normal computation worker-local
           allows CUDA normal work and CPU segmentation to overlap across spins.
           The heavy numpy reductions release the GIL during compute, so
           threads parallelize effectively. Each worker is pinned to a single
           OpenMP thread to avoid oversubscription.
        3. Merge per-spin arrays in idx order so concatenated outputs match
           the serial path byte-for-byte.
        """
        # Step 1: drain the generator into plain-numpy per-spin payloads.
        drain_start = perf_counter()
        payloads: List[Tuple] = []
        color_valid_masks_by_idx: Dict[int, NDArray[np.bool_]] = {}
        has_colors: Optional[bool] = None
        for idx, point_cloud in enumerate(point_cloud_generator):
            ray_starts = point_cloud.xyz_start.cpu().numpy()
            ray_ends = point_cloud.xyz_end.cpu().numpy()
            frame_colors = point_cloud.color.cpu().numpy() if point_cloud.color is not None else None
            color_valid_masks_by_idx[idx] = (
                torch.bitwise_and(unpack_optional(point_cloud.flags), RayFlags.RGB_LABEL.value)
                .cpu()
                .numpy()
                .astype(bool)
                if self.respect_rgb_label_validity
                else np.ones(len(ray_ends), dtype=bool)
            )
            if has_colors is None:
                has_colors = frame_colors is not None
            elif has_colors and frame_colors is None:
                logger.warning("Frame has no colors despite previous frames having colors, disabling color collection")
                has_colors = False
            if len(ray_ends) == 0:
                logger.warning(f"Frame {idx} contains no rays, skipping frame")
                continue
            flags_road_prior = (
                torch.bitwise_and(unpack_optional(point_cloud.flags), RayFlags.ROAD_SEMANTIC.value)
                .cpu()
                .numpy()
                .astype(bool)
            )
            payloads.append((idx, ray_starts, ray_ends, flags_road_prior, frame_colors))

        if not payloads:
            raise ValueError("No points in any of the lidar frames")
        logger.info(
            f"Drained {len(payloads)} lidar frames into segmentation payloads in {perf_counter() - drain_start:.3f} s"
        )

        config = _PerSpinConfig(
            min_ray_length=self.min_ray_length,
            ground_control_point=self.ground_control_point,
            ground_compat_max_distance=self.ground_compat_max_distance,
            ground_compat_max_angle_deg=self.ground_compat_max_angle_deg,
            num_plane_hypotheses=self.num_plane_hypotheses,
            plane_max_distance=self.plane_max_distance,
            plane_max_angle_deg=self.plane_max_angle_deg,
            plane_min_eval_points=self.plane_min_eval_points,
            plane_max_eval_points=self.plane_max_eval_points,
            plane_min_inliers=self.plane_min_inliers,
            enable_plane_extension=self.enable_plane_extension,
            spin_mesh_device=self._spin_mesh_device,
            transformation=transformation,
        )

        # Step 2: dispatch via ThreadPoolExecutor.
        n_workers = min(self.num_workers, len(payloads))
        worker_start = perf_counter()

        results_by_idx: Dict[int, _PerSpinResult] = {}
        with _single_threaded_blas_context():
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(
                        _process_single_spin_worker,
                        idx,
                        ray_starts,
                        ray_ends,
                        road_prior_mask,
                        frame_colors,
                        config,
                    )
                    for idx, ray_starts, ray_ends, road_prior_mask, frame_colors in payloads
                ]
                for future in futures:
                    result = future.result()
                    results_by_idx[result.idx] = result
        logger.info(
            f"Processed {len(results_by_idx)} lidar segmentation payloads with {n_workers} workers in "
            f"{perf_counter() - worker_start:.3f} s"
        )

        # Step 3: collect outputs in idx order so concatenated arrays match the
        # serial path byte-for-byte (the serial path's concat order is the
        # generator's yield order, which is monotonic idx).
        collected_points: List[NDArray[np.float32]] = []
        collected_normals: List[NDArray[np.float32]] = []
        collected_road_masks: List[NDArray[np.bool_]] = []
        collected_road_prior_masks: List[NDArray[np.bool_]] = []
        collected_colors: List[NDArray[np.uint8]] = []
        collected_color_valid_masks: List[NDArray[np.bool_]] = []
        for idx in sorted(results_by_idx.keys()):
            r = results_by_idx[idx]
            if r.skip_reason is not None:
                _log_pooled_skip_warning(idx, r, config)
            collected_points.append(r.points)
            collected_normals.append(r.normals)
            collected_road_prior_masks.append(r.road_prior_mask)
            collected_road_masks.append(r.road_mask)
            if has_colors and r.frame_colors is not None:
                collected_colors.append(r.frame_colors)
                collected_color_valid_masks.append(color_valid_masks_by_idx[idx])

        if collected_points[0].size == 0:
            raise ValueError("No points in any of the lidar frames")

        points = np.concatenate(collected_points)
        normals = np.concatenate(collected_normals)
        road_mask = np.concatenate(collected_road_masks)
        road_prior_mask = np.concatenate(collected_road_prior_masks)
        merged_colors = np.concatenate(collected_colors) if has_colors and collected_colors else None
        if merged_colors is not None and self.respect_rgb_label_validity:
            merged_color_valid_mask = np.concatenate(collected_color_valid_masks)
            merged_colors = _fill_invalid_road_colors(
                points,
                merged_colors,
                merged_color_valid_mask,
                road_mask,
            )
        return points, normals, road_mask, road_prior_mask, merged_colors

    @staticmethod
    def _render_points(
        points: NDArray[np.float32],
        colors: NDArray[np.uint8],
        camera_target: NDArray[np.float32] = np.array([0, 0, 0], dtype=np.float32),
        camera_position: NDArray[np.float32] = np.array([0, 0, 100], dtype=np.float32),
        resolution_px: Tuple[int, int] = (1280, 1024),
        hfov_deg: float = 75.0,
        near_plane=1e-3,
        bgcolor: ArrayLike = np.array([100, 100, 100], dtype=np.uint8),
    ) -> PILImage.Image:
        """Simplified rendering of colored points into an image"""
        assert points.shape[1] == 3
        assert colors.shape == points.shape

        # Calculate world to camera transformation
        z_axis = camera_target - camera_position
        z_axis /= np.linalg.norm(z_axis)
        up_vector = np.array([0, 0, 1])
        x_axis = np.cross(z_axis, up_vector)
        x_axis_norm = np.linalg.norm(x_axis)
        # Handle camera looking straight up or down, as the cross product is null.
        x_axis = np.array([1, 0, 0]) if x_axis_norm < 1e-3 else x_axis / x_axis_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)  # Just for numerical precision
        camera_to_world_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        assert np.allclose(camera_to_world_rotation @ camera_to_world_rotation.T, np.identity(3))

        # Transform points to camera space
        points_cam_space = (points - camera_position.reshape(1, 3)) @ camera_to_world_rotation

        # Cull points that are not in front of the near plane
        indices_front = np.where(points_cam_space[:, 2] > near_plane)[0]

        # Project points to the image
        width, height = resolution_px
        focal_length_px = width / 2 / np.tan(np.radians(hfov_deg) / 2)
        x = focal_length_px * points_cam_space[indices_front, 0] / points_cam_space[indices_front, 2] + width / 2.0
        y = focal_length_px * points_cam_space[indices_front, 1] / points_cam_space[indices_front, 2] + height / 2.0
        projects_inside = (x > 0.0) & (x < float(width)) & (y > 0.0) & (y < float(height))

        # Cull points that do not project within the image frame
        x = x[projects_inside]
        y = y[projects_inside]
        xi = x.astype(np.int32)
        yi = y.astype(np.int32)
        visible_point_indices = indices_front[projects_inside]

        colors = colors[visible_point_indices]

        image = np.empty(shape=(height, width, 3), dtype=np.uint8)
        image[:] = np.asarray(bgcolor)
        # Each point being splatted into a single pixel without any care to depth ordering
        image[yi, xi] = colors
        return PILImage.fromarray(image)


def downsample_3d_points_on_2d_grid(
    points: NDArray[np.float32],
    cell_size: float,
    colors: Optional[NDArray[np.uint8]] = None,
    height_statistic: Literal["mean", "median"] = "mean",
    color_statistic: Literal["mean", "median"] = "mean",
    min_points_per_cell: int = 1,
) -> Tuple[NDArray[np.float32], Optional[NDArray[np.uint8]]]:
    """Downsample points on a 2D grid in the XY plane.

    Constructs a 2D grid within the bounding box of the point cloud (discarding the Z coordinate in this step).
    A point is returned at the center of each occupied cell (producing a more regular output point distribution),
    and the output point's height is aggregated from all input points falling into the cell.

    ``height_statistic`` and ``color_statistic`` support ``mean`` and ``median``. Median height is useful
    for noisy camera-depth road points because it is robust to isolated spike points inside a BEV cell.

    Cells with fewer than ``min_points_per_cell`` input points are dropped. Default of 1 keeps every
    occupied cell (legacy behavior); raise to 2-5 to suppress sparse, noise-dominated cells.
    """
    assert len(points.shape) == 2 and points.shape[1] == 3
    assert len(points) > 0
    assert cell_size > 0.0
    assert min_points_per_cell >= 1
    assert colors is None or colors.shape == points.shape
    if height_statistic not in ("mean", "median"):
        raise ValueError(f"Unsupported height_statistic={height_statistic!r}; expected 'mean' or 'median'")
    if color_statistic not in ("mean", "median"):
        raise ValueError(f"Unsupported color_statistic={color_statistic!r}; expected 'mean' or 'median'")

    bbox_min = np.min(points[:, 0:2], axis=0)
    bbox_max = np.max(points[:, 0:2], axis=0) + cell_size * 1e-4  # Workaround to include boundary points

    # At least 1 cell along each dimension.
    cell_counts = np.maximum(np.ceil((bbox_max - bbox_min) / cell_size), [1, 1]).astype(int)
    scale = 1 / cell_size

    logger.info(
        f"bbox_size=({bbox_max[0] - bbox_min[0]:.3f}, {bbox_max[1] - bbox_min[1]:.3f}), "
        f"num_cells={tuple(cell_counts)}, "
        f"total_cells={np.prod(cell_counts) / 1e3:.1f}k"
    )

    # Point to cell assignments.
    normalized_points = scale * (points[:, 0:2] - bbox_min)
    cell_indices = np.floor(normalized_points).astype(int)
    cell_ids = cell_indices[:, 1] * cell_counts[0] + cell_indices[:, 0]  # Linear cell index per point.

    # Group points by cell. ``np.bincount`` gives a fast vectorized sum
    # for the mean path; the median path needs grouped values, so we
    # share an ``np.unique``+``inverse`` decomposition.
    unique_cell_ids, inverse = np.unique(cell_ids, return_inverse=True)
    num_groups = len(unique_cell_ids)
    point_counts_per_group = np.bincount(inverse, minlength=num_groups).astype(np.float32)

    if height_statistic == "mean":
        z_sum_per_group = np.bincount(inverse, weights=points[:, 2], minlength=num_groups)
        z_per_group = (z_sum_per_group / point_counts_per_group).astype(np.float32)
    else:
        # Median needs grouped values; sort once by cell-id and split.
        sorting_indices = np.argsort(cell_ids, kind="stable")
        sorted_inverse = inverse[sorting_indices]
        sorted_points_z = points[sorting_indices, 2]
        group_start_indices = np.searchsorted(sorted_inverse, np.arange(num_groups))
        points_per_group_z = np.split(sorted_points_z, group_start_indices[1:])
        z_per_group = np.array([np.median(g) for g in points_per_group_z], dtype=np.float32)

    # Recover (col, row) cell indices from the linear ids (cell_id = row * W + col).
    width_cells = int(cell_counts[0])
    unique_cols = unique_cell_ids % width_cells
    unique_rows = unique_cell_ids // width_cells
    group_cell_indices = np.stack([unique_cols, unique_rows], axis=1).astype(np.float32)
    cell_centers = (cell_size * (group_cell_indices + 0.5) + bbox_min).astype(np.float32)
    downsampled_points = np.hstack((cell_centers, z_per_group.reshape(-1, 1)))

    downsampled_colors: Optional[NDArray[np.uint8]] = None
    if colors is not None:
        colors_f = colors.astype(np.float32)
        if color_statistic == "mean":
            # Per-channel sum via bincount (cheaper than an np.add.at-style 3-axis scatter).
            c_sum_r = np.bincount(inverse, weights=colors_f[:, 0], minlength=num_groups)
            c_sum_g = np.bincount(inverse, weights=colors_f[:, 1], minlength=num_groups)
            c_sum_b = np.bincount(inverse, weights=colors_f[:, 2], minlength=num_groups)
            avg = np.stack([c_sum_r, c_sum_g, c_sum_b], axis=1) / point_counts_per_group[:, None]
            downsampled_colors = avg.astype(np.uint8)
        else:
            # Reuse the same sort decomposition as the height-median branch
            # if it ran; otherwise compute it now.
            if height_statistic != "median":
                sorting_indices = np.argsort(cell_ids, kind="stable")
                sorted_inverse = inverse[sorting_indices]
                group_start_indices = np.searchsorted(sorted_inverse, np.arange(num_groups))
            sorted_colors = colors_f[sorting_indices]
            colors_per_group = np.split(sorted_colors, group_start_indices[1:], axis=0)
            downsampled_colors = np.array([np.median(c, axis=0) for c in colors_per_group], dtype=np.uint8)

    # Drop sparse cells. Default ``min_points_per_cell=1`` is a no-op.
    if min_points_per_cell > 1:
        keep_mask = point_counts_per_group >= min_points_per_cell
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            logger.info(f"Dropped {n_dropped} / {num_groups} cells with fewer than {min_points_per_cell} points")
        downsampled_points = downsampled_points[keep_mask]
        if downsampled_colors is not None:
            downsampled_colors = downsampled_colors[keep_mask]

    return downsampled_points, downsampled_colors


def downsample_3d_points_on_2d_grid_no_clamp(points: NDArray[np.float32], cell_size: float) -> NDArray[np.float32]:
    """Merge points in each occupied XY grid cell by averaging their XYZ coordinates."""
    assert points.ndim == 2 and points.shape[1] == 3
    assert cell_size > 0.0
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)

    cell_indices = np.floor(points[:, :2] / cell_size).astype(np.int64)
    _, inverse = np.unique(cell_indices, axis=0, return_inverse=True)
    point_counts = np.bincount(inverse)
    point_sums = np.zeros((len(point_counts), 3), dtype=np.float32)
    np.add.at(point_sums, inverse, points)
    return (point_sums / point_counts[:, None]).astype(np.float32)


def filter_local_height_outliers(
    vertices: NDArray[np.float32],
    cell_size: float,
    threshold: float,
    min_neighbors: int = 4,
    radius: int = 1,
) -> NDArray[np.bool_]:
    """Identify and mark vertex spikes that deviate strongly from their BEV neighbors.

    For each vertex, look at vertices within ``radius`` cells in BEV (excluding self). If the
    neighbor count is at least ``min_neighbors``, compute their height median and drop the
    current vertex when ``|z - median| > threshold``. Cells with sparse neighborhoods (e.g. road
    boundary) are kept as-is to avoid over-aggressive filtering.

    Assumes ``vertices`` are placed at BEV cell centers (the output of
    :func:`downsample_3d_points_on_2d_grid`).
    """
    n = len(vertices)
    if n == 0 or threshold <= 0.0:
        return np.ones(n, dtype=bool)

    # Recover integer cell indices from XY positions. Vertices sit at cell centers, so the
    # difference between two cell centers is exactly an integer multiple of ``cell_size`` and
    # ``round(delta / cell_size)`` snaps to that integer modulo float noise.
    bbox_min_xy = vertices[:, :2].min(axis=0)
    cell_idx = np.round((vertices[:, :2] - bbox_min_xy) / cell_size).astype(np.int64)

    cell_to_vertex: dict = {}
    for i in range(n):
        cell_to_vertex[(int(cell_idx[i, 0]), int(cell_idx[i, 1]))] = i

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        cx = int(cell_idx[i, 0])
        cy = int(cell_idx[i, 1])
        neighbor_zs: List[float] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                j = cell_to_vertex.get((cx + dx, cy + dy))
                if j is not None:
                    neighbor_zs.append(float(vertices[j, 2]))
        if len(neighbor_zs) >= min_neighbors:
            ref_z = float(np.median(neighbor_zs))
            if abs(float(vertices[i, 2]) - ref_z) > threshold:
                keep[i] = False
    return keep


def filter_triangles_by_max_edge_length(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int32],
    max_edge_length: float,
    protected_triangle_mask: Optional[NDArray[np.bool_]] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_]]:
    """Drop triangles whose BEV (XY) max edge exceeds ``max_edge_length`` and compact vertices.

    Convex-hull / large-gap bridging triangles introduced by 2D Delaunay (e.g., across non-road
    regions or sensor occlusions) have no real point support inside; their interiors are pure
    interpolation. Removing them prevents downstream rendering and normal estimation from being
    polluted by these synthetic faces.

    ``protected_triangle_mask`` exempts the marked triangles from the length test. Scope it
    narrowly: a broad exemption re-admits the hull-spanning faces this filter exists to remove.

    Returns ``(new_vertices, new_triangles, vertex_keep_mask)``. ``vertex_keep_mask`` indexes into
    the original ``vertices`` array and is intended for filtering parallel per-vertex attributes
    (e.g., colors) on the caller side.
    """
    n_vertices = len(vertices)
    if len(triangles) == 0:
        return vertices.copy(), triangles.copy(), np.ones(n_vertices, dtype=bool)
    if protected_triangle_mask is not None and len(protected_triangle_mask) != len(triangles):
        raise ValueError(
            f"protected_triangle_mask has {len(protected_triangle_mask)} entries for {len(triangles)} triangles"
        )

    tri_xy = vertices[triangles, :2]  # (T, 3, 2)
    e0 = tri_xy[:, 1] - tri_xy[:, 0]
    e1 = tri_xy[:, 2] - tri_xy[:, 1]
    e2 = tri_xy[:, 0] - tri_xy[:, 2]
    max_edge_per_tri = np.maximum(
        np.maximum(np.linalg.norm(e0, axis=1), np.linalg.norm(e1, axis=1)),
        np.linalg.norm(e2, axis=1),
    )
    keep_tri_mask = max_edge_per_tri < max_edge_length
    if protected_triangle_mask is not None:
        keep_tri_mask |= np.asarray(protected_triangle_mask, dtype=bool)
    kept_triangles = triangles[keep_tri_mask]

    if len(kept_triangles) == 0:
        return (
            np.zeros((0, vertices.shape[1]), dtype=vertices.dtype),
            kept_triangles.astype(triangles.dtype, copy=False),
            np.zeros(n_vertices, dtype=bool),
        )

    referenced = np.unique(kept_triangles.ravel())
    vertex_keep_mask = np.zeros(n_vertices, dtype=bool)
    vertex_keep_mask[referenced] = True

    old_to_new = -np.ones(n_vertices, dtype=np.int64)
    old_to_new[referenced] = np.arange(len(referenced))

    new_vertices = vertices[referenced].copy()
    new_triangles = old_to_new[kept_triangles].astype(triangles.dtype, copy=False)
    return new_vertices, new_triangles, vertex_keep_mask


def filter_triangles_by_max_normal_angle(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int32],
    max_angle_deg: float,
    axis: int = 2,
) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_]]:
    """Drop triangles whose face normal is too far from the elevation axis and compact vertices.

    Short BEV edges can still form near-vertical curtain triangles when road-segmented boundary
    cells contain curb, facade, vegetation, or other off-surface points. Unlike the long-edge
    filter, this catches steep local faces in both Delaunay and grid topology.
    """
    n_vertices = len(vertices)
    if len(triangles) == 0:
        return vertices.copy(), triangles.copy(), np.ones(n_vertices, dtype=bool)

    tri_vertices = vertices[triangles]
    normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    normal_norms = np.linalg.norm(normals, axis=1)
    axis_alignment = np.zeros_like(normal_norms)
    valid_normals = normal_norms > 1.0e-12
    axis_alignment[valid_normals] = np.abs(normals[valid_normals, axis]) / normal_norms[valid_normals]
    min_axis_alignment = np.cos(np.deg2rad(max_angle_deg))
    keep_tri_mask = valid_normals & (axis_alignment >= min_axis_alignment)
    kept_triangles = triangles[keep_tri_mask]

    if len(kept_triangles) == 0:
        return (
            np.zeros((0, vertices.shape[1]), dtype=vertices.dtype),
            kept_triangles.astype(triangles.dtype, copy=False),
            np.zeros(n_vertices, dtype=bool),
        )

    referenced = np.unique(kept_triangles.ravel())
    vertex_keep_mask = np.zeros(n_vertices, dtype=bool)
    vertex_keep_mask[referenced] = True

    old_to_new = -np.ones(n_vertices, dtype=np.int64)
    old_to_new[referenced] = np.arange(len(referenced))

    new_vertices = vertices[referenced].copy()
    new_triangles = old_to_new[kept_triangles].astype(triangles.dtype, copy=False)
    return new_vertices, new_triangles, vertex_keep_mask


def build_grid_constrained_mesh(
    vertices: NDArray[np.float32],
    cell_size: float,
) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_]]:
    """Build a 2.5D mesh by triangulating only adjacent occupied BEV grid cells.

    Unlike 2D Delaunay, this never fills the full convex hull of the input. Triangles are emitted
    only inside occupied 2x2 grid blocks, so large road gaps and separate road islands remain
    disconnected instead of being bridged by long synthetic faces.

    Returns ``(new_vertices, triangles, vertex_keep_mask)``. ``vertex_keep_mask`` indexes into the
    input ``vertices`` array and lets callers compact parallel per-vertex attributes.
    """
    n_vertices = len(vertices)
    if n_vertices < 3:
        return (
            np.zeros((0, vertices.shape[1]), dtype=vertices.dtype),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros(n_vertices, dtype=bool),
        )

    bbox_min_xy = vertices[:, :2].min(axis=0)
    cell_idx = np.round((vertices[:, :2] - bbox_min_xy) / cell_size).astype(np.int64)

    cell_to_vertex: dict[tuple[int, int], int] = {}
    for i in range(n_vertices):
        cell = (int(cell_idx[i, 0]), int(cell_idx[i, 1]))
        # The downsampler produces one vertex per cell. If a caller supplies duplicate cells
        # manually, keep the first one so triangulation remains deterministic.
        cell_to_vertex.setdefault(cell, i)

    triangles: List[List[int]] = []
    cells = set(cell_to_vertex)
    candidate_blocks = set()
    for cx, cy in cells:
        candidate_blocks.add((cx, cy))
        candidate_blocks.add((cx - 1, cy))
        candidate_blocks.add((cx, cy - 1))
        candidate_blocks.add((cx - 1, cy - 1))

    for cx, cy in sorted(candidate_blocks):
        corners = [
            cell_to_vertex.get((cx, cy)),
            cell_to_vertex.get((cx + 1, cy)),
            cell_to_vertex.get((cx, cy + 1)),
            cell_to_vertex.get((cx + 1, cy + 1)),
        ]
        present = [idx is not None for idx in corners]
        if sum(present) == 4:
            v00_full, v10_full, v01_full, v11_full = [idx for idx in corners if idx is not None]
            triangles.append([v00_full, v10_full, v11_full])
            triangles.append([v00_full, v11_full, v01_full])
        elif sum(present) == 3:
            # A single missing corner in a 2x2 block is a small local gap; emit the one supported
            # triangle. Larger gaps do not form any 2x2 block and are left open.
            v00_opt, v10_opt, v01_opt, v11_opt = corners
            if v00_opt is None:
                assert v10_opt is not None and v01_opt is not None and v11_opt is not None
                triangles.append([v10_opt, v11_opt, v01_opt])
            elif v10_opt is None:
                assert v00_opt is not None and v01_opt is not None and v11_opt is not None
                triangles.append([v00_opt, v11_opt, v01_opt])
            elif v01_opt is None:
                assert v00_opt is not None and v10_opt is not None and v11_opt is not None
                triangles.append([v00_opt, v10_opt, v11_opt])
            else:
                assert v00_opt is not None and v10_opt is not None and v01_opt is not None
                triangles.append([v00_opt, v10_opt, v01_opt])

    if len(triangles) == 0:
        return (
            np.zeros((0, vertices.shape[1]), dtype=vertices.dtype),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros(n_vertices, dtype=bool),
        )

    raw_triangles = np.array(triangles, dtype=np.int32)
    referenced = np.unique(raw_triangles.ravel())
    vertex_keep_mask = np.zeros(n_vertices, dtype=bool)
    vertex_keep_mask[referenced] = True

    old_to_new = -np.ones(n_vertices, dtype=np.int64)
    old_to_new[referenced] = np.arange(len(referenced))
    new_vertices = vertices[referenced].copy()
    new_triangles = old_to_new[raw_triangles].astype(np.int32, copy=False)
    return new_vertices, new_triangles, vertex_keep_mask


def _generate_ground_points_from_cuboids_unfiltered(
    cuboid_tracks: CuboidTracks,
    num_samples: int,
    cuboids_base_margin: float,
) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Sample ground points at the base of each cuboid pose in world frame and return the associated track index per sample."""
    unpacked = Tracks.Unpacked.from_data(cuboid_tracks.tracks_data)
    cuboids_dims = cuboid_tracks.cuboids_dims.cpu().numpy()  # (N_tracks, 3)

    samples_per_track: List[NDArray[np.float32]] = []
    track_indices_per_point: List[NDArray[np.int64]] = []
    for track_idx in range(len(unpacked.tracks_id)):
        cuboid_dims = cuboids_dims[track_idx]
        tquats = np.array(unpacked.tracks_poses[track_idx], dtype=np.float32)  # (n_poses, 7)
        if len(tquats) == 0:
            continue
        T_track_worlds = tquat_to_se3_matrix(tquats, unbatch=False).cpu().numpy()  # (n_poses, 4, 4)

        expanded_dims = cuboid_dims + np.array(
            [2 * cuboids_base_margin, 2 * cuboids_base_margin, 0.0], dtype=np.float32
        )
        base_samples_local = cuboid_base_grid_local(expanded_dims, num_samples)  # (n*n, 3)
        base_samples_world = [
            ((T[:3, :3] @ base_samples_local.T).T + T[:3, 3]).astype(np.float32) for T in T_track_worlds
        ]
        samples_per_track.append(np.concatenate(base_samples_world, axis=0))
        track_indices_per_point.append(
            np.full(len(T_track_worlds) * len(base_samples_local), track_idx, dtype=np.int64)
        )

    cuboid_ground_points = (
        np.concatenate(samples_per_track, axis=0) if len(samples_per_track) > 0 else np.empty((0, 3), dtype=np.float32)
    )
    track_indices = (
        np.concatenate(track_indices_per_point, axis=0)
        if len(track_indices_per_point) > 0
        else np.empty((0,), dtype=np.int64)
    )
    return cuboid_ground_points, track_indices


def generate_ground_points_from_cuboids(
    cuboid_tracks: CuboidTracks,
    scanned_ground_points: Optional[NDArray[np.float32]] = None,
    num_samples: int = 4,
    cuboids_base_margin: float = 0.5,
    exclude_points_radius: float = 1.0,
    exclude_points_knns: int = 1,
    cuboid_point_merge_cell_size: float = 0.5,
) -> NDArray[np.float32]:
    """Sample ground points at the base of each cuboid pose in world frame.

    1. Samples a num_samples x num_samples grid at the base rectangle of each cuboid in its local frame per pose.
    The base rectangle is enlarged by cuboids_base_margin on each side before sampling.
    Samples are transformed to world frame.
    2. Concatenates sample points over all cuboid poses.
    3. Discards cuboid ground points that have their k-th nearest scanned ground point within exclude_points_radius
       in the 2D top view, keeping only points in areas not covered by scanned data. If fewer than k scanned points
       are available, filtering is skipped because the k-th nearest scanned point does not exist.
    4. Merges remaining samples within each occupied XY grid cell by averaging their XYZ coordinates.

    Output points cover all areas covered by input cuboids, except areas already covered by scanned ground points.

    Args:
        cuboid_tracks: Cuboid tracks in world frame.
        scanned_ground_points: Dense set of scanned ground points (e.g. road points). Cuboid
            ground points whose k-th nearest scanned point is within exclude_points_radius in the
            XY projection are removed. If None, no filtering is applied.
        num_samples: Number of grid samples along each axis of the base rectangle.
        cuboids_base_margin: Per-side margin in meters added to the cuboid base before sampling.
        exclude_points_radius: XY-plane radius in meters; a cuboid ground point is excluded if its
            k-th nearest scanned point is within this radius.
        exclude_points_knns: Number of nearest scanned points to query per cuboid ground point (k). If fewer than k
            scanned points are available, filtering is skipped.
        cuboid_point_merge_cell_size: XY grid cell size in meters used to merge cuboid ground samples.

    Returns:
        Array of shape (N, 3) containing all sampled world-frame ground points (empty (0, 3)
        array if there are no tracks or poses).
    """
    num_poses = cuboid_tracks.tracks_poses.shape[0]
    logger.info("%d cuboid tracks and %d poses inbound", len(cuboid_tracks.tracks_id), num_poses)

    start_timestamp = perf_counter()
    cuboid_ground_points, _ = _generate_ground_points_from_cuboids_unfiltered(
        cuboid_tracks,
        num_samples,
        cuboids_base_margin,
    )
    logger.info(
        "%d ground points created (margin: %.2f m) from %d cuboid poses in %.3f seconds",
        len(cuboid_ground_points),
        cuboids_base_margin,
        num_poses,
        perf_counter() - start_timestamp,
    )

    if scanned_ground_points is not None and len(scanned_ground_points) > 0 and len(cuboid_ground_points) > 0:
        # Discard cuboid ground points where the k-th nearest scanned point is within exclude_points_radius in 2D BEV.
        # Only points in areas not covered by scanned data are kept.
        if len(scanned_ground_points) < exclude_points_knns:
            logger.warning(
                "Skipping cuboid ground point filtering: requested %d nearest scanned points, but only %d available",
                exclude_points_knns,
                len(scanned_ground_points),
            )
        else:
            kdtree = cKDTree(scanned_ground_points[:, :2])
            distances, _ = kdtree.query(cuboid_ground_points[:, :2], k=exclude_points_knns)
            # distances shape: (N,) for k=1, (N, k) for k>1. Reshape to (N, k) uniformly before taking the row-wise max.
            max_distances = distances.reshape(len(cuboid_ground_points), exclude_points_knns).max(axis=1)
            keep_mask = max_distances > exclude_points_radius
            num_removed = int((~keep_mask).sum())
            logger.info(
                "%d of %d (%.1f%%) cuboid ground points removed (k=%d nearest scan point within %.2f m); %d kept",
                num_removed,
                len(cuboid_ground_points),
                (num_removed / len(cuboid_ground_points)) * 100,
                exclude_points_knns,
                exclude_points_radius,
                int(keep_mask.sum()),
            )
            cuboid_ground_points = cuboid_ground_points[keep_mask]

    # Merge cuboid ground samples because they can be highly redundant due to nearby poses.
    # Instead of filtering or interpolating the poses with heuristics, points are sampled from all available poses
    # and points are merged per cell. This guarantees that points cover the same areas covered by cuboids, minus areas covered by scanned points.
    cuboid_ground_points_merged = downsample_3d_points_on_2d_grid_no_clamp(
        cuboid_ground_points, cuboid_point_merge_cell_size
    )
    logger.info(
        "Merged %d cuboid ground samples into %d occupied %.2f m grid cells",
        len(cuboid_ground_points),
        len(cuboid_ground_points_merged),
        cuboid_point_merge_cell_size,
    )
    return cuboid_ground_points_merged


def _try_load_ground_mesh_extension() -> Optional[Any]:
    """Return the loaded ``nurec.libs.ground_mesh`` CUDA extension, or ``None``."""
    if "extension" in _ground_mesh_extension_cache:
        return _ground_mesh_extension_cache["extension"]
    try:
        from nurec.libs.ground_mesh.interface import ground_mesh as ext  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised by environment
        logger.info(
            "nurec.libs.ground_mesh CUDA extension unavailable (%s); using Python ground-mesh fallback.",
            exc,
        )
        _ground_mesh_extension_cache["extension"] = None
        return None
    _ground_mesh_extension_cache["extension"] = ext
    return ext


@dataclass(frozen=True)
class _GridMeshCudaResult:
    """Mesh produced by the CUDA grid fast path.

    ``smooth_vertices`` is ``None`` when GPU smoothing did not run, i.e. the caller still has to smooth.
    """

    vertices: NDArray[np.float32]
    triangles: NDArray[np.int32]
    vertex_colors: Optional[NDArray[np.uint8]]
    smooth_vertices: Optional[NDArray[np.float32]]


def _try_build_grid_mesh_mean_cuda(
    points: NDArray[np.float32],
    colors: Optional[NDArray[np.uint8]],
    cell_size: float,
    min_points_per_cell: int,
    hole_fill_min_neighbors: int,
    smoothing_passes: int,
    elevation_axis: int,
) -> Optional[_GridMeshCudaResult]:
    """Build mean-aggregated grid mesh via CUDA when the extension is available."""
    if not torch.cuda.is_available():
        return None
    ext = _try_load_ground_mesh_extension()
    if ext is None:
        return None

    points_t = torch.as_tensor(np.ascontiguousarray(points, dtype=np.float32), device=torch.device("cuda"))
    colors_t = (
        torch.as_tensor(np.ascontiguousarray(colors, dtype=np.uint8), device=torch.device("cuda"))
        if colors is not None
        else None
    )
    vertices_t, triangles_t, vertex_colors_t = ext.build_grid_mesh_mean(
        points_t,
        colors_t,
        cell_size=float(cell_size),
        min_points_per_cell=int(min_points_per_cell),
        hole_fill_min_neighbors=int(hole_fill_min_neighbors),
    )
    smooth_vertices_t = None
    if smoothing_passes > 0 and len(triangles_t) > 0 and hasattr(ext, "smooth_mesh_along_axis"):
        edge_blocks = torch.cat(
            (
                triangles_t[:, [0, 1]],
                triangles_t[:, [1, 2]],
                triangles_t[:, [2, 0]],
            ),
            dim=0,
        ).to(dtype=torch.int64)
        edges_t = torch.sort(edge_blocks, dim=1).values
        edges_t = torch.unique(edges_t, dim=0)
        smooth_vertices_t = ext.smooth_mesh_along_axis(
            vertices_t,
            edges_t,
            axis=int(elevation_axis),
            smoothness=1.0,
            num_passes=int(smoothing_passes),
        )
    vertices = vertices_t.detach().cpu().numpy().astype(np.float32, copy=False)
    smooth_vertices = (
        smooth_vertices_t.detach().cpu().numpy().astype(np.float32, copy=False)
        if smooth_vertices_t is not None
        else None
    )
    triangles = triangles_t.detach().cpu().numpy().astype(np.int32, copy=False)
    vertex_colors = vertex_colors_t.detach().cpu().numpy().astype(np.uint8, copy=False) if colors is not None else None
    return _GridMeshCudaResult(
        vertices=vertices,
        triangles=triangles,
        vertex_colors=vertex_colors,
        smooth_vertices=smooth_vertices,
    )


@dataclass
class DelaunayElevationMeshingAlgorithm:
    """Mesh a terrain/elevation-type 2.5D point cloud that does not contain vertical or overhanging structures

    Produces a mesh from input elevation point cloud by performing the following steps:
    1. Downsample the point cloud on a 2D grid laid over the XY plane (Bird's Eye View, BEV).
    2. Construct a Delaunay 2D mesh on the input points in the BEV (XY components of the points) and
       lift the 2D vertices to the 3D points in Z, so the mesh vertices become the downsampled points.
       A simple and relatively fast method to produce a watertight mesh (no holes) over elevation data.
       The meshed domain is the convex hull of the input points in the BEV.
       Noise and gross outlier (off-surface) points in the downsampled point cloud directly appear in the mesh.
       For this reason, the input is assumed to be free of gross outliers, otherwise they cause spikes in the mesh.
    3. A mesh smoothing is applied along the z-direction only to denoise the vertices as a post-processing step.
    """

    # TODO: Replace direct alignment of vertices to points with a linear least squares minimization of a fitting
    # and a smoothness term. This would allow to add extra points to the 2D mesh whose elevation is unknown,
    # e.g. extend the mesh smoothly at the edges to cover a larger area.

    enable_downsampling: bool = True
    voxel_size: float = 1.0
    min_points_per_voxel: int = 1
    grid_hole_fill_min_neighbors: int = 6
    smoothing_passes: int = 1
    elevation_axis: int = 2
    mesh_topology: Literal["delaunay", "grid"] = "delaunay"
    smoothing_method: Literal["neighbor_average", "local_plane"] = "neighbor_average"
    local_plane_min_neighbors: int = 3
    local_plane_include_self: bool = False

    # Robustness knobs. Defaults preserve legacy behavior (mean aggregation, no extra filtering) so existing
    # tests and external callers are unaffected. The production camera-ground-mesh workflow enables
    # these via YAML config.
    height_statistic: Literal["mean", "median"] = "mean"
    color_statistic: Literal["mean", "median"] = "mean"
    outlier_height_threshold: Optional[float] = None  # None=disabled; e.g. 0.35m to drop spike cells
    outlier_neighbor_radius: int = 1
    outlier_min_neighbors: int = 4
    max_triangle_edge_length: Optional[float] = None  # None=disabled; e.g. 1.0m to drop bridging tris
    max_triangle_normal_angle_deg: Optional[float] = None  # None=disabled; e.g. 60 deg drops vertical curtains

    # Called with (vertices, triangles) after triangulation; returns a per-triangle bool mask of
    # faces to exempt from the long-edge filter. A callable, not a precomputed mask, because those
    # arrays only exist inside build_mesh_from_points. None means no exemption.
    protected_triangle_mask_fn: Optional[Callable[[NDArray[np.floating], NDArray[np.integer]], NDArray[np.bool_]]] = (
        None
    )

    def __post_init__(self) -> None:
        if self.mesh_topology not in ("delaunay", "grid"):
            raise ValueError(f"Unsupported mesh_topology={self.mesh_topology!r}; expected 'delaunay' or 'grid'")
        if self.smoothing_method not in ("neighbor_average", "local_plane"):
            raise ValueError(
                f"Unsupported smoothing_method={self.smoothing_method!r}; expected 'neighbor_average' or 'local_plane'"
            )
        if self.local_plane_min_neighbors < 1:
            raise ValueError("local_plane_min_neighbors must be positive")
        if not (0 <= self.grid_hole_fill_min_neighbors <= 8):
            raise ValueError("grid_hole_fill_min_neighbors must be in [0, 8]")
        if self.height_statistic not in ("mean", "median"):
            raise ValueError(f"Unsupported height_statistic={self.height_statistic!r}; expected 'mean' or 'median'")
        if self.color_statistic not in ("mean", "median"):
            raise ValueError(f"Unsupported color_statistic={self.color_statistic!r}; expected 'mean' or 'median'")
        if self.max_triangle_normal_angle_deg is not None and not (0.0 < self.max_triangle_normal_angle_deg <= 90.0):
            raise ValueError("max_triangle_normal_angle_deg must be in (0, 90], or None to disable")

    def _try_build_grid_topology_cuda(
        self,
        points: NDArray[np.float32],
        colors: Optional[NDArray[np.uint8]],
    ) -> Optional[_GridMeshCudaResult]:
        """Build downsampling and grid topology on the GPU, or return ``None`` if that is not applicable."""
        if not (
            self.enable_downsampling
            and self.mesh_topology == "grid"
            and self.height_statistic == "mean"
            and self.color_statistic == "mean"
            and self.outlier_height_threshold is None
        ):
            return None

        logger.info(
            f"CUDA grid meshing candidate over {len(points)} points "
            f"(cell_size={self.voxel_size}, min_points_per_cell={self.min_points_per_voxel}, "
            f"hole_fill_min_neighbors={self.grid_hole_fill_min_neighbors})"
        )
        start_time = perf_counter()
        cuda_grid_result = _try_build_grid_mesh_mean_cuda(
            points,
            colors=colors,
            cell_size=self.voxel_size,
            min_points_per_cell=self.min_points_per_voxel,
            hole_fill_min_neighbors=self.grid_hole_fill_min_neighbors,
            smoothing_passes=self.smoothing_passes if self.smoothing_method == "neighbor_average" else 0,
            elevation_axis=self.elevation_axis,
        )
        elapsed_time = perf_counter() - start_time
        if cuda_grid_result is None:
            return None
        logger.info(
            f"CUDA grid meshing produced {len(cuda_grid_result.vertices)} vertices and "
            f"{len(cuda_grid_result.triangles)} triangles in {elapsed_time:.3f} s"
        )
        return cuda_grid_result

    def build_mesh_from_points(
        self,
        points: NDArray[np.float32],
        colors: Optional[NDArray[np.uint8]] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], Optional[NDArray[np.uint8]]]:
        # Effective point cloud denoising and reduction of the number of points by orders of magnitude
        # to make 2D Delaunay meshing run faster.
        # 2D grid-based downsampling replaces voxel-based downsampling via pcu.downsample_point_cloud_on_voxel_grid()
        # because it produces a better quality mesh, free of the Z-quantization artifacts of the latter approach.
        # The additional regularization effect of clamping output points to cell-centers is beneficial
        # for smoothing algorithms that are not normalizing by vertex density.
        # A ``None`` value means no valid smoothed vertices are available yet, so smoothing still has to run.
        smooth_vertices: Optional[NDArray[np.float32]] = None
        cuda_grid_result = self._try_build_grid_topology_cuda(points, colors)
        if cuda_grid_result is not None:
            vertices = cuda_grid_result.vertices
            triangles = cuda_grid_result.triangles
            vertex_colors = cuda_grid_result.vertex_colors
            smooth_vertices = cuda_grid_result.smooth_vertices

        if cuda_grid_result is None:
            logger.info(f"Downsampling {len(points)} points on voxels (cell_size={self.voxel_size})")
            start_time = perf_counter()
            if self.enable_downsampling:
                vertices, vertex_colors = downsample_3d_points_on_2d_grid(
                    points,
                    self.voxel_size,
                    colors=colors,
                    height_statistic=self.height_statistic,
                    color_statistic=self.color_statistic,
                    min_points_per_cell=self.min_points_per_voxel,
                )
            else:
                vertices, vertex_colors = points, colors
            elapsed_time = perf_counter() - start_time
            logger.info(f"Downsampling produced {len(vertices)} points in {elapsed_time:.3f} s")

        # Local height outlier filter: drop cells whose Z deviates strongly from BEV neighbors.
        if self.enable_downsampling and self.outlier_height_threshold is not None and len(vertices) > 0:
            keep_mask = filter_local_height_outliers(
                vertices,
                cell_size=self.voxel_size,
                threshold=self.outlier_height_threshold,
                min_neighbors=self.outlier_min_neighbors,
                radius=self.outlier_neighbor_radius,
            )
            n_dropped = int((~keep_mask).sum())
            if n_dropped > 0:
                logger.info(
                    f"Local height outlier filter dropped {n_dropped} / {len(vertices)} vertices "
                    f"(threshold={self.outlier_height_threshold}m, radius={self.outlier_neighbor_radius})"
                )
            vertices = vertices[keep_mask]
            if vertex_colors is not None:
                vertex_colors = vertex_colors[keep_mask]

        # Degenerate guard: Delaunay requires at least 3 non-collinear points. The count check
        # below catches "too few"; the QhullError catch below covers the harder case where >= 3
        # remaining points are collinear or nearly so (can happen when outlier / min-points
        # filters prune the cloud down to a narrow strip).
        if len(vertices) < 3:
            logger.warning(
                f"Too few vertices ({len(vertices)}) for {self.mesh_topology} meshing; returning empty mesh."
            )
            empty_triangles = np.zeros((0, 3), dtype=np.int32)
            return (
                vertices.copy(),
                empty_triangles,
                vertices.copy(),
                vertex_colors.copy() if vertex_colors is not None else None,
            )

        if cuda_grid_result is None:
            start_time = perf_counter()
            if self.mesh_topology == "delaunay":
                logger.info(f"2D Delaunay triangulation over {len(vertices)} points")
                try:
                    triangles = Delaunay(vertices[:, 0:2]).simplices.astype(np.int32, copy=False)
                except QhullError as exc:
                    logger.warning(
                        f"Delaunay triangulation failed on {len(vertices)} vertices "
                        f"(likely collinear or degenerate XY layout: {exc}); returning empty mesh."
                    )
                    empty_vertices = np.zeros((0, vertices.shape[1]), dtype=vertices.dtype)
                    empty_triangles = np.zeros((0, 3), dtype=np.int32)
                    empty_colors = (
                        np.zeros((0, vertex_colors.shape[1]), dtype=vertex_colors.dtype)
                        if vertex_colors is not None
                        else None
                    )
                    return empty_vertices, empty_triangles, empty_vertices.copy(), empty_colors
            else:
                logger.info(f"Grid-constrained triangulation over {len(vertices)} points")
                new_vertices, triangles, vertex_keep_mask = build_grid_constrained_mesh(vertices, self.voxel_size)
                vertices = new_vertices
                if vertex_colors is not None:
                    vertex_colors = vertex_colors[vertex_keep_mask]
            elapsed_time = perf_counter() - start_time
            logger.info(
                f"{self.mesh_topology} triangulation produced {len(triangles)} triangles in {elapsed_time:.3f} s"
            )

        # Drop bridging long triangles (e.g., convex-hull edges across non-road regions or gaps).
        if self.mesh_topology == "delaunay" and self.max_triangle_edge_length is not None and len(triangles) > 0:
            protected_mask = None
            if self.protected_triangle_mask_fn is not None:
                protected_mask = self.protected_triangle_mask_fn(vertices, triangles)
                if len(protected_mask) != len(triangles):
                    raise ValueError(
                        f"protected_triangle_mask_fn returned {len(protected_mask)} entries "
                        f"for {len(triangles)} triangles"
                    )
                logger.info(
                    f"Long-edge filter: {int(np.asarray(protected_mask).sum())} / {len(triangles)} "
                    "triangles exempted by protected_triangle_mask_fn"
                )
            new_vertices, new_triangles, vertex_keep_mask = filter_triangles_by_max_edge_length(
                vertices, triangles, self.max_triangle_edge_length, protected_triangle_mask=protected_mask
            )
            n_dropped_tri = len(triangles) - len(new_triangles)
            n_dropped_v = int((~vertex_keep_mask).sum())
            if len(new_triangles) == 0 or len(new_vertices) < 3:
                logger.warning(
                    f"Long-edge triangle filter would empty mesh "
                    f"({len(vertices)} vertices, {len(triangles)} triangles); keeping unfiltered mesh."
                )
            else:
                if n_dropped_tri > 0 or n_dropped_v > 0:
                    logger.info(
                        f"Long-edge triangle filter dropped {n_dropped_tri} / {len(triangles)} triangles "
                        f"and {n_dropped_v} unreferenced vertices "
                        f"(max_edge={self.max_triangle_edge_length}m)"
                    )
                vertices = new_vertices
                triangles = new_triangles
                if vertex_colors is not None:
                    vertex_colors = vertex_colors[vertex_keep_mask]

        # Drop near-vertical faces, e.g. road-edge curtains from short XY edges with large Z jumps.
        if self.max_triangle_normal_angle_deg is not None and len(triangles) > 0:
            new_vertices, new_triangles, vertex_keep_mask = filter_triangles_by_max_normal_angle(
                vertices, triangles, self.max_triangle_normal_angle_deg, axis=self.elevation_axis
            )
            n_dropped_tri = len(triangles) - len(new_triangles)
            n_dropped_v = int((~vertex_keep_mask).sum())
            if len(new_triangles) == 0 or len(new_vertices) < 3:
                logger.warning(
                    f"Normal-angle triangle filter would empty mesh "
                    f"({len(vertices)} vertices, {len(triangles)} triangles); keeping unfiltered mesh."
                )
            else:
                if n_dropped_tri > 0 or n_dropped_v > 0:
                    logger.info(
                        f"Normal-angle triangle filter dropped {n_dropped_tri} / {len(triangles)} triangles "
                        f"and {n_dropped_v} unreferenced vertices "
                        f"(max_angle={self.max_triangle_normal_angle_deg}deg)"
                    )
                vertices = new_vertices
                triangles = new_triangles
                if vertex_colors is not None:
                    vertex_colors = vertex_colors[vertex_keep_mask]
                # TODO: Refactor the grid path so topology-changing filters run before any backend
                # smoothing, or otherwise make the backend boundary explicit. This filter compacts
                # vertices after CUDA smoothing may already have run, so GPU-smoothed heights still
                # index the pre-filter vertex array and must be invalidated.
                smooth_vertices = None

        # Degenerate guard: smoothing / downstream consumers cannot handle empty meshes.
        if len(triangles) == 0 or len(vertices) < 3:
            logger.warning(
                f"Mesh empty after triangle filtering "
                f"({len(vertices)} vertices, {len(triangles)} triangles); returning empty mesh."
            )
            return (
                vertices.copy(),
                triangles,
                vertices.copy(),
                vertex_colors.copy() if vertex_colors is not None else None,
            )

        # Only want to smooth mesh vertices along the vertical direction.
        # Laplacian smoothing is not suitable as it also moves vertices parallel to the surface
        # and has a well known shrinking bias.
        if smooth_vertices is None:
            smooth_vertices = vertices
            if self.smoothing_passes > 0:
                logger.info(f"Mesh smoothing over {len(vertices)} vertices ({self.smoothing_passes} passes)")
                start_time = perf_counter()
                if self.smoothing_method == "neighbor_average":
                    smooth_vertices = smooth_mesh_along_axis_torch(
                        vertices,
                        triangles,
                        axis=self.elevation_axis,
                        num_passes=self.smoothing_passes,
                    )
                else:
                    logger.info(f"Collecting vertex neighbors for {len(vertices)} vertices")
                    neighbor_start_time = perf_counter()
                    vertex_neighbors, vertex_distances = get_vertex_neighbors(vertices, triangles)
                    neighbor_elapsed_time = perf_counter() - neighbor_start_time
                    logger.info(f"Collected vertex neighbors in {neighbor_elapsed_time:.3f} s")
                    for _ in range(self.smoothing_passes):
                        smooth_vertices = mesh_smoothing_along_axis(
                            smooth_vertices,
                            vertex_neighbors,
                            vertex_distances,
                            axis=self.elevation_axis,
                            method=self.smoothing_method,
                            local_plane_min_neighbors=self.local_plane_min_neighbors,
                            local_plane_include_self=self.local_plane_include_self,
                        )
                elapsed_time = perf_counter() - start_time
                logger.info(f"Mesh smoothing took {elapsed_time:.3f} s")

        # Smoothing changes vertex heights but not topology. Re-apply the steep-face guard on
        # the final smoothed mesh so road-edge curtains introduced or exposed by smoothing do
        # not reach Gaussian initialization / artifact export.
        if self.smoothing_passes > 0 and self.max_triangle_normal_angle_deg is not None and len(triangles) > 0:
            new_smooth_vertices, new_triangles, vertex_keep_mask = filter_triangles_by_max_normal_angle(
                smooth_vertices, triangles, self.max_triangle_normal_angle_deg, axis=self.elevation_axis
            )
            n_dropped_tri = len(triangles) - len(new_triangles)
            n_dropped_v = int((~vertex_keep_mask).sum())
            if len(new_triangles) == 0 or len(new_smooth_vertices) < 3:
                logger.warning(
                    f"Post-smoothing normal-angle triangle filter would empty mesh "
                    f"({len(smooth_vertices)} vertices, {len(triangles)} triangles); keeping unfiltered mesh."
                )
            else:
                if n_dropped_tri > 0 or n_dropped_v > 0:
                    logger.info(
                        f"Post-smoothing normal-angle triangle filter dropped {n_dropped_tri} / {len(triangles)} "
                        f"triangles and {n_dropped_v} unreferenced vertices "
                        f"(max_angle={self.max_triangle_normal_angle_deg}deg)"
                    )
                smooth_vertices = new_smooth_vertices
                triangles = new_triangles
                vertices = vertices[vertex_keep_mask]
                if vertex_colors is not None:
                    vertex_colors = vertex_colors[vertex_keep_mask]

        return smooth_vertices, triangles, vertices, vertex_colors


def _save_image(image_path: str, image: PILImage.Image) -> None:
    logger.info(f"Saving {image_path}")
    image.save(image_path)


def get_mesh_edges(triangles: NDArray[np.int32]) -> NDArray[np.int32]:
    """Return a unique list of undirected edges on a mesh"""
    v0 = triangles[:, [0]]
    v1 = triangles[:, [1]]
    v2 = triangles[:, [2]]
    edges = np.vstack([np.hstack([v0, v1]), np.hstack([v1, v2]), np.hstack([v2, v0])])
    assert edges.shape == (3 * triangles.shape[0], 2)
    assert edges.dtype == triangles.dtype
    # Sort edges and make them unique
    swap = edges[:, 0] > edges[:, 1]
    edges[swap, :] = np.flip(edges[swap, :], axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def calculate_edge_lengths(vertices: NDArray[np.float32], edges: NDArray[np.int32]) -> NDArray[np.float32]:
    assert vertices.shape[1] == 3
    assert edges.shape[1] == 2
    assert np.max(edges) < vertices.shape[0]
    return np.linalg.norm(vertices[edges[:, 0], :] - vertices[edges[:, 1], :], ord=2, axis=1)


def get_vertex_neighbors(
    vertices: NDArray[np.float32], triangles: NDArray[np.int32]
) -> Tuple[List[List[int]], List[List[float]]]:
    """List all adjacent vertices (1-ring neighbors) of every vertex

    Returns the list of indices of neighbors and the list of neighbor distances for every vertex.
    """
    num_vertices = len(vertices)
    assert vertices.shape[1] == 3
    assert triangles.shape[1] == 3
    neighbors: List[List[int]] = [[] for _ in range(num_vertices)]
    distances: List[List[float]] = [[] for _ in range(num_vertices)]
    if len(triangles) == 0:
        # Empty mesh after triangle filtering: every vertex is isolated.
        return neighbors, distances
    assert np.max(triangles) < num_vertices
    edges = get_mesh_edges(triangles)
    edge_lengths = calculate_edge_lengths(vertices, edges)
    for edge_idx in range(len(edges)):
        i, j = edges[edge_idx, :]
        edge_length = edge_lengths[edge_idx]
        neighbors[i].append(j)
        neighbors[j].append(i)
        distances[i].append(edge_length)
        distances[j].append(edge_length)
    return neighbors, distances


def _weighted_neighbor_average_height(
    vertices: NDArray[np.float32],
    neighbor_indices: List[int],
    neighbor_distances: List[float],
    axis: int,
) -> float:
    neighbor_distances_np = np.array(neighbor_distances, dtype=np.float32)
    neighbor_heights = vertices[neighbor_indices, axis]
    neighbor_weights = 1.0 / np.maximum(neighbor_distances_np, 1.0e-8)
    return float(np.average(neighbor_heights, weights=neighbor_weights))


def _local_plane_projected_height(
    vertices: NDArray[np.float32],
    vertex_idx: int,
    neighbor_indices: List[int],
    neighbor_distances: List[float],
    axis: int,
    min_neighbors: int,
    include_self: bool,
) -> Optional[float]:
    """Fit a local plane to neighbors and project ``vertex_idx`` onto it along ``axis``."""
    if len(neighbor_indices) < min_neighbors:
        return None

    fit_indices = list(neighbor_indices)
    weights = 1.0 / np.maximum(np.array(neighbor_distances, dtype=np.float32), 1.0e-8)
    if include_self:
        fit_indices.append(vertex_idx)
        weights = np.concatenate([weights, np.array([weights.max() if len(weights) > 0 else 1.0], dtype=np.float32)])

    fixed_axes = [0, 1, 2]
    fixed_axes.remove(axis)
    fit_points = vertices[fit_indices]
    A = np.column_stack([fit_points[:, fixed_axes[0]], fit_points[:, fixed_axes[1]], np.ones(len(fit_indices))])
    if np.linalg.matrix_rank(A) < 3:
        return None

    sqrt_weights = np.sqrt(weights).reshape(-1, 1)
    weighted_A = A * sqrt_weights
    weighted_b = fit_points[:, axis] * sqrt_weights[:, 0]
    coeffs, *_ = np.linalg.lstsq(weighted_A, weighted_b, rcond=None)
    target = vertices[vertex_idx]
    return float(coeffs[0] * target[fixed_axes[0]] + coeffs[1] * target[fixed_axes[1]] + coeffs[2])


def mesh_smoothing_along_axis(
    vertices: NDArray[np.float32],
    vertex_neighbors: List[List[int]],
    vertex_distances: List[List[float]],
    axis: int = 2,
    smoothness: float = 1.0,
    method: Literal["neighbor_average", "local_plane"] = "neighbor_average",
    local_plane_min_neighbors: int = 3,
    local_plane_include_self: bool = False,
) -> NDArray[np.float32]:
    """Smooth a mesh along a single axis, in a single pass.

    Avoids the mesh shrinking bias of Laplacian smoothing, and keeps the vertices locked in the birds eye view.
    The legacy method moves each vertex towards the weighted average height of the 1-ring neighbors.
    The ``local_plane`` method regresses a local plane on the 1-ring and vertically projects the
    current vertex to that plane, preserving local slope better than height averaging.
    The amount of smoothness is controlled by a smoothness parameter linearly between the original height
    (smoothness=0.0) and the target height (smoothness=1.0).
    """
    num_vertices = len(vertices)
    assert vertices.shape[1] == 3
    assert len(vertex_neighbors) == num_vertices
    assert len(vertex_distances) == num_vertices
    if method not in ("neighbor_average", "local_plane"):
        raise ValueError(f"Unsupported smoothing method {method!r}; expected 'neighbor_average' or 'local_plane'")
    target_heights = np.empty((num_vertices,), dtype=np.float32)

    for i in range(num_vertices):
        neighbor_indices = vertex_neighbors[i]
        if len(neighbor_indices) == 0:
            # Isolated vertex (e.g., after long-edge triangle filter removed every incident
            # triangle). Keep its original height instead of averaging an empty neighborhood.
            target_heights[i] = vertices[i, axis]
            continue

        if method == "local_plane":
            projected_height = _local_plane_projected_height(
                vertices,
                vertex_idx=i,
                neighbor_indices=neighbor_indices,
                neighbor_distances=vertex_distances[i],
                axis=axis,
                min_neighbors=local_plane_min_neighbors,
                include_self=local_plane_include_self,
            )
            if projected_height is not None:
                target_heights[i] = projected_height
                continue

        # Reduce the influence of distant neighbors. Works much better than simple average
        # (a.k.a. mean curvature flow / Laplacian smoothing with uniform weights).
        target_heights[i] = _weighted_neighbor_average_height(
            vertices, neighbor_indices, vertex_distances[i], axis=axis
        )

    smooth_vertices = vertices.copy()
    smooth_vertices[:, axis] = target_heights * smoothness + vertices[:, axis] * (1.0 - smoothness)
    return smooth_vertices


def _smooth_mesh_along_axis_numpy(
    vertices: NDArray[np.float32],
    edges: NDArray[np.int32],
    axis: int,
    smoothness: float,
    num_passes: int,
) -> NDArray[np.float32]:
    """CPU-native vectorized implementation using ``np.bincount`` for the scatter.

    Faster than eager-mode torch on CPU (no per-op Python overhead, no tensor
    allocation per pass). PyTorch wins on GPU, numpy wins on CPU.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    num_vertices = v.shape[0]
    i = edges[:, 0].astype(np.int64)
    j = edges[:, 1].astype(np.int64)

    # Edge weights are computed once on the initial vertex positions and reused
    # across all passes.
    diff = v[i] - v[j]
    dist = np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)
    w = (1.0 / dist).astype(np.float32)  # [E]

    # Symmetric scatter: each undirected edge contributes its weight at both endpoints.
    indices = np.concatenate([i, j])
    weights_concat = np.concatenate([w, w])
    weight_sum = np.bincount(indices, weights=weights_concat, minlength=num_vertices).astype(np.float32)
    connected = weight_sum > 0

    heights = v[:, axis].astype(np.float32, copy=True)
    for _ in range(num_passes):
        # Weighted average of neighbor heights via bincount.
        values_to_i = w * heights[j]
        values_to_j = w * heights[i]
        values = np.concatenate([values_to_i, values_to_j])
        weighted_sum = np.bincount(indices, weights=values, minlength=num_vertices).astype(np.float32)
        target = heights.copy()
        target[connected] = weighted_sum[connected] / weight_sum[connected]
        heights = target * smoothness + heights * (1.0 - smoothness)

    out = v.copy()
    out[:, axis] = heights
    return out


def _smooth_mesh_along_axis_torch_cuda(
    vertices: NDArray[np.float32],
    edges: NDArray[np.int32],
    axis: int,
    smoothness: float,
    num_passes: int,
    device: "torch.device",
) -> NDArray[np.float32]:
    """CUDA path. Always do this on GPU. Calling on CPU is wasteful vs. numpy."""
    ext = _try_load_ground_mesh_extension()
    if ext is not None:
        vertices_t = torch.as_tensor(np.ascontiguousarray(vertices, dtype=np.float32), device=device)
        edges_t = torch.as_tensor(np.ascontiguousarray(edges, dtype=np.int64), device=device)
        return (
            ext.smooth_mesh_along_axis(
                vertices_t,
                edges_t,
                axis=int(axis),
                smoothness=float(smoothness),
                num_passes=int(num_passes),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    v_np = np.ascontiguousarray(vertices, dtype=np.float32)
    num_vertices = v_np.shape[0]

    v_t = torch.from_numpy(v_np).to(device)
    e = torch.from_numpy(edges.astype(np.int64)).to(device)
    i = e[:, 0]
    j = e[:, 1]

    diff = v_t[i] - v_t[j]
    dist = torch.linalg.norm(diff, dim=1)
    w = 1.0 / dist

    scatter_index = torch.cat([i, j])
    neighbors = torch.cat([j, i])
    # Sort by destination, then neighbor index, to fix each vertex's reduction order.
    order = torch.argsort(scatter_index * num_vertices + neighbors)
    neighbors = neighbors[order]
    weights = torch.cat([w, w])[order]
    counts = torch.bincount(scatter_index, minlength=num_vertices)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    # Offsets delimit each destination vertex's sorted neighbor range.
    weight_sum = torch.segment_reduce(weights, "sum", offsets=offsets)
    connected = weight_sum > 0

    heights = v_t[:, axis].clone()
    for _ in range(num_passes):
        weighted = torch.segment_reduce(weights * heights[neighbors], "sum", offsets=offsets)
        target = heights.clone()
        target[connected] = weighted[connected] / weight_sum[connected]
        heights = target * smoothness + heights * (1.0 - smoothness)

    out = v_t.clone()
    out[:, axis] = heights
    return out.detach().cpu().numpy()


def smooth_mesh_along_axis_torch(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int32],
    axis: int = 2,
    smoothness: float = 1.0,
    num_passes: int = 1,
    device: Optional["torch.device"] = None,
) -> NDArray[np.float32]:
    """Smooth a mesh along a single axis, ``num_passes`` times.

    Each vertex's coordinate along ``axis`` moves towards the inverse-distance-
    weighted average of its 1-ring neighbor coordinates, blended with the
    original value by ``smoothness`` (0 = no change, 1 = full average). Edge
    weights are pre-computed once from the initial vertex positions.

    Auto-selects the fastest backend for the host:

    - **CUDA available**: fixed-order neighbor reductions on GPU, using the
      CUDA extension when available or sorted PyTorch segments otherwise.
    - **CPU only**: ``np.bincount``, a C-level reduction that beats eager-mode
      torch on CPU (per-op Python overhead drowns out the parallel scatter benefit).
    """
    edges_np = get_mesh_edges(triangles)

    if device is None:
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = device.type == "cuda"

    if use_cuda:
        if device is None:
            device = torch.device("cuda")
        return _smooth_mesh_along_axis_torch_cuda(vertices, edges_np, axis, smoothness, num_passes, device)
    return _smooth_mesh_along_axis_numpy(vertices, edges_np, axis, smoothness, num_passes)


def get_nominal_ground_point_under_lidar(
    datasource: NCOREDataSource,
    rig_z_offset: float = 0.0,
    lidar_id: str | None = None,
) -> NDArray[np.float32]:
    """Return a nominal ground control point in the selected LiDAR's reference frame.

    ``rig_z_offset`` is the ground height in the ego/rig frame. The returned point lies on the vertical
    line through the lidar at height ``rig_z_offset - sensor_height``, where ``sensor_height`` is the
    lidar's Z translation in the rig frame. If ``lidar_id`` is omitted, this matches ``get_point_clouds()``
    defaulting to the first sensor in sorted ``all_lidar_ids`` order.
    """

    # TODO: Add missing sequence_id, sensor_id and lidar_frame_idx to PointCloud when it contains a single Lidar frame.
    # Until then, using the simplifying assumptions below.

    dataset_lidar_sensor_ids = list(datasource.lidar_sensors)
    if not dataset_lidar_sensor_ids:
        raise ValueError("At least one lidar is required to compute a nominal ground point")
    if lidar_id is None:
        lidar_id = dataset_lidar_sensor_ids[0]
    elif lidar_id not in datasource.lidar_sensors:
        raise ValueError(f"Unknown lidar id '{lidar_id}'")
    lidar_sensor = datasource.lidar_sensors[lidar_id]
    sensor_to_rig_transf = unpack_optional(lidar_sensor.T_sensor_rig)
    # The sensor position in the rig frame is in the last column if and only if the transformation is affine.
    if not np.allclose(sensor_to_rig_transf[3, 0:4], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("get_T_sensor_rig() does not return an affine transformation")
    sensor_height = sensor_to_rig_transf[2, 3]
    ground_point = np.array([0, 0, rig_z_offset - sensor_height], dtype=np.float32)
    logger.info(
        f"Nominal ground point in a sensor-centric reference frame for lidar '{lidar_id}': {ground_point.tolist()} "
    )
    return ground_point


def xyz_to_spherical(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the point cloud to spherical coordinates, assuming that points are centered and the z axis points up"""
    assert points.shape[1] == 3
    radii = np.linalg.norm(points, axis=1)
    azimuth = np.arctan2(points[:, 1], points[:, 0])  # Ranges [-pi, +pi]
    elevation = np.arcsin(points[:, 2] / radii)
    return azimuth, elevation, radii


def _auto_detect_bin_counts(azimuth: NDArray[np.float32], elevation: NDArray[np.float32]) -> Tuple[int, int]:
    """Pick scan-grid-matched bin counts from the data.

    Spinning LiDARs emit returns at a fixed set of elevation angles (one
    per laser channel) and a fixed set of azimuth steps per revolution.
    Counting the unique quantized values in the input gives the natural
    grid size; oversized bins would scatter neighbouring returns into
    non-adjacent cells and starve the quad-emission step of full quads.

    Quantization tolerance: ``elevation`` is rounded to ``1e-4 rad``
    (~0.006°), which suppresses float-arithmetic noise while preserving
    distinct laser channels. ``azimuth`` is rounded to ``1e-3 rad``
    (~0.057°); the looser tolerance reflects realistic motion-compensation
    jitter on per-azimuth samples.

    Counts are clamped to ``[8, 512]`` (elevation) and ``[128, 8192]``
    (azimuth) to keep the downstream occupancy buffer to a sane size on
    pathological inputs.
    """
    n_el = int(np.unique(np.round(elevation * 1e4).astype(np.int64)).shape[0])
    n_el = int(np.clip(n_el, 8, 512))
    n_az = int(np.unique(np.round(azimuth * 1e3).astype(np.int64)).shape[0])
    n_az = int(np.clip(n_az, 128, 8192))
    return n_el, n_az


def construct_lidar_spin_mesh(
    centered_points: np.ndarray,
    n_elevation_bins: Optional[int] = None,
    n_azimuth_bins: Optional[int] = None,
):
    """Construct a triangle mesh over a set of input lidar points scanned from the origin.

    Recovers mesh topology from the LiDAR scan pattern: in
    ``(azimuth, elevation)`` space the returns lie on a regular 2D grid
    (one row per laser channel, one column per azimuth sample). The
    implementation quantizes the (az, el) plane to
    ``n_elevation_bins × n_azimuth_bins`` cells, picks a representative
    point per occupied cell, and emits two triangles per fully-occupied
    quad in that grid.

    Bin counts default to ``None``, which triggers an auto-detection pass
    (``_auto_detect_bin_counts``) that sizes the grid to the input's scan
    structure. Explicit values can be supplied to override the heuristic.
    Cells that receive zero points are simply absent from the mesh —
    holes in the topology are correct for dropped LiDAR returns and the
    downstream vertex-normal estimator falls back to a default normal for
    unconnected vertices.

    Returns:
        ``(triangles, azimuth_elevation)`` where ``triangles`` are int32
        vertex-index triplets into ``centered_points`` and
        ``azimuth_elevation`` is the per-point ``(az, el, 0)`` array
        retained for downstream diagnostics.
    """
    azimuth, elevation, _ = xyz_to_spherical(centered_points)
    if n_elevation_bins is None or n_azimuth_bins is None:
        auto_n_el, auto_n_az = _auto_detect_bin_counts(azimuth, elevation)
        if n_elevation_bins is None:
            n_elevation_bins = auto_n_el
        if n_azimuth_bins is None:
            n_azimuth_bins = auto_n_az
    azimuth_elevation = np.stack([azimuth, elevation, np.zeros_like(azimuth)], axis=1)
    triangles = _scanline_triangulate(azimuth, elevation, n_elevation_bins, n_azimuth_bins)
    return triangles, azimuth_elevation


def _scanline_triangulate(
    azimuth: NDArray[np.float32],
    elevation: NDArray[np.float32],
    n_elevation_bins: int,
    n_azimuth_bins: int,
) -> NDArray[np.int32]:
    """Vectorized scanline triangulation in (azimuth, elevation) space.

    - Azimuth bin range is fixed at ``[-π, π]`` (the canonical
      ``np.arctan2`` output range).
    - Elevation bin range is taken from the data extents on the assumption
      that the input represents a single LiDAR spin (a few tens of degrees
      vertical FOV in practice).
    - For cells receiving more than one point, the *lowest-indexed* point is
      kept as the cell's representative. The other points remain as mesh
      vertices but participate in no triangle; their normals fall back to
      the default ``(0, 0, 1)`` downstream.
    - Triangles are emitted only for quads where all four corners are
      occupied. Partial quads produce holes, which is the correct outcome
      for dropped LiDAR returns.
    """
    n_points = azimuth.shape[0]
    if n_points < 3:
        return np.empty((0, 3), dtype=np.int32)

    el_min = float(elevation.min())
    el_max = float(elevation.max())
    el_span = max(el_max - el_min, 1e-9)
    az_span = 2.0 * np.pi + 1e-9  # span over [-π, π]

    el_bin = np.floor((elevation - el_min) / el_span * n_elevation_bins).astype(np.int32)
    az_bin = np.floor((azimuth - (-np.pi)) / az_span * n_azimuth_bins).astype(np.int32)
    np.clip(el_bin, 0, n_elevation_bins - 1, out=el_bin)
    np.clip(az_bin, 0, n_azimuth_bins - 1, out=az_bin)

    cell_id = el_bin.astype(np.int64) * n_azimuth_bins + az_bin.astype(np.int64)

    # Stable sort gives the lowest-indexed point per cell deterministically.
    sort_idx = np.argsort(cell_id, kind="stable")
    sorted_cells = cell_id[sort_idx]
    first_in_cell = np.concatenate(([True], sorted_cells[1:] != sorted_cells[:-1]))

    occupancy = np.full(n_elevation_bins * n_azimuth_bins, -1, dtype=np.int32)
    occupancy[sorted_cells[first_in_cell]] = sort_idx[first_in_cell].astype(np.int32)
    occupancy_2d = occupancy.reshape(n_elevation_bins, n_azimuth_bins)

    # ``p0..p3`` are int32 grids of vertex indices at the four corners of
    # each ``(r, c, r+1, c+1)`` quad; ``-1`` entries mark empty cells.
    p0 = occupancy_2d[:-1, :-1]
    p1 = occupancy_2d[:-1, 1:]
    p2 = occupancy_2d[1:, :-1]
    p3 = occupancy_2d[1:, 1:]
    valid_quad = (p0 >= 0) & (p1 >= 0) & (p2 >= 0) & (p3 >= 0)
    if not np.any(valid_quad):
        return np.empty((0, 3), dtype=np.int32)

    p0v, p1v, p2v, p3v = p0[valid_quad], p1[valid_quad], p2[valid_quad], p3[valid_quad]
    tri1 = np.stack([p0v, p1v, p2v], axis=1)
    tri2 = np.stack([p1v, p3v, p2v], axis=1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int32)


def flip_normals_towards_viewpoint(points: np.ndarray, normals: np.ndarray, viewpoint: np.ndarray) -> np.ndarray:
    """Flip point normals to point towards a user-defined reference point (viewpoint)"""
    assert points.shape[1] == 3
    assert normals.shape == points.shape
    assert viewpoint.size == 3
    viewpoint = viewpoint.reshape((1, 3))
    needs_flipping = np.sum(normals * (viewpoint - points), axis=1) < 0.0
    flipped_normals = normals.copy()
    flipped_normals[needs_flipping] = -normals[needs_flipping]
    return flipped_normals


# Sentinel cache for the lazily loaded spin_mesh extension.
# None = not tried, False = tried and unavailable, else the loaded module.
# Kept as a module global (not a function attribute) so it resolves reliably.
_spin_mesh_extension: Any = None


def _try_load_spin_mesh_extension() -> Optional[Any]:
    """Return the loaded ``nurec.libs.spin_mesh`` CUDA extension, or ``None``.

    Lazily imported and cached in a module global so the first call pays the
    import cost once and subsequent calls return the cached handle. A
    failure to import (missing build artifact, undefined symbols, etc.)
    is logged once and the cached value is set to ``False`` so we don't
    keep retrying.
    """
    global _spin_mesh_extension
    if _spin_mesh_extension is not None:
        return _spin_mesh_extension if _spin_mesh_extension is not False else None
    try:
        from nurec.libs.spin_mesh.interface import spin_mesh as ext  # type: ignore
    except Exception as exc:  # pragma: no cover — exercised by environment
        logger.info(
            "nurec.libs.spin_mesh CUDA extension unavailable (%s); using torch-on-CUDA fallback "
            "for compute_mesh_normals_in_lidar_spin(device='cuda').",
            exc,
        )
        _spin_mesh_extension = False
        return None
    _spin_mesh_extension = ext
    return ext


def _scanline_triangulate_torch(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    n_elevation_bins: int,
    n_azimuth_bins: int,
) -> torch.Tensor:
    """Torch version of ``_scanline_triangulate``. Returns int32 triangle indices.

    Quantizes (az, el) onto an ``n_elevation_bins × n_azimuth_bins`` grid,
    keeps the lowest-indexed point per occupied cell, and enumerates all
    fully-occupied quads. Runs on whatever device the inputs live on.

    Binning math is promoted to float64 so the partitioning of points
    into cells does not depend on whether the upstream
    ``azimuth``/``elevation`` were produced in float32 or float64; this
    keeps the result stable across backends and across the CPU/GPU split.
    """
    device = azimuth.device
    n_points = azimuth.shape[0]
    if n_points < 3:
        return torch.empty((0, 3), dtype=torch.int32, device=device)

    azimuth_f64 = azimuth.to(torch.float64)
    elevation_f64 = elevation.to(torch.float64)
    el_min = elevation_f64.min()
    el_max = elevation_f64.max()
    el_span = torch.clamp(el_max - el_min, min=1e-9)
    az_span = 2.0 * np.pi + 1e-9

    el_bin = torch.floor((elevation_f64 - el_min) / el_span * n_elevation_bins).to(torch.int64)
    az_bin = torch.floor((azimuth_f64 - (-np.pi)) / az_span * n_azimuth_bins).to(torch.int64)
    el_bin.clamp_(0, n_elevation_bins - 1)
    az_bin.clamp_(0, n_azimuth_bins - 1)

    cell_id = el_bin * n_azimuth_bins + az_bin

    sort_idx = torch.argsort(cell_id, stable=True)
    sorted_cells = cell_id[sort_idx]
    if sorted_cells.numel() == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=device)
    first_in_cell = torch.cat(
        [
            torch.ones(1, dtype=torch.bool, device=device),
            sorted_cells[1:] != sorted_cells[:-1],
        ]
    )

    occupancy = torch.full((n_elevation_bins * n_azimuth_bins,), -1, dtype=torch.int32, device=device)
    occupancy[sorted_cells[first_in_cell]] = sort_idx[first_in_cell].to(torch.int32)
    occupancy_2d = occupancy.view(n_elevation_bins, n_azimuth_bins)

    p0 = occupancy_2d[:-1, :-1]
    p1 = occupancy_2d[:-1, 1:]
    p2 = occupancy_2d[1:, :-1]
    p3 = occupancy_2d[1:, 1:]
    valid_quad = (p0 >= 0) & (p1 >= 0) & (p2 >= 0) & (p3 >= 0)
    if not torch.any(valid_quad):
        return torch.empty((0, 3), dtype=torch.int32, device=device)

    p0v = p0[valid_quad]
    p1v = p1[valid_quad]
    p2v = p2[valid_quad]
    p3v = p3[valid_quad]
    tri1 = torch.stack([p0v, p1v, p2v], dim=1)
    tri2 = torch.stack([p1v, p3v, p2v], dim=1)
    return torch.cat([tri1, tri2], dim=0)


def _compute_vertex_normals_area_weighted_torch(points: torch.Tensor, triangles_int64: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex-normal computation on torch.

    Each face's unnormalized cross product ``(b-a) × (c-a)`` (whose
    magnitude is ``2 × triangle_area``) is scattered onto its three
    vertices via ``index_add_``; the result is then normalized per
    vertex. Unconnected vertices accumulate to zero and normalize to
    NaN, which the caller replaces with the ``(0, 0, 1)`` fallback.
    """
    n = points.shape[0]
    a = points[triangles_int64[:, 0]]
    b = points[triangles_int64[:, 1]]
    c = points[triangles_int64[:, 2]]
    face_normal = torch.linalg.cross(b - a, c - a)

    accum = torch.zeros((n, 3), dtype=points.dtype, device=points.device)
    accum.index_add_(0, triangles_int64[:, 0], face_normal)
    accum.index_add_(0, triangles_int64[:, 1], face_normal)
    accum.index_add_(0, triangles_int64[:, 2], face_normal)

    norms = torch.linalg.norm(accum, dim=1, keepdim=True)
    return accum / norms  # → NaN rows where accum is the zero vector


def compute_mesh_normals_in_lidar_spin_torch(
    points: torch.Tensor,
    n_elevation_bins: Optional[int] = None,
    n_azimuth_bins: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-spin pipeline (xyz → spherical → scanline mesh → area-weighted
    normals → viewpoint flip) running entirely on ``points.device``.

    Returns torch tensors on the same device. The numpy entry point
    ``compute_mesh_normals_in_lidar_spin`` dispatches into this function
    when ``device`` selects a CUDA device.

    ``n_elevation_bins`` / ``n_azimuth_bins`` default to ``None``; when
    either is unset the helper auto-detects the grid size from the
    spherical coordinates (see ``_auto_detect_bin_counts``). The detection
    runs on CPU after a small ``.cpu()`` transfer of the (az, el) arrays.
    """
    assert points.dim() == 2 and points.shape[1] == 3
    if points.dtype != torch.float32:
        points = points.to(torch.float32)

    radii = torch.linalg.norm(points, dim=1).clamp_min(1e-12)
    azimuth = torch.atan2(points[:, 1], points[:, 0])
    elevation = torch.asin((points[:, 2] / radii).clamp(-1.0, 1.0))

    if n_elevation_bins is None or n_azimuth_bins is None:
        # Auto-detection runs on numpy arrays; transfer is cheap relative
        # to the kernel work that follows.
        az_np = azimuth.detach().cpu().numpy()
        el_np = elevation.detach().cpu().numpy()
        auto_n_el, auto_n_az = _auto_detect_bin_counts(az_np, el_np)
        if n_elevation_bins is None:
            n_elevation_bins = auto_n_el
        if n_azimuth_bins is None:
            n_azimuth_bins = auto_n_az

    triangles_int32 = _scanline_triangulate_torch(azimuth, elevation, n_elevation_bins, n_azimuth_bins)
    triangles_int64 = triangles_int32.to(torch.int64)

    normals = _compute_vertex_normals_area_weighted_torch(points, triangles_int64)

    # Replace NaN normals (unconnected or degenerate vertices) with +z.
    nan_rows = torch.isnan(normals).any(dim=1)
    if torch.any(nan_rows):
        fallback = normals.new_tensor([0.0, 0.0, 1.0])
        normals = torch.where(nan_rows.unsqueeze(1), fallback.unsqueeze(0), normals)

    # Flip normals to point toward the sensor (origin, in centered-frame
    # coordinates). Equivalent to ``flip_normals_towards_viewpoint(points,
    # normals, [0, 0, 0])``.
    dot_to_origin = -(normals * points).sum(dim=1, keepdim=True)
    normals = torch.where(dot_to_origin < 0.0, -normals, normals)

    return normals, triangles_int32


def compute_mesh_normals_in_lidar_spin(
    points: NDArray[np.float32],
    verbose: bool = False,
    n_elevation_bins: Optional[int] = None,
    n_azimuth_bins: Optional[int] = None,
    device: Union[str, "torch.device"] = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normals for points from a Lidar spin via triangle mesh construction over the scanning pattern.

    Input points are assumed to be in cartesian coordinates and scanned with a Lidar spinning at the origin.
    This function performs the following steps:
    1. Convert points to spherical coordinates (azimuth, elevation, radius).
    2. Construct a triangle mesh in azimuth-elevation (2D) space over the points as vertices.
    3. Compute vertex normals of the mesh.
    4. Flip normals to point outside of the surface, given the (ray) origin.

    Bin counts default to ``None`` and are auto-detected from the input's
    scan structure when unset (see ``_auto_detect_bin_counts``).

    ``device`` selects the backend. ``"cpu"`` runs the numpy + ``pcu``
    path and is safe to call concurrently from multiple
    ``ThreadPoolExecutor`` workers. ``"cuda"`` routes through
    ``compute_mesh_normals_in_lidar_spin_torch``, which fuses all four
    steps into torch ops on the GPU. If a CUDA device is requested but
    not available, the call falls back to the CPU path with a warning
    rather than failing.
    """
    assert len(points.shape) == 2 and points.shape[1] == 3

    device_str = str(device)
    use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()
    if device_str.startswith("cuda") and not use_cuda:
        logger.warning("compute_mesh_normals_in_lidar_spin: CUDA requested but unavailable; falling back to CPU.")

    if use_cuda:
        if verbose:
            logger.info(f"Computing lidar spin mesh + normals on CUDA over {len(points)} points")
        # Resolve auto-detect once at the dispatcher: the ``.cu`` binding
        # requires explicit int arguments (no ``Optional[int]`` support), and
        # we want a single detection pass shared by both the kernel and the
        # torch fallback.
        elevation_np = None
        if n_elevation_bins is None or n_azimuth_bins is None:
            radii = np.linalg.norm(points, axis=1)
            radii = np.maximum(radii, 1e-12)
            azimuth_np = np.arctan2(points[:, 1], points[:, 0])
            elevation_np = np.arcsin(np.clip(points[:, 2] / radii, -1.0, 1.0))
            auto_n_el, auto_n_az = _auto_detect_bin_counts(azimuth_np, elevation_np)
            if n_elevation_bins is None:
                n_elevation_bins = auto_n_el
            if n_azimuth_bins is None:
                n_azimuth_bins = auto_n_az
        target_device = torch.device(device_str)
        points_t = torch.as_tensor(points, dtype=torch.float32, device=target_device)
        cu_extension = _try_load_spin_mesh_extension()
        if cu_extension is not None:
            # CUDA path: fused binning + cell claim, triangle emission, and occupancy-neighborhood normals in ``nurec/libs/spin_mesh``.
            # Pass elevation bounds only when auto-detection already computed them.
            # Explicit-bin callers avoid an extra CPU pass and use the extension's
            # compatibility path.
            el_min = float(np.min(elevation_np)) if elevation_np is not None and len(elevation_np) > 0 else float("nan")
            el_max = float(np.max(elevation_np)) if elevation_np is not None and len(elevation_np) > 0 else float("nan")
            normals_t, triangles_t = cu_extension.compute_mesh_normals_in_lidar_spin(
                points_t,
                n_elevation_bins=n_elevation_bins,
                n_azimuth_bins=n_azimuth_bins,
                el_min=el_min,
                el_max=el_max,
            )
        else:
            # Torch-on-CUDA fallback when the extension hasn't been built
            # for this environment (e.g. running outside the Bazel-built
            # wheel during local development).
            normals_t, triangles_t = compute_mesh_normals_in_lidar_spin_torch(
                points_t,
                n_elevation_bins=n_elevation_bins,
                n_azimuth_bins=n_azimuth_bins,
            )
        normals = normals_t.detach().cpu().numpy().astype(np.float32, copy=False)
        triangles = triangles_t.detach().cpu().numpy().astype(np.int32, copy=False)
        return normals, triangles

    if verbose:
        logger.info(f"Reconstructing lidar spin mesh over {len(points)} points")
        start_time = perf_counter()
    triangles, azimuth_elevation = construct_lidar_spin_mesh(
        points,
        n_elevation_bins=n_elevation_bins,
        n_azimuth_bins=n_azimuth_bins,
    )
    if verbose:
        elapsed_time = perf_counter() - start_time
        logger.info(f"{len(triangles)} triangles reconstructed in {elapsed_time:.3f} s")

    if verbose:
        logger.info(f"Computing normals for {len(points)} mesh vertices")
        start_time = perf_counter()
    if triangles.shape[0] == 0:
        # ``pcu`` rejects empty triangle inputs; produce the +z fallback directly.
        normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (len(points), 1))
    else:
        normals = pcu.estimate_mesh_vertex_normals(points, triangles)
        invalid_normals = np.any(np.isnan(normals), axis=1)
        normals[invalid_normals] = np.array([0, 0, 1], dtype=np.float32)
    normals = flip_normals_towards_viewpoint(points, normals, np.zeros((1, 3), dtype=np.float32))
    if verbose:
        elapsed_time = perf_counter() - start_time
        logger.info(f"{len(normals)} normals computed in {elapsed_time:.3f} s")

    return normals, triangles


def _find_ground_compatible_points_numpy(
    points: NDArray[np.float32],
    normals: NDArray[np.float32],
    ground_control_point: NDArray[np.float32],
    ref_normal: NDArray[np.float32],
    max_distance: float,
    max_angle_deg: float,
) -> NDArray[np.bool_]:
    """CPU-native implementation of the ground-compatible point filter.

    The threshold comparisons must stay identical to ``ground_compatible_kernel`` in
    ``libs/ground_mesh/ground_mesh.cu`` (``distance <= max_distance``, ``cos_angle > min_cos_angle``)
    so that the CUDA fast path and this fallback agree on points sitting exactly on either bound.
    """
    plane_normals = normals
    plane_offsets = -np.sum(plane_normals * points, axis=1)
    # The distance is simple and can be calculated in batch because normals are all unit-norm.
    plane_distances_from_gcp = np.abs(np.sum(plane_normals * ground_control_point.flatten(), axis=1) + plane_offsets)

    cos_angles = np.sum(plane_normals * ref_normal.flatten(), axis=1)
    min_cos_angle = np.cos(np.radians(max_angle_deg))

    return np.logical_and(plane_distances_from_gcp <= max_distance, cos_angles > min_cos_angle)


def find_ground_compatible_points(
    points: NDArray[np.float32],  # Input points to cull
    normals: NDArray[np.float32],  # Normals of the input points
    ground_control_point: NDArray[np.float32],  # A known point on the ground in a sensor-centric frame
    ref_normal: NDArray[np.float32],  # Reference normal for culling ground-compatible points
    max_distance: float,  # Distance tolerance w.r.t. the ground control points for plane hypotheses
    max_angle_deg: float,  # Angle tolerance w.r.t. the ref normal for plane hypotheses
) -> NDArray[np.bool_]:
    """Return a mask of oriented points compatible with the ground model.

    Args:
        points: Float32 point positions with shape ``(N, 3)``.
        normals: Unit-length float32 point normals with shape ``(N, 3)``.
        ground_control_point: A known ground point containing exactly three values.
        ref_normal: A unit-length reference normal containing exactly three values.
        max_distance: Positive maximum distance from a point's tangent plane to
            ``ground_control_point``.
        max_angle_deg: Maximum angle from ``ref_normal`` in the range ``(0, 180]`` degrees.

    Returns:
        A boolean array with shape ``(N,)`` indicating the compatible points.
    """
    assert points.shape[1] == 3
    assert ref_normal.size == 3
    assert normals.shape == points.shape
    assert ground_control_point.size == 3
    assert max_distance > 0.0
    assert max_angle_deg > 0.0 and max_angle_deg <= 180.0
    assert np.allclose(np.sum(normals**2, axis=1), 1.0)  # Point normals expected to be unit-norm
    assert np.isclose(np.linalg.norm(ref_normal.flatten()), 1.0)  # Reference normal expected to be unit-norm

    return _find_ground_compatible_points_numpy(
        points, normals, ground_control_point, ref_normal, max_distance, max_angle_deg
    )


def subsample(num_items, max_samples: int) -> np.ndarray:
    num_samples = min(max_samples, num_items)
    # Use a local RandomState rather than mutating numpy's global state so
    # callers that run segment_road's per-spin loop concurrently across
    # threads do not race on the global RNG. ``RandomState(seed).choice(...)``
    # is bit-equivalent to ``np.random.seed(seed); np.random.choice(...)`` in
    # serial.
    rng = np.random.RandomState(GROUND_MESH_RANDOM_SEED)
    return rng.choice(num_items, num_samples, replace=False) if num_items > max_samples else np.arange(num_items)


def detect_dominant_plane(
    points: NDArray[np.float32],
    normals: NDArray[np.float32],
    distance_threshold: float,
    angle_threshold_deg: float,
    num_hypotheses: int,
    eval_point_indices: Optional[ArrayLike] = None,
) -> Tuple[Plane, NDArray[np.bool_], PlaneDetectorRansac]:
    ransac = PlaneDetectorRansac(distance_threshold, angle_threshold_deg)

    # Local ``RandomState`` rather than seeding numpy's global state. Numerically
    # identical to ``np.random.seed(seed); np.random.choice(...)`` in serial,
    # but thread-safe when callers run multiple instances concurrently.
    rng = np.random.RandomState(GROUND_MESH_RANDOM_SEED)

    # Generate plane hypotheses from individual points and their normals.
    # Assumes normals of decent quality but it is faster to evaluate than hypotheses from point triplets:
    # - Needs less hypotheses to achieve the same chance of an uncontaminated one (needs 1 inlier instead of 3).
    # - Enables prefiltering points based on geometric priors to avoid evaluating hypotheses that do not satisfy them.
    planes, hypothesis_indices = ransac.generate_planes_from_single_points(
        points, normals, num_planes=num_hypotheses, rng=rng
    )

    # Optionally subsample points for plane scoring because a spin can contain hundreds of thousands of points.
    # There must remain enough samples on the actual plane to be safely detectable.
    if eval_point_indices is None:
        eval_points = points
        eval_normals = normals
    else:
        indices = np.asarray(eval_point_indices)
        assert indices.size > 1
        eval_points = points[indices]
        eval_normals = normals[indices]

    ext = _try_load_ground_mesh_extension()
    if ext is not None and torch.cuda.is_available():
        native_points = np.ascontiguousarray(points, dtype=np.float32)
        native_normals = np.ascontiguousarray(normals, dtype=np.float32)
        native_eval_indices = (
            np.arange(len(points), dtype=np.int64)
            if eval_point_indices is None
            else np.asarray(eval_point_indices, dtype=np.int64)
        )
        best_normal_t, best_offset_t, inlier_mask_t = ext.detect_dominant_plane(
            torch.as_tensor(native_points, device="cuda"),
            torch.as_tensor(native_normals, device="cuda"),
            torch.as_tensor(np.ascontiguousarray(hypothesis_indices, dtype=np.int64), device="cuda"),
            torch.as_tensor(np.ascontiguousarray(native_eval_indices, dtype=np.int64), device="cuda"),
            distance_threshold=float(distance_threshold),
            angle_threshold_deg=float(angle_threshold_deg),
        )
        best_plane = Plane(
            best_normal_t.detach().cpu().numpy().astype(np.float32, copy=False),
            float(best_offset_t.item()),
        )
        return best_plane, inlier_mask_t.detach().cpu().numpy().astype(bool, copy=False), ransac

    # Score all hypotheses in a single batched matrix op instead of looping in
    # Python. Matches the per-hypothesis result within float32 tolerance.
    plane_normals_batched = np.stack([p.normal for p in planes], axis=0).astype(np.float32)
    plane_offsets_batched = np.array([p.offset for p in planes], dtype=np.float32)
    plane_scores = ransac.evaluate_plane_hypotheses_batched(
        plane_normals_batched, plane_offsets_batched, eval_points, eval_normals
    )
    assert not np.any(np.isnan(plane_scores))

    # Pick the plane with the highest score ~ most support from the points and normals.
    best_plane_idx = int(np.argmax(plane_scores))
    best_plane = planes[best_plane_idx]

    # Find inliers in the full set of input points
    is_inlier = ransac.find_inliers(best_plane, points, normals)

    # TODO: do at least 1 round of plane refitting (regression) to all inliers and recomputing inliers
    # because candidate planes are generated from single points whose normals are inaccurate,
    # although the best that has the most support is picked.

    return best_plane, is_inlier, ransac


def reconstruct_ground_mesh_from_points(
    point_cloud_generator: Generator[PointCloud, None, None],  # Input point cloud
    min_ray_length: float,
    ground_control_point: NDArray[np.float32],
    ground_compat_max_distance: float,
    ground_compat_max_angle_deg: float,
    num_plane_hypotheses: int,  # Number of plane hypotheses to generate for dominant plane detection
    plane_max_distance: float,
    plane_max_angle_deg: float,
    plane_min_eval_points: int,
    plane_max_eval_points: int,
    plane_min_inliers: int,
    enable_plane_extension: bool,
    voxel_size: float,
    min_points_per_voxel: int,
    smoothing_passes: int,
    grid_hole_fill_min_neighbors: int = 6,
    export_per_frame_diagnostics: bool = False,
    output_path: Path = Path("."),
    points_to_world_transf: Optional[FrameConversion] = None,
    verbose: bool = False,
    # Mesh robustness knobs.
    mesh_topology: Literal["delaunay", "grid"] = "delaunay",
    smoothing_method: Literal["neighbor_average", "local_plane"] = "neighbor_average",
    local_plane_min_neighbors: int = 3,
    local_plane_include_self: bool = False,
    height_statistic: Literal["mean", "median"] = "mean",
    color_statistic: Literal["mean", "median"] = "mean",
    outlier_height_threshold: Optional[float] = None,
    outlier_neighbor_radius: int = 1,
    outlier_min_neighbors: int = 4,
    max_triangle_edge_length: Optional[float] = None,
    max_triangle_normal_angle_deg: Optional[float] = None,
    spin_mesh_device: Literal["auto", "cpu", "cuda"] = "auto",
    num_workers: int = 1,
    respect_rgb_label_validity: bool = False,
    cuboid_tracks: Optional[CuboidTracks] = None,
    cuboids_base_margin: float = 0.5,
    cuboids_exclude_points_radius: float = 1.0,
    cuboids_exclude_points_knns: int = 1,
    cuboid_point_merge_cell_size: float = 0.5,
) -> Tuple[
    NDArray[np.float32],  # Mesh vertices after smoothing (V,3)
    NDArray[np.int32],  # Mesh triangles as vertex index triplets (T,3)
    NDArray[np.float32],  # Vertices before smoothing (V,3)
    NDArray[np.float32],  # Point positions for meshing (N,3)
    NDArray[np.float32],  # Point normals for meshing (N,3)
    NDArray[np.bool_],  # Road mask (N,), indicates for each point whether it has been identified as road internally
    NDArray[np.bool_],  # Initial road mask (N,) from the input point cloud
    Optional[NDArray[np.uint8]],  # Per-vertex RGB colors (V,3), or None if colors were not available
    Optional[NDArray[np.float32]],  # Cuboid ground points used for mesh extension, or None
]:
    # Remove outliers from the road points by robust dominant plane fitting to the points per spin
    # such that plane hypotheses pass through the provided ground control point.
    road_segmenter = DominantPlaneRoadSegmenter(
        min_ray_length=min_ray_length,
        ground_control_point=ground_control_point,
        ground_compat_max_distance=ground_compat_max_distance,
        ground_compat_max_angle_deg=ground_compat_max_angle_deg,
        num_plane_hypotheses=num_plane_hypotheses,
        plane_max_distance=plane_max_distance,
        plane_max_angle_deg=plane_max_angle_deg,
        plane_min_eval_points=plane_min_eval_points,
        plane_max_eval_points=plane_max_eval_points,
        plane_min_inliers=plane_min_inliers,
        enable_plane_extension=enable_plane_extension,
        respect_rgb_label_validity=respect_rgb_label_validity,
        export_per_frame_diagnostics=export_per_frame_diagnostics,
        output_path=output_path,
        verbose=verbose,
        spin_mesh_device=spin_mesh_device,
        num_workers=num_workers,
    )
    points, normals, road_mask, initial_road_mask, point_colors = road_segmenter.segment_road(
        point_cloud_generator, transformation=points_to_world_transf
    )

    logger.info("Separating merged point cloud into road and non-road points")
    points_road = points[road_mask]
    colors_road = point_colors[road_mask] if point_colors is not None else None

    if len(points_road) < 3:
        raise ValueError(f"Too few road points ({len(points_road)}) to mesh")

    cuboid_ground_points: Optional[NDArray[np.float32]] = None
    if cuboid_tracks is not None:
        cuboid_ground_points = generate_ground_points_from_cuboids(
            cuboid_tracks,
            points_road,
            cuboids_base_margin=cuboids_base_margin,
            exclude_points_radius=cuboids_exclude_points_radius,
            exclude_points_knns=cuboids_exclude_points_knns,
            cuboid_point_merge_cell_size=cuboid_point_merge_cell_size,
        )

    points_to_mesh: NDArray[np.float32] = points_road
    colors_to_mesh = colors_road
    if cuboid_ground_points is not None and len(cuboid_ground_points) > 0:
        logger.info("Extending road points with %d extra ground points", len(cuboid_ground_points))
        points_to_mesh = np.concatenate([points_road, cuboid_ground_points], axis=0)
        if colors_road is not None:
            grey = np.full((len(cuboid_ground_points), colors_road.shape[1]), 128, dtype=colors_road.dtype)
            colors_to_mesh = np.concatenate([colors_road, grey], axis=0)

    # Apply meshing to road points by assuming no outliers in the road segment at this point.
    meshing = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=voxel_size,
        min_points_per_voxel=min_points_per_voxel,
        grid_hole_fill_min_neighbors=grid_hole_fill_min_neighbors,
        smoothing_passes=smoothing_passes,
        elevation_axis=2,
        mesh_topology=mesh_topology,
        smoothing_method=smoothing_method,
        local_plane_min_neighbors=local_plane_min_neighbors,
        local_plane_include_self=local_plane_include_self,
        height_statistic=height_statistic,
        color_statistic=color_statistic,
        outlier_height_threshold=outlier_height_threshold,
        outlier_neighbor_radius=outlier_neighbor_radius,
        outlier_min_neighbors=outlier_min_neighbors,
        max_triangle_edge_length=max_triangle_edge_length,
        max_triangle_normal_angle_deg=max_triangle_normal_angle_deg,
    )
    smooth_vertices, triangles, vertices, vertex_colors = meshing.build_mesh_from_points(
        points_to_mesh, colors=colors_to_mesh
    )

    return (
        smooth_vertices,
        triangles,
        vertices,
        points,
        normals,
        road_mask,
        initial_road_mask,
        vertex_colors,
        cuboid_ground_points,
    )


def _triangle_edges(triangles: NDArray[np.int32]) -> NDArray[np.int32]:
    """Return all directed triangle edges as ``(3*T, 2)`` in (v0,v1),(v1,v2),(v2,v0) order.

    Row ``e * T + k`` is edge ``e`` of triangle ``k``, so the owning triangle index of every
    row is ``np.arange(3 * T) % T``. Shared by the connectivity, boundary and densification
    passes; callers sort each row (``np.sort(..., axis=1)``) for an undirected edge key.
    """
    return np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0)


def _triangle_components(triangles: NDArray[np.int32]) -> NDArray[np.int64]:
    """Per-triangle connected-component label; triangles sharing an edge are connected.

    For non-manifold inputs (an edge shared by >2 triangles), chaining triangles within each
    edge group keeps connectivity correct without an all-pairs blow-up. Label values are the
    arbitrary ids returned by ``scipy.sparse.csgraph.connected_components``.
    """
    n_tris = len(triangles)
    if n_tris == 0:
        return np.zeros(0, dtype=np.int64)

    # Undirected edge of each triangle, tagged with the owning triangle index, sorted so
    # triangles sharing an edge land in adjacent rows; connect each to the next in its group.
    edges = np.sort(_triangle_edges(triangles), axis=1)
    tri_idx = np.tile(np.arange(n_tris, dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges_sorted = edges[order]
    tri_sorted = tri_idx[order]
    same_group = np.all(edges_sorted[1:] == edges_sorted[:-1], axis=1)
    src = tri_sorted[:-1][same_group]
    dst = tri_sorted[1:][same_group]

    graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n_tris, n_tris))
    _, component = connected_components(graph, directed=False)
    return component


def keep_largest_connected_component(
    triangles: NDArray[np.int32],
) -> NDArray[np.bool_]:
    """Return a per-triangle bool mask selecting only the largest face-connected component.

    Triangles are face-connected when they share an edge (see :func:`_triangle_components`).
    """
    if len(triangles) == 0:
        return np.zeros(0, dtype=bool)

    component = _triangle_components(triangles)
    sizes = np.bincount(component)
    largest = int(sizes.argmax())
    logger.info(
        f"largest_connected_component: kept {sizes[largest]}/{len(triangles)} tris "
        f"({len(sizes)} components, dropped {int(sizes.sum() - sizes[largest])})"
    )
    return component == largest


def _trace_boundary_loop(
    start: int, next_vertex: "dict[int, list[int]]", used: "set[tuple[int, int]]"
) -> Optional[list[int]]:
    """Follow oriented boundary edges from ``start`` until the loop closes, marking edges used.

    Returns the closed vertex loop, or None if the walk dead-ends (an open chain).
    """
    loop = [start]
    cur = start
    while candidates := [v for v in next_vertex.get(cur, ()) if (cur, v) not in used]:
        nxt = candidates[0]
        used.add((cur, nxt))
        cur = nxt
        if cur == start:
            return loop
        loop.append(cur)
    return None


def find_boundary_loops(triangles: NDArray[np.int32]) -> list[list[int]]:
    """Return closed vertex-index loops along the mesh boundary.

    A boundary edge is one shared by exactly one triangle. Edges are walked in the
    direction they appear in their owning triangle's CCW winding, so each output loop
    is oriented (outer loops are CCW around the meshed region, interior holes are CW).
    """
    from collections import defaultdict

    if len(triangles) == 0:
        return []

    # A boundary edge belongs to exactly one triangle. Count undirected edges vectorised, then
    # keep the (single) directed edge of each boundary edge so the walk below stays oriented.
    directed = _triangle_edges(triangles)
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse.ravel()] == 1]

    next_vertex: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary.tolist():
        next_vertex[a].append(b)

    used: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for start in list(next_vertex.keys()):
        if all((start, v) in used for v in next_vertex[start]):
            continue
        loop = _trace_boundary_loop(start, next_vertex, used)
        if loop is not None and len(loop) >= 3:
            loops.append(loop)

    logger.info(f"find_boundary_loops: {len(loops)} closed loops, sizes={[len(loop) for loop in loops][:10]}")
    return loops


def _polygon_xy_area(xy: NDArray[np.float32]) -> float:
    """Unsigned polygon area in XY (shoelace), computed in float64 after a local mean-shift.

    Camera ground meshes are world-space (absolute coords ~1e5) where float32 shoelace
    products (~1e10) lose the area to catastrophic cancellation -- a 100m^2 outer loop and a
    64m^2 inner loop can both collapse to the same float32 value, inverting the largest-area
    outer-hull selection in :func:`_fill_component_interior_loops` and triangulating the
    exterior as a hole. Translation is area-invariant, so the shift does not change the result.
    """
    p = np.asarray(xy, dtype=np.float64)
    p = p - p.mean(axis=0)
    x = p[:, 0]
    y = p[:, 1]
    return float(0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _triangulate_simple_polygon_xy(pts: NDArray[np.float32]) -> NDArray[np.int32]:
    """Ear-clipping triangulation of a simple (non-self-intersecting) polygon in XY.

    Returns ``(T, 3)`` triangles as indices into ``pts``. Correct for convex and concave
    polygons alike: it triangulates exactly the polygon interior (no over-fill across a
    concave notch, no spurious triangles outside), referencing only the input vertices.
    Runs in O(n^2) time and O(n) memory over the (small) loop, unlike filtering an
    unconstrained Delaunay of the loop vertices.
    """
    if len(pts) < 3:
        return np.zeros((0, 3), dtype=np.int32)

    # Shift to a local origin and promote to float64: camera ground meshes are world-space
    # (absolute coords ~1e5) where float32 orientation math collapses to zero-area degeneracy.
    # Translation is orientation/area-invariant, so this does not change the result.
    xy = np.asarray(pts, dtype=np.float64)
    xy = xy - xy.mean(axis=0)
    orig = np.arange(len(xy))
    # Drop consecutive duplicate vertices (incl. a closing repeat of the first vertex).
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.any(xy[1:] != xy[:-1], axis=1)
    if np.array_equal(xy[0], xy[-1]):
        keep[-1] = False
    xy, orig = xy[keep], orig[keep]
    n = len(xy)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int32)

    # Orient CCW so a positive cross product at a vertex marks a convex corner.
    signed_area2 = float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    remaining = list(range(n)) if signed_area2 >= 0.0 else list(range(n - 1, -1, -1))

    def cross(o: int, a: int, b: int) -> float:
        return float((xy[a, 0] - xy[o, 0]) * (xy[b, 1] - xy[o, 1]) - (xy[a, 1] - xy[o, 1]) * (xy[b, 0] - xy[o, 0]))

    def in_triangle(p: int, a: int, b: int, c: int) -> bool:
        # Vertex p inside or on CCW triangle (a, b, c).
        return cross(a, b, p) >= 0.0 and cross(b, c, p) >= 0.0 and cross(c, a, p) >= 0.0

    def find_ear() -> Optional[int]:
        m = len(remaining)
        for k in range(m):
            i0, i1, i2 = remaining[(k - 1) % m], remaining[k], remaining[(k + 1) % m]
            if cross(i0, i1, i2) <= 0.0:  # reflex or collinear corner -> not an ear tip
                continue
            if any(in_triangle(j, i0, i1, i2) for j in remaining if j not in (i0, i1, i2)):
                continue
            return k
        return None  # no ear: numerically degenerate / non-simple loop

    tris: list[list[int]] = []
    # Each clip removes one vertex, so this terminates in <= n-3 steps; the walrus exits cleanly
    # when no ear is found (degenerate / non-simple loop) -- no while-True or break needed.
    while len(remaining) > 3 and (ear := find_ear()) is not None:
        m = len(remaining)
        i0, i1, i2 = remaining[(ear - 1) % m], remaining[ear], remaining[(ear + 1) % m]
        tris.append([i0, i1, i2])
        del remaining[ear]
    if len(remaining) == 3:
        tris.append(list(remaining))

    # A simple polygon yields exactly n-2 triangles. Fewer means the loop was not simple
    # (self-touching / pinched); a partial ear fan would leave an invalid, half-filled hole
    # for densify and smoothing, so discard the whole loop rather than emit partial triangles.
    if len(tris) < n - 2:
        logger.warning(
            "ear-clip produced %d/%d triangles for a %d-vertex loop; loop is likely "
            "non-simple (self-touching) -- skipping it (no partial fill)",
            len(tris),
            n - 2,
            n,
        )
        return np.zeros((0, 3), dtype=np.int32)
    return orig[np.asarray(tris, dtype=np.int64)].astype(np.int32)


def _fill_component_interior_loops(
    vertices: NDArray[np.float32],
    component_triangles: NDArray[np.int32],
) -> list[NDArray[np.int32]]:
    """Ear-clip the interior holes of ONE face-connected mesh component.

    The largest-area boundary loop is that component's outer hull; every other loop is an
    interior hole and is triangulated in XY (filling exactly the loop interior even when
    concave). Returns the hole-fill triangle chunks indexing the global vertex array; empty
    when the component is hole-free.
    """
    loops = find_boundary_loops(component_triangles)
    if len(loops) <= 1:
        return []

    loop_areas = [_polygon_xy_area(vertices[loop, :2]) for loop in loops]
    outer_idx = int(np.argmax(loop_areas))

    chunks: list[NDArray[np.int32]] = []
    for li, loop in enumerate(loops):
        if li == outer_idx or len(loop) < 3:
            continue
        loop_arr = np.asarray(loop, dtype=np.int64)
        local = _triangulate_simple_polygon_xy(vertices[loop_arr, :2])
        if len(local) > 0:
            chunks.append(loop_arr[local].astype(np.int32, copy=False))
    return chunks


def fill_interior_boundary_loops(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Triangulate every interior hole and concatenate to the input triangles.

    Holes are found per face-connected component: within each component the largest-area
    boundary loop is its outer hull and every other loop is an interior hole. Filling
    per-component is required because a single global "largest loop = outer" rule would treat
    the outer boundary of every component but the biggest as a hole and wrongly triangulate
    it (reachable when ``drop_floating_components=False`` leaves several road patches). Filled
    triangles reference existing vertices (no new vertices added).
    """
    if len(triangles) == 0:
        return triangles

    # Single face-connected component (the production path, which drops floating components
    # first) needs no per-component grouping -- fill its loops directly. Only when several
    # disconnected patches survive (drop_floating_components=False) must each be handled on
    # its own so a patch's outer boundary is not mistaken for another's hole.
    components = _triangle_components(triangles)
    component_ids = np.unique(components)
    if len(component_ids) == 1:
        fill_chunks = _fill_component_interior_loops(vertices, triangles)
    else:
        fill_chunks = [
            chunk
            for comp in component_ids
            for chunk in _fill_component_interior_loops(vertices, triangles[components == comp])
        ]

    n_filled = sum(len(chunk) for chunk in fill_chunks)
    logger.info(f"fill_interior_boundary_loops: {len(component_ids)} components, added {n_filled} hole-fill triangles")
    if not fill_chunks:
        return triangles
    return np.concatenate([triangles, *fill_chunks], axis=0).astype(np.int32, copy=False)


def densify_mesh_midpoint(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int32],
    vertex_colors: Optional[NDArray[np.uint8]],
    max_edge_length: float,
    max_iters: int = 6,
    color_kdtree_points: Optional[NDArray[np.float32]] = None,
    color_kdtree_colors: Optional[NDArray[np.uint8]] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.int32], Optional[NDArray[np.uint8]]]:
    """T-junction-aware midpoint refinement (red/green) on triangles with too-long XY edges.

    Each iteration:
    1. Identify "bad" triangles whose longest XY edge exceeds ``max_edge_length``.
    2. Mark all three edges of every bad triangle for midpoint insertion.
    3. For every triangle, count its marked edges and subdivide:
       - 0 marked: keep unchanged.
       - 1 marked: bisect via the single midpoint (green refinement, 2 sub-tris).
       - 2 marked: 3 sub-tris.
       - 3 marked: full red refinement (4 sub-tris).

    This eliminates T-junctions where a subdivided edge meets an undivided neighbor.

    If ``color_kdtree_points`` and ``color_kdtree_colors`` are both provided, each new
    midpoint vertex's RGB is sampled from the nearest input point in XY (sharper texture
    than averaging the two parent vertex colors). Otherwise midpoint color = mean of the
    two parent colors.
    """
    if len(triangles) == 0:
        empty_c = None if vertex_colors is None else vertex_colors.copy()
        return vertices.copy(), triangles.copy(), empty_c
    V = vertices.copy().astype(np.float32)
    T = triangles.copy().astype(np.int32)
    C = None if vertex_colors is None else vertex_colors.copy()

    kdt = None
    kdt_colors = None
    if color_kdtree_points is not None and color_kdtree_colors is not None and len(color_kdtree_points) > 0:
        kdt = cKDTree(color_kdtree_points[:, :2])
        kdt_colors = color_kdtree_colors

    for it in range(max_iters):
        tri_xy = V[T, :2]
        e0 = np.linalg.norm(tri_xy[:, 1] - tri_xy[:, 0], axis=1)
        e1 = np.linalg.norm(tri_xy[:, 2] - tri_xy[:, 1], axis=1)
        e2 = np.linalg.norm(tri_xy[:, 0] - tri_xy[:, 2], axis=1)
        max_e = np.maximum(np.maximum(e0, e1), e2)
        bad = max_e > max_edge_length
        if not bad.any():
            logger.info(f"densify_mesh_midpoint: converged at iter {it}")
            break
        logger.info(f"densify_mesh_midpoint iter {it}: {int(bad.sum())} tris > {max_edge_length}m")

        # Mark every edge of every bad triangle; one midpoint vertex is inserted per unique
        # (sorted) marked edge. np.unique sorts lexically, so the marked-edge table is sorted.
        bad_edges = np.sort(_triangle_edges(T[bad]), axis=1)
        edge_list = np.unique(bad_edges, axis=0).astype(np.int64)
        if len(edge_list) == 0:
            break

        n_old = len(V)
        mid_pos = (V[edge_list[:, 0]] + V[edge_list[:, 1]]) * 0.5
        V = np.concatenate([V, mid_pos.astype(V.dtype)], axis=0)
        if C is not None:
            if kdt is not None:
                assert kdt_colors is not None  # paired with kdt above
                _, idx = kdt.query(mid_pos[:, :2], k=1)
                mid_col = kdt_colors[idx].astype(C.dtype, copy=False)
            else:
                mid_col = ((C[edge_list[:, 0]].astype(np.int32) + C[edge_list[:, 1]].astype(np.int32)) // 2).astype(
                    C.dtype, copy=False
                )
            C = np.concatenate([C, mid_col], axis=0)

        # Vectorised midpoint lookup: encode each sorted edge as a single int (both endpoints
        # are < n_old) and binary-search it against the sorted marked-edge codes.
        edge_codes = edge_list[:, 0] * n_old + edge_list[:, 1]

        def midvert(p: NDArray[np.int64], q: NDArray[np.int64]) -> NDArray[np.int64]:
            codes = np.minimum(p, q) * n_old + np.maximum(p, q)
            pos = np.clip(np.searchsorted(edge_codes, codes), 0, len(edge_codes) - 1)
            return np.where(edge_codes[pos] == codes, n_old + pos, -1)

        a, b, c = T[:, 0].astype(np.int64), T[:, 1].astype(np.int64), T[:, 2].astype(np.int64)
        m_ab, m_bc, m_ca = midvert(a, b), midvert(b, c), midvert(c, a)

        # Per-triangle subdivision pattern from which edges carry a midpoint. A shared edge is
        # split identically for both owning triangles, so the result is T-junction-free.
        pattern = (m_ab >= 0).astype(np.int8) | ((m_bc >= 0).astype(np.int8) << 1) | ((m_ca >= 0).astype(np.int8) << 2)

        chunks: list[NDArray[np.int64]] = []
        keys: list[NDArray[np.int64]] = []

        def emit(
            mask: NDArray[np.bool_], *tris: Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]
        ) -> None:
            if not mask.any():
                return
            src = np.flatnonzero(mask)  # original triangle indices (ascending)
            for sub, (v0, v1, v2) in enumerate(tris):
                chunks.append(np.stack([v0[mask], v1[mask], v2[mask]], axis=1))
                keys.append(src * 8 + sub)  # (original triangle, sub-triangle) ordering key

        emit(pattern == 0, (a, b, c))  # unchanged
        emit(pattern == 1, (a, m_ab, c), (m_ab, b, c))  # green refinement (1 edge)
        emit(pattern == 2, (a, b, m_bc), (a, m_bc, c))
        emit(pattern == 4, (a, b, m_ca), (b, c, m_ca))
        emit(pattern == 3, (a, m_ab, c), (m_ab, m_bc, c), (m_ab, b, m_bc))  # 2 edges
        emit(pattern == 5, (a, m_ab, m_ca), (m_ab, b, m_ca), (m_ca, b, c))
        emit(pattern == 6, (a, b, m_bc), (a, m_bc, m_ca), (m_ca, m_bc, c))
        emit(pattern == 7, (a, m_ab, m_ca), (m_ab, b, m_bc), (m_ca, m_bc, c), (m_ab, m_bc, m_ca))  # red (4 sub-tris)

        # Re-order sub-triangles back into original per-triangle emission order so face order
        # (hence seeded surface sampling on a reused mesh) matches the pre-vectorisation loop.
        emit_order = np.argsort(np.concatenate(keys), kind="stable")
        T = np.concatenate(chunks, axis=0)[emit_order].astype(np.int32)

    return V, T, C


def patch_and_densify_ground_mesh(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int32],
    vertex_colors: Optional[NDArray[np.uint8]],
    *,
    road_points_xyz: Optional[NDArray[np.float32]] = None,
    road_colors_rgb: Optional[NDArray[np.uint8]] = None,
    drop_floating_components: bool = True,
    fill_interior_holes: bool = True,
    densify_max_edge_length: Optional[float] = 0.3,
    densify_max_iters: int = 4,
    postdensify_smooth_passes: int = 5,
    elevation_axis: int = 2,
) -> Tuple[NDArray[np.float32], NDArray[np.int32], Optional[NDArray[np.uint8]]]:
    """Patch + densify a 2.5D ground mesh produced by DelaunayElevationMeshingAlgorithm.

    Steps (each optional):
    1. Drop floating components, keeping only the largest. Callers wanting a different retention
       rule should apply it to ``triangles`` first and pass ``drop_floating_components=False``;
       this is the first step, so the result is the same.
    2. Fill every interior boundary loop (closes gaps from parked cars / occluders).
    3. T-junction-aware midpoint densification of the remaining long triangles (typically
       the new hole-fill bridges), with optional kdtree-based midpoint coloring from
       ``road_points_xyz``/``road_colors_rgb``.
    4. Laplacian smoothing along ``elevation_axis`` to remove sub-triangle ridges.

    Returns ``(vertices, triangles, vertex_colors)`` with unreferenced vertices compacted out.
    """
    if len(triangles) == 0:
        return vertices.copy(), triangles.copy(), (None if vertex_colors is None else vertex_colors.copy())

    V = vertices
    T = triangles
    C = vertex_colors

    if drop_floating_components:
        keep = keep_largest_connected_component(T)
        T = T[keep]
        if len(T) == 0:
            return V.copy(), T.copy(), (None if C is None else C.copy())

    if fill_interior_holes:
        T = fill_interior_boundary_loops(V, T)

    if densify_max_edge_length is not None and densify_max_edge_length > 0.0:
        # Compact vertex set before densify so the cKDTree-free path stays bounded by
        # the actually-referenced vertices.
        referenced = np.unique(T.ravel())
        old_to_new = -np.ones(len(V), dtype=np.int64)
        old_to_new[referenced] = np.arange(len(referenced))
        V = V[referenced].astype(np.float32, copy=False)
        T = old_to_new[T].astype(np.int32, copy=False)
        if C is not None:
            C = C[referenced]
        logger.info(f"patch_and_densify_ground_mesh: pre-densify compact -> {len(V)} verts, {len(T)} tris")

        V, T, C = densify_mesh_midpoint(
            V,
            T,
            C,
            max_edge_length=densify_max_edge_length,
            max_iters=densify_max_iters,
            color_kdtree_points=road_points_xyz,
            color_kdtree_colors=road_colors_rgb,
        )
        logger.info(f"patch_and_densify_ground_mesh: post-densify {len(V)} verts, {len(T)} tris")

    if postdensify_smooth_passes > 0 and len(T) > 0:
        vertex_neighbors, vertex_distances = get_vertex_neighbors(V, T)
        for _ in range(postdensify_smooth_passes):
            V = mesh_smoothing_along_axis(
                V,
                vertex_neighbors,
                vertex_distances,
                axis=elevation_axis,
                method="neighbor_average",
            )
        logger.info(f"patch_and_densify_ground_mesh: post-densify smoothing {postdensify_smooth_passes} passes")

    return V.astype(np.float32, copy=False), T.astype(np.int32, copy=False), C
