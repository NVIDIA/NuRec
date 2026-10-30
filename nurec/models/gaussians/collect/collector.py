# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gaussian Parameter Collector Interface.

This module defines the interface for collecting and transforming Gaussian parameters
from multiple layers into a unified representation suitable for rendering.

Overview:
---------
The Gaussian parameter collector is responsible for:
1. Gathering Gaussian parameters (positions, rotations, scales, densities, features) from multiple layers
2. Applying activations (e.g., exp to scales, sigmoid to densities, normalization to rotations)
3. Transforming rigid and deformable layers using track poses
4. Concatenating all layers into unified output tensors
5. Computing time-based embeddings for temporal features (Fourier features)
6. Preparing input embeddings for hash grids (xyz + instance + time)

Architecture:
-------------
The collector supports multiple layer types:
- Base layers: Static Gaussians with basic properties
- Spherical Harmonics (SH) layers: Add view-dependent features (albedo/specular)
- Rigid layers: Gaussians attached to moving tracks (rigid transformations)
- Deformable layers: Gaussians with learned deformations on top of rigid motion

Workflow:
---------
1. Setup (once at collector creation):
   - Configuration: Define layer types and their properties via LayersConfig
2. Per-collection (every training step or rendered frame):
   - Data preparation: Pack per-layer data into LayerData structures
   - Collection: Call collect() to transform and concatenate all layers
   - Output: Receive unified tensors ready for rendering

The implementation uses Slang GPU kernels for high-performance parallel processing.
"""

import math

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Tuple

import torch

from nurec.models.tracks_calib_data import TracksCalibData, TracksCalibResiduals


# Type of activation supported by this library.
class RotationActivation(Enum):
    """Activation functions applied to rotation quaternions.

    NORMALIZE: Normalize quaternions to unit length to ensure valid rotations.
    """

    NORMALIZE = 0


class ScaleActivation(Enum):
    """Activation functions applied to scale parameters.

    EXP: Apply exponential to ensure positive scales.
    """

    EXP = 0


class DensityActivation(Enum):
    """Activation functions applied to density/opacity values.

    SIGMOID: Apply sigmoid to constrain densities to [0, 1] range.
    """

    SIGMOID = 0


#
# Config classes, describes what is in the layer.
#
@dataclass(slots=True)
class LayerConfigBase:
    """Base configuration for a Gaussian layer.

    Defines the activation functions to apply during collection.

    Attributes:
        rotation_activation: Activation function for rotations (e.g., normalization)
        scale_activation: Activation function for scales (e.g., exp)
        density_activation: Activation function for densities (e.g., sigmoid)
    """

    rotation_activation: RotationActivation
    scale_activation: ScaleActivation
    density_activation: DensityActivation


@dataclass(slots=True)
class EmbeddingConfig:
    """Base class for time embedding configurations."""

    pass


@dataclass(slots=True)
class IndividualRemapTimeInputEmbeddingConfig(EmbeddingConfig):
    """Time embedding with per-instance timestamp ranges.

    Each instance (e.g., dynamic object) has its own time range that gets
    remapped to a normalized interval for embedding.

    Attributes:
        timestamps_us_ranges: Tensor of shape [num_instances, 2] containing
            [min_timestamp, max_timestamp] in microseconds for each instance
        remap_min: Minimum value of the remapped time range
        remap_max: Maximum value of the remapped time range
    """

    timestamps_us_ranges: torch.Tensor
    remap_min: float
    remap_max: float


@dataclass(slots=True)
class HolisticRemapTimeInputEmbeddingConfig(EmbeddingConfig):
    """Time embedding with a single global timestamp range.

    All instances share the same time range for normalization.

    Attributes:
        timestamps_us_min: Minimum timestamp in microseconds
        timestamps_us_max: Maximum timestamp in microseconds
        remap_min: Minimum value of the remapped time range
        remap_max: Maximum value of the remapped time range
    """

    timestamps_us_min: int
    timestamps_us_max: int
    remap_min: float
    remap_max: float


@dataclass(slots=True)
class IndividualStepTimeInputEmbeddingConfig(EmbeddingConfig):
    """Step-function time embedding with per-instance timestamp ranges.

    Uses a differentiable step function from NSC paper (https://arxiv.org/abs/2306.07970) for temporal embedding,
    with per-instance time ranges.

    Attributes:
        timestamps_us_ranges: Tensor of shape [num_instances, 2] (int64) containing
            [min_timestamp, max_timestamp] in microseconds for each instance
        n_steps: Number of steps in the step function
        n_dims: Number of output dimensions
    """

    timestamps_us_ranges: torch.Tensor  # [num_instances, 2], int64
    n_steps: int
    n_dims: int


def swap_time_embedding_table(
    embed_config: "IndividualRemapTimeInputEmbeddingConfig | IndividualStepTimeInputEmbeddingConfig",
    new_table: torch.Tensor,
    *,
    min_rows: int,
) -> torch.Tensor:
    """Re-point a live handler's per-instance time table at ``new_table``, validating it first.

    Handlers hold ``embed_config`` by reference and re-read ``timestamps_us_ranges`` every collect, so
    reassigning the field takes effect on the next render with no rebuild/recompile (table length is not a
    kernel-spec key). The Slang kernel gathers ``timestamps_us_ranges[instance_idx]`` with NO bounds check
    and the handler's build-time dtype/shape/order asserts never re-run — this helper is the only safety
    net for any table the kernel can see after construction:

    - int64, shape ``[n, 2]``, ``requires_grad=False`` (the build-time invariants);
    - same device as the table it replaces — a cross-device table passes every value/shape check and then
      dies at the kernel launch that gathers ``timestamps_us_ranges`` (``.contiguous()`` does not move
      devices, and the rebuilt table's device follows the request's tracks);
    - strict ``col0 < col1`` — degenerate single-timestamp rows (min == max, e.g. a one-frame track from
      the segment-min/max oracle) are widened by +1 µs on a copy rather than rejected;
    - ``n >= min_rows`` — the caller passes the maximum live ``instance_idx`` + 1 so no gather can run
      past the end.

    Returns the installed table (``new_table`` itself, or the widened copy).
    """
    if not isinstance(embed_config, (IndividualRemapTimeInputEmbeddingConfig, IndividualStepTimeInputEmbeddingConfig)):
        raise TypeError(f"swap_time_embedding_table: config carries no per-instance table: {type(embed_config)}")
    old_device = embed_config.timestamps_us_ranges.device
    if new_table.device != old_device:
        raise ValueError(
            f"swap_time_embedding_table: device change not allowed: existing table on {old_device} but new "
            f"table on {new_table.device}; the Slang gather runs on the collector's device and a cross-device "
            f"table faults at kernel launch"
        )
    if new_table.dtype != torch.int64:
        raise ValueError(f"timestamps_us_ranges must be int64, got {new_table.dtype}")
    if new_table.ndim != 2 or new_table.shape[1] != 2:
        raise ValueError(f"timestamps_us_ranges must be [n, 2], got {tuple(new_table.shape)}")
    if new_table.requires_grad:
        raise ValueError("timestamps_us_ranges must not require grad")
    if int(new_table.shape[0]) < min_rows:
        raise ValueError(
            f"timestamps_us_ranges has {int(new_table.shape[0])} rows but must cover instance indices up to "
            f"{min_rows - 1} (min_rows={min_rows}); an under-length table is an out-of-bounds kernel gather"
        )
    degenerate = new_table[:, 0] == new_table[:, 1]
    if bool(degenerate.any()):
        new_table = new_table.clone()
        new_table[:, 1] = torch.where(degenerate, new_table[:, 1] + 1, new_table[:, 1])
    if not bool((new_table[:, 0] < new_table[:, 1]).all()):
        raise ValueError("timestamps_us_ranges must satisfy col0 < col1 for every row")
    embed_config.timestamps_us_ranges = new_table
    return new_table


@dataclass(slots=True)
class LayerConfigSH(LayerConfigBase):
    """Configuration for a Spherical Harmonics layer with view-dependent features.

    Extends base layer with albedo and specular features, optionally with
    Fourier-based temporal modulation.

    Attributes:
        fourier_features_dim: Number of Fourier frequency bands for temporal features.
            If > 1, features are modulated over time using Fourier basis.
            If == 1, features are static (no temporal modulation).
        embed_config: Time embedding configuration (required if fourier_features_dim > 1,
            None otherwise)
    """

    fourier_features_dim: int
    embed_config: EmbeddingConfig | None


@dataclass(slots=True)
class LayerConfigRigid(LayerConfigSH):
    """Configuration for a rigid layer.

    Gaussians are attached to tracks (moving objects) and undergo rigid
    transformations based on track poses.
    """

    pass


@dataclass(slots=True)
class LayerConfigDeformable(LayerConfigRigid):
    """Configuration for a deformable layer.

    Extends rigid layer with learned per-Gaussian deformations applied
    on top of the rigid track motion.
    """

    pass


@dataclass(slots=True)
class LayersConfig:
    """Complete configuration for all layers in the collector.

    Attributes:
        layers: List of per-layer configurations
        extra_signal_dim: Dimension of extra signal features (shared across all layers)
        camera_extra_signal_dim: Dimension of camera-specific features (shared across all layers)
        lidar_extra_signal_dim: Dimension of lidar-specific features (shared across all layers)
        albedo_dim: Dimension of albedo (diffuse color) features
        specular_dim: Dimension of specular (view-dependent) features
        sh_features_dtype: Storage dtype for SH features ("fp32" or "fp16")
    """

    layers: list[LayerConfigBase]
    extra_signal_dim: int
    camera_extra_signal_dim: int
    lidar_extra_signal_dim: int
    albedo_dim: int
    specular_dim: int
    sh_features_dtype: Literal["fp32", "fp16"] = "fp32"


#
# Each layer's interface to the per-layer data needed by the collector.
#
@dataclass(slots=True)
class LayerDataBase:
    """Base data for a Gaussian layer.

    Contains the fundamental parameters for all Gaussians in a layer.

    Attributes:
        positions: Tensor of shape [num_gaussians, 3] - 3D positions in local space
        rotations: Tensor of shape [num_gaussians, 4] - Quaternions in wxyz format
        scales: Tensor of shape [num_gaussians, 3] - Log-space scales (before exp activation)
        densities: Tensor of shape [num_gaussians, 1] - Logit-space densities (before sigmoid)
        extra_signal: Tensor of shape [num_gaussians, extra_signal_dim] - Generic extra features
        camera_extra_signal: Tensor of shape [num_gaussians, camera_extra_signal_dim] - Camera features
        lidar_extra_signal: Tensor of shape [num_gaussians, lidar_extra_signal_dim] - Lidar features
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    extra_signal: torch.Tensor
    camera_extra_signal: torch.Tensor
    lidar_extra_signal: torch.Tensor


@dataclass(slots=True)
class EmbeddingData:
    """Base class for per-layer embedding data."""

    pass


@dataclass(slots=True)
class IndividualRemapTimeInputEmbeddingData(EmbeddingData):
    """Per-Gaussian instance indices for individual time embedding.

    Attributes:
        instance_idx: Tensor of shape [num_gaussians] (int32) - Instance ID for each Gaussian,
            used to look up per-instance timestamp ranges
    """

    instance_idx: torch.Tensor


@dataclass(slots=True)
class IndividualStepTimeInputEmbeddingData(EmbeddingData):
    """Per-Gaussian data for step time embedding.

    Attributes:
        instance_idx: Tensor of shape [num_gaussians] (int32)
        u: Tensor of shape [num_instances, n_dims, n_steps] - current step positions (trainable)
        beta: Current beta value for stepness
    """

    instance_idx: torch.Tensor
    u: torch.Tensor  # Trainable!
    beta: torch.Tensor


@dataclass(slots=True)
class LayerDataSH(LayerDataBase):
    """Data for a Spherical Harmonics layer with view-dependent features.

    Extends base layer with appearance features.

    Attributes:
        features_albedo: Tensor of shape [num_gaussians, fourier_dim, albedo_dim] if using
            Fourier features, or [num_gaussians, albedo_dim] otherwise - Diffuse color features
        features_specular: Tensor of shape [num_gaussians, specular_dim] - View-dependent features
        embed_data: Optional embedding data for time-varying features
    """

    features_albedo: torch.Tensor
    features_specular: torch.Tensor
    embed_data: EmbeddingData | None


@dataclass(slots=True, frozen=True)
class DirectTracksCalibData(TracksCalibData):
    """Direct parameterization of track calibration.

    Applies learned delta transformations in either the world or cuboid-local
    frame.

    Attributes:
        gradient_mask: Tensor of shape [num_poses] (bool) - Whether to allow gradients per pose.
            Typically used to fix the first and last pose of a track, preventing them from
            being modified by optimization.
        tracks_delta_q: Tensor of shape [num_poses, 4] - Quaternion deltas (xyzw format)
        tracks_delta_t: Tensor of shape [num_poses, 3] - Translation deltas
        left_multiply: Whether to express deltas in the world frame by left
            multiplication. Right multiplication expresses them in the cuboid-local frame.
    """

    gradient_mask: torch.Tensor
    tracks_delta_q: torch.Tensor
    tracks_delta_t: torch.Tensor
    left_multiply: bool = True


@dataclass(slots=True)
class TracksTimestampsData:
    """Base class for timestamp selection strategies during track interpolation."""

    pass


@dataclass(slots=True)
class TracksTimestampsGlobalData(TracksTimestampsData):
    """Use a single timestamp for all tracks.

    Attributes:
        timestamp: Timestamp in microseconds (int)
    """

    timestamp: int


@dataclass(slots=True)
class TracksTimestampsPerTrackData(TracksTimestampsData):
    """Use pre-specified timestamps per track.

    Attributes:
        timestamps: Tensor of shape [num_tracks] (int64) - Timestamp in microseconds per track
    """

    timestamps: torch.Tensor


@dataclass(slots=True)
class TracksFrustumCullData:
    """Data for the optional track frustum pre-cull.

    Attributes:
        rays: Flattened frustum rays of shape ``[height * width, 6]``.
        rays_height: Frustum ray-grid row count.
        rays_width: Frustum ray-grid column count.
        margin_rad: Angular slack in radians applied to each of the 4
            frustum side planes. Default 5 degrees.
    """

    rays: torch.Tensor
    rays_height: int
    rays_width: int
    margin_rad: float = math.radians(5.0)


@dataclass(slots=True)
class TracksTimestampsEstimationData(TracksTimestampsData):
    """Estimate timestamps per track via ray-cuboid intersection.

    ``rays`` are always the rays used for ray-cuboid timestamp estimation.
    ``frustum_cull`` optionally supplies a separate frustum ray grid for the
    spatial pre-cull.  When it is ``None``, the pre-pass only applies
    time-range culling.

    Camera timestamp estimation normally uses the same flattened camera rays
    for ``rays`` and ``frustum_cull.rays``.  Lidar timestamp estimation uses
    lidar rays for ``rays``.  Since a normal lidar bundle covers 360 degrees,
    the lidar pre-pass usually leaves ``frustum_cull`` unset.  When active
    tile rendering limits the lidar pass to tiles overlapping the active
    camera view, ``frustum_cull.rays`` can be set to camera frustum rays so
    the lidar pass culls tracks outside the camera-visible region before
    running lidar ray-cuboid intersections.

    Attributes:
        rays: Tensor of shape [num_rays, 6] - Ray origins (3) and directions (3)
        rays_timestamps: Tensor of shape [num_rays] (int64) - Timestamp per ray in microseconds
        cuboids_dimensions: Tensor of shape [num_tracks, 3] - Bounding box dimensions per track
        default_timestamp: Default timestamp in microseconds for tracks with no intersections
        frame_start_timestamp: Frame exposure start in microseconds, used by
            the culling pre-pass to interpolate per-track poses inside the
            rolling-shutter window.  Required.
        frame_end_timestamp: Frame exposure end in microseconds.  Required.
        rays_timestamp_layout: Optional producer guarantee for shared timestamps per row,
            column, or image. None retains uncached estimation. This describes the
            estimation rays, independently of the optional frustum rays.
        rays_grid_shape: Optional (batch, height, width) before flattening, after any
            rectangular subsampling. Required for caching; unknown/mismatched shapes
            retain uncached estimation. Arbitrary spatial reordering must clear layout.
    """

    rays: torch.Tensor
    rays_timestamps: torch.Tensor
    cuboids_dimensions: torch.Tensor
    default_timestamp: int
    frame_start_timestamp: int
    frame_end_timestamp: int
    frustum_cull: Optional[TracksFrustumCullData] = None
    rays_timestamp_layout: Optional[Literal["rows", "columns", "global"]] = None
    rays_grid_shape: Optional[Tuple[int, int, int]] = None


@dataclass(slots=True)
class TracksInterpolationData:
    """Data for interpolating track poses at query timestamps.

    Tracks are represented as time-series of poses, packed into flat arrays.
    Endpoint visibility uses a per-track cadence plus the current frame exposure.
    The cadence is topology metadata and should be computed when the tracks are
    constructed, then reused for every interpolation query.

    Attributes:
        tracks_poses: Tensor of shape [total_num_poses, 7] - All poses concatenated,
            format: [translation(3), quaternion_xyzw(4)]
        tracks_timestamps: Tensor of shape [total_num_poses] (int64) - Timestamp per pose
        tracks_packinfo: Tensor of shape [num_tracks, 2] (int32) - [start_index, length] per track
        timestamps_data: Strategy for selecting query timestamps
        nearest_neighbor: If True, use nearest neighbor instead of interpolation
        max_positive_timestamp_gap_us: Tensor of shape [num_tracks] (int64) containing the derived maximum
            strictly-positive timestamp gap for each track.
        frame_exposure_us: Non-negative frame exposure duration. The interpolation kernel adds this scalar to
            each track's cadence before deciding whether an out-of-range timestamp may clamp to an endpoint.
    """

    tracks_poses: torch.Tensor
    tracks_timestamps: torch.Tensor
    tracks_packinfo: torch.Tensor
    timestamps_data: TracksTimestampsData
    nearest_neighbor: bool
    max_positive_timestamp_gap_us: torch.Tensor
    frame_exposure_us: int = 0


@dataclass(slots=True)
class LayerDataRigid(LayerDataSH):
    """Data for a rigid layer attached to moving tracks.

    Each Gaussian is associated with a track and transforms according to
    the track's pose.

    Attributes:
        poses: Tensor of shape [num_tracks, 7] - Interpolated/calibrated poses for tracks,
            format: [translation(3), quaternion_xyzw(4)]
        keep_mask: Tensor of shape [num_tracks] (bool) - Whether each track is visible/active
        tracks_ids: Tensor of shape [num_gaussians] (int32) - Track ID for each Gaussian
    """

    poses: torch.Tensor
    keep_mask: torch.Tensor
    tracks_ids: torch.Tensor


@dataclass(slots=True)
class SceneContractorData:
    """Configuration for scene space contraction.

    Contracts world space into a bounded domain for hash grid encoding.

    Attributes:
        aabb_blb: Tensor of shape [num_instances, 3] - Axis-aligned bounding box
            bottom-left-back corner per instance
        aabb_trf: Tensor of shape [num_instances, 3] - Axis-aligned bounding box
            top-right-front corner per instance
        degree: Lp-norm degree for contraction (e.g., 2.0 for L2, inf for L-infinity),
            or None for no contraction
        is_merf: If True, use MERF-style contraction; otherwise use standard contraction
    """

    aabb_blb: torch.Tensor
    aabb_trf: torch.Tensor
    degree: float | None
    is_merf: bool


@dataclass(slots=True)
class InstanceEmbeddingData:
    """Base class for instance embedding data."""

    pass


@dataclass(slots=True)
class WeightedInstanceInputEmbeddingData(InstanceEmbeddingData):
    """Instance embeddings via learned weight vectors.

    Attributes:
        instance_embedding_weights: Tensor of shape [num_instances, embedding_dim] -
            Learned embedding vectors per instance
    """

    instance_embedding_weights: torch.Tensor


@dataclass(slots=True)
class InputEmbeddingData:
    """Data for computing input embeddings for hash grid (xyz + instance_emb + time_emb).

    Prepares concatenated feature vectors suitable for hash grid input, combining
    spatial coordinates, instance identity, and temporal information.

    Attributes:
        xyzs: Tensor of shape [num_gaussians, 3] - 3D world coordinates
        instance_idx: Tensor of shape [num_gaussians] (int32) - Instance ID per Gaussian
        timestamps_us: Timestamp in microseconds, either a single value for all
            Gaussians or a compact tensor of shape [num_instances] with one
            timestamp per track/instance. ``instance_idx`` maps each Gaussian
            to its timestamp.
        scene_contractor: Configuration for spatial contraction
        instance_embedding_weights: Instance embedding configuration
        time_embedding_config: Time embedding configuration
        timestamps_delta: Optional tensor of shape [num_gaussians] (int64) - Per-Gaussian timestamp deltas
            in microseconds. If provided, output shape becomes [num_gaussians*3, base_dim] with embeddings
            stacked as [t, t-delta, t+delta] along the batch dimension
    """

    xyzs: torch.Tensor
    instance_idx: torch.Tensor
    timestamps_us: int | torch.Tensor

    scene_contractor: SceneContractorData
    instance_embedding_weights: InstanceEmbeddingData
    time_embedding_config: EmbeddingConfig

    timestamps_delta: torch.Tensor | None = None


@dataclass(slots=True)
class LayerDataDeformable(LayerDataRigid):
    """Data for a deformable layer with learned per-Gaussian deformations.

    Extends rigid layer with optional deformation offsets applied in local space
    before the rigid transformation.

    Attributes:
        deform_positions: Optional tensor of shape [num_gaussians, 3] - Position offsets
        deform_rotations: Optional tensor of shape [num_gaussians, 4] - Rotation offsets
            (quaternion xyzw format, as offset from identity)
    """

    deform_positions: torch.Tensor | None
    deform_rotations: torch.Tensor | None


@dataclass(slots=True, frozen=True)
class LayerSegments:
    """Plain data describing which layer-local rows a collect writes.

    ``starts`` ([W] int32) holds the first row of each window and ``window_size_prefix``
    ([W + 1] int32) the exclusive prefix of the window sizes. Both are ``None`` for a
    whole-layer or empty collect because the specialized dense kernel binds neither.

    A windowed collect writes only those rows in place; rows outside the windows keep
    whatever the destination held, which makes it inference-only. ``n_segments`` is 0
    for a whole layer or an empty collect and W otherwise. ``active_count``
    distinguishes whole from empty.
    """

    starts: torch.Tensor | None
    window_size_prefix: torch.Tensor | None
    n_segments: int
    active_count: int


def windowed_layer_segments(starts: torch.Tensor, window_size_prefix: torch.Tensor, active_count: int) -> LayerSegments:
    """Describe a windowed layer collect."""
    return LayerSegments(
        starts=starts,
        window_size_prefix=window_size_prefix,
        n_segments=int(starts.shape[0]),
        active_count=int(active_count),
    )


def whole_layer_segments(row_count: int) -> LayerSegments:
    """Describe a dense whole-layer collect, or no work when ``row_count == 0``."""
    return LayerSegments(starts=None, window_size_prefix=None, n_segments=0, active_count=int(row_count))


@dataclass(slots=True, frozen=True)
class TrackTimeData:
    """Pose timestamps of a tracked layer, as seen by the pose interpolation.

    These are the same three tensors ``TracksInterpolationData`` carries (there under
    ``tracks_timestamps``); this is the subset that describes a track's time topology,
    with none of the per-frame query strategy, so the host can decide which tracks a
    frame can reach without holding a frame's interpolation request.

    Attributes:
        tracks_timestamps_us: Tensor of shape [total_num_poses] (int64)
        tracks_packinfo: Tensor of shape [num_tracks, 2] (int32) - [start_index, length] per track
        max_positive_timestamp_gap_us: Tensor of shape [num_tracks] (int64) - endpoint tolerance per track
    """

    tracks_timestamps_us: torch.Tensor
    tracks_packinfo: torch.Tensor
    max_positive_timestamp_gap_us: torch.Tensor


@dataclass(slots=True)
class LayersData:
    """Complete input data for all layers.

    Attributes:
        layers: List of per-layer data structures
        frame_timestamp_us: Global frame timestamp in microseconds (used for Fourier features)
    """

    layers: list[LayerDataBase]
    frame_timestamp_us: int


# The returned concatenated tensors.
@dataclass(slots=True)
class CollectorResult:
    """Unified output from the collector.

    All tensors are concatenations across all layers, with activations applied.

    Attributes:
        positions: Tensor of shape [total_gaussians, 3] - World-space positions
        rotations: Tensor of shape [total_gaussians, 4] - Normalized quaternions (wxyz format)
        scales: Tensor of shape [total_gaussians, 3] - Positive scales (after exp)
        densities: Tensor of shape [total_gaussians, 1] - Densities in [0,1] (after sigmoid)
        extra_signal: Tensor of shape [total_gaussians, extra_signal_dim]
        camera_extra_signal: Tensor of shape [total_gaussians, camera_extra_signal_dim]
        lidar_extra_signal: Tensor of shape [total_gaussians, lidar_extra_signal_dim]
        features: Tensor of shape [total_gaussians, albedo_dim + specular_dim] -
            Concatenated appearance features (with Fourier applied if configured)
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    extra_signal: torch.Tensor
    camera_extra_signal: torch.Tensor
    lidar_extra_signal: torch.Tensor
    features: torch.Tensor


# The actual collector interface.
class GaussianParameterCollector(ABC):
    """Abstract interface for Gaussian parameter collection and transformation.

    This is the main interface for collecting, transforming, and preparing Gaussian
    parameters for rendering. Implementations use GPU kernels for high performance.
    """

    # Every implementation captures its construction config; handlers hold the contained layer configs by
    # reference, so post-construction table swaps (swap_time_embedding_table) go through this object.
    layers_config: LayersConfig

    @abstractmethod
    def __init__(self, layers_config: LayersConfig):
        """Initialize the collector with layer configuration.

        Args:
            layers_config: Configuration specifying layer types and dimensions
        """
        ...

    @abstractmethod
    def collect(
        self,
        layers_data: LayersData,
        layer_indices: Optional[List[int]] = None,
        out: Optional[CollectorResult] = None,
        segments: Optional[List[LayerSegments]] = None,
    ) -> CollectorResult:
        """Collect and transform Gaussians from all layers.

        This is the main entry point that:
        1. Applies activations to base parameters
        2. Transforms rigid/deformable layers using track poses
        3. Computes time-varying features (Fourier)
        4. Concatenates all layers into unified output tensors

        Args:
            layers_data: Input data for all layers
            layer_indices: Optional mapping from each entry in layers_data.layers
                to the corresponding layer_handler index. When None, layers_data.layers
                must contain all layers in order. When provided, enables collecting
                a subset of layers using the correct handler and layers_data.layers should
                only contain the subset of the collected layers, in the order matching the indices
            out: Optional caller-owned destination. When given, the collected layers are
                written straight into its tensors instead of freshly allocated ones, and it
                is returned as-is. Its rows must match the collected gaussian count exactly
                and each field must be contiguous. Inference-only: the destination is
                typically an aliased view, so gradients would not flow back to it.
            segments: Optional per-layer LayerSegments, aligned with layers_data.layers.
                ``None`` covers every layer whole, which is the dense collect; a caller
                that has a plan passes one LayerSegments per layer, using
                ``whole_layer_segments`` for the layers it wants collected whole. A
                layer given ``windowed_layer_segments`` is written in place and only
                within its windows, so a windowed collect is inference-only.

        Returns:
            CollectorResult containing transformed and concatenated parameters
        """
        ...

    @abstractmethod
    def calibrate_tracks_poses(
        self,
        tracks_poses: torch.Tensor,
        tracks_calib_data: TracksCalibData,
    ) -> torch.Tensor:
        """Apply calibration corrections to track poses.

        Adjusts track poses using learned extrinsic calibration parameters.

        Args:
            tracks_poses: Base poses as ``[translation(3), quaternion_xyzw(4)]``.
            tracks_calib_data: Calibration parameters and fixed track metadata.

        Returns:
            Calibrated poses tensor of shape [num_poses, 7] in format
            [translation(3), quaternion_xyzw(4)]
        """
        ...

    @abstractmethod
    def compute_tracks_calib_residuals(self, tracks_calib_data: TracksCalibData) -> TracksCalibResiduals:
        """Compute residuals for a structured calibrator supported by this collector.

        Args:
            tracks_calib_data: Calibration parameters and fixed track metadata.

        Returns:
            The consistency, smoothness, and pose-regularization residuals.

        Raises:
            ValueError: If this collector does not support the supplied calibrator data.
        """
        ...

    @abstractmethod
    def interpolate_tracks_poses(
        self, tracks_interpolation_data: TracksInterpolationData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpolate track poses at query timestamps.

        Given time-series of poses per track, interpolates to find poses at
        specified timestamps using either linear interpolation or nearest neighbor.

        Args:
            tracks_interpolation_data: Packed poses, timestamps, and query strategy

        Returns:
            Tuple of:
            - poses: Tensor of shape [num_tracks, 7] - Interpolated poses
            - inside: Tensor of shape [num_tracks] (bool) - Whether the timestamp
              is accepted for the track, including endpoint clamping within the
              configured cadence-plus-exposure tolerance
            - timestamps: Tensor of shape [num_tracks] (int64) - Effective
              deformation-conditioning timestamp for each track. Interpolation
              and nearest-neighbor pose lookup retain the query timestamp while
              an accepted out-of-range query uses the clamped endpoint timestamp.
        """
        ...

    @abstractmethod
    def prepare_input_embeddings(self, input_embedding_data: InputEmbeddingData) -> torch.Tensor:
        """Prepare input embeddings for hash grid encoding.

        Computes concatenated feature vectors combining spatial coordinates,
        instance embeddings, and time embeddings.

        Args:
            input_embedding_data: Input coordinates, instance IDs, and embedding configs

        Returns:
            Tensor of shape:
            - [num_gaussians, base_dim] if timestamps_delta is None
            - [num_gaussians*3, base_dim] if timestamps_delta is provided (stacked as [t, t-delta, t+delta])
            where base_dim = 3 (xyz) + instance_embedding_dim + 1 (time)
        """
        ...
