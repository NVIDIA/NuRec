# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometric multi-view deformable attention for Kelvin point queries.

Every retained 3D point is projected into every context camera. The exact
projected pixel is always sampled; additional query-conditioned offsets provide
a small deformable neighborhood. Only in-frustum, depth-visible query/view
pairs enter feature sampling and attention. Fusion contains no depth-confidence
input.
"""

import math

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from torch import nn

from nurec.utils.geometry import se3_matrix_inverse, se3_matrix_to_tquat, tquat_to_se3_matrix


class _ProjectionReturn(Protocol):
    image_points: torch.Tensor
    valid_indices: torch.Tensor | None
    timestamps_us: torch.Tensor | None
    T_world_sensors: torch.Tensor | None


ProjectionFn = Callable[[Any, torch.Tensor, torch.Tensor, torch.Tensor, bool], _ProjectionReturn]


@dataclass(frozen=True, kw_only=True, slots=True)
class ProjectedPointReferences:
    """Exact rolling-shutter projection of queries into all context views.

    Shapes are ``[B,V,N,*]``. Invalid entries are finite zeros.
    """

    image_points: torch.Tensor
    valid_mask: torch.Tensor
    timestamps_us: torch.Tensor
    query_metric_distance: torch.Tensor


@dataclass(frozen=True, kw_only=True, slots=True)
class DeformableFusionOutput:
    features: torch.Tensor
    valid_observation_count: torch.Tensor
    view_valid_query_count: torch.Tensor
    view_attention_mass: torch.Tensor
    in_frustum_observation_count: torch.Tensor
    depth_tested_observation_count: torch.Tensor
    occluded_observation_count: torch.Tensor
    deformed_sample_attention_mass: torch.Tensor
    learned_offset_magnitude_sum: torch.Tensor
    attention_entropy: torch.Tensor


def _default_projection_fn(
    camera_model_parameters: Any,
    world_points: torch.Tensor,
    T_world_sensors: torch.Tensor,
    timestamps_us: torch.Tensor,
    return_all_projections: bool,
) -> _ProjectionReturn:
    # Lazy import keeps the tensor-only unit tests CPU-compatible.
    from nurec.libs.vren.interface import world_points_to_image_points_shutter_pose

    return world_points_to_image_points_shutter_pose(
        camera_model_parameters,
        world_points,
        T_world_sensors,
        timestamps_us,
        return_all_projections=return_all_projections,
    )


def _sensor_to_world_to_world_to_sensor(sensor_to_world_tquats: torch.Tensor) -> torch.Tensor:
    sensor_to_world = tquat_to_se3_matrix(sensor_to_world_tquats, unbatch=False)
    world_to_sensor = se3_matrix_inverse(sensor_to_world, unbatch=False)
    return se3_matrix_to_tquat(world_to_sensor, unbatch=False)


def _metric_distance_in_sensor_frame(points_world: torch.Tensor, world_to_sensor_tquats: torch.Tensor) -> torch.Tensor:
    world_to_sensor = tquat_to_se3_matrix(world_to_sensor_tquats, unbatch=False)
    points_sensor = torch.bmm(world_to_sensor[:, :3, :3], points_world.unsqueeze(-1)).squeeze(-1)
    points_sensor = points_sensor + world_to_sensor[:, :3, 3]
    return torch.linalg.vector_norm(points_sensor, dim=-1)


@torch.no_grad()
def project_points_to_context_views(
    query_xyz_world: torch.Tensor,
    query_valid_mask: torch.Tensor,
    camera_model_parameters: Sequence[Sequence[Any]],
    sensor_to_world_tquats_startend: torch.Tensor,
    frame_timestamps_startend_us: torch.Tensor,
    *,
    projection_fn: ProjectionFn | None = None,
) -> ProjectedPointReferences:
    """Project every valid query once per view using the exact NCore camera.

    Projection is intentionally outside the feature-fusion chunks: this makes
    one CUDA projection launch per batch/view rather than one per view/chunk.
    The camera kernel is non-differentiable; query XYZ still receives render
    gradients directly as the emitted Gaussian position and depth supervision.
    """
    if query_xyz_world.ndim != 3 or query_xyz_world.shape[-1] != 3:
        raise ValueError(f"query_xyz_world must be [B,N,3], got {tuple(query_xyz_world.shape)}")
    batch_size, num_queries, _ = query_xyz_world.shape
    if query_valid_mask.shape != (batch_size, num_queries) or query_valid_mask.dtype != torch.bool:
        raise ValueError("query_valid_mask must be bool [B,N]")
    if sensor_to_world_tquats_startend.ndim != 4 or sensor_to_world_tquats_startend.shape[:1] != (batch_size,):
        raise ValueError("sensor_to_world_tquats_startend must be [B,V,2,7]")
    if sensor_to_world_tquats_startend.shape[2:] != (2, 7):
        raise ValueError("sensor_to_world_tquats_startend must be [B,V,2,7]")
    if sensor_to_world_tquats_startend.device != query_xyz_world.device:
        raise ValueError("query points and camera poses must be on the same device")
    num_views = sensor_to_world_tquats_startend.shape[1]
    if frame_timestamps_startend_us.shape != (batch_size, num_views, 2):
        raise ValueError("frame_timestamps_startend_us must be [B,V,2]")
    if frame_timestamps_startend_us.dtype != torch.int64:
        raise TypeError("frame timestamps must be int64")
    if len(camera_model_parameters) != batch_size or any(len(row) != num_views for row in camera_model_parameters):
        raise ValueError("camera_model_parameters must be nested [B][V]")

    project = projection_fn or _default_projection_fn
    device = query_xyz_world.device
    image_points = torch.zeros((batch_size, num_views, num_queries, 2), device=device, dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, num_views, num_queries), device=device, dtype=torch.bool)
    timestamps_us = torch.zeros((batch_size, num_views, num_queries), device=device, dtype=torch.int64)
    metric_distance = torch.zeros((batch_size, num_views, num_queries), device=device, dtype=torch.float32)
    finite_query = torch.isfinite(query_xyz_world).all(dim=-1)

    for batch_idx in range(batch_size):
        candidate_indices = torch.where(query_valid_mask[batch_idx] & finite_query[batch_idx])[0]
        if candidate_indices.numel() == 0:
            continue
        candidate_xyz = query_xyz_world[batch_idx, candidate_indices].detach().float()
        for view_idx in range(num_views):
            world_to_sensor_startend = _sensor_to_world_to_world_to_sensor(
                sensor_to_world_tquats_startend[batch_idx, view_idx].float()
            )
            projection = project(
                camera_model_parameters[batch_idx][view_idx],
                candidate_xyz,
                world_to_sensor_startend,
                frame_timestamps_startend_us[batch_idx, view_idx],
                True,
            )
            if (
                projection.valid_indices is None
                or projection.timestamps_us is None
                or projection.T_world_sensors is None
            ):
                raise ValueError("projection must return valid indices, timestamps, and per-point poses")
            compact = projection.valid_indices.to(device=device, dtype=torch.int64)
            if bool(((compact < 0) | (compact >= candidate_indices.numel())).any()):
                raise IndexError("projection returned an out-of-range valid index")
            if projection.image_points.shape != (candidate_indices.numel(), 2):
                raise ValueError("return_all_projections must return one image point per candidate")
            target = candidate_indices[compact]
            if target.numel() == 0:
                continue
            poses = projection.T_world_sensors.to(device=device, dtype=torch.float32)
            if poses.shape != (target.numel(), 7):
                raise ValueError("projection T_world_sensors must be [num_valid,7]")
            if projection.timestamps_us.shape != (target.numel(),):
                raise ValueError("projection timestamps_us must be [num_valid]")
            distances = _metric_distance_in_sensor_frame(candidate_xyz[compact], poses)
            compact_image_points = projection.image_points[compact].to(device).float()
            finite = torch.isfinite(distances) & torch.isfinite(compact_image_points).all(dim=-1)
            target = target[finite]
            compact = compact[finite]
            distances = distances[finite]
            image_points[batch_idx, view_idx, target] = projection.image_points[compact].to(device).float()
            timestamps_us[batch_idx, view_idx, target] = projection.timestamps_us.to(device)[finite]
            metric_distance[batch_idx, view_idx, target] = distances
            valid_mask[batch_idx, view_idx, target] = True

    return ProjectedPointReferences(
        image_points=image_points,
        valid_mask=valid_mask,
        timestamps_us=timestamps_us,
        query_metric_distance=metric_distance,
    )


def bilinear_sample_image_points(
    image: torch.Tensor,
    image_points: torch.Tensor,
    full_resolution: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample ``[B,H,W,C]`` at ``[B,Q,P,2]`` full-image pixels."""
    if image.ndim != 4 or image_points.ndim != 4 or image_points.shape[-1] != 2:
        raise ValueError("expected image [B,H,W,C] and image_points [B,Q,P,2]")
    batch_size = image.shape[0]
    if image_points.shape[0] != batch_size or full_resolution.shape != (batch_size, 2):
        raise ValueError("sampling batch dimensions or full_resolution [B,2] do not match")
    finite = torch.isfinite(image_points).all(dim=-1)
    in_bounds = finite & (image_points >= 0).all(dim=-1) & (image_points < full_resolution[:, None, None]).all(dim=-1)
    safe_points = torch.where(finite.unsqueeze(-1), image_points, torch.zeros_like(image_points))
    grid = safe_points / full_resolution[:, None, None].to(safe_points.dtype) * 2.0 - 1.0
    values = F.grid_sample(
        image.movedim(-1, 1).float(),
        grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).movedim(1, -1)
    return values, in_bounds


def build_context_image_atlas(image: torch.Tensor) -> torch.Tensor:
    """Stack ``[B,V,H,W,C]`` views into one vertically padded sampling atlas.

    A zero row on both sides of every view preserves ``grid_sample`` zero-padding
    semantics at image boundaries, so interpolation can never bleed between
    adjacent cameras or timestamps.
    """
    if image.ndim != 5:
        raise ValueError("context image must be [B,V,H,W,C]")
    batch_size, num_views, height, width, channels = image.shape
    padded = F.pad(image.movedim(-1, 2).float(), (0, 0, 1, 1))
    return padded.permute(2, 0, 1, 3, 4).reshape(
        1,
        channels,
        batch_size * num_views * (height + 2),
        width,
    )


def bilinear_sample_context_atlas(
    atlas: torch.Tensor,
    view_height: int,
    image_points: torch.Tensor,
    batch_indices: torch.Tensor,
    view_indices: torch.Tensor,
    full_resolution: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample packed query/view pairs from a padded context atlas.

    The packed path retains CUDA ``grid_sample`` for efficient backward while
    omitting camera/query pairs rejected before feature attention. Inputs are
    ``image_points[K,P,2]`` and matching integer ``[K]`` batch/view indices.
    """
    if atlas.ndim != 4 or atlas.shape[0] != 1:
        raise ValueError("atlas must be [1,C,B*V*(H+2),W]")
    if image_points.ndim != 3 or image_points.shape[-1] != 2:
        raise ValueError("image_points must be [K,P,2]")
    if batch_indices.ndim != 1 or view_indices.shape != batch_indices.shape:
        raise ValueError("batch_indices and view_indices must be matching [K] tensors")
    if image_points.shape[0] != batch_indices.shape[0]:
        raise ValueError("packed indices and image points must have the same K dimension")
    if batch_indices.dtype != torch.int64 or view_indices.dtype != torch.int64:
        raise TypeError("packed image indices must be int64")
    if full_resolution.ndim != 3 or full_resolution.shape[-1] != 2:
        raise ValueError("full_resolution must be [B,V,2]")
    if view_height <= 0:
        raise ValueError("view_height must be positive")

    batch_size, num_views = full_resolution.shape[:2]
    expected_atlas_height = batch_size * num_views * (view_height + 2)
    if atlas.shape[2] != expected_atlas_height:
        raise ValueError("atlas height does not match B, V, and padded view height")
    if image_points.shape[0] == 0:
        return (
            atlas.new_empty((0, image_points.shape[1], atlas.shape[1])),
            torch.empty((0, image_points.shape[1]), dtype=torch.bool, device=atlas.device),
        )

    indexed_resolution = full_resolution[batch_indices, view_indices].to(image_points.dtype)
    finite = torch.isfinite(image_points).all(dim=-1)
    in_bounds = finite & (image_points >= 0).all(dim=-1) & (image_points < indexed_resolution[:, None]).all(dim=-1)
    safe_points = torch.where(finite.unsqueeze(-1), image_points, torch.zeros_like(image_points))

    combined_view = batch_indices * num_views + view_indices
    atlas_height = atlas.shape[2]
    grid_x = safe_points[..., 0] / indexed_resolution[:, None, 0] * 2.0 - 1.0
    atlas_y_numerator = (
        combined_view[:, None].to(safe_points.dtype) * (view_height + 2)
        + 1.0
        + safe_points[..., 1] / indexed_resolution[:, None, 1] * view_height
    )
    grid_y = atlas_y_numerator / atlas_height * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    values = F.grid_sample(
        atlas,
        grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].permute(1, 2, 0)
    return values, in_bounds


def masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax with an exactly-zero result when every candidate is invalid."""
    safe_logits = torch.where(valid_mask, logits.float(), torch.full_like(logits.float(), -torch.inf))
    has_valid = valid_mask.any(dim=dim, keepdim=True)
    safe_logits = torch.where(has_valid, safe_logits, torch.zeros_like(safe_logits))
    weights = torch.softmax(safe_logits, dim=dim)
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


def apply_bounded_rgb_logit_residual(
    source_rgb: torch.Tensor,
    raw_residual: torch.Tensor,
    *,
    max_logit_delta: float,
) -> torch.Tensor:
    """Apply a bounded logit-space color correction with an exact zero baseline.

    The correction is expressed as a change relative to the numerically
    reconstructed source color. Subtracting that same reconstruction makes a
    zero residual return ``source_rgb`` exactly, including colors at 0 and 1.
    """
    if source_rgb.shape != raw_residual.shape or source_rgb.shape[-1] != 3:
        raise ValueError("source_rgb and raw_residual must be matching [...,3] tensors")
    if max_logit_delta <= 0.0:
        raise ValueError("max_logit_delta must be positive")
    source = source_rgb.float()
    eps = torch.finfo(source.dtype).eps
    source_logits = torch.logit(source.clamp(min=eps, max=1.0 - eps))
    reconstructed_source = torch.sigmoid(source_logits)
    logit_delta = max_logit_delta * torch.tanh(raw_residual.float())
    corrected = source + (torch.sigmoid(source_logits + logit_delta) - reconstructed_source)
    return corrected.clamp(0.0, 1.0).to(source_rgb.dtype)


class PointQueryDeformAttn(nn.Module):
    """Geometric query projection followed by local deformable/view attention."""

    base_offsets: torch.Tensor

    def __init__(
        self,
        *,
        encoder_dim: int,
        fusion_dim: int,
        num_sampling_points: int,
        offset_radius_feature_px: float,
        depth_residual_scale_m: float,
        occlusion_abs_tolerance_m: float,
        occlusion_rel_tolerance: float,
        same_time_tolerance_us: int,
        view_fusion_mode: str = "scalar",
        view_interaction_dim: int = 32,
    ) -> None:
        super().__init__()
        if num_sampling_points < 1:
            raise ValueError("num_sampling_points must be positive")
        if occlusion_abs_tolerance_m < 0.0 or occlusion_rel_tolerance < 0.0:
            raise ValueError("occlusion tolerances must be non-negative")
        if view_fusion_mode not in {"scalar", "depth-aware-interaction"}:
            raise ValueError(f"unsupported view_fusion_mode: {view_fusion_mode}")
        if view_interaction_dim <= 0:
            raise ValueError("view_interaction_dim must be positive")
        self.fusion_dim = fusion_dim
        self.num_sampling_points = num_sampling_points
        self.offset_radius_feature_px = offset_radius_feature_px
        self.depth_residual_scale_m = depth_residual_scale_m
        self.occlusion_abs_tolerance_m = occlusion_abs_tolerance_m
        self.occlusion_rel_tolerance = occlusion_rel_tolerance
        self.same_time_tolerance_us = same_time_tolerance_us
        self.view_fusion_mode = view_fusion_mode

        self.feature_projection = nn.Sequential(
            nn.Linear(encoder_dim, fusion_dim, bias=False),
            nn.LayerNorm(fusion_dim),
        )
        self.query_embed = nn.Sequential(
            nn.Linear(12, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.offset_head = nn.Linear(fusion_dim, max(0, num_sampling_points - 1) * 2)
        self.offset_weight_head = nn.Linear(fusion_dim, num_sampling_points)
        if view_fusion_mode == "depth-aware-interaction":
            self.query_interaction_projection = nn.Linear(fusion_dim, view_interaction_dim, bias=False)
            self.observation_interaction_projection = nn.Linear(fusion_dim, view_interaction_dim, bias=False)
            self.depth_residual_embedding = nn.Sequential(
                nn.Linear(1, view_interaction_dim),
                nn.GELU(),
                nn.Linear(view_interaction_dim, view_interaction_dim),
            )
            self.observation_interaction_norm = nn.LayerNorm(view_interaction_dim)
            self.view_score = nn.Sequential(
                nn.Linear(view_interaction_dim + 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
        else:
            self.view_score = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 1))
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
        self.output_norm = nn.LayerNorm(fusion_dim)
        self.mlp_norm = nn.LayerNorm(fusion_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Linear(fusion_dim * 4, fusion_dim),
        )

        if num_sampling_points > 1:
            angles = torch.arange(num_sampling_points - 1, dtype=torch.float32) * (
                2.0 * math.pi / (num_sampling_points - 1)
            )
            base_offsets = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        else:
            base_offsets = torch.empty((0, 2), dtype=torch.float32)
        self.register_buffer("base_offsets", base_offsets, persistent=False)

    def project_context_features(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Project one VGGT feature tap before point sampling."""
        return self.feature_projection(encoder_features.float())

    def embed_point_queries(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.query_embed(descriptor.float())

    def _sampling_offsets(self, queries: torch.Tensor) -> torch.Tensor:
        batch_size, num_queries, _ = queries.shape
        exact = queries.new_zeros((batch_size, num_queries, 1, 2))
        if self.num_sampling_points == 1:
            return exact
        learned = self.offset_head(queries).reshape(batch_size, num_queries, self.num_sampling_points - 1, 2)
        deformed = self.base_offsets.to(learned) + torch.tanh(learned) * self.offset_radius_feature_px
        return torch.cat([exact, deformed], dim=2)

    def fuse_chunk(
        self,
        queries: torch.Tensor,
        query_timestamps_us: torch.Tensor,
        query_dynamic_mask: torch.Tensor,
        query_valid_mask: torch.Tensor,
        context_feature_atlas: torch.Tensor,
        context_distance_atlas: torch.Tensor,
        reference_image_points: torch.Tensor,
        reference_valid_mask: torch.Tensor,
        reference_timestamps_us: torch.Tensor,
        query_metric_distance: torch.Tensor,
        full_resolution: torch.Tensor,
    ) -> DeformableFusionOutput:
        """Fuse a contiguous query chunk; caller may activation-checkpoint it."""
        batch_size, num_views = full_resolution.shape[:2]
        _, num_queries, query_dim = queries.shape
        feature_dim = context_feature_atlas.shape[1]
        if query_dim != self.fusion_dim or feature_dim != self.fusion_dim:
            raise ValueError("query/context feature widths must equal fusion_dim")
        total_views = batch_size * num_views
        if total_views <= 0:
            raise ValueError("at least one batch item and context view are required")
        feature_height = context_feature_atlas.shape[2] // total_views - 2
        distance_height = context_distance_atlas.shape[2] // total_views - 2
        if feature_height <= 0 or distance_height <= 0:
            raise ValueError("context atlases do not contain positive-height padded views")

        reference_finite = torch.isfinite(reference_image_points).all(dim=-1)
        reference_in_bounds = (
            reference_finite
            & (reference_image_points >= 0).all(dim=-1)
            & (reference_image_points < full_resolution[:, :, None]).all(dim=-1)
        )
        in_frustum = query_valid_mask[:, None] & reference_valid_mask & reference_in_bounds
        frustum_pairs = torch.nonzero(in_frustum, as_tuple=False)
        frustum_batch, frustum_view, frustum_query = frustum_pairs.unbind(dim=1)
        exact_points = reference_image_points[frustum_batch, frustum_view, frustum_query].unsqueeze(1)
        sampled_distance, distance_in_bounds = bilinear_sample_context_atlas(
            context_distance_atlas,
            distance_height,
            exact_points,
            frustum_batch,
            frustum_view,
            full_resolution,
        )
        sampled_distance = sampled_distance.reshape(-1)
        distance_in_bounds = distance_in_bounds.reshape(-1)
        pair_query_distance = query_metric_distance[frustum_batch, frustum_view, frustum_query]
        finite_distance = (
            torch.isfinite(sampled_distance)
            & (sampled_distance > 0.0)
            & torch.isfinite(pair_query_distance)
            & (pair_query_distance > 0.0)
        )
        depth_tested = distance_in_bounds & finite_distance
        visibility_distance = sampled_distance.detach()
        occlusion_margin = self.occlusion_abs_tolerance_m + self.occlusion_rel_tolerance * visibility_distance
        occluded = depth_tested & (pair_query_distance > visibility_distance + occlusion_margin)

        timestamp_delta_us = (
            reference_timestamps_us[frustum_batch, frustum_view, frustum_query]
            - query_timestamps_us[frustum_batch, frustum_query]
        ).abs()
        same_time = timestamp_delta_us <= self.same_time_tolerance_us
        time_eligible = ~query_dynamic_mask[frustum_batch, frustum_query] | same_time
        attention_eligible = depth_tested & (~occluded) & time_eligible
        attention_pairs = frustum_pairs[attention_eligible]
        pair_batch, pair_view, pair_query = attention_pairs.unbind(dim=1)

        in_frustum_observation_count = torch.bincount(frustum_batch, minlength=batch_size)
        depth_tested_observation_count = torch.bincount(frustum_batch[depth_tested], minlength=batch_size)
        occluded_observation_count = torch.bincount(frustum_batch[occluded], minlength=batch_size)

        exact_sampling_points = reference_image_points[pair_batch, pair_view, pair_query].unsqueeze(1)
        if self.num_sampling_points == 1:
            # A one-element deformable neighborhood is exactly the projected
            # pixel.  Avoid constructing offsets and scales whose contribution
            # is identically zero.
            pair_sampling_points = exact_sampling_points
        else:
            offsets = self._sampling_offsets(queries)
            feature_width = context_feature_atlas.shape[3]
            feature_resolution = queries.new_tensor([feature_width, feature_height])
            pair_pixel_scale = full_resolution[pair_batch, pair_view].to(queries.dtype) / feature_resolution
            pair_sampling_points = exact_sampling_points + (
                offsets[pair_batch, pair_query] * pair_pixel_scale.unsqueeze(1)
            )
        sampled_features, sampled_in_bounds = bilinear_sample_context_atlas(
            context_feature_atlas,
            feature_height,
            pair_sampling_points,
            pair_batch,
            pair_view,
            full_resolution,
        )
        finite_features = torch.isfinite(sampled_features).all(dim=-1)
        offset_valid = sampled_in_bounds & finite_features
        safe_features = torch.where(offset_valid.unsqueeze(-1), sampled_features, torch.zeros_like(sampled_features))

        pair_normalized_queries: torch.Tensor | None = None
        if self.num_sampling_points == 1:
            # For one valid candidate, masked softmax is exactly one.  Skipping
            # query/sample normalization, the offset head, and softmax is
            # numerically equivalent and materially reduces E2 runtime.
            observation_features = safe_features[:, 0].float()
            observation_has_value = offset_valid[:, 0]
        else:
            normalized_queries = F.normalize(queries.float(), dim=-1)
            pair_normalized_queries = normalized_queries[pair_batch, pair_query]
            normalized_samples = F.normalize(safe_features.float(), dim=-1)
            content_logits = (normalized_samples * pair_normalized_queries.unsqueeze(1)).sum(dim=-1)
            learned_offset_logits = self.offset_weight_head(queries).float()[pair_batch, pair_query]
            offset_weights = masked_softmax(content_logits + learned_offset_logits, offset_valid, dim=-1)
            observation_features = (offset_weights.unsqueeze(-1) * safe_features.float()).sum(dim=1)
            observation_has_value = offset_valid.any(dim=-1)

        pair_depth_residual = (sampled_distance[attention_eligible] - pair_query_distance[attention_eligible]).abs()
        pair_timestamp_delta_us = timestamp_delta_us[attention_eligible]
        pair_same_time = same_time[attention_eligible]
        pair_view_valid = observation_has_value
        safe_depth_residual = torch.where(
            pair_view_valid,
            pair_depth_residual,
            torch.zeros_like(pair_depth_residual),
        )
        safe_timestamp_delta_us = torch.where(
            pair_view_valid,
            pair_timestamp_delta_us,
            torch.zeros_like(pair_timestamp_delta_us),
        )
        safe_observation = torch.where(
            pair_view_valid.unsqueeze(-1),
            observation_features,
            torch.zeros_like(observation_features),
        )
        normalized_depth = torch.log1p(safe_depth_residual / self.depth_residual_scale_m)
        delta_seconds = safe_timestamp_delta_us.float() / 1_000_000.0
        if self.view_fusion_mode == "depth-aware-interaction":
            query_interaction = self.query_interaction_projection(queries.float())[pair_batch, pair_query]
            observation_interaction = self.observation_interaction_norm(
                self.observation_interaction_projection(safe_observation.float())
                + self.depth_residual_embedding(normalized_depth.unsqueeze(-1))
            )
            score_inputs = torch.cat(
                [
                    query_interaction * observation_interaction,
                    pair_same_time.float().unsqueeze(-1),
                    torch.tanh(delta_seconds).unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            if pair_normalized_queries is None:
                pair_normalized_queries = F.normalize(queries.float(), dim=-1)[pair_batch, pair_query]
            view_similarity = (F.normalize(safe_observation, dim=-1) * pair_normalized_queries).sum(dim=-1)
            score_inputs = torch.stack(
                [view_similarity, normalized_depth, delta_seconds, pair_same_time.float()],
                dim=-1,
            )
        learned_view_score = self.view_score(score_inputs).squeeze(-1)
        # Retain the established metric-depth reliability prior at initialization;
        # the richer scorer learns a residual over it rather than starting from a
        # random view-selection policy.
        pair_view_logits = learned_view_score - normalized_depth
        view_valid = torch.zeros(
            (batch_size, num_views, num_queries),
            dtype=torch.bool,
            device=queries.device,
        ).index_put((pair_batch, pair_view, pair_query), pair_view_valid)
        view_logits = queries.new_zeros((batch_size, num_views, num_queries), dtype=torch.float32).index_put(
            (pair_batch, pair_view, pair_query),
            pair_view_logits,
        )
        view_weights = masked_softmax(view_logits, view_valid, dim=1)
        if self.view_fusion_mode == "depth-aware-interaction":
            attention_entropy = -(
                view_weights * torch.log(view_weights.clamp_min(torch.finfo(view_weights.dtype).tiny))
            ).sum(dim=1)
            attention_entropy = attention_entropy.detach()
        else:
            attention_entropy = queries.new_zeros((batch_size, num_queries), dtype=torch.float32)
        pair_view_weights = view_weights[pair_batch, pair_view, pair_query]
        flat_query_indices = pair_batch * num_queries + pair_query
        fused = (
            safe_observation.new_zeros((batch_size * num_queries, feature_dim))
            .index_add(
                0,
                flat_query_indices,
                pair_view_weights.unsqueeze(-1) * safe_observation,
            )
            .reshape(batch_size, num_queries, feature_dim)
        )

        if self.num_sampling_points > 1:
            deformed_sample_fraction = offset_weights[..., 1:].sum(dim=-1)
            deformed_sample_attention_mass = queries.new_zeros((batch_size,), dtype=torch.float32).index_add(
                0,
                pair_batch,
                pair_view_weights * deformed_sample_fraction,
            )
            learned_offset_delta = offsets[:, :, 1:] - self.base_offsets.to(offsets)
            learned_offset_magnitude_sum = (
                torch.linalg.vector_norm(learned_offset_delta, dim=-1).mean(dim=-1) * query_valid_mask
            ).sum(dim=1)
        else:
            deformed_sample_attention_mass = queries.new_zeros((batch_size,))
            learned_offset_magnitude_sum = queries.new_zeros((batch_size,))

        output = self.output_norm(queries + self.output_projection(fused.to(queries.dtype)))
        output = output + self.mlp(self.mlp_norm(output))
        return DeformableFusionOutput(
            features=output,
            valid_observation_count=view_valid.sum(dim=1),
            view_valid_query_count=view_valid.sum(dim=2),
            view_attention_mass=view_weights.sum(dim=2),
            in_frustum_observation_count=in_frustum_observation_count,
            depth_tested_observation_count=depth_tested_observation_count,
            occluded_observation_count=occluded_observation_count,
            deformed_sample_attention_mass=deformed_sample_attention_mass,
            learned_offset_magnitude_sum=learned_offset_magnitude_sum,
            attention_entropy=attention_entropy,
        )
