# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic camera-frame eligibility for component-only training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from nurec.datasets.samplers.holdout import (
    CoverageFrameSampler,
    CoveragePlan,
    CoveragePlanRestriction,
    CoverageSensorSampler,
)
from nurec.datasets.samplers.image_crop import ImageCropCameraPixelSampler
from nurec.utils.batch import DataBatch, RectSubsampled
from nurec.utils.component_only_mask import build_camera_loss_mask
from nurec.utils.types import RayFlags


if TYPE_CHECKING:
    from nurec.datasets.ncore import NCORETrainDataset


@dataclass(frozen=True, slots=True)
class SupervisedCameraSamplers:
    """Validated sampler bundle used to build and install an eligible-frame index."""

    dataset: NCORETrainDataset
    sensor_sampler: CoverageSensorSampler
    frame_sampler: CoverageFrameSampler
    pixel_sampler: ImageCropCameraPixelSampler
    plan: CoveragePlan


def resolve_supervised_camera_samplers(dataset: NCORETrainDataset) -> SupervisedCameraSamplers:
    """Require the deterministic full-image coverage path supported by ``supervised_only``.

    Random crops, preset ROIs and sparse pixel samplers make frame-level eligibility ambiguous: a
    nominally eligible frame can still produce an empty sampled crop. Fail during setup instead of
    silently admitting zero-mask optimizer steps.
    """
    if len(dataset.batch_samplers) != 1:
        raise ValueError(
            "component_only camera frame_selection='supervised_only' requires exactly one training batch sampler"
        )
    batch_sampler = dataset.batch_samplers[0]
    if batch_sampler.n_train_sample_camera_rays <= 0:
        raise ValueError(
            "component_only camera frame_selection='supervised_only' requires camera rays in every training batch"
        )
    sensor_sampler = batch_sampler.camera_sensor_sampler
    frame_sampler = batch_sampler.camera_frame_sampler
    pixel_sampler = batch_sampler.camera_pixel_sampler
    if not isinstance(sensor_sampler, CoverageSensorSampler) or not isinstance(frame_sampler, CoverageFrameSampler):
        raise ValueError(
            "component_only camera frame_selection='supervised_only' requires coverage camera sensor and frame samplers"
        )
    if not isinstance(pixel_sampler, ImageCropCameraPixelSampler) or pixel_sampler.crop_type != "full_image":
        raise ValueError(
            "component_only camera frame_selection='supervised_only' currently requires the full-image image-crop "
            "camera pixel sampler; random crops, preset ROIs, and sparse pixel sampling are unsupported"
        )
    return SupervisedCameraSamplers(
        dataset=dataset,
        sensor_sampler=sensor_sampler,
        frame_sampler=frame_sampler,
        pixel_sampler=pixel_sampler,
        plan=frame_sampler.coverage_plan,
    )


def component_camera_supervision_mask(model: Any, camera: DataBatch.Camera, rendering_camera: Any) -> torch.Tensor:
    """Build the final depth-aware component mask and intersect pre-existing camera validity."""
    spec = model._component_only_mask_spec
    assert spec is not None, "component-only camera supervision requires an active mask specification"
    track_regions = model.build_component_only_bbox_masks(rendering_camera)
    supervision_mask = build_camera_loss_mask(spec, camera.labels.semantic, track_regions)
    if camera.labels.flags is None:
        return supervision_mask
    return supervision_mask & ((camera.labels.flags.squeeze(-1) & RayFlags.INVALID.value) == 0)


def find_local_eligible_camera_pairs(
    samplers: SupervisedCameraSamplers,
    model: Any,
    *,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> list[tuple[str, int]]:
    """Evaluate this rank's deterministic shard of reachable camera/frame pairs."""
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed eligibility shard rank={rank}, world_size={world_size}")

    dataset = samplers.dataset
    datasource = dataset.datasource
    rng = np.random.default_rng(0)  # full-image sampling is deterministic; required only by the sampler API
    eligible: list[tuple[str, int]] = []
    with torch.no_grad():
        for pair_index, (sensor_id, frame_idx) in enumerate(samplers.plan.iter_pairs()):
            if pair_index % world_size != rank:
                continue
            frame_range = datasource.camera_frame_ranges[sensor_id]
            unique_sensor_id = datasource.camera_unique_ids[sensor_id].id
            valid_pixels = datasource.cameras_frame_valid_pixels_masks[sensor_id][frame_idx].unpacked()
            sampled = samplers.pixel_sampler.sample_camera_pixels(
                rng=rng,
                batch_idx=pair_index,
                frame_range=frame_range,
                n_frame_pixel_samples=samplers.dataset.n_train_sample_camera_rays,
                frame_all_pixels=datasource.cameras_all_pixels[sensor_id],
                frame_valid_pixels_mask=valid_pixels,
                unique_camera_id=unique_sensor_id,
                camera_frame_idx=frame_idx,
            ).sampled_pixels
            if not isinstance(sampled, RectSubsampled):
                raise TypeError("supervised_only full-image sampling must produce RectSubsampled camera pixels")
            camera = dataset.get_camera_data_batch(
                sensor_id,
                frame_idx,
                sampled,
                load_rgb=False,
                load_aux_geometry=False,
            ).to(device)
            data = DataBatch(
                idx=pair_index,
                sequence_id=[datasource.sequence_id],
                camera=camera,
            )
            rendering = model.calib(data, skip_calib=False)
            if rendering.camera is None:
                raise RuntimeError("component-only supervised frame indexing requires calibrated camera rays")
            if component_camera_supervision_mask(model, camera, rendering.camera).any().item():
                eligible.append((sensor_id, frame_idx))
    return eligible


def gather_eligible_camera_pairs(
    local_pairs: list[tuple[str, int]], *, local_error: Exception | None = None
) -> set[tuple[str, int]]:
    """Union rank-local results, or raise a synchronized error on every DDP worker."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        if local_error is not None:
            raise local_error
        return set(local_pairs)
    local_failure = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    local_result = (local_pairs, local_failure)
    gathered: list[tuple[list[tuple[str, int]], str | None] | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_result)
    failures = [f"rank {rank}: {result[1]}" for rank, result in enumerate(gathered) if result and result[1]]
    if failures:
        raise RuntimeError("component_only supervised frame indexing failed on " + "; ".join(failures))
    return {pair for result in gathered if result is not None for pair in result[0]}


def install_eligible_camera_plan(
    samplers: SupervisedCameraSamplers,
    eligible_pairs: set[tuple[str, int]],
) -> CoveragePlanRestriction:
    """Restrict and re-inject the shared plan into both halves of the camera sampler."""
    restriction = samplers.plan.restrict_to_pairs(eligible_pairs)
    samplers.sensor_sampler.set_coverage_plan(restriction.plan)
    samplers.frame_sampler.set_coverage_plan(restriction.plan)
    return restriction


__all__ = [
    "SupervisedCameraSamplers",
    "component_camera_supervision_mask",
    "find_local_eligible_camera_pairs",
    "gather_eligible_camera_pairs",
    "install_eligible_camera_plan",
    "resolve_supervised_camera_samplers",
]
