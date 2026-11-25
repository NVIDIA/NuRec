# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mode-independent input contracts and preparation for fused loss modules."""

import logging

from collections.abc import Iterable, Mapping
from typing import Any, Literal, NamedTuple

import torch

from nurec.datasets.tracks import CuboidTracks
from nurec.instant_nurec.models.base import BaseInstantNuRec
from nurec.models.background import SkyEnvMapBackground
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.models.post_processing import BilateralGridPerCamera, BilateralGridPerFrame, BilateralGridT
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.misc import get_pack_info_from_n, unpack_optional


log = logging.getLogger(__name__)

# Empty IDs selecting matches penalize no points; selecting nonmatches penalizes all.
NODE_SEMANTIC_SELECT_NONE: tuple[tuple[int, ...], bool] = ((), True)

FUSED_LOSS_NAMES = (
    "rgb_l1_mean",
    "ssim_ssim_mean",
    "lidar_l1_mean",
    "background_mse_mean",
    "background_lidar_mse_mean",
    "bilateral_grid_drift_identity_distance_mean",
    "bilateral_grid_per_camera_tv_total_variation_spatial_mean",
    "bilateral_grid_per_frame_spatial_tv_total_variation_spatial_mean",
    "gaussian_scale_abs_mean",
    "sky_env_map_background_total_variation_spatial_mean",
    "gaussian_density_abs_mean",
    "out_of_bound_l1_mean",
    "gaussian_z_scale_abs_mean",
    "intensity_mse_mean",
    "raydrop_mse_mean",
    "background_in_track_gaussian_bce_with_logits_mean",
    "node_semantic_gaussians_bce_with_logits_mean",
    "semantic_cross_entropy_mean",
    "deform_smoothness_abs_mean",
)


def _get_bilateral_grids(
    model: BaseModel | BaseInstantNuRec, bilateral_grid_class: type[BilateralGridT]
) -> list[BilateralGridT]:
    grids: list[BilateralGridT] = []
    match model:
        case GaussiansComposite():
            for post_processing in model.post_processings:
                if isinstance(post_processing, bilateral_grid_class):
                    grids.append(post_processing)
        case BaseInstantNuRec():
            pass
        case _:
            raise RuntimeError(f"{_get_bilateral_grids.__qualname__} got unsupported model type {type(model)}")
    return grids


def loss_inputs_available(
    loss_name: str,
    model: BaseModel,
    camera_grids: list[BilateralGridPerCamera],
    frame_grids: list[BilateralGridPerFrame],
    target: DataAndRenderingBatch,
    out_of_bound_gaussian_nodes: list[RigidGaussianModel],
    loss: Any,
) -> bool:
    if loss_name == "rgb_l1_mean":
        if target.data.camera is not None:
            assert target.data.camera.labels.flags is not None and target.data.camera.labels.rgb is not None, (
                "Slang losses requires camera labels flags and rgb!"
            )
            return True
    elif loss_name == "ssim_ssim_mean":
        if target.data.camera is not None:
            assert target.data.camera.labels.flags is not None and target.data.camera.labels.rgb is not None, (
                "SSIM loss requires camera labels flags and rgb."
            )
            return True
    elif loss_name == "lidar_l1_mean":
        if target.data.lidar is not None:
            assert target.data.lidar.labels.flags is not None and target.data.lidar.labels.distance is not None, (
                "Slang losses requires lidar labels flags and distance!"
            )
            return True
    elif loss_name == "background_mse_mean":
        if target.data.camera is not None:
            assert target.data.camera.labels.flags is not None, "Slang losses requires camera labels flags!"
            return True
    elif loss_name == "background_lidar_mse_mean":
        if target.data.lidar is not None:
            assert target.data.lidar.labels.flags is not None, "Slang losses requires lidar labels flags!"
            return True
    elif loss_name == "bilateral_grid_drift_identity_distance_mean":
        if len(camera_grids) > 0 or len(frame_grids) > 0:
            return True
    elif loss_name == "bilateral_grid_per_camera_tv_total_variation_spatial_mean":
        if len(camera_grids) > 0:
            return True
    elif loss_name == "bilateral_grid_per_frame_spatial_tv_total_variation_spatial_mean":
        if len(frame_grids) > 0:
            return True
    elif loss_name == "gaussian_scale_abs_mean":
        return True
    elif loss_name == "sky_env_map_background_total_variation_spatial_mean":
        # The sky model may be disabled via a component-only overlay
        # (a SkipBackground stand-in replaces it) — there are no
        # env-map textures to regularize then: no-op this loss.
        return isinstance(getattr(model, "background", None), SkyEnvMapBackground)
    elif loss_name == "gaussian_density_abs_mean":
        return True
    elif loss_name == "out_of_bound_l1_mean":
        assert isinstance(model, GaussiansComposite), "Slang losses requires GaussiansComposite model for out_of_bound!"

        return len(out_of_bound_gaussian_nodes) > 0
    elif loss_name == "gaussian_z_scale_abs_mean":
        return True
    elif loss_name == "intensity_mse_mean":
        if target.data.lidar is not None:
            return True
    elif loss_name == "raydrop_mse_mean":
        if target.data.lidar is not None:
            return True
    elif loss_name == "background_in_track_gaussian_bce_with_logits_mean":
        assert isinstance(model, GaussiansComposite), (
            "background_in_track_gaussian loss requires model to be GaussiansComposite"
        )
        if target.rendering is None:
            return False
        if target.rendering.camera is None:
            return False
        has_tracks = any(
            isinstance(model.gaussians_nodes[node_id], RigidGaussianModel) for node_id in model.get_gaussians_node_ids()
        )
        has_background_points = any(
            node_id in loss.layer_names and not isinstance(model.gaussians_nodes[node_id], RigidGaussianModel)
            for node_id in model.get_gaussians_node_ids()
        )
        return has_tracks and has_background_points
    elif loss_name == "node_semantic_gaussians_bce_with_logits_mean":
        assert isinstance(model, GaussiansComposite), (
            "node_semantic_gaussians loss requires model to be GaussiansComposite"
        )
        return True
    elif loss_name == "semantic_cross_entropy_mean":
        if target.data.camera is None:
            return False
        labels = target.data.camera.labels
        if labels.flags is None or labels.semantic is None:
            raise ValueError("semantic loss requires camera label flags and semantic labels")
        return True
    elif loss_name == "road_gaussians_abs_mean":
        if target.rendering is None or target.rendering.camera is None:
            return False
        assert target.rendering.camera.b == 1, "road loss requires a single camera"
        assert isinstance(model, GaussiansComposite), "road loss requires GaussiansComposite"
        return loss.layer_name in model.gaussians_nodes
    elif loss_name == "deform_smoothness_abs_mean":
        return True
    return False


def node_semantic_metadata(loss: Any) -> dict[str, tuple[tuple[int, ...], bool]]:
    node_metadata: dict[str, tuple[tuple[int, ...], bool]] = {}
    configured_node_ids = loss.layer_labels_to_use.keys() | loss.layer_labels_to_exclude.keys()
    for node_id in configured_node_ids:
        if node_id in loss.layer_labels_to_use:
            labels = loss.layer_labels_to_use[node_id]
            select_matches = False
        else:
            labels = loss.layer_labels_to_exclude[node_id]
            select_matches = True
        class_ids = tuple(
            sorted({loss.semantic_classes_map[label] for label in labels if loss.semantic_classes_map[label] >= 0})
        )
        node_metadata[node_id] = (class_ids, select_matches)

    return node_metadata


class BackgroundLossTracks(NamedTuple):
    """Only the packed track tensors consumed by background losses."""

    tracks_packinfo: torch.Tensor
    tracks_poses: torch.Tensor
    tracks_timestamps_us: torch.Tensor
    cuboids_dims: torch.Tensor


def _prepare_loss_tracks(tracks: list[CuboidTracks]) -> BackgroundLossTracks:
    if len(tracks) == 1:
        track = tracks[0]
        return BackgroundLossTracks(
            track.tracks_packinfo, track.tracks_poses.data, track.tracks_timestamps_us, track.cuboids_dims
        )
    lengths = torch.cat([track.tracks_packinfo[:, 1] for track in tracks], dim=0)
    return BackgroundLossTracks(
        get_pack_info_from_n(lengths).int(),
        torch.cat([track.tracks_poses.data for track in tracks], dim=0),
        torch.cat([track.tracks_timestamps_us for track in tracks], dim=0),
        torch.cat([track.cuboids_dims for track in tracks], dim=0),
    )


class BackgroundTrackInputs(NamedTuple):
    filtered_entries: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]
    unfiltered_entries: list[tuple[torch.Tensor, torch.Tensor]]
    positions: torch.Tensor
    density_logits: torch.Tensor
    semantic_logits: torch.Tensor | None
    n_semantic_points: int
    semantic_segments: tuple[tuple[int, tuple[int, ...], torch.Tensor], ...]
    tracks: BackgroundLossTracks
    camera_timestamps_startend_us: torch.Tensor


def prepare_background_tracks(
    model: GaussiansComposite,
    target: DataAndRenderingBatch,
    loss: Any,
) -> BackgroundTrackInputs | None:
    filtered_entries: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    unfiltered_entries: list[tuple[torch.Tensor, torch.Tensor]] = []
    tracks_list: list[CuboidTracks] = []
    layer_labels_to_use = loss.layer_labels_to_use
    for node_id in model.get_gaussians_node_ids():
        bg_node = model.gaussians_nodes[node_id]
        if isinstance(bg_node, RigidGaussianModel):
            if bg_node.owns_tracks:
                tracks_list.append(bg_node.cuboid_tracks)
            continue
        if node_id not in loss.layer_names:
            continue

        bg_positions = bg_node.get_positions().float()
        density_logits = bg_node.get_densities(preactivation=True).flatten().float()
        assert bg_positions.ndim == 2 and bg_positions.shape[1] == 3, "background track positions must have shape [N,3]"
        assert density_logits.numel() == bg_positions.shape[0], (
            "background density logits must contain one value per position"
        )
        node_semantic_logits = None
        if node_id in layer_labels_to_use:
            try:
                node_semantic_logits = bg_node.get_extra_signal_by_key("semantic_logits")
            except ValueError:
                node_semantic_logits = None
            if node_semantic_logits is None:
                log.warning(
                    "Node '%s' is in layer_labels_to_use but missing 'semantic_logits', "
                    "collecting all points without semantic filtering",
                    node_id,
                )

        if node_semantic_logits is not None:
            assert node_semantic_logits.ndim == 2 and node_semantic_logits.shape[0] == bg_positions.shape[0], (
                f"semantic_logits for node '{node_id}' must have shape [N,C] matching positions"
            )
            filtered_entries.append((node_id, bg_positions, density_logits, node_semantic_logits))
        else:
            unfiltered_entries.append((bg_positions, density_logits))

    if tracks_list and (filtered_entries or unfiltered_entries):
        bg_track_n_semantic_points = 0
        bg_track_semantic_logits = None
        bg_track_semantic_segments: tuple[tuple[int, tuple[int, ...], torch.Tensor], ...] = ()
        ordered_positions = [entry[1] for entry in filtered_entries] + [entry[0] for entry in unfiltered_entries]
        ordered_density_logits = [entry[2] for entry in filtered_entries] + [entry[1] for entry in unfiltered_entries]
        bg_track_positions = (
            ordered_positions[0] if len(ordered_positions) == 1 else torch.cat(ordered_positions, dim=0)
        )
        bg_track_density_logits = (
            ordered_density_logits[0] if len(ordered_density_logits) == 1 else torch.cat(ordered_density_logits, dim=0)
        )

        if filtered_entries:
            bg_track_semantic_logits = filtered_entries[0][3]
            segment_end = 0
            semantic_segments: list[tuple[int, tuple[int, ...], torch.Tensor]] = []
            semantic_classes_map = loss.semantic_classes_map
            for node_id, _, _, node_semantic_logits in filtered_entries:
                node_point_count = int(node_semantic_logits.shape[0])
                segment_end += node_point_count
                # Empty layers have no points to classify, so omitting their
                # zero-width segment preserves behavior and keeps endpoints ordered.
                if node_point_count == 0:
                    continue
                allowed_ids = tuple(
                    sorted(
                        {
                            semantic_classes_map[label]
                            for label in layer_labels_to_use[node_id]
                            if semantic_classes_map[label] >= 0
                        }
                    )
                )
                semantic_segments.append((segment_end, allowed_ids, node_semantic_logits))
            bg_track_n_semantic_points = segment_end
            bg_track_semantic_segments = tuple(semantic_segments)

        tracks = _prepare_loss_tracks(tracks_list)
        timestamps = unpack_optional(unpack_optional(target.rendering).camera).timestamps_startend_us
        return BackgroundTrackInputs(
            filtered_entries,
            unfiltered_entries,
            bg_track_positions,
            bg_track_density_logits,
            bg_track_semantic_logits,
            bg_track_n_semantic_points,
            bg_track_semantic_segments,
            tracks,
            timestamps,
        )
    return None


class GaussianSegmentInputs(NamedTuple):
    empty_input: torch.Tensor | None
    segments: tuple[torch.Tensor, ...]
    weights: tuple[float, ...]
    n_elements: int


def prepare_gaussian_segments(
    model: GaussiansComposite,
    node_ids: Iterable[str],
    layer_weights: Mapping[str, float],
    *,
    member: Literal["scales", "densities"],
    densities_are_logits: bool = False,
) -> GaussianSegmentInputs:
    """Retain node storage, layer weights, and the first enabled empty input.

    Zero-weight nonempty nodes remain present so visibility offsets keep the
    model's ordering. Scale inputs retain their float32 [N,3] validation; density
    inputs retain their existing flattening and selected activation contract.
    """
    segments: list[torch.Tensor] = []
    weights: list[float] = []
    empty = None
    for node_id in node_ids:
        node = model.gaussians_nodes[node_id]
        if member == "scales":
            values = node.get_scales(preactivation=True)
            assert values.ndim == 2 and values.shape[1] == 3 and values.dtype == torch.float32, (
                "Gaussian scales must be float32 [N,3]"
            )
        else:
            values = node.get_densities(preactivation=densities_are_logits).flatten()
        if values.shape[0] == 0:
            if empty is None:
                empty = values
            continue
        segments.append(values)
        weights.append(float(layer_weights.get(node_id, 1.0)))
    count = sum(segment.shape[0] for segment in segments)
    if count:
        return GaussianSegmentInputs(None, tuple(segments), tuple(weights), count)
    if empty is None:
        empty = torch.empty((0, 3) if member == "scales" else (0,), dtype=torch.float32, device=model.device)
    return GaussianSegmentInputs(empty, (), (), 0)


def gaussian_visibility_flags(
    scale_loss: Any, density_loss: Any, run_scale: bool, run_density: bool
) -> tuple[bool, bool]:
    scale = run_scale and bool(getattr(scale_loss, "visibility_filter", False))
    density = run_density and bool(getattr(density_loss, "visibility_filter", False))
    if scale != density and run_scale and run_density:
        raise ValueError(
            "gaussian_scale and gaussian_density must have matching visibility_filter when both are active. "
            f"Got scale={scale}, density={density}. "
            "The Slang kernel applies a single shared visibility mask to both."
        )
    return scale, density


def prepare_node_semantics(
    model: GaussiansComposite, loss: Any
) -> list[tuple[str, torch.Tensor, torch.Tensor, tuple[int, ...], bool]]:
    node_metadata = node_semantic_metadata(loss)
    node_entries: list[tuple[str, torch.Tensor, torch.Tensor, tuple[int, ...], bool]] = []
    for node_id in model.get_gaussians_node_ids():
        if node_id not in node_metadata:
            continue
        semantic_node = model.gaussians_nodes[node_id]
        try:
            semantic_input = semantic_node.get_extra_signal_by_key("semantic_logits")
        except ValueError:
            semantic_input = None
        if semantic_input is None:
            log.warning(
                "Node '%s' is configured for node-semantic but missing 'semantic_logits', skipping it",
                node_id,
            )
            continue
        density_input = semantic_node.get_densities(preactivation=True).flatten().float()
        semantic_input = semantic_input.float()
        class_ids, select_matches = node_metadata[node_id]
        node_entries.append((node_id, density_input, semantic_input, class_ids, select_matches))
    return node_entries


class JointNodeInputs(NamedTuple):
    primary_node_id: str | None
    primary_density: torch.Tensor | None
    primary_semantic: torch.Tensor | None
    background_allowed_class_ids: tuple[int, ...]
    primary_predicate: tuple[tuple[int, ...], bool] | None
    other_density: torch.Tensor
    other_segments: tuple[tuple[int, tuple[int, ...], bool], ...]
    other_semantic: tuple[torch.Tensor, ...]


def prepare_joint_node_semantics(
    node_entries: list[tuple[str, torch.Tensor, torch.Tensor, tuple[int, ...], bool]],
    background: BackgroundTrackInputs | None,
    background_loss: Any,
) -> JointNodeInputs:
    """Select shared semantic domains; callers retain their own position contract."""
    primary_node_id = None
    primary_density = primary_semantic = None
    allowed_ids: tuple[int, ...] = ()
    primary_predicate: tuple[tuple[int, ...], bool] | None = None
    other_entries = node_entries
    if background is not None:
        if len(background.filtered_entries) == 1 and not background.unfiltered_entries:
            primary_node_id, _, primary_density, primary_semantic = background.filtered_entries[0]
            primary_entry = next((entry for entry in node_entries if entry[0] == primary_node_id), None)
            if primary_entry is not None:
                primary_predicate = (primary_entry[3], primary_entry[4])
                other_entries = [entry for entry in node_entries if entry[0] != primary_node_id]
            else:
                primary_predicate = NODE_SEMANTIC_SELECT_NONE
            classes = background_loss.semantic_classes_map
            allowed_ids = tuple(
                sorted(
                    {
                        classes[label]
                        for label in background_loss.layer_labels_to_use[primary_node_id]
                        if classes[label] >= 0
                    }
                )
            )
    else:
        non_road_entries = [entry for entry in node_entries if entry[0] != "road"]
        primary_entry = (
            non_road_entries[0] if len(node_entries) == 2 and len(non_road_entries) == 1 else node_entries[0]
        )
        primary_node_id, primary_density, primary_semantic, class_ids, select_matches = primary_entry
        primary_predicate = (class_ids, select_matches)
        other_entries = [entry for entry in node_entries if entry is not primary_entry]
    tensors = [entry[1] for entry in other_entries]
    other_density = (tensors[0] if len(tensors) == 1 else torch.cat(tensors)) if tensors else node_entries[0][1][:0]
    segment_end = 0
    segments: list[tuple[int, tuple[int, ...], bool]] = []
    for _, density, _, class_ids, select_matches in other_entries:
        segment_end += density.numel()
        segments.append((segment_end, class_ids, select_matches))
    return JointNodeInputs(
        primary_node_id,
        primary_density,
        primary_semantic,
        allowed_ids,
        primary_predicate,
        other_density,
        tuple(segments),
        tuple(entry[2] for entry in other_entries),
    )
