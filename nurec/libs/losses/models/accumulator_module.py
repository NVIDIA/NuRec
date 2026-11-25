# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 2: Losses model layer - Neural network modules for fused CUDA losses."""

from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

from nurec.libs.losses.functional.cuda_losses_accumulator import (
    BG_GRID_LOSS_NAMES,
    BG_TRACK_NODE_LOSS_NAMES,
    CAMERA_LOSS_NAMES,
    GAUSSIAN_LOSS_NAMES,
    LIDAR_LOSS_NAMES,
    N_LOSS_SLOTS,
    GsplatLossAdapter,
)
from nurec.libs.losses.models.fused_loss_inputs import (
    FUSED_LOSS_NAMES,
    _get_bilateral_grids,
    gaussian_visibility_flags,
    loss_inputs_available,
    prepare_background_tracks,
    prepare_gaussian_segments,
    prepare_joint_node_semantics,
    prepare_node_semantics,
)
from nurec.libs.losses.models.utils import (
    _get_out_of_bound_gaussian_nodes,
    _maybe_update_mcmc_visibility_counters,
    get_rendered_visibility_mask,
)
from nurec.models.background import SkyEnvMapBackground
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.models.post_processing import (
    BilateralGridPerCamera,
    BilateralGridPerFrame,
)
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.misc import unpack_optional
from nurec.utils.types import GaussiansCompositeReturn


if TYPE_CHECKING:
    from nurec.libs.losses.models.render_losses import (
        GaussianDensityLoss,
        GaussianScaleLoss,
        SSIMLoss,
    )


class AccumulatorModuleLosses(torch.nn.Module):
    """
    Stateful class for all fused CUDA losses model layer

    Wraps fused losses for NRE training; see nurec/libs/losses/README.md.
    """

    # Class attribute so CPU-only tests can read the CUDA-eligible loss names
    # without constructing the owned dispatch modules.
    # CUDA accumulators compute each member mean and apply its objective weight.
    available: ClassVar[list[str]] = list(FUSED_LOSS_NAMES) + ["road_gaussians_abs_mean"]
    losses: list[Any]
    accumulator: GsplatLossAdapter

    def __init__(self):
        super().__init__()
        self.losses = []
        self.accumulator = GsplatLossAdapter()

    def is_enabled(
        self,
        loss_name: str,
        model: BaseModel,
        camera_grids: list[BilateralGridPerCamera],
        frame_grids: list[BilateralGridPerFrame],
        target: DataAndRenderingBatch,
        out_of_bound_gaussian_nodes: list[RigidGaussianModel],
        loss: Any,
    ) -> bool:
        return loss_inputs_available(
            loss_name, model, camera_grids, frame_grids, target, out_of_bound_gaussian_nodes, loss
        )

    def accumulate(
        self,
        step: int,
        model: BaseModel,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return weighted CUDA slots and reporting weights without scalar view objects."""
        assert results is not None and target is not None, "Fused losses require results and target!"
        camera_grids: list[BilateralGridPerCamera] | None = None
        frame_grids: list[BilateralGridPerFrame] | None = None
        out_of_bound_gaussian_nodes: list[RigidGaussianModel] | None = None

        # Snapshot each scheduled config and its current signed weight once.
        # Available zero-weight entries remain visible to logging, but their groups do not
        # prepare inputs or call a producer for them.
        selected_losses: dict[str, tuple[Any, float]] = {}
        for loss in self.losses:
            if not loss.should_run_fn(step):
                continue
            if (
                loss.name
                in (
                    "bilateral_grid_drift_identity_distance_mean",
                    "bilateral_grid_per_camera_tv_total_variation_spatial_mean",
                    "bilateral_grid_per_frame_spatial_tv_total_variation_spatial_mean",
                )
                and camera_grids is None
            ):
                camera_grids = _get_bilateral_grids(model, BilateralGridPerCamera)
                frame_grids = _get_bilateral_grids(model, BilateralGridPerFrame)
            if loss.name == "out_of_bound_l1_mean" and out_of_bound_gaussian_nodes is None:
                out_of_bound_gaussian_nodes = _get_out_of_bound_gaussian_nodes(model)
            if self.is_enabled(
                loss.name,
                model,
                camera_grids or [],
                frame_grids or [],
                target,
                out_of_bound_gaussian_nodes or [],
                loss,
            ):
                selected_losses[loss.name] = (loss, float(loss.lambda_))

        losses_vec = torch.zeros((N_LOSS_SLOTS,), dtype=torch.float32, device=model.device)
        losses_vec = self._accumulate_camera(losses_vec, results, target, selected_losses)
        losses_vec = self._accumulate_ssim(losses_vec, results, target, selected_losses)
        losses_vec = self._accumulate_lidar(losses_vec, results, target, selected_losses)
        losses_vec = self._accumulate_gaussian(
            losses_vec,
            model,
            results,
            out_of_bound_gaussian_nodes or [],
            selected_losses,
        )
        losses_vec = self._accumulate_bg_grid(losses_vec, model, camera_grids or [], frame_grids or [], selected_losses)
        losses_vec = self._accumulate_bg_track_node(losses_vec, model, target, selected_losses)
        losses_vec = self._accumulate_ground(losses_vec, model, target, selected_losses)
        return losses_vec, {name: weight for name, (_, weight) in selected_losses.items()}

    def _accumulate_camera(
        self,
        losses_vec: torch.Tensor,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        selected_losses: dict[str, tuple[Any, float]],
    ) -> torch.Tensor:
        names = CAMERA_LOSS_NAMES
        weights = tuple(selected_losses.get(name, (None, 0.0))[1] for name in names)
        if not any(weight != 0.0 for weight in weights):
            return losses_vec
        if results.rendered_cam is None:
            raise ValueError("Fused camera losses require results.rendered_cam")
        camera = unpack_optional(target.data.camera)
        rendered = results.rendered_cam
        flags = unpack_optional(camera.labels.flags)
        rgb_pred = rgb_gt = bg_pred = semantic_logits = semantic_targets = None
        if weights[0] != 0.0:
            rgb_gt = camera.labels.rgb
            rgb_pred = rendered.rgb
        if weights[1] != 0.0:
            bg_pred = rendered.opacity
        if weights[2] != 0.0:
            semantic_targets = camera.labels.semantic
            semantic_logits = unpack_optional(rendered.extra_ray_signals).semantic_logits
        return self.accumulator.accumulate_camera(
            losses_vec,
            weights,
            flags=flags,
            rgb_pred=rgb_pred,
            rgb_gt=rgb_gt,
            bg_pred=bg_pred,
            semantic_logits=semantic_logits,
            semantic_targets=semantic_targets,
        )

    def _accumulate_ssim(
        self,
        losses_vec: torch.Tensor,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        selected_losses: dict[str, tuple[Any, float]],
    ) -> torch.Tensor:
        loss, weight = selected_losses.get("ssim_ssim_mean", (None, 0.0))
        if weight == 0.0:
            return losses_vec
        if results.rendered_cam is None:
            raise ValueError("Fused camera losses require results.rendered_cam")
        camera = unpack_optional(target.data.camera)
        flags = unpack_optional(camera.labels.flags)
        rgb_gt = unpack_optional(camera.labels.rgb)
        rgb_pred = unpack_optional(results.rendered_cam.rgb)
        ssim_loss: SSIMLoss = loss
        return self.accumulator.accumulate_ssim(
            losses_vec,
            weight,
            flags=flags,
            rgb_pred=rgb_pred,
            rgb_gt=rgb_gt,
            channel=ssim_loss.channel,
            window_size=ssim_loss.window_size,
            mask_mode="constant" if ssim_loss.mask_mode.lower() == "constant" else "gt",
            constant_mask_value=float(ssim_loss.mask_value),
        )

    def _accumulate_lidar(
        self,
        losses_vec: torch.Tensor,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        selected_losses: dict[str, tuple[Any, float]],
    ) -> torch.Tensor:
        names = LIDAR_LOSS_NAMES
        weights = tuple(selected_losses.get(name, (None, 0.0))[1] for name in names)
        if not any(weight != 0.0 for weight in weights):
            return losses_vec
        lidar = unpack_optional(target.data.lidar)
        rendered = unpack_optional(results.rendered_lidar)
        flags = unpack_optional(lidar.labels.flags)
        distance_pred = distance_gt = intensity_pred = intensity_gt = raydrop_pred = raydrop_gt = bg_pred = None
        if weights[0] != 0.0:
            distance_pred, distance_gt = rendered.distance, lidar.labels.distance
        if weights[1] != 0.0:
            intensity_pred = unpack_optional(rendered.extra_ray_signals).intensity
            intensity_gt = lidar.labels.intensity
        if weights[2] != 0.0:
            raydrop_pred = unpack_optional(rendered.extra_ray_signals).raydrop
            raydrop_gt = lidar.labels.raydrop
        if weights[3] != 0.0:
            bg_pred = rendered.opacity
        return self.accumulator.accumulate_lidar(
            losses_vec,
            weights,
            flags=flags,
            distance_pred=distance_pred,
            distance_gt=distance_gt,
            intensity_pred=intensity_pred,
            intensity_gt=intensity_gt,
            raydrop_pred=raydrop_pred,
            raydrop_gt=raydrop_gt,
            bg_pred=bg_pred,
        )

    def _accumulate_gaussian(
        self,
        losses_vec: torch.Tensor,
        model: BaseModel,
        results: GaussiansCompositeReturn,
        out_of_bound_gaussian_nodes: list[RigidGaussianModel],
        selected_losses: dict[str, tuple[Any, float]],
    ) -> torch.Tensor:
        names = GAUSSIAN_LOSS_NAMES
        # Availability controls logging independently of a member's weight.
        # Check metadata before the zero-weight shortcut without loading scales
        # or converting/broadcasting deformation tensors.
        z_scale_config = selected_losses.get("gaussian_z_scale_abs_mean", (None, 0.0))[0]
        if z_scale_config is not None:
            assert isinstance(model, GaussiansComposite), (
                "gaussian_z_scale loss requires model to be GaussiansComposite"
            )
            if z_scale_config.layer_name not in model.gaussians_nodes:
                selected_losses.pop("gaussian_z_scale_abs_mean")
        if "deform_smoothness_abs_mean" in selected_losses and results.deform_smoothness is None:
            selected_losses.pop("deform_smoothness_abs_mean")
        weights = [selected_losses.get(name, (None, 0.0))[1] for name in names]
        if not any(weight != 0.0 for weight in weights):
            return losses_vec
        scale_config = selected_losses[names[0]][0] if weights[0] != 0.0 else None
        density_config = selected_losses[names[1]][0] if weights[1] != 0.0 else None
        run_gaussian_scale = weights[0] != 0.0
        run_gaussian_density = weights[1] != 0.0
        run_gaussian_z_scale = selected_losses[names[2]][0] if weights[2] != 0.0 else None
        run_out_of_bound = weights[3] != 0.0
        run_deform = selected_losses[names[4]][0] if weights[4] != 0.0 else None

        # Load Gaussian scale inputs
        gaussian_scales: torch.Tensor | None = None
        gaussian_scale_segments: tuple[torch.Tensor, ...] = ()
        gaussian_scale_segment_weights: tuple[float, ...] = ()
        gaussian_scale_count = 0

        if run_gaussian_scale:
            # Type narrow: gaussian_scale loss requires GaussiansComposite
            assert isinstance(model, GaussiansComposite), "gaussian_scale loss requires model to be GaussiansComposite"

            # Find the gaussian_scale loss config to get layer_lambdas
            gaussian_scale_loss: GaussianScaleLoss | None = scale_config

            if gaussian_scale_loss is not None:
                gaussian_scales, gaussian_scale_segments, gaussian_scale_segment_weights, gaussian_scale_count = (
                    prepare_gaussian_segments(
                        model,
                        model.get_gaussians_node_ids(),
                        gaussian_scale_loss.layer_lambdas or {},
                        member="scales",
                    )
                )

        # Load Gaussian density inputs
        gaussian_densities: torch.Tensor | None = None
        gaussian_density_segments: tuple[torch.Tensor, ...] = ()
        gaussian_density_segment_weights: tuple[float, ...] = ()
        gaussian_density_count = 0
        gaussian_densities_are_logits = False

        if run_gaussian_density:
            # Type narrow: gaussian_density loss requires GaussiansComposite
            assert isinstance(model, GaussiansComposite), (
                "gaussian_density loss requires model to be GaussiansComposite"
            )
            # Find the gaussian_density loss config to get layer_lambdas
            gaussian_density_loss: GaussianDensityLoss | None = density_config

            if gaussian_density_loss is not None:
                density_node_ids = model.get_gaussians_node_ids()
                # Fuse only the sigmoid contract supported by the accumulator.
                # Other activations and segment-capacity overflow retain the
                # activated-input path, including its per-layer weighting.
                gaussian_densities_are_logits = self.accumulator.can_fuse_density_activation(
                    len(density_node_ids)
                ) and all(
                    getattr(
                        getattr(model.gaussians_nodes[node_id], "config", None),
                        "density_activation",
                        None,
                    )
                    == "sigmoid"
                    for node_id in density_node_ids
                )

                (
                    gaussian_densities,
                    gaussian_density_segments,
                    gaussian_density_segment_weights,
                    gaussian_density_count,
                ) = prepare_gaussian_segments(
                    model,
                    density_node_ids,
                    gaussian_density_loss.layer_lambdas or {},
                    member="densities",
                    densities_are_logits=gaussian_densities_are_logits,
                )

        # Visibility mask for gaussian_scale / gaussian_density (same logic as PyTorch path)
        visibility_gaussian: torch.Tensor | None = None
        if run_gaussian_scale or run_gaussian_density:
            scale_loss_cfg: GaussianScaleLoss | None = scale_config
            density_loss_cfg: GaussianDensityLoss | None = density_config
            use_visibility_scale, use_visibility_density = gaussian_visibility_flags(
                scale_loss_cfg, density_loss_cfg, run_gaussian_scale, run_gaussian_density
            )
            active_visibility_scale = use_visibility_scale and gaussian_scale_count > 0
            active_visibility_density = use_visibility_density and gaussian_density_count > 0

            if active_visibility_scale or active_visibility_density:
                element_count = max(
                    gaussian_scale_count if active_visibility_scale else 0,
                    gaussian_density_count if active_visibility_density else 0,
                )
                if active_visibility_scale and gaussian_scale_segments:
                    ref_tensor = gaussian_scale_segments[0]
                elif active_visibility_density and gaussian_density_segments:
                    ref_tensor = gaussian_density_segments[0]
                else:
                    ref_tensor = unpack_optional(gaussian_scales if active_visibility_scale else gaussian_densities)
                loss_cfg = scale_loss_cfg if active_visibility_scale else density_loss_cfg
                assert loss_cfg is not None
                occlusion_aware = bool(loss_cfg.occlusion_aware)
                visibility_gaussian = get_rendered_visibility_mask(
                    results=results,
                    element_count=element_count,
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                    visibility_filter=True,
                    occlusion_aware=occlusion_aware,
                )
                _maybe_update_mcmc_visibility_counters(model=model, results=results, visibility=visibility_gaussian)

        # Visibility is optional in gsplat. Keep it absent when filtering is
        # disabled instead of allocating a full all-ones tensor every step.
        gaussian_visibility = visibility_gaussian

        # Load out_of_bound inputs
        out_of_bound_position_segments: tuple[torch.Tensor, ...] = ()
        out_of_bound_cuboid_dims_tables: tuple[torch.Tensor, ...] = ()
        out_of_bound_cuboid_ids_segments: tuple[torch.Tensor, ...] = ()

        if run_out_of_bound:
            positions_list: list[torch.Tensor] = []
            dims_tables: list[torch.Tensor] = []
            cuboid_ids_list: list[torch.Tensor] = []
            for node in out_of_bound_gaussian_nodes:
                positions = node.get_positions()
                cuboid_dims = node.cuboid_tracks.cuboids_dims
                gaussian_cuboid_ids = node.gaussian_cuboid_ids
                assert positions.ndim == 2 and positions.shape[1] == 3 and positions.dtype == torch.float32, (
                    "out_of_bound_positions must be float32 [N,3]"
                )
                assert cuboid_dims.ndim == 2 and cuboid_dims.shape[1] == 3 and cuboid_dims.dtype == torch.float32, (
                    "out_of_bound_cuboid_dims must be a float32 [T,3] table"
                )
                assert gaussian_cuboid_ids.dtype in (
                    torch.int32,
                    torch.int64,
                ), "gaussian_cuboid_ids must be int32 or int64"
                assert gaussian_cuboid_ids.numel() == positions.shape[0], (
                    "gaussian_cuboid_ids must contain one index per position"
                )
                positions_list.append(positions)
                dims_tables.append(cuboid_dims)
                cuboid_ids_list.append(gaussian_cuboid_ids)
            out_of_bound_position_segments = tuple(positions_list)
            out_of_bound_cuboid_dims_tables = tuple(dims_tables)
            out_of_bound_cuboid_ids_segments = tuple(cuboid_ids_list)

        # Load Gaussian Z-scale inputs
        # Z-scale uses one configured layer without per-layer weighting. The
        # scale member instead supplies all node tensors and layer weights to
        # Layer 1, even when that includes the same node (e.g. "road").
        gaussian_z_scales: torch.Tensor | None = None
        z_scale_threshold = 0.0

        if run_gaussian_z_scale is not None:
            assert isinstance(model, GaussiansComposite)
            z_scale_threshold = run_gaussian_z_scale.road_z_scale
            # An enabled empty layer retains its original device and graph.
            gaussian_z_scales = model.gaussians_nodes[run_gaussian_z_scale.layer_name].get_scales(preactivation=True)

        # The deformation domain is independent of the Gaussian parameters.
        deformation = None
        deform_mask_source: torch.Tensor | None = None
        if run_deform is not None:
            deformation = unpack_optional(results.deform_smoothness)
            deform_mask_source = results.deform_smoothness_mask

        return self.accumulator.accumulate_gaussian(
            losses_vec,
            tuple(weights),
            scales=gaussian_scales,
            densities=gaussian_densities,
            z_scales=gaussian_z_scales,
            visibility=gaussian_visibility,
            deformation=deformation,
            deform_mask=deform_mask_source,
            scales_segments=gaussian_scale_segments,
            scale_segment_weights=gaussian_scale_segment_weights,
            densities_segments=gaussian_density_segments,
            density_segment_weights=gaussian_density_segment_weights,
            oob_positions_segments=out_of_bound_position_segments,
            oob_cuboid_dims_tables=out_of_bound_cuboid_dims_tables,
            oob_cuboid_ids_segments=out_of_bound_cuboid_ids_segments,
            z_scale_threshold=z_scale_threshold,
            densities_are_logits=gaussian_densities_are_logits,
        )

    def _accumulate_bg_grid(
        self,
        losses_vec: torch.Tensor,
        model: BaseModel,
        camera_grids: list[BilateralGridPerCamera],
        frame_grids: list[BilateralGridPerFrame],
        selected_losses: dict[str, tuple[Any, float]],
    ) -> torch.Tensor:
        names = BG_GRID_LOSS_NAMES
        weights = tuple(selected_losses.get(name, (None, 0.0))[1] for name in names)
        if not any(weight != 0.0 for weight in weights):
            return losses_vec
        grids_per_camera = grids_per_frame = bg_tex = None
        if camera_grids and (weights[1] != 0.0 or weights[2] != 0.0):
            assert len(camera_grids) == 1, "Fused losses require exactly one BilateralGridPerCamera"
            grids_per_camera = cast(torch.Tensor, camera_grids[0].bilateral_grid.grid)
        if frame_grids and (weights[1] != 0.0 or weights[3] != 0.0):
            assert len(frame_grids) == 1, "Fused losses require exactly one BilateralGridPerFrame"
            grids_per_frame = cast(torch.Tensor, frame_grids[0].bilateral_grid.grid)
        if weights[0] != 0.0:
            assert isinstance(model, GaussiansComposite), "SkyEnvMap loss requires GaussiansComposite model"
            assert isinstance(model.background, SkyEnvMapBackground), "SkyEnvMap loss requires SkyEnvMapBackground"
            bg_tex = unpack_optional(model.background.textures)
        return self.accumulator.accumulate_bg_grid(
            losses_vec,
            weights,
            bg_tex=bg_tex,
            grids_camera=grids_per_camera,
            grids_frame=grids_per_frame,
        )

    def _accumulate_bg_track_node(
        self,
        losses_vec: torch.Tensor,
        model: BaseModel,
        target: DataAndRenderingBatch,
        selected_losses: dict[str, tuple[Any, float]],
    ) -> torch.Tensor:
        names = BG_TRACK_NODE_LOSS_NAMES
        node_config, node_weight = selected_losses.get(names[1], (None, 0.0))
        if node_config is not None and node_weight == 0.0:
            # Keep the existing missing-input omission at zero weight. The
            # getter returns a view; only presence is inspected, with no tensor
            # value reads, casts, density loading, or semantic classification.
            assert isinstance(model, GaussiansComposite)
            configured_node_ids = node_config.layer_labels_to_use.keys() | node_config.layer_labels_to_exclude.keys()
            for node_id in model.get_gaussians_node_ids():
                if node_id not in configured_node_ids:
                    continue
                try:
                    semantic_input = model.gaussians_nodes[node_id].get_extra_signal_by_key("semantic_logits")
                except ValueError:
                    semantic_input = None
                if semantic_input is not None:
                    break
            else:
                selected_losses.pop(names[1])
        weights = [selected_losses.get(name, (None, 0.0))[1] for name in names]
        if not any(weight != 0.0 for weight in weights):
            return losses_vec
        run_background_in_track = selected_losses[names[0]][0] if weights[0] != 0.0 else None
        run_node_semantic = selected_losses[names[1]][0] if weights[1] != 0.0 else None

        # Semantic-filtered
        # layers are packed first. Their host-known shapes and configured labels become
        # immutable by-value launch metadata; unfiltered and missing-semantic layers form
        # an include-all suffix.
        bg_track_positions: torch.Tensor | None = None
        bg_track_density_logits: torch.Tensor | None = None
        bg_track_semantic_logits: torch.Tensor | None = None
        bg_track_n_semantic_points: int | None = None
        bg_track_semantic_segments: tuple[tuple[int, tuple[int, ...], torch.Tensor], ...] = ()
        bg_track_camera_timestamps_startend_us: torch.Tensor | None = None
        bg_track_tracks_packinfo: torch.Tensor | None = None
        bg_track_tracks_poses: torch.Tensor | None = None
        bg_track_tracks_timestamps_us: torch.Tensor | None = None
        bg_track_cuboids_dims: torch.Tensor | None = None
        bg_track_density_logits_min = -20.0
        bg_track_prepared = False
        background = None
        if run_background_in_track is not None:
            assert isinstance(model, GaussiansComposite), (
                "background_in_track_gaussian loss requires model to be GaussiansComposite"
            )
            background = prepare_background_tracks(model, target, run_background_in_track)
            if background is not None:
                bg_track_positions = background.positions
                bg_track_density_logits = background.density_logits
                bg_track_semantic_logits = background.semantic_logits
                bg_track_n_semantic_points = background.n_semantic_points
                bg_track_semantic_segments = background.semantic_segments
                bg_track_tracks_packinfo = background.tracks.tracks_packinfo
                bg_track_tracks_poses = background.tracks.tracks_poses
                bg_track_tracks_timestamps_us = background.tracks.tracks_timestamps_us
                bg_track_cuboids_dims = background.tracks.cuboids_dims
                bg_track_camera_timestamps_startend_us = background.camera_timestamps_startend_us
                bg_track_density_logits_min = run_background_in_track.density_logits_min
                bg_track_prepared = True

        other_node_density_input = None
        background_allowed_class_ids: tuple[int, ...] = ()
        primary_predicate: tuple[tuple[int, ...], bool] | None = None
        other_segments: tuple[tuple[int, tuple[int, ...], bool], ...] = ()
        other_semantic_logits: tuple[torch.Tensor, ...] = ()

        if run_node_semantic is not None:
            assert isinstance(model, GaussiansComposite)
            node_entries = prepare_node_semantics(model, run_node_semantic)

            if not node_entries:
                selected_losses.pop("node_semantic_gaussians_bce_with_logits_mean", None)
                weights[1] = 0.0
                run_node_semantic = None
            else:
                joint = prepare_joint_node_semantics(node_entries, background, run_background_in_track)
                other_node_density_input = joint.other_density
                background_allowed_class_ids = joint.background_allowed_class_ids
                primary_predicate = joint.primary_predicate
                other_segments = joint.other_segments
                other_semantic_logits = joint.other_semantic
                if joint.primary_node_id is not None:
                    # Node-only accumulator semantics needs no position getter or tensor.
                    bg_track_positions = background.filtered_entries[0][1] if background is not None else None
                    bg_track_density_logits = joint.primary_density
                    bg_track_semantic_logits = joint.primary_semantic
                    bg_track_n_semantic_points = None

        if not bg_track_prepared:
            # A configured track loss with no owned tracks remains a logged zero.
            weights[0] = 0.0
        if not any(weight != 0.0 for weight in weights):
            return losses_vec
        return self.accumulator.accumulate_bg_track_node(
            losses_vec,
            tuple(weights),
            positions=bg_track_positions if weights[0] != 0.0 else None,
            density_logits=unpack_optional(bg_track_density_logits),
            camera_timestamps_startend_us=bg_track_camera_timestamps_startend_us if weights[0] != 0.0 else None,
            tracks_packinfo=bg_track_tracks_packinfo if weights[0] != 0.0 else None,
            tracks_poses=bg_track_tracks_poses if weights[0] != 0.0 else None,
            tracks_timestamps_us=bg_track_tracks_timestamps_us if weights[0] != 0.0 else None,
            cuboids_dims=bg_track_cuboids_dims if weights[0] != 0.0 else None,
            semantic_logits=bg_track_semantic_logits,
            n_semantic_points=bg_track_n_semantic_points,
            background_semantic_segments=bg_track_semantic_segments,
            background_allowed_class_ids=background_allowed_class_ids,
            node_primary_predicate=primary_predicate,
            density_logits_min=bg_track_density_logits_min,
            other_density_logits=other_node_density_input if weights[1] != 0.0 else None,
            other_semantic_logits=other_semantic_logits if weights[1] != 0.0 else (),
            other_segments=other_segments if weights[1] != 0.0 else (),
        )

    def _accumulate_ground(
        self,
        losses_vec: torch.Tensor,
        model: BaseModel,
        target: DataAndRenderingBatch,
        selected_losses: dict[str, tuple[Any, float]],
    ) -> torch.Tensor:
        road_loss, weight = selected_losses.get("road_gaussians_abs_mean", (None, 0.0))
        if weight == 0.0:
            return losses_vec
        assert isinstance(model, GaussiansComposite)
        road_node = model.gaussians_nodes[road_loss.layer_name]
        road_positions = road_node.get_positions()
        road_rotations = road_node.get_rotations(quaternion_format="wxyz", preactivation=True)
        rendering_camera = unpack_optional(unpack_optional(target.rendering).camera)
        poses = rendering_camera.poses_tquat_startend
        # Reject invalid camera metadata before Ground sampling advances the RNG.
        assert poses.ndim == 3 and poses.shape == (
            1,
            2,
            7,
        ), "road loss requires camera poses of shape [1,2,7]"
        random_values = road_loss._make_random_values(road_positions.device, road_positions.dtype)
        return self.accumulator.accumulate_ground(
            losses_vec,
            weight,
            positions=road_positions,
            rotations=road_rotations,
            camera_poses=poses,
            random_values=random_values,
            min_bias=road_loss.min,
            range_bias=road_loss.range,
            grid_len=road_loss.grid_len,
            rotation_lambda=road_loss.rotation_lambda,
        )
