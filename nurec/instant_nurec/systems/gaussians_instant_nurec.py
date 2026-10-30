# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from einops import rearrange
from tqdm import tqdm

import ncore.impl.common.transformations as ncore_transformations

from nurec.compression.usdz import write_artifact_cache
from nurec.instant_nurec.config.export_paths import resolve_export_ply_base_dir
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.instant_nurec.models import make_instant_nurec_model
from nurec.instant_nurec.models.base import BaseInstantNuRecSupervisionPack
from nurec.instant_nurec.predict.export_ply import export_ply
from nurec.instant_nurec.predict.export_usdz import build_artifact_cache
from nurec.instant_nurec.predict.prediction_runner import (
    reconstruct_instant_nurec_context,
    run_instant_nurec_prediction,
)
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.instant_nurec.systems.base import BaseInstantNuRecSystem
from nurec.instant_nurec.utils.trajectory import SensorOverride, pad_rig_timestamps
from nurec.render.render import CameraRenderOptions, PoseRange, RenderableModel
from nurec.utils.batch import InstantNuRecDataBatch
from nurec.utils.geometry import se3_matrix_to_tquat
from nurec.utils.misc import unpack_optional
from nurec.utils.types import (
    GaussiansCompositeReturn,
    RigTrajectories,
)
from nurec.utils.visualize import scalar2img


logger = logging.getLogger(__name__)


class GaussiansInstantNuRecSystem(BaseInstantNuRecSystem):
    """
    Inference-only Gaussians instant_nurec system: model construction, forward pass, and the
    `predict_step`/`on_predict_batch_end` export pipeline. Training/validation/test loops
    and media-logging visualizations live on `TrainGaussiansInstantNuRecSystem` in a separate
    downstream training package.
    """

    def __init__(self, config: InstantNuRecConfig) -> None:
        super().__init__(config)
        self.model = make_instant_nurec_model(config.model)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> Any:
        """Keep validation-only LPIPS state compatible without relaxing model loading.

        Older training checkpoints persisted LPIPS backbone tensors. Inference
        does not construct that metric, and training constructs it during setup,
        after weights-only initialization. Conversely, pre-LPIPS checkpoints do
        not contain its fixed pretrained tensors. Only this exact system-level
        metric prefix is optional; reconstruction/loss weights remain strict.
        """
        current_state = self.state_dict()
        metric_prefix = "validation_lpips."
        filtered_state = OrderedDict(state_dict)
        if (metadata := getattr(state_dict, "_metadata", None)) is not None:
            setattr(filtered_state, "_metadata", metadata.copy())

        removed = 0
        for key in tuple(filtered_state):
            if key.startswith(metric_prefix) and key not in current_state:
                del filtered_state[key]
                removed += 1
        backfilled = 0
        for key, value in current_state.items():
            if key.startswith(metric_prefix) and key not in filtered_state:
                filtered_state[key] = value
                backfilled += 1
        if removed or backfilled:
            logger.info(
                "Adapted validation-only LPIPS checkpoint state: ignored %d obsolete tensors, "
                "initialized %d missing tensors; reconstruction weights remain strict",
                removed,
                backfilled,
            )
        return super().load_state_dict(filtered_state, strict=strict, assign=assign)

    def forward(
        self, batch: InstantNuRecDataBatch
    ) -> tuple[list[BaseInstantNuRecPrimitive], list[BaseInstantNuRecSupervisionPack], list[GaussiansCompositeReturn]]:
        primitives, supervision_packs, cuboid_tracks = reconstruct_instant_nurec_context(
            model=self.model,
            batch=batch,
            media_logger=self.media_logger,
            compute_supervision_pack=batch.supervision is not None,
        )

        # If no supervision is provided, we don't need to render at those poses.
        if batch.supervision is None:
            return primitives, [] if supervision_packs is None else supervision_packs, []

        # Prepare supervision signals
        batch.supervision, prepared_supervision_packs = self.model.prepare_supervision(
            batch.context, batch.supervision, cuboid_tracks, unpack_optional(supervision_packs), self.media_logger
        )

        # If no rendering is needed, we don't need to forward the primitives.
        if self.global_step < self.config.enable_render_global_step:
            return primitives, prepared_supervision_packs, [GaussiansCompositeReturn() for _ in primitives]

        # Forward the primitives
        traced_returns: list[GaussiansCompositeReturn] = []
        for primitive, supervision in zip(primitives, unpack_optional(batch.supervision)):
            assert supervision.rendering is not None, "Rendering must be provided"
            assert supervision.data is not None, "Data must be provided"
            traced_returns.append(
                primitive.forward(
                    rendering_cam_data=supervision.rendering.camera,
                    frames_cam_meta=supervision.data.camera.meta if supervision.data.camera is not None else None,
                    rendering_lidar_data=supervision.rendering.lidar,
                    frames_lidar_meta=supervision.data.lidar.meta if supervision.data.lidar is not None else None,
                )
            )

        return primitives, prepared_supervision_packs, traced_returns

    @staticmethod
    def _rendered_to_distance_img(
        rendered_distance: torch.Tensor | np.ndarray,
        rendered_opacity: torch.Tensor | np.ndarray,
        img_width: int,
        img_height: int,
        start_n_rays: int = 0,
    ) -> np.ndarray:
        img_n_pixels = img_width * img_height
        # Expected distance with opacity mask
        rendered_distance = rendered_distance[start_n_rays : start_n_rays + img_n_pixels]
        rendered_opacity = rendered_opacity[start_n_rays : start_n_rays + img_n_pixels]
        if isinstance(rendered_distance, torch.Tensor):
            rendered_distance = rendered_distance.cpu().numpy()
        if isinstance(rendered_opacity, torch.Tensor):
            rendered_opacity = rendered_opacity.cpu().numpy()
        # Compute inverse expected distance, weighted by opacity (ZipNeRF visualization formulation)
        expected_distance = (rendered_opacity / np.clip(rendered_distance, a_min=1.0e-6, a_max=None)) * rendered_opacity
        return scalar2img(rearrange(expected_distance, "(h w) -> h w", h=img_height), vmin=0.0, vmax=0.5)

    def _extract_render_timestamps_us(self, rig_trajectories: RigTrajectories) -> tuple[int, int]:
        min_timestamps_us: int = 2**63 - 1
        max_timestamps_us: int = 0
        for context_single_rig in rig_trajectories.rig_trajectories:
            for camera_timestamps_us in context_single_rig.cameras_frame_timestamps_us.values():
                min_timestamps_us = min(min_timestamps_us, int(camera_timestamps_us.min().item()))
                max_timestamps_us = max(max_timestamps_us, int(camera_timestamps_us.max().item()))

        return min_timestamps_us, max_timestamps_us

    def render_rig_trajectories_video(
        self,
        camera_unique_id: str,
        primitive: BaseInstantNuRecPrimitive,
        supervision_rig: RigTrajectories,
        timestamps_us_range: tuple[int, int],
        fps: int,
        sensor_override: SensorOverride | None = None,
    ):
        # Extend rig trajectory to cover the render timestamps (use constant padding).
        supervision_rig = pad_rig_timestamps(supervision_rig, *timestamps_us_range)

        # Prepare the renderable model.
        renderable_model = RenderableModel.load_from_instant_nurec_primitive(primitive, supervision_rig.world_to_nurec)

        # Prepare rendering timestamps
        render_timestamps_us = np.arange(*timestamps_us_range, 1_000_000 // fps)
        # If too few frames (e.g. timestamps_us is actually img index), just sample a 1s video
        if len(render_timestamps_us) < fps:
            render_timestamps_us = np.linspace(*timestamps_us_range, num=fps, dtype=np.int64)

        camera_calibration = supervision_rig.camera_calibrations[camera_unique_id]
        matched_trajectories = [
            traj for traj in supervision_rig.rig_trajectories if camera_calibration.sequence_id == traj.sequence_id
        ]
        assert len(matched_trajectories) == 1, "Expected exactly one trajectory for each camera calibration."
        trajectory = matched_trajectories[0]

        # Interpolate the rig trajectory to get poses from the rendering timestamps.
        interpolator = ncore_transformations.PoseInterpolator(
            trajectory.T_rig_worlds.cpu().numpy(), trajectory.T_rig_world_timestamps_us.cpu().numpy().astype(np.int64)
        )
        T_rig_world = (
            torch.from_numpy(interpolator.interpolate_to_timestamps(render_timestamps_us)).float().to(self.device)
        )  # [t, 4, 4]
        with torch.autocast(device_type="cuda", enabled=False):
            T_sensor_world = T_rig_world @ camera_calibration.T_sensor_rig
            if sensor_override is not None:
                T_sensor_world = sensor_override.apply_sensor_to_world(T_sensor_world)
            camera_world_tquat = se3_matrix_to_tquat(T_sensor_world)

        parameters = camera_calibration.camera_model_parameters
        if sensor_override is not None:
            parameters = sensor_override.apply_camera_model_parameters(parameters)
        img_width, img_height = parameters.resolution.tolist()

        # Yield the rendered frames when ready
        for frame_tquat, render_timestamp_us in zip(camera_world_tquat, render_timestamps_us):
            with torch.inference_mode():
                yield renderable_model.render_camera_frame(
                    camera_intrinsics=parameters,
                    camera_to_world=PoseRange(
                        frame_tquat,
                        frame_tquat,
                        render_timestamp_us,
                        render_timestamp_us,
                    ),
                    resolution=(img_width, img_height),
                    unique_sensor_idx=camera_calibration.unique_sensor_idx,
                    unique_frame_idx=0,  # Dummy data since we don't have this
                    render_options=CameraRenderOptions(
                        color_image=True,
                        distance_image=True,
                        opacity_image=True,
                    ),
                )

    def predict_step(
        self, batch: InstantNuRecDataBatch, batch_local_idx: int
    ) -> dict[str, list[BaseInstantNuRecPrimitive] | InstantNuRecDataBatch]:
        output = run_instant_nurec_prediction(
            batch=batch,
            device=self.device,
            export_preprocess_config=self.cached_config.model.export_preprocess,
            predict_config=self.predict_config,
            forward_chunk=self.forward,
        )
        return {"primitives": output.primitives, "batch": output.batch}

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)

        # Ensure outputs are not None and contain the required keys
        assert outputs is not None and "primitives" in outputs and "batch" in outputs

        out_batch: InstantNuRecDataBatch = outputs["batch"]
        primitives_list: list[BaseInstantNuRecPrimitive] = outputs["primitives"]
        n_chunks = len(primitives_list)
        assert len(out_batch) == n_chunks, "batch context length must match number of primitives"

        if out_batch.meta is None or out_batch.context_rig is None:
            return

        # Helper to export PLY and USDZ for one chunk
        def export_chunk(
            primitive: BaseInstantNuRecPrimitive, rig: RigTrajectories, meta: dict, chunk_suffix: str
        ) -> None:
            if self.predict_config.export_ply.enabled:
                export_ply_base_dir = resolve_export_ply_base_dir(
                    self.predict_config.export_ply.save_to_dir,
                    out_dir=self.out_dir,
                    run_id=self.run_id,
                    meta=meta,
                )
                path = os.path.join(
                    export_ply_base_dir,
                    "ply",
                    meta["sequence_id"],
                    meta["sequence_id"] + chunk_suffix + ".ply",
                )
                export_ply(
                    config=self.predict_config.export_ply,
                    primitives=primitive,
                    rig_trajectories=rig,
                    path=Path(path),
                )
            if self.predict_config.artifact.enabled:
                path = os.path.join(
                    self.out_dir,
                    self.run_id,
                    "artifacts",
                    meta["sequence_id"],
                    "last" + chunk_suffix + ".usdz",
                )
                artifact_cache = build_artifact_cache(
                    instant_nurec_config=self.cached_config,
                    artifact_config=self.predict_config.artifact,
                    primitive=primitive,
                    rig_trajectories=rig,
                    meta_data=meta,
                )
                write_artifact_cache(artifact_cache, Path(path))

        for chunk_idx in range(n_chunks):
            meta = out_batch.meta[chunk_idx]
            assert "sequence_id" in meta, f"sequence_id key must be provided, only got {meta.keys()}"
            chunk_suffix = "" if self.predict_config.primitive_merge.enabled else f"_chunk{chunk_idx}"
            export_chunk(
                primitives_list[chunk_idx],
                out_batch.context_rig[chunk_idx],
                meta,
                chunk_suffix,
            )

        # Export rendered videos only for merged output (single primitive/batch)
        if n_chunks != 1:
            return

        merged_primitive: BaseInstantNuRecPrimitive = primitives_list[0]
        merged_rig: RigTrajectories = out_batch.context_rig[0]
        meta = out_batch.meta[0]

        if self.predict_config.render_video.enabled:
            os.makedirs(
                video_dir := os.path.join(self.out_dir, self.run_id, "videos", meta["sequence_id"]),
                exist_ok=True,
            )
            timestamps_us_range = self._extract_render_timestamps_us(merged_rig)
            # Determine what cameras to render
            camera_unique_ids: list[str] = []
            camera_overrides: list[SensorOverride | None] = []
            camera_save_names: list[str] = []
            # Check if any camera overrides are provided
            if self.predict_config.render_video.override_sensors is None:
                camera_unique_ids = list(merged_rig.camera_calibrations.keys())
                camera_overrides = [None for _ in camera_unique_ids]
                camera_save_names = camera_unique_ids[:]
            else:
                for idx, override_sensor in enumerate(self.predict_config.render_video.override_sensors):
                    if override_sensor.sensor_id not in merged_rig.camera_calibrations:
                        logger.warning(f"Camera {override_sensor.sensor_id} not found in merged rig, skipping")
                        continue
                    camera_unique_ids.append(override_sensor.sensor_id)
                    camera_overrides.append(SensorOverride.from_config(override_sensor))
                    camera_save_names.append(f"override_{idx}")
            # Perform actual rendering
            for camera_unique_id, camera_override, camera_save_name in zip(
                camera_unique_ids, camera_overrides, camera_save_names
            ):
                ims: list[np.ndarray] = []
                for rendered_camera_frame in tqdm(
                    self.render_rig_trajectories_video(
                        camera_unique_id,
                        merged_primitive,
                        merged_rig,
                        timestamps_us_range,
                        self.predict_config.render_video.fps,
                        camera_override,
                    ),
                    desc=f"Rendering {camera_unique_id} ({camera_save_name})",
                    unit=" frame",
                ):
                    im_pd_rgb = np.clip(rendered_camera_frame.color_image.cpu().numpy() * 255.0, 0, 255).astype(
                        np.uint8
                    )
                    if self.predict_config.render_video.depth:
                        im_pd_distance = self._rendered_to_distance_img(
                            rendered_camera_frame.distance_image.reshape(-1),
                            rendered_camera_frame.opacity_image.reshape(-1),
                            im_pd_rgb.shape[1],
                            im_pd_rgb.shape[0],
                        )
                        ims.append(np.concatenate([im_pd_rgb, im_pd_distance], axis=1))
                    else:
                        ims.append(im_pd_rgb)
                imageio.v2.mimwrite(
                    os.path.join(video_dir, f"{camera_save_name}.mp4"),
                    ims,  # type: ignore
                    fps=self.predict_config.render_video.fps,
                    macro_block_size=1,
                )
                logger.info(f"Exported video for camera {camera_unique_id} ({camera_save_name}) under {video_dir}")
