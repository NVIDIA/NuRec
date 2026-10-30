# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

from abc import abstractmethod
from typing import Any, Callable, List, Literal, Optional, Self, TypeVar

import torch
import torch.utils.checkpoint

from omegaconf import DictConfig, OmegaConf

from nurec.config.version import get_version
from nurec.harmonizer.model import BaseHarmonizerModel
from nurec.instant_nurec.config.models import PrimitiveExportPreprocessConfig
from nurec.instant_nurec.supervision_pack import BaseInstantNuRecSupervisionPack
from nurec.models.gaussians.renderers import BaseGaussianRenderer
from nurec.utils.batch import DataAndRenderingBatch, FrameMeta, RenderingData
from nurec.utils.misc import map_optional, strip_none_from_config
from nurec.utils.profiling import ScopedTimer
from nurec.utils.types import (
    Checkpoint,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    RigTrajectories,
)


class BaseInstantNuRecPrimitive:
    """
    Base class for all renderable primitives reconstructed by an instant_nurec.
    """

    @abstractmethod
    def forward(
        self,
        rendering_cam_data: Optional[RenderingData] = None,
        frames_cam_meta: Optional[List[FrameMeta]] = None,
        rendering_lidar_data: Optional[RenderingData] = None,
        frames_lidar_meta: Optional[List[FrameMeta]] = None,
    ) -> GaussiansCompositeReturn:
        pass

    @abstractmethod
    def state_dict_and_config(self) -> tuple[dict[str, Any], DictConfig]:
        """Returns (state_dict, config) consumed by serialize_to_json_dict to build the model config blob."""

    @abstractmethod
    def get_checkpoint(self) -> Checkpoint:
        """Used for serialization to be used in render."""

    @torch.no_grad()
    def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict[str, Any]:
        """Serializes model config for consumption by GSplatRenderer.__init__ at initialization time."""

        state_dict, config = self.state_dict_and_config()
        config_dict = strip_none_from_config(OmegaConf.to_container(config))
        json_dict = {
            "nurec_data": {
                "version": map_optional(get_version(), lambda v: v.semantic_string()),
                "model": "nurec",
                # Empty to be filled by the derived class
                "config": config_dict,
                "state_dict": {f".{key}": value for key, value in state_dict.items()} if with_state_dict else {},
            }
        }

        if with_state_dict:
            assert isinstance(json_dict["nurec_data"]["state_dict"], dict)
            # add shape entries for every tensor
            json_dict["nurec_data"]["state_dict"].update(
                {
                    key + ".shape": list(value.size())
                    for key, value in json_dict["nurec_data"]["state_dict"].items()
                    if isinstance(value, torch.Tensor) and isinstance(key, str)
                }
            )

            # convert tensor to bytes
            def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
                # default conversion to half for single precision tensor
                tensor = tensor.to(dtype=torch.float16) if tensor.dtype == torch.float32 else tensor
                return tensor.flatten().cpu().numpy().tobytes()

            json_dict["nurec_data"]["state_dict"] = {
                key: tensor_to_bytes(value) if isinstance(value, torch.Tensor) else value
                for key, value in json_dict["nurec_data"]["state_dict"].items()
            }

        return json_dict

    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def detach(self) -> BaseInstantNuRecPrimitive: ...

    @abstractmethod
    def rigid_transform(self, T_new: torch.Tensor) -> Self: ...

    @abstractmethod
    def preprocess_for_export(
        self,
        context_batch: DataAndRenderingBatch,
        config: PrimitiveExportPreprocessConfig,
        context_rig: RigTrajectories | None = None,
        supervision_pack: BaseInstantNuRecSupervisionPack | None = None,
    ) -> Self:
        """
        Filter and preprocess the primitive for export (e.g. density/sky/road masking).
        Called per chunk after forward; when merging is enabled, merge will then apply
        rigid_transform to align chunks. Implementations must not apply rigid_transform.
        ``context_rig`` is optional; Celsius may require it when project_to_z_offset is True.
        ``supervision_pack`` is optional; Kelvin uses it (when present) as the preferred source
        for predicted per-image semantics during sky-mask precomputation.
        """

    @abstractmethod
    def __len__(self) -> int: ...

    def get_all_gaussian_positions(self) -> torch.Tensor | None:
        return None


InstantNuRecPrimitiveType = TypeVar("InstantNuRecPrimitiveType", bound=BaseInstantNuRecPrimitive)


class BaseGaussiansInstantNuRecPrimitive(BaseInstantNuRecPrimitive):
    """
    instant_nurec primitives that is rendered using Gaussians and potentially postprocessings (e.g. sky and PPISPs).
    """

    # Here we allow the gaussians_renderer to be None to just carry the data.
    # The main use case is when we want to pass in this model for renderer initialization (to break circular dependency).
    gaussians_renderer: BaseGaussianRenderer | None

    # Harmonizer model that is applied after the final rendering is completed.
    harmonizer_model: BaseHarmonizerModel | None
    harmonizer_use_color_transfer: bool

    def __init__(
        self,
        gaussians_renderer: BaseGaussianRenderer | None,
        checkpointing: Literal["render", "all", "none"] = "render",
        shared_gaussian_parameters: bool = False,
        harmonizer_model: BaseHarmonizerModel | None = None,
        harmonizer_use_color_transfer: bool = True,
    ):
        self.gaussians_renderer = gaussians_renderer
        self.checkpointing = checkpointing
        # If True, when rendering multiple frames within a single forward call, the first frame's FramesMeta will be used to get the gaussian parameters for all frames.
        # Useful to save memory when rendering multiple frames.
        self.shared_gaussian_parameters = shared_gaussian_parameters
        self.harmonizer_model = harmonizer_model
        self.harmonizer_use_color_transfer = harmonizer_use_color_transfer

    @abstractmethod
    def get_gaussian_parameters(self, timestamps_us: torch.Tensor | None) -> dict[str, torch.Tensor]: ...

    def get_all_gaussian_positions(self) -> torch.Tensor | None:
        """
        Get all Gaussian center positions from the primitive.

        Returns:
            torch.Tensor | None: Positions of all Gaussians, shape (N, 3), or None if not available
        """
        gaussian_params = self.get_gaussian_parameters(None)
        if "positions" in gaussian_params:
            return gaussian_params["positions"]
        else:
            return None

    def get_extra_ray_signal_infos(self) -> tuple[list[str], list[int], list[Callable]]:
        return ([], [], [])

    def postprocess_rendering(
        self, out: GaussiansRenderReturn, rendering_data: RenderingData, unique_sensor_idx: int | None
    ) -> GaussiansRenderReturn:
        """This is supposed to be called one image at a time."""
        return out

    @ScopedTimer("BaseGaussiansInstantNuRecPrimitive.forward")
    def forward(
        self,
        rendering_cam_data: Optional[RenderingData] = None,
        frames_cam_meta: Optional[List[FrameMeta]] = None,
        rendering_lidar_data: Optional[RenderingData] = None,
        frames_lidar_meta: Optional[List[FrameMeta]] = None,
    ) -> GaussiansCompositeReturn:
        # Rendering logic should resemble GaussiansComposite.forward() but conducted in batch.

        # Precompute the gaussian parameters for all frames.
        shared_gaussian_parameters: dict[str, torch.Tensor] = {}
        if self.shared_gaussian_parameters:
            if rendering_cam_data is not None:
                # Use CPU copy to avoid GPU->CPU sync when .item() is called inside get_gaussian_parameters
                shared_gaussian_parameters = self.get_gaussian_parameters(
                    rendering_cam_data.timestamps_startend_us_cpu[0:1]
                )
            elif rendering_lidar_data is not None:
                # Use CPU copy to avoid GPU->CPU sync when .item() is called inside get_gaussian_parameters
                shared_gaussian_parameters = self.get_gaussian_parameters(
                    rendering_lidar_data.timestamps_startend_us_cpu[0:1]
                )

        # Render the images
        out_cam_list: list[GaussiansRenderReturn] = []
        if frames_cam_meta is not None:
            assert rendering_cam_data is not None, (
                f"{self.__class__.__name__} All camera information has to be provided for rendering."
            )
            out_cam_list = self._render_cameras(rendering_cam_data, frames_cam_meta, shared_gaussian_parameters)

        # Render the lidar
        out_lidar_list: list[GaussiansRenderReturn] = []
        if frames_lidar_meta is not None:
            assert rendering_lidar_data is not None, (
                f"{self.__class__.__name__} All lidar information has to be provided for rendering."
            )
            out_lidar_list = self._render_lidars(rendering_lidar_data, frames_lidar_meta, shared_gaussian_parameters)

        return GaussiansCompositeReturn(
            rendered_cam=GaussiansRenderReturn.collate_fn(out_cam_list, device=self.device()) if out_cam_list else None,
            rendered_lidar=GaussiansRenderReturn.collate_fn(out_lidar_list, device=self.device())
            if out_lidar_list
            else None,
        )

    def _render_cameras(
        self,
        rendering_data: RenderingData,
        frame_meta: List[FrameMeta],
        shared_gaussian_parameters: dict[str, torch.Tensor],
    ) -> list[GaussiansRenderReturn]:
        """Render camera frames."""
        assert rendering_data.b == len(frame_meta), (
            f"Expected batch size to be {len(frame_meta)}, but got {rendering_data.b}"
        )

        render_all_fn = (
            functools.partial(torch.utils.checkpoint.checkpoint, self._render_single_camera_frame, use_reentrant=False)
            if self.checkpointing == "all"
            else self._render_single_camera_frame
        )

        B = rendering_data.b
        out_cam_list: list[GaussiansRenderReturn] = []
        for batch_idx in range(B):
            out_cam = render_all_fn(rendering_data[batch_idx], frame_meta[batch_idx], shared_gaussian_parameters)
            out_cam_list.append(out_cam)

        return out_cam_list

    def _render_single_camera_frame(
        self,
        rendering_data: RenderingData,
        frame_meta: FrameMeta,
        shared_gaussian_parameters: dict[str, torch.Tensor],
    ) -> GaussiansRenderReturn:
        """Render a single camera frame."""
        assert self.gaussians_renderer is not None, f"{self.__class__.__name__}: gaussians_renderer has to be provided."
        assert rendering_data.b == 1, f"Expected batch size to be 1, but got {rendering_data.b}"

        gaussian_parameters = (
            shared_gaussian_parameters
            if self.shared_gaussian_parameters
            else self.get_gaussian_parameters(rendering_data.timestamps_startend_us)
        )
        render_fn = (
            self.gaussians_renderer.render_with_deferred_bp
            if self.checkpointing == "render"
            else self.gaussians_renderer.render
        )

        out = render_fn(
            rendering_data=rendering_data,
            gaussian_parameters=gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=self.get_extra_ray_signal_infos(),
            frame_meta=[frame_meta],
        )
        out = self.postprocess_rendering(out, rendering_data, frame_meta.unique_sensor_idx)

        # Apply Harmonizer model (only for camera frames)
        if self.harmonizer_model is not None:
            out.rgb = self.harmonizer_model(
                out.rgb,
                torch.Size([rendering_data.h, rendering_data.w]),
                self.harmonizer_use_color_transfer,
            )

        return out

    def _render_lidars(
        self,
        rendering_data: RenderingData,
        frame_meta: List[FrameMeta],
        shared_gaussian_parameters: dict[str, torch.Tensor],
    ) -> list[GaussiansRenderReturn]:
        """Render lidar frames."""
        assert rendering_data.b == len(frame_meta), (
            f"Expected batch size to be {len(frame_meta)}, but got {rendering_data.b}"
        )

        render_all_fn = (
            functools.partial(torch.utils.checkpoint.checkpoint, self._render_single_lidar_frame, use_reentrant=False)
            if self.checkpointing == "all"
            else self._render_single_lidar_frame
        )

        B = rendering_data.b
        out_lidar_list: list[GaussiansRenderReturn] = []
        for batch_idx in range(B):
            out_lidar = render_all_fn(rendering_data[batch_idx], frame_meta[batch_idx], shared_gaussian_parameters)
            out_lidar_list.append(out_lidar)
        return out_lidar_list

    def _render_single_lidar_frame(
        self, rendering_data: RenderingData, frame_meta: FrameMeta, shared_gaussian_parameters: dict[str, torch.Tensor]
    ) -> GaussiansRenderReturn:
        """Render a single lidar frame."""
        assert self.gaussians_renderer is not None, f"{self.__class__.__name__}: gaussians_renderer has to be provided."
        assert rendering_data.b == 1, f"Expected batch size to be 1, but got {rendering_data.b}"

        gaussian_parameters = (
            shared_gaussian_parameters
            if self.shared_gaussian_parameters
            else self.get_gaussian_parameters(rendering_data.timestamps_startend_us)
        )
        render_fn = (
            self.gaussians_renderer.render_with_deferred_bp
            if self.checkpointing == "render"
            else self.gaussians_renderer.render
        )
        out = render_fn(
            rendering_data=rendering_data,
            gaussian_parameters=gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=self.get_extra_ray_signal_infos(),
            frame_meta=[frame_meta],
        )
        return self.postprocess_rendering(out, rendering_data, frame_meta.unique_sensor_idx)
