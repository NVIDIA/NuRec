# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import os

from dataclasses import dataclass, field
from typing import Callable, Optional, Self

import imageio.v2 as iio
import torch

from einops import rearrange
from torchvision.transforms.functional import InterpolationMode, resize

from nurec.grpc.protos.sensorsim_pb2 import (
    AvailableEgoMasksReturn,
    CameraRenderProduct,
    CameraRenderRequest,
    EgoMaskId,
    RGBRenderRequest,
)


logger = logging.getLogger("nurec.grpc.serve")


@dataclass(frozen=True)
class EgocarHood:
    """
    Describes an image of the egocar hood (visible in video but not in NuRec renders)
    that is pasted over the rendered view at the end of view generation.

    Only the file location is recorded, so a bank with thousands of hoods costs nothing
    until a hood is used. `EgocarRigBank.overlay_on_image` decodes and caches the image.
    """

    camera_logical_id: str
    rig_id: str
    path: str

    def load(self) -> torch.Tensor:
        """
        Reads the hood as a (4, H, W) float tensor in [0, 1].

        Assumes an RGBA image which is transparent outside of the car hood.
        """
        hood_rgba = iio.imread(self.path)
        if hood_rgba.ndim != 3 or hood_rgba.shape[-1] != 4:
            raise ValueError(f"Egocar hood {self.path} is not an RGBA image (shape {hood_rgba.shape})")
        return rearrange(torch.from_numpy(hood_rgba), "h w c -> c h w").float() / 255.0

    @property
    def metadata(self) -> AvailableEgoMasksReturn.EgoMaskMetadata:
        return AvailableEgoMasksReturn.EgoMaskMetadata(
            ego_mask_id=EgoMaskId(camera_logical_id=self.camera_logical_id, rig_config_id=self.rig_id),
        )


@dataclass
class EgocarRig:
    """
    Holds a collection of EgocarHoods, keyed by camera logical ID.
    """

    rig_id: str
    _hoods: dict[str, EgocarHood]

    def get(self, camera_logical_id: str) -> Optional[EgocarHood]:
        return self._hoods.get(camera_logical_id, None)

    @classmethod
    def from_dir(cls, rig_id: str, path: str) -> Self:
        """Lists all ego hood images at <path>/<camera_logical_id>.png without decoding them."""
        SUFFIX = ".png"

        hoods = {}
        for filename in sorted(os.listdir(path)):
            hood_path = os.path.join(path, filename)
            if not filename.endswith(SUFFIX) or not os.path.isfile(hood_path):
                continue
            camera_logical_id = filename.removesuffix(SUFFIX)
            hoods[camera_logical_id] = EgocarHood(camera_logical_id, rig_id, hood_path)

        return cls(rig_id, hoods)

    def __repr__(self) -> str:
        return f"EgocarRig({', '.join(self._hoods.keys())})"

    def available_metadata(self) -> list[AvailableEgoMasksReturn.EgoMaskMetadata]:
        return [hood.metadata for hood in self._hoods.values()]


@dataclass
class EgocarRigBank:
    """
    Holds EgocarRigs corresponding to different rig configurations.

    Listing a bank is a directory walk; the images stay on disk. `overlay_on_image`
    decodes a hood on first use and keeps it, resized and on the render device, in a
    least-recently-used cache of at most `cache_size` entries keyed by hood, render size
    and device. Memory therefore follows the hoods in active use, not the size of the
    bank. A cached 1920x1080 hood is 33 MB of float32.
    """

    _rigs: dict[str, EgocarRig]
    cache_size: int = 32
    _resized_hood: Callable[[EgocarHood, tuple[int, int], torch.device], torch.Tensor] = field(
        init=False,
        repr=False,
    )
    _logged_mask_selection_keys: set[tuple[str, str, str, bool]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def __post_init__(self):
        assert self.cache_size >= 1, f"cache_size must be positive, got {self.cache_size}"
        # A per-bank cache; lru_cache is thread-safe, and concurrent misses on one key only duplicate work.
        self._resized_hood = functools.lru_cache(maxsize=self.cache_size)(self._load_resized_hood)

    @classmethod
    def from_dir(cls, path: str, cache_size: int = 32) -> Self:
        """Lists all egocar rigs from <path>/<rig_id>/<camera_logical_id>.png without decoding them."""
        rigs = {}
        for rig_id in sorted(os.listdir(path)):
            rig_path = os.path.join(path, rig_id)
            if not os.path.isdir(rig_path):
                continue
            rigs[rig_id] = EgocarRig.from_dir(rig_id, rig_path)

        return cls(rigs, cache_size=cache_size)

    @classmethod
    def empty(cls) -> Self:
        return cls({})

    def __repr__(self) -> str:
        return f"EgocarRigBank({', '.join(self._rigs.keys())})"

    def available_metadata(self) -> list[AvailableEgoMasksReturn.EgoMaskMetadata]:
        return [metadata for rig in self._rigs.values() for metadata in rig.available_metadata()]

    @staticmethod
    def _load_resized_hood(hood: EgocarHood, size: tuple[int, int], device: torch.device) -> torch.Tensor:
        logger.info(f"EgocarHood: decoding {hood.path} at {size} on {device}.")
        hood_rgba = hood.load()
        if hood_rgba.shape[1:3] != size:
            hood_rgba = resize(hood_rgba, list(size), interpolation=InterpolationMode.BILINEAR)
        return hood_rgba.to(device)

    def overlay_on_image(self, hood: EgocarHood, rendered_image: torch.Tensor) -> torch.Tensor:
        """
        Overlay the egocar hood on a rendered image.

        Args:
            hood: A hood from this bank, usually from `select_from_request`.
            rendered_image: (H, W, C), RGB, floating point in [0, 1]

        Returns:
            (H, W, C), RGB, floating point in [0, 1]
        """
        height, width = rendered_image.shape[0:2]
        hood_rgba = self._resized_hood(hood, (height, width), rendered_image.device)
        hood_rgb = rearrange(hood_rgba[:3], "c h w -> h w c")
        mask = hood_rgba[3].unsqueeze(-1)  # (h, w, 1)

        return rendered_image * (1 - mask) + hood_rgb * mask

    def _log_mask_selection_once(self, request: RGBRenderRequest, hood: Optional[EgocarHood]) -> None:
        render_camera_logical_id = request.camera_intrinsics.logical_id or request.ego_mask_id.camera_logical_id
        if not render_camera_logical_id:
            render_camera_logical_id = "<unspecified>"

        requested_rig_id = request.ego_mask_id.rig_config_id
        requested_camera_logical_id = request.ego_mask_id.camera_logical_id
        log_key = (
            render_camera_logical_id,
            requested_rig_id,
            requested_camera_logical_id,
            request.insert_ego_mask,
        )
        if log_key in self._logged_mask_selection_keys:
            return
        self._logged_mask_selection_keys.add(log_key)

        if not request.insert_ego_mask:
            logger.info(
                f"New logical camera id '{render_camera_logical_id}' requested no ego mask; rendering without ego mask."
            )
            return

        requested_mask = f"rig_config_id='{requested_rig_id}', camera_logical_id='{requested_camera_logical_id}'"
        if hood is None:
            logger.info(
                f"New logical camera id '{render_camera_logical_id}' requested ego mask {requested_mask}; "
                "no matching mask was found."
            )
            return

        logger.info(
            f"New logical camera id '{render_camera_logical_id}' using ego mask "
            f"rig_config_id='{hood.rig_id}', camera_logical_id='{hood.camera_logical_id}' "
            f"(requested {requested_mask})."
        )

    def select_from_request(self, request: RGBRenderRequest | CameraRenderRequest) -> Optional[EgocarHood]:
        if not request.insert_ego_mask:
            self._log_mask_selection_once(request, None)
            return None

        if isinstance(request, CameraRenderRequest):
            products = list(request.render_products) or [CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR]
            if CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR not in products:
                return None

        if request.ego_mask_id.rig_config_id not in self._rigs:
            self._log_mask_selection_once(request, None)
            raise KeyError(f"No egocar rig found for config ID {request.ego_mask_id.rig_config_id=}")

        hood = self._rigs[request.ego_mask_id.rig_config_id].get(request.ego_mask_id.camera_logical_id)
        self._log_mask_selection_once(request, hood)
        if hood is None:
            raise KeyError(
                "No egocar hood found for "
                f"{request.ego_mask_id.rig_config_id=} and {request.ego_mask_id.camera_logical_id=}"
            )
        return hood
