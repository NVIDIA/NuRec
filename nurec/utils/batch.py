# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, List, Literal, Optional, Self, Sequence, Tuple, TypeAlias, TypeVar, Union, cast

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image as PILImage
import torch
import torch.nn.functional as F

from ncore.data import (
    ConcreteCameraModelParametersUnion,
    ConcreteLidarModelParametersUnion,
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
)
from ncore.impl.common.transformations import PoseInterpolator
from ncore.sensors import (
    CameraModel,
    FThetaCameraModel,
    LidarModel,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
    RowOffsetStructuredSpinningLidarModel,
    camera_model_from_parameters,
    maybe_lidar_model_from_parameters,
)
from nurec.libs.geometry.kernels.pose import se3pose_from_matrix
from nurec.libs.sensors.kernels.cameras import ShutterType, image_points_to_world_rays_shutter_pose
from nurec.libs.sensors.kernels.lidars import generate_spinning_lidar_rays
from nurec.utils.geometry import tquat_to_se3_matrix
from nurec.utils.misc import assert_same_type, collate_fn, to_torch, unpack_optional
from nurec.utils.prober import get_global_prober
from nurec.utils.profiling import ScopedTimer
from nurec.utils.sensors import (
    RectSubsampledSensor,
    SensorModelComputations,
)
from nurec.utils.sensors.ncore_sensors_converters import (
    CameraModelConverter,
    DynamicPose,
    LidarModelConverter,
    Pose,
)
from nurec.utils.types import (
    CuboidTracksDataPack,
    RayFlags,
    RigTrajectories,
)


_CAMERA_TIMESTAMP_LAYOUTS: dict[ShutterType, Literal["rows", "columns", "global"]] = {
    ShutterType.ROLLING_TOP_TO_BOTTOM: "rows",
    ShutterType.ROLLING_BOTTOM_TO_TOP: "rows",
    ShutterType.ROLLING_LEFT_TO_RIGHT: "columns",
    ShutterType.ROLLING_RIGHT_TO_LEFT: "columns",
    ShutterType.GLOBAL: "global",
}


ConcreteCameraModelsUnion: TypeAlias = FThetaCameraModel | OpenCVFisheyeCameraModel | OpenCVPinholeCameraModel
ConcreteLidarModelsUnion: TypeAlias = RowOffsetStructuredSpinningLidarModel
ConcreteSensorModelParametersUnion: TypeAlias = ConcreteCameraModelParametersUnion | ConcreteLidarModelParametersUnion


@dataclass(kw_only=True, slots=True)
class RectSubsampled(RectSubsampledSensor):
    """
    Subsampled rectangular pixel region with offset i/j and dimension height/width.

    Note that the offset i/j and dimension height/width are relative to the scaled pixel domain. I.e., subsampling is applied first, then cropping.

    The fields are:
    - original_width: The original width of the sensor. [int]
    - original_height: The original height of the sensor. [int]
    - width: The width of the pixel region. [int]
    - height: The height of the pixel region. [int]
    - i: Optional. The offset in the x-direction. Default is 0. [int]
    - j: Optional. The offset in the y-direction. Default is 0. [int]
    - subsample_factor: Optional. The amount of isotropic subsampling. The larger the value is, the smaller the sampled region will be. 1 means no subsampling. Default is 1. [float]
    """

    def to_json(self) -> dict:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "width": self.width,
            "height": self.height,
            "i": self.i,
            "j": self.j,
            "subsample_factor": self.subsample_factor,
        }

    def __hash__(self) -> int:
        return hash(
            (self.original_width, self.original_height, self.width, self.height, self.i, self.j, self.subsample_factor)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RectSubsampled):
            return False
        return self.to_json() == other.to_json()

    def pixel_coordinates(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Compute pixel center coordinates in the downsampled coordinate frame.

        Returns continuous pixel coordinates with 0.5 offset (pixel center convention),
        consistent with ncore's ``pixels_to_image_points()``. The coordinates are in the
        downsampled image space: for pixel index ``k``, the center is at ``k + 0.5``.

        Use together with `downsampled_resolution` to get the matching resolution.

        Returns:
            [height, width, 2] torch.Tensor[float32] with (x, y) pixel center coordinates.
        """
        ix = torch.arange(self.i, self.i + self.width, device=device, dtype=torch.float32)
        iy = torch.arange(self.j, self.j + self.height, device=device, dtype=torch.float32)

        # Pixel centers at +0.5 offset
        grid_x, grid_y = torch.meshgrid(ix + 0.5, iy + 0.5, indexing="xy")
        return torch.stack([grid_x, grid_y], dim=-1)

    def downsampled_resolution(self) -> tuple[int, int]:
        """Return the resolution of the downsampled image as (width, height).

        This is the resolution that matches the coordinates from `pixel_coordinates`.
        """
        w = round(self.original_width / self.subsample_factor)
        h = round(self.original_height / self.subsample_factor)
        return (w, h)

    def crop_array(self, arr: np.ndarray) -> np.ndarray:
        """Crop an array to the pixel region.

        Args:
            arr: The sensor array to crop. Should be a 3D array with shape (H-scaled, W-scaled, C) that is already subsampled.

        Returns:
            The cropped sensor array.
        """
        assert arr.ndim == 3, f"Array must be 3D shape, but got {arr.ndim}D of shape {arr.shape}"
        height, width, _ = arr.shape
        assert height * self.subsample_factor == self.original_height, (
            f"Unscaled height ({height}) does not match the original height ({self.original_height})"
        )
        assert width * self.subsample_factor == self.original_width, (
            f"Unscaled width ({width}) does not match the original width ({self.original_width})"
        )

        # crop
        box = (self.i, self.j, self.i + self.width, self.j + self.height)  # (left, top, right, bottom)
        if width < box[2]:
            raise ValueError(f"Image width {width} is smaller than the width of the pixel region {box[2]}")
        if height < box[3]:
            raise ValueError(f"Image height {height} is smaller than the height of the pixel region {box[3]}")
        return arr[box[1] : box[3], box[0] : box[2], :]

    def apply_to_array(self, arr: np.ndarray, interpolation: int | None, use_pil_lanczos: bool = False) -> np.ndarray:
        """Apply the subsampling (resizing and cropping) to a sensor array.

        Args:
            arr: The sensor array to apply the subsampling to. Should be a 3D array with shape (H, W, C)
            interpolation: The interpolation method to use. See `cv2.resize` for more details. Can be None if `use_pil_lanczos` is True.
            use_pil_lanczos: Whether to use PIL's LANCZOS instead of cv2.INTER_LANCZOS4. Default is False. Note enabling
              this will ignore the interpolation argument.

        Returns:
            The subsampled sensor array.
        """
        assert arr.ndim == 3, f"Array must be 3D shape, but got {arr.ndim}D of shape {arr.shape}"
        height, width, dim = arr.shape
        assert height == self.original_height, (
            f"Image height ({height}) does not match the original height ({self.original_height})"
        )
        assert width == self.original_width, (
            f"Image width ({width}) does not match the original width ({self.original_width})"
        )

        # resize
        if self.subsample_factor != 1.0:
            height = round(height / self.subsample_factor)
            width = round(width / self.subsample_factor)
            if use_pil_lanczos:
                pil_image = PILImage.fromarray(arr)
                pil_image = pil_image.resize((width, height), PILImage.Resampling.LANCZOS)
                arr = np.asarray(pil_image)
            else:
                assert interpolation is not None, "interpolation is required when use_pil_lanczos is False"
                arr = cv2.resize(arr, (width, height), interpolation=interpolation)
            # In the case of arr is (height, width, 1), cv2.resize will return (height, width),
            # so we need to reshape it back to (height, width, 1)
            arr = arr.reshape(height, width, dim)

        # crop
        return self.crop_array(arr)

    def apply_to_camera_model(self, sensor_model: CameraModel) -> CameraModel:
        """Apply the subsampling to the camera sensor model.

        Args:
            sensor_model: The model before subsampling.

        Returns:
            The camera model after subsampling.
        """
        assert self.original_width == int(sensor_model.resolution[0].item()), (
            "Image width does not match the original width"
        )
        assert self.original_height == int(sensor_model.resolution[1].item()), (
            "Image height does not match the original height"
        )

        sensor_model_parameters = cast(ConcreteCameraModelsUnion, sensor_model).get_parameters()
        ncore_camera_model_parameters_json = sensor_model_parameters.to_json()
        ncore_camera_model_parameter_type = sensor_model_parameters.type()

        # deserialize NCore camera model parameters
        ncore_camera_model_parameters: ConcreteCameraModelParametersUnion
        if ncore_camera_model_parameter_type == FThetaCameraModelParameters.type():
            ncore_camera_model_parameters = FThetaCameraModelParameters.from_json(ncore_camera_model_parameters_json)
        elif ncore_camera_model_parameter_type == OpenCVFisheyeCameraModelParameters.type():
            ncore_camera_model_parameters = OpenCVFisheyeCameraModelParameters.from_json(
                ncore_camera_model_parameters_json
            )
        elif ncore_camera_model_parameter_type == OpenCVPinholeCameraModelParameters.type():
            ncore_camera_model_parameters = OpenCVPinholeCameraModelParameters.from_json(
                ncore_camera_model_parameters_json
            )
        else:
            raise Exception(f"Unsupported camera model type: {ncore_camera_model_parameter_type}")

        transformed_parameters = ncore_camera_model_parameters.transform(
            image_domain_scale=1.0 / self.subsample_factor,
            image_domain_offset=(self.i, self.j),
            new_resolution=(self.width, self.height),
        )

        return camera_model_from_parameters(transformed_parameters, device=sensor_model.device, dtype=torch.float32)


@torch.autocast(device_type="cuda", enabled=False)
def compute_pixel_footprint(camera_rays: torch.Tensor | np.ndarray, pixel_offset: int = 10, mode: str = "cone"):
    """Computes the footprint of a pixel at unit length - footprint is defined as the diameter of the circle at unit length
    Since the computation is usually precision sensitive, we disable autocast to avoid mis-using 16-bit precision.
    Input:
        camera_rays - normalized rays in the camera coordinate system [h,w,3] (torch.Tensor | np.ndarray)
        pixel_offset - offset used to compute the solid angle. When the fov is small and the image resolution is high, small pixel offset can lead to problems with numerical precision
        mode - pixel footprint definition (cone - diameter of a cone carved by the pixel, plane_intersect - diameter of the intersection of the pixel cone with the plane parallel to the image )
    """

    assert pixel_offset != 0, "pixel_offset must not be zero to avoid division by zero in footprint calculation"

    is_input_numpy = isinstance(camera_rays, np.ndarray)
    if is_input_numpy:
        camera_rays = to_torch(cast(npt.NDArray, camera_rays), device="cpu", dtype=torch.float32)
    assert isinstance(camera_rays, torch.Tensor)

    # Convert to float32 on CUDA unconditionally to avoid mixed precision issues
    if camera_rays.device.type == "cuda":
        camera_rays = camera_rays.float()

    # Normalize camera rays to unit length
    camera_rays = F.normalize(camera_rays, dim=-1)

    if mode == "cone":
        # Bottom right corner
        dot_product = torch.einsum(
            "ijk, ijk -> ij", camera_rays[pixel_offset:, pixel_offset:], camera_rays[:-pixel_offset, :-pixel_offset]
        )
        # Clamp between [-1, 1] to avoid NaN in arccos.
        # These are due to numerical precision issues in F.normalize and torch.einsum
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        solid_angle_bottom_right = torch.arccos(
            F.pad(
                dot_product.unsqueeze(0),
                (0, pixel_offset, 0, pixel_offset),
                mode="replicate",
            ).squeeze(0)
        )

        # Top right corner
        dot_product = torch.einsum(
            "ijk, ijk -> ij",
            camera_rays[pixel_offset:, :-pixel_offset],
            camera_rays[:-pixel_offset, pixel_offset:],
        )
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        solid_angle_top_right = torch.arccos(
            F.pad(
                dot_product.unsqueeze(0),
                (0, pixel_offset, pixel_offset, 0),
                mode="replicate",
            ).squeeze(0)
        )

        solid_angle = torch.mean(
            torch.cat([solid_angle_bottom_right[:, :, None], solid_angle_top_right[:, :, None]], dim=-1), dim=-1
        )

        # Due to numerical precision issues, some angles might be zero.
        # Let's replace them with a small value to avoid zero footprint.
        # The footprint will end up influencing the gaussian scale parameter,
        # which uses the exp/log activation functions, which aren't defined at zero.
        # This avoids NaNs propagating in many places.
        solid_angle.masked_fill_(solid_angle == 0, 1e-12)

        # Divide the solid angle by the pixel offset and multiply by 2 to get the diameter
        footprint = 2 * torch.tan(0.5 * solid_angle) / pixel_offset

    elif mode == "plane_intersect":
        # Point where the rays intersect the plane at depth 1
        plane_intersection_points = camera_rays * (1 / camera_rays[:, :, 2:3])
        dist_on_plane_bottom_right = F.pad(
            torch.norm(
                plane_intersection_points[pixel_offset:, pixel_offset:]
                - plane_intersection_points[:-pixel_offset, :-pixel_offset],
                dim=-1,
            ).unsqueeze(0)
            / pixel_offset,
            (0, pixel_offset, 0, pixel_offset),
            mode="replicate",
        ).squeeze(0)
        dist_on_plane_top_right = F.pad(
            torch.norm(
                plane_intersection_points[pixel_offset:, :-pixel_offset]
                - plane_intersection_points[:-pixel_offset, pixel_offset:],
                dim=-1,
            ).unsqueeze(0)
            / pixel_offset,
            (0, pixel_offset, pixel_offset, 0),
            mode="replicate",
        ).squeeze(0)

        footprint = torch.mean(
            torch.cat([dist_on_plane_bottom_right[:, :, None], dist_on_plane_top_right[:, :, None]], dim=-1),
            dim=-1,
        )

    else:
        raise ValueError("Pixel footprint computation mode must be one of [cone, plane_intersect]")

    if is_input_numpy:
        return footprint.numpy()
    else:
        return footprint


def generate_grid_2d_indices(
    resolution: Tuple[int, int], order: Literal["xy", "yx"] = "xy", device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Computes pixel coordinates for all pixels in the sensor frame. [Applicable to both camera and lidar]

    Args:
        resolution: (w, h) A tuple containing the width and height of the sensor frame.
        order: The order of the returned coordinates. "xy" means (x, y) and "yx" means (y, x). Default is "xy".

    Returns:
        torch.Tensor: A 2D tensor of shape (N, 2) where N = width * height, containing
            (x, y) / (y, x) pixel coordinates for every pixel in the sensor frame. The coordinates
            are zero-based indices with x in [0, width-1] and y in [0, height-1].
    """
    w, h = resolution
    sensor_pixels_x, sensor_pixels_y = torch.meshgrid(
        torch.arange(w, dtype=torch.int16, device=device),
        torch.arange(h, dtype=torch.int16, device=device),
        indexing="xy",
    )  # [0, w-1] x [0, h-1]
    if order == "xy":
        return torch.stack([sensor_pixels_x.flatten(), sensor_pixels_y.flatten()], dim=1)
    elif order == "yx":
        return torch.stack([sensor_pixels_y.flatten(), sensor_pixels_x.flatten()], dim=1)
    else:
        raise ValueError(f"Invalid order: {order}")


@dataclass(kw_only=True, slots=True)
class UnstructuredLidarTilingData:
    """Plain-tensor payload of a gsplat ``UnstructuredLidarFrameContext``, built
    once in the dataloader worker (CPU) and carried through the batch so the
    frame-context construction (analytic projection + occupancy tiling +
    auto-resolution) is kept OUT of the render loop.

    Produced by ``UnstructuredLidarFrameContext.to_tensors()`` and consumed by
    ``UnstructuredLidarFrameContext.from_tensors()`` render-side (see the renderer;
    the worker's context object itself is discarded -- only these tensors + meta
    travel, and the render rebuilds a context from them at zero compute).

    The occupancy tiling is per-frame and its tensors are variable-length (N rays,
    n_tiles differ per frame), so this is only valid for a SINGLE-frame batch
    (the unstructured lidar render asserts B == 1). ``meta`` holds the scalar
    fields (FOV bounds, bin counts, periodicity, time-tiling) needed to rebuild.
    """

    directions: torch.Tensor  # [N, 3] sensor-frame ray directions (unit)
    az: torch.Tensor  # [N]
    el: torch.Tensor  # [N]
    relative_timestamps: torch.Tensor  # [N] relative frame time in [0, 1]
    occupancy: torch.Tensor  # [n_el, n_az] bool
    tiles_pack_info: torch.Tensor  # [n_tiles, 2] int32
    tiles_to_elements_map: torch.Tensor  # [N, 2] int32
    ray_index: torch.Tensor  # [N] int64
    time_mask: torch.Tensor  # [n_el*n_az] int32
    tile_mean_time: torch.Tensor  # [n_el*n_az] float32 per-cell mean rel time (-1 empty)
    meta: dict  # scalar metadata (frame_start/end_us, n_bins_*, az/el bounds, ...)

    _TENSOR_FIELDS = (
        "directions",
        "az",
        "el",
        "relative_timestamps",
        "occupancy",
        "tiles_pack_info",
        "tiles_to_elements_map",
        "ray_index",
        "time_mask",
        "tile_mean_time",
    )

    def to(self, *args, **kwargs) -> "UnstructuredLidarTilingData":
        return UnstructuredLidarTilingData(
            **{f: getattr(self, f).to(*args, **kwargs) for f in self._TENSOR_FIELDS},
            meta=self.meta,
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for f in self._TENSOR_FIELDS:
            t = getattr(self, f)
            if t.is_cuda:
                t.record_stream(stream)

    @classmethod
    def collate_fn(
        cls,
        seq: List["UnstructuredLidarTilingData"],
        device: torch.device = torch.device("cpu"),
    ) -> "UnstructuredLidarTilingData":
        # Per-frame variable-length tiling -> only single-frame batches supported
        # (matches the unstructured lidar render's B == 1 assertion).
        assert len(seq) == 1, "UnstructuredLidarTilingData supports only B == 1 batches"
        return seq[0].to(device)

    def __getitem__(self, item: Union[int, slice]) -> "UnstructuredLidarTilingData":
        # Single-frame payload: only the trivial B==1 slice/index is meaningful.
        return self


@dataclass(kw_only=True, slots=True)
class RenderingData:
    """Data for rendering. Shared between camera and lidar.

    Note: The `rays`, `poses_tquat_startend` and the underlying scene representation (e.g. 3D Gaussians)
    should be in the same coordinate system. For example, the most common case is that it carries `rays`
    in the NuRec space, the `poses_tquat_startend` is the transform from sensor to NuRec space, and the underlying
    scene representation (e.g. 3D Gaussians) is also in the NuRec space.

    The fields are:

    - rays: Ray origins and directions for the camera or lidar. [Tensor[float32]]. (B, height, width, 6).
    - sensor_model_parameters: List of model parameters for the camera or lidar if applicable. List of length B.
    - poses_tquat_startend: Start and end poses of the frame. [Tensor[float32]]. (B, 2, 7)
    - timestamps_startend_us: Start and end timestamps of the frame in microseconds. [Tensor[int64]]. (B, 2)
    - rays_timestamp_layout: Optional producer guarantee that timestamps are shared by each
        image row, column, or entire image. Unknown and reordered rays use None. Images
        may have different timestamp values, but must share the same layout in a batch.
    - ray_flags: Optional. Per-pixel RayFlags bitmask. [Tensor[int32]]. (B, height, width, 1). Populated from DataBatch labels.
    - active_tile_mask: Optional. Per-frame boolean mask of active tiles. [Tensor[bool]].
        Shape (B, tile_axis_0, tile_axis_1). Camera path: (B, tiles_y, tiles_x). Lidar path:
        elevation-major (B, n_bins_elevation, n_bins_azimuth), i.e.
        `CameraToLidarTileMapping.lidar_tile_grid_shape`.
        Can be provided externally; if None, the composite computes the camera mask from
        ray_flags (renderers do not self-compute).
    """

    rays: torch.Tensor
    sensor_model_parameters: list[ConcreteSensorModelParametersUnion | None]
    poses_tquat_startend: torch.Tensor  # (B, 2, 7)
    timestamps_startend_us: torch.Tensor  # (B, 2) - kept on GPU for GPU operations
    rays_timestamps_us: torch.Tensor | None = None  # (B, height, width, 1)
    # Producer guarantee for the row-major image grid; None means unknown/random.
    # Arbitrary spatial reordering must clear this metadata. Batch slicing is safe.
    rays_timestamp_layout: Literal["rows", "columns", "global"] | None = None
    _rays_footprints: torch.Tensor | None = None  # (B, height, width, 1)
    timestamps_startend_us_cpu: torch.Tensor  # (B, 2) - cpu copy to avoid .item() calls
    _distance_to_depth_scale: torch.Tensor | None = None  # [Tensor[float32]] (B, height, width, 1)
    _rays_is_sky: torch.Tensor | None = (
        None  # [Tensor[bool]] (B, height, width, 1) — sky rays mask for gradient detachment
    )
    ray_flags: torch.Tensor | None = None  # [Tensor[int32]] (B, height, width, 1) — RayFlags bitmask
    active_tile_mask: torch.Tensor | None = None  # [Tensor[bool]] (B, tile_axis_0, tile_axis_1)
    # Model-free unstructured LiDAR: prebuilt frame-context tiling tensors (built
    # in the dataloader worker; None for camera / structured lidar / no-model).
    unstructured_lidar_tiling: Optional[UnstructuredLidarTilingData] = None

    def __post_init__(self):
        B = self.rays.shape[0]
        assert self.rays.ndim == 4 and self.rays.shape[3] == 6, "Rays must be a 4D tensor (B, height, width, 6)"
        assert len(self.sensor_model_parameters) == B, "Model parameters must be a list of length B"
        assert self.poses_tquat_startend.shape == (B, 2, 7), "Poses must be a 3D tensor (B, 2, 7)"
        assert self.timestamps_startend_us.shape == (B, 2), "Timestamps must be a 2D tensor (B, 2)"
        assert self.timestamps_startend_us_cpu.shape == (B, 2), "CPU timestamps must be a 2D tensor (B, 2)"
        if self.rays_timestamps_us is not None:
            assert self.rays_timestamps_us.ndim == 4 and self.rays_timestamps_us.shape[3] == 1, (
                f"Rays timestamps must be a 4D tensor (B, height, width, 1), but got {self.rays_timestamps_us.shape}"
            )
        if self._rays_footprints is not None:
            assert self._rays_footprints.ndim == 4 and self._rays_footprints.shape[3] == 1, (
                f"Rays footprints must be a 4D tensor (B, height, width, 1), but got {self._rays_footprints.shape}"
            )
        if self._distance_to_depth_scale is not None:
            assert (
                self._distance_to_depth_scale.ndim == 4
                and self._distance_to_depth_scale.shape[:3] == self.rays.shape[:3]
            ), (
                f"Depth to distance scale must be a 4D tensor (B, height, width, 1) and match the rays shape (B, height, width, 6), but got {self._distance_to_depth_scale.shape}"
            )
        if self.ray_flags is not None:
            assert (
                self.ray_flags.ndim == 4
                and self.ray_flags.shape[:3] == self.rays.shape[:3]
                and self.ray_flags.shape[3] == 1
            ), (
                f"ray_flags must be a 4D tensor (B, height, width, 1) matching rays shape, but got {self.ray_flags.shape}"
            )
        if self.active_tile_mask is not None:
            assert (
                self.active_tile_mask.ndim == 3
                and self.active_tile_mask.shape[0] == B
                and self.active_tile_mask.dtype == torch.bool
            ), (
                f"active_tile_mask must be a 3D bool tensor (B, tile_axis_0, tile_axis_1) with B={B}, but got "
                f"shape {self.active_tile_mask.shape} dtype {self.active_tile_mask.dtype}"
            )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self.rays.record_stream(stream)
        # skip self.sensor_model_parameters. Assumption: it doesn't contain any tensors.
        self.poses_tquat_startend.record_stream(stream)
        self.timestamps_startend_us.record_stream(stream)
        if self.rays_timestamps_us is not None:
            self.rays_timestamps_us.record_stream(stream)
        if self._rays_footprints is not None:
            self._rays_footprints.record_stream(stream)
        if self._rays_is_sky is not None:
            self._rays_is_sky.record_stream(stream)
        if self.ray_flags is not None:
            self.ray_flags.record_stream(stream)
        if self.active_tile_mask is not None:
            self.active_tile_mask.record_stream(stream)
        if self.unstructured_lidar_tiling is not None:
            self.unstructured_lidar_tiling.record_stream(stream)

    @property
    def b(self) -> int:
        return self.rays.shape[0]

    @property
    def h(self) -> int:
        return self.rays.shape[1]

    @property
    def w(self) -> int:
        return self.rays.shape[2]

    @property
    @torch.autocast(device_type="cuda", enabled=False)
    def distance_to_depth_scale(self) -> torch.Tensor:
        """Compute the multiplication factor to convert depth to distance.
        Shape: (B, height, width, 1)"""
        if self._distance_to_depth_scale is None:
            scales: list[torch.Tensor] = []
            for bidx in range(self.b):
                sensor_model = camera_model_from_parameters(
                    cast(ConcreteCameraModelParametersUnion, self.sensor_model_parameters[bidx]),
                    device=self.rays.device,
                )
                width, height = sensor_model.resolution.tolist()
                elements = generate_grid_2d_indices((width, height), device=self.rays.device)
                sensor_rays = sensor_model.pixels_to_camera_rays(elements).reshape(height, width, 3)
                scales.append(sensor_rays[..., 2].reshape(height, width))
            self._distance_to_depth_scale = torch.stack(scales, dim=0).unsqueeze(-1)
        return self._distance_to_depth_scale

    @property
    def ray_footprints(self) -> torch.Tensor:
        assert self._rays_footprints is not None, "Rays footprints must be set"
        return self._rays_footprints

    @property
    @torch.autocast(device_type="cuda", enabled=False)
    def uv_directions_frame_end(self) -> torch.Tensor:
        """Compute the UV directions for the pixels in the frame.
        Shape: (B, 2, 3), here the 2nd dimension is U and V directions, normalized"""
        frame_end_se3 = tquat_to_se3_matrix(self.poses_tquat_startend[:, 1], unbatch=False)  # (B, 4, 4)
        return frame_end_se3[:, :3, :2].transpose(1, 2)

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        if any(item.rays_timestamps_us is None for item in seq):
            rays_timestamps_us = None
        else:
            rays_timestamps_us = collate_fn([item.rays_timestamps_us for item in seq], device)
        if any(item._rays_footprints is None for item in seq):
            _rays_footprints = None
        else:
            _rays_footprints = collate_fn([item._rays_footprints for item in seq], device)
        # Keep GPU version on target device, CPU copy on CPU
        timestamps_startend_us = collate_fn([item.timestamps_startend_us for item in seq], device)
        timestamps_startend_us_cpu = collate_fn([item.timestamps_startend_us_cpu for item in seq], torch.device("cpu"))
        if any(item._distance_to_depth_scale is None for item in seq):
            _distance_to_depth_scale = None
        else:
            _distance_to_depth_scale = collate_fn([item._distance_to_depth_scale for item in seq], device)
        if any(item._rays_is_sky is None for item in seq):
            _rays_is_sky = None
        else:
            _rays_is_sky = collate_fn([item._rays_is_sky for item in seq], device)
        if any(item.ray_flags is None for item in seq):
            ray_flags = None
        else:
            ray_flags = collate_fn([item.ray_flags for item in seq], device)
        if any(item.active_tile_mask is None for item in seq):
            active_tile_mask = None
        else:
            active_tile_mask = collate_fn([item.active_tile_mask for item in seq], device)
        if any(item.unstructured_lidar_tiling is None for item in seq):
            unstructured_lidar_tiling = None
        else:
            unstructured_lidar_tiling = UnstructuredLidarTilingData.collate_fn(
                [cast(UnstructuredLidarTilingData, item.unstructured_lidar_tiling) for item in seq], device
            )
        return cls(
            rays=collate_fn([item.rays for item in seq], device),
            sensor_model_parameters=[p for item in seq for p in item.sensor_model_parameters],
            poses_tquat_startend=collate_fn([item.poses_tquat_startend for item in seq], device),
            timestamps_startend_us=timestamps_startend_us,
            rays_timestamps_us=rays_timestamps_us,
            rays_timestamp_layout=seq[0].rays_timestamp_layout
            if all(item.rays_timestamp_layout == seq[0].rays_timestamp_layout for item in seq)
            else None,
            _rays_footprints=_rays_footprints,
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
            _distance_to_depth_scale=_distance_to_depth_scale,
            _rays_is_sky=_rays_is_sky,
            ray_flags=ray_flags,
            active_tile_mask=active_tile_mask,
            unstructured_lidar_tiling=unstructured_lidar_tiling,
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            rays=self.rays.to(*args, **kwargs),
            sensor_model_parameters=self.sensor_model_parameters,
            rays_timestamp_layout=self.rays_timestamp_layout,
            poses_tquat_startend=self.poses_tquat_startend.to(*args, **kwargs),
            timestamps_startend_us=self.timestamps_startend_us.to(*args, **kwargs),
            rays_timestamps_us=self.rays_timestamps_us.to(*args, **kwargs)
            if self.rays_timestamps_us is not None
            else None,
            _rays_footprints=self._rays_footprints.to(*args, **kwargs) if self._rays_footprints is not None else None,
            # CPU copy stays on CPU - don't move it
            timestamps_startend_us_cpu=self.timestamps_startend_us_cpu,
            _distance_to_depth_scale=self._distance_to_depth_scale.to(*args, **kwargs)
            if self._distance_to_depth_scale is not None
            else None,
            _rays_is_sky=self._rays_is_sky.to(*args, **kwargs) if self._rays_is_sky is not None else None,
            ray_flags=self.ray_flags.to(*args, **kwargs) if self.ray_flags is not None else None,
            active_tile_mask=self.active_tile_mask.to(*args, **kwargs) if self.active_tile_mask is not None else None,
            unstructured_lidar_tiling=self.unstructured_lidar_tiling.to(*args, **kwargs)
            if self.unstructured_lidar_tiling is not None
            else None,
        )

    def __getitem__(self, item: Union[int, slice]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            rays=self.rays[item],
            sensor_model_parameters=self.sensor_model_parameters[item],
            rays_timestamp_layout=self.rays_timestamp_layout,
            poses_tquat_startend=self.poses_tquat_startend[item],
            timestamps_startend_us=self.timestamps_startend_us[item],
            rays_timestamps_us=self.rays_timestamps_us[item] if self.rays_timestamps_us is not None else None,
            _rays_footprints=self._rays_footprints[item] if self._rays_footprints is not None else None,
            timestamps_startend_us_cpu=self.timestamps_startend_us_cpu[item],
            _distance_to_depth_scale=self._distance_to_depth_scale[item]
            if self._distance_to_depth_scale is not None
            else None,
            _rays_is_sky=self._rays_is_sky[item] if self._rays_is_sky is not None else None,
            ray_flags=self.ray_flags[item] if self.ray_flags is not None else None,
            active_tile_mask=self.active_tile_mask[item] if self.active_tile_mask is not None else None,
            unstructured_lidar_tiling=self.unstructured_lidar_tiling[item]
            if self.unstructured_lidar_tiling is not None
            else None,
        )


@dataclass(kw_only=True, slots=True)
class FrameMeta:
    """Metadata for a frame. Shared between camera and lidar.

    The fields are:
    - unique_sensor_idx: Index of the sensor that captured the frame. int32
    - unique_frame_idx: 0-based linear frame index over *all* frames of one sensor
        type (cameras or lidars), training and holdout combined. Used by
        CameraFreePoseViewGeometry for pose lookup and calibration. int32
    - training_frame_idx: 0-based linear frame index over *training* frames only
        of one sensor type, or -1 for holdout / novel-view frames. The training-
        frame space is contiguous (no gaps for excluded holdout frames), so
        per-frame parameter tensors in PPISP and bilateral grids can be tightly
        sized to N_training_frames. When no holdout sampler is used,
        training_frame_idx equals unique_frame_idx for every frame. int32
    - logical_sensor_id: Optional stable dataset sensor name. Unlike
        ``unique_sensor_idx``, this preserves the semantic camera identity needed
        by name-keyed model components (for example VGGT's camera embeddings).
    - subsample: Subsampled pixel region (crop and resize info). Default to None.
    """

    unique_sensor_idx: int
    unique_frame_idx: int
    training_frame_idx: int
    logical_sensor_id: str | None = None
    subsample: RectSubsampled | None = None
    # [TODO]: temporary solution to align with the current design, should be improved in the future
    T_offset_nurec_startend: torch.Tensor | None = None  # (2, 4, 4) offset in the NuRec frame or None

    # Tensor version that will be used for cuda, populated automatically
    unique_frame_idx_tensor: torch.Tensor | None = None

    # Pre-built 2-element tensor [unique_sensor_idx, unique_frame_idx] used by
    # renderers to avoid a fresh `torch.tensor([...], device=cuda)` allocation
    # on every render call (not permitted during `torch.cuda.graph` capture).
    unique_sensor_frame_idx_tensor: torch.Tensor | None = None

    # Stringified version of the unique sensor index, used to index sensor_models as a nn.ModuleDict.
    unique_sensor_idx_str: str | None = None

    def __post_init__(self):
        if self.unique_frame_idx_tensor is None:
            self.unique_frame_idx_tensor = (
                torch.tensor([self.unique_frame_idx], dtype=torch.int32) if self.unique_frame_idx != -1 else None
            )
        if self.unique_sensor_frame_idx_tensor is None:
            # Always populated (no `unique_frame_idx == -1` sentinel gating):
            # downstream consumers in `renderers.py` and `gsplat_renderer.py` read
            # this field unconditionally and pass it to gsplat kernels
            # that expect an int32 `[sensor_idx, frame_idx]` tensor — the same
            # `[sensor_idx, -1]` shape the pre-MR call sites unconditionally
            # constructed. Does NOT mirror the sibling `unique_frame_idx_tensor`
            # gating: that field's consumers (e.g. `nurec/utils/sensors/sensors.py`)
            # explicitly handle `None`; this field's consumers don't.
            self.unique_sensor_frame_idx_tensor = torch.tensor(
                [self.unique_sensor_idx, self.unique_frame_idx], dtype=torch.int32
            )
        if self.unique_sensor_idx_str is None:
            self.unique_sensor_idx_str = str(self.unique_sensor_idx)
        if self.T_offset_nurec_startend is not None:
            assert self.T_offset_nurec_startend.shape == (2, 4, 4), (
                "T_offset_nurec_startend must be a 3D tensor (2, 4, 4)"
            )
            assert self.T_offset_nurec_startend.dtype == torch.float32, (
                "T_offset_nurec_startend must be a float32 tensor"
            )

    @property
    def h(self) -> int | None:
        if self.subsample is None:
            return None
        else:
            return self.subsample.height

    @property
    def w(self) -> int | None:
        if self.subsample is None:
            return None
        else:
            return self.subsample.width

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> List[Self]:
        return [s.to(device) for s in seq]

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            unique_sensor_idx=self.unique_sensor_idx,
            unique_frame_idx=self.unique_frame_idx,
            training_frame_idx=self.training_frame_idx,
            logical_sensor_id=self.logical_sensor_id,
            subsample=self.subsample.to(*args, **kwargs) if self.subsample is not None else None,
            T_offset_nurec_startend=self.T_offset_nurec_startend.to(*args, **kwargs)
            if self.T_offset_nurec_startend is not None
            else None,
            unique_frame_idx_tensor=self.unique_frame_idx_tensor.to(*args, **kwargs)
            if self.unique_frame_idx_tensor is not None
            else None,
            unique_sensor_frame_idx_tensor=self.unique_sensor_frame_idx_tensor.to(*args, **kwargs)
            if self.unique_sensor_frame_idx_tensor is not None
            else None,
            unique_sensor_idx_str=self.unique_sensor_idx_str,
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        if self.T_offset_nurec_startend is not None:
            self.T_offset_nurec_startend.record_stream(stream)
        if self.unique_frame_idx_tensor is not None:
            self.unique_frame_idx_tensor.record_stream(stream)
        if self.unique_sensor_frame_idx_tensor is not None:
            self.unique_sensor_frame_idx_tensor.record_stream(stream)


@dataclass(kw_only=True, slots=True, frozen=True)
class PoseCorrection:
    """A correction applied to sensor start/end poses before rays are generated.

    `modifies_poses` must be False only when `apply` is numerically the identity. Consumers
    holding rays precomputed in world space from the uncorrected poses (`use_real_lidar_rays`)
    cannot follow a pose change, so they use this to tell a real correction from a
    pass-through instead of assuming any supplied callback calibrates.
    """

    apply: Callable[[torch.Tensor, torch.Tensor, FrameMeta], torch.Tensor]
    modifies_poses: bool

    def __call__(
        self, poses_startend: torch.Tensor, timestamps_startend_us: torch.Tensor, meta: FrameMeta
    ) -> torch.Tensor:
        return self.apply(poses_startend, timestamps_startend_us, meta)


@dataclass(kw_only=True, slots=True)
class CameraFrameLabels:
    """Labels for a camera frame.

    The fields are:
    - flags: Optional. Bitmask integer value (see RayFlags). Default is None. [Tensor[int32]]. (B, height, width, 1).
    - rgb: Optional. RGB value within [0, 1]. Default is None. [Tensor[float32]]. (B, height, width, 3).
    - distance: Optional. Metric ray-depth in NuRec scale (not z-depth). Default is None. [Tensor[float32]]. (B, height, width, 1).
    - metric_distance: Optional. Non-metric ray depth from a depth estimation model. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - relative_distance: Optional. Non-metric ray depth from a depth estimation model. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - alpha: Optional. Transparency of the pixel valued in [0, 1]. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - semantic: Optional. Semantic class of the pixel. Default is None. [Tensor[uint8]]. (B, height, width, 1).
    - normals: Optional. Per-pixel normal vectors relative to the world frame. Default is None. [Tensor[float32]]. (B, height, width, 3).
    - velocity: Optional. Velocity vector of the pixel (unit is m/s). Default is None. [Tensor[float32]]. (B, height, width, 3/6).
    - _n_valid_rgb: Optional. Number of valid RGB channels as cache for n_valid_rgb property. Default is None. int32.
    - _n_valid: Optional. Number of valid pixels as cache for n_valid property. Default is None. int32.
    - _n_harmonized: Optional. Number of Harmonized as cache for n_harmonized property. Default is None. int32.
    - _n_valid_bg: Optional. Number of valid pixels for background loss (excludes INVALID, HARMONIZED, SYNTHETIC). Default is None. int32.
    """

    flags: torch.Tensor | None = None
    rgb: torch.Tensor | None = None
    metric_distance: torch.Tensor | None = None
    relative_distance: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    semantic: torch.Tensor | None = None
    normals: torch.Tensor | None = None
    velocity: torch.Tensor | None = None
    _n_valid_rgb: int | None = None
    _n_valid: int | None = None
    _n_harmonized: int | None = None
    _n_valid_bg: int | None = None
    # TODO: add other labels from ExtraSignals

    def __post_init__(self):
        if self.flags is not None:
            assert self.flags.ndim == 4 and self.flags.shape[3] == 1, "Flags must be a 4D tensor (B, height, width, 1)"
            assert self.flags.dtype == torch.int32, "Flags must be a int32 tensor"
        if self.rgb is not None:
            assert self.rgb.ndim == 4 and self.rgb.shape[3] == 3, "RGB must be a 4D tensor (B, height, width, 3)"
            assert self.rgb.dtype == torch.float32, "RGB must be a float32 tensor"
        if self.metric_distance is not None:
            assert self.metric_distance.ndim == 4 and self.metric_distance.shape[3] == 1, (
                "Metric distance must be a 4D tensor (B, height, width, 1)"
            )
            assert self.metric_distance.dtype == torch.float32, "Metric distance must be a float32 tensor"
        if self.relative_distance is not None:
            assert self.relative_distance.ndim == 4 and self.relative_distance.shape[3] == 1, (
                "Relative distance must be a 4D tensor (B, height, width, 1)"
            )
            assert self.relative_distance.dtype == torch.float32, "Relative distance must be a float32 tensor"
        if self.alpha is not None:
            assert self.alpha.ndim == 4 and self.alpha.shape[3] == 1, "Alpha must be a 4D tensor (B, height, width, 1)"
            assert self.alpha.dtype == torch.float32, "Alpha must be a float32 tensor"
        if self.semantic is not None:
            assert self.semantic.ndim == 4 and self.semantic.shape[3] == 1, (
                "Semantic must be a 4D tensor (B, height, width, 1)"
            )
            assert self.semantic.dtype == torch.uint8, "Semantic must be a uint8 tensor"
        if self.normals is not None:
            assert self.normals.ndim == 4 and self.normals.shape[3] == 3, (
                "Normals must be a 4D tensor (B, height, width, 3)"
            )
            assert self.normals.dtype == torch.float32, "Normals must be a float32 tensor"
        if self.velocity is not None:
            assert self.velocity.ndim == 4 and self.velocity.shape[3] in [
                3,
                6,
            ], "Velocity must be a 4D tensor (B, height, width, 3/6)"
            assert self.velocity.dtype == torch.float32, "Velocity must be a float32 tensor"
        if self._n_valid_rgb is not None:
            assert isinstance(self._n_valid_rgb, int), "N valid RGB must be an int"
        if self._n_valid is not None:
            assert isinstance(self._n_valid, int), "N valid must be an int"
        if self._n_harmonized is not None:
            assert isinstance(self._n_harmonized, int), "N Harmonized must be an int"
        if self._n_valid_bg is not None:
            assert isinstance(self._n_valid_bg, int), "N valid bg must be an int"

    @property
    def n_valid_rgb(self) -> int:
        if self._n_valid_rgb is None:
            rgb_valid_mask = self.get_mask_flags_all(RayFlags.RGB_LABEL) & self.get_mask_flags_none(RayFlags.INVALID)
            self._n_valid_rgb = int(rgb_valid_mask.sum().item()) * 3
        return self._n_valid_rgb

    @property
    def n_valid(self) -> int:
        if self._n_valid is None:
            valid_mask = self.get_mask_flags_none(RayFlags.INVALID)
            self._n_valid = int(valid_mask.sum().item())
        return self._n_valid

    @property
    def n_harmonized(self) -> int:
        if self._n_harmonized is None:
            harmonized_mask = self.get_mask_flags_all(RayFlags.HARMONIZED)
            self._n_harmonized = int(harmonized_mask.sum().item())
        return self._n_harmonized

    @property
    def n_valid_bg(self) -> int:
        """Number of valid pixels for background loss (excludes INVALID, HARMONIZED, SYNTHETIC)."""
        if self._n_valid_bg is None:
            valid_mask = (
                self.get_mask_flags_none(RayFlags.INVALID)
                & self.get_mask_flags_none(RayFlags.HARMONIZED)
                & self.get_mask_flags_none(RayFlags.SYNTHETIC)
            )
            self._n_valid_bg = int(valid_mask.sum().item())
        return self._n_valid_bg

    @property
    def b(self) -> int | None:
        batch_sizes = [
            getattr(self, attr).shape[0]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(batch_sizes) == 0:
            return None
        else:
            assert all(b == batch_sizes[0] for b in batch_sizes), "All batch sizes must be the same"
            return batch_sizes[0]

    @property
    def h(self) -> int | None:
        heights = [
            getattr(self, attr).shape[1]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(heights) == 0:
            return None
        else:
            assert all(h == heights[0] for h in heights), "All heights must be the same"
            return heights[0]

    @property
    def w(self) -> int | None:
        widths = [
            getattr(self, attr).shape[2]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(widths) == 0:
            return None
        else:
            assert all(w == widths[0] for w in widths), "All widths must be the same"
            return widths[0]

    def get_mask_flags_all(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *all* flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).eq(flags.value)

    def get_mask_flags_any(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *any* flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).ne(0)

    def get_mask_flags_none(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *none* of the flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).eq(0)

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        # For metric distance, if any is not None, we set the others to zeros
        metric_distance_seq = [item.metric_distance for item in seq]
        if (
            first_not_none_distance := next(
                (distance for distance in metric_distance_seq if distance is not None), None
            )
        ) is not None:
            metric_distance_seq = [
                torch.zeros_like(first_not_none_distance) if distance is None else distance
                for distance in metric_distance_seq
            ]

        return cls(
            flags=collate_fn([item.flags for item in seq], device),
            rgb=collate_fn([item.rgb for item in seq], device),
            metric_distance=collate_fn(metric_distance_seq, device),
            relative_distance=collate_fn([item.relative_distance for item in seq], device),
            alpha=collate_fn([item.alpha for item in seq], device),
            semantic=collate_fn([item.semantic for item in seq], device),
            normals=collate_fn([item.normals for item in seq], device),
            velocity=collate_fn([item.velocity for item in seq], device),
            _n_valid_rgb=sum([item.n_valid_rgb for item in seq]),
            _n_valid=sum([item.n_valid for item in seq]),
            _n_harmonized=sum([item.n_harmonized for item in seq]),
            _n_valid_bg=sum([item.n_valid_bg for item in seq]),
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            flags=self.flags.to(*args, **kwargs) if self.flags is not None else None,
            rgb=self.rgb.to(*args, **kwargs) if self.rgb is not None else None,
            metric_distance=self.metric_distance.to(*args, **kwargs) if self.metric_distance is not None else None,
            relative_distance=self.relative_distance.to(*args, **kwargs)
            if self.relative_distance is not None
            else None,
            alpha=self.alpha.to(*args, **kwargs) if self.alpha is not None else None,
            semantic=self.semantic.to(*args, **kwargs) if self.semantic is not None else None,
            normals=self.normals.to(*args, **kwargs) if self.normals is not None else None,
            velocity=self.velocity.to(*args, **kwargs) if self.velocity is not None else None,
            _n_valid_rgb=self._n_valid_rgb,
            _n_valid=self._n_valid,
            _n_harmonized=self._n_harmonized,
            _n_valid_bg=self._n_valid_bg,
        )

    def __getitem__(self, item: Union[int, slice, torch.Tensor]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            flags=self.flags[item] if self.flags is not None else None,
            rgb=self.rgb[item] if self.rgb is not None else None,
            metric_distance=self.metric_distance[item] if self.metric_distance is not None else None,
            relative_distance=self.relative_distance[item] if self.relative_distance is not None else None,
            alpha=self.alpha[item] if self.alpha is not None else None,
            semantic=self.semantic[item] if self.semantic is not None else None,
            normals=self.normals[item] if self.normals is not None else None,
            velocity=self.velocity[item] if self.velocity is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class LidarFrameLabels:
    """Labels for a lidar frame.

    The fields are:
    - flags: Optional. Bitmask integer value (see RayFlags). Default is None. [Tensor[int32]]. (B, height, width, 1).
    - distance: Optional. Metric ray-depth in NuRec scale (not z-depth). Default is None. [Tensor[float32]]. (B, height, width, 1).
    - intensity: Optional. Intensity of the lidar response. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - raydrop: Optional. The possiblity that the ray should be dropped. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - _n_valid_lidar: Optional. The number of valid lidar rays as cache for n_valid_lidar property. Default is None. int32.
    - _n_valid_raydrop: Optional. The number of raydrop-valid rays (INVALID-none) as cache for n_valid_raydrop property. Default is None. int32.
    - sparse_rays: Optional. The rays from the lidar points. Default is None. [Tensor[float32]]. (B, n_sparse_rays, 6).
    - sparse_timestamps: Optional. The timestamps of the lidar points. Default is None. [Tensor[int64]]. (B, n_sparse_rays, 1).
    - sparse_elements: Optional. The elements (row, col) of the lidar points. Default is None. [Tensor[int64]]. (B, n_sparse_rays, 2).
    """

    flags: torch.Tensor | None = None
    distance: torch.Tensor | None = None
    intensity: torch.Tensor | None = None
    raydrop: torch.Tensor | None = None
    _n_valid_lidar: int | None = None
    _n_valid_raydrop: int | None = None
    # TODO: add other labels from ExtraSignals

    # To support cases where rays are computed from the lidar points.
    sparse_rays: torch.Tensor | None = None
    sparse_timestamps: torch.Tensor | None = None
    sparse_elements: torch.Tensor | None = None

    # Ray delivery mode this frame's labels were built for: "structured"
    # (sensor-model (n_rows, n_columns) range-view grid) or "unstructured"
    # (model-independent (N, 1) return list). Set explicitly by the dataset; read by
    # LidarFreePoseViewGeometry.to_rendering_data to pick the ray-construction path.
    # It is intentionally NOT inferred from sparse_* presence, because
    # structured + use_real_lidar_rays also carries sparse_* (to override the grid's
    # returning cells with measured geometry).
    ray_mode: Literal["structured", "unstructured"] = "structured"

    # Model-free unstructured LiDAR: frame-context tiling built in the dataloader
    # worker (CPU) so its construction is off the render loop. None for structured
    # / camera / datasources without a spinning lidar model. B == 1 only.
    unstructured_lidar_tiling: Optional["UnstructuredLidarTilingData"] = None

    def __post_init__(self):
        if self.flags is not None:
            assert self.flags.ndim == 4 and self.flags.shape[3] == 1, "Flags must be a 4D tensor (B, height, width, 1)"
            assert self.flags.dtype == torch.int32, "Flags must be a int32 tensor"
        if self.distance is not None:
            assert self.distance.ndim == 4 and self.distance.shape[3] == 1, (
                "Distance must be a 4D tensor (B, height, width, 1)"
            )
            assert self.distance.dtype == torch.float32, "Distance must be a float32 tensor"
        if self.intensity is not None:
            assert self.intensity.ndim == 4 and self.intensity.shape[3] == 1, (
                "Intensity must be a 4D tensor (B, height, width, 1)"
            )
            assert self.intensity.dtype == torch.float32, "Intensity must be a float32 tensor"
        if self.raydrop is not None:
            assert self.raydrop.ndim == 4 and self.raydrop.shape[3] == 1, (
                "Raydrop must be a 4D tensor (B, height, width, 1)"
            )
            assert self.raydrop.dtype == torch.float32, "Raydrop must be a float32 tensor"
        if self._n_valid_lidar is not None:
            assert isinstance(self._n_valid_lidar, int), "N valid must be an int"
        if self._n_valid_raydrop is not None:
            assert isinstance(self._n_valid_raydrop, int), "N valid raydrop must be an int"
        if self.sparse_rays is not None:
            assert self.sparse_rays.ndim == 3 and self.sparse_rays.shape[2] == 6, (
                "Rays must be a 4D tensor (B, n_sparse_rays, 6)"
            )
            assert self.sparse_rays.dtype == torch.float32, "Rays must be a float32 tensor"
        if self.sparse_timestamps is not None:
            assert self.sparse_timestamps.ndim == 3 and self.sparse_timestamps.shape[2] == 1, (
                "Timestamps must be a 4D tensor (B, n_sparse_rays, 1)"
            )
            assert self.sparse_timestamps.dtype == torch.int64, "Timestamps must be a int64 tensor"
        if self.sparse_elements is not None:
            assert self.sparse_elements.ndim == 3 and self.sparse_elements.shape[2] == 2, (
                "Elements must be a 4D tensor (B, n_sparse_rays, 2)"
            )
            assert self.sparse_elements.dtype == torch.int64, "Elements must be a int64 tensor"

    @property
    def n_valid_lidar(self) -> int:
        if self._n_valid_lidar is None:
            valid_lidar_mask = self.get_mask_flags_none(RayFlags.INVALID) & self.get_mask_flags_none(RayFlags.DROPPED)
            self._n_valid_lidar = int(valid_lidar_mask.sum().item())
        return self._n_valid_lidar

    @property
    def n_valid_raydrop(self) -> int:
        """Number of rays valid for the raydrop loss (INVALID-none; DROPPED allowed)."""
        if self._n_valid_raydrop is None:
            raydrop_valid_mask = self.get_mask_flags_none(RayFlags.INVALID)
            self._n_valid_raydrop = int(raydrop_valid_mask.sum().item())
        return self._n_valid_raydrop

    @property
    def b(self) -> int | None:
        batch_sizes = [
            getattr(self, attr).shape[0]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(batch_sizes) == 0:
            return None
        else:
            assert all(b == batch_sizes[0] for b in batch_sizes), "All batch sizes must be the same"
            return batch_sizes[0]

    @property
    def h(self) -> int | None:
        heights = [
            getattr(self, attr).shape[1]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(heights) == 0:
            return None
        else:
            assert all(h == heights[0] for h in heights), "All heights must be the same"
            return heights[0]

    @property
    def w(self) -> int | None:
        widths = [
            getattr(self, attr).shape[2]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(widths) == 0:
            return None
        else:
            assert all(w == widths[0] for w in widths), "All widths must be the same"
            return widths[0]

    def get_mask_flags_all(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *all* flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).eq(flags.value)

    def get_mask_flags_any(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *any* flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).ne(0)

    def get_mask_flags_none(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *none* of the flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).eq(0)

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        return cls(
            flags=collate_fn([item.flags for item in seq], device),
            distance=collate_fn([item.distance for item in seq], device),
            intensity=collate_fn([item.intensity for item in seq], device),
            raydrop=collate_fn([item.raydrop for item in seq], device),
            _n_valid_lidar=sum([item.n_valid_lidar for item in seq]),
            _n_valid_raydrop=sum([item.n_valid_raydrop for item in seq]),
            sparse_rays=collate_fn([item.sparse_rays for item in seq], device),
            sparse_timestamps=collate_fn([item.sparse_timestamps for item in seq], device),
            sparse_elements=collate_fn([item.sparse_elements for item in seq], device),
            ray_mode=seq[0].ray_mode,
            unstructured_lidar_tiling=(
                UnstructuredLidarTilingData.collate_fn(
                    [cast(UnstructuredLidarTilingData, item.unstructured_lidar_tiling) for item in seq], device
                )
                if all(item.unstructured_lidar_tiling is not None for item in seq)
                else None
            ),
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            flags=self.flags.to(*args, **kwargs) if self.flags is not None else None,
            distance=self.distance.to(*args, **kwargs) if self.distance is not None else None,
            intensity=self.intensity.to(*args, **kwargs) if self.intensity is not None else None,
            raydrop=self.raydrop.to(*args, **kwargs) if self.raydrop is not None else None,
            _n_valid_lidar=self._n_valid_lidar,
            _n_valid_raydrop=self._n_valid_raydrop,
            sparse_rays=self.sparse_rays.to(*args, **kwargs) if self.sparse_rays is not None else None,
            sparse_timestamps=self.sparse_timestamps.to(*args, **kwargs)
            if self.sparse_timestamps is not None
            else None,
            sparse_elements=self.sparse_elements.to(*args, **kwargs) if self.sparse_elements is not None else None,
            ray_mode=self.ray_mode,
            unstructured_lidar_tiling=self.unstructured_lidar_tiling.to(*args, **kwargs)
            if self.unstructured_lidar_tiling is not None
            else None,
        )

    def __getitem__(self, item: Union[int, slice, torch.Tensor]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            flags=self.flags[item] if self.flags is not None else None,
            distance=self.distance[item] if self.distance is not None else None,
            intensity=self.intensity[item] if self.intensity is not None else None,
            raydrop=self.raydrop[item] if self.raydrop is not None else None,
            sparse_rays=self.sparse_rays[item] if self.sparse_rays is not None else None,
            sparse_timestamps=self.sparse_timestamps[item] if self.sparse_timestamps is not None else None,
            sparse_elements=self.sparse_elements[item] if self.sparse_elements is not None else None,
            ray_mode=self.ray_mode,
            unstructured_lidar_tiling=self.unstructured_lidar_tiling,
        )


@dataclass(kw_only=True, slots=True)
class DataBatch:
    """Data for both camera and lidar frames."""

    @dataclass(kw_only=True, slots=True)
    class Camera:
        """Data for a camera frame. Includes the frame meta and labels."""

        meta: List[FrameMeta]
        labels: CameraFrameLabels

        @property
        def b(self) -> int:
            return len(self.meta)

        @property
        def h(self) -> int | None:
            heights = [meta.h for meta in self.meta] + [self.labels.h]
            heights = [h for h in heights if h is not None]
            if len(heights) == 0:
                return None
            else:
                assert all(h == heights[0] for h in heights), "All heights must be the same"
                return heights[0]

        @property
        def w(self) -> int | None:
            widths = [meta.w for meta in self.meta] + [self.labels.w]
            widths = [w for w in widths if w is not None]
            if len(widths) == 0:
                return None
            else:
                assert all(w == widths[0] for w in widths), "All widths must be the same"
                return widths[0]

        @classmethod
        def collate_fn(
            cls,
            seq: List[DataBatch.Camera],
            device: torch.device = torch.device("cpu"),
        ) -> DataBatch.Camera:
            return cls(
                meta=FrameMeta.collate_fn([meta for item in seq for meta in item.meta], device),
                labels=CameraFrameLabels.collate_fn([item.labels for item in seq], device),
            )

        def to(self, *args, **kwargs) -> Self:
            return self.__class__(
                meta=[meta.to(*args, **kwargs) for meta in self.meta],
                labels=self.labels.to(*args, **kwargs),
            )

        def __getitem__(self, item: Union[int, slice]) -> Self:
            """Allows indexing into the dataclass to get a subset of the data."""
            if isinstance(item, int):
                item = slice(item, item + 1)

            return self.__class__(meta=self.meta[item], labels=self.labels[item])

    @dataclass(kw_only=True, slots=True)
    class Lidar:
        """Data for a lidar frame. Includes the frame meta and labels."""

        meta: List[FrameMeta]
        labels: LidarFrameLabels

        @property
        def b(self) -> int:
            return len(self.meta)

        @property
        def h(self) -> int | None:
            heights = [meta.h for meta in self.meta] + [self.labels.h]
            heights = [h for h in heights if h is not None]
            if len(heights) == 0:
                return None
            else:
                assert all(h == heights[0] for h in heights), "All heights must be the same"
                return heights[0]

        @property
        def w(self) -> int | None:
            widths = [meta.w for meta in self.meta] + [self.labels.w]
            widths = [w for w in widths if w is not None]
            if len(widths) == 0:
                return None
            else:
                assert all(w == widths[0] for w in widths), "All widths must be the same"
                return widths[0]

        @classmethod
        def collate_fn(
            cls,
            seq: List[DataBatch.Lidar],
            device: torch.device = torch.device("cpu"),
        ) -> DataBatch.Lidar:
            return cls(
                meta=FrameMeta.collate_fn([meta for item in seq for meta in item.meta], device),
                labels=LidarFrameLabels.collate_fn([item.labels for item in seq], device),
            )

        def to(self, *args, **kwargs) -> Self:
            return self.__class__(
                meta=[meta.to(*args, **kwargs) for meta in self.meta],
                labels=self.labels.to(*args, **kwargs),
            )

        def __getitem__(self, item: Union[int, slice]) -> Self:
            """Allows indexing into the dataclass to get a subset of the data."""
            if isinstance(item, int):
                item = slice(item, item + 1)

            return self.__class__(
                meta=self.meta[item],
                labels=self.labels[item],
            )

    idx: int | None = None
    worker_id: List[int] | None = None
    sequence_id: List[str] | None = None

    camera: Camera | None = None
    lidar: Lidar | None = None

    @classmethod
    def collate_fn(
        cls,
        seq: List[DataBatch],
        device: torch.device = torch.device("cpu"),
    ) -> DataBatch:
        assert all(item.idx == seq[0].idx for item in seq), "All items must have the same idx"

        if any(item.worker_id is None for item in seq):
            worker_id = None
        else:
            worker_id = [worker_id for item in seq for worker_id in unpack_optional(item.worker_id)]

        if any(item.sequence_id is None for item in seq):
            sequence_id = None
        else:
            sequence_id = [sequence_id for item in seq for sequence_id in unpack_optional(item.sequence_id)]

        if any(item.camera is None for item in seq):
            camera = None
        else:
            camera = DataBatch.Camera.collate_fn([unpack_optional(item.camera) for item in seq], device)

        if any(item.lidar is None for item in seq):
            lidar = None
        else:
            lidar = DataBatch.Lidar.collate_fn([unpack_optional(item.lidar) for item in seq], device)

        return cls(
            idx=seq[0].idx,
            worker_id=worker_id,
            sequence_id=sequence_id,
            camera=camera,
            lidar=lidar,
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            idx=self.idx,
            worker_id=self.worker_id,
            sequence_id=self.sequence_id,
            camera=self.camera.to(*args, **kwargs) if self.camera is not None else None,
            lidar=self.lidar.to(*args, **kwargs) if self.lidar is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class RenderingBatch:
    """
    A RenderingBatch is a collection of RenderingData for camera and lidar.
    """

    camera: RenderingData | None = None
    lidar: RenderingData | None = None

    @classmethod
    def collate_fn(
        cls,
        seq: List[RenderingBatch],
        device: torch.device = torch.device("cpu"),
    ) -> RenderingBatch:
        if any(item.camera is None for item in seq):
            camera = None
        else:
            camera = RenderingData.collate_fn([unpack_optional(item.camera) for item in seq], device)

        if any(item.lidar is None for item in seq):
            lidar = None
        else:
            lidar = RenderingData.collate_fn([unpack_optional(item.lidar) for item in seq], device)

        return cls(camera=camera, lidar=lidar)

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            camera=self.camera.to(*args, **kwargs) if self.camera is not None else None,
            lidar=self.lidar.to(*args, **kwargs) if self.lidar is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class DataAndRenderingBatch:
    """
    A DataAndRenderingBatch is a compounding type used for training and validation.

    The fields are:
    - data: The information from the dataset that are required for training and validation. [DataBatch]
    - rendering: The information that will be fed into the renderer. [RenderingBatch]

    Note the design of the workflow for training and validation is:

    1. None -> [dataloader] -> DataAndRenderingBatch(data: DataBatch, rendering: RenderingBatch | None)
    2. DataAndRenderingBatch -> [pre-processing] -> DataAndRenderingBatch(data: DataBatch, rendering: RenderingBatch)
    3. RenderingBatch  -> [renderer] -> RenderingOutput
    4. (DataAndRenderingBatch, RenderingOutput) -> [loss] -> loss

    For inference, the workflow is:

    rendering: RenderingBatch -> [renderer] -> RenderingOutput
    """

    # DataBatch is always provided by the dataloader, whereas the RenderingBatch is optional.
    # In some cases we would like to compute the rendering data in the dataloader.
    # For example, for the Harmonizer sampler, or to hide the latency when not performing camera pose optimization.
    data: DataBatch
    rendering: RenderingBatch | None = None

    @classmethod
    def collate_fn(
        cls,
        seq: List[DataAndRenderingBatch],
        device: torch.device = torch.device("cpu"),
    ) -> DataAndRenderingBatch:
        if all(seq_isnone := [item.rendering is None for item in seq]):
            rendering = None
        elif not any(seq_isnone):
            rendering = RenderingBatch.collate_fn(cast(list[RenderingBatch], [item.rendering for item in seq]), device)
        else:
            raise ValueError("DataAndRenderingBatch.collate_fn: All items must have either a rendering or no rendering")

        return cls(
            data=DataBatch.collate_fn([item.data for item in seq], device),
            rendering=rendering,
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            data=self.data.to(*args, **kwargs),
            rendering=self.rendering.to(*args, **kwargs) if self.rendering is not None else None,
        )

    def pin_memory(self):
        """
        Enable pinned memory for async data transfer in PyTorch.

        When using a DataLoader with `pin_memory=True`, PyTorch calls
        `pin_memory()` on each object it retrieves from the dataset.
        Implementing this method ensures that the returned object is moved
        into pinned (page-locked) memory.
        """
        q = queue.Queue()
        q.put(self)
        while not q.empty():
            dataclass_obj = q.get()
            for field in fields(dataclass_obj):
                f = getattr(dataclass_obj, field.name)
                pin_memory_attr = getattr(f, "pin_memory", None)
                if callable(pin_memory_attr):
                    setattr(dataclass_obj, field.name, pin_memory_attr())
                elif is_dataclass(f):
                    q.put(f)
        return self


# Note(ruilong): temporarily place this class here because NCORETrainDataset depends on it.
# Should move it to nurec.models.view_geometry after migration.
class CameraFreePoseViewGeometry(torch.nn.Module):
    """
    FreePoseViewGeometry for camera sensors. It stores raw (un-subsampled) camera extrinsics and intrinsics for all frames & views. [Exist for the new batch format]
    """

    def __init__(
        self,
        T_sensor_world_startend_allviews: torch.Tensor,  # (n_frames, 2, 4, 4)
        timestamps_startend_us_allviews: torch.Tensor,  # (n_frames, 2)
        sensor_models: dict[str, CameraModel],  # mapping from unique_sensor_idx to CameraModel
        # Maps a sensor id to a range of unique frame indices that can be used to recover the slices of
        # T_sensor_world_startend_allviews and timestamps_startend_us_allviews belonging to a specific sensor.
        sensor_ids_to_frame_range: dict[str, range],
        enable_calib: bool = False,
    ):
        super().__init__()
        assert T_sensor_world_startend_allviews.shape[0] == timestamps_startend_us_allviews.shape[0], (
            "T_sensor_world_startend_allviews and timestamps_startend_us_allviews must have the same number of frames"
        )
        self.T_sensor_world_startend_allviews = torch.nn.Buffer(T_sensor_world_startend_allviews, persistent=False)
        self.timestamps_startend_us_allviews = torch.nn.Buffer(timestamps_startend_us_allviews, persistent=False)
        self.timestamps_startend_us_allviews_cpu = timestamps_startend_us_allviews.clone().cpu()
        self.sensor_models = torch.nn.ModuleDict(sensor_models)
        self.sensor_ids_to_frame_range = sensor_ids_to_frame_range

        # When `cache_sensor_params` is True, cache per-sensor subsampled data for `to_rendering_data`:
        # footprints, ncore `parameters`, and `sensorlib_parameters` from `CameraModelConverter`
        # (world rays use these with `image_points_to_world_rays_shutter_pose`).
        # `rect_subsampled` is stored to invalidate the cache when subsampling changes.
        self.cached_sensor_params: dict[str, dict] = {
            k: {
                "rect_subsampled": None,
                "footprints": None,
                "parameters": None,
                "sensorlib_parameters": None,
            }
            for k in sensor_models.keys()
        }

        self.enable_calib = enable_calib
        self.embeds: torch.nn.Embedding | None = None
        if enable_calib:
            # Delta positions (3D) + Delta rotations (6D) = 9-D embedding space.
            # Setup a single delta transformation per camera frame to be optimized in 9-D space.
            # Any 9-D vector in the embedding space corresponds to an SE(3) transformation.
            # Each SE(3) transformation will be applied to both the frame start and end poses of the same frame.
            self.embeds = torch.nn.Embedding(T_sensor_world_startend_allviews.shape[0], 9)
            torch.nn.init.zeros_(self.embeds.weight)

        self.cached_sensor_subsample: dict[tuple[Optional[RectSubsampled], int], ConcreteCameraModelsUnion] = {}

    @staticmethod
    def from_rig_trajectories(
        rig_trajectories: RigTrajectories, enable_calib: bool = False, interp_with_rig: bool = True
    ) -> CameraFreePoseViewGeometry:
        """
        Initialize a `CameraFreePoseViewGeometry` from a `NCOREDataSource`.
        """
        camera_calibrations = rig_trajectories.camera_calibrations
        world_to_nurec = rig_trajectories.world_to_nurec

        # collect all extrinsics and intrinsics
        T_sensor_world_startend_allviews = []
        timestamps_startend_us_allviews = []
        sensor_models: dict[str, CameraModel] = {}
        sensor_ids_to_frame_range: dict[str, range] = {}  # camera_id -> range of unique frame indices
        unique_frame_start_index = 0

        for sensor_id, camera_calibration in camera_calibrations.items():
            unique_sensor_idx = camera_calibration.unique_sensor_idx if camera_calibration.unique_sensor_idx >= 0 else 0

            sensor_model = camera_model_from_parameters(
                camera_calibration.camera_model_parameters, device=torch.device("cpu"), dtype=torch.float32
            )
            sensor_models[str(unique_sensor_idx)] = sensor_model

            # Match on frame-timestamp membership rather than sequence_id: this sensor's own
            # timestamps and poses are indexed below, so a trajectory that does not carry them
            # cannot serve this sensor. Several trajectories may share one sequence_id.
            candidate_trajectories = [
                trajectory
                for trajectory in rig_trajectories.rig_trajectories
                if sensor_id in trajectory.cameras_frame_timestamps_us
            ]
            assert len(candidate_trajectories) == 1, (
                f"Expected exactly one rig trajectory to carry frame timestamps for camera {sensor_id} "
                f"(sequence {camera_calibration.sequence_id}), found {len(candidate_trajectories)}"
            )
            rig_trajectory = candidate_trajectories[0]

            if interp_with_rig:
                pose_interpolator = PoseInterpolator(
                    rig_trajectory.T_rig_worlds.cpu(), rig_trajectory.T_rig_world_timestamps_us.cpu()
                )
            elif rig_trajectory.cameras_frame_T_rig_worlds is not None:
                poses = rig_trajectory.cameras_frame_T_rig_worlds[sensor_id]
            else:
                raise ValueError("Expected cameras_frame_T_rig_worlds to be available when interp_with_rig is False")

            timestamps_us = rig_trajectory.cameras_frame_timestamps_us[sensor_id]
            assert timestamps_us.ndim == 2 and timestamps_us.shape[1] == 2, (
                "timestamps_us is expected to be a 2D tensor with shape (n_frames, 2)"
            )
            timestamps_startend_us_allviews.append(timestamps_us)

            T_sensor_rig_np = camera_calibration.T_sensor_rig.cpu().numpy()
            for frame_idx, timestamp_us in enumerate(timestamps_us):
                if interp_with_rig:
                    T_rig_world_startend = pose_interpolator.interpolate_to_timestamps(timestamp_us.cpu())
                else:
                    # directly fetch the per-frame pose
                    T_rig_world_startend = poses[frame_idx].numpy()
                T_sensor_world_startend = world_to_nurec.transform_poses(T_rig_world_startend @ T_sensor_rig_np)
                T_sensor_world_startend_allviews.append(torch.from_numpy(T_sensor_world_startend).to(torch.float32))

            # Build map from sensor id to a range of unique frame indices that can be used to recover the slices of
            # T_sensor_world_startend_allviews and timestamps_startend_us_allviews belonging to a specific sensor.
            num_frames = timestamps_us.shape[0]
            sensor_ids_to_frame_range[sensor_id] = range(
                unique_frame_start_index, unique_frame_start_index + num_frames
            )
            unique_frame_start_index += num_frames

        return CameraFreePoseViewGeometry(
            T_sensor_world_startend_allviews=torch.stack(T_sensor_world_startend_allviews, dim=0),
            timestamps_startend_us_allviews=torch.cat(timestamps_startend_us_allviews, dim=0),
            sensor_models=sensor_models,
            sensor_ids_to_frame_range=sensor_ids_to_frame_range,
            enable_calib=enable_calib,
        )

    def get_frame_ranges_per_sensor(self) -> dict[str, range]:
        """Get the frame ranges per sensor."""
        return self.sensor_ids_to_frame_range

    def get_timestamps(self, unique_frame_idx: int | None = None) -> torch.Tensor:
        """Get frame start/end timestamps of either a selected training frame or all training frames from all cameras.

        Args:
            unique_frame_idx: The index of a specfic camera frame or -1 or None to return all timestamps.

        Returns:
            A (2,) tensor containing the start and end timestamps of a frame if unique_frame_idx is provided,
            otherwise a tensor of shape (n_frames, 2) containing the start and end timestamp of each camera frame.
        """
        if unique_frame_idx is None or unique_frame_idx == -1:
            return self.timestamps_startend_us_allviews  # (n_frames, 2)
        else:
            return self.timestamps_startend_us_allviews[unique_frame_idx]  # (2,)

    def get_sensor_model(self, frame_meta: FrameMeta) -> CameraModel:
        """
        Getter to request sensor model for a given sensor index.

        Args:
            frame_meta: The frame metadata. See `FrameMeta` for more details.

        Returns:
            The sensor model in the subsampled domain.
        """
        subsample = frame_meta.subsample
        unique_sensor_idx = frame_meta.unique_sensor_idx if frame_meta.unique_sensor_idx >= 0 else 0
        sensor_model = cast(ConcreteCameraModelsUnion, self.sensor_models[str(unique_sensor_idx)])
        if subsample is not None:
            return subsample.apply_to_camera_model(sensor_model)
        else:
            return sensor_model

    def _compute_elements_and_sensor_rays(
        self, sensor_model: CameraModel, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the camera rays for a given frame.
        """
        width, height = sensor_model.resolution.tolist()
        elements = generate_grid_2d_indices((width, height), device=device)
        sensor_rays = sensor_model.pixels_to_camera_rays(elements)
        elements = elements.reshape(height, width, 2)
        sensor_rays = sensor_rays.reshape(height, width, 3)
        return elements, sensor_rays

    def get_poses(self, unique_frame_idx: int | None = None, skip_calib: bool = False) -> torch.Tensor:
        """
        Get the calibrated poses for a given frame.
        """
        return SensorModelComputations.get_poses_calib(
            embeds=self.embeds,
            T_sensor_world_startend_allviews=self.T_sensor_world_startend_allviews,
            unique_frame_idx=unique_frame_idx,
            unique_frame_idx_tensor=torch.tensor([unique_frame_idx], dtype=torch.int32)
            if unique_frame_idx is not None and unique_frame_idx != -1
            else None,
            enable_calib=self.enable_calib and not skip_calib,
        )

    def get_poses_and_timestamps_startend(
        self,
        meta: FrameMeta,
        enable_calib: Optional[bool] = None,
        enable_torch_compile: bool = False,
    ) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
        """
        Get the poses and timestamps for a given frame.
        """
        return SensorModelComputations.get_poses_and_timestamps_startend(
            subsample=meta.subsample,
            embeds=self.embeds,
            T_offset_nurec_startend=meta.T_offset_nurec_startend,
            T_sensor_world_startend_allviews=self.T_sensor_world_startend_allviews,
            timestamps_startend_us_allviews=self.timestamps_startend_us_allviews,
            timestamps_startend_us_allviews_cpu=self.timestamps_startend_us_allviews_cpu,
            sensor_models=self.sensor_models,
            unique_frame_idx=meta.unique_frame_idx,
            unique_frame_idx_tensor=meta.unique_frame_idx_tensor,
            unique_sensor_idx_str=unpack_optional(meta.unique_sensor_idx_str),
            enable_calib=enable_calib if enable_calib is not None else self.enable_calib,
            is_lidar=False,
            enable_torch_compile=enable_torch_compile,
        )

    @ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data")
    def to_rendering_data(
        self,
        data_batch: DataBatch.Camera,
        cache_sensor_params: bool = False,
        skip_calib: bool = False,
        global_step_for_prober: Optional[int] = None,
        enable_torch_compile: bool = False,
        pose_correction: Optional[PoseCorrection] = None,
    ) -> RenderingData:
        """
        Convert a `DataBatch.Camera` to a `RenderingData`.

        This function will compute the followings and store them in the `RenderingData` object:
        * rays in world space via sensorlib (`sensorlib_parameters` from `CameraModelConverter` and `image_points_to_world_rays_shutter_pose`).
        * startend poses and timestamps for the given frame in the subsampled domain.
        * sensor model in the subsampled domain.
        """
        if data_batch.b == 1:
            return self._to_rendering_data_single_batch(
                data_batch,
                cache_sensor_params=cache_sensor_params,
                skip_calib=skip_calib,
                global_step_for_prober=global_step_for_prober,
                enable_torch_compile=enable_torch_compile,
                pose_correction=pose_correction,
            )
        else:
            rendering_data_list: list[RenderingData] = []
            for bidx in range(data_batch.b):
                rendering_data_list.append(
                    self._to_rendering_data_single_batch(
                        data_batch[bidx],
                        cache_sensor_params=cache_sensor_params,
                        skip_calib=skip_calib,
                        global_step_for_prober=global_step_for_prober,
                        enable_torch_compile=enable_torch_compile,
                        pose_correction=pose_correction,
                    )
                )
            return RenderingData.collate_fn(rendering_data_list, device=rendering_data_list[0].rays.device)

    def _to_rendering_data_single_batch(
        self,
        data_batch: DataBatch.Camera,
        cache_sensor_params: bool = False,
        skip_calib: bool = False,
        global_step_for_prober: Optional[int] = None,
        enable_torch_compile: bool = False,
        pose_correction: Optional[PoseCorrection] = None,
    ) -> RenderingData:
        """
        Internal single batch version of `to_rendering_data`.
        """
        assert data_batch.b == 1, "Only one frame is supported"
        meta = data_batch.meta[0]

        with ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data/get_poses_and_timestamps_startend"):
            enable_calib = self.enable_calib and not skip_calib

            # Mimic querying data stored in the rig module.
            pose_and_timestamps_startend_return = self.get_poses_and_timestamps_startend(
                meta, enable_calib, enable_torch_compile=enable_torch_compile
            )

            T_sensor_world_startend = pose_and_timestamps_startend_return.T_sensor_world_startend
            timestamps_startend_us = pose_and_timestamps_startend_return.timestamps_startend_us
            timestamps_startend_us_gpu = pose_and_timestamps_startend_return.timestamps_startend_us_gpu
            timestamps_startend_us_cpu = pose_and_timestamps_startend_return.timestamps_startend_us_cpu
            if pose_correction is not None:
                # Calibration must see timestamps after rolling-shutter crop adjustment.
                T_sensor_world_startend = pose_correction(T_sensor_world_startend, timestamps_startend_us, meta)

        with ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data/cache_sensor_params"):
            subsample_unique_sensor_idx = (meta.subsample, meta.unique_sensor_idx)
            if subsample_unique_sensor_idx in self.cached_sensor_subsample:
                sensor_model = self.cached_sensor_subsample[subsample_unique_sensor_idx]
            else:
                sensor_model = cast(ConcreteCameraModelsUnion, self.get_sensor_model(meta))
                self.cached_sensor_subsample[subsample_unique_sensor_idx] = sensor_model

            if not cache_sensor_params:
                _, sensor_rays = self._compute_elements_and_sensor_rays(sensor_model, T_sensor_world_startend.device)
                # Note(ruilong): footprints are only required when AAA is on. Should be optional but here we always compute
                # it to be aligned with the old batch format.
                footprints = compute_pixel_footprint(sensor_rays)[..., None]  # (height, width, 1)
                sensor_model_parameters = sensor_model.get_parameters()
                sensorlib_parameters = CameraModelConverter.convert(sensor_model, device=T_sensor_world_startend.device)
            else:
                subsample = meta.subsample
                unique_sensor_idx = meta.unique_sensor_idx if meta.unique_sensor_idx >= 0 else 0
                cached_sensor_params = self.cached_sensor_params[str(unique_sensor_idx)]
                if (cached_sensor_params["rect_subsampled"] == subsample) and (
                    cached_sensor_params["sensorlib_parameters"] is not None
                ):
                    # Cache hit: subsampled footprints, ncore parameters, and sensorlib parameters.

                    footprints = cached_sensor_params["footprints"]
                    sensor_model_parameters = cached_sensor_params["parameters"]
                    sensorlib_parameters = cached_sensor_params["sensorlib_parameters"]
                else:
                    # Subsample changed or cache empty: recompute footprints and sensorlib parameters.
                    _, sensor_rays = self._compute_elements_and_sensor_rays(
                        sensor_model, T_sensor_world_startend.device
                    )
                    footprints = compute_pixel_footprint(sensor_rays)[..., None]  # (height, width, 1)
                    sensor_model_parameters = sensor_model.get_parameters()
                    sensorlib_parameters = CameraModelConverter.convert(
                        sensor_model, device=T_sensor_world_startend.device
                    )
                    self.cached_sensor_params[str(unique_sensor_idx)] = {
                        "rect_subsampled": subsample,
                        "footprints": footprints,
                        "parameters": sensor_model_parameters,
                        "sensorlib_parameters": sensorlib_parameters,
                    }

        with ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data/se3pose_from_matrix"):
            translations, rotations = se3pose_from_matrix(T_sensor_world_startend)
            poses_tquat_startend = torch.cat([translations, rotations], dim=1)
            poses_tquat_startend = poses_tquat_startend.unsqueeze(0)

        with ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data/pixels_to_world_rays_shutter_pose"):
            timestamps_cpu = timestamps_startend_us_cpu.flatten()

            (world_rays, timestamps_us, _, _) = image_points_to_world_rays_shutter_pose(
                image_points=None,
                projection=sensorlib_parameters.projection,
                external_distortion=sensorlib_parameters.external_distortion,
                resolution=sensorlib_parameters.resolution,
                shutter_type=sensorlib_parameters.shutter_type,
                dynamic_pose=DynamicPose(
                    start_pose=Pose(translation=translations[0], rotation=rotations[0]),
                    end_pose=Pose(translation=translations[1], rotation=rotations[1]),
                ),
                start_timestamp_us=int(timestamps_cpu[0].item()),
                end_timestamp_us=int(timestamps_cpu[1].item()),
                return_timestamps=True,
            )

        if global_step_for_prober is not None:
            if (
                prober_result := get_global_prober()(
                    global_step_for_prober,
                    "sensor_to_world_rays_shutter_pose",
                    T_sensor_world_startend=T_sensor_world_startend,
                    timestamps_startend_us=timestamps_startend_us,
                    world_rays_grad=world_rays,
                )
            ) is not None:
                # Connect the gradient probing to the rest of the computation graph
                (world_rays,) = prober_result

        rays = unpack_optional(world_rays)  # camera rays are in nurec space
        rays = rays.reshape(sensorlib_parameters.resolution[1], sensorlib_parameters.resolution[0], 6)
        timestamps = unpack_optional(timestamps_us)
        timestamps = timestamps.reshape(sensorlib_parameters.resolution[1], sensorlib_parameters.resolution[0], 1)

        return RenderingData(
            rays=rays.unsqueeze(0),
            # FIXME: This will be on cpu in numpy array.
            sensor_model_parameters=[sensor_model_parameters],
            poses_tquat_startend=poses_tquat_startend,
            timestamps_startend_us=timestamps_startend_us_gpu,
            rays_timestamps_us=timestamps.unsqueeze(0),
            # This producer generated the complete row-major camera grid above.
            # Do not infer this guarantee from tensor contiguity at the collector.
            rays_timestamp_layout=_CAMERA_TIMESTAMP_LAYOUTS.get(sensorlib_parameters.shutter_type),
            _rays_footprints=footprints.unsqueeze(0),
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
        )


# Note(ruilong): temporarily place this class here because NCORETrainDataset depends on it.
# Should move it to nurec.models.view_geometry after migration.
class LidarFreePoseViewGeometry(torch.nn.Module):
    """
    FreePoseViewGeometry for lidar sensors. It stores raw (un-subsampled) lidar extrinsics and intrinsics for all frames. [Exist for the new batch format]
    """

    def __init__(
        self,
        T_sensor_world_startend_allviews: torch.Tensor,  # (n_frames, 2, 4, 4)
        timestamps_startend_us_allviews: torch.Tensor,  # (n_frames, 2)
        sensor_models: dict[str, LidarModel],  # mapping from unique_sensor_idx to LidarModel
        enable_calib: bool = False,
    ):
        super().__init__()
        assert T_sensor_world_startend_allviews.shape[0] == timestamps_startend_us_allviews.shape[0], (
            "T_sensor_world_startend_allviews and timestamps_startend_us_allviews must have the same number of frames"
        )
        self.T_sensor_world_startend_allviews = torch.nn.Buffer(T_sensor_world_startend_allviews, persistent=False)
        self.timestamps_startend_us_allviews = torch.nn.Buffer(timestamps_startend_us_allviews, persistent=False)
        self.timestamps_startend_us_allviews_cpu = timestamps_startend_us_allviews.clone().cpu()
        self.sensor_models = torch.nn.ModuleDict(sensor_models)

        # When `cache_sensor_params` is True, cache ncore `parameters` and `sensorlib_parameters`
        # (`LidarModelConverter`) for `generate_spinning_lidar_rays`.
        self.cached_sensor_params: dict[str, dict] = {
            k: {
                "parameters": None,
                "sensorlib_parameters": None,
            }
            for k in sensor_models.keys()
        }

        self.enable_calib = enable_calib
        self.embeds: torch.nn.Embedding | None = None
        if enable_calib:
            n_frames = T_sensor_world_startend_allviews.shape[0]
            # Delta positions (3D) + Delta rotations (6D)
            self.embeds = torch.nn.Embedding(n_frames, 9)
            torch.nn.init.zeros_(self.embeds.weight)

    @staticmethod
    def empty() -> LidarFreePoseViewGeometry:
        return LidarFreePoseViewGeometry(
            T_sensor_world_startend_allviews=torch.empty((0, 2, 4, 4)),
            timestamps_startend_us_allviews=torch.empty((0, 2), dtype=torch.int64),
            sensor_models={},
        )

    @staticmethod
    def from_rig_trajectories(
        rig_trajectories: RigTrajectories, enable_calib: bool = False
    ) -> LidarFreePoseViewGeometry:
        """
        Initialize a `LidarFreePoseViewGeometry` from a `NCOREDataSource`.
        """
        lidar_calibrations = rig_trajectories.lidar_calibrations
        if len(lidar_calibrations.items()) == 0:
            return LidarFreePoseViewGeometry.empty()

        world_to_nurec = rig_trajectories.world_to_nurec

        # collect all extrinsics and intrinsics
        T_sensor_world_startend_allviews = []
        timestamps_startend_us_allviews = []
        sensor_models: dict[str, LidarModel] = {}

        for sensor_id, lidar_calibration in lidar_calibrations.items():
            # See the camera path: selection must guarantee that lidars_frame_timestamps_us[sensor_id]
            # below resolves, which a sequence_id match does not.
            candidate_trajectories = [
                trajectory
                for trajectory in rig_trajectories.rig_trajectories
                if sensor_id in trajectory.lidars_frame_timestamps_us
            ]
            if len(candidate_trajectories) != 1:
                raise ValueError(
                    f"Expected exactly one rig trajectory to carry frame timestamps for lidar {sensor_id} "
                    f"(sequence {lidar_calibration.sequence_id}), found {len(candidate_trajectories)}"
                )
            rig_trajectory = candidate_trajectories[0]
            pose_interpolator = PoseInterpolator(
                rig_trajectory.T_rig_worlds.cpu(), rig_trajectory.T_rig_world_timestamps_us.cpu()
            )
            maybe_sensor_model = maybe_lidar_model_from_parameters(
                lidar_calibration.lidar_model_parameters, device="cpu", dtype=torch.float32
            )
            unique_sensor_idx = lidar_calibration.unique_sensor_idx if lidar_calibration.unique_sensor_idx >= 0 else 0

            # The pure unstructured + measured-ray path does not derive rays from
            # intrinsics. Keep its poses and timestamps even when typed LiDAR
            # intrinsics are absent (for example ET30X V4), and let the modeled
            # paths fail at get_sensor_model() if they actually require a model.
            if maybe_sensor_model is not None:
                sensor_models[str(unique_sensor_idx)] = maybe_sensor_model

            timestamps_us = rig_trajectory.lidars_frame_timestamps_us[sensor_id]
            assert timestamps_us.ndim == 2 and timestamps_us.shape[1] == 2, (
                "timestamps_us is expected to be a 2D tensor with shape (n_frames, 2)"
            )
            timestamps_startend_us_allviews.append(timestamps_us)

            T_sensor_rig_np = lidar_calibration.T_sensor_rig.cpu().numpy()
            for timestamp_us in timestamps_us:
                T_rig_world_startend = pose_interpolator.interpolate_to_timestamps(timestamp_us.cpu())
                T_sensor_world_startend = world_to_nurec.transform_poses(T_rig_world_startend @ T_sensor_rig_np)
                T_sensor_world_startend_allviews.append(torch.from_numpy(T_sensor_world_startend).to(torch.float32))

        return LidarFreePoseViewGeometry(
            T_sensor_world_startend_allviews=torch.stack(T_sensor_world_startend_allviews, dim=0),
            timestamps_startend_us_allviews=torch.cat(timestamps_startend_us_allviews, dim=0),
            sensor_models=sensor_models,
            enable_calib=enable_calib,
        )

    def get_sensor_model(self, frame_meta: FrameMeta) -> LidarModel:
        """
        Getter to request sensor model for a given sensor index.

        Args:
            frame_meta: The frame metadata. See `FrameMeta` for more details.

        Returns:
            The sensor model in the subsampled domain.
        """
        subsample = frame_meta.subsample
        unique_sensor_idx = frame_meta.unique_sensor_idx if frame_meta.unique_sensor_idx >= 0 else 0
        key = str(unique_sensor_idx)
        if key not in self.sensor_models:
            # Reached when a datasource supplies no typed LiDAR intrinsics. Say so,
            # instead of surfacing a bare KeyError from the lookup below.
            raise ValueError(
                f"no LiDAR sensor model for unique_sensor_idx={key}: this datasource provides no "
                "typed LiDAR intrinsics, so any path that needs the fitted spinning-lidar model "
                "(structured rays, or modeled unstructured rays) cannot run. Use "
                "dataset.lidar_ray_mode='unstructured' with dataset.use_real_lidar_rays=True, or "
                "supply lidar_model_parameters for this sensor."
            )
        sensor_model = cast(ConcreteLidarModelsUnion, self.sensor_models[key])
        if subsample is not None:
            raise NotImplementedError("subsample on sensor model is not supported for Lidar")
        return sensor_model

    def get_poses_and_timestamps_startend(
        self,
        meta: FrameMeta,
        enable_calib: bool,
        enable_torch_compile: bool = False,
    ) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
        """
        Get the poses and timestamps for a given frame.
        """
        return SensorModelComputations.get_poses_and_timestamps_startend(
            subsample=meta.subsample,
            embeds=self.embeds,
            T_offset_nurec_startend=meta.T_offset_nurec_startend,
            T_sensor_world_startend_allviews=self.T_sensor_world_startend_allviews,
            timestamps_startend_us_allviews=self.timestamps_startend_us_allviews,
            timestamps_startend_us_allviews_cpu=self.timestamps_startend_us_allviews_cpu,
            sensor_models=self.sensor_models,
            unique_frame_idx=meta.unique_frame_idx,
            unique_frame_idx_tensor=meta.unique_frame_idx_tensor,
            unique_sensor_idx_str=unpack_optional(meta.unique_sensor_idx_str),
            enable_calib=enable_calib,
            is_lidar=True,
            enable_torch_compile=enable_torch_compile,
        )

    def _compute_elements_and_sensor_rays(
        self, sensor_model: RowOffsetStructuredSpinningLidarModel, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the lidar rays for a given frame.
        """
        width, height = sensor_model.n_columns, sensor_model.n_rows
        # Note(ruilong): be careful that elements in NCore are expected to be (row, col) indices
        # which is effectively y-x order.
        elements = generate_grid_2d_indices((width, height), order="yx", device=device)
        sensor_rays = sensor_model.elements_to_sensor_rays(elements)
        elements = elements.reshape(height, width, 2)
        sensor_rays = sensor_rays.reshape(height, width, 3)
        return elements, sensor_rays

    @ScopedTimer("LidarFreePoseViewGeometry/to_rendering_data")
    def to_rendering_data(
        self,
        data_batch: DataBatch.Lidar,
        cache_sensor_params: bool = False,
        skip_calib: bool = False,
        global_step_for_prober: Optional[int] = None,
        enable_torch_compile: bool = False,
        pose_correction: Optional[PoseCorrection] = None,
    ) -> RenderingData:
        """
        Convert a `DataBatch.Lidar` to a `RenderingData`.

        This function will compute the followings and store them in the `RenderingData` object:
        * rays in world space via sensorlib (`sensorlib_parameters` from `LidarModelConverter` and `generate_spinning_lidar_rays`).
        * startend poses and timestamps for the given frame in the subsampled domain.
        * sensor model in the subsampled domain.
        """
        with ScopedTimer("LidarFreePoseViewGeometry/to_rendering_data/recursivity"):
            if data_batch.b > 1:
                rendering_data_list: list[RenderingData] = []
                for bidx in range(data_batch.b):
                    rendering_data_list.append(
                        self.to_rendering_data(
                            data_batch[bidx],
                            cache_sensor_params=cache_sensor_params,
                            skip_calib=skip_calib,
                            global_step_for_prober=global_step_for_prober,
                            enable_torch_compile=enable_torch_compile,
                            pose_correction=pose_correction,
                        )
                    )
                return RenderingData.collate_fn(rendering_data_list, device=rendering_data_list[0].rays.device)

            assert data_batch.b == 1, "Only one frame is supported"
            meta = data_batch.meta[0]

        with ScopedTimer("LidarFreePoseViewGeometry/to_rendering_data/get_poses_and_timestamps_startend"):
            # Mimic querying data stored in the rig module.
            enable_calib = self.enable_calib and not skip_calib

            pose_and_timestamps_startend_return = self.get_poses_and_timestamps_startend(
                meta, enable_calib, enable_torch_compile=enable_torch_compile
            )

            T_sensor_world_startend = pose_and_timestamps_startend_return.T_sensor_world_startend
            timestamps_startend_us = pose_and_timestamps_startend_return.timestamps_startend_us
            timestamps_startend_us_gpu = pose_and_timestamps_startend_return.timestamps_startend_us_gpu
            timestamps_startend_us_cpu = pose_and_timestamps_startend_return.timestamps_startend_us_cpu
            if pose_correction is not None:
                T_sensor_world_startend = pose_correction(T_sensor_world_startend, timestamps_startend_us, meta)

        # Ray delivery is an explicit (mode x measured) matrix -- NOT inferred
        # from sparse_* presence (structured + measured also carries sparse_*):
        #   (A) structured, modeled : sensor-model (H, W) grid.
        #   (A') structured, measured: (H, W) grid with returning cells OVERRIDDEN by
        #        the measured world rays (sparse_*).
        #   (B) unstructured, measured: model-free (N, 1) list of measured rays.
        #   (C) unstructured, modeled: model-free (N, 1) list of rays generated at the
        #       return elements.
        # INVARIANT: rays are ALWAYS in NuRec world frame (the renderer transforms to
        # the sensor frame internally). Never emit sensor-local rays.
        unstructured = data_batch.labels.ray_mode == "unstructured"
        measured_rays = data_batch.labels.sparse_rays is not None

        # Anything that moves the poses -- free-pose `embeds`, or a rig-calib `pose_correction`
        # (supplied with skip_calib=True, so `enable_calib` is False there) -- cannot move the
        # measured rays, which were precomputed in world space; leaving the two inconsistent is
        # what this guards. A' compensates below; B still asserts.
        poses_moved = enable_calib or (pose_correction is not None and pose_correction.modifies_poses)

        def recalibrate_measured_rays(measured: torch.Tensor) -> torch.Tensor:
            """Re-express precomputed world rays (N, 6) under the moved pose.

            datasets/ncore.py builds them from the stored pose, so anything that moved it --
            free-pose `embeds` or a rig-calib `pose_correction` -- must be re-applied here or
            it is silently discarded. `enable_calib=False` below is the stored pose under both,
            since `pose_correction` is applied after it.
            """
            if not poses_moved:
                return measured
            # A single rigid transform covers the sweep: motion_compensation=True puts every
            # point in the END-timepoint sensor frame. float64 throughout, cast once -- `m` is
            # near identity, so its translation is a difference of two large world translations.
            T_orig_startend = self.get_poses_and_timestamps_startend(
                meta, enable_calib=False, enable_torch_compile=enable_torch_compile
            ).T_sensor_world_startend
            m = torch.matmul(
                T_sensor_world_startend[1].double(),
                torch.linalg.inv(T_orig_startend[1].double()),
            ).to(T_sensor_world_startend.dtype)
            return torch.cat(
                [
                    torch.addmm(m[:3, 3], measured[:, :3], m[:3, :3].transpose(0, 1)),
                    torch.matmul(measured[:, 3:], m[:3, :3].transpose(0, 1)),
                ],
                dim=-1,
            )

        # The sensor model is needed whenever any ray is DERIVED from it: the
        # structured grid (A/A'), the structured-override base grid (A'), and the
        # unstructured modeled-at-elements variant (C). It is required only in the
        # pure model-free measured case (B: unstructured + measured).
        needs_model = not (unstructured and measured_rays)
        sensor_model_parameters = None
        sensorlib_parameters = None
        if needs_model:
            sensor_model = self.get_sensor_model(meta)

            assert isinstance(sensor_model, RowOffsetStructuredSpinningLidarModel), (
                "Only RowOffsetStructuredSpinningLidarModel is supported for now."
            )

            # Generate the rays from the sensor model.
            if not cache_sensor_params:
                sensor_model_parameters = sensor_model.get_parameters()
                sensorlib_parameters = LidarModelConverter.convert(sensor_model, device=T_sensor_world_startend.device)
            else:
                unique_sensor_idx = meta.unique_sensor_idx if meta.unique_sensor_idx >= 0 else 0
                cached_sensor_params = self.cached_sensor_params[str(unique_sensor_idx)]
                if cached_sensor_params["sensorlib_parameters"] is None:
                    sensor_model_parameters = sensor_model.get_parameters()
                    sensorlib_parameters = LidarModelConverter.convert(
                        sensor_model, device=T_sensor_world_startend.device
                    )
                    self.cached_sensor_params[str(unique_sensor_idx)] = {
                        "parameters": sensor_model_parameters,
                        "sensorlib_parameters": sensorlib_parameters,
                    }
                else:
                    sensor_model_parameters = cached_sensor_params["parameters"]
                    sensorlib_parameters = cached_sensor_params["sensorlib_parameters"]

        with ScopedTimer("LidarFreePoseViewGeometry/to_rendering_data/se3pose_from_matrix"):
            translations, rotations = se3pose_from_matrix(T_sensor_world_startend)
            poses_tquat_startend = torch.cat([translations, rotations], dim=1)
            poses_tquat_startend = poses_tquat_startend.unsqueeze(0)

        sparse_rays = data_batch.labels.sparse_rays
        sparse_timestamps = data_batch.labels.sparse_timestamps
        sparse_elements = data_batch.labels.sparse_elements

        with ScopedTimer("LidarFreePoseViewGeometry/to_rendering_data/elements_to_world_rays_shutter_pose"):
            timestamps_cpu = timestamps_startend_us_cpu.flatten()

            if unstructured and measured_rays:
                # (B) measured (N, 1) rays: nothing to generate from the model.
                world_rays, timestamps_us = None, None
            else:
                # Generate modeled rays: the full (H, W) grid for structured (A/A'),
                # or only the N return elements for unstructured-modeled (C).
                gen_elements = None
                if unstructured:
                    gen_elements = unpack_optional(sparse_elements).squeeze(0).to(torch.int32)  # (C)
                (world_rays, timestamps_us, _, _) = generate_spinning_lidar_rays(
                    projection=unpack_optional(sensorlib_parameters).projection,
                    elements=gen_elements,
                    dynamic_pose=DynamicPose(
                        start_pose=Pose(translation=translations[0], rotation=rotations[0]),
                        end_pose=Pose(translation=translations[1], rotation=rotations[1]),
                    ),
                    start_timestamp_us=int(timestamps_cpu[0].item()),
                    end_timestamp_us=int(timestamps_cpu[1].item()),
                    return_timestamps=True,
                )

        # Gradient prober only applies to the modeled grid path (structured).
        if global_step_for_prober is not None and not unstructured:
            if (
                prober_result := get_global_prober()(
                    global_step_for_prober,
                    "sensor_to_world_rays_shutter_pose",
                    lidar_T_sensor_world_startend=T_sensor_world_startend,
                    lidar_timestamps_startend_us=timestamps_startend_us,
                    lidar_world_rays_grad=world_rays,
                )
            ) is not None:
                # Connect the gradient probing to the rest of the computation graph
                (world_rays,) = prober_result

        if unstructured:
            # Model-free (N, 1, *) grid: dims come from the return list, NOT the
            # sensor model. sensor_model_parameters is None -> no model dependency.
            if measured_rays:
                # (B) measured world rays. Re-expressing them would not unblock calibration
                # here: the unstructured renderer takes its geometry from the dataloader-built
                # tiling and reads `rays` only for its shape, and gsplat backprops into neither
                # that tiling nor `viewmats`, so nothing would reach the pose deltas. Unblocking
                # it needs a world-ray input on UnstructuredLidarFrameContext.
                assert not poses_moved, (
                    "use_real_lidar_rays=True with lidar_ray_mode='unstructured' does not support "
                    "pose calibration: the unstructured render path cannot propagate pose gradients."
                )
                rays = unpack_optional(sparse_rays).squeeze(0).reshape(-1, 1, 6)
            else:
                # (C) modeled world rays generated at the return elements.
                gen = unpack_optional(world_rays)  # (N, 6) in nurec space
                rays = gen.reshape(-1, 1, 6)
            assert sparse_timestamps is not None, "unstructured mode requires per-ray timestamps"
            timestamps = sparse_timestamps.squeeze(0).reshape(-1, 1, 1)
            return RenderingData(
                rays=rays.unsqueeze(0),
                sensor_model_parameters=[None],  # model-free: no sensor model
                poses_tquat_startend=poses_tquat_startend,
                timestamps_startend_us=timestamps_startend_us_gpu,
                rays_timestamps_us=timestamps.unsqueeze(0),
                timestamps_startend_us_cpu=timestamps_startend_us_cpu,
                # Frame-context tiling prebuilt in the dataloader worker (if the
                # datasource exposed a spinning lidar model). Carried through so the
                # renderer rebuilds the context via from_tensors (no build in loop).
                unstructured_lidar_tiling=data_batch.labels.unstructured_lidar_tiling,
            )

        # STRUCTURED: modeled (n_rows, n_columns, 6) world-ray grid. needs_model is
        # True on this path, so sensorlib_parameters / sensor_model_parameters are set.
        projection = unpack_optional(sensorlib_parameters).projection
        rays = unpack_optional(world_rays)  # lidar rays are in nurec space
        rays = rays.reshape(projection.n_rows, projection.n_columns, 6)
        timestamps = unpack_optional(timestamps_us)
        timestamps = timestamps.reshape(projection.n_rows, projection.n_columns, 1)

        # (A') structured + measured: override the returning cells with the measured
        # world rays / timestamps (the modeled grid otherwise fills every cell).
        if measured_rays:
            measured = recalibrate_measured_rays(unpack_optional(sparse_rays).squeeze(0))
            # Override into copies: the generated rays are saved for
            # GenerateSpinningLidarRaysFunction's backward, so an in-place write bumps their
            # version counter and autograd rejects the graph. Unconditional on purpose -- with
            # `skip_first_pose_delta`, frame 0 has `enable_calib=False` yet still carries a
            # zero-weighted `embeds` term, so `poses_moved` would wrongly skip the copy there.
            rays = rays.clone()
            timestamps = timestamps.clone()
            se = unpack_optional(sparse_elements).squeeze(0)
            rows, cols = se[:, 0], se[:, 1]
            rays[rows, cols] = measured
            timestamps[rows, cols] = unpack_optional(sparse_timestamps).squeeze(0)

        return RenderingData(
            rays=rays.unsqueeze(0),
            sensor_model_parameters=[sensor_model_parameters],
            poses_tquat_startend=poses_tquat_startend,
            timestamps_startend_us=timestamps_startend_us_gpu,
            rays_timestamps_us=timestamps.unsqueeze(0),
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
        )


# Collate all attributes, *except* batch.idx.
# The input batch.idx's must be either be None or be the same index.
# The resulting batch.idx is this unique index, or None if all input indices are None.
def batch_collate_fn(
    item_or_seq: Union[DataBatch, DataAndRenderingBatch, Sequence[DataBatch], Sequence[DataAndRenderingBatch]],
    device: torch.device = torch.device("cpu"),
) -> DataBatch | DataAndRenderingBatch:
    # If it's a single batch, just return it
    if isinstance(item_or_seq, DataBatch | DataAndRenderingBatch):
        return item_or_seq

    if len(item_or_seq) == 1:
        return item_or_seq[0]

    # Short cuts for DataBatch and DataAndRenderingBatch
    if isinstance(item_or_seq[0], DataBatch):
        return DataBatch.collate_fn(cast(list[DataBatch], item_or_seq), device)
    elif isinstance(item_or_seq[0], DataAndRenderingBatch):
        return DataAndRenderingBatch.collate_fn(cast(list[DataAndRenderingBatch], item_or_seq), device)
    else:
        raise ValueError("Unsupported batch type")


@dataclass(slots=False, kw_only=True)
class InstantNuRecDataBatch:
    """
    A batch that contains (B,) groups of context and supervision DataBatch(es).
    We also precompute RenderingBatch altogether to hide latency for data preprocessing.

    Contains
        - context: list of context images (A DataAndRenderingBatch)
        - supervision: list of supervision images (A DataAndRenderingBatch)
        - cuboid_tracks: list of cuboid tracks
        - context_rig: list of rig trajectories for context (intrinsics subsampled already to context images)
        - supervision_rig: list of rig trajectories for supervision (intrinsics subsampled already to supervision images)
        - meta: list of dictionaries of metadata for each batch, e.g. sequence_id, ncore_json_path, etc.
    """

    context: list[DataAndRenderingBatch]
    supervision: list[DataAndRenderingBatch] | None = None
    cuboid_tracks: list[CuboidTracksDataPack] | None = None
    context_rig: list[RigTrajectories] | None = None
    supervision_rig: list[RigTrajectories] | None = None
    meta: list[dict[str, Any]] | None = None

    def __post_init__(self):
        if self.supervision is not None:
            assert len(self.context) == len(self.supervision), "Number of context and supervision batches must match"
        if self.cuboid_tracks is not None:
            assert len(self.context) == len(self.cuboid_tracks), "Number of context and cuboid tracks must match"
        if self.context_rig is not None:
            assert len(self.context) == len(self.context_rig), "Number of context and context_rig must match"
        if self.supervision_rig is not None:
            assert len(self.context) == len(self.supervision_rig), (
                "Number of supervision and supervision_rig must match"
            )

    def __getitem__(self, item: Union[int, slice]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            context=self.context[item],
            supervision=self.supervision[item] if self.supervision is not None else None,
            cuboid_tracks=self.cuboid_tracks[item] if self.cuboid_tracks is not None else None,
            context_rig=self.context_rig[item] if self.context_rig is not None else None,
            supervision_rig=self.supervision_rig[item] if self.supervision_rig is not None else None,
            meta=self.meta[item] if self.meta is not None else None,
        )

    def __len__(self) -> int:
        return len(self.context)

    @torch.autocast(device_type="cuda", enabled=False)
    def maybe_compute_rendering_data(self, device: torch.device):
        """Populates self.{context,supervision}[...].data.rendering unless already present"""

        for batch, rigs in ((self.context, self.context_rig), (self.supervision, self.supervision_rig)):
            if rigs is None or batch is None:
                continue
            for data, rig in zip(batch, rigs):
                # Do not re-compute if rendering data already exists.
                if data.rendering is not None:
                    continue
                # Note here that data.data.{camera,lidar} are supposed to be fp32 in their post-init checks.
                camera_rendering_data = (
                    CameraFreePoseViewGeometry.from_rig_trajectories(rig)
                    .to(device=device)
                    .to_rendering_data(data.data.camera.to(device), cache_sensor_params=True, enable_torch_compile=True)
                    if data.data.camera is not None
                    else None
                )
                lidar_rendering_data = (
                    LidarFreePoseViewGeometry.from_rig_trajectories(rig)
                    .to(device=device)
                    .to_rendering_data(data.data.lidar.to(device), cache_sensor_params=True, enable_torch_compile=True)
                    if data.data.lidar is not None
                    else None
                )
                data.rendering = RenderingBatch(camera=camera_rendering_data, lidar=lidar_rendering_data)

    @classmethod
    def collate_fn(
        cls,
        seq: Sequence[InstantNuRecDataBatch],
        device: torch.device = torch.device("cpu"),
        unsqueeze_if_zero_dim: bool = True,
    ) -> Self:
        T = TypeVar("T")

        def _collate_vals(*vals: list[T] | None) -> list[T] | None:
            assert len(vals) > 0, "At least one value must be provided"
            if any(val is None for val in vals):
                return None

            vals_arr = sum([unpack_optional(val) for val in vals], [])
            assert_same_type(vals_arr)

            for vi, v in enumerate(vals_arr):
                if isinstance(v, DataBatch):
                    vals_arr[vi] = cast(T, v.to(device))

                elif hasattr(v, "to_device"):
                    vals_arr[vi] = cast(T, v.to_device(device))

            return vals_arr

        return cls(
            context=unpack_optional(_collate_vals(*[batch.context for batch in seq])),
            supervision=_collate_vals(*[batch.supervision for batch in seq]),
            cuboid_tracks=_collate_vals(*[batch.cuboid_tracks for batch in seq]),
            context_rig=_collate_vals(*[batch.context_rig for batch in seq]),
            supervision_rig=_collate_vals(*[batch.supervision_rig for batch in seq]),
            meta=_collate_vals(*[batch.meta for batch in seq]),
        )
