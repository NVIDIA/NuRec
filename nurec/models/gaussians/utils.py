# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math

from pathlib import Path
from typing import Callable, ClassVar, List, Literal, Optional, TypeAlias, TypeVar

import numpy as np
import point_cloud_utils as pcu
import torch

from torch import nn
from torch.nn import functional as F

from nurec.utils.geometry import quat_to_so3_matrix, so3_matrix_to_quat
from nurec.utils.types import PointCloud


log = logging.getLogger(__name__)

QuaternionFormat: TypeAlias = Literal["xyzw", "wxyz"]

## NOTE: SPH code from gaussian-splatting, from plenoctree, from ???
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

T = TypeVar("T", np.ndarray, torch.Tensor)
RGBPostProcessing: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

# Must match the forward CRF (`applyCRF`) in
# nurec/utils/io/export/ppisp_export/spg_sidecars/ppisp_usd_spg.cu:155-176.
_PPISP_TOE_SHOULDER_MIN = 0.3
_PPISP_GAMMA_MIN = 0.1
_PPISP_CENTER_EPS = 1.0e-6
_PPISP_DENOM_EPS = 1.0e-6


def RGB2SH(rgb: T) -> T:
    return (rgb - 0.5) / C0


def SH2RGB(sh: T) -> T:
    return sh * C0 + 0.5


def _bounded_positive_forward(raw: torch.Tensor, minimum: float) -> torch.Tensor:
    return minimum + F.softplus(raw)


def ppisp_crf_inverse(rgb: torch.Tensor, crf_params: torch.Tensor) -> torch.Tensor:
    """Invert the new PPISP CRF pointwise for RGB values in [0, 1].

    ``crf_params`` is a per-channel ``(3, 4)`` tensor with last-dim layout
    ``[toe_raw, shoulder_raw, gamma_raw, center_raw]``, matching the forward CRF (``applyCRF``)
    in nurec/utils/io/export/ppisp_export/spg_sidecars/ppisp_usd_spg.cu:155-176. The inverse is
    exact only on the non-saturated interior: values the forward mapped to 0/1 cannot be
    recovered.
    """
    crf_params = crf_params.to(device=rgb.device, dtype=rgb.dtype)
    toe = _bounded_positive_forward(crf_params[:, 0], _PPISP_TOE_SHOULDER_MIN)
    shoulder = _bounded_positive_forward(crf_params[:, 1], _PPISP_TOE_SHOULDER_MIN)
    gamma = _bounded_positive_forward(crf_params[:, 2], _PPISP_GAMMA_MIN)
    center = torch.clamp(torch.sigmoid(crf_params[:, 3]), _PPISP_CENTER_EPS, 1.0 - _PPISP_CENTER_EPS)

    lerp = torch.clamp((shoulder - toe) * center + toe, min=_PPISP_DENOM_EPS)
    a = (shoulder * center) / lerp
    # Forward leaves b = 1 - a unclamped; a < 1 always since center < 1, so this clamp only
    # guards the div-by-zero in (1 - u) / b at the degenerate limit.
    b = torch.clamp(1.0 - a, min=_PPISP_DENOM_EPS)

    y = torch.clamp(rgb, 0.0, 1.0)
    u = torch.pow(y, 1.0 / gamma)
    below = center * torch.pow(torch.clamp(u / a, min=0.0), 1.0 / toe)
    above = 1.0 - (1.0 - center) * torch.pow(torch.clamp((1.0 - u) / b, min=0.0), 1.0 / shoulder)
    return torch.clamp(torch.where(u <= a, below, above), 0.0, 1.0)


def make_ppisp_crf_inverse_rgb_post_processing(crf_params: torch.Tensor, camera_index: int = 0) -> RGBPostProcessing:
    """Build the RGB correction that bakes a destination-camera PPISP CRF inverse into albedo.

    Only the CRF is inverted: it is the dominant tone stage, and unlike exposure/color/vignetting
    it is a global pointwise curve that can be folded into per-Gaussian base color.
    """
    if crf_params.dim() == 3:
        if camera_index < 0 or camera_index >= crf_params.shape[0]:
            raise ValueError(
                f"camera_index={camera_index} out of bounds for crf_params shape {tuple(crf_params.shape)}"
            )
        crf_params = crf_params[camera_index]
    if crf_params.shape != (3, 4):
        raise ValueError(f"Expected crf_params shape (3, 4), got {tuple(crf_params.shape)}")

    def rgb_post_processing(rgb: torch.Tensor) -> torch.Tensor:
        return ppisp_crf_inverse(rgb, crf_params)

    return rgb_post_processing


def apply_rgb_post_processing_to_features_dc(
    features_albedo: torch.Tensor, rgb_post_processing: RGBPostProcessing
) -> None:
    """Bake an RGB correction into the view-independent SH DC albedo band.

    Higher-order SH/specular terms are left unchanged: a pointwise nonlinear RGB curve is
    not a coefficient-wise SH transform.
    """
    if features_albedo.dim() == 2:
        dc_albedo = features_albedo
    elif features_albedo.dim() == 3:
        dc_albedo = features_albedo[:, 0, :]
    else:
        raise ValueError(f"Expected features_albedo rank 2 or 3, got {features_albedo.dim()}")

    corrected_rgb = rgb_post_processing(SH2RGB(dc_albedo))
    dc_albedo.copy_(RGB2SH(corrected_rgb).to(dtype=dc_albedo.dtype))


def sh_degree_to_specular_dim(degree: int) -> int:
    """Number of dimensions used by SH of deg [1..degree], inclusive"""
    return 3 * ((degree + 1) ** 2 - 1)


def sh_degree_to_num_features(degree: int) -> int:
    """Number of dimensions used by SH of deg [0..degree], inclusive"""
    return sh_degree_to_specular_dim(degree) + 3


def num_features_to_sh_degree(num_features: int) -> int:
    """
    Given num_features from sh_degree_to_num_features(d) = 3 * (d + 1)^2, compute the integer degree d.
    """
    # 1) Check that num_features is a multiple of 3
    assert num_features % 3 == 0, (
        f"num_features = {num_features} is not a multiple of 3, so it cannot match 3*(d+1)^2 for integer d"
    )

    # 2) Divide by 3
    squared_part = num_features // 3

    # 3) Check that squared_part is a perfect square:
    candidate = math.isqrt(squared_part)
    assert candidate * candidate == squared_part, (
        f"num_features = {num_features} implies {squared_part} is not a perfect square, so it cannot match (d+1)^2 for integer d"
    )

    # 4) Subtract 1 to get the degree
    degree = candidate - 1

    # 5) Optional: check for negative degree (if candidate == 0, that means no valid degree).
    assert degree >= 0, f"num_features = {num_features} is too small to represent degree 0 or higher"

    return degree


def cube_root(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3)


def spherical_to_cartesian(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def uniform_sample_sphere(
    num_samples: int, device: torch.device, inverse: bool = False, radius: float = 1.0
) -> torch.Tensor:
    """
    refer to https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    sample points uniformly inside a sphere
    """
    if not inverse:
        dist = torch.rand((num_samples,)).to(device)
        dist = cube_root(dist) * radius
    else:
        dist = torch.rand((num_samples,)).to(device)
        dist = 1 / dist.clamp_min(1 / radius)
    thetas = torch.arccos(2 * torch.rand((num_samples,)) - 1).to(device)
    phis = 2 * torch.pi * torch.rand((num_samples,)).to(device)
    pts = spherical_to_cartesian(dist, thetas, phis)
    return pts


def subsample_points(
    num_points: int,
    point_clouds: List[PointCloud],
    gaussian_cuboid_ids: List[torch.Tensor],
    gaussian_scales: List[torch.Tensor],
    observed_counts: List[int],
) -> tuple[List[PointCloud], List[torch.Tensor], List[torch.Tensor], List[int]]:
    assert num_points > 0, "num_points must be greater than 0"

    total_points = sum([pc.n_points for pc in point_clouds])
    if total_points > num_points:
        layer_sample_ratio = float(num_points) / total_points
        subsampled_point_clouds: List[PointCloud] = []
        subsampled_gaussian_cuboid_ids: List[torch.Tensor] = []
        subsampled_gaussian_scales: List[torch.Tensor] = []
        subsampled_observed_counts: List[int] = []

        for pc, gid, scale, observed_count in zip(point_clouds, gaussian_cuboid_ids, gaussian_scales, observed_counts):
            num_sampled_points = int(pc.n_points * layer_sample_ratio)
            if num_sampled_points == 0:
                continue
            random_idxs = torch.tensor(
                np.random.choice(pc.n_points, num_sampled_points, replace=False), device=pc.xyz_start.device
            )
            subsampled_point_clouds.append(pc[random_idxs])
            subsampled_gaussian_cuboid_ids.append(gid[random_idxs])
            subsampled_gaussian_scales.append(scale[random_idxs])
            subsampled_observed_counts.append(observed_count)

        point_clouds = subsampled_point_clouds
        gaussian_cuboid_ids = subsampled_gaussian_cuboid_ids
        gaussian_scales = subsampled_gaussian_scales
        observed_counts = subsampled_observed_counts

    return point_clouds, gaussian_cuboid_ids, gaussian_scales, observed_counts


def track_random_initialization(
    num_points: int, track_index: int, cuboid_extent: torch.Tensor, default_scale: float, device: torch.device
) -> tuple[PointCloud, torch.Tensor, torch.Tensor]:
    xyz_e_r_local = torch.rand((num_points, 3), device=device) * cuboid_extent - cuboid_extent / 2
    color_r = (torch.rand((num_points, 3), device=device) * 255.0).byte()
    # We don't currently use xyz_start for initialization so setting it to xyz_e_r_local, but if that changes
    # we need to determine what the right starting point for random initializations should be
    scale_r = torch.full((num_points,), default_scale, device=device)
    # we use scale_r as a point cloud argument before otherwise PointCloud.collate_fn will fail due to
    # inconsistent shapes
    point_cloud = PointCloud(
        xyz_start=xyz_e_r_local,
        xyz_end=xyz_e_r_local,
        color=color_r,
        camera_footprint_scale=scale_r,
    )
    gaussian_cuboid_id = torch.full(
        (num_points, 1),
        fill_value=track_index,
        dtype=torch.int32,
        device=device,
    )
    gaussian_scale = scale_r.unsqueeze(-1)
    return point_cloud, gaussian_cuboid_id, gaussian_scale


class PLYGaussianLoader:
    """
    Loads ply files generated by https://github.com/3DTopia/LGM/blob/main/core/gs.py#L101,
    which should be the format used by the original 3DGS implementation.

    This has been expanded to import additional properties exported by instant_nurec, such as a road_mask
    (1 for each gaussian associated with the road) and sky_mask (float per gaussian, used for
    sky filtering). This class makes no assumptions about whether the gaussian properties
    (scale, density) are activated or not.
    """

    cuboids_dims: Optional[torch.Tensor]
    TENSOR_FIELDS: ClassVar[tuple[str, ...]] = (
        "positions",
        "rotations",
        "densities",
        "scales",
        "features_albedo",
        "features_specular",
        "road_mask",
        "sky_mask",
        "cuboids_dims",
    )

    def __init__(self, path: Path, quaternion_format: str = "wxyz", device: str = "cuda") -> None:
        """
        quaternion_format (str): specifies the format of the rotation read from the PLY file.
            Must be either "wxyz" or "xyzw". This is used when transforming.
        """
        self.device = device

        assert (Path(path)).is_file(), f"{self.__class__.__name__} provided path {path} not a file"
        data = self._load_ply(path)

        self.positions = data["positions"]
        self.rotations = data["rotations"]
        self.densities = data["densities"]
        self.scales = data["scales"]
        self.features_albedo = data["features_albedo"]
        self.features_specular = data["features_specular"] if "features_specular" in data else None
        self.road_mask = data["road_mask"] if "road_mask" in data else None
        self.sky_mask = data["sky_mask"] if "sky_mask" in data else None
        self.cuboids_dims = None

        assert quaternion_format in ["xyzw", "wxyz"]
        self.quaternion_format = quaternion_format

    def clone(self) -> "PLYGaussianLoader":
        cloned = self.__class__.__new__(self.__class__)
        cloned.device = self.device
        cloned.positions = self.positions.clone()
        cloned.rotations = self.rotations.clone()
        cloned.densities = self.densities.clone()
        cloned.scales = self.scales.clone()
        cloned.features_albedo = self.features_albedo.clone()
        cloned.features_specular = self.features_specular.clone() if self.features_specular is not None else None
        cloned.road_mask = self.road_mask.clone() if self.road_mask is not None else None
        cloned.sky_mask = self.sky_mask.clone() if self.sky_mask is not None else None
        cloned.cuboids_dims = self.cuboids_dims.clone() if self.cuboids_dims is not None else None
        cloned.quaternion_format = self.quaternion_format
        return cloned

    def transform(self, transform: torch.Tensor) -> None:
        """Applies a rigid transform to the gaussians in-place.

        transform (torch.Tensor): Expects the transform in the form of a homogeneous source -> target
            transformation:
            ⎡ R  -o ⎤
            ⎣ 0 1/s ⎦
        """
        self.positions = (self.positions @ transform[:3, :3].T) + transform[:3, 3]

        # quat_to_so3_matrix expects XYZW format
        if self.quaternion_format == "xyzw":
            quaternions = self.rotations
        else:
            quaternions = self.rotations[:, [1, 2, 3, 0]]

        rotations_matrix = quat_to_so3_matrix(quaternions, unbatch=False)
        quaternions = so3_matrix_to_quat(transform[:3, :3] @ rotations_matrix, unbatch=False)

        if self.quaternion_format == "xyzw":
            self.rotations = quaternions
        else:
            self.rotations = quaternions[:, [3, 0, 1, 2]]

    def scale(self, scale_factor: float, scale_activation: nn.Module, scale_activation_inv: nn.Module) -> None:
        """Scales the gaussians in-place"""
        self.positions *= scale_factor
        self.scales = scale_activation_inv(scale_activation(self.scales) * scale_factor)

    def _load_ply(self, path: Path) -> dict[str, torch.Tensor]:
        ply_data = pcu.load_triangle_mesh(path).vertex_data
        ply_metadata = ply_data.custom_attributes

        positions = torch.FloatTensor(ply_data.positions).to(self.device)
        densities = torch.FloatTensor(ply_metadata["opacity"]).unsqueeze(-1).to(self.device)

        features_albedo = torch.cat(
            [
                torch.FloatTensor(ply_metadata["f_dc_0"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["f_dc_1"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["f_dc_2"]).unsqueeze(-1),
            ],
            dim=1,
        ).to(self.device)

        scales = torch.cat(
            [
                torch.FloatTensor(ply_metadata["scale_0"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["scale_1"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["scale_2"]).unsqueeze(-1),
            ],
            dim=1,
        ).to(self.device)

        rotations = torch.cat(
            [
                torch.FloatTensor(ply_metadata["rot_0"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["rot_1"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["rot_2"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["rot_3"]).unsqueeze(-1),
            ],
            dim=1,
        ).to(self.device)

        # Load all f_rest_x parameters into a buffer, ordered by their coefficient index.
        # The reader hands back custom_attributes in hash order rather than file order, so
        # taking the keys as they come permutes the SH coefficients -- even for files this
        # module's own write_ply_3dgs produced.
        f_rest_keys = sorted(
            (key for key in ply_metadata.keys() if key.startswith("f_rest_")),
            key=lambda key: int(key.removeprefix("f_rest_")),
        )
        f_rest_tensors = [torch.FloatTensor(ply_metadata[key]).unsqueeze(-1) for key in f_rest_keys]

        ret = {
            "positions": positions,
            "rotations": rotations,
            "densities": densities,
            "scales": scales,
            "features_albedo": features_albedo,
        }

        num_gaussians = positions.shape[0]
        num_speculars = len(f_rest_tensors) // 3
        features_specular: Optional[torch.Tensor] = None
        if f_rest_tensors:
            features_specular = (
                torch.cat(f_rest_tensors, dim=1)
                .to(self.device)
                .reshape(num_gaussians, 3, num_speculars)
                .transpose(2, 1)
                .reshape(num_gaussians, num_speculars * 3)
            )
            ret["features_specular"] = features_specular

        if "road_mask" in ply_metadata.keys():
            ret["road_mask"] = torch.BoolTensor(ply_metadata["road_mask"]).to(self.device)

        if "sky_mask" in ply_metadata.keys():  # "Mask" is a float per gaussian describing the sky probability
            sky = torch.FloatTensor(ply_metadata["sky_mask"]).to(self.device)
            ret["sky_mask"] = sky.unsqueeze(-1) if sky.dim() == 1 else sky

        return ret

    @classmethod
    def from_ply_bytes(
        cls, ply_bytes: bytes, device: str = "cuda", cuboids_dims: Optional[torch.Tensor] = None
    ) -> "PLYGaussianLoader":
        """
        Create loader from PLY bytes by writing to temporary file.

        Args:
            ply_bytes: PLY file contents as bytes
            device: Device to load tensors onto

        Returns:
            PLYGaussianLoader instance
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_file.write(ply_bytes)
            tmp_path = Path(tmp_file.name)

        try:
            loader = cls(path=tmp_path, device=device)
            loader.cuboids_dims = cuboids_dims
            return loader
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def from_tensors(
        cls,
        *,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        densities: torch.Tensor,
        scales: torch.Tensor,
        features_albedo: torch.Tensor,
        features_specular: Optional[torch.Tensor] = None,
        road_mask: Optional[torch.Tensor] = None,
        sky_mask: Optional[torch.Tensor] = None,
        cuboids_dims: Optional[torch.Tensor] = None,
        quaternion_format: QuaternionFormat = "wxyz",
        device: str | torch.device | None = None,
        copy: bool = True,
    ) -> "PLYGaussianLoader":
        """Create a loader from an in-memory Gaussian tensor payload.

        Inputs are detached and moved to one device. By default they are copied; ``copy=False``
        transfers ownership of already-ordinary tensors, such as newly allocated distributed
        receive buffers. Callers that need ordinary tensors must invoke this constructor outside
        inference mode. No Gaussian activation or quaternion-layout conversion is performed here.
        """
        if quaternion_format not in ("xyzw", "wxyz"):
            raise ValueError(f"Unsupported quaternion_format {quaternion_format!r}; expected 'xyzw' or 'wxyz'")

        num_gaussians = positions.shape[0] if positions.ndim > 0 else 0
        required_shapes = {
            "positions": (positions, (num_gaussians, 3)),
            "rotations": (rotations, (num_gaussians, 4)),
            "densities": (densities, (num_gaussians, 1)),
            "scales": (scales, (num_gaussians, 3)),
            "features_albedo": (features_albedo, (num_gaussians, 3)),
        }
        for name, (value, expected_shape) in required_shapes.items():
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(value.shape)}")
        if features_specular is not None and (
            features_specular.ndim != 2 or features_specular.shape[0] != num_gaussians
        ):
            raise ValueError(
                "features_specular must be a two-dimensional tensor with "
                f"{num_gaussians} rows, got {tuple(features_specular.shape)}"
            )
        if road_mask is not None:
            if tuple(road_mask.shape) != (num_gaussians,):
                raise ValueError(f"road_mask must have shape ({num_gaussians},), got {tuple(road_mask.shape)}")
            if road_mask.dtype != torch.bool:
                raise ValueError(f"road_mask must have dtype bool, got {road_mask.dtype}")
        if sky_mask is not None and tuple(sky_mask.shape) != (num_gaussians, 1):
            raise ValueError(f"sky_mask must have shape ({num_gaussians}, 1), got {tuple(sky_mask.shape)}")

        target_device = positions.device if device is None else torch.device(device)

        def copied(value: torch.Tensor) -> torch.Tensor:
            value = value.detach()
            if not copy and value.is_inference():
                raise ValueError("copy=False cannot take ownership of an inference tensor")
            return value.to(device=target_device, copy=copy)

        loader = cls.__new__(cls)
        loader.device = str(target_device)
        loader.positions = copied(positions)
        loader.rotations = copied(rotations)
        loader.densities = copied(densities)
        loader.scales = copied(scales)
        loader.features_albedo = copied(features_albedo)
        loader.features_specular = copied(features_specular) if features_specular is not None else None
        loader.road_mask = copied(road_mask) if road_mask is not None else None
        loader.sky_mask = copied(sky_mask) if sky_mask is not None else None
        loader.cuboids_dims = copied(cuboids_dims) if cuboids_dims is not None else None
        loader.quaternion_format = quaternion_format
        return loader


# Type alias for cleaner API
Asset: TypeAlias = PLYGaussianLoader


def write_ply_3dgs(
    path: Path,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    features_albedo: torch.Tensor,
    features_specular: torch.Tensor | None = None,
    color: torch.Tensor | None = None,
    normals: torch.Tensor | None = None,
    custom_attributes: dict[str, torch.Tensor] = {},
) -> None:
    """
    Writes a PLY file from the given tensors in the original 3DGS format.

    Note that the format should be compatible with the original 3DGS implementation but differences
    between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
    3rd-party 3DGS viewers.
    Note2: The given tensors should be the raw Gaussian parameters, not the activated ones (e.g., sigmoid, exp, relu, etc.).
    """
    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = positions.cpu().numpy()

    if color is not None:
        mesh.vertex_data.colors = color.cpu().numpy()

    if normals is not None:
        assert normals.shape == positions.shape, "normals must have the same shape as positions"
        mesh.vertex_data.normals = normals.cpu().numpy()

    rotations_numpy = rotations.cpu().numpy()
    for attr_i in range(4):
        mesh.vertex_data.custom_attributes[f"rot_{attr_i}"] = rotations_numpy[..., attr_i]

    scales_numpy = scales.cpu().numpy()
    for attr_i in range(3):
        mesh.vertex_data.custom_attributes[f"scale_{attr_i}"] = scales_numpy[..., attr_i]

    mesh.vertex_data.custom_attributes["opacity"] = densities.cpu().numpy()

    features_albedo_numpy = features_albedo.cpu().numpy()
    for attr_i in range(3):
        mesh.vertex_data.custom_attributes[f"f_dc_{attr_i}"] = features_albedo_numpy[..., attr_i]

    num_gaussians = positions.shape[0]
    if features_specular is not None:
        num_speculars = features_specular.shape[-1] // 3
        features_specular_numpy = (
            features_specular.reshape((num_gaussians, num_speculars, 3))
            .transpose(2, 1)
            .reshape((num_gaussians, num_speculars * 3))
            .cpu()
            .numpy()
        )
        for attr_i in range(features_specular.shape[-1]):
            mesh.vertex_data.custom_attributes[f"f_rest_{attr_i}"] = features_specular_numpy[..., attr_i]

    for key, value in custom_attributes.items():
        mesh.vertex_data.custom_attributes[key] = value.cpu().numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    log.info(f"Wrote {path.suffix}-file: {path.absolute()}")


def write_ply_3dgrt(
    path: Path,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    percentage_gaussians: float = 100,
) -> None:
    """
    Write a specified percentage of Gaussians into a PLY file, each Gaussian as a pseudo-colored mesh octahedron.

    Each Gaussian is represented as an octahedron transformed by the Gaussian parameters:
    gaussian = alpha * S @ R.t() @ v + mu, where S is the scale matrix, R is the rotation matrix, v are the canonical vertices of the octahedron, mu is the positions vector.
    alpha affects the size of each Gaussian as it's based on the density, alpha = sqrt(2 * log(density / 0.01)).
    The color of each Gaussian is based its direction, i.e. the sign of its own coordinate system:
    (+x, +y, +z, -x, -y, -z) = (red, green, blue, white, white, white)

    Note that the given tensors should be the Gaussian parameters after applying the activation function (e.g., sigmoid, exp, relu, etc.).

    Args:
        positions[torch.Tensor]: Positions of the 3D Gaussians (x, y, z) of shape [n_gaussians, 3]
        rotations[torch.Tensor]: Rotation of each Gaussian represented as a unit quaternion (xyzw) of shape [n_gaussians, 4]
        scales[torch.Tensor]: Anisotropic scale of each Gaussian of shape [n_gaussians, 3]
        densities[torch.Tensor]: Density of each Gaussian of shape [n_gaussians, 1]
        percentage_gaussians[float]: Use only n% of gaussians to reduce the number of vertices. Range (0, 100]. Default is 100.

    Returns:
        None
    """
    assert 0 < percentage_gaussians <= 100

    # Find invalid values (NaN, +-inf) in the three input tensors and remove them
    valid_mask = ~(
        ~torch.isfinite(positions).all(dim=1)
        | ~torch.isfinite(rotations).all(dim=1)
        | ~torch.isfinite(scales).all(dim=1)
        | ~torch.isfinite(densities).all(dim=1)
        | torch.isnan(positions).any(dim=1)
        | torch.isnan(rotations).any(dim=1)
        | torch.isnan(scales).any(dim=1)
        | torch.isnan(densities).any(dim=1)
    )

    n_gaussians_all = positions.shape[0]
    n_valid = int(torch.sum(valid_mask).item())
    if n_gaussians_all - n_valid > 0:
        log.info(
            f"Found {n_gaussians_all - n_valid} invalid gaussians out of {n_gaussians_all} total. Keeping {n_valid} valid gaussians"
        )
    if n_valid == 0:
        log.warning("No valid gaussians remaining after filtering! Exiting mesh construction early.")
        return

    positions = positions[valid_mask, ...]  # [n_valid, 3]
    rotations = rotations[valid_mask, ...]  # [n_valid, 4]
    scales = scales[valid_mask, ...]  # [n_valid, 3]
    densities = densities[valid_mask, ...]  # [n_valid, 1]

    # Remove Gaussians according to percentage_gaussians, but keep at least 1 Gaussian
    n_gaussians_valid = positions.shape[0]
    n_gaussians_to_keep = max(1, int(n_gaussians_valid * percentage_gaussians / 100))
    log.info(
        f"Using {n_gaussians_to_keep} gaussians out of {n_gaussians_valid} ({percentage_gaussians:.1f}%) via random sampling"
    )

    # Generate random indices for uniform sampling
    if n_gaussians_to_keep < n_gaussians_valid:
        random_indices = torch.randperm(n_gaussians_valid, device=positions.device)[:n_gaussians_to_keep]
        random_indices = torch.sort(random_indices)[0]  # Sort for better memory access patterns

        positions = positions[random_indices, :]  # [n_gaussians_to_keep, 3]
        rotations = rotations[random_indices, :]  # [n_gaussians_to_keep, 4]
        scales = scales[random_indices, :]  # [n_gaussians_to_keep, 3]
        densities = densities[random_indices, :]  # [n_gaussians_to_keep, 1]

    # Build the canonical octahedron
    octa_canonical = torch.cat(
        [torch.eye(3, device=scales.device), -torch.eye(3, device=scales.device)], dim=-1
    )  # [3, 6]
    n_vertices_per_gaussian = octa_canonical.shape[1]

    # Vertices of the mesh
    # Transform octahedron using alpha * S @ R.t() @ v + mu formula from 3DGRT paper
    alpha_min = 0.005  # default opacity threshold
    densities_clamped = densities.clamp_min(alpha_min)
    alpha = torch.sqrt(2 * torch.log(densities_clamped / alpha_min))  # [n_gaussians_to_keep, 1]
    rot_mats = quat_to_so3_matrix(rotations, unbatch=False)  # [n_gaussians_to_keep, 3, 3]
    scale_mats = torch.diag_embed(scales)  # [n_gaussians_to_keep, 3, 3]
    gaussians = (
        alpha[:, None] * scale_mats @ rot_mats.transpose(-1, -2) @ octa_canonical[None, :, :] + positions[:, :, None]
    )  # [n_gaussians_to_keep, 3, n_vertices_per_gaussian]
    gaussians = gaussians.transpose(dim0=-1, dim1=-2)  # [n_gaussians_to_keep, n_vertices_per_gaussian, 3]

    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = gaussians.reshape(-1, 3).cpu().numpy()  # [V, 3]

    # Faces of the mesh
    faces = generate_octahedron_faces(octa_canonical.transpose(1, 0)[None, :, :])  # [1, 8, 3]
    # Compute the vertices index offsets for each gaussian
    vertex_offsets = (
        torch.arange(gaussians.shape[0], device=faces.device) * n_vertices_per_gaussian
    )  # [n_gaussians_to_keep]
    faces = faces + vertex_offsets[:, None, None]  # [n_gaussians_to_keep, 8, 3]
    mesh.face_data.vertex_ids = faces.reshape(-1, 3).cpu().numpy()

    # Color (RGBA) the gaussians in the mesh (red is x+, green is y+, blue is z+, rest is white)
    colors = np.array(
        gaussians.shape[0]
        * [
            [1, 0, 0, 1],  # (1,0,0) x+
            [0, 1, 0, 1],  # (0,1,0) y+
            [0, 0, 1, 1],  # (0,0,1) z+
            [1, 1, 1, 1],  # (-1,0,0) x-
            [1, 1, 1, 1],  # (0,-1,0) y-
            [1, 1, 1, 1],  # (0,0,-1) z-
        ]
    )  # [n_gaussians_to_keep * 6, 4]
    mesh.vertex_data.colors = colors

    # Save the mesh
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    logging.info(f"Saved mesh file: {path}")


def generate_octahedron_faces(vertices: torch.Tensor) -> torch.Tensor:
    """
    Generating octahedron faces from batches of six 3D vertices.

    Args:
        vertices: torch tensor of shape (..., 6, 3) containing sets of six 3D points

    Returns:
        torch tensor of shape (..., 8, 3) containing face indices with correct winding
    """
    original_shape = vertices.shape
    assert original_shape[-2:] == (6, 3), "Last two dimensions must be (6, 3)"

    # Reshape to (N, 6, 3) for batch processing
    batch_shape = original_shape[:-2]
    n_octahedra = int(torch.prod(torch.tensor(batch_shape)).item()) if batch_shape else 1
    vertices_flat = vertices.reshape(n_octahedra, 6, 3)

    # Define the 8 triangular faces of the octahedron
    # Note: The faces have the correct winding order in the canonical octahedron: (+-1,0,0), (0,+-1,0), (0,0,+-1).
    face_templates = [
        # Top pyramid (around +z vertex, index 2)
        [0, 1, 2],  # +x, +y, +z
        [1, 3, 2],  # +y, -x, +z
        [3, 4, 2],  # -x, -y, +z
        [4, 0, 2],  # -y, +x, +z
        # Bottom pyramid (around -z vertex, index 5)
        [0, 4, 5],  # +x, -y, -z
        [4, 3, 5],  # -y, -x, -z
        [3, 1, 5],  # -x, +y, -z
        [1, 0, 5],  # +y, +x, -z
    ]

    # Compute face indices for given octahedra defined by vertices such that the normals point outward (correct winding order).
    # Initialize output array
    faces_flat = torch.zeros((n_octahedra, 8, 3), dtype=torch.long, device=vertices.device)

    # Process each face template
    for face_idx, face_template in enumerate(face_templates):
        v0, v1, v2 = face_template

        # Vectorized computation for all octahedra
        edge1 = vertices_flat[:, v1] - vertices_flat[:, v0]  # (N, 3)
        edge2 = vertices_flat[:, v2] - vertices_flat[:, v0]  # (N, 3)

        # Cross product for all octahedra
        normals = torch.cross(edge1, edge2)  # (N, 3)

        # Calculate face centers
        face_centers = (vertices_flat[:, v0] + vertices_flat[:, v1] + vertices_flat[:, v2]) / 3  # (N, 3)

        # Outward direction (normalized face center for octahedron)
        face_center_norms = torch.norm(face_centers, dim=1, keepdim=True)  # (N, 1)
        outward_directions = face_centers / (face_center_norms + 1e-8)  # (N, 3)

        # Dot product to check if normal points outward
        dot_products = torch.sum(normals * outward_directions, dim=1)  # (N,)

        # Set face indices based on dot product sign
        faces_flat[:, face_idx] = torch.where(
            dot_products[:, None] < 0,  # If normal points inward
            torch.tensor([v0, v2, v1], device=vertices.device),  # Reverse winding
            torch.tensor([v0, v1, v2], device=vertices.device),  # Keep original winding
        )

    # Reshape back to original batch shape
    output_shape = batch_shape + (8, 3)
    return faces_flat.reshape(output_shape)
