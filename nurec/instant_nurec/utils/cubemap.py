# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from einops import rearrange

from ncore.impl.data.types import CameraModelParameters
from nurec.libs.vren.interface import camera_rays_to_image_points
from nurec.models.background_sampling import sample_cubemap_seamless
from nurec.utils.profiling import ScopedTimer


def cubemap_ray_directions(size: int, device: torch.device) -> torch.Tensor:
    """
    Compute (6, size, size, 3) ray directions corresponding to the sky texture.
    """
    # Corresponds to pixel centers (not corners)
    px = (torch.arange(size, device=device) + 0.5) / size * 2 - 1
    uu, vv = torch.meshgrid(px, px, indexing="xy")
    front_dirs = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
    front_dirs = front_dirs / front_dirs.norm(dim=-1, keepdim=True)

    xx, yy, zz = front_dirs.unbind(-1)
    right_dirs = torch.stack([zz, yy, -xx], dim=-1)
    left_dirs = torch.stack([-zz, yy, xx], dim=-1)
    top_dirs = torch.stack([xx, -zz, yy], dim=-1)
    bottom_dirs = torch.stack([xx, zz, -yy], dim=-1)
    back_dirs = torch.stack([-xx, yy, -zz], dim=-1)

    return torch.stack([right_dirs, left_dirs, top_dirs, bottom_dirs, front_dirs, back_dirs], dim=0)


@ScopedTimer("instant_nurec.utils.unproject_to_sky_cubemap")
@torch.compile
def unproject_to_sky_cubemap(
    sky_cubemap_size: int,
    R_camera_world: torch.Tensor,
    camera_model_parameters: list[CameraModelParameters],
    feature: torch.Tensor,
    feature_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unproject the RGB image to the sky cubemap.
    Args:
        R_camera_world: (N, 3, 3) world rotation matrix of the camera
        camera_model_parameters: The camera model parameters. [N,]
        feature: The feature image. [N, H, W, C]
        feature_mask: The mask of the feature image. [N, H, W, 1]
    Returns:
        The sky cubemap feature image. [6, self.sky_cubemap_size, self.sky_cubemap_size, C]
        The mask corresponding to the cubemap. [6, self.sky_cubemap_size, self.sky_cubemap_size, 1]
    """
    # TODO: This function is very slow. Consider downsample the feature image or use batched implementation
    # of camera_rays_to_image_points.

    sky_rays_d = cubemap_ray_directions(sky_cubemap_size, feature.device)
    feature_dim = feature.shape[-1]
    sky_cubemap_shape = (6, sky_cubemap_size, sky_cubemap_size)
    sky_cubemap_feature = torch.zeros((*sky_cubemap_shape, feature_dim), device=feature.device)
    sky_cubemap_valid_counts = torch.zeros(sky_cubemap_shape, device=feature.device, dtype=torch.int32)
    for vidx in range(feature.shape[0]):
        resolution = torch.from_numpy(camera_model_parameters[vidx].resolution).to(feature.device)
        with torch.autocast("cuda", enabled=False):
            # Fastest implementation we found. Don't change.
            image_points_return = camera_rays_to_image_points(
                camera_model_parameters[vidx], (sky_rays_d @ R_camera_world[vidx, :3, :3].float()).reshape(-1, 3)
            )
        image_points_valid_inds: torch.Tensor = torch.where(image_points_return.valid_flag)[0]
        valid_samples_uv = (image_points_return.image_points[image_points_valid_inds] / resolution) * 2 - 1
        if feature_mask is not None:
            valid_samples_mask = (
                torch.nn.functional.grid_sample(
                    rearrange(feature_mask[vidx].float(), "H W 1 -> 1 1 H W"),
                    valid_samples_uv[None, None],
                    padding_mode="border",
                    align_corners=False,
                ).reshape(-1)
                > 0.9
            )
            valid_samples_uv = valid_samples_uv[valid_samples_mask]
            image_points_valid_inds = image_points_valid_inds[valid_samples_mask]

        sky_cubemap_feature.view(-1, feature_dim)[image_points_valid_inds] += torch.nn.functional.grid_sample(
            rearrange(feature[vidx], "H W C -> 1 C H W"),
            valid_samples_uv[None, None],
            padding_mode="border",
            align_corners=False,
        )[0, :, 0].T
        sky_cubemap_valid_counts.view(-1)[image_points_valid_inds] += 1

    # Average the features from multiple views
    sky_cubemap_feature /= torch.clamp(sky_cubemap_valid_counts[..., None].float(), min=1e-3)
    sky_cubemap_valid_mask = sky_cubemap_valid_counts > 0

    return sky_cubemap_feature, sky_cubemap_valid_mask[..., None]


def rotate_sky_cubemap(cubemap: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Rotate the cubemap by the given rotation matrix.
    Note that due to aliasing, rotating the cubemap first and then back is not the same as the original.
    Args:
        cubemap: (6, cubemap_size, cubemap_size, C)
        rotation: (3, 3)
    Returns:
        (6, cubemap_size, cubemap_size, C)
    """
    cubemap_size = cubemap.shape[1]
    query_rays = cubemap_ray_directions(cubemap_size, device=cubemap.device) @ rotation.float()
    query_rays = query_rays.reshape(-1, 3)
    opengl_rays_d = torch.stack([query_rays[:, 0], -query_rays[:, 1], query_rays[:, 2]], dim=-1)
    return sample_cubemap_seamless(cubemap, opengl_rays_d).reshape_as(cubemap)


def soften_cubemap_mask(
    cubemap_mask: torch.Tensor,
    dilate_kernel_size: int = 31,
    blur_sigma: float = 8.0,
) -> torch.Tensor:
    """
    Soften a per-face cubemap mask via morphological dilation followed by a Gaussian blur,
    producing smooth weights in [0, 1] that fade out around the original mask edges.

    Each cubemap face is processed independently; smoothing does not cross face seams.

    Note on defaults:
        ``dilate_kernel_size=31`` and ``blur_sigma=8.0`` are tuned for the typical Kelvin
        cubemap face size (~256 px). They produce a softening band of roughly 30-50 px,
        which is a small fraction of a 256-px face. If you call this on a much smaller
        cubemap (e.g. the 16x16 placeholders used in ``KelvinInstantNuRecPrimitive.random``),
        these defaults will dilate the mask to (nearly) the entire face and blur it to a
        roughly-uniform field; pass smaller values explicitly for those use cases.

    Args:
        cubemap_mask: (6, H, W, 1) boolean or float mask.
        dilate_kernel_size: Square structuring element side length (in pixels) for dilation.
            Use <= 1 to skip dilation.
        blur_sigma: Standard deviation (in pixels) of the separable Gaussian blur.
            Use <= 0 to skip blurring.
    Returns:
        (6, H, W, 1) float tensor with values in [0, 1].
    """
    x = cubemap_mask.float().permute(0, 3, 1, 2)  # (6, 1, H, W)
    if dilate_kernel_size > 1:
        pad = dilate_kernel_size // 2
        x = torch.nn.functional.max_pool2d(x, kernel_size=dilate_kernel_size, stride=1, padding=pad)
        # max_pool2d with even kernels can shift the spatial size; crop back to original H, W.
        x = x[..., : cubemap_mask.shape[1], : cubemap_mask.shape[2]]
    if blur_sigma > 0:
        # Defensive ``int(...)``: ``round()`` on a Python ``float`` already returns ``int``,
        # but ``round()`` on a 0-D tensor returns a tensor and ``torch.arange(kernel_size, ...)``
        # below requires a Python int. Keep the cast so future tensor call sites don't silently
        # break here.
        kernel_size = 2 * int(round(3 * blur_sigma)) + 1
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
        kernel_1d = torch.exp(-(coords**2) / (2 * blur_sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        pad = kernel_size // 2
        # Use 'replicate' padding (not conv2d's default zero-padding) so that the Gaussian
        # blur does not attenuate the mask near face boundaries -- zero-padding would create
        # a dark seam of width ~3*sigma around each cubemap face edge. This is not a true
        # cubemap-aware pad across neighboring faces, but it avoids the artifact.
        x = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="replicate")
        x = torch.nn.functional.conv2d(x, kernel_1d.view(1, 1, 1, -1))
        x = torch.nn.functional.pad(x, (0, 0, pad, pad), mode="replicate")
        x = torch.nn.functional.conv2d(x, kernel_1d.view(1, 1, -1, 1))
    return x.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def layout_sky_cubemap(cubemap: torch.Tensor) -> torch.Tensor:
    """
    Layout the cubemap into a 2x3 grid, suitable for visualization. Grid has the following layout:
    [
        [left, front, right],
        [back, bottom, top],
    ]
    Args:
        cubemap: (6, cubemap_size, cubemap_size, 3)
    Returns:
        (2 * cubemap_size, 3 * cubemap_size, 3)
    """
    right, left, top, bottom, front, back = cubemap.unbind(0)
    return torch.cat(
        [
            torch.cat([left, front, right], dim=1),
            torch.cat([back, bottom, top], dim=1),
        ],
        dim=0,
    )
