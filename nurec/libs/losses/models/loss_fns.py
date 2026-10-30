# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from gsplat.losses import BG_GRID_NUM_COLS, BG_GRID_NUM_ROWS
from gsplat.losses import functional as gsplat_fn
from ppisp import PPISP  # type: ignore[import-not-found]

from nurec.libs.losses.models.registry import register_loss_fn
from nurec.libs.losses.models.utils import torch_ssim
from nurec.models.custom_modules import trunc_bce
from nurec.utils.lpips_network import LPIPSNetwork
from nurec.utils.packed_ops import packed_weighted_sum


register_loss_fn("bce_clipped")(gsplat_fn.bce_clipped)


@register_loss_fn("bce_truncated")
def bce_loss_truncated(input: torch.Tensor, target: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """
    Returns equivalent results as `bce_loss_clipped`,
    only that the clipped-out areas will still hold clipped gradients to avoid dead zone.
    See the documentation in `trunc_bce` for more details.
    """
    return trunc_bce(input, target, eps)


def dirac_delta_approx(x, mu=0, sigma=1e-5):
    """
    Approximates the Dirac delta function with a Gaussian distribution.

    Args:
        x (torch.Tensor): The input tensor.
        mu (float, optional): The mean of the Gaussian distribution. Defaults to 0.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 1e-5.

    Returns:
        torch.Tensor: The output tensor.
    """
    return (1 / (math.sqrt(2 * torch.pi * sigma**2))) * torch.exp(-((x - mu) ** 2) / (2 * sigma**2))


@register_loss_fn("los")
def line_of_sight_loss(
    rays_mask: torch.Tensor,
    gt_depth: torch.Tensor,
    weights: torch.Tensor,
    t_vals: torch.Tensor,
    pack_info: torch.Tensor,
    epsilon: float,
    depth_upper_bound: float | None = None,
) -> torch.Tensor:
    # evaluate sample-specific values unconditionally on all samples (unmasked)
    gt_depth_samples, t_vals = (
        gt_depth.squeeze().repeat_interleave(pack_info[:, 1], dim=0),
        t_vals.squeeze().detach(),
    )  # currently t_vals don't propagate gradients, still explicitly detaching to future-proof
    depth_mask = gt_depth_samples > 0

    # apply depth_upper_bound
    if depth_upper_bound is not None:
        depth_mask &= gt_depth_samples < depth_upper_bound

    empty_mask = t_vals < gt_depth_samples - epsilon
    near_mask = (t_vals > (gt_depth_samples - epsilon)) & (t_vals < gt_depth_samples + epsilon)

    empty_mask &= depth_mask
    near_mask &= depth_mask

    # evaluate per-ray loss values on the masked active rays only
    empty_loss = packed_weighted_sum(empty_mask.unsqueeze(-1).float(), weights.square(), pack_info[rays_mask, :])
    near_loss = packed_weighted_sum(
        near_mask.unsqueeze(-1).float(),
        (weights - dirac_delta_approx(t_vals - gt_depth_samples, sigma=epsilon / 3)).square(),
        pack_info[rays_mask, :],
    )

    """
    The far_loss term is not used as it's not needed for training.
    see (Eq 18) of the "Urban Radiance Fields" paper, as well as EmerNeRF implementation
    https://github.com/NVlabs/EmerNeRF/commit/17386ee765de69f9b815115af7930a464f2a3404#diff-2dba32fe1191d4d3b5eac91d8a1ba37d7a238fb6dc16f7abe4c718ceed6c2390R462

    far_mask = t_vals > gt_depth + epsilon
    far_mask &= depth_mask
    far_loss = torch.mean(packed_weighted_sum(
        far_mask.unsqueeze(-1).float(),
        weights.square(),
        pack_info
    )) # type: ignore
    """

    depth_error = empty_loss + near_loss
    return depth_error


register_loss_fn("weights_reg")(gsplat_fn.weights_reg)


register_loss_fn("relu_sum")(gsplat_fn.relu_sum)


@register_loss_fn("total_variation_spatial")
def total_variation_spatial(x: torch.Tensor, wrap_around: bool = False) -> torch.Tensor:
    """
    Compute total variation loss for a tensor x across spatial dimensions.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        wrap_around: If True, use cyclic differences across spatial boundaries.

    Returns:
        tv_loss (torch.Tensor): Total variation loss, a tensor of shape (B,).
    """

    def diff(dim: int) -> torch.Tensor:
        return x - torch.roll(x, shifts=-1, dims=dim) if wrap_around else x.diff(dim=dim)

    tv_z = diff(dim=2).square().mean(dim=(1, 2, 3, 4)) if x.shape[2] > 1 else x.new_zeros(x.shape[0])
    tv_y = diff(dim=3).square().mean(dim=(1, 2, 3, 4)) if x.shape[3] > 1 else x.new_zeros(x.shape[0])
    tv_x = diff(dim=4).square().mean(dim=(1, 2, 3, 4)) if x.shape[4] > 1 else x.new_zeros(x.shape[0])

    return tv_z + tv_y + tv_x


register_loss_fn("total_variation_temporal")(gsplat_fn.total_variation_temporal)


@register_loss_fn("identity_distance")
def identity_distance(grid: torch.Tensor) -> torch.Tensor:
    """Frobenius distance of the background grid from the identity transform.

    gsplat's defaults are the generic 3x4 affine shape, not this grid's layout,
    so the layout is passed explicitly.
    """
    return gsplat_fn.identity_distance(grid, BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS)


@register_loss_fn("ssim")
def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: torch.Tensor,
    mask_value: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
) -> torch.Tensor:
    img1 = img1 * mask + mask_value * (~mask)
    img2 = img2 * mask + mask_value * (~mask)

    ssim_map = torch_ssim(img1, img2, window=window, window_size=window_size, channel=channel)

    # Filter out the SSIM map with the mask so that the loss and hence bakward will only be applied to valid pixels
    # Average across the channels to obtain per ray SSIM loss and flatten it out
    num_mask = mask.sum()
    scale_factor = torch.where(
        num_mask > 0, 1 / num_mask, 0
    )  # pre-scale the mask to compute masked mean without using loss[mask], avoiding CUDA synchronization
    return (1 - ssim_map).mean(dim=1, keepdim=True) * mask * scale_factor


@register_loss_fn("depth_inverse_mse")
def depth_inverse_mse_loss(
    pred_value: torch.Tensor,
    target_value: torch.Tensor,
    eps: float,
    **kwargs,
) -> torch.Tensor:
    """Thin shim: the registry calls loss fns with keyword extras gsplat does not take."""
    return gsplat_fn.depth_inverse_mse(pred_value, target_value, eps)


@register_loss_fn("log_l1")
def log_l1(pred_value: torch.Tensor, target_value: torch.Tensor, **kwargs) -> torch.Tensor:
    """Thin shim: the registry calls loss fns with keyword extras gsplat does not take."""
    return gsplat_fn.log_l1(pred_value, target_value)


@register_loss_fn("lpips")
def lpips_loss(img1: torch.Tensor, img2: torch.Tensor, lpips: LPIPSNetwork) -> torch.Tensor:
    # [B, 1, 1, 1] -> [B]
    return lpips.forward(img1, img2).view(-1)


register_loss_fn("normal_cosine")(gsplat_fn.normal_cosine_loss)


@register_loss_fn("ppisp_loss")
def ppisp_loss(ppisp_module: PPISP) -> torch.Tensor:
    """Compute regularization loss for PPISP via the library's built-in method.

    Delegates to ``ppisp.PPISP.get_regularization_loss()`` which computes weighted
    regularization terms for exposure, vignetting, color correction, and CRF
    parameters.  The per-component weights are configured through ``PPISPConfig``.

    Args:
        ppisp_module: A ``ppisp.PPISP`` instance that exposes ``get_regularization_loss()``.

    Returns:
        Scalar regularization loss.
    """
    return ppisp_module.get_regularization_loss()
