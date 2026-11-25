# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference entropy for channel-first Mask2Former logits."""

import importlib

from typing import Any

import torch


_native: Any = importlib.import_module("apps.av.aux_gen.mask2former.libmask2former_cc")


# Match the deployed class count and pinned Torch vectorized reduction domain.
_CLASSES = 20
_REDUCTION_WIDTH = 4
_MIN_VECTORIZED_PIXELS = 128


def _quantile(tensor: torch.Tensor, q: float, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    """Sort and interpolate without torch.quantile's 2^24-element limit.

    Args:
        tensor: Floating-point tensor on CPU or CUDA, with any shape/layout.
        q: Scalar quantile in [0,1].
        dim: Dimension to reduce; None flattens the tensor first.
        keepdim: Whether to retain the reduced dimension with size one.

    Returns:
        Quantile values on the input device and dtype, using the legacy
        Python-rank and weighted-sum interpolation arithmetic.
    """
    assert 0 <= q <= 1, "\n\nquantile value should be a float between 0 and 1.\n\n"

    if dim is None:
        tensor = tensor.flatten()
        dim = 0

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    num_elements = sorted_tensor.size(dim)
    index = q * (num_elements - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, num_elements - 1)
    lower_value = sorted_tensor.select(dim, lower_index)
    upper_value = sorted_tensor.select(dim, upper_index)
    # linear interpolation
    weight = index - lower_index
    quantile_value = (1 - weight) * lower_value + weight * upper_value

    return quantile_value.unsqueeze(dim) if keepdim else quantile_value


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return per-pixel entropy, preserving the existing generic/autograd path.

    Args:
        logits: Floating point [C,H,W] class logits. The native inference path
            accepts contiguous CUDA FP32 twenty-class logits whose spatial size
            selects PyTorch's vectorized four-way reduction (>=128, multiple of4).

    Returns:
        [H,W] entropy on the input device, with the input floating dtype.
        Other shapes/layouts/devices and gradient-bearing inputs use PyTorch.
    """
    if logits.ndim != 3:
        raise ValueError("entropy logits must have shape [C,H,W]")
    pixels = logits.shape[1] * logits.shape[2]
    if (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.shape[0] == _CLASSES
        and logits.is_contiguous()
        and not logits.requires_grad
        and pixels >= _MIN_VECTORIZED_PIXELS
        and pixels % _REDUCTION_WIDTH == 0
    ):
        output = logits.new_empty(logits.shape[1:])
        _native.categorical_entropy_out(logits, output)
        return output
    return torch.special.entr(torch.softmax(logits, dim=0)).sum(dim=0)


def mask2former_postprocess(
    logits: torch.Tensor, predictions: torch.Tensor, building_id: int, estimate_logits: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter the most uncertain pixels using the existing 90th-percentile rule.

    Args:
        logits: FP32 [C,H,W] class logits; uncertain pixels are overwritten
            with the building one-hot vector only when estimate_logits is true.
        predictions: Integer [1,H,W] labels on the logits device.
        building_id: Fallback class index in [0,C).
        estimate_logits: Whether to update logits as well as labels.

    Returns:
        uint8 [H,W] labels, the input logits, and bool [H,W] uncertainty mask.
        The CUDA twenty-class specialization returns fresh labels. Other inputs
        retain the generic PyTorch path and its existing label aliasing behavior.
        Native selection uses pinned Torch FP32 rank/lerp at every size.
        The generic path retains the legacy large-image interpolation rule.
    """
    pixels = logits.shape[1] * logits.shape[2]
    if (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.shape[0] == _CLASSES
        and logits.is_contiguous()
        and not logits.requires_grad
        and predictions.device == logits.device
        and predictions.is_contiguous()
        and predictions.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        and tuple(predictions.shape) == (1, *logits.shape[1:])
        and pixels >= _MIN_VECTORIZED_PIXELS
        and pixels % _REDUCTION_WIDTH == 0
    ):
        labels = predictions.new_empty(logits.shape[1:], dtype=torch.uint8)
        mask = predictions.new_empty(logits.shape[1:], dtype=torch.bool)
        entropy = logits.new_empty(logits.shape[1:])
        _native.mask2former_postprocess_out(logits, predictions, building_id, estimate_logits, entropy, labels, mask)
        return labels, logits, mask
    entropy = categorical_entropy(logits)
    quantile = 0.90
    # torch.quantile's FP32 rank limit; larger inputs use the legacy helper rule.
    torch_quantile_limit = 1 << 24
    if pixels <= torch_quantile_limit:
        threshold = torch.quantile(entropy, quantile)
    else:
        threshold = _quantile(entropy, quantile)
    mask = entropy > threshold
    labels = predictions.to(dtype=torch.uint8).squeeze(0)
    labels[mask] = building_id
    if estimate_logits:
        logits[:, mask] = 0
        logits[building_id, mask] = 1
    return labels, logits, mask
