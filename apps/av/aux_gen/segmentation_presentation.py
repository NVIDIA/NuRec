# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only conversion and visualization helpers for segmentation outputs."""

from pathlib import Path
from typing import Any

import numpy as np

from numpy.typing import NDArray
from PIL import Image as PILImage


def is_dark_image(image: Any) -> bool:
    """Return whether a host image should skip promptless SAM2 inference.

    Args:
        image: RGB-compatible host image with shape ``[H, W, C]``.

    Returns:
        Whether the stride-16 sample has mean intensity below eight.
    """
    image_array = np.asarray(image)
    return bool(image_array[::16, ::16].mean(axis=(0, 1)).mean() < 8)


def labels_to_paletted_image(labels: NDArray[Any]) -> PILImage.Image:
    """Convert semantic labels to the legacy paletted host representation."""
    semantic_seg = PILImage.fromarray(np.asarray(labels, dtype=np.uint8), mode="P")
    semantic_seg.putpalette(np.linspace(0, 255, 256, dtype=np.uint8).tolist())
    return semantic_seg


def save_mask2former_visualization(
    image_rgb: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    stuff_colors: NDArray[np.uint8],
    vis_path: Path,
    uncertainty_mask: NDArray[Any] | None = None,
) -> None:
    """Save semantic labels blended over RGB and optional uncertainty.

    Args:
        image_rgb: Source RGB image with shape ``[H, W, 3]``.
        labels: Semantic class IDs with shape ``[H, W]``.
        stuff_colors: RGB class palette with shape ``[C, 3]``.
        vis_path: Extension-free path identifying the output frame.
        uncertainty_mask: Optional boolean-compatible mask with shape
            ``[H, W]``. It is saved beside the semantic overlay.

    Returns:
        None.

    Raises:
        ValueError: If image, labels, uncertainty, or palette have
            incompatible shapes, or labels reference a class outside the
            palette.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"Mask2Former visualization expects RGB [H, W, 3], got {image_rgb.shape}")
    if labels.shape != image_rgb.shape[:2]:
        raise ValueError(f"Mask2Former visualization labels {labels.shape} do not match image {image_rgb.shape[:2]}")
    if stuff_colors.ndim != 2 or stuff_colors.shape[1] != 3:
        raise ValueError(f"Mask2Former visualization expects an RGB class palette, got {stuff_colors.shape}")
    if labels.size and int(labels.max()) >= len(stuff_colors):
        raise ValueError(f"Mask2Former visualization label {int(labels.max())} exceeds {len(stuff_colors)} classes")

    semantic_rgb = stuff_colors[labels]
    blended = np.rint(image_rgb.astype(np.float32) * 0.5 + semantic_rgb.astype(np.float32) * 0.5).astype(np.uint8)
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(blended, mode="RGB").save(vis_path.with_suffix(".png"))
    if uncertainty_mask is not None:
        if uncertainty_mask.shape != labels.shape:
            raise ValueError(
                f"Mask2Former uncertainty mask {uncertainty_mask.shape} does not match labels {labels.shape}"
            )
        uncertainty_image = np.asarray(uncertainty_mask, dtype=np.bool_).astype(np.uint8) * 255
        PILImage.fromarray(uncertainty_image, mode="L").save(vis_path.parent / f"{vis_path.name}_uncertainty_mask.png")
