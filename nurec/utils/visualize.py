# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from typing import Dict, Optional, Sequence

import flow_vis
import imageio
import matplotlib as mpl
import numpy as np
import torch

from einops import rearrange
from lightning_fabric.loggers.logger import Logger
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger


logger = logging.getLogger(__name__)


def scalar2img(
    scalar: np.ndarray,
    vmin: float | None = 0.1,
    vmax: float | None = 10.0,
    cmap: str = "turbo",
) -> np.ndarray:
    """
    Converts a (H, W) array into an RGB array suitable for viewing (by applying the colormap).

    Args:
        - scalar: 2d (H, W) array of scalars to be converted by a colormap
        - vmin: the value to correspond to start of the colormap (lower values clip), None uses `scalar.min()`
        - vmax: the value to correspond to end of the colormap (higher values clip), None uses `scalar.max()`
        - cmap: name of the matplotlib colormap to be used (see https://matplotlib.org/stable/users/explain/colors/colormaps.html)

    Returns:
        - A (H, W, 3) uint8 array of rgb colors
    """
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    # TODO: check why this doesn't work with type annotation
    colored_distance = (mapper.to_rgba(scalar)[:, :, :3] * 255).astype(np.uint8)  # type: ignore
    return colored_distance


def scalar2rgb(
    scalar: np.ndarray,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    cmap: str = "bwr",
) -> np.ndarray:
    """
    Converts a array into an RGB array suitable for viewing (by applying the colormap).

    Args:
        - scalar: (N, 1) array of scalars to be converted by a colormap
        - vmin: the value to correspond to start of the colormap (lower values clip), None uses `scalar.min()`
        - vmax: the value to correspond to end of the colormap (higher values clip), None uses `scalar.max()`
        - cmap: name of the matplotlib colormap to be used (see https://matplotlib.org/stable/users/explain/colors/colormaps.html)

    Returns:
        - A (N, 3) float16 array of rgb colors
    """
    normalizer = mpl.colors.Normalize(vmin=np.percentile(scalar, 5), vmax=np.percentile(scalar, 99))
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colored_distance = (mapper.to_rgba(scalar)[:, :3]).astype(np.float16)  # type: ignore
    return colored_distance


def sem2img(sem_label: np.ndarray, color_remap: np.ndarray) -> np.ndarray:
    """Map a (H, W) class-id array to (H, W, 3) uint8 RGB via `color_remap[id]`."""
    return color_remap[sem_label].astype(np.uint8, copy=False)


# Default LUT size when ``semantic_class_names`` is absent (class id -> row; golden-ratio hues).
SEMANTICS_COLOR_LUT_DEFAULT_SIZE = 256


def build_semantics_class_color_lut(n: int) -> np.ndarray:
    """(n, 3) uint8 RGB: one color per class id, deterministic (matches CLI ``render --output-type semantics-colorized``).

    Uses golden-ratio spacing on the hue wheel so consecutive class indices are easy to tell apart;
    light saturation/value dithering adds separation when many classes are used.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    phi = 0.6180339887498949
    i = np.arange(n, dtype=np.float64)
    h = (i * phi) % 1.0
    s = np.clip(0.72 + 0.2 * ((i % 5) / 4.0), 0.0, 1.0)
    v = np.clip(0.78 + 0.2 * (((i // 5) % 3) / 2.0), 0.0, 1.0)
    hsv = np.stack([h, s, v], axis=-1)
    rgb = mpl.colors.hsv_to_rgb(hsv)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def build_semantics_class_colors(lut_size: int, artifact_class_colors: np.ndarray | None) -> np.ndarray:
    """Build a (lut_size, 3) uint8 RGB LUT for semantic visualization.

    Rows ``0 .. min(lut_size, N)-1`` are taken from ``artifact_class_colors`` when it is a valid
    ``(N, 3) uint8`` array (NCORE ``stuff_colors`` / validation colormap). Remaining rows use
    :func:`build_semantics_class_color_lut` so overflow class ids still get distinct colors.
    """
    lut = build_semantics_class_color_lut(lut_size)
    if artifact_class_colors is None or artifact_class_colors.size == 0:
        return lut
    ac = np.asarray(artifact_class_colors, dtype=np.uint8)
    if ac.ndim != 2 or ac.shape[1] != 3:
        return lut
    n = min(lut_size, ac.shape[0])
    if n > 0:
        lut[:n] = ac[:n]
    return lut


# Legend strip size when drawing semantics legend (swatch + labels).
_SEG_LEGEND_SWATCH_SIZE = 20
_SEG_LEGEND_PAD = 8
_SEG_LEGEND_FONT_SIZE = 12


def _load_semantics_legend_font(size: int):
    """Load DejaVu Sans from matplotlib's bundled fonts (hermetic; ships with the wheel).

    Avoids depending on host system fonts (``/usr/share/fonts/...`` /
    ``/System/Library/Fonts/...``) which are absent from minimal containers and
    Bazel runfiles. Falls back to ``ImageFont.load_default()`` only if matplotlib's
    bundled font is itself missing — i.e. a broken matplotlib install.
    """
    font_path = os.path.join(mpl.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(font_path, size)
    except OSError as exc:
        logger.warning(
            "Could not load matplotlib's bundled DejaVuSans (%s); falling back to PIL default — "
            "legend may have incorrect proportions.",
            exc,
        )
        return ImageFont.load_default()


def create_semantics_legend(
    class_colors: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Create a legend image for semantics class colors.

    Args:
        class_colors: (num_classes, 3) uint8 colors for classes 0..num_classes-1.
        class_names: Optional ``{class_id: name}`` mapping. Missing ids fall back to ``str(i)``.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    num_classes = int(class_colors.shape[0])
    if num_classes <= 0:
        return np.full((1, 1, 3), 255, dtype=np.uint8)

    swatch = _SEG_LEGEND_SWATCH_SIZE
    pad = _SEG_LEGEND_PAD
    font = _load_semantics_legend_font(_SEG_LEGEND_FONT_SIZE)
    _bb = font.getbbox("0")
    line_height = max(swatch, int(_bb[3] - _bb[1]) + 2)
    if class_names is not None:
        text_w = 0
        for i in range(num_classes):
            label = class_names.get(i, str(i))
            bbox = font.getbbox(f"{i}: {label}")
            text_w = max(text_w, int(bbox[2] - bbox[0]))
        legend_width = swatch + pad + max(text_w + 8, 100)
    else:
        legend_width = swatch + pad + 80

    legend_h = int(pad + (num_classes - 1) * (line_height + 2) + swatch + pad)
    out = np.full((legend_h, legend_width, 3), 255, dtype=np.uint8)
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    for i in range(num_classes):
        y0 = pad + i * (line_height + 2)
        y1 = y0 + swatch
        x0 = pad
        x1 = x0 + swatch
        draw.rectangle([x0, y0, x1, y1], fill=tuple(int(c) for c in class_colors[i]))
        label = class_names.get(i, str(i)) if class_names is not None else str(i)
        draw.text((x1 + 4, y0), f"{i}: {label}", fill=(0, 0, 0), font=font)
    return np.asarray(pil)


def flow2img(flow: np.ndarray, rad_max: float | None = None) -> np.ndarray:
    """
    Converts a UV flow field into an RGB image suitable for viewing.

    Args:
        - flow: (H, W, 2) array of UV flow vectors
        - rad_max: the maximum magnitude of the flow vectors, None uses max of flow radius
    """

    assert flow.ndim == 3, "input flow must have three dimensions"
    assert flow.shape[2] == 2, "input flow must have shape [H,W,2]"
    # Sanitize NaN/Inf: when upstream heads diverge during training the input flow can
    # contain NaN/Inf which become huge ints in `flow_vis.flow_uv_to_colors`, raising
    # IndexError. Replace with 0 so the visualization stays valid (best-effort frame).
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))

    if rad_max is None:
        rad_max = np.max(rad).item()
    else:
        # Renormalize the flow vectors
        renorm_mask = rad > rad_max
        u[renorm_mask] = u[renorm_mask] * (rad_max / rad[renorm_mask])
        v[renorm_mask] = v[renorm_mask] * (rad_max / rad[renorm_mask])

    u = u / (rad_max + 1e-5)
    v = v / (rad_max + 1e-5)
    return flow_vis.flow_uv_to_colors(u, v, False)


def instance2color(instance, color_map, h, w):
    n_inst = len(instance["scores"])
    instance_color = np.full((h, w, 3), 255, dtype=np.uint8)
    for inst in range(n_inst):
        instance_color[np.where(instance["instance_masks"][inst])] = color_map[instance["classes"][inst]]
    return instance_color


def generate_colors(
    values: np.ndarray,
    cmap_name: str = "jet_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    return_hex: bool = False,
) -> np.ndarray:
    """Convert a range of values to the given color map.

    Args:
        values: a 1d array of values used to generate the color map, shape (num_values,).
        cmap_name: any color map of matplotlib.
        vmin: lower bound of the normalisation range used on the values: [vmin, vmax].
        vmax: upper bound of the normalisation range used on the values: [vmin, vmax].
        return_hex: if True, return the color map as hex values.

    Returns:
        colors as floats in the range [0, 1], shape (num_values, 3[r, g, b]).
            If `return_hex` is True: return hex values as string e.g. '#000080', shape (num_values,).
    """

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba_colors = cmap(norm(values))
    if return_hex:
        return np.array([mpl.colors.rgb2hex(v) for v in rgba_colors])
    return rgba_colors[:, :3]  # remove the final column which is a constant alpha=1 value.


def save_im(im: torch.Tensor, h: int, filename: str) -> None:
    """Save an image to a file.

    Args:
        im (torch.Tensor): The image to save, tensor with shape [(h w) c].
        h (int): The height of the image.
        filename (str): The filename to save the image to.
    """
    np_im: np.ndarray = rearrange(im.cpu().numpy(), "(h w) c -> h w c", h=h)
    np_im = (np_im * 255).astype(np.uint8)
    imageio.v2.imsave(filename, np_im)


def log_image(
    logger: Logger | None,
    image: torch.Tensor | np.ndarray,
    caption: str,
    step: int | None = None,
) -> None:
    """Logs an image to the specified logger if available.

    Args:
        logger (Logger | None): Logger instance, or None to skip logging
        image (torch.Tensor | np.ndarray): Image to log
        caption (str): Name/tag for the image
        step (Optional[int]): Current training step
    """

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image.copy())
        if image.dim() == 3:
            image = rearrange(image, "h w c -> c h w")
        elif image.dim() == 2:
            image = rearrange(image, "h w -> 1 h w")
    assert isinstance(image, torch.Tensor) and image.dim() == 3

    match logger:
        case WandbLogger():
            logger.log_image(caption, [image], step=step)

        case TensorBoardLogger():
            logger.experiment.add_image(caption, image, step, dataformats="CHW" if image.dim() == 3 else "HW")

        case DummyLogger() | None:
            pass

        case _:
            # log_image now accepts any lightning_fabric Logger; an unhandled
            # subtype would otherwise fall through and silently drop the image.
            logging.getLogger(__name__).warning(
                "log_image: unsupported logger type %s; image %r not logged",
                type(logger).__name__,
                caption,
            )


def make_image_grid(images: list[np.ndarray], grid_width: int, subsample: int = 1) -> np.ndarray:
    """
    Arrange a list of images into a grid with given width.

    Args:
        images (list of np.ndarray): List of images as numpy arrays (H, W, C or H, W).
        grid_width (int): Number of images per row in the grid.
        subsample (int): Stride-subsample each tile before compositing (default 1 = no-op).

    Returns:
        np.ndarray: The resulting grid image.
    """
    if subsample > 1:
        min_h = min(img.shape[0] for img in images)
        min_w = min(img.shape[1] for img in images)
        subsample = max(1, min(subsample, min_h, min_w))
        images = [img[::subsample, ::subsample] for img in images]

    # Pre-process dimensions in a single pass
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    channels = max(img.shape[2] if img.ndim == 3 else 1 for img in images)

    # Calculate grid dimensions
    num_images = len(images)
    grid_height = (num_images + grid_width - 1) // grid_width

    # Pre-allocate the output grid
    grid_image = np.zeros(
        (grid_height * max_height, grid_width * max_width, channels),
        dtype=images[0].dtype,
    )

    for idx, img in enumerate(images):
        # Convert to 3D if needed
        if img.ndim == 2:
            img = img[..., None]

        h, w = img.shape[:2]
        row, col = divmod(idx, grid_width)

        # Calculate center offsets
        y_offset = (max_height - h) // 2
        x_offset = (max_width - w) // 2

        # Direct slice assignment (no intermediate padding)
        y_start = row * max_height + y_offset
        x_start = col * max_width + x_offset
        grid_image[y_start : y_start + h, x_start : x_start + w, : img.shape[2]] = img

    # If single-channel, squeeze to remove unnecessary dimensions
    return grid_image.squeeze(-1) if channels == 1 else grid_image


def save_video(
    video_file: str,
    images: Sequence[np.ndarray],
    fps: int = 30,
    macro_block_size: int = 1,
    format: str = "mp4",
) -> None:
    """Save a sequence of images as a video file, ensuring images have even dimensions for ffmpeg.

    Args:
        video_file (str): Path to save the video file.
        images (list[np.ndarray]): List of images to be saved as video frames.
        fps (int, optional): Frames per second for the video. Defaults to 30.
        macro_block_size (int, optional): Macro block size for the video codec. Defaults to 1.
        format (str, optional): Video format. Defaults to "mp4".
    """
    # Chop images if dimensions not divisible by 2 to fix ffmpeg errors
    processed_images = []
    unaligned_warning = False
    for img in images:
        # Remove last row if height is odd
        if img.shape[0] % 2 == 1:
            img = img[:-1, :]
        # Remove last column if width is odd
        if img.shape[1] % 2 == 1:
            img = img[:, :-1]
        processed_images.append(img)

        if img.shape[0] % 16 != 0:
            unaligned_warning = True

    if unaligned_warning:
        logger.warning("Video width not multiple of 16 - ffmpeg will warn about unaligned data")

    # Results in an Array is not the same as ArrayLike annotation error, so we suppress it
    imageio.v2.mimwrite(video_file, processed_images, fps=fps, macro_block_size=macro_block_size, format=format)  # type: ignore
    logger.info(f"Saved {len(images)} frames into '{video_file}'")


def draw_text_overlay(im: np.ndarray, overlay_text: str):
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(im)
    draw = ImageDraw.Draw(pil_img)

    # Try to use a default font, fallback to default if not available
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
        except Exception:
            font = ImageFont.load_default()

    # Get text size for positioning
    bbox = draw.textbbox((0, 0), overlay_text, font=font)
    text_width = bbox[2] - bbox[0]

    # Position text in top right corner with padding
    x = im.shape[1] - text_width - 15
    y = 15

    # Draw text with black outline for better visibility
    outline_color = (0, 0, 0)
    text_color = (255, 255, 255)

    # Draw outline
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), overlay_text, font=font, fill=outline_color)

    # Draw main text
    draw.text((x, y), overlay_text, font=font, fill=text_color)

    # Convert back to numpy array and update the original array
    im[:] = np.array(pil_img)


DEFAULT_PS_VIEW_JSON_STR = """
{
    "farClipRatio": 1000.0,
    "nearClipRatio": 0.005,
    "fov": 45.0,
    "projectionMode": "Perspective",
    "viewMat": [0.0, -1.0, 0.0, -0.0, -0.0, 0.0, 1.0, -0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "windowHeight": 1080,
    "windowWidth": 1920
}"""
