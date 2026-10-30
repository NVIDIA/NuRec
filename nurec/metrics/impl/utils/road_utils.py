# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless helpers for road-pixel quality assessment.

Four sub-metrics scoring a rendered image against ground truth, restricted
to a road mask. Each exposes a ``compute_<name>(gt, rendered, valid_mask,
**kw)`` plus a matching ``debug_<name>(...)`` for visualisations.

Core Metrics:

- Edge fidelity (``ref``): adaptive Canny edges, then P/R/F1 within a
  pixel tolerance plus symmetric Chamfer distance.
- Local-contrast fidelity (``rlcf``): sliding-window std-dev correlation,
  preservation ratio, and high-contrast pixel fractions.
- Lane-marking detection (``lm``): adaptive bright-on-dark mask
  (``pixel > local_mean + k · local_std``) with morphological cleanup,
  then P/R/F1/IoU on the binary masks.
- Multi-scale texture fidelity (``rmtf``): Difference-of-Gaussians band
  energies with per-band ratios (clipped at 2.0) and a weighted aggregate.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt
from torchvision.transforms.functional import rgb_to_grayscale


log = logging.getLogger(__name__)


# ---- Constants ---------------------------------------------------------
_EPS = 1e-12  # divide-by-zero guard

# Per-DoG-band weights, finest first. Tuned for n_scales=4;
# compute_rmtf renormalises the slice when n_scales < 4.
_RMTF_BAND_WEIGHTS: tuple[float, ...] = (0.15, 0.30, 0.30, 0.25)


# ---- Shared helpers ----------------------------------------------------


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) RGB to (H, W) grayscale; pass (H, W) input through.

    Args:
        img: Image as [H, W, 3] RGB or [H, W] grayscale.

    Returns:
        Grayscale image [H, W], same dtype as input.

    Raises:
        ValueError: if ``img`` is neither (H, W) nor (H, W, 3).
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"_to_gray expects (H, W) or (H, W, 3), got shape {img.shape}")


def _nan_dict(keys: tuple[str, ...] | list[str]) -> dict[str, float]:
    """Build an all-NaN result dict for the given keys.

    Used by the empty-mask early-return branches of the ``compute_*``
    helpers so the output schema is preserved when no road pixels exist.

    Args:
        keys: Output-dict keys for this sub-metric.

    Returns:
        Mapping ``{key: float('nan')}`` for every key in ``keys``.
    """
    return {k: float("nan") for k in keys}


def select_viz_frames(frame_names: list[str], every_nth: int = 30) -> set[str]:
    """Pick which frames of a clip to visualise (each costs many full-res debug images).

    Randomly samples ~1 in ``every_nth`` frames but at least 10; clips of up
    to 10 frames are returned in full, and ``every_nth <= 0`` disables
    sampling. The RNG seed is fixed, so reruns select the same frames.
    Not road-specific — move to ``nurec.metrics.utils`` if another metric
    family adopts it.

    Args:
        frame_names: Frame identifiers of the clip.
        every_nth: Average number of clip frames per visualised frame.

    Returns:
        The subset of ``frame_names`` to visualise.
    """
    if every_nth <= 0 or len(frame_names) <= 10:
        return set(frame_names)
    n_viz = min(len(frame_names), max(10, round(len(frame_names) / every_nth)))
    rng = np.random.default_rng(0)
    return set(rng.choice(sorted(frame_names), size=n_viz, replace=False))


# ---- Input adapters and filtering primitives ---------------------------


def _resolve_device(device: str | torch.device | None, *arrays: np.ndarray | torch.Tensor) -> torch.device:
    """Pick the compute device: explicit arg > first input tensor's device > auto.

    Args:
        device: Explicit device request, or None to follow the inputs.
        arrays: Candidate inputs; the first torch.Tensor wins.

    Returns:
        The torch.device to run the metric on.
    """
    if device is not None:
        return torch.device(device)
    for a in arrays:
        if isinstance(a, torch.Tensor):
            return a.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_gray_tensor(img: np.ndarray | torch.Tensor, device: torch.device, name: str) -> torch.Tensor:
    """Normalise an image to a (H, W) float32 grayscale tensor on ``device``.

    Accepts numpy or torch in (H, W), (H, W, 3), (3, H, W) or (1, 3, H, W)
    layout.

    Args:
        img: Image array/tensor, float values in [0, 1].
        device: Target device.
        name: Argument name for error messages.

    Returns:
        (H, W) float32 tensor on ``device``.

    Raises:
        TypeError: if ``img`` is neither np.ndarray nor torch.Tensor.
        ValueError: if the layout is not one of the accepted shapes.
    """
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(img))
    elif isinstance(img, torch.Tensor):
        t = img.detach()
    else:
        raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(img).__name__}")
    t = t.to(device=device, dtype=torch.float32)
    if t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 2:
        return t
    if t.ndim == 3 and t.shape[0] == 3:  # CHW
        return rgb_to_grayscale(t).squeeze(0)
    if t.ndim == 3 and t.shape[-1] == 3:  # HWC
        return rgb_to_grayscale(t.permute(2, 0, 1)).squeeze(0)
    raise ValueError(f"{name} must be (H, W), (H, W, 3) or (3, H, W); got shape {tuple(img.shape)}")


def _as_mask_tensor(mask: np.ndarray | torch.Tensor, device: torch.device, name: str = "valid_mask") -> torch.Tensor:
    """Normalise a road mask to a (H, W) boolean tensor on ``device``.

    Args:
        mask: Boolean-ish mask, numpy or torch, (H, W) or (1, H, W).
        device: Target device.
        name: Argument name for error messages.

    Returns:
        (H, W) bool tensor on ``device``.

    Raises:
        TypeError: if ``mask`` is neither np.ndarray nor torch.Tensor.
        ValueError: if the mask is not 2-D after squeezing a leading 1.
    """
    if isinstance(mask, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(mask))
    elif isinstance(mask, torch.Tensor):
        t = mask.detach()
    else:
        raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(mask).__name__}")
    t = t.to(device=device)
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 2:
        raise ValueError(f"{name} must be (H, W); got shape {tuple(mask.shape)}")
    return t.bool()


def _gaussian_kernel1d(ksize: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Build a normalised 1-D Gaussian kernel.

    Args:
        ksize: Kernel length (odd).
        sigma: Gaussian standard deviation.
        device: Target device.

    Returns:
        (ksize,) float64 tensor summing to 1.
    """
    x = torch.arange(ksize, dtype=torch.float64, device=device) - (ksize - 1) / 2.0
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_blur_t(x: torch.Tensor, sigma: float, ksize: int | None = None) -> torch.Tensor:
    """Separable Gaussian blur of a (1, 1, H, W) tensor with reflect padding.

    Hand-rolled on ``conv2d`` instead of ``torchvision.gaussian_blur``, which
    materialises the full 2-D kernel — O(k²) work per pixel vs O(k) separable,
    prohibitive at the ~100-tap kernels these metrics use. The kernel
    follows ``x``'s dtype.

    Args:
        x: Input tensor (1, 1, H, W).
        sigma: Gaussian standard deviation.
        ksize: Kernel size; None auto-sizes to ``round(6 * sigma + 1) | 1``.
            The kernel is clamped to the image size so small inputs stay valid.

    Returns:
        Blurred tensor (1, 1, H, W).
    """
    if ksize is None:
        ksize = int(round(sigma * 6 + 1)) | 1
    max_ksize = min(x.shape[-2:])
    if max_ksize % 2 == 0:
        max_ksize -= 1
    ksize = max(3, min(ksize, max_ksize))
    pad = ksize // 2
    kernel = _gaussian_kernel1d(ksize, sigma, x.device).to(x.dtype)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    # The vertical pass runs as a horizontal conv on the transposed tensor:
    # a (ksize, 1) kernel hits a much slower CUDA path in float64.
    x = F.conv2d(x.transpose(-1, -2).contiguous(), kernel.view(1, 1, 1, -1))
    return x.transpose(-1, -2).contiguous()


def _box_mean_t(x: torch.Tensor, ksize: int) -> torch.Tensor:
    """Box-filter mean of a (1, 1, H, W) tensor with reflect padding.

    Args:
        x: Input tensor (1, 1, H, W).
        ksize: Side length of the square window.

    Returns:
        Filtered tensor (1, 1, H, W).
    """
    pad_before, pad_after = ksize // 2, ksize - 1 - ksize // 2
    x = F.pad(x, (pad_before, pad_after, pad_before, pad_after), mode="reflect")
    return F.avg_pool2d(x, ksize, stride=1)


def _local_std_t(gray: torch.Tensor, window_size: int) -> torch.Tensor:
    """Per-pixel local std-dev via the box-filter identity ``Var(x) = E[x²] − E[x]²``.

    Runs in float64: the ``E[x²] − E[x]²`` cancellation loses too much
    precision in float32, and the f64 box filter is bandwidth-bound,
    not compute-bound.

    Args:
        gray: Grayscale tensor (H, W).
        window_size: Side length of the square sliding window.

    Returns:
        (H, W) float64 tensor of per-pixel std-deviations.
    """
    x = gray[None, None].double()
    mean = _box_mean_t(x, window_size)
    mean_sq = _box_mean_t(x * x, window_size)
    variance = (mean_sq - mean * mean).clamp_min_(0.0)
    return variance.sqrt_()[0, 0]


def _binary_open_t(mask: torch.Tensor, ksize: int) -> torch.Tensor:
    """Morphological open (erode then dilate) of a boolean (H, W) tensor.

    Erosion pads with True and dilation with False, so pixels at the image
    border behave as if the mask continued unchanged past it.

    Args:
        mask: Boolean tensor (H, W).
        ksize: Side length of the rectangular structuring element.

    Returns:
        Opened boolean tensor (H, W).
    """
    if ksize == 1:
        return mask
    pad_before, pad_after = ksize // 2, ksize - 1 - ksize // 2
    m = mask[None, None].float()
    m = F.pad(m, (pad_before, pad_after, pad_before, pad_after), value=1.0)
    m = -F.max_pool2d(-m, ksize, stride=1)  # erosion
    m = F.pad(m, (pad_before, pad_after, pad_before, pad_after), value=0.0)
    m = F.max_pool2d(m, ksize, stride=1)  # dilation
    return m[0, 0] > 0.5


def _detect_markings_t(
    gray: torch.Tensor, valid_mask: torch.Tensor, k: float, blur_sigma: float, morph_ksize: int
) -> torch.Tensor:
    """Adaptive bright-on-dark threshold + morphological cleanup → binary mask.

    A pixel is a "lane marking" candidate if it sits more than
    ``k * local_std`` above the wide-Gaussian local mean; single-pixel noise
    is then removed by a morphological open.

    Args:
        gray: Grayscale tensor (H, W), float32 in [0, 1].
        valid_mask: Boolean road mask (H, W).
        k: Sigma multiplier above the local mean for the threshold.
        blur_sigma: Standard deviation of the wide local-statistics Gaussian.
        morph_ksize: Side length of the rectangular morph-open kernel.

    Returns:
        Boolean (H, W) tensor, True where a lane-marking candidate survives.
    """
    ksize = int(blur_sigma * 4) | 1
    # float64: the threshold is a binary decision, so float32 blur noise
    # would flip borderline pixels.
    x = gray[None, None].double()
    local_mean = _gaussian_blur_t(x, blur_sigma, ksize)
    local_sq_mean = _gaussian_blur_t(x * x, blur_sigma, ksize)
    local_std = (local_sq_mean - local_mean * local_mean).clamp_min_(0.0).sqrt_()
    marking = (x > local_mean + k * local_std)[0, 0] & valid_mask
    return _binary_open_t(marking, morph_ksize)


def _dog_bands_t(gray: torch.Tensor, n_scales: int, base_sigma: float) -> list[torch.Tensor]:
    """Difference-of-Gaussians decomposition of ``gray``, finest band first.

    Band ``i`` captures detail at roughly the spatial frequency of
    ``base_sigma * 2**i``, as the difference of two successive blurs.

    Args:
        gray: Grayscale tensor (H, W), float32.
        n_scales: Number of octaves to extract.
        base_sigma: Standard deviation of the finest Gaussian.

    Returns:
        List of ``n_scales`` float32 (H, W) tensors.
    """
    x = gray[None, None]
    bands: list[torch.Tensor] = []
    prev = _gaussian_blur_t(x, base_sigma)
    for i in range(n_scales):
        sigma = base_sigma * (2 ** (i + 1))
        cur = _gaussian_blur_t(x, sigma)
        bands.append((prev - cur)[0, 0])
        prev = cur
    return bands


def _safe_corrcoef_t(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation with safe handling of degenerate inputs.

    Falls back to 1.0 when both inputs are constant (matching constants are
    a perfect correlation by convention) and 0.0 when only one is constant
    or fewer than two samples are available.

    Args:
        a: First sample tensor [N], float64.
        b: Second sample tensor [N], float64, same length as ``a``.

    Returns:
        Pearson correlation coefficient in [-1, 1].
    """
    a_std = float(a.std(correction=0))
    b_std = float(b.std(correction=0))
    if len(a) < 2 or a_std < _EPS or b_std < _EPS:
        return 1.0 if (a_std < _EPS and b_std < _EPS) else 0.0
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = torch.sqrt((a_centered * a_centered).sum() * (b_centered * b_centered).sum())
    return float((a_centered * b_centered).sum() / denom)


# -----------------------------------------------------------------------
# ref — Road Edge Fidelity (Canny P/R/F1 + Chamfer)
# -----------------------------------------------------------------------


def _canny_with_adaptive_thresholds(gray: np.ndarray, mask: np.ndarray, canny_sigma: float) -> np.ndarray:
    """Run Canny edge detection with thresholds set from the masked-median intensity.

    The Canny low/high thresholds are picked as ``(1 ± canny_sigma) * median``
    of the in-mask pixels, so darker or brighter scenes get proportionally
    scaled thresholds. Edges falling outside ``mask`` are zeroed.

    Args:
        gray: Grayscale image [H, W], float32 in [0, 1].
        mask: Boolean region of interest [H, W]; True where edges count.
        canny_sigma: Fractional spread around the median for the lo/hi
            thresholds.

    Returns:
        uint8 edge map [H, W], non-zero where Canny marked an edge.
    """
    gray_u8 = (gray * 255).clip(0, 255).astype(np.uint8)
    vals = gray_u8[mask]
    if len(vals) == 0:
        return np.zeros_like(gray_u8)
    median = float(np.median(vals))
    # Dark / near-uniform regions: the median is uninformative and the
    # (1 ± σ) · median scaling collapses to a 1-2 px-wide threshold band
    # after int() truncation, which makes Canny detect noise as edges.
    # Fall back to OpenCV's traditional 50/150 defaults instead.
    if median < 10.0:
        lower, upper = 50, 150
    else:
        lower = int(max(0, (1.0 - canny_sigma) * median))
        upper = int(min(255, (1.0 + canny_sigma) * median))
    edges = cv2.Canny(gray_u8, lower, upper)
    edges[~mask] = 0
    return edges


def compute_ref(
    gt: np.ndarray,
    rendered: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tolerance_px: int = 3,
    canny_sigma: float = 0.33,
) -> dict[str, float]:
    """Compute Canny edge precision/recall/F1 and symmetric Chamfer on road pixels.

    Detects edges in ``gt`` and ``rendered`` with adaptive Canny inside the
    road mask, then matches the two edge sets with a pixel-tolerance disc to
    derive precision/recall/F1. The symmetric Chamfer distance averages each
    edge pixel's distance to the nearest opposite-set edge over both sides.

    Args:
        gt: Ground-truth image [H, W, 3], float32 in [0, 1].
        rendered: Rendered image [H, W, 3], float32 in [0, 1].
        valid_mask: Road mask [H, W], True where edges should contribute.
        tolerance_px: Distance within which a rendered edge counts as
            matching a GT edge for P/R/F1.
        canny_sigma: Spread around the masked median that picks Canny's
            low/high thresholds.

    Returns:
        Dict with keys ``ref_chamfer``, ``ref_precision``, ``ref_recall``,
        ``ref_f1``. All NaN if ``valid_mask`` is empty; degenerate
        edge sets return saturated edge-cases (see implementation).
    """
    keys = ("ref_chamfer", "ref_precision", "ref_recall", "ref_f1")
    if valid_mask.sum() == 0:
        return _nan_dict(keys)
    try:
        gt_gray = _to_gray(gt)
        rendered_gray = _to_gray(rendered)
        gt_edges = _canny_with_adaptive_thresholds(gt_gray, valid_mask, canny_sigma)
        rendered_edges = _canny_with_adaptive_thresholds(rendered_gray, valid_mask, canny_sigma)

        gt_edge_mask = gt_edges > 0
        rendered_edge_mask = rendered_edges > 0
        gt_edge_count = gt_edge_mask.sum()
        rendered_edge_count = rendered_edge_mask.sum()

        if gt_edge_count == 0 and rendered_edge_count == 0:
            return {"ref_chamfer": 0.0, "ref_precision": 1.0, "ref_recall": 1.0, "ref_f1": 1.0}
        # One side has no edges → Chamfer is undefined. Emit a bounded, finite
        # penalty (10x the match tolerance) instead of NaN: the frame should
        # hurt the mean (a render with no edges is catastrophically blurry),
        # but by a resolution-independent amount that can't dominate the
        # per-camera aggregate the way an image-diagonal sentinel would.
        max_chamfer = 10.0 * tolerance_px
        if gt_edge_count == 0:
            return {"ref_chamfer": max_chamfer, "ref_precision": 0.0, "ref_recall": 1.0, "ref_f1": 0.0}
        if rendered_edge_count == 0:
            return {"ref_chamfer": max_chamfer, "ref_precision": 1.0, "ref_recall": 0.0, "ref_f1": 0.0}

        # Restrict the EDT to the union-edge bounding box: every query pixel
        # (an edge in one set) and every nearest neighbour (an edge in the
        # other set) lies inside it, so the crop is exact while skipping the
        # road-free majority of the frame.
        edge_rows = np.flatnonzero((gt_edge_mask | rendered_edge_mask).any(axis=1))
        edge_cols = np.flatnonzero((gt_edge_mask | rendered_edge_mask).any(axis=0))
        crop = (slice(edge_rows[0], edge_rows[-1] + 1), slice(edge_cols[0], edge_cols[-1] + 1))
        gt_edge_mask = gt_edge_mask[crop]
        rendered_edge_mask = rendered_edge_mask[crop]
        gt_dist = distance_transform_edt(~gt_edge_mask)
        rendered_dist = distance_transform_edt(~rendered_edge_mask)

        precision = float((gt_dist[rendered_edge_mask] <= tolerance_px).sum() / rendered_edge_count)
        recall = float((rendered_dist[gt_edge_mask] <= tolerance_px).sum() / gt_edge_count)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        chamfer = float(0.5 * (gt_dist[rendered_edge_mask].mean() + rendered_dist[gt_edge_mask].mean()))

        return {"ref_chamfer": chamfer, "ref_precision": precision, "ref_recall": recall, "ref_f1": f1}
    except (cv2.error, np.linalg.LinAlgError, ZeroDivisionError):
        log.warning("compute_ref failed on degenerate input", exc_info=True)
        return _nan_dict(keys)


def debug_ref(
    gt: np.ndarray,
    rendered: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tolerance_px: int = 3,
    canny_sigma: float = 0.33,
) -> dict[str, np.ndarray]:
    """Build ``ref`` debug visualisations: TP/FP/FN overlay and raw edge maps.

    Args:
        gt: Ground-truth image [H, W, 3], float32 in [0, 1].
        rendered: Rendered image [H, W, 3], float32 in [0, 1].
        valid_mask: Road mask [H, W], True where edges should be considered.
        tolerance_px: Disc radius for the TP/FP/FN classification.
        canny_sigma: Same as ``compute_ref``.

    Returns:
        Dict with three uint8 arrays:
            ``edges_gt`` — GT edge map [H, W].
            ``edges_rendered`` — rendered edge map [H, W].
            ``edges_overlay`` — [H, W, 3] RGB with green=TP, red=FP,
            blue=FN.
    """
    try:
        gt_gray = _to_gray(gt)
        rendered_gray = _to_gray(rendered)
        gt_edges = _canny_with_adaptive_thresholds(gt_gray, valid_mask, canny_sigma)
        rendered_edges = _canny_with_adaptive_thresholds(rendered_gray, valid_mask, canny_sigma)

        gt_edge_mask = gt_edges > 0
        rendered_edge_mask = rendered_edges > 0
        h, w = valid_mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        if gt_edge_mask.sum() > 0 and rendered_edge_mask.sum() > 0:
            gt_dist = distance_transform_edt(~gt_edge_mask)
            rendered_dist = distance_transform_edt(~rendered_edge_mask)
            tp = rendered_edge_mask & (gt_dist <= tolerance_px)
            fp = rendered_edge_mask & (gt_dist > tolerance_px)
            fn = gt_edge_mask & (rendered_dist > tolerance_px)
            overlay[tp] = [0, 255, 0]
            overlay[fp] = [255, 0, 0]
            overlay[fn] = [0, 0, 255]

        return {"edges_gt": gt_edges, "edges_rendered": rendered_edges, "edges_overlay": overlay}
    except (cv2.error, np.linalg.LinAlgError, ZeroDivisionError):
        log.warning("debug_ref failed on degenerate input", exc_info=True)
        return {}


# -----------------------------------------------------------------------
# rlcf — Road Local Contrast Fidelity (sliding-window std-dev)
# -----------------------------------------------------------------------


def compute_rlcf(
    gt: np.ndarray | torch.Tensor,
    rendered: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    window_size: int = 11,
    high_contrast_threshold: float = 0.05,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Compute local-contrast correlation, preservation, and high-contrast fractions.

    Compares per-pixel local std-deviation between GT and rendered inside
    the road mask. ``preservation`` is the mean-ratio
    (``mean(rendered_std) / mean(gt_std)``) — values < 1 indicate softer
    rendering, > 1 indicates over-sharpening.

    Args:
        gt: Ground-truth RGB image, numpy (H, W, 3) or torch (3, H, W) /
            (1, 3, H, W), float values in [0, 1].
        rendered: Rendered image, same accepted layouts as ``gt``.
        valid_mask: Road mask [H, W].
        window_size: Side length of the sliding window used for local σ.
        high_contrast_threshold: Pixels with σ above this count as
            high-contrast.
        device: Compute device; None follows the input tensors (or picks
            CUDA when available for numpy inputs).

    Returns:
        Dict with keys ``rlcf_correlation``, ``rlcf_preservation``,
        ``rlcf_high_contrast_gt``, ``rlcf_high_contrast_rendered``.
        All NaN when ``valid_mask`` is empty.

    Raises:
        ValueError: if ``window_size`` < 1 (config error, not a per-frame condition).
    """
    if window_size < 1:
        raise ValueError(f"window_size={window_size} must be >= 1")

    keys = ("rlcf_correlation", "rlcf_preservation", "rlcf_high_contrast_gt", "rlcf_high_contrast_rendered")
    dev = _resolve_device(device, gt, rendered, valid_mask)
    valid_t = _as_mask_tensor(valid_mask, dev)
    if not bool(valid_t.any()):
        return _nan_dict(keys)
    try:
        gt_gray = _as_gray_tensor(gt, dev, "gt")
        rendered_gray = _as_gray_tensor(rendered, dev, "rendered")
        gt_std = _local_std_t(gt_gray, window_size)
        rendered_std = _local_std_t(rendered_gray, window_size)

        # Masked statistics reduce in float64 to hold precision over
        # millions of pixels.
        gt_vals = gt_std[valid_t].double()
        rendered_vals = rendered_std[valid_t].double()
        correlation = _safe_corrcoef_t(gt_vals, rendered_vals)

        gt_mean = float(gt_vals.mean())
        if gt_mean < _EPS:
            # GT has no local contrast on road pixels. If the renderer also
            # has none, both sides agree → preservation = 1.0. If the renderer
            # added contrast where GT had none, there's no GT structure to
            # have preserved → 0.0 (matches the `compute_rmtf` choice of
            # collapsing degenerate-band ratios to a finite penalty).
            preservation = 1.0 if float(rendered_vals.mean()) < _EPS else 0.0
        else:
            preservation = float(rendered_vals.mean()) / gt_mean

        n_valid = float(valid_t.sum())
        hc_gt = float((gt_vals > high_contrast_threshold).sum()) / n_valid
        hc_rendered = float((rendered_vals > high_contrast_threshold).sum()) / n_valid

        return {
            "rlcf_correlation": correlation,
            "rlcf_preservation": preservation,
            "rlcf_high_contrast_gt": hc_gt,
            "rlcf_high_contrast_rendered": hc_rendered,
        }
    except (cv2.error, np.linalg.LinAlgError, ZeroDivisionError):
        log.warning("compute_rlcf failed on degenerate input", exc_info=True)
        return _nan_dict(keys)


def debug_rlcf(
    gt: np.ndarray | torch.Tensor,
    rendered: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    window_size: int = 11,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Build ``rlcf`` local-contrast heatmaps for GT and rendered.

    Runs the same helpers as ``compute_rlcf``, so the heatmaps show exactly
    the maps that were scored.

    Args:
        gt: Ground-truth RGB image, numpy (H, W, 3) or torch (3, H, W) /
            (1, 3, H, W), float values in [0, 1].
        rendered: Rendered image, same accepted layouts as ``gt``.
        valid_mask: Road mask [H, W].
        window_size: Same as ``compute_rlcf``.
        device: Same as ``compute_rlcf``.

    Returns:
        Dict with two uint8 turbo-mapped images:
            ``local_contrast_gt`` and ``local_contrast_rendered``
            (both [H, W, 3]). Returns ``{}`` when ``valid_mask`` is empty.
    """
    dev = _resolve_device(device, gt, rendered, valid_mask)
    valid_t = _as_mask_tensor(valid_mask, dev)
    if not bool(valid_t.any()):
        return {}
    try:
        from nurec.utils.visualize import scalar2img  # only for debug visualizations

        gt_std = _local_std_t(_as_gray_tensor(gt, dev, "gt"), window_size)
        rendered_std = _local_std_t(_as_gray_tensor(rendered, dev, "rendered"), window_size)
        vmax = max(float(gt_std[valid_t].max()), float(rendered_std[valid_t].max()), 1e-6)
        return {
            "local_contrast_gt": scalar2img(gt_std.cpu().numpy(), vmin=0.0, vmax=vmax, cmap="turbo"),
            "local_contrast_rendered": scalar2img(rendered_std.cpu().numpy(), vmin=0.0, vmax=vmax, cmap="turbo"),
        }
    except (cv2.error, np.linalg.LinAlgError, ZeroDivisionError):
        log.warning("debug_rlcf failed on degenerate input", exc_info=True)
        return {}


# -----------------------------------------------------------------------
# lm — Lane Marking Detection (P/R/F1/IoU)
# -----------------------------------------------------------------------


def compute_lm(
    gt: np.ndarray | torch.Tensor,
    rendered: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    k: float = 1.5,
    blur_sigma: float = 21.0,
    morph_ksize: int = 3,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Compute lane-marking detection P/R/F1/IoU on road pixels.

    Detects bright-on-dark candidates in ``gt`` and ``rendered`` with an
    adaptive local threshold (see ``_detect_markings``), then computes
    precision/recall/F1 plus IoU on the resulting binary marking masks.

    Args:
        gt: Ground-truth RGB image, numpy (H, W, 3) or torch (3, H, W) /
            (1, 3, H, W), float values in [0, 1].
        rendered: Rendered image, same accepted layouts as ``gt``.
        valid_mask: Road mask [H, W].
        k: Sigma multiplier above the local mean for the marking threshold.
        blur_sigma: Standard deviation of the Gaussian for local statistics.
        morph_ksize: Side length of the rectangular morph-open kernel.
        device: Compute device; None follows the input tensors (or picks
            CUDA when available for numpy inputs).

    Returns:
        Dict with keys ``lm_iou``, ``lm_f1``, ``lm_precision``,
        ``lm_recall``. All NaN if ``valid_mask`` is empty; degenerate
        detections return saturated edge-cases (see implementation).

    Raises:
        ValueError: if ``morph_ksize`` < 1 or ``blur_sigma`` <= 0 (config errors).
    """
    if morph_ksize < 1:
        raise ValueError(f"morph_ksize={morph_ksize} must be >= 1")
    if blur_sigma <= 0:
        raise ValueError(f"blur_sigma={blur_sigma} must be > 0")

    keys = ("lm_iou", "lm_f1", "lm_precision", "lm_recall")
    dev = _resolve_device(device, gt, rendered, valid_mask)
    valid_t = _as_mask_tensor(valid_mask, dev)
    if not bool(valid_t.any()):
        return _nan_dict(keys)
    try:
        gt_gray = _as_gray_tensor(gt, dev, "gt")
        rendered_gray = _as_gray_tensor(rendered, dev, "rendered")
        gt_markings = _detect_markings_t(gt_gray, valid_t, k, blur_sigma, morph_ksize)
        rendered_markings = _detect_markings_t(rendered_gray, valid_t, k, blur_sigma, morph_ksize)

        gt_count = int(gt_markings.sum())
        rendered_count = int(rendered_markings.sum())
        if gt_count == 0 and rendered_count == 0:
            return {"lm_iou": 1.0, "lm_f1": 1.0, "lm_precision": 1.0, "lm_recall": 1.0}
        if gt_count == 0:
            return {"lm_iou": 0.0, "lm_f1": 0.0, "lm_precision": 0.0, "lm_recall": 1.0}
        if rendered_count == 0:
            return {"lm_iou": 0.0, "lm_f1": 0.0, "lm_precision": 1.0, "lm_recall": 0.0}

        intersection = float(int((gt_markings & rendered_markings).sum()))
        union = float(int((gt_markings | rendered_markings).sum()))
        iou = intersection / union if union > 0 else 0.0
        precision = intersection / float(rendered_count)
        recall = intersection / float(gt_count)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"lm_iou": iou, "lm_f1": f1, "lm_precision": precision, "lm_recall": recall}
    except (cv2.error, np.linalg.LinAlgError, ZeroDivisionError):
        log.warning("compute_lm failed on degenerate input", exc_info=True)
        return _nan_dict(keys)


def debug_lm(
    gt: np.ndarray | torch.Tensor,
    rendered: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    k: float = 1.5,
    blur_sigma: float = 21.0,
    morph_ksize: int = 3,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Build ``lm`` debug visualisations: per-side marking masks and TP/FP/FN overlay.

    Runs the same helpers as ``compute_lm``, so the masks show exactly the
    detections that were scored.

    Args:
        gt: Ground-truth RGB image, numpy (H, W, 3) or torch (3, H, W) /
            (1, 3, H, W), float values in [0, 1].
        rendered: Rendered image, same accepted layouts as ``gt``.
        valid_mask: Road mask [H, W].
        k: Same as ``compute_lm``.
        blur_sigma: Same as ``compute_lm``.
        morph_ksize: Same as ``compute_lm``.
        device: Same as ``compute_lm``.

    Returns:
        Dict with three uint8 arrays:
            ``markings_gt`` — 0/255 GT marking mask [H, W].
            ``markings_rendered`` — 0/255 rendered marking mask [H, W].
            ``markings_overlay`` — [H, W, 3] RGB with green=TP, red=FP,
            blue=FN.
    """
    try:
        dev = _resolve_device(device, gt, rendered, valid_mask)
        valid_t = _as_mask_tensor(valid_mask, dev)
        gt_gray = _as_gray_tensor(gt, dev, "gt")
        rendered_gray = _as_gray_tensor(rendered, dev, "rendered")
        gt_markings = _detect_markings_t(gt_gray, valid_t, k, blur_sigma, morph_ksize).cpu().numpy()
        rendered_markings = _detect_markings_t(rendered_gray, valid_t, k, blur_sigma, morph_ksize).cpu().numpy()
        h, w = gt_markings.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[gt_markings & rendered_markings] = [0, 255, 0]
        overlay[rendered_markings & ~gt_markings] = [255, 0, 0]
        overlay[gt_markings & ~rendered_markings] = [0, 0, 255]
        return {
            "markings_gt": (gt_markings.astype(np.uint8) * 255),
            "markings_rendered": (rendered_markings.astype(np.uint8) * 255),
            "markings_overlay": overlay,
        }
    except (cv2.error, np.linalg.LinAlgError, ZeroDivisionError):
        log.warning("debug_lm failed on degenerate input", exc_info=True)
        return {}


# -----------------------------------------------------------------------
# rmtf — Road Multi-Scale Texture Fidelity (DoG bands)
# -----------------------------------------------------------------------


def compute_rmtf(
    gt: np.ndarray | torch.Tensor,
    rendered: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    n_scales: int = 4,
    base_sigma: float = 1.0,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Compute multi-scale DoG energy ratios plus a weighted aggregate.

    For each DoG band, compares the rendered-vs-GT energy on road pixels.
    The aggregate weights per-band ratios by ``_RMTF_BAND_WEIGHTS`` and
    clips each ratio at 2.0 before weighting, so a single over-sharpened
    band can't dominate the score.

    Note on ``n_scales``: ``_RMTF_BAND_WEIGHTS`` is tuned for the default
    ``n_scales=4``. When fewer scales are requested, the slice
    ``_RMTF_BAND_WEIGHTS[:n_scales]`` is renormalised to sum to 1.0, which
    rebalances which bands dominate the aggregate — e.g. ``n_scales=2``
    yields effective weights ``(1/3, 2/3)``, so the coarser band weighs
    more heavily than under the n=4 tuning. Configurations beyond 4 raise.

    Args:
        gt: Ground-truth RGB image, numpy (H, W, 3) or torch (3, H, W) /
            (1, 3, H, W), float values in [0, 1].
        rendered: Rendered image, same accepted layouts as ``gt``.
        valid_mask: Road mask [H, W].
        n_scales: Number of DoG octaves.
        base_sigma: Standard deviation of the finest Gaussian; band ``i``
            uses ``base_sigma * 2**i``.
        device: Compute device; None follows the input tensors (or picks
            CUDA when available for numpy inputs).

    Returns:
        Dict with one ``rmtf_ratio_s{σ}`` key per band plus
        ``rmtf_aggregate``. All NaN when ``valid_mask`` is empty.

    Raises:
        ValueError: if ``n_scales`` is outside ``[1, len(_RMTF_BAND_WEIGHTS)]``
            or ``base_sigma`` <= 0 (config errors).
    """
    if not 1 <= n_scales <= len(_RMTF_BAND_WEIGHTS):
        raise ValueError(f"n_scales={n_scales} outside supported range [1, {len(_RMTF_BAND_WEIGHTS)}]")
    if base_sigma <= 0:
        raise ValueError(f"base_sigma={base_sigma} must be > 0")

    band_keys = [f"rmtf_ratio_s{base_sigma * (2**i):g}" for i in range(n_scales)] + ["rmtf_aggregate"]
    dev = _resolve_device(device, gt, rendered, valid_mask)
    valid_t = _as_mask_tensor(valid_mask, dev)
    if not bool(valid_t.any()):
        return _nan_dict(band_keys)
    try:
        gt_gray = _as_gray_tensor(gt, dev, "gt")
        rendered_gray = _as_gray_tensor(rendered, dev, "rendered")
        gt_bands = _dog_bands_t(gt_gray, n_scales, base_sigma)
        rendered_bands = _dog_bands_t(rendered_gray, n_scales, base_sigma)

        weights = np.array(_RMTF_BAND_WEIGHTS[:n_scales])
        weights = weights / weights.sum()

        results: dict[str, float] = {}
        agg = 0.0
        for i, (gb, rb) in enumerate(zip(gt_bands, rendered_bands)):
            sigma_label = f"{base_sigma * (2**i):g}"
            # Band energies reduce in float64 (see compute_rlcf note).
            gt_energy = float(gb[valid_t].double().pow(2).mean())
            rendered_energy = float(rb[valid_t].double().pow(2).mean())
            if gt_energy < _EPS:
                # Renderer adds energy where GT has none → maximum band penalty.
                # Use 2.0 (the per-band aggregate cap) so the value written to
                # metadata stays finite, and the aggregate contribution matches
                # what min(ratio, 2.0) would yield for any over-energy ratio.
                ratio = 1.0 if rendered_energy < _EPS else 2.0
            else:
                ratio = rendered_energy / gt_energy
            results[f"rmtf_ratio_s{sigma_label}"] = ratio
            agg += float(weights[i]) * min(ratio, 2.0)
        results["rmtf_aggregate"] = agg
        return results
    except (cv2.error, np.linalg.LinAlgError, ZeroDivisionError):
        log.warning("compute_rmtf failed on degenerate input", exc_info=True)
        return _nan_dict(band_keys)


def debug_rmtf(
    gt: np.ndarray | torch.Tensor,
    rendered: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    n_scales: int = 4,
    base_sigma: float = 1.0,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Build per-band DoG magnitude visualisations for ``rmtf``.

    Runs the same helpers as ``compute_rmtf``, so the band images show
    exactly the maps that were scored.

    Args:
        gt: Ground-truth RGB image, numpy (H, W, 3) or torch (3, H, W) /
            (1, 3, H, W), float values in [0, 1].
        rendered: Rendered image, same accepted layouts as ``gt``.
        valid_mask: Road mask [H, W].
        n_scales: Same as ``compute_rmtf``.
        base_sigma: Same as ``compute_rmtf``.
        device: Same as ``compute_rmtf``.

    Returns:
        Dict with two uint8 turbo-mapped [H, W, 3] images per band:
        ``dog_s{σ}_gt`` and ``dog_s{σ}_rendered``. Returns ``{}`` when
        ``valid_mask`` is empty.
    """
    dev = _resolve_device(device, gt, rendered, valid_mask)
    valid_t = _as_mask_tensor(valid_mask, dev)
    if not bool(valid_t.any()):
        return {}
    try:
        from nurec.utils.visualize import scalar2img  # only for debug visualizations

        gt_bands = _dog_bands_t(_as_gray_tensor(gt, dev, "gt"), n_scales, base_sigma)
        rendered_bands = _dog_bands_t(_as_gray_tensor(rendered, dev, "rendered"), n_scales, base_sigma)
        images: dict[str, np.ndarray] = {}
        for i, (gb, rb) in enumerate(zip(gt_bands, rendered_bands)):
            sigma_label = f"{base_sigma * (2**i):g}"
            gb_abs, rb_abs = gb.abs(), rb.abs()
            vmax = max(float(gb_abs[valid_t].max()), float(rb_abs[valid_t].max()), 1e-6)
            images[f"dog_s{sigma_label}_gt"] = scalar2img(gb_abs.cpu().numpy(), vmin=0.0, vmax=vmax, cmap="turbo")
            images[f"dog_s{sigma_label}_rendered"] = scalar2img(rb_abs.cpu().numpy(), vmin=0.0, vmax=vmax, cmap="turbo")
        return images
    except (cv2.error, np.linalg.LinAlgError, ZeroDivisionError):
        log.warning("debug_rmtf failed on degenerate input", exc_info=True)
        return {}
