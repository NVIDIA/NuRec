# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Road-pixel quality metrics — four BaseMetric subclasses.

Each metric scores rendering quality between a rendered image and its
ground-truth counterpart, restricted to road pixels. The four sub-metrics
are registered as independent ``MetricType`` entries (``REF``, ``RLCF``,
``LM``, ``RMTF``), sharing their plumbing through ``_RoadMetricBase``.

For each per-frame ``compute()`` call, a metric returns a ``MetricResult``
with ONE headline scalar in ``values`` and its per-frame diagnostic
sub-scores in ``metadata``. Each class declares its output region via
``SEMANTIC_CLASS`` (``"road"`` for these four); the CLI routes both the
headline and the diagnostics to that region, tagging diagnostics with
``metric_group="diagnostic"`` so YAML can filter them out cleanly while
CSV/parquet preserve them for debugging.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torch._prims_common import DeviceLikeType

from nurec.metrics.impl.utils.road_utils import (
    compute_lm,
    compute_ref,
    compute_rlcf,
    compute_rmtf,
    debug_lm,
    debug_ref,
    debug_rlcf,
    debug_rmtf,
)
from nurec.metrics.metric import BaseMetric, MetricResult
from nurec.metrics.types import MetricType
from nurec.metrics.utils import AggregationMethod, aggregate_tensors


# ---- Shared input-normalisation helpers --------------------------------


def _to_numpy_hwc(arr: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    """Normalise a (C, H, W) tensor / (H, W, C) ndarray pred-or-target to (H, W, C) float32 numpy.

    Tensors are permuted to HWC; ndarrays are assumed to already be HWC and
    only have their dtype cast. Callers MUST pre-normalise pixel values to
    float [0, 1] -- this function only changes dtype, it does NOT rescale.
    Downstream helpers (Canny, DoG, local-σ correlation) assume [0, 1] and
    would saturate / overflow on raw uint8-range values.
    """
    if isinstance(arr, torch.Tensor):
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"{name} tensor must be (3, H, W); got shape {tuple(arr.shape)}")
        return arr.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32, copy=False)
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32, copy=False)
    raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(arr).__name__}")


def _to_numpy_bool_mask(arr: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    """Normalise a boolean mask argument to a 2-D boolean numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(bool)
    if isinstance(arr, np.ndarray):
        return arr if arr.dtype == bool else arr.astype(bool)
    raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(arr).__name__}")


def _image_spatial_shape(arr: np.ndarray | torch.Tensor, name: str) -> tuple[int, int]:
    """Validate an image argument's layout and return its (H, W), without copying data off its device."""
    if isinstance(arr, torch.Tensor):
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"{name} tensor must be (3, H, W); got shape {tuple(arr.shape)}")
        return int(arr.shape[1]), int(arr.shape[2])
    if isinstance(arr, np.ndarray):
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"{name} must be (H, W, 3); got shape {arr.shape}")
        return int(arr.shape[0]), int(arr.shape[1])
    raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(arr).__name__}")


def _validate_road_inputs(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
) -> None:
    """Shared shape/type validation for the 4 road sub-metric classes."""
    pred_shape = _image_spatial_shape(pred, "pred")
    target_shape = _image_spatial_shape(target, "target")
    if pred_shape != target_shape:
        raise ValueError(f"pred {pred_shape} and target {target_shape} spatial shapes must match")
    if not isinstance(mask, (np.ndarray, torch.Tensor)):
        raise TypeError(f"mask must be np.ndarray or torch.Tensor, got {type(mask).__name__}")
    mask_shape = tuple(int(s) for s in mask.shape)
    if mask_shape != pred_shape:
        raise ValueError(f"mask shape {mask_shape} must match image spatial dims {pred_shape}")


# ---- The four road-quality metric classes ------------------------------


class _RoadMetricBase(BaseMetric):
    """Shared plumbing for the four road sub-metric classes.

    Subclasses declare ``_NAME`` and ``HEADLINE_KEY``, implement ``type()``,
    and override ``_score`` / ``_debug`` with direct calls to their helpers.
    """

    _NAME: str
    HEADLINE_KEY: str
    SEMANTIC_CLASS = "road"
    MASK_KIND = "road"
    EMIT_DIAGNOSTICS = True
    OVERLAY_HEADLINE = False

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.MEAN,
        sub_metric_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(device, aggregation_methods)
        self.sub_metric_params: dict[str, Any] = sub_metric_params or {}

    def _score(
        self,
        gt: np.ndarray | torch.Tensor,
        rendered: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        **kw: Any,
    ) -> dict[str, float]:
        """Run this metric's ``compute_*`` helper; the caller merges ``sub_metric_params`` into ``kw``."""
        raise NotImplementedError

    def _debug(
        self,
        gt: np.ndarray | torch.Tensor,
        rendered: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        **kw: Any,
    ) -> dict[str, np.ndarray]:
        """Run this metric's ``debug_*`` helper; the caller merges ``sub_metric_params`` into ``kw``."""
        raise NotImplementedError

    def validate_inputs(self, pred, target, mask, **kw):
        _validate_road_inputs(pred, target, mask)

    def _compute(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        **kw: Any,
    ) -> MetricResult:
        scores = self._score(target, pred, mask, **self.sub_metric_params, **kw)
        headline = torch.tensor(float(scores[self.HEADLINE_KEY]), dtype=torch.float32, device=self.device)
        metadata = {k: float(v) for k, v in scores.items() if k != self.HEADLINE_KEY}
        return MetricResult(values={self.HEADLINE_KEY: headline}, metadata=metadata)

    def compute_debug_images(
        self,
        gt: np.ndarray | torch.Tensor,
        rendered: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        **kw: Any,
    ) -> dict[str, np.ndarray]:
        return self._debug(gt, rendered, mask, **self.sub_metric_params, **kw)

    def aggregate(self) -> dict[AggregationMethod, MetricResult]:
        out: dict[AggregationMethod, MetricResult] = {}
        if not self._values:
            return out
        for method in self._aggregation_methods:
            per_frame = [
                mr.values[self.HEADLINE_KEY]
                for mr in self._values
                if self.HEADLINE_KEY in mr.values and torch.isfinite(mr.values[self.HEADLINE_KEY])
            ]
            if not per_frame:
                agg = torch.tensor(float("nan"), dtype=torch.float32, device=self.device)
            else:
                agg = aggregate_tensors(per_frame, method=method)
            out[method] = MetricResult(values={self.HEADLINE_KEY: agg})
        return out

    def metadata(self) -> dict[str, Any]:
        return {"metric_name": self._NAME, "sub_metric_params": dict(self.sub_metric_params)}

    def reset(self) -> None:
        """No internal state to reset (per-frame values are owned by BaseMetric.clear())."""


class RefMetric(_RoadMetricBase):
    """Road Edge Fidelity — adaptive Canny edges + symmetric Chamfer on road pixels.

    Returns headline ``ref_chamfer`` (lower-is-better, pixel distance) plus
    diagnostic ``ref_precision``, ``ref_recall``, ``ref_f1`` in metadata.
    """

    _NAME = "ref"
    HEADLINE_KEY = "ref_chamfer"

    def type(self) -> MetricType:
        return MetricType.REF

    def _score(self, gt, rendered, mask, **kw):
        # compute_ref runs on cv2/scipy: numpy conversion happens at this boundary.
        return compute_ref(
            _to_numpy_hwc(gt, "target"), _to_numpy_hwc(rendered, "pred"), _to_numpy_bool_mask(mask, "mask"), **kw
        )

    def _debug(self, gt, rendered, mask, **kw):
        return debug_ref(
            _to_numpy_hwc(gt, "gt"), _to_numpy_hwc(rendered, "rendered"), _to_numpy_bool_mask(mask, "mask"), **kw
        )


class RlcfMetric(_RoadMetricBase):
    """Road Local Contrast Fidelity — sliding-window σ correlation + preservation ratio.

    Returns headline ``rlcf_preservation`` (≈1.0 is matched, <1 mushy, >1 over-sharp)
    plus diagnostic ``rlcf_correlation``, ``rlcf_high_contrast_gt``,
    ``rlcf_high_contrast_rendered`` in metadata.
    """

    _NAME = "rlcf"
    HEADLINE_KEY = "rlcf_preservation"

    def type(self) -> MetricType:
        return MetricType.RLCF

    def _score(self, gt, rendered, mask, **kw):
        return compute_rlcf(gt, rendered, mask, **kw)

    def _debug(self, gt, rendered, mask, **kw):
        return debug_rlcf(gt, rendered, mask, **kw)


class LmMetric(_RoadMetricBase):
    """Lane Markings — adaptive bright-on-dark detection (P/R/F1/IoU).

    Returns headline ``lm_iou`` (0..1, higher is better) plus diagnostic
    ``lm_precision``, ``lm_recall``, ``lm_f1`` in metadata.
    """

    _NAME = "lm"
    HEADLINE_KEY = "lm_iou"

    def type(self) -> MetricType:
        return MetricType.LM

    def _score(self, gt, rendered, mask, **kw):
        return compute_lm(gt, rendered, mask, **kw)

    def _debug(self, gt, rendered, mask, **kw):
        return debug_lm(gt, rendered, mask, **kw)


class RmtfMetric(_RoadMetricBase):
    """Road Multi-scale Texture Fidelity — DoG band energy ratios + weighted aggregate.

    Returns headline ``rmtf_aggregate`` (≈1.0 is matched, capped at 2.0) plus
    diagnostic per-band ratios (``rmtf_ratio_s1``, ``rmtf_ratio_s2``, …) in metadata.
    """

    _NAME = "rmtf"
    HEADLINE_KEY = "rmtf_aggregate"

    def type(self) -> MetricType:
        return MetricType.RMTF

    def _score(self, gt, rendered, mask, **kw):
        return compute_rmtf(gt, rendered, mask, **kw)

    def _debug(self, gt, rendered, mask, **kw):
        return debug_rmtf(gt, rendered, mask, **kw)
