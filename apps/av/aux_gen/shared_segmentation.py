# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segmentation adapter for the shared device-frame service."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from numpy.typing import NDArray
from PIL import Image as PILImage

from apps.av.aux_gen.device_frame_service import (
    DeviceFrameSharedArrayDescriptor,
    PendingDeviceFrame,
    SharedDeviceFrameClient,
)
from apps.av.aux_gen.segmentation_device_pipeline import (
    MASK2FORMER_PROCESSOR,
    SEMANTIC_LABELS_OUTPUT,
    SEMANTIC_LOGITS_OUTPUT,
    SEMANTIC_UNCERTAINTY_OUTPUT,
)
from apps.av.aux_gen.segmentation_presentation import (
    labels_to_paletted_image,
    save_mask2former_visualization,
)


class AuxiliaryFrameStatistics(Protocol):
    """Fold an extension processor's per-frame counters into camera totals.

    One instance is created per camera worker, so implementations own mutable
    totals and never coordinate across processes.
    """

    def accumulate(self, statistics: NDArray[Any]) -> None:
        """Add one frame's counter vector to this camera's running totals.

        Args:
            statistics: The frame's raw counter output, as declared by the
                extension processor's statistics output spec.
        """
        ...

    def camera_record(self, camera_id: str) -> Mapping[str, Any]:
        """Return this camera's accumulated counters.

        Args:
            camera_id: Camera identifier to record with the counters.

        Returns:
            A picklable record published to the parent process.
        """
        ...


@dataclass(kw_only=True, slots=True, frozen=True)
class AuxiliarySegmentationOutputs:
    """Describe how one extension processor's frame outputs are consumed.

    Attributes:
        processor_name: Device-pipeline processor key enabled for sampled
            frames.
        mask_output: Output-buffer key holding the ``uint8[H, W]`` mask stored
            as a grayscale PNG.
        mask_root: Root directory receiving ``<camera_id>/<frame>.png`` masks.
        statistics_output: Optional output-buffer key holding per-frame
            counters.
        create_statistics: Optional per-camera counter accumulator factory.
            Required when ``statistics_output`` is set.
    """

    processor_name: str
    mask_output: str
    mask_root: Path
    statistics_output: str | None = None
    create_statistics: Callable[[], AuxiliaryFrameStatistics] | None = None


@dataclass(kw_only=True, slots=True, frozen=True)
class SegmentationFrameOutput:
    """Expose host outputs for one completed segmentation frame.

    Fields may alias storage owned or reused by a producer. Consumers handing
    them to asynchronous work must take an ownership copy first.
    """

    semantic_seg: PILImage.Image | None = None
    semantic_seg_logits: NDArray[Any] | None = None
    auxiliary_mask: NDArray[np.uint8] | None = None
    auxiliary_mask_path: Path | None = None


def save_auxiliary_mask(mask_path: Path, mask: NDArray[np.uint8]) -> None:
    """Atomically encode and write one auxiliary mask as a grayscale PNG.

    Args:
        mask_path: Final PNG destination, which must not already exist.
        mask: Single-channel uint8 mask to encode.

    Raises:
        FileExistsError: If ``mask_path`` already exists.
        RuntimeError: If Pillow cannot encode or store the temporary PNG.
    """
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    # Existing masks imply stale or mixed output; never silently clobber them.
    if mask_path.exists():
        raise FileExistsError(mask_path)
    temporary = mask_path.with_name(f".{mask_path.stem}.tmp{mask_path.suffix}")
    try:
        PILImage.fromarray(mask).save(temporary, format="PNG")
        temporary.replace(mask_path)
    except OSError as error:
        raise RuntimeError(f"Could not write auxiliary mask: {mask_path}") from error


@dataclass(frozen=True, slots=True)
class PendingSharedSegmentationFrame:
    """Add segmentation presentation state to a pending device frame."""

    device_frame: PendingDeviceFrame
    vis_path: Path | None


@dataclass(kw_only=True, slots=True)
class SharedSegmentationFrameProcessor:
    """Submit segmentation frames and resolve their outputs in order."""

    stuff_colors: NDArray[np.uint8]
    processor_inputs: tuple[DeviceFrameSharedArrayDescriptor, ...]
    client: SharedDeviceFrameClient
    include_mask2former: bool = True
    auxiliary_outputs: AuxiliarySegmentationOutputs | None = None
    auxiliary_sample_indices: frozenset[int] = frozenset()
    camera_id: str | None = None
    camera_frame_range_start: int = 0
    _auxiliary_statistics: AuxiliaryFrameStatistics | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create this camera's counter accumulator when one is configured."""
        if self.auxiliary_outputs is not None and self.auxiliary_outputs.create_statistics is not None:
            self._auxiliary_statistics = self.auxiliary_outputs.create_statistics()

    def __call__(
        self,
        request_id: int,
        _frame_timestamp_us: int,
        decoded_image: Any,
        vis_path: Path | None,
    ) -> PendingSharedSegmentationFrame | None:
        """Submit one decoded frame without waiting for its device result."""
        enabled_processors = {MASK2FORMER_PROCESSOR} if self.include_mask2former else set()
        if self.auxiliary_outputs is not None and request_id in self.auxiliary_sample_indices:
            enabled_processors.add(self.auxiliary_outputs.processor_name)
        pending_frame = self.client.submit(
            request_id=request_id,
            enabled_processors=frozenset(enabled_processors),
            image=decoded_image,
            processor_inputs=self.processor_inputs,
        )
        if pending_frame is None:
            return None
        return PendingSharedSegmentationFrame(
            device_frame=pending_frame,
            vis_path=vis_path,
        )

    def resolve(self, pending_frame: PendingSharedSegmentationFrame) -> SegmentationFrameOutput:
        """Resolve one submitted frame into host-owned segmentation outputs."""
        resolved = self.client.resolve(
            pending_frame.device_frame,
            include_input=pending_frame.vis_path is not None,
        )
        labels = resolved.outputs.get(SEMANTIC_LABELS_OUTPUT)
        if self.include_mask2former and labels is None:
            raise RuntimeError("Enabled Mask2Former processor did not return semantic labels")
        semantic_seg_logits = resolved.outputs.get(SEMANTIC_LOGITS_OUTPUT)
        if pending_frame.vis_path is not None and labels is not None:
            if resolved.input_image is None:
                raise RuntimeError("Shared device-frame client omitted the requested input image")
            save_mask2former_visualization(
                resolved.input_image,
                labels,
                self.stuff_colors,
                pending_frame.vis_path,
                resolved.outputs.get(SEMANTIC_UNCERTAINTY_OUTPUT),
            )
        auxiliary_mask = None
        auxiliary_mask_path = None
        if self.auxiliary_outputs is not None:
            if pending_frame.device_frame.request.request_id in self.auxiliary_sample_indices:
                if self.auxiliary_outputs.mask_output not in resolved.outputs:
                    raise RuntimeError(
                        f"Sampled frame omitted auxiliary mask output: {self.auxiliary_outputs.mask_output}"
                    )
            auxiliary_mask = resolved.outputs.get(self.auxiliary_outputs.mask_output)
        if auxiliary_mask is not None:
            assert self.auxiliary_outputs is not None
            if self.camera_id is None:
                raise RuntimeError("Auxiliary mask output received without a camera ID")
            source_frame_index = self.camera_frame_range_start + pending_frame.device_frame.request.request_id
            auxiliary_mask_path = self.auxiliary_outputs.mask_root / self.camera_id / f"{source_frame_index:06d}.png"
            self._accumulate_auxiliary_statistics(resolved.outputs)
        return SegmentationFrameOutput(
            semantic_seg=labels_to_paletted_image(labels) if labels is not None else None,
            semantic_seg_logits=semantic_seg_logits,
            auxiliary_mask=auxiliary_mask,
            auxiliary_mask_path=auxiliary_mask_path,
        )

    def _accumulate_auxiliary_statistics(self, outputs: Mapping[str, NDArray[Any]]) -> None:
        """Fold one frame's extension counters into this camera's totals.

        Each camera is resolved by one worker, so it can accumulate locally and
        publish a single camera record when processing finishes.
        """
        assert self.auxiliary_outputs is not None
        statistics_output = self.auxiliary_outputs.statistics_output
        if statistics_output is None:
            return
        if self._auxiliary_statistics is None:
            raise RuntimeError("Auxiliary statistics output requires a statistics accumulator")
        statistics = outputs.get(statistics_output)
        if statistics is None:
            raise RuntimeError("Auxiliary mask output arrived without its statistics output")
        self._auxiliary_statistics.accumulate(statistics)

    def auxiliary_camera_statistics(self) -> Mapping[str, Any] | None:
        """Return one aggregate record for this camera's sampled frames.

        Returns:
            The extension's accumulated counters for this camera, or ``None``
            when no auxiliary frames were sampled or no counters are collected.

        Raises:
            RuntimeError: If samples were configured without a camera ID.
        """
        if not self.auxiliary_sample_indices or self._auxiliary_statistics is None:
            return None
        if self.camera_id is None:
            raise RuntimeError("Auxiliary statistics require a camera ID")
        return self._auxiliary_statistics.camera_record(self.camera_id)

    def close(self) -> None:
        """Close every shared-memory attachment borrowed by this processor."""
        self.client.close()
