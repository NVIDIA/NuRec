# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight configuration for shared segmentation device pipelines."""

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch

from apps.av.aux_gen.device_frame import (
    DeviceFrameOutputSpec,
    DeviceFramePipeline,
    DeviceFrameProcessor,
    processor_directory,
)


MASK2FORMER_PROCESSOR = "mask2former"
SAM2_PROCESSOR = "sam2"
CAMERA_MASK_INPUT = "camera_mask"
SEMANTIC_LABELS_OUTPUT = "semantic_labels"
SEMANTIC_LOGITS_OUTPUT = "semantic_logits"
SEMANTIC_UNCERTAINTY_OUTPUT = "semantic_uncertainty"
SAM2_MASK_OUTPUT = "sam2_mask"


class AuxiliaryProcessorFactory(Protocol):
    """Create one extension-owned processor for the shared device pipeline.

    An extending entrypoint supplies an implementation so this module stays
    independent of any particular auxiliary model. Instances travel to GPU
    service processes inside the pipeline factory, so they must be picklable
    and must not import model code before they are called.
    """

    def output_specs(self, resolution: tuple[int, int]) -> tuple[DeviceFrameOutputSpec, ...]:
        """Describe the outputs the created processor produces.

        Args:
            resolution: Raw image resolution as ``(width, height)`` pixels.

        Returns:
            Ordered output specifications for parent allocation.
        """
        ...

    def __call__(self, *, semantic_class_names: Sequence[str]) -> DeviceFrameProcessor:
        """Create the processor on the current CUDA device.

        Args:
            semantic_class_names: Ordered Mask2Former semantic class names,
                or an empty sequence when Mask2Former is disabled.

        Returns:
            A processor implementing the shared device-frame protocol.
        """
        ...


def _mask2former_output_specs(
    resolution: tuple[int, int],
    *,
    estimate_logits: bool,
    include_uncertainty: bool = False,
    num_classes: int,
) -> tuple[DeviceFrameOutputSpec, ...]:
    """Build the Mask2Former output contract for one frame resolution.

    Args:
        resolution: Raw image resolution as ``(width, height)`` pixels.
        estimate_logits: Whether to include per-class semantic logits.
        include_uncertainty: Whether to include the semantic uncertainty mask.
        num_classes: Number of Mask2Former semantic classes. Required to be
            positive when ``estimate_logits`` is enabled.

    Returns:
        Ordered output specs for Mask2Former. The required labels output is
        ``semantic_labels`` with dtype ``torch.uint8`` and shape ``[H, W]``.
        When requested, logits are ``semantic_logits`` with dtype
        ``torch.float16`` and shape ``[H, W, C]`` where ``C == num_classes``.
        When requested, uncertainty is ``semantic_uncertainty`` with dtype
        ``torch.uint8`` and shape ``[H, W]``.

    Raises:
        ValueError: If logits are requested with ``num_classes <= 0``.
    """
    width, height = resolution
    specs = [DeviceFrameOutputSpec(SEMANTIC_LABELS_OUTPUT, (height, width), torch.uint8)]
    if estimate_logits:
        if num_classes <= 0:
            raise ValueError(f"Mask2Former logits require a positive class count, got {num_classes}")
        specs.append(
            DeviceFrameOutputSpec(
                SEMANTIC_LOGITS_OUTPUT,
                (height, width, num_classes),
                torch.float16,
            )
        )
    if include_uncertainty:
        specs.append(DeviceFrameOutputSpec(SEMANTIC_UNCERTAINTY_OUTPUT, (height, width), torch.uint8))
    return tuple(specs)


def _sam2_output_specs(resolution: tuple[int, int]) -> tuple[DeviceFrameOutputSpec, ...]:
    """Build the promptless SAM2 output contract for one frame resolution.

    Args:
        resolution: Raw image resolution as ``(width, height)`` pixels.

    Returns:
        A single ``sam2_mask`` output spec with dtype ``torch.uint8`` and
        shape ``[H, W]``.
    """
    width, height = resolution
    return (DeviceFrameOutputSpec(SAM2_MASK_OUTPUT, (height, width), torch.uint8),)


def segmentation_output_specs(
    resolution: tuple[int, int],
    *,
    estimate_logits: bool,
    include_uncertainty: bool = False,
    num_classes: int,
    include_sam2: bool,
    auxiliary_specs: tuple[DeviceFrameOutputSpec, ...] = (),
    include_mask2former: bool = True,
) -> tuple[DeviceFrameOutputSpec, ...]:
    """Compose output specs for the shared segmentation service contract.

    Args:
        resolution: Raw image resolution as ``(width, height)`` pixels. Each
            returned output shape uses ``height`` before ``width``.
        estimate_logits: Whether Mask2Former should produce
            ``semantic_logits``.
        include_uncertainty: Whether Mask2Former should produce
            ``semantic_uncertainty``.
        num_classes: Number of Mask2Former semantic classes. Required to be
            positive when ``estimate_logits`` is enabled.
        include_sam2: Whether to append the promptless SAM2 ``sam2_mask``
            output.
        auxiliary_specs: Outputs declared by an extension-owned processor.
        include_mask2former: Whether to allocate semantic-label outputs.

    Returns:
        Ordered output specs for parent allocation and service validation:
        Mask2Former labels first, optional Mask2Former logits and uncertainty
        next, followed by optional SAM2 and extension outputs. Shapes are
        unbatched per-frame tensors and dtypes match the CUDA/shared-memory
        staging buffers.

    Raises:
        ValueError: If logits are requested with ``num_classes <= 0``.
    """
    return (
        *(
            _mask2former_output_specs(
                resolution,
                estimate_logits=estimate_logits,
                include_uncertainty=include_uncertainty,
                num_classes=num_classes,
            )
            if include_mask2former
            else ()
        ),
        *(_sam2_output_specs(resolution) if include_sam2 else ()),
        *auxiliary_specs,
    )


@dataclass(slots=True)
class SegmentationDeviceFramePipelineFactory:
    """Create resolution-specific Mask2Former and SAM2 pipelines lazily.

    An extending entrypoint may supply one additional processor through
    ``auxiliary_processor``; this module never names or imports it.

    The estimator import intentionally occurs only when a GPU service invokes
    the factory. Parent and CPU camera-worker processes can import and pickle
    this configuration without importing MMSeg, SAM2, or other model stacks.
    """

    estimate_logits: bool
    enable_trt: bool
    include_sam2: bool
    include_mask2former: bool = True
    output_uncertainty_mask: bool = False
    auxiliary_processor: AuxiliaryProcessorFactory | None = None
    _sam2_processor: Any | None = field(default=None, init=False, repr=False)
    _auxiliary_processor: Any | None = field(default=None, init=False, repr=False)

    def output_specs(
        self,
        resolution: tuple[int, int],
        *,
        num_classes: int,
    ) -> tuple[DeviceFrameOutputSpec, ...]:
        """Return the transport contract produced by this factory.

        Args:
            resolution: Raw image resolution as ``(width, height)``.
            num_classes: Number of Mask2Former semantic classes.

        Returns:
            Ordered output specifications for all configured processors.
        """
        return segmentation_output_specs(
            resolution,
            estimate_logits=self.estimate_logits,
            include_uncertainty=self.output_uncertainty_mask,
            num_classes=num_classes,
            include_sam2=self.include_sam2,
            include_mask2former=self.include_mask2former,
            auxiliary_specs=(
                () if self.auxiliary_processor is None else self.auxiliary_processor.output_specs(resolution)
            ),
        )

    def __call__(self, resolution: tuple[int, int]) -> DeviceFramePipeline:
        """Create a device pipeline for one image resolution.

        Args:
            resolution: Raw image resolution as ``(width, height)``.

        Returns:
            A pipeline containing optional Mask2Former, shared SAM2, and an
            optional extension-owned processor initialized on the current CUDA
            device.
        """
        mask2former = None
        if self.include_mask2former:
            from apps.av.aux_gen.estimators import Mask2FormerSegmentationEstimator

            mask2former = Mask2FormerSegmentationEstimator(
                list(resolution),
                estimate_logits=self.estimate_logits,
                enable_trt=self.enable_trt,
                output_uncertainty_mask=self.output_uncertainty_mask,
            )
        if self.include_sam2 and self._sam2_processor is None:
            from apps.av.aux_gen.estimators import PromptlessSAM2EgoMaskEstimator

            self._sam2_processor = PromptlessSAM2EgoMaskEstimator()
        if self.auxiliary_processor is not None and self._auxiliary_processor is None:
            self._auxiliary_processor = self.auxiliary_processor(
                semantic_class_names=tuple(mask2former.stuff_classes) if mask2former is not None else (),
            )
        processors = processor_directory(
            *((mask2former,) if mask2former is not None else ()),
            *((self._sam2_processor,) if self._sam2_processor is not None else ()),
            *((self._auxiliary_processor,) if self._auxiliary_processor is not None else ()),
        )
        return DeviceFramePipeline(
            input_spec=(
                mask2former.get_device_input_spec()
                if mask2former is not None
                else ((1, 3, resolution[1], resolution[0]), torch.uint8)
            ),
            processors=processors,
        )
