# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic contracts for processing shared image frames on a device."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DeviceFrameOutputSpec:
    """Describe one output produced by a device-frame processor.

    Attributes:
        name: Stable semantic name used to route the output.
        shape: Unbatched output tensor shape.
        dtype: Torch dtype used for device and shared-memory staging.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class DeviceFrameProcessorMetadata:
    """One processor's metadata, split by who consumes it.

    A live processor owns CUDA state and cannot cross a process boundary, so
    the service sends these dictionaries in its readiness message instead.
    Consumers read the subset they need by name; no consumer merges them.

    Attributes:
        storage: Writer-facing metadata persisted with the auxiliary data.
        backend: Inference metadata used only to validate service readiness.
    """

    storage: dict[str, Any]
    backend: dict[str, Any]

    @classmethod
    def directory(cls, processors: Mapping[str, "DeviceFrameProcessor"]) -> dict[str, "DeviceFrameProcessorMetadata"]:
        """Collect metadata from every processor in a pipeline directory.

        Args:
            processors: Live processors keyed by ``processor.name``.

        Returns:
            The same keys, each mapped to that processor's metadata.
        """
        return {name: processor.metadata() for name, processor in processors.items()}


def processor_directory(*processors: "DeviceFrameProcessor") -> dict[str, "DeviceFrameProcessor"]:
    """Build an insertion-ordered processor directory keyed by name.

    Args:
        processors: Processors in execution order.

    Returns:
        Mapping from ``processor.name`` to the processor.

    Raises:
        ValueError: If two processors share a name.
    """
    directory: dict[str, DeviceFrameProcessor] = {}
    for processor in processors:
        if processor.name in directory:
            raise ValueError(f"Duplicate device-frame processor name {processor.name}")
        directory[processor.name] = processor
    return directory


class DeviceFrameProcessor(Protocol):
    """Process a frame already staged on a device."""

    name: str

    def metadata(self) -> DeviceFrameProcessorMetadata:
        """Return serializable metadata describing this processor.

        Returns:
            Storage and backend metadata suitable for transport to the parent
            process.
        """
        ...

    def output_specs(self, resolution: tuple[int, int]) -> tuple[DeviceFrameOutputSpec, ...]:
        """Describe every output the processor may produce.

        Args:
            resolution: Frame resolution as ``(width, height)``.

        Returns:
            Ordered output specifications for this processor.
        """
        ...

    def process_device(
        self,
        image_bgr_nchw: torch.Tensor,
        *,
        outputs: Mapping[str, torch.Tensor],
        processor_inputs: Mapping[str, torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> None:
        """Enqueue device processing into caller-owned outputs.

        Args:
            image_bgr_nchw: Contiguous BGR device frame with shape
                ``[1, 3, H, W]`` and the pipeline-declared dtype.
            outputs: Caller-owned CUDA tensors keyed by declared output name.
                Each tensor matches the processor's declared shape and dtype
                and remains valid for the pipeline-slot lifetime.
            processor_inputs: Read-only CUDA tensors keyed by semantic name.
                Each retains the shape and dtype of its static shared input
                and remains valid for the service lifetime.
            stream: CUDA stream owning processing and output writes.

        Returns:
            None. Work is enqueued into ``stream``.

        Raises:
            ValueError: If an input or output violates the processor contract.
            RuntimeError: If device processing cannot produce a required
                output.
        """
        ...


def process_device_frame_from_host(
    processor: DeviceFrameProcessor,
    image_rgb_hwc: NDArray[np.uint8],
    *,
    input_spec: tuple[tuple[int, ...], torch.dtype],
    device: torch.device,
    processor_inputs: Mapping[str, NDArray[Any] | torch.Tensor] | None = None,
    stream: torch.cuda.Stream | None = None,
) -> dict[str, NDArray[Any]]:
    """Run one device-frame processor from host RGB input.

    Args:
        processor: Generic processor implementing the device-frame contract.
        image_rgb_hwc: Host ``uint8[H, W, 3]`` image in RGB channel order.
        input_spec: Processor input shape and dtype as ``([1, 3, H, W],
            dtype)``.
        device: Device receiving the staged input, outputs, and processor
            inputs.
        processor_inputs: Optional host arrays or tensors keyed by semantic
            processor-input name.
        stream: Optional CUDA stream. The current stream on ``device`` is used
            by default.

    Returns:
        Host-owned NumPy outputs keyed by the processor's semantic output
        names.

    Raises:
        ValueError: If the image, input shape, or output names violate the
            processor contract.
    """
    image_rgb = np.asarray(image_rgb_hwc)
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError(f"Host frame must be RGB uint8[H, W, 3], got {image_rgb.dtype} {image_rgb.shape}")

    input_shape, input_dtype = input_spec
    height, width = image_rgb.shape[:2]
    if input_shape != (1, 3, height, width):
        raise ValueError(f"Processor input shape {input_shape} does not match host frame {image_rgb.shape}")

    resolution = (width, height)
    output_specs = processor.output_specs(resolution)
    output_names = [spec.name for spec in output_specs]
    if len(output_names) != len(set(output_names)):
        raise ValueError(f"Host processor output names must be unique, got {output_names}")

    image_rgb_tensor = torch.from_numpy(np.array(image_rgb, copy=True))
    image_bgr_nchw = (
        image_rgb_tensor.to(device=device, dtype=input_dtype).permute(2, 0, 1).flip(0).unsqueeze(0).contiguous()
    )
    outputs = {spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in output_specs}
    inputs_device = {
        name: value.to(device=device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
        for name, value in (processor_inputs or {}).items()
    }
    processing_stream = stream or torch.cuda.current_stream(device=device)
    processor.process_device(
        image_bgr_nchw,
        outputs=outputs,
        processor_inputs=inputs_device,
        stream=processing_stream,
    )
    processing_stream.synchronize()
    return {name: output.cpu().numpy().copy() for name, output in outputs.items()}


@dataclass(frozen=True, slots=True)
class DeviceFramePipeline:
    """Bind a frame input contract to an ordered processor composition.

    Attributes:
        input_spec: Device input shape and dtype as ``(shape, dtype)``.
        processors: Named processors in registration order, keyed by
            ``processor.name``.
    """

    input_spec: tuple[tuple[int, ...], torch.dtype]
    processors: dict[str, DeviceFrameProcessor]

    def __post_init__(self) -> None:
        """Reject directory keys that do not match ``processor.name``."""
        mismatched = [name for name, processor in self.processors.items() if processor.name != name]
        if mismatched:
            raise ValueError(f"Processor directory keys must match processor names, got {mismatched}")
