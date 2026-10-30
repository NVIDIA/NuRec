# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory transport and CUDA service for generic frame processors."""

import os
import queue
import time
import traceback

from contextlib import nullcontext
from dataclasses import dataclass, field
from math import prod
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Mapping

import numpy as np
import torch

from numpy.typing import NDArray

from apps.av.aux_gen.device_frame import (
    DeviceFrameOutputSpec,
    DeviceFramePipeline,
    DeviceFrameProcessor,
    DeviceFrameProcessorMetadata,
)


@dataclass(frozen=True, slots=True)
class DeviceFrameSharedArrayDescriptor:
    """Describe one named array stored in shared memory."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    shared_memory_name: str

    @property
    def output_spec(self) -> DeviceFrameOutputSpec:
        """Return this descriptor's processor output contract."""
        return DeviceFrameOutputSpec(self.name, self.shape, self.dtype)


@dataclass(frozen=True, slots=True)
class DeviceFrameSharedMemorySlot:
    """Describe parent-owned input and processor output buffers.

    Attributes:
        input_name: Name of an ``[H, W, 3]`` RGB ``uint8`` input buffer.
        resolution: Image resolution as ``(width, height)``.
        outputs: Ordered processor output descriptors.
    """

    input_name: str
    resolution: tuple[int, int]
    outputs: tuple[DeviceFrameSharedArrayDescriptor, ...]

    @property
    def output_specs(self) -> tuple[DeviceFrameOutputSpec, ...]:
        """Return the output contract independently of shared-memory names.

        Returns:
            Ordered processor output names, shapes, and torch dtypes.
        """
        return tuple(output.output_spec for output in self.outputs)


@dataclass(slots=True)
class DeviceFrameSharedBuffers:
    """Expose arrays and handles backing one generic frame transport.

    Attributes:
        slot: Serializable descriptor passed between service processes.
        input_buffer: RGB ``uint8[H, W, 3]`` input array.
        output_buffers: Processor outputs keyed by stable output name.
    """

    slot: DeviceFrameSharedMemorySlot
    input_buffer: NDArray[np.uint8]
    output_buffers: dict[str, NDArray[Any]]
    _shared_memories: tuple[SharedMemory, ...]
    _unlink_on_close: bool

    @classmethod
    def _from_handles(
        cls,
        slot: DeviceFrameSharedMemorySlot,
        shared_memories: tuple[SharedMemory, ...],
        *,
        unlink_on_close: bool,
    ) -> "DeviceFrameSharedBuffers":
        """Build typed views over shared-memory handles.

        Args:
            slot: Descriptor defining output names, shapes, and dtypes.
            shared_memories: Handles ordered as input followed by ``outputs``.
            unlink_on_close: Whether :meth:`close` owns unlinking the handles.

        Returns:
            Typed shared arrays with their lifetime-owning handles.

        Raises:
            ValueError: If the handle count does not match ``slot``.
        """
        expected_count = 1 + len(slot.outputs)
        if len(shared_memories) != expected_count:
            raise ValueError(f"Shared-memory slot needs {expected_count} handles, received {len(shared_memories)}")

        width, height = slot.resolution
        output_buffers: dict[str, NDArray[Any]] = {
            output.name: np.ndarray(
                output.shape,
                dtype=_numpy_dtype(output.dtype),
                buffer=shared_memory.buf,
            )
            for output, shared_memory in zip(slot.outputs, shared_memories[1:])
        }

        return cls(
            slot=slot,
            input_buffer=np.ndarray(
                (height, width, 3),
                dtype=np.uint8,
                buffer=shared_memories[0].buf,
            ),
            output_buffers=output_buffers,
            _shared_memories=shared_memories,
            _unlink_on_close=unlink_on_close,
        )

    @classmethod
    def create(
        cls,
        resolution: tuple[int, int],
        output_specs: tuple[DeviceFrameOutputSpec, ...],
    ) -> "DeviceFrameSharedBuffers":
        """Allocate a parent-owned shared-memory transport.

        Args:
            resolution: Camera resolution as ``(width, height)``.
            output_specs: Unique processor output contracts.

        Returns:
            Owned transport whose :meth:`close` also unlinks all buffers.

        Raises:
            ValueError: If output specifications contain duplicate names.
            Exception: Propagates allocation failures after releasing any
                buffers already allocated by this call.
        """
        output_names = [spec.name for spec in output_specs]
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"Shared-memory output names must be unique, got {output_names}")

        width, height = resolution
        shared_memories: list[SharedMemory] = []
        try:
            input_shared_memory = SharedMemory(
                create=True,
                size=width * height * 3 * np.dtype(np.uint8).itemsize,
            )
            shared_memories.append(input_shared_memory)
            shared_outputs: list[DeviceFrameSharedArrayDescriptor] = []
            for spec in output_specs:
                output_shared_memory = SharedMemory(
                    create=True,
                    size=prod(spec.shape) * _numpy_dtype(spec.dtype).itemsize,
                )
                shared_memories.append(output_shared_memory)
                shared_outputs.append(
                    DeviceFrameSharedArrayDescriptor(
                        name=spec.name,
                        shape=spec.shape,
                        dtype=spec.dtype,
                        shared_memory_name=output_shared_memory.name,
                    )
                )

            slot = DeviceFrameSharedMemorySlot(
                input_name=input_shared_memory.name,
                resolution=resolution,
                outputs=tuple(shared_outputs),
            )
            return cls._from_handles(slot, tuple(shared_memories), unlink_on_close=True)
        except Exception:
            for shared_memory in shared_memories:
                shared_memory.close()
                try:
                    shared_memory.unlink()
                except FileNotFoundError:
                    pass
            raise

    @classmethod
    def attach(cls, slot: DeviceFrameSharedMemorySlot) -> "DeviceFrameSharedBuffers":
        """Attach typed views to an existing parent-owned transport.

        Args:
            slot: Descriptor carrying shared-memory names and array shapes.

        Returns:
            Attached transport whose :meth:`close` does not unlink buffers.

        Raises:
            Exception: Propagates attachment failures after closing handles
                opened earlier in this call.
        """
        names = [slot.input_name, *(output.shared_memory_name for output in slot.outputs)]

        shared_memories: list[SharedMemory] = []
        try:
            # ``track=False`` prevents child resource trackers from unlinking
            # memory owned by the parent process.
            for name in names:
                shared_memories.append(SharedMemory(name=name, create=False, track=False))
            return cls._from_handles(slot, tuple(shared_memories), unlink_on_close=False)
        except Exception:
            for shared_memory in shared_memories:
                shared_memory.close()
            raise

    def close(self) -> None:
        """Close handles and unlink them when this instance owns the memory.

        Returns:
            None.
        """
        for shared_memory in self._shared_memories:
            shared_memory.close()
            if self._unlink_on_close:
                try:
                    shared_memory.unlink()
                except FileNotFoundError:
                    pass


@dataclass(slots=True)
class DeviceFrameSharedArray:
    """Own or attach a typed named shared array.

    Attributes:
        descriptor: Serializable array contract sent with frame requests.
        array: NumPy view over the shared-memory storage.
    """

    descriptor: DeviceFrameSharedArrayDescriptor
    array: NDArray[Any]
    _shared_memory: SharedMemory
    _unlink_on_close: bool

    @classmethod
    def _from_handle(
        cls,
        descriptor: DeviceFrameSharedArrayDescriptor,
        shared_memory: SharedMemory,
        *,
        unlink_on_close: bool,
    ) -> "DeviceFrameSharedArray":
        """Build a typed array view over one shared-memory handle.

        Args:
            descriptor: Semantic name, shape, dtype, and shared-memory name.
            shared_memory: Open handle backing the returned array.
            unlink_on_close: Whether :meth:`close` unlinks the object.

        Returns:
            A shared array retaining the handle for the view's lifetime.
        """
        return cls(
            descriptor=descriptor,
            array=np.ndarray(
                descriptor.shape,
                dtype=_numpy_dtype(descriptor.dtype),
                buffer=shared_memory.buf,
            ),
            _shared_memory=shared_memory,
            _unlink_on_close=unlink_on_close,
        )

    @classmethod
    def create(cls, name: str, array: Any) -> "DeviceFrameSharedArray":
        """Create and populate an owned static processor input.

        Args:
            name: Stable semantic input name exposed to processors.
            array: Contiguous-compatible source array with a supported dtype.

        Returns:
            Owned shared array whose :meth:`close` unlinks its storage.

        Raises:
            ValueError: If the input is empty or has an unsupported dtype.
        """
        source = np.asarray(array)
        if source.size == 0:
            raise ValueError(f"Static processor input {name} must not be empty")
        dtype = _torch_dtype(source.dtype)
        shared_memory = SharedMemory(create=True, size=source.nbytes)
        descriptor = DeviceFrameSharedArrayDescriptor(
            name=name,
            shape=tuple(source.shape),
            dtype=dtype,
            shared_memory_name=shared_memory.name,
        )
        try:
            shared_array = cls._from_handle(descriptor, shared_memory, unlink_on_close=True)
            np.copyto(shared_array.array, source)
            return shared_array
        except Exception:
            shared_memory.close()
            try:
                shared_memory.unlink()
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def attach(cls, descriptor: DeviceFrameSharedArrayDescriptor) -> "DeviceFrameSharedArray":
        """Attach to an existing static processor input.

        Args:
            descriptor: Serializable shared-array contract.

        Returns:
            Attached array whose :meth:`close` leaves ownership unchanged.
        """
        shared_memory = SharedMemory(
            name=descriptor.shared_memory_name,
            create=False,
            track=False,
        )
        try:
            return cls._from_handle(descriptor, shared_memory, unlink_on_close=False)
        except Exception:
            shared_memory.close()
            raise

    def close(self) -> None:
        """Close this handle and unlink storage when owned.

        Returns:
            None.
        """
        self._shared_memory.close()
        if self._unlink_on_close:
            try:
                self._shared_memory.unlink()
            except FileNotFoundError:
                pass


@dataclass(frozen=True, slots=True)
class DeviceFrameRequest:
    """Represent one request whose pixels already occupy a worker's slot.

    Attributes:
        worker_id: Stable index used to select per-worker queues and staging.
        request_id: Monotonic identifier within one worker.
        slot: Shared-memory buffers owned by the parent process.
        enabled_processors: Processor names selected for this frame.
        processor_inputs: Static shared inputs selected for this request.
    """

    worker_id: int
    request_id: int
    slot: DeviceFrameSharedMemorySlot
    enabled_processors: frozenset[str]
    processor_inputs: tuple[DeviceFrameSharedArrayDescriptor, ...] = ()


@dataclass(frozen=True, slots=True)
class DeviceFrameResult:
    """Report completion after output shared memory is safe to read.

    Attributes:
        request_id: Identifier copied from the completed request.
        error: Formatted traceback on failure, otherwise ``None``.
        output_names: Names of outputs written for this request.
    """

    request_id: int
    error: str | None
    output_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PendingDeviceFrame:
    """Retain one submitted request and its borrowed transport."""

    request: DeviceFrameRequest
    shared_buffers: DeviceFrameSharedBuffers


@dataclass(frozen=True, slots=True)
class ResolvedDeviceFrame:
    """Own copied outputs from one completed device-frame request."""

    outputs: dict[str, NDArray[Any]]
    input_image: NDArray[np.uint8] | None


@dataclass(kw_only=True, slots=True)
class SharedDeviceFrameClient:
    """Submit host images to shared device-frame services and resolve outputs."""

    worker_id: int
    slot_queue: Any
    request_queue: Any
    result_queue: Any
    stop_event: Any
    _shared_buffers_by_name: dict[str, DeviceFrameSharedBuffers] = field(default_factory=dict, init=False, repr=False)
    _completed_results: dict[int, DeviceFrameResult] = field(default_factory=dict, init=False, repr=False)

    def submit(
        self,
        request_id: int,
        image: Any,
        enabled_processors: frozenset[str],
        processor_inputs: tuple[DeviceFrameSharedArrayDescriptor, ...] = (),
    ) -> PendingDeviceFrame | None:
        """Copy one image into a borrowed slot and queue its descriptor.

        Args:
            request_id: Monotonic request identifier for this worker.
            image: RGB image convertible to ``uint8[H, W, 3]``.
            enabled_processors: Processor names selected for this frame.
            processor_inputs: Static shared inputs selected for this request.

        Returns:
            Pending request retaining its slot, or ``None`` when no compatible
            slot is currently available.

        Raises:
            RuntimeError: If the shared service has stopped.
            TypeError: If the slot queue contains an invalid descriptor.
            ValueError: If the image does not match the borrowed slot.
        """
        if self.stop_event.is_set():
            raise RuntimeError("Shared device-frame service stopped")
        try:
            slot = self.slot_queue.get_nowait()
        except queue.Empty:
            return None
        if not isinstance(slot, DeviceFrameSharedMemorySlot):
            raise TypeError(f"Unexpected device-frame shared-memory slot type: {type(slot)}")

        submitted = False
        try:
            shared_buffers = self._shared_buffers_by_name.get(slot.input_name)
            if shared_buffers is None:
                shared_buffers = DeviceFrameSharedBuffers.attach(slot)
                self._shared_buffers_by_name[slot.input_name] = shared_buffers
            input_image = np.asarray(image, dtype=np.uint8)
            if input_image.shape != shared_buffers.input_buffer.shape:
                raise ValueError(
                    f"Image shape {input_image.shape} for worker {self.worker_id} does not match "
                    f"{shared_buffers.input_buffer.shape}"
                )
            np.copyto(shared_buffers.input_buffer, input_image)

            request = DeviceFrameRequest(
                worker_id=self.worker_id,
                request_id=request_id,
                slot=shared_buffers.slot,
                enabled_processors=enabled_processors,
                processor_inputs=processor_inputs,
            )
            while True:
                try:
                    self.request_queue.put(request, timeout=0.2)
                    submitted = True
                    return PendingDeviceFrame(request=request, shared_buffers=shared_buffers)
                except queue.Full:
                    if self.stop_event.is_set():
                        raise RuntimeError("Shared device-frame service stopped before accepting a request")
        finally:
            if not submitted:
                self.slot_queue.put(slot)

    def resolve(
        self,
        pending_frame: PendingDeviceFrame,
        *,
        include_input: bool = False,
    ) -> ResolvedDeviceFrame:
        """Wait for a request and copy only the outputs produced for it.

        Args:
            pending_frame: Submitted request retaining a borrowed slot.
            include_input: Whether to retain a copy of the original RGB image.

        Returns:
            Owned output arrays and optional input image. The shared slot has
            already been returned to the pool.

        Raises:
            RuntimeError: If the service stops, fails, or returns inconsistent
                completion state.
            TypeError: If the result queue contains an invalid payload.
        """
        request = pending_frame.request
        shared_buffers = pending_frame.shared_buffers
        try:
            while request.request_id not in self._completed_results:
                try:
                    result = self.result_queue.get(timeout=0.2)
                except queue.Empty:
                    if not self.stop_event.is_set():
                        continue
                    try:
                        result = self.result_queue.get(timeout=0.2)
                    except queue.Empty as error:
                        raise RuntimeError("Shared device-frame service failed while a request was pending") from error

                if not isinstance(result, DeviceFrameResult):
                    raise TypeError(f"Unexpected device-frame result type: {type(result)}")
                if result.error is not None:
                    raise RuntimeError(f"Shared device-frame service failed:\n{result.error}")
                if result.request_id in self._completed_results:
                    raise RuntimeError(f"Received duplicate device-frame result {result.request_id}")
                self._completed_results[result.request_id] = result

            result = self._completed_results.pop(request.request_id)
            if result.request_id != request.request_id:
                raise RuntimeError(f"Expected device-frame result {request.request_id}, received {result.request_id}")

            try:
                outputs = {name: shared_buffers.output_buffers[name].copy() for name in result.output_names}
            except KeyError as error:
                raise RuntimeError(f"Completed output {error.args[0]} has no shared-memory buffer") from error
            input_image = shared_buffers.input_buffer.copy() if include_input else None
            return ResolvedDeviceFrame(
                outputs=outputs,
                input_image=input_image,
            )
        finally:
            self.slot_queue.put(shared_buffers.slot)

    def close(self) -> None:
        """Close every shared-memory attachment cached by this client.

        Returns:
            None.
        """
        for shared_buffers in self._shared_buffers_by_name.values():
            shared_buffers.close()


@dataclass(frozen=True, slots=True)
class DeviceFrameServiceReady:
    """Report processor metadata or a startup error to the parent process.

    Attributes:
        service_id: Stable process identity within the shared service pool.
        device_id: Logical CUDA device owned by the reporting service.
        processor_metadata: Metadata subsets for each registered processor,
            keyed by ``processor.name``, or ``None`` when startup failed.
        error: Formatted startup traceback, otherwise ``None``.
    """

    service_id: int
    device_id: int
    processor_metadata: dict[str, DeviceFrameProcessorMetadata] | None
    error: str | None


_ENABLE_NVTX = os.getenv("NUREC_MASK2FORMER_NVTX") == "1"


def _numpy_dtype(dtype: torch.dtype) -> np.dtype[Any]:
    """Return the shared-memory NumPy dtype for a device-frame tensor.

    Args:
        dtype: Torch output dtype.

    Returns:
        Equivalent NumPy dtype.

    Raises:
        ValueError: If shared-memory transport does not support ``dtype``.
    """
    supported = {
        torch.bool: np.dtype(np.bool_),
        torch.uint8: np.dtype(np.uint8),
        torch.int32: np.dtype(np.int32),
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
    }
    try:
        return supported[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported shared-memory output dtype {dtype}") from error


def _torch_dtype(dtype: np.dtype[Any]) -> torch.dtype:
    """Return the device-frame Torch dtype for a shared array.

    Args:
        dtype: NumPy array dtype.

    Returns:
        Equivalent supported Torch dtype.

    Raises:
        ValueError: If shared-memory transport does not support ``dtype``.
    """
    supported = {
        np.dtype(np.bool_): torch.bool,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
    }
    try:
        return supported[np.dtype(dtype)]
    except KeyError as error:
        raise ValueError(f"Unsupported shared-memory input dtype {dtype}") from error


def _nvtx_range(message: str) -> Any:
    """Create an optional NVTX range.

    Args:
        message: Label shown in an Nsight Systems trace.

    Returns:
        An NVTX context manager when tracing is enabled, otherwise a no-op
        context manager.
    """
    return torch.cuda.nvtx.range(message) if _ENABLE_NVTX else nullcontext()


def _run_device_frame_processors(
    processors: Mapping[str, DeviceFrameProcessor],
    request: DeviceFrameRequest,
    image_bgr_nchw: torch.Tensor,
    outputs: Mapping[str, torch.Tensor],
    processor_inputs: Mapping[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> tuple[str, ...]:
    """Enqueue every enabled processor and return its ordered output names.

    Args:
        processors: Registered processors in execution order.
        request: Frame request controlling optional processors.
        image_bgr_nchw: Contiguous BGR ``[1, 3, H, W]`` CUDA image with the
            pipeline-declared dtype.
        outputs: Caller-owned CUDA staging tensors matching every processor's
            declared output shape and dtype. They remain valid for the
            pipeline-slot lifetime.
        processor_inputs: Read-only CUDA tensors retaining their static shared
            input shape and dtype for the service lifetime.
        stream: CUDA stream owning all processor work.

    Returns:
        Stable output names to copy back after processing.

    Raises:
        RuntimeError: If processors declare duplicate outputs.
        ValueError: If the request enables an unregistered processor or a
            declared output has no CUDA staging tensor.
    """
    unknown_processors = request.enabled_processors - processors.keys()
    if unknown_processors:
        raise ValueError(f"Frame request enabled unregistered processors {sorted(unknown_processors)}")

    output_names: list[str] = []
    for processor in processors.values():
        if processor.name not in request.enabled_processors:
            continue
        processor_outputs = tuple(spec.name for spec in processor.output_specs(request.slot.resolution))
        duplicate_outputs = set(output_names).intersection(processor_outputs)
        if duplicate_outputs:
            raise RuntimeError(f"Device processors declared duplicate outputs {sorted(duplicate_outputs)}")
        missing_outputs = [name for name in processor_outputs if name not in outputs]
        if missing_outputs:
            raise ValueError(f"Device processor {processor.name} has no staging for outputs {missing_outputs}")
        with _nvtx_range(f"{processor.name}/processing/{request.worker_id}/{request.request_id}"):
            processor.process_device(
                image_bgr_nchw,
                outputs=outputs,
                processor_inputs=processor_inputs,
                stream=stream,
            )
        output_names.extend(processor_outputs)
    return tuple(output_names)


def _get_device_frame_output_specs(
    processors: Mapping[str, DeviceFrameProcessor],
    resolution: tuple[int, int],
) -> tuple[DeviceFrameOutputSpec, ...]:
    """Collect unique output staging specifications from processors.

    Args:
        processors: Registered processors in execution order.
        resolution: Frame resolution as ``(width, height)``.

    Returns:
        Ordered output specifications used to allocate CUDA and pinned staging.

    Raises:
        RuntimeError: If processors declare the same output name.
    """
    specs: list[DeviceFrameOutputSpec] = []
    names: set[str] = set()
    for processor in processors.values():
        for spec in processor.output_specs(resolution):
            if spec.name in names:
                raise RuntimeError(f"Device processors declared duplicate output {spec.name}")
            names.add(spec.name)
            specs.append(spec)
    return tuple(specs)


def _shaped_storage_view(
    storage: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Return a shaped prefix view after validating fixed storage.

    Args:
        storage: Capacity-sized backing tensor.
        shape: Requested view shape.
        dtype: Required storage dtype.
        name: Contract name included in validation errors.

    Returns:
        A view with the requested shape.

    Raises:
        ValueError: If storage has the wrong dtype or insufficient capacity.
    """
    numel = prod(shape)
    if storage.dtype != dtype or numel > storage.numel():
        raise ValueError(f"{name} exceeds fixed staging contract")
    return storage[:numel].view(shape)


@dataclass(frozen=True, slots=True)
class _HostDeviceStorage:
    """Pair fixed host and device storage under one tensor contract."""

    host: torch.Tensor
    device: torch.Tensor

    @classmethod
    def allocate(cls, shape: tuple[int, ...], dtype: torch.dtype) -> "_HostDeviceStorage":
        """Allocate capacity-sized pinned host and CUDA device storage.

        Args:
            shape: Maximum tensor shape.
            dtype: Shared host and device dtype.

        Returns:
            Paired flat storage with capacity for ``shape``.
        """
        numel = prod(shape)
        return cls(
            host=torch.empty(numel, dtype=dtype, pin_memory=True),
            device=torch.empty(numel, dtype=dtype, device="cuda"),
        )

    def shaped(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        name: str,
    ) -> "_HostDeviceStorage":
        """Return matching prefix views into host and device storage.

        Args:
            shape: Requested tensor shape.
            dtype: Required dtype for both storage tensors.
            name: Contract name included in validation errors.

        Returns:
            Paired host and device views with the requested shape.

        Raises:
            ValueError: If either storage tensor has the wrong dtype or
                insufficient capacity.
        """
        return type(self)(
            host=_shaped_storage_view(self.host, shape, dtype, f"{name} host storage"),
            device=_shaped_storage_view(self.device, shape, dtype, f"{name} device storage"),
        )


@dataclass(slots=True)
class _DevicePipelineSlot:
    """Hold capacity-sized processor staging and completion events.

    Storage tensors are allocated once for the largest allowed resolution.
    Request-facing tensors are shaped prefix views into that fixed storage.
    """

    resolution: tuple[int, int]
    output_specs: tuple[DeviceFrameOutputSpec, ...]
    input_rgb_storage: _HostDeviceStorage
    processor_input_device_storage: torch.Tensor
    outputs_storage: dict[str, _HostDeviceStorage]
    input_rgb: _HostDeviceStorage
    processor_input_device: torch.Tensor
    outputs: dict[str, _HostDeviceStorage]
    processor_inputs_device: dict[str, torch.Tensor]
    h2d_done: torch.cuda.Event
    processing_done: torch.cuda.Event
    d2h_done: torch.cuda.Event
    active_output_names: tuple[str, ...] = ()
    request: DeviceFrameRequest | None = None

    def bind(
        self,
        resolution: tuple[int, int],
        input_shape: tuple[int, ...],
        input_dtype: torch.dtype,
        output_specs: tuple[DeviceFrameOutputSpec, ...],
    ) -> None:
        """Bind request-shaped views without reallocating backing storage.

        Args:
            resolution: Request resolution as ``(width, height)``.
            input_shape: Processor input shape for this request.
            input_dtype: Processor input dtype for this request.
            output_specs: Processor output contracts for this request.

        Returns:
            None.

        Raises:
            ValueError: If the request exceeds or disagrees with the fixed
                storage contract.
        """
        width, height = resolution
        input_rgb = self.input_rgb_storage.shaped(
            (height, width, 3),
            torch.uint8,
            "RGB input",
        )
        processor_input_device = _shaped_storage_view(
            self.processor_input_device_storage,
            input_shape,
            input_dtype,
            "Processor input device storage",
        )
        outputs: dict[str, _HostDeviceStorage] = {}
        for spec in output_specs:
            output_storage = self.outputs_storage.get(spec.name)
            if output_storage is None:
                raise ValueError(f"Output {spec.name} exceeds fixed staging contract")
            outputs[spec.name] = output_storage.shaped(
                spec.shape,
                spec.dtype,
                f"Output {spec.name}",
            )

        self.resolution = resolution
        self.output_specs = output_specs
        self.input_rgb = input_rgb
        self.processor_input_device = processor_input_device
        self.outputs = outputs


def _schedule_cuda_pipeline_stages(
    *,
    h2d_stream: Any,
    processing_stream: Any,
    d2h_stream: Any,
    h2d_done: Any,
    processing_done: Any,
    d2h_done: Any,
    h2d_task: Callable[[Any], None],
    processing_task: Callable[[Any], None],
    d2h_task: Callable[[Any], None],
) -> None:
    """Enqueue H2D, processing, and D2H tasks with event dependencies.

    Args:
        h2d_stream: Stream passed to ``h2d_task``.
        processing_stream: Stream that waits for H2D completion.
        d2h_stream: Stream that waits for processing completion.
        h2d_done: Event recorded after ``h2d_task`` is enqueued.
        processing_done: Event recorded after ``processing_task`` is enqueued.
        d2h_done: Event recorded after ``d2h_task`` is enqueued.
        h2d_task: Callable that enqueues the host-to-device stage.
        processing_task: Callable that enqueues model/post-processing work.
        d2h_task: Callable that enqueues the device-to-host stage.

    Returns:
        None. This function does not synchronize the host.

    Raises:
        Exception: Propagates task or stream/event enqueue failures.
    """
    h2d_task(h2d_stream)
    h2d_done.record(h2d_stream)
    processing_stream.wait_event(h2d_done)
    processing_task(processing_stream)
    processing_done.record(processing_stream)
    d2h_stream.wait_event(processing_done)
    d2h_task(d2h_stream)
    d2h_done.record(d2h_stream)


def run_device_frame_service_process(
    *,
    service_id: int,
    device_id: int,
    initial_resolution: tuple[int, int],
    request_queue: Any,
    result_queues: list[Any],
    ready_queue: Any,
    stop_event: Any,
    pipeline_factory: Callable[[tuple[int, int]], DeviceFramePipeline],
    allowed_resolutions: frozenset[tuple[int, int]],
    max_staging_slots: int = 3,
) -> None:
    """Own CUDA model state and serve bounded shared-memory requests.

    Args:
        service_id: Stable process identity within the shared service pool.
        device_id: CUDA device index owned by this process.
        initial_resolution: First ``(width, height)`` model resolution to load.
        request_queue: Bounded multi-producer request queue; ``None`` closes it.
        result_queues: One bounded completion queue per worker.
        ready_queue: Startup queue receiving one
            :class:`DeviceFrameServiceReady`.
        stop_event: Cross-process event requesting failure shutdown.
        pipeline_factory: Picklable callable constructing one generic
            resolution-specific device-frame pipeline.
        max_staging_slots: Maximum pinned/device staging sets retained by this
            service, independent of global shared-memory transport IDs.
        allowed_resolutions: Finite set of resolutions this service may cache
            pipelines for and uses to size fixed staging storage.

    Returns:
        None. Successful outputs are written to request shared memory before
        their completion result is queued.

    Startup errors are returned through ``ready_queue`` and request-time
    errors through ``result_queues``; they do not escape this process entry
    point.
    """
    if max_staging_slots < 1:
        raise ValueError(f"max_staging_slots must be positive, got {max_staging_slots}")
    if not allowed_resolutions:
        raise ValueError("allowed_resolutions must not be empty")
    if initial_resolution not in allowed_resolutions:
        raise ValueError(f"Initial resolution {initial_resolution} is not allowed")
    staging_resolution = max(allowed_resolutions, key=lambda value: value[0] * value[1])
    pipelines: dict[tuple[int, int], DeviceFramePipeline] = {}
    shared_buffers_by_slot: dict[str, DeviceFrameSharedBuffers] = {}
    shared_inputs_by_memory_name: dict[str, DeviceFrameSharedArray] = {}
    shared_input_descriptors_by_memory_name: dict[str, DeviceFrameSharedArrayDescriptor] = {}
    shared_inputs_device_by_memory_name: dict[str, torch.Tensor] = {}
    pipeline_slots: list[_DevicePipelineSlot] = []
    pipeline_slots_by_transport: dict[str, _DevicePipelineSlot] = {}
    h2d_stream: torch.cuda.Stream
    processing_stream: torch.cuda.Stream
    d2h_stream: torch.cuda.Stream

    def get_pipeline(resolution: tuple[int, int]) -> DeviceFramePipeline:
        """Return a cached resolution-specific pipeline.

        Args:
            resolution: Frame resolution as ``(width, height)``.

        Returns:
            The cached or newly constructed generic device-frame pipeline.

        Raises:
            TypeError: If the factory does not return a pipeline contract.
            ValueError: If the resolution is outside the configured set.
        """
        if resolution not in allowed_resolutions:
            raise ValueError(
                f"Device-frame service received unexpected resolution {resolution}; "
                f"expected one of {sorted(allowed_resolutions)}"
            )
        pipeline = pipelines.get(resolution)
        if pipeline is None:
            pipeline = pipeline_factory(resolution)
            if not isinstance(pipeline, DeviceFramePipeline):
                raise TypeError(f"Pipeline factory returned unexpected type {type(pipeline)}")
            pipelines[resolution] = pipeline
        return pipeline

    def get_shared_buffers(request: DeviceFrameRequest) -> DeviceFrameSharedBuffers:
        """Return typed shared-memory views for one request slot.

        Args:
            request: Request carrying the worker ID and slot descriptor.

        Returns:
            A cached attachment that remains open until service shutdown.

        Raises:
            ValueError: If a shared-memory name is reused for a different slot.
        """
        slot_key = request.slot.input_name
        shared_buffers = shared_buffers_by_slot.get(slot_key)
        if shared_buffers is None:
            shared_buffers = DeviceFrameSharedBuffers.attach(request.slot)
            shared_buffers_by_slot[slot_key] = shared_buffers
        elif shared_buffers.slot != request.slot:
            raise ValueError(f"Shared-memory input {slot_key} was reused for a different slot")
        return shared_buffers

    def get_processor_inputs_device(request: DeviceFrameRequest) -> dict[str, torch.Tensor]:
        """Attach and stage this request's unique static inputs once.

        Args:
            request: Frame request carrying static shared-array descriptors.

        Returns:
            CUDA tensors keyed by semantic processor name. Each retains its
            descriptor's shape and dtype on the service device and is cached
            for the service lifetime.

        Raises:
            ValueError: If names are duplicated or a shared-memory name is
                reused with a different contract.
        """
        inputs_device: dict[str, torch.Tensor] = {}
        for descriptor in request.processor_inputs:
            if descriptor.name in inputs_device:
                raise ValueError(f"Frame request contains duplicate processor input {descriptor.name}")
            memory_name = descriptor.shared_memory_name
            cached_descriptor = shared_input_descriptors_by_memory_name.get(memory_name)
            if cached_descriptor is not None and cached_descriptor != descriptor:
                raise ValueError(f"Shared processor input {memory_name} was reused for a different contract")
            input_device = shared_inputs_device_by_memory_name.get(memory_name)
            if input_device is None:
                shared_input = DeviceFrameSharedArray.attach(descriptor)
                shared_inputs_by_memory_name[memory_name] = shared_input
                shared_input_descriptors_by_memory_name[memory_name] = descriptor
                input_device = torch.as_tensor(shared_input.array).to(device="cuda", non_blocking=False)
                shared_inputs_device_by_memory_name[memory_name] = input_device
            inputs_device[descriptor.name] = input_device
        return inputs_device

    def put_result(worker_id: int, result: DeviceFrameResult) -> None:
        """Queue a completion unless another process requested shutdown.

        Args:
            worker_id: Index selecting the destination result queue.
            result: Completion or failure payload to send.

        Returns:
            None.
        """
        while not stop_event.is_set():
            try:
                result_queues[worker_id].put(result, timeout=0.2)
                return
            except queue.Full:
                pass

    def get_pipeline_slot(
        request: DeviceFrameRequest,
        pipeline: DeviceFramePipeline,
    ) -> _DevicePipelineSlot:
        """Acquire bounded staging independently of the transport identity.

        Args:
            request: Request defining worker ID, resolution, and optional
                output buffers.
            pipeline: Resolution-specific input and processor contract.

        Returns:
            Idle compatible staging, or a newly allocated bounded slot.

        Raises:
            RuntimeError: If no staging is available.
            ValueError: If the pipeline input or output contract does not
                match the request.
        """
        slot_key = request.slot.input_name
        if slot_key in pipeline_slots_by_transport:
            raise RuntimeError(f"Shared-memory slot {slot_key} already has an in-flight request")

        width, height = request.slot.resolution
        input_shape, input_dtype = pipeline.input_spec
        expected_input_shape = (1, 3, height, width)
        if input_shape != expected_input_shape:
            raise ValueError(f"Pipeline input shape {input_shape} does not match expected {expected_input_shape}")

        output_specs = _get_device_frame_output_specs(pipeline.processors, request.slot.resolution)
        if output_specs != request.slot.output_specs:
            raise ValueError(
                f"Processor output contract {output_specs} does not match shared-memory contract "
                f"{request.slot.output_specs}"
            )

        idle_slot_index = next((index for index, slot in enumerate(pipeline_slots) if slot.request is None), None)
        if idle_slot_index is None and len(pipeline_slots) >= max_staging_slots:
            raise RuntimeError("Device-frame service has no idle staging slot")
        if idle_slot_index is not None:
            slot = pipeline_slots[idle_slot_index]
            slot.bind(request.slot.resolution, input_shape, input_dtype, output_specs)
            return slot

        staging_width, staging_height = staging_resolution
        capacity_pipeline = get_pipeline(staging_resolution)
        capacity_input_shape, capacity_input_dtype = capacity_pipeline.input_spec
        expected_capacity_input_shape = (1, 3, staging_height, staging_width)
        if capacity_input_shape != expected_capacity_input_shape:
            raise ValueError(
                f"Capacity pipeline input shape {capacity_input_shape} does not match "
                f"expected {expected_capacity_input_shape}"
            )
        if capacity_input_dtype != input_dtype:
            raise ValueError(
                f"Capacity pipeline input dtype {capacity_input_dtype} does not match request dtype {input_dtype}"
            )
        staging_input_numel = prod(capacity_input_shape)
        capacity_output_specs = _get_device_frame_output_specs(capacity_pipeline.processors, staging_resolution)
        outputs_storage = {
            spec.name: _HostDeviceStorage.allocate(spec.shape, spec.dtype) for spec in capacity_output_specs
        }
        input_rgb_storage = _HostDeviceStorage.allocate(
            (staging_height, staging_width, 3),
            torch.uint8,
        )
        processor_input_device_storage = torch.empty(staging_input_numel, dtype=input_dtype, device="cuda")

        slot = _DevicePipelineSlot(
            resolution=request.slot.resolution,
            output_specs=output_specs,
            input_rgb_storage=input_rgb_storage,
            processor_input_device_storage=processor_input_device_storage,
            outputs_storage=outputs_storage,
            input_rgb=input_rgb_storage,
            processor_input_device=processor_input_device_storage,
            outputs={},
            processor_inputs_device={},
            h2d_done=torch.cuda.Event(),
            processing_done=torch.cuda.Event(),
            d2h_done=torch.cuda.Event(),
        )
        slot.bind(request.slot.resolution, input_shape, input_dtype, output_specs)
        pipeline_slots.append(slot)
        return slot

    def schedule_pipeline_request(request: DeviceFrameRequest, pipeline: DeviceFramePipeline) -> None:
        """Enqueue one worker request across the three pipeline streams.

        Args:
            request: Request whose shared input is ready to stage.
            pipeline: Device-frame pipeline for the request resolution.

        Returns:
            None.

        Raises:
            RuntimeError: If the shared-memory slot already has an in-flight request.
            ValueError: If staging contracts do not match the request.
        """
        slot = get_pipeline_slot(request, pipeline)

        input_buffer = get_shared_buffers(request).input_buffer
        # Keep the host stage as a contiguous RGB uint8 memcpy. Layout, channel
        # order, and backend dtype conversion run on CUDA below.
        np.copyto(slot.input_rgb.host.numpy(), input_buffer)
        slot.processor_inputs_device = get_processor_inputs_device(request)
        slot.request = request
        pipeline_slots_by_transport[request.slot.input_name] = slot

        def h2d_task(stream: torch.cuda.Stream) -> None:
            """Copy pinned input to CUDA.

            Args:
                stream: CUDA stream that owns the non-blocking copy.

            Returns:
                None.
            """
            with torch.cuda.stream(stream):
                with _nvtx_range(f"device_frame/h2d/{request.worker_id}/{request.request_id}"):
                    slot.input_rgb.device.copy_(slot.input_rgb.host, non_blocking=True)

        def processing_task(stream: torch.cuda.Stream) -> None:
            """Enqueue registered device processors.

            Args:
                stream: CUDA stream shared by all enabled processors.

            Returns:
                None.

            """
            with torch.cuda.stream(stream):
                slot.processor_input_device.copy_(
                    slot.input_rgb.device.permute(2, 0, 1).flip(0).unsqueeze(0),
                    non_blocking=True,
                )
                slot.active_output_names = _run_device_frame_processors(
                    pipeline.processors,
                    request,
                    slot.processor_input_device,
                    {name: output.device for name, output in slot.outputs.items()},
                    slot.processor_inputs_device,
                    stream,
                )

        def d2h_task(stream: torch.cuda.Stream) -> None:
            """Copy compact outputs into pinned host staging.

            Args:
                stream: CUDA stream that owns the non-blocking output copies.

            Returns:
                None.

            """
            with torch.cuda.stream(stream):
                with _nvtx_range(f"processors/d2h/{request.worker_id}/{request.request_id}"):
                    for output_name in slot.active_output_names:
                        output = slot.outputs[output_name]
                        output.host.copy_(
                            output.device,
                            non_blocking=True,
                        )

        _schedule_cuda_pipeline_stages(
            h2d_stream=h2d_stream,
            processing_stream=processing_stream,
            d2h_stream=d2h_stream,
            h2d_done=slot.h2d_done,
            processing_done=slot.processing_done,
            d2h_done=slot.d2h_done,
            h2d_task=h2d_task,
            processing_task=processing_task,
            d2h_task=d2h_task,
        )

    def complete_pipeline_requests(pending_slot_keys: set[str]) -> bool:
        """Publish completed slots.

        Args:
            pending_slot_keys: Mutable set of shared-memory slots with
                in-flight requests; completed keys are removed in place.

        Returns:
            Whether at least one completion was published.

        Raises:
            RuntimeError: If completed slot state or requested outputs are
                missing.
        """
        completed_any = False
        for slot_key in tuple(pending_slot_keys):
            slot = pipeline_slots_by_transport[slot_key]
            if not slot.d2h_done.query():
                continue
            request = slot.request
            if request is None:
                raise RuntimeError(f"Pipeline slot {slot_key} completed without a request")

            shared_buffers = get_shared_buffers(request)
            for output_name in slot.active_output_names:
                try:
                    output_buffer = shared_buffers.output_buffers[output_name]
                    output_host = slot.outputs[output_name].host
                except KeyError as error:
                    raise RuntimeError(f"Completed output {output_name} has no transport staging") from error
                np.copyto(output_buffer, output_host.numpy())

            put_result(
                request.worker_id,
                DeviceFrameResult(
                    request_id=request.request_id,
                    error=None,
                    output_names=slot.active_output_names,
                ),
            )
            slot.active_output_names = ()
            slot.request = None
            del pipeline_slots_by_transport[slot_key]
            pending_slot_keys.remove(slot_key)
            completed_any = True
        return completed_any

    try:
        torch.cuda.set_device(device_id)

        first_pipeline = get_pipeline(initial_resolution)
        h2d_stream = torch.cuda.Stream()
        processing_stream = torch.cuda.Stream()
        d2h_stream = torch.cuda.Stream()
        ready_queue.put(
            DeviceFrameServiceReady(
                service_id=service_id,
                device_id=device_id,
                processor_metadata=DeviceFrameProcessorMetadata.directory(first_pipeline.processors),
                error=None,
            )
        )
    except Exception:
        ready_queue.put(
            DeviceFrameServiceReady(
                service_id=service_id,
                device_id=device_id,
                processor_metadata=None,
                error=traceback.format_exc(),
            )
        )
        stop_event.set()
        return

    try:
        pending_slot_keys: set[str] = set()
        input_closed = False
        while not stop_event.is_set() and (not input_closed or pending_slot_keys):
            completed_any = complete_pipeline_requests(pending_slot_keys)

            if input_closed or len(pending_slot_keys) >= max_staging_slots:
                if pending_slot_keys and not completed_any:
                    time.sleep(0.001)
                continue

            try:
                request = request_queue.get(timeout=0.01 if pending_slot_keys else 0.2)
            except queue.Empty:
                continue

            if request is None:
                input_closed = True
                continue
            if not isinstance(request, DeviceFrameRequest):
                raise TypeError(f"Unexpected device frame request type: {type(request)}")

            pipeline = get_pipeline(request.slot.resolution)
            schedule_pipeline_request(request, pipeline)
            pending_slot_keys.add(request.slot.input_name)
    except Exception:
        error = f"Device-frame service {service_id} on CUDA device {device_id} failed:\n{traceback.format_exc()}"
        for result_queue in result_queues:
            try:
                result_queue.put_nowait(DeviceFrameResult(request_id=-1, error=error))
            except queue.Full:
                pass
        stop_event.set()
    finally:
        try:
            torch.cuda.synchronize(device_id)
        except Exception:
            pass
        for shared_buffers in shared_buffers_by_slot.values():
            shared_buffers.close()
        for shared_input in shared_inputs_by_memory_name.values():
            shared_input.close()
        pipeline_slots.clear()
        pipeline_slots_by_transport.clear()
        shared_inputs_device_by_memory_name.clear()
        shared_input_descriptors_by_memory_name.clear()
        pipelines.clear()
        torch.cuda.empty_cache()
