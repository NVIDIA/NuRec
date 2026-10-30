# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import queue
import threading

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from apps.av.aux_gen.device_frame import (
    DeviceFrameOutputSpec,
    DeviceFramePipeline,
    DeviceFrameProcessorMetadata,
    processor_directory,
)
from apps.av.aux_gen.device_frame_service import (
    DeviceFrameRequest,
    DeviceFrameResult,
    DeviceFrameSharedArray,
    DeviceFrameSharedArrayDescriptor,
    DeviceFrameSharedBuffers,
    DeviceFrameSharedMemorySlot,
    SharedDeviceFrameClient,
    _run_device_frame_processors,
    _schedule_cuda_pipeline_stages,
    run_device_frame_service_process,
)


MASK2FORMER_PROCESSOR = "test_primary"
SAM2_PROCESSOR = "test_secondary"
CAMERA_MASK_INPUT = "test_auxiliary_input"
SEMANTIC_LABELS_OUTPUT = "test_labels"
SEMANTIC_LOGITS_OUTPUT = "test_logits"
SAM2_MASK_OUTPUT = "test_mask"


def _output_specs(
    resolution: tuple[int, int],
    *,
    logits: bool = False,
    sam2: bool = False,
) -> tuple[DeviceFrameOutputSpec, ...]:
    """Build the test transport contract."""
    width, height = resolution
    specs = [DeviceFrameOutputSpec(SEMANTIC_LABELS_OUTPUT, (height, width), torch.uint8)]
    if logits:
        specs.append(DeviceFrameOutputSpec(SEMANTIC_LOGITS_OUTPUT, (height, width, 2), torch.float16))
    if sam2:
        specs.append(DeviceFrameOutputSpec(SAM2_MASK_OUTPUT, (height, width), torch.uint8))
    return tuple(specs)


def _device_pipeline_staging_nbytes(
    input_shape: tuple[int, ...],
    input_dtype: torch.dtype,
    output_specs: tuple[DeviceFrameOutputSpec, ...],
) -> tuple[int, int]:
    """Pinned-host and CUDA bytes for one staging slot, matching service layout."""

    def tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        return int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()

    _, _, height, width = input_shape
    rgb_nbytes = tensor_nbytes((height, width, 3), torch.uint8)
    output_nbytes = sum(tensor_nbytes(spec.shape, spec.dtype) for spec in output_specs)
    return (
        rgb_nbytes + output_nbytes,
        rgb_nbytes + tensor_nbytes(input_shape, input_dtype) + output_nbytes,
    )


@pytest.mark.parametrize("num_classes", [0, 20])
def test_full_resolution_staging_memory_is_explicitly_accounted(num_classes: int) -> None:
    width, height = 3840, 2160
    output_specs = [DeviceFrameOutputSpec(SEMANTIC_LABELS_OUTPUT, (height, width), torch.uint8)]
    if num_classes:
        output_specs.append(DeviceFrameOutputSpec(SEMANTIC_LOGITS_OUTPUT, (height, width, num_classes), torch.float16))

    host_bytes, device_bytes = _device_pipeline_staging_nbytes(
        (1, 3, height, width),
        torch.uint8,
        tuple(output_specs),
    )

    output_bytes_per_pixel = 1 + 2 * num_classes
    assert host_bytes == width * height * (3 + output_bytes_per_pixel)
    assert device_bytes == width * height * (3 + 3 + output_bytes_per_pixel)
    if num_classes:
        assert 300 * 1024**2 < host_bytes < device_bytes < 400 * 1024**2
    else:
        assert host_bytes < device_bytes < 64 * 1024**2


class _FakeEstimator:
    def __init__(self, resolution: list[int], *, estimate_logits: bool, enable_trt: bool) -> None:
        self.resolution = resolution
        self.estimate_logits = estimate_logits
        self.enable_trt = enable_trt
        self.name = MASK2FORMER_PROCESSOR

    def metadata(self) -> DeviceFrameProcessorMetadata:
        return DeviceFrameProcessorMetadata(
            storage={
                "resolution": self.resolution,
                "dataset_name": "test",
                "method": "mask2former",
                "pretrained_checkpoint": "test.pth",
                "stuff_classes": ["background", "foreground"],
                "stuff_colors": [[0, 0, 0], [255, 255, 255]],
            },
            backend={"inference_backend": "tensorrt" if self.enable_trt else "pytorch"},
        )

    def output_specs(self, resolution: tuple[int, int]) -> tuple[DeviceFrameOutputSpec, ...]:
        return _output_specs(resolution, logits=self.estimate_logits)

    def process_device(self, image_bgr_nchw, *, outputs, processor_inputs, stream) -> None:
        raise NotImplementedError


class _FakePipelineFactory:
    """Build test pipelines while reusing one optional secondary processor."""

    def __init__(
        self,
        estimator_factory,
        *,
        estimate_logits: bool = False,
        enable_trt: bool = False,
        sam2_factory=None,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.estimate_logits = estimate_logits
        self.enable_trt = enable_trt
        self.sam2_factory = sam2_factory
        self.sam2_processor = None

    def __call__(self, resolution: tuple[int, int]) -> DeviceFramePipeline:
        estimator = self.estimator_factory(
            list(resolution),
            estimate_logits=self.estimate_logits,
            enable_trt=self.enable_trt,
        )
        if self.sam2_factory is not None and self.sam2_processor is None:
            self.sam2_processor = self.sam2_factory()
        processors = processor_directory(
            estimator,
            *((self.sam2_processor,) if self.sam2_processor is not None else ()),
        )
        return DeviceFramePipeline(
            input_spec=estimator.get_device_input_spec(),
            processors=processors,
        )


def test_shared_memory_create_releases_each_successful_partial_allocation() -> None:
    input_shared_memory = MagicMock(name="input_shared_memory")
    with (
        patch(
            "apps.av.aux_gen.device_frame_service.SharedMemory",
            side_effect=[input_shared_memory, OSError("label allocation failed")],
        ),
        pytest.raises(OSError, match="label allocation failed"),
    ):
        DeviceFrameSharedBuffers.create((4, 3), _output_specs((4, 3)))

    input_shared_memory.close.assert_called_once_with()
    input_shared_memory.unlink.assert_called_once_with()


def test_shared_memory_attach_closes_each_successful_partial_attachment() -> None:
    input_shared_memory = MagicMock(name="input_shared_memory")
    slot = DeviceFrameSharedMemorySlot(
        input_name="input",
        resolution=(4, 3),
        outputs=(
            DeviceFrameSharedArrayDescriptor(
                name=SEMANTIC_LABELS_OUTPUT,
                shape=(3, 4),
                dtype=torch.uint8,
                shared_memory_name="labels",
            ),
        ),
    )
    with (
        patch(
            "apps.av.aux_gen.device_frame_service.SharedMemory",
            side_effect=[input_shared_memory, OSError("label attachment failed")],
        ),
        pytest.raises(OSError, match="label attachment failed"),
    ):
        DeviceFrameSharedBuffers.attach(slot)

    input_shared_memory.close.assert_called_once_with()
    input_shared_memory.unlink.assert_not_called()


def test_shared_device_frame_client_submits_and_resolves_only_active_outputs() -> None:
    buffers = DeviceFrameSharedBuffers.create((4, 3), _output_specs((4, 3), sam2=True))
    camera_mask = DeviceFrameSharedArray.create(
        CAMERA_MASK_INPUT,
        np.array([[True, False, True, False]] * 3, dtype=np.bool_),
    )
    slot_queue: queue.Queue = queue.Queue(maxsize=1)
    request_queue: queue.Queue = queue.Queue(maxsize=1)
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    slot_queue.put(buffers.slot)
    client = SharedDeviceFrameClient(
        worker_id=2,
        slot_queue=slot_queue,
        request_queue=request_queue,
        result_queue=result_queue,
        stop_event=stop_event,
    )
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)

    try:
        pending = client.submit(
            7,
            image,
            frozenset({MASK2FORMER_PROCESSOR}),
            (camera_mask.descriptor,),
        )
        assert pending is not None
        request = request_queue.get_nowait()
        assert request.worker_id == 2
        assert request.request_id == 7
        assert request.processor_inputs == (camera_mask.descriptor,)
        np.testing.assert_array_equal(buffers.input_buffer, image)

        buffers.output_buffers[SEMANTIC_LABELS_OUTPUT].fill(5)
        buffers.output_buffers[SAM2_MASK_OUTPUT].fill(9)
        result_queue.put(
            DeviceFrameResult(
                request_id=7,
                error=None,
                output_names=(SEMANTIC_LABELS_OUTPUT,),
            )
        )
        resolved = client.resolve(pending, include_input=True)

        assert set(resolved.outputs) == {SEMANTIC_LABELS_OUTPUT}
        np.testing.assert_array_equal(resolved.outputs[SEMANTIC_LABELS_OUTPUT], np.full((3, 4), 5, dtype=np.uint8))
        assert resolved.input_image is not None
        np.testing.assert_array_equal(resolved.input_image, image)
        assert slot_queue.get_nowait() == buffers.slot
    finally:
        client.close()
        camera_mask.close()
        buffers.close()


@pytest.mark.parametrize(
    ("results", "error_type", "expected_error"),
    [
        ([object()], TypeError, "Unexpected device-frame result type"),
        (
            [DeviceFrameResult(request_id=7, error="inference failed", output_names=())],
            RuntimeError,
            "inference failed",
        ),
        (
            [
                DeviceFrameResult(request_id=8, error=None, output_names=()),
                DeviceFrameResult(request_id=8, error=None, output_names=()),
            ],
            RuntimeError,
            "duplicate device-frame result 8",
        ),
    ],
)
def test_shared_device_frame_client_returns_slot_on_resolve_error(
    results: list[Any],
    error_type: type[Exception],
    expected_error: str,
) -> None:
    buffers = DeviceFrameSharedBuffers.create((4, 3), _output_specs((4, 3)))
    slot_queue: queue.Queue = queue.Queue(maxsize=1)
    request_queue: queue.Queue = queue.Queue(maxsize=1)
    result_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    slot_queue.put(buffers.slot)
    client = SharedDeviceFrameClient(
        worker_id=2,
        slot_queue=slot_queue,
        request_queue=request_queue,
        result_queue=result_queue,
        stop_event=stop_event,
    )

    try:
        pending = client.submit(7, np.zeros((3, 4, 3), dtype=np.uint8), frozenset({MASK2FORMER_PROCESSOR}))
        assert pending is not None
        for result in results:
            result_queue.put(result)

        with pytest.raises(error_type, match=expected_error):
            client.resolve(pending)

        assert slot_queue.get_nowait() == buffers.slot
        assert slot_queue.empty()
    finally:
        client.close()
        buffers.close()


def test_cuda_pipeline_stages_have_explicit_dependencies_without_synchronizing() -> None:
    operations = []

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            operations.append(f"record:{self.name}:{stream.name}")

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            operations.append(f"wait:{self.name}:{event.name}")

    h2d_stream = FakeStream("h2d")
    processing_stream = FakeStream("processing")
    d2h_stream = FakeStream("d2h")
    h2d_done = FakeEvent("h2d-done")
    processing_done = FakeEvent("processing-done")
    d2h_done = FakeEvent("d2h-done")

    _schedule_cuda_pipeline_stages(
        h2d_stream=h2d_stream,
        processing_stream=processing_stream,
        d2h_stream=d2h_stream,
        h2d_done=h2d_done,
        processing_done=processing_done,
        d2h_done=d2h_done,
        h2d_task=lambda stream: operations.append(f"task:{stream.name}"),
        processing_task=lambda stream: operations.append(f"task:{stream.name}"),
        d2h_task=lambda stream: operations.append(f"task:{stream.name}"),
    )

    assert operations == [
        "task:h2d",
        "record:h2d-done:h2d",
        "wait:processing:h2d-done",
        "task:processing",
        "record:processing-done:processing",
        "wait:d2h:processing-done",
        "task:d2h",
        "record:d2h-done:d2h",
    ]


def test_cuda_pipeline_does_not_schedule_d2h_after_processing_enqueue_error() -> None:
    operations = []

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            operations.append(("record", self.name, stream.name))

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            operations.append(("wait", self.name, event.name))

    def fail_processing(stream):
        operations.append(("task", stream.name))
        raise RuntimeError("enqueue boom")

    with pytest.raises(RuntimeError, match="enqueue boom"):
        _schedule_cuda_pipeline_stages(
            h2d_stream=FakeStream("h2d"),
            processing_stream=FakeStream("processing"),
            d2h_stream=FakeStream("d2h"),
            h2d_done=FakeEvent("h2d-done"),
            processing_done=FakeEvent("processing-done"),
            d2h_done=FakeEvent("d2h-done"),
            h2d_task=lambda stream: operations.append(("task", stream.name)),
            processing_task=fail_processing,
            d2h_task=lambda stream: operations.append(("task", stream.name)),
        )

    assert not any(operation[:2] == ("task", "d2h") for operation in operations)


def test_device_frame_processors_compose_in_order_with_declared_outputs() -> None:
    operations: list[str] = []

    class FakeProcessor:
        def __init__(self, name: str, output_name: str) -> None:
            self.name = name
            self._output_name = output_name

        def metadata(self) -> DeviceFrameProcessorMetadata:
            return DeviceFrameProcessorMetadata(storage={}, backend={})

        def output_specs(self, resolution: tuple[int, int]) -> tuple[DeviceFrameOutputSpec, ...]:
            del resolution
            return (DeviceFrameOutputSpec(self._output_name, (3, 4), torch.uint8),)

        def process_device(self, image_bgr_nchw, *, outputs, processor_inputs, stream) -> None:
            operations.append(self.name)
            assert outputs[self._output_name] is not None

    request = DeviceFrameRequest(
        worker_id=0,
        request_id=1,
        slot=DeviceFrameSharedMemorySlot(
            input_name="input",
            resolution=(4, 3),
            outputs=(),
        ),
        enabled_processors=frozenset({"first", "second"}),
    )
    output_names = _run_device_frame_processors(
        processor_directory(
            FakeProcessor("first", "first_output"),
            FakeProcessor("disabled", "disabled_output"),
            FakeProcessor("second", "second_output"),
        ),
        request,
        MagicMock(),
        {"first_output": MagicMock(), "second_output": MagicMock()},
        {},
        MagicMock(),
    )

    assert operations == ["first", "second"]
    assert output_names == ("first_output", "second_output")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pipeline requires a GPU")
def test_gpu_service_pipelines_mixed_resolutions_and_reuses_worker_slots() -> None:
    created_resolutions = []
    staging_pointers: list[tuple[int, int, int]] = []
    sam2_request_count = 0

    class FakeDeviceEstimator(_FakeEstimator):
        def get_device_input_spec(self):
            width, height = self.resolution
            return (1, 3, height, width), torch.uint8

        def process_device(self, image_bgr_nchw, *, outputs, processor_inputs, stream):
            semantic_seg_out = outputs[SEMANTIC_LABELS_OUTPUT]
            semantic_seg_logits_out = outputs.get(SEMANTIC_LOGITS_OUTPUT)
            ego_mask = processor_inputs.get(CAMERA_MASK_INPUT)
            assert image_bgr_nchw.device.type == "cuda"
            assert semantic_seg_logits_out is not None
            assert ego_mask is None
            staging_pointers.append(
                (
                    image_bgr_nchw.data_ptr(),
                    semantic_seg_out.data_ptr(),
                    semantic_seg_logits_out.data_ptr(),
                )
            )
            semantic_seg_out.fill_(self.resolution[0])
            semantic_seg_logits_out.fill_(self.resolution[0])

    class FakeDeviceSam2Estimator:
        name = SAM2_PROCESSOR

        def metadata(self) -> DeviceFrameProcessorMetadata:
            return DeviceFrameProcessorMetadata(
                storage={
                    "dataset_name": "test",
                    "method": "sam2",
                    "pretrained_checkpoint": "sam2.pt",
                },
                backend={},
            )

        def output_specs(self, resolution):
            width, height = resolution
            return (DeviceFrameOutputSpec(SAM2_MASK_OUTPUT, (height, width), torch.uint8),)

        def process_device(self, image_bgr_nchw, *, outputs, processor_inputs, stream):
            nonlocal sam2_request_count
            del processor_inputs, stream
            assert image_bgr_nchw.device.type == "cuda"
            assert image_bgr_nchw.dtype == torch.uint8
            assert image_bgr_nchw[:, 0].eq(3).all()
            assert image_bgr_nchw[:, 1].eq(2).all()
            assert image_bgr_nchw[:, 2].eq(1).all()
            sam2_request_count += 1
            outputs[SAM2_MASK_OUTPUT].fill_(7)

    def estimator_factory(resolution, **kwargs):
        created_resolutions.append(tuple(resolution))
        return FakeDeviceEstimator(resolution, **kwargs)

    request_queue: queue.Queue = queue.Queue(maxsize=2)
    result_queues: list[queue.Queue] = [queue.Queue(maxsize=1), queue.Queue(maxsize=1)]
    ready_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    shared_buffers = [
        DeviceFrameSharedBuffers.create((4, 3), _output_specs((4, 3), logits=True, sam2=True)),
        DeviceFrameSharedBuffers.create((2, 2), _output_specs((2, 2), logits=True, sam2=True)),
    ]
    service_thread = threading.Thread(
        target=run_device_frame_service_process,
        kwargs={
            "service_id": 0,
            "device_id": 0,
            "initial_resolution": (4, 3),
            "request_queue": request_queue,
            "result_queues": result_queues,
            "ready_queue": ready_queue,
            "stop_event": stop_event,
            "pipeline_factory": _FakePipelineFactory(
                estimator_factory,
                estimate_logits=True,
                enable_trt=True,
                sam2_factory=FakeDeviceSam2Estimator,
            ),
            "max_staging_slots": 1,
            "allowed_resolutions": frozenset({(4, 3), (2, 2)}),
        },
    )
    try:
        service_thread.start()
        ready = ready_queue.get(timeout=5.0)
        assert ready.device_id == 0
        assert ready.error is None

        first_buffers = shared_buffers[0]
        first_buffers.input_buffer[:] = np.array([1, 2, 3], dtype=np.uint8)
        request_queue.put(
            DeviceFrameRequest(
                worker_id=0,
                request_id=-1,
                slot=first_buffers.slot,
                enabled_processors=frozenset({SAM2_PROCESSOR}),
            )
        )
        assert result_queues[0].get(timeout=5.0).request_id == -1
        np.testing.assert_array_equal(
            first_buffers.output_buffers[SAM2_MASK_OUTPUT],
            np.full((3, 4), 7, dtype=np.uint8),
        )

        for worker_id, buffers in enumerate(shared_buffers):
            buffers.input_buffer.fill(worker_id)
            request_queue.put(
                DeviceFrameRequest(
                    worker_id=worker_id,
                    request_id=0,
                    slot=buffers.slot,
                    enabled_processors=frozenset({MASK2FORMER_PROCESSOR}),
                )
            )

        for worker_id, buffers in enumerate(shared_buffers):
            result = result_queues[worker_id].get(timeout=5.0)
            assert result.error is None
            width, height = buffers.slot.resolution
            np.testing.assert_array_equal(
                buffers.output_buffers[SEMANTIC_LABELS_OUTPUT],
                np.full((height, width), width, dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                buffers.output_buffers[SEMANTIC_LOGITS_OUTPUT],
                np.full((height, width, 2), width, dtype=np.float16),
            )
        request_queue.put(
            DeviceFrameRequest(
                worker_id=0,
                request_id=1,
                slot=first_buffers.slot,
                enabled_processors=frozenset({MASK2FORMER_PROCESSOR}),
            )
        )
        assert result_queues[0].get(timeout=5.0).request_id == 1
        np.testing.assert_array_equal(
            first_buffers.output_buffers[SAM2_MASK_OUTPUT],
            np.full((3, 4), 7, dtype=np.uint8),
        )
    finally:
        request_queue.put(None)
        service_thread.join(timeout=5.0)
        for buffers in shared_buffers:
            buffers.close()

    assert not service_thread.is_alive()
    assert not stop_event.is_set()
    assert created_resolutions == [(4, 3), (2, 2)]
    assert len(set(staging_pointers)) == 1
    assert sam2_request_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pipeline requires a GPU")
def test_gpu_service_reuses_global_slot_across_camera_workers() -> None:
    class FakeDeviceEstimator(_FakeEstimator):
        def get_device_input_spec(self):
            width, height = self.resolution
            return (1, 3, height, width), torch.uint8

        def process_device(self, image_bgr_nchw, *, outputs, processor_inputs, stream):
            semantic_seg_out = outputs[SEMANTIC_LABELS_OUTPUT]
            semantic_seg_logits_out = outputs.get(SEMANTIC_LOGITS_OUTPUT)
            ego_mask = processor_inputs.get(CAMERA_MASK_INPUT)
            assert image_bgr_nchw.device.type == "cuda"
            assert semantic_seg_logits_out is None
            assert ego_mask is not None
            assert ego_mask.device.type == "cuda"
            assert ego_mask.dtype == torch.bool
            semantic_seg_out.fill_(int(ego_mask[0, 0]))

    request_queue: queue.Queue = queue.Queue(maxsize=1)
    result_queues: list[queue.Queue] = [queue.Queue(maxsize=1), queue.Queue(maxsize=1)]
    ready_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    shared_buffers = DeviceFrameSharedBuffers.create((2, 2), _output_specs((2, 2)))
    camera_masks = [
        DeviceFrameSharedArray.create(CAMERA_MASK_INPUT, np.zeros((2, 2), dtype=np.bool_)),
        DeviceFrameSharedArray.create(CAMERA_MASK_INPUT, np.ones((2, 2), dtype=np.bool_)),
    ]
    service_thread = threading.Thread(
        target=run_device_frame_service_process,
        kwargs={
            "service_id": 0,
            "device_id": 0,
            "initial_resolution": (2, 2),
            "request_queue": request_queue,
            "result_queues": result_queues,
            "ready_queue": ready_queue,
            "stop_event": stop_event,
            "pipeline_factory": _FakePipelineFactory(
                FakeDeviceEstimator,
                enable_trt=True,
            ),
            "allowed_resolutions": frozenset({(2, 2)}),
        },
    )
    try:
        service_thread.start()
        assert ready_queue.get(timeout=5).error is None
        for worker_id in range(2):
            request_queue.put(
                DeviceFrameRequest(
                    worker_id=worker_id,
                    request_id=worker_id,
                    slot=shared_buffers.slot,
                    enabled_processors=frozenset({MASK2FORMER_PROCESSOR}),
                    processor_inputs=(camera_masks[worker_id].descriptor,),
                )
            )
            assert result_queues[worker_id].get(timeout=5).error is None
            np.testing.assert_array_equal(
                shared_buffers.output_buffers[SEMANTIC_LABELS_OUTPUT],
                np.full((2, 2), worker_id, dtype=np.uint8),
            )
    finally:
        request_queue.put(None)
        service_thread.join(timeout=30)
        for camera_mask in camera_masks:
            camera_mask.close()
        shared_buffers.close()

    assert not service_thread.is_alive()
    assert not stop_event.is_set()


@pytest.mark.parametrize(
    "service_device_ids",
    [(0, 0), (0, 1)],
    ids=["two-services-one-gpu", "one-service-per-gpu"],
)
def test_gpu_services_dynamically_share_one_request_queue(service_device_ids: tuple[int, int]) -> None:
    if torch.cuda.device_count() <= max(service_device_ids):
        pytest.skip(f"Routing to devices {service_device_ids} requires more visible GPUs")
    processing_barrier = threading.Barrier(2, timeout=30)

    class FakeDeviceEstimator(_FakeEstimator):
        def __init__(self, resolution: list[int], *, estimate_logits: bool, enable_trt: bool) -> None:
            super().__init__(resolution, estimate_logits=estimate_logits, enable_trt=enable_trt)
            self.device_id = torch.cuda.current_device()

        def get_device_input_spec(self):
            width, height = self.resolution
            return (1, 3, height, width), torch.uint8

        def process_device(self, image_bgr_nchw, *, outputs, processor_inputs, stream):
            semantic_seg_out = outputs[SEMANTIC_LABELS_OUTPUT]
            semantic_seg_logits_out = outputs.get(SEMANTIC_LOGITS_OUTPUT)
            ego_mask = processor_inputs.get(CAMERA_MASK_INPUT)
            assert image_bgr_nchw.device.index == self.device_id
            assert semantic_seg_logits_out is None
            assert ego_mask is None
            processing_barrier.wait()
            semantic_seg_out.fill_(self.device_id + 1)

    request_queue: queue.Queue = queue.Queue(maxsize=2)
    result_queues: list[queue.Queue] = [queue.Queue(maxsize=2)]
    ready_queue: queue.Queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    shared_buffers = [
        DeviceFrameSharedBuffers.create((2, 2), _output_specs((2, 2))),
        DeviceFrameSharedBuffers.create((3, 2), _output_specs((3, 2))),
    ]
    service_threads = [
        threading.Thread(
            target=run_device_frame_service_process,
            kwargs={
                "service_id": service_id,
                "device_id": device_id,
                "initial_resolution": (2, 2),
                "request_queue": request_queue,
                "result_queues": result_queues,
                "ready_queue": ready_queue,
                "stop_event": stop_event,
                "pipeline_factory": _FakePipelineFactory(
                    FakeDeviceEstimator,
                    enable_trt=True,
                ),
                "allowed_resolutions": frozenset({(2, 2), (3, 2)}),
            },
        )
        for service_id, device_id in enumerate(service_device_ids)
    ]
    try:
        for service_thread in service_threads:
            service_thread.start()
        ready_services = {
            (ready.service_id, ready.device_id) for ready in (ready_queue.get(timeout=30) for _ in service_threads)
        }
        assert ready_services == set(enumerate(service_device_ids))

        for request_id, buffers in enumerate(shared_buffers):
            buffers.input_buffer.fill(request_id)
            request_queue.put(
                DeviceFrameRequest(
                    worker_id=0,
                    request_id=request_id,
                    slot=buffers.slot,
                    enabled_processors=frozenset({MASK2FORMER_PROCESSOR}),
                )
            )
        completed_request_ids = {result_queues[0].get(timeout=30).request_id for _ in shared_buffers}
        assert completed_request_ids == {0, 1}

        assert {int(buffers.output_buffers[SEMANTIC_LABELS_OUTPUT][0, 0]) for buffers in shared_buffers} == {
            device_id + 1 for device_id in service_device_ids
        }
    finally:
        for _ in service_threads:
            request_queue.put(None)
        for service_thread in service_threads:
            service_thread.join(timeout=30)
        for buffers in shared_buffers:
            buffers.close()

    assert all(not service_thread.is_alive() for service_thread in service_threads)
    assert not stop_event.is_set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pipeline requires a GPU")
def test_gpu_service_propagates_inference_errors() -> None:
    class FailingEstimator(_FakeEstimator):
        def get_device_input_spec(self):
            width, height = self.resolution
            return (1, 3, height, width), torch.uint8

        def process_device(self, *args, **kwargs):
            raise RuntimeError("inference boom")

    request_queue: queue.Queue = queue.Queue(maxsize=1)
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    ready_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    shared_buffers = DeviceFrameSharedBuffers.create((2, 2), _output_specs((2, 2)))
    service_thread = threading.Thread(
        target=run_device_frame_service_process,
        kwargs={
            "service_id": 0,
            "device_id": 0,
            "initial_resolution": (2, 2),
            "request_queue": request_queue,
            "result_queues": [result_queue],
            "ready_queue": ready_queue,
            "stop_event": stop_event,
            "pipeline_factory": _FakePipelineFactory(FailingEstimator),
            "allowed_resolutions": frozenset({(2, 2)}),
        },
    )
    try:
        service_thread.start()
        ready = ready_queue.get(timeout=1.0)
        assert ready.error is None
        request_queue.put(
            DeviceFrameRequest(
                worker_id=0,
                request_id=0,
                slot=shared_buffers.slot,
                enabled_processors=frozenset({MASK2FORMER_PROCESSOR}),
            )
        )
        result = result_queue.get(timeout=5.0)
        assert result.error is not None
        assert "inference boom" in result.error
        service_thread.join(timeout=1.0)
        assert stop_event.is_set()
        assert not service_thread.is_alive()
    finally:
        shared_buffers.close()
