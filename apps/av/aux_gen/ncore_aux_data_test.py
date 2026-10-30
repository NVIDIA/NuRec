# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ncore_aux_data module."""

import itertools
import pickle
import queue
import subprocess
import sys
import threading

from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import mmseg.apis  # type: ignore[import-not-found]
import numpy as np
import pytest
import torch

from click.testing import CliRunner
from mmseg.trt.mask2former_inferencer import Mask2formerInferencer  # type: ignore[import-not-found]
from PIL import Image as PILImage
from python.runfiles import runfiles
from scipy.ndimage import binary_dilation, median_filter

from apps.av.aux_gen.device_frame import (
    DeviceFrameOutputSpec,
    DeviceFrameProcessor,
    DeviceFrameProcessorMetadata,
    process_device_frame_from_host,
)
from apps.av.aux_gen.device_frame_service import (
    DeviceFrameResult,
    DeviceFrameServiceReady,
    DeviceFrameSharedArray,
    DeviceFrameSharedBuffers,
)
from apps.av.aux_gen.estimators import (
    Mask2FormerSegmentationEstimator,
    PromptlessSAM2EgoMaskEstimator,
    find_network_file_path,
    get_gpu_info,
)
from apps.av.aux_gen.ncore_aux_data import (
    AsyncShardWriter,
    AuxDataWriter,
    DepthGenerationRequest,
    LoaderWorkerPayload,
    OpenedSequence,
    _merge_mask2former_outputs,
    _run_ego_mask_for_camera,
    _run_segmentation_camera_frames,
    _run_serial_mask2former_segmentation,
    _run_shared_segmentation,
    _segmentation_host_worker,
    _SegmentationHostWorkerConfig,
    _select_ego_mask_sample_indices,
    _stop_processes_if_alive,
    _wait_for_segmentation_services,
    build_ncore_aux_data_cli,
    cli,
)
from apps.av.aux_gen.segmentation_device_pipeline import (
    CAMERA_MASK_INPUT,
    MASK2FORMER_PROCESSOR,
    SAM2_MASK_OUTPUT,
    SAM2_PROCESSOR,
    SEMANTIC_LABELS_OUTPUT,
    SEMANTIC_LOGITS_OUTPUT,
    SEMANTIC_UNCERTAINTY_OUTPUT,
    SegmentationDeviceFramePipelineFactory,
)
from apps.av.aux_gen.segmentation_presentation import is_dark_image, save_mask2former_visualization
from apps.av.aux_gen.shared_segmentation import (
    AuxiliarySegmentationOutputs,
    SegmentationFrameOutput,
    SharedSegmentationFrameProcessor,
    save_auxiliary_mask,
)


RUNFILES = runfiles.Create()
SEQUENCE = "clipgt-9048443e-c482-4228-8326-5b3dff3be711"


def _processor_metadata(
    storage: dict[str, Any],
    backend: dict[str, Any] | None = None,
) -> DeviceFrameProcessorMetadata:
    return DeviceFrameProcessorMetadata(storage=storage, backend={} if backend is None else backend)


def test_imports():
    """Verify ncore_aux_data module and its external dependencies can be imported."""
    import apps.av.aux_gen.ncore_aux_data  # noqa: F401

    assert apps.av.aux_gen.ncore_aux_data.cli is not None


@pytest.mark.parametrize(
    "checkpoint",
    [
        "mask2former_dinov2_nv_private12k_backboneTTA_76000.pth",
        "sam2_hiera_l_avfinetuned_v1.pt",
    ],
)
def test_pretrained_checkpoint_resolves_from_http_file_runfiles(checkpoint: str) -> None:
    assert find_network_file_path(checkpoint).is_file()


def test_built_cli_forwards_configured_depth_generation() -> None:
    depth_generation = DepthGenerationRequest(
        create_estimator=MagicMock(),
        store_as_png=False,
        description="test depth estimation",
    )
    configure_depth_generation = MagicMock(return_value=depth_generation)
    command = build_ncore_aux_data_cli(configure_depth_generation)

    with patch("apps.av.aux_gen.ncore_aux_data.ncore_aux_data") as run_aux_data:
        result = CliRunner().invoke(
            command,
            [
                "--output-dir=/tmp/ncore_aux_data_test",
                "--segmentation-backend=none",
                "--no-ego-mask",
                "--no-lidar-seg-camvis",
            ],
        )

    assert result.exit_code == 0, result.output
    configure_depth_generation.assert_called_once()
    assert run_aux_data.call_args.args[2] is depth_generation
    assert run_aux_data.call_args.args[3] == {}


def test_cli_processes_v4_manifest(tmp_path: Path) -> None:
    manifest_path = Path(RUNFILES.Rlocation(f"test_data_ncore_v4/{SEQUENCE}.json"))
    result = CliRunner().invoke(
        cli,
        [
            f"--dataset-path={manifest_path}",
            "--camera-id=camera_front_wide_120fov",
            f"--output-dir={tmp_path}",
            "--segmentation-backend=none",
            "--no-lidar-seg-camvis",
            "--no-ego-mask",
            "--store-meta",
            "offset",
            "--sequence-duration-sec=0.4",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert any(tmp_path.iterdir())


def test_cli_requires_exactly_one_sequence_input(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            f"--output-dir={tmp_path}",
            "--segmentation-backend=none",
            "offset",
        ],
    )

    assert result.exit_code == 2
    assert "Exactly one sequence input must be provided" in result.output


def test_ncore_aux_data_import_does_not_load_estimators() -> None:
    """Keep spawned host workers independent of GPU estimator imports."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import apps.av.aux_gen.ncore_aux_data; assert 'apps.av.aux_gen.estimators' not in sys.modules"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    ("mode_args", "expected"),
    [
        ([], "serial"),
        (["--serial-mode"], "serial"),
        (["--no-parallel-mode"], "serial"),
        (["--parallel-mode"], "parallel"),
    ],
)
def test_execution_mode_cli(mode_args: list[str], expected: str) -> None:
    with patch("apps.av.aux_gen.ncore_aux_data.ncore_aux_data") as run_aux_data:
        result = CliRunner().invoke(
            cli,
            ["--output-dir=/tmp/ncore_aux_data_test", *mode_args, "offset"],
        )

    assert result.exit_code == 0, result.output
    assert run_aux_data.call_args.args[0].execution_mode == expected


def test_execution_modes_are_mutually_exclusive() -> None:
    result = CliRunner().invoke(
        cli,
        ["--output-dir=/tmp/ncore_aux_data_test", "--serial-mode", "--parallel-mode", "offset"],
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_workers_per_gpu_cli() -> None:
    with patch("apps.av.aux_gen.ncore_aux_data.ncore_aux_data") as run_aux_data:
        result = CliRunner().invoke(
            cli,
            [
                "--output-dir=/tmp/ncore_aux_data_test",
                "--parallel-mode",
                "--workers-per-gpu=3",
                "offset",
            ],
        )

    assert result.exit_code == 0, result.output
    assert run_aux_data.call_args.args[0].workers_per_gpu == 3


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            ["--parallel-mode", "--segmentation-backend=none"],
            "--parallel-mode requires --segmentation-backend=mask2former",
        ),
        (
            ["--serial-mode", "--workers-per-gpu=2"],
            "--workers-per-gpu requires --parallel-mode",
        ),
        (
            ["--parallel-mode", "--zarr-store-type=directory"],
            "--parallel-mode requires --zarr-store-type=itar",
        ),
    ],
)
def test_execution_mode_rejects_options_that_would_be_ignored(options: list[str], message: str) -> None:
    result = CliRunner().invoke(
        cli,
        ["--output-dir=/tmp/ncore_aux_data_test", *options, "offset"],
    )

    assert result.exit_code == 2
    assert message in result.output


def test_visualize_accepts_parallel_mode() -> None:
    with patch("apps.av.aux_gen.ncore_aux_data.ncore_aux_data") as run_aux_data:
        result = CliRunner().invoke(
            cli,
            ["--output-dir=/tmp/ncore_aux_data_test", "--parallel-mode", "--visualize", "offset"],
        )

    assert result.exit_code == 0, result.output
    params = run_aux_data.call_args.args[0]
    assert params.parallel_mode
    assert params.visualize


def test_run_ego_mask_for_camera():
    """Test per-camera ego mask logic: default all, or only listed cameras."""
    # --no-ego-mask: never run
    assert _run_ego_mask_for_camera(False, None, "cam1") is False
    assert _run_ego_mask_for_camera(False, ["cam1"], "cam1") is False

    # --ego-mask, no --ego-mask-camera-id: run for every camera
    assert _run_ego_mask_for_camera(True, None, "cam1") is True
    assert _run_ego_mask_for_camera(True, [], "cam1") is True
    assert _run_ego_mask_for_camera(True, (), "cam2") is True  # tuple from Click multiple=True

    # --ego-mask --ego-mask-camera-id cam1: run only for cam1
    assert _run_ego_mask_for_camera(True, ["cam1"], "cam1") is True
    assert _run_ego_mask_for_camera(True, ["cam1"], "cam2") is False
    assert _run_ego_mask_for_camera(True, ["cam1", "cam2"], "cam1") is True
    assert _run_ego_mask_for_camera(True, ["cam1", "cam2"], "cam3") is False


def test_select_ego_mask_sample_indices_preserves_buffered_sampling() -> None:
    timestamps = [index * 1_000_000 for index in range(10)]

    assert _select_ego_mask_sample_indices(timestamps, samples_per_second=0.25, camera_id="camera") == (2, 5)
    assert _select_ego_mask_sample_indices(timestamps[:2], samples_per_second=10.0, camera_id="camera") == (0, 1)
    with pytest.raises(ValueError, match="No frames available"):
        _select_ego_mask_sample_indices([], samples_per_second=1.0, camera_id="camera")


def test_is_dark_image_uses_stride_sample_mean() -> None:
    dark = np.full((32, 32, 3), 7, dtype=np.uint8)
    bright = dark.copy()
    bright[::16, ::16] = 8

    assert is_dark_image(dark)
    assert not is_dark_image(bright)


@pytest.mark.parametrize("include_auxiliary", [False, True])
def test_segmentation_pipeline_factory_is_picklable_and_reuses_processors(include_auxiliary: bool) -> None:
    mask2former_processors = []
    sam2_processor = MagicMock()
    sam2_processor.name = SAM2_PROCESSOR

    class FakeMask2Former:
        name = MASK2FORMER_PROCESSOR
        stuff_classes = ("road", "car")

        def __init__(self, resolution, *, estimate_logits, enable_trt, output_uncertainty_mask) -> None:
            self.resolution = resolution
            self.estimate_logits = estimate_logits
            self.enable_trt = enable_trt
            self.output_uncertainty_mask = output_uncertainty_mask
            mask2former_processors.append(self)

        def get_device_input_spec(self):
            width, height = self.resolution
            return (1, 3, height, width), torch.uint8

    factory = SegmentationDeviceFramePipelineFactory(
        estimate_logits=True,
        enable_trt=False,
        include_sam2=True,
    )
    pickle.dumps(factory)
    auxiliary_factory = MagicMock()
    auxiliary_factory.return_value.name = "extension"
    auxiliary_spec = DeviceFrameOutputSpec("extension_mask", (3, 4), torch.uint8)
    auxiliary_factory.output_specs.return_value = (auxiliary_spec,)
    if include_auxiliary:
        factory.auxiliary_processor = auxiliary_factory
    with (
        patch("apps.av.aux_gen.estimators.Mask2FormerSegmentationEstimator", FakeMask2Former),
        patch(
            "apps.av.aux_gen.estimators.PromptlessSAM2EgoMaskEstimator",
            return_value=sam2_processor,
        ) as create_sam2,
    ):
        first_pipeline = factory((4, 3))
        second_pipeline = factory((8, 6))

    assert [processor.resolution for processor in mask2former_processors] == [[4, 3], [8, 6]]
    assert all(not processor.output_uncertainty_mask for processor in mask2former_processors)
    assert first_pipeline.processors == {
        mask2former_processors[0].name: mask2former_processors[0],
        sam2_processor.name: sam2_processor,
        **({"extension": auxiliary_factory.return_value} if include_auxiliary else {}),
    }
    assert second_pipeline.processors == {
        mask2former_processors[1].name: mask2former_processors[1],
        sam2_processor.name: sam2_processor,
        **({"extension": auxiliary_factory.return_value} if include_auxiliary else {}),
    }
    output_specs = factory.output_specs((4, 3), num_classes=2)
    assert [spec.name for spec in output_specs] == [
        SEMANTIC_LABELS_OUTPUT,
        SEMANTIC_LOGITS_OUTPUT,
        SAM2_MASK_OUTPUT,
        *(["extension_mask"] if include_auxiliary else []),
    ]
    if include_auxiliary:
        auxiliary_factory.assert_called_once_with(semantic_class_names=("road", "car"))
        auxiliary_factory.output_specs.assert_called_once_with((4, 3))
        assert output_specs[-1] == auxiliary_spec
    else:
        auxiliary_factory.assert_not_called()
    create_sam2.assert_called_once_with()


def test_wait_for_segmentation_services_accepts_consistent_metadata() -> None:
    semantic_metadata = {"stuff_classes": ["road"]}
    sam2_metadata = {"method": "sam2"}
    ready_queue: queue.Queue = queue.Queue()
    ready_queue.put(
        DeviceFrameServiceReady(
            service_id=1,
            device_id=0,
            processor_metadata={
                MASK2FORMER_PROCESSOR: _processor_metadata(semantic_metadata),
                SAM2_PROCESSOR: _processor_metadata(sam2_metadata),
            },
            error=None,
        )
    )
    ready_queue.put(
        DeviceFrameServiceReady(
            service_id=0,
            device_id=0,
            processor_metadata={
                MASK2FORMER_PROCESSOR: _processor_metadata(semantic_metadata),
                SAM2_PROCESSOR: _processor_metadata(sam2_metadata),
            },
            error=None,
        )
    )

    ready = _wait_for_segmentation_services(
        {0: MagicMock(), 1: MagicMock()},
        ready_queue,
        sam2_enabled=True,
    )

    assert ready.device_id == 0
    assert ready.processor_metadata == {
        MASK2FORMER_PROCESSOR: _processor_metadata(semantic_metadata),
        SAM2_PROCESSOR: _processor_metadata(sam2_metadata),
    }


@pytest.mark.parametrize(
    ("second_metadata", "second_sam2_metadata", "expected_error"),
    [
        ({"stuff_classes": ["vehicle"]}, {"method": "sam2"}, "inconsistent processor metadata"),
        ({"stuff_classes": ["road"]}, {"method": "other"}, "inconsistent processor metadata"),
    ],
)
def test_wait_for_segmentation_services_rejects_inconsistent_metadata(
    second_metadata: dict,
    second_sam2_metadata: dict,
    expected_error: str,
) -> None:
    ready_queue: queue.Queue = queue.Queue()
    ready_queue.put(
        DeviceFrameServiceReady(
            service_id=0,
            device_id=0,
            processor_metadata={
                MASK2FORMER_PROCESSOR: _processor_metadata({"stuff_classes": ["road"]}),
                SAM2_PROCESSOR: _processor_metadata({"method": "sam2"}),
            },
            error=None,
        )
    )
    ready_queue.put(
        DeviceFrameServiceReady(
            service_id=1,
            device_id=1,
            processor_metadata={
                MASK2FORMER_PROCESSOR: _processor_metadata(second_metadata),
                SAM2_PROCESSOR: _processor_metadata(second_sam2_metadata),
            },
            error=None,
        )
    )

    with pytest.raises(RuntimeError, match=expected_error):
        _wait_for_segmentation_services(
            {0: MagicMock(), 1: MagicMock()},
            ready_queue,
            sam2_enabled=True,
        )


def test_wait_for_segmentation_services_rejects_mixed_inference_backends() -> None:
    ready_queue: queue.Queue = queue.Queue()
    metadata_cases: tuple[dict[str, Any], ...] = (
        {
            "stuff_classes": ["road"],
            "inference_backend": "tensorrt",
            "inference_engine": "mask2former-a40.engine",
            "device_input_dtype": "torch.float16",
        },
        {
            "stuff_classes": ["road"],
            "inference_backend": "pytorch",
            "inference_engine": None,
            "device_input_dtype": "torch.uint8",
        },
    )
    for service_id, metadata in enumerate(metadata_cases):
        ready_queue.put(
            DeviceFrameServiceReady(
                service_id=service_id,
                device_id=service_id,
                processor_metadata={
                    MASK2FORMER_PROCESSOR: _processor_metadata(
                        {"stuff_classes": metadata["stuff_classes"]},
                        {
                            "inference_backend": metadata["inference_backend"],
                            "inference_engine": metadata["inference_engine"],
                            "device_input_dtype": metadata["device_input_dtype"],
                        },
                    )
                },
                error=None,
            )
        )

    with pytest.raises(RuntimeError, match="mixed inference backends are not supported"):
        _wait_for_segmentation_services(
            {0: MagicMock(), 1: MagicMock()},
            ready_queue,
            sam2_enabled=False,
        )


def test_wait_for_segmentation_services_reports_startup_failures() -> None:
    ready_queue: queue.Queue = queue.Queue()
    ready_queue.put(
        DeviceFrameServiceReady(
            service_id=0,
            device_id=0,
            processor_metadata=None,
            error="model load failed",
        )
    )

    with pytest.raises(RuntimeError, match="model load failed"):
        _wait_for_segmentation_services(
            {0: MagicMock()},
            ready_queue,
            sam2_enabled=False,
        )


def test_wait_for_segmentation_services_reports_unready_process_exit() -> None:
    process = MagicMock(exitcode=7)
    process.is_alive.return_value = False

    with pytest.raises(RuntimeError, match=r"service 0 \(exit code 7\)"):
        _wait_for_segmentation_services(
            {0: process},
            queue.Queue(),
            sam2_enabled=False,
        )


def test_wait_for_segmentation_services_reports_startup_timeout() -> None:
    process = MagicMock()
    process.is_alive.return_value = True

    with pytest.raises(RuntimeError, match=r"within 0s: \[0\]"):
        _wait_for_segmentation_services(
            {0: process},
            queue.Queue(),
            sam2_enabled=False,
            timeout_s=0,
        )


def test_segmentation_service_start_failure_cleans_up_started_services() -> None:
    processes = []

    class FakeProcess:
        def __init__(self, *, name, target, kwargs) -> None:
            self.name = name
            self.target = target
            self.kwargs = kwargs
            self.exitcode: int | None = None
            self.started = False
            self.joined = False

        def start(self) -> None:
            if self.name.startswith("segmentation-service-1"):
                raise OSError("process start failed")
            self.started = True

        def is_alive(self) -> bool:
            return self.started and not self.joined

        def join(self, timeout=None) -> None:
            del timeout
            self.joined = True
            self.exitcode = 0

    class FakeContext:
        def __init__(self) -> None:
            self.stop_event = threading.Event()

        def Queue(self, maxsize=0):
            return queue.Queue(maxsize=maxsize)

        def Event(self):
            return self.stop_event

        def Process(self, *, name, target, kwargs):
            process = FakeProcess(name=name, target=target, kwargs=kwargs)
            processes.append(process)
            return process

    context = FakeContext()
    camera_sensor = MagicMock()
    camera_sensor.get_frames_timestamps_us.return_value = np.array([100, 200])
    camera_sensor.model_parameters.resolution = np.array([4, 3])
    sequence_loader = MagicMock()
    sequence_loader.get_camera_sensor.return_value = camera_sensor
    opened_sequence = OpenedSequence(
        backend=MagicMock(),
        loader=sequence_loader,
        worker_payload=MagicMock(spec=LoaderWorkerPayload),
    )
    sensor_ranges = MagicMock(camera_frame_ranges={"camera": range(2)})
    params = MagicMock(
        workers_per_gpu=2,
        ego_mask=False,
        ego_mask_camera_ids=None,
        ego_mask_samples_per_second=0.2,
        seg_logits=False,
        enable_trt=True,
        poses_component_group="poses",
        intrinsics_component_group="intrinsics",
        masks_component_group="masks",
        cuboids_component_group="cuboids",
        zarr_store_type="itar",
        open_consolidated=True,
        ego_mask_aggregation_method="majority",
        parallel_mode=True,
    )

    with (
        patch("apps.av.aux_gen.ncore_aux_data.torch.cuda.device_count", return_value=1),
        patch("apps.av.aux_gen.ncore_aux_data.multiprocessing.get_context", return_value=context),
        patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None),
        pytest.raises(OSError, match="process start failed"),
    ):
        _run_shared_segmentation(sensor_ranges, opened_sequence, params, Path("/output"))

    assert processes[0].started
    assert processes[0].joined
    assert not processes[1].started
    assert context.stop_event.is_set()


@pytest.mark.parametrize("startup_fails", [False, True])
@pytest.mark.parametrize("mask2former_enabled", [False, True])
def test_shared_segmentation_extension_lifecycle_with_empty_camera(
    startup_fails: bool, mask2former_enabled: bool
) -> None:
    """Forward extension state and snapshot statistics before shutting down their owner."""
    context = MagicMock()
    context.Process.return_value.start.side_effect = OSError("stop after sampling") if startup_fails else None
    context.Process.return_value.exitcode = 0
    context.Queue.return_value.get_nowait.side_effect = queue.Empty
    context.Event.return_value.is_set.return_value = False
    context.Process.return_value.is_alive.return_value = False
    statistics = [{"camera_id": "active", "pixels": 3}]
    expected_statistics = tuple(statistics)
    context.Manager.return_value.list.return_value = statistics
    lifecycle = MagicMock()
    lifecycle.attach_mock(context.Process.return_value.join, "join")
    lifecycle.attach_mock(context.Manager.return_value.shutdown, "shutdown")
    context.Manager.return_value.shutdown.side_effect = statistics.clear

    camera_sensors = {}
    for camera_id, timestamps in {"empty": [], "active": [100, 200]}.items():
        camera_sensor = MagicMock()
        camera_sensor.get_frames_timestamps_us.return_value = np.asarray(timestamps)
        camera_sensor.model_parameters.resolution = np.array([4, 3])
        camera_sensors[camera_id] = camera_sensor
    sequence_loader = MagicMock()
    sequence_loader.get_camera_sensor.side_effect = camera_sensors.__getitem__
    opened_sequence = OpenedSequence(
        backend=MagicMock(),
        loader=sequence_loader,
        worker_payload=MagicMock(spec=LoaderWorkerPayload),
    )
    sensor_ranges = MagicMock(camera_frame_ranges={"empty": range(0), "active": range(2)})
    params = MagicMock(
        workers_per_gpu=1,
        ego_mask=True,
        ego_mask_camera_ids=None,
        ego_mask_samples_per_second=1.0,
        seg_logits=False,
        enable_trt=True,
        dataset_path="dataset.json",
        segmentation_backend="mask2former" if mask2former_enabled else "none",
    )
    auxiliary_segmentation = MagicMock()
    auxiliary_segmentation.frame_outputs.return_value = AuxiliarySegmentationOutputs(
        processor_name="extension",
        mask_output="extension_mask",
        mask_root=Path("/output/extension"),
    )
    auxiliary_segmentation.select_camera_frames.return_value = {1: (0,)}
    lifecycle.attach_mock(auxiliary_segmentation.finalize, "finalize")
    extension_metadata = {"model": "extension-v1"}
    ready = DeviceFrameServiceReady(
        service_id=0,
        device_id=0,
        processor_metadata={
            **(
                {MASK2FORMER_PROCESSOR: _processor_metadata({"stuff_classes": ["road"]})} if mask2former_enabled else {}
            ),
            SAM2_PROCESSOR: _processor_metadata({"model": "sam2"}),
            "extension": _processor_metadata(extension_metadata),
        },
        error=None,
    )

    with (
        patch("apps.av.aux_gen.ncore_aux_data.torch.cuda.device_count", return_value=1),
        patch("apps.av.aux_gen.ncore_aux_data.multiprocessing.get_context", return_value=context),
        patch("apps.av.aux_gen.ncore_aux_data.SegmentationDeviceFramePipelineFactory") as create_pipeline,
        patch("apps.av.aux_gen.ncore_aux_data._wait_for_segmentation_services", return_value=ready),
        patch("apps.av.aux_gen.ncore_aux_data.DeviceFrameSharedBuffers.create"),
        patch("apps.av.aux_gen.ncore_aux_data.sequence_loader_shard_info", return_value=("clip", None, None)),
        patch("apps.av.aux_gen.ncore_aux_data._merge_mask2former_outputs"),
        patch("apps.av.aux_gen.ncore_aux_data.time.perf_counter", side_effect=[10.0, 12.0]),
        patch(
            "apps.av.aux_gen.ncore_aux_data._select_ego_mask_sample_indices",
            wraps=_select_ego_mask_sample_indices,
        ) as select_ego_mask_sample_indices,
        pytest.raises(OSError, match="stop after sampling") if startup_fails else nullcontext(),
    ):
        _run_shared_segmentation(
            sensor_ranges,
            opened_sequence,
            params,
            Path("/output"),
            auxiliary_segmentation,
        )

    auxiliary_segmentation.select_camera_frames.assert_called_once_with(
        sequence_loader,
        [("empty", range(0)), ("active", range(2))],
    )
    select_ego_mask_sample_indices.assert_called_once_with(
        [100, 200],
        samples_per_second=1.0,
        camera_id="active",
    )
    context.Manager.return_value.shutdown.assert_called_once_with()
    create_pipeline.assert_called_once_with(
        estimate_logits=False,
        enable_trt=True,
        include_sam2=True,
        include_mask2former=mask2former_enabled,
        output_uncertainty_mask=params.visualize,
        auxiliary_processor=auxiliary_segmentation.processor_factory.return_value,
    )
    if startup_fails:
        auxiliary_segmentation.finalize.assert_not_called()
    else:
        worker_configs = [entry.kwargs["args"][0] for entry in context.Process.call_args_list if "args" in entry.kwargs]
        assert [(config.camera_id, config.auxiliary_sample_indices) for config in worker_configs] == [
            ("empty", ()),
            ("active", (0,)),
        ]
        assert all(
            config.auxiliary_outputs is auxiliary_segmentation.frame_outputs.return_value for config in worker_configs
        )
        assert all(config.auxiliary_camera_stats is statistics for config in worker_configs)
        auxiliary_segmentation.finalize.assert_called_once_with(
            camera_ids=("empty", "active"),
            camera_statistics=expected_statistics,
            processor_metadata=extension_metadata,
            wall_time_seconds=2.0,
        )
        assert [entry[0] for entry in lifecycle.mock_calls] == ["join", "join", "join", "shutdown", "finalize"]


def test_stop_processes_if_alive_escalates_unresponsive_process() -> None:
    class FakeProcess:
        name = "stuck-service"

        def __init__(self) -> None:
            self.alive = True
            self.join_timeouts: list[float] = []
            self.terminated = False
            self.killed = False

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            self.join_timeouts.append(timeout)

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True
            self.alive = False

    process = FakeProcess()

    _stop_processes_if_alive([process], timeout_s=0.25)

    assert process.join_timeouts == [0.25, 0.25, 0.25]
    assert process.terminated
    assert process.killed


def test_sam2_device_post_processing_matches_scipy() -> None:
    mask = np.random.default_rng(0).integers(0, 2, size=(32, 40), dtype=np.uint8)
    expected = binary_dilation(
        median_filter(mask, size=15),
        structure=np.ones((11, 11)),
    ).astype(np.uint8)

    actual = PromptlessSAM2EgoMaskEstimator._post_process_device_mask(torch.from_numpy(mask))

    np.testing.assert_array_equal(actual.numpy(), expected)


def test_process_device_frame_from_host_uses_generic_processor_contract() -> None:
    class FakeProcessor(DeviceFrameProcessor):
        name = "fake"

        def metadata(self) -> DeviceFrameProcessorMetadata:
            return DeviceFrameProcessorMetadata(storage={}, backend={})

        def output_specs(self, resolution: tuple[int, int]) -> tuple[DeviceFrameOutputSpec, ...]:
            width, height = resolution
            return (DeviceFrameOutputSpec("blue", (height, width), torch.uint8),)

        def process_device(
            self,
            image_bgr_nchw: torch.Tensor,
            *,
            outputs,
            processor_inputs,
            stream,
        ) -> None:
            assert stream is fake_stream
            torch.testing.assert_close(processor_inputs["mask"], torch.ones((2, 3), dtype=torch.bool))
            outputs["blue"].copy_(image_bgr_nchw[0, 0])

    image_rgb = np.empty((2, 3, 3), dtype=np.uint8)
    image_rgb[..., 0] = 30
    image_rgb[..., 1] = 20
    image_rgb[..., 2] = 10
    fake_stream = MagicMock()

    outputs = process_device_frame_from_host(
        FakeProcessor(),
        image_rgb,
        input_spec=((1, 3, 2, 3), torch.uint8),
        device=torch.device("cpu"),
        processor_inputs={"mask": np.ones((2, 3), dtype=np.bool_)},
        stream=fake_stream,
    )

    fake_stream.synchronize.assert_called_once_with()
    np.testing.assert_array_equal(outputs["blue"], np.full((2, 3), 10, dtype=np.uint8))


def test_sam2_estimator_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        PromptlessSAM2EgoMaskEstimator(device="cpu")


def test_get_gpu_info_uses_current_cuda_device() -> None:
    with (
        patch("apps.av.aux_gen.estimators.torch.cuda.is_available", return_value=True),
        patch("apps.av.aux_gen.estimators.torch.cuda.current_device", return_value=3),
        patch("apps.av.aux_gen.estimators.torch.cuda.get_device_capability", return_value=(8, 9)) as get_capability,
        patch("apps.av.aux_gen.estimators.torch.cuda.get_device_name", return_value="GPU 3") as get_name,
    ):
        assert get_gpu_info() == ("GPU 3", "8.9")

    get_capability.assert_called_once_with(3)
    get_name.assert_called_once_with(3)


def test_sam2_device_inference_rejects_stream_on_another_device() -> None:
    estimator = PromptlessSAM2EgoMaskEstimator.__new__(PromptlessSAM2EgoMaskEstimator)
    image = MagicMock(
        ndim=4,
        shape=(1, 3, 16, 16),
        device=torch.device("cuda:1"),
        dtype=torch.uint8,
    )
    output = MagicMock(
        shape=(16, 16),
        device=torch.device("cuda:1"),
        dtype=torch.uint8,
    )
    stream = MagicMock(device=torch.device("cuda:0"))

    with pytest.raises(ValueError, match="SAM2 stream is on cuda:0"):
        estimator.process_device(
            image,
            outputs={SAM2_MASK_OUTPUT: output},
            processor_inputs={},
            stream=stream,
        )


def test_mask2former_visualization_preserves_uncertainty_mask(tmp_path: Path) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    labels = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    palette = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)
    uncertainty = np.array([[False, True, False], [False, False, True]])
    vis_path = tmp_path / "frame"

    save_mask2former_visualization(image, labels, palette, vis_path, uncertainty)

    assert vis_path.with_suffix(".png").is_file()
    uncertainty_path = tmp_path / "frame_uncertainty_mask.png"
    np.testing.assert_array_equal(
        np.asarray(PILImage.open(uncertainty_path)),
        uncertainty.astype(np.uint8) * 255,
    )


def test_mask2former_infer_device_delegates_to_mmseg() -> None:
    estimator = Mask2FormerSegmentationEstimator.__new__(Mask2FormerSegmentationEstimator)
    predictor = MagicMock()
    device_prediction = MagicMock(backbone_feats=None)
    logits = torch.zeros((2, 2, 3))
    labels = torch.zeros((1, 2, 3), dtype=torch.uint8)
    device_prediction.seg_logits.data = logits
    device_prediction.pred_sem_seg.data = labels
    estimator.predictor = predictor
    estimator.resolution = [3, 2]
    estimator.stuff_classes = ["building", "road"]
    estimator.estimate_logits = False
    image = torch.zeros((1, 3, 2, 3), dtype=torch.uint8)
    stream = MagicMock()
    uncertainty = MagicMock()

    with (
        patch(
            "apps.av.aux_gen.estimators.mmseg.apis.inference_model_device",
            return_value=device_prediction,
        ) as inference_model_device,
        patch.object(estimator, "_post_process_device_tensors", return_value=(labels[0], logits, uncertainty)),
        patch("apps.av.aux_gen.estimators.torch.cuda.stream"),
    ):
        actual_labels, actual_logits, actual_backbone, actual_uncertainty = estimator.infer_device(image, None, stream)

    inference_model_device.assert_called_once_with(predictor, image, stream)
    assert actual_logits is logits
    torch.testing.assert_close(actual_labels, labels[0])
    assert actual_backbone is None
    assert actual_uncertainty is uncertainty


def test_mask2former_post_processing_preserves_device_and_handles_softmax_underflow() -> None:
    estimator = Mask2FormerSegmentationEstimator.__new__(Mask2FormerSegmentationEstimator)
    estimator.stuff_classes = ["building", "road"]
    estimator.estimate_logits = False
    logits = torch.tensor(
        [
            [[1000.0, 0.0]],
            [[-1000.0, 0.0]],
        ]
    )
    labels = torch.ones((1, 1, 2), dtype=torch.int64)

    actual_labels, actual_logits, uncertainty = estimator._post_process_device_tensors(logits, labels)

    assert actual_labels.device == logits.device
    assert actual_logits.device == logits.device
    torch.testing.assert_close(uncertainty, torch.tensor([[False, True]]))
    torch.testing.assert_close(actual_labels, torch.tensor([[1, 0]], dtype=torch.uint8))


def test_mask2former_processor_returns_uncertainty_when_requested() -> None:
    estimator = Mask2FormerSegmentationEstimator.__new__(Mask2FormerSegmentationEstimator)
    estimator.estimate_logits = False
    estimator.output_uncertainty_mask = True
    labels = torch.zeros((2, 3), dtype=torch.uint8)
    logits = torch.zeros((2, 2, 3))
    uncertainty = torch.tensor(
        [[False, True, False], [True, False, True]],
        dtype=torch.bool,
    )
    outputs = {
        SEMANTIC_LABELS_OUTPUT: torch.empty((2, 3), dtype=torch.uint8),
        SEMANTIC_UNCERTAINTY_OUTPUT: torch.empty((2, 3), dtype=torch.uint8),
    }

    with (
        patch.object(estimator, "infer_device", return_value=(labels, logits, None, uncertainty)),
        patch("apps.av.aux_gen.estimators.torch.cuda.stream"),
    ):
        estimator.process_device(
            torch.empty((1, 3, 2, 3)),
            outputs=outputs,
            processor_inputs={},
            stream=MagicMock(),
        )

    torch.testing.assert_close(outputs[SEMANTIC_LABELS_OUTPUT], labels)
    torch.testing.assert_close(outputs[SEMANTIC_UNCERTAINTY_OUTPUT], uncertainty.to(dtype=torch.uint8))


def test_host_inference_uses_common_device_path() -> None:
    estimator = Mask2FormerSegmentationEstimator.__new__(Mask2FormerSegmentationEstimator)
    image = PILImage.fromarray(np.zeros((2, 3, 3), dtype=np.uint8), mode="RGB")
    estimator.predictor = MagicMock()
    estimator.resolution = [3, 2]
    estimator.estimate_logits = False
    estimator.estimate_dinov2_feats = False
    estimator.output_uncertainty_mask = False
    estimator.stuff_classes = ["building", "road"]
    labels = torch.zeros((2, 3), dtype=torch.uint8)
    logits = torch.zeros((2, 2, 3))
    uncertainty = torch.zeros((2, 3), dtype=torch.bool)

    with (
        patch(
            "apps.av.aux_gen.estimators.mmseg.apis.get_model_device_input_spec",
            return_value=((1, 3, 2, 3), torch.uint8, torch.device("cpu")),
        ),
        patch.object(estimator, "infer_device", return_value=(labels, logits, None, uncertainty)) as device_inference,
        patch("apps.av.aux_gen.estimators.torch.cuda.current_stream"),
        patch("apps.av.aux_gen.estimators.torch.cuda.stream"),
        patch("apps.av.aux_gen.estimators.mmseg.apis.inference_model") as legacy_inference,
    ):
        result = estimator.predict(image)

    device_inference.assert_called_once()
    legacy_inference.assert_not_called()
    np.testing.assert_array_equal(np.asarray(result.semantic_seg), np.zeros((2, 3), dtype=np.uint8))


def test_host_inference_visualization_uses_borrowed_numpy_frame(tmp_path: Path) -> None:
    estimator = Mask2FormerSegmentationEstimator.__new__(Mask2FormerSegmentationEstimator)
    image_rgb = np.empty((2, 3, 3), dtype=np.uint8)
    image_rgb[..., 0] = 30
    image_rgb[..., 1] = 20
    image_rgb[..., 2] = 10
    image = PILImage.fromarray(image_rgb, mode="RGB")
    estimator.predictor = MagicMock()
    estimator.resolution = [3, 2]
    estimator.estimate_logits = False
    estimator.estimate_dinov2_feats = False
    estimator.output_uncertainty_mask = True
    estimator.stuff_classes = ["building", "road"]
    estimator.stuff_colors = [[0, 0, 0], [255, 0, 0]]
    labels = np.ones((2, 3), dtype=np.uint8)
    uncertainty = np.zeros((2, 3), dtype=np.uint8)

    with (
        patch(
            "apps.av.aux_gen.estimators.mmseg.apis.get_model_device_input_spec",
            return_value=((1, 3, 2, 3), torch.uint8, torch.device("cpu")),
        ),
        patch(
            "apps.av.aux_gen.estimators.process_device_frame_from_host",
            return_value={
                SEMANTIC_LABELS_OUTPUT: labels,
                SEMANTIC_UNCERTAINTY_OUTPUT: uncertainty,
            },
        ) as process_frame,
        patch("apps.av.aux_gen.estimators._save_mask2former_visualization") as save_visualization,
    ):
        result = estimator.predict(image, vis_path=tmp_path / "frame")

    process_frame.assert_called_once()
    saved_image = save_visualization.call_args.args[0]
    assert isinstance(saved_image, np.ndarray)
    np.testing.assert_array_equal(saved_image, image_rgb)
    assert save_visualization.call_args.args[4] is uncertainty
    np.testing.assert_array_equal(np.asarray(result.semantic_seg), labels)


def test_host_inference_feature_path_converts_numpy_frame_to_tensor() -> None:
    height, width, num_classes = 2, 3, 3
    estimator = Mask2FormerSegmentationEstimator.__new__(Mask2FormerSegmentationEstimator)
    estimator.predictor = MagicMock()
    estimator.resolution = [width, height]
    estimator.stuff_classes = ["building", "egocar", "road"]
    estimator.estimate_logits = False
    estimator.estimate_dinov2_feats = True
    estimator.output_uncertainty_mask = False
    image_rgb = np.empty((height, width, 3), dtype=np.uint8)
    image_rgb[..., 0] = 30
    image_rgb[..., 1] = 20
    image_rgb[..., 2] = 10
    fake_stream = MagicMock()
    captured_inputs = []
    labels = torch.full((height, width), 2, dtype=torch.uint8)
    logits = torch.zeros((num_classes, height, width), dtype=torch.float32)
    backbone_feats = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    uncertainty = torch.zeros((height, width), dtype=torch.bool)

    def infer_device(image_bgr_nchw, ego_mask, stream):
        captured_inputs.append(image_bgr_nchw.clone())
        assert ego_mask is None
        assert stream is fake_stream
        return labels, logits, backbone_feats, uncertainty

    with (
        patch(
            "apps.av.aux_gen.estimators.mmseg.apis.get_model_device_input_spec",
            return_value=((1, 3, height, width), torch.float32, torch.device("cpu")),
        ),
        patch("apps.av.aux_gen.estimators.torch.cuda.current_stream", return_value=fake_stream),
        patch.object(estimator, "infer_device", side_effect=infer_device) as device_predict,
    ):
        result = estimator.predict(PILImage.fromarray(image_rgb, mode="RGB"))

    device_predict.assert_called_once()
    assert captured_inputs[0].shape == (1, 3, height, width)
    assert captured_inputs[0].dtype == torch.float32
    torch.testing.assert_close(captured_inputs[0][0, 0], torch.full((height, width), 10, dtype=torch.float32))
    torch.testing.assert_close(captured_inputs[0][0, 2], torch.full((height, width), 30, dtype=torch.float32))
    np.testing.assert_array_equal(np.asarray(result.semantic_seg), np.full((height, width), 2, dtype=np.uint8))
    assert result.semantic_dinov2_feats is not None
    np.testing.assert_array_equal(result.semantic_dinov2_feats, backbone_feats.permute(1, 2, 0).numpy())
    assert result.semantic_seg_logits is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-resident SAM2 path requires a GPU")
def test_sam2_device_inference_reuses_bgr_cuda_image() -> None:
    captured_inputs = []

    class FakeTransforms:
        def transforms(self, image):
            return image

    class FakePredictor:
        _bb_feat_sizes = [(1, 1), (1, 1), (1, 1)]

        def __init__(self):
            self._transforms = FakeTransforms()

        def reset_predictor(self):
            self._is_image_set = False

        def _predict(self, point_coords, point_labels, *, multimask_output, return_logits):
            assert point_coords.device.type == "cuda"
            assert point_labels.device.type == "cuda"
            masks = torch.ones((1, 3, 16, 16), dtype=torch.bool, device="cuda")
            scores = torch.tensor([[0.1, 0.9, 0.2]], device="cuda")
            return masks, scores, torch.empty(0, device="cuda")

    class FakeSam2Model:
        directly_add_no_mem_embed = False

        def forward_image(self, image):
            captured_inputs.append(image.clone())
            return image

        def _prepare_backbone_features(self, backbone_out):
            features = [torch.zeros((1, 1, 1), device="cuda") for _ in range(3)]
            return None, features, None, None

    estimator = object.__new__(PromptlessSAM2EgoMaskEstimator)
    estimator.predictor = FakePredictor()
    estimator.sam2_model = FakeSam2Model()

    image_bgr = torch.empty((1, 3, 16, 16), dtype=torch.float32, device="cuda")
    image_bgr[:, 0].fill_(10)
    image_bgr[:, 1].fill_(20)
    image_bgr[:, 2].fill_(30)
    ego_mask_out = torch.empty((16, 16), dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()

    estimator.process_device(
        image_bgr,
        outputs={SAM2_MASK_OUTPUT: ego_mask_out},
        processor_inputs={},
        stream=stream,
    )
    stream.synchronize()

    assert captured_inputs[0].device.type == "cuda"
    assert torch.allclose(captured_inputs[0][0, 0], torch.full((16, 16), 30 / 255, device="cuda"))
    assert torch.allclose(captured_inputs[0][0, 2], torch.full((16, 16), 10 / 255, device="cuda"))
    torch.testing.assert_close(ego_mask_out, torch.ones_like(ego_mask_out))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-resident SAM2 path requires a GPU")
def test_sam2_host_inference_delegates_to_device_path() -> None:
    class FakePredictor:
        device = torch.device("cuda")

    estimator = object.__new__(PromptlessSAM2EgoMaskEstimator)
    estimator.predictor = FakePredictor()
    captured_inputs = []

    def process_device(image_bgr_nchw, *, outputs, processor_inputs, stream) -> None:
        assert not processor_inputs
        captured_inputs.append(image_bgr_nchw.clone())
        assert isinstance(stream, torch.cuda.Stream)
        outputs[SAM2_MASK_OUTPUT].fill_(7)

    image_rgb = np.empty((16, 20, 3), dtype=np.uint8)
    image_rgb[..., 0] = 30
    image_rgb[..., 1] = 20
    image_rgb[..., 2] = 10
    with patch.object(estimator, "process_device", side_effect=process_device) as device_process:
        actual = estimator.predict(PILImage.fromarray(image_rgb))

    device_process.assert_called_once()
    assert captured_inputs[0].shape == (1, 3, 16, 20)
    assert captured_inputs[0].device.type == "cuda"
    torch.testing.assert_close(captured_inputs[0][0, 0], torch.full((16, 20), 10, dtype=torch.uint8, device="cuda"))
    torch.testing.assert_close(captured_inputs[0][0, 2], torch.full((16, 20), 30, dtype=torch.uint8, device="cuda"))
    np.testing.assert_array_equal(actual, np.full((16, 20), 7, dtype=np.uint8))


def test_trt_device_inference_uses_binding_tensor_device() -> None:
    """The TensorRT binding stores a tensor whose device identifies the engine."""
    input_tensor = torch.empty(1)
    predictor = MagicMock(inputs=[MagicMock(device=input_tensor)], max_batch_size=1, _input_shape=(3, 1, 1))

    with patch("mmseg.apis.inference.Mask2formerInferencer", type(predictor)):
        _, _, device = mmseg.apis.get_model_device_input_spec(predictor, (1, 1))

    assert device == input_tensor.device


def test_merge_mask2former_outputs_includes_optional_logits():
    output_path = Path("/output")
    camera_ids = ("front", "rear")
    semantic_output_path = output_path / "sequence.aux.sseg.zarr.itar"
    logits_output_path = output_path / "sequence.aux.sseg-logits.zarr.itar"
    ego_mask_output_path = output_path / "sequence.aux.egomask.zarr.itar"
    with patch(
        "apps.av.aux_gen.ncore_aux_data._merge_zarr_itar_files",
        side_effect=[semantic_output_path, logits_output_path, ego_mask_output_path],
    ) as merge:
        assert (
            _merge_mask2former_outputs(
                output_path,
                camera_ids,
                "sequence",
                seg_logits=True,
                ego_mask_camera_ids=("front",),
            )
            == semantic_output_path
        )

    assert merge.call_args_list[0].args == (
        [
            output_path / "cam" / "front" / semantic_output_path.name,
            output_path / "cam" / "rear" / semantic_output_path.name,
        ],
        semantic_output_path,
    )
    assert merge.call_args_list[1].args == (
        [
            output_path / "cam" / "front" / logits_output_path.name,
            output_path / "cam" / "rear" / logits_output_path.name,
        ],
        logits_output_path,
    )
    assert merge.call_args_list[2].args == (
        [output_path / "cam" / "front" / ego_mask_output_path.name],
        ego_mask_output_path,
    )


def test_run_mask2former_segmentation_does_not_call_get_egomask_when_has_ego_mask_false():
    """When has_ego_mask is False, writer.get_egomask must not be called (avoids creating empty egomask store)."""
    estimator = MagicMock()
    estimator.metadata.return_value = _processor_metadata({"stuff_classes": [], "stuff_colors": []})
    writer = MagicMock()
    camera_model_parameters = MagicMock()
    camera_model_parameters.resolution = np.array([64, 64])  # .tolist() is called on it
    camera_sensor = MagicMock()

    with (
        patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None),
        patch(
            "apps.av.aux_gen.ncore_aux_data._run_segmentation_camera_frames",
            wraps=_run_segmentation_camera_frames,
        ) as run_camera_frames,
    ):
        _run_serial_mask2former_segmentation(
            estimator,
            writer,
            camera_id="cam1",
            camera_model_parameters=camera_model_parameters,
            camera_sensor=camera_sensor,
            timestamped_image_frame_handles=[],  # no frames so no predict loop
            backend=MagicMock(),
            visualize=False,
            has_ego_mask=False,
        )
    run_camera_frames.assert_called_once()
    writer.get_egomask.assert_not_called()


def test_run_mask2former_segmentation_calls_get_egomask_when_has_ego_mask_true_and_no_sensor_mask():
    """When has_ego_mask is True and get_camera_sensor_mask returns None, writer.get_egomask is used."""
    estimator = MagicMock()
    estimator.metadata.return_value = _processor_metadata({"stuff_classes": [], "stuff_colors": []})
    writer = MagicMock()
    writer.get_egomask.return_value = None
    camera_model_parameters = MagicMock()
    camera_model_parameters.resolution = np.array([64, 64])
    camera_sensor = MagicMock()

    with patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None):
        _run_serial_mask2former_segmentation(
            estimator,
            writer,
            camera_id="cam1",
            camera_model_parameters=camera_model_parameters,
            camera_sensor=camera_sensor,
            timestamped_image_frame_handles=[],
            backend=MagicMock(),
            visualize=False,
            has_ego_mask=True,
        )
    writer.get_egomask.assert_called_once_with("cam1", 0)


def _segmentation_mocks(labels: np.ndarray, logits: np.ndarray | None = None):
    """Mocks for one Mask2Former frame: (estimator, writer, camera params, sensor, handles)."""
    estimator = MagicMock()
    estimator.metadata.return_value = _processor_metadata({"stuff_classes": [], "stuff_colors": []})
    estimator.predict.return_value = MagicMock(
        semantic_seg=PILImage.fromarray(labels, mode="P"), semantic_seg_logits=logits
    )
    writer = MagicMock()
    camera_model_parameters = MagicMock()
    camera_model_parameters.resolution = np.array([labels.shape[1], labels.shape[0]])
    return estimator, writer, camera_model_parameters, MagicMock(), [(1000, MagicMock())]


def test_async_shard_writer_preserves_submission_order():
    """One consumer thread must keep tar entries in submission order."""
    written: list[int] = []
    with AsyncShardWriter(max_pending_bytes=64) as shard_writer:
        for i in range(50):
            shard_writer.submit(written.append, i, nbytes=32)
    assert written == list(range(50))


def test_async_shard_writer_enforces_byte_budget():
    """Queued payloads must not exceed the configured byte budget."""
    first_write_started = threading.Event()
    release_first_write = threading.Event()
    two_items_submitted = threading.Event()

    def write(index: int) -> None:
        if index == 0:
            first_write_started.set()
            release_first_write.wait()

    def submit_items(shard_writer: AsyncShardWriter) -> None:
        for i in range(50):
            shard_writer.submit(write, i, nbytes=32)
            if i == 1:
                two_items_submitted.set()

    with AsyncShardWriter(max_pending_bytes=64) as shard_writer:
        producer = threading.Thread(target=submit_items, args=(shard_writer,))
        producer.start()
        try:
            assert first_write_started.wait(timeout=30)
            assert two_items_submitted.wait(timeout=30)
            with shard_writer._space:
                assert shard_writer._pending_bytes == 64
        finally:
            release_first_write.set()
            producer.join(timeout=30)
        assert not producer.is_alive()


def test_async_shard_writer_close_reports_writer_error():
    """A failed write must surface on the producer instead of exiting cleanly."""
    shard_writer = AsyncShardWriter()
    cause = OSError("disk full")

    def boom(_):
        raise cause

    shard_writer.submit(boom, "frame", nbytes=8)
    with pytest.raises(RuntimeError) as excinfo:
        shard_writer.close()
    assert excinfo.value.__cause__ is cause


def test_async_shard_writer_rejects_submit_after_close():
    """Submitting after close() must fail loudly rather than drop the write."""
    shard_writer = AsyncShardWriter()
    shard_writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        shard_writer.submit(print, nbytes=8)


def test_async_shard_writer_accepts_payload_larger_than_budget():
    """An oversized payload must still be written, not deadlock on backpressure."""
    written: list[str] = []
    with AsyncShardWriter(max_pending_bytes=16) as shard_writer:
        shard_writer.submit(written.append, "logits", nbytes=1 << 30)
        shard_writer.submit(written.append, "labels", nbytes=8)
    assert written == ["logits", "labels"]


def test_async_shard_writer_body_error_is_not_replaced_by_write_error():
    """When the loop itself fails, its exception must survive the drain."""
    with pytest.raises(ValueError, match="frame decode"):
        with AsyncShardWriter() as shard_writer:
            shard_writer.submit(MagicMock(side_effect=OSError("disk full")), nbytes=8)
            raise ValueError("frame decode")


def test_run_mask2former_segmentation_flushes_writes_before_returning():
    """Writes must reach the store before the function returns, i.e. before finalize()."""
    estimator, writer, camera_model_parameters, camera_sensor, handles = _segmentation_mocks(
        np.zeros((4, 4), dtype=np.uint8)
    )
    with patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None):
        _run_serial_mask2former_segmentation(
            estimator,
            writer,
            camera_id="cam1",
            camera_model_parameters=camera_model_parameters,
            camera_sensor=camera_sensor,
            timestamped_image_frame_handles=handles,
            backend=MagicMock(),
            visualize=False,
            has_ego_mask=False,
        )
    writer.store_semantic_segmentation.assert_called_once()


def test_run_mask2former_segmentation_accounts_for_all_image_bands():
    """The pending-byte budget must include every band in the queued image."""
    estimator, writer, camera_model_parameters, camera_sensor, handles = _segmentation_mocks(
        np.zeros((4, 4), dtype=np.uint8)
    )
    image = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")
    estimator.predict.return_value.semantic_seg = image

    with (
        patch.object(AsyncShardWriter, "submit") as submit,
        patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None),
    ):
        _run_serial_mask2former_segmentation(
            estimator,
            writer,
            camera_id="cam1",
            camera_model_parameters=camera_model_parameters,
            camera_sensor=camera_sensor,
            timestamped_image_frame_handles=handles,
            backend=MagicMock(),
            visualize=False,
            has_ego_mask=False,
        )

    assert submit.call_args.kwargs["nbytes"] == image.width * image.height * 3


def test_run_segmentation_camera_frames_queues_owned_auxiliary_mask(tmp_path: Path) -> None:
    """Auxiliary PNG encoding must use the bounded writer with an owned buffer."""
    writer = MagicMock()
    writer.output_dir_path = tmp_path
    mask = np.full((4, 5), 255, dtype=np.uint8)
    mask_path = tmp_path / "extension" / "masks" / "cam1" / "000000.png"
    output = SegmentationFrameOutput(
        semantic_seg=PILImage.fromarray(np.zeros((4, 5), dtype=np.uint8), mode="P"),
        auxiliary_mask=mask,
        auxiliary_mask_path=mask_path,
    )
    handle = MagicMock()

    with patch.object(AsyncShardWriter, "submit") as submit:
        _run_segmentation_camera_frames(
            writer=writer,
            camera_id="cam1",
            timestamped_image_frame_handles=[(1000, handle)],
            semantic_metadata={},
            visualize=False,
            process_frame=lambda *_args: output,
        )

    mask_call = next(call for call in submit.call_args_list if call.args[0] is save_auxiliary_mask)
    assert mask_call.args[1] == mask_path
    np.testing.assert_array_equal(mask_call.args[2], mask)
    assert mask_call.args[2] is not mask
    assert mask_call.kwargs["nbytes"] == mask.nbytes


def test_save_auxiliary_mask_does_not_publish_partial_png(tmp_path: Path) -> None:
    """A failed PNG encode should never expose a partial final mask."""
    mask_path = tmp_path / "masks" / "000000.png"
    image = MagicMock()

    def fail_after_partial_write(path: Path, *, format: str) -> None:
        assert format == "PNG"
        path.write_bytes(b"partial")
        raise OSError("disk full")

    image.save.side_effect = fail_after_partial_write
    with (
        patch("apps.av.aux_gen.shared_segmentation.PILImage.fromarray", return_value=image),
        pytest.raises(RuntimeError, match="Could not write auxiliary mask"),
    ):
        save_auxiliary_mask(mask_path, np.zeros((2, 3), dtype=np.uint8))

    assert not mask_path.exists()


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("has_mask", [False, True])
def test_shared_segmentation_requires_mask_for_sampled_frames(sampled: bool, has_mask: bool) -> None:
    """Sampled frames require a mask; existing mask handling stays unchanged."""
    mask = np.ones((2, 3), dtype=np.uint8)
    client = MagicMock()
    client.resolve.return_value.outputs = {SEMANTIC_LABELS_OUTPUT: np.zeros_like(mask)}
    if has_mask:
        client.resolve.return_value.outputs["extension_mask"] = mask
    pending = MagicMock()
    pending.device_frame.request.request_id = 0
    pending.vis_path = None
    processor = SharedSegmentationFrameProcessor(
        stuff_colors=np.zeros((1, 3), dtype=np.uint8),
        processor_inputs=(),
        client=client,
        auxiliary_outputs=AuxiliarySegmentationOutputs(
            processor_name="extension",
            mask_output="extension_mask",
            mask_root=Path("/output/extension"),
        ),
        auxiliary_sample_indices=frozenset({0}) if sampled else frozenset(),
        camera_id="camera_front",
        camera_frame_range_start=10,
    )
    if sampled and not has_mask:
        with pytest.raises(RuntimeError, match="Sampled frame omitted auxiliary mask output: extension_mask"):
            processor.resolve(pending)
    else:
        result = processor.resolve(pending)
        assert result.auxiliary_mask is (mask if has_mask else None)
        assert result.auxiliary_mask_path == (Path("/output/extension/camera_front/000010.png") if has_mask else None)


def test_shared_segmentation_delegates_statistics_to_the_extension() -> None:
    """Per-frame counters collapse into one extension-owned camera record."""

    class FakeStatistics:
        def __init__(self) -> None:
            self.totals = 0

        def accumulate(self, statistics: np.ndarray) -> None:
            self.totals += int(statistics.sum())

        def camera_record(self, camera_id: str) -> dict[str, object]:
            return {"camera_id": camera_id, "totals": self.totals}

    processor = SharedSegmentationFrameProcessor(
        stuff_colors=np.zeros((1, 3), dtype=np.uint8),
        processor_inputs=(),
        client=MagicMock(),
        auxiliary_outputs=AuxiliarySegmentationOutputs(
            processor_name="extension",
            mask_output="extension_mask",
            mask_root=Path("/output/extension"),
            statistics_output="extension_stats",
            create_statistics=FakeStatistics,
        ),
        auxiliary_sample_indices=frozenset({0, 1}),
        camera_id="camera_front",
    )

    processor._accumulate_auxiliary_statistics({"extension_stats": np.asarray([10, 2, 30], dtype=np.int32)})
    processor._accumulate_auxiliary_statistics({"extension_stats": np.asarray([0, 1, 5], dtype=np.int32)})

    assert processor.auxiliary_camera_statistics() == {"camera_id": "camera_front", "totals": 48}


def test_auxiliary_only_factory_never_initializes_mask2former_or_sam2() -> None:
    auxiliary = MagicMock()
    auxiliary.return_value.name = "extension"
    auxiliary.output_specs.return_value = (DeviceFrameOutputSpec("extension_mask", (3, 4), torch.uint8),)
    factory = SegmentationDeviceFramePipelineFactory(
        estimate_logits=False,
        enable_trt=True,
        include_sam2=False,
        include_mask2former=False,
        auxiliary_processor=auxiliary,
    )
    with (
        patch("apps.av.aux_gen.estimators.Mask2FormerSegmentationEstimator") as mask2former,
        patch("apps.av.aux_gen.estimators.PromptlessSAM2EgoMaskEstimator") as sam2,
    ):
        pipeline = factory((4, 3))
        factory((8, 6))
    mask2former.assert_not_called()
    sam2.assert_not_called()
    auxiliary.assert_called_once_with(semantic_class_names=())
    assert list(pipeline.processors) == ["extension"]
    assert [spec.name for spec in factory.output_specs((4, 3), num_classes=0)] == ["extension_mask"]


def test_auxiliary_only_skips_unselected_decode_and_semantic_storage(tmp_path: Path) -> None:
    writer = MagicMock()
    writer.output_dir_path = tmp_path
    handles = [MagicMock() for _ in range(4)]
    labels = np.asarray([[0, 7], [255, 7]], dtype=np.uint8)
    process = MagicMock(
        return_value=SegmentationFrameOutput(
            auxiliary_mask=labels,
            auxiliary_mask_path=tmp_path / "labels" / "000002.png",
        )
    )
    with patch.object(AsyncShardWriter, "submit") as submit:
        _run_segmentation_camera_frames(
            writer=writer,
            camera_id="camera",
            semantic_metadata=None,
            visualize=False,
            timestamped_image_frame_handles=[(100 + i, handle) for i, handle in enumerate(handles)],
            process_frame=process,
            selected_frame_indices=frozenset({2}),
        )
    assert [handle.get_data.call_count for handle in handles] == [0, 0, 1, 0]
    assert process.call_args.args[:2] == (2, 102)
    writer.store_semantic_meta.assert_not_called()
    writer.store_semantic_segmentation.assert_not_called()
    submit.assert_called_once()
    assert submit.call_args.args[0] is save_auxiliary_mask
    np.testing.assert_array_equal(submit.call_args.args[2], labels)
    assert submit.call_args.args[2] is not labels


def test_auxiliary_only_does_not_merge_semantic_archives(tmp_path: Path) -> None:
    with patch("apps.av.aux_gen.ncore_aux_data._merge_zarr_itar_files") as merge:
        assert _merge_mask2former_outputs(tmp_path, ["camera"], "clip", False, include_mask2former=False) is None
    merge.assert_not_called()


def test_shared_segmentation_requires_statistics_alongside_an_auxiliary_mask() -> None:
    """A missing counters buffer must fail loudly instead of writing zeros."""
    processor = SharedSegmentationFrameProcessor(
        stuff_colors=np.zeros((1, 3), dtype=np.uint8),
        processor_inputs=(),
        client=MagicMock(),
        auxiliary_outputs=AuxiliarySegmentationOutputs(
            processor_name="extension",
            mask_output="extension_mask",
            mask_root=Path("/output/extension"),
            statistics_output="extension_stats",
            create_statistics=MagicMock(),
        ),
        auxiliary_sample_indices=frozenset({0}),
        camera_id="camera_front",
    )

    with pytest.raises(RuntimeError, match="without its statistics output"):
        processor._accumulate_auxiliary_statistics({})


def test_run_mask2former_segmentation_propagates_store_failure():
    """A store failure must escape, so parallel workers exit non-zero."""
    estimator, writer, camera_model_parameters, camera_sensor, handles = _segmentation_mocks(
        np.zeros((4, 4), dtype=np.uint8)
    )
    writer.store_semantic_segmentation.side_effect = OSError("disk full")
    with patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None):
        with pytest.raises(RuntimeError, match="async shard write failed"):
            _run_serial_mask2former_segmentation(
                estimator,
                writer,
                camera_id="cam1",
                camera_model_parameters=camera_model_parameters,
                camera_sensor=camera_sensor,
                timestamped_image_frame_handles=handles,
                backend=MagicMock(),
                visualize=False,
                has_ego_mask=False,
            )


def test_run_mask2former_segmentation_copies_labels_before_submit():
    """PILImage.fromarray aliases the estimator's label buffer, so the submitted image must be a copy.

    Two frames, with the writer thread held until the second frame has overwritten
    the shared buffer: without the copy, frame one's queued image would be encoded
    after the overwrite and the wrong labels would be stored.
    """
    labels = np.zeros((4, 4), dtype=np.uint8)
    estimator, writer, camera_model_parameters, camera_sensor, _ = _segmentation_mocks(labels)
    buffer_overwritten = threading.Event()
    frames = itertools.count()
    stored_maxima: list[int] = []

    def predict(*_args, **_kwargs):
        if next(frames):  # the second frame reuses the same buffer
            labels[:] = 7
            buffer_overwritten.set()
        return MagicMock(semantic_seg=PILImage.fromarray(labels, mode="P"), semantic_seg_logits=None)

    def store(_camera_id, _timestamp, image, _fmt):
        assert buffer_overwritten.wait(timeout=30)
        stored_maxima.append(int(np.asarray(image).max()))

    estimator.predict.side_effect = predict
    writer.store_semantic_segmentation.side_effect = store
    with patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None):
        _run_serial_mask2former_segmentation(
            estimator,
            writer,
            camera_id="cam1",
            camera_model_parameters=camera_model_parameters,
            camera_sensor=camera_sensor,
            timestamped_image_frame_handles=[(1000, MagicMock()), (2000, MagicMock())],
            backend=MagicMock(),
            visualize=False,
            has_ego_mask=False,
        )
    assert stored_maxima == [0, 7]


def test_run_mask2former_segmentation_copies_reused_float16_logits_before_submit():
    """A contiguous float16 producer buffer must not alias queued logits."""
    labels = np.zeros((4, 4), dtype=np.uint8)
    logits = np.zeros((4, 4, 3), dtype=np.float16)
    estimator, writer, camera_model_parameters, camera_sensor, _ = _segmentation_mocks(labels, logits=logits)
    buffer_overwritten = threading.Event()
    frames = itertools.count()
    stored_maxima: list[int] = []

    def predict(*_args, **_kwargs):
        if next(frames):  # the second frame reuses the same buffer
            logits[:] = 7
            buffer_overwritten.set()
        return MagicMock(semantic_seg=PILImage.fromarray(labels, mode="P"), semantic_seg_logits=logits)

    def store(_camera_id, _timestamp, queued_logits):
        assert buffer_overwritten.wait(timeout=30)
        stored_maxima.append(int(queued_logits.max()))

    estimator.predict.side_effect = predict
    writer.store_semantic_logits.side_effect = store
    with patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None):
        _run_serial_mask2former_segmentation(
            estimator,
            writer,
            camera_id="cam1",
            camera_model_parameters=camera_model_parameters,
            camera_sensor=camera_sensor,
            timestamped_image_frame_handles=[(1000, MagicMock()), (2000, MagicMock())],
            backend=MagicMock(),
            visualize=False,
            has_ego_mask=False,
        )
    assert stored_maxima == [0, 7]


def test_store_semantic_logits_reuses_float16_input():
    """The real store method must not duplicate producer-owned float16 logits."""
    writer = AuxDataWriter.__new__(AuxDataWriter)
    writer.zlib_compressor = MagicMock()
    base_group = MagicMock()
    logits = np.zeros((4, 4, 3), dtype=np.float16)

    with (
        patch.object(AuxDataWriter, "get_base_group", return_value=base_group),
        patch("nurec.utils.ncore_utils.zarr.Group", MagicMock),
    ):
        writer.store_semantic_logits("cam1", 1000, logits)

    stored = base_group.require_group.return_value.create_dataset.call_args.kwargs["data"]
    assert stored is logits


def test_run_mask2former_segmentation_casts_logits_to_float16():
    """Logits are queued as float16: same stored values as the store's own cast, half the bytes."""
    logits = np.full((4, 4, 3), 0.5, dtype=np.float32)
    estimator, writer, camera_model_parameters, camera_sensor, handles = _segmentation_mocks(
        np.zeros((4, 4), dtype=np.uint8), logits=logits
    )
    with patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None):
        _run_serial_mask2former_segmentation(
            estimator,
            writer,
            camera_id="cam1",
            camera_model_parameters=camera_model_parameters,
            camera_sensor=camera_sensor,
            timestamped_image_frame_handles=handles,
            backend=MagicMock(),
            visualize=False,
            has_ego_mask=False,
        )
    stored = writer.store_semantic_logits.call_args.args[2]
    assert stored.dtype == np.float16
    np.testing.assert_array_equal(stored, logits.astype(np.float16))


def test_shared_host_worker_preserves_frame_order_and_writes_complete_outputs(tmp_path: Path):
    resolution = (4, 3)
    output_specs = (
        DeviceFrameOutputSpec(SEMANTIC_LABELS_OUTPUT, (3, 4), torch.uint8),
        DeviceFrameOutputSpec(SEMANTIC_UNCERTAINTY_OUTPUT, (3, 4), torch.uint8),
        DeviceFrameOutputSpec(SAM2_MASK_OUTPUT, (3, 4), torch.uint8),
    )
    shared_buffers = [
        DeviceFrameSharedBuffers.create(resolution, output_specs),
        DeviceFrameSharedBuffers.create(resolution, output_specs),
    ]
    slot_queue: queue.Queue = queue.Queue(maxsize=2)
    for buffers in shared_buffers:
        slot_queue.put(buffers.slot)
    request_queue: queue.Queue = queue.Queue(maxsize=2)
    result_queue: queue.Queue = queue.Queue(maxsize=2)
    error_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    sequence_loader = MagicMock()
    sequence_loader.sequence_id = "sequence"
    camera_sensor = sequence_loader.get_camera_sensor.return_value
    camera_sensor.frames_count = 2
    camera_sensor.get_frames_timestamps_us.return_value = [100, 200]
    frame_handles = []
    for value in (1, 2):
        frame_handle = MagicMock()
        frame_handle.get_data.return_value.get_decoded_image.return_value = PILImage.fromarray(
            np.full((3, 4, 3), value, dtype=np.uint8)
        )
        frame_handles.append(frame_handle)
    camera_sensor.get_frame_handle.side_effect = frame_handles

    writer = MagicMock()
    writer.output_dir_path = tmp_path
    writer.store_base_name = "base"

    def respond() -> None:
        buffers_by_input_name = {buffers.slot.input_name: buffers for buffers in shared_buffers}
        for sample_request_id in range(2):
            request = request_queue.get(timeout=1.0)
            assert request.request_id == sample_request_id
            assert request.enabled_processors == frozenset({SAM2_PROCESSOR})
            assert request.processor_inputs == ()
            buffers = buffers_by_input_name[request.slot.input_name]
            buffers.output_buffers[SAM2_MASK_OUTPUT].fill(request.request_id)
            result_queue.put(
                DeviceFrameResult(
                    request_id=request.request_id,
                    error=None,
                    output_names=(SAM2_MASK_OUTPUT,),
                )
            )

        requests = [request_queue.get(timeout=1.0) for _ in range(2)]
        assert [request.request_id for request in requests] == [0, 1]
        for request in requests:
            assert request.enabled_processors == frozenset({MASK2FORMER_PROCESSOR})
            assert len(request.processor_inputs) == 1
            assert request.processor_inputs[0].name == CAMERA_MASK_INPUT
            generated_mask = DeviceFrameSharedArray.attach(request.processor_inputs[0])
            try:
                assert generated_mask.array.dtype == np.bool_
                np.testing.assert_array_equal(
                    generated_mask.array,
                    np.ones((3, 4), dtype=np.bool_),
                )
            finally:
                generated_mask.close()
            buffers = buffers_by_input_name[request.slot.input_name]
            buffers.output_buffers[SEMANTIC_LABELS_OUTPUT].fill(request.request_id)
            buffers.output_buffers[SEMANTIC_UNCERTAINTY_OUTPUT].fill(request.request_id)
        for request in reversed(requests):
            result_queue.put(
                DeviceFrameResult(
                    request_id=request.request_id,
                    error=None,
                    output_names=(SEMANTIC_LABELS_OUTPUT, SEMANTIC_UNCERTAINTY_OUTPUT),
                )
            )

    responder = threading.Thread(target=respond)
    responder.start()
    try:
        with (
            patch(
                "apps.av.aux_gen.ncore_aux_data._open_worker_sequence",
                return_value=(MagicMock(), sequence_loader),
            ),
            patch("apps.av.aux_gen.ncore_aux_data.sequence_loader_shard_info", return_value=("base", 0, 1)),
            patch("apps.av.aux_gen.ncore_aux_data.AuxDataWriter", return_value=writer),
            patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None),
            patch("apps.av.aux_gen.ncore_aux_data.is_dark_image", return_value=False),
            patch(
                "apps.av.aux_gen.ncore_aux_data._run_segmentation_camera_frames",
                wraps=_run_segmentation_camera_frames,
            ) as run_camera_frames,
        ):
            _segmentation_host_worker(
                _SegmentationHostWorkerConfig(
                    worker_id=0,
                    loader_payload=MagicMock(spec=LoaderWorkerPayload),
                    zarr_store_type="itar",
                    output_path=MagicMock(),
                    camera_id="camera",
                    camera_frame_range=range(2),
                    semantic_metadata={
                        "resolution": list(resolution),
                        "dataset_name": "test",
                        "method": "mask2former",
                        "pretrained_checkpoint": "test.pth",
                        "stuff_classes": ["background", "foreground"],
                        "stuff_colors": [[0, 0, 0], [255, 255, 255]],
                    },
                    resolution=resolution,
                    visualize=True,
                    slot_queue=slot_queue,
                    sam2_sample_indices=(0, 1),
                    sam2_metadata={
                        "dataset_name": "test",
                        "method": "sam2",
                        "pretrained_checkpoint": "sam2.pt",
                    },
                    ego_mask_aggregation_method="majority",
                    request_queue=request_queue,
                    result_queue=result_queue,
                    error_queue=error_queue,
                    stop_event=stop_event,
                )
            )
        run_camera_frames.assert_called_once()
        responder.join(timeout=1.0)

        assert not stop_event.is_set()
        assert error_queue.empty()
        assert slot_queue.qsize() == 2

        assert [call.args[1] for call in writer.store_semantic_segmentation.call_args_list] == [100, 200]
        stored_labels = [np.asarray(call.args[2]) for call in writer.store_semantic_segmentation.call_args_list]
        np.testing.assert_array_equal(stored_labels[0], np.zeros((3, 4), dtype=np.uint8))
        np.testing.assert_array_equal(stored_labels[1], np.ones((3, 4), dtype=np.uint8))
        first_visualization = np.asarray(PILImage.open(tmp_path / "seg_logits/camera/base/100.png"))
        second_visualization = np.asarray(PILImage.open(tmp_path / "seg_logits/camera/base/200.png"))
        np.testing.assert_array_equal(first_visualization, np.zeros((3, 4, 3), dtype=np.uint8))
        np.testing.assert_array_equal(second_visualization, np.full((3, 4, 3), 128, dtype=np.uint8))
        first_uncertainty = np.asarray(PILImage.open(tmp_path / "seg_logits/camera/base/100_uncertainty_mask.png"))
        second_uncertainty = np.asarray(PILImage.open(tmp_path / "seg_logits/camera/base/200_uncertainty_mask.png"))
        np.testing.assert_array_equal(first_uncertainty, np.zeros((3, 4), dtype=np.uint8))
        np.testing.assert_array_equal(second_uncertainty, np.full((3, 4), 255, dtype=np.uint8))
        assert [call.args[1] for call in writer.store_egomask.call_args_list] == [100, 200, 0]
        stored_ego_masks = [np.asarray(call.args[2]) for call in writer.store_egomask.call_args_list]
        np.testing.assert_array_equal(stored_ego_masks[0], np.zeros((3, 4), dtype=np.uint8))
        np.testing.assert_array_equal(stored_ego_masks[1], np.ones((3, 4), dtype=np.uint8))
        np.testing.assert_array_equal(stored_ego_masks[2], np.ones((3, 4), dtype=np.uint8))
        writer.store_egomask_meta.assert_called_once()
        writer.finalize.assert_called_once_with()
    finally:
        responder.join(timeout=1.0)
        for buffers in shared_buffers:
            buffers.close()


def test_shared_host_worker_finalizes_partial_writer_after_failure() -> None:
    sequence_loader = MagicMock(sequence_id="sequence")
    camera_sensor = sequence_loader.get_camera_sensor.return_value
    camera_sensor.frames_count = 0
    camera_sensor.get_frames_timestamps_us.return_value = []
    writer = MagicMock()
    error_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    with (
        patch(
            "apps.av.aux_gen.ncore_aux_data._open_worker_sequence",
            return_value=(MagicMock(), sequence_loader),
        ),
        patch("apps.av.aux_gen.ncore_aux_data.sequence_loader_shard_info", return_value=("base", 0, 1)),
        patch("apps.av.aux_gen.ncore_aux_data.AuxDataWriter", return_value=writer),
        patch("apps.av.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None),
        patch(
            "apps.av.aux_gen.ncore_aux_data._run_segmentation_camera_frames",
            side_effect=RuntimeError("inference failed"),
        ),
        pytest.raises(RuntimeError, match="inference failed"),
    ):
        _segmentation_host_worker(
            _SegmentationHostWorkerConfig(
                worker_id=0,
                loader_payload=MagicMock(spec=LoaderWorkerPayload),
                zarr_store_type="itar",
                output_path=Path("."),
                camera_id="camera",
                camera_frame_range=range(0),
                semantic_metadata={
                    "resolution": [4, 3],
                    "dataset_name": "test",
                    "method": "mask2former",
                    "pretrained_checkpoint": "test.pth",
                    "stuff_classes": ["background"],
                    "stuff_colors": [[0, 0, 0]],
                },
                resolution=(4, 3),
                visualize=False,
                slot_queue=queue.Queue(),
                sam2_sample_indices=(),
                sam2_metadata=None,
                ego_mask_aggregation_method="majority",
                request_queue=queue.Queue(),
                result_queue=queue.Queue(),
                error_queue=error_queue,
                stop_event=stop_event,
            )
        )

    assert stop_event.is_set()
    assert error_queue.get_nowait()[0] == "camera"
    writer.finalize.assert_called_once_with()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-resident estimator path requires a GPU")
@pytest.mark.parametrize("backend", ["tensorrt", "pytorch"])
def test_mask2former_device_inference_keeps_outputs_on_cuda(backend: str) -> None:
    height, width, num_classes = 2, 3, 3
    captured_inputs = []

    class FakeDevicePredictor:
        inputs = [MagicMock(device=torch.empty(18, dtype=torch.uint8, device="cuda"))]
        max_batch_size = 1
        _input_shape = [3, height, width]
        dataset_meta = {"classes": ["building", "egocar", "road"]}
        cfg = MagicMock(trt_settings=MagicMock(backbone_feats_shape=(0,)))

        def infer_device(self, image, stream):
            assert image.device.type == "cuda"
            logits = torch.zeros((num_classes, height, width), dtype=torch.float32, device="cuda")
            labels = torch.full((1, height, width), 2, dtype=torch.int8, device="cuda")
            backbone = torch.empty(0, device="cuda")
            return [logits.reshape(-1), labels.reshape(-1), backbone]

    class FakePyTorchPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), device="cuda"))
            self.cfg = MagicMock(
                test_pipeline=[
                    {
                        "type": "Resize",
                        "scale": (width * 2, height * 2),
                        "keep_ratio": False,
                    }
                ]
            )

        def data_preprocessor(self, data, *, training):
            assert not training
            captured_inputs.append(data["inputs"][0].clone())
            return {
                "inputs": torch.stack(data["inputs"]),
                "data_samples": data["data_samples"],
            }

        def predict(self, inputs, data_samples):
            assert inputs.device.type == "cuda"
            assert data_samples[0].metainfo["ori_shape"] == (height, width)
            prediction = MagicMock()
            prediction.seg_logits.data = torch.zeros(
                (num_classes, height, width),
                dtype=torch.float32,
                device="cuda",
            )
            prediction.pred_sem_seg.data = torch.full(
                (1, height, width),
                2,
                dtype=torch.uint8,
                device="cuda",
            )
            prediction.backbone_feats = None
            return [prediction]

        def test_step(self, data):
            preprocessed = self.data_preprocessor(data, training=False)
            return self.predict(preprocessed["inputs"], preprocessed["data_samples"])

    estimator = Mask2FormerSegmentationEstimator.__new__(Mask2FormerSegmentationEstimator)
    estimator.predictor = FakeDevicePredictor() if backend == "tensorrt" else FakePyTorchPredictor()
    estimator.resolution = [width, height]
    estimator.stuff_classes = ["building", "egocar", "road"]
    estimator.estimate_logits = True
    estimator.output_uncertainty_mask = False
    estimator.name = MASK2FORMER_PROCESSOR

    stream = torch.cuda.Stream()
    image = torch.zeros((1, 3, height, width), dtype=torch.uint8, device="cuda")
    labels_out = torch.empty((height, width), dtype=torch.uint8, device="cuda")
    logits_out = torch.empty((height, width, num_classes), dtype=torch.float16, device="cuda")
    ego_mask = torch.zeros((height, width), dtype=torch.bool, device="cuda")
    ego_mask[0, 0] = True
    backend_type_patch: Any = (
        patch("mmseg.apis.inference.BaseSegmentor", FakePyTorchPredictor)
        if backend == "pytorch"
        else patch("mmseg.apis.inference.Mask2formerInferencer", FakeDevicePredictor)
    )
    with backend_type_patch:
        estimator.process_device(
            image,
            outputs={
                SEMANTIC_LABELS_OUTPUT: labels_out,
                SEMANTIC_LOGITS_OUTPUT: logits_out,
            },
            processor_inputs={CAMERA_MASK_INPUT: ego_mask},
            stream=stream,
        )
    stream.synchronize()

    expected_labels = torch.full((height, width), 2, dtype=torch.uint8, device="cuda")
    expected_labels[0, 0] = 1
    torch.testing.assert_close(labels_out, expected_labels)
    torch.testing.assert_close(logits_out[0, 0], torch.tensor([0, 1, 0], dtype=torch.float16, device="cuda"))
    if backend == "pytorch":
        assert captured_inputs[0].shape == (3, height * 2, width * 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-resident estimator path requires a GPU")
def test_mask2former_host_inference_delegates_to_device_path() -> None:
    height, width, num_classes = 2, 3, 3
    predictor = MagicMock()
    predictor.inputs = [MagicMock(device=torch.empty(18, dtype=torch.uint8, device="cuda"))]
    predictor.max_batch_size = 1
    predictor._input_shape = [3, height, width]

    estimator = Mask2FormerSegmentationEstimator.__new__(Mask2FormerSegmentationEstimator)
    estimator.predictor = predictor
    estimator.resolution = [width, height]
    estimator.stuff_classes = ["building", "egocar", "road"]
    estimator.estimate_logits = True
    estimator.estimate_dinov2_feats = False
    estimator.output_uncertainty_mask = False
    captured_inputs = []

    def infer_device(image_bgr_nchw, ego_mask, stream):
        captured_inputs.append(image_bgr_nchw.clone())
        assert ego_mask is not None
        assert isinstance(stream, torch.cuda.Stream)
        labels = torch.full((height, width), 2, dtype=torch.uint8, device="cuda")
        labels[ego_mask] = 1
        logits = torch.full((num_classes, height, width), 3, dtype=torch.float32, device="cuda")
        return labels, logits, None, torch.zeros((height, width), dtype=torch.bool, device="cuda")

    image_rgb = np.empty((height, width, 3), dtype=np.uint8)
    image_rgb[..., 0] = 30
    image_rgb[..., 1] = 20
    image_rgb[..., 2] = 10
    ego_mask = np.zeros((height, width), dtype=bool)
    ego_mask[0, 0] = True
    with (
        patch(
            "apps.av.aux_gen.estimators.mmseg.apis.get_model_device_input_spec",
            return_value=((1, 3, height, width), torch.uint8, torch.device("cuda")),
        ),
        patch.object(estimator, "infer_device", side_effect=infer_device) as device_predict,
    ):
        result = estimator.predict(PILImage.fromarray(image_rgb), ego_mask=ego_mask)

    device_predict.assert_called_once()
    assert captured_inputs[0].shape == (1, 3, height, width)
    assert captured_inputs[0].device.type == "cuda"
    torch.testing.assert_close(
        captured_inputs[0][0, 0],
        torch.full((height, width), 10, dtype=torch.uint8, device="cuda"),
    )
    torch.testing.assert_close(
        captured_inputs[0][0, 2],
        torch.full((height, width), 30, dtype=torch.uint8, device="cuda"),
    )
    expected_labels = np.full((height, width), 2, dtype=np.uint8)
    expected_labels[0, 0] = 1
    np.testing.assert_array_equal(np.asarray(result.semantic_seg), expected_labels)
    assert result.semantic_seg_logits is not None
    np.testing.assert_array_equal(
        result.semantic_seg_logits,
        np.full((height, width, num_classes), 3, dtype=np.float16),
    )
    assert result.semantic_dinov2_feats is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TensorRT device binding test requires CUDA")
@pytest.mark.parametrize("execute_v3", [False, True])
@pytest.mark.parametrize("enqueue_succeeds", [False, True])
def test_mask2former_device_inference_binds_request_tensor_directly(
    execute_v3: bool,
    enqueue_succeeds: bool,
) -> None:
    engine_input_device = torch.zeros(18, dtype=torch.float32, device="cuda")
    engine_input = MagicMock()
    engine_input.device = engine_input_device
    engine_input.device_ptr = engine_input_device.data_ptr()
    engine_input.name = "images"
    output_device = torch.empty(1, dtype=torch.float32, device="cuda")
    inferencer = object.__new__(Mask2formerInferencer)
    inferencer.max_batch_size = 1
    inferencer._input_shape = [3, 2, 3]
    inferencer.inputs = [engine_input]
    inferencer.outputs = [MagicMock(device=output_device)]
    inferencer.bindings = [engine_input.device_ptr, output_device.data_ptr()]
    inferencer.execute_v3 = execute_v3
    inferencer.context = MagicMock()
    inferencer.context.set_tensor_address.return_value = True
    inferencer.context.execute_async.return_value = enqueue_succeeds
    inferencer.context.execute_async_v3.return_value = enqueue_succeeds
    inferencer.trt_runtime = None
    inferencer.engine = None
    inferencer.stream = torch.cuda.Stream()

    image = torch.ones((1, 3, 2, 3), dtype=torch.float32, device="cuda")
    if enqueue_succeeds:
        outputs = inferencer.infer_device(image, torch.cuda.Stream())
        assert outputs[0] is output_device
    else:
        with pytest.raises(RuntimeError, match="failed to enqueue"):
            inferencer.infer_device(image, torch.cuda.Stream())
    assert torch.count_nonzero(engine_input_device).item() == 0
    if execute_v3:
        assert inferencer.context.set_tensor_address.call_args_list == [
            call("images", image.data_ptr()),
            call("images", engine_input.device_ptr),
        ]
        inferencer.context.execute_async_v3.assert_called_once()
        inferencer.context.execute_async.assert_not_called()
    else:
        inferencer.context.set_tensor_address.assert_not_called()
        inferencer.context.execute_async_v3.assert_not_called()
        inferencer.context.execute_async.assert_called_once()
        assert inferencer.context.execute_async.call_args.kwargs["bindings"] == [
            image.data_ptr(),
            output_device.data_ptr(),
        ]
        assert inferencer.bindings == [engine_input.device_ptr, output_device.data_ptr()]


@pytest.mark.parametrize(
    "range_args", [[], ["offset", "--sequence-duration-sec=0.4"], ["sensor-frames", "--main-sensor-id=cam"]]
)
def test_cli_forwards_instance_generation(range_args) -> None:
    request = MagicMock()
    configure = MagicMock(return_value=request)
    command = build_ncore_aux_data_cli(configure_instance_generation=configure)
    with patch("apps.av.aux_gen.ncore_aux_data.ncore_aux_data") as run:
        result = CliRunner().invoke(command, ["--output-dir=/tmp/unused", "--segmentation-backend=none", *range_args])
    assert result.exit_code == 0, result.output
    configure.assert_called_once()
    assert run.call_args.args[5] is request
