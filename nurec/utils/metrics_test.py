# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
import yaml

from pydantic import TypeAdapter

from nurec.config.version import Version
from nurec.utils import metrics as metrics_module
from nurec.utils.metrics import (
    AggregatedMetricRow,
    FrameMetricSample,
    HardwareInfo,
    MetricAggregate,
    MetricRow,
    MetricSample,
    MetricsCollector,
    MetricsFileContent,
    MetricsFileRunInfo,
    MetricsSamples,
    MetricsSamplesPerSequencePerSensor,
    StepwiseMetricSample,
    aggregated_metric_rows_from_content,
    create_metric_sample,
    metric_rows_from_content,
    write_aggregated_metrics_to_parquet,
    write_metrics_to_parquet,
)


REPORTING_DATE = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
SEQUENCE_ID = "clipgt-01234567-89ab-cdef-0123-456789abcdef"


def make_metrics_content(
    sensor_type: str = "lidar",
    metrics: MetricsSamples | None = None,
    aggregated_metrics: dict[str, MetricAggregate] | None = None,
) -> MetricsFileContent:
    """Build a representative camera or LiDAR validation result with frame metadata."""

    if metrics is None:
        sensor_metrics: dict[str, list[MetricSample | StepwiseMetricSample]] = {
            "test/chamfer_distance": [
                StepwiseMetricSample(
                    name="test/chamfer_distance",
                    value=0.25,
                    timestamp_us_begin=100,
                    timestamp_us_end=200,
                    unique_frame_idx=3,
                )
            ]
        }
        metrics = MetricsSamples(
            general={
                "test/elapsed_time_s": [MetricSample(name="test/elapsed_time_s", value=1.5)],
            },
            per_sequence={
                SEQUENCE_ID: MetricsSamplesPerSequencePerSensor(
                    per_camera={"camera_front": sensor_metrics} if sensor_type == "camera" else {},
                    per_lidar={"lidar_top_360fov": sensor_metrics} if sensor_type == "lidar" else {},
                )
            },
        )

    if aggregated_metrics is None:
        aggregated_metrics = {
            "test/chamfer_distance": MetricAggregate(value=0.25, aggregation_method="mean"),
        }

    return MetricsFileContent(
        aggregated_metrics=aggregated_metrics,
        metrics=metrics,
        program_version=Version(
            version_major=26,
            version_minor=6,
            version_patch=0,
            git_commit_sha_short="abc1234",
            git_commit_date=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        run_info=MetricsFileRunInfo(
            train_config_name="configs/apps/prod/Hyperion-8.1/car2sim_lidar.yaml",
            mode="val",
            run_id="lidar-metrics",
        ),
    )


def test_write_metrics_to_parquet_preserves_lidar_sample_metadata(tmp_path: Path) -> None:
    """The parquet output has the same normalized rows as Kratos telemetry."""

    content = make_metrics_content()
    rows = metric_rows_from_content(content, reporting_date=REPORTING_DATE)

    assert len(rows) == 2
    lidar_row = next(row for row in rows if row.lidar_id is not None)
    assert lidar_row.sequence_id == "clipgt-01234567-89ab-cdef-0123-456789abcdef"
    assert lidar_row.lidar_id == "lidar_top_360fov"
    assert lidar_row.camera_id is None
    assert lidar_row.metric_name == "test/chamfer_distance"
    assert lidar_row.timestamp_us_begin == 100
    assert lidar_row.timestamp_us_end == 200
    assert lidar_row.unique_frame_idx == 3
    assert lidar_row.reporting_date == REPORTING_DATE.isoformat()
    # Generic and timestamped stepwise samples keep their original identity fields.
    assert all((row.frame, row.iteration, row.valid) == (None, None, True) for row in rows)
    assert list(lidar_row.to_dict()) == MetricRow.column_names()

    output_path = write_metrics_to_parquet(content, tmp_path / "metrics.parquet", reporting_date=REPORTING_DATE)
    table = pq.read_table(output_path)

    assert table.column_names == MetricRow.column_names()
    assert table.num_rows == len(rows)
    assert table.to_pylist() == [row.to_dict() for row in rows]
    parquet_lidar_row = next(row for row in table.to_pylist() if row["lidar_id"] == "lidar_top_360fov")
    assert parquet_lidar_row["value"] == lidar_row.value
    assert parquet_lidar_row["timestamp_us_begin"] == lidar_row.timestamp_us_begin
    assert parquet_lidar_row["run_id"] == "lidar-metrics"


def test_write_aggregated_metrics_to_parquet_preserves_nurec_reductions(tmp_path: Path) -> None:
    """Dashboard aggregates must retain the values already computed by NuRec."""

    content = make_metrics_content()
    rows = aggregated_metric_rows_from_content(content, reporting_date=REPORTING_DATE)

    assert len(rows) == 1
    aggregate_row = rows[0]
    assert aggregate_row.sequence_id == "clipgt-01234567-89ab-cdef-0123-456789abcdef"
    assert aggregate_row.metric_name == "test/chamfer_distance"
    assert aggregate_row.aggregation_method == "mean"
    assert aggregate_row.value == 0.25
    assert list(aggregate_row.to_dict()) == AggregatedMetricRow.column_names()

    output_path = write_aggregated_metrics_to_parquet(
        content,
        tmp_path / "aggregated_metrics.parquet",
        reporting_date=REPORTING_DATE,
    )
    table = pq.read_table(output_path)

    assert table.column_names == AggregatedMetricRow.column_names()
    assert table.num_rows == len(rows)
    parquet_row = table.to_pylist()[0]
    assert parquet_row["value"] == aggregate_row.value
    assert parquet_row["aggregation_method"] == "mean"
    assert parquet_row["reporting_date"] == REPORTING_DATE.isoformat()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_metrics_parquet_schema_is_stable_for_sparse_and_empty_content(tmp_path: Path, nonfinite: float) -> None:
    """Parquet column types must not depend on the sensors or values in one run."""

    nonfinite_content = make_metrics_content(
        metrics=MetricsSamples(
            general={"test/nonfinite": [MetricSample(name="test/nonfinite", value=nonfinite)]},
            per_sequence={},
        ),
        aggregated_metrics={},
    )
    empty_content = make_metrics_content(metrics=MetricsSamples(general={}, per_sequence={}), aggregated_metrics={})

    output_paths = [
        write_metrics_to_parquet(make_metrics_content("lidar"), tmp_path / "lidar.parquet", REPORTING_DATE),
        write_metrics_to_parquet(make_metrics_content("camera"), tmp_path / "camera.parquet", REPORTING_DATE),
        write_metrics_to_parquet(nonfinite_content, tmp_path / "nonfinite.parquet", REPORTING_DATE),
        write_metrics_to_parquet(empty_content, tmp_path / "empty.parquet", REPORTING_DATE),
    ]
    tables = [pq.read_table(output_path) for output_path in output_paths]

    assert all(table.schema == tables[0].schema for table in tables)
    assert tables[0].column_names == MetricRow.column_names()
    assert tables[0].schema.field("camera_id").type == pa.string()
    assert tables[0].schema.field("lidar_id").type == pa.string()
    assert tables[0].schema.field("value").type == pa.float64()
    assert tables[0].schema.field("frame").type == pa.int64()
    assert tables[0].schema.field("iteration").type == pa.int64()
    assert tables[0].schema.field("valid").type == pa.bool_()
    nonfinite_row = tables[2].to_pylist()[0]
    assert nonfinite_row["value"] is None
    assert (nonfinite_row["frame"], nonfinite_row["iteration"], nonfinite_row["valid"]) == (None, None, True)
    assert tables[3].num_rows == 0


def test_aggregated_metrics_parquet_schema_is_stable_for_empty_content(tmp_path: Path) -> None:
    """Aggregate parquet retains its dashboard schema even when no reductions exist."""

    content = make_metrics_content(metrics=MetricsSamples(general={}, per_sequence={}), aggregated_metrics={})
    output_path = write_aggregated_metrics_to_parquet(content, tmp_path / "empty_aggregates.parquet", REPORTING_DATE)
    table = pq.read_table(output_path)

    assert table.column_names == AggregatedMetricRow.column_names()
    assert table.schema.field("sequence_id").type == pa.string()
    assert table.schema.field("value").type == pa.float64()
    assert table.num_rows == 0


def test_aggregate_metrics_merges_scalar_samples_without_cuda(tmp_path: Path, monkeypatch) -> None:
    """Scalar metric aggregation must work on a no-GPU host.

    Reproduces the multi-process serve metrics-merge path: pool real per-frame samples into a
    ``MetricsCollector`` and then ``write_metrics_to_yaml`` -> ``aggregate_metrics``. With CUDA
    forced unavailable, the historical ``torch.tensor(values, device="cuda")`` raised
    ``RuntimeError: No CUDA GPUs are available``; the device-flexible path must fall back to CPU
    and still emit ``aggregated_metrics`` with the correct reductions.
    """

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    version = Version(
        version_major=26,
        version_minor=6,
        version_patch=0,
        git_commit_sha_short="abc1234",
        git_commit_date=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    collector = MetricsCollector(train_config_name="cfg", mode="val", run_id="merge", version=version)

    # Real pooled samples across camera and lidar sensors, plus a general metric.
    camera_values = [0.1, 0.2, 0.3, 0.4]
    for value in camera_values:
        collector.save_metric(
            create_metric_sample("test/psnr", value),
            unique_sensor_id="camera_front",
            sequence_id=SEQUENCE_ID,
        )
    for value in [10.0, 20.0]:
        collector.save_metric(
            create_metric_sample("test/chamfer_distance", value),
            unique_sensor_id="lidar_top",
            sequence_id=SEQUENCE_ID,
            is_lidar=True,
        )
    collector.save_metric(create_metric_sample("test/elapsed_time_s", 1.5))

    def sum_in_order(values: torch.Tensor) -> torch.Tensor:
        assert values.tolist() == [2.0, 3.0, 5.0]
        return values.sum()

    for sensor_id, is_lidar, value in (("camera_front", False, 2.0), ("lidar_top", True, 3.0), (None, False, 5.0)):
        collector.save_metric(
            create_metric_sample("shared", value),
            unique_sensor_id=sensor_id,
            sequence_id=SEQUENCE_ID if sensor_id is not None else None,
            is_lidar=is_lidar,
            reduce_fx=sum_in_order,
        )
    collector.save_metric(
        FrameMetricSample(name="shared", value=0.0, frame=0, iteration=1, valid=False),
        reduce_fx=sum_in_order,
    )

    collector.write_metrics_to_yaml(str(tmp_path))

    metrics_file = tmp_path / "metrics.yaml"
    assert metrics_file.is_file()
    content = yaml.safe_load(metrics_file.read_text())
    aggregated = content["aggregated_metrics"]

    assert aggregated["test/psnr"]["value"] == pytest.approx(sum(camera_values) / len(camera_values))
    assert aggregated["test/psnr"]["aggregation_method"] == "mean"
    assert aggregated["test/psnr_min"]["value"] == pytest.approx(min(camera_values))
    assert aggregated["test/psnr_max"]["value"] == pytest.approx(max(camera_values))
    assert "test/psnr_std" in aggregated
    assert "test/psnr_var" in aggregated
    assert aggregated["test/chamfer_distance"]["value"] == pytest.approx(15.0)
    assert aggregated["test/elapsed_time_s"]["value"] == pytest.approx(1.5)
    assert aggregated["shared"] == {"value": 10.0, "aggregation_method": "sum_in_order"}


def test_write_metrics_to_yaml_writes_samples_without_aggregates(tmp_path: Path, monkeypatch) -> None:
    """YAML and per-sample parquet do not require aggregation from a caller."""

    content = make_metrics_content()
    collector = MetricsCollector(
        train_config_name=content.run_info.train_config_name,
        mode="val",
        run_id=content.run_info.run_id,
        version=content.program_version,
    )
    collector._samples = content.metrics
    monkeypatch.setattr(collector, "aggregate_metrics", lambda _: None)

    collector.write_metrics_to_yaml(str(tmp_path), write_parquet=True)

    assert (tmp_path / "metrics.parquet").is_file()
    assert (tmp_path / "metrics.yaml").is_file()
    assert not (tmp_path / "aggregated_metrics.parquet").exists()


@pytest.mark.parametrize("device_index, expected_index", [(None, 3), (1, 1)])
def test_hardware_info_uses_selected_cuda_device(monkeypatch, device_index, expected_index) -> None:
    properties = SimpleNamespace(
        name="Test GPU", major=8, minor=9, total_memory=24 * 1024**3, multi_processor_count=128
    )
    current_device = Mock(return_value=3)
    get_properties = Mock(return_value=properties)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_properties)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_86", "compute_90"])
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch._C, "_cuda_getDriverVersion", lambda: 12090, raising=False)

    hardware = metrics_module.collect_hardware_info(device_index)

    assert hardware.model_dump() == {
        "torch_cuda_available": True,
        "gpu_device_index": expected_index,
        "gpu_name": "Test GPU",
        "gpu_compute_capability": "8.9",
        "gpu_total_memory_bytes": 24 * 1024**3,
        "gpu_multiprocessor_count": 128,
        "compiled_architectures": ["sm_80", "sm_86", "compute_90"],
        "cuda_runtime_version": "12.8",
        "cuda_driver_version": "12.9",
        "torch_version": torch.__version__,
        "cpu": None,
        "os": None,
    }
    get_properties.assert_called_once_with(expected_index)
    if device_index is None:
        current_device.assert_called_once_with()
    else:
        current_device.assert_not_called()


@pytest.mark.parametrize("cuda_runtime", [None, "12.8"])
def test_hardware_info_without_gpu_keeps_build_metadata(monkeypatch, cuda_runtime) -> None:
    current_device = Mock(side_effect=AssertionError("No CUDA device should be selected"))
    driver_version = Mock(side_effect=AssertionError("No driver query is needed"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: [])
    monkeypatch.setattr(torch.version, "cuda", cuda_runtime)
    monkeypatch.setattr(metrics_module, "collect_cuda_driver_version", driver_version)

    hardware = metrics_module.collect_hardware_info()

    assert hardware.torch_cuda_available is False
    assert hardware.gpu_device_index is None
    assert hardware.gpu_name is None
    assert hardware.compiled_architectures == []
    assert hardware.cuda_runtime_version == cuda_runtime
    assert hardware.cuda_driver_version is None
    assert hardware.torch_version == torch.__version__
    current_device.assert_not_called()
    driver_version.assert_not_called()


def test_hardware_query_failure_preserves_available_build_metadata(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", Mock(side_effect=RuntimeError("Device unavailable")))
    monkeypatch.setattr(torch.cuda, "get_arch_list", Mock(side_effect=RuntimeError("Architectures unavailable")))
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(metrics_module, "collect_cuda_driver_version", lambda: None)

    hardware = metrics_module.collect_hardware_info()

    assert hardware.torch_cuda_available is True
    assert hardware.gpu_device_index == 2
    assert hardware.gpu_name is None
    assert hardware.compiled_architectures == []
    assert hardware.cuda_runtime_version == "12.8"
    assert hardware.cuda_driver_version is None
    assert hardware.torch_version == torch.__version__


@pytest.mark.parametrize("driver_result, expected_version", [(0, "12.9"), (999, None)])
def test_cuda_driver_version_falls_back_to_native_api(monkeypatch, driver_result, expected_version) -> None:
    monkeypatch.delattr(torch._C, "_cuda_getDriverVersion", raising=False)

    def get_version(pointer):
        metrics_module.ctypes.cast(
            pointer, metrics_module.ctypes.POINTER(metrics_module.ctypes.c_int)
        ).contents.value = 12090
        return driver_result

    driver = SimpleNamespace(cuDriverGetVersion=Mock(side_effect=get_version))
    load_driver = Mock(return_value=driver)
    monkeypatch.setattr(metrics_module.ctypes, "CDLL", load_driver)

    assert metrics_module.collect_cuda_driver_version() == expected_version
    load_driver.assert_called_once_with("nvcuda.dll" if metrics_module.os.name == "nt" else "libcuda.so.1")
    driver.cuDriverGetVersion.assert_called_once()


def test_missing_cuda_driver_does_not_fail_metadata_collection(monkeypatch) -> None:
    monkeypatch.delattr(torch._C, "_cuda_getDriverVersion", raising=False)
    monkeypatch.setattr(metrics_module.ctypes, "CDLL", Mock(side_effect=OSError("Driver unavailable")))

    assert metrics_module.collect_cuda_driver_version() is None


def test_written_yaml_carries_timing_context(tmp_path: Path, monkeypatch) -> None:
    """A timing metric is only interpretable with its method and the hardware it ran on."""

    content = make_metrics_content()
    collector = MetricsCollector(
        train_config_name=content.run_info.train_config_name,
        mode="render",
        run_id=content.run_info.run_id,
        version=content.program_version,
    )
    collector._samples = content.metrics
    collector.set_metric_metadata("render_time_ms", timing_method="wall_clock", renderer="gsplat")
    collector.set_metric_metadata("render_time_ms", resolution=[3840, 2160])
    monkeypatch.setattr(collector, "aggregate_metrics", lambda _: None)

    collector.write_metrics_to_yaml(str(tmp_path))

    written = yaml.safe_load((tmp_path / "metrics.yaml").read_text())
    assert written["schema_version"] == 5
    assert written["run_info"]["mode"] == "render"
    # set_metric_metadata merges rather than replacing, so a later call keeps the earlier fields.
    assert written["metric_metadata"]["render_time_ms"] == {
        "timing_method": "wall_clock",
        "renderer": "gsplat",
        "resolution": [3840, 2160],
    }
    # Populated only where a GPU is visible; the key is always present so consumers can rely on it.
    assert "hardware" in written
    assert written["hardware"]["torch_version"] == torch.__version__


@pytest.mark.parametrize("torch_cuda_available", [True, False, None])
def test_schema_v5_yaml_still_validates_for_kratos(
    tmp_path: Path, monkeypatch, torch_cuda_available: bool | None
) -> None:
    """The Kratos uploader parses metrics.yaml through MetricsFileContent; v5 must survive that."""

    monkeypatch.setattr(
        metrics_module, "collect_hardware_info", lambda: HardwareInfo(torch_cuda_available=torch_cuda_available)
    )
    content = make_metrics_content()
    collector = MetricsCollector(
        train_config_name=content.run_info.train_config_name,
        mode="render",
        run_id=content.run_info.run_id,
        version=content.program_version,
    )
    collector._samples = content.metrics
    collector.set_metric_metadata("render_time_ms", timing_method="wall_clock")
    monkeypatch.setattr(collector, "aggregate_metrics", lambda _: content.aggregated_metrics)

    collector.write_metrics_to_yaml(str(tmp_path))

    written = yaml.safe_load((tmp_path / "metrics.yaml").read_text())
    assert written["hardware"]["torch_cuda_available"] is torch_cuda_available
    assert "cuda_available" not in written["hardware"]
    # Serialized YAML must validate as MetricsFileContent and support metric row conversion.
    reparsed = MetricsFileContent.model_validate(written)
    assert reparsed.schema_version == 5
    assert reparsed.hardware is not None
    assert reparsed.hardware.torch_cuda_available is torch_cuda_available
    assert reparsed.metric_metadata["render_time_ms"]["timing_method"] == "wall_clock"
    assert metric_rows_from_content(reparsed, reporting_date=REPORTING_DATE)
    assert aggregated_metric_rows_from_content(reparsed, reporting_date=REPORTING_DATE)


@pytest.mark.parametrize("sensor_types", [(), (False,), (True,), (False, True)])
def test_profile_yaml_is_additive_and_keeps_kratos_input_compatible(
    collector: MetricsCollector, tmp_path: Path, sensor_types: tuple[bool, ...]
) -> None:
    # General and ordinary sensor metrics must survive even when profiling summaries are enabled.
    collector.save_metric(MetricSample(name="setup_ms", value=7.0))
    expected: list[tuple[str | None, str | None, str, float]] = [(None, None, "setup_ms", 7.0)]
    for is_lidar in sensor_types:
        sensor = "shared-name"
        camera_id, lidar_id = (None, sensor) if is_lidar else (sensor, None)
        collector.save_metric(
            MetricSample(name="quality", value=0.5),
            sequence_id="sequence",
            unique_sensor_id=sensor,
            is_lidar=is_lidar,
        )
        expected.append((camera_id, lidar_id, "quality", 0.5))
        for iteration in (1, 2):
            value = float(iteration * (10 if is_lidar else 2))
            collector.record_frame(
                sequence_id="sequence",
                sensor_id=sensor,
                frame=4,
                iteration=iteration,
                is_lidar=is_lidar,
                cuda_ms=value,
            )
            expected.append((camera_id, lidar_id, "cuda_ms", value))
    collector.set_metric_metadata("cuda_ms", warmup_frames=3, iterations=2, trajectory_id=0)
    report = write_report(collector, tmp_path)

    # This is the same YAML -> model validation boundary used by both Kratos readers.
    reparsed = MetricsFileContent.model_validate(report)
    assert reparsed.schema_version == 5
    assert "schema" not in report
    assert reparsed.metric_metadata["cuda_ms"]["warmup_frames"] == 3
    rows = metric_rows_from_content(reparsed, reporting_date=REPORTING_DATE)
    assert sorted(((row.camera_id, row.lidar_id, row.metric_name, row.value) for row in rows), key=str) == sorted(
        expected, key=str
    )

    if not sensor_types:
        assert report["metrics"]["per_sequence"] == {}
        return
    sensors = report["metrics"]["per_sequence"]["sequence"]
    # The pre-profiling sample union ignores extra frame/iteration fields while retaining values.
    legacy_parser = TypeAdapter(dict[str, dict[str, list[MetricSample | StepwiseMetricSample]]])
    for group, is_lidar in (("per_camera", False), ("per_lidar", True)):
        legacy_sensors = legacy_parser.validate_python(sensors[group])
        if is_lidar not in sensor_types:
            assert legacy_sensors == {}
            continue
        current_samples = getattr(reparsed.metrics.per_sequence["sequence"], group)["shared-name"]["cuda_ms"]
        assert [
            (sample.frame, sample.iteration) for sample in current_samples if isinstance(sample, FrameMetricSample)
        ] == [(4, 1), (4, 2)]
        legacy_samples = legacy_sensors["shared-name"]["cuda_ms"]
        assert [sample.value for sample in legacy_samples] == [sample.value for sample in current_samples]
        assert all(type(sample) is MetricSample for sample in legacy_samples)


def test_pre_v4_yaml_still_validates(tmp_path: Path) -> None:
    """Files written before the new fields existed keep parsing, so old artifacts stay uploadable."""

    legacy = make_metrics_content().model_dump()
    legacy["schema_version"] = 2
    del legacy["metric_metadata"]
    del legacy["hardware"]
    del legacy["profile_type"]

    reparsed = MetricsFileContent.model_validate(legacy)
    assert reparsed.schema_version == 2
    assert reparsed.metric_metadata == {}
    assert reparsed.hardware is None


def test_metrics_json_schema_declares_optional_torch_cuda_availability() -> None:
    hardware_schema = MetricsFileContent.model_json_schema()["$defs"]["HardwareInfo"]
    properties = hardware_schema["properties"]
    availability = properties["torch_cuda_available"]
    assert "cuda_available" not in properties
    assert "torch_cuda_available" not in hardware_schema.get("required", [])
    assert availability["default"] is None
    assert {variant["type"] for variant in availability["anyOf"]} == {"boolean", "null"}


def test_schema_v4_yaml_without_profile_fields_still_validates() -> None:
    """Existing artifacts may omit the profiling metadata declared in schema v5."""

    legacy = make_metrics_content().model_dump()
    legacy["schema_version"] = 4
    del legacy["profile_type"]
    legacy["hardware"] = {"gpu_name": "Test GPU", "torch_version": "2.7.0"}

    reparsed = MetricsFileContent.model_validate(legacy)
    assert reparsed.schema_version == 4
    assert reparsed.profile_type is None
    assert reparsed.hardware is not None
    assert reparsed.hardware.gpu_name == "Test GPU"
    assert reparsed.hardware.torch_cuda_available is None
    assert reparsed.hardware.torch_version == "2.7.0"
    assert reparsed.hardware.cpu is None
    assert reparsed.hardware.os is None
    assert reparsed.metrics == make_metrics_content().metrics


def test_write_metrics_to_yaml_uses_one_reporting_date_for_both_parquets(tmp_path: Path, monkeypatch) -> None:
    """Sample and aggregate artifacts from one validation share a reporting date."""

    content = make_metrics_content()
    collector = MetricsCollector(
        train_config_name=content.run_info.train_config_name,
        mode="val",
        run_id=content.run_info.run_id,
        version=content.program_version,
    )
    collector._samples = content.metrics
    monkeypatch.setattr(collector, "aggregate_metrics", lambda _: content.aggregated_metrics)

    collector.write_metrics_to_yaml(str(tmp_path), write_parquet=True)

    sample_dates = {row["reporting_date"] for row in pq.read_table(tmp_path / "metrics.parquet").to_pylist()}
    aggregate_dates = {
        row["reporting_date"] for row in pq.read_table(tmp_path / "aggregated_metrics.parquet").to_pylist()
    }
    assert (tmp_path / "metrics.yaml").is_file()
    assert len(sample_dates) == 1
    assert sample_dates == aggregate_dates


@pytest.fixture
def collector(monkeypatch: pytest.MonkeyPatch) -> MetricsCollector:
    version = Version(
        version_major=26,
        version_minor=9,
        version_patch=0,
        git_commit_sha_short="abc1234",
        git_commit_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(metrics_module, "__version__", version)
    monkeypatch.setattr("nurec.utils.metrics.get_version", lambda **kwargs: version)
    monkeypatch.setattr("nurec.utils.metrics.collect_hardware_info", lambda: HardwareInfo(gpu_name="Test GPU"))
    monkeypatch.setattr("nurec.utils.metrics.torch.cuda.is_available", lambda: False)
    return MetricsCollector(mode="render", frame_metrics=True, run_id="test-run")


def write_report(collector: MetricsCollector, output: Path) -> dict:
    collector.write_metrics_to_yaml(str(output))
    return yaml.safe_load((output / "metrics.yaml").read_text())


def test_collector_reuses_loaded_version_without_querying_source_control(
    collector: MetricsCollector,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    query_version = Mock(side_effect=AssertionError("Version was already loaded at startup"))
    monkeypatch.setattr("nurec.utils.metrics.get_version", query_version)

    measured = MetricsCollector(mode="render", frame_metrics=True, run_id="version-test")
    report = write_report(measured, tmp_path)

    query_version.assert_not_called()
    assert report["program_version"]["git_commit_sha_short"] == "abc1234"
    assert report["program_version"]["version_major"] == 26
    assert report["hardware"]["gpu_name"] == "Test GPU"


@pytest.mark.parametrize(
    ("is_lidar", "sensor_id", "group", "opposite_group", "modality"),
    [
        (False, "front", "per_camera", "per_lidar", "camera"),
        (True, "pandar128", "per_lidar", "per_camera", "lidar"),
    ],
)
def test_frame_samples_keep_distinct_clock_extrema(
    collector: MetricsCollector,
    tmp_path: Path,
    is_lidar: bool,
    sensor_id: str,
    group: str,
    opposite_group: str,
    modality: str,
) -> None:
    for frame, iteration, wall, cuda in ((8, 1, 30.0, 1.0), (8, 2, 10.0, 3.0), (12, 1, 20.0, 2.0)):
        collector.record_frame(
            sequence_id="sequence",
            sensor_id=sensor_id,
            is_lidar=is_lidar,
            frame=frame,
            iteration=iteration,
            render_time_ms=wall,
            cuda_ms=cuda,
        )
    for metric in ("render_time_ms", "cuda_ms"):
        collector.set_metric_metadata(metric, warmup_frames=3, iterations=2, renderer="test")

    report = write_report(collector, tmp_path)
    assert "schema" not in report
    assert report["schema_version"] == 5
    assert report["hardware"]["gpu_name"] == "Test GPU"
    assert report["program_version"]["git_commit_sha_short"] == "abc1234"
    assert report["run_info"]["mode"] == "render"
    assert report["run_info"]["run_id"] == "test-run"
    sensors = report["metrics"]["per_sequence"]["sequence"]
    assert sensors[opposite_group] == {}
    assert report["metrics"]["general"] == {}
    assert sensors[group][sensor_id] == {
        "render_time_ms": [
            {"name": "render_time_ms", "value": 30.0, "frame": 8, "iteration": 1, "valid": True},
            {"name": "render_time_ms", "value": 10.0, "frame": 8, "iteration": 2, "valid": True},
            {"name": "render_time_ms", "value": 20.0, "frame": 12, "iteration": 1, "valid": True},
        ],
        "cuda_ms": [
            {"name": "cuda_ms", "value": 1.0, "frame": 8, "iteration": 1, "valid": True},
            {"name": "cuda_ms", "value": 3.0, "frame": 8, "iteration": 2, "valid": True},
            {"name": "cuda_ms", "value": 2.0, "frame": 12, "iteration": 1, "valid": True},
        ],
    }
    for metric, mean, std, minimum, maximum in (
        ("render_time_ms", 20.0, (200.0 / 3) ** 0.5, (10.0, 2), (30.0, 1)),
        ("cuda_ms", 2.0, (2.0 / 3) ** 0.5, (1.0, 1), (3.0, 2)),
    ):
        metadata = report["metric_metadata"][metric]
        assert metadata["warmup_frames"] == 3
        assert metadata["iterations"] == 2
        assert metadata["renderer"] == "test"
        per_sensor = metadata["per_sequence"]["sequence"][group][sensor_id]
        for summary in (metadata, per_sensor):
            assert summary["mean"] == mean
            assert summary["std"] == pytest.approx(std)
            assert "avg" not in summary
        assert report["aggregated_metrics"][metric]["value"] == mean
        for name, (value, iteration) in (("min", minimum), ("max", maximum)):
            assert metadata[name] == {
                "value": value,
                "frame": 8,
                "iteration": iteration,
                modality: sensor_id,
                "sequence": "sequence",
            }
            assert per_sensor[name] == {"value": value, "frame": 8, "iteration": iteration}


@pytest.mark.parametrize("is_lidar", [False, True])
@pytest.mark.parametrize("cuda_values", [(2.0, 0.0, 6.0), (0.0, 0.0, 0.0)])
def test_invalid_cuda_samples_preserve_alignment_without_affecting_statistics(
    collector: MetricsCollector, tmp_path: Path, is_lidar: bool, cuda_values: tuple[float, ...]
) -> None:
    """YAML, telemetry rows and Parquet preserve the same CUDA values, validity and frame identity."""
    for iteration, cuda_ms in enumerate(cuda_values, 1):
        collector.record_frame(
            sequence_id="sequence",
            sensor_id="front",
            frame=4,
            iteration=iteration,
            is_lidar=is_lidar,
            metric_validity={"cuda_ms": cuda_ms > 0},
            render_time_ms=iteration * 4.0,
            cuda_ms=cuda_ms,
        )
    collector.set_metric_metadata("cuda_ms", warmup_frames=1)

    report = write_report(collector, tmp_path)
    group = "per_lidar" if is_lidar else "per_camera"
    samples = report["metrics"]["per_sequence"]["sequence"][group]["front"]
    assert len(samples["cuda_ms"]) == len(samples["render_time_ms"]) == 3
    assert [(row["frame"], row["iteration"]) for row in samples["cuda_ms"]] == [
        (row["frame"], row["iteration"]) for row in samples["render_time_ms"]
    ]
    assert [(row["value"], row["valid"]) for row in samples["cuda_ms"]] == [
        (cuda_ms, cuda_ms > 0) for cuda_ms in cuda_values
    ]
    assert [(row["value"], row["valid"]) for row in samples["render_time_ms"]] == [
        (4.0, True),
        (8.0, True),
        (12.0, True),
    ]
    assert report["aggregated_metrics"]["render_time_ms"]["value"] == 8.0
    assert report["metric_metadata"]["render_time_ms"]["mean"] == 8.0

    valid_cuda = [value for value in cuda_values if value > 0]
    if valid_cuda:
        for name, value in (
            ("cuda_ms", 4.0),
            ("cuda_ms_min", 2.0),
            ("cuda_ms_max", 6.0),
            ("cuda_ms_std", 2.0),
            ("cuda_ms_var", 4.0),
        ):
            assert report["aggregated_metrics"][name]["value"] == value
        metadata = report["metric_metadata"]["cuda_ms"]
        for summary in (metadata, metadata["per_sequence"]["sequence"][group]["front"]):
            assert summary["mean"] == 4.0
            assert summary["std"] == 2.0
            assert summary["min"]["value"] == 2.0
            assert summary["min"]["iteration"] == 1
            assert summary["max"]["value"] == 6.0
            assert summary["max"]["iteration"] == 3
    else:
        assert not any(name.startswith("cuda_ms") for name in report["aggregated_metrics"])
        assert report["metric_metadata"]["cuda_ms"] == {"warmup_frames": 1}

    reparsed = MetricsFileContent.model_validate(report)
    assert reparsed.model_dump() == report
    reparsed_samples = getattr(reparsed.metrics.per_sequence["sequence"], group)["front"]["cuda_ms"]
    assert all(isinstance(sample, FrameMetricSample) for sample in reparsed_samples)
    assert [(sample.value, sample.valid) for sample in reparsed_samples if isinstance(sample, FrameMetricSample)] == [
        (cuda_ms, cuda_ms > 0) for cuda_ms in cuda_values
    ]
    rows = metric_rows_from_content(reparsed, reporting_date=REPORTING_DATE)
    assert len(rows) == 6
    assert [row.value for row in rows if row.metric_name == "cuda_ms"] == list(cuda_values)
    assert [row.valid for row in rows if row.metric_name == "cuda_ms"] == [value > 0 for value in cuda_values]
    assert [row.value for row in rows if row.metric_name == "render_time_ms"] == [4.0, 8.0, 12.0]
    camera_id, lidar_id = (None, "front") if is_lidar else ("front", None)
    yaml_samples = [
        (
            "sequence",
            camera_id,
            lidar_id,
            metric,
            sample["frame"],
            sample["iteration"],
            sample["value"],
            sample["valid"],
        )
        for metric, metric_samples in samples.items()
        for sample in metric_samples
    ]
    # These are the exact dictionaries sent by the Kratos uploader.
    telemetry_rows = [row.to_dict() for row in rows]
    columns = ("sequence_id", "camera_id", "lidar_id", "metric_name", "frame", "iteration", "value", "valid")
    assert [tuple(row[column] for column in columns) for row in telemetry_rows] == yaml_samples

    output_path = write_metrics_to_parquet(reparsed, tmp_path / "metrics.parquet", REPORTING_DATE)
    parquet_table = pq.read_table(output_path)
    parquet_rows = parquet_table.to_pylist()
    assert parquet_table.column_names == MetricRow.column_names()
    assert len(parquet_rows) == 6
    assert [tuple(row[column] for column in columns) for row in parquet_rows] == yaml_samples
    assert parquet_rows == telemetry_rows


def test_frame_sample_validity_defaults_true_when_absent(collector: MetricsCollector, tmp_path: Path) -> None:
    collector.record_frame(sequence_id="sequence", sensor_id="front", frame=4, iteration=1, cuda_ms=2.0)
    report = write_report(collector, tmp_path)
    del report["metrics"]["per_sequence"]["sequence"]["per_camera"]["front"]["cuda_ms"][0]["valid"]

    reparsed = MetricsFileContent.model_validate(report)
    sample = reparsed.metrics.per_sequence["sequence"].per_camera["front"]["cuda_ms"][0]
    assert isinstance(sample, FrameMetricSample)
    assert sample.valid is True
    assert sample.value == 2.0
    row = metric_rows_from_content(reparsed, reporting_date=REPORTING_DATE)[0]
    assert (row.frame, row.iteration, row.value, row.valid) == (4, 1, 2.0, True)


def test_summaries_distinguish_cameras_and_sequences(collector: MetricsCollector, tmp_path: Path) -> None:
    for sequence, camera, values in (
        ("first", "front", (1.0, 3.0)),
        ("first", "back", (10.0,)),
        ("second", "front", (30.0,)),
    ):
        for iteration, value in enumerate(values, 1):
            collector.record_frame(
                sequence_id=sequence,
                sensor_id=camera,
                frame=0,
                iteration=iteration,
                render_time_ms=value * 2,
                cuda_ms=value,
            )
    report = write_report(collector, tmp_path)
    for metric, scale in (("cuda_ms", 1), ("render_time_ms", 2)):
        metadata = report["metric_metadata"][metric]
        assert metadata["mean"] == 11.0 * scale
        assert metadata["std"] == pytest.approx(131.5**0.5 * scale)
        assert metadata["min"]["camera"] == "front"
        assert metadata["min"]["sequence"] == "first"
        assert metadata["max"]["camera"] == "front"
        assert metadata["max"]["sequence"] == "second"
        for sequence, camera, mean, std in (
            ("first", "front", 2.0, 1.0),
            ("first", "back", 10.0, 0.0),
            ("second", "front", 30.0, 0.0),
        ):
            summary = metadata["per_sequence"][sequence]["per_camera"][camera]
            assert summary["mean"] == mean * scale
            assert summary["std"] == std * scale
    assert len(report["metrics"]["per_sequence"]["first"]["per_camera"]["front"]["cuda_ms"]) == 2


def test_tied_extrema_keep_first_measured_iteration(collector: MetricsCollector, tmp_path: Path) -> None:
    for frame, iteration in ((9, 2), (0, 1)):
        collector.record_frame(
            sequence_id="sequence",
            sensor_id="front",
            frame=frame,
            iteration=iteration,
            render_time_ms=5.0,
            cuda_ms=3.0,
        )
    report = write_report(collector, tmp_path)
    for metric, mean in (("render_time_ms", 5.0), ("cuda_ms", 3.0)):
        metadata = report["metric_metadata"][metric]
        for summary in (metadata, metadata["per_sequence"]["sequence"]["per_camera"]["front"]):
            assert summary["mean"] == mean
            assert summary["std"] == 0.0
        for name in ("min", "max"):
            assert metadata[name]["frame"] == 9
            assert metadata[name]["iteration"] == 2


def test_empty_report_has_no_fabricated_summary(collector: MetricsCollector, tmp_path: Path) -> None:
    collector.set_metric_metadata("cuda_ms", warmup_frames=1)
    report = write_report(collector, tmp_path)
    assert report["metrics"] == {"general": {}, "per_sequence": {}}
    assert report["aggregated_metrics"] == {}
    assert report["metric_metadata"] == {"cuda_ms": {"warmup_frames": 1}}


def test_reset_clears_both_frame_rows_and_native_samples(collector: MetricsCollector, tmp_path: Path) -> None:
    collector.record_frame(
        sequence_id="sequence", sensor_id="front", frame=4, iteration=1, render_time_ms=5.0, cuda_ms=3.0
    )
    collector.reset()
    report = write_report(collector, tmp_path)
    assert report["metrics"] == {"general": {}, "per_sequence": {}}
    assert report["aggregated_metrics"] == {}
    assert report["metric_metadata"] == {}


def test_default_formatter_keeps_shared_schema(collector: MetricsCollector, tmp_path: Path) -> None:
    native = MetricsCollector(run_id="native")
    content = MetricsFileContent(
        aggregated_metrics={},
        metrics=MetricsSamples(general={}, per_sequence={}),
        program_version=Version(version_major=26, version_minor=9, version_patch=0),
        run_info=MetricsFileRunInfo(train_config_name="config", mode="val", run_id="native"),
        metric_metadata={"test": {"unit": "ms"}},
        hardware=HardwareInfo(gpu_name="Test GPU"),
    )
    assert native.format_metrics_content(content) == content.model_dump()
    native.save_metric(create_metric_sample("test", 2.0))
    report = write_report(native, tmp_path)
    assert report["schema_version"] == 5
    assert "schema" not in report
    assert report["metrics"]["general"]["test"] == [{"name": "test", "value": 2.0}]


def test_nsight_report_identifies_instrumented_run(collector: MetricsCollector, tmp_path: Path) -> None:
    measured = MetricsCollector(mode="render", frame_metrics=True, run_id="nsight-test", nsight=True)
    measured.record_frame(
        sequence_id="sequence", sensor_id="front", frame=0, iteration=1, render_time_ms=5.0, cuda_ms=3.0
    )
    report = write_report(measured, tmp_path)
    assert report["profile_type"] == "nsight"
    assert report["program_version"] == measured._version.model_dump()
    assert report["run_info"] == {"train_config_name": "unknown", "mode": "render", "run_id": "nsight-test"}
    assert {"aggregated_metrics", "metrics", "metric_metadata", "hardware", "schema_version"} <= report.keys()
    assert "nsight_captures" not in report
    assert "nsight_trace" not in report
    assert report["metrics"]["per_sequence"]["sequence"]["per_camera"]["front"]["cuda_ms"][0]["value"] == 3.0


@pytest.mark.parametrize("nsight", [False, True])
def test_profile_yaml_preserves_declared_metadata_when_round_tripped(
    collector: MetricsCollector,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    nsight: bool,
) -> None:
    """Parsing and saving a profiling report must retain host details and its profile marker."""

    host = {
        "cpu": {"model": "Test CPU", "logical_processors": 16, "available_processors": 4},
        "os": {
            "system": "Linux",
            "is_wsl": True,
            "distribution": {"id": "ubuntu", "version_id": "24.04", "pretty_name": None},
        },
    }
    monkeypatch.setattr(metrics_module, "collect_host_info", lambda: host)
    measured = MetricsCollector(mode="render", frame_metrics=True, run_id="round-trip", nsight=nsight)
    measured.record_frame(sequence_id="sequence", sensor_id="front", frame=0, iteration=1, cuda_ms=3.0)
    report = write_report(measured, tmp_path)

    reparsed = MetricsFileContent.model_validate(report)
    saved_again = reparsed.model_dump()

    assert report["schema_version"] == 5
    assert reparsed.schema_version == 5
    assert reparsed.hardware is not None
    assert reparsed.hardware.cpu == host["cpu"]
    assert reparsed.hardware.os == host["os"]
    assert reparsed.hardware.gpu_name == "Test GPU"
    assert reparsed.profile_type == ("nsight" if nsight else None)
    assert saved_again["hardware"]["cpu"] == host["cpu"]
    assert saved_again["hardware"]["os"] == host["os"]
    assert saved_again["profile_type"] == ("nsight" if nsight else None)
    assert saved_again == report


def test_host_info_identifies_wsl_distribution_and_cpu_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(metrics_module.platform, "release", lambda: "6.6.87.2-microsoft-standard-WSL2")
    monkeypatch.setattr(metrics_module.platform, "version", lambda: "#1 SMP WSL2")
    monkeypatch.setattr(metrics_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(metrics_module.platform, "processor", lambda: "fallback CPU")
    monkeypatch.setattr(
        metrics_module.Path, "read_text", lambda *args, **kwargs: "processor: 0\nmodel name: Test CPU\n"
    )
    monkeypatch.setattr(
        metrics_module.platform,
        "freedesktop_os_release",
        lambda: {"ID": "ubuntu", "NAME": "Ubuntu", "VERSION_ID": "24.04", "PRETTY_NAME": "Ubuntu 24.04.3 LTS"},
    )
    monkeypatch.setattr(metrics_module.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(metrics_module.os, "sched_getaffinity", lambda pid: {2, 3, 4, 5}, raising=False)

    result = metrics_module.collect_host_info()

    assert result["cpu"] == {
        "model": "Test CPU",
        "architecture": "x86_64",
        "logical_processors": 16,
        "available_processors": 4,
    }
    assert result["os"] == {
        "system": "Linux",
        "release": "6.6.87.2-microsoft-standard-WSL2",
        "version": "#1 SMP WSL2",
        "machine": "x86_64",
        "is_wsl": True,
        "distribution": {"id": "ubuntu", "name": "Ubuntu", "version_id": "24.04", "pretty_name": "Ubuntu 24.04.3 LTS"},
    }


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_host_info_uses_fallbacks_when_platform_details_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
) -> None:
    monkeypatch.setattr(metrics_module.platform, "system", lambda: system)
    monkeypatch.setattr(metrics_module.platform, "release", lambda: "test-release")
    monkeypatch.setattr(metrics_module.platform, "version", lambda: "test-version")
    monkeypatch.setattr(metrics_module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(metrics_module.platform, "processor", lambda: "fallback CPU")
    cpuinfo = Mock(side_effect=OSError("not available"))
    distribution = Mock(side_effect=OSError("not available"))
    monkeypatch.setattr(metrics_module.Path, "read_text", cpuinfo)
    monkeypatch.setattr(metrics_module.platform, "freedesktop_os_release", distribution)
    monkeypatch.setattr(metrics_module.os, "cpu_count", lambda: 8)
    if system == "Linux":
        monkeypatch.setattr(
            metrics_module.os, "sched_getaffinity", Mock(side_effect=OSError("not available")), raising=False
        )
    else:
        monkeypatch.delattr(metrics_module.os, "sched_getaffinity", raising=False)

    result = metrics_module.collect_host_info()

    assert result["cpu"] == {
        "model": "fallback CPU",
        "architecture": "aarch64",
        "logical_processors": 8,
        "available_processors": 8,
    }
    assert result["os"]["system"] == system
    assert result["os"]["distribution"] is None
    assert result["os"]["is_wsl"] is False
    if system != "Linux":
        cpuinfo.assert_not_called()
        distribution.assert_not_called()


def test_host_info_is_collected_once_and_preserves_gpu_metadata(
    collector: MetricsCollector,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = {"cpu": {"model": "Test CPU"}, "os": {"system": "Linux"}}
    collect_host = Mock(return_value=host)
    monkeypatch.setattr(metrics_module, "collect_host_info", collect_host)
    measured = MetricsCollector(mode="render", frame_metrics=True, run_id="host-test", nsight=True)
    report = write_report(measured, tmp_path)
    write_report(measured, tmp_path)

    collect_host.assert_called_once_with()
    assert report["hardware"]["cpu"] == host["cpu"]
    assert report["hardware"]["os"] == host["os"]
    assert report["hardware"]["gpu_name"] == "Test GPU"
    assert "nsight_captures" not in report


def test_frame_metrics_accept_generic_values_and_keep_camera_and_lidar_groups(
    collector: MetricsCollector, tmp_path: Path
) -> None:
    for is_lidar, values in ((False, (2.0, 0.25)), (True, (8.0, 0.75))):
        collector.record_frame(
            sequence_id="sequence",
            sensor_id="shared-name",
            frame=5,
            iteration=2,
            is_lidar=is_lidar,
            latency_ms=values[0],
            quality=values[1],
        )
    report = write_report(collector, tmp_path)
    sensors = report["metrics"]["per_sequence"]["sequence"]
    for group, latency, quality in (("per_camera", 2.0, 0.25), ("per_lidar", 8.0, 0.75)):
        assert sensors[group]["shared-name"] == {
            "latency_ms": [{"name": "latency_ms", "value": latency, "frame": 5, "iteration": 2, "valid": True}],
            "quality": [{"name": "quality", "value": quality, "frame": 5, "iteration": 2, "valid": True}],
        }
        samples = getattr(collector._samples.per_sequence["sequence"], group)["shared-name"]
        assert all(isinstance(values[0], FrameMetricSample) for values in samples.values())
        for metric, mean in (("latency_ms", latency), ("quality", quality)):
            summary = report["metric_metadata"][metric]["per_sequence"]["sequence"][group]["shared-name"]
            assert summary["mean"] == mean
            assert summary["std"] == 0.0
    summary = report["metric_metadata"]["latency_ms"]
    assert summary["mean"] == 5.0
    assert summary["std"] == 3.0
    assert summary["min"] == {"value": 2.0, "frame": 5, "iteration": 2, "camera": "shared-name", "sequence": "sequence"}
    assert summary["max"] == {"value": 8.0, "frame": 5, "iteration": 2, "lidar": "shared-name", "sequence": "sequence"}
    assert report["metric_metadata"]["quality"]["mean"] == 0.5
    assert report["metric_metadata"]["quality"]["std"] == 0.25
    assert "cuda_ms" not in report["metric_metadata"]


def test_frame_report_uses_combined_samples_and_preserves_duplicate_identities_and_parquet(
    collector: MetricsCollector, tmp_path: Path
) -> None:
    collector.record_frame(
        sequence_id="sequence", sensor_id="front", frame=5, iteration=2, latency_ms=3.0, quality=0.25
    )
    remote = MetricsCollector(mode="render", frame_metrics=True)
    remote.record_frame(sequence_id="sequence", sensor_id="front", frame=5, iteration=2, latency_ms=9.0, quality=0.75)
    combined = collector._samples.model_copy(deep=True)
    combined_sensor = combined.per_sequence["sequence"].per_camera["front"]
    for metric, samples in remote._samples.per_sequence["sequence"].per_camera["front"].items():
        combined_sensor[metric].extend(samples)
    # Distributed serialization must preserve the new sample subtype and its identity fields.
    combined = MetricsSamples.model_validate(combined.model_dump())
    assert isinstance(combined.per_sequence["sequence"].per_camera["front"]["latency_ms"][1], FrameMetricSample)

    collector.write_metrics_outputs(
        combined,
        collector.aggregate_metrics(combined),
        str(tmp_path),
        write_parquet=True,
        reporting_date=REPORTING_DATE,
    )
    report = yaml.safe_load((tmp_path / "metrics.yaml").read_text())
    assert report["metrics"]["per_sequence"]["sequence"]["per_camera"]["front"] == {
        "latency_ms": [
            {"name": "latency_ms", "value": 3.0, "frame": 5, "iteration": 2, "valid": True},
            {"name": "latency_ms", "value": 9.0, "frame": 5, "iteration": 2, "valid": True},
        ],
        "quality": [
            {"name": "quality", "value": 0.25, "frame": 5, "iteration": 2, "valid": True},
            {"name": "quality", "value": 0.75, "frame": 5, "iteration": 2, "valid": True},
        ],
    }
    for metric, mean, std in (("latency_ms", 6.0, 3.0), ("quality", 0.5, 0.25)):
        metadata = report["metric_metadata"][metric]
        for summary in (metadata, metadata["per_sequence"]["sequence"]["per_camera"]["front"]):
            assert summary["mean"] == mean
            assert summary["std"] == std
    assert report["aggregated_metrics"]["latency_ms"]["value"] == 6.0
    assert len(collector._samples.per_sequence["sequence"].per_camera["front"]["latency_ms"]) == 1
    table = pq.read_table(tmp_path / "metrics.parquet")
    assert table.column_names == MetricRow.column_names()
    assert table.num_rows == 4
    rows = [row for row in table.to_pylist() if row["metric_name"] == "latency_ms"]
    assert [row["value"] for row in rows] == [3.0, 9.0]
    assert all(row["camera_id"] == "front" and row["lidar_id"] is None for row in rows)
    assert pq.read_table(tmp_path / "aggregated_metrics.parquet").column_names == AggregatedMetricRow.column_names()


def test_explicit_version_overrides_loaded_version_without_a_query(
    collector: MetricsCollector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = Version(version_major=27, version_minor=1, version_patch=2, git_commit_sha_short="def5678")
    query = Mock(side_effect=AssertionError("Explicit versions must not query source control"))
    monkeypatch.setattr(metrics_module, "get_version", query)
    measured = MetricsCollector(version=explicit)
    report = write_report(measured, tmp_path)
    query.assert_not_called()
    assert report["program_version"]["git_commit_sha_short"] == "def5678"
    assert report["program_version"]["version_major"] == 27
    assert report["schema_version"] == 5


def test_missing_loaded_version_uses_existing_surrogate_fallback(
    collector: MetricsCollector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fallback = Version(version_major=26, version_minor=9, version_patch=0)
    monkeypatch.setattr(metrics_module, "__version__", None)
    query = Mock(return_value=fallback)
    monkeypatch.setattr(metrics_module, "get_version", query)
    measured = MetricsCollector(mode="render", frame_metrics=True)
    report = write_report(measured, tmp_path)
    query.assert_called_once_with(allow_empty=True)
    assert report["program_version"]["version_major"] == 26
    assert report["program_version"]["version_minor"] == 9


def test_legacy_collector_reuses_loaded_version_without_collecting_frame_host_metadata(
    collector: MetricsCollector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = Mock(side_effect=AssertionError("Version was already loaded"))
    collect_host = Mock(side_effect=AssertionError("Host details require frame_metrics=True"))
    monkeypatch.setattr(metrics_module, "get_version", query)
    monkeypatch.setattr(metrics_module, "collect_host_info", collect_host)
    measured = MetricsCollector(mode="render")
    measured.save_metric(create_metric_sample("custom", 3.0))
    report = write_report(measured, tmp_path)
    query.assert_not_called()
    collect_host.assert_not_called()
    assert report["schema_version"] == 5
    assert "schema" not in report
    assert report["metrics"]["general"]["custom"] == [{"name": "custom", "value": 3.0}]
    assert report["program_version"]["git_commit_sha_short"] == "abc1234"
