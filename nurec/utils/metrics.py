# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import logging
import math
import os
import platform

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch
import yaml

from pydantic import BaseModel, Field

from nurec import __version__
from nurec.config.version import Version, get_version
from nurec.utils.batch import FrameMeta
from nurec.utils.misc import distributed_all_gather_nested, rank_zero_only, unpack_optional


logger = logging.getLogger(__name__)


class MetricsRunConfig(BaseModel):
    """Minimal config class for metrics collection, avoiding circular dependency."""

    version: Version
    train_config_name: str
    mode: str
    run_id: str


class MetricSample(BaseModel):
    """A class to represent a metric sample."""

    name: Optional[str] = Field(default=None, description="The name of the metric")
    value: float = Field(description="The value of the metric")

    @classmethod
    def from_value(cls, name: str, value: float) -> MetricSample:
        """Create a metric sample from a value."""

        return cls(name=name, value=value)


class StepwiseMetricSample(MetricSample):
    """A class to represent a stepwise metric sample."""

    timestamp_us_begin: int | None = Field(description="The optional start timestamp of the metric")
    timestamp_us_end: int | None = Field(description="The optional end timestamp of the metric")
    unique_frame_idx: int = Field(description="The unique frame index of the metric")

    @classmethod
    def from_value_and_metadata(
        cls,
        name: str,
        value: float,
        frame_meta: FrameMeta,
        timestamps_startend_us: torch.Tensor | None = None,
    ) -> StepwiseMetricSample:
        """Create a stepwise metric sample from a value and metadata."""

        timestamp_us_begin = None
        timestamp_us_end = None
        if timestamps_startend_us is not None:
            # timestamps_startend_us is (1,2) tensor; index directly
            timestamp_us_begin = int(timestamps_startend_us[0, 0].item())
            timestamp_us_end = int(timestamps_startend_us[0, 1].item())

        return cls(
            name=name,
            value=value,
            timestamp_us_begin=timestamp_us_begin,
            timestamp_us_end=timestamp_us_end,
            unique_frame_idx=frame_meta.unique_frame_idx,
        )


class FrameMetricSample(MetricSample):
    """A metric value associated with one sensor frame and measured iteration."""

    frame: int = Field(description="The source frame index within the sensor")
    iteration: int = Field(description="The measured iteration, starting at one")
    valid: bool = Field(default=True, description="Whether this measurement may contribute to timing statistics")


GenericMetricSample = Union[MetricSample, StepwiseMetricSample, FrameMetricSample]


def create_metric_sample(
    name: str,
    value: float,
    frame_meta: FrameMeta | None = None,
    timestamps_startend_us: torch.Tensor | None = None,
) -> GenericMetricSample:
    """Create a metric sample from a value and metadata."""

    if frame_meta is not None:
        return StepwiseMetricSample.from_value_and_metadata(name, value, frame_meta, timestamps_startend_us)
    else:
        return MetricSample.from_value(name, value)


class MetricAggregate(BaseModel):
    """A class to represent a metric aggregate."""

    value: float = Field(description="The value of the metric")
    aggregation_method: str = Field(description="The aggregation method of the metric")


class HardwareInfo(BaseModel):
    """The hardware a run was measured on.

    Timing metrics cannot be interpreted without it and it cannot be recovered from stored
    artifacts, so it is captured at collection time.
    """

    torch_cuda_available: Optional[bool] = Field(
        default=None,
        description="Whether CUDA was available to PyTorch in the process collecting these metrics; None if not recorded",
    )
    gpu_device_index: Optional[int] = Field(default=None, description="The selected CUDA device index")
    gpu_name: Optional[str] = Field(default=None, description="The GPU model name")
    gpu_compute_capability: Optional[str] = Field(default=None, description="The GPU compute capability")
    gpu_total_memory_bytes: Optional[int] = Field(default=None, description="The total GPU memory in bytes")
    gpu_multiprocessor_count: Optional[int] = Field(default=None, description="The GPU multiprocessor count")
    compiled_architectures: List[str] = Field(
        default_factory=list, description="CUDA architectures torch was built for"
    )
    cuda_runtime_version: Optional[str] = Field(default=None, description="The CUDA version torch was built with")
    cuda_driver_version: Optional[str] = Field(default=None, description="The CUDA API version supported by the driver")
    torch_version: Optional[str] = Field(default=None, description="The torch version")
    cpu: Optional[Dict[str, Any]] = Field(default=None, description="CPU model, architecture, and processor counts")
    os: Optional[Dict[str, Any]] = Field(default=None, description="Operating system, kernel, and distribution details")


def collect_cuda_driver_version() -> Optional[str]:
    """Read the CUDA driver's supported API version without launching a subprocess."""

    try:
        try:
            raw_version = torch._C._cuda_getDriverVersion()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            # Recent torch builds do not expose the accessor, but the native driver API is available.
            driver = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
            get_version = driver.cuDriverGetVersion
            get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
            get_version.restype = ctypes.c_int
            version = ctypes.c_int()
            if get_version(ctypes.byref(version)) != 0:
                return None
            raw_version = version.value
        return f"{raw_version // 1000}.{(raw_version % 1000) // 10}"
    except (AttributeError, OSError, RuntimeError):
        return None


def collect_hardware_info(device_index: Optional[int] = None) -> HardwareInfo:
    """Collect the selected device and build description, best effort."""

    hardware = HardwareInfo(
        torch_cuda_available=torch.cuda.is_available(),
        cuda_runtime_version=torch.version.cuda,
        torch_version=torch.__version__,
    )
    try:
        hardware.compiled_architectures = torch.cuda.get_arch_list()
    except (AttributeError, RuntimeError):
        pass
    if not hardware.torch_cuda_available:
        return hardware

    hardware.cuda_driver_version = collect_cuda_driver_version()
    try:
        hardware.gpu_device_index = torch.cuda.current_device() if device_index is None else device_index
        properties = torch.cuda.get_device_properties(hardware.gpu_device_index)
        hardware.gpu_name = properties.name
        hardware.gpu_compute_capability = f"{properties.major}.{properties.minor}"
        hardware.gpu_total_memory_bytes = properties.total_memory
        hardware.gpu_multiprocessor_count = properties.multi_processor_count
    except (AssertionError, RuntimeError, ValueError):
        # Metadata failure must not discard measurements or the build details collected above.
        pass
    return hardware


class MetricsFileRunInfo(BaseModel):
    """A class to represent the run informations."""

    train_config_name: str = Field(description="The name of the train config")
    mode: str = Field(description="The mode of the run")
    run_id: str = Field(description="The id of the run")


# Metrics grouped by sequence, sensor, metric
MetricsGeneralSampleDict = Dict[str, List[GenericMetricSample]]
MetricsPerSensorSampleDict = Dict[str, Dict[str, List[GenericMetricSample]]]
MetricsAggregateDict = Dict[str, MetricAggregate]


class MetricsSamplesPerSequencePerSensor(BaseModel):
    """A class to represent a metric samples grouped per sensor (camera or lidar)."""

    per_camera: MetricsPerSensorSampleDict = Field(description="Per camera metrics samples")
    per_lidar: MetricsPerSensorSampleDict = Field(description="Per lidar metrics samples")


MetricsSamplesPerSequenceDict = Dict[str, MetricsSamplesPerSequencePerSensor]


class MetricsSamples(BaseModel):
    """A class to represent a single value of a named performance or quality metric."""

    general: MetricsGeneralSampleDict = Field(description="General metrics samples")
    per_sequence: MetricsSamplesPerSequenceDict = Field(description="Per sequence metrics samples")


class MetricsFileContent(BaseModel):
    """A class to represent the content of a metrics file that will be saved to a file."""

    aggregated_metrics: MetricsAggregateDict = Field(description="Aggregated metrics")
    metrics: MetricsSamples = Field(description="List of metrics samples")
    program_version: Version = Field(description="Version informations of the program")
    run_info: MetricsFileRunInfo = Field(description="Run informations")
    metric_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Free-form metadata keyed by metric name, e.g. timing method, renderer, resolution",
    )
    hardware: Optional[HardwareInfo] = Field(default=None, description="Hardware the run was measured on")
    profile_type: Optional[Literal["nsight"]] = Field(
        default=None, description="Profiler instrumentation used when collecting the metrics"
    )
    # Update the schema version here when the metrics file format changes.
    # 3 is skipped: it belongs to the unrelated schema written by nurec/metrics/metric.py.
    # 5 declares profiling instrumentation, CPU/OS metadata, and nullable torch_cuda_available in the shared model.
    # Schema 5 also retains invalid CUDA frame timings with an explicit validity flag.
    schema_version: int = Field(description="Schema version of the metrics file", default=5)


@dataclass
class MetricRow:
    """Canonical normalized metric sample for Kratos telemetry and dashboard parquet."""

    sequence_id: Optional[str]
    camera_id: Optional[str]
    lidar_id: Optional[str]
    metric_name: str
    timestamp_us_begin: Optional[int]
    timestamp_us_end: Optional[int]
    unique_frame_idx: Optional[int]
    value: Optional[Union[float, int]]
    git_commit_date: Optional[str]
    git_commit_sha_short: Optional[str]
    git_tree_dirty: Optional[bool]
    version_major: Optional[int]
    version_minor: Optional[int]
    version_patch: Optional[int]
    version_string: Optional[str]
    mode: Optional[str]
    run_id: Optional[str]
    train_config_name: Optional[str]
    reporting_date: Optional[str]
    frame: int | None = None
    iteration: int | None = None
    valid: bool = True

    @classmethod
    def column_names(cls) -> list[str]:
        """Return the stable column order used by the parquet export."""

        return list(cls.__dataclass_fields__.keys())

    def to_dict(self) -> dict:
        """Convert this row into a column-name keyed dictionary."""

        return asdict(self)


@dataclass
class AggregatedMetricRow:
    """An NuRec-computed aggregate suitable for dashboard ingestion."""

    sequence_id: Optional[str]
    metric_name: str
    aggregation_method: str
    value: Optional[Union[float, int]]
    git_commit_date: Optional[str]
    git_commit_sha_short: Optional[str]
    git_tree_dirty: Optional[bool]
    version_major: Optional[int]
    version_minor: Optional[int]
    version_patch: Optional[int]
    version_string: Optional[str]
    mode: Optional[str]
    run_id: Optional[str]
    train_config_name: Optional[str]
    reporting_date: Optional[str]

    @classmethod
    def column_names(cls) -> list[str]:
        """Return the stable column order used by the aggregate parquet export."""

        return list(cls.__dataclass_fields__.keys())

    def to_dict(self) -> dict:
        """Convert this row into a column-name keyed dictionary."""

        return asdict(self)


def _normalize_metric_value(value: float | int, metric_name: str, log: logging.Logger) -> float | int | None:
    if math.isnan(value):
        log.warning("NaN value found for %s, converting to NULL", metric_name)
        return None
    if math.isinf(value):
        log.warning("Infinity value found for %s, converting to NULL", metric_name)
        return None
    return value


def metric_rows_from_content(
    metrics_content: MetricsFileContent,
    reporting_date: datetime | None = None,
    log: logging.Logger | None = None,
) -> list[MetricRow]:
    """Flatten an NuRec metrics file into the long-form telemetry row schema.

    YAML, Parquet, and the Kratos payload preserve the same frame identity, value,
    and validity. Invalid CUDA timings retain their zero placeholder and valid=False;
    consumers must exclude invalid samples when computing statistics. Legacy non-finite
    metric values still normalize to NULL.
    """

    reporting_date = reporting_date or datetime.now(timezone.utc)
    reporting_date_str = reporting_date.isoformat()
    log = log or logger
    version_info = metrics_content.program_version
    run_info = metrics_content.run_info
    rows: list[MetricRow] = []

    def append_samples(
        metric_name: str,
        samples: list[GenericMetricSample],
        sequence_id: str | None = None,
        camera_id: str | None = None,
        lidar_id: str | None = None,
    ) -> None:
        for sample in samples:
            try:
                rows.append(
                    MetricRow(
                        sequence_id=sequence_id or None,
                        camera_id=camera_id or None,
                        lidar_id=lidar_id or None,
                        metric_name=metric_name,
                        timestamp_us_begin=sample.timestamp_us_begin
                        if isinstance(sample, StepwiseMetricSample)
                        else None,
                        timestamp_us_end=sample.timestamp_us_end if isinstance(sample, StepwiseMetricSample) else None,
                        unique_frame_idx=sample.unique_frame_idx if isinstance(sample, StepwiseMetricSample) else None,
                        value=_normalize_metric_value(sample.value, metric_name, log),
                        frame=sample.frame if isinstance(sample, FrameMetricSample) else None,
                        iteration=sample.iteration if isinstance(sample, FrameMetricSample) else None,
                        valid=sample.valid if isinstance(sample, FrameMetricSample) else True,
                        git_commit_date=version_info.git_commit_date.isoformat(),
                        git_commit_sha_short=version_info.git_commit_sha_short,
                        git_tree_dirty=version_info.git_tree_dirty,
                        version_major=version_info.version_major,
                        version_minor=version_info.version_minor,
                        version_patch=version_info.version_patch,
                        version_string=version_info.version_string,
                        mode=run_info.mode,
                        run_id=run_info.run_id,
                        train_config_name=run_info.train_config_name,
                        reporting_date=reporting_date_str,
                    )
                )
            except (AttributeError, ValueError) as error:
                log.error("Error processing metric %s: %s", metric_name, error)

    for metric_name, samples in metrics_content.metrics.general.items():
        append_samples(metric_name, samples)

    for sequence_id, sensors in metrics_content.metrics.per_sequence.items():
        for camera_id, metrics in sensors.per_camera.items():
            for metric_name, samples in metrics.items():
                append_samples(metric_name, samples, sequence_id=sequence_id, camera_id=camera_id)
        for lidar_id, metrics in sensors.per_lidar.items():
            for metric_name, samples in metrics.items():
                append_samples(metric_name, samples, sequence_id=sequence_id, lidar_id=lidar_id)

    return rows


def aggregated_metric_rows_from_content(
    metrics_content: MetricsFileContent,
    reporting_date: datetime | None = None,
    log: logging.Logger | None = None,
) -> list[AggregatedMetricRow]:
    """Flatten NuRec's authoritative aggregates into dashboard-ready rows.

    Aggregates span all samples in a metrics file. When the file contains a
    single sequence, retain that sequence id; otherwise leave it null to make
    the global scope explicit.
    """

    reporting_date = reporting_date or datetime.now(timezone.utc)
    reporting_date_str = reporting_date.isoformat()
    log = log or logger
    version_info = metrics_content.program_version
    run_info = metrics_content.run_info
    sequence_ids = list(metrics_content.metrics.per_sequence)
    sequence_id = (sequence_ids[0] or None) if len(sequence_ids) == 1 else None

    return [
        AggregatedMetricRow(
            sequence_id=sequence_id,
            metric_name=metric_name,
            aggregation_method=aggregate.aggregation_method,
            value=_normalize_metric_value(aggregate.value, metric_name, log),
            git_commit_date=version_info.git_commit_date.isoformat(),
            git_commit_sha_short=version_info.git_commit_sha_short,
            git_tree_dirty=version_info.git_tree_dirty,
            version_major=version_info.version_major,
            version_minor=version_info.version_minor,
            version_patch=version_info.version_patch,
            version_string=version_info.version_string,
            mode=run_info.mode,
            run_id=run_info.run_id,
            train_config_name=run_info.train_config_name,
            reporting_date=reporting_date_str,
        )
        for metric_name, aggregate in metrics_content.aggregated_metrics.items()
    ]


def _metric_row_parquet_schema():
    """Return the fixed Arrow schema for normalized metric samples."""

    import pyarrow as pa  # type: ignore[import-untyped]

    return pa.schema(
        [
            pa.field("sequence_id", pa.string()),
            pa.field("camera_id", pa.string()),
            pa.field("lidar_id", pa.string()),
            pa.field("metric_name", pa.string()),
            pa.field("timestamp_us_begin", pa.int64()),
            pa.field("timestamp_us_end", pa.int64()),
            pa.field("unique_frame_idx", pa.int64()),
            pa.field("value", pa.float64()),
            pa.field("git_commit_date", pa.string()),
            pa.field("git_commit_sha_short", pa.string()),
            pa.field("git_tree_dirty", pa.bool_()),
            pa.field("version_major", pa.int64()),
            pa.field("version_minor", pa.int64()),
            pa.field("version_patch", pa.int64()),
            pa.field("version_string", pa.string()),
            pa.field("mode", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("train_config_name", pa.string()),
            pa.field("reporting_date", pa.string()),
            pa.field("frame", pa.int64()),
            pa.field("iteration", pa.int64()),
            pa.field("valid", pa.bool_(), nullable=False),
        ]
    )


def _aggregated_metric_row_parquet_schema():
    """Return the fixed Arrow schema for NuRec-computed aggregates."""

    import pyarrow as pa  # type: ignore[import-untyped]

    return pa.schema(
        [
            pa.field("sequence_id", pa.string()),
            pa.field("metric_name", pa.string()),
            pa.field("aggregation_method", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("git_commit_date", pa.string()),
            pa.field("git_commit_sha_short", pa.string()),
            pa.field("git_tree_dirty", pa.bool_()),
            pa.field("version_major", pa.int64()),
            pa.field("version_minor", pa.int64()),
            pa.field("version_patch", pa.int64()),
            pa.field("version_string", pa.string()),
            pa.field("mode", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("train_config_name", pa.string()),
            pa.field("reporting_date", pa.string()),
        ]
    )


def _write_rows_to_parquet(
    rows: list[MetricRow] | list[AggregatedMetricRow], output_path: str | Path, schema: Any
) -> Path:
    """Write rows using an explicit schema, including when all values are null or no rows exist."""

    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([row.to_dict() for row in rows], schema=schema)
    pq.write_table(table, output_path)
    return output_path


def write_metrics_to_parquet(
    metrics_content: MetricsFileContent,
    output_path: str | Path,
    reporting_date: datetime | None = None,
) -> Path:
    """Write NuRec metric samples to a stable long-form parquet file."""

    rows = metric_rows_from_content(metrics_content, reporting_date=reporting_date)
    output_path = _write_rows_to_parquet(rows, output_path, _metric_row_parquet_schema())
    logger.info("Metrics parquet saved to %s (%d rows)", output_path, len(rows))
    return output_path


def write_aggregated_metrics_to_parquet(
    metrics_content: MetricsFileContent,
    output_path: str | Path,
    reporting_date: datetime | None = None,
) -> Path:
    """Write NuRec-computed aggregate metrics without recomputing reductions."""

    rows = aggregated_metric_rows_from_content(metrics_content, reporting_date=reporting_date)
    output_path = _write_rows_to_parquet(rows, output_path, _aggregated_metric_row_parquet_schema())
    logger.info("Aggregated metrics parquet saved to %s (%d rows)", output_path, len(rows))
    return output_path


def collect_host_info() -> dict[str, Any]:
    """Describe the CPU and operating system once, before rendering starts."""
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    model = platform.processor() or None
    distribution = None
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    model = line.partition(":")[2].strip() or model
                    break
        except (OSError, UnicodeError):
            pass
        try:
            values = platform.freedesktop_os_release()
            distribution = {key: values.get(key.upper(), "") for key in ("id", "name", "version_id", "pretty_name")}
        except OSError:
            pass
    logical_processors = os.cpu_count()
    try:
        available_processors: int | None = len(os.sched_getaffinity(0))
    except (AttributeError, OSError, NotImplementedError):
        available_processors = logical_processors
    return {
        "cpu": {
            "model": model,
            "architecture": machine,
            "logical_processors": logical_processors,
            "available_processors": available_processors,
        },
        "os": {
            "system": system,
            "release": release,
            "version": version,
            "machine": machine,
            "distribution": distribution,
            "is_wsl": system == "Linux" and ("microsoft" in release.lower() or "wsl" in version.lower()),
        },
    }


def summarize_frame_metrics(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Summarize measured renders with population std; extrema ties keep the first.

    Rows must be nonempty, and every row must contain the requested metric.
    """
    values = [row[metric] for row in rows]
    mean = sum(values) / len(values)
    summary: dict[str, Any] = {
        "mean": mean,
        "std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
    }
    for name, reducer in (("min", min), ("max", max)):
        row = reducer(rows, key=lambda sample: sample[metric])
        summary[name] = {
            "value": row[metric],
            **{key: row[key] for key in ("frame", "iteration", "camera", "lidar", "sequence") if key in row},
        }
    return summary


class MetricsCollector:
    """A class to collect and save metrics to YAML files."""

    def __init__(
        self,
        train_config_name: str = "unknown",
        mode: Literal["train", "val", "render", "grpc_server", "unknown"] = "unknown",
        run_id: str = "unknown",
        version: Optional[Version] = None,
        *,
        frame_metrics: bool = False,
        nsight: bool = False,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            train_config_name: Name of the training configuration
            mode: Mode of the run (e.g., "train", "val", "grpc_server")
            run_id: Unique identifier for the run
            version: Explicit version; otherwise reuse NuRec's startup version, with a surrogate in build/test environments.
            frame_metrics: Add frame/iteration summaries and host details to the standard metrics report.
            nsight: Identify measurements collected under Nsight Systems instrumentation.
        """
        self._train_config_name = train_config_name
        self._mode = mode
        self._run_id = run_id
        # Avoid another source-control subprocess query after the GPU model has loaded.
        resolved_version = version if version is not None else __version__
        if resolved_version is None:
            resolved_version = unpack_optional(get_version(allow_empty=True))
        self._version = resolved_version
        self._frame_metrics = frame_metrics
        self._nsight = nsight
        self._samples: MetricsSamples = MetricsSamples(general={}, per_sequence={})
        self._reduce_functions: Dict[str, Union[str, Callable[[torch.Tensor], torch.Tensor]]] = {}
        self._metric_metadata: Dict[str, Dict[str, Any]] = {}
        self._hardware = collect_hardware_info()
        self._host_info = collect_host_info() if frame_metrics else {}

    def record_frame(
        self,
        *,
        sequence_id: str,
        sensor_id: str,
        frame: int,
        iteration: int,
        is_lidar: bool = False,
        metric_validity: Mapping[str, bool] | None = None,
        **metrics: float,
    ) -> None:
        """Keep every per-frame metric; invalid measurements remain present but do not affect statistics."""
        for name, value in metrics.items():
            self.save_metric(
                FrameMetricSample(
                    name=name,
                    value=value,
                    frame=frame,
                    iteration=iteration,
                    valid=metric_validity.get(name, True) if metric_validity is not None else True,
                ),
                unique_sensor_id=sensor_id,
                sequence_id=sequence_id,
                is_lidar=is_lidar,
            )

    def set_metric_metadata(self, name: str, **fields: Any) -> None:
        """Attach free-form metadata to a metric, merging with anything already recorded.

        Timing metrics in particular need ``timing_method`` so a wall-clock span and a CUDA-event
        span are distinguishable in the written file.
        """

        self._metric_metadata.setdefault(name, {}).update(fields)

    def reset(self) -> None:
        """Reset the metrics collector by clearing all collected metrics."""
        self._samples.general.clear()
        self._samples.per_sequence.clear()

    def save_metric(
        self,
        metric_sample: GenericMetricSample,
        unique_sensor_id: Optional[str] = None,
        sequence_id: Union[list[str], str, None] = None,
        reduce_fx: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
        is_lidar: bool = False,
    ) -> None:
        """Save a metric value for later aggregation.

        Args:
            name: Name of the metric
            value: Value of the metric
            metadata: Metadata of the metric
            sequence_id: Sequence identifier
            reduce_fx: Reduction function to apply to the metric values
        """
        if metric_sample.name is None:
            raise ValueError("Metric sample name is None")

        if sequence_id is None and unique_sensor_id is None:
            if metric_sample.name not in self._samples.general:
                self._samples.general[metric_sample.name] = []
            self._samples.general[metric_sample.name].append(metric_sample)
        else:
            if sequence_id is None:
                sequence_id = ""
            if isinstance(sequence_id, list):
                sequence_id = str.join("+", sequence_id)
            # sequence_id: str

            if sequence_id not in self._samples.per_sequence:
                self._samples.per_sequence[sequence_id] = MetricsSamplesPerSequencePerSensor(
                    per_camera={}, per_lidar={}
                )
            sensor_dict = (
                self._samples.per_sequence[sequence_id].per_camera
                if not is_lidar
                else self._samples.per_sequence[sequence_id].per_lidar
            )

            unique_sensor_id = unique_sensor_id or ""
            # Remove the sequence_id from the unique_sensor_id if present
            # e.g:
            # unique_sensor_id = camera_front_wide_120fov@3c291d58-15ad-11ed-b911-00044bf65f0e@1659807950800159-1659807956200057
            # sequence_id = 3c291d58-15ad-11ed-b911-00044bf65f0e@1659807950800159-1659807956200057
            # -> unique_sensor_id = camera_front_wide_120fov
            if (
                sequence_id is not None
                and unique_sensor_id[-len(sequence_id) :] == sequence_id
                and unique_sensor_id[-len(sequence_id) - 1] in ["@", "+", "-", "_", "|"]
            ):
                unique_sensor_id = unique_sensor_id[: -len(sequence_id) - 1]

            if unique_sensor_id not in sensor_dict:
                sensor_dict[unique_sensor_id] = {}
            if metric_sample.name not in sensor_dict[unique_sensor_id]:
                sensor_dict[unique_sensor_id][metric_sample.name] = []
            sensor_dict[unique_sensor_id][metric_sample.name].append(metric_sample)

        if metric_sample.name not in self._reduce_functions:
            self._reduce_functions[metric_sample.name] = torch.mean

        if reduce_fx is not None:
            if metric_sample.name in self._reduce_functions and self._reduce_functions[metric_sample.name] != reduce_fx:
                logger.warning(
                    f"Metric {metric_sample.name} already has a different reduce function: {self._reduce_functions[metric_sample.name]} -> {reduce_fx}, overriding"
                )
            self._reduce_functions[metric_sample.name] = reduce_fx

    def combine_samples_across_ranks(self, rank_samples: MetricsSamples) -> MetricsSamples:
        if torch.distributed.is_initialized():
            # Use distributed_all_gather_nested to directly handle the nested structure
            # only_on_rank_zero=False because we need the combined data for sorting
            combined_samples = distributed_all_gather_nested(rank_samples, only_on_rank_zero=False)

            # Sort metrics by timestamp_us_begin
            for sequence_data in combined_samples.per_sequence.values():
                for sensor_type in ["per_camera", "per_lidar"]:
                    for sensor_data in getattr(sequence_data, sensor_type).values():
                        for metric_name, metric_values in sensor_data.items():
                            if metric_values and isinstance(metric_values[0], StepwiseMetricSample):
                                sensor_data[metric_name] = sorted(metric_values, key=lambda x: x.unique_frame_idx)

            return combined_samples
        else:
            return rank_samples

    @rank_zero_only
    def aggregate_metrics(self, combined_samples: MetricsSamples) -> MetricsAggregateDict:
        # Aggregate the combined metrics
        aggregated_metrics: MetricsAggregateDict = {}

        # We allow strings or torch functions as reduce_fx to remain compatible with LightningModule.log() arguments
        resolved_reduce_functions: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for metric_name, reduce_fx in self._reduce_functions.items():
            resolved_reduce_fx: Callable[[torch.Tensor], torch.Tensor]
            if isinstance(reduce_fx, str):
                match reduce_fx:
                    case "min":
                        resolved_reduce_fx = torch.min
                    case "max":
                        resolved_reduce_fx = torch.max
                    case "mean":
                        resolved_reduce_fx = torch.mean
                    case "sum":
                        resolved_reduce_fx = torch.sum
                    case _:
                        raise ValueError(f"Unknown reduction function: {reduce_fx}")
            elif reduce_fx is not None:
                resolved_reduce_fx = reduce_fx
            else:
                resolved_reduce_fx = torch.mean

            resolved_reduce_functions[metric_name] = resolved_reduce_fx

        aggregated_metrics_values: Dict[str, List[float]] = {}

        sensor_metrics = (
            metrics
            for sequence in combined_samples.per_sequence.values()
            for metrics in chain(sequence.per_camera.values(), sequence.per_lidar.values())
        )
        for metrics in chain(sensor_metrics, (combined_samples.general,)):
            for metric_name, samples in metrics.items():
                aggregated_metrics_values.setdefault(metric_name, []).extend(
                    sample.value for sample in samples if not isinstance(sample, FrameMetricSample) or sample.valid
                )

        # Aggregation is scalar reduction math, so it does not require a GPU. Use CUDA when it is
        # available (unchanged from the historical path) but fall back to CPU otherwise, so scalar
        # metric merging (e.g. the multi-process serve metrics merge) works on a no-GPU host.
        aggregate_device = "cuda" if torch.cuda.is_available() else "cpu"

        for metric_name, values in aggregated_metrics_values.items():
            if not values:
                continue

            values_tensor = torch.tensor(values, device=aggregate_device)

            # Default aggregate using configured reduce function (backwards compatible)
            reduce_fx = (
                resolved_reduce_functions[metric_name] if metric_name in resolved_reduce_functions else torch.mean
            )
            aggregated_metrics[metric_name] = MetricAggregate(
                value=reduce_fx(values_tensor).item(),
                aggregation_method=reduce_fx.__name__,
            )

            # Additional aggregates: min, max, std, var (population stats for stability)
            aggregated_metrics[f"{metric_name}_min"] = MetricAggregate(
                value=torch.min(values_tensor).item(),
                aggregation_method="min",
            )
            aggregated_metrics[f"{metric_name}_max"] = MetricAggregate(
                value=torch.max(values_tensor).item(),
                aggregation_method="max",
            )
            aggregated_metrics[f"{metric_name}_std"] = MetricAggregate(
                value=torch.std(values_tensor, unbiased=False).item(),
                aggregation_method="std",
            )
            aggregated_metrics[f"{metric_name}_var"] = MetricAggregate(
                value=torch.var(values_tensor, unbiased=False).item(),
                aggregation_method="var",
            )

        return aggregated_metrics

    def format_metrics_content(self, content: MetricsFileContent) -> dict[str, Any]:
        """Add profiling metadata while preserving the canonical metrics schema."""
        if not self._frame_metrics:
            return content.model_dump()
        content = content.model_copy()
        hardware = content.hardware.model_dump() if content.hardware is not None else {}
        content.hardware = HardwareInfo(**{**hardware, **self._host_info})
        if self._nsight:
            content.profile_type = "nsight"
        report = content.model_dump()

        all_rows: dict[str, list[dict[str, Any]]] = {}
        # Summarize gathered samples independently; keep the original per-metric lists intact.
        for sequence, sensors in content.metrics.per_sequence.items():
            for group, sensor_key in (("per_camera", "camera"), ("per_lidar", "lidar")):
                for sensor, metrics in getattr(sensors, group).items():
                    for metric, samples in metrics.items():
                        rows = [
                            {"frame": sample.frame, "iteration": sample.iteration, metric: sample.value}
                            for sample in samples
                            if isinstance(sample, FrameMetricSample) and sample.valid
                        ]
                        if not rows:
                            continue
                        metadata = report["metric_metadata"].setdefault(metric, {})
                        summaries = metadata.setdefault("per_sequence", {})
                        summaries.setdefault(sequence, {}).setdefault(group, {})[sensor] = summarize_frame_metrics(
                            rows, metric
                        )
                        all_rows.setdefault(metric, []).extend(
                            {**row, sensor_key: sensor, "sequence": sequence} for row in rows
                        )
        for metric, rows in all_rows.items():
            report["metric_metadata"][metric].update(summarize_frame_metrics(rows, metric))
        return report

    @rank_zero_only
    def write_metrics_outputs(
        self,
        combined_samples: MetricsSamples,
        aggregated_metrics: MetricsAggregateDict | None,
        val_dir: str,
        write_parquet: bool = False,
        reporting_date: datetime | None = None,
    ) -> None:
        """Write rank-zero validation artifacts from a single metrics snapshot."""

        content = MetricsFileContent(
            aggregated_metrics=aggregated_metrics or {},
            metrics=combined_samples,
            program_version=self._version,
            run_info=MetricsFileRunInfo(
                train_config_name=self._train_config_name,
                mode=self._mode,
                run_id=self._run_id,
            ),
            metric_metadata=self._metric_metadata,
            hardware=self._hardware,
        )

        # Per-sample metrics remain useful even if a caller intentionally skips
        # aggregation. This method is rank-zero-only, so distributed runs never
        # race to write the shared artifact path.
        if write_parquet:
            write_metrics_to_parquet(content, Path(val_dir) / "metrics.parquet", reporting_date=reporting_date)

        # Write metrics to YAML file
        filename = os.path.join(val_dir, "metrics.yaml")
        with open(filename, "w") as f:
            yaml.dump(self.format_metrics_content(content), f)

        logger.info(f"Metrics saved to {filename}")

        if aggregated_metrics is None:
            return

        if write_parquet:
            write_aggregated_metrics_to_parquet(
                content,
                Path(val_dir) / "aggregated_metrics.parquet",
                reporting_date=reporting_date,
            )

    def write_metrics_to_yaml(self, val_dir: str, write_parquet: bool = False) -> None:
        """Write collected metrics to YAML and, optionally, long-form parquet.

        Args:
            val_dir: Directory to save validation metrics
            write_parquet: Also save metric samples in ``metrics.parquet`` and
                NuRec-computed aggregates in ``aggregated_metrics.parquet``.
        """

        combined_samples = self.combine_samples_across_ranks(self._samples)
        aggregated_metrics = self.aggregate_metrics(combined_samples)
        reporting_date = datetime.now(timezone.utc) if write_parquet else None
        self.write_metrics_outputs(
            combined_samples,
            aggregated_metrics,
            val_dir,
            write_parquet=write_parquet,
            reporting_date=reporting_date,
        )

        self._samples = combined_samples
