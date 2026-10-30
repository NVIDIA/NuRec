# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# compare against the verbose definition of the old serve.py command
# and make sure the dynamically added options are the same

import json
import logging

from typing import Optional
from unittest import mock

import click
import pytest

from click.testing import CliRunner
from pydantic import ValidationError

import nurec.utils.cli as cli

from nurec.config.version import get_version
from nurec.grpc import serve as serve_module
from nurec.grpc.grpc_server_config import (
    ALLOWED_IMAGE_FORMATS_DESCRIPTION,
    DEFAULT_DOWNLOAD_CACHE_DIR,
    HARMONIZER_TEMPORAL_COSMOS_BASE_MODEL_URL,
    HARMONIZER_TEMPORAL_COSMOS_TOKENIZER_URL,
    HARMONIZER_TEMPORAL_MODEL_FILENAME,
    HARMONIZER_TEMPORAL_MODEL_URL,
    HARMONIZER_TEMPORAL_RESOLUTION,
    SUPPORTED_GRPC_IMAGE_FORMATS,
    GrpcServerConfig,
    grpc_cli_options,
)
from nurec.grpc.protos.sensorsim_pb2 import ImageFormat
from nurec.grpc.serve import serve_grpc
from nurec.utils.misc import unpack_optional


log = logging.getLogger("nurec.grpc.grpc_server_config_test")


# deprecated call
@click.command("serve-grpc")
@click.option(
    "--artifact-glob",
    type=str,
    help="Glob expression to find artifacts. Must end in .usdz to find relevant files.",
    required=False,
    default=None,  # NOTE: this was added to the old command to make it compatible with the new command
)
@click.option(
    "--host",
    type=str,
    help="GRPC server host",
    default="localhost",
    required=False,
)
@click.option(
    "--port",
    type=int,
    help="GRPC server port",
    default=8080,  # NOTE: this was a string in the old command, fixed to int here
    required=False,
)
@click.option(
    "--health-port",
    type=int,
    help=(
        "Port for gRPC health checks (grpc.health.v1.Health). "
        "If unset, health is served on the main gRPC server/port (legacy behavior). "
        "Set to a different port to run a dedicated health-only gRPC server."
    ),
    default=None,
    required=False,
)
@click.option(
    "--test-scenes-are-valid/--no-test-scenes-are-valid",
    help="Try to load each detected scene before coming online. Helps avoid crashes during long running jobs.",
    default=False,
)
@click.option(
    "--renderer",
    type=click.Choice(["default", "gsplat"], case_sensitive=False),
    help=(
        "Renderer backend: 'default' uses the artifact's trained renderer (PyTorch forward pass), "
        "'gsplat' forces GSplatRenderer."
    ),
    default="default",
)
@click.option(
    "--allowed-image-formats",
    type=str,
    help=ALLOWED_IMAGE_FORMATS_DESCRIPTION,
    default=None,
)
@click.option(
    "--enable-harmonizer/--no-enable-harmonizer",  # NOTE: added --no- prefix to be consistent with other flag declarations
    is_flag=True,
    help="Load non-temporal Harmonizer postprocessing; render requests opt in with apply_harmonizer.",
    default=False,
)
@click.option(
    "--harmonizer-url",
    type=str,
    help="URL of Harmonizer checkpoint.",
    default="https://huggingface.co/nvidia/Harmonizer/blob/main/harmonizer_nontemporal.pt",
)
@click.option(
    "--harmonizer-cache",
    type=str,
    help="Full path to local Harmonizer cache dir.",
    default="~/.cache/nurec/harmonizer",
)
@click.option(
    "--harmonizer-model-filename",
    type=str,
    help="Filename of Harmonizer checkpoint.",
    default="harmonizer_nontemporal.pt",
)
@click.option(
    "--harmonizer-resolution",
    type=click.Tuple([int, int]),
    help="Resolution for Harmonizer processing.",
    default=(576, 1024),
)
@click.option(
    "--enable-harmonizer-temporal/--no-enable-harmonizer-temporal",
    is_flag=True,
    help=(
        "Load temporal Harmonizer postprocessing; opting into a colour render requires "
        "harmonizer_session_id, while an opt-out can supply one to reset its history. "
        "Mutually exclusive with --enable-harmonizer."
    ),
    default=False,
)
@click.option(
    "--harmonizer-temporal-url",
    type=str,
    help="URL of the temporal Harmonizer checkpoint.",
    default=HARMONIZER_TEMPORAL_MODEL_URL,
)
@click.option(
    "--harmonizer-temporal-model-filename",
    type=str,
    help="Filename of the temporal Harmonizer checkpoint.",
    default=HARMONIZER_TEMPORAL_MODEL_FILENAME,
)
@click.option(
    "--harmonizer-temporal-resolution",
    type=click.Tuple([int, int]),
    help="Resolution for temporal Harmonizer processing.",
    default=HARMONIZER_TEMPORAL_RESOLUTION,
)
@click.option(
    "--harmonizer-temporal-cosmos-base-model-url",
    type=str,
    help="URL of the Cosmos Predict2 base model required by checkpointed temporal Harmonizer.",
    default=HARMONIZER_TEMPORAL_COSMOS_BASE_MODEL_URL,
)
@click.option(
    "--harmonizer-temporal-cosmos-tokenizer-url",
    type=str,
    help="URL of the Cosmos Predict2 tokenizer required by checkpointed temporal Harmonizer.",
    default=HARMONIZER_TEMPORAL_COSMOS_TOKENIZER_URL,
)
@click.option(
    "--harmonizer-temporal-cosmos-cache-dir",
    type=str,
    help=(
        "Optional local Cosmos Predict2 cache dir. If unset, temporal Harmonizer uses a "
        "Cosmos subdirectory under the Harmonizer cache."
    ),
    default=None,
)
@click.option(
    "--max-harmonizer-sessions",
    type=int,
    help=(
        "Maximum concurrent temporal Harmonizer sessions. create_harmonizer_session beyond "
        "this returns RESOURCE_EXHAUSTED."
    ),
    default=32,
)
@cli.scopedtimer_cli_options(print_func=log.info)
@cli.tracks_timestamp_ray_stride_cli_option()
@click.option(
    "--ray-chunk-size",
    type=int,
    help="Maximum number of rays processed in a single forward pass. Default: effectively unlimited. Set if seeing GPU OOMs.",
    default=2**62,
)
@click.option(
    "--egocar-hood-dir",
    type=str,
    help="Directory with egocar hood images.",
    default=None,
)
@click.option(
    "--download-cache-dir",
    type=str,
    help="Directory for downloaded scene files",
    default=DEFAULT_DOWNLOAD_CACHE_DIR,
)
@click.option(
    "--download-cache-size",
    type=int,
    help="Maximum number of downloaded scenes to keep in cache",
    default=5,
)
@click.option(
    "--max-workers",
    type=int,
    help="Maximum number of workers for the grpc server",
    default=1,
)
@click.option(
    "--metrics-output-dir",
    type=str,
    help="Directory to save render time metrics. If not specified, metrics collection is disabled.",
    default=None,
)
@click.option(
    "--enable-editing-actors/--no-enable-editing-actors",  # NOTE: added --no- prefix to be consistent with other flag declarations
    is_flag=True,
    help="Enable editing the actors in a scene or use the original actor poses.",
    default=False,
)
@click.option(
    "--cache-size",
    type=int,
    help="Maximum number of models to cache (count-based LRU). If OOM occurs during loading, spare backends are automatically evicted and load is retried.",
    default=10,
)
@click.option(
    "--max-backends-per-scene",
    type=int,
    help=(
        "Maximum number of cached backends allowed per scene. Must be positive and no greater than "
        "--cache-size. If unset, no per-scene limit or capacity waiting is applied. When set, saturated "
        "requests wait for a reusable backend until cancellation or a deadline; waiter order is not guaranteed."
    ),
    default=None,
)
@click.option(
    "--cuda-ipc-max-buffer-count",
    type=int,
    help=(
        "Maximum number of GPU frame buffers a single CUDA IPC session's open_ipc_session may "
        "request (bounds GPU memory pinned per session -- e.g. a 4K RGB_UINT8_PLANAR buffer is "
        "~25MB, so the default allows up to ~1.6GB per session)."
    ),
    default=64,
)
@click.option(
    "--use-gsplat/--no-use-gsplat",
    default=None,
    hidden=True,
    help="Deprecated: use --renderer instead.",
)
@click.option(
    "--enable-difix/--no-enable-difix",
    default=None,
    hidden=True,
    help="Deprecated: use --enable-harmonizer instead.",
)
@click.option(
    "--difix-url",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated: use --harmonizer-url instead.",
)
@click.option(
    "--difix-cache",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated: use --harmonizer-cache instead.",
)
@click.option(
    "--difix-model-filename",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated: use --harmonizer-model-filename instead.",
)
@click.option(
    "--difix-resolution",
    type=click.Tuple([int, int]),
    default=None,
    hidden=True,
    help="Deprecated: use --harmonizer-resolution instead.",
)
@click.version_option(version=str(unpack_optional(get_version(), default="version-not-available")))
def serve_grpc_old(
    artifact_glob: Optional[str],
    host: str,
    port: int,
    health_port: Optional[int],
    test_scenes_are_valid: bool,
    renderer: str,
    allowed_image_formats: Optional[str],
    use_gsplat: Optional[bool],
    enable_harmonizer: bool,
    harmonizer_url: str,
    harmonizer_cache: str,
    harmonizer_model_filename: str,
    harmonizer_resolution: tuple[int, int],
    enable_harmonizer_temporal: bool,
    harmonizer_temporal_url: str,
    harmonizer_temporal_model_filename: str,
    harmonizer_temporal_resolution: tuple[int, int],
    harmonizer_temporal_cosmos_base_model_url: str,
    harmonizer_temporal_cosmos_tokenizer_url: str,
    harmonizer_temporal_cosmos_cache_dir: Optional[str],
    max_harmonizer_sessions: int,
    enable_difix: Optional[bool],
    difix_url: Optional[str],
    difix_cache: Optional[str],
    difix_model_filename: Optional[str],
    difix_resolution: Optional[tuple[int, int]],
    enable_timing: bool,
    timing_verbosity: str,
    timing_logfile: Optional[str],
    timing_synchronize: bool,
    profiling_backend: str,
    ray_chunk_size: int,
    egocar_hood_dir: Optional[str],
    download_cache_dir: str,
    download_cache_size: int,
    max_workers: int,
    cache_size: int,
    max_backends_per_scene: Optional[int],
    cuda_ipc_max_buffer_count: int,
    metrics_output_dir: Optional[str],
    enable_editing_actors: Optional[bool],
    tracks_timestamp_ray_stride: int,
) -> click.Context:
    """Neural Rendering gRPC server"""
    return click.get_current_context()


def test_dynamic_grpc_server_command_equals_old_command():
    dynamic_command = serve_grpc
    old_command = serve_grpc_old

    assert isinstance(dynamic_command, click.Command)
    assert isinstance(old_command, click.Command)

    # also assert the help messages are the same
    assert dynamic_command.help == old_command.help
    assert dynamic_command.name == old_command.name

    # Added launch-only flags (--multiproc-workers / --require-mps /
    # --multiproc-require-scene-cache / --multiproc-bringup-concurrency / --multiproc-max-in-flight /
    # --multiproc-admission-timeout-s) that are NOT part of the
    # GrpcServerConfig surface; exclude them from the old-vs-generated comparison (asserted separately).
    launch_only_flags = {
        "multiproc_workers",
        "require_mps",
        "multiproc_require_scene_cache",
        "multiproc_bringup_concurrency",
        "multiproc_max_in_flight",
        "multiproc_admission_timeout_s",
    }

    # Compare the parameter-name sets first so a missing/extra option fails with a
    # clear diff instead of an opaque strict-zip length error.
    dynamic_names = {str(p.name) for p in dynamic_command.params if p.name not in launch_only_flags}
    old_names = {str(p.name) for p in old_command.params}
    assert dynamic_names == old_names, (
        f"serve-grpc option sets diverged. "
        f"only in dynamic: {sorted(dynamic_names - old_names)}; "
        f"only in old reference: {sorted(old_names - dynamic_names)}"
    )

    # then sort params by name
    dynamic_command_params_sorted = sorted(
        (p for p in dynamic_command.params if p.name not in launch_only_flags), key=lambda x: str(x.name)
    )
    old_command_params_sorted = sorted(old_command.params, key=lambda x: str(x.name))

    for dynamic_param, old_param in zip(dynamic_command_params_sorted, old_command_params_sorted, strict=True):
        print(f"# dynamic_param.name: {dynamic_param.name} - old_param.name: {old_param.name}")
        assert dynamic_param.name == old_param.name
        assert dynamic_param.default == old_param.default
        print(f"# dynamic_param.type: {type(dynamic_param.type)} - old_param.type: {type(old_param.type)}")
        assert type(dynamic_param.type) is type(old_param.type)
        assert dynamic_param.required == old_param.required
        assert dynamic_param.get_help_record(click.Context(dynamic_command)) == old_param.get_help_record(
            click.Context(old_command)
        )


def test_multiproc_launch_flags_present_with_defaults():
    # Drop-in switch: serve-grpc gains launch-only flags (not GrpcServerConfig fields).
    params = {p.name: p for p in serve_grpc.params}
    assert "multiproc_workers" in params
    assert params["multiproc_workers"].default == 0  # single-process by default
    assert "require_mps" in params
    assert params["require_mps"].default is False
    assert "multiproc_require_scene_cache" in params
    assert params["multiproc_require_scene_cache"].default is False  # Auto-align capacities by default
    # Admission options, all OFF (unlimited) by default.
    assert params["multiproc_max_in_flight"].default == 0
    assert params["multiproc_admission_timeout_s"].default == 0.0
    # Firing patterns are read at launch to build the LidarBank, not carried in the config;
    # repeatable so a multi-lidar artifact can supply one per model-free unit.


@click.command("serve-grpc-test")
@grpc_cli_options()
def serve_grpc_test_command(**kwargs):  # pragma: no cover - invoked by CliRunner
    # Instantiate config to trigger validation
    config = GrpcServerConfig(**kwargs)
    click.echo(json.dumps(config.model_dump(mode="json"), sort_keys=True))


def _invoke_grpc_test_command(args: list[str]) -> dict:
    runner = CliRunner()
    result = runner.invoke(serve_grpc_test_command, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output.strip())


def _invoke_grpc_test_command_failure(args: list[str]) -> str:
    runner = CliRunner()
    result = runner.invoke(serve_grpc_test_command, args)
    assert result.exit_code != 0
    return result.output or str(result.exception)


def test_renderer_default_is_default():
    params = _invoke_grpc_test_command([])
    assert params["renderer"] == "default"


def test_renderer_gsplat():
    params = _invoke_grpc_test_command(["--renderer", "gsplat"])
    assert params["renderer"] == "gsplat"


def test_allowed_image_formats_default_is_unrestricted():
    params = _invoke_grpc_test_command([])
    assert params["allowed_image_formats"] is None


def test_allowed_image_formats_are_canonicalized_from_cli():
    params = _invoke_grpc_test_command(["--allowed-image-formats", "jpeg, rgb"])
    assert params["allowed_image_formats"] == "JPEG,RGB_UINT8_PLANAR"


def test_allowed_image_formats_are_canonicalized_programmatically():
    config = GrpcServerConfig(allowed_image_formats="png,JPEG")
    assert config.allowed_image_formats == "PNG,JPEG"


def test_supported_grpc_image_formats_match_proto_enum():
    for name in SUPPORTED_GRPC_IMAGE_FORMATS:
        assert ImageFormat.Name(ImageFormat.Value(name)) == name


@pytest.mark.parametrize(
    ("value", "expected_message"),
    [
        pytest.param("", "empty entries", id="empty-list"),
        pytest.param("JPEG,", "empty entries", id="empty-entry"),
        pytest.param("AV1", "unsupported image format", id="unsupported-format"),
        pytest.param("JPEG,JPEG", "duplicate format JPEG", id="duplicate-format"),
        pytest.param(
            "RGB,RGB_UINT8_PLANAR",
            "duplicate format RGB_UINT8_PLANAR",
            id="duplicate-format-after-alias-normalization",
        ),
    ],
)
def test_allowed_image_formats_reject_invalid_values(value: str, expected_message: str):
    with pytest.raises(ValidationError, match=expected_message):
        GrpcServerConfig(allowed_image_formats=value)


def test_deprecated_use_gsplat_redirects_to_renderer():
    params = _invoke_grpc_test_command(["--use-gsplat"])
    assert params["renderer"] == "gsplat"


def test_deprecated_difix_options_redirect_to_harmonizer_options():
    params = _invoke_grpc_test_command(
        [
            "--enable-difix",
            "--difix-url",
            "https://example.com/harmonizer.pt",
            "--difix-cache",
            "/tmp/harmonizer-cache",
            "--difix-model-filename",
            "harmonizer.pt",
            "--difix-resolution",
            "720",
            "1280",
        ]
    )

    assert params["enable_harmonizer"] is True
    assert params["harmonizer_url"] == "https://example.com/harmonizer.pt"
    assert params["harmonizer_cache"] == "/tmp/harmonizer-cache"
    assert params["harmonizer_model_filename"] == "harmonizer.pt"
    assert params["harmonizer_resolution"] == [720, 1280]


def test_harmonizer_options():
    params = _invoke_grpc_test_command(
        [
            "--enable-harmonizer",
            "--harmonizer-url",
            "https://example.com/harmonizer.pt",
            "--harmonizer-cache",
            "/tmp/harmonizer-cache",
            "--harmonizer-model-filename",
            "harmonizer.pt",
            "--harmonizer-resolution",
            "720",
            "1280",
        ]
    )

    assert params["enable_harmonizer"] is True
    assert params["harmonizer_url"] == "https://example.com/harmonizer.pt"
    assert params["harmonizer_cache"] == "/tmp/harmonizer-cache"
    assert params["harmonizer_model_filename"] == "harmonizer.pt"
    assert params["harmonizer_resolution"] == [720, 1280]


def test_harmonizer_options_override_deprecated_difix_options():
    params = _invoke_grpc_test_command(
        [
            "--no-enable-harmonizer",
            "--enable-difix",
            "--harmonizer-url",
            "https://example.com/new-harmonizer.pt",
            "--difix-url",
            "https://example.com/old-difix.pt",
            "--harmonizer-resolution",
            "720",
            "1280",
            "--difix-resolution",
            "1",
            "2",
        ]
    )

    assert params["enable_harmonizer"] is False
    assert params["harmonizer_url"] == "https://example.com/new-harmonizer.pt"
    assert params["harmonizer_resolution"] == [720, 1280]


# Non-temporal and temporal harmonizer are distinct checkpoints/code paths; enabling both must fail.
def test_enable_harmonizer_and_temporal_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        GrpcServerConfig(enable_harmonizer=True, enable_harmonizer_temporal=True)


def test_max_backends_per_scene_unset():
    params = _invoke_grpc_test_command([])
    assert params["max_backends_per_scene"] is None


def test_max_backends_per_scene_valid():
    params = _invoke_grpc_test_command(["--cache-size", "3", "--max-backends-per-scene", "2"])
    assert params["cache_size"] == 3
    assert params["max_backends_per_scene"] == 2


@pytest.mark.parametrize("invalid_value", [0, -1])
def test_max_backends_per_scene_must_be_positive(invalid_value):
    with pytest.raises(ValidationError, match="max_backends_per_scene must be greater than 0"):
        GrpcServerConfig(max_backends_per_scene=invalid_value)


@pytest.mark.parametrize("invalid_value", ["0", "-1"])
def test_max_backends_per_scene_cli_must_be_positive(invalid_value):
    output = _invoke_grpc_test_command_failure(["--max-backends-per-scene", invalid_value])
    assert "max_backends_per_scene must be greater than 0" in output


def test_max_backends_per_scene_must_not_exceed_cache_size():
    with pytest.raises(ValidationError, match="max_backends_per_scene must be less than or equal to cache_size"):
        GrpcServerConfig(cache_size=2, max_backends_per_scene=3)


def test_max_backends_per_scene_cli_must_not_exceed_cache_size():
    output = _invoke_grpc_test_command_failure(["--cache-size", "2", "--max-backends-per-scene", "3"])
    assert "max_backends_per_scene must be less than or equal to cache_size" in output


def test_cuda_ipc_max_buffer_count_default():
    params = _invoke_grpc_test_command([])
    assert params["cuda_ipc_max_buffer_count"] == 64


def test_cuda_ipc_max_buffer_count_valid():
    params = _invoke_grpc_test_command(["--cuda-ipc-max-buffer-count", "128"])
    assert params["cuda_ipc_max_buffer_count"] == 128


@pytest.mark.parametrize("invalid_value", [0, -1])
def test_cuda_ipc_max_buffer_count_must_be_positive(invalid_value):
    with pytest.raises(ValidationError, match="cuda_ipc_max_buffer_count must be greater than 0"):
        GrpcServerConfig(cuda_ipc_max_buffer_count=invalid_value)


@pytest.mark.parametrize("invalid_value", ["0", "-1"])
def test_cuda_ipc_max_buffer_count_cli_must_be_positive(invalid_value):
    output = _invoke_grpc_test_command_failure(["--cuda-ipc-max-buffer-count", invalid_value])
    assert "cuda_ipc_max_buffer_count must be greater than 0" in output


def test_sensor_sim_service_wires_max_backends_per_scene_to_backend_cache():
    server_config = GrpcServerConfig(cache_size=3, max_backends_per_scene=2)

    with mock.patch.object(serve_module, "BackendCache") as backend_cache_cls:
        service = serve_module.SensorSimService(
            server=None,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=3,
            max_backends_per_scene=2,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=server_config,
        )

    backend_cache_cls.assert_called_once_with(maxsize=3, max_per_key=2)
    assert service.backend_cache == backend_cache_cls.return_value


def test_serve_grpc_wires_max_backends_per_scene_to_service():
    fake_server = mock.MagicMock()
    fake_server.add_insecure_port.return_value = 1
    fake_service = mock.MagicMock()

    with (
        mock.patch.object(serve_module.futures, "ThreadPoolExecutor", return_value=mock.MagicMock()),
        mock.patch.object(serve_module.grpc, "server", return_value=fake_server),
        mock.patch.object(serve_module, "SceneCache"),
        mock.patch.object(serve_module, "Downloader"),
        mock.patch.object(serve_module, "SensorSimService", return_value=fake_service) as service_cls,
        mock.patch.object(serve_module, "add_SensorsimServiceServicer_to_server"),
        mock.patch.object(serve_module.health_pb2_grpc, "add_HealthServicer_to_server"),
    ):
        result = CliRunner().invoke(
            serve_grpc,
            ["--cache-size", "3", "--max-backends-per-scene", "2"],
        )

    assert result.exit_code == 0, result.output
    service_cls.assert_called_once()
    assert service_cls.call_args.kwargs["cache_size"] == 3
    assert service_cls.call_args.kwargs["max_backends_per_scene"] == 2


def test_tracks_timestamp_ray_stride_rejects_below_one():
    @click.command()
    @cli.tracks_timestamp_ray_stride_cli_option()
    def _cmd(tracks_timestamp_ray_stride: int) -> None:  # pragma: no cover - invoked by CliRunner
        click.echo(tracks_timestamp_ray_stride)

    result = CliRunner().invoke(_cmd, ["--tracks-timestamp-ray-stride", "0"])
    assert result.exit_code != 0
    assert "x>=1" in result.output  # IntRange(min=1) range message

    ok = CliRunner().invoke(_cmd, ["--tracks-timestamp-ray-stride", "16"])
    assert ok.exit_code == 0
    assert ok.output.strip() == "16"


def test_cache_capacity_warning_when_cache_smaller_than_workers():
    warning = GrpcServerConfig(cache_size=4, max_workers=8).cache_capacity_warning()
    assert warning is not None
    assert "cache_size" in warning and "max_workers" in warning


def test_no_cache_capacity_warning_when_cache_ge_workers():
    # Equal is fine (each worker can hold exactly one replica).
    assert GrpcServerConfig(cache_size=8, max_workers=8).cache_capacity_warning() is None
    # Larger cache is fine.
    assert GrpcServerConfig(cache_size=16, max_workers=8).cache_capacity_warning() is None
    # Defaults (cache_size=10, max_workers=1) are fine.
    assert GrpcServerConfig().cache_capacity_warning() is None


@pytest.mark.parametrize("value", [0, -1])
def test_cache_size_must_be_positive(value: int) -> None:
    # The resident-scene cap needs room for at least one scene, so 0 and negatives are rejected.
    with pytest.raises(ValueError, match=r"cache_size must be a positive integer"):
        GrpcServerConfig(cache_size=value)


def test_download_cache_size_rejects_negative() -> None:
    # A negative on-disk cap is always a misconfiguration.
    with pytest.raises(ValueError, match=r"download_cache_size must be a non-negative integer"):
        GrpcServerConfig(download_cache_size=-1)


def test_download_cache_size_zero_is_accepted_no_persist() -> None:
    # 0 = the historical "never persist" / evict-immediately disk cache; restored per D-N2 (single-process
    # parity), so it must construct cleanly rather than hard-failing at startup.
    config = GrpcServerConfig(download_cache_size=0)
    assert config.download_cache_size == 0


def test_default_download_cache_size_is_five() -> None:
    # Single-process default restored to 5 (the multiprocess path auto-bumps to the resident budget itself).
    assert GrpcServerConfig().download_cache_size == 5


def test_cache_capacity_positive_boundary_is_valid() -> None:
    config = GrpcServerConfig(cache_size=1, download_cache_size=1)
    assert config.cache_size == 1
    assert config.download_cache_size == 1
