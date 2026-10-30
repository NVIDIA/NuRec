# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

from contextlib import nullcontext
from functools import wraps
from typing import Any, Optional

import click

from omegaconf import DictConfig, open_dict
from pydantic import field_validator, model_validator

from nurec.config.base_schema import BaseConfigSchema, Field
from nurec.config.model import RendererBackend
from nurec.utils.cli import config_to_cli_options


DEFAULT_DOWNLOAD_CACHE_DIR = "~/.cache/nurec/downloaded_scenes"

# Default coordinates for the Harmonizer checkpoint fields below (all CLI-overridable).
HARMONIZER_CACHE_DIR = "~/.cache/nurec/harmonizer"

HARMONIZER_NONTEMPORAL_MODEL_URL = "https://huggingface.co/nvidia/Harmonizer/blob/main/harmonizer_nontemporal.pt"
HARMONIZER_NONTEMPORAL_MODEL_FILENAME = "harmonizer_nontemporal.pt"
HARMONIZER_NONTEMPORAL_RESOLUTION = (576, 1024)

HARMONIZER_TEMPORAL_MODEL_URL = "https://huggingface.co/nvidia/Harmonizer/resolve/main/diffusion_harmonizer.pkl"
HARMONIZER_TEMPORAL_MODEL_FILENAME = "diffusion_harmonizer.pkl"
HARMONIZER_TEMPORAL_RESOLUTION = (576, 1024)
HARMONIZER_TEMPORAL_COSMOS_BASE_MODEL_URL = (
    "https://huggingface.co/nvidia/Cosmos-Predict2-0.6B-Text2Image/resolve/main/model.pt"
)
HARMONIZER_TEMPORAL_COSMOS_TOKENIZER_URL = (
    "https://huggingface.co/nvidia/Cosmos-Predict2-0.6B-Text2Image/resolve/main/tokenizer/tokenizer.pth"
)

# Legacy `difix_*` config keys → modern `harmonizer_*` / `enable_harmonizer`
# equivalents.  Used by the schema-level migration validator below and kept
# parallel to the CLI shim in :func:`grpc_cli_options`.
_DIFIX_TO_HARMONIZER_KEYS = {
    "enable_difix": "enable_harmonizer",
    "difix_url": "harmonizer_url",
    "difix_cache": "harmonizer_cache",
    "difix_model_filename": "harmonizer_model_filename",
    "difix_resolution": "harmonizer_resolution",
}

SUPPORTED_GRPC_IMAGE_FORMATS = ("PNG", "JPEG", "RGB_UINT8_PLANAR")
_GRPC_IMAGE_FORMAT_ALIASES = {"RGB": "RGB_UINT8_PLANAR"}
ALLOWED_IMAGE_FORMATS_DESCRIPTION = (
    "Comma-separated allow-list of image formats accepted by gRPC camera render requests: "
    "PNG, JPEG, or RGB_UINT8_PLANAR (RGB is accepted as an alias). Unset adds no format admission "
    "restriction, preserving the existing behavior."
)


def normalize_allowed_image_formats(value: Optional[str]) -> Optional[str]:
    """Validate and canonicalize the operator-facing image-format allow-list."""
    if value is None:
        return None

    normalized_formats = []
    for requested_format in value.split(","):
        normalized_format = requested_format.strip().upper()
        if not normalized_format:
            raise ValueError("allowed_image_formats must not contain empty entries")
        normalized_format = _GRPC_IMAGE_FORMAT_ALIASES.get(normalized_format, normalized_format)
        if normalized_format not in SUPPORTED_GRPC_IMAGE_FORMATS:
            raise ValueError(
                f"unsupported image format {requested_format.strip()!r}; expected one of "
                f"{', '.join(SUPPORTED_GRPC_IMAGE_FORMATS)}"
            )
        if normalized_format in normalized_formats:
            raise ValueError(f"allowed_image_formats contains duplicate format {normalized_format}")
        normalized_formats.append(normalized_format)

    return ",".join(normalized_formats)


def allowed_image_format_names(value: Optional[str]) -> Optional[frozenset[str]]:
    """Return the canonical allow-list, or ``None`` when admission is unrestricted."""
    normalized = normalize_allowed_image_formats(value)
    return None if normalized is None else frozenset(normalized.split(","))


class GrpcServerConfig(BaseConfigSchema):
    """Settings for the gRPC server."""

    artifact_glob: Optional[str] = Field(
        description="Glob expression to find artifacts. Must end in .usdz to find relevant files.", default=None
    )
    host: Optional[str] = Field(description="GRPC server host", default="localhost")
    port: int = Field(description="GRPC server port", default=8080)
    health_port: Optional[int] = Field(
        description="Port for gRPC health checks (grpc.health.v1.Health). If unset, health is served on the main gRPC server/port (legacy behavior). Set to a different port to run a dedicated health-only gRPC server.",
        default=None,
    )
    test_scenes_are_valid: bool = Field(
        description="Try to load each detected scene before coming online. Helps avoid crashes during long running jobs.",
        default=False,
    )
    renderer: RendererBackend = Field(
        description=(
            "Renderer backend: 'default' uses the artifact's trained renderer (PyTorch forward pass), "
            "'gsplat' forces GSplatRenderer."
        ),
        default=RendererBackend.DEFAULT,
    )
    allowed_image_formats: Optional[str] = Field(
        description=ALLOWED_IMAGE_FORMATS_DESCRIPTION,
        default=None,
    )
    enable_harmonizer: bool = Field(
        description="Load non-temporal Harmonizer postprocessing; render requests opt in with apply_harmonizer.",
        default=False,
    )
    harmonizer_url: str = Field(
        description="URL of Harmonizer checkpoint.",
        default=HARMONIZER_NONTEMPORAL_MODEL_URL,
    )
    harmonizer_cache: str = Field(description="Full path to local Harmonizer cache dir.", default=HARMONIZER_CACHE_DIR)
    harmonizer_model_filename: str = Field(
        description="Filename of Harmonizer checkpoint.", default=HARMONIZER_NONTEMPORAL_MODEL_FILENAME
    )
    harmonizer_resolution: tuple[int, int] = Field(
        description="Resolution for Harmonizer processing.", default=HARMONIZER_NONTEMPORAL_RESOLUTION
    )
    enable_harmonizer_temporal: bool = Field(
        description=(
            "Load temporal Harmonizer postprocessing; opting into a colour render requires "
            "harmonizer_session_id, while an opt-out can supply one to reset its history. "
            "Mutually exclusive with --enable-harmonizer."
        ),
        default=False,
    )
    harmonizer_temporal_url: str = Field(
        description="URL of the temporal Harmonizer checkpoint.",
        default=HARMONIZER_TEMPORAL_MODEL_URL,
    )
    harmonizer_temporal_model_filename: str = Field(
        description="Filename of the temporal Harmonizer checkpoint.",
        default=HARMONIZER_TEMPORAL_MODEL_FILENAME,
    )
    harmonizer_temporal_resolution: tuple[int, int] = Field(
        description="Resolution for temporal Harmonizer processing.",
        default=HARMONIZER_TEMPORAL_RESOLUTION,
    )
    harmonizer_temporal_cosmos_base_model_url: str = Field(
        description="URL of the Cosmos Predict2 base model required by checkpointed temporal Harmonizer.",
        default=HARMONIZER_TEMPORAL_COSMOS_BASE_MODEL_URL,
    )
    harmonizer_temporal_cosmos_tokenizer_url: str = Field(
        description="URL of the Cosmos Predict2 tokenizer required by checkpointed temporal Harmonizer.",
        default=HARMONIZER_TEMPORAL_COSMOS_TOKENIZER_URL,
    )
    harmonizer_temporal_cosmos_cache_dir: Optional[str] = Field(
        description=(
            "Optional local Cosmos Predict2 cache dir. If unset, temporal Harmonizer uses a "
            "Cosmos subdirectory under the Harmonizer cache."
        ),
        default=None,
    )
    max_harmonizer_sessions: int = Field(
        description=(
            "Maximum concurrent temporal Harmonizer sessions. create_harmonizer_session beyond "
            "this returns RESOURCE_EXHAUSTED."
        ),
        default=32,
    )
    ray_chunk_size: int = Field(
        description="Maximum number of rays processed in a single forward pass. Default: effectively unlimited. Set if seeing GPU OOMs.",
        default=2**62,
    )
    egocar_hood_dir: Optional[str] = Field(description="Directory with egocar hood images.", default=None)
    download_cache_dir: str = Field(
        description="Directory for downloaded scene files", default=DEFAULT_DOWNLOAD_CACHE_DIR
    )
    # Single-process default is 5 (unchanged from historical `serve`). The multiprocess path aligns the
    # on-disk cache to the resident-scene budget on its own, via `_resolve_download_cache_size`'s auto-bump,
    # so the shared default does not need to carry the multiprocess floor.
    download_cache_size: int = Field(description="Maximum number of downloaded scenes to keep in cache", default=5)
    max_workers: int = Field(description="Maximum number of workers for the grpc server", default=1)
    enable_editing_actors: bool = Field(
        description="Enable editing the actors in a scene or use the original actor poses.", default=False
    )
    cache_size: int = Field(
        description="Maximum number of models to cache (count-based LRU). If OOM occurs during loading, spare backends are automatically evicted and load is retried.",
        default=10,
    )
    max_backends_per_scene: Optional[int] = Field(
        description=(
            "Maximum number of cached backends allowed per scene. Must be positive and no greater than "
            "--cache-size. If unset, no per-scene limit or capacity waiting is applied. When set, saturated "
            "requests wait for a reusable backend until cancellation or a deadline; waiter order is not guaranteed."
        ),
        default=None,
    )
    cuda_ipc_max_buffer_count: int = Field(
        description=(
            "Maximum number of GPU frame buffers a single CUDA IPC session's open_ipc_session may "
            "request (bounds GPU memory pinned per session -- e.g. a 4K RGB_UINT8_PLANAR buffer is "
            "~25MB, so the default allows up to ~1.6GB per session)."
        ),
        default=64,
    )
    metrics_output_dir: Optional[str] = Field(
        description="Directory to save render time metrics. If not specified, metrics collection is disabled.",
        default=None,
    )

    def cache_capacity_warning(self) -> Optional[str]:
        """Warn when the backend cache is too small for the worker concurrency.

        Each in-flight render checks out a (per-scene) backend replica from the
        ``BackendCache``; with ``cache_size < max_workers`` a burst of concurrent
        same-scene requests can occupy every cache slot and make new checkouts raise
        ``CacheFullError`` (see ``serve.py`` ``BackendCache.put``). Returns a warning
        string in that case, else ``None``.
        """
        if self.cache_size < self.max_workers:
            return (
                f"cache_size ({self.cache_size}) < max_workers ({self.max_workers}): a burst of "
                f"concurrent same-scene requests can exhaust the backend cache and raise "
                f"CacheFullError. Set --cache-size >= --max-workers."
            )
        return None

    @field_validator("allowed_image_formats")
    @classmethod
    def _normalize_allowed_image_formats(cls, value: Optional[str]) -> Optional[str]:
        return normalize_allowed_image_formats(value)

    @model_validator(mode="after")
    def _validate_positive_cache_sizes(self) -> "GrpcServerConfig":
        """Guard the cache capacities. The resident-scene cap (`cache_size`) must be a positive integer — the
        pool needs room for at least one resident scene. The on-disk download cap (`download_cache_size`)
        must be non-negative: 0 is the historical "never persist" / evict-immediately disk cache and stays
        accepted (single-process parity), while a negative value is always a misconfiguration."""
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be a positive integer; got {self.cache_size}")
        if self.download_cache_size < 0:
            raise ValueError(f"download_cache_size must be a non-negative integer; got {self.download_cache_size}")
        return self

    @model_validator(mode="before")
    @classmethod
    def _migrate_difix_keys(cls, data: Any) -> Any:
        """Backward compat: route legacy ``difix_*`` config keys to their
        ``harmonizer_*`` / ``enable_harmonizer`` equivalents.

        The CLI shim in :func:`grpc_cli_options` handles command-line flags;
        this validator covers YAML, env-var, and programmatic dict inputs that
        bypass the CLI layer.  When both the legacy and the modern key are
        supplied, the modern key wins.
        """
        if not isinstance(data, (dict, DictConfig)):
            return data
        ctx = open_dict(data) if isinstance(data, DictConfig) else nullcontext()
        with ctx:
            for legacy_key, modern_key in _DIFIX_TO_HARMONIZER_KEYS.items():
                if legacy_key not in data:
                    continue
                warnings.warn(
                    f"`{legacy_key}` is deprecated, use `{modern_key}` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if modern_key not in data:
                    data[modern_key] = data[legacy_key]
        return data

    @model_validator(mode="after")
    def _check_harmonizer_modes_exclusive(self) -> "GrpcServerConfig":
        """Non-temporal and temporal Harmonizer are distinct checkpoints/code paths; enable at most one."""
        if self.enable_harmonizer and self.enable_harmonizer_temporal:
            raise ValueError(
                "enable_harmonizer (non-temporal) and enable_harmonizer_temporal are mutually exclusive; "
                "enable only one."
            )
        return self

    @model_validator(mode="after")
    def _validate_max_backends_per_scene(self) -> "GrpcServerConfig":
        if self.max_backends_per_scene is None:
            return self

        if self.max_backends_per_scene <= 0:
            raise ValueError("max_backends_per_scene must be greater than 0 when set")
        if self.max_backends_per_scene > self.cache_size:
            raise ValueError("max_backends_per_scene must be less than or equal to cache_size")
        return self

    @model_validator(mode="after")
    def _validate_cuda_ipc_max_buffer_count(self) -> "GrpcServerConfig":
        if self.cuda_ipc_max_buffer_count <= 0:
            raise ValueError("cuda_ipc_max_buffer_count must be greater than 0")
        return self


def grpc_cli_options():
    options = config_to_cli_options(GrpcServerConfig())

    # Deprecated flags kept for backward compatibility.
    deprecated_options = [
        click.option(
            "--use-gsplat/--no-use-gsplat",
            default=None,
            hidden=True,
            help="Deprecated: use --renderer instead.",
        ),
        click.option(
            "--enable-difix/--no-enable-difix",
            default=None,
            hidden=True,
            help="Deprecated: use --enable-harmonizer instead.",
        ),
        click.option(
            "--difix-url",
            type=str,
            default=None,
            hidden=True,
            help="Deprecated: use --harmonizer-url instead.",
        ),
        click.option(
            "--difix-cache",
            type=str,
            default=None,
            hidden=True,
            help="Deprecated: use --harmonizer-cache instead.",
        ),
        click.option(
            "--difix-model-filename",
            type=str,
            default=None,
            hidden=True,
            help="Deprecated: use --harmonizer-model-filename instead.",
        ),
        click.option(
            "--difix-resolution",
            type=click.Tuple([int, int]),
            default=None,
            hidden=True,
            help="Deprecated: use --harmonizer-resolution instead.",
        ),
    ]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Redirect deprecated flags to --renderer, warn, then strip them.
            use_gsplat = kwargs.pop("use_gsplat", None)
            enable_difix = kwargs.pop("enable_difix", None)
            difix_url = kwargs.pop("difix_url", None)
            difix_cache = kwargs.pop("difix_cache", None)
            difix_model_filename = kwargs.pop("difix_model_filename", None)
            difix_resolution = kwargs.pop("difix_resolution", None)

            def is_default_param(param_name: str) -> bool:
                ctx = click.get_current_context(silent=True)
                get_parameter_source = getattr(ctx, "get_parameter_source", None) if ctx is not None else None
                if get_parameter_source is None:
                    return True
                return get_parameter_source(param_name) == click.core.ParameterSource.DEFAULT

            def redirect_deprecated_difix_option(
                old_value,
                old_option: str,
                new_option: str,
                new_param_name: str,
            ) -> None:
                if old_value is None:
                    return
                warnings.warn(
                    f"{old_option} is deprecated, use {new_option} instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                if is_default_param(new_param_name):
                    kwargs[new_param_name] = old_value

            if use_gsplat is not None:
                warnings.warn(
                    "--use-gsplat/--no-use-gsplat is deprecated, use --renderer instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if use_gsplat and kwargs.get("renderer") == RendererBackend.DEFAULT:
                    kwargs["renderer"] = RendererBackend.GSPLAT

            redirect_deprecated_difix_option(
                enable_difix,
                "--enable-difix/--no-enable-difix",
                "--enable-harmonizer/--no-enable-harmonizer",
                "enable_harmonizer",
            )
            redirect_deprecated_difix_option(
                difix_url,
                "--difix-url",
                "--harmonizer-url",
                "harmonizer_url",
            )
            redirect_deprecated_difix_option(
                difix_cache,
                "--difix-cache",
                "--harmonizer-cache",
                "harmonizer_cache",
            )
            redirect_deprecated_difix_option(
                difix_model_filename,
                "--difix-model-filename",
                "--harmonizer-model-filename",
                "harmonizer_model_filename",
            )
            redirect_deprecated_difix_option(
                difix_resolution,
                "--difix-resolution",
                "--harmonizer-resolution",
                "harmonizer_resolution",
            )

            return func(*args, **kwargs)

        wrapper.__click_params__ = list(getattr(func, "__click_params__", []))
        for option in reversed(deprecated_options + options):
            wrapper = option(wrapper)
        return wrapper

    return decorator
