# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and restore compact NuRec USDZ artifacts."""

import dataclasses
import fnmatch
import io
import json
import operator
import tempfile
import typing
import zipfile

from pathlib import Path

import safetensors
import safetensors.torch
import torch
import yaml

from omegaconf import DictConfig, OmegaConf

import nurec.artifact.artifact
import nurec.compression.gaussian_codec as gaussian_codec
import nurec.datasets.summary
import nurec.utils.upgrade

from nurec.utils.io.artifact_cache import ArtifactCache
from nurec.utils.types import ArtifactContents, NamedSerialized


COMPACT_PAYLOAD_NAME = "compact.safetensors"
DEFAULT_COMPRESSION_LEVEL = 9
DEFAULT_OPACITY_THRESHOLD = 0.005

_CHECKPOINT_NAME = "checkpoint.ckpt"
_PARSED_CONFIG_NAME = "parsed_config.yaml"
_DATASOURCE_SUMMARY_NAME = "datasource_summary.json"
_RIG_TRAJECTORIES_NAME = "rig_trajectories.json"
_SEQUENCE_TRACKS_NAME = "sequence_tracks.json"
_METADATA_NAME = "metadata.yaml"
_PROTECTED_SOURCE_MEMBERS = frozenset({_METADATA_NAME})
_CHECKPOINT_METADATA_KEYS = ("epoch", "global_step", "pytorch-lightning_version")
_JSON_SEPARATORS = (",", ":")
_GENERATED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_T = typing.TypeVar("_T")
_ZipWrite = typing.Callable[[zipfile.ZipFile], None]


@dataclasses.dataclass(frozen=True)
class _CompressionSource:
    state_dict: dict[str, typing.Any]
    parsed_config: DictConfig
    datasource_summary: dict[str, typing.Any]
    rig_trajectories: dict[str, typing.Any]
    sequence_tracks: dict[str, typing.Any]
    checkpoint_metadata: dict[str, typing.Any]


def _default_output_path(input_path: Path) -> Path:
    """Return a compact filename candidate for input_path."""
    return input_path.with_name(f"{input_path.stem}.compact.usdz")


def _is_compact_usdz(path: Path) -> bool:
    """Return true if path is a compact USDZ archive."""
    if path.suffix != ".usdz":
        return False
    with zipfile.ZipFile(path, "r") as zf:
        return COMPACT_PAYLOAD_NAME in zf.namelist()


def _read_compact_payload(path: Path) -> bytes:
    """Return compact safetensors payload bytes from a compact USDZ archive."""
    with zipfile.ZipFile(path, "r") as zf:
        try:
            return zf.read(COMPACT_PAYLOAD_NAME)
        except KeyError as exc:
            raise ValueError(f"{path} does not contain {COMPACT_PAYLOAD_NAME!r}") from exc


def load_compact_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Return a renderable checkpoint from a compact USDZ artifact.

    Packed tensors are moved to device before unpacking, so callers can
    materialize the dense state directly on CUDA without writing a temporary
    checkpoint.
    """
    tensors, metadata = _read_compact_tensors(path, device=device)
    state_dict = gaussian_codec.unpack(tensors, metadata)
    return _make_checkpoint(metadata, state_dict)


def compact_preserved_extra_signal_tensors(path: Path) -> frozenset[str]:
    """Return extra-signal tensors preserved by value in a compact USDZ."""
    metadata = _read_compact_metadata(path)
    return _preserved_extra_signal_tensors_from_metadata(metadata)


def write_artifact_cache(artifact_cache: ArtifactCache, output_path: Path) -> Path:
    """Write artifact_cache as normal or compact USDZ according to its config."""
    compression = artifact_cache.artifact_config.compression
    if not compression.enabled:
        artifact_cache.write_to_usdz(output_path)
        return output_path

    return compress_artifact_cache(
        artifact_cache,
        output_path,
        opacity_threshold=compression.opacity_threshold,
        compression_level=compression.level,
        preserve_extra_signal_tensors=compression.preserve_extra_signal_tensors,
        exclude_patterns=tuple(compression.exclude),
    )


def compress_artifact_cache(
    artifact_cache: ArtifactCache,
    output_path: Path,
    *,
    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    preserve_extra_signal_tensors: typing.Sequence[str] = (),
    exclude_patterns: tuple[str, ...] = (),
    renderer: str | None = None,
) -> Path:
    """Write a compact USDZ artifact from an in-memory artifact cache.

    `renderer`, if given, overrides `model.renderer.name` before the config upgrade runs. It is
    the recovery path for artifacts trained with a legacy NRend renderer that has no automatic
    migration path (see `nurec.utils.upgrade.apply_renderer_override`).
    """
    if output_path.suffix != ".usdz":
        raise ValueError("output must be a compact .usdz artifact")

    source = _load_artifact_cache_source(artifact_cache, renderer=renderer)
    payload = _pack_source(
        source,
        opacity_threshold=opacity_threshold,
        compression_level=compression_level,
        preserve_extra_signal_tensors=preserve_extra_signal_tensors,
    )
    _write_compact_artifact_cache(
        artifact_cache,
        output_path,
        payload,
        generated_members=_generated_members(source),
        exclude_patterns=exclude_patterns,
    )
    return output_path


def compress_usdz(
    input_path: Path,
    output_path: Path | None = None,
    *,
    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    preserve_extra_signal_tensors: typing.Sequence[str] = (),
    exclude_patterns: tuple[str, ...] = (),
    renderer: str | None = None,
) -> Path:
    """Compress a NuRec USDZ artifact.

    Normal USDZ input is loaded and upgraded through NuRec before tensor
    compression. Compact input is rejected; decompress it before creating a new
    compact artifact.

    `renderer`, if given, overrides `model.renderer.name` before the config upgrade runs. It is
    the recovery path for artifacts trained with a legacy NRend renderer that has no automatic
    migration path (see `nurec.utils.upgrade.apply_renderer_override`).

    Writes output_path and returns it. Raises ValueError if the artifact,
    settings, or requested paths are invalid.
    """
    if input_path.suffix != ".usdz":
        raise ValueError("input must be a .usdz artifact")
    if output_path is None:
        output_path = _default_output_path(input_path)
    if output_path.suffix != ".usdz":
        raise ValueError("output must be a compact .usdz artifact")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different files")
    if _is_compact_usdz(input_path):
        raise ValueError("input is already compact; decompress it before compressing again")

    source = _load_normal_source(input_path, renderer=renderer)
    payload = _pack_source(
        source,
        opacity_threshold=opacity_threshold,
        compression_level=compression_level,
        preserve_extra_signal_tensors=preserve_extra_signal_tensors,
    )
    _write_compact_usdz(
        input_path,
        output_path,
        payload,
        generated_members=_generated_members(source),
        exclude_patterns=exclude_patterns,
    )
    return output_path


def _pack_source(
    source: _CompressionSource,
    *,
    opacity_threshold: float,
    compression_level: int,
    preserve_extra_signal_tensors: typing.Sequence[str],
) -> bytes:
    """Pack source tensors and return serialized safetensors bytes."""
    tensors, metadata = gaussian_codec.pack(
        source.state_dict,
        opacity_threshold=opacity_threshold,
        compression_level=compression_level,
        preserve_extra_signal_tensors=preserve_extra_signal_tensors,
    )
    metadata.update(_metadata_fields(source))
    return safetensors.torch.save(tensors, metadata=metadata)


def _load_normal_source(input_path: Path, *, renderer: str | None = None) -> _CompressionSource:
    """Load and upgrade a normal NuRec USDZ artifact."""
    artifact = nurec.artifact.artifact.Artifact(input_path)
    checkpoint = _torch_load_checkpoint(artifact.checkpoint, _CHECKPOINT_NAME)

    orig_config = OmegaConf.create(artifact.parsed_config)
    return _build_compression_source(
        checkpoint,
        orig_config,
        artifact.datasource_summary,
        artifact.rig_trajectories,
        artifact.sequence_tracks,
        renderer=renderer,
    )


def _load_artifact_cache_source(artifact_cache: ArtifactCache, *, renderer: str | None = None) -> _CompressionSource:
    """Return source state from an in-memory artifact cache."""
    checkpoint_member = _require_serialized_member(artifact_cache.checkpoint, _CHECKPOINT_NAME)
    config_member = _require_serialized_member(artifact_cache.parsed_config, _PARSED_CONFIG_NAME)
    datasource_summary_member = _require_serialized_member(
        artifact_cache.datasource_summary,
        _DATASOURCE_SUMMARY_NAME,
    )
    rig_trajectories_member = _require_archive_member(artifact_cache.rig_trajectories, _RIG_TRAJECTORIES_NAME)
    sequence_tracks_member = _require_archive_member(artifact_cache.sequence_tracks, _SEQUENCE_TRACKS_NAME)

    checkpoint = _torch_load_member(checkpoint_member)
    orig_config = _config_from_member(config_member)
    datasource_summary_json = _json_from_member(datasource_summary_member)

    return _build_compression_source(
        checkpoint,
        orig_config,
        datasource_summary_json,
        rig_trajectories=_json_from_member(rig_trajectories_member),
        sequence_tracks=_json_from_member(sequence_tracks_member),
        renderer=renderer,
    )


def _build_compression_source(
    checkpoint: dict[str, typing.Any],
    orig_config: DictConfig,
    datasource_summary_json: dict[str, typing.Any],
    rig_trajectories: dict[str, typing.Any],
    sequence_tracks: dict[str, typing.Any],
    *,
    renderer: str | None = None,
) -> _CompressionSource:
    # `orig_config` (pristine, un-overridden) stays the record of what the artifact was trained
    # with and is what upgrade_model below keys off. Only the config upgrade gets the renderer
    # override, mirroring RenderableModel.load_from_artifact.
    upgraded_config = nurec.utils.upgrade.upgrade_config(
        nurec.utils.upgrade.apply_renderer_override(orig_config, renderer)
    )
    OmegaConf.resolve(upgraded_config)

    datasource_summary = nurec.datasets.summary.DataSourceSummary.from_dict(
        datasource_summary_json,
        infer_missing=True,
    )
    checkpoint_state_dict = _checkpoint_state_dict(checkpoint)
    raw_state_dict = {k.removeprefix("model."): v for k, v in checkpoint_state_dict.items() if k.startswith("model.")}
    state_dict = nurec.utils.upgrade.upgrade_model(raw_state_dict, orig_config, datasource_summary=datasource_summary)

    return _CompressionSource(
        state_dict={f"model.{k}": v for k, v in state_dict.items()},
        parsed_config=upgraded_config,
        datasource_summary=datasource_summary_json,
        rig_trajectories=rig_trajectories,
        sequence_tracks=sequence_tracks,
        checkpoint_metadata=_checkpoint_metadata(checkpoint),
    )


def _checkpoint_state_dict(checkpoint: typing.Any) -> dict[str, typing.Any]:
    """Return the state_dict from a Gaussian checkpoint."""
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("compact USDZ compression requires checkpoint.ckpt with a state_dict")
    return checkpoint["state_dict"]


def _require_archive_member(contents: ArtifactContents | None, filename: str) -> NamedSerialized:
    """Return a serialized archive member by filename."""
    if contents is None:
        raise ValueError(f"artifact cache is missing {filename}")
    for member in contents:
        if member.filename == filename and isinstance(member, NamedSerialized):
            return member
    raise ValueError(f"artifact cache is missing {filename}")


def _require_serialized_member(member: NamedSerialized | None, filename: str) -> NamedSerialized:
    """Return member if it is present with the expected filename."""
    if member is None:
        raise ValueError(f"artifact cache is missing {filename}")
    if member.filename != filename:
        raise ValueError(f"artifact cache has {member.filename}, expected {filename}")
    return member


def _member_bytes(member: NamedSerialized) -> bytes:
    """Return serialized member bytes."""
    return member.serialized if isinstance(member.serialized, bytes) else member.serialized.encode("utf-8")


def _torch_load_checkpoint(
    file: typing.IO[bytes],
    member_name: str,
    owner: str = "artifact",
) -> dict[str, typing.Any]:
    """Load a trusted NuRec checkpoint member."""
    try:
        return typing.cast(
            dict[str, typing.Any],
            torch.load(file, weights_only=False, map_location="cpu"),
        )
    except Exception as exc:
        raise ValueError(f"{owner} has corrupt {member_name}") from exc


def _torch_load_member(member: NamedSerialized) -> dict[str, typing.Any]:
    """Load a trusted NuRec checkpoint member."""
    return _torch_load_checkpoint(io.BytesIO(_member_bytes(member)), member.filename, "artifact cache")


def _config_from_member(member: NamedSerialized) -> DictConfig:
    """Return a config from a serialized YAML member."""
    return _parse_config_bytes(_member_bytes(member), member.filename, "artifact cache")


def _json_from_member(member: NamedSerialized) -> dict[str, typing.Any]:
    """Return JSON object from a serialized member."""
    return _parse_json_object_bytes(_member_bytes(member), member.filename, "artifact cache")


def _metadata_fields(source: _CompressionSource) -> dict[str, str]:
    """Return safetensors metadata fields for restored checkpoints."""
    return {key: _json_string(value) for key, value in source.checkpoint_metadata.items()}


def _generated_members(source: _CompressionSource) -> dict[str, bytes]:
    """Return upgraded USDZ members generated by compression."""
    return _sidecar_members(
        source.parsed_config,
        source.datasource_summary,
        source.rig_trajectories,
        source.sequence_tracks,
    )


def _generated_members_from_compact(input_path: Path) -> dict[str, bytes]:
    """Return generated USDZ members from compact archive sidecars."""
    parsed_config, datasource_summary, rig_trajectories, sequence_tracks = _read_compact_sidecars(input_path)
    # Decompression keeps using the standard config upgrade path.
    upgraded_config = nurec.utils.upgrade.upgrade_config(parsed_config)
    OmegaConf.resolve(upgraded_config)
    return _sidecar_members(upgraded_config, datasource_summary, rig_trajectories, sequence_tracks)


def _sidecar_members(
    parsed_config: DictConfig,
    datasource_summary: dict[str, typing.Any],
    rig_trajectories: dict[str, typing.Any],
    sequence_tracks: dict[str, typing.Any],
) -> dict[str, bytes]:
    """Return canonical runtime sidecar members."""
    return {
        _PARSED_CONFIG_NAME: OmegaConf.to_yaml(parsed_config).encode("utf-8"),
        _DATASOURCE_SUMMARY_NAME: _json_bytes(datasource_summary),
        _RIG_TRAJECTORIES_NAME: _json_bytes(rig_trajectories),
        _SEQUENCE_TRACKS_NAME: _json_bytes(sequence_tracks),
    }


def _read_compact_sidecars(
    input_path: Path,
) -> tuple[DictConfig, dict[str, typing.Any], dict[str, typing.Any], dict[str, typing.Any]]:
    """Return compact archive sidecars."""
    with zipfile.ZipFile(input_path, "r") as zf:
        parsed_config = _read_config_member(zf)
        datasource_summary = _read_json_member(zf, _DATASOURCE_SUMMARY_NAME)
        rig_trajectories = _read_json_member(zf, _RIG_TRAJECTORIES_NAME)
        sequence_tracks = _read_json_member(zf, _SEQUENCE_TRACKS_NAME)

    return parsed_config, datasource_summary, rig_trajectories, sequence_tracks


def _read_config_member(zf: zipfile.ZipFile) -> DictConfig:
    """Return parsed_config from a compact archive."""
    try:
        data = zf.read(_PARSED_CONFIG_NAME)
    except KeyError as exc:
        raise ValueError(f"compact archive is missing {_PARSED_CONFIG_NAME}") from exc
    return _parse_config_bytes(data, _PARSED_CONFIG_NAME, "compact archive")


def _read_json_member(zf: zipfile.ZipFile, member_name: str) -> dict[str, typing.Any]:
    """Return a JSON member from a compact archive."""
    try:
        data = zf.read(member_name)
    except KeyError as exc:
        raise ValueError(f"compact archive is missing {member_name}") from exc
    return _parse_json_object_bytes(data, member_name, "compact archive")


def _parse_config_bytes(data: bytes, filename: str, owner: str) -> DictConfig:
    try:
        value = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise ValueError(f"{owner} has corrupt {filename}") from exc
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"{owner} member {filename} must be a YAML object")
    return OmegaConf.create(value)


def _parse_json_object_bytes(data: bytes, filename: str, owner: str) -> dict[str, typing.Any]:
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{owner} has corrupt {filename}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{owner} member {filename} must be a JSON object")
    return typing.cast(dict[str, typing.Any], value)


def _checkpoint_metadata(checkpoint: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Return scalar checkpoint metadata preserved across compact artifacts."""
    return {
        "epoch": _checkpoint_metadata_int(checkpoint.get("epoch"), "epoch"),
        "global_step": _checkpoint_metadata_int(checkpoint.get("global_step"), "global_step"),
        "pytorch-lightning_version": _checkpoint_metadata_str(checkpoint.get("pytorch-lightning_version")),
    }


def _checkpoint_metadata_int(value: typing.Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"checkpoint metadata {key} must be a scalar integer")
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"checkpoint metadata {key} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"checkpoint metadata {key} must be a scalar integer") from exc


def _checkpoint_metadata_str(value: typing.Any) -> str | None:
    return None if value is None else str(value)


def _checkpoint_metadata_from_safetensors(metadata: dict[str, str]) -> dict[str, typing.Any]:
    """Return checkpoint metadata stored in safetensors metadata."""
    return {key: json.loads(metadata[key]) if key in metadata else None for key in _CHECKPOINT_METADATA_KEYS}


def decompress_usdz(input_path: Path, output_path: Path) -> None:
    """Restore an eval-ready USDZ artifact from lossy compact tensors.

    Writes output_path and raises ValueError if the compact artifact or paths
    are invalid.
    """
    if input_path.suffix != ".usdz":
        raise ValueError("input must be a compact .usdz artifact")
    if output_path.suffix != ".usdz":
        raise ValueError("output must be a .usdz artifact")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different files")

    checkpoint_bytes, generated = _unpack_generated_members(input_path)
    generated[_CHECKPOINT_NAME] = checkpoint_bytes
    _write_restored_usdz(input_path, output_path, generated)


def _stored_zip_info(src: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """Return a ZIP_STORED ZipInfo preserving stable metadata from src."""
    info = zipfile.ZipInfo(src.filename, src.date_time)
    info.comment = src.comment
    info.extra = src.extra
    info.internal_attr = src.internal_attr
    info.external_attr = src.external_attr
    info.create_system = src.create_system
    info.compress_type = zipfile.ZIP_STORED
    return info


def _generated_zip_info(filename: str) -> zipfile.ZipInfo:
    """Return deterministic metadata for generated ZIP members."""
    info = zipfile.ZipInfo(filename, _GENERATED_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    return info


def _writestr_generated(dst: zipfile.ZipFile, filename: str, data: bytes) -> None:
    """Write a generated member with deterministic ZIP metadata."""
    dst.writestr(_generated_zip_info(filename), data)


def _write_compact_usdz(
    input_path: Path,
    output_path: Path,
    payload: bytes,
    *,
    generated_members: dict[str, bytes],
    exclude_patterns: tuple[str, ...],
) -> None:
    """Write a compact USDZ archive."""
    _write_archive_atomically(
        output_path,
        lambda tmp_path: _write_compact_archive(
            input_path,
            tmp_path,
            payload,
            generated_members=generated_members,
            exclude_patterns=exclude_patterns,
        ),
    )


def _write_compact_artifact_cache(
    artifact_cache: ArtifactCache,
    output_path: Path,
    payload: bytes,
    *,
    generated_members: dict[str, bytes],
    exclude_patterns: tuple[str, ...],
) -> None:
    """Write compact USDZ from an in-memory artifact cache."""
    _write_archive_atomically(
        output_path,
        lambda tmp_path: _write_compact_artifact_cache_archive(
            artifact_cache,
            tmp_path,
            payload,
            generated_members=generated_members,
            exclude_patterns=exclude_patterns,
        ),
    )


def _write_compact_artifact_cache_archive(
    artifact_cache: ArtifactCache,
    output_path: Path,
    payload: bytes,
    *,
    generated_members: dict[str, bytes],
    exclude_patterns: tuple[str, ...],
) -> None:
    """Write compact artifact cache archive to output_path."""
    source_members = [
        (member.filename, lambda dst, member=member: member.save_to_zip(dst))
        for member in artifact_cache.archive_members()
    ]
    _write_compact_archive_members(
        output_path,
        source_members,
        payload,
        generated_members=generated_members,
        exclude_patterns=exclude_patterns,
    )


def _write_compact_archive(
    input_path: Path,
    output_path: Path,
    payload: bytes,
    *,
    generated_members: dict[str, bytes],
    exclude_patterns: tuple[str, ...],
) -> None:
    """Write a compact USDZ archive to output_path."""
    with zipfile.ZipFile(input_path, "r") as src:
        source_members = [
            (
                info.filename,
                lambda dst, info=info: dst.writestr(_stored_zip_info(info), src.read(info.filename)),
            )
            for info in src.infolist()
        ]
        _write_compact_archive_members(
            output_path,
            source_members,
            payload,
            generated_members=generated_members,
            exclude_patterns=exclude_patterns,
        )


def _write_compact_archive_members(
    output_path: Path,
    source_members: typing.Iterable[tuple[str, _ZipWrite]],
    payload: bytes,
    *,
    generated_members: dict[str, bytes],
    exclude_patterns: tuple[str, ...],
) -> None:
    reserved = {COMPACT_PAYLOAD_NAME, _CHECKPOINT_NAME}

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as dst:
        # The payload and generated sidecars are one coupled artifact, just
        # as checkpoint.ckpt and sidecars are coupled in a normal USDZ.
        for name, data in generated_members.items():
            _writestr_generated(dst, name, data)

        for name, write_member in source_members:
            if name in reserved or name in generated_members:
                continue
            if _should_exclude(name, exclude_patterns) and name not in _PROTECTED_SOURCE_MEMBERS:
                continue
            write_member(dst)

        _writestr_generated(dst, COMPACT_PAYLOAD_NAME, payload)


def _should_exclude(name: str, exclude_patterns: tuple[str, ...]) -> bool:
    """Return true if name matches an exclude pattern."""
    return any(_matches_exclude(name, pattern) for pattern in exclude_patterns)


def _matches_exclude(name: str, pattern: str) -> bool:
    """Return true if name matches a shell-style file or folder pattern."""
    if pattern.endswith("/"):
        return name == pattern.removesuffix("/") or name == pattern or name.startswith(pattern)
    return fnmatch.fnmatchcase(name, pattern)


def _unpack_generated_members(input_path: Path) -> tuple[bytes, dict[str, bytes]]:
    """Return restored checkpoint bytes and generated sidecar members."""
    checkpoint = load_compact_checkpoint(input_path)
    generated = _generated_members_from_compact(input_path)
    return _torch_save_bytes(checkpoint), generated


def _read_compact_tensors(
    input_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Return safetensors payload tensors and metadata from a compact USDZ."""
    payload = _read_compact_payload(input_path)
    metadata = _safetensors_metadata(payload)
    tensors = _load_safetensors_payload(payload)
    target = torch.device(device)
    if target.type != "cpu":
        tensors = {key: tensor.to(target) for key, tensor in tensors.items()}
    return tensors, metadata


def _read_compact_metadata(input_path: Path) -> dict[str, str]:
    """Return compact safetensors metadata without reading tensor data."""
    with zipfile.ZipFile(input_path, "r") as zf:
        try:
            with zf.open(COMPACT_PAYLOAD_NAME, "r") as fp:
                return _safetensors_metadata_from_header(_read_safetensors_header(fp))
        except KeyError as exc:
            raise ValueError(f"{input_path} does not contain {COMPACT_PAYLOAD_NAME!r}") from exc


def _load_safetensors_payload(payload: bytes) -> dict[str, torch.Tensor]:
    try:
        return safetensors.torch.load(payload)
    except safetensors.SafetensorError as exc:
        raise ValueError(f"compact archive has corrupt {COMPACT_PAYLOAD_NAME}") from exc


def _safetensors_metadata(payload: bytes) -> dict[str, str]:
    """Return safetensors metadata from serialized bytes."""
    return _safetensors_metadata_from_header(_safetensors_header(payload))


def _safetensors_header(payload: bytes) -> bytes:
    """Return the serialized safetensors header."""
    return _read_safetensors_header(io.BytesIO(payload))


def _read_safetensors_header(file: typing.IO[bytes]) -> bytes:
    """Read a serialized safetensors header from file."""
    header_prefix = file.read(8)
    if len(header_prefix) < 8:
        raise ValueError("compact safetensors payload is too small")
    header_len = int.from_bytes(header_prefix, "little")
    header = file.read(header_len)
    if len(header) < header_len:
        raise ValueError("compact safetensors payload has a truncated header")
    return header


def _safetensors_metadata_from_header(header_bytes: bytes) -> dict[str, str]:
    """Return safetensors metadata from a serialized header."""
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("compact safetensors payload has corrupt header JSON") from exc
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("compact safetensors metadata must be a JSON object")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("compact safetensors metadata must contain string values")
    return typing.cast(dict[str, str], metadata)


def _preserved_extra_signal_tensors_from_metadata(metadata: dict[str, str]) -> frozenset[str]:
    raw = metadata.get(gaussian_codec._METADATA_PRESERVED_EXTRA_SIGNAL_TENSORS, "[]")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("compact safetensors metadata has corrupt preserved signal tensor list") from exc
    if not isinstance(parsed, list) or not all(isinstance(value, str) for value in parsed):
        raise ValueError("compact safetensors metadata preserved signal tensor list must be a JSON string list")
    return frozenset(parsed)


def _make_checkpoint(metadata: dict[str, str], state_dict: dict[str, torch.Tensor]) -> dict:
    """Build the smallest normal checkpoint needed by artifact render/eval paths."""
    # Eval/restored artifacts should load into freshly built models, matching
    # the upgraded compact state instead of mutating an existing module in place.
    checkpoint: dict[str, object] = {"state_dict": state_dict, "load_in_place": False}
    for key, value in _checkpoint_metadata_from_safetensors(metadata).items():
        if value is not None:
            checkpoint[key] = value
    return checkpoint


def _torch_save_bytes(obj: object) -> bytes:
    """Serialize a torch object to bytes."""
    with io.BytesIO() as buff:
        torch.save(obj, buff)
        return buff.getvalue()


def _json_string(value: typing.Any) -> str:
    """Return compact JSON for archive metadata."""
    return json.dumps(value, separators=_JSON_SEPARATORS)


def _json_bytes(value: typing.Any) -> bytes:
    """Return compact JSON bytes for archive sidecars."""
    return _json_string(value).encode("utf-8")


def _write_restored_usdz(input_path: Path, output_path: Path, generated: dict[str, bytes]) -> None:
    """Write restored USDZ members plus any sidecars kept in the compact archive."""

    def write_archive(tmp_path: Path) -> None:
        _write_restored_archive(input_path, tmp_path, generated)

    _write_archive_atomically(output_path, write_archive)


def _write_restored_archive(input_path: Path, output_path: Path, generated: dict[str, bytes]) -> None:
    """Write restored USDZ archive to output_path."""
    with (
        zipfile.ZipFile(input_path, "r") as src,
        zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as dst,
    ):
        for name, data in generated.items():
            _writestr_generated(dst, name, data)

        for info in src.infolist():
            name = info.filename
            if name in generated or name == COMPACT_PAYLOAD_NAME:
                continue
            dst.writestr(_stored_zip_info(info), src.read(name))


def _write_archive_atomically(output_path: Path, write: typing.Callable[[Path], _T]) -> _T:
    """Write output_path through a sibling temp file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = write(tmp_path)
        tmp_path.replace(output_path)
        return result
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
