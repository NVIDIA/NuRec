# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate a NuRec Onboard input YAML.

Provides ``parse_input``, ``get_nested``, and ``is_unresolved`` as an
importable library (used by the validators) and a CLI. The CLI wraps
``yaml.safe_load`` with a dotted-key lookup helper so the shell dispatcher
(``run.sh``) can pull individual fields out of the parsed document without
doing its own YAML parsing. Example, run from the NuRec Onboard directory:

    python3 -m lib.parse_input my_input.yaml data.path ncore.converter_dir

prints one value per line in the requested order. Missing keys print an
empty string (preserving the contract the shell caller expects from
``mapfile``).

Validated on every load:

- ``data.name``: must match ``[A-Za-z0-9][A-Za-z0-9_-]{0,63}`` (used as a
  filesystem path component).
- ``ncore.camera_ids`` and ``ncore.lidar_ids``: when present, must be lists of
  non-empty sensor ID strings matching the same charset constraint.
- ``nurec.camera_ids``: when present, must be a list of sensor ID strings.
- ``nurec.lidar_ids``: when present, must be a one-element list of sensor ID
  strings and each concrete ID must appear in ``ncore.lidar_ids`` when both
  fields are concrete.
- ``ncore.converter_dir``, ``ncore.output_dir``, ``nurec.output_dir``,
  ``nurec.dev_deps_image``, and ``nurec.source_dir``: when present, must be non-blank
  strings.
- ``execution`` iteration limits: when present, must be positive integers.
- ``audit``: when present, must contain only the documented keys with the
  expected scalar types.
- ``data.profile``: when present, must be a non-empty mapping carrying every
  section in ``REQUIRED_PROFILE_SECTIONS``. Presence of this block is what
  tells the orchestrator a dataset is already audited, so an empty or
  half-authored profile has to fail here rather than silently skip the audit
  step. Per-field profile checks live in ``audit_dataset_validator``.
"""

from __future__ import annotations

import re
import sys

from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


# Fields that are consumed as filesystem path components or other contexts
# where unconstrained content is unsafe. Validated inside `parse_input()`
# on every load so shell callers (and any future Python importer) can't
# silently hand an agent-supplied value with shell metacharacters, path
# separators, or traversal segments to a downstream `cp`, `mkdir`, or
# `docker -v` invocation.
_DATA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SENSOR_ID_RE = _DATA_NAME_RE  # same charset; sensor IDs appear in docker -v mounts and Hydra overrides
_FIELD_VALIDATORS: dict[str, tuple[re.Pattern[str], str]] = {
    "data.name": (
        _DATA_NAME_RE,
        "data.name must be 1-64 chars, start with alphanumeric, and contain"
        " only [A-Za-z0-9_-] (it is used verbatim as a filesystem path"
        " component: <workdir>/ncore-<data.name>)",
    ),
}
_AUDIT_BLOCK_FIELDS: frozenset[str] = frozenset({"enabled", "output_dir", "fail_on_unresolved"})
# Keep in sync with nurec-onboard/steps/audit-dataset/input-skeleton.yaml and
# nurec-onboard/steps/audit-dataset/SKILL.md. Imported by
# `audit_dataset_validator`, which checks the fields within each section.
REQUIRED_PROFILE_SECTIONS: tuple[str, ...] = (
    "file_layout",
    "calibration_and_frames",
    "pose_sources",
    "timestamp_semantics",
    "cameras",
    "lidars",
    "cuboids",
    "known_gaps",
)
_EXECUTION_BLOCK_FIELDS: frozenset[str] = frozenset(
    {"max_inner_iterations", "max_outer_iterations", "max_config_iterations"}
)
_DEFAULTS_PATH = Path(__file__).parents[1] / "defaults.yaml"


def parse_input(path: str | Path) -> dict[str, Any]:
    """Load a workflow input YAML file into a dict.

    Args:
        path: Input YAML path.

    Returns:
        Parsed input document, or an empty dict for empty or whitespace-only
        files.

    Raises:
        TypeError: If the top-level YAML document is not a mapping.
        ValueError: If any validated field is present but malformed.
        yaml.YAMLError: If the YAML is malformed.
    """
    text = Path(path).read_text()
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected the top-level YAML document in {path} to be a mapping, got {type(data).__name__}")
    runtime = data.get("nurec")
    if isinstance(runtime, dict):
        images = [runtime[key] for key in ("image", "tools_image", "dev_deps_image") if runtime.get(key) is not None]
        if images and any(image != images[0] for image in images):
            raise ValueError("NuRec uses one dev_deps_image; image overrides must agree")
        if images:
            runtime["dev_deps_image"] = images[0]
        runtime.pop("image", None)
        runtime.pop("tools_image", None)
    data = _merge_defaults(data)
    for block in ("data", "ncore", "nurec", "execution", "audit"):
        _validate_optional_mapping(block, data.get(block))
    for field in _FIELD_VALIDATORS:
        _validate_field(field, get_nested(data, field))
    _validate_profile_block(data)
    _validate_profile_scope(data)
    _validate_runtime_settings(data)
    _validate_execution_block(data.get("execution"))
    _validate_audit_block(data.get("audit"))
    return data


def load_defaults(path: str | Path = _DEFAULTS_PATH) -> dict[str, Any]:
    """Load workflow input defaults.

    Args:
        path: Defaults YAML path.

    Returns:
        Parsed defaults mapping.

    Raises:
        TypeError: If the defaults YAML top-level document is not a mapping.
        yaml.YAMLError: If the defaults YAML is malformed.
    """
    defaults = yaml.safe_load(Path(path).read_text())
    if defaults is None:
        return {}
    if not isinstance(defaults, dict):
        raise TypeError(
            f"Expected the top-level YAML document in {path} to be a mapping, got {type(defaults).__name__}"
        )
    return defaults


def get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    """Look up a (possibly dotted) key path in a nested dict.

    Args:
        data: Parsed YAML mapping to inspect.
        dotted_key: Dot-separated key path.

    Returns:
        The nested value, or ``None`` if any segment is missing or the traversal
        hits a non-mapping value before the final segment.
    """
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _merge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` overlaid on repository workflow defaults."""
    merged = deepcopy(load_defaults())
    _deep_update(merged, data)
    return merged


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Recursively update ``base`` with ``overlay`` values.

    A ``None`` overlay value for a key whose default is a mapping is treated as
    "inherit the default" and skipped. This keeps an explicit-null *block* in an
    input document (e.g. ``nurec:`` with no children, which YAML parses to
    ``None``) from replacing the defaulted block merged from ``defaults.yaml``
    with ``None`` and silently dropping its image pins. A ``None`` overlay value
    for a non-mapping field is still written, so present-but-null placeholder
    fields (e.g. an audit-generated ``camera_ids: null``) keep their key.

    Args:
        base: Mapping to update in place (the merged defaults).
        overlay: Mapping whose values override ``base``.
    """
    for key, value in overlay.items():
        base_value = base.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            _deep_update(base_value, value)
        elif value is None and isinstance(base_value, dict):
            continue
        else:
            base[key] = value


def _stringify(value: Any) -> str:
    """Render a value for the shell caller (one field per line)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _validate_field(field: str, value: Any) -> None:
    """Enforce charset/length constraints on security-sensitive fields.

    Raises ``ValueError`` when ``field`` has a registered validator and
    ``value`` is present but doesn't match. ``None`` (missing field) passes
    through so callers can still distinguish "absent" from "present but
    invalid" at higher layers.
    """
    validator = _FIELD_VALIDATORS.get(field)
    if validator is None or value is None:
        return
    pattern, message = validator
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"Invalid {field}={value!r}: {message}")


def _validate_profile_block(data: dict[str, Any]) -> None:
    """Validate the optional ``data.profile`` dataset-profile block.

    Args:
        data: Parsed workflow input document.

    Raises:
        ValueError: If ``data.profile`` is present but is not a mapping, is
            empty, omits a required section, or carries a section with no body.
    """
    data_block = data.get("data")
    if not isinstance(data_block, dict) or "profile" not in data_block:
        return
    profile = data_block["profile"]
    if not isinstance(profile, dict):
        raise ValueError(f"Invalid data.profile={profile!r}: must be a mapping")
    if not profile:
        raise ValueError(
            "Invalid data.profile: must not be empty; a profile block is what marks a dataset as"
            " already audited, so omit the key entirely to run audit-dataset instead"
        )
    missing = [section for section in REQUIRED_PROFILE_SECTIONS if section not in profile]
    if missing:
        raise ValueError(
            f"Invalid data.profile: missing required sections {missing!r};"
            f" a complete profile carries {list(REQUIRED_PROFILE_SECTIONS)!r}"
        )
    # A null section states no fact, so a profile of bare section keys would
    # skip the audit while carrying nothing to audit. An empty list is left
    # alone: `lidars: []` is a real statement about a camera-only dataset.
    empty = [section for section in REQUIRED_PROFILE_SECTIONS if profile[section] is None]
    if empty:
        raise ValueError(
            f"Invalid data.profile: sections {empty!r} are present but empty; fill in the dataset"
            " facts, or omit data.profile entirely to run audit-dataset instead"
        )


def _validate_profile_scope(data: dict[str, Any]) -> None:
    """Validate scoped sensor IDs and conversion window fields.

    Args:
        data: Parsed workflow input document.

    Raises:
        ValueError: If a scoped field has the wrong shape or value.
    """
    _validate_ids_field(
        "ncore.camera_ids",
        get_nested(data, "ncore.camera_ids"),
    )
    ncore_lidar_ids = get_nested(data, "ncore.lidar_ids")
    _validate_ids_field("ncore.lidar_ids", ncore_lidar_ids)
    _validate_ids_field(
        "nurec.camera_ids",
        get_nested(data, "nurec.camera_ids"),
    )
    nurec_lidar_ids = get_nested(data, "nurec.lidar_ids")
    _validate_ids_field("nurec.lidar_ids", nurec_lidar_ids, exact_length=1)
    if isinstance(nurec_lidar_ids, list) and isinstance(ncore_lidar_ids, list):
        concrete_ncore_lidar_ids = {
            item for item in ncore_lidar_ids if isinstance(item, str) and not is_unresolved(item)
        }
        ncore_lidar_ids_fully_concrete = all(
            isinstance(item, str) and not is_unresolved(item) for item in ncore_lidar_ids
        )
        if ncore_lidar_ids_fully_concrete:
            for i, lidar_id in enumerate(nurec_lidar_ids):
                if (
                    isinstance(lidar_id, str)
                    and not is_unresolved(lidar_id)
                    and (not concrete_ncore_lidar_ids or lidar_id not in concrete_ncore_lidar_ids)
                ):
                    raise ValueError(f"Invalid nurec.lidar_ids[{i}]={lidar_id!r}: must appear in ncore.lidar_ids")
    seek_sec = get_nested(data, "ncore.seek_sec")
    duration_sec = get_nested(data, "ncore.duration_sec")
    _validate_optional_number("ncore.seek_sec", seek_sec)
    _validate_optional_number(
        "ncore.duration_sec",
        duration_sec,
    )
    if _is_concrete(seek_sec) != _is_concrete(duration_sec):
        raise ValueError("Invalid ncore time window: seek_sec and duration_sec must both be present or both be absent")


def _validate_audit_block(value: Any) -> None:
    """Validate the optional ``audit`` workflow-input block.

    Args:
        value: Value of the top-level ``audit`` key.

    Raises:
        ValueError: If the block has unknown keys or invalid value types.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"Invalid audit={value!r}: must be a mapping")
    unknown = sorted(set(value) - _AUDIT_BLOCK_FIELDS)
    if unknown:
        raise ValueError(f"Invalid audit keys {unknown!r}: allowed keys are {sorted(_AUDIT_BLOCK_FIELDS)!r}")
    if "enabled" in value and not isinstance(value["enabled"], bool):
        raise ValueError(f"Invalid audit.enabled={value['enabled']!r}: must be a bool")
    if "fail_on_unresolved" in value and not isinstance(value["fail_on_unresolved"], bool):
        raise ValueError(f"Invalid audit.fail_on_unresolved={value['fail_on_unresolved']!r}: must be a bool")
    if "output_dir" in value and (not isinstance(value["output_dir"], str) or not value["output_dir"].strip()):
        raise ValueError(f"Invalid audit.output_dir={value['output_dir']!r}: must be a non-blank string")


def _validate_runtime_settings(data: dict[str, Any]) -> None:
    """Validate optional grouped runtime path and image settings.

    Args:
        data: Parsed workflow input document.

    Raises:
        ValueError: If any present runtime setting has the wrong type.
    """
    for field in (
        "ncore.converter_dir",
        "ncore.output_dir",
        "nurec.output_dir",
        "nurec.dev_deps_image",
        "nurec.source_dir",
    ):
        _validate_optional_string(field, get_nested(data, field))


def _validate_execution_block(value: Any) -> None:
    """Validate the optional ``execution`` workflow-control block.

    Args:
        value: Value of the top-level ``execution`` key.

    Raises:
        ValueError: If the block has unknown keys or invalid value types.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"Invalid execution={value!r}: must be a mapping")
    unknown = sorted(set(value) - _EXECUTION_BLOCK_FIELDS)
    if unknown:
        raise ValueError(f"Invalid execution keys {unknown!r}: allowed keys are {sorted(_EXECUTION_BLOCK_FIELDS)!r}")
    for field in _EXECUTION_BLOCK_FIELDS:
        if field in value:
            _validate_positive_int(f"execution.{field}", value[field])


def _validate_ids_field(field: str, value: Any, *, exact_length: int | None = None) -> None:
    """Validate an optional list-of-sensor-ID field.

    Raises ``ValueError`` when *value* is present but:
    - is not a list,
    - contains a non-string or charset-violating element.

    ``None`` (absent field) passes through silently.
    """
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(f"Invalid {field}={value!r}: must be a list of sensor ID strings")
    for i, item in enumerate(value):
        if item is None:
            raise ValueError(
                f"Invalid {field}[{i}]=null: sensor IDs must be non-empty strings or UNRESOLVED: placeholders"
            )
        _validate_sensor_id_field(f"{field}[{i}]", item)
    if exact_length is not None and len(value) != exact_length:
        noun = "element" if exact_length == 1 else "elements"
        reason = "; NuRec trains on a single LiDAR per run" if field == "nurec.lidar_ids" else ""
        raise ValueError(f"Invalid {field}: must contain exactly {exact_length} {noun} (got {len(value)}){reason}")


def _validate_sensor_id_field(field: str, value: Any) -> None:
    """Validate an optional single sensor-ID field.

    Args:
        field: Diagnostic field path.
        value: Field value.

    Raises:
        ValueError: If ``value`` is present but not a valid sensor ID or
            ``UNRESOLVED:`` placeholder.
    """
    if value is None:
        return
    if is_unresolved(value):
        return
    if not isinstance(value, str) or not _SENSOR_ID_RE.fullmatch(value):
        raise ValueError(
            f"Invalid {field}={value!r}: each sensor ID must be"
            " 1-64 chars, start with alphanumeric, and contain only"
            " [A-Za-z0-9_-]"
        )


def _validate_optional_mapping(field: str, value: Any) -> None:
    """Validate an optional top-level mapping block."""
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"Invalid {field}={value!r}: must be a mapping")


def _validate_optional_number(field: str, value: Any) -> None:
    """Validate an optional non-negative numeric field.

    Args:
        field: Diagnostic field path.
        value: Field value.

    Raises:
        ValueError: If ``value`` is present but is not non-negative numeric or
            an ``UNRESOLVED:`` placeholder.
    """
    if value is None or is_unresolved(value):
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or not isfinite(value):
        raise ValueError(f"Invalid {field}={value!r}: must be a non-negative number")


def _validate_optional_string(field: str, value: Any) -> None:
    """Validate an optional non-blank string field."""
    if value is None or is_unresolved(value):
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid {field}={value!r}: must be a non-blank string")


def _validate_positive_int(field: str, value: Any) -> None:
    """Validate a positive integer field."""
    if is_unresolved(value):
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Invalid {field}={value!r}: must be a positive integer")


def is_unresolved(value: Any) -> bool:
    """Return whether ``value`` is an ``UNRESOLVED:`` placeholder string.

    Args:
        value: Any parsed YAML value.

    Returns:
        True when ``value`` is a string starting with ``UNRESOLVED:``.
    """
    return isinstance(value, str) and value.startswith("UNRESOLVED:")


def _is_concrete(value: Any) -> bool:
    """Return whether ``value`` is a concrete non-placeholder value."""
    return value is not None and not is_unresolved(value)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <input.yaml> [field ...]\nFields may be dotted (e.g. 'data.path').\n")
        sys.exit(1)

    try:
        doc = parse_input(sys.argv[1])
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        sys.stderr.write(f"parse_input: {exc}\n")
        sys.exit(2)

    if len(sys.argv) > 2:
        for field in sys.argv[2:]:
            print(_stringify(get_nested(doc, field)))
    else:
        for key, val in doc.items():
            print(f"{key}={_stringify(val)}")
