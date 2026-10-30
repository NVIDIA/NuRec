# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate artifacts emitted by the ``audit-dataset`` step.

Usage:
``cd "$NUREC_ONBOARD_DIR" && bazel run //skills/nurec-onboard/validators:audit_dataset_validator -- <input.yaml> <report.md> <audit_profile.json>``.

Exit codes: ``0`` pass, ``1`` fail, ``2`` malformed input.
"""

from __future__ import annotations

import json
import re
import sys

from pathlib import Path
from typing import Any

import yaml

from lib.parse_input import REQUIRED_PROFILE_SECTIONS, is_unresolved, parse_input

from validators._validation_base import ValidationError, placeholder_label, run_validator


_REQUIRED_REPORT_SECTIONS: tuple[str, ...] = (
    "## Summary",
    "## Documentation Coverage",
    "## Minimum NCore/NuRec Facts",
    "## Sensor Facts",
    "## Confirmed Parameters",
    "## Critical Gaps",
    "## Warnings",
    "## Operator Next Steps",
)

_CRITICAL_GAPS_HEADER_RE = re.compile(r"^## Critical Gaps\s*$", re.MULTILINE)
_CRITICAL_GAP_KEY_RE = re.compile(
    r"""^\s*(?:[-*]\s*)?(?:`(?P<backtick>[^`]+)`|["'](?P<quote>[^"']+)["']|(?P<bare>(?:data|ncore|nurec|execution|audit)(?:\.[A-Za-z0-9_]+|\[[0-9]+\])*))(?:\s*:|\s+-|\s*$)"""
)
_UNRESOLVED_RE = re.compile(r"UNRESOLVED:")
_UNRESOLVED_EXPLICIT_KEY_RE = re.compile(r"UNRESOLVED:\s*(data\.[A-Za-z0-9_.\-\[\]]+)")
_UNKNOWN_NUREC_ONBOARD_VERSION = "0.0.0+unknown"
# The `version:` key of the workflow SKILL.md's YAML frontmatter, optionally quoted.
_FRONTMATTER_VERSION_RE = re.compile(
    r"""\A---\n(?:.*\n)*?version:\s*["']?([^"'\s#]+)["']?\s*(?:#.*)?\n(?:.*\n)*?---\n"""
)


def _read_nurec_onboard_version() -> str:
    """Read NuRec Onboard's release version from the workflow SKILL.md frontmatter."""
    skill_md = Path(__file__).parents[1] / "nurec-onboard" / "SKILL.md"
    try:
        text = skill_md.read_text()
    except OSError:
        return _UNKNOWN_NUREC_ONBOARD_VERSION
    match = _FRONTMATTER_VERSION_RE.match(text)
    if match is None:
        return _UNKNOWN_NUREC_ONBOARD_VERSION
    return match.group(1)


NUREC_ONBOARD_VERSION = _read_nurec_onboard_version()

# Keep these profile schema fragments in sync with
# nurec-onboard/steps/audit-dataset/input-skeleton.yaml and nurec-onboard/steps/audit-dataset/SKILL.md.
_REQUIRED_PROFILE_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "file_layout": (
        "summary",
        "root",
        "sequence_id_source",
        "raw_inputs",
        "calibration_files",
        "metadata_files",
    ),
    "calibration_and_frames": (
        "summary",
        "calibration_sources",
        "rig_frame",
        "world_frame",
        "camera_frame",
        "lidar_frame",
        "frames",
        "static_transforms",
        "dynamic_transforms",
        "transform_direction_conventions",
        "known_mismatches",
    ),
    "timestamp_semantics": (
        "summary",
        "clock",
        "unit",
        "camera_timestamp_meaning",
        "lidar_frame_timestamp_meaning",
        "per_ray_timestamps",
        "synchronization",
    ),
    "cuboids": (
        "summary",
        "exists",
        "raw_stream",
        "centroid",
        "dimensions",
        "yaw",
        "reference_frame",
        "timestamp_meaning",
        "track_id_source",
        "class_vocabulary",
        "label_source",
        "coverage",
    ),
}
_REQUIRED_NCORE_FIELDS: tuple[str, ...] = (
    "camera_ids",
    "lidar_ids",
    "seek_sec",
    "duration_sec",
    "converter_dir",
    "output_dir",
)
_REQUIRED_NUREC_FIELDS: tuple[str, ...] = ("camera_ids", "lidar_ids", "output_dir")
_REQUIRED_EXECUTION_FIELDS: tuple[str, ...] = (
    "max_inner_iterations",
    "max_outer_iterations",
    "max_config_iterations",
)
# Keep these profile schema fragments in sync with
# nurec-onboard/steps/audit-dataset/input-skeleton.yaml and nurec-onboard/steps/audit-dataset/SKILL.md.
_REQUIRED_PROFILE_LIST_ENTRY_FIELDS: dict[str, tuple[str, ...]] = {
    "pose_sources": (
        "id",
        "raw_stream",
        "source_frame",
        "target_frame",
        "direction",
        "timestamp_unit",
        "coverage",
        "global_frame",
    ),
    "cameras": (
        "id",
        "raw_stream",
        "model",
        "resolution",
        "image_format",
        "model_family",
        "intrinsics_source",
        "distortion_model",
        "distortion_coeff_order",
        "image_state",
        "shutter",
        "masks",
    ),
    "lidars": (
        "id",
        "raw_stream",
        "model",
        "type",
        "raw_format",
        "model_source",
        "beam_structure",
        "spinning",
        "timing",
        "per_ray_metadata",
        "intensity",
        "motion_compensation",
    ),
}
# Keep these profile schema fragments in sync with
# nurec-onboard/steps/audit-dataset/input-skeleton.yaml and nurec-onboard/steps/audit-dataset/SKILL.md.
_REQUIRED_PROFILE_NESTED_FIELDS: dict[str, tuple[str, ...]] = {
    "cameras": (
        "shutter.type",
        "shutter.direction",
        "shutter.readout_duration_sec",
        "shutter.timestamp_meaning",
        "masks.available",
        "masks.entries",
    ),
    "lidars": (
        "beam_structure.rows",
        "beam_structure.columns",
        "beam_structure.row_elevations",
        "beam_structure.column_azimuths",
        "beam_structure.row_azimuth_offsets",
        "spinning.direction",
        "spinning.frequency_hz",
        "timing.frame_timestamp_meaning",
        "timing.per_ray_timestamps",
    ),
}
_REQUIRED_PROFILE_NESTED_FIELD_TYPES: dict[str, dict[str, str]] = {
    "cameras": {
        "shutter.type": "description",
        "shutter.direction": "description_or_null",
        "shutter.readout_duration_sec": "number_or_description_or_null",
        "shutter.timestamp_meaning": "description",
        "masks.available": "bool",
        "masks.entries": "list",
    },
    "lidars": {
        "beam_structure.rows": "number_or_description_or_null",
        "beam_structure.columns": "number_or_description_or_null",
        "beam_structure.row_elevations": "description_or_null",
        "beam_structure.column_azimuths": "description_or_null",
        "beam_structure.row_azimuth_offsets": "description_or_null",
        # null passes this type gate but is rejected by the high-risk check
        # (_high_risk_fact_diagnostics), which requires concrete-or-UNRESOLVED.
        "spinning.direction": "description_or_null",
        "spinning.frequency_hz": "number_or_description_or_null",
        "timing.frame_timestamp_meaning": "description_or_null",
        "timing.per_ray_timestamps": "description_or_null",
    },
}


def _extract_unresolved_keys(value: Any, *, root_path: str = "data.profile") -> list[str]:
    """Collect unresolved fact keys from a generated structured value.

    Args:
        value: Generated YAML subtree to scan.
        root_path: Dot path for ``value`` in the generated input.

    Returns:
        YAML paths for each ``UNRESOLVED:`` marker, in occurrence order. A
        marker may still supply an explicit ``data.*`` key for unusual cases.
    """
    keys: list[str] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if _UNRESOLVED_RE.search(node):
                explicit = _UNRESOLVED_EXPLICIT_KEY_RE.search(node)
                keys.append(explicit.group(1) if explicit is not None else path)
        elif isinstance(node, dict):
            for key, child in node.items():
                _walk(child, f"{path}.{key}")
        elif isinstance(node, list):
            for i, child in enumerate(node):
                _walk(child, f"{path}[{i}]")

    _walk(value, root_path)
    return keys


def _placeholder_diagnostics(path: str, value: Any) -> list[str]:
    """Find placeholder markers outside sanctioned ``UNRESOLVED`` strings.

    Args:
        path: Diagnostic path to ``value``.
        value: YAML subtree to scan.

    Returns:
        Diagnostics for placeholder markers.
    """
    diagnostics: list[str] = []
    if isinstance(value, str):
        if "UNRESOLVED:" in value:
            return diagnostics
        label = placeholder_label(value)
        if label is not None:
            diagnostics.append(f"{path} contains {label}: {value.strip()!r}")
    elif isinstance(value, dict):
        for key, child in value.items():
            diagnostics.extend(_placeholder_diagnostics(f"{path}.{key}", child))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            diagnostics.extend(_placeholder_diagnostics(f"{path}[{i}]", child))
    return diagnostics


def _list_section_diagnostics(section: str, entries: Any) -> list[str]:
    """Validate standardized list-section entry shapes.

    Args:
        section: Profile section name.
        entries: Profile section value.

    Returns:
        Diagnostics for malformed list sections or entries.
    """
    diagnostics: list[str] = []
    required_fields = _REQUIRED_PROFILE_LIST_ENTRY_FIELDS[section]
    path = f"data.profile.{section}"
    if not isinstance(entries, list):
        return [f"{path} must be a list"]
    for i, entry in enumerate(entries):
        entry_path = f"{path}[{i}]"
        if not isinstance(entry, dict):
            diagnostics.append(f"{entry_path} must be a mapping")
            continue
        for field in required_fields:
            if field not in entry:
                diagnostics.append(f"{entry_path}.{field} is missing")
        for field_path in _REQUIRED_PROFILE_NESTED_FIELDS.get(section, ()):
            nested_value = _get_nested_field(entry, field_path)
            if nested_value is _MISSING:
                diagnostics.append(f"{entry_path}.{field_path} is missing")
                continue
            value_kind = _REQUIRED_PROFILE_NESTED_FIELD_TYPES.get(section, {}).get(field_path)
            if value_kind is not None:
                diagnostics.extend(_value_type_diagnostics(f"{entry_path}.{field_path}", nested_value, value_kind))
    return diagnostics


_MISSING = object()


def _get_nested_field(value: dict[str, Any], field_path: str) -> Any:
    """Return a dotted nested field value or ``_MISSING``.

    Args:
        value: Mapping to inspect.
        field_path: Dot-separated path inside ``value``.

    Returns:
        The field value, or ``_MISSING`` when any path segment is absent.
    """
    node: Any = value
    for part in field_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _is_non_empty_string(value: Any) -> bool:
    """Return whether ``value`` is a non-blank string."""
    return isinstance(value, str) and bool(value.strip())


def _is_present_or_unresolved(value: Any) -> bool:
    """Return whether ``value`` is concrete or explicitly unresolved."""
    if is_unresolved(value):
        return True
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def _value_type_diagnostics(path: str, value: Any, kind: str) -> list[str]:
    """Validate a schema field's value kind.

    Args:
        path: Diagnostic path to ``value``.
        value: Field value.
        kind: One of the local schema value-kind names.

    Returns:
        Diagnostics for wrong value types.
    """
    if is_unresolved(value):
        return []
    if kind == "description":
        if _is_non_empty_string(value):
            return []
        return [f"{path} must be a non-empty string or UNRESOLVED marker"]
    if kind == "description_or_null":
        if value is None or _is_non_empty_string(value):
            return []
        return [f"{path} must be null, a non-empty string, or UNRESOLVED marker"]
    if kind == "number_or_description_or_null":
        if (
            value is None
            or (isinstance(value, (int, float)) and not isinstance(value, bool))
            or _is_non_empty_string(value)
        ):
            return []
        return [f"{path} must be null, numeric, a non-empty string, or UNRESOLVED marker"]
    if kind == "bool":
        if isinstance(value, bool):
            return []
        return [f"{path} must be a bool or UNRESOLVED marker"]
    if kind == "list":
        if isinstance(value, list):
            return []
        return [f"{path} must be a list or UNRESOLVED marker"]
    return []


def _profile_semantic_diagnostics(profile: dict[str, Any]) -> list[str]:
    """Validate cross-field and canonical-shape profile invariants.

    Args:
        profile: ``data.profile`` mapping.

    Returns:
        Diagnostics for profile values that have the required keys but an
        invalid relationship or canonical shape.
    """
    diagnostics: list[str] = []
    cuboids = profile.get("cuboids")
    if isinstance(cuboids, dict) and cuboids.get("exists") is not False:
        # `dimensions` needs an exact length/width/height key-set check, not
        # only the nested presence/type checks used for camera and LiDAR fields.
        diagnostics.extend(_cuboid_dimensions_diagnostics(cuboids.get("dimensions")))
    diagnostics.extend(_high_risk_fact_diagnostics(profile))
    return diagnostics


def _high_risk_fact_diagnostics(profile: dict[str, Any]) -> list[str]:
    """Validate high-risk facts are concrete or explicitly unresolved.

    Enforces shape only: every high-risk field must hold a concrete value or an
    ``UNRESOLVED:`` marker, never blank or null. Whether an unresolved marker
    then *blocks* the workflow is a separate question the orchestrator decides
    from the blocking/converter-derivable split in the audit SKILL, so do not
    read membership here as "unresolved here stops the run."

    Args:
        profile: ``data.profile`` mapping.

    Returns:
        Diagnostics for high-risk facts that are blank or null. These facts are
        expensive to guess wrong; audit-dataset should emit ``UNRESOLVED:`` when
        documentation does not answer them.
    """
    diagnostics: list[str] = []

    def require(path: str, value: Any) -> None:
        if not _is_present_or_unresolved(value):
            diagnostics.append(f"{path} must be concrete or an UNRESOLVED marker")

    calibration = profile.get("calibration_and_frames")
    if isinstance(calibration, dict):
        for field in ("rig_frame", "camera_frame", "lidar_frame"):
            # These are required section fields, so an absent one is already
            # reported as "is missing" by the required-field check; guard on
            # presence here to add the concrete-or-UNRESOLVED check only, not a
            # duplicate missing-field diagnostic.
            if field in calibration:
                require(f"data.profile.calibration_and_frames.{field}", calibration.get(field))
        static_transforms = calibration.get("static_transforms")
        if isinstance(static_transforms, list):
            for i, transform in enumerate(static_transforms):
                if isinstance(transform, dict):
                    require(
                        f"data.profile.calibration_and_frames.static_transforms[{i}].direction",
                        transform.get("direction"),
                    )

    cameras = profile.get("cameras")
    if isinstance(cameras, list):
        for i, camera in enumerate(cameras):
            if isinstance(camera, dict):
                for field in ("model_family", "distortion_model"):
                    require(f"data.profile.cameras[{i}].{field}", camera.get(field))

    lidars = profile.get("lidars")
    if isinstance(lidars, list):
        for i, lidar in enumerate(lidars):
            if isinstance(lidar, dict):
                require(f"data.profile.lidars[{i}].motion_compensation", lidar.get("motion_compensation"))
                spinning = lidar.get("spinning")
                if isinstance(spinning, dict):
                    require(f"data.profile.lidars[{i}].spinning.direction", spinning.get("direction"))

    cuboids = profile.get("cuboids")
    if isinstance(cuboids, dict) and cuboids.get("exists") is not False:
        for field in ("reference_frame", "centroid"):
            require(f"data.profile.cuboids.{field}", cuboids.get(field))

    return diagnostics


def _audit_scope_block_diagnostics(path: str, value: Any, required_fields: tuple[str, ...]) -> list[str]:
    """Validate a top-level scope block in an audit-generated input YAML.

    Args:
        path: Top-level scope block name.
        value: Scope block value.
        required_fields: Keys required in audit-generated input YAMLs.

    Returns:
        Diagnostics for missing or malformed audit-output scope blocks.
    """
    if not isinstance(value, dict):
        return [f"{path} is missing or not a mapping"]
    diagnostics: list[str] = []
    for field in required_fields:
        if field not in value:
            diagnostics.append(f"{path}.{field} is missing")
    diagnostics.extend(_placeholder_diagnostics(path, value))
    return diagnostics


def _cuboid_dimensions_diagnostics(value: Any) -> list[str]:
    """Validate canonical cuboid dimension mapping shape.

    Args:
        value: ``data.profile.cuboids.dimensions`` value.

    Returns:
        Diagnostics when the dimensions field is not a ``length`` / ``width`` /
        ``height`` source-field mapping.
    """
    path = "data.profile.cuboids.dimensions"
    if is_unresolved(value):
        return []
    if not isinstance(value, dict):
        return [f"{path} must be a mapping with length, width, and height keys"]
    expected = {"length", "width", "height"}
    actual = set(value)
    if actual != expected:
        return [f"{path} keys must be exactly {sorted(expected)!r}"]
    diagnostics: list[str] = []
    for key, item in value.items():
        if not (is_unresolved(item) or _is_non_empty_string(item)):
            diagnostics.append(f"{path}.{key} must be a non-empty string or UNRESOLVED marker")
    return diagnostics


def _critical_gaps_section(report_text: str) -> str:
    """Return the report's Critical Gaps section body.

    Args:
        report_text: Full Markdown report text.

    Returns:
        Text between ``## Critical Gaps`` and the next level-2 header, or an
        empty string if the section is absent.
    """
    match = _CRITICAL_GAPS_HEADER_RE.search(report_text)
    if match is None:
        return ""
    body_start = report_text.find("\n", match.end())
    if body_start == -1:
        return ""
    next_header = report_text.find("\n## ", body_start + 1)
    if next_header == -1:
        return report_text[body_start + 1 :]
    return report_text[body_start + 1 : next_header]


def _critical_gap_keys(report_text: str) -> set[str]:
    """Return exact gap keys listed in the Critical Gaps section.

    Args:
        report_text: Full Markdown report text.

    Returns:
        Gap keys parsed from bullet lines or standalone key lines.
    """
    keys: set[str] = set()
    for line in _critical_gaps_section(report_text).splitlines():
        match = _CRITICAL_GAP_KEY_RE.match(line)
        if match is not None:
            keys.add(match.group("backtick") or match.group("quote") or match.group("bare"))
    return keys


def _has_report_header(report_text: str, header: str) -> bool:
    """Return whether ``report_text`` contains an exact Markdown header line.

    Args:
        report_text: Full Markdown report text.
        header: Header line to find.

    Returns:
        True when ``header`` appears as its own line.
    """
    return re.search(rf"^{re.escape(header)}\s*$", report_text, flags=re.MULTILINE) is not None


def _validate_generated_input(input_yaml: str | Path) -> tuple[dict[str, Any], list[str]]:
    """Validate the generated workflow input YAML.

    Args:
        input_yaml: Path to the generated YAML.

    Returns:
        Parsed YAML and diagnostics.

    Raises:
        ValidationError: If the YAML cannot be read or parsed by the shared
            launcher parser.
    """
    try:
        doc = parse_input(input_yaml)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValidationError([f"generated input YAML failed parse_input validation: {exc}"]) from exc

    diagnostics: list[str] = []
    data = doc.get("data")
    if not isinstance(data, dict):
        diagnostics.append("data block is missing or not a mapping")
        return doc, diagnostics

    for field in ("name", "path", "description"):
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            diagnostics.append(f"data.{field} is missing or blank")
        elif not isinstance(value, str):
            diagnostics.append(f"data.{field} must be a string, got {type(value).__name__}")

    docs = data.get("docs")
    if docs is not None:
        if isinstance(docs, str):
            pass
        elif isinstance(docs, list) and all(isinstance(item, str) for item in docs):
            pass
        else:
            diagnostics.append("data.docs must be a string or list of strings when present")

    diagnostics.extend(_placeholder_diagnostics("data.description", data.get("description")))

    profile = data.get("profile")
    if not isinstance(profile, dict):
        diagnostics.append("data.profile is missing or not a mapping")
    else:
        for section in REQUIRED_PROFILE_SECTIONS:
            if section not in profile:
                diagnostics.append(f"data.profile.{section} is missing")
                continue
            section_value = profile[section]
            if section in _REQUIRED_PROFILE_LIST_ENTRY_FIELDS:
                diagnostics.extend(_list_section_diagnostics(section, section_value))
                continue
            required_fields = _REQUIRED_PROFILE_SECTION_FIELDS.get(section)
            if required_fields is None:
                if not isinstance(section_value, list):
                    diagnostics.append(f"data.profile.{section} must be a list")
                continue
            if not isinstance(section_value, dict):
                diagnostics.append(f"data.profile.{section} must be a mapping")
                continue
            for field in required_fields:
                if field not in section_value:
                    diagnostics.append(f"data.profile.{section}.{field} is missing")
            for field_path in _REQUIRED_PROFILE_NESTED_FIELDS.get(section, ()):
                if _get_nested_field(section_value, field_path) is _MISSING:
                    diagnostics.append(f"data.profile.{section}.{field_path} is missing")
        diagnostics.extend(_profile_semantic_diagnostics(profile))
        diagnostics.extend(_placeholder_diagnostics("data.profile", profile))

    diagnostics.extend(_audit_scope_block_diagnostics("ncore", doc.get("ncore"), _REQUIRED_NCORE_FIELDS))
    diagnostics.extend(_audit_scope_block_diagnostics("nurec", doc.get("nurec"), _REQUIRED_NUREC_FIELDS))
    diagnostics.extend(_audit_scope_block_diagnostics("execution", doc.get("execution"), _REQUIRED_EXECUTION_FIELDS))

    return doc, diagnostics


def _validate_report(report_path: str | Path, unresolved_keys: list[str]) -> list[str]:
    """Validate the data-quality report shape and gap coverage.

    Args:
        report_path: Path to the Markdown report.
        unresolved_keys: Keys discovered in the generated input YAML.

    Returns:
        Diagnostics for missing sections or missing gap explanations.
    """
    path = Path(report_path)
    diagnostics: list[str] = []
    if not path.exists():
        return [f"data-quality report does not exist: {path}"]
    if not path.is_file():
        return [f"data-quality report is not a regular file: {path}"]
    try:
        text = path.read_text()
    except OSError as exc:
        return [f"failed to read data-quality report {path}: {exc}"]

    for section in _REQUIRED_REPORT_SECTIONS:
        if not _has_report_header(text, section):
            diagnostics.append(f"data-quality report is missing required section {section!r}")

    gaps = _critical_gap_keys(text)
    for key in unresolved_keys:
        if key not in gaps:
            diagnostics.append(
                f"generated input YAML has UNRESOLVED: {key!r}, but "
                "the report's Critical Gaps section does not mention it"
            )
    return diagnostics


def _load_profile(profile_path: str | Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Read the audit profile JSON.

    Args:
        profile_path: Path to ``audit_profile.json``.

    Returns:
        Parsed profile and diagnostics. The profile is ``None`` when reading or
        decoding failed.
    """
    path = Path(profile_path)
    if not path.exists():
        return None, [f"audit profile does not exist: {path}"]
    if not path.is_file():
        return None, [f"audit profile is not a regular file: {path}"]
    try:
        with path.open() as fh:
            profile = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"failed to read audit profile {path}: {exc}"]
    if not isinstance(profile, dict):
        return None, [f"audit profile must be a JSON object, got {type(profile).__name__}"]
    return profile, []


def _validate_profile(
    profile_path: str | Path,
    *,
    generated_doc: dict[str, Any],
    unresolved_keys: list[str],
) -> list[str]:
    """Validate the machine-readable audit profile.

    Args:
        profile_path: Path to ``audit_profile.json``.
        generated_doc: Parsed generated workflow input YAML.
        unresolved_keys: Keys discovered in ``data.profile``.

    Returns:
        Diagnostics for missing or inconsistent profile fields.
    """
    profile, diagnostics = _load_profile(profile_path)
    if profile is None:
        return diagnostics

    if profile.get("nurec_onboard_version") != NUREC_ONBOARD_VERSION:
        diagnostics.append(f"audit_profile.nurec_onboard_version must be {NUREC_ONBOARD_VERSION!r}")

    data = generated_doc.get("data")
    if not isinstance(data, dict):
        data = {}
    dataset = profile.get("dataset")
    if not isinstance(dataset, dict):
        diagnostics.append("audit_profile.dataset must be a mapping")
    else:
        for field in ("name", "path"):
            if dataset.get(field) != data.get(field):
                diagnostics.append(
                    f"audit_profile.dataset.{field}={dataset.get(field)!r} "
                    f"does not match generated input data.{field}={data.get(field)!r}"
                )

    minimum_facts = profile.get("minimum_facts")
    if not isinstance(minimum_facts, dict):
        diagnostics.append("audit_profile.minimum_facts must be a mapping")
    else:
        unresolved = minimum_facts.get("unresolved")
        if not isinstance(unresolved, list):
            diagnostics.append("audit_profile.minimum_facts.unresolved must be a list")
        else:
            missing = [key for key in unresolved_keys if key not in unresolved]
            if missing:
                diagnostics.append(
                    "audit_profile.minimum_facts.unresolved must include every "
                    f"UNRESOLVED key from the generated input YAML: missing {missing!r}"
                )
        if "confirmed" in minimum_facts and not isinstance(minimum_facts["confirmed"], list):
            diagnostics.append("audit_profile.minimum_facts.confirmed must be a list when present")

    for field in ("dataset_probe_hints",):
        if not isinstance(profile.get(field), dict):
            diagnostics.append(f"audit_profile.{field} must be a mapping")
    profile_copy = profile.get("profile")
    data_profile = data.get("profile")
    if profile_copy is not None and profile_copy != data_profile:
        diagnostics.append("audit_profile.profile must match generated input data.profile when present")
    return diagnostics


def validate_audit_artifacts(input_yaml: str | Path, report_md: str | Path, profile_json: str | Path) -> None:
    """Validate all artifacts emitted by ``audit-dataset``.

    Args:
        input_yaml: Generated `nurec-onboard` input YAML.
        report_md: Generated data-quality report Markdown file.
        profile_json: Generated `audit_profile.json`.

    Raises:
        ValidationError: If any artifact is malformed or inconsistent.
    """
    doc, diagnostics = _validate_generated_input(input_yaml)
    data = doc.get("data")
    if not isinstance(data, dict):
        data = {}
    unresolved_keys = (
        _extract_unresolved_keys(data.get("profile"))
        + _extract_unresolved_keys(doc.get("ncore"), root_path="ncore")
        + _extract_unresolved_keys(doc.get("nurec"), root_path="nurec")
        + _extract_unresolved_keys(doc.get("execution"), root_path="execution")
    )

    diagnostics.extend(_validate_report(report_md, unresolved_keys))
    diagnostics.extend(
        _validate_profile(
            profile_json,
            generated_doc=doc,
            unresolved_keys=unresolved_keys,
        )
    )

    if diagnostics:
        raise ValidationError(diagnostics)


def _validate_from_args(args: list[str]) -> None:
    """Validate CLI positional arguments.

    Args:
        args: Positional arguments after the program name.

    Raises:
        ValidationError: If the invocation or artifacts are invalid.
    """
    if args == ["--print-nurec-onboard-version"]:
        print(NUREC_ONBOARD_VERSION)
        return
    if len(args) != 3:
        raise ValidationError(
            [
                "expected exactly 3 positional arguments: "
                "<generated_input.yaml> <data_quality_report.md> <audit_profile.json>"
            ]
        )
    validate_audit_artifacts(args[0], args[1], args[2])


_USAGE = (
    'Usage: cd "$NUREC_ONBOARD_DIR" && bazel run '
    "//skills/nurec-onboard/validators:audit_dataset_validator -- "
    "<generated_input.yaml> <data_quality_report.md> <audit_profile.json>\n"
    '       cd "$NUREC_ONBOARD_DIR" && bazel run '
    "//skills/nurec-onboard/validators:audit_dataset_validator -- --print-nurec-onboard-version\n"
    "Validates the generated workflow input, audit report, and machine-readable "
    "audit profile emitted by audit-dataset.\n"
)


def main(argv: list[str]) -> int:
    """Run the audit artifact validator CLI.

    Args:
        argv: Full command-line argument vector, including program name.

    Returns:
        Standard validator exit code.
    """
    return run_validator(argv, _validate_from_args, usage=_USAGE, progname="audit_dataset_validator")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
