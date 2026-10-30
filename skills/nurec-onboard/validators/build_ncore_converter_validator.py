# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validator for build critical-parameter justifications.

Checks required fields, placeholders, authoritative-source blocks, and
worked examples. With ``--store``, verifies worked examples against the
NCore store.

Usage: ``cd "$NUREC_ONBOARD_DIR" && bazel run //skills/nurec-onboard/validators:build_ncore_converter_validator -- <summary.yaml> [--store <path>]``.
Exit codes: ``0`` pass, ``1`` fail, ``2`` malformed input.
"""

from __future__ import annotations

import re
import sys

from pathlib import Path
from typing import Any

from validators._validation_base import (
    ValidationError,
    as_dict,
    as_list,
    is_int,
    is_number,
    load_yaml,
    numeric_vec_diagnostics,
    step_validator_main,
    string_field_diagnostic,
)


_PACKAGE_NAME_RE = re.compile(r"^ncore_[a-z][a-z0-9_]*$")
_PLACEHOLDER_PACKAGE = "ncore_example"

REQUIRED_PARAMETERS: tuple[str, ...] = (
    # LiDAR model parameters. spinning_direction is column-array monotonicity.
    "spinning_direction",
    "row_elevations_rad",
    "column_azimuths_rad",
    "row_azimuth_offsets_rad",
    "spinning_lidar_model_frame_choice",
    # Extrinsic transforms.
    "T_lidar_rig",
    # Geometry and camera timing/model choices.
    "cuboid_dim_axis_ordering",
    "camera_timestamp_meaning",
    "T_camera_rig",
    "camera_frame_convention",
    "rig_frame_convention",
    # Timestamps + shutter.
    "timestamp_unit",
    "shutter_type_and_direction",
    # Data-state assumptions.
    "lidar_raw_format",
    "lidar_motion_compensation",
    "image_distortion_state",
    # Camera masks and intrinsics.
    "camera_model_family",
    "camera_static_mask_handling",
    # Pose trajectory + LiDAR point frame.
    "pose_trajectory_source",
    "lidar_point_frame",
    "intensity_normalization_method",
    # Annotations.
    "cuboid_centroid_convention",
    "cuboid_world_anchored_frame",
    "lidar_frame_timestamp_meaning",
    # Inventory, provenance, and run-scope decisions.
    "sensor_inventory_decisions",
    "lidar_model_source",
    "nurec_lidar_selection",
    "sequence_frame_range_decision",
)

REQUIRED_ENTRY_FIELDS: tuple[str, ...] = (
    "parameter",
    "value",
    "evidence",
    "wrong_value_consequence",
)

_STRING_REQUIRED_ENTRY_FIELDS: frozenset[str] = frozenset({"parameter", "evidence", "wrong_value_consequence"})
"""Entry fields that must be non-blank, placeholder-free strings."""

PARAMETERS_REQUIRING_WORKED_EXAMPLE: frozenset[str] = frozenset(
    {
        "T_lidar_rig",
        "spinning_lidar_model_frame_choice",
        "T_camera_rig",
        "camera_model_family",
        "sequence_frame_range_decision",
    }
)
"""Parameters that need store-verifiable ``worked_example`` blocks."""

PARAMETERS_REQUIRING_AUTHORITATIVE_SOURCE: frozenset[str] = frozenset({"T_lidar_rig", "T_camera_rig"})
"""Parameters that must record declared vs. loaded extrinsic source."""

PARAMETERS_ALLOWING_EMPTY_VALUE: frozenset[str] = frozenset({"camera_static_mask_handling"})
"""Critical parameters where an empty ``value:`` container is a documented legitimate state.

``camera_static_mask_handling`` permits ``value: []`` when the dataset ships no per-camera
mask files (see ``critical-parameters.md``); ``evidence`` must cite the probe enumeration
and ``wrong_value_consequence`` must reference the NuRec quality impact of writing ``{}``.
"""

AUTHORITATIVE_SOURCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "declared_in_input",
    "used_by_converter",
    "source_match",
    "verification",
)
"""Required keys on ``authoritative_source``."""

AUTHORITATIVE_SOURCE_VERIFICATION_REQUIRED_FIELDS: tuple[str, ...] = (
    "per_image_data_available",
    "summary",
)
"""Required keys on ``authoritative_source.verification``."""

WORKED_EXAMPLE_FIELDS: tuple[str, ...] = (
    "column_index",
    "row_index",
    "d_rig_expected",
    "tolerance_deg",
)
"""Fields for a LiDAR ``(row, column) -> d_rig`` prediction."""
CAMERA_WORKED_EXAMPLE_REQUIRED_FIELDS: tuple[str, ...] = (
    "camera_id",
    "axis_in_camera",
    "axis_in_rig_expected",
    "tolerance_deg",
)
"""Required rotation-check fields for each camera worked example."""

CAMERA_WORKED_EXAMPLE_TRANSLATION_FIELDS: tuple[str, ...] = (
    "position_in_rig_expected",
    "position_tolerance_m",
    "position_in_rig_expected_derivation",
)
"""All-or-none translation-check fields for camera worked examples."""

CAMERA_MODEL_WORKED_EXAMPLE_REQUIRED_FIELDS: tuple[str, ...] = (
    "camera_id",
    "model",
    "camera_rays",
    "expected_image_points",
    "tolerance_px",
    "sample_source",
)
"""Required projection-check fields for each camera-model worked example."""

CAMERA_MODEL_NAMES: tuple[str, ...] = ("opencv_pinhole", "opencv_fisheye", "ftheta")
"""Supported NCore camera model family names for ``camera_model_family``."""

SEQUENCE_FRAME_RANGE_COUNT_FIELDS: tuple[str, ...] = (
    "source_track_count_in_coverage",
    "stored_track_count",
    "source_observation_count_in_coverage",
    "stored_observation_count",
)
"""Cuboid counts required on ``sequence_frame_range_decision.value``.

Tracks are counted by exact identity: cuboid component, track id, label
source, and source version.
"""

CUBOID_SAMPLED_TRACK_FIELDS: tuple[str, ...] = (
    "component",
    "track_id",
    "label_source",
    "source_version",
    "source_timestamps_us",
    "sample_source",
)
"""Required fields for each ``sequence_frame_range_decision`` sampled track."""

CuboidTrackKey = tuple[str, str, str, str | None]
"""``(cuboid component, track_id, label source name, source version)``."""


def _as_list_of_entries(value: Any) -> list[dict[str, Any]]:
    """Coerce the parsed YAML value into a list of mapping entries.

    Args:
        value: Parsed ``critical_parameter_justifications`` value.

    Returns:
        List of justification mappings.

    Raises:
        ValidationError: If ``value`` is not a list of mappings.
    """
    if not isinstance(value, list):
        raise ValidationError([f"critical_parameter_justifications must be a YAML list, got {type(value).__name__}"])
    entries: list[dict[str, Any]] = []
    for i, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValidationError(
                [f"critical_parameter_justifications[{i}] must be a mapping, got {type(entry).__name__}"]
            )
        entries.append(entry)
    return entries


def validate_entries(entries: list[dict[str, Any]]) -> None:
    """Enforce structural completeness of the justifications list.

    Args:
        entries: Parsed critical-parameter justification entries.

    Returns:
        None.

    Raises:
        ValidationError: If required parameters, fields, worked examples, or
            authoritative-source blocks are missing or malformed.
    """
    diagnostics: list[str] = []

    by_name: dict[str, dict[str, Any]] = {}
    for i, entry in enumerate(entries):
        name = entry.get("parameter")
        if name is None or (isinstance(name, str) and not name.strip()):
            diagnostics.append(f"critical_parameter_justifications[{i}] is missing a 'parameter' key")
            continue
        if not isinstance(name, str):
            diagnostics.append(
                f"critical_parameter_justifications[{i}].parameter must be a string, got {type(name).__name__}"
            )
            continue
        if name in by_name:
            diagnostics.append(f"critical_parameter_justifications contains duplicate entries for parameter {name!r}")
            continue
        by_name[name] = entry

    for required in REQUIRED_PARAMETERS:
        if required not in by_name:
            diagnostics.append(
                f"Missing justification for required parameter "
                f"{required!r} (see nurec-onboard/steps/build-ncore-converter/"
                f"critical-parameters.md for the entry schema and "
                f"required evidence)"
            )
            continue
        entry = by_name[required]
        for field in REQUIRED_ENTRY_FIELDS:
            if field in _STRING_REQUIRED_ENTRY_FIELDS:
                diag = string_field_diagnostic(f"{required!r}: {field!r}", entry.get(field))
            else:
                diag = _value_field_diagnostic(f"{required!r}: {field!r}", entry.get(field), parameter=required)
            if diag is not None:
                diagnostics.append(diag)
        if required in PARAMETERS_REQUIRING_WORKED_EXAMPLE:
            diagnostics.extend(_validate_worked_example_structure(required, entry))
        if required in PARAMETERS_REQUIRING_AUTHORITATIVE_SOURCE:
            diagnostics.extend(_validate_authoritative_source(required, entry))

    if diagnostics:
        raise ValidationError(diagnostics)


def _value_field_diagnostic(prefix: str, value: Any, *, parameter: str | None = None) -> str | None:
    """Validate a critical-parameter ``value:`` field, return ``None`` if OK.

    ``value:`` is schema-untyped per ``critical-parameters.md`` 'Entry
    schema' (``value: <any>``), so each parameter uses its natural YAML
    shape (string, list, mapping, matrix) and downstream tooling does
    not need to re-parse block-literal strings.

    Args:
        prefix: User-facing diagnostic path for the field (e.g.
            ``"'T_lidar_rig': 'value'"``).
        value: Parsed ``value:`` field from the justification entry.
        parameter: Critical-parameter name owning this value. When the
            parameter is listed in ``PARAMETERS_ALLOWING_EMPTY_VALUE``,
            an empty list passes the shape check (matching the
            documented ``value: []`` shape); an empty dict is still
            rejected so ``value: {}`` -- which the parameter's
            ``wrong_value_consequence`` cites as the failure mode --
            does not get the same exception. ``None`` (default)
            enforces the empty-container rejection unconditionally.

    Returns:
        Diagnostic line when ``value`` is missing, blank, or carries a
        placeholder marker; ``None`` when ``value`` is present and
        non-empty (any structural type passes -- per-parameter coverage
        checks handle finer-grained shape constraints separately).
    """
    if value is None:
        return (
            f"{prefix} is missing or null (declare the chosen value; "
            f"see critical-parameters.md 'Entry schema' -- the field "
            f"accepts any YAML type: string, number, list, mapping)"
        )
    if isinstance(value, str):
        return string_field_diagnostic(prefix, value)
    if isinstance(value, (list, dict)) and len(value) == 0:
        if isinstance(value, list) and parameter in PARAMETERS_ALLOWING_EMPTY_VALUE:
            return None
        return (
            f"{prefix} is an empty {type(value).__name__} (declare the "
            f"chosen value; see critical-parameters.md 'Entry schema')"
        )
    return None


def _validate_authoritative_source(parameter: str, entry: dict[str, Any]) -> list[str]:
    """Validate ``authoritative_source`` shape and placeholder-free fields.

    Args:
        parameter: Critical parameter name owning the block.
        entry: Parsed critical-parameter justification entry.

    Returns:
        Diagnostics for missing, malformed, or placeholder-filled
        ``authoritative_source`` fields.
    """
    diagnostics: list[str] = []
    block = entry.get("authoritative_source")
    if block is None:
        diagnostics.append(
            f"{parameter!r}: 'authoritative_source' is missing "
            f"(see conventions.md 'When the input YAML names an "
            f"authoritative source')."
        )
        return diagnostics
    if not isinstance(block, dict):
        diagnostics.append(f"{parameter!r}: 'authoritative_source' must be a mapping, got {type(block).__name__}")
        return diagnostics
    for field in AUTHORITATIVE_SOURCE_REQUIRED_FIELDS:
        if field not in block:
            diagnostics.append(f"{parameter!r}.authoritative_source: {field!r} is missing")
            continue
        raw = block[field]
        if field == "source_match":
            if not isinstance(raw, bool):
                diagnostics.append(
                    f"{parameter!r}.authoritative_source.source_match must "
                    f"be a boolean (true/false), got {type(raw).__name__}"
                )
            continue
        if field == "verification":
            diagnostics.extend(_validate_authoritative_source_verification(parameter, raw))
            continue
        # String fields: declared_in_input, used_by_converter.
        diag = string_field_diagnostic(
            f"{parameter!r}.authoritative_source.{field}",
            raw,
            blank_suffix=(
                "(when the input declares no authoritative source, set "
                "'declared_in_input: \"none -- no authoritative source "
                "declared\"' rather than leaving the field empty)"
            ),
        )
        if diag is not None:
            diagnostics.append(diag)
    return diagnostics


def _validate_authoritative_source_verification(parameter: str, block: Any) -> list[str]:
    """Validate ``authoritative_source.verification`` shape.

    Args:
        parameter: Critical parameter name owning the block.
        block: Parsed ``authoritative_source.verification`` value.

    Returns:
        Diagnostics for missing, malformed, or placeholder-filled verification
        fields.
    """
    diagnostics: list[str] = []
    if block is None:
        diagnostics.append(
            f"{parameter!r}.authoritative_source: 'verification' is missing "
            f"(see conventions.md 'When the input YAML names an "
            f"authoritative source')."
        )
        return diagnostics
    if not isinstance(block, dict):
        diagnostics.append(
            f"{parameter!r}.authoritative_source.verification must be a mapping, got {type(block).__name__}"
        )
        return diagnostics
    for field in AUTHORITATIVE_SOURCE_VERIFICATION_REQUIRED_FIELDS:
        if field not in block:
            diagnostics.append(f"{parameter!r}.authoritative_source.verification: {field!r} is missing")
            continue
        raw = block[field]
        if field == "per_image_data_available":
            if not isinstance(raw, bool):
                diagnostics.append(
                    f"{parameter!r}.authoritative_source.verification."
                    f"per_image_data_available must be a boolean "
                    f"(true/false), got {type(raw).__name__}"
                )
            continue
        diag = string_field_diagnostic(
            f"{parameter!r}.authoritative_source.verification.{field}",
            raw,
            blank_suffix=(
                "(when no per-image pose data is available, set summary "
                "to 'no per-image pose data available for this sensor; "
                "trusted static source as the only available evidence' "
                "rather than leaving the field empty)"
            ),
        )
        if diag is not None:
            diagnostics.append(diag)
    return diagnostics


def _validate_worked_example_structure(parameter: str, entry: dict[str, Any]) -> list[str]:
    """Validate ``worked_example`` presence and shape.

    Args:
        parameter: Critical parameter name owning the worked example.
        entry: Parsed critical-parameter justification entry.

    Returns:
        Diagnostics for missing or malformed worked-example fields.
    """
    if parameter == "T_camera_rig":
        return _validate_worked_example_structure_camera(parameter, entry)
    if parameter == "camera_model_family":
        return _validate_worked_example_structure_camera_model(parameter, entry)
    if parameter == "sequence_frame_range_decision":
        return _validate_worked_example_structure_cuboid_tracks(parameter, entry)
    diagnostics: list[str] = []
    worked = entry.get("worked_example")
    if worked is None:
        diagnostics.append(
            f"{parameter!r}: 'worked_example' is missing (see SKILL.md 'critical-parameter justifications')."
        )
        return diagnostics
    if not isinstance(worked, dict):
        diagnostics.append(f"{parameter!r}: 'worked_example' must be a mapping, got {type(worked).__name__}")
        return diagnostics
    for field in WORKED_EXAMPLE_FIELDS:
        if field not in worked:
            diagnostics.append(f"{parameter!r}.worked_example: {field!r} is missing")
    d_rig = worked.get("d_rig_expected")
    if d_rig is not None:
        diagnostics.extend(
            numeric_vec_diagnostics(
                f"{parameter!r}.worked_example.d_rig_expected",
                d_rig,
                hint="(e.g. [1.0, 0.0, 0.0])",
            )
        )
    for numeric_field in ("column_index", "row_index"):
        v = worked.get(numeric_field)
        if v is not None and not is_int(v):
            diagnostics.append(
                f"{parameter!r}.worked_example.{numeric_field} must be an integer, got {type(v).__name__}"
            )
    tol = worked.get("tolerance_deg")
    if tol is not None and not is_number(tol):
        diagnostics.append(f"{parameter!r}.worked_example.tolerance_deg must be numeric, got {type(tol).__name__}")
    lidar_id = worked.get("lidar_id")
    if lidar_id is not None and (not isinstance(lidar_id, str) or not lidar_id.strip()):
        diagnostics.append(
            f"{parameter!r}.worked_example.lidar_id must be a non-blank "
            f"string when present, got {type(lidar_id).__name__}"
        )
    return diagnostics


def _validate_worked_example_structure_cuboid_tracks(parameter: str, entry: dict[str, Any]) -> list[str]:
    """Validate cuboid counts and sampled tracks on ``sequence_frame_range_decision``.

    Args:
        parameter: Critical parameter name owning the worked example.
        entry: Parsed critical-parameter justification entry.

    Returns:
        Diagnostics for malformed counts, source tracks neither stored nor
        excluded, a recorded observation count over the bound, missing
        per-track check evidence, or malformed sampled tracks.
    """
    diagnostics: list[str] = []
    value = as_dict(entry.get("value"))
    counts: dict[str, int] = {}
    for field in SEQUENCE_FRAME_RANGE_COUNT_FIELDS:
        count = value.get(field)
        if not is_int(count) or count < 0:
            diagnostics.append(f"{parameter!r}.value.{field} must be a non-negative integer, got {count!r}")
        else:
            counts[field] = count

    excluded = value.get("excluded_tracks")
    excluded_count: int | None = None
    if not isinstance(excluded, dict) or not all(
        isinstance(reason, str) and reason.strip() and is_int(n) and n > 0 for reason, n in excluded.items()
    ):
        diagnostics.append(
            f"{parameter!r}.value.excluded_tracks must map each exclusion reason to a positive track count "
            f"({{}} when none), got {excluded!r}"
        )
    else:
        excluded_count = sum(excluded.values())

    if len(counts) == len(SEQUENCE_FRAME_RANGE_COUNT_FIELDS) and excluded_count is not None:
        source_tracks, stored_tracks = counts["source_track_count_in_coverage"], counts["stored_track_count"]
        accounted = stored_tracks + excluded_count
        if accounted < source_tracks:
            diagnostics.append(
                f"{parameter!r}.value: {source_tracks - accounted} source track(s) in coverage are neither stored "
                f"nor excluded (source_track_count_in_coverage={source_tracks}, stored_track_count={stored_tracks}, "
                f"excluded={excluded_count})"
            )
        elif accounted > source_tracks:
            diagnostics.append(
                f"{parameter!r}.value: stored plus excluded tracks exceed the source by {accounted - source_tracks} "
                f"(source_track_count_in_coverage={source_tracks}, stored_track_count={stored_tracks}, "
                f"excluded={excluded_count})"
            )
    if len(counts) == len(SEQUENCE_FRAME_RANGE_COUNT_FIELDS):
        bound = counts["source_observation_count_in_coverage"] + 2 * counts["stored_track_count"]
        if counts["stored_observation_count"] > bound:
            diagnostics.append(
                f"{parameter!r}.value.stored_observation_count={counts['stored_observation_count']} exceeds "
                f"source_observation_count_in_coverage + 2 * stored_track_count = {bound}; the converter "
                "stored observations the source does not have"
            )

    worked = entry.get("worked_example")
    if not isinstance(worked, dict):
        diagnostics.append(f"{parameter!r}: 'worked_example' must be a mapping with tracks_checked and sampled_tracks")
        return diagnostics
    tracks_checked = worked.get("tracks_checked")
    if not is_int(tracks_checked):
        diagnostics.append(
            f"{parameter!r}.worked_example.tracks_checked must be the number of tracks the converter passed "
            f"through check_cuboid_track_timestamps, got {tracks_checked!r}"
        )
    elif "stored_track_count" in counts and tracks_checked != counts["stored_track_count"]:
        diagnostics.append(
            f"{parameter!r}.worked_example.tracks_checked={tracks_checked} but stored_track_count="
            f"{counts['stored_track_count']}; check_cuboid_track_timestamps must run on every stored track"
        )

    sampled = worked.get("sampled_tracks")
    if not isinstance(sampled, list):
        diagnostics.append(f"{parameter!r}.worked_example.sampled_tracks must be a list of source tracks")
        return diagnostics
    if not sampled and counts.get("stored_track_count", 0) > 0:
        diagnostics.append(
            f"{parameter!r}.worked_example.sampled_tracks is empty but stored_track_count > 0; "
            "record at least one source track"
        )
    for i, track in enumerate(sampled):
        path = f"{parameter!r}.worked_example.sampled_tracks[{i}]"
        if not isinstance(track, dict):
            diagnostics.append(f"{path} must be a mapping, got {type(track).__name__}")
            continue
        for field in CUBOID_SAMPLED_TRACK_FIELDS:
            if field not in track:
                diagnostics.append(f"{path}: {field!r} is missing")
        for field in ("component", "track_id", "label_source", "sample_source"):
            if field in track:
                diag = string_field_diagnostic(f"{path}.{field}", track[field])
                if diag is not None:
                    diagnostics.append(diag)
        version = track.get("source_version")
        if version is not None and (not isinstance(version, str) or not version.strip()):
            diagnostics.append(f"{path}.source_version must be a non-blank string or null")
        source_us = track.get("source_timestamps_us")
        if "source_timestamps_us" in track:
            if not isinstance(source_us, list) or not source_us or not all(is_int(t) for t in source_us):
                diagnostics.append(f"{path}.source_timestamps_us must be a non-empty list of integers")
            elif len(set(source_us)) != len(source_us):
                diagnostics.append(f"{path}.source_timestamps_us repeats a timestamp")
    return diagnostics


def _expected_cuboid_track_timestamps(source_us: list[int], coverage_lo: int, coverage_hi: int) -> list[int]:
    """Return the timestamps a converter must store for one source track.

    Every source timestamp inside ``[coverage_lo, coverage_hi]``, plus each
    coverage boundary the source span crosses (see ``conventions.md``
    "Tracks lost to interval clipping").
    """
    ordered = sorted(source_us)
    expected = {t for t in ordered if coverage_lo <= t <= coverage_hi}
    if ordered and ordered[0] < coverage_lo <= ordered[-1]:
        expected.add(coverage_lo)
    if ordered and ordered[0] <= coverage_hi < ordered[-1]:
        expected.add(coverage_hi)
    return sorted(expected)


def _cuboid_track_store_diagnostics(
    parameter: str,
    entry: dict[str, Any],
    stored: dict[CuboidTrackKey, list[int]],
    coverage: tuple[int, int] | None,
) -> list[str]:
    """Compare recorded cuboid counts and sampled tracks against stored observations.

    Sampled tracks are a representative sample, not an exhaustive one. An
    unsampled track is covered here only by the aggregate counts; its exact
    timestamps are checked by ``check_cuboid_track_timestamps`` during
    conversion.

    Args:
        parameter: Critical parameter name owning the worked example.
        entry: Parsed ``sequence_frame_range_decision`` entry.
        stored: Stored observation timestamps keyed by exact track identity.
        coverage: ``(first frame start, last frame end)`` across stored
            sensors, or ``None`` when the store has no sensor frames.

    Returns:
        Diagnostics for recorded counts that disagree with the store, a store
        over the observation bound, and sampled tracks whose stored timestamps
        differ from their source selection.
    """
    diagnostics: list[str] = []
    value = as_dict(entry.get("value"))
    stored_observations = sum(len(ts) for ts in stored.values())

    for field, actual in (("stored_track_count", len(stored)), ("stored_observation_count", stored_observations)):
        recorded = value.get(field)
        if is_int(recorded) and recorded != actual:
            diagnostics.append(f"{parameter!r}.value.{field}={recorded} but the store holds {actual}")
    source_in_coverage = value.get("source_observation_count_in_coverage")
    if is_int(source_in_coverage) and stored_observations > source_in_coverage + 2 * len(stored):
        diagnostics.append(
            f"store holds {stored_observations} cuboid observations, more than "
            f"source_observation_count_in_coverage + 2 * stored tracks = "
            f"{source_in_coverage + 2 * len(stored)}; the converter stored observations the source does not have"
        )

    sampled = as_list(as_dict(entry.get("worked_example")).get("sampled_tracks"))
    if not sampled:
        return diagnostics
    if coverage is None:
        diagnostics.append(f"{parameter!r}: store has no sensor frames to define cuboid coverage")
        return diagnostics
    coverage_lo, coverage_hi = coverage
    for i, track in enumerate(sampled):
        if not isinstance(track, dict):
            continue
        component, track_id, label_source = track.get("component"), track.get("track_id"), track.get("label_source")
        source_us = track.get("source_timestamps_us")
        if not (isinstance(component, str) and isinstance(track_id, str) and isinstance(label_source, str)):
            continue
        if not isinstance(source_us, list):
            continue
        key = (component, track_id, label_source, track.get("source_version"))
        path = (
            f"{parameter!r}.worked_example.sampled_tracks[{i}] "
            f"({key[0]}:{key[1]} {key[2]}@{'any' if key[3] is None else key[3]})"
        )
        stored_us = sorted(stored.get(key, []))
        expected_us = _expected_cuboid_track_timestamps(source_us, coverage_lo, coverage_hi)
        if stored_us != expected_us:
            extra = sorted(set(stored_us) - set(expected_us))
            missing = sorted(set(expected_us) - set(stored_us))
            repeated = len(stored_us) - len(set(stored_us))
            diagnostics.append(
                f"{path}: stores {len(stored_us)} observations but its source selection is {len(expected_us)} "
                f"({len(extra)} not from the source, {len(missing)} missing, {repeated} repeated); "
                f"first extra {extra[:3]}, first missing {missing[:3]}"
            )
    return diagnostics


def _numeric_matrix_diagnostics(path: str, value: Any, *, columns: int) -> list[str]:
    """Validate a numeric matrix with a fixed column count.

    Args:
        path: Diagnostic path to ``value``.
        value: Parsed matrix value.
        columns: Required number of columns.

    Returns:
        Diagnostics for malformed, non-numeric, empty, or ragged matrices.
    """
    if not isinstance(value, list):
        return [f"{path} must be a list of rows"]
    if not value:
        return [f"{path} must contain at least one row"]
    diagnostics: list[str] = []
    for i, row in enumerate(value):
        row_path = f"{path}[{i}]"
        if not isinstance(row, list):
            diagnostics.append(f"{row_path} must be a list")
            continue
        if len(row) != columns:
            diagnostics.append(f"{row_path} must contain exactly {columns} numeric values, got {len(row)}")
            continue
        for j, item in enumerate(row):
            if not is_number(item):
                diagnostics.append(f"{row_path}[{j}] must be numeric, got {type(item).__name__}")
    return diagnostics


def _camera_model_value_models(entry: dict[str, Any]) -> dict[str, set[str]]:
    """Return camera model families declared by ``camera_model_family.value``.

    Args:
        entry: Parsed ``camera_model_family`` justification entry.

    Returns:
        Mapping from camera ID to declared model names. Other value shapes
        return an empty mapping and skip consistency cross-checking.
    """
    raw_value = entry.get("value")
    if not isinstance(raw_value, list):
        return {}
    models: dict[str, set[str]] = {}
    for item in raw_value:
        if not isinstance(item, dict):
            continue
        camera_id = item.get("camera_id")
        model = item.get("model")
        if isinstance(camera_id, str) and camera_id.strip() and isinstance(model, str) and model.strip():
            models.setdefault(camera_id, set()).add(model)
    return models


def _camera_model_value_diagnostics(parameter: str, entry: dict[str, Any]) -> list[str]:
    """Validate the per-camera ``{camera_id, model}`` shape of ``value``.

    Args:
        parameter: Critical parameter name owning the entry.
        entry: Parsed ``camera_model_family`` justification entry.

    Returns:
        Diagnostics for a non-list ``value`` (a present-but-wrong shape that
        the generic ``value`` check passes), and for ``value`` list entries
        that are not mappings, omit a non-blank ``camera_id``, repeat a
        ``camera_id``, or omit a valid ``model``. A missing/``None`` ``value``
        is left to the generic ``value`` check, which reports it.
    """
    raw_value = entry.get("value")
    diagnostics: list[str] = []
    if not isinstance(raw_value, list):
        if raw_value is not None:
            diagnostics.append(f"{parameter}.value must be a list, got {type(raw_value).__name__}")
        return diagnostics
    seen_ids: set[str] = set()
    for i, item in enumerate(raw_value):
        path = f"{parameter}.value[{i}]"
        if not isinstance(item, dict):
            diagnostics.append(f"{path} must be a mapping, got {type(item).__name__}")
            continue
        camera_id = item.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id.strip():
            diagnostics.append(f"{path}.camera_id must be a non-blank string")
        elif camera_id in seen_ids:
            diagnostics.append(f"{parameter}.value contains duplicate camera_id={camera_id!r}")
        else:
            seen_ids.add(camera_id)
        model = item.get("model")
        if model is None:
            diagnostics.append(f"{path}.model is missing (one of {', '.join(CAMERA_MODEL_NAMES)})")
        elif model not in CAMERA_MODEL_NAMES:
            diagnostics.append(f"{path}.model must be one of {', '.join(CAMERA_MODEL_NAMES)}; got {model!r}")
    return diagnostics


def _camera_model_mismatch_diagnostic(
    parameter: str,
    camera_id: str,
    declared: set[str],
    observed: set[str],
    *,
    observed_path: str,
) -> str | None:
    """Return a model-family mismatch diagnostic for one camera, if needed."""
    if declared == observed:
        return None
    return (
        f"{parameter}.value declares {camera_id!r} model {sorted(declared)!r} "
        f"but {observed_path} declares {sorted(observed)!r}"
    )


def _validate_worked_example_structure_camera_model(parameter: str, entry: dict[str, Any]) -> list[str]:
    """Validate per-camera projection worked examples for camera model family.

    Args:
        parameter: Critical parameter name owning the worked example.
        entry: Parsed ``camera_model_family`` justification entry.

    Returns:
        Diagnostics for missing or malformed projection samples.
    """
    diagnostics: list[str] = _camera_model_value_diagnostics(parameter, entry)
    worked = entry.get("worked_example")
    if worked is None:
        diagnostics.append(
            f"{parameter!r}: 'worked_example' is missing (see SKILL.md 'critical-parameter justifications')."
        )
        return diagnostics
    if not isinstance(worked, dict):
        diagnostics.append(f"{parameter!r}: 'worked_example' must be a mapping, got {type(worked).__name__}")
        return diagnostics
    per_camera = worked.get("per_camera")
    if per_camera is None:
        diagnostics.append(
            f"{parameter!r}.worked_example: 'per_camera' is missing "
            f"(required: a list of projection-check entries, one per camera in value)."
        )
        return diagnostics
    if not isinstance(per_camera, list):
        diagnostics.append(f"{parameter!r}.worked_example.per_camera must be a list, got {type(per_camera).__name__}")
        return diagnostics
    if not per_camera:
        diagnostics.append(f"{parameter!r}.worked_example.per_camera must contain at least one entry")
        return diagnostics

    seen_ids: set[str] = set()
    observed_models: dict[str, set[str]] = {}
    for i, cam in enumerate(per_camera):
        if not isinstance(cam, dict):
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] must be a mapping, got {type(cam).__name__}"
            )
            continue
        for field in CAMERA_MODEL_WORKED_EXAMPLE_REQUIRED_FIELDS:
            if field not in cam:
                diagnostics.append(f"{parameter!r}.worked_example.per_camera[{i}]: {field!r} is missing")
        cam_id = cam.get("camera_id")
        is_duplicate = False
        if cam_id is not None:
            if not isinstance(cam_id, str) or not cam_id.strip():
                diagnostics.append(
                    f"{parameter!r}.worked_example.per_camera[{i}].camera_id "
                    f"must be a non-blank string, got {type(cam_id).__name__}"
                )
            elif cam_id in seen_ids:
                is_duplicate = True
                diagnostics.append(f"{parameter!r}.worked_example.per_camera contains duplicate camera_id={cam_id!r}")
            else:
                seen_ids.add(cam_id)
        model = cam.get("model")
        if model is not None:
            if model not in CAMERA_MODEL_NAMES:
                diagnostics.append(
                    f"{parameter!r}.worked_example.per_camera[{i}].model must be one of "
                    f"{', '.join(CAMERA_MODEL_NAMES)}; got {model!r}"
                )
            elif isinstance(cam_id, str) and cam_id.strip() and not is_duplicate:
                # Only the first occurrence contributes to the model cross-check;
                # a duplicate entry is already reported above, so accumulating it
                # would add a spurious mismatch diagnostic for the same camera.
                observed_models.setdefault(cam_id, set()).add(model)
        if "camera_rays" in cam:
            diagnostics.extend(
                _numeric_matrix_diagnostics(
                    f"{parameter!r}.worked_example.per_camera[{i}].camera_rays",
                    cam["camera_rays"],
                    columns=3,
                )
            )
        if "expected_image_points" in cam:
            diagnostics.extend(
                _numeric_matrix_diagnostics(
                    f"{parameter!r}.worked_example.per_camera[{i}].expected_image_points",
                    cam["expected_image_points"],
                    columns=2,
                )
            )
        rays = cam.get("camera_rays")
        points = cam.get("expected_image_points")
        if isinstance(rays, list) and isinstance(points, list) and len(rays) != len(points):
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}]: camera_rays and "
                f"expected_image_points must have the same row count"
            )
        if "tolerance_px" in cam:
            # An absent key is reported by the required-fields check above; a
            # present-but-null value is not (it is in `cam`), so flag it here
            # rather than letting float(None) raise at store-check time.
            tol = cam.get("tolerance_px")
            if not is_number(tol):
                diagnostics.append(
                    f"{parameter!r}.worked_example.per_camera[{i}].tolerance_px must be numeric, got {type(tol).__name__}"
                )
            elif tol < 0:
                diagnostics.append(f"{parameter!r}.worked_example.per_camera[{i}].tolerance_px must be non-negative")
        if "sample_source" in cam:
            diag = string_field_diagnostic(
                f"{parameter!r}.worked_example.per_camera[{i}].sample_source",
                cam["sample_source"],
            )
            if diag is not None:
                diagnostics.append(diag)

    declared_models = _camera_model_value_models(entry)
    if declared_models:
        declared_cameras = set(declared_models)
        missing = declared_cameras - seen_ids
        extra = seen_ids - declared_cameras
        if missing:
            diagnostics.append(
                f"{parameter}.value declares cameras {sorted(missing)!r} but "
                f"worked_example.per_camera does not cover them"
            )
        if extra:
            diagnostics.append(
                f"{parameter}.worked_example.per_camera covers cameras {sorted(extra)!r} not declared in {parameter}.value"
            )
        for camera_id in sorted(declared_cameras & seen_ids):
            declared = declared_models.get(camera_id, set())
            observed = observed_models.get(camera_id, set())
            if not observed:
                # A missing or unknown per_camera model is already reported
                # above; skip the cross-check to avoid a confusing "declares []".
                continue
            diag = _camera_model_mismatch_diagnostic(
                parameter,
                camera_id,
                declared,
                observed,
                observed_path="worked_example.per_camera",
            )
            if diag is not None:
                diagnostics.append(diag)
    return diagnostics


def _validate_worked_example_structure_camera(parameter: str, entry: dict[str, Any]) -> list[str]:
    """Validate per-camera worked-example shape for ``T_camera_rig``.

    Args:
        parameter: Critical parameter name owning the worked example.
        entry: Parsed ``T_camera_rig`` justification entry.

    Returns:
        Diagnostics for missing or malformed per-camera worked-example fields.
    """
    diagnostics: list[str] = []
    worked = entry.get("worked_example")
    if worked is None:
        diagnostics.append(
            f"{parameter!r}: 'worked_example' is missing (see SKILL.md 'critical-parameter justifications')."
        )
        return diagnostics
    if not isinstance(worked, dict):
        diagnostics.append(f"{parameter!r}: 'worked_example' must be a mapping, got {type(worked).__name__}")
        return diagnostics
    per_camera = worked.get("per_camera")
    if per_camera is None:
        diagnostics.append(
            f"{parameter!r}.worked_example: 'per_camera' is missing "
            f"(required: a list of {{camera_id, axis_in_camera, "
            f"axis_in_rig_expected, tolerance_deg, ...}} entries, one "
            f"per camera the converter writes a T_camera_rig for)."
        )
        return diagnostics
    if not isinstance(per_camera, list):
        diagnostics.append(f"{parameter!r}.worked_example.per_camera must be a list, got {type(per_camera).__name__}")
        return diagnostics
    if not per_camera:
        diagnostics.append(f"{parameter!r}.worked_example.per_camera must contain at least one entry")
        return diagnostics

    seen_ids: set[str] = set()
    for i, cam in enumerate(per_camera):
        if not isinstance(cam, dict):
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] must be a mapping, got {type(cam).__name__}"
            )
            continue
        for field in CAMERA_WORKED_EXAMPLE_REQUIRED_FIELDS:
            if field not in cam:
                diagnostics.append(f"{parameter!r}.worked_example.per_camera[{i}]: {field!r} is missing")
        cam_id = cam.get("camera_id")
        if cam_id is not None:
            if not isinstance(cam_id, str) or not cam_id.strip():
                diagnostics.append(
                    f"{parameter!r}.worked_example.per_camera[{i}].camera_id "
                    f"must be a non-blank string, got {type(cam_id).__name__}"
                )
            elif cam_id in seen_ids:
                diagnostics.append(f"{parameter!r}.worked_example.per_camera contains duplicate camera_id={cam_id!r}")
            else:
                seen_ids.add(cam_id)
        for vec_field in ("axis_in_camera", "axis_in_rig_expected"):
            v = cam.get(vec_field)
            if v is None:
                continue
            diagnostics.extend(
                numeric_vec_diagnostics(
                    f"{parameter!r}.worked_example.per_camera[{i}].{vec_field}",
                    v,
                    hint="(e.g. [0.0, 0.0, 1.0])",
                )
            )
        tol = cam.get("tolerance_deg")
        if tol is not None and not is_number(tol):
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}].tolerance_deg must be numeric, got {type(tol).__name__}"
            )
        # Translation block: present iff all three fields are present.
        # Conditionally mandatory when per-image camera pose data is
        # available -- see _per_image_data_required_for_translation below.
        present = [f for f in CAMERA_WORKED_EXAMPLE_TRANSLATION_FIELDS if f in cam]
        translation_required = _translation_block_required(entry)
        if translation_required and not present:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}]: "
                f"translation block is required when "
                f"authoritative_source.verification."
                f"per_image_data_available is true; declare "
                f"position_in_rig_expected, position_tolerance_m, and "
                f"position_in_rig_expected_derivation (see conventions.md "
                f"'When the input YAML names an authoritative source')."
            )
        elif present and len(present) != len(CAMERA_WORKED_EXAMPLE_TRANSLATION_FIELDS):
            missing = [f for f in CAMERA_WORKED_EXAMPLE_TRANSLATION_FIELDS if f not in cam]
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}]: "
                f"translation block is incomplete (declared {present!r}, "
                f"missing {missing!r}). Declare all of "
                f"position_in_rig_expected, position_tolerance_m, and "
                f"position_in_rig_expected_derivation, or none."
            )
        if "position_in_rig_expected" in cam:
            diagnostics.extend(
                numeric_vec_diagnostics(
                    f"{parameter!r}.worked_example.per_camera[{i}].position_in_rig_expected",
                    cam["position_in_rig_expected"],
                    hint="(meters in rig frame)",
                )
            )
        if "position_tolerance_m" in cam:
            v = cam["position_tolerance_m"]
            if not is_number(v):
                diagnostics.append(
                    f"{parameter!r}.worked_example.per_camera[{i}]."
                    f"position_tolerance_m must be numeric, got "
                    f"{type(v).__name__}"
                )
            elif v < 0:
                diagnostics.append(
                    f"{parameter!r}.worked_example.per_camera[{i}].position_tolerance_m must be non-negative, got {v}"
                )
        if "position_in_rig_expected_derivation" in cam:
            diag = string_field_diagnostic(
                f"{parameter!r}.worked_example.per_camera[{i}].position_in_rig_expected_derivation",
                cam["position_in_rig_expected_derivation"],
                blank_suffix=(
                    "(free-text describing how position_in_rig_expected "
                    "was obtained from a stream other than "
                    "authoritative_source.used_by_converter, so the "
                    "cross-check is independent)"
                ),
            )
            if diag is not None:
                diagnostics.append(diag)

    diagnostics.extend(_check_camera_coverage(parameter, entry, per_camera))
    return diagnostics


def _translation_block_required(entry: dict[str, Any]) -> bool:
    """Return True when per-camera translation checks are mandatory.

    Args:
        entry: Parsed ``T_camera_rig`` justification entry.

    Returns:
        ``True`` when authoritative-source verification reports per-image data
        is available.
    """
    auth = entry.get("authoritative_source")
    if not isinstance(auth, dict):
        return False
    verification = auth.get("verification")
    if not isinstance(verification, dict):
        return False
    return verification.get("per_image_data_available") is True


def _check_camera_coverage(
    parameter: str,
    entry: dict[str, Any],
    per_camera: list[Any],
) -> list[str]:
    """Check camera-ID coverage between value and worked examples.

    Args:
        parameter: Critical parameter name owning the worked example.
        entry: Parsed critical-parameter justification entry.
        per_camera: Parsed ``worked_example.per_camera`` list.

    Returns:
        Diagnostics for cameras declared in ``value`` but missing from
        ``worked_example.per_camera``, or vice versa.
    """
    diagnostics: list[str] = []
    raw_value = entry.get("value")
    if not isinstance(raw_value, dict):
        # Cross-check only operates on the documented dict-keyed-by-camera-id
        # shape (``critical-parameters.md`` 'Value field'); other shapes fall
        # to ``assert_worked_example_camera``'s store check.
        return diagnostics
    declared_cameras = {k for k in raw_value.keys() if isinstance(k, str) and k.strip()}
    camera_ids = (cam.get("camera_id") for cam in per_camera if isinstance(cam, dict))
    covered_cameras = {cam_id for cam_id in camera_ids if isinstance(cam_id, str) and cam_id.strip()}
    missing = declared_cameras - covered_cameras
    extra = covered_cameras - declared_cameras
    if missing:
        diagnostics.append(
            f"{parameter}.value declares cameras "
            f"{sorted(missing)!r} but worked_example.per_camera does not "
            f"cover them (every declared camera must carry a worked-example "
            f"entry)"
        )
    if extra:
        diagnostics.append(
            f"{parameter}.worked_example.per_camera covers cameras {sorted(extra)!r} not declared in {parameter}.value"
        )
    return diagnostics


def _open_store_reader(store_path: str | Path) -> Any:
    """Open an NCore SequenceComponentGroupsReader, raising ValidationError.

    Lazy-imports ncore so the structural path does not require it.
    """
    try:
        from ncore.data.v4 import (  # type: ignore[import-not-found]
            SequenceComponentGroupsReader,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            [f"failed to import ncore ({exc}). Run under the converter workdir's Bazel env, or omit --store."]
        ) from exc
    try:
        return SequenceComponentGroupsReader([Path(store_path)], open_consolidated=True)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError([f"failed to open NCore store at {store_path!s}: {exc}"]) from exc


def _unit_3vec(np_mod: Any, arr: Any, label: str) -> tuple[Any, str | None]:
    """Normalize a 3-vector or report a non-finite/zero error.

    Returns ``(unit_array, None)`` on success or ``(None, error_msg)``
    when the norm is non-positive or non-finite. ``np_mod`` is the
    caller's numpy module (lazy-imported to keep the structural path
    free of ncore deps).
    """
    n = float(np_mod.linalg.norm(arr))
    if n <= 0.0 or not np_mod.isfinite(n):
        return None, f"{label} must be a non-zero 3-vector"
    return arr / n, None


def assert_worked_example(
    store_path: str | Path,
    entry: dict[str, Any],
    *,
    reader: Any | None = None,
) -> None:
    """Verify a LiDAR worked-example ray prediction against the store.

    Args:
        store_path: Path to the converted NCore store (used in diagnostics
            and to open ``reader`` if not supplied).
        entry: Parsed LiDAR critical-parameter justification entry.
        reader: Optional shared ``SequenceComponentGroupsReader``. When
            ``None``, this helper opens its own.

    Raises:
        ValidationError: If NCore imports fail, the store cannot be opened, or
            the worked-example prediction does not match the store.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]

        from ncore.data import (  # type: ignore[import-not-found]
            RowOffsetStructuredSpinningLidarModelParameters,
        )
        from ncore.data.v4 import (  # type: ignore[import-not-found]
            IntrinsicsComponent,
            LidarSensorComponent,
            PosesComponent,
        )
        from ncore.sensors import (  # type: ignore[import-not-found]
            RowOffsetStructuredSpinningLidarModel,
        )
    except Exception as exc:  # noqa: BLE001 - any import failure is reported.
        raise ValidationError(
            [f"failed to import ncore ({exc}). Run under the converter workdir's Bazel env, or omit --store."]
        ) from exc

    parameter = entry.get("parameter", "<unknown>")
    worked = as_dict(entry.get("worked_example"))
    row_idx = int(worked.get("row_index", 0))
    col_idx = int(worked.get("column_index", 0))
    d_rig_raw = worked.get("d_rig_expected")
    if d_rig_raw is None:
        raise ValidationError(
            [f"{parameter!r}.worked_example.d_rig_expected is missing (row={row_idx}, col={col_idx})"]
        )
    try:
        d_expected = np.asarray(d_rig_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            [
                f"{parameter!r}.worked_example.d_rig_expected is not a valid"
                f" 3-vector (row={row_idx}, col={col_idx}): {exc}"
            ]
        ) from exc
    tol_deg = float(worked.get("tolerance_deg", 0.0))
    requested_lidar_id = worked.get("lidar_id")
    if requested_lidar_id is not None and not isinstance(requested_lidar_id, str):
        raise ValidationError(
            [
                f"{parameter!r}.worked_example.lidar_id must be a string "
                f"when present, got {type(requested_lidar_id).__name__}"
            ]
        )

    d_expected, err = _unit_3vec(np, d_expected, f"{parameter!r}.worked_example.d_rig_expected")
    if err is not None:
        raise ValidationError([err])

    if reader is None:
        reader = _open_store_reader(store_path)

    intrinsics_readers = reader.open_component_readers(IntrinsicsComponent.Reader)
    lidar_readers = reader.open_component_readers(LidarSensorComponent.Reader)
    poses_readers = reader.open_component_readers(PosesComponent.Reader)

    candidates: list[tuple[str, Any, Any]] = []
    for lidar_id in lidar_readers:
        params = None
        for ir in intrinsics_readers.values():
            params = ir.get_lidar_model_parameters(lidar_id)
            if params is not None:
                break
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            continue

        T_lidar_rig = None
        for pr in poses_readers.values():
            try:
                T_lidar_rig = pr.get_static_pose(lidar_id, "rig")
                break
            except KeyError:
                continue
        if T_lidar_rig is None:
            continue

        candidates.append((lidar_id, params, T_lidar_rig))

    if not candidates:
        raise ValidationError(
            [
                f"{parameter!r}.worked_example: no spinning LiDAR with both "
                f"intrinsics and rig extrinsic in {store_path!s}"
            ]
        )

    if requested_lidar_id is not None:
        filtered = [c for c in candidates if c[0] == requested_lidar_id]
        if not filtered:
            available = ", ".join(repr(c[0]) for c in candidates)
            raise ValidationError(
                [
                    f"{parameter!r}.worked_example.lidar_id="
                    f"{requested_lidar_id!r} not found among spinning "
                    f"LiDARs in {store_path!s} (available: {available})"
                ]
            )
        candidates = filtered
    elif len(candidates) > 1:
        available = ", ".join(repr(c[0]) for c in candidates)
        raise ValidationError(
            [
                f"{parameter!r}.worked_example: store contains multiple "
                f"spinning LiDARs ({available}); add 'lidar_id: <id>' to "
                f"worked_example to disambiguate."
            ]
        )

    lidar_id, params, T_lidar_rig = candidates[0]

    if not (0 <= row_idx < int(params.n_rows)) or not (0 <= col_idx < int(params.n_columns)):
        raise ValidationError(
            [
                f"{parameter!r}.worked_example: (row={row_idx}, "
                f"col={col_idx}) is outside intrinsics grid "
                f"({params.n_rows} x {params.n_columns}) for {lidar_id}"
            ]
        )

    element = np.array([[row_idx, col_idx]], dtype=np.int64)
    model = RowOffsetStructuredSpinningLidarModel(params, device="cpu")
    d_model = model.elements_to_sensor_rays(element).cpu().numpy().astype(np.float64)[0]
    d_model_norm = float(np.linalg.norm(d_model))
    if d_model_norm > 0:
        d_model = d_model / d_model_norm
    T_lidar_arr = np.asarray(T_lidar_rig, dtype=np.float64)
    if T_lidar_arr.shape != (4, 4):
        raise ValidationError(
            [
                f"{parameter!r}.worked_example: static pose "
                f"{lidar_id!r}->'rig' must be a 4x4 transform, got "
                f"shape {T_lidar_arr.shape!r}"
            ]
        )
    R = T_lidar_arr[:3, :3]
    d_rig = R @ d_model
    dot = float(np.clip(d_rig @ d_expected, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot)))
    if angle_deg > tol_deg:
        raise ValidationError(
            [
                f"{parameter!r}.worked_example: (row={row_idx}, "
                f"col={col_idx}) on {lidar_id!r} -> d_rig="
                f"[{d_rig[0]:+.3f}, {d_rig[1]:+.3f}, {d_rig[2]:+.3f}] "
                f"vs expected "
                f"[{d_expected[0]:+.3f}, {d_expected[1]:+.3f}, "
                f"{d_expected[2]:+.3f}] (angle={angle_deg:.2f} > "
                f"{tol_deg:.2f} deg); see conventions.md "
                f"'Static extrinsics over empirical solving'."
            ]
        )


def assert_worked_example_camera(
    store_path: str | Path,
    entry: dict[str, Any],
    *,
    reader: Any | None = None,
) -> None:
    """Verify camera worked-example rotation/translation predictions.

    Args:
        store_path: Path to the converted NCore store (used in diagnostics
            and to open ``reader`` if not supplied).
        entry: Parsed ``T_camera_rig`` critical-parameter justification entry.
        reader: Optional shared ``SequenceComponentGroupsReader``. When
            ``None``, this helper opens its own.

    Raises:
        ValidationError: If NCore imports fail, the store cannot be opened, or
            any per-camera prediction does not match the store.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]

        from ncore.data.v4 import (  # type: ignore[import-not-found]
            PosesComponent,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            [f"failed to import ncore ({exc}). Run under the converter workdir's Bazel env, or omit --store."]
        ) from exc

    parameter = entry.get("parameter", "<unknown>")
    worked = as_dict(entry.get("worked_example"))
    per_camera = as_list(worked.get("per_camera"))

    if reader is None:
        reader = _open_store_reader(store_path)

    poses_readers = reader.open_component_readers(PosesComponent.Reader)

    diagnostics: list[str] = []
    for i, cam in enumerate(per_camera):
        if not isinstance(cam, dict):
            continue
        cam_id = cam.get("camera_id")
        if not isinstance(cam_id, str) or not cam_id.strip():
            continue
        T = None
        for pr in poses_readers.values():
            try:
                T = pr.get_static_pose(cam_id, "rig")
                break
            except KeyError:
                continue
        if T is None:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}]: no static "
                f"pose <{cam_id}>-><rig> found in {store_path!s} (the "
                f"converter must register the camera's static T_camera_rig "
                f"under the same camera_id used in worked_example)"
            )
            continue

        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.shape != (4, 4):
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] "
                f"({cam_id!r}): static pose <{cam_id}>-><rig> must be "
                f"a 4x4 transform, got shape {T_arr.shape!r}"
            )
            continue
        R = T_arr[:3, :3]
        t = T_arr[:3, 3]

        # Required rotation check.
        try:
            axis_cam = np.asarray(cam.get("axis_in_camera"), dtype=np.float64)
            axis_rig_expected = np.asarray(cam.get("axis_in_rig_expected"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] "
                f"({cam_id!r}): axis vectors are not valid numeric "
                f"3-vectors: {exc}"
            )
            continue
        cam_label = f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): axis_in_camera"
        rig_label = f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): axis_in_rig_expected"
        axis_cam_n, err = _unit_3vec(np, axis_cam, cam_label)
        if err is not None:
            diagnostics.append(err)
            continue
        axis_rig_expected_n, err = _unit_3vec(np, axis_rig_expected, rig_label)
        if err is not None:
            diagnostics.append(err)
            continue
        d_rig = R @ axis_cam_n
        d_rig_norm = float(np.linalg.norm(d_rig))
        if d_rig_norm > 0:
            d_rig = d_rig / d_rig_norm
        dot = float(np.clip(d_rig @ axis_rig_expected_n, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(dot)))
        tol_deg = float(cam.get("tolerance_deg", 0.0))
        if angle_deg > tol_deg:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] "
                f"({cam_id!r}): rotation prediction failed -- "
                f"axis_in_camera=[{axis_cam_n[0]:+.3f}, "
                f"{axis_cam_n[1]:+.3f}, {axis_cam_n[2]:+.3f}] -> "
                f"d_rig=[{d_rig[0]:+.3f}, {d_rig[1]:+.3f}, "
                f"{d_rig[2]:+.3f}] vs expected "
                f"[{axis_rig_expected_n[0]:+.3f}, "
                f"{axis_rig_expected_n[1]:+.3f}, "
                f"{axis_rig_expected_n[2]:+.3f}] "
                f"(angle={angle_deg:.2f} > {tol_deg:.2f} deg); see "
                f"conventions.md 'Static extrinsics over empirical solving'."
            )

        # Translation check. Runs iff the full translation block is
        # declared. The all-or-nothing structural rule is enforced
        # separately by _validate_worked_example_structure_camera, so
        # by the time we get here we trust that all three fields appear
        # together (or none of them do). The derivation field is not
        # consumed at runtime -- it's a free-text audit field preserved
        # into the diagnosis YAML for the anti-circularity review.
        if "position_in_rig_expected" in cam and "position_tolerance_m" in cam:
            try:
                pos_rig_expected = np.asarray(cam["position_in_rig_expected"], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                diagnostics.append(
                    f"{parameter!r}.worked_example.per_camera[{i}] "
                    f"({cam_id!r}): position_in_rig_expected is not a "
                    f"valid numeric 3-vector: {exc}"
                )
                continue
            pos_tol_m = float(cam["position_tolerance_m"])
            delta = t - pos_rig_expected
            dist = float(np.linalg.norm(delta))
            if dist > pos_tol_m:
                diagnostics.append(
                    f"{parameter!r}.worked_example.per_camera[{i}] "
                    f"({cam_id!r}): translation prediction failed -- "
                    f"T_camera_rig.t=[{t[0]:+.3f}, {t[1]:+.3f}, "
                    f"{t[2]:+.3f}] vs expected "
                    f"[{pos_rig_expected[0]:+.3f}, "
                    f"{pos_rig_expected[1]:+.3f}, "
                    f"{pos_rig_expected[2]:+.3f}] m "
                    f"(distance={dist:.3f} > {pos_tol_m:.3f} m); see "
                    f"conventions.md 'Static extrinsics over empirical "
                    f"solving'."
                )

    if diagnostics:
        raise ValidationError(diagnostics)


def _get_camera_model_parameters(intrinsics_readers: dict[str, Any], cam_id: str) -> Any | None:
    """Look up camera model parameters across intrinsics component readers.

    NCore's ``IntrinsicsComponent.Reader.get_camera_model_parameters`` raises
    ``KeyError`` for a camera it does not hold, so absence is signaled by the
    exception, not a ``None`` return.
    """
    for ir in intrinsics_readers.values():
        try:
            return ir.get_camera_model_parameters(cam_id)
        except KeyError:
            continue
    return None


def assert_worked_example_camera_model(
    store_path: str | Path,
    entry: dict[str, Any],
    *,
    reader: Any | None = None,
) -> None:
    """Verify camera model family with source-derived projection samples.

    Args:
        store_path: Path to the converted NCore store (used in diagnostics and
            to open ``reader`` if not supplied).
        entry: Parsed ``camera_model_family`` critical-parameter justification
            entry.
        reader: Optional shared ``SequenceComponentGroupsReader``. When
            ``None``, this helper opens its own.

    Raises:
        ValidationError: If NCore imports fail, the store cannot be opened, or
            any camera model projection does not match the worked example.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]

        from ncore.data import (  # type: ignore[import-not-found]
            FThetaCameraModelParameters,
            OpenCVFisheyeCameraModelParameters,
            OpenCVPinholeCameraModelParameters,
        )
        from ncore.data.v4 import (  # type: ignore[import-not-found]
            IntrinsicsComponent,
        )
        from ncore.sensors import CameraModel  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            [f"failed to import ncore ({exc}). Run under the converter workdir's Bazel env, or omit --store."]
        ) from exc

    expected_types = {
        "opencv_pinhole": OpenCVPinholeCameraModelParameters,
        "opencv_fisheye": OpenCVFisheyeCameraModelParameters,
        "ftheta": FThetaCameraModelParameters,
    }

    parameter = entry.get("parameter", "<unknown>")
    worked = as_dict(entry.get("worked_example"))
    per_camera = as_list(worked.get("per_camera"))

    if reader is None:
        reader = _open_store_reader(store_path)

    intrinsics_readers = reader.open_component_readers(IntrinsicsComponent.Reader)

    diagnostics: list[str] = []
    for i, cam in enumerate(per_camera):
        if not isinstance(cam, dict):
            continue
        cam_id = cam.get("camera_id")
        if not isinstance(cam_id, str) or not cam_id.strip():
            continue
        model_name = cam.get("model")
        if model_name not in expected_types:
            continue

        params = _get_camera_model_parameters(intrinsics_readers, cam_id)
        if params is None:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                f"no camera intrinsics found in {store_path!s}"
            )
            continue

        expected_type = expected_types[model_name]
        if not isinstance(params, expected_type):
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                f"stored camera model parameters are {type(params).__name__}, "
                f"but worked example declares {model_name!r}"
            )
            continue

        try:
            rays = np.asarray(cam["camera_rays"], dtype=np.float32)
            expected = np.asarray(cam["expected_image_points"], dtype=np.float32)
            tol_px = float(cam["tolerance_px"])
        except (KeyError, TypeError, ValueError) as exc:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): projection samples are malformed: {exc}"
            )
            continue

        if rays.ndim != 2 or rays.shape[1] != 3:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                f"camera_rays must have shape (N, 3), got {rays.shape!r}"
            )
            continue
        if expected.shape != (rays.shape[0], 2):
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                f"expected_image_points must have shape ({rays.shape[0]}, 2), got {expected.shape!r}"
            )
            continue
        if rays.shape[0] == 0:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                "camera_rays and expected_image_points must contain at least one sample"
            )
            continue
        if not np.isfinite(rays).all() or not np.isfinite(expected).all():
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                "camera_rays and expected_image_points must contain only finite values"
            )
            continue
        if not np.isfinite(tol_px) or tol_px < 0.0:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                f"tolerance_px must be finite and non-negative, got {tol_px}"
            )
            continue

        try:
            model = CameraModel.from_parameters(params, device="cpu")
            projected = model.camera_rays_to_image_points(rays)
            actual = projected.image_points.detach().cpu().numpy().astype(np.float64)
            valid = projected.valid_flag.detach().cpu().numpy().astype(bool)
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                f"failed to project camera rays through stored model: {exc}"
            )
            continue

        if not bool(np.all(valid)):
            invalid = np.flatnonzero(~valid)
            preview = ", ".join(str(int(idx)) for idx in invalid[:8])
            if invalid.size > 8:
                preview += f", ... ({invalid.size - 8} more)"
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                f"stored camera model marked {invalid.size}/{rays.shape[0]} "
                f"projection samples invalid (indices: {preview})"
            )
            continue

        errors = np.linalg.norm(actual - expected.astype(np.float64), axis=1)
        max_idx = int(np.argmax(errors))
        max_err = float(errors[max_idx])
        if max_err > tol_px:
            diagnostics.append(
                f"{parameter!r}.worked_example.per_camera[{i}] ({cam_id!r}): "
                f"projection mismatch from {cam.get('sample_source')!r} -- "
                f"max error {max_err:.3f}px at sample {max_idx} exceeds "
                f"tolerance {tol_px:.3f}px; check camera model family, "
                f"coefficient ordering, and source-to-NCore parameter mapping"
            )

    if diagnostics:
        raise ValidationError(diagnostics)


def assert_worked_example_cuboid_tracks(
    store_path: str | Path,
    entry: dict[str, Any],
    *,
    reader: Any | None = None,
) -> None:
    """Verify recorded cuboid counts and sampled source tracks against the store.

    Args:
        store_path: Path to the converted NCore store (used in diagnostics
            and to open ``reader`` if not supplied).
        entry: Parsed ``sequence_frame_range_decision`` justification entry.
        reader: Optional shared ``SequenceComponentGroupsReader``. When
            ``None``, this helper opens its own.

    Raises:
        ValidationError: If NCore imports fail, the store cannot be read, or
            the stored cuboids disagree with the recorded source data.
    """
    try:
        from ncore.data.v4 import (  # type: ignore[import-not-found]
            CameraSensorComponent,
            CuboidsComponent,
            LidarSensorComponent,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            [f"failed to import ncore ({exc}). Run under the converter workdir's Bazel env, or omit --store."]
        ) from exc

    if reader is None:
        reader = _open_store_reader(store_path)

    stored_by_track: dict[CuboidTrackKey, list[int]] = {}
    try:
        for component, cuboid_reader in reader.open_component_readers(CuboidsComponent.Reader).items():
            for obs in cuboid_reader.get_observations():
                version = obs.source_version
                key = (component, str(obs.track_id), obs.source.name, None if version is None else str(version))
                stored_by_track.setdefault(key, []).append(int(obs.timestamp_us))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError([f"failed to read cuboid observations from {store_path!s}: {exc}"]) from exc

    starts: list[int] = []
    ends: list[int] = []
    for reader_type in (CameraSensorComponent.Reader, LidarSensorComponent.Reader):
        for sensor_reader in reader.open_component_readers(reader_type).values():
            frames = sensor_reader.frames_timestamps_us
            if len(frames):
                starts.append(int(frames[:, 0].min()))
                ends.append(int(frames[:, 1].max()))
    coverage = (min(starts), max(ends)) if starts else None

    diagnostics = _cuboid_track_store_diagnostics(entry["parameter"], entry, stored_by_track, coverage)
    if diagnostics:
        raise ValidationError(diagnostics)


def assert_worked_examples(store_path: str | Path, entries: list[dict[str, Any]]) -> None:
    """Run worked-example store checks for every applicable entry.

    Opens the NCore store once and reuses the reader across LiDAR and
    camera assertions; aggregates ``ValidationError`` diagnostics.
    """
    applicable = [
        e
        for e in entries
        if isinstance(e.get("parameter"), str)
        and e["parameter"] in PARAMETERS_REQUIRING_WORKED_EXAMPLE
        and isinstance(e.get("worked_example"), dict)
    ]
    if not applicable:
        return

    reader = _open_store_reader(store_path)

    diagnostics: list[str] = []
    for entry in applicable:
        name = entry["parameter"]
        try:
            if name == "T_camera_rig":
                assert_worked_example_camera(store_path, entry, reader=reader)
            elif name == "camera_model_family":
                assert_worked_example_camera_model(store_path, entry, reader=reader)
            elif name == "sequence_frame_range_decision":
                assert_worked_example_cuboid_tracks(store_path, entry, reader=reader)
            else:
                assert_worked_example(store_path, entry, reader=reader)
        except ValidationError as exc:
            diagnostics.extend(exc.diagnostics)
    if diagnostics:
        raise ValidationError(diagnostics)


def _check_package_rename(doc: dict[str, Any], summary_path: str | Path) -> None:
    """Verify the agent renamed the placeholder Python package.

    The bundled template at ``ncore_template/ncore_example/`` is
    materialized into the converter workdir verbatim. The single-token rename
    step (see ``nurec-onboard/steps/build-ncore-converter/SKILL.md``) requires the agent
    to substitute ``example`` for the dataset token, producing a
    package directory named ``ncore_<dataset>/``. This check fails if either
    (a) the resulting top-level converter package directory does not match
    ``^ncore_[a-z][a-z0-9_]*$``, or (b) the placeholder name ``ncore_example``
    was never replaced.

    Args:
        doc: Parsed summary YAML document.
        summary_path: Path of the summary YAML (used for error messages).

    Raises:
        ValidationError: If ``ncore_converter_dir`` is missing/blank, does not
            resolve to an existing directory, contains no ``ncore_<dataset>/``
            package, or still carries the unrenamed placeholder / a
            pattern-violating package name.
    """
    # `ncore_converter_dir` is a required field in both the scaffold-mode and
    # fix-mode build summary schemas (nurec-onboard/steps/build-ncore-converter/SKILL.md).
    # Fail closed on absence/blankness so the rename guard cannot be bypassed
    # by simply omitting the field.
    converter_dir_raw = doc.get("ncore_converter_dir")
    if not isinstance(converter_dir_raw, str) or not converter_dir_raw.strip():
        raise ValidationError(
            [
                f"{summary_path}: 'ncore_converter_dir' is missing or blank. "
                f"The build summary must record the converter workdir so the "
                f"placeholder-rename guard rail can confirm "
                f"'{_PLACEHOLDER_PACKAGE}/' was renamed (see "
                f"nurec-onboard/steps/build-ncore-converter/SKILL.md)."
            ]
        )
    converter_dir = Path(converter_dir_raw).expanduser()
    if not converter_dir.is_dir():
        raise ValidationError(
            [
                f"{summary_path}: 'ncore_converter_dir'={converter_dir_raw!r} "
                f"is not an existing directory; the placeholder-rename guard "
                f"rail cannot inspect the converter workdir. Record the real "
                f"converter workdir path (see "
                f"nurec-onboard/steps/build-ncore-converter/SKILL.md)."
            ]
        )
    candidates: list[str] = []
    for entry in converter_dir.iterdir():
        if not entry.is_dir() or entry.is_symlink():
            continue
        name = entry.name
        if name.startswith(".") or not name.startswith("ncore_"):
            continue
        if not (entry / "__init__.py").is_file():
            continue
        candidates.append(name)
    if not candidates:
        # This validator is the build step's final gate -- it runs only after
        # the template was materialized (which always ships an 'ncore_example/'
        # package) and the conversion produced a store (--store). A workdir with
        # no 'ncore_<dataset>/' package therefore is not a legitimate
        # intermediate state: the package is missing, lacks __init__.py, or was
        # renamed without the required 'ncore_' prefix (which would otherwise
        # silently bypass the placeholder-rename guard below).
        raise ValidationError(
            [
                f"{summary_path}: converter workdir {converter_dir} contains "
                f"no top-level 'ncore_<dataset>/' Python package (a directory "
                f"named 'ncore_*' with an __init__.py). The single-token "
                f"rename must keep the 'ncore_' prefix "
                f"('{_PLACEHOLDER_PACKAGE}/' -> 'ncore_<dataset>/'); a package "
                f"renamed without that prefix, a missing package, or a package "
                f"without __init__.py all fail this guard. See "
                f"nurec-onboard/steps/build-ncore-converter/SKILL.md."
            ]
        )
    errors: list[str] = []
    if _PLACEHOLDER_PACKAGE in candidates:
        errors.append(
            f"{summary_path}: converter workdir {converter_dir} still "
            f"contains the placeholder package '{_PLACEHOLDER_PACKAGE}/'. "
            f"Apply the single-token rename described in "
            f"nurec-onboard/steps/build-ncore-converter/SKILL.md (replace 'example' "
            f"with the dataset token in the directory name, class "
            f"names, and imports)."
        )
    invalid = [name for name in candidates if not _PACKAGE_NAME_RE.match(name)]
    if invalid:
        errors.append(
            f"{summary_path}: converter workdir {converter_dir} contains "
            f"top-level package directory(ies) {invalid} that do not match "
            f"the required pattern '^ncore_[a-z][a-z0-9_]*$'. Rename to "
            f"'ncore_<dataset>' per the single-token rename rule in "
            f"nurec-onboard/steps/build-ncore-converter/SKILL.md."
        )
    if errors:
        raise ValidationError(errors)


def validate_summary(summary_path: str | Path) -> list[dict[str, Any]]:
    """Parse a summary YAML and validate critical-parameter justifications.

    Args:
        summary_path: Path to the build step summary YAML file.

    Returns:
        Parsed critical-parameter justification entries for optional
        downstream store checks.

    Raises:
        ValidationError: If the YAML is malformed or the justifications fail
            structural validation.
    """
    doc = load_yaml(summary_path)
    if not isinstance(doc, dict):
        raise ValidationError(
            [f"Top-level YAML document in {summary_path} must be a mapping, got {type(doc).__name__}"]
        )
    raw = doc.get("critical_parameter_justifications")
    if raw is None:
        raise ValidationError(
            [
                f"{summary_path} has no 'critical_parameter_justifications' "
                f"key (required by the critical-parameter justification step "
                f"in scaffold-mode and fix-mode re-justification; see "
                f"nurec-onboard/steps/build-ncore-converter/critical-parameters.md)."
            ]
        )
    entries = _as_list_of_entries(raw)
    validate_entries(entries)
    _check_package_rename(doc, summary_path)
    return entries


_USAGE = (
    'Usage: cd "$NUREC_ONBOARD_DIR" && bazel run '
    "//skills/nurec-onboard/validators:build_ncore_converter_validator -- "
    "<summary.yaml> [--store <ncore-store-path>]\n"
    "Validates the 'critical_parameter_justifications' list in a\n"
    "build-ncore-converter step summary. With --store, also\n"
    "runs assert_worked_example for entries that carry one.\n"
)


main = step_validator_main(
    progname="build_ncore_converter_validator",
    usage=_USAGE,
    validate_fn=validate_summary,
    allow_store=True,
    post_store_fn=assert_worked_examples,
)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
