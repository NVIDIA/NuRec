# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validator for the ``run-nurec`` diagnosis YAML.

Checks reconstruction status, issue-bucket structure, metrics shape, resolved
image subsampling, and NuRec LiDAR-selection provenance.

Usage: ``cd "$NUREC_ONBOARD_DIR" && bazel run //skills/nurec-onboard/validators:run_nurec_validator -- <diagnosis.yaml>``.
Exit codes: ``0`` pass, ``1`` fail, ``2`` malformed input.
"""

from __future__ import annotations

import sys

from pathlib import Path
from typing import Any

from validators._validation_base import (
    ValidationError,
    blank_or_placeholder_diagnostic,
    is_finite_number,
    load_yaml,
    step_validator_main,
    string_field_diagnostic,
)


_VALID_RECONSTRUCTION_STATUSES = ("success", "issues_detected")
_VALID_SEVERITIES = ("error", "warning")
_VALID_SUGGESTED_FIX_KINDS = ("custom_config_edit", "hydra_override")


def _validate_issue_entries(doc: dict[str, Any], bucket: str, *, require_suggested_fix: bool) -> list[str]:
    """Validate a run-nurec issue bucket.

    Args:
        doc: Parsed run-nurec diagnosis YAML document.
        bucket: Issue-bucket key to validate.
        require_suggested_fix: Whether each entry must include a structured
            NuRec config fix.

    Returns:
        Diagnostics for missing or malformed issue entries.
    """
    diagnostics: list[str] = []
    raw = doc.get(bucket)
    if raw is None:
        diagnostics.append(f"'{bucket}' is missing (required list)")
        return diagnostics
    if not isinstance(raw, list):
        return [f"'{bucket}' must be a list, got {type(raw).__name__}"]
    for i, entry in enumerate(raw):
        prefix = f"{bucket}[{i}]"
        if not isinstance(entry, dict):
            diagnostics.append(f"{prefix} must be a mapping, got {type(entry).__name__}")
            continue
        for field in ("symptom", "likely_cause", "severity"):
            if field not in entry:
                diagnostics.append(f"{prefix}: {field!r} is missing")
        for field in ("symptom", "likely_cause"):
            if field in entry:
                diag = blank_or_placeholder_diagnostic(f"{prefix}.{field}", entry[field])
                if diag is not None:
                    diagnostics.append(diag)
                elif not isinstance(entry[field], str):
                    diagnostics.append(f"{prefix}.{field} must be a string, got {type(entry[field]).__name__}")
        # Only validate when the key is actually present: missing ``severity``
        # is already reported by the required-keys loop above, and the explicit
        # ``null`` case is *not* handled there (the key is present), so it
        # still falls into this branch and surfaces as "got None" instead of
        # silently passing.
        if "severity" in entry and entry["severity"] not in _VALID_SEVERITIES:
            diagnostics.append(f"{prefix}.severity must be one of {_VALID_SEVERITIES!r}, got {entry['severity']!r}")
        if require_suggested_fix:
            diagnostics.extend(_validate_suggested_fix(prefix, entry))
    return diagnostics


def _validate_suggested_fix(prefix: str, entry: dict[str, Any]) -> list[str]:
    """Validate a structured NuRec config suggested-fix block.

    Args:
        prefix: User-facing diagnostic path for the issue entry.
        entry: Parsed issue entry.

    Returns:
        Diagnostics for a missing or malformed ``suggested_fix`` block.
    """
    diagnostics: list[str] = []
    fix = entry.get("suggested_fix")
    if fix is None:
        return [f"{prefix}: 'suggested_fix' is missing (required for nurec_config_issues)"]
    if not isinstance(fix, dict):
        return [f"{prefix}.suggested_fix must be a mapping, got {type(fix).__name__}"]
    kind = fix.get("kind")
    if kind not in _VALID_SUGGESTED_FIX_KINDS:
        diagnostics.append(f"{prefix}.suggested_fix.kind must be one of {_VALID_SUGGESTED_FIX_KINDS!r}, got {kind!r}")
    value = fix.get("value")
    diag = blank_or_placeholder_diagnostic(f"{prefix}.suggested_fix.value", value)
    if diag is not None:
        diagnostics.append(diag)
    elif not isinstance(value, str):
        diagnostics.append(f"{prefix}.suggested_fix.value must be a string, got {type(value).__name__}")
    return diagnostics


def _validate_metrics(doc: dict[str, Any]) -> list[str]:
    """Validate optional reconstruction metrics.

    Args:
        doc: Parsed run-nurec diagnosis YAML document.

    Returns:
        Diagnostics for missing success metrics or malformed metric values.
    """
    metrics = doc.get("metrics")
    if metrics is None:
        if doc.get("reconstruction_status") == "success":
            return ["'metrics' is missing (required when reconstruction_status=success)"]
        return []
    if not isinstance(metrics, dict):
        return [f"'metrics' must be a mapping, got {type(metrics).__name__}"]
    diagnostics: list[str] = []
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            diagnostics.append(f"'metrics' contains non-string or blank key {key!r}")
            continue
        if value is None:
            continue
        if not is_finite_number(value):
            diagnostics.append(f"metrics.{key} must be a finite number or null, got {value!r}")
    return diagnostics


def _positive_int_diagnostic(field: str, value: Any) -> str | None:
    """Validate a field is a positive integer.

    Args:
        field: User-facing diagnostic path.
        value: Parsed YAML value.

    Returns:
        A diagnostic string when invalid, otherwise None.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{field} must be a positive integer, got {type(value).__name__}"
    if value <= 0:
        return f"{field} must be a positive integer, got {value!r}"
    return None


def _validate_subsampling(doc: dict[str, Any]) -> list[str]:
    """Validate resolved image subsampling and selected camera dimensions.

    Args:
        doc: Parsed run-nurec diagnosis YAML document.

    Returns:
        Diagnostics for missing, malformed, or non-divisible subsampling data.
    """
    block = doc.get("subsampling")
    if block is None:
        return ["'subsampling' is missing (required for run-nurec)"]
    if not isinstance(block, dict):
        return [f"'subsampling' must be a mapping, got {type(block).__name__}"]

    diagnostics: list[str] = []
    factor_fields = (
        "camera_pixel_sampler_subsample",
        "n_val_image_subsample",
        "n_train_sequential_image_subsample",
    )
    factors: dict[str, int] = {}
    for field in factor_fields:
        if field not in block:
            diagnostics.append(f"subsampling.{field} is missing")
            continue
        value = block[field]
        diag = _positive_int_diagnostic(f"subsampling.{field}", value)
        if diag is not None:
            diagnostics.append(diag)
            continue
        factors[field] = value

    if len(set(factors.values())) > 1:
        diagnostics.append(
            "subsampling camera_pixel_sampler_subsample, n_val_image_subsample, and n_train_sequential_image_subsample must resolve to the same value"
        )

    cameras = block.get("selected_camera_resolutions")
    if "selected_camera_resolutions" not in block:
        diagnostics.append("subsampling.selected_camera_resolutions is missing")
        return diagnostics
    if not isinstance(cameras, list):
        diagnostics.append(f"subsampling.selected_camera_resolutions must be a list, got {type(cameras).__name__}")
        return diagnostics
    if not cameras:
        diagnostics.append("subsampling.selected_camera_resolutions must contain at least one camera")
        return diagnostics

    divisor = factors.get("camera_pixel_sampler_subsample")
    for i, camera in enumerate(cameras):
        prefix = f"subsampling.selected_camera_resolutions[{i}]"
        if not isinstance(camera, dict):
            diagnostics.append(f"{prefix} must be a mapping, got {type(camera).__name__}")
            continue
        diag = string_field_diagnostic(f"{prefix}.camera_id", camera.get("camera_id"))
        if diag is not None:
            diagnostics.append(diag)
        for dim in ("width", "height"):
            if dim not in camera:
                diagnostics.append(f"{prefix}.{dim} is missing")
                continue
            value = camera[dim]
            dim_diag = _positive_int_diagnostic(f"{prefix}.{dim}", value)
            if dim_diag is not None:
                diagnostics.append(dim_diag)
                continue
            if divisor is not None and value % divisor != 0:
                diagnostics.append(f"{prefix}.{dim}={value} is not divisible by resolved subsampling factor {divisor}")
    return diagnostics


def _validate_lidar_selection(doc: dict[str, Any]) -> list[str]:
    """Validate NuRec LiDAR-selection provenance.

    Args:
        doc: Parsed run-nurec diagnosis YAML document.

    Returns:
        Diagnostics for missing or malformed ``nurec_lidar_selection`` data.
    """
    block = doc.get("nurec_lidar_selection")
    if block is None:
        return ["'nurec_lidar_selection' is missing (required for run-nurec)"]
    if not isinstance(block, dict):
        return [f"'nurec_lidar_selection' must be a mapping, got {type(block).__name__}"]
    diagnostics: list[str] = []
    for field in ("selected_id", "rationale"):
        if field not in block:
            diagnostics.append(f"nurec_lidar_selection.{field} is missing")
            continue
        diag = string_field_diagnostic(f"nurec_lidar_selection.{field}", block[field])
        if diag is not None:
            diagnostics.append(diag)
    excluded = block.get("excluded_ids")
    if "excluded_ids" not in block:
        diagnostics.append("nurec_lidar_selection.excluded_ids is missing")
    elif not isinstance(excluded, list):
        diagnostics.append(f"nurec_lidar_selection.excluded_ids must be a list, got {type(excluded).__name__}")
    else:
        for i, item in enumerate(excluded):
            if not isinstance(item, str) or not item.strip():
                diagnostics.append(f"nurec_lidar_selection.excluded_ids[{i}] must be a non-blank string")
    if "operator_override" not in block:
        diagnostics.append("nurec_lidar_selection.operator_override is missing")
    elif not isinstance(block["operator_override"], bool):
        diagnostics.append(
            f"nurec_lidar_selection.operator_override must be a bool, got {type(block['operator_override']).__name__}"
        )
    return diagnostics


def _validate_shape(doc: dict[str, Any]) -> list[str]:
    """Check required keys and types before cross-field invariants.

    Args:
        doc: Parsed run-nurec diagnosis YAML document.

    Returns:
        Diagnostics for missing or malformed top-level fields.
    """
    diagnostics: list[str] = []
    status = doc.get("reconstruction_status")
    if "reconstruction_status" not in doc:
        diagnostics.append("'reconstruction_status' is missing (one of: success | issues_detected)")
    elif status not in _VALID_RECONSTRUCTION_STATUSES:
        diagnostics.append(f"'reconstruction_status' must be one of {_VALID_RECONSTRUCTION_STATUSES!r}, got {status!r}")
    diagnostics.extend(_validate_metrics(doc))
    diagnostics.extend(_validate_subsampling(doc))
    diagnostics.extend(_validate_lidar_selection(doc))
    diagnostics.extend(_validate_issue_entries(doc, "ncore_data_issues", require_suggested_fix=False))
    diagnostics.extend(_validate_issue_entries(doc, "nurec_config_issues", require_suggested_fix=True))
    return diagnostics


def _validate_invariants(doc: dict[str, Any]) -> list[str]:
    """Check run-nurec status and issue-bucket invariants.

    The statuses partition on error severity, not on whether an entry
    exists: a warning-severity finding did not stop the reconstruction,
    so it rides along on a successful run.

    Args:
        doc: Parsed run-nurec diagnosis YAML document.

    Returns:
        Diagnostics for inconsistent reconstruction status and issue state.
    """
    diagnostics: list[str] = []
    status = doc.get("reconstruction_status")
    error_buckets = [
        bucket
        for bucket in ("ncore_data_issues", "nurec_config_issues")
        if isinstance(doc.get(bucket), list)
        and any(isinstance(e, dict) and e.get("severity") == "error" for e in doc[bucket])
    ]
    if status == "success" and error_buckets:
        diagnostics.append(
            f"reconstruction_status='success' requires no severity='error' entries, found one in "
            f"{' and '.join(error_buckets)}; downgrade it to severity='warning' if the run survived it"
        )
    if status == "issues_detected" and not error_buckets:
        diagnostics.append(
            "reconstruction_status='issues_detected' requires at least one severity='error' entry in "
            "ncore_data_issues or nurec_config_issues; a warning-only run is a success"
        )
    return diagnostics


def validate_diagnosis(diag_path: str | Path) -> None:
    """Validate a run-nurec diagnosis YAML file.

    Args:
        diag_path: Path to the run-nurec diagnosis YAML file.

    Returns:
        None.

    Raises:
        ValidationError: If the YAML is malformed or violates run-nurec
            invariants.
    """
    doc = load_yaml(diag_path)
    if not isinstance(doc, dict):
        raise ValidationError([f"Top-level YAML document in {diag_path} must be a mapping, got {type(doc).__name__}"])
    diagnostics: list[str] = []
    diagnostics.extend(_validate_shape(doc))
    diagnostics.extend(_validate_invariants(doc))
    if diagnostics:
        raise ValidationError(diagnostics)


_USAGE = (
    'Usage: cd "$NUREC_ONBOARD_DIR" && bazel run '
    "//skills/nurec-onboard/validators:run_nurec_validator -- <diagnosis.yaml>\n"
    "Asserts cross-field invariants on the run-nurec diagnosis\n"
    "YAML, including reconstruction status, issue-bucket structure,\n"
    "metrics shape, resolved image subsampling, and NuRec LiDAR-selection provenance.\n"
)


main = step_validator_main(
    progname="run_nurec_validator",
    usage=_USAGE,
    validate_fn=validate_diagnosis,
)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
