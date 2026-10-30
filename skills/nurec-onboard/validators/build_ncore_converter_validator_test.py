# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``validators/build_ncore_converter_validator.py``.

Covers the four bool-as-numeric loopholes in this file's numeric-field
checks, sibling to the leak originally flagged in
``numeric_vec_diagnostics``:

1. ``worked_example.column_index`` and ``row_index`` (must be int).
2. Single-camera ``worked_example.tolerance_deg`` (must be numeric).
3. Per-camera ``worked_example.per_camera[*].tolerance_deg``.
4. Per-camera ``worked_example.per_camera[*].position_tolerance_m``.

Each loophole test checks both the loophole-closure (bool flagged) and
a positive case (valid numeric value accepted) so the diagnostic stays
pinned to the right field.

Also covers ``_value_field_diagnostic`` (the ``value:`` field accepts any
YAML type; the helper rejects only missing / blank / placeholder /
empty-container values) and ``_check_camera_coverage`` (cross-checks
``T_camera_rig.value`` keys against ``worked_example.per_camera`` ids,
no-ops on non-dict ``value:`` shapes).
"""

from __future__ import annotations

import tempfile
import unittest

from pathlib import Path
from typing import Any

from validators._validation_base import ValidationError
from validators.build_ncore_converter_validator import (
    _PLACEHOLDER_PACKAGE,
    _camera_model_mismatch_diagnostic,
    _camera_model_value_diagnostics,
    _check_camera_coverage,
    _check_package_rename,
    _cuboid_track_store_diagnostics,
    _expected_cuboid_track_timestamps,
    _get_camera_model_parameters,
    _validate_worked_example_structure,
    _validate_worked_example_structure_camera,
    _validate_worked_example_structure_camera_model,
    _value_field_diagnostic,
)
from validators.validator_test_helpers import ValidatorTestCase


def _non_camera_entry(**worked_overrides: Any) -> dict[str, Any]:
    """Return a critical-parameter entry with non-camera worked_example shape.

    Args:
        worked_overrides: Field-level overrides applied inside ``worked_example``.

    Returns:
        Entry dict consumable by ``_validate_worked_example_structure``.
    """
    base: dict[str, Any] = {"worked_example": {"a_native_expected": [1.0, 0.0, 0.0]}}
    base["worked_example"].update(worked_overrides)
    return base


def _camera_entry(**cam_overrides: Any) -> dict[str, Any]:
    """Return a T_camera_rig entry with one per_camera worked_example.

    Args:
        cam_overrides: Field-level overrides applied inside ``per_camera[0]``.

    Returns:
        Entry dict consumable by ``_validate_worked_example_structure_camera``.
    """
    cam: dict[str, Any] = {
        "camera_id": "front",
        "z_axis_optical_expected": [0.0, 0.0, 1.0],
        "tolerance_deg": 0.5,
    }
    cam.update(cam_overrides)
    return {"worked_example": {"per_camera": [cam]}}


def _per_camera(*ids: str) -> list[dict[str, Any]]:
    """Build a minimal ``worked_example.per_camera`` list with the given ids."""
    return [{"camera_id": cid} for cid in ids]


def _camera_model_entry(**cam_overrides: Any) -> dict[str, Any]:
    """Return a camera_model_family entry with one projection sample block.

    Args:
        cam_overrides: Field-level overrides applied inside ``per_camera[0]``.

    Returns:
        Entry dict consumable by
        ``_validate_worked_example_structure_camera_model``.
    """
    cam: dict[str, Any] = {
        "camera_id": "front",
        "model": "opencv_pinhole",
        "camera_rays": [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
        "expected_image_points": [[960.5, 540.5], [1060.5, 540.5]],
        "tolerance_px": 1.0,
        "sample_source": "LiDAR-to-image correspondence samples from source data",
    }
    cam.update(cam_overrides)
    return {
        "value": [{"camera_id": "front", "model": "opencv_pinhole"}],
        "worked_example": {"per_camera": [cam]},
    }


class TestWorkedExampleColumnIndex(ValidatorTestCase):
    """``column_index`` must be an int (not bool)."""

    def test_bool_true_flagged(self) -> None:
        """``column_index: true`` is flagged as non-int."""
        diags = _validate_worked_example_structure("scratch_param", _non_camera_entry(column_index=True))
        self.assertOneDiagnosticMatching(diags, "column_index must be", "got bool")

    def test_bool_false_flagged(self) -> None:
        """``column_index: false`` is also flagged.

        Distinct from the ``True`` case because ``False`` would coerce
        to ``0`` under a naive ``isinstance(_, int)`` check that doesn't
        exclude bool, and ``0`` is otherwise a perfectly valid column
        index; this exercises the same code path as ``True`` but with a
        value that would NOT trip a separate range check.
        """
        diags = _validate_worked_example_structure("scratch_param", _non_camera_entry(column_index=False))
        self.assertOneDiagnosticMatching(diags, "column_index must be")

    def test_valid_int_accepted(self) -> None:
        """A real int ``column_index`` is accepted."""
        diags = _validate_worked_example_structure("scratch_param", _non_camera_entry(column_index=5))
        self.assertNoDiagnosticMatching(diags, "column_index must be")


class TestWorkedExampleRowIndex(ValidatorTestCase):
    """``row_index`` must be an int (not bool)."""

    def test_bool_flagged(self) -> None:
        """``row_index: true`` is flagged via ``is_int``."""
        diags = _validate_worked_example_structure("scratch_param", _non_camera_entry(row_index=True))
        self.assertOneDiagnosticMatching(diags, "row_index must be")


class TestWorkedExampleToleranceDeg(ValidatorTestCase):
    """Single-camera ``tolerance_deg`` must be numeric (not bool)."""

    def test_bool_flagged(self) -> None:
        """``tolerance_deg: true`` is flagged via ``is_number``."""
        diags = _validate_worked_example_structure("scratch_param", _non_camera_entry(tolerance_deg=True))
        self.assertOneDiagnosticMatching(diags, "tolerance_deg must be numeric")

    def test_valid_float_accepted(self) -> None:
        """``tolerance_deg: 0.5`` passes."""
        diags = _validate_worked_example_structure("scratch_param", _non_camera_entry(tolerance_deg=0.5))
        self.assertNoDiagnosticMatching(diags, "tolerance_deg must be numeric")


class TestWorkedExamplePerCameraToleranceDeg(ValidatorTestCase):
    """Per-camera ``tolerance_deg`` must be numeric (not bool)."""

    def test_bool_flagged(self) -> None:
        """Per-camera ``tolerance_deg: true`` is flagged via ``is_number``."""
        diags = _validate_worked_example_structure_camera("T_camera_rig", _camera_entry(tolerance_deg=True))
        self.assertOneDiagnosticMatching(diags, "tolerance_deg must be numeric")

    def test_valid_float_accepted(self) -> None:
        """Per-camera ``tolerance_deg: 0.5`` passes shape check."""
        diags = _validate_worked_example_structure_camera("T_camera_rig", _camera_entry(tolerance_deg=0.5))
        self.assertNoDiagnosticMatching(diags, "tolerance_deg must be numeric")


class TestWorkedExamplePerCameraPositionToleranceM(ValidatorTestCase):
    """Per-camera ``position_tolerance_m`` must be numeric (not bool)."""

    def test_bool_flagged(self) -> None:
        """``position_tolerance_m: false`` is flagged via ``is_number``."""
        diags = _validate_worked_example_structure_camera("T_camera_rig", _camera_entry(position_tolerance_m=False))
        self.assertOneDiagnosticMatching(diags, "position_tolerance_m must be numeric")

    def test_valid_float_accepted(self) -> None:
        """``position_tolerance_m: 0.05`` passes shape check."""
        diags = _validate_worked_example_structure_camera("T_camera_rig", _camera_entry(position_tolerance_m=0.05))
        self.assertNoDiagnosticMatching(diags, "position_tolerance_m must be numeric")


class TestCameraModelWorkedExample(ValidatorTestCase):
    """``camera_model_family`` requires per-camera projection samples."""

    def test_valid_projection_sample_shape_passes(self) -> None:
        """A complete projection sample block passes structural validation."""
        self.assertEqual(
            _validate_worked_example_structure_camera_model("camera_model_family", _camera_model_entry()), []
        )

    def test_rejects_unknown_model_name(self) -> None:
        """Camera model names are limited to supported NCore camera families."""
        diags = _validate_worked_example_structure_camera_model(
            "camera_model_family",
            _camera_model_entry(model="thin_lens"),
        )
        self.assertOneDiagnosticMatching(diags, "model must be one of", "thin_lens")

    def test_requires_matching_sample_counts(self) -> None:
        """Rays and expected image points must have the same row count."""
        diags = _validate_worked_example_structure_camera_model(
            "camera_model_family",
            _camera_model_entry(expected_image_points=[[960.5, 540.5]]),
        )
        self.assertOneDiagnosticMatching(diags, "same row count")

    def test_requires_coverage_for_declared_cameras(self) -> None:
        """Every camera declared in value needs a worked-example block."""
        entry = _camera_model_entry()
        entry["value"].append({"camera_id": "rear", "model": "opencv_fisheye"})
        diags = _validate_worked_example_structure_camera_model("camera_model_family", entry)
        self.assertOneDiagnosticMatching(diags, "declares cameras", "rear")

    def test_requires_matching_model_for_declared_camera(self) -> None:
        """The value model and worked-example model must agree per camera."""
        entry = _camera_model_entry(model="opencv_fisheye")

        diags = _validate_worked_example_structure_camera_model("camera_model_family", entry)

        self.assertOneDiagnosticMatching(diags, "front", "opencv_pinhole", "opencv_fisheye")

    def test_missing_worked_example_model_skips_mismatch_diagnostic(self) -> None:
        """A missing per_camera model reports once, without a spurious mismatch."""
        entry = _camera_model_entry()
        del entry["worked_example"]["per_camera"][0]["model"]

        diags = _validate_worked_example_structure_camera_model("camera_model_family", entry)

        self.assertOneDiagnosticMatching(diags, "per_camera[0]", "model", "missing")
        self.assertNoDiagnosticMatching(diags, "but worked_example.per_camera declares")

    def test_duplicate_per_camera_entry_skips_spurious_mismatch(self) -> None:
        """A duplicate per_camera entry reports once, without a model mismatch."""
        entry = _camera_model_entry()
        first = entry["worked_example"]["per_camera"][0]
        entry["worked_example"]["per_camera"].append({**first, "model": "opencv_fisheye"})

        diags = _validate_worked_example_structure_camera_model("camera_model_family", entry)

        self.assertOneDiagnosticMatching(diags, "duplicate camera_id", "front")
        self.assertNoDiagnosticMatching(diags, "value declares", "front")

    def test_null_tolerance_px_is_flagged_as_non_numeric(self) -> None:
        """A present-but-null ``tolerance_px`` is reported, not passed to store."""
        diags = _validate_worked_example_structure_camera_model(
            "camera_model_family",
            _camera_model_entry(tolerance_px=None),
        )
        self.assertOneDiagnosticMatching(diags, "tolerance_px", "must be numeric", "NoneType")

    def test_absent_tolerance_px_reports_missing_only(self) -> None:
        """An absent ``tolerance_px`` reports as missing without a numeric diagnostic."""
        entry = _camera_model_entry()
        del entry["worked_example"]["per_camera"][0]["tolerance_px"]

        diags = _validate_worked_example_structure_camera_model("camera_model_family", entry)

        self.assertOneDiagnosticMatching(diags, "tolerance_px", "missing")
        self.assertNoDiagnosticMatching(diags, "tolerance_px", "must be numeric")


class TestCameraModelParameterLookup(ValidatorTestCase):
    """Camera model parameter lookup handles absent cameras cleanly."""

    def test_keyerror_is_treated_as_missing_camera(self) -> None:
        """A KeyError from an intrinsics reader does not escape lookup."""

        class _IntrinsicsReader:
            def get_camera_model_parameters(self, _camera_id: str) -> object:
                raise KeyError("front")

        self.assertIsNone(_get_camera_model_parameters({"intrinsics": _IntrinsicsReader()}, "front"))

    def test_returns_first_reader_with_parameters(self) -> None:
        """Lookup continues past missing readers and returns the first hit."""

        class _MissingReader:
            def get_camera_model_parameters(self, _camera_id: str) -> object:
                raise KeyError("front")

        class _PresentReader:
            def __init__(self) -> None:
                self.params = object()

            def get_camera_model_parameters(self, _camera_id: str) -> object:
                return self.params

        present = _PresentReader()

        self.assertIs(
            _get_camera_model_parameters({"missing": _MissingReader(), "present": present}, "front"),
            present.params,
        )


class TestCameraModelMismatchDiagnostic(ValidatorTestCase):
    """Model mismatch diagnostics are shared by structure and store checks."""

    def test_matching_models_return_no_diagnostic(self) -> None:
        """No diagnostic is emitted when the declared and observed models match."""
        self.assertIsNone(
            _camera_model_mismatch_diagnostic(
                "camera_model_family",
                "front",
                {"opencv_pinhole"},
                {"opencv_pinhole"},
                observed_path="worked_example.per_camera",
            )
        )

    def test_mismatched_models_report_camera_and_models(self) -> None:
        """Mismatch diagnostics include the camera ID and both model names."""
        diag = _camera_model_mismatch_diagnostic(
            "camera_model_family",
            "front",
            {"opencv_pinhole"},
            {"opencv_fisheye"},
            observed_path="worked_example.per_camera[0].model",
        )

        self.assertIsNotNone(diag)
        assert diag is not None
        self.assertIn("front", diag)
        self.assertIn("opencv_pinhole", diag)
        self.assertIn("opencv_fisheye", diag)
        self.assertIn("worked_example.per_camera[0].model", diag)


class TestCameraModelValueDiagnostics(ValidatorTestCase):
    """``camera_model_family.value`` entries need a camera_id and valid model."""

    def test_valid_value_entries_pass(self) -> None:
        """A well-formed ``{camera_id, model}`` list emits no diagnostics."""
        self.assertEqual(_camera_model_value_diagnostics("camera_model_family", _camera_model_entry()), [])

    def test_missing_model_is_flagged(self) -> None:
        """A value entry without ``model`` is reported rather than dropped."""
        entry = {"value": [{"camera_id": "front"}]}
        diags = _camera_model_value_diagnostics("camera_model_family", entry)
        self.assertOneDiagnosticMatching(diags, "value[0].model", "missing")

    def test_unknown_model_is_flagged(self) -> None:
        """A value entry with an unsupported ``model`` is reported."""
        entry = {"value": [{"camera_id": "front", "model": "thin_lens"}]}
        diags = _camera_model_value_diagnostics("camera_model_family", entry)
        self.assertOneDiagnosticMatching(diags, "value[0].model", "thin_lens")

    def test_blank_camera_id_is_flagged(self) -> None:
        """A value entry with a blank ``camera_id`` is reported."""
        entry = {"value": [{"camera_id": "  ", "model": "opencv_pinhole"}]}
        diags = _camera_model_value_diagnostics("camera_model_family", entry)
        self.assertOneDiagnosticMatching(diags, "value[0].camera_id", "non-blank")

    def test_duplicate_camera_id_is_flagged(self) -> None:
        """Repeated ``camera_id`` entries are reported, not silently merged."""
        entry = {
            "value": [
                {"camera_id": "front", "model": "opencv_pinhole"},
                {"camera_id": "front", "model": "opencv_fisheye"},
            ]
        }
        diags = _camera_model_value_diagnostics("camera_model_family", entry)
        self.assertOneDiagnosticMatching(diags, "duplicate camera_id", "front")

    def test_non_list_value_is_flagged(self) -> None:
        """A present-but-non-list ``value`` is reported as needing a list."""
        diags = _camera_model_value_diagnostics("camera_model_family", {"value": "scalar"})
        self.assertOneDiagnosticMatching(diags, "value must be a list")

    def test_missing_value_defers_to_generic_check(self) -> None:
        """A missing/``None`` ``value`` is left to the generic value validation."""
        self.assertEqual(_camera_model_value_diagnostics("camera_model_family", {}), [])


class TestValueFieldDiagnostic(ValidatorTestCase):
    """``_value_field_diagnostic``: ``value:`` accepts any YAML type per
    ``critical-parameters.md`` 'Entry schema'. The helper passes
    structural types (string, number, list, mapping) through and
    rejects only missing / blank / placeholder / empty-container
    values."""

    def test_accepts_string(self) -> None:
        """Enum-like string ``value:`` (e.g. ``spinning_direction: "cw"``) passes."""
        self.assertIsNone(_value_field_diagnostic("p", "cw"))

    def test_accepts_list_of_lists_matrix(self) -> None:
        """A 4x4 matrix as ``value:`` for ``T_lidar_rig`` passes."""
        matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.81],
            [0.0, 0.0, 0.0, 1.0],
        ]
        self.assertIsNone(_value_field_diagnostic("p", matrix))

    def test_accepts_list_of_dicts(self) -> None:
        """``camera_model_family`` as a list of dicts passes."""
        self.assertIsNone(_value_field_diagnostic("p", [{"camera_id": "front", "model": "opencv_pinhole"}]))

    def test_accepts_nested_mapping(self) -> None:
        """``sensor_inventory_decisions`` shape (dict-of-lists / nested mapping) passes."""
        value = {
            "cameras": [{"path": "front", "include": True}],
            "lidars": [{"path": "top", "include": True}],
        }
        self.assertIsNone(_value_field_diagnostic("p", value))

    def test_accepts_number(self) -> None:
        """A bare numeric ``value:`` (e.g. an offset constant) passes."""
        self.assertIsNone(_value_field_diagnostic("p", 1.81))

    def test_rejects_none(self) -> None:
        """Explicit ``None`` reports missing/null with a hint at the documented shape."""
        diag = _value_field_diagnostic("p", None)
        assert diag is not None
        self.assertIn("missing or null", diag)

    def test_rejects_blank_string(self) -> None:
        """A blank string ``value:`` is still rejected (string-typed missing)."""
        diag = _value_field_diagnostic("p", "   ")
        assert diag is not None
        self.assertIn("missing or blank", diag)

    def test_rejects_placeholder_string(self) -> None:
        """A string ``value:`` carrying a ``FILL IN`` marker is rejected."""
        diag = _value_field_diagnostic("p", "FILL IN")
        assert diag is not None
        self.assertIn("placeholder", diag)

    def test_rejects_empty_list(self) -> None:
        """An empty list ``value:`` is rejected -- declared shape with no content."""
        diag = _value_field_diagnostic("p", [])
        assert diag is not None
        self.assertIn("empty list", diag)

    def test_rejects_empty_dict(self) -> None:
        """An empty dict ``value:`` is rejected for the same reason as empty list."""
        diag = _value_field_diagnostic("p", {})
        assert diag is not None
        self.assertIn("empty dict", diag)

    def test_accepts_empty_list_for_documented_exception(self) -> None:
        """``camera_static_mask_handling: value: []`` is allowed (documented carve-out)."""
        self.assertIsNone(_value_field_diagnostic("p", [], parameter="camera_static_mask_handling"))

    def test_rejects_empty_dict_for_documented_exception_parameter(self) -> None:
        """``camera_static_mask_handling: value: {}`` is still rejected.

        The documented carve-out shape is ``value: []`` (no per-camera
        mask files shipped); ``value: {}`` is what the parameter's
        ``wrong_value_consequence`` cites as the failure mode (NuRec
        silently disables the mask channel). Allowing ``{}`` here
        would make the validator accept the very state the parameter
        was added to flag.
        """
        diag = _value_field_diagnostic("p", {}, parameter="camera_static_mask_handling")
        assert diag is not None
        self.assertIn("empty dict", diag)

    def test_rejects_empty_list_for_other_parameter(self) -> None:
        """A parameter outside ``PARAMETERS_ALLOWING_EMPTY_VALUE`` still rejects ``[]``."""
        diag = _value_field_diagnostic("p", [], parameter="T_lidar_rig")
        assert diag is not None
        self.assertIn("empty list", diag)


class TestCheckCameraCoverage(ValidatorTestCase):
    """``_check_camera_coverage``: cross-checks ``T_camera_rig.value``
    keys against ``worked_example.per_camera`` ids. Operates only on
    the documented dict shape (see ``critical-parameters.md`` 'Value
    field')."""

    def test_dict_shape_happy_path(self) -> None:
        """Dict-shaped value with matching per_camera produces no diagnostics."""
        entry: dict[str, Any] = {"value": {"front": [[]], "rear": [[]]}}
        self.assertEqual(_check_camera_coverage("T_camera_rig", entry, _per_camera("front", "rear")), [])

    def test_dict_shape_missing_camera_diag(self) -> None:
        """Dict-shaped value with a camera missing from per_camera reports a diag."""
        entry: dict[str, Any] = {"value": {"front": [[]], "rear": [[]]}}
        diags = _check_camera_coverage("T_camera_rig", entry, _per_camera("front"))
        self.assertOneDiagnosticMatching(diags, "declares cameras", "rear")

    def test_dict_shape_extra_camera_diag(self) -> None:
        """Dict-shaped value with an extra worked_example camera reports a diag."""
        entry: dict[str, Any] = {"value": {"front": [[]]}}
        diags = _check_camera_coverage("T_camera_rig", entry, _per_camera("front", "rear"))
        self.assertOneDiagnosticMatching(diags, "not declared", "rear")

    def test_non_dict_shape_skips_silently(self) -> None:
        """Coverage no-ops when ``value`` is any non-dict shape (string, list, ...)."""
        entry = {"value": "front,rear"}
        self.assertEqual(_check_camera_coverage("T_camera_rig", entry, _per_camera("front")), [])


_RANGE = "sequence_frame_range_decision"

_CAR = {
    "component": "default",
    "track_id": "car-1",
    "label_source": "EXTERNAL",
    "source_version": None,
    "source_timestamps_us": [500, 1200, 1800, 2500],
    "sample_source": "tracks/car-1",
}
"""Boundary-crossing track: source ``[500, 1200, 1800, 2500]`` over coverage ``[1000, 2000]``."""

_CAR_KEY = ("default", "car-1", "EXTERNAL", None)
_CAR_SELECTION = [1000, 1200, 1800, 2000]


def _range_entry(
    sampled: list[Any] | None = None, *, tracks_checked: Any = 1, **value_overrides: Any
) -> dict[str, Any]:
    """Return a ``sequence_frame_range_decision`` entry storing ``_CAR`` alone."""
    value: dict[str, Any] = {
        "source_track_count_in_coverage": 1,
        "stored_track_count": 1,
        "excluded_tracks": {},
        "source_observation_count_in_coverage": 2,
        "stored_observation_count": 4,
    }
    value.update(value_overrides)
    return {
        "parameter": _RANGE,
        "value": value,
        "worked_example": {"tracks_checked": tracks_checked, "sampled_tracks": [_CAR] if sampled is None else sampled},
    }


class TestCuboidTrackWorkedExampleStructure(ValidatorTestCase):
    """``_validate_worked_example_structure`` on ``sequence_frame_range_decision``."""

    def _diags(self, entry: dict[str, Any]) -> list[str]:
        return _validate_worked_example_structure(_RANGE, entry)

    def test_valid_entry_passes(self) -> None:
        self.assertEqual(self._diags(_range_entry()), [])

    def test_missing_count_is_reported(self) -> None:
        self.assertOneDiagnosticMatching(
            self._diags(_range_entry(stored_observation_count=None)), "stored_observation_count", "non-negative"
        )

    def test_source_tracks_neither_stored_nor_excluded_fail(self) -> None:
        entry = _range_entry(source_track_count_in_coverage=3)
        self.assertOneDiagnosticMatching(self._diags(entry), "2 source track(s) in coverage are neither stored")

    def test_stored_tracks_beyond_the_source_fail(self) -> None:
        entry = _range_entry(source_track_count_in_coverage=0, source_observation_count_in_coverage=4)
        self.assertOneDiagnosticMatching(self._diags(entry), "exceed the source by 1")

    def test_excluded_tracks_with_reasons_balance_the_count(self) -> None:
        entry = _range_entry(source_track_count_in_coverage=4, excluded_tracks={"class out of scope": 3})
        self.assertEqual(self._diags(entry), [])

    def test_excluded_tracks_must_be_a_reason_to_count_mapping(self) -> None:
        for excluded in (None, 3, {"": 1}, {"invalid box": 0}):
            with self.subTest(excluded=excluded):
                entry = _range_entry(excluded_tracks=excluded)
                self.assertOneDiagnosticMatching(self._diags(entry), "excluded_tracks must map")

    def test_recorded_counts_over_the_bound_fail(self) -> None:
        self.assertOneDiagnosticMatching(self._diags(_range_entry(stored_observation_count=5)), "exceeds", "= 4")

    def test_tracks_checked_must_equal_stored_tracks(self) -> None:
        self.assertOneDiagnosticMatching(self._diags(_range_entry(tracks_checked=0)), "tracks_checked=0")
        self.assertOneDiagnosticMatching(self._diags(_range_entry(tracks_checked=None)), "tracks_checked must be")

    def test_stored_tracks_require_a_sampled_track(self) -> None:
        self.assertOneDiagnosticMatching(self._diags(_range_entry(sampled=[])), "sampled_tracks is empty")

    def test_no_cuboids_allows_an_empty_sample(self) -> None:
        entry = _range_entry(
            sampled=[],
            tracks_checked=0,
            source_track_count_in_coverage=0,
            stored_track_count=0,
            source_observation_count_in_coverage=0,
            stored_observation_count=0,
        )
        self.assertEqual(self._diags(entry), [])

    def test_sampled_track_requires_its_label_source_identity(self) -> None:
        track = {k: v for k, v in _CAR.items() if k != "label_source"}
        self.assertOneDiagnosticMatching(self._diags(_range_entry(sampled=[track])), "'label_source' is missing")

    def test_repeated_source_timestamp_is_reported(self) -> None:
        track = dict(_CAR, source_timestamps_us=[1, 1])
        self.assertOneDiagnosticMatching(self._diags(_range_entry(sampled=[track])), "repeats a timestamp")


class TestCuboidTrackStoreDiagnostics(ValidatorTestCase):
    """``_cuboid_track_store_diagnostics`` against stored observations over coverage ``[1000, 2000]``."""

    COVERAGE = (1000, 2000)

    def _diags(self, stored: dict[tuple, list[int]], entry: dict[str, Any] | None = None) -> list[str]:
        return _cuboid_track_store_diagnostics(_RANGE, entry or _range_entry(), stored, self.COVERAGE)

    def test_matching_store_passes(self) -> None:
        self.assertEqual(self._diags({_CAR_KEY: [2000, 1000, 1800, 1200]}), [])

    def test_resampled_sampled_track_fails_counts_bound_and_selection(self) -> None:
        diags = self._diags({_CAR_KEY: list(range(1000, 2001, 100))})
        self.assertTrue(any("stored_observation_count=4 but the store holds 11" in d for d in diags), diags)
        self.assertTrue(any("more than source_observation_count_in_coverage" in d for d in diags), diags)
        self.assertTrue(any("car-1" in d and "7 not from the source" in d for d in diags), diags)

    def test_dropped_boundary_is_reported(self) -> None:
        diags = self._diags({_CAR_KEY: [1000, 1200, 1800]}, _range_entry(stored_observation_count=3))
        self.assertOneDiagnosticMatching(diags, "car-1", "1 missing", "[2000]")

    def test_sampled_track_absent_from_store_is_reported(self) -> None:
        stored = {("default", "other", "EXTERNAL", None): [1100, 1200]}
        diags = self._diags(stored, _range_entry(stored_observation_count=2))
        self.assertOneDiagnosticMatching(diags, "car-1", "stores 0 observations")

    def test_alternative_sources_and_versions_are_compared_separately(self) -> None:
        """The same component and track id under other sources or versions is not merged into the sample."""
        stored = {
            _CAR_KEY: _CAR_SELECTION,
            ("default", "car-1", "AUTOLABEL", None): [1100, 1500, 1900],
            ("default", "car-1", "EXTERNAL", "v2"): [1300, 1600],
        }
        entry = _range_entry(stored_track_count=3, stored_observation_count=9, source_observation_count_in_coverage=7)
        self.assertEqual(self._diags(stored, entry), [])

        v2 = dict(_CAR, source_version="v2", source_timestamps_us=[1300, 1600])
        entry["worked_example"]["sampled_tracks"] = [_CAR, v2]
        self.assertEqual(self._diags(stored, entry), [])

    def test_unsampled_resampled_track_is_not_verified_per_track(self) -> None:
        """Only sampled tracks are compared exactly; an unsampled one is covered by the aggregate bound alone."""
        truck_key = ("default", "truck-1", "EXTERNAL", None)
        within_bound = {_CAR_KEY: _CAR_SELECTION, truck_key: [1000, 1100, 1200]}
        entry = _range_entry(stored_track_count=2, stored_observation_count=7, source_observation_count_in_coverage=3)
        self.assertEqual(self._diags(within_bound, entry), [])

        over_bound = {_CAR_KEY: _CAR_SELECTION, truck_key: list(range(1000, 2001, 100))}
        entry = _range_entry(stored_track_count=2, stored_observation_count=15, source_observation_count_in_coverage=3)
        self.assertOneDiagnosticMatching(self._diags(over_bound, entry), "more than source_observation_count")

    def test_expected_selection_matches_the_conversion_rule(self) -> None:
        self.assertEqual(
            _expected_cuboid_track_timestamps([500, 1100, 1200, 2500], 1000, 2000), [1000, 1100, 1200, 2000]
        )
        self.assertEqual(_expected_cuboid_track_timestamps([1500], 1000, 2000), [1500])
        self.assertEqual(_expected_cuboid_track_timestamps([100, 500], 1000, 2000), [])


def _make_package(parent: Path, name: str) -> None:
    """Create a Python package directory ``name`` with an ``__init__.py``.

    Args:
        parent: Directory to create the package under.
        name: Package directory name.
    """
    pkg = parent / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")


class TestCheckPackageRename(ValidatorTestCase):
    """``_check_package_rename``: fail-closed guard rail for the single-token
    placeholder rename. ``ncore_converter_dir`` is a required build-summary
    field (nurec-onboard/steps/build-ncore-converter/SKILL.md), so an absent/blank value
    or a path that is not an existing directory is a hard error rather than a
    silent skip."""

    def test_missing_converter_dir_raises(self) -> None:
        """An absent ``ncore_converter_dir`` fails closed."""
        with self.assertRaises(ValidationError) as ctx:
            _check_package_rename({}, "summary.yaml")
        self.assertOneDiagnosticMatching(ctx.exception.diagnostics, "ncore_converter_dir", "missing or blank")

    def test_blank_converter_dir_raises(self) -> None:
        """A blank ``ncore_converter_dir`` fails closed."""
        with self.assertRaises(ValidationError) as ctx:
            _check_package_rename({"ncore_converter_dir": "   "}, "summary.yaml")
        self.assertOneDiagnosticMatching(ctx.exception.diagnostics, "ncore_converter_dir", "missing or blank")

    def test_non_string_converter_dir_raises(self) -> None:
        """A non-string ``ncore_converter_dir`` is treated as missing/blank."""
        with self.assertRaises(ValidationError) as ctx:
            _check_package_rename({"ncore_converter_dir": 123}, "summary.yaml")
        self.assertOneDiagnosticMatching(ctx.exception.diagnostics, "ncore_converter_dir", "missing or blank")

    def test_nonexistent_converter_dir_raises(self) -> None:
        """A present-but-nonexistent ``ncore_converter_dir`` fails closed."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does_not_exist")
            with self.assertRaises(ValidationError) as ctx:
                _check_package_rename({"ncore_converter_dir": missing}, "summary.yaml")
        self.assertOneDiagnosticMatching(ctx.exception.diagnostics, "ncore_converter_dir", "not an existing directory")

    def test_placeholder_package_not_renamed_raises(self) -> None:
        """A surviving ``ncore_example/`` placeholder package fails closed."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_package(Path(tmp), _PLACEHOLDER_PACKAGE)
            with self.assertRaises(ValidationError) as ctx:
                _check_package_rename({"ncore_converter_dir": tmp}, "summary.yaml")
        self.assertOneDiagnosticMatching(ctx.exception.diagnostics, _PLACEHOLDER_PACKAGE, "still")

    def test_invalid_package_name_raises(self) -> None:
        """A package directory that breaks the ``ncore_<dataset>`` pattern fails."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_package(Path(tmp), "ncore_Bad-Name")
            with self.assertRaises(ValidationError) as ctx:
                _check_package_rename({"ncore_converter_dir": tmp}, "summary.yaml")
        self.assertOneDiagnosticMatching(ctx.exception.diagnostics, "ncore_Bad-Name", "do not match")

    def test_renamed_package_passes(self) -> None:
        """A correctly renamed ``ncore_<dataset>/`` package passes."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_package(Path(tmp), "ncore_demo_dataset")
            _check_package_rename({"ncore_converter_dir": tmp}, "summary.yaml")

    def test_no_ncore_package_raises(self) -> None:
        """A converter dir with no ``ncore_*`` package fails closed.

        At the build step's final gate the template has been materialized and
        renamed, so a missing ``ncore_<dataset>/`` package is a defect, not an
        intermediate state.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _make_package(Path(tmp), "tools")  # only a non-ncore package present
            with self.assertRaises(ValidationError) as ctx:
                _check_package_rename({"ncore_converter_dir": tmp}, "summary.yaml")
        self.assertOneDiagnosticMatching(ctx.exception.diagnostics, "no top-level", "ncore_<dataset>")

    def test_rename_dropping_ncore_prefix_raises(self) -> None:
        """Renaming the package without the ``ncore_`` prefix cannot bypass the guard."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_package(Path(tmp), "waymo")  # placeholder renamed but prefix dropped
            with self.assertRaises(ValidationError) as ctx:
                _check_package_rename({"ncore_converter_dir": tmp}, "summary.yaml")
        self.assertOneDiagnosticMatching(ctx.exception.diagnostics, "no top-level", "ncore_<dataset>")


if __name__ == "__main__":
    unittest.main()
