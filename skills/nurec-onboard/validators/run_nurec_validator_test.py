# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``validators/run_nurec_validator.py``.

Covers:

* The ``severity: null`` loophole: an explicit ``null`` severity used to
  slip past both the required-keys loop (key present) and the validity
  check (``severity is not None`` short-circuit). The test pins the
  fixed behavior and checks the missing-key case still reports a single
  diagnostic, not a double-fire.
* ``_validate_metrics`` per-key bool-as-numeric exclusion (a metric
  value of ``True`` must not be accepted as numeric, even though
  ``bool`` subclasses ``int`` in Python).
* The decisions ``run-nurec/config-skeleton.yaml`` encodes, which are
  otherwise only prose in the step's SKILL and are what the authored
  config inherits on every run.
* The ``reconstruction_status`` partition on error severity: a
  warning-severity finding belongs on a successful run, so only a
  ``severity: error`` entry forces ``issues_detected``.
"""

from __future__ import annotations

import unittest

from pathlib import Path
from typing import Any

import yaml

from validators.run_nurec_validator import (
    _validate_invariants,
    _validate_issue_entries,
    _validate_metrics,
    _validate_subsampling,
)
from validators.validator_test_helpers import ValidatorTestCase


_STEP_DIR = Path(__file__).parents[1] / "nurec-onboard" / "steps" / "run-nurec"
_CONFIG_SKELETON = _STEP_DIR / "config-skeleton.yaml"


def _issue_entry(**overrides: object) -> dict[str, object]:
    """Return a minimally-valid issue entry with overrides applied.

    The base entry passes ``symptom`` / ``likely_cause`` / ``severity``
    so that overriding a single field exposes only that field's
    diagnostic in tests.

    Args:
        overrides: Field-level overrides.

    Returns:
        A dict suitable for inclusion in a ``bucket: [...]`` list.
    """
    base: dict[str, object] = {
        "symptom": "frame 0 cuboid spike",
        "likely_cause": "stale extrinsic in justifications.yaml",
        "severity": "error",
    }
    base.update(overrides)
    return base


def _doc(entry: dict[str, object]) -> dict[str, object]:
    """Wrap a single issue entry in a one-bucket diagnosis doc.

    Args:
        entry: Issue entry to place in ``ncore_data_issues[0]``.

    Returns:
        A doc that ``_validate_issue_entries`` can consume directly.
    """
    return {"ncore_data_issues": [entry]}


class TestIssueEntrySeverity(ValidatorTestCase):
    """Tests for ``_validate_issue_entries`` ``severity`` field handling."""

    def test_valid_error_passes(self) -> None:
        """A ``severity: error`` entry produces no severity diagnostic."""
        diags = _validate_issue_entries(
            _doc(_issue_entry(severity="error")),
            "ncore_data_issues",
            require_suggested_fix=False,
        )
        self.assertNoDiagnosticMatching(diags, "severity")

    def test_valid_warning_passes(self) -> None:
        """A ``severity: warning`` entry produces no severity diagnostic."""
        diags = _validate_issue_entries(
            _doc(_issue_entry(severity="warning")),
            "ncore_data_issues",
            require_suggested_fix=False,
        )
        self.assertNoDiagnosticMatching(diags, "severity")

    def test_invalid_string_flagged(self) -> None:
        """An out-of-whitelist string is flagged with ``got 'x'``."""
        diags = _validate_issue_entries(
            _doc(_issue_entry(severity="critical")),
            "ncore_data_issues",
            require_suggested_fix=False,
        )
        self.assertOneDiagnosticMatching(diags, "severity must be one of", "got 'critical'")

    def test_explicit_null_flagged(self) -> None:
        """``severity: null`` is reported as an invalid value, not silently accepted.

        The validity check uses ``"severity" in entry and ...`` rather than
        ``severity is not None and ...`` so that explicit ``null`` (key
        present, value is ``None``) takes the same path as a typo'd
        severity value -- the prior ``is not None`` short-circuit let
        ``severity: null`` slip through with no diagnostic.
        """
        diags = _validate_issue_entries(
            _doc(_issue_entry(severity=None)),
            "ncore_data_issues",
            require_suggested_fix=False,
        )
        self.assertOneDiagnosticMatching(diags, "severity must be one of", "got None")

    def test_missing_key_single_diagnostic(self) -> None:
        """A missing severity key produces only the missing-key diagnostic.

        Regression guard against a naive fix to the explicit-null path:
        changing the validity check to ``is None or ...`` would
        double-fire on missing keys (one for missing, one for "got
        None"). The ``"severity" in entry and ...`` form keeps missing
        keys flowing only through the required-keys loop above, so the
        diagnostic count stays at one.
        """
        entry = _issue_entry()
        del entry["severity"]
        diags = _validate_issue_entries(
            _doc(entry),
            "ncore_data_issues",
            require_suggested_fix=False,
        )
        self.assertOneDiagnosticMatching(diags, "'severity' is missing")
        self.assertNoDiagnosticMatching(diags, "severity must be one of")

    def test_integer_typo_flagged(self) -> None:
        """An int written for severity (typo / YAML coercion) is flagged."""
        diags = _validate_issue_entries(
            _doc(_issue_entry(severity=2)),
            "ncore_data_issues",
            require_suggested_fix=False,
        )
        self.assertOneDiagnosticMatching(diags, "severity must be one of", "got 2")


class TestValidateMetrics(ValidatorTestCase):
    """Tests for ``_validate_metrics`` value-shape checks."""

    def test_value_bool_flagged(self) -> None:
        """A bool metric value is flagged via ``is_finite_number``."""
        diags = _validate_metrics({"reconstruction_status": "success", "metrics": {"psnr": 25.0, "ssim": True}})
        self.assertOneDiagnosticMatching(diags, "metrics.ssim must be a finite number or null, got True")

    def test_non_finite_value_flagged(self) -> None:
        """A head that failed to compute reports NaN, not a missing key."""
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                diags = _validate_metrics({"reconstruction_status": "success", "metrics": {"psnr": value}})
                self.assertOneDiagnosticMatching(diags, "metrics.psnr must be a finite number or null")

    def test_valid_floats_pass(self) -> None:
        """All-numeric metrics produce no value-shape diagnostics."""
        diags = _validate_metrics({"reconstruction_status": "success", "metrics": {"psnr": 25.0, "ssim": 0.85}})
        self.assertNoDiagnosticMatching(diags, "must be numeric")

    def test_null_value_accepted(self) -> None:
        """``null`` is a valid sentinel for any metric (e.g. ssim/lpips when the metric was not computed)."""
        diags = _validate_metrics(
            {"reconstruction_status": "success", "metrics": {"psnr": 25.0, "ssim": None, "lpips": None}}
        )
        self.assertNoDiagnosticMatching(diags, "must be numeric")

    def test_missing_when_success_required(self) -> None:
        """``metrics`` missing under ``reconstruction_status=success`` is required."""
        diags = _validate_metrics({"reconstruction_status": "success"})
        self.assertOneDiagnosticMatching(diags, "'metrics' is missing")

    def test_missing_when_status_not_success_ok(self) -> None:
        """``metrics`` is optional when reconstruction did not succeed."""
        diags = _validate_metrics({"reconstruction_status": "issues_detected"})
        self.assertEqual(diags, [])

    def test_non_string_key_emits_single_diagnostic(self) -> None:
        """A non-string key paired with a non-numeric value emits one diagnostic, not two.

        Without a ``continue`` after the bad-key branch, the loop falls
        through to the numeric check on the bad entry and produces a
        second ``metrics.<key> must be numeric`` diagnostic whose
        ``metrics.<key>`` path is meaningless (the key isn't a string).
        Pin the single-diagnostic contract so a regression in the
        control flow re-introduces the duplicate.
        """
        diags = _validate_metrics({"reconstruction_status": "success", "metrics": {42: "not-a-number"}})
        self.assertOneDiagnosticMatching(diags, "'metrics' contains non-string or blank key")
        self.assertNoDiagnosticMatching(diags, "must be numeric or null")


class TestValidateSubsampling(ValidatorTestCase):
    """Tests for resolved NuRec image-subsampling validation."""

    def test_matching_divisible_subsampling_passes(self) -> None:
        """Matching factors that divide every selected camera dimension pass."""
        diags = _validate_subsampling(
            {
                "subsampling": {
                    "camera_pixel_sampler_subsample": 2,
                    "n_val_image_subsample": 2,
                    "n_train_sequential_image_subsample": 2,
                    "selected_camera_resolutions": [
                        {"camera_id": "front", "width": 1920, "height": 886},
                        {"camera_id": "side", "width": 1280, "height": 720},
                    ],
                }
            }
        )
        self.assertEqual(diags, [])

    def test_mismatched_factors_flagged(self) -> None:
        """Training and validation decode factors must resolve to the same value."""
        diags = _validate_subsampling(
            {
                "subsampling": {
                    "camera_pixel_sampler_subsample": 4,
                    "n_val_image_subsample": 2,
                    "n_train_sequential_image_subsample": 2,
                    "selected_camera_resolutions": [{"camera_id": "front", "width": 1920, "height": 888}],
                }
            }
        )
        self.assertOneDiagnosticMatching(diags, "must resolve to the same value")

    def test_non_divisible_dimension_flagged(self) -> None:
        """A selected camera dimension must be divisible by the resolved factor."""
        diags = _validate_subsampling(
            {
                "subsampling": {
                    "camera_pixel_sampler_subsample": 4,
                    "n_val_image_subsample": 4,
                    "n_train_sequential_image_subsample": 4,
                    "selected_camera_resolutions": [{"camera_id": "front", "width": 1920, "height": 886}],
                }
            }
        )
        self.assertOneDiagnosticMatching(diags, "height=886 is not divisible")

    def test_bool_factor_flagged(self) -> None:
        """A YAML bool is not accepted as an integer subsampling factor."""
        diags = _validate_subsampling(
            {
                "subsampling": {
                    "camera_pixel_sampler_subsample": True,
                    "n_val_image_subsample": 1,
                    "n_train_sequential_image_subsample": 1,
                    "selected_camera_resolutions": [{"camera_id": "front", "width": 1920, "height": 886}],
                }
            }
        )
        self.assertOneDiagnosticMatching(diags, "subsampling.camera_pixel_sampler_subsample must be a positive integer")


class TestStatusSeverityPartition(ValidatorTestCase):
    """``reconstruction_status`` partitions on error severity.

    A warning-severity finding did not stop the reconstruction, so it
    belongs on a successful run rather than forcing ``issues_detected``.
    """

    def _doc(self, status: str, **buckets: list[Any]) -> dict[str, object]:
        """Return a diagnosis with the given status and issue buckets."""
        doc: dict[str, object] = {
            "reconstruction_status": status,
            "ncore_data_issues": [],
            "nurec_config_issues": [],
        }
        doc.update(buckets)
        return doc

    def test_success_passes_without_error_entries(self) -> None:
        """A success raises nothing whether its buckets are empty or hold warnings."""
        cases = {
            "empty buckets": {},
            "warnings in both": {
                "ncore_data_issues": [_issue_entry(severity="warning")],
                "nurec_config_issues": [_issue_entry(severity="warning")],
            },
        }
        for name, buckets in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_validate_invariants(self._doc("success", **buckets)), [])

    def test_success_with_error_entry_flagged(self) -> None:
        """An error entry is incompatible with success, and names its bucket."""
        doc = self._doc("success", nurec_config_issues=[_issue_entry(severity="error")])
        self.assertOneDiagnosticMatching(
            _validate_invariants(doc),
            "reconstruction_status='success' requires no severity='error' entries",
            "nurec_config_issues",
        )

    def test_issues_detected_with_only_warnings_flagged(self) -> None:
        """A warning-only run must claim success, not issues_detected."""
        doc = self._doc("issues_detected", ncore_data_issues=[_issue_entry(severity="warning")])
        self.assertOneDiagnosticMatching(
            _validate_invariants(doc),
            "reconstruction_status='issues_detected' requires at least one",
            "a warning-only run is a success",
        )

    def test_issues_detected_with_error_entry_passes(self) -> None:
        """An error entry is what issues_detected reports, wherever it sits in the bucket."""
        cases = {
            "error alone": [_issue_entry(severity="error")],
            "error behind a warning": [_issue_entry(severity="warning"), _issue_entry(severity="error")],
        }
        for name, entries in cases.items():
            with self.subTest(case=name):
                doc = self._doc("issues_detected", ncore_data_issues=entries)
                self.assertEqual(_validate_invariants(doc), [])

    def test_malformed_entry_does_not_crash(self) -> None:
        """A non-mapping entry is left to the shape pass, not crashed on."""
        doc = self._doc("issues_detected", ncore_data_issues=["not a mapping"])
        self.assertOneDiagnosticMatching(
            _validate_invariants(doc),
            "reconstruction_status='issues_detected' requires at least one",
        )


class TestConfigSkeleton(unittest.TestCase):
    """Pin the decisions ``config-skeleton.yaml`` encodes.

    The skeleton is what every authored config starts from, so a field
    silently dropped or flipped here changes every subsequent run with
    nothing else to catch it. These assertions cover choices NuRec Onboard
    made; the shard-dependent values stay placeholders and are the
    agent's job to resolve.
    """

    doc: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = yaml.safe_load(_CONFIG_SKELETON.read_text())

    def test_parses_with_expected_top_level_blocks(self) -> None:
        """The skeleton is loadable YAML carrying its four blocks."""
        self.assertEqual(sorted(self.doc), ["dataset", "defaults", "model", "system"])

    def test_ssim_and_lpips_are_enabled(self) -> None:
        """Both metrics NuRec leaves off are turned on.

        A config inheriting NuRec's defaults reports PSNR alone. The
        workflow documents and diagnoses against all three, and the
        workflow summary requires all three at ``workflow_status:
        success``, so the skeleton has to enable the other two.
        """
        metrics = self.doc["system"]["test"]["metrics"]
        self.assertIs(metrics["ssim"]["enabled"], True)
        self.assertIs(metrics["lpips"]["enabled"], True)

    def test_all_three_subsample_factors_are_full_resolution(self) -> None:
        """Every image-subsample knob is pinned to 1.

        NuRec expects the validation and train-sequential factors to
        match the pixel sampler's. Two of the three inherit 4 from
        ``/dataset: ncore``, so a dropped field here quietly returns the
        run to quarter resolution rather than failing.
        """
        dataset = self.doc["dataset"]
        self.assertEqual(dataset["n_val_image_subsample"], 1)
        self.assertEqual(dataset["n_train_sequential_image_subsample"], 1)
        self.assertEqual(dataset["samplers"]["batch_sampler"]["camera_pixel_sampler"]["subsample"], 1)

    def test_prod_option_blocks_are_present(self) -> None:
        """Every SQA option block the step's prose explains is listed.

        The subsampling block is deliberately absent: NuRec ships only
        ``batch_sampler_2`` and ``batch_sampler_4``, so full resolution
        has no block to select.
        """
        listed = {key for entry in self.doc["defaults"] if isinstance(entry, dict) for key in entry}
        self.assertEqual(
            listed,
            {
                "/apps/prod/options/artifact",
                "/apps/prod/options/harmonizer/3dgut",
                "/apps/prod/options/mesh",
            },
        )

    def test_skill_references_the_skeleton(self) -> None:
        """The SKILL points at the file, so extracting it left a link."""
        self.assertIn("config-skeleton.yaml", (_STEP_DIR / "SKILL.md").read_text())


if __name__ == "__main__":
    unittest.main()
