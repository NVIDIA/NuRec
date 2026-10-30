# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``validators/validate_ncore_data_validator.py``.

Covers the ``validate_run.exit_code`` ``is_int`` check: a bool exit code
(``True`` / ``False``) must not slip past this validator as ``1`` / ``0``
under Python's ``bool`` subclasses ``int`` quirk. The tests pin the
``is_int``-based shape check against regression.

Also covers the cross-check between the diagnosis-reported
``validate_run.exit_code`` and the sidecar's own ``exit_code`` field.
"""

from __future__ import annotations

import unittest

from typing import Any

from validators.validate_ncore_data_validator import (
    _validate_input_data_issue_entries,
    _validate_invariants,
    _validate_shape,
)
from validators.validator_test_helpers import ValidatorTestCase


def _base_diag() -> dict[str, Any]:
    """Return a minimally-valid validate-ncore-data diagnosis doc."""
    return {
        "status": "pass",
        "validation_passed": True,
        "validate_run": {"exit_code": 0, "result_path": "/tmp/validate_result.yaml"},
        "build_run": {
            "converter_repo": "/tmp/converter",
            "exit_code": 0,
            "stdout_path": "/tmp/b.stdout",
        },
        "unresolved_issues": {},
        "iterations": 1,
        "iteration_history": [],
    }


class TestValidateRunExitCode(ValidatorTestCase):
    """``validate_run.exit_code`` is an int (not bool, not string)."""

    def test_zero_passes(self) -> None:
        """A clean exit code 0 passes the shape check."""
        diags = _validate_shape(_base_diag())
        self.assertNoDiagnosticMatching(diags, "validate_run.exit_code", "must be an int")

    def test_nonzero_int_passes_shape(self) -> None:
        """A non-zero int exit code passes shape; severity is decided elsewhere."""
        diag = _base_diag()
        diag["validate_run"]["exit_code"] = 1
        diags = _validate_shape(diag)
        self.assertNoDiagnosticMatching(diags, "validate_run.exit_code", "must be an int")

    def test_bool_rejected(self) -> None:
        """``exit_code: True`` (bool) is flagged via ``is_int``."""
        diag = _base_diag()
        diag["validate_run"]["exit_code"] = True
        diags = _validate_shape(diag)
        self.assertOneDiagnosticMatching(diags, "validate_run.exit_code", "must be an int")

    def test_string_rejected(self) -> None:
        """``exit_code: '0'`` (string) is flagged."""
        diag = _base_diag()
        diag["validate_run"]["exit_code"] = "0"
        diags = _validate_shape(diag)
        self.assertOneDiagnosticMatching(diags, "validate_run.exit_code", "must be an int")

    def test_missing_flagged(self) -> None:
        """Missing ``exit_code`` is reported by the required-keys diagnostic."""
        diag = _base_diag()
        del diag["validate_run"]["exit_code"]
        diags = _validate_shape(diag)
        self.assertOneDiagnosticMatching(diags, "validate_run.exit_code")


class TestValidateRunResultPath(ValidatorTestCase):
    """``validate_run.result_path`` must be an absolute path."""

    def test_absolute_passes(self) -> None:
        """An absolute path passes the shape check."""
        diags = _validate_shape(_base_diag())
        self.assertNoDiagnosticMatching(diags, "validate_run.result_path", "absolute path")

    def test_relative_rejected(self) -> None:
        """A relative path is rejected as cwd-dependent."""
        diag = _base_diag()
        diag["validate_run"]["result_path"] = "validate_result.yaml"
        diags = _validate_shape(diag)
        self.assertOneDiagnosticMatching(diags, "validate_run.result_path", "absolute path")

    def test_blank_string_rejected_before_absolute_check(self) -> None:
        """A blank string fires the non-blank diagnostic, not the absolute one."""
        diag = _base_diag()
        diag["validate_run"]["result_path"] = "   "
        diags = _validate_shape(diag)
        self.assertOneDiagnosticMatching(diags, "validate_run.result_path", "non-blank string")
        self.assertNoDiagnosticMatching(diags, "validate_run.result_path", "absolute path")


def _converter_issues_diag(error_message: str = "") -> dict[str, Any]:
    """Return a diagnosis claiming validate.py failed.

    Coherent by default: ``validation_passed=false`` paired with
    ``validate_run.exit_code=1``. Tests that need the exit-code mismatch
    case override ``validate_run.exit_code = 0`` explicitly.
    """
    return {
        "status": "converter_issues",
        "validation_passed": False,
        "validate_run": {"exit_code": 1, "result_path": "/tmp/validate_result.yaml"},
        "build_run": {
            "converter_repo": "/tmp/converter",
            "exit_code": 0,
            "stdout_path": "/tmp/b.stdout",
        },
        "unresolved_issues": {
            "converter": [
                {
                    "error_message": error_message,
                    "demotion_attempted": False,
                    "demotion_criteria_failed": [],
                    "iterations_before_giving_up": 0,
                }
            ],
        },
        "iterations": 1,
        "iteration_history": [
            {
                "errors": [error_message] if error_message else [],
                "fixes_applied": [],
                "validation_passed_after": False,
            }
        ],
    }


class TestExitCodeMismatch(ValidatorTestCase):
    """``validate_run.exit_code`` must agree with ``validate_result.yaml`` exit_code."""

    def test_mismatch_is_diagnosed(self) -> None:
        """Diagnosis says exit_code=0 but the sidecar says 1: hard diagnostic."""
        diag = _converter_issues_diag(error_message="some failure")
        diag["validate_run"]["exit_code"] = 0
        parsed_result = {
            "schema_valid": True,
            "passed": False,
            "result_exit_code": 1,
            "failed_check_messages": ["some failure"],
            "warning_messages": [],
        }
        diagnostics = _validate_invariants(diag, parsed_result)
        self.assertOneDiagnosticMatching(diagnostics, "disagrees with validate_result.yaml exit_code")

    def test_match_is_clean(self) -> None:
        """Diagnosis exit_code matches sidecar exit_code: no mismatch diagnostic."""
        diag = _converter_issues_diag(error_message="some failure")
        parsed_result = {
            "schema_valid": True,
            "passed": False,
            "result_exit_code": 1,
            "failed_check_messages": ["some failure"],
            "warning_messages": [],
        }
        diagnostics = _validate_invariants(diag, parsed_result)
        self.assertNoDiagnosticMatching(diagnostics, "disagrees with validate_result.yaml exit_code")

    def test_missing_result_exit_code_does_not_diagnose_mismatch(self) -> None:
        """No mismatch diagnostic when the sidecar's exit_code field is missing.

        That case is already reported via ``schema_invalid=True`` (the
        typed-field check rejects sidecars lacking ``exit_code``); a
        second mismatch diagnostic would just add noise.
        """
        diag = _converter_issues_diag(error_message="some failure")
        parsed_result = {
            "schema_valid": False,
            "passed": False,
            "result_exit_code": None,
            "failed_check_messages": ["some failure"],
            "warning_messages": [],
        }
        diagnostics = _validate_invariants(diag, parsed_result)
        self.assertNoDiagnosticMatching(diagnostics, "disagrees with validate_result.yaml exit_code")

    def test_missing_diagnosis_exit_code_does_not_diagnose_mismatch(self) -> None:
        """No mismatch diagnostic when the diagnosis's exit_code is missing or non-int.

        That case is reported by the shape pass; the mismatch check stays
        silent rather than emitting a second confusing diagnostic.
        """
        diag = _converter_issues_diag(error_message="some failure")
        del diag["validate_run"]["exit_code"]
        parsed_result = {
            "schema_valid": True,
            "passed": True,
            "result_exit_code": 0,
            "failed_check_messages": [],
            "warning_messages": [],
        }
        diagnostics = _validate_invariants(diag, parsed_result)
        self.assertNoDiagnosticMatching(diagnostics, "disagrees with validate_result.yaml exit_code")


class TestStatusRequiresNamedBucket(ValidatorTestCase):
    """A non-pass status must have entries in the bucket it names."""

    def test_converter_issues_with_empty_converter_bucket(self) -> None:
        """``converter_issues`` after a passing validate.py still needs converter entries."""
        diag = _converter_issues_diag()
        diag["validation_passed"] = True
        diag["validate_run"]["exit_code"] = 0
        diag["unresolved_issues"]["converter"] = []
        diagnostics = _validate_invariants(diag, None)
        self.assertOneDiagnosticMatching(
            diagnostics, "status='converter_issues'", "unresolved_issues.converter is empty"
        )

    def test_input_data_issues_with_empty_input_data_bucket(self) -> None:
        """``input_data_issues`` needs input-data entries."""
        diag = _converter_issues_diag()
        diag["status"] = "input_data_issues"
        diag["validation_passed"] = True
        diag["validate_run"]["exit_code"] = 0
        diag["unresolved_issues"] = {"converter": [], "input_data": []}
        diagnostics = _validate_invariants(diag, None)
        self.assertOneDiagnosticMatching(
            diagnostics, "status='input_data_issues'", "unresolved_issues.input_data is empty"
        )


class TestInputDataIssueSeverity(ValidatorTestCase):
    """Input-data issues carry a severity for downstream routing."""

    def test_warning_and_error_pass(self) -> None:
        """Recognized severities pass the input-data issue shape check."""
        diag = _base_diag()
        diag["status"] = "input_data_issues"
        diag["unresolved_issues"] = {
            "input_data": [
                {
                    "issue": "Rolling shutter timing is unpublished.",
                    "severity": "warning",
                    "required_quality": "Camera timing model documented.",
                },
                {
                    "issue": "No ego poses overlap the selected interval.",
                    "severity": "error",
                    "required_quality": "Pose coverage over the sequence interval.",
                },
            ]
        }

        diagnostics = _validate_input_data_issue_entries(diag)

        self.assertNoDiagnosticMatching(diagnostics, "severity")

    def test_missing_severity_rejected(self) -> None:
        """Missing severity is rejected because the orchestrator gates on it."""
        diag = _base_diag()
        diag["status"] = "input_data_issues"
        diag["unresolved_issues"] = {
            "input_data": [
                {
                    "issue": "Rolling shutter timing is unpublished.",
                    "required_quality": "Camera timing model documented.",
                }
            ]
        }

        diagnostics = _validate_input_data_issue_entries(diag)

        self.assertOneDiagnosticMatching(diagnostics, "input_data[0].severity", "missing")

    def test_unknown_severity_rejected(self) -> None:
        """Only error and warning severities are routing-safe."""
        diag = _base_diag()
        diag["status"] = "input_data_issues"
        diag["unresolved_issues"] = {
            "input_data": [
                {
                    "issue": "Rolling shutter timing is unpublished.",
                    "severity": "info",
                    "required_quality": "Camera timing model documented.",
                }
            ]
        }

        diagnostics = _validate_input_data_issue_entries(diag)

        self.assertOneDiagnosticMatching(diagnostics, "severity", "error or warning")


if __name__ == "__main__":
    unittest.main()
