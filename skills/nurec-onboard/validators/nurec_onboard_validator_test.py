# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``validators/nurec_onboard_validator.py``.

Covers:

* The bidirectional ``steps_completed`` <-> ``runs[*]`` invariant: every
  step in ``steps_completed`` must have non-empty ``runs:``, and every
  step with non-empty ``runs:`` must appear in ``steps_completed``.
  The legitimate-skip path (omit from ``steps_completed`` AND empty
  ``runs:``) passes.
* The numeric-isinstance sites that route through ``is_int``:
  top-level ``execution.max_*_iterations``,
  ``steps.validate-ncore-data.summary.iterations``, the
  ``validate_run.exit_code`` shape check, the ``iteration_history``
  length-coupling, ``steps.run-nurec.summary.iterations`` /
  ``config_iterations``, and the
  ``iterations==0 + validation_passed=false`` hard-error invariant.
* The split between "missing key" and "wrong type" diagnostics on
  ``steps.<x>.{runs, summary, ...}`` (and on the top-level ``steps``
  block), and the contract that the invariants pass does not crash on
  a corrupted block that the shape pass has already diagnosed.
* The ``steps_completed`` duplicate-detection diagnostic and its
  interaction with the per-entry enum check.
* The top-level ``data_profile_present`` bool and the auto-audit ordering
  invariant: a profile-less input (``data_profile_present=false``) must
  have run ``audit-dataset`` before scope checks or any later step.
* The quality-metric requirement at ``workflow_status: success``: each of
  ``psnr``, ``ssim``, and ``lpips`` must carry a numeric value, so a run
  whose configured head reported nothing cannot claim success.
"""

from __future__ import annotations

import copy
import unittest

from typing import Any

from validators.nurec_onboard_validator import (
    _validate_invariants,
    _validate_shape,
)
from validators.validator_test_helpers import ValidatorTestCase


# Per-step ``runs[*].status`` values come from
# ``nurec_onboard_validator._STEP_RUN_STATUSES``: build/run-nurec use
# the ``success`` sentinel, validate-ncore-data uses ``pass``.
# ``steps.<step>.unresolved_issues`` is a mapping (typically bucketed
# by issue category), not a list. ``workflow_status: success`` requires
# ``steps.run-nurec.val_artifact_backstop.metrics_yaml_present: true``
# so the validator can confirm a metrics.yaml was produced before the
# workflow claims success.
_BASE_SUMMARY: dict[str, Any] = {
    "workflow_status": "success",
    "steps_completed": [
        "build-ncore-converter",
        "validate-ncore-data",
        "run-nurec",
    ],
    "data_profile_present": True,
    "execution": {
        "max_inner_iterations": 10,
        "max_outer_iterations": 5,
        "max_config_iterations": 5,
    },
    "steps": {
        "audit-dataset": {
            "runs": [],
            "summary": {"blocking_unresolved_count": 0},
            "unresolved_issues": {"critical_gaps": []},
        },
        "build-ncore-converter": {
            "runs": [
                {
                    "attempt": 1,
                    "status": "success",
                    "started_at": "2026-05-06T00:00:00Z",
                    "ended_at": "2026-05-06T00:00:01Z",
                    "duration_sec": 1.0,
                }
            ],
            "summary": {},
            "unresolved_issues": {},
        },
        "validate-ncore-data": {
            "runs": [
                {
                    "attempt": 1,
                    "status": "pass",
                    "started_at": "2026-05-06T00:00:02Z",
                    "ended_at": "2026-05-06T00:00:03Z",
                    "duration_sec": 1.0,
                }
            ],
            "summary": {
                "iterations": 1,
                "iteration_history": [
                    {
                        "errors": [],
                        "fixes_applied": [],
                        "validation_passed_after": True,
                    }
                ],
                "validation_passed": True,
                "validate_run": {"exit_code": 0, "result_path": "/tmp/validate_result.yaml"},
            },
            "unresolved_issues": {},
        },
        "run-nurec": {
            "runs": [
                {
                    "attempt": 1,
                    "status": "success",
                    "started_at": "2026-05-06T00:00:04Z",
                    "ended_at": "2026-05-06T00:00:05Z",
                    "duration_sec": 1.0,
                }
            ],
            "summary": {
                "reconstruction_status": "success",
                "metrics": {"psnr": 24.5, "ssim": 0.82, "lpips": 0.21},
                "iterations": 1,
                "config_iterations": 0,
                "config_fixes_applied": [],
            },
            "unresolved_issues": {},
            "val_artifact_backstop": {"metrics_yaml_present": True},
        },
    },
    "unresolved_issues": {
        "ncore_data_issues": [],
        "nurec_config_issues": [],
        "diagnostic_issues": [],
    },
}


def _base_summary() -> dict[str, Any]:
    """Return a fresh deep-copy of the happy-path workflow summary.

    Tests mutate the returned dict to provoke specific failures, so
    sharing the module-level base would leak state between tests.
    """
    return copy.deepcopy(_BASE_SUMMARY)


class _WorkflowLintMixin:
    """Helpers for running both shape and invariants passes on a doc.

    Used by every TestCase in this file so test bodies can call
    ``self._lint(doc)`` / ``self._matches(doc, substring)`` without
    each class redefining the bridge to the two validator entry points.
    """

    def _lint(self, doc: dict[str, Any]) -> list[str]:
        """Run shape + invariant checks and return concatenated diagnostics."""
        return _validate_shape(doc) + _validate_invariants(doc, None)

    def _matches(self, doc: dict[str, Any], substring: str) -> list[str]:
        """Return diagnostics from ``self._lint(doc)`` containing ``substring``."""
        return [d for d in self._lint(doc) if substring in d]


class TestHappyPath(ValidatorTestCase, _WorkflowLintMixin):
    """The base summary passes shape + invariants with zero diagnostics."""

    def test_happy_path_no_diagnostics(self) -> None:
        """The base summary passes with zero diagnostics."""
        self.assertEqual(self._lint(_base_summary()), [])


class TestStepsCompletedRunsInvariant(ValidatorTestCase, _WorkflowLintMixin):
    """Bidirectional ``steps_completed`` <-> ``runs[*]`` invariant."""

    def test_listed_but_empty_runs_flagged(self) -> None:
        """Forward direction: a listed step with no runs is flagged."""
        doc = _base_summary()
        doc["steps"]["run-nurec"]["runs"] = []
        self.assertOneDiagnosticMatching(self._lint(doc), "steps_completed lists 'run-nurec'")

    def test_runs_present_but_not_in_steps_completed_flagged(self) -> None:
        """Reverse direction: a step with runs but absent from list is flagged."""
        doc = _base_summary()
        doc["steps_completed"] = [
            "build-ncore-converter",
            "validate-ncore-data",
        ]  # run-nurec dropped while runs[*] still present
        self.assertOneDiagnosticMatching(self._lint(doc), "is not listed in steps_completed")

    def test_legitimate_skip_passes(self) -> None:
        """Skipped step: omit from steps_completed AND ``runs: []`` passes.

        Plausible scenario: ``validate-ncore-data`` returned
        ``input_data_issues``, the orchestrator skipped ``run-nurec``
        rather than launching a doomed reconstruction, and the workflow
        ended in ``partial_success`` status. The skipped step must be omitted
        from ``steps_completed`` and have ``runs: []``;
        ``reconstruction_status`` is ``not_run``.
        """
        doc = _base_summary()
        doc["steps_completed"] = [
            "build-ncore-converter",
            "validate-ncore-data",
        ]
        doc["steps"]["run-nurec"]["runs"] = []
        doc["steps"]["run-nurec"]["summary"]["reconstruction_status"] = "not_run"
        doc["workflow_status"] = "partial_success"
        self.assertEqual(self._matches(doc, "steps_completed"), [])

    def test_audit_step_can_be_skipped(self) -> None:
        """A skipped audit step needs no step block of its own.

        Asserted on a summary that does not claim success, which is where the
        blocking-count requirement applies.
        """
        doc = _base_summary()
        doc["workflow_status"] = "partial_success"
        del doc["steps"]["audit-dataset"]
        self.assertEqual(self._matches(doc, "audit-dataset"), [])

    def test_listed_audit_step_must_have_step_block(self) -> None:
        """Listed audit step must still emit a step block."""
        doc = _base_summary()
        doc["steps_completed"] = ["audit-dataset", *doc["steps_completed"]]
        del doc["steps"]["audit-dataset"]
        self.assertOneDiagnosticMatching(self._lint(doc), "steps_completed lists 'audit-dataset'", "is missing")

    def test_run_nurec_summary_can_be_absent_when_step_did_not_run(self) -> None:
        """Skipped run-nurec does not require a summary block."""
        doc = _base_summary()
        doc["workflow_status"] = "partial_success"
        doc["steps_completed"] = ["build-ncore-converter", "validate-ncore-data"]
        doc["steps"]["run-nurec"]["runs"] = []
        del doc["steps"]["run-nurec"]["summary"]
        doc["steps"]["validate-ncore-data"]["unresolved_issues"] = {"input_data": ["upstream issue"]}
        doc["steps"]["run-nurec"]["val_artifact_backstop"] = {}
        self.assertNoDiagnosticMatching(self._lint(doc), "steps.run-nurec.summary")

    def test_audit_step_with_runs_must_be_listed(self) -> None:
        """Audit runs must be reflected in ``steps_completed``."""
        doc = _base_summary()
        doc["steps"]["audit-dataset"]["runs"] = [
            {
                "attempt": 1,
                "status": "success",
                "started_at": "2026-05-06T00:00:00Z",
                "ended_at": "2026-05-06T00:00:01Z",
                "duration_sec": 1.0,
            }
        ]
        doc["steps"]["audit-dataset"]["summary"] = {
            "generated_input_yaml": "/tmp/input.yaml",
            "data_quality_report": "/tmp/report.md",
            "audit_profile": "/tmp/audit_profile.json",
            "unresolved_count": 0,
            "blocking_unresolved_count": 0,
        }
        self.assertOneDiagnosticMatching(self._lint(doc), "audit-dataset", "is not listed in steps_completed")

    def test_audit_step_partial_can_justify_partial_success(self) -> None:
        """Audit critical gaps are valid unresolved issues for partial_success."""
        doc = _base_summary()
        doc["workflow_status"] = "partial_success"
        doc["steps_completed"] = ["audit-dataset"]
        for step in ("build-ncore-converter", "validate-ncore-data", "run-nurec"):
            doc["steps"][step]["runs"] = []
        doc["steps"]["validate-ncore-data"]["summary"] = {}
        doc["steps"]["run-nurec"]["summary"]["reconstruction_status"] = "not_run"
        doc["steps"]["run-nurec"]["val_artifact_backstop"] = {}
        doc["steps"]["audit-dataset"] = {
            "runs": [
                {
                    "attempt": 1,
                    "status": "partial",
                    "started_at": "2026-05-06T00:00:00Z",
                    "ended_at": "2026-05-06T00:00:01Z",
                    "duration_sec": 1.0,
                }
            ],
            "summary": {
                "generated_input_yaml": "/tmp/input.yaml",
                "data_quality_report": "/tmp/report.md",
                "audit_profile": "/tmp/audit_profile.json",
                "unresolved_count": 1,
                "blocking_unresolved_count": 1,
            },
            "unresolved_issues": {"critical_gaps": ["data.camera_details"]},
        }
        self.assertEqual(self._lint(doc), [])

    def test_audit_blocking_unresolved_count_blocks_success(self) -> None:
        """Audit blocking_unresolved_count must affect workflow success invariants."""
        doc = _base_summary()
        doc["steps_completed"] = ["audit-dataset", *doc["steps_completed"]]
        doc["steps"]["audit-dataset"] = {
            "runs": [
                {
                    "attempt": 1,
                    "status": "partial",
                    "started_at": "2026-05-06T00:00:00Z",
                    "ended_at": "2026-05-06T00:00:01Z",
                    "duration_sec": 1.0,
                }
            ],
            "summary": {
                "generated_input_yaml": "/tmp/input.yaml",
                "data_quality_report": "/tmp/report.md",
                "audit_profile": "/tmp/audit_profile.json",
                "unresolved_count": 1,
                "blocking_unresolved_count": 1,
            },
            "unresolved_issues": {"critical_gaps": []},
        }
        self.assertOneDiagnosticMatching(self._lint(doc), "workflow_status='success'", "blocking_unresolved_count=1")

    def test_hand_authored_profile_blocking_count_blocks_success(self) -> None:
        """A blocking count gates the workflow even when audit-dataset skipped.

        The orchestrator applies the unresolved gate to a hand-authored
        ``data.profile`` too, so the count is authoritative on a summary whose
        ``audit-dataset`` step has no runs.
        """
        doc = _base_summary()
        doc["steps"]["audit-dataset"]["summary"] = {
            "unresolved_count": 1,
            "blocking_unresolved_count": 1,
        }
        self.assertOneDiagnosticMatching(self._lint(doc), "workflow_status='success'", "blocking_unresolved_count=1")

    def test_hand_authored_profile_blocking_count_must_be_an_int(self) -> None:
        """A non-int blocking count is rejected when audit-dataset skipped."""
        doc = _base_summary()
        doc["steps"]["audit-dataset"]["summary"] = {"blocking_unresolved_count": "1"}
        self.assertOneDiagnosticMatching(self._lint(doc), "blocking_unresolved_count", "must be an int")

    def test_success_must_record_the_blocking_count(self) -> None:
        """An absent count reads as zero, so success has to carry the field.

        Both shapes that record nothing are covered: the summary block present
        with the field dropped, and no ``audit-dataset`` step block at all.
        """
        no_field = _base_summary()
        del no_field["steps"]["audit-dataset"]["summary"]["blocking_unresolved_count"]
        no_block = _base_summary()
        del no_block["steps"]["audit-dataset"]
        for label, doc in (("field dropped", no_field), ("block dropped", no_block)):
            with self.subTest(shape=label):
                self.assertOneDiagnosticMatching(
                    self._lint(doc),
                    "workflow_status='success'",
                    "blocking_unresolved_count is missing",
                )

    def test_audit_advisory_unresolved_allows_success(self) -> None:
        """Advisory unresolved facts do not block success.

        This is the severity split: ``unresolved_count`` is non-zero but every
        one of those facts is advisory, so the run still completes.
        """
        doc = _base_summary()
        doc["steps_completed"] = ["audit-dataset", *doc["steps_completed"]]
        doc["steps"]["audit-dataset"] = {
            "runs": [
                {
                    "attempt": 1,
                    "status": "success",
                    "started_at": "2026-05-06T00:00:00Z",
                    "ended_at": "2026-05-06T00:00:01Z",
                    "duration_sec": 1.0,
                }
            ],
            "summary": {
                "generated_input_yaml": "/tmp/input.yaml",
                "data_quality_report": "/tmp/report.md",
                "audit_profile": "/tmp/audit_profile.json",
                "unresolved_count": 2,
                "blocking_unresolved_count": 0,
                "advisory_unresolved": [
                    "data.profile.cameras[0].image_format",
                    "data.profile.file_layout.metadata_files",
                ],
            },
            "unresolved_issues": {"critical_gaps": []},
        }
        self.assertEqual(self._lint(doc), [])

    def test_audit_blocking_count_cannot_exceed_total(self) -> None:
        """Blocking facts are a subset, so the blocking count is bounded."""
        doc = _base_summary()
        doc["steps_completed"] = ["audit-dataset", *doc["steps_completed"]]
        doc["steps"]["audit-dataset"] = {
            "runs": [
                {
                    "attempt": 1,
                    "status": "partial",
                    "started_at": "2026-05-06T00:00:00Z",
                    "ended_at": "2026-05-06T00:00:01Z",
                    "duration_sec": 1.0,
                }
            ],
            "summary": {
                "generated_input_yaml": "/tmp/input.yaml",
                "data_quality_report": "/tmp/report.md",
                "audit_profile": "/tmp/audit_profile.json",
                "unresolved_count": 1,
                "blocking_unresolved_count": 3,
            },
            "unresolved_issues": {"critical_gaps": []},
        }
        self.assertOneDiagnosticMatching(self._lint(doc), "blocking_unresolved_count", "exceeds")

    def test_audit_advisory_unresolved_must_be_a_list(self) -> None:
        """A non-list ``advisory_unresolved`` is rejected."""
        doc = _base_summary()
        doc["steps_completed"] = ["audit-dataset", *doc["steps_completed"]]
        doc["steps"]["audit-dataset"] = {
            "runs": [
                {
                    "attempt": 1,
                    "status": "success",
                    "started_at": "2026-05-06T00:00:00Z",
                    "ended_at": "2026-05-06T00:00:01Z",
                    "duration_sec": 1.0,
                }
            ],
            "summary": {
                "generated_input_yaml": "/tmp/input.yaml",
                "data_quality_report": "/tmp/report.md",
                "audit_profile": "/tmp/audit_profile.json",
                "unresolved_count": 1,
                "blocking_unresolved_count": 0,
                "advisory_unresolved": "data.profile.cameras[0].image_format",
            },
            "unresolved_issues": {"critical_gaps": []},
        }
        self.assertOneDiagnosticMatching(self._lint(doc), "advisory_unresolved' must be a list")

    def test_audit_blocking_unresolved_count_justifies_partial_success(self) -> None:
        """Audit blocking count can explain partial_success without duplicate issue buckets."""
        doc = _base_summary()
        doc["workflow_status"] = "partial_success"
        doc["steps_completed"] = ["audit-dataset", *doc["steps_completed"]]
        doc["steps"]["audit-dataset"] = {
            "runs": [
                {
                    "attempt": 1,
                    "status": "partial",
                    "started_at": "2026-05-06T00:00:00Z",
                    "ended_at": "2026-05-06T00:00:01Z",
                    "duration_sec": 1.0,
                }
            ],
            "summary": {
                "generated_input_yaml": "/tmp/input.yaml",
                "data_quality_report": "/tmp/report.md",
                "audit_profile": "/tmp/audit_profile.json",
                "unresolved_count": 1,
                "blocking_unresolved_count": 1,
            },
            "unresolved_issues": {"critical_gaps": []},
        }
        self.assertNoDiagnosticMatching(self._lint(doc), "no step-local unresolved_issues are recorded")


class TestNumericIsIntSites(ValidatorTestCase, _WorkflowLintMixin):
    """``is_int`` shape checks on the workflow summary's numeric fields."""

    def test_execution_caps_required(self) -> None:
        """Workflow summaries must carry effective execution caps."""
        doc = _base_summary()
        del doc["execution"]
        self.assertOneDiagnosticMatching(self._lint(doc), "'execution' is missing")

    def test_execution_caps_must_be_mapping(self) -> None:
        """The top-level ``execution`` block must be a mapping."""
        doc = _base_summary()
        doc["execution"] = []
        self.assertOneDiagnosticMatching(self._lint(doc), "'execution' must be a mapping")

    def test_execution_caps_bool_flagged(self) -> None:
        """``execution.max_inner_iterations: true`` is rejected."""
        doc = _base_summary()
        doc["execution"]["max_inner_iterations"] = True
        self.assertOneDiagnosticMatching(self._lint(doc), "'execution.max_inner_iterations' must be an int")

    def test_execution_caps_must_be_positive(self) -> None:
        """``execution.max_outer_iterations: 0`` is rejected."""
        doc = _base_summary()
        doc["execution"]["max_outer_iterations"] = 0
        self.assertOneDiagnosticMatching(self._lint(doc), "'execution.max_outer_iterations' must be > 0")

    def test_validate_iterations_bool_flagged(self) -> None:
        """``steps.validate-ncore-data.summary.iterations: true`` is rejected."""
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["iterations"] = True
        self.assertOneDiagnosticMatching(
            self._lint(doc), "steps.validate-ncore-data.summary.iterations' must be an int"
        )

    def test_validate_iterations_string_flagged(self) -> None:
        """``iterations: '1'`` (string typo) is rejected."""
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["iterations"] = "1"
        self.assertOneDiagnosticMatching(
            self._lint(doc), "steps.validate-ncore-data.summary.iterations' must be an int"
        )

    def test_validate_run_exit_code_bool_flagged(self) -> None:
        """``validate_run.exit_code: false`` is rejected."""
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["validate_run"]["exit_code"] = False
        self.assertOneDiagnosticMatching(self._lint(doc), "validate_run.exit_code' must be an int")

    def test_validate_run_result_path_relative_flagged(self) -> None:
        """A relative ``validate_run.result_path`` is rejected as cwd-dependent."""
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["validate_run"]["result_path"] = "validate_result.yaml"
        self.assertOneDiagnosticMatching(self._lint(doc), "validate_run.result_path", "absolute path")

    def test_validate_run_result_path_blank_skips_absolute_check(self) -> None:
        """A blank ``result_path`` fires only the non-blank diagnostic."""
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["validate_run"]["result_path"] = "   "
        diagnostics = self._lint(doc)
        self.assertOneDiagnosticMatching(diagnostics, "validate_run.result_path", "non-blank string")
        self.assertNoDiagnosticMatching(diagnostics, "validate_run.result_path", "absolute path")

    def test_iteration_history_len_must_match_iterations(self) -> None:
        """``len(iteration_history) != iterations`` is flagged."""
        doc = _base_summary()
        # iteration_history has length 1; bump iterations to 2.
        doc["steps"]["validate-ncore-data"]["summary"]["iterations"] = 2
        self.assertOneDiagnosticMatching(self._lint(doc), "iteration_history) must equal")

    def test_validate_advisory_warnings_non_list_flagged(self) -> None:
        """A non-list ``advisory_warnings`` is rejected."""
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["advisory_warnings"] = "global-shutter fallback"
        self.assertOneDiagnosticMatching(
            self._lint(doc), "steps.validate-ncore-data.summary.advisory_warnings' must be a list"
        )

    def test_validate_advisory_warnings_list_accepted(self) -> None:
        """A list ``advisory_warnings`` leaves an otherwise-success summary clean.

        Asserts zero diagnostics (not just the absence of an
        ``advisory_warnings`` message) to prove the advisory field does not
        perturb the success gate.
        """
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["advisory_warnings"] = [
            "global-shutter fallback (T_rs unpublished)",
        ]
        self.assertEqual(self._lint(doc), [])

    def test_nurec_iterations_bool_flagged(self) -> None:
        """``steps.run-nurec.summary.iterations: true`` is rejected."""
        doc = _base_summary()
        doc["steps"]["run-nurec"]["summary"]["iterations"] = True
        self.assertOneDiagnosticMatching(self._lint(doc), "steps.run-nurec.summary.iterations' must be an int")

    def test_nurec_config_iterations_bool_flagged(self) -> None:
        """``steps.run-nurec.summary.config_iterations: false`` is rejected."""
        doc = _base_summary()
        doc["steps"]["run-nurec"]["summary"]["config_iterations"] = False
        self.assertOneDiagnosticMatching(self._lint(doc), "steps.run-nurec.summary.config_iterations' must be an int")


class TestIterationsZeroInvariant(ValidatorTestCase, _WorkflowLintMixin):
    """``iterations == 0 ∧ validation_passed=false`` hard-error invariant."""

    def test_iterations_zero_with_failed_validation_flagged(self) -> None:
        """``iterations == 0 ∧ validation_passed=false`` is a hard error."""
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["iterations"] = 0
        doc["steps"]["validate-ncore-data"]["summary"]["iteration_history"] = []
        doc["steps"]["validate-ncore-data"]["summary"]["validation_passed"] = False
        self.assertOneDiagnosticMatching(self._lint(doc), "iterations=0 with")

    def test_iterations_bool_does_not_mask_zero_invariant(self) -> None:
        """Bool ``iterations`` must NOT match the ``== 0`` branch of the invariant.

        ``True == 1`` and ``False == 0`` under bare ``==`` comparison,
        which would have made ``iterations=False + validation_passed=false``
        look like the 0-iterations hard error. ``is_int`` excludes bool,
        so the hard-error branch never fires for bool inputs (the shape
        check above already flagged the bool separately).
        """
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["iterations"] = False
        doc["steps"]["validate-ncore-data"]["summary"]["validation_passed"] = False
        self.assertNoDiagnosticMatching(self._lint(doc), "iterations=0 with")


class TestPartialSuccessRequiresCapExhaustion(ValidatorTestCase, _WorkflowLintMixin):
    """Unresolved converter partial_success must exhaust the inner-loop cap."""

    def _failed_validate_partial(self, iterations: int, *, max_inner_iterations: int = 10) -> dict[str, Any]:
        """Return a partial-success summary with unresolved converter failures.

        Args:
            iterations: Number of inner-loop fix cycles recorded.
            max_inner_iterations: Effective configured inner-loop cap.

        Returns:
            Workflow summary document for validator tests.
        """
        doc = _base_summary()
        doc["workflow_status"] = "partial_success"
        doc["steps_completed"] = ["build-ncore-converter", "validate-ncore-data"]
        doc["execution"]["max_inner_iterations"] = max_inner_iterations
        doc["steps"]["validate-ncore-data"]["runs"][0]["status"] = "converter_issues"
        doc["steps"]["validate-ncore-data"]["summary"]["iterations"] = iterations
        doc["steps"]["validate-ncore-data"]["summary"]["iteration_history"] = [
            {
                "errors": ["validate.py failure"],
                "fixes_applied": ["attempted converter fix"],
                "validation_passed_after": False,
            }
            for _ in range(iterations)
        ]
        doc["steps"]["validate-ncore-data"]["summary"]["validation_passed"] = False
        doc["steps"]["validate-ncore-data"]["summary"]["validate_run"]["exit_code"] = 1
        doc["steps"]["validate-ncore-data"]["unresolved_issues"] = {
            "converter": [
                {
                    "error_message": "validate.py failure",
                    "demotion_attempted": False,
                    "demotion_criteria_failed": [],
                    "iterations_before_giving_up": iterations,
                }
            ],
            "input_data": [],
        }
        doc["steps"]["run-nurec"]["runs"] = []
        doc["steps"]["run-nurec"]["summary"]["reconstruction_status"] = "not_run"
        doc["steps"]["run-nurec"]["val_artifact_backstop"] = {}
        return doc

    def test_unresolved_converter_partial_before_cap_flagged(self) -> None:
        """``partial_success`` cannot stop the inner loop before its cap."""
        doc = self._failed_validate_partial(iterations=1, max_inner_iterations=10)
        self.assertOneDiagnosticMatching(
            self._lint(doc),
            "workflow_status='partial_success'",
            "iterations >= execution.max_inner_iterations",
            "iterations=1",
            "max_inner_iterations=10",
        )

    def test_unresolved_converter_partial_at_cap_passes_cap_invariant(self) -> None:
        """Cap-exhausted converter failures may report ``partial_success``."""
        doc = self._failed_validate_partial(iterations=10, max_inner_iterations=10)
        self.assertNoDiagnosticMatching(self._lint(doc), "iterations >= execution.max_inner_iterations")


class TestSuccessRequiresMetricHeads(ValidatorTestCase, _WorkflowLintMixin):
    """``workflow_status: success`` needs all three metric heads reported.

    ``metrics_yaml_present`` says the file exists, not that it carries
    values. The authored config enables the ssim and lpips heads and
    NuRec always emits psnr, so a head with no value means it was
    configured and then produced nothing.
    """

    def test_all_three_present_passes(self) -> None:
        """The base summary's three numeric heads raise nothing."""
        self.assertNoDiagnosticMatching(self._lint(_base_summary()), ".metrics")

    def test_missing_head_flagged(self) -> None:
        """A dropped head is named, with where to read its value."""
        doc = _base_summary()
        del doc["steps"]["run-nurec"]["summary"]["metrics"]["lpips"]
        self.assertOneDiagnosticMatching(
            self._lint(doc),
            "steps.run-nurec.summary.metrics.lpips is missing",
            "test/lpips",
        )

    def test_null_head_flagged(self) -> None:
        """An explicit null is a configured head that reported nothing."""
        doc = _base_summary()
        doc["steps"]["run-nurec"]["summary"]["metrics"]["ssim"] = None
        self.assertOneDiagnosticMatching(
            self._lint(doc),
            "steps.run-nurec.summary.metrics.ssim=None",
            "requires a finite numeric value",
        )

    def test_bool_head_is_not_numeric(self) -> None:
        """``True`` is not a metric value, though bool subclasses int."""
        doc = _base_summary()
        doc["steps"]["run-nurec"]["summary"]["metrics"]["psnr"] = True
        self.assertOneDiagnosticMatching(
            self._lint(doc),
            "steps.run-nurec.summary.metrics.psnr=True",
            "requires a finite numeric value",
        )

    def test_non_finite_heads_flagged(self) -> None:
        """NaN and the infinities are non-values that YAML loads as floats."""
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                doc = _base_summary()
                doc["steps"]["run-nurec"]["summary"]["metrics"]["lpips"] = value
                self.assertOneDiagnosticMatching(
                    self._lint(doc),
                    "steps.run-nurec.summary.metrics.lpips=",
                    "requires a finite numeric value",
                )

    def test_absent_metrics_block_flagged(self) -> None:
        """A summary with no metrics block at all names all three heads."""
        doc = _base_summary()
        del doc["steps"]["run-nurec"]["summary"]["metrics"]
        self.assertOneDiagnosticMatching(
            self._lint(doc),
            "steps.run-nurec.summary.metrics is missing",
            "psnr, ssim, lpips",
        )

    def test_non_mapping_metrics_reported_once(self) -> None:
        """A malformed block is a type diagnostic, not three missing heads."""
        doc = _base_summary()
        doc["steps"]["run-nurec"]["summary"]["metrics"] = [24.5, 0.82, 0.21]
        self.assertOneDiagnosticMatching(self._lint(doc), "steps.run-nurec.summary.metrics must be a mapping")

    def test_partial_success_tolerates_a_missing_head(self) -> None:
        """Only success gates on the heads; partial_success may lack one."""
        doc = _base_summary()
        doc["workflow_status"] = "partial_success"
        doc["steps"]["run-nurec"]["unresolved_issues"] = {"nurec_config": ["lpips head reported nothing"]}
        doc["steps"]["run-nurec"]["summary"]["metrics"]["lpips"] = None
        self.assertNoDiagnosticMatching(self._lint(doc), ".metrics.lpips")


class TestDataProfilePresent(ValidatorTestCase, _WorkflowLintMixin):
    """``data_profile_present`` shape and the auto-audit ordering invariant."""

    def _profile_absent_with_audit(self) -> dict[str, Any]:
        """Return a summary for a profile-less input that audited first.

        Returns:
            Workflow summary with ``data_profile_present`` false and a
            completed ``audit-dataset`` step.
        """
        doc = _base_summary()
        doc["data_profile_present"] = False
        doc["steps_completed"] = ["audit-dataset", *doc["steps_completed"]]
        doc["steps"]["audit-dataset"]["runs"] = [
            {
                "attempt": 1,
                "status": "success",
                "started_at": "2026-05-06T00:00:00Z",
                "ended_at": "2026-05-06T00:00:01Z",
                "duration_sec": 1.0,
            }
        ]
        doc["steps"]["audit-dataset"]["summary"] = {
            "generated_input_yaml": "/tmp/audit/input.yaml",
            "data_quality_report": "/tmp/audit/report.md",
            "audit_profile": "/tmp/audit/audit_profile.json",
            "unresolved_count": 0,
            "blocking_unresolved_count": 0,
        }
        return doc

    def test_data_profile_present_required(self) -> None:
        """The summary must carry ``data_profile_present``."""
        doc = _base_summary()
        del doc["data_profile_present"]
        self.assertOneDiagnosticMatching(self._lint(doc), "'data_profile_present' is missing")

    def test_data_profile_present_must_be_bool(self) -> None:
        """A non-bool ``data_profile_present`` is rejected."""
        doc = _base_summary()
        doc["data_profile_present"] = "yes"
        self.assertOneDiagnosticMatching(self._lint(doc), "'data_profile_present' must be a bool")

    def test_profile_absent_without_audit_flagged(self) -> None:
        """A profile-less input that skipped audit is flagged."""
        doc = _base_summary()
        doc["data_profile_present"] = False
        self.assertOneDiagnosticMatching(
            self._lint(doc),
            "data_profile_present=false requires audit-dataset to have run",
        )

    def test_profile_absent_with_audit_passes(self) -> None:
        """A profile-less input that audited first passes the invariant."""
        doc = self._profile_absent_with_audit()
        self.assertNoDiagnosticMatching(
            self._lint(doc),
            "data_profile_present=false requires audit-dataset to have run",
        )

    def test_profile_present_without_audit_passes(self) -> None:
        """A profile-bearing input may skip audit (no ordering diagnostic)."""
        doc = _base_summary()
        self.assertNoDiagnosticMatching(
            self._lint(doc),
            "data_profile_present=false requires audit-dataset to have run",
        )


class TestMissingKeyVsWrongType(ValidatorTestCase, _WorkflowLintMixin):
    """Absent keys must emit ``is missing``, not a wrong-type diagnostic.

    Background: ``dict.get`` defaults to ``None``, so a missing key
    used to surface as ``"... must be a list, got NoneType"`` because
    the shape check ran on the ``None`` default. The required-fields
    style used by sibling validators is ``"... is missing"``; tests
    here pin both halves: the missing-key diagnostic fires, AND the
    wrong-type diagnostic does NOT.
    """

    def test_runs_absent_diagnosed_as_missing(self) -> None:
        """Absent ``runs`` key emits ``is missing``, not a wrong-type diagnostic."""
        doc = _base_summary()
        del doc["steps"]["build-ncore-converter"]["runs"]
        self.assertOneDiagnosticMatching(self._lint(doc), "steps.build-ncore-converter.runs is missing")
        self.assertNoDiagnosticMatching(self._lint(doc), "must be a list, got NoneType")

    def test_runs_wrong_type_still_flagged(self) -> None:
        """Non-None, non-list ``runs`` still hits the wrong-type branch."""
        doc = _base_summary()
        doc["steps"]["build-ncore-converter"]["runs"] = "not-a-list"
        self.assertOneDiagnosticMatching(self._lint(doc), "steps.build-ncore-converter.runs must be a list, got str")

    def test_build_unresolved_issues_wrong_type_flagged(self) -> None:
        """Build-step unresolved issues must use the same mapping shape."""
        doc = _base_summary()
        doc["steps"]["build-ncore-converter"]["unresolved_issues"] = []
        self.assertOneDiagnosticMatching(
            self._lint(doc), "steps.build-ncore-converter.unresolved_issues", "must be a mapping"
        )

    def test_summary_absent_diagnosed_as_missing(self) -> None:
        """Absent ``summary`` key emits ``is missing``, not wrong-type."""
        doc = _base_summary()
        del doc["steps"]["validate-ncore-data"]["summary"]
        self.assertOneDiagnosticMatching(self._lint(doc), "steps.validate-ncore-data.summary is missing")
        self.assertNoDiagnosticMatching(self._lint(doc), "summary must be a mapping, got NoneType")

    def test_summary_wrong_type_still_flagged(self) -> None:
        """Non-None, non-dict ``summary`` still hits the wrong-type branch."""
        doc = _base_summary()
        doc["steps"]["validate-ncore-data"]["summary"] = ["not", "a", "mapping"]
        self.assertOneDiagnosticMatching(
            self._lint(doc), "steps.validate-ncore-data.summary must be a mapping, got list"
        )

    def test_reconstruction_status_absent_after_run_nurec_ran(self) -> None:
        """A run-nurec step with runs must record ``reconstruction_status``."""
        doc = _base_summary()
        self.assertTrue(doc["steps"]["run-nurec"]["runs"])
        del doc["steps"]["run-nurec"]["summary"]["reconstruction_status"]
        self.assertOneDiagnosticMatching(self._lint(doc), "'steps.run-nurec.summary.reconstruction_status' is required")

    def test_not_run_status_does_not_bypass_metrics_backstop(self) -> None:
        """Missing metrics after recorded run-nurec runs require failure, whatever the status."""
        doc = _base_summary()
        self.assertTrue(doc["steps"]["run-nurec"]["runs"])
        doc["steps"]["run-nurec"]["summary"]["reconstruction_status"] = "not_run"
        doc["steps"]["run-nurec"]["val_artifact_backstop"] = {"metrics_yaml_present": False}
        doc["workflow_status"] = "partial_success"
        self.assertOneDiagnosticMatching(
            self._lint(doc), "metrics_yaml_present=false requires workflow_status='failure'"
        )

    def test_steps_absent_diagnosed_as_missing(self) -> None:
        """Top-level ``steps`` key absent emits ``is missing``."""
        doc = _base_summary()
        del doc["steps"]
        self.assertOneDiagnosticMatching(self._lint(doc), "'steps' is missing")

    def test_steps_wrong_type_still_flagged(self) -> None:
        """Non-None, non-dict ``steps`` still hits the wrong-type branch."""
        doc = _base_summary()
        doc["steps"] = ["not", "a", "mapping"]
        self.assertOneDiagnosticMatching(self._lint(doc), "'steps' must be a mapping keyed by step name, got list")


class TestWrongTypeDoesNotCrashInvariants(ValidatorTestCase, _WorkflowLintMixin):
    """Wrong-type nested blocks must be diagnosed without crashing the invariants pass.

    Two-pass contract: ``_validate_shape`` emits the wrong-type
    diagnostic, and ``_validate_invariants`` does not mask it by
    raising on a subsequent ``.get(...)`` read on the corrupted block.
    Each test mutates one nested block to a non-dict / non-mapping
    value and asserts the shape diagnostic fires exactly once -- which
    only succeeds if both passes complete on the mutated document.
    """

    def _assert_wrong_type_diagnostic_without_crash(
        self,
        path: tuple[str, ...],
        bad_value: Any,
        expected_substring: str,
    ) -> None:
        """Inject ``bad_value`` at ``doc[*path]`` and assert exactly one matching diagnostic.

        Args:
            path: Sequence of keys naming the block to corrupt
                (e.g. ``("steps", "run-nurec", "summary")``).
            bad_value: Non-dict value to assign at ``doc[*path]``.
            expected_substring: Substring of the shape-validator
                diagnostic that must appear exactly once.
        """
        doc = _base_summary()
        target: Any = doc
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad_value
        self.assertOneDiagnosticMatching(self._lint(doc), expected_substring)

    def test_nurec_summary_wrong_type_does_not_crash(self) -> None:
        """Non-dict ``steps.run-nurec.summary`` does not crash the invariants pass.

        The invariants pass reads ``ns.get("reconstruction_status")`` and
        must treat a non-dict block as absent rather than raising.
        ``_validate_shape`` is responsible for the wrong-type diagnostic
        on the same block.
        """
        self._assert_wrong_type_diagnostic_without_crash(
            ("steps", "run-nurec", "summary"),
            ["not", "a", "mapping"],
            "'steps.run-nurec.summary' must be a mapping, got list",
        )

    def test_backstop_wrong_type_does_not_crash(self) -> None:
        """Non-dict ``val_artifact_backstop`` does not crash the invariants pass.

        The invariants pass reads ``backstop.get("metrics_yaml_present")``
        and must treat a non-dict block as absent rather than raising.
        """
        self._assert_wrong_type_diagnostic_without_crash(
            ("steps", "run-nurec", "val_artifact_backstop"),
            "string-not-mapping",
            "'steps.run-nurec.val_artifact_backstop' must be a mapping, got str",
        )

    def test_validate_run_wrong_type_does_not_crash(self) -> None:
        """Non-dict ``summary.validate_run`` does not crash the invariants pass.

        ``vs`` here is a mapping, so the invariants pass reaches the
        ``vs.get("validate_run")`` read at the anti-lying backstop; a
        non-dict child block must be treated as absent rather than
        raising. The shape validator diagnoses the ``validate_run``
        block separately via the ``is missing or not a mapping`` branch.
        """
        self._assert_wrong_type_diagnostic_without_crash(
            ("steps", "validate-ncore-data", "summary", "validate_run"),
            ["not", "a", "mapping"],
            "'steps.validate-ncore-data.summary.validate_run' is missing or not a",
        )


class TestStepsCompletedDuplicates(ValidatorTestCase, _WorkflowLintMixin):
    """``steps_completed`` duplicate-detection diagnostic + enum interaction."""

    def test_duplicate_flagged(self) -> None:
        """Duplicate entries in ``steps_completed`` are flagged.

        Previously a document with ``steps_completed: [build,
        build, ...]`` satisfied both the per-entry enum check (each
        entry is a valid step) and the bidirectional invariant (the
        deduped set contains the step with non-empty runs), so the
        duplicate was silently swallowed by the set comprehension.
        """
        doc = _base_summary()
        doc["steps_completed"] = [
            "build-ncore-converter",
            "build-ncore-converter",
            "validate-ncore-data",
            "run-nurec",
        ]
        self.assertOneDiagnosticMatching(
            self._lint(doc), "is a duplicate", "steps_completed[1]='build-ncore-converter'"
        )

    def test_invalid_name_does_not_pollute_dedup(self) -> None:
        """An invalid ``steps_completed`` entry is flagged, but its repeats
        are not double-flagged as duplicates.

        Repeats of an invalid name only trip the per-entry enum check
        (once per occurrence) and never enter the duplicate-tracking
        set. This keeps a single typo from cascading into a confusing
        duplicate diagnostic on top of the enum diagnostic.
        """
        doc = _base_summary()
        doc["steps_completed"] = [
            "build-ncore-converter",
            "validate-ncore-data",
            "run-nurec",
            "not-a-step",
            "not-a-step",
        ]
        enum_hits = self._matches(doc, "must be one of")
        self.assertEqual(len(enum_hits), 2, msg=f"expected one enum diag per typo, got {enum_hits}")
        self.assertNoDiagnosticMatching(self._lint(doc), "is a duplicate")


def _failing_summary() -> dict[str, Any]:
    """Return a happy-path summary mutated to look like validate.py failed.

    Coherent end-to-end: ``workflow_status="failure"`` with
    ``validation_passed=False`` and ``validate_run.exit_code=1``, plus
    the downstream ``run-nurec`` step marked as not run (validate.py
    failure gates reconstruction, so empty runs / ``reconstruction_status
    = "not_run"`` / ``metrics_yaml_present = False`` are the expected
    shape). Tests that need the exit-code mismatch case override
    ``validate_run.exit_code = 0`` explicitly.
    """
    doc = _base_summary()
    doc["workflow_status"] = "failure"
    # ``steps_completed`` must stay in lockstep with which steps have runs:
    # the bidirectional invariant rejects either direction's drift.
    doc["steps_completed"] = ["build-ncore-converter", "validate-ncore-data"]
    vs_step = doc["steps"]["validate-ncore-data"]
    vs_step["runs"][0]["status"] = "converter_issues"
    vs = vs_step["summary"]
    vs["validation_passed"] = False
    vs["validate_run"] = {"exit_code": 1, "result_path": "/tmp/validate_result.yaml"}
    vs["iteration_history"][0]["validation_passed_after"] = False
    ns_step = doc["steps"]["run-nurec"]
    ns_step["runs"] = []
    ns_step["summary"] = {
        "reconstruction_status": "not_run",
        "iterations": 0,
        "config_iterations": 0,
        "config_fixes_applied": [],
    }
    ns_step["val_artifact_backstop"] = {"metrics_yaml_present": False}
    return doc


class TestExitCodeMismatch(ValidatorTestCase):
    """Workflow-level cross-check: validate_run.exit_code MUST agree with the sidecar."""

    def test_mismatch_is_diagnosed(self) -> None:
        """Summary says exit_code=0 but sidecar says 1: hard diagnostic."""
        doc = _failing_summary()
        doc["steps"]["validate-ncore-data"]["summary"]["validate_run"]["exit_code"] = 0
        parsed_result = {
            "schema_valid": True,
            "passed": False,
            "result_exit_code": 1,
            "failed_check_messages": [],
            "warning_messages": [],
        }
        diagnostics = _validate_invariants(doc, parsed_result)
        self.assertOneDiagnosticMatching(diagnostics, "disagrees with validate_result.yaml exit_code")

    def test_match_is_clean(self) -> None:
        """Summary exit_code matches sidecar exit_code: no mismatch diagnostic."""
        doc = _failing_summary()
        parsed_result = {
            "schema_valid": True,
            "passed": False,
            "result_exit_code": 1,
            "failed_check_messages": [],
            "warning_messages": [],
        }
        diagnostics = _validate_invariants(doc, parsed_result)
        self.assertNoDiagnosticMatching(diagnostics, "disagrees with validate_result.yaml exit_code")

    def test_missing_result_exit_code_does_not_diagnose_mismatch(self) -> None:
        """No mismatch diagnostic when the sidecar lacks an exit_code field.

        That case already trips ``schema_invalid=True``; the mismatch
        check stays silent rather than emitting a second diagnostic.
        """
        doc = _failing_summary()
        parsed_result: dict[str, Any] = {
            "schema_valid": False,
            "passed": False,
            "result_exit_code": None,
            "failed_check_messages": [],
            "warning_messages": [],
        }
        diagnostics = _validate_invariants(doc, parsed_result)
        self.assertNoDiagnosticMatching(diagnostics, "disagrees with validate_result.yaml exit_code")


if __name__ == "__main__":
    unittest.main()
