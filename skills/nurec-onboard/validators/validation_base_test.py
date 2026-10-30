# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``validators/_validation_base.py``.

Covers:

* ``is_int`` and ``is_number`` predicates (Python's bool-as-int subclass
  quirk, valid inputs, invalid inputs).
* ``is_finite_number``, which adds the rejection of the NaN and infinity
  floats YAML resolves from ``.nan`` and ``.inf``.
* ``validate_step_run_entry`` shape check, including explicit-null
  treatment (a missing key and an explicit ``null`` produce the same
  single missing-key diagnostic).
* ``numeric_vec_diagnostics`` element-level numeric check (bool exclusion
  is part of the contract).
* ``evaluate_validate_run`` exit-code interpretation through ``is_int``.
* ``parse_validate_result`` schema strictness: missing or mis-typed
  ``passed``/``exit_code``/``failed_checks``/``warnings`` register as
  ``schema_valid=False`` so a minimal stub cannot bypass the sidecar
  receipt check.
"""

from __future__ import annotations

import tempfile
import unittest

from pathlib import Path
from typing import Any

import yaml

from validators._validation_base import (
    VALIDATE_RESULT_PROGRAM,
    VALIDATE_RESULT_SCHEMA_VERSION,
    ValidationError,
    evaluate_validate_run,
    is_finite_number,
    is_int,
    is_number,
    numeric_vec_diagnostics,
    parse_validate_result,
    validate_result_internal_consistency_diagnostic,
    validate_step_run_entry,
)
from validators.validator_test_helpers import ValidatorTestCase


class TestIsInt(ValidatorTestCase):
    """Tests for the ``is_int`` predicate."""

    def test_excludes_bool(self) -> None:
        """``is_int`` rejects bool because ``bool`` subclasses ``int`` in Python."""
        self.assertFalse(is_int(True))
        self.assertFalse(is_int(False))

    def test_accepts_real_ints(self) -> None:
        """``is_int`` accepts positive, zero, and negative ints."""
        self.assertTrue(is_int(0))
        self.assertTrue(is_int(1))
        self.assertTrue(is_int(-3))

    def test_rejects_other_types(self) -> None:
        """``is_int`` rejects float, str, None, and other non-int types."""
        self.assertFalse(is_int(1.5))
        self.assertFalse(is_int("1"))
        self.assertFalse(is_int(None))
        self.assertFalse(is_int([1]))


class TestIsNumber(ValidatorTestCase):
    """Tests for the ``is_number`` predicate."""

    def test_excludes_bool(self) -> None:
        """``is_number`` rejects bool, matching the ``is_int`` rule."""
        self.assertFalse(is_number(True))
        self.assertFalse(is_number(False))

    def test_accepts_int_and_float(self) -> None:
        """``is_number`` accepts both ints and floats including negatives."""
        self.assertTrue(is_number(0))
        self.assertTrue(is_number(1))
        self.assertTrue(is_number(0.5))
        self.assertTrue(is_number(-1.5))

    def test_rejects_other_types(self) -> None:
        """``is_number`` rejects str, None, and non-numeric containers."""
        self.assertFalse(is_number("1"))
        self.assertFalse(is_number(None))
        self.assertFalse(is_number([1.0]))


class TestIsFiniteNumber(ValidatorTestCase):
    """Tests for the ``is_finite_number`` predicate."""

    def test_rejects_non_finite_floats(self) -> None:
        """NaN and the infinities are the values ``is_number`` lets through."""
        self.assertTrue(is_number(float("nan")))
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertFalse(is_finite_number(value))

    def test_accepts_real_measurements(self) -> None:
        """Ordinary ints and floats, including zero and negatives, pass."""
        for value in (0, 1, 0.5, -1.5):
            with self.subTest(value=value):
                self.assertTrue(is_finite_number(value))

    def test_inherits_the_is_number_rejections(self) -> None:
        """Bool, str, and None are rejected before the finiteness check."""
        for value in (True, "1", None, [1.0]):
            with self.subTest(value=value):
                self.assertFalse(is_finite_number(value))


def _step_run(**overrides: object) -> dict[str, object]:
    """Return a minimally-valid step-run entry with overrides applied.

    Args:
        overrides: Field-level overrides.

    Returns:
        A dict suitable for passing to ``validate_step_run_entry``.
    """
    base: dict[str, object] = {
        "attempt": 1,
        "status": "success",
        "started_at": "2026-05-06T22:00:00Z",
        "ended_at": "2026-05-06T22:00:01Z",
        "duration_sec": 1.0,
    }
    base.update(overrides)
    return base


class TestValidateStepRunEntry(ValidatorTestCase):
    """Tests for ``validate_step_run_entry``."""

    def test_happy_path(self) -> None:
        """A complete, valid run entry produces no diagnostics."""
        self.assertEqual(validate_step_run_entry("run", _step_run()), [])

    def test_missing_attempt_single_diagnostic(self) -> None:
        """A missing ``attempt`` key is reported as missing exactly once."""
        entry = _step_run()
        del entry["attempt"]
        diags = validate_step_run_entry("run", entry)
        self.assertOneDiagnosticMatching(diags, "attempt is missing")
        self.assertEqual(sum(1 for d in diags if "attempt" in d), 1)

    def test_explicit_null_attempt_treated_as_missing(self) -> None:
        """Explicit ``None`` is treated the same as a missing key (no slip-through).

        The explicit-null and missing-key paths produce the same single
        ``... is missing`` diagnostic; without this fold-together, the
        per-field shape check below would silently let ``attempt: null``
        pass while the missing-key case still failed.
        """
        entry = _step_run(attempt=None)
        diags = validate_step_run_entry("run", entry)
        self.assertOneDiagnosticMatching(diags, "attempt is missing")
        self.assertEqual(sum(1 for d in diags if "attempt" in d), 1)

    def test_attempt_bool_rejected(self) -> None:
        """``attempt: true`` (bool) is flagged via ``is_int``."""
        diags = validate_step_run_entry("run", _step_run(attempt=True))
        self.assertOneDiagnosticMatching(diags, "attempt must be an integer >= 1")

    def test_attempt_zero_rejected(self) -> None:
        """``attempt: 0`` violates the ``>= 1`` constraint."""
        diags = validate_step_run_entry("run", _step_run(attempt=0))
        self.assertOneDiagnosticMatching(diags, "attempt must be an integer >= 1")

    def test_explicit_null_started_at_reported_as_missing(self) -> None:
        """Explicit ``started_at: null`` is reported as missing.

        Same fold-together as the ``attempt: null`` case: the per-field
        non-blank-string check would short-circuit on ``None``, so the
        explicit-null case must be caught by the required-keys loop.
        """
        diags = validate_step_run_entry("run", _step_run(started_at=None))
        self.assertOneDiagnosticMatching(diags, "started_at is missing")

    def test_duration_bool_rejected(self) -> None:
        """``duration_sec: true`` is flagged via ``is_number``."""
        diags = validate_step_run_entry("run", _step_run(duration_sec=True))
        self.assertOneDiagnosticMatching(diags, "duration_sec must be numeric and >= 0")

    def test_duration_negative_rejected(self) -> None:
        """Negative ``duration_sec`` violates the ``>= 0`` constraint."""
        diags = validate_step_run_entry("run", _step_run(duration_sec=-0.5))
        self.assertOneDiagnosticMatching(diags, "duration_sec must be numeric and >= 0")

    def test_status_whitelist_enforced(self) -> None:
        """``valid_statuses`` whitelists the exact ``status`` values allowed."""
        diags = validate_step_run_entry(
            "run",
            _step_run(status="weird"),
            valid_statuses=("success", "failure"),
        )
        self.assertOneDiagnosticMatching(diags, "must be one of")


class TestNumericVecDiagnostics(ValidatorTestCase):
    """Tests for ``numeric_vec_diagnostics``."""

    def test_rejects_bools(self) -> None:
        """Bool elements are flagged as non-numeric."""
        diags = numeric_vec_diagnostics("p", [True, False, True])
        self.assertEqual(len(diags), 3)
        for d in diags:
            self.assertIn("must be numeric, got bool", d)

    def test_bool_in_middle_position_still_caught(self) -> None:
        """A bool sneaking into a mostly-valid vector is still caught."""
        diags = numeric_vec_diagnostics("p", [1.0, True, 2.0])
        self.assertEqual(len(diags), 1)
        self.assertIn("p[1] must be numeric, got bool", diags[0])

    def test_accepts_valid_floats(self) -> None:
        """All-float vector at the expected length passes."""
        self.assertEqual(numeric_vec_diagnostics("p", [0.1, 0.2, 0.3]), [])

    def test_accepts_valid_ints(self) -> None:
        """All-int vector at the expected length passes."""
        self.assertEqual(numeric_vec_diagnostics("p", [1, 2, 3]), [])

    def test_wrong_length(self) -> None:
        """A short list reports a single length-mismatch diagnostic."""
        diags = numeric_vec_diagnostics("p", [1.0, 2.0])
        self.assertOneDiagnosticMatching(diags, "must be a 3-element list")

    def test_not_a_list(self) -> None:
        """A non-list value reports the length-mismatch diagnostic."""
        diags = numeric_vec_diagnostics("p", "not-a-list")
        self.assertOneDiagnosticMatching(diags, "must be a 3-element list")


class TestEvaluateValidateRun(ValidatorTestCase):
    """Tests for ``evaluate_validate_run`` exit-code interpretation."""

    def test_zero_exit_no_parsed_clean(self) -> None:
        """A clean exit with no parsed signals reports no failure."""
        had_fail, _ = evaluate_validate_run(None, 0)
        self.assertFalse(had_fail)

    def test_nonzero_exit_means_fail(self) -> None:
        """A non-zero int exit code is a real failure even with no parsed signals."""
        had_fail, _ = evaluate_validate_run(None, 1)
        self.assertTrue(had_fail)

    def test_bool_exit_does_not_leak_as_fail(self) -> None:
        """``exit_code: True`` (bool) must NOT register as a non-zero int.

        Naive ``isinstance(exit_code, int) and exit_code != 0`` would treat
        ``True`` as ``1`` and lift had_fail; ``is_int`` excludes bool, so
        the only failure path for bool inputs is the shape validator
        rejecting them upstream.
        """
        had_fail, _ = evaluate_validate_run(None, True)
        self.assertFalse(had_fail)

    def test_failed_check_messages_imply_fail(self) -> None:
        """Parsed ``failed_check_messages`` lift had_fail even on a 0 exit code."""
        had_fail, _ = evaluate_validate_run({"failed_check_messages": ["x"]}, 0)
        self.assertTrue(had_fail)

    def test_passed_false_implies_fail(self) -> None:
        """Typed ``passed: False`` lifts had_fail even when no FAIL messages were captured.

        Defense-in-depth: validate.py sets ``passed`` and
        ``failed_check_messages`` consistently from the same ctx.errors
        list, so any divergence between the two is itself a tampering
        signal. Both signals independently lift had_fail.
        """
        had_fail, _ = evaluate_validate_run({"passed": False, "failed_check_messages": []}, 0)
        self.assertTrue(had_fail)

    def test_passed_true_clean_no_fail(self) -> None:
        """Typed ``passed: True`` with empty failed_check_messages and zero exit is clean."""
        had_fail, _ = evaluate_validate_run({"passed": True, "failed_check_messages": []}, 0)
        self.assertFalse(had_fail)

    def test_schema_invalid_flag_lifted(self) -> None:
        """A parsed result missing ``schema_valid`` is flagged as schema-invalid."""
        _, schema_invalid = evaluate_validate_run({"passed": True, "failed_check_messages": []}, 0)
        self.assertTrue(schema_invalid)

    def test_schema_valid_flag_clears(self) -> None:
        """A parsed result with ``schema_valid: True`` clears the schema-invalid flag."""
        _, schema_invalid = evaluate_validate_run(
            {"schema_valid": True, "passed": True, "failed_check_messages": []}, 0
        )
        self.assertFalse(schema_invalid)

    def test_none_parsed_does_not_report_schema_invalid(self) -> None:
        """A missing parsed result does NOT double-report schema_invalid.

        The missing-file diagnostic from ``parse_validate_result``
        already carries that signal, so ``evaluate_validate_run`` only
        flags ``schema_invalid`` when given a dict that is structurally
        readable but misidentified.
        """
        _, schema_invalid = evaluate_validate_run(None, 0)
        self.assertFalse(schema_invalid)

    def test_internal_unexpected_exception_failed_check_lifts_had_fail(self) -> None:
        """A sidecar recording validate.py's internal crash flags had_fail.

        Internal validation crashes are represented as failed checks, so
        the parsed result must be treated like any other failed run.
        """
        parsed = {
            "schema_valid": True,
            "passed": False,
            "failed_check_messages": ["unexpected exception during checks: ValueError: bad"],
        }
        had_fail, schema_invalid = evaluate_validate_run(parsed, 1)
        self.assertTrue(had_fail)
        self.assertFalse(schema_invalid)

    def test_internal_interrupted_failed_check_lifts_had_fail(self) -> None:
        """A sidecar recording validate.py's interruption flags had_fail.

        Interruptions are represented as failed checks, so the parsed
        result must be treated like any other failed run.
        """
        parsed = {
            "schema_valid": True,
            "passed": False,
            "failed_check_messages": ["validation interrupted: KeyboardInterrupt: "],
        }
        had_fail, schema_invalid = evaluate_validate_run(parsed, 1)
        self.assertTrue(had_fail)
        self.assertFalse(schema_invalid)


def _well_formed_sidecar_doc(**overrides: Any) -> dict[str, Any]:
    """Return a minimally-valid sidecar YAML document, with overrides applied.

    Args:
        overrides: Field-level overrides. ``__delete__=[<keys>]`` removes
            top-level keys to test the missing-field code paths.

    Returns:
        A dict suitable for ``yaml.safe_dump`` into a sidecar fixture file.
    """
    base: dict[str, Any] = {
        "schema_version": VALIDATE_RESULT_SCHEMA_VERSION,
        "program": VALIDATE_RESULT_PROGRAM,
        "program_version": "1.2.3",
        "started_at": "2026-05-06T22:00:00Z",
        "ended_at": "2026-05-06T22:00:01Z",
        "duration_sec": 1.0,
        "exit_code": 0,
        "passed": True,
        "component_groups": ["/tmp/x.json"],
        "sample_frames": None,
        "failed_checks": [],
        "warnings": [],
    }
    to_delete = overrides.pop("__delete__", ())
    base.update(overrides)
    for key in to_delete:
        base.pop(key, None)
    return base


class TestParseValidateResult(ValidatorTestCase):
    """Tests for ``parse_validate_result``'s schema-strictness contract.

    A sidecar must include the receipt fields and typed pass/fail fields
    before it can be treated as schema-valid.
    """

    def _write_sidecar(self, tmpdir: Path, doc: dict[str, Any]) -> Path:
        """Serialize ``doc`` to a sidecar fixture file inside ``tmpdir``."""
        path = tmpdir / "validate_result.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        return path

    def test_well_formed_sidecar_is_schema_valid(self) -> None:
        """A fully-typed sidecar passes ``schema_valid`` and exposes typed fields."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_sidecar(Path(td), _well_formed_sidecar_doc())
            parsed = parse_validate_result(path)
        self.assertTrue(parsed["schema_valid"])
        self.assertIs(parsed["passed"], True)
        self.assertEqual(parsed["result_exit_code"], 0)

    def test_malformed_sidecars_are_schema_invalid(self) -> None:
        """Missing or mistyped required fields fail schema validation."""
        cases = [
            ("wrong schema_version", _well_formed_sidecar_doc(schema_version=999), True, 0),
            ("wrong program", _well_formed_sidecar_doc(program="other/tool.py"), True, 0),
            (
                "minimal stub",
                {"schema_version": VALIDATE_RESULT_SCHEMA_VERSION, "program": VALIDATE_RESULT_PROGRAM},
                None,
                None,
            ),
            ("missing passed", _well_formed_sidecar_doc(__delete__=["passed"]), None, 0),
            ("missing exit_code", _well_formed_sidecar_doc(__delete__=["exit_code"]), True, None),
            ("missing failed_checks", _well_formed_sidecar_doc(__delete__=["failed_checks"]), True, 0),
            ("missing warnings", _well_formed_sidecar_doc(__delete__=["warnings"]), True, 0),
            ("passed wrong type", _well_formed_sidecar_doc(passed="true"), None, 0),
            ("exit_code wrong type", _well_formed_sidecar_doc(exit_code=True), True, None),
            ("failed_checks wrong type", _well_formed_sidecar_doc(failed_checks={}), True, 0),
        ]
        for name, doc, expected_passed, expected_exit_code in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                path = self._write_sidecar(Path(td), doc)
                parsed = parse_validate_result(path)
            self.assertFalse(parsed["schema_valid"])
            self.assertEqual(parsed["passed"], expected_passed)
            self.assertEqual(parsed["result_exit_code"], expected_exit_code)

    def test_result_exit_code_round_trips_nonzero(self) -> None:
        """``result_exit_code`` mirrors a non-zero sidecar ``exit_code``."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_sidecar(Path(td), _well_formed_sidecar_doc(exit_code=1, passed=False))
            parsed = parse_validate_result(path)
        self.assertTrue(parsed["schema_valid"])
        self.assertEqual(parsed["result_exit_code"], 1)
        self.assertIs(parsed["passed"], False)

    def test_missing_file_raises(self) -> None:
        """A missing sidecar file raises ``ValidationError`` (not schema_invalid)."""
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nope.yaml"
            with self.assertRaises(ValidationError):
                parse_validate_result(missing)

    def test_non_mapping_yaml_raises(self) -> None:
        """A YAML file that is not a mapping raises ``ValidationError``."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "list.yaml"
            path.write_text("- a\n- b\n")
            with self.assertRaises(ValidationError):
                parse_validate_result(path)


class TestParseValidateResultEntryShapes(ValidatorTestCase):
    """Entries in ``failed_checks`` / ``warnings`` must carry non-blank messages.

    Without per-entry validation, a crafted sidecar with
    ``failed_checks: [{}, {"foo": "bar"}]`` would round-trip through
    ``_messages_from_entries`` as an empty message list while still
    satisfying the top-level list-type check, silently suppressing the
    cross-check invariants in the step validators.
    """

    def _parse(self, **overrides: Any) -> dict[str, Any]:
        """Write a sidecar with overrides and return the parsed dict."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "validate_result.yaml"
            path.write_text(yaml.safe_dump(_well_formed_sidecar_doc(**overrides), sort_keys=False))
            return parse_validate_result(path)

    def test_well_formed_entries_pass(self) -> None:
        """Entries with non-blank string messages are schema_valid."""
        parsed = self._parse(
            failed_checks=[{"tier": "Tier 1", "name": "shape", "message": "structural failure"}],
            warnings=[{"tier": "Tier 0", "name": "n", "message": "non-blocking warning"}],
            passed=False,
            exit_code=1,
        )
        self.assertTrue(parsed["schema_valid"])
        self.assertEqual(parsed["failed_check_messages"], ["structural failure"])
        self.assertEqual(parsed["warning_messages"], ["non-blocking warning"])

    def test_failed_check_without_message_rejects_sidecar(self) -> None:
        """A failed_check entry missing ``message`` flips schema_valid to False."""
        parsed = self._parse(failed_checks=[{"tier": "Tier 1", "name": "shape"}], passed=False, exit_code=1)
        self.assertFalse(parsed["schema_valid"])
        self.assertEqual(parsed["failed_check_messages"], [])

    def test_non_dict_entry_rejects_sidecar(self) -> None:
        """A non-mapping entry in ``failed_checks`` flips schema_valid to False."""
        parsed = self._parse(failed_checks=["just a string"], passed=False, exit_code=1)
        self.assertFalse(parsed["schema_valid"])

    def test_blank_message_rejects_sidecar(self) -> None:
        """A blank ``message`` value flips schema_valid to False."""
        parsed = self._parse(
            failed_checks=[{"tier": "Tier 1", "name": "shape", "message": "   "}],
            passed=False,
            exit_code=1,
        )
        self.assertFalse(parsed["schema_valid"])

    def test_non_string_message_rejects_sidecar(self) -> None:
        """A non-string ``message`` value flips schema_valid to False."""
        parsed = self._parse(failed_checks=[{"message": 42}], passed=False, exit_code=1)
        self.assertFalse(parsed["schema_valid"])

    def test_warnings_entries_are_also_validated(self) -> None:
        """The same per-entry check applies to ``warnings``."""
        parsed = self._parse(warnings=[{"tier": "Tier 0"}])
        self.assertFalse(parsed["schema_valid"])


class TestValidateResultInternalConsistency(ValidatorTestCase):
    """Tests for ``validate_result_internal_consistency_diagnostic``.

    validate.py writes ``passed = (exit_code == 0)`` as a single
    statement, so any disagreement in a captured sidecar indicates the
    file was hand-edited or fabricated after the run.
    """

    def test_consistent_clean(self) -> None:
        """``passed=True, exit_code=0`` and ``passed=False, exit_code=1`` are clean."""
        self.assertIsNone(
            validate_result_internal_consistency_diagnostic(
                {"passed": True, "result_exit_code": 0},
            )
        )
        self.assertIsNone(
            validate_result_internal_consistency_diagnostic(
                {"passed": False, "result_exit_code": 1},
            )
        )

    def test_passed_true_with_nonzero_exit_flagged(self) -> None:
        """``passed=True, exit_code=1`` is flagged as a tampering signal."""
        diag = validate_result_internal_consistency_diagnostic(
            {"passed": True, "result_exit_code": 1},
        )
        self.assertIsNotNone(diag)
        assert diag is not None  # for type narrowing
        self.assertIn("passed=True", diag)
        self.assertIn("exit_code=1", diag)

    def test_passed_false_with_zero_exit_flagged(self) -> None:
        """``passed=False, exit_code=0`` is flagged as a tampering signal."""
        diag = validate_result_internal_consistency_diagnostic(
            {"passed": False, "result_exit_code": 0},
        )
        self.assertIsNotNone(diag)
        assert diag is not None
        self.assertIn("passed=False", diag)
        self.assertIn("exit_code=0", diag)

    def test_none_parsed_result_silent(self) -> None:
        """A missing parsed result emits no diagnostic (covered by parse-time error)."""
        self.assertIsNone(validate_result_internal_consistency_diagnostic(None))

    def test_missing_fields_silent(self) -> None:
        """Missing or mis-typed ``passed`` / ``result_exit_code`` are silent here.

        Those cases are already reported via ``schema_valid=False`` in
        ``parse_validate_result``; double-reporting them would just add
        noise.
        """
        self.assertIsNone(
            validate_result_internal_consistency_diagnostic(
                {"passed": None, "result_exit_code": 0},
            )
        )
        self.assertIsNone(
            validate_result_internal_consistency_diagnostic(
                {"passed": True, "result_exit_code": None},
            )
        )
        # Bool exit_code (which Python treats as int) must not be accepted
        # via the ``is_int`` check; mirrors the rest of the sidecar contract.
        self.assertIsNone(
            validate_result_internal_consistency_diagnostic(
                {"passed": True, "result_exit_code": True},
            )
        )


if __name__ == "__main__":
    unittest.main()
