# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validator for the ``validate-ncore-data`` diagnosis YAML.

Checks status/issue-bucket invariants and cross-checks
``validate_run.result_path`` (the typed ``validate_result.yaml`` sidecar
written by ``ncore_validation/validate.py``) so failed
validate.py output cannot be reported as a clean pass.

Usage: ``cd "$NUREC_ONBOARD_DIR" && bazel run //skills/nurec-onboard/validators:validate_ncore_data_validator -- <diagnosis.yaml>``.
Exit codes: ``0`` pass, ``1`` fail, ``2`` malformed input.
"""

from __future__ import annotations

import sys

from pathlib import Path
from typing import Any

from validators._validation_base import (
    ValidationError,
    as_dict,
    as_list,
    evaluate_validate_run,
    is_empty_container,
    is_int,
    load_yaml,
    parse_absolute_validate_result,
    step_validator_main,
    validate_converter_issue_entries,
    validate_result_exit_code_diagnostic,
    validate_result_internal_consistency_diagnostic,
)


_INPUT_DATA_SEVERITIES = frozenset({"error", "warning"})


def _validate_input_data_issue_entries(diag: dict[str, Any]) -> list[str]:
    """Validate structured input-data issue entries.

    Args:
        diag: Parsed validate-ncore-data diagnosis YAML document.

    Returns:
        Diagnostics for malformed input-data issue entries.
    """
    diagnostics: list[str] = []
    unresolved = diag.get("unresolved_issues")
    if not isinstance(unresolved, dict) or "input_data" not in unresolved:
        return diagnostics

    input_data = unresolved["input_data"]
    if not isinstance(input_data, list):
        return [f"'unresolved_issues.input_data' must be a list, got {type(input_data).__name__}"]

    for i, entry in enumerate(input_data):
        prefix = f"unresolved_issues.input_data[{i}]"
        if not isinstance(entry, dict):
            diagnostics.append(f"{prefix} must be a mapping, got {type(entry).__name__}")
            continue
        if "severity" not in entry:
            diagnostics.append(f"{prefix}.severity is missing (error | warning)")
            continue
        severity = entry["severity"]
        if severity not in _INPUT_DATA_SEVERITIES:
            diagnostics.append(f"{prefix}.severity must be error or warning, got {severity!r}")

    return diagnostics


def _validate_shape(diag: dict[str, Any]) -> list[str]:
    """Check required keys and types before cross-field invariants.

    Args:
        diag: Parsed validate-ncore-data diagnosis YAML document.

    Returns:
        Diagnostics for missing or malformed top-level fields.
    """
    diagnostics: list[str] = []
    if "validation_passed" not in diag:
        diagnostics.append("'validation_passed' is missing (bool, required)")
    elif not isinstance(diag["validation_passed"], bool):
        diagnostics.append(f"'validation_passed' must be a bool, got {type(diag['validation_passed']).__name__}")
    if "status" not in diag:
        diagnostics.append("'status' is missing (one of: pass | converter_issues | input_data_issues | both)")
    elif diag["status"] not in (
        "pass",
        "converter_issues",
        "input_data_issues",
        "both",
    ):
        diagnostics.append(
            f"'status' must be one of pass / converter_issues / input_data_issues / both, got {diag['status']!r}"
        )
    if "validate_run" not in diag:
        diagnostics.append(
            "'validate_run' block is missing (required: at least "
            "exit_code:int and result_path:str so the validator can "
            "cross-check the diagnosis against validate.py's typed "
            "validate_result.yaml output)"
        )
    elif not isinstance(diag["validate_run"], dict):
        diagnostics.append(f"'validate_run' must be a mapping, got {type(diag['validate_run']).__name__}")
    else:
        run = diag["validate_run"]
        if "exit_code" not in run:
            diagnostics.append("'validate_run.exit_code' is missing (int, required)")
        elif not is_int(run["exit_code"]):
            diagnostics.append(f"'validate_run.exit_code' must be an int, got {type(run['exit_code']).__name__}")
        if "result_path" not in run:
            diagnostics.append(
                "'validate_run.result_path' is missing (str, required: "
                "absolute path to validate.py's validate_result.yaml sidecar)"
            )
        elif not isinstance(run["result_path"], str) or not run["result_path"].strip():
            diagnostics.append("'validate_run.result_path' must be a non-blank string")
        elif not Path(run["result_path"]).is_absolute():
            diagnostics.append(f"'validate_run.result_path'={run['result_path']!r} must be an absolute path")
    if "unresolved_issues" in diag and not isinstance(diag["unresolved_issues"], dict):
        diagnostics.append(f"'unresolved_issues' must be a mapping, got {type(diag['unresolved_issues']).__name__}")
    return diagnostics


def _validate_invariants(diag: dict[str, Any], parsed_result: dict[str, Any] | None) -> list[str]:
    """Check diagnosis invariants and parsed-result honesty.

    Args:
        diag: Parsed validate-ncore-data diagnosis YAML document.
        parsed_result: Parsed ``validate_result.yaml`` sidecar from
            ``validate.py``, or ``None`` when no result file was
            available.

    Returns:
        Diagnostics for inconsistent status, issue buckets, or
        validate_result.yaml mismatches.
    """
    diagnostics: list[str] = []
    validation_passed = diag.get("validation_passed")
    status = diag.get("status")
    unresolved = as_dict(diag.get("unresolved_issues"))
    converter_issues = unresolved.get("converter")

    # validation_passed=false -> status in {converter_issues, both} +
    # unresolved.converter non-empty.
    if validation_passed is False:
        if status not in ("converter_issues", "both"):
            diagnostics.append(
                f"validation_passed=false but status={status!r}; status "
                f"MUST be 'converter_issues' or 'both' when validate.py "
                f"reports a failure (validate.py's Tier 1/2 checks cover "
                f"V4 structural correctness, which is the converter's "
                f"responsibility)"
            )
        if not converter_issues:
            diagnostics.append(
                "validation_passed=false but unresolved_issues.converter "
                "is empty; every failed validate.py check must be recorded "
                "as a structured entry under unresolved_issues.converter"
            )

    # status=pass -> validation_passed=true + every unresolved_issues.* empty.
    if status == "pass":
        if validation_passed is not True:
            diagnostics.append(
                f"status='pass' but validation_passed={validation_passed!r}; "
                f"a 'pass' status requires validation_passed=true"
            )
        for k, v in unresolved.items():
            if not is_empty_container(v):
                diagnostics.append(
                    f"status='pass' but unresolved_issues.{k} is "
                    f"non-empty; clear all unresolved issues or "
                    f"choose a different status"
                )

    # A non-pass status must be backed by entries in the bucket it names.
    # validation_passed=false already requires converter entries above.
    if status in ("converter_issues", "both") and validation_passed is not False and not converter_issues:
        diagnostics.append(
            f"status={status!r} but unresolved_issues.converter is empty; "
            f"record the converter issues or choose a different status"
        )
    if status in ("input_data_issues", "both") and is_empty_container(unresolved.get("input_data")):
        diagnostics.append(
            f"status={status!r} but unresolved_issues.input_data is empty; "
            f"record the input-data issues or choose a different status"
        )

    # validate_run / result cross-check. The exit-code branch still runs when
    # result parsing failed so missing sidecars cannot hide failed runs.
    run = as_dict(diag.get("validate_run"))
    exit_code = run.get("exit_code")
    had_fail, schema_invalid = evaluate_validate_run(parsed_result, exit_code)
    if had_fail and validation_passed is True:
        diagnostics.append(
            "validate_run reports failure (non-zero exit code, "
            "passed=false, or failed_checks in validate_result.yaml) "
            "but validation_passed=true; the diagnosis MUST set "
            "validation_passed=false when validate.py did not pass"
        )
    if schema_invalid:
        result_path = run.get("result_path", "<unknown>")
        diagnostics.append(
            f"validate_run.result_path={result_path!r} is not a valid "
            f"validate_result.yaml (wrong schema_version, program, or "
            f"missing typed fields passed/exit_code/failed_checks/warnings); "
            f"either validate.py was not actually run or the wrong "
            f"file was captured"
        )

    exit_code_diag = validate_result_exit_code_diagnostic("validate_run", parsed_result, exit_code)
    if exit_code_diag is not None:
        diagnostics.append(exit_code_diag)

    internal_diag = validate_result_internal_consistency_diagnostic(parsed_result)
    if internal_diag is not None:
        diagnostics.append(internal_diag)

    # Verbatim FAIL lines must appear in the diagnosis. Every entry in
    # validate_result.yaml's failed_checks[*].message must appear inside
    # some unresolved_issues.converter[*].error_message string so the
    # agent cannot paraphrase failures away.
    if parsed_result is not None:
        failures = as_list(parsed_result.get("failed_check_messages"))
        if isinstance(converter_issues, list):
            messages = [e.get("error_message", "") for e in converter_issues if isinstance(e, dict)]
        else:
            messages = []
        for failure in failures:
            if not any(failure in m for m in messages if isinstance(m, str)):
                diagnostics.append(
                    f"validate.py reported a failure ('{failure}') but it "
                    f"does not appear verbatim in any "
                    f"unresolved_issues.converter[*].error_message; the "
                    f"agent must quote validate.py errors exactly rather "
                    f"than paraphrasing them away"
                )

    return diagnostics


def validate_diagnosis(diag_path: str | Path) -> None:
    """Validate a diagnosis YAML file.

    Args:
        diag_path: Path to the validate-ncore-data diagnosis YAML file.

    Returns:
        None.

    Raises:
        ValidationError: If the YAML is malformed or violates diagnosis
            invariants.
    """
    doc = load_yaml(diag_path)
    if not isinstance(doc, dict):
        raise ValidationError([f"Top-level YAML document in {diag_path} must be a mapping, got {type(doc).__name__}"])

    diagnostics: list[str] = []
    diagnostics.extend(_validate_shape(doc))
    diagnostics.extend(validate_converter_issue_entries(doc))
    diagnostics.extend(_validate_input_data_issue_entries(doc))

    run = doc.get("validate_run")
    result_path = run.get("result_path") if isinstance(run, dict) else None
    parsed_result = parse_absolute_validate_result(result_path, diagnostics)

    diagnostics.extend(_validate_invariants(doc, parsed_result))

    if diagnostics:
        raise ValidationError(diagnostics)


_USAGE = (
    'Usage: cd "$NUREC_ONBOARD_DIR" && bazel run '
    "//skills/nurec-onboard/validators:validate_ncore_data_validator -- "
    "<diagnosis.yaml>\n"
    "Asserts cross-field invariants on the validate-ncore-data\n"
    "diagnosis YAML, plus anti-lying invariants that cross-check the\n"
    "diagnosis against validate.py's typed validate_result.yaml\n"
    "sidecar referenced by validate_run.result_path.\n"
)


main = step_validator_main(
    progname="validate_ncore_data_validator",
    usage=_USAGE,
    validate_fn=validate_diagnosis,
)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
