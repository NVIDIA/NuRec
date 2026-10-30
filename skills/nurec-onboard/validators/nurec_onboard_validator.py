# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validator for the ``nurec-onboard`` Workflow Summary.

Checks success/partial-success invariants, step timing,
final validate.py result honesty (via the typed validate_result.yaml
sidecar), iteration history shape, unresolved converter issue shape,
and the NuRec metrics backstop.

Usage: ``cd "$NUREC_ONBOARD_DIR" && bazel run //skills/nurec-onboard/validators:nurec_onboard_validator -- <workflow_summary.yaml>``.
Exit codes: ``0`` pass, ``1`` fail, ``2`` malformed input.
"""

from __future__ import annotations

import sys

from pathlib import Path
from typing import Any

from validators._validation_base import (
    ValidationError,
    as_dict,
    evaluate_validate_run,
    get_path,
    is_empty_container,
    is_finite_number,
    is_int,
    load_yaml,
    parse_absolute_validate_result,
    step_validator_main,
    validate_converter_issue_entries,
    validate_result_exit_code_diagnostic,
    validate_result_internal_consistency_diagnostic,
    validate_step_run_entry,
)


_VALID_WORKFLOW_STATUSES = ("success", "partial_success", "failure")
_VALID_RECONSTRUCTION_STATUSES = ("success", "issues_detected", "not_run")
_OPTIONAL_STEP_RUN_SECTIONS = ("audit-dataset",)
_REQUIRED_STEP_RUN_SECTIONS = ("build-ncore-converter", "validate-ncore-data", "run-nurec")
_STEP_RUN_SECTIONS = _OPTIONAL_STEP_RUN_SECTIONS + _REQUIRED_STEP_RUN_SECTIONS

# Per-step status enums for `runs[*].status`. Every step accepts `error` as
# an explicit fallback for orchestrator-side crashes/aborts (an attempt that
# could not reach a step-native outcome).
_STEP_RUN_STATUSES: dict[str, tuple[str, ...]] = {
    "audit-dataset": ("success", "partial", "error"),
    "build-ncore-converter": ("success", "failed", "error"),
    "validate-ncore-data": (
        "pass",
        "converter_issues",
        "input_data_issues",
        "both",
        "error",
    ),
    "run-nurec": ("success", "issues_detected", "not_run", "error"),
}

_VALIDATE_SUMMARY = "steps.validate-ncore-data.summary"
_AUDIT_SUMMARY = "steps.audit-dataset.summary"
_NUREC_SUMMARY = "steps.run-nurec.summary"
_NUREC_BACKSTOP = "steps.run-nurec.val_artifact_backstop"
_CONVERTER_ISSUES = "steps.validate-ncore-data.unresolved_issues.converter"
_VALIDATE_RESULT = f"{_VALIDATE_SUMMARY}.validate_run.result_path"
_EXECUTION_CAPS = ("max_inner_iterations", "max_outer_iterations", "max_config_iterations")

# Quality-metric heads a successful run must report. NuRec emits test/psnr
# whenever a validation frame has valid rays, and the authored config enables
# the ssim and lpips heads, so all three have a value on a healthy run. A null
# means the head was configured and then produced nothing -- the LPIPS weight
# fetch failing is the known case -- which is exactly what success must not
# hide. Excludes cpsnr: it needs per-camera semantic labels most shards do not
# carry, and emits per-class keys rather than one scalar.
_REQUIRED_METRIC_HEADS = ("psnr", "ssim", "lpips")


def _step_has_runs(doc: dict[str, Any], step: str) -> bool:
    """Return True when a workflow step recorded at least one run attempt.

    Args:
        doc: Parsed workflow-summary YAML document.
        step: Step name under ``steps``.

    Returns:
        Whether ``steps.<step>.runs`` is a non-empty list.
    """
    runs = get_path(doc, f"steps.{step}.runs")
    return isinstance(runs, list) and len(runs) > 0


def _validate_steps(doc: dict[str, Any]) -> list[str]:
    """Validate step sections and per-attempt run stats.

    A missing or non-mapping ``steps`` block returns early with a single
    diagnostic; downstream shape checks in :func:`_validate_shape` still run
    (they look up nested step paths via :func:`get_path` and surface their
    own missing-block diagnostics), so the operator gets a complete picture
    in one validator pass.

    Args:
        doc: Parsed workflow-summary YAML document.

    Returns:
        Diagnostics for missing or malformed ``steps`` entries, including
        per-step status-enum violations.
    """
    steps = doc.get("steps")
    if steps is None:
        return ["'steps' is missing"]
    if not isinstance(steps, dict):
        return [f"'steps' must be a mapping keyed by step name, got {type(steps).__name__}"]
    diagnostics: list[str] = []
    total_attempts = 0
    for step in _STEP_RUN_SECTIONS:
        step_block = steps.get(step)
        prefix = f"steps.{step}"
        if step_block is None:
            if step in _REQUIRED_STEP_RUN_SECTIONS:
                diagnostics.append(f"{prefix} is missing")
            continue
        if not isinstance(step_block, dict):
            diagnostics.append(f"{prefix} must be a mapping, got {type(step_block).__name__}")
            continue
        attempts = step_block.get("runs")
        if attempts is None:
            diagnostics.append(f"{prefix}.runs is missing")
            continue
        if not isinstance(attempts, list):
            diagnostics.append(f"{prefix}.runs must be a list, got {type(attempts).__name__}")
            continue
        summary = step_block.get("summary")
        summary_required = step != "run-nurec" or len(attempts) > 0
        if summary is None:
            if summary_required:
                diagnostics.append(f"{prefix}.summary is missing")
        elif not isinstance(summary, dict):
            diagnostics.append(f"{prefix}.summary must be a mapping, got {type(summary).__name__}")
        total_attempts += len(attempts)
        valid_statuses = _STEP_RUN_STATUSES.get(step)
        for i, entry in enumerate(attempts):
            diagnostics.extend(
                validate_step_run_entry(
                    f"{prefix}.runs[{i}]",
                    entry,
                    valid_statuses=valid_statuses,
                )
            )
    if total_attempts == 0:
        diagnostics.append("'steps' must contain at least one step run attempt")
    for step in steps:
        if step not in _STEP_RUN_SECTIONS:
            diagnostics.append(f"steps contains unknown step {step!r}; expected only {_STEP_RUN_SECTIONS!r}")
    return diagnostics


def _validate_shape(doc: dict[str, Any]) -> list[str]:
    """Check required keys and types before cross-field invariants.

    Args:
        doc: Parsed workflow-summary YAML document.

    Returns:
        Diagnostics for missing or malformed top-level fields.
    """
    diagnostics: list[str] = []
    if "workflow_status" not in doc:
        diagnostics.append("'workflow_status' is missing (one of: success | partial_success | failure)")
    elif doc["workflow_status"] not in _VALID_WORKFLOW_STATUSES:
        diagnostics.append(
            f"'workflow_status' must be one of {_VALID_WORKFLOW_STATUSES!r}, got {doc['workflow_status']!r}"
        )
    if "steps_completed" not in doc:
        diagnostics.append("'steps_completed' is missing (list of: " + " | ".join(_STEP_RUN_SECTIONS) + ")")
    else:
        steps_completed = doc["steps_completed"]
        if not isinstance(steps_completed, list):
            diagnostics.append(f"'steps_completed' must be a list, got {type(steps_completed).__name__}")
        else:
            seen: set[str] = set()
            for i, name in enumerate(steps_completed):
                if name not in _STEP_RUN_SECTIONS:
                    diagnostics.append(f"steps_completed[{i}]={name!r} must be one of {list(_STEP_RUN_SECTIONS)!r}")
                    continue
                if name in seen:
                    diagnostics.append(
                        f"steps_completed[{i}]={name!r} is a duplicate; 'steps_completed' must not repeat entries"
                    )
                else:
                    seen.add(name)
    execution = doc.get("execution")
    if "execution" not in doc:
        diagnostics.append("'execution' is missing (required: effective max_*_iterations caps)")
    elif not isinstance(execution, dict):
        diagnostics.append(f"'execution' must be a mapping, got {type(execution).__name__}")
    else:
        for field in _EXECUTION_CAPS:
            value = execution.get(field)
            if not is_int(value):
                diagnostics.append(f"'execution.{field}' must be an int")
            elif value <= 0:
                diagnostics.append(f"'execution.{field}' must be > 0")
    if "data_profile_present" not in doc:
        diagnostics.append("'data_profile_present' is missing (bool: whether the input carried data.profile at entry)")
    elif not isinstance(doc["data_profile_present"], bool):
        diagnostics.append(f"'data_profile_present' must be a bool, got {type(doc['data_profile_present']).__name__}")
    diagnostics.extend(_validate_steps(doc))

    audit_ran = _step_has_runs(doc, "audit-dataset")
    audit_summary = get_path(doc, _AUDIT_SUMMARY)
    if audit_summary is not None and not isinstance(audit_summary, dict):
        diagnostics.append(f"'{_AUDIT_SUMMARY}' must be a mapping, got {type(audit_summary).__name__}")
    elif isinstance(audit_summary, dict):
        for field in ("generated_input_yaml", "data_quality_report", "audit_profile"):
            value = audit_summary.get(field)
            if audit_ran and (not isinstance(value, str) or not value.strip()):
                diagnostics.append(f"'{_AUDIT_SUMMARY}.{field}' must be a non-blank string when audit-dataset ran")
        unresolved_count = audit_summary.get("unresolved_count")
        if audit_ran and not is_int(unresolved_count):
            diagnostics.append(f"'{_AUDIT_SUMMARY}.unresolved_count' must be an int when audit-dataset ran")
        elif is_int(unresolved_count) and unresolved_count < 0:
            diagnostics.append(f"'{_AUDIT_SUMMARY}.unresolved_count' must be >= 0")
        # Validated on both paths, not just when audit ran: the orchestrator
        # records a blocking count for a hand-authored profile too, and that
        # value gates the workflow.
        blocking_count = audit_summary.get("blocking_unresolved_count")
        if audit_ran and blocking_count is None:
            diagnostics.append(f"'{_AUDIT_SUMMARY}.blocking_unresolved_count' must be an int when audit-dataset ran")
        elif blocking_count is not None and not is_int(blocking_count):
            diagnostics.append(
                f"'{_AUDIT_SUMMARY}.blocking_unresolved_count' must be an int, got {type(blocking_count).__name__}"
            )
        elif is_int(blocking_count) and blocking_count < 0:
            diagnostics.append(f"'{_AUDIT_SUMMARY}.blocking_unresolved_count' must be >= 0")
        # The blocking facts are a subset of the unresolved ones, so a blocking
        # count above the total means one of the two was miscounted.
        if is_int(blocking_count) and is_int(unresolved_count) and blocking_count > unresolved_count:
            diagnostics.append(
                f"'{_AUDIT_SUMMARY}.blocking_unresolved_count'={blocking_count} exceeds "
                f"'{_AUDIT_SUMMARY}.unresolved_count'={unresolved_count}; blocking facts are a "
                "subset of the unresolved ones"
            )
        if "advisory_unresolved" in audit_summary and not isinstance(audit_summary["advisory_unresolved"], list):
            diagnostics.append(
                f"'{_AUDIT_SUMMARY}.advisory_unresolved' must be a list, got "
                f"{type(audit_summary['advisory_unresolved']).__name__}"
            )

    vs = get_path(doc, _VALIDATE_SUMMARY)
    if vs is None:
        if _step_has_runs(doc, "validate-ncore-data"):
            diagnostics.append(
                f"'{_VALIDATE_SUMMARY}' block is missing (required: at least validation_passed, iterations)"
            )
    elif not isinstance(vs, dict):
        diagnostics.append(f"'{_VALIDATE_SUMMARY}' must be a mapping, got {type(vs).__name__}")
    elif _step_has_runs(doc, "validate-ncore-data"):
        if "validation_passed" not in vs:
            diagnostics.append(f"'{_VALIDATE_SUMMARY}.validation_passed' is missing (bool, required)")
        elif not isinstance(vs["validation_passed"], bool):
            diagnostics.append(
                f"'{_VALIDATE_SUMMARY}.validation_passed' must be a bool, got {type(vs['validation_passed']).__name__}"
            )
        if "iterations" not in vs:
            diagnostics.append(
                f"'{_VALIDATE_SUMMARY}.iterations' is missing (int, "
                "required: count of validate -> build(fix) -> validate "
                "cycles run)"
            )
        elif not is_int(vs["iterations"]):
            diagnostics.append(
                f"'{_VALIDATE_SUMMARY}.iterations' must be an int, got {type(vs['iterations']).__name__}"
            )
        run = vs.get("validate_run")
        if not isinstance(run, dict):
            diagnostics.append(
                f"'{_VALIDATE_SUMMARY}.validate_run' is missing or not a "
                "mapping (required: exit_code:int, result_path:str)"
            )
        else:
            exit_code = run.get("exit_code")
            if not is_int(exit_code):
                diagnostics.append(f"'{_VALIDATE_SUMMARY}.validate_run.exit_code' must be an int")
            result_path = run.get("result_path")
            if not isinstance(result_path, str) or not result_path.strip():
                diagnostics.append(f"'{_VALIDATE_SUMMARY}.validate_run.result_path' must be a non-blank string")
            elif not Path(result_path).is_absolute():
                diagnostics.append(
                    f"'{_VALIDATE_SUMMARY}.validate_run.result_path'={result_path!r} must be an absolute path"
                )
        iterations = vs.get("iterations")
        history = vs.get("iteration_history")
        if is_int(iterations):
            if iterations >= 1 and not isinstance(history, list):
                diagnostics.append(f"'{_VALIDATE_SUMMARY}.iteration_history' must be a list when iterations >= 1")
            elif isinstance(history, list) and len(history) != iterations:
                diagnostics.append(
                    f"len({_VALIDATE_SUMMARY}.iteration_history) must equal {_VALIDATE_SUMMARY}.iterations"
                )
        if isinstance(history, list):
            for i, entry in enumerate(history):
                item = f"{_VALIDATE_SUMMARY}.iteration_history[{i}]"
                if not isinstance(entry, dict):
                    diagnostics.append(f"{item} must be a mapping, got {type(entry).__name__}")
                    continue
                if not isinstance(entry.get("errors"), list):
                    diagnostics.append(f"{item}.errors must be a list")
                if not isinstance(entry.get("fixes_applied"), list):
                    diagnostics.append(f"{item}.fixes_applied must be a list")
                if not isinstance(entry.get("validation_passed_after"), bool):
                    diagnostics.append(f"{item}.validation_passed_after must be a bool")
        if "advisory_warnings" in vs and not isinstance(vs["advisory_warnings"], list):
            diagnostics.append(
                f"'{_VALIDATE_SUMMARY}.advisory_warnings' must be a list, got {type(vs['advisory_warnings']).__name__}"
            )

    ns = get_path(doc, _NUREC_SUMMARY)
    if ns is not None and not isinstance(ns, dict):
        diagnostics.append(f"'{_NUREC_SUMMARY}' must be a mapping, got {type(ns).__name__}")
    elif isinstance(ns, dict):
        if "reconstruction_status" in ns:
            recon = ns["reconstruction_status"]
            if recon not in _VALID_RECONSTRUCTION_STATUSES:
                diagnostics.append(
                    f"{_NUREC_SUMMARY}.reconstruction_status must be one of "
                    f"{_VALID_RECONSTRUCTION_STATUSES!r}, got {recon!r}"
                )
        for field in ("iterations", "config_iterations"):
            if field in ns and not is_int(ns[field]):
                diagnostics.append(f"'{_NUREC_SUMMARY}.{field}' must be an int, got {type(ns[field]).__name__}")
        if "config_fixes_applied" in ns and not isinstance(ns["config_fixes_applied"], list):
            diagnostics.append(
                f"'{_NUREC_SUMMARY}.config_fixes_applied' must be a list, got "
                f"{type(ns['config_fixes_applied']).__name__}"
            )
    if _step_has_runs(doc, "run-nurec") and (
        ns is None or (isinstance(ns, dict) and "reconstruction_status" not in ns)
    ):
        diagnostics.append(
            f"'{_NUREC_SUMMARY}.reconstruction_status' is required when steps.run-nurec.runs is non-empty"
        )
    for step in ("audit-dataset", "build-ncore-converter", "validate-ncore-data", "run-nurec"):
        unresolved = get_path(doc, f"steps.{step}.unresolved_issues")
        if unresolved is not None and not isinstance(unresolved, dict):
            diagnostics.append(f"'steps.{step}.unresolved_issues' must be a mapping, got {type(unresolved).__name__}")
    backstop = get_path(doc, _NUREC_BACKSTOP)
    if backstop is not None:
        if not isinstance(backstop, dict):
            diagnostics.append(f"'{_NUREC_BACKSTOP}' must be a mapping, got {type(backstop).__name__}")
        elif "metrics_yaml_present" in backstop and not isinstance(backstop["metrics_yaml_present"], bool):
            diagnostics.append(f"'{_NUREC_BACKSTOP}.metrics_yaml_present' must be a bool")
    return diagnostics


def _any_unresolved(doc: dict[str, Any]) -> bool:
    """Return True if any ``unresolved_issues.*`` bucket is non-empty.

    Args:
        doc: Parsed workflow-summary YAML document.

    Returns:
        ``True`` when at least one unresolved issue bucket has content.
    """
    steps = doc.get("steps")
    if not isinstance(steps, dict):
        return False
    for step_block in steps.values():
        if not isinstance(step_block, dict):
            continue
        unresolved = step_block.get("unresolved_issues")
        if isinstance(unresolved, dict) and any(not is_empty_container(v) for v in unresolved.values()):
            return True
    return False


def _audit_blocking_unresolved_count(doc: dict[str, Any]) -> int:
    """Return the count of blocking unresolved facts reported for the profile.

    Reads ``blocking_unresolved_count`` rather than ``unresolved_count``: only
    blocking facts gate the workflow. Unresolved facts a converter can derive
    are recorded in ``advisory_unresolved`` and do not block.

    Read regardless of whether ``audit-dataset`` ran. The orchestrator applies
    the unresolved gate to a hand-authored ``data.profile`` as well, recording
    the result in the same field, so keying this on the step having run would
    leave the hand-authored profile ungated.

    Args:
        doc: Parsed workflow-summary YAML document.

    Returns:
        Positive blocking count, or ``0`` when absent or invalid. An absent
        field is not a clean gate: ``workflow_status: success`` separately
        requires the count to have been recorded.
    """
    blocking_count = get_path(doc, f"{_AUDIT_SUMMARY}.blocking_unresolved_count")
    if is_int(blocking_count) and blocking_count > 0:
        return blocking_count
    return 0


def _missing_metric_head_diagnostics(nurec_summary: dict[str, Any]) -> list[str]:
    """Diagnose quality-metric heads a successful run failed to report.

    Args:
        nurec_summary: The ``steps.run-nurec.summary`` mapping.

    Returns:
        Diagnostics for a malformed metrics block or for any head in
        ``_REQUIRED_METRIC_HEADS`` that is absent or non-numeric.
    """
    metrics = nurec_summary.get("metrics")
    if metrics is None:
        return [
            f"workflow_status='success' but {_NUREC_SUMMARY}.metrics is missing; "
            f"success requires a reported value for each of {', '.join(_REQUIRED_METRIC_HEADS)}"
        ]
    if not isinstance(metrics, dict):
        return [f"{_NUREC_SUMMARY}.metrics must be a mapping, got {type(metrics).__name__}"]
    diagnostics: list[str] = []
    for head in _REQUIRED_METRIC_HEADS:
        if head not in metrics:
            diagnostics.append(
                f"workflow_status='success' but {_NUREC_SUMMARY}.metrics.{head} is missing; "
                f"read it from the aggregated_metrics 'test/{head}' key in metrics.yaml"
            )
        elif not is_finite_number(metrics[head]):
            diagnostics.append(
                f"workflow_status='success' but {_NUREC_SUMMARY}.metrics.{head}={metrics[head]!r}; "
                f"success requires a finite numeric value (the head was configured and reported nothing)"
            )
    return diagnostics


def _has_unresolved_converter_issues(doc: dict[str, Any]) -> bool:
    """Return True when validate-ncore-data recorded converter issues.

    Args:
        doc: Parsed workflow-summary YAML document.

    Returns:
        Whether the converter unresolved-issues bucket contains content.
    """
    converter_issues = get_path(doc, _CONVERTER_ISSUES)
    return not is_empty_container(converter_issues)


def _validate_invariants(doc: dict[str, Any], parsed_result: dict[str, Any] | None) -> list[str]:
    """Cross-field invariants on the Workflow Summary YAML.

    Args:
        doc: Parsed workflow-summary YAML document.
        parsed_result: Parsed final ``validate_result.yaml`` sidecar from
            ``validate.py``, or ``None`` when no result file was available.

    Returns:
        Diagnostics for inconsistent status, validation, reconstruction, or
        unresolved-issue state.
    """
    diagnostics: list[str] = []
    workflow_status = doc.get("workflow_status")
    vs = as_dict(get_path(doc, _VALIDATE_SUMMARY))
    ns = as_dict(get_path(doc, _NUREC_SUMMARY))
    backstop = as_dict(get_path(doc, _NUREC_BACKSTOP))
    iterations = vs.get("iterations")
    validation_passed = vs.get("validation_passed")
    audit_blocking_count = _audit_blocking_unresolved_count(doc)

    # Auto-audit ordering: a profile-less input must run audit-dataset before
    # scope checks or any later step. data_profile_present is recorded as seen
    # at entry (before auto-audit), so false with no audit run means the
    # workflow skipped the mandatory audit pre-step.
    if doc.get("data_profile_present") is False and not _step_has_runs(doc, "audit-dataset"):
        diagnostics.append(
            "data_profile_present=false requires audit-dataset to have run "
            "(non-empty steps.audit-dataset.runs); a profile-less input must "
            "audit before scope checks or any later step"
        )

    # `steps_completed` and steps with non-empty `runs:` must agree
    # bidirectionally: a step in `steps_completed` MUST have at least one
    # `runs[*]` entry (the orchestrator claimed it ran -- timing must back
    # that claim), and a step with at least one `runs[*]` entry MUST be
    # listed in `steps_completed` (it produced attempts -- claim it
    # honestly). The shape check in `_validate_shape` already rejects
    # unknown step names, so the iteration here is over validated names.
    steps_completed = doc.get("steps_completed")
    steps = doc.get("steps")
    if isinstance(steps_completed, list) and isinstance(steps, dict):
        steps_completed_set = {name for name in steps_completed if name in _STEP_RUN_SECTIONS}
        for name in _STEP_RUN_SECTIONS:
            step_block = steps.get(name)
            if not isinstance(step_block, dict):
                if step_block is None and name in steps_completed_set:
                    diagnostics.append(
                        f"steps_completed lists {name!r} but "
                        f"steps.{name} is missing; every completed step "
                        "must emit a step block with at least one runs[*] entry"
                    )
                continue
            runs = step_block.get("runs")
            if not isinstance(runs, list):
                runs = []
            has_runs = len(runs) > 0
            if name in steps_completed_set and not has_runs:
                diagnostics.append(
                    f"steps_completed lists {name!r} but "
                    f"steps.{name}.runs is empty; every completed step "
                    "must emit at least one runs[*] entry "
                    "(omit the step from steps_completed if it did not run)"
                )
            elif has_runs and name not in steps_completed_set:
                diagnostics.append(
                    f"steps.{name}.runs has {len(runs)} entries but "
                    f"{name!r} is not listed in steps_completed; "
                    "every step that produced runs[*] must appear in "
                    "steps_completed"
                )

    # iterations == 0 ∧ !validation_passed -> HARD ERROR.
    if is_int(iterations) and iterations == 0 and validation_passed is False:
        diagnostics.append(
            f"{_VALIDATE_SUMMARY}.iterations=0 with "
            "validation_passed=false; "
            "the inner fix loop must run at least one validate -> "
            "build(fix) -> validate cycle when validate.py reports a "
            "failure (orchestrator skipped the 'Handle validation "
            "diagnosis' entry rule)"
        )

    if workflow_status == "success":
        # validation_passed must be true.
        if validation_passed is not True:
            diagnostics.append(
                f"workflow_status='success' but "
                f"{_VALIDATE_SUMMARY}.validation_passed="
                f"{validation_passed!r}; success requires validation "
                f"to have passed"
            )
        # reconstruction must have succeeded.
        recon = ns.get("reconstruction_status")
        if recon != "success":
            diagnostics.append(
                f"workflow_status='success' but "
                f"{_NUREC_SUMMARY}.reconstruction_status={recon!r}; "
                f"success requires reconstruction_status='success'"
            )
        # metrics.yaml backstop must have fired and confirmed presence.
        metrics_present = backstop.get("metrics_yaml_present")
        if metrics_present is not True:
            diagnostics.append(
                f"workflow_status='success' but "
                f"{_NUREC_BACKSTOP}.metrics_yaml_present="
                f"{metrics_present!r}; the val-artifact existence "
                f"backstop must confirm metrics.yaml was produced "
                f"before claiming success"
            )
        diagnostics.extend(_missing_metric_head_diagnostics(ns))
        # every unresolved bucket must be empty.
        if _any_unresolved(doc):
            diagnostics.append(
                "workflow_status='success' but a step has non-empty "
                "unresolved_issues; clear all unresolved issues or "
                "downgrade to partial_success"
            )
        # The count is what shows the unresolved gate was applied. A successful
        # run either audited or carried a data.profile, and both record the
        # count, so reading an absent field as zero would let silence clear the
        # gate.
        if get_path(doc, f"{_AUDIT_SUMMARY}.blocking_unresolved_count") is None:
            diagnostics.append(
                "workflow_status='success' but "
                f"{_AUDIT_SUMMARY}.blocking_unresolved_count is missing; record the count the "
                "unresolved gate produced, for a hand-authored data.profile as well as a "
                "generated one"
            )
        if audit_blocking_count > 0:
            diagnostics.append(
                "workflow_status='success' but "
                f"{_AUDIT_SUMMARY}.blocking_unresolved_count={audit_blocking_count}; "
                "resolve audit critical gaps or downgrade to partial_success"
            )

    # Recorded run-nurec runs, not reconstruction_status, decide whether NuRec
    # ran, so a reported status cannot switch off the metrics-absence backstop.
    if (
        backstop.get("metrics_yaml_present") is False
        and _step_has_runs(doc, "run-nurec")
        and workflow_status != "failure"
    ):
        diagnostics.append(
            f"{_NUREC_BACKSTOP}.metrics_yaml_present=false "
            "requires workflow_status='failure' (NuRec produced no validation metrics)"
        )

    if workflow_status == "partial_success":
        # Per the SKILL exit table, every partial_success row -- including an
        # iteration-cap hit -- must record its remaining issues in some
        # step-local unresolved_issues.* bucket (or the audit unresolved
        # count).
        if not (_any_unresolved(doc) or audit_blocking_count > 0):
            diagnostics.append(
                "workflow_status='partial_success' but no step-local "
                "unresolved_issues are recorded; record the remaining issues "
                "in some step-local unresolved_issues.* bucket (an iteration "
                "cap hit must record its remaining failures there too), "
                "otherwise re-classify as success or failure"
            )
        max_inner_iterations = get_path(doc, "execution.max_inner_iterations")
        if (
            validation_passed is False
            and _has_unresolved_converter_issues(doc)
            and is_int(iterations)
            and is_int(max_inner_iterations)
            and iterations < max_inner_iterations
        ):
            diagnostics.append(
                "workflow_status='partial_success' with "
                f"{_VALIDATE_SUMMARY}.validation_passed=false and unresolved "
                f"converter issues requires {_VALIDATE_SUMMARY}.iterations "
                f">= execution.max_inner_iterations; got iterations={iterations} "
                f"and execution.max_inner_iterations={max_inner_iterations}"
            )

    # Cross-check the final validate-ncore-data result sidecar against the
    # summary. The exit-code branch still runs when result parsing failed.
    run = as_dict(vs.get("validate_run"))
    exit_code = run.get("exit_code")
    had_fail, schema_invalid = evaluate_validate_run(parsed_result, exit_code)
    if had_fail and validation_passed is True:
        diagnostics.append(
            f"{_VALIDATE_SUMMARY}.validate_run reports failure "
            "(non-zero exit code, passed=false, or failed_checks in "
            "validate_result.yaml) but "
            f"{_VALIDATE_SUMMARY}.validation_passed=true; the upstream "
            "validate-ncore-data validator would normally catch this -- "
            "if it did not, an iteration was skipped"
        )
    if schema_invalid:
        diagnostics.append(
            f"{_VALIDATE_RESULT} is not a valid validate_result.yaml "
            "(wrong schema_version, program, or missing typed fields "
            "passed/exit_code/failed_checks/warnings); either validate.py "
            "was not run on the final iteration or the wrong file was captured"
        )

    exit_code_diag = validate_result_exit_code_diagnostic(f"{_VALIDATE_SUMMARY}.validate_run", parsed_result, exit_code)
    if exit_code_diag is not None:
        diagnostics.append(exit_code_diag)

    internal_diag = validate_result_internal_consistency_diagnostic(parsed_result)
    if internal_diag is not None:
        diagnostics.append(internal_diag)

    return diagnostics


def validate_workflow_summary(summary_path: str | Path) -> None:
    """Validate a Workflow Summary YAML file.

    Args:
        summary_path: Path to the workflow-summary YAML file.

    Returns:
        None.

    Raises:
        ValidationError: If the YAML is malformed or violates workflow
            invariants.
    """
    doc = load_yaml(summary_path)
    if not isinstance(doc, dict):
        raise ValidationError(
            [f"Top-level YAML document in {summary_path} must be a mapping, got {type(doc).__name__}"]
        )

    diagnostics: list[str] = []
    diagnostics.extend(_validate_shape(doc))
    diagnostics.extend(validate_converter_issue_entries(doc, _CONVERTER_ISSUES))

    parsed_result = parse_absolute_validate_result(get_path(doc, _VALIDATE_RESULT), diagnostics)

    diagnostics.extend(_validate_invariants(doc, parsed_result))

    if diagnostics:
        raise ValidationError(diagnostics)


_USAGE = (
    'Usage: cd "$NUREC_ONBOARD_DIR" && bazel run '
    "//skills/nurec-onboard/validators:nurec_onboard_validator -- "
    "<workflow_summary.yaml>\n"
    "Asserts cross-field invariants on the nurec-onboard Workflow\n"
    "Summary YAML, plus a backstop anti-lying invariant that\n"
    "cross-checks the validate-ncore-data summary against the typed\n"
    "validate_result.yaml sidecar referenced by its validate_run.result_path.\n"
)


main = step_validator_main(
    progname="nurec_onboard_validator",
    usage=_USAGE,
    validate_fn=validate_workflow_summary,
)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
