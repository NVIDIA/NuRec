<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Workflow Summary schema and backstop validator

Full field schema for the structured Workflow Summary and the mandatory emit-time
validator. See [SKILL.md "Workflow Summary"](SKILL.md#5-workflow-summary) for the
contract overview, timestamp-capture rules, and the `steps_completed` / `runs[*]`
invariants. The shown fields are the contract; `validators/nurec_onboard_validator.py`
is the executable authority for field shape and cross-field invariants.

## Contents

- [Summary schema](#summary-schema)
- [Backstop validator](#backstop-validator)

## Summary schema

```yaml
workflow_status: success | partial_success | failure
steps_completed:
  [audit-dataset, build-ncore-converter, validate-ncore-data, run-nurec]

# Whether the input nurec-onboard received carried data.profile (recorded as
# seen at entry, before any auto-audit). false requires audit-dataset to have
# run, so the validator can reject a profile-less input that reached a scope
# check or any later step without auditing first.
data_profile_present: <bool>

execution:
  # Effective cap used for validate -> build(fix) -> validate cycles.
  max_inner_iterations: <int>
  # Effective cap used for run-nurec -> build(fix) -> validate -> run-nurec
  # cycles.
  max_outer_iterations: <int>
  # Effective cap used for run-nurec config-fix retries.
  max_config_iterations: <int>

steps:
  audit-dataset:
    runs:
      - attempt: <int>
        status: success | partial | error
        started_at: <ISO8601 string>
        ended_at: <ISO8601 string>
        duration_sec: <float>
    summary:
      generated_input_yaml: <string>
      data_quality_report: <string>
      audit_profile: <string>
      # Every unresolved fact, both severities.
      unresolved_count: <int>
      # The blocking subset; this is what gates workflow_status. Must be
      # <= unresolved_count. Recorded on both paths, since a hand-authored
      # data.profile gets the same gate: a summary claiming success carries
      # this count even when audit-dataset did not run, and it is the only
      # field required in that case.
      blocking_unresolved_count: <int>
      # YAML paths of the unresolved facts a converter can derive. Advisory
      # only: does not populate unresolved_issues.*, so a run carrying only
      # these can still exit workflow_status: success.
      advisory_unresolved: [<string>, ...]
    unresolved_issues:
      # YAML paths of the blocking facts.
      critical_gaps: [<string>, ...]

  build-ncore-converter:
    runs:
      # attempt is 1-based within this step.
      - attempt: <int>
        status: success | failed | error
        started_at: <ISO8601 string>
        ended_at: <ISO8601 string>
        duration_sec: <float>
    summary:
      mode: scaffold | scaffold+fix
      inferred_parameters: { ... }
      fixes_applied: [<string>]

  validate-ncore-data:
    runs:
      - attempt: <int>
        status: pass | converter_issues | input_data_issues | both | error
        started_at: <ISO8601 string>
        ended_at: <ISO8601 string>
        duration_sec: <float>
    summary:
      # Last validate.py exit code: 0 -> true, nonzero -> false.
      validation_passed: <bool>
      # Non-blocking warning-severity input-data findings from an
      # `input_data_issues` diagnosis that did NOT stop run-nurec. Advisory
      # only: does not populate unresolved_issues.*, so a run carrying only
      # these can still exit workflow_status: success.
      advisory_warnings: [<string>, ...]
      eval_metrics: { ... }
      # Inner-loop fix cycles only (validate -> build(fix) -> validate);
      # outer-loop validate invocations may appear in `runs`, not here.
      iterations: <int>
      # One entry per cycle; length must equal iterations.
      iteration_history:
        # Exact validate.py error lines for this cycle.
        - errors: [<string>, ...]
          fixes_applied: [<string>, ...]
          validation_passed_after: <bool>
      # Last validate.py run.
      validate_run:
        # Required.
        exit_code: <int>
        # Required; absolute path to validate.py's typed validate_result.yaml
        # sidecar (schema_version, program, passed, failed_checks[*].message,
        # warnings[*].message).
        result_path: <string>
        # Optional; human-debug log capture (not consumed by the validator).
        log_path: <string>
        # Optional.
        invoked_at: <ISO8601 string>
    unresolved_issues:
      converter:
        # Exact validate_result.yaml failed_checks[*].message.
        - error_message: <string>
          demotion_attempted: <bool>
          demotion_criteria_failed: [<string>, ...]
          iterations_before_giving_up: <int>
      input_data: [<string>, ...]

  run-nurec:
    runs:
      - attempt: <int>
        status: success | issues_detected | not_run | error
        started_at: <ISO8601 string>
        ended_at: <ISO8601 string>
        duration_sec: <float>
    summary:
      reconstruction_status: success | issues_detected | not_run
      # From metrics.yaml aggregated_metrics keys test/psnr, test/ssim,
      # test/lpips; success requires all three finite. A head that failed to
      # compute reports NaN, which is not a reported value.
      metrics:
        psnr: <float>
        ssim: <float>
        lpips: <float>
      subsampling:
        camera_pixel_sampler_subsample: <int>
        n_val_image_subsample: <int>
        n_train_sequential_image_subsample: <int>
        selected_camera_resolutions:
          - camera_id: <string>
            width: <int>
            height: <int>
      # Outer (build-fix) iterations.
      iterations: <int>
      # Config-fix retries on the final outer pass.
      config_iterations: <int>
      # E.g. "switched to dynamic_gaussians",
      # "override track_label_sources=null".
      config_fixes_applied: [<string>]
    val_artifact_backstop:
      # Required; <nurec_output_dir>/*/val/metrics.yaml exists.
      metrics_yaml_present: <bool>
      # Optional; whether <nurec_output_dir>/*/val/pred_*.mp4 exists.
      pred_videos_present: <bool>
      # Optional; the run-id whose val/ dir was checked.
      most_recent_run_id: <string>
      # Optional; the glob-expanded paths actually checked.
      checked_paths: [<string>, ...]
    unresolved_issues:
      ncore_data: [<string>, ...]
      nurec_config: [<string>, ...]

output_paths:
  ncore_output_dir: <string>
  sequence_meta: <string>
  nurec_output_dir: <string>
```

## Backstop validator

**Workflow Summary backstop validator (mandatory)**: before emitting the Workflow
Summary, write the YAML to a temporary file and run

```bash
cd "$NUREC_ONBOARD_DIR" && bazel run //skills/nurec-onboard/validators:nurec_onboard_validator -- <workflow_summary.yaml>
```

The validator asserts success/partial-success invariants, rejects
`validation_passed=true` when validate.py's typed `validate_result.yaml`
sidecar reports failure (non-zero exit code, `passed: false`, or non-empty
`failed_checks`), requires an absolute
`steps.validate-ncore-data.summary.validate_run.result_path` and rejects
sidecars whose `schema_version` / `program` does not match the expected
receipt, treats `validation_passed=false` with
`steps.validate-ncore-data.summary.iterations==0` as a hard error, requires
`audit-dataset` to have run whenever `data_profile_present` is `false`, and
rejects any disagreement between `steps_completed` and the set of steps
with non-empty `runs:` (in either direction).

A non-zero exit on the validator is a hard failure -- the summary must be
fixed before being returned to the operator.
