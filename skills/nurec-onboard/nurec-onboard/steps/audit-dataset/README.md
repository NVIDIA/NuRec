<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# audit-dataset

`audit-dataset` is a standalone, operator-reviewed step that explores a raw sensor dataset and writes a workflow-ready input YAML.

Run it before `nurec-onboard` when a dataset has not yet been described for NCore and NuRec:

```bash
export NUREC_WORKDIR=/path/to/workdir
./run.sh /path/to/audit.yaml --agent <claude|codex> --step audit-dataset
./run.sh \
  "$NUREC_WORKDIR/audit-my_dataset/my_dataset_input.yaml" --agent <claude|codex>
```

The audit does not scaffold a converter, run conversion, or launch NuRec. It produces facts and gaps for review.

## Audit Input

```yaml
data:
  name: my_dataset
  path: /path/to/raw/sequence
  docs:
    - /path/to/dataset_docs.md
    - https://example.com/dataset-paper
  hints: >
    Optional operator-known facts to verify, such as calibration file
    locations, sensor inventory, or timestamp meanings.

audit_output_dir: /path/to/output # optional; defaults to $NUREC_WORKDIR/audit-<data.name>
```

## Outputs

The step writes:

- `<audit_output_dir>/<data.name>_input.yaml`: generated input for `nurec-onboard`.
- `<audit_output_dir>/<data.name>_data_quality_report.md`: confirmed facts, warnings, and unresolved gaps.
- `<audit_output_dir>/audit_profile.json`: machine-readable audit findings that can inform the later converter-local `dataset_probe.py`.

Unknown required facts are written as `UNRESOLVED:` descriptions in the generated YAML and repeated by YAML path in the report's Critical gaps section. Resolve those before running the full workflow unless you explicitly want `build-ncore-converter` to rediscover or diagnose them later.

## Relationship To `dataset_probe.py`

The audit is upstream of converter creation. It may emit `audit_profile.json`, but it must not write converter workdir files or depend on the converter template being materialized.

`dataset_probe.py` remains owned by `build-ncore-converter`: once the converter workdir exists, the agent fills that probe with dataset-specific code to produce precise measurements for converter implementation and critical-parameter justifications.
