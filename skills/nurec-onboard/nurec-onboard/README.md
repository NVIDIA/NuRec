<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec Onboard

NuRec Onboard workflow: build an NCore converter, validate its output, and run NuRec reconstruction -- all autonomously.

## Quick start

Create an input YAML (see the root [README](../README.md#input-format)) and run:

```bash
./run.sh my_input.yaml --agent <claude|codex>
```

`--agent` is required and takes `claude` or `codex`.

For a new dataset, prefer running `audit-dataset` first:

```bash
./run.sh /path/to/audit.yaml --agent <claude|codex> --step audit-dataset
./run.sh \
  "$NUREC_WORKDIR/audit-my_dataset/my_dataset_input.yaml" --agent <claude|codex>
```

Review and resolve any `UNRESOLVED:` entries in the generated YAML before
starting the full workflow.

## What it does

The workflow has four steps. The first, **audit-dataset**, is conditional (see
below); the other three always run, in sequence with automatic feedback loops:

1. **build-ncore-converter** (scaffold mode) -- explore raw data, implement converter
2. **validate-ncore-data** -- validate output, run eval metrics
3. **run-nurec** -- generate aux data, run reconstruction

If validation or reconstruction finds issues traceable to the converter, the workflow
automatically loops back to `build-ncore-converter` (fix mode) to apply fixes.

The workflow runs `audit-dataset` inline whenever `data.profile` is absent,
then reloads the generated input before converter work starts. Standalone audit
remains the recommended path when unresolved dataset facts need operator or
data-owner review.

## Input

Same as `build-ncore-converter`:

```yaml
ncore:
  converter_dir: /path/to/ncore-myformat
data:
  path: /data/raw/my_dataset
  description: >
    Autonomous driving dataset with surround cameras, LiDAR, and ego poses.
  profile:
    cameras: [...]
    lidars: [...]
    pose_sources: [...]
    timestamp_semantics: { ... }
  docs: /data/raw/my_dataset/README.md # optional

audit:
  enabled: true # optional force-on re-audit override
  fail_on_unresolved: true
```

## Prerequisites

- Bazel (for building/running the converter)
- Docker with NVIDIA Container Toolkit (`nvidia-docker`) and GPU

## Iteration limits

Configurable via the input YAML (see root [README](../README.md#input-format)); defaults live in [`../defaults.yaml`](../defaults.yaml):

- Inner loop (validate -> fix -> validate): `execution.max_inner_iterations`
- Outer loop (nurec -> fix -> validate -> nurec): `execution.max_outer_iterations`
- Config-fix loop (run-nurec config retries per outer iteration): `execution.max_config_iterations`

## Reference files

- [SKILL.md](SKILL.md) -- full workflow orchestration specification
- Steps: `steps/audit-dataset/`, `steps/build-ncore-converter/`,
  `steps/validate-ncore-data/`, `steps/run-nurec/`
