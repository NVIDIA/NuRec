<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# validate-ncore-data

Validate NCore V4 output, run eval metrics, and produce a structured diagnosis.

## Usage

Most users should run the full workflow via `nurec-onboard` rather than this step directly.
For standalone use, create an input YAML (see the root [README](../../../README.md#input-format))
and run:

```bash
./run.sh my_input.yaml --agent <claude|codex> --step validate-ncore-data
```

## What it does

1. Runs `validate.py` against the NCore V4 data (three-tier validation)
2. Runs `eval.py` to compute quality metrics
3. Assesses input data quality (upstream sensor data issues)
4. Produces a structured **diagnosis** -- it does NOT edit converter code

The diagnosis is consumed by `build-ncore-converter` (fix mode) or the `nurec-onboard` orchestrator.

## Input

```yaml
ncore:
  converter_dir: /path/to/ncore-myformat
data:
  ncore_path: /output/ncore/seq_001.json
```

## Reference files

- [SKILL.md](SKILL.md) -- full step specification
- [common-failures.md](common-failures.md) -- failure-to-fix mapping
