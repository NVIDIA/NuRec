<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# build-ncore-converter

Scaffold and implement an NCore V4 data converter from the bundled template at `ncore_template/`, or fix diagnosed issues in an existing converter.

## Usage

Most users should run the full workflow via `nurec-onboard` rather than this step directly.
For standalone use, create an input YAML (see the root [README](../../../README.md#input-format))
and run:

```bash
./run.sh my_input.yaml --agent <claude|codex> --step build-ncore-converter
```

## Modes

- **Scaffold mode**: First run. Explores raw sensor data, materializes the bundled template into the converter workdir, and applies the single-token `example` → `<dataset>` rename to produce the implementation.
- **Fix mode**: Re-invoked with a diagnosis from `validate-ncore-data` or `run-nurec`. Applies targeted fixes and re-runs conversion.

## Input

See `SKILL.md` for the full input schema. Minimal example:

```yaml
ncore:
  converter_dir: /path/to/ncore-myformat
data:
  path: /data/raw/my_dataset
  description: >
    3 spinning LiDARs, 6 rolling-shutter cameras, timestamps in microseconds.
```

## Reference files

- [SKILL.md](SKILL.md) -- full step specification
- [conventions.md](conventions.md) -- NCore V4 conventions quick-reference
- [critical-parameters.md](critical-parameters.md) -- `critical_parameter_justifications` schema (the critical-parameter justification step)
