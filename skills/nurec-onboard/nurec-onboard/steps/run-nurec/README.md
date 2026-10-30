<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# run-nurec

Run NuRec (Neural Reconstruction Engine) on validated NCore V4 data.

## Usage

Most users should run the full workflow via `nurec-onboard` rather than this step directly.
For standalone use, create an input YAML (see the root [README](../../../README.md#input-format))
and run:

```bash
./run.sh my_input.yaml --agent <claude|codex> --step run-nurec
```

## What it does

1. Generates auxiliary data (semantic segmentation, LiDAR segmentation) via `ncore-aux-data`
2. Selects or creates an app config matching the sensor suite
3. Runs NuRec reconstruction via Docker
4. Interprets output metrics (PSNR, SSIM, LPIPS)
5. Produces a validator-checked structured diagnosis (routes NCore data issues back to `build-ncore-converter`)

## Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- GPU with 24+ GB VRAM
- NuRec dev dependencies images selected in [`../../../defaults.yaml`](../../../defaults.yaml)

## Input

```yaml
ncore:
  converter_dir: /path/to/ncore-myformat
data:
  ncore_path: /output/ncore/seq_001.json
  description: >
    3 spinning LiDARs, 6 rolling-shutter cameras.
```

## Reference files

- [SKILL.md](SKILL.md) -- full step specification
