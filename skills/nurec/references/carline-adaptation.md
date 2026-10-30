<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Carline Adaptation

Carline adaptation is a local NuRec rendering utility for an existing NuRec
USDZ and an already-augmented target vehicle rig JSON. It composes existing
NuRec and Harmonizer workflows; it does not introduce another runtime or
command wrapper.

## Workflow

1. Resolve the source USDZ, augmented target rig JSON, output directory, and
   target-rig camera IDs.
2. Follow [Option 2 in `local-render.md`](local-render.md#option-2-custom-rig-trajectory-full-flow):
   - Pass the augmented rig to `export-custom-rig-trajectory` with
     `--rig-json`.
   - Pass the exported JSON to `render` with `--custom-rig-trajectory` and
     render the requested `--camera-id` values.
3. Inspect and preserve the raw rendered frames.
4. For requested enhancement, follow [public Harmonizer setup](harmonizer.md)
   and the public model repository for each rendered camera. Preserve raw frames
   and do not assume a separate Harmonizer skill is installed.

## Inputs and outputs

- **Inputs:** source USDZ, augmented target rig JSON, requested target-rig
  camera IDs.
- **NuRec outputs:** custom rig trajectory JSON and raw rendered frames.
- **Harmonizer output:** enhanced frames produced from the raw render
  directories by the public Harmonizer workflow while preserving the raw frames.

Example original and augmented rig files remain under
[`rig-json/`](rig-json/). Use the user's augmented rig when one is supplied.
