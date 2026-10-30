<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Custom wheel change records (copies)

Apache-2.0 section 4(b) change records for the `+nre` wheels built from
NVIDIA's wheel build, kept here so the diffs are inspectable from the public `deps/`
tree and linkable from `THIRD_PARTY_NOTICES.md`.

The patches and `NUREC_CHANGES.txt` files in this directory are the change records for the published
wheels. The `.patch` files are the single source of truth: NVIDIA's wheel build applies them from
here. The `NUREC_CHANGES.txt` files are kept in sync with the copies the build packs into each wheel. Each
directory also ships, with the wheel's upstream sources, under `/licenses/sources` in the
`nurec_dev_deps` image.

| Wheel                               | Directory             |
| ----------------------------------- | --------------------- |
| `cosmos-predict2 1.0.9+nre2`        | `cosmos_predict2/`    |
| `decord 0.6.0+nre4`                 | `decord/`             |
| `opencv-python 4.10.0.84+nre5`      | `opencv-python/`      |
| `transformer_engine 1.11.0+...nre1` | `transformer_engine/` |
| `viser 0.2.23+nre5` (built by hand) | `viser/`              |
| `causal-conv1d 1.5.2+...`           | `causal-conv1d/`      |
| `imageio-ffmpeg 0.6.0+nre6`         | `imageio-ffmpeg/`     |
| `lietorch 0.3+...nre1`              | `lietorch/`           |
| `mamba-ssm 2.3.1+...`               | `mamba-ssm/`          |
| `mmcv 2.2.0+...`                    | `mmcv/`               |
| `point-cloud-utils 0.34.0+nre1`     | `point-cloud-utils/`  |
| `poisson-recon 13.74.1`             | `poisson-recon/`      |
| `sam2 1.1.0+...`                    | `sam2/`               |
| `torch 2.11.0+cu128.nre1`           | `torch/`              |
| `yourdfpy 0.0.58+nre1`              | `yourdfpy/`           |
