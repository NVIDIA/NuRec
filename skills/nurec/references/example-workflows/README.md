<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

-->

# Source-based examples

Run from the released checkout after [setup](../install.md).

- `bash/`: small single-stage wrappers. Each sources the shared `nurec_dev.sh`
  helper; export `NUREC_SRC`, `NUREC_IMAGE`, and the inputs documented in the script.
- `bash/nurec_workflow_pai.md`: routes the full PAI flow to the shipped benchmark.

The standalone Hydra runner examples are replaced by the runner shipped at
`//workflows/av_benchmark:run`. Configuration selection remains in
[configuration.md](../configuration.md). No additional workflow repository is needed.
