<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting source workflows

| Symptom                                  | Check / remedy                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Dependency image unavailable             | Verify the GitHub release's matching image tag/digest; report the pull error if unpublished.            |
| Build looks idle                         | First-time Bazel/CUDA compilation can be slow. Preserve logs and poll the existing process.             |
| Files exist on host but cannot be opened | Use absolute paths under `NUREC_SRC` or add a same-path mount to `NUREC_DOCKER_ARGS`.                   |
| Permission error in HOME or ccache       | Source the setup helper as your normal user; use a writable persistent output base.                     |
| Gated model or dataset denied            | Verify browser terms acceptance and the read token for the named Hugging Face repository.               |
| Unexpected dataset/sensor errors         | Check actual manifest IDs and choose a matching config; PAI and Waymo rigs differ.                      |
| Auxiliary tool missing                   | Inspect `//apps:nurec_tools --help` and the build's tools selection.                                    |
| Out of GPU memory                        | Reduce the supported workload/resolution/batch setting and record the change; preserve failed-run logs. |
| Artifact has no controllable actors      | Confirm training/export included actor tracks; enabling editing alone cannot add them.                  |
| Custom trajectory fails                  | Regenerate for this exact USDZ and rig; parse the JSON and use non-training-view rendering.             |
| gRPC client/server mismatch              | Generate client stubs from the same release as the server.                                              |

Query the source-built command's `--help` before adapting a recipe. Never resolve a
missing public command by substituting an unpublished application image or private
configuration.
