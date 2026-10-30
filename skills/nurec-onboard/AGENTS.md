<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

Project-level conventions for autonomous workflow execution.

## Docker

- Always assign container names.
- Preserve the resource, GPU, environment, user, and container-side mount flags
  in the workflow's Docker templates. Substitute only documented placeholders.
- Issue each Docker command as a single `docker` invocation built from literal
  words. Docker runs on the host rather than in the command sandbox, and that
  applies only to a command in this shape. A chained, piped, redirected,
  wildcard-bearing, environment-assignment-prefixed, `$(...)`-bearing, or
  `$VAR`-bearing Docker command loses host execution and fails as a socket
  connection error rather than as a policy refusal, so resolve placeholders,
  uid, gid, and paths in separate commands and write the results in literally.
  `-e VAR` with no `=value` passes the value through from the environment.
- Run long agent-managed containers detached and inspect their state with
  separate, short-lived `docker` commands. Commands presented for an operator
  to run must stay in the foreground.
- Issue Docker commands separately from host filesystem setup, output reads,
  and cleanup. Docker access is outside the command sandbox's filesystem and
  network boundaries.

## NCore

Never use system `python` or `python3` to import NCore. Run NCore operations
through the Bazel targets specified by the workflow.

## Shell safety

- Do not use `sudo` for destructive commands.
- Do not clear directories with variable-plus-glob forms such as
  `rm -rf "$dir"/*`. Remove a validated, fully qualified directory and recreate
  it, or use `find "$dir" -mindepth 1 -delete`.
- Treat ownership recovery requiring `sudo chown` as an operator action.

## Workflow execution

- Execute workflows autonomously without requesting input.
- Follow the selected `SKILL.md` as the operating contract.
- Execute workflow implementation inline. Do not delegate converter creation,
  validation, reconstruction, or diagnosis to subagents.
