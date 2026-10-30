<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `.codex/config.toml` — annotated reference

This project configuration defines the static half of the Codex autorun
boundary. `run.sh` supplies paths and input-derived domains at launch so the
tracked file does not contain machine-specific values.

Codex loads project `.codex` configuration only after the checkout is trusted.
The launcher checks that trust is recorded before starting an agent.

## Approval and permission profile

`approval_policy = "never"` makes denied actions fail immediately instead of
waiting for an approval that an unattended workflow cannot provide.

`default_permissions = "nurec-onboard"` selects a custom permission profile.
Custom profiles require Codex 0.138.0 or newer and do not compose with the
legacy `sandbox_mode` and `sandbox_workspace_write` settings. The launcher
rejects those legacy settings in active machine or user configuration.

## Model and reasoning effort

`model_reasoning_effort = "xhigh"` raises deliberation above the `"medium"`
default that the current model catalog declares. Each step reads a long
specification, infers dataset conventions from raw sensor data, and only then
writes converter code, so the default level under-reasons relative to the work.

`"ultra"` is available on the current models but is not a valid choice for this
workflow: it delegates work to sub-agents automatically, and the workflow
requires converter implementation, validation, and diagnosis to run inline (see
[`AGENTS.md`](../AGENTS.md)).

The launcher passes no `--model` for Codex, so the model itself is whichever one
the CLI defaults to. Select one per run with `run.sh --model <slug>`; the slugs
available to the signed-in account are listed by Codex's own model catalog.

## Filesystem

The static profile grants Codex's minimal runtime paths read access and nothing
else. A restricted profile leaves every ungranted path unreachable, so the
default is closed without an explicit deny entry — and the profile deliberately
holds none, because Codex can only enforce a denied read from inside the sandbox
and therefore refuses to run _any_ command outside it while one is configured.
That would include the Docker commands the workflow depends on. On each launch,
`run.sh` adds:

- read access to the checkout, raw dataset, input YAML, local `data.docs`
  files, Codex's installed runtime and temporary helper directory, and the
  Bazelisk binary cache;
- write access to `/tmp`, the Bazel and uv caches, and `NUREC_WORKDIR`.

Every path is canonicalized before it becomes a permission override. The
checkout and raw dataset remain read-only even when they are beneath another
readable root.

## Network

The static domain list contains the package, source, container-registry, and
build hosts used by the workflow. `run.sh` adds the hostname of every
URL-valued `data.docs` entry, public or not. Destinations named in neither
list remain blocked.

`features.network_proxy = true` starts the proxy that enforces those domain
rules. The setting is load-bearing in both directions: without an active proxy
the rules are inert, and sandboxed commands cannot resolve hostnames at all,
because the proxy is what resolves them. A sandboxed command has no resolver of
its own — where `/etc/resolv.conf` points into `/run`, as it does under
systemd-resolved, the filesystem profile never grants the target. The launcher
fetches one allowlisted URL from inside the sandbox during pre-flight so a dead
network fails before a workflow starts rather than surfacing as a
name-resolution error inside a later build.

Any higher-precedence configuration layer that turns the proxy back off fails
that probe rather than the workflow.

Native web search is disabled so documentation retrieval uses the same
sandboxed network policy as other commands.

## Docker

For a manifest with `requires_docker: true`, `run.sh` allows the Docker daemon
socket for that invocation. Other manifests do not receive socket access. On
Linux that entry grants nothing on its own, because the sandbox namespace omits
`/run` and the socket path does not exist inside it.

What actually reaches the daemon is the `docker` rule in
[`rules/nurec-onboard.rules`](rules/nurec-onboard.rules). Its `allow` decision
makes Codex run a matching command on the host rather than in the sandbox, and it
needs no approval, so it holds under `approval_policy = "never"`. The rule only
takes effect for a single Docker command built from literal words: a chained,
piped, redirected, or `$(...)`-bearing command loses the host route and fails
with an ordinary socket connection error. `run.sh` asserts both the rule and the
absence of deny entries during pre-flight.

Docker is therefore an explicit sandbox escape, and a broader one than the socket
grant implies: the command runs on the host with full filesystem access and
unfiltered network, and a container can additionally use its own network and
write through host mounts. The `docker` rule, the workflow's Docker templates,
and the documented host mount paths are all part of the trust boundary. This
matches the Claude configuration, which exempts `docker *` from its sandbox for
the same reason (see
[`.claude/settings.local.README.md`](../.claude/settings.local.README.md)).

Any other `allow` rule receives the same host execution, including rules from a
user-level `$CODEX_HOME/rules` file that NuRec Onboard does not control.
