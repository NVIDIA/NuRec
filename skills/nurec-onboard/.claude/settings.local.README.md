<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `.claude/settings.local.json` — annotated reference

Key-by-key annotation of [`settings.local.template.json`](settings.local.template.json), the committed template from which [`run.sh`](../run.sh) generates the git-ignored `settings.local.json` on every Claude Code launch. Make policy changes in the template: the generated file is replaced wholesale on each launch, including any rules Claude Code added to it during a session. The upstream definition of each `sandbox.*` key lives in [Claude Code's sandbox settings documentation](https://code.claude.com/docs/en/settings#sandbox-settings). Runtime trust boundaries are summarized in the top-level [`README.md`](../README.md#security). These files target Claude Code's built-in sandbox (Limited autorun mode).

## `permissions.allow`

Each entry gates one tool the agent may invoke. The standard Claude Code tool names (`Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, `Task`, `WebFetch`) map to the obvious tool of the same name. The less-obvious entries (`Shell`, `SubAgent`, `Monitor`) are internal Claude Code permission names that Claude Code itself prompted for in a prior session; the per-entry notes below explain what each gates.

- `Bash` — run shell commands (most heavily used; the SKILLs run `bazel`, `docker`, `python3`, `pdftotext`, etc. through it).
- `Edit` — modify files in the writable areas listed under `sandbox.filesystem.allowWrite`.
- `Glob` — file-name pattern search.
- `Grep` — content search (ripgrep).
- `Monitor` — internal Claude Code permission name for the monitoring / observability surface (e.g. progress reporting from long-running tools); the multi-step pipeline depends on it for clean status output.
- `Read` — read files from the local filesystem.
- `Shell` — internal Claude Code permission name for shell-like operations surfaced separately from `Bash` in some Claude Code versions; the SKILL files depend on it for shell commands to run without a per-call prompt under the `/sandbox` autorun policy.
- `SubAgent` — internal Claude Code permission name for spawning subagents surfaced separately from `Task` in some Claude Code versions; the pipeline orchestrator depends on it to launch helpers without per-call prompts.
- `Task` — launch subagents (`Task` tool). The pipeline SKILLs use this for the `Explore`/`Plan` helpers; subagents inherit this same `permissions.allow` set and the same sandbox.
- `WebFetch(domain:<host>)` _(one rule per `data.docs` URL hostname, written per launch)_ — restricts WebFetch to the exact hostnames declared in the input YAML's `data.docs` field. [`lib/claude_config.py`](../lib/claude_config.py) strips all prior `WebFetch*` entries and replaces them with one `WebFetch(domain:<host>)` rule per unique hostname. When `data.docs` has zero URLs the agent cannot WebFetch anywhere. **Not constrained by `sandbox.network.allowedDomains`** (sandbox network restrictions apply only to the Bash tool and its child processes; WebFetch is executed by Claude Code's own process). The actual fetch is performed locally; only the requested hostname is preflighted to `api.anthropic.com` against a safety blocklist. See the [`README.md` Security section](../README.md#security) for the full implication.
- `Write` — create files in the writable areas listed under `sandbox.filesystem.allowWrite`.

## `sandbox.allowUnsandboxedCommands: false`

Disables Claude Code's built-in `dangerouslyDisableSandbox` escape hatch. By default (`true`), when a sandboxed command fails due to sandbox restrictions, Claude Code may offer to retry it outside the sandbox; setting this to `false` closes that route entirely. The only configured sandbox escape in NuRec Onboard is `docker *` in `excludedCommands` (explicit, documented below). [`run.sh`](../run.sh) verifies this key is `false` before launching. See [Claude Code sandboxing docs](https://code.claude.com/docs/en/sandboxing) for the default behavior this overrides.

## `sandbox.autoAllowBashIfSandboxed: true`

Deliberate. The sandbox itself is the trust boundary, so prompting on every Bash call adds friction without security benefit — the file-write and network-egress restrictions still apply.

## `sandbox.enabled: true`

Turns on Claude Code's built-in sandbox (Limited autorun mode). [`run.sh`](../run.sh) reads back `settings.local.json` on every launch and refuses to start the agent if this becomes `false`. See the [`README.md` Security section](../README.md#security) for the runtime trust-boundary summary.

## `sandbox.excludedCommands: ["docker *"]`

Deliberate **escape hatch** for the GPU + host-mount Docker workload. Any command matching `docker *` runs on the host, outside the sandbox, with whatever flags the SKILL prescribes (typically `--gpus all`, host-path volume mounts via `-v`, and on some hosts `--network host`). The trust boundary for Docker actions therefore shifts to the SKILL files plus operator review of the printed command.

See the [`README.md` Security section](../README.md#security) for the full implication.

## `sandbox.failIfUnavailable: true`

Companion to `sandbox.enabled`. Without it, Claude Code's documented default behavior is to _warn and run unsandboxed_ when the sandbox cannot start (e.g. WSL1, missing kernel features, unsupported platform). With `failIfUnavailable: true`, Claude Code hard-fails instead, so the trust boundary cannot vanish silently on a misconfigured host. [`run.sh`](../run.sh) also verifies this key is `true` before launching.

## `sandbox.filesystem.allowWrite`

An empty list (`[]`) in the template; **the launcher is the source of truth**, not the template. [`lib/claude_config.py`](../lib/claude_config.py) sets this key in the generated file on every launch to the canonical resolution (symlinks followed, `~` expanded) of:

- The defaults declared in `SANDBOX_DEFAULT_WRITABLE_PATHS` near the top of [`run.sh`](../run.sh) -- edit that constant to add or change always-on writable paths for a different host setup. The shipped defaults are `/tmp` (short-lived scratch), `~/.cache/bazel` (the Bazel local repository cache, also pre-populated by the `warm_bazel_cache` step before the agent starts), and `~/.cache/uv` (used by `bazel run //deps/pip:requirements` when the converter regenerates its pip lockfile).
- `$NUREC_WORKDIR` -- the operator-chosen workdir (rejected if it resolves to `/`, `$HOME`, `/root`, `/etc`, `/usr`, `/var` without `NUREC_WORKDIR_ALLOW_ANY=1`).

Canonical-path resolution matters because the Claude Code sandbox enforces writability against the resolved path, not the as-supplied path -- a symlinked write target appears read-only inside the sandbox unless the symlink target itself is in the allow list. Without resolution, a host whose `~/.cache/bazel` (or any other listed path) is a symlink would silently get a read-only cache and force every Bazel invocation to re-fetch upstream artifacts on every run.

Entries added to this key in the template are replaced on every launch.

## `sandbox.network.allowedDomains`

The shipped allowlist below is the minimum set required for the workflow to operate.

- `bcr.bazel.build` — Bazel Central Registry. Bzlmod (`MODULE.bazel`) module-extension evaluation fetches module metadata and source archives from here when the inherited `MODULE.bazel.lock` is missing or invalidated. The `warm_bazel_cache` step in [`run.sh`](../run.sh) populates this in the host cache before the agent starts; the in-sandbox allowlist entry is the fallback for the case where Bazel re-evaluates a module inside the sandbox.
- `download.pytorch.org` — PyTorch's CPU package index, where the converter template's lock file gets its CPU-only `torch` build.
- `download-r2.pytorch.org` — PyTorch's wheel storage, where `download.pytorch.org`'s index links its `torch` wheels.
- `files.pythonhosted.org` — pip wheel + sdist downloads. Bare host, not `*.pythonhosted.org`: pip only fetches from this exact host.
- `github.com` — git pulls and Bazel rules sourced from GitHub.
- `objects.githubusercontent.com` — GitHub Releases CDN, one of the redirect targets for `http_archive` rules sourced from `github.com/<org>/<repo>/releases/download/...`.
- `pypi.org` — pip package index. Bare host, not `*.pypi.org`: pip never uses pypi.org subdomains.
- `release-assets.githubusercontent.com` — GitHub Releases CDN (current generation, supplements/replaces `objects.githubusercontent.com` for newer release assets); same role: redirect target for `http_archive` rules pointing at `github.com/<org>/<repo>/releases/download/...`.
- `releases.bazel.build` — Bazel binary distribution. Where `bazel` is `bazelisk` (a version-pinning wrapper), it resolves the version in `.bazelversion` from its own cache, which the sandbox can read because reads are unrestricted. This entry covers the case where the pinned version is not cached yet and `bazelisk` has to download it.
