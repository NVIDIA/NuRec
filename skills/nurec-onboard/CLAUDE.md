<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLAUDE.md

Project-level conventions for autonomous workflow execution.

## Docker conventions

- **Never use `-d` (detached mode)** when presenting `docker run` commands to the operator. The operator wants to see container output directly in the terminal. Run all containers in foreground (no `-d` flag).
- Always assign container names (`--name`) for easier monitoring and cleanup.
- Do **not** modify resource/runtime flags in SKILL `docker run` templates — specifically `--shm-size`, `--gpus`, `-e` env vars, `--user`, and the container-side mount points referenced elsewhere in the command. Substitute only placeholder values (host paths, uid/gid, sequence/container names, sensor IDs). Reducing `--shm-size=64g` below 64g, in particular, will crash PyTorch dataloader workers with `No space left on device`. Dropping `--user <uid>:<gid>` leaves outputs owned by `root:root` and forces an operator-side `sudo chown` recovery before normal cleanup can proceed.

## System Python and NCore

**Never use system `python3` (or `python`) to import `ncore`.** The converter repos pin `nvidia-ncore` via Bazel; system Python does not have access to this version. Any `python3 -c "import ncore ..."` or `python3 <script>` that imports ncore will silently pick up the wrong version or fail. Use `bazel run` targets inside the converter repo for all NCore operations.

```bash
# CORRECT
cd "$NUREC_ONBOARD_DIR" && bazel run //skills/nurec-onboard/ncore_validation:validate -- ...
cd "$ncore_converter_dir" && bazel run //ncore_<dataset>:convert -- ...

# WRONG — system python sees wrong/missing ncore version
python3 -c "import ncore; ..."
python3 ncore_validation/validate.py
```

## Bash tool constraints

- The Bash tool has a maximum timeout of 10 minutes (`BASH_MAX_TIMEOUT_MS` ceiling of 600000ms per the [Bash tool docs](https://code.claude.com/docs/en/tools-reference#bash)). Never run long Docker training commands in foreground Bash tasks.
- For commands expected to complete within 10 minutes, use `run_in_background: true` ([same docs](https://code.claude.com/docs/en/tools-reference#bash)).
- For longer commands the agent needs to run itself, use `docker run -d` and monitor with `docker logs`. This does **not** apply to commands presented to the operator (see Docker conventions above).
- **Avoid `!` in any Bash-tool command line; write the script to `/tmp/claude/<name>.{sh,py}` and invoke that file.** Claude Code's Bash tool unconditionally escapes `!` → `\!` at the transport layer (the `shell-quote` library's `quote()` function), regardless of quoting context, before passing the command to `bash -c`. Upstream tracking is broken: the canonical root-cause writeup [anthropics/claude-code#35701](https://github.com/anthropics/claude-code/issues/35701) was stale-bot-closed despite proposing a one-line regex fix, and the most-upvoted symptom report [#10153](https://github.com/anthropics/claude-code/issues/10153) was closed in Apr 2026 with "fixed in 2.1.75" — a claim contradicted by direct binary inspection of every cached version through 2.1.145. The only currently-open public tracker is the broad "crazy escaping" meta [#7387](https://github.com/anthropics/claude-code/issues/7387), kept alive only by the author defeating the stale bot every two weeks. The `\!` leaks into the eventual command and breaks downstream parsers. Two observed cases:

  - `python3 -c '... b"itar" != foo ...'` → Python (or `jq`, `gh`, etc.) receives the literal `\!=` token and rejects it as a syntax error; burns a round-trip.
  - `nohup foo & echo $!` → the `$!` PID expansion arrives as literal `$!`, the launch silently no-ops, and a downstream "wait for PID" hangs.

  Both have the same fix: write the command body to `/tmp/claude/<name>.sh` (or `.py` for Python) and invoke that file -- file contents arrive via the filesystem, not via the command-line escape. Quoting tricks (single quotes, `set +H`, manual `\!`) do not help; the documented inline alternative is ANSI-C hex (`$'\x21'`), but a file is cleaner. For background launches, the cleaner pattern is the runner's own `run_in_background: true` argument, which avoids needing `$!` at all.

- **Never bundle non-`docker` work into a `docker`-bearing Bash call.** Issue each `docker ...` invocation as its own Bash call. Pre-docker setup (`mkdir`, `rm`, `cp`), post-docker reads (`docker logs`, `docker inspect`), and cleanup (`docker rm`) each go in separate Bash calls. The Claude Code sandbox treats `docker` as incompatible and routes any Bash call containing `docker *` outside the sandbox (per [Sandboxing docs](https://code.claude.com/docs/en/sandboxing) -- `docker` is the documented example of an incompatible tool, typically placed in `excludedCommands` or escaped via `dangerouslyDisableSandbox`), which bypasses `sandbox.filesystem.allowWrite`; bundling pulls cleanup-shaped commands out of the sandbox and exposes them to the host filesystem.
- **Avoid `rm -rf $VAR/*` and `${VAR}/*` shell forms.** Bash glob expansion plus an empty/unset `$VAR` becomes `rm -rf /*`, which Claude Code's built-in safety layer detects on shape (not on dataflow) and prompts on -- even in `bypassPermissions` mode (per the [Permissions docs](https://code.claude.com/docs/en/permissions): "Removals targeting the filesystem root or home directory, such as `rm -rf /` and `rm -rf ~`, still prompt as a circuit breaker against model error"). When clearing a directory's contents use `rm -rf "$VAR" && mkdir -p "$VAR"` (no glob; target is a fully-qualified path string) or `find "$VAR" -mindepth 1 -delete` (no glob; walks contents). Both also abort cleanly if `$VAR` is unset under `set -u`.
- **Never invoke `sudo rm` (or any `sudo`-elevated destructive command) autonomously.** SKILL prose that mentions root-owned artifacts requiring elevation describes an operator task, not an agent task -- `sudo` bypasses the shape-detection heuristics that catch `rm -rf $VAR/*` and similar footguns. When the agent encounters root-owned outputs (typically because a `docker run` fallback dropped `--user`), the recovery path is `sudo chown -R $(id -u):$(id -g) <path>` (run by the operator), after which a normal user-mode `rm -rf <path>` from the agent runs under the standard sandbox/safety layer. Pair the `chown` with the path narrowing rule above ("`$VAR`-free", fully-qualified path) regardless of who runs it.

## Workflow execution

- Workflows are fully autonomous -- do not prompt the user for input.
- When a workflow SKILL.md says to run a command, adapt the execution strategy (foreground vs. detached) based on expected runtime.

## Sub-agent policy

- When executing the `nurec-onboard` workflow (or any of its steps: `build-ncore-converter`, `validate-ncore-data`, `run-nurec`), do **not** delegate implementation work to [subagents](https://code.claude.com/docs/en/sub-agents) via the `Agent` tool (Claude Code's [Task tool](https://code.claude.com/docs/en/tools-reference#task)). Execute every step inline in the main session: probe the raw data, write the converter, run conversion, run validation, author the NuRec custom config, launch training, build the diagnosis. Subagent prompts re-scope the SKILL's context behind a written brief, which has caused implementation deviations (e.g. operator-scoped sensor lists leaking from "runtime output filter" into the converter's implementation domain, leaving the code unable to handle non-listed sensors on a future run). The SKILL is the contract; an inline executor can read it directly, a subagent can only read what the orchestrator paraphrases. Inline execution costs more main-context tokens but trades that against deviation risk on a multi-step workflow that can't easily be re-run.
- The built-in [`Explore` and `Plan` subagents](https://code.claude.com/docs/en/sub-agents#built-in-subagents) are exempt from this rule — they are read-only / planning helpers that don't execute SKILL steps, so they can't deviate from the SKILL contract.
