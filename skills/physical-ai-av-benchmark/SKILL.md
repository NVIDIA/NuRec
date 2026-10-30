---
name: physical-ai-av-benchmark
description: >-
  Use when the user wants to run the NuRec AV benchmark workflow on one
  NVIDIA PhysicalAI-Autonomous-Vehicles clip on a new Brev GPU instance,
  from a source checkout and the pinned dev dependencies image. Thin
  pointer to the existing docs, skills and workflow. Do NOT use for other
  NuRec work such as training, rendering or export (use `nurec`), custom
  datasets (use `nurec-onboard`), or dataset browsing and download-only
  requests (use `physical-ai-datasets`).
version: "0.1.0"
tools:
  - Shell
  - Read
license: Apache-2.0
compatibility: >-
  Brev CLI, a Brev GPU instance (Linux x86_64, one GPU), Docker and the
  NVIDIA Container Toolkit on the instance host, and a Hugging Face
  account that has accepted the PhysicalAI-Autonomous-Vehicles license.
  Requirements are defined by the hardware and software pages of the
  published NuRec documentation.
dependencies:
  - bash
  - brev
  - docker
  - tmux
metadata:
  author: NVIDIA NRS
  tags:
    - nurec
    - autonomous-vehicles
    - benchmark
    - brev
    - physical-ai
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PhysicalAI-AV Benchmark

Run `workflows/av_benchmark` on a fresh Brev instance. This page only routes you: read each document it names instead of expecting it to be restated here. The agent drives Brev and checks results; the user owns the interactive logins and the shell that holds each secret, and the agent never handles a secret.

**1. Prepare.** The user completes the interactive `brev login` in their own terminal. Install the Brev CLI only from Brev's own install page (https://docs.nvidia.com/brev/cli/getting-started); never reuse an unknown binary found on the machine. Then follow Brev's pages (https://docs.nvidia.com/brev/latest/cli/instance-creation.md, https://docs.nvidia.com/brev/latest/cli/connectivity.md; with a Brev CLI skill, use it, see https://docs.nvidia.com/brev/latest/guides/ai-agents/cli-with-agents.md) and confirm the active Brev organization before creating anything. Read GPU count, architecture and whether a type can stop from `brev search --wide --json` (the architecture column only shows with `--wide`).

The NuRec hardware page (https://docs.nvidia.com/nurec/basics/hardware.html) gives the GPU requirement and recommendation: use one GPU on the platform that page states, and offer the recommended class first. A smaller card that meets the stated minimum is the user's choice, after a plain warning that an out-of-memory run is not a valid benchmark. No doc gives a disk figure, so recommend a generous disk, check free space on the filesystems the run will actually use (root, the container image store, any local NVMe) before setup, and warn plainly when space looks tight; if it is too small, look for a local NVMe before recreating, because Brev's `--type` creation may not apply a disk size.

**2. Confirm and create.** Preview with `--dry-run --type <selected-type>`. Then ask the user for one confirmation before any spend, showing the selected `--type`, its hourly price, the clip (the spec's default or one they name) and the run mode: the benchmark defaults, or a change the user names as specific variables and values, which you record as `--set` overrides and show in this confirmation (to help them choose, use the sources named in stage 6). In the same confirmation, say plainly that charges accrue while the instance runs, that a stopped instance (where the type allows it) may still incur storage charges, that only deleting it ends all charges, and that this session cannot clean it up once it has ended; the confirmation covers this. Create with the selected `--type` only after the user confirms, then confirm the instance in `brev ls` (create can exit 0 with zero instances). If create fails, stop and ask: the confirmed type, price, clip and run mode are one decision.

**3. Host and prerequisites.** Find out which machine your shell reaches. No container marker (for example `/.dockerenv`) and PID 1 being systemd are indicators, not proof. Also run a check the user can see, for example that the shell reaches the machine whose GPU, driver and container runtime the later steps use. If the indicators and the check disagree, or you are unsure, ask the user before changing anything. Pass `--host` to `brev shell`, `exec` and `copy` only when the default shell is a container. On the host, pass the prerequisite checks under "Required software" on the NuRec software page (https://docs.nvidia.com/nurec/basics/software.html) and on the hardware page. Show any system change (package installs, container-runtime or CDI refresh) exactly as you will run it, and get the user's approval for each separately.

Read both the software page and the hardware page for the NVIDIA driver. If they differ, tell the user the difference, and follow the stated recommendation for the GPU. If the host is below what a doc requires, say so and ask.

**4. Source and dev image.** Use only the clone command and the pinned dev image given under "Build and run from source" on the software page, and treat that checkout and its image as one release set. If the checkout's own documents describe a different setup, follow the software page and tell the user the difference; the branch, image and setup block you used here are the ones the user replays in stage 6. If the documented source or image is unreachable, stop and ask the user; any source the user places must be a git checkout, not an archive, and the agent never copies private source without explicit approval. Run that section's build step before any data download (the user-owned shell in stage 5 replays that section's whole setup block); the build step catches mismatches that show up at build time, not every mismatch. If the build fails, stop and tell the user.

If a clone or pull is denied, stop and ask the user. Only if the user confirms they have access, the user logs in to that service themselves in their own host shell, entering any secret with the visible prompt in stage 5; the agent never handles it. Otherwise stop. Never fall back to a floating tag.

**5. Data from Hugging Face, in a shell the user owns.** The workflow downloads the clip itself (see "Input clip" in `workflows/av_benchmark/README.md`) and needs the user's `HF_TOKEN`. The user opens a shell on the Brev machine (`brev shell <instance>`, with `--host` as decided in stage 3), starts `tmux` so it survives a dropped connection, accepts the dataset license on the Hugging Face page, and enters the token there with `read -rsp 'Enter your Hugging Face token (input is hidden), then press Enter: ' HF_TOKEN; echo; export HF_TOKEN`. It waits for one pasted line, shows nothing while you paste, and ends on Enter. The token needs read access to the gated dataset. Check it by length only; see "Verifying secrets safely" in `skills/physical-ai-datasets/SKILL.md`. Use the same visible-prompt pattern for any other secret; the agent never handles one.

**6. Configure, run and watch.** Apply the run mode the user confirmed in stage 2; do not ask again. As a recap, state the clip, duration, training steps and camera set as the spec's `vars:` give them (`workflows/av_benchmark/specs/av_benchmark_7cam.yaml`: `CLIP_ID`, `DURATION_SEC`, `TRAIN_SAMPLES`, `CAMERAS`). Any change is applied as exactly the `--set` overrides recorded in stage 2, and that run is labelled a non-benchmark diagnostic. Answer "what configs exist" or "what does X do" from the NuRec configuration page (https://docs.nvidia.com/nurec/nurec/configs.html), the workflow README and the spec's annotated `vars:`, not from memory.

From the user's shell, the user replays the whole setup block of "Build and run from source" (its variables, directories, build configuration and the `nurec_dev` helper together; the helper alone does nothing, and it is defined per terminal) and runs the README command that includes `--with download` (the second command under "Input clip"). That command carries a placeholder `--set CLIP_ID=your-clip-id` line: omit it to use the spec's default clip, or set it to the clip id the user named. Recommend `set -o pipefail` with output teed to a log, for example `$NUREC_SRC/outputs/av_benchmark.log`, so a failure stays visible, and have the command write a durable exit marker (for example `RUN_EXIT=<code>`) to that log when it ends.

The agent watches that log with short repeated `brev exec` checks every few minutes (with `--host` per stage 3) and reports one line of status each time; do not use a single blocking wait of tens of minutes. The exit marker is the end signal. A traceback in the log is a trigger to look closer, for example to check whether the process has exited, not the end. An out-of-memory failure means the run is no longer a valid benchmark: stop and tell the user, and never change benchmark variables or the camera set to dodge it; any reduced-settings run is a non-benchmark diagnostic and must be labelled so (see `skills/nurec/references/troubleshooting.md` for OOM).

Whenever you stop in stages 3 to 6, for a failure or a question, tell the user the instance may still be running and billing, and offer stage 7 with whatever exists (for example the logs).

**7. Results and wrap-up.**

1. Check the instance first with `brev ls --json` (the exact instance in the confirmed organization; plain `brev ls` shows only names when piped). If the check fails or the state is unclear, retry once; if it still fails or is unclear, say the state is unverified and charges may continue, and go to item 5 with the final state unverified. Otherwise follow the state. If the instance no longer exists, say so and go to item 5. If it is not running, say what Brev reports; reading and copying need a running instance, so offer `brev start` for a stopped one, or skip to item 3. After a start, or for an instance that is still starting, repeat this check up to three times, about a minute apart, and handle each result by the rules above; continue here once it runs, and if the start fails or an observed non-running state remains after the last check, say so and go to item 3. If it is running, capture the report inputs that exist: task times from the manifest, key metrics from `metrics/benchmark_row.json`, and where the files are; list each input the run did not produce as not produced.
2. Ask the user what to copy back; suggest those of the outputs named under "What it produces" in the README and by the spec's report outputs that exist, plus the logs, and never the downloaded clip, whose license forbids redistribution. Say the destination, check local free space first, copy with Brev's file-transfer page (https://docs.nvidia.com/brev/latest/guides/development-tools/file-transfer-scp.md), and verify completeness and a checksum. Check the dataset license before sharing outputs outside your organization.
3. Only then, whether or not anything existed to copy, offer the lifecycle actions the selected type supports (the search listing says whether a type can stop; see Brev's instance-management page, https://docs.nvidia.com/brev/latest/cli/instance-management.md), with one line on what each loses or costs, and ask the user; never infer deletion.
4. Execute the chosen action and verify the resulting state.
5. End with a short report: task times and key metrics (or that they were not produced), where the files are, what was not copied, any failure and what was done about it, and the final instance state, marked unverified if it could not be checked. Unless the verified final state is deleted, say plainly what still applies (a running instance is still billing; a stopped one may still incur storage charges; an unverified one may still be billing) and point to Brev's instance-management page for stopping or deleting it.
