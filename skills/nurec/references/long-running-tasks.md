<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Long-Running Tasks — Subagent + Cron + 5-Minute Status

A copy of the cross-skill long-running-task convention from
the canonical NuRec skill repo, restated here so the `nurec`
skill (and any skill that nests it) is self-contained for the cases
where the rule applies (NuRec training, multi-clip batch renders, harmonizer
runs, `export-custom-rig-trajectory` bakes that miss the cache, large HF
dataset downloads).

---

## Rule

**Any task expected to run ≥ 5 minutes must be delegated to a spawned
subagent** (e.g. via the `Task` tool) so the parent agent stays
responsive. The subagent launches the work as a backgrounded job
under a cron-style ticker that reports status (state, elapsed, last
log line, next-step hint) **every 5 minutes** until the job exits.

- Never block the parent agent on a long foreground command.
- Never sleep blindly between checks — always report status on the
  cron tick.
- Status reports MUST include: current state (`running` / `failed` /
  `done`), elapsed wallclock, the last log line the parent can act
  on, and a one-sentence hint for what the parent should do next
  (continue polling, take over manually, surface a known failure,
  etc.).

The 5-minute cadence is a ceiling, not a floor — a sub-task that
needs faster checkpoints (e.g. boot of `nurec serve-grpc`, which is
~80 s) can tick more often, but the parent must never wait longer
than 5 minutes without a status update.

---

## NuRec-flow cases this rule applies to

- **NuRec training.** Multi-hour runs should always be delegated and polled.
- **Multi-clip batch renders.** Local loops can exceed 5 minutes once the
  clip count grows beyond a few.
- **Harmonizer runs** after render (when the caller opts in).
- **`export-custom-rig-trajectory` generation** for the selected scene and rig.
  ~2-3 minutes on A100-class — borderline; treat as long-running
  whenever wall-clock is non-trivial.
- **Large HF dataset downloads** during dataset preparation or PAI clip staging.

The local `nurec render` / `nurec serve-grpc` warm boot is _not_ in
this list — it's ~80 s on A100, well under the 5-minute threshold.
A parent skill that orchestrates the warm-boot in a chat surface
should still surface progress on a tighter cadence (no >30 s
silent window) so the user doesn't read the wait as a hang, but
the spawned-subagent + cron-tick machinery isn't required.

---

## What "status every 5 minutes" looks like in practice

A minimal compliant tick:

```text
[14:03:21] training (running, 12m18s)
  last:  Epoch 4/100 | loss=0.1843 | psnr=27.4 | lpips=0.082
  next:  Keep polling; checkpoint expected at epoch 10 (~9 min).
```

A failure tick:

```text
[14:48:09] training (failed, 57m41s)
  last:  RuntimeError: CUDA out of memory. Tried to allocate 2.31 GiB
  next:  Surface to user; suggest reducing batch size in the
         training Hydra override and resubmit.
```

A completion tick (final report, no further polling):

```text
[15:22:04] training (done, 1h31m12s)
  artifact:  $NUREC_SRC/output/<run_id>/artifacts/last.usdz
  next:      Render the exported USDZ using the local render recipe.
```

Keep the ticks compact — they should be readable in one screen, not
a wall of logs. Streaming the entire NuRec training log into the
parent every 5 minutes defeats the point of delegating.

---

## Subagent + cron in practice

Two pieces have to be true:

1. **The work runs in a spawned subagent**, not in the parent's main
   tool-call loop. Use the `Task` tool for this. The parent issues
   one `Task` call and then resumes other work / waits on the user;
   the subagent owns the long job end-to-end.
2. **The subagent runs the work as a backgrounded job under a
   ticker.** Inside the subagent, launch the actual command with the
   shell tool in the background (e.g. `block_until_ms: 0` for a
   `Shell` call, or a backgrounded `nohup` + PID file inside that
   call), then loop `AwaitShell` / `sleep` + status read on a 5-min
   cadence. The subagent emits one report per tick.

This is what keeps the parent responsive: the parent's tool-call
budget isn't pinned by a single multi-hour shell, and the user can
still ask follow-up questions while training runs.
