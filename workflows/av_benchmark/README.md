<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AV Benchmark workflow

`specs/av_benchmark_7cam.yaml` is the definition: an ordered task list, each naming a binary and
its full argument list. `run.py` resolves `$VAR` placeholders, runs the tasks in order, and writes
`manifest.json` with per-task timings and a per-phase roll-up. It has no DAG, retries, parallelism
or scheduling -- an external orchestrator owns those.

## Input clip

For a source build, complete [Software Setup](https://docs.nvidia.com/nurec/basics/software.html)
first and run the commands below from your host shell. Keep the clip and work
directories under `$NUREC_SRC` so the container can access them.

The chain needs a raw PhysicalAI-AV clip, which this repository does not ship. Download one yourself
from the [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset, under the dataset's license (which does not allow redistributing it), then point the runner
at it:

```bash
nurec_dev bazelisk run //workflows/av_benchmark:run -- \
  --spec "$NUREC_SRC/workflows/av_benchmark/specs/av_benchmark_7cam.yaml" \
  --clip-root "$NUREC_SRC/data/clip" \
  --work-dir "$NUREC_SRC/outputs/av_benchmark"
```

`--clip-root` is required when using a clip you already downloaded. Set `CLIP_ID` to that clip.

Alternatively, enable the spec's `download` task to acquire the clip under the same dataset license.
Authenticate with `HF_TOKEN` and omit `--clip-root`; the download task supplies the clip root to
later tasks:

```bash
nurec_dev bazelisk run //workflows/av_benchmark:run -- \
  --spec "$NUREC_SRC/workflows/av_benchmark/specs/av_benchmark_7cam.yaml" \
  --with download \
  --set CLIP_ID=your-clip-id \
  --work-dir "$NUREC_SRC/outputs/av_benchmark"
```

The spec's defaults are the **benchmark's**: a full clip, the preset's own training budget, and
every metric the chain can produce. CI narrows the two expensive variables with `--set`. There is
one definition, so CI cannot drift from what the benchmark publishes.

`CAMERAS` is the trained and validated camera set, as a Hydra list defaulting to the preset's seven.
The six cameras of the published NuRec dataset leave out the rear tele camera; quote the value, since
an unquoted `[` is a shell glob:

```bash
--set 'CAMERAS=[camera_front_wide_120fov,camera_front_tele_30fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_rear_left_70fov,camera_rear_right_70fov]'
```

`convert` and `aux` prepare all seven cameras either way. The set must include `RENDER_CAMERA`, or
`render` finds no such camera in the trained model.

`GROUND_MESH=true` adds the ground mesh to the exported USDZ, as the published NuRec dataset ships it.

`inject_reference_map` (optional; `--with inject_reference_map`) writes `release/<clip>.usdz`: the
exported USDZ plus `map.xodr`, `pose_record.json` and the `clipgt/` layers of `REFERENCE_USDZ`, any
USDZ of the same clip that carries them (a local path, or an `hf://` path read with `HF_TOKEN`), and
an ECEF `T_world_base`. A reconstruction from a raw clip has no map and an identity `T_world_base`,
and consumers place the map through that pose. The scene keeps its own timestamps.

`T_world_base` is solved from the reference's `pose_record.json`, the map-localised ego trajectory:
the scene's poses are paired with the record's in time and `T_world_base` is the rigid transform that
fits them best. The task fails when the fit leaves more than 1 m median or 3 m 95th-percentile
position residual or 3° median rotation residual, or under 90% of the poses match in time, and skips
with a warning when the reference is missing or cannot place the map. A placed USDZ is re-placed by
passing it as both scene and reference, with the output written to another path. `release/<clip>.reference_map.json` records the outcome.

## What it produces

`manifest.json` carries per-task timing and the reconstruction-time roll-up. Tasks that measure the
result rather than produce it carry no `phase` and are excluded from that total.

Metric files are written by the stages themselves:

| file                               | written by               | covers                                              |
| ---------------------------------- | ------------------------ | --------------------------------------------------- |
| `train/<run_id>/val/metrics.yaml`  | `validate`               | LiDAR fidelity, whole-frame PSNR, per-class `cpsnr` |
| `metrics/rgb/`                     | `render_metrics`         | whole-frame PSNR, SSIM, LPIPS                       |
| `metrics/objects/`                 | `render_metrics_objects` | per-instance crops of tracked actors                |
| `metrics/gaussian_statistics.yaml` | `gaussian_stats`         | gaussian counts and distributions per layer         |
| `metrics/benchmark_row.json`       | `benchmark_row`          | the published row for the clip                      |

`render_video` writes the published video, `release_video/color/<camera>/video.mp4`, at 3840x2160.

`benchmark_row` runs before the runner writes `manifest.json`, so its reconstruction-time fields
carry an absence reason. The orchestrator re-runs it with the task's own arguments plus
`--manifest=$WORK/manifest.json` to fill them.

## What it does not produce

Cohort-level results. One invocation is one clip; the cohort loop, its retries and its aggregation
belong to the orchestrator.

## Failure

The runner stops at the first task that exits non-zero and returns that code, so the CI job fails
when a stage fails. It asserts nothing about metric values: this chain establishes that metrics are
produced, and thresholds belong to whatever consumes them.
