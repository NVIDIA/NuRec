<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec — CLI cookbook

Complete [source setup](install.md) before running the commands below.

Most-used source-built invocations. The single-line summary lives in
`SKILL.md`. See `references/cli-reference.md` for the full command
inventory, the per-sub-command `--help` examples, and the authoritative
flag tables.

## Train + validate (Waymo Open Dataset, dynamic 3DGUT)

> **Recipe scope.** `apps/AV/Waymo/3dgut_dynamic.yaml` is for
> the **Waymo Open Dataset** only — it bakes in the Waymo sensor rig.
> For the NVIDIA Physical AI Autonomous Vehicles (PAI) dataset, use
> the Hyperion-8.1 recipe in the next section instead.

```bash
NAME=my-clip-name  # dataset JSON basename, without .json

nurec_dev bazelisk run //:run -- \
  main \
  --config-name=apps/AV/Waymo/3dgut_dynamic.yaml \
  mode=trainval \
  dataset.path=$NUREC_SRC/dataset/"${NAME:?}".json \
  out_dir=$NUREC_SRC/output \
  logger=tensorboard \
  checkpoint.artifact.enabled=true \
  checkpoint.artifact.rig_trajectories.enabled=true \
  checkpoint.artifact.sequence_tracks.enabled=true
```

## Train + validate (Physical AI Autonomous Vehicles — Hyperion-8.1)

For the gated NVIDIA dataset
[`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles),
train with the AV Benchmark preset shipped in the checkout and the
arguments of the `train` task in
`workflows/av_benchmark/specs/av_benchmark_7cam.yaml` — **not**
the Waymo recipe above. A clip trained this way matches an AV
Benchmark run of the same source release.

```bash
NAME=my-clip-name  # dataset JSON basename, without .json
PRESET_DIR=$NUREC_SRC/workflows/av_benchmark/configs  # absolute path in the released checkout
CAMERAS='[camera_front_wide_120fov,camera_front_tele_30fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_rear_left_70fov,camera_rear_right_70fov,camera_rear_tele_30fov]'

nurec_dev bazelisk run //:run -- \
  main \
  --config-name=${PRESET_DIR}/av_benchmark_lidar_7cam.yaml \
  dataset.path=$NUREC_SRC/dataset/"${NAME:?}".json \
  out_dir=$NUREC_SRC/output \
  mode=train \
  logger.name=dummy \
  logger.run_id=latest \
  dataset.n_samples_per_epoch=30000 \
  "dataset.camera_ids=${CAMERAS}" \
  "dataset.train_camera_ids=${CAMERAS}" \
  "dataset.val_camera_ids=${CAMERAS}"
```

For the full host-side PAI walkthrough see
[`example-workflows/bash/nurec_workflow_pai.md`](example-workflows/bash/nurec_workflow_pai.md).

## Re-validate with a 3 m lateral shift

```bash
RUN_ID=latest  # run dir under out_dir (e.g. latest, 2026-09-03_12-00-00)

nurec_dev bazelisk run //:run -- \
  main \
  --config-name=$NUREC_SRC/output/"${RUN_ID:?}"/config/parsed.yaml \
  mode=val \
  resume=$NUREC_SRC/output/"${RUN_ID:?}"/checkpoints/last.ckpt \
  out_dir=$NUREC_SRC/output \
  dataset.val_sensor_transl_delta_m="[0,3,0]"
```

## Render frames locally (no gRPC) at quarter resolution

```bash
RUN_ID=latest  # run dir under out_dir (e.g. latest, 2026-09-03_12-00-00)

nurec_dev bazelisk run //:run -- \
  render \
  --artifact-path $NUREC_SRC/output/"${RUN_ID:?}"/artifacts/last.usdz \
  --output-dir   $NUREC_SRC/output/"${RUN_ID:?}"/renders \
  --camera-id camera_front_wide_120fov \
  --image-scale 0.25 \
  --image-format png \
  --frame-naming frame-end-timestamp \
  --replicate-training-views \
  --export-video --video-fps 30 --video-crf 20
```

## Render at native resolution along original sensor poses

`--replicate-training-views` (default ON) locks per-frame poses,
intrinsics, and ISP to the training rig, and overrides
`--calib-source` to `training-sensor-poses-calib` (the optimised
sensor-to-world poses) for the most faithful playback. Repeat
`--camera-id` to render multiple sensors. Budget on a 48 GB
Ampere/Ada GPU: ~270 ms per 1920x1080 frame (~5x slower wall
time vs the 0.25 quarter-res preset above).

```bash
NAME=my-clip-name  # scene artifact basename, without .usdz
RUN_ID=latest  # run dir under out_dir (e.g. latest, 2026-09-03_12-00-00)

nurec_dev bazelisk run //:run -- \
  render \
  --artifact-path $NUREC_SRC/scene/"${NAME:?}".usdz \
  --output-dir   $NUREC_SRC/output/"${RUN_ID:?}"/renders \
  --camera-id camera_front_wide_120fov \
  --image-scale 1.0 \
  --image-format png \
  --frame-naming contiguous-output-index \
  --replicate-training-views \
  --renderer default

# Stitch into visually-lossless H.264. Use this fallback whenever
# the NuRec container predates release 26.04 (where the in-container
# --export-video / --video-fps / --video-crf flags landed); on
# 26.04+ append those flags to the render command above and skip
# ffmpeg entirely.
ffmpeg -framerate 30 \
  -i /path/to/output/"${RUN_ID:?}"/renders/color/camera_front_wide_120fov/%06d.png \
  -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \
  /path/to/output/"${RUN_ID:?}"/renders/camera_front_wide_120fov.mp4
```

> **Quality knobs** for `render`:
> `--image-scale 1.0` (native, never upscale) · `--image-format
jpg` (`render` defaults to `png`; only `render-grpc` defaults to
> jpeg. `jpg` is ~10x lighter; use `png` when you
> need bit-exact pixels — GT comparison via
> `eval-rendering-metrics`, perceptual-loss training, or downstream
> perception ingest) · `--replicate-training-views` (keep ON for
> fidelity; flip OFF only when applying rig offsets / actor edits /
> custom rig trajectory) · `--calib-source
training-sensor-poses-calib` (optimised per-sensor poses;
> auto-selected by `--replicate-training-views`; highest fidelity)
> · `--renderer default` (the artifact's trained renderer; also the
> default) · `--frame-naming contiguous-output-index` (drop-in for
> `ffmpeg %06d`) / `frame-end-timestamp` (deterministic global
> timestamps for benchmarking against GT). Post-encode target
> `crf 18` (visually lossless) or `crf 14` (perceptually lossless).

## Launch sensorsim gRPC server with editing enabled

```bash
RUN_ID=latest  # run dir under out_dir (e.g. latest, 2026-09-03_12-00-00)

nurec_dev bazelisk run //:run -- \
  serve-grpc \
  --port="${NUREC_GRPC_PORT:-8080}" \
  --artifact-glob $NUREC_SRC/output/"${RUN_ID:?}"/artifacts/last.usdz \
  --renderer default \
  --enable-harmonizer \
  --enable-editing-actors \
  --test-scenes-are-valid
```

## Render a LiDAR sweep from a running `serve-grpc`

```bash
RUN_ID=latest  # run dir under out_dir (e.g. latest, 2026-09-03_12-00-00)

nurec_dev bazelisk run //:run -- \
  render-grpc \
  --port="${NUREC_GRPC_PORT:-8080}" \
  --artifact-path $NUREC_SRC/output/"${RUN_ID:?}"/artifacts/last.usdz \
  --output-dir $NUREC_SRC/output/"${RUN_ID:?}"/lidar-renders \
  --lidar --lidar-id lidar_top --lidar-format ply
```
