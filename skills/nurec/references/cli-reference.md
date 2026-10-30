<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec CLI Reference

Complete [source setup](install.md) before running the commands below.

This file enumerates the public CLI entry points exposed by the
`//:run` and `//apps:nurec_tools` Bazel targets.

The checkout's own `--help` is always authoritative:

```bash
SUB_COMMAND=render                               # any sub-command from the index below

nurec_dev bazelisk run //:run -- --help
nurec_dev bazelisk run //:run -- "${SUB_COMMAND:?}" --help
nurec_dev bazelisk run //apps:nurec_tools -- --help
```

## Sub-command index — `//:run`

| Sub-command                    | What it does                                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main`                         | Hydra training/validation app (`mode=train                                                                                              \| val \| trainval`). |
| `render`                       | Render frames locally from a USDZ along the training rig (or a custom rig) with optional XYZ rig offsets — **no gRPC server required**.                       |
| `serve-grpc`                   | Stand up the `SensorsimService` gRPC server backed by one or more USDZ artifacts.                                                                             |
| `render-grpc`                  | Demo gRPC client: render RGB or LiDAR along the training trajectory; supports `--edit-assets`.                                                                |
| `render-novel-trajectory`      | Demo gRPC client: scripted novel trajectory (replay → spinning actors → spiral → stop).                                                                       |
| `export-gaussian-plys`         | Per-layer Gaussian `.ply` files in 3DGS or 3DGRT format.                                                                                                      |
| `export-gaussian-usd-asset`    | Standalone Gaussian USD asset for downstream USD pipelines.                                                                                                   |
| `export-mesh`                  | Triangle mesh (Poisson) from fused LiDAR/camera point clouds, optional smoothing + road segmentation.                                                         |
| `export-ground-mesh`           | Plane-segmentation-based Delaunay ground mesh.                                                                                                                |
| `eval-ground-mesh`             | Compute deviation metrics of a ground mesh against the rig trajectory.                                                                                        |
| `export-point-cloud`           | Fused or per-frame LiDAR PLY, colorized by RGB / semantic / road.                                                                                             |
| `export-depth`                 | Per-frame depth NPZ rendered from the trained Gaussians (euclidean or z-depth).                                                                               |
| `export-ego-mask`              | RGBA ego-vehicle hood masks per camera.                                                                                                                       |
| `export-mask-overlay`          | Visualization: blend a directory of binary masks onto a directory of frames.                                                                                  |
| `export-rig-trajectories`      | Per-sensor rig trajectory JSON / USDA.                                                                                                                        |
| `export-sequence-tracks`       | Cuboid / actor track JSON or USDA.                                                                                                                            |
| `export-ncore-tracks`          | Per-sensor per-frame poses (NCore "tracks") for downstream pipelines.                                                                                         |
| `export-custom-rig-trajectory` | Build a `rig_trajectories.json` for `render --custom-rig-trajectory`, optionally driven by an NDAS rig JSON.                                                  |
| `export-external-assets`       | Repackage Asset-Harvester output into a USDZ + emit `edit-assets.json` stub.                                                                                  |
| `export-usdz-artifact`         | Re-export a USDZ from a config + checkpoint (e.g. after re-tuning artifact options).                                                                          |
| `export-ncore-diagnostic`      | Diagnostic dump of an NCore shard (RGB frames, lidar PLYs, semantic overlays, metadata) for QA.                                                               |
| `export-ncore-benchmark-gt`    | Build the ground-truth folder layout consumed by `eval-rendering-metrics`.                                                                                    |
| `export-parsed-config`         | Dump the resolved Hydra config of a YAML file or USDZ to stdout / file (with optional `--upgrade`).                                                           |
| `export-artifact-structure`    | Dump the tensor structure (shape/dtype) of a `.usdz` checkpoint as JSON — useful when authoring upgrade functions.                                            |
| `export-camera-ground-mesh`    | Export a per-camera ground mesh from a trained artifact.                                                                                                      |
| `compress-usdz`                | Compress a USDZ artifact for distribution.                                                                                                                    |
| `decompress-usdz`              | Restore a compressed USDZ artifact.                                                                                                                           |
| `compile-asset-edits`          | Compile an `edit-assets.json` into the form `serve-grpc` consumes.                                                                                            |
| `upgrade-config`               | Persistently upgrade a `parsed.yaml` (or USDZ-extracted config) to a target NuRec version.                                                                    |
| `upgrade-artifact`             | Persistently upgrade a `.usdz` (config + checkpoint) to a target NuRec version.                                                                               |
| `gaussian-statistics`          | Per-class / spatial-density statistics (and a heatmap PNG) for a trained 3DGUT model.                                                                         |
| `eval-rendering-metrics`       | Compute PSNR / SSIM / LPIPS (image-level) and semantic / perceptual similarity of matched object crops (object-level) over rendered vs GT.                    |
| `compute-metrics`              | Single-image or single-video metric (PSNR, SSIM, LPIPS, cpsnr, FID, drift, fcs_adaptive, ntd, perceptual, temporal_coherence, …).                             |
| `viewer`                       | Launch the in-browser USDZ viewer on a port.                                                                                                                  |
| `ply_viewer`                   | Launch the in-browser PLY-only viewer.                                                                                                                        |
| `profile-dataloader`           | Profile the dataloader for a config (training-loop hot path).                                                                                                 |
| `run-script`                   | Run a custom Python script inside the container's NuRec Python env (Bazel runfiles aware).                                                                    |

See §2 for the `//apps:nurec_tools` entry points.

---

## 1. Top-level entry — `//:run`

When the container is invoked without a sub-command keyword (just
`--config-name=...`), it runs the **Hydra training app**. When the
first positional argument is a sub-command (`export-*`, `serve-grpc`,
`render-grpc`, `render`, `viewer`, …), the matching utility runs
instead.

### `main` — training / validation

```bash
CONFIG_NAME=apps/AV/Waymo/3dgut_dynamic.yaml     # Hydra recipe path inside the container
MODE=trainval                                    # train | val | trainval
NAME=my-clip-name                                # NCore store / manifest base name

nurec_dev bazelisk run //:run -- \
  main \
  --config-name="${CONFIG_NAME:?}" \
  mode="${MODE:?}" \
  dataset.path="$NUREC_SRC/dataset/${NAME:?}.json" \
  out_dir=$NUREC_SRC/output
# ...append further Hydra overrides as extra key=value arguments,
# e.g. trainer.max_epochs=5 datamodule.train_batch_size=2
```

| Override                                              | Default                        | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--config-name`                                       | (required)                     | Hydra config to load. Use a recipe shipped in the source checkout at training time, then re-pass `$NUREC_SRC/output/<RUN-ID>/config/parsed.yaml` for validation / export. Pick by source dataset: **Waymo Open Dataset** → `apps/AV/Waymo/3dgut_dynamic.yaml` (Waymo-only; do not use for PAI); **NVIDIA Physical AI Autonomous Vehicles (PAI)** → the AV Benchmark preset `$NUREC_SRC/workflows/av_benchmark/configs/av_benchmark_lidar_7cam.yaml`; PandaSet / NV / Tesla / Alpasim → matching `apps/...` paths. Full table in [`configuration.md`](configuration.md). |
| `mode`                                                | `trainval`                     | `train`, `val`, or `trainval`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `dataset.path`                                        | (required at train)            | NCore JSON manifest inside the container (`$NUREC_SRC/dataset/<NAME>.json`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `out_dir`                                             | `???` (must be set)            | Output root inside the container; bind-mount it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `logger`                                              | `wandb`                        | Pass `logger=tensorboard` to skip the `wandb` prompt; `logger=dummy` to disable logging.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `logger.run_id`                                       | random                         | Set explicitly to make `<RUN-ID>` deterministic. Honors `$NUREC_ENV_RUN_ID` (used by SLURM/cluster wrappers).                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `seed`                                                | `42`                           | Global RNG seed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `matmul_precision`                                    | `medium`                       | `highest`, `high`, or `medium` for `torch.set_float32_matmul_precision`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `trainer.world_size`                                  | `1`                            | GPUs per node. `0` enables SLURM-style auto-detect (when `trainer.num_nodes=0` too).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `trainer.num_nodes`                                   | `1`                            | Number of nodes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `trainer.precision`                                   | `32` (AV recipe) / `16` (base) | Mixed-precision setting.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `trainer.max_epochs`                                  | `1`                            | Training epoch budget.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `trainer.check_val_every_n_epoch`                     | `1`                            | Validate every N epochs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `trainer.batch_size_scaling_factor`                   | `1.0`                          | Scale effective batch size in distributed training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `trainer.training_step_scaling_factor`                | `1.0`                          | Scale effective step count in distributed training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `trainer.relative_lr`                                 | `true`                         | Auto-scale LRs by world size to mimic `world_size=1`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `trainer.relative_schedule`                           | `true`                         | Auto-scale LR schedules by world size.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `trainer.relative_num_workers`                        | `true`                         | Auto-scale dataloader worker count by world size.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `trainer.nccl_timeout_minutes`                        | `30`                           | NCCL collective timeout. Bump on long-running multi-node jobs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `trainer.detect_anomaly`                              | `false`                        | Enable PyTorch Lightning anomaly detection (NaN/Inf).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `dataset.camera_ids`                                  | recipe default                 | E.g. `dataset.camera_ids="['camera_front_wide_120fov','camera_cross_left_120fov']"` (no spaces).                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `dataset.lidar_ids`                                   | recipe default                 | LiDAR ID whitelist.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `dataset.train_camera_ids` / `dataset.val_camera_ids` | `${dataset.camera_ids}`        | Independently override the train and val sensor sets (e.g. render a sensor that wasn't trained on).                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `dataset.seek_offset_sec` / `dataset.duration_sec`    | unset                          | Restrict the time range of an NCore sequence.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `dataset.val_sensor_transl_delta_m`                   | `null`                         | XYZ metres translation delta for novel-view validation. No spaces inside the brackets.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `dataset.val_sensor_rot_delta_deg`                    | `null`                         | Roll-pitch-yaw degree deltas relative to the rig.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `dataset.val_camera_frame_step`                       | `1`                            | Validate every Nth camera frame. Mutually exclusive with `dataset.val_camera_exclude_frame_step`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `dataset.val_camera_exclude_frame_step`               | `null`                         | Exclude every Nth camera frame from validation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `dataset.val_lidar`                                   | `false`                        | Render LiDAR rays in val/test.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `dataset.val_camera`                                  | `true`                         | Render camera rays in val/test.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `dataset.aux_data`                                    | `true`                         | Use the auxiliary `<NAME>.aux.*.zarr.itar` shards. Disable for camera-only debugging.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `dataset.valid_measurements_method`                   | recipe default                 | One of `EGO`, `EGO_CUBOIDTRACKS`, `EGO_SCENEFLOW`, `EGO_CUBOIDTRACKS_SCENEFLOW`, `EGO_CUBOIDTRACKS_TRAFFICLIGHT`, `EGO_FRAMEMASKS`.                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `resume`                                              | `null`                         | Path or filename to a `.ckpt` to resume / validate / export from. Accepts `last`, `last.ckpt`, `epoch=4`, `epoch=4.ckpt`, or an absolute path.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `resume_weights_only`                                 | `false`                        | Load weights only (e.g. for transfer fine-tuning).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `force_validate`                                      | `false`                        | Force a validation pass at the start.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `+log_level`                                          | `2`                            | `0`=FATAL, `1`=ERROR, `2`=WARN, `3`=INFO, `4`=DEBUG. The `+` adds the key.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `checkpoint.every_n_train_steps`                      | `1000`                         | Step interval between checkpoints.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `checkpoint.save_top_k`                               | `1`                            | Keep best-K (`-1` = keep all).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `checkpoint.monitor`                                  | `train/psnr`                   | Metric to monitor for `save_top_k`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `checkpoint.artifact.enabled`                         | `false`                        | Bundle a USDZ on each save. Required for `serve-grpc`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `checkpoint.artifact.checkpoint.enabled`              | `true`                         | Embed the `.ckpt` inside the USDZ.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `checkpoint.artifact.rig_trajectories.enabled`        | `true`                         | Embed rig trajectories.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `checkpoint.artifact.sequence_tracks.enabled`         | `true`                         | Embed cuboid/actor tracks.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `system.test.save_extra_signals`                      | `false`                        | Save per-frame extra signals (semantic, intensity, …) at validation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `system.test.video_fps`                               | `30`                           | FPS used when assembling validation MP4s.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `system.test.metrics.cpsnr.enabled`                   | `true`                         | Class-conditional PSNR.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `system.test.metrics.ssim.enabled`                    | `false`                        | Toggle SSIM at validation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `system.test.metrics.lpips.enabled`                   | `false`                        | Toggle LPIPS at validation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `system.test.lidar.raydrop_threshold`                 | `0.5`                          | LiDAR ray-drop probability cutoff.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

Artifacts (under `${out_dir}/${RUN_ID}/`):

- `config/parsed.yaml` — resolved Hydra config.
- `checkpoints/*.ckpt` — training snapshots; `last.ckpt` is the final one.
- `val/metrics.yaml` — `test/psnr`, `test/ssim`, `test/lpips`.
- `val/*.mp4`, `val/<frame>/*.png` — depth, opacity, segmentation, RGB.
- `logs/` — training and validation logs.
- `artifacts/last.usdz` — final renderable scene (training mode, when
  `checkpoint.artifact.enabled=true`).

#### Resuming a prior run

`resume` resolves relative to `<out_dir>/<RUN-ID>/checkpoints/`. Common
recipes:

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

... resume=last                                    # latest snapshot
... resume="epoch=4.ckpt"                          # specific epoch (no shell expansion)
... resume="$NUREC_SRC/output/${RUN_ID:?}/checkpoints/last.ckpt"   # explicit
```

> Resuming from a different `out_dir` is not allowed — the parsed
> config from the original run is the source of truth.

### `render` (in-container, no gRPC)

Render frames from a USDZ along the training rig (or an externally
authored `rig_trajectories.json`) with optional fixed rig offsets — no
gRPC server, no editing actors required.

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  render \
  --artifact-path "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz" \
  --output-dir "$NUREC_SRC/output/${RUN_ID:?}/renders" \
  --camera-id camera_front_wide_120fov \
  --image-format png \
  --image-scale 0.25 \
  --replicate-training-views
```

Required: `--artifact-path` (USDZ to render) and `--output-dir`
(frames land at `<output_dir>/<output_type>/<sensor_id>/<frame_name>.<ext>`
— e.g. `.../color/camera_front_wide_120fov/000000.png`; `--output-type`
defaults to `color` and `--image-format` defaults to `png`).

Optional flags:

| Flag                                                           | Default                      | Purpose                                                                                                                                                                                                   |
| -------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--camera-id` (repeatable)                                     | unset                        | Camera(s) to render. Omit to render ALL available cameras.                                                                                                                                                |
| `--height`                                                     | unset                        | Output image height. Mutually exclusive with `--image-scale`; one of the two is required.                                                                                                                 |
| `--image-scale`                                                | unset                        | Scale factor in `(0, 1]` applied to each camera's native resolution.                                                                                                                                      |
| `--frame-step`                                                 | `1`                          | Frame stride.                                                                                                                                                                                             |
| `--rolling-shutter-duration`                                   | unset                        | Override per-frame rolling-shutter duration (microseconds).                                                                                                                                               |
| `--image-format [png\|jpg\|jpeg]`                              | `png`                        | Output codec.                                                                                                                                                                                             |
| `--export-video`                                               | off                          | Encode an H.264 MP4 per camera after rendering completes.                                                                                                                                                 |
| `--video-fps`                                                  | `30.0`                       | FPS for `--export-video`.                                                                                                                                                                                 |
| `--video-crf`                                                  | `20`                         | x264 CRF for `--export-video` (0–51, lower = better).                                                                                                                                                     |
| `--renderer [default\|gsplat]`                                 | `default`                    | Renderer backend.                                                                                                                                                                                         |
| `--enable-editing-actors`                                      | off                          | Allow modifications to dynamic actor poses (only renders dynamic actors active in the current frame).                                                                                                     |
| `--demo-actor-transform`                                       | off                          | Demo: apply a Z-axis rotation to editable actors. Requires `--enable-editing-actors`.                                                                                                                     |
| `--rig-translation-offset tx ty tz`                            | `0 0 0`                      | Static rig-frame XYZ offset in metres.                                                                                                                                                                    |
| `--rig-rotation-offset yaw -roll -pitch`                       | `0 0 0`                      | Static rig-frame rotation in degrees (axis quirk: see flag help).                                                                                                                                         |
| `--replicate-training-views` / `--no-replicate-training-views` | `--replicate-training-views` | Replicate the per-frame ISP / camera intrinsics from training. Set `--no-replicate-training-views` whenever you apply rig offsets, edit actors, override rolling shutter, or use a custom rig trajectory. |
| `--calib-source`                                               | `training-rig-poses`         | Pose source: `training-rig-poses`, `training-rig-poses-per-frame`, `training-sensor-poses-nocalib`, `training-sensor-poses-calib`.                                                                        |
| `--frame-naming`                                               | `contiguous-output-index`    | `frame-end-timestamp` (microseconds, recommended for benchmarking) or `contiguous-output-index` (0,1,2,…).                                                                                                |
| `--custom-rig-trajectory`                                      | unset                        | Path to a `rig_trajectories.json` (e.g. produced by `export-custom-rig-trajectory`) to override the embedded trajectory.                                                                                  |
| `--max-pending-save-tasks`                                     | `64`                         | Backpressure for the disk-save thread pool.                                                                                                                                                               |
| `--save-workers`                                               | `16`                         | Number of disk-save worker threads.                                                                                                                                                                       |

#### Quality presets

Two ready-made flag combinations cover ~all batch-render use cases.
Pick by purpose; both keep `--replicate-training-views` ON so the
playback follows the original sensor poses.

| Preset                           | Use it when…                                                                                                      | Flags                                                                                                                                                                  |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Quick preview / iteration**    | dialing in a rig offset, smoke-testing a new artifact, or producing a low-bandwidth proxy MP4                     | `--image-scale 0.25 --image-format jpg --replicate-training-views --renderer default`                                                                                  |
| **Native resolution / archival** | producing the deliverable render, or input to a downstream perception/eval pipeline                               | `--image-scale 1.0 --image-format jpg --replicate-training-views --calib-source training-sensor-poses-calib --renderer default --frame-naming contiguous-output-index` |
| **Native resolution / lossless** | GT comparison via `eval-rendering-metrics`, perceptual-loss training, or any pipeline that needs bit-exact pixels | swap `--image-format jpg` → `--image-format png` in the row above (~10× more disk per frame)                                                                           |

Notes for the native-resolution presets:

- `--image-scale 1.0` matches the camera's native training resolution
  (typically 1920×1080 for the AV front-wide); never pass a value > 1
  — Gaussians are not super-resolving above the training footprint.
- `--image-format jpg` is the default for batch/archival use because
  it's ~10× lighter on disk than `png` at perceptually negligible
  cost. Switch to `--image-format png` whenever you need bit-exact
  output — GT comparison, perceptual-loss training, or downstream
  perception ingest.
- `--replicate-training-views` (default ON) implicitly sets
  `--calib-source` to `training-sensor-poses-calib`, the optimised
  per-sensor poses; passing it explicitly makes the contract obvious
  in scripts.
- `--renderer default` uses the renderer the artifact was trained with
  (a PyTorch forward pass); it is also the CLI default. `gsplat` forces
  the GSplat rasterizer regardless of how the artifact was trained —
  useful for cross-checks, at the cost of frame-to-frame variation.
- `--frame-naming contiguous-output-index` is the drop-in for
  `ffmpeg -i %06d.png`; switch to `frame-end-timestamp` only if you
  need filenames that match an external benchmark/GT layout (e.g.
  `eval-rendering-metrics`).
- Encode wisely: `ffmpeg -framerate 30 -i %06d.png -c:v libx264
-crf 18 -preset slow -pix_fmt yuv420p out.mp4` is visually
  lossless; `crf 14` is perceptually lossless; `libx265` halves the
  bitrate at the same quality. On NuRec 26.04+ you can skip ffmpeg and
  pass `--export-video --video-fps 30 --video-crf 18` directly to
  `render` — but earlier containers (e.g. `:latest` 26.2.150) reject
  those flags, so the `ffmpeg` post-step is the portable path.

### `serve-grpc`

Start the sensorsim gRPC server backed by one or more NuRec USDZs.

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  serve-grpc \
  --artifact-glob "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz"
```

Required flags:

| Flag              | Default | Purpose                                                                         |
| ----------------- | ------- | ------------------------------------------------------------------------------- |
| `--artifact-glob` | `None`  | Glob pattern matching `.usdz` files to load. Quote it to avoid shell expansion. |

Optional flags:

| Flag                                                     | Default                                                                        | Purpose                                                                                                           |
| -------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `--host`                                                 | `localhost`                                                                    | gRPC bind host. Use `0.0.0.0` to expose to the LAN (with `-p 8080:8080` instead of `--net=host`).                 |
| `--port`                                                 | `8080`                                                                         | gRPC bind port.                                                                                                   |
| `--health-port`                                          | unset                                                                          | If set, run a dedicated `grpc.health.v1.Health` server on that port; otherwise health is multiplexed on `--port`. |
| `--test-scenes-are-valid` / `--no-test-scenes-are-valid` | `False`                                                                        | Load + validate every scene at startup before serving.                                                            |
| `--renderer [default\|gsplat]`                           | `default`                                                                      | `default` = the renderer the artifact was trained with; `gsplat` = force the GSplat rasterizer.                   |
| `--enable-harmonizer`                                    | `False`                                                                        | Apply the Harmonizer in-server. Mutually exclusive with `--enable-harmonizer-temporal`.                           |
| `--harmonizer-url`                                       | `https://huggingface.co/nvidia/Harmonizer/blob/main/harmonizer_nontemporal.pt` | Checkpoint URL.                                                                                                   |
| `--harmonizer-cache`                                     | `~/.cache/nurec/harmonizer`                                                    | Local checkpoint cache directory.                                                                                 |
| `--harmonizer-model-filename`                            | `harmonizer_nontemporal.pt`                                                    | Checkpoint filename.                                                                                              |
| `--harmonizer-resolution`                                | `(576, 1024)`                                                                  | (H, W) the Harmonizer runs at. Takes TWO ints, e.g. `--harmonizer-resolution 576 1024`.                           |
| `--enable-timing`                                        | `False`                                                                        | Print per-stage rendering timings.                                                                                |
| `--ray-chunk-size`                                       | `2^62`                                                                         | Max rays per forward pass; lower it to fit smaller GPUs.                                                          |
| `--egocar-hood-dir`                                      | unset                                                                          | Override directory of egocar hood images.                                                                         |
| `--enable-editing-actors`                                | `False`                                                                        | Required for `render-grpc --edit-assets`.                                                                         |
| `--download-cache-dir`                                   | `~/.cache/nurec/downloaded_scenes`                                             | Cache for scenes downloaded on demand by the server.                                                              |
| `--download-cache-size`                                  | `5`                                                                            | Max number of cached downloaded scenes (LRU).                                                                     |
| `--cache-size`                                           | `10`                                                                           | Max number of loaded backends (count-based LRU). On OOM, spare backends are auto-evicted and load is retried.     |
| `--max-workers`                                          | `1`                                                                            | Thread-pool worker count for the gRPC server.                                                                     |
| `--metrics-output-dir`                                   | unset                                                                          | Directory to write per-request rendering-time metrics.                                                            |

Deprecated (still accepted with a warning, hidden from `--help`). Each of
these defaults to `None` and only takes effect if you pass it explicitly; the
live `--harmonizer-*` defaults above apply otherwise:

- `--use-gsplat` → `--renderer gsplat`
- `--enable-difix` → `--enable-harmonizer`
- `--difix-url` → `--harmonizer-url`
- `--difix-cache` → `--harmonizer-cache`
- `--difix-model-filename` → `--harmonizer-model-filename`
- `--difix-resolution` → `--harmonizer-resolution`

### `render-grpc`

Demo gRPC client that walks the embedded training trajectory of a USDZ
and writes one frame per sample (RGB by default; LiDAR with `--lidar`).
Supports asset editing.

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  render-grpc \
  --artifact-path "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz" \
  --output-dir "$NUREC_SRC/output/${RUN_ID:?}/renders" \
  --camera-id camera_front_wide_120fov
```

Required: `--artifact-path` (NuRec `.usdz` to render) and
`--output-dir` (rendered images go here).

Optional flags (RGB):

| Flag                                     | Default                    | Purpose                                                                                              |
| ---------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--host`                                 | `localhost`                | gRPC server host.                                                                                    |
| `--port`                                 | `8080`                     | gRPC server port.                                                                                    |
| `--height`                               | `300`                      | Output image height (px).                                                                            |
| `--camera-id`                            | `camera_front_wide_120fov` | Camera identifier.                                                                                   |
| `--image-format [png\|jpeg]`             | `jpeg`                     | Output codec.                                                                                        |
| `--frame-step`                           | `1`                        | Frame stride.                                                                                        |
| `--disable-rolling-shutter`              | unset                      | Replace rolling-shutter timestamps with whole-frame end timestamps.                                  |
| `--enable-editing-actors`                | off                        | Send dynamic actor updates each frame (only renders actors active in the current frame).             |
| `--demo-actor-transform`                 | off                        | Demo: rotate editable actors per frame. Requires `--enable-editing-actors`.                          |
| `--shutdown-server-on-completion`        | unset                      | Stop the server when rendering finishes.                                                             |
| `--rig-name`                             | unset                      | E.g. `hyperion8.0`, `hyperion8.1` to inpaint the matching ego hood. Set unset / `None` to disable.   |
| `--rig-translation-offset tx ty tz`      | `0 0 0`                    | Static rig-frame XYZ offset in metres.                                                               |
| `--rig-rotation-offset yaw -roll -pitch` | `0 0 0`                    | Static rig-frame rotation in degrees.                                                                |
| `--frame-naming`                         | `contiguous-output-index`  | `frame-end-timestamp` (microseconds) or `contiguous-output-index` (0,1,2,…).                         |
| `--edit-assets`                          | unset                      | Path to `edit-assets.json` from `export-external-assets`. Requires server `--enable-editing-actors`. |
| `--sequential`                           | off                        | Send requests one-by-one (no async batching after warmup). Use to debug latency.                     |

LiDAR-mode flags (require `--lidar`):

| Flag                                                     | Default          | Purpose                                                                   |
| -------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------- |
| `--lidar`                                                | off              | Switch to `render_lidar` and write point clouds.                          |
| `--lidar-id`                                             | first available  | LiDAR sensor ID to use for frame timestamps.                              |
| `--lidar-format [bin\|ply]`                              | `bin`            | Binary blob (`xyz` then `intensity`) or PLY with intensity-mapped colors. |
| `--lidar-raydrop-threshold`                              | `0.5`            | Drop rays with `raydrop > T`.                                             |
| `--lidar-opacity-threshold`                              | `0.0`            | Drop rays with `opacity <= T`. `0.0` disables the filter.                 |
| `--lidar-distance-filter` / `--no-lidar-distance-filter` | artifact default | Toggle distance-based edge filtering (removes floating points).           |
| `--lidar-distance-filter-threshold`                      | artifact default | `[0, 1]`; higher = fewer points filtered.                                 |

`render-grpc` writes `timestamps.json` next to the rendered frames
listing the per-frame `frame_start_timestamp_us` /
`frame_end_timestamp_us` and the file name on disk. It also writes
`render_grpc_cli_args.json` capturing the exact CLI invocation.

### `render-novel-trajectory`

Demo gRPC client that scripts a more interesting playback (replay,
spinning actors, spiral, stop-ego, …) over an existing trained USDZ.
Useful for visualising a checkpoint without writing your own client.

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  render-novel-trajectory \
  --artifact-path "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz" \
  --output-dir "$NUREC_SRC/output/${RUN_ID:?}/renders" \
  --camera-id camera_front_wide_120fov \
  --height 540 \
  --store-video
```

| Flag                                 | Default                    | Purpose                                  |
| ------------------------------------ | -------------------------- | ---------------------------------------- |
| `--artifact-path`                    | (required)                 | USDZ to drive.                           |
| `--output-dir`                       | (required)                 | Where to save frames + MP4.              |
| `--host`                             | `localhost`                | gRPC server host.                        |
| `--port`                             | `8080`                     | gRPC server port.                        |
| `--height`                           | `300`                      | Output image height (px).                |
| `--camera-id`                        | `camera_front_wide_120fov` | Camera identifier.                       |
| `--shutdown-server-on-completion`    | `False`                    | Stop the server when rendering finishes. |
| `--store-video` / `--no-store-video` | `--store-video`            | Encode an MP4 of the result.             |

### Export sub-commands

All export sub-commands share the same general pattern: bind the
training `dataset` and `output` directories at the same paths used at
training time, re-pass `parsed.yaml` as `--config-name`, and reference
`last.ckpt` (or another snapshot) when the command needs Gaussians.

#### `export-gaussian-plys`

```bash
NAME=my-clip-name                                # NCore store / manifest base name
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  export-gaussian-plys \
  --config-name "$NUREC_SRC/output/${RUN_ID:?}/config/parsed.yaml" \
  --checkpoint-name last.ckpt
```

| Flag                       | Default      | Purpose                                                                                                       |
| -------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------- |
| `--config-name`            | (required)   | The training `parsed.yaml`.                                                                                   |
| `--checkpoint-name`        | `last.ckpt`  | Checkpoint filename inside the run's `checkpoints/` dir.                                                      |
| `--output-dir`             | `<run>/plys` | Output directory.                                                                                             |
| `--format [_3DGS\|_3DGRT]` | `_3DGS`      | `_3DGS` is third-party-viewer compatible; `_3DGRT` exports oriented octahedra reflecting Gaussian parameters. |
| `--percentage-gaussians`   | `100`        | `(0, 100]` — sub-sample factor (\_3DGRT only).                                                                |

#### `export-gaussian-usd-asset`

Standalone Gaussian USD asset (PPISP-compatible) for downstream USD
pipelines. Same `--config-name` / `--checkpoint-name` / `--output-dir`
contract as `export-gaussian-plys`. Run `--help` for the long flag list.

#### `export-mesh`

Triangular mesh (Poisson) from fused LiDAR/camera point clouds.

```bash
NAME=my-clip-name                                # NCore store / manifest base name
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  export-mesh \
  --config-name "$NUREC_SRC/output/${RUN_ID:?}/config/parsed.yaml" \
  --output-dir "$NUREC_SRC/output/${RUN_ID:?}/meshes"
```

| Flag                                                         | Default       | Purpose                                                      |
| ------------------------------------------------------------ | ------------- | ------------------------------------------------------------ |
| `--config-name`                                              | (required)    | Hydra config with a dataset spec.                            |
| `--output-dir`                                               | (required)    | Output directory.                                            |
| `--mesh-basename`                                            | `mesh`        | Base filename.                                               |
| `--camera-id` (repeatable)                                   | all           | Cameras to fuse for the point cloud.                         |
| `--lidar-id` (repeatable)                                    | all           | LiDARs to fuse for the point cloud.                          |
| `--format [ply\|usd]` (repeatable)                           | `ply usd`     | Output formats.                                              |
| `--smooth` / `--no-smooth`                                   | `--no-smooth` | Apply rig-trajectory-based mesh smoothing.                   |
| `--n-neighbors`                                              | `200`         | k-NNs used for normal estimation.                            |
| `--step-frame`                                               | `1`           | Frame stride.                                                |
| `--trim-distance`                                            | `0.225`       | Vertex-to-2nd-nearest-point distance threshold for trimming. |
| `--apply-road-segmentation` / `--no-apply-road-segmentation` | off           | Segment road vs non-road faces.                              |
| `--export-disjoint-meshes`                                   | off           | Export segmented road / non-road meshes as separate files.   |
| `--coord-space [nurec\|world]`                               | `world`       | Output coordinate frame.                                     |

#### `export-ground-mesh`

Plane-segmentation-based Delaunay ground mesh; useful for downstream
"ground hugging" simulators.

| Notable flag                   | Default | Purpose                         |
| ------------------------------ | ------- | ------------------------------- |
| `--num-plane-hypotheses`       | `100`   | RANSAC-style plane sampling.    |
| `--plane-max-distance`         | `0.3`   | Inlier threshold (m).           |
| `--plane-max-angle-deg`        | `30.0`  | Max normal angle.               |
| `--smoothing-passes`           | `10`    | Mesh smoothing iterations.      |
| `--voxel-size`                 | `0.1`   | Voxel-grid downsample size (m). |
| `--export-meshing-diagnostics` | off     | Dump intermediate PLYs for QA.  |

#### `eval-ground-mesh`

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  eval-ground-mesh \
  --usdz-path "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz" \
  --output-dir "$NUREC_SRC/output/${RUN_ID:?}/ground-mesh-eval"
```

| Flag                    | Purpose                                         |
| ----------------------- | ----------------------------------------------- |
| `--output-dir`          | (required) Where to write the metrics.          |
| `--usdz-path`           | USDZ to read ground mesh + rig trajectory from. |
| `--ground-mesh-path`    | Override: load a PLY ground mesh directly.      |
| `--rig-trajectory-path` | Override: load a JSON rig trajectory directly.  |

#### `export-point-cloud`

Fused (or per-frame) LiDAR PLY, colorized by RGB, semantic class, or road/non-road.

| Flag                                      | Default    | Purpose                                                           |
| ----------------------------------------- | ---------- | ----------------------------------------------------------------- |
| `--config-name`                           | (required) | Hydra config.                                                     |
| `--output-dir`                            | (required) | Output directory.                                                 |
| `--colorizer [none\|rgb\|semantic\|road]` | `rgb`      | Colorization mode. `rgb` keeps only camera-visible points.        |
| `--per-frame`                             | off        | Export one PLY per LiDAR frame instead of a single fused cloud.   |
| `--per-class`                             | off        | Split into one PLY per semantic class (when `semantic` / `road`). |
| `--frame-step`                            | `50`       | Frame stride for `--per-frame`.                                   |
| `--valid-points-only`                     | off        | Filter to "valid" points only.                                    |

#### `export-depth`

Per-frame depth NPZ rendered from the trained Gaussians.

| Flag                                | Default       | Purpose                                                                           |
| ----------------------------------- | ------------- | --------------------------------------------------------------------------------- |
| `--config-name`                     | (required)    | `parsed.yaml`.                                                                    |
| `--checkpoint-name`                 | `last.ckpt`   | Checkpoint filename.                                                              |
| `--depth-type [euclidean\|z-depth]` | (required)    | `euclidean` returns the ray-distance; `z-depth` projects along the camera Z-axis. |
| `--output-dir`                      | `<run>/depth` | Output directory. **Currently ignored — see below.**                              |

> **`--output-dir` does not take effect.** `nurec/utils/io/export/depth.py`
> selects `Path(config.out_dir)` in the branch that runs when `--output-dir` > _is_ supplied, so passing it silently writes to the config's `out_dir`
> instead. Set `out_dir` in the config, or move the files afterwards. This is
> an upstream code bug, not a doc one.

Output: `depth_000000.npy.npz` (compressed numpy, key `depth`) per frame.

#### `export-ego-mask`

```bash
NAME=my-clip-name          # NCore store base name
CAM1=camera_front_wide_120fov
CAM2=camera_cross_left_120fov

nurec_dev bazelisk run //:run -- \
  export-ego-mask \
  --shard-file-pattern "$NUREC_SRC/dataset/${NAME:?}.zarr.itar" \
  --output-dir "$NUREC_SRC/output" \
  --camera-ids "${CAM1:?}" --camera-ids "${CAM2:?}"
```

| Flag                                 | Default            | Purpose                                                     |
| ------------------------------------ | ------------------ | ----------------------------------------------------------- |
| `--shard-file-pattern`               | (required)         | Glob for NCore zarr.itar shards.                            |
| `--output-dir`                       | (required)         | RGBA PNGs land in `<output_dir>/ego-hoods/<camera_id>.png`. |
| `--camera-ids` (repeatable)          | all available      | Cameras to export.                                          |
| `--invert-mask` / `--no-invert-mask` | `--no-invert-mask` | Invert the masked region.                                   |
| `--camera-frame-idx`                 | `50`               | Frame index used as the RGB backdrop.                       |

#### `export-mask-overlay`

Visualization helper that overlays a directory of binary masks on top of
a directory of frame sequences.

| Flag                        | Purpose                                                              |
| --------------------------- | -------------------------------------------------------------------- |
| `--image-dir`               | Root containing one subdirectory per image sequence.                 |
| `--mask-dir`                | Directory of binary PNG masks (one per sequence; named `<seq>.png`). |
| `--output-dir`              | Output directory mirroring `--image-dir`.                            |
| `--mask-color`              | RGB triple in `[0, 1]`.                                              |
| `--mask-alpha`              | Blend strength in `[0, 1]`.                                          |
| `--invert-mask`             | Treat zero pixels as masked.                                         |
| `--image-format [png\|jpg]` | Output codec.                                                        |
| `--jpeg-quality`            | `0–100`.                                                             |
| `--png-compression`         | `0–9`.                                                               |

#### `export-rig-trajectories`

```bash
NAME=my-clip-name                                # NCore store / manifest base name
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  export-rig-trajectories \
  --config-name "$NUREC_SRC/output/${RUN_ID:?}/config/parsed.yaml" \
  --output-dir "$NUREC_SRC/output/${RUN_ID:?}"
```

Produces `rig_trajectories.json` and `.usda` next to the run.

#### `export-sequence-tracks`

Export cuboid / actor track data for a trained run.

| Flag                                   | Default         | Purpose                                                   |
| -------------------------------------- | --------------- | --------------------------------------------------------- |
| `--config-name`                        | (required)      | `parsed.yaml`.                                            |
| `--checkpoint-path`                    | unset           | Optional `.ckpt`; required to merge model-updated tracks. |
| `--output-dir`                         | (required)      | Output directory.                                         |
| `--dynamic-only` / `--all-tracks`      | `--all-tracks`  | Restrict to dynamic actors.                               |
| `--world-frame` / `--nurec-frame`      | `--world-frame` | Coordinate frame.                                         |
| `--format [json\|usda]` (repeatable)   | `json usda`     | Output formats.                                           |
| `--controllable-only` / `--all-tracks` | `--all-tracks`  | Only export controllable actors.                          |

The doc snippet you may have seen elsewhere uses `--checkpoint-name`;
the modern flag is `--checkpoint-path`.

#### `export-ncore-tracks`

Per-sensor per-frame poses (NCore "tracks") for downstream pipelines.

```bash
NAME=my-clip-name
CAM1=camera_front_wide_120fov
CAM2=camera_cross_left_120fov
LIDAR=lidar_top_360fov

nurec_dev bazelisk run //:run -- \
  export-ncore-tracks \
  --shard-file-pattern "$NUREC_SRC/dataset/${NAME:?}.zarr.itar" \
  --model-tracks-json "$NUREC_SRC/output/sequence_tracks.json" \
  --output-dir "$NUREC_SRC/output" \
  --camera-id "${CAM1:?}" --camera-id "${CAM2:?}" \
  --lidar-id "${LIDAR:?}"
```

| Flag                                   | Default                    | Purpose                                                                    |
| -------------------------------------- | -------------------------- | -------------------------------------------------------------------------- |
| `--shard-file-pattern`                 | (required)                 | Glob for NCore zarr.itar shards.                                           |
| `--model-tracks-json`                  | (required)                 | JSON written by `export-sequence-tracks`.                                  |
| `--output-dir`                         | (required)                 | Output directory.                                                          |
| `--seek-offset-sec`                    | unset                      | Initial pose-timestamp offset (seconds).                                   |
| `--duration-sec`                       | unset                      | Sequence duration (seconds, `-1` for all).                                 |
| `--camera-id` (repeatable)             | `camera_front_wide_120fov` | Cameras to include. The first camera is the reference for frame numbering. |
| `--lidar-id`                           | `lidar`                    | LiDAR sensor for ego-trajectory rays.                                      |
| `--enable-lidars` / `--disable-lidars` | `--enable-lidars`          | Include LiDAR data.                                                        |

Output: `sensor_tracks_<start_us>_<end_us>.json`.

#### `export-custom-rig-trajectory`

Build a `rig_trajectories.json` for `render --custom-rig-trajectory`,
optionally rewriting camera calibrations from an NDAS rig JSON.

| Flag                         | Purpose                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `--artifact-path`            | Source USDZ (one of `--artifact-path` or `--reference-rig-trajectory` is required). |
| `--reference-rig-trajectory` | Source rig trajectory JSON.                                                         |
| `--rig-json`                 | Optional NDAS rig JSON used to rewrite camera calibrations.                         |
| `--output`                   | Output JSON path.                                                                   |

#### `export-external-assets`

Repackage a target USDZ with assets produced by the Asset Harvester.
See `references/asset-editing.md` for the JSON schema and the full
edit / render flow.

```bash
nurec_dev bazelisk run //:run -- \
  export-external-assets \
  --artifact-path $NUREC_SRC/output/target.usdz \
  --external-assets-dir $NUREC_SRC/assets \
  --output-edit-file $NUREC_SRC/output/edit-assets.json \
  --output-artifact-path $NUREC_SRC/output/target-external-assets.usdz
```

| Flag                     | Required | Purpose                                                                                                          |
| ------------------------ | -------- | ---------------------------------------------------------------------------------------------------------------- |
| `--artifact-path`        | yes      | Input USDZ.                                                                                                      |
| `--external-assets-dir`  | yes      | Asset-Harvester output directory (must contain `metadata.yaml`).                                                 |
| `--output-edit-file`     | yes      | Stub `edit-assets.json` to fill in.                                                                              |
| `--output-artifact-path` | no       | If omitted, only the JSON is written; otherwise the USDZ is repackaged with the AH assets in `external_assets/`. |

#### `export-usdz-artifact`

Re-export a USDZ from a config + checkpoint (e.g. after toggling
`checkpoint.artifact.*` options).

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  export-usdz-artifact \
  --config-name "$NUREC_SRC/output/${RUN_ID:?}/config/parsed.yaml" \
  --checkpoint-name last.ckpt \
  --output-dir "$NUREC_SRC/output/${RUN_ID:?}/usd-out"
```

#### `export-ncore-diagnostic`

Diagnostic dump of an NCore shard for QA — RGB frames, LiDAR PLYs,
semantic overlays, rig metadata.

| Flag                                                               | Default     | Purpose                                    |
| ------------------------------------------------------------------ | ----------- | ------------------------------------------ |
| `--dataset-path` (preferred) / `--shard-file-pattern` (deprecated) | (required)  | NCore manifest.                            |
| `--output-dir`                                                     | unset       | Where to write artifacts.                  |
| `--frame-step-camera`                                              | `50`        | Camera frame stride.                       |
| `--frame-step-lidar`                                               | `50`        | LiDAR frame stride.                        |
| `--frame-naming [index\|timestamp]`                                | `timestamp` | Per-frame naming scheme.                   |
| `--video-fps`                                                      | `15`        | FPS for any exported video.                |
| `--meta`                                                           | off         | Export NCore metadata as YAML.             |
| `--camera-images`                                                  | off         | Export camera RGB frames.                  |
| `--lidar-points`                                                   | off         | Per-frame LiDAR PLYs (per spin).           |
| `--lidar-points-fused`                                             | off         | Single fused LiDAR PLY per sensor.         |
| `--semantic-labelmaps`                                             | off         | Per-camera semantic PNGs (when available). |
| `--semantic-overlays`                                              | off         | Semantic labelmaps blended on RGB.         |

V4 component-group selectors: `--poses-component-group`,
`--intrinsics-component-group`, `--masks-component-group`,
`--cuboids-component-group`. Default is `default`.

#### `export-ncore-benchmark-gt`

Build the ground-truth folder layout consumed by `eval-rendering-metrics`.
Run `--help` for the full flag list.

#### `export-parsed-config`

Dump the resolved Hydra config of a YAML file or USDZ. Useful for
diffing recipes between releases or extracting a `parsed.yaml` from a
foreign USDZ.

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)
CONFIG_NAME=apps/AV/Waymo/3dgut_dynamic.yaml     # Hydra recipe path inside the container

# From an artifact (auto-loads the embedded parsed config).
nurec_dev bazelisk run //:run -- \
  export-parsed-config --input "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz"

# From a YAML recipe with an upgrade pass.
nurec_dev bazelisk run //:run -- \
  export-parsed-config --config-name apps/AV/Waymo/3dgut_dynamic.yaml --upgrade

# Save to file with sorted keys.
nurec_dev bazelisk run //:run -- \
  export-parsed-config --config-name "${CONFIG_NAME:?}" --output $NUREC_SRC/output/parsed.yaml
```

Pass Hydra overrides after `--`:

```bash
CONFIG_NAME=apps/AV/Waymo/3dgut_dynamic.yaml     # Hydra recipe path inside the container

... export-parsed-config --config-name "${CONFIG_NAME:?}" -- datamodule.train_batch_size=2
```

#### `export-artifact-structure`

Dump the tensor structure (shape/dtype) of a `.usdz` checkpoint as
JSON — most useful when authoring upgrade functions or debugging
checkpoint shape mismatches.

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  export-artifact-structure --input "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz" \
  --output $NUREC_SRC/output/structure.json
```

Accepts both `.usdz` and raw `.ckpt` inputs.

### `upgrade-config` / `upgrade-artifact`

Older artifacts are auto-upgraded **in memory** every time they are
loaded; use these commands when you want to persist the upgrade
(faster reuse, easier diffing).

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

# Persistently upgrade a YAML (or USDZ-extracted) config.
nurec_dev bazelisk run //:run -- \
  upgrade-config \
  --input "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz" \
  --output "$NUREC_SRC/output/${RUN_ID:?}/upgraded.yaml" \
  --target-version 26.4

# Persistently upgrade a full USDZ (config + checkpoint).
nurec_dev bazelisk run //:run -- \
  upgrade-artifact \
  --input "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz" \
  --output "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.upgraded.usdz"
```

Common flags:

| Flag                                                     | Purpose                                                                   |
| -------------------------------------------------------- | ------------------------------------------------------------------------- |
| `--input`                                                | Path to the source `.usdz` or `.yaml`.                                    |
| `--output`                                               | Output path (must not exist for `upgrade-artifact`).                      |
| `--target-version`                                       | `<major>.<minor>[.<patch>]`. Defaults to the container's current version. |
| `--sort-keys` / `--no-sort-keys` (`upgrade-config` only) | Sort YAML keys (helps diffing).                                           |
| `--debug` (`upgrade-artifact` only)                      | Verbose logging.                                                          |

### `gaussian-statistics`

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  gaussian-statistics \
  --config-name "$NUREC_SRC/output/${RUN_ID:?}/config/parsed.yaml" \
  --checkpoint-name last.ckpt \
  --tile-size 10.0 \
  --heatmap \
  --output-file "$NUREC_SRC/output/${RUN_ID:?}/gaussian-stats.yaml"
```

| Flag                         | Default     | Purpose                                           |
| ---------------------------- | ----------- | ------------------------------------------------- |
| `--config-name`              | (required)  | Training `parsed.yaml`.                           |
| `--checkpoint-name`          | `last.ckpt` | Snapshot to analyse.                              |
| `--tile-size`                | `10.0`      | Tile (m²) for spatial-density binning.            |
| `--heatmap` / `--no-heatmap` | `--heatmap` | Save a tile-density heatmap PNG.                  |
| `--zoom-threshold`           | `1`         | Zoom factor for the heatmap.                      |
| `--output-file`              | unset       | Persist the per-class stats (`.json` or `.yaml`). |

### `eval-rendering-metrics`

Object-level metrics are **opt-in**: pass `--object-level-metrics` _and_ a
source for the object data. Without both, only image-level metrics are
produced.

Compute image-level (PSNR / SSIM / LPIPS) and object-level (semantic and
perceptual similarity of matched object crops) metrics over a directory of
rendered frames vs ground truth.

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  eval-rendering-metrics \
  --render-dir "$NUREC_SRC/output/${RUN_ID:?}/renders/color" \
  --gt-dir "$NUREC_SRC/output/${RUN_ID:?}/gt" \
  --output-dir "$NUREC_SRC/output/${RUN_ID:?}/metrics" \
  --metrics psnr --metrics ssim --metrics lpips
```

The expected GT layout is `<gt-dir>/camera_images/<seq>/<frame_ts>.<ext>`
plus optional ego masks at `<gt-dir>/camera_ego_masks/<seq>.png`.
Generate it with `export-ncore-benchmark-gt`.

| Flag                                                                                                         | Default    | Purpose                                                                |
| ------------------------------------------------------------------------------------------------------------ | ---------- | ---------------------------------------------------------------------- |
| `--render-dir`                                                                                               | (required) | Rendered frame root (or a directory of MP4s for object-level metrics). |
| `--gt-dir`                                                                                                   | (required) | GT root. (Pass any directory when only running object-level metrics.)  |
| `--output-dir`                                                                                               | (required) | Where to write metrics + visualizations.                               |
| `--rendered-image-extension`                                                                                 | `png`      | Rendered frame extension.                                              |
| `--metrics psnr/ssim/lpips` (repeatable)                                                                     | `psnr`     | Image-level metrics.                                                   |
| `--max-frames`                                                                                               | unset      | Quick-test cap.                                                        |
| `--visualize` / `--visualization-multiplier` / `--visualization-normalized` / `--visualization-jpeg-quality` | off        | Per-frame difference visualizations.                                   |

### `compute-metrics`

Single image / video pair metric. Useful for CI / scripted benchmarking.

```bash
nurec_dev bazelisk run //:run -- \
  compute-metrics \
  --pred-path $NUREC_SRC/data/render.png \
  --target-path $NUREC_SRC/data/gt.png \
  --output-path $NUREC_SRC/data/metrics.json \
  --metric psnr
```

Supported `--metric` values: `psnr`, `ssim`, `lpips`, `fid`, `drift`,
`fcs_adaptive`, `ntd`, `d_skew`, `d_kurt`, `perceptual`,
`temporal_coherence`.

> **`cpsnr` is not usable here.** `compute-metrics` accepts it but
> `nurec/metrics/compute_metrics.py` raises
> `NotImplementedError("CPSNR computation is not implemented yet")` for that
> case. CPSNR _does_ work during training/validation via
> `system.test.metrics.cpsnr.enabled` — that is a different code path. `--device` defaults to `cuda`. `--lpips-net-type`
> selects between `alex`, `vgg`, `squeeze`.

### `viewer` and `ply_viewer`

Browser-based viewers for trained USDZs and standalone PLYs.

```bash
RUN_ID=latest                                    # run directory name under out_dir (logger.run_id)

nurec_dev bazelisk run //:run -- \
  viewer \
  --artifact-path "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz" \
  --port 8080
```

Common flags (both):

| Flag     | Default     | Purpose                                    |
| -------- | ----------- | ------------------------------------------ |
| `--host` | `127.0.0.1` | Bind host. Use `0.0.0.0` to expose to LAN. |
| `--port` | `8080`      | Bind port.                                 |

`viewer` requires `--artifact-path <usdz>`. `ply_viewer` requires
`--ply-path <ply>` and accepts `--config <yaml>` to swap the renderer
config.

### `profile-dataloader`

Profiles the dataloader hot path for a given config; produces a CPU/GPU
trace under `<out_dir>/<RUN-ID>`. Run `--help` for the full surface.

### `run-script`

Run an arbitrary Python script inside the container with the NuRec
Python environment available (Bazel-runfiles aware). Useful for
small one-off jobs that import from the `nurec.*` packages.

```bash
nurec_dev bazelisk run //:run -- \
  run-script $NUREC_SRC/scripts/my_script.py
```

## 2. Auxiliary-data CLI — `//apps:nurec_tools`

The public tools target exposes `ncore-aux-data` and `pai-convert`, plus
benchmark helpers. Query `nurec_dev bazelisk run //apps:nurec_tools -- --help`
for the enabled commands. Asset capture is a separate workflow.

### NuRec Auxiliary Data

```bash
NAME=my-clip-name                                # NCore store / manifest base name
CAM1=camera_front_wide_120fov                    # camera sensor id
CAM2=camera_cross_left_120fov                    # second camera sensor id

nurec_dev bazelisk run //apps:nurec_tools -- ncore-aux-data \
  --dataset-path="$NUREC_SRC/dataset/${NAME:?}.json" \
  --output-dir=$NUREC_SRC/output \
  --camera-id="${CAM1:?}" --camera-id="${CAM2:?}" \
  --store-meta --no-seg-logits \
  --lidar-seg-camvis
```

See `references/aux-data.md` for the full flag matrix and a glossary
of each auxiliary-data type.

Asset capture is a separate workflow, not an exported `//apps:nurec_tools`
sub-command. NuRec accepts compatible captured assets via
`export-external-assets`; see [asset-editing.md](asset-editing.md).

---

## 3. Hydra recipe paths shipped in the source checkout

Inspect `configs/` in the released source tree for available recipes. Pick by source dataset:

- **Waymo Open Dataset** → `apps/AV/Waymo/*.yaml` (Waymo-only).
- **NVIDIA Physical AI Autonomous Vehicles (PAI)** →
  the AV Benchmark preset
  `$NUREC_SRC/workflows/av_benchmark/configs/av_benchmark_lidar_7cam.yaml`.
- **PandaSet / NV / Tesla / Alpasim** → matching paths below.

Do **not** use the Waymo recipes for PAI clips — the sensor rig,
validation cameras, and lidar IDs differ.

The most-used recipes for AV reconstruction:

```text
# Waymo Open Dataset (Waymo-only)
apps/AV/Waymo/3dgut_dynamic.yaml                 # canonical Waymo Open Dataset dynamic 3DGUT
apps/AV/Waymo/3dgut_dynamic_mcmc.yaml            # MCMC densification (default since 25.07)
apps/AV/Waymo/3dgut_dynamic_road_semantic.yaml   # adds the road semantic loss
apps/AV/Waymo/3dgut_dynamic_road_semantic_mcmc.yaml
apps/AV/Waymo/3dgut_static.yaml                  # static-scene baseline
apps/AV/Waymo/3dgut_static_mcmc.yaml
apps/AV/Waymo/3dgut_static_calib.yaml            # static + camera calibration optim
apps/AV/Waymo/3dgut_static_road_semantic.yaml

# Physical AI Autonomous Vehicles (PAI) — Hyperion-8.1
workflows/av_benchmark/configs/av_benchmark_lidar_7cam.yaml  # PAI reconstruction recipe (the AV Benchmark preset)

apps/AV/PandaSet/3dgut_dynamic.yaml
apps/AV/PandaSet/3dgut_dynamic_mcmc.yaml
apps/AV/PandaSet/3dgut_dynamic_mcmc_calib.yaml

apps/AV/NV/3dgut_dynamic.yaml
apps/AV/NV/3dgut_dynamic_mcmc.yaml
apps/AV/NV/3dgut_dynamic_road_semantic.yaml
apps/AV/NV/3dgut_dynamic_road_semantic_mcmc.yaml
apps/AV/NV/3dgut_dynamic_temporal_appearance.yaml
apps/AV/NV/3dgut_static.yaml
apps/AV/NV/3dgut_static_calib.yaml
apps/AV/NV/3dgut_static_mcmc.yaml
apps/AV/NV/3dgut_static_road_semantic.yaml
apps/AV/NV/3dgut_static_road_semantic_mcmc.yaml

apps/AV/Tesla/3dgut_static.yaml
apps/AV/Tesla/3dgut_static_road_semantic.yaml
apps/AV/Tesla/3dgut_static_road_semantic_calib.yaml
apps/AV/Tesla/3dgut_static_road_semantic_mcmc.yaml
apps/AV/Tesla/3dgut_static_road_semantic_mcmc_calib.yaml

apps/Alpasim/alpasim_3dgut.yaml
apps/Alpasim/alpasim_3dgut_fullres.yaml
apps/Alpasim/alpasim_3dgut_speed.yaml

```

For command help in this source release, run:

```bash
nurec_dev bazelisk run //:run -- --help
```

Inspect `configs/` for other recipes shipped in this source release.
Treat any other path you may have seen elsewhere as unverified.

## 4. gRPC client protobufs

Generate stubs from the matching source release; follow
[grpc-api.md](grpc-api.md#2-client-protobufs-from-source).

## 5. Help flags everywhere

Every sub-command supports `--help`:

```bash
nurec_dev bazelisk run //:run -- --help
nurec_dev bazelisk run //:run -- export-gaussian-plys --help
nurec_dev bazelisk run //:run -- serve-grpc --help
nurec_dev bazelisk run //:run -- render --help
nurec_dev bazelisk run //:run -- render-grpc --help
nurec_dev bazelisk run //:run -- upgrade-artifact --help
nurec_dev bazelisk run //apps:nurec_tools -- --help
```

Always cross-check the container's own help output when a flag here
seems missing or stale — releases land roughly monthly and the image
is the source of truth.
