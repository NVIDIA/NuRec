<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec Configuration Matrix (Hydra)

Complete [source setup](install.md) before running the commands below.

The NuRec training app is a Hydra application. Every flag below is a
Hydra override appended to the `nurec_dev bazelisk run //:run -- main` command — never a CLI
flag. Required overrides:

- `--config-name` (top-level Hydra; either a recipe path baked into
  the container or a `parsed.yaml` produced by a prior training run).
- `mode=<train|val|trainval>`
- `dataset.path=<NCore JSON inside the container>`
- `out_dir=<output dir inside the container>`

For the source-build invocation syntax (same-path mounts and dependency
image), see [`cli-reference.md`](cli-reference.md) §1 "Training /
validation". The rest of this file enumerates the Hydra-side options
you append after `main`.

## Recipe paths shipped in the source checkout

The container resolves recipes itself; the path you pass to
`--config-name` is interpreted relative to the in-container
`configs/` tree.

### Picking the right recipe by source dataset

| Source dataset                                   | Use this recipe family                                                                                                                       |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Waymo Open Dataset**                           | `apps/AV/Waymo/*.yaml` (see below). **Waymo-only** — bakes in the Waymo sensor rig.                                                          |
| **NVIDIA Physical AI Autonomous Vehicles (PAI)** | The AV Benchmark preset `workflows/av_benchmark/configs/av_benchmark_lidar_7cam.yaml` (see below). **Do not use the Waymo recipes for PAI.** |
| **NVIDIA Hyperion / NV-AV captures (non-PAI)**   | `apps/AV/NV/*.yaml`.                                                                                                                         |
| **PandaSet**                                     | `apps/AV/PandaSet/*.yaml`.                                                                                                                   |
| **Tesla research clips**                         | `apps/AV/Tesla/*.yaml`.                                                                                                                      |
| **Alpasim**                                      | `apps/Alpasim/*.yaml`.                                                                                                                       |

### AV — Waymo (Waymo Open Dataset only)

These recipes target the Waymo Open Dataset sensor rig and
conventions. **Do not pick them for PAI clips, NV/Hyperion captures,
PandaSet, or any non-Waymo source** — the sensor IDs, intrinsics, and
validation cameras will be wrong.

| Recipe                                                | What it sets up                                      |
| ----------------------------------------------------- | ---------------------------------------------------- |
| `apps/AV/Waymo/3dgut_dynamic.yaml`                    | Canonical Waymo Open Dataset dynamic 3DGUT recipe.   |
| `apps/AV/Waymo/3dgut_dynamic_mcmc.yaml`               | Same, with MCMC densification (default since 25.07). |
| `apps/AV/Waymo/3dgut_dynamic_road_semantic.yaml`      | Adds the road-semantic supervision.                  |
| `apps/AV/Waymo/3dgut_dynamic_road_semantic_mcmc.yaml` | Road semantic + MCMC.                                |
| `apps/AV/Waymo/3dgut_static.yaml`                     | Static-scene baseline.                               |
| `apps/AV/Waymo/3dgut_static_calib.yaml`               | Static + camera calibration optim.                   |
| `apps/AV/Waymo/3dgut_static_mcmc.yaml`                | Static + MCMC.                                       |
| `apps/AV/Waymo/3dgut_static_road_semantic.yaml`       | Static + road semantic.                              |

Default Waymo cameras: `camera_front_50fov`,
`camera_front_right_50fov`, `camera_front_left_50fov`. Default LiDAR:
`lidar_top`. (`camera_side_left_50fov` / `camera_side_right_50fov` are
also available.)

### AV — Physical AI Autonomous Vehicles (PAI) / Hyperion-8.1

For the gated NVIDIA dataset
[`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
(after running `pai-to-ncore` to land NCore V4 shards), train with the
AV Benchmark preset shipped in the checkout, **not** any Waymo recipe.

| Recipe                                                        | What it sets up                                                                                                                    |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `workflows/av_benchmark/configs/av_benchmark_lidar_7cam.yaml` | The AV Benchmark training preset: Hyperion-8.1 seven cameras plus `lidar_top_360fov`. Its lidar models are in `sensor/` beside it. |

Standard usage:

1. Pass `--config-name=$NUREC_SRC/workflows/av_benchmark/configs/av_benchmark_lidar_7cam.yaml`
   to the NuRec entrypoint (an absolute path inside the image, whose
   working directory is `$NUREC_SRC`).
2. Add the overrides of the `train` task in
   `workflows/av_benchmark/specs/av_benchmark_7cam.yaml`:
   `dataset.path`, `out_dir`, `mode=train`, `logger.name=dummy`,
   `logger.run_id`, `dataset.n_samples_per_epoch=30000` and the camera
   list as `dataset.camera_ids`, `dataset.train_camera_ids` and
   `dataset.val_camera_ids`.
3. Export with `--config-name=<out_dir>/<run_id>/config/parsed.yaml`
   (again an absolute path inside the image).

The Bash recipe in
[`example-workflows/bash/nurec_workflow_pai.md`](example-workflows/bash/nurec_workflow_pai.md)
covers the full host-side workflow.

### AV — NV (NVIDIA Hyperion / NV-AV stack)

| Recipe                                              | What it sets up                                                   |
| --------------------------------------------------- | ----------------------------------------------------------------- |
| `apps/AV/NV/3dgut_dynamic.yaml`                     | Dynamic 3DGUT.                                                    |
| `apps/AV/NV/3dgut_dynamic_mcmc.yaml`                | Dynamic 3DGUT + MCMC.                                             |
| `apps/AV/NV/3dgut_dynamic_road_semantic.yaml`       | Dynamic + road semantic loss.                                     |
| `apps/AV/NV/3dgut_dynamic_road_semantic_mcmc.yaml`  | Dynamic + road semantic + MCMC.                                   |
| `apps/AV/NV/3dgut_dynamic_temporal_appearance.yaml` | Adds the temporal-appearance head (traffic lights, brake lights). |
| `apps/AV/NV/3dgut_static.yaml`                      | Static baseline.                                                  |
| `apps/AV/NV/3dgut_static_calib.yaml`                | Static + sensor calibration optim.                                |
| `apps/AV/NV/3dgut_static_mcmc.yaml`                 | Static + MCMC.                                                    |
| `apps/AV/NV/3dgut_static_road_semantic.yaml`        | Static + road semantic.                                           |
| `apps/AV/NV/3dgut_static_road_semantic_mcmc.yaml`   | Static + road semantic + MCMC.                                    |

### AV — PandaSet

| Recipe                                           | What it sets up                     |
| ------------------------------------------------ | ----------------------------------- |
| `apps/AV/PandaSet/3dgut_dynamic.yaml`            | Dynamic 3DGUT, PandaSet sensor rig. |
| `apps/AV/PandaSet/3dgut_dynamic_mcmc.yaml`       | + MCMC.                             |
| `apps/AV/PandaSet/3dgut_dynamic_mcmc_calib.yaml` | + MCMC + calibration.               |

PandaSet cameras: `front_camera`, `front_left_camera`,
`front_right_camera`, `left_camera`, `right_camera`, `back_camera`
(with a 260-pixel bottom mask border on `back_camera`). LiDAR:
`main_pandar64`.

### AV — Tesla (research workflow)

| Recipe                                                     | What it sets up         |
| ---------------------------------------------------------- | ----------------------- |
| `apps/AV/Tesla/3dgut_static.yaml`                          | Static.                 |
| `apps/AV/Tesla/3dgut_static_road_semantic.yaml`            | Static + road semantic. |
| `apps/AV/Tesla/3dgut_static_road_semantic_calib.yaml`      | + calibration.          |
| `apps/AV/Tesla/3dgut_static_road_semantic_mcmc.yaml`       | + MCMC.                 |
| `apps/AV/Tesla/3dgut_static_road_semantic_mcmc_calib.yaml` | + MCMC + calibration.   |

### Alpasim

| Recipe                                    | What it sets up          |
| ----------------------------------------- | ------------------------ |
| `apps/Alpasim/alpasim_3dgut.yaml`         | Default Alpasim 3DGUT.   |
| `apps/Alpasim/alpasim_3dgut_fullres.yaml` | Full-resolution variant. |
| `apps/Alpasim/alpasim_3dgut_speed.yaml`   | Speed-tuned variant.     |

### Reusing a parsed config

`$NUREC_SRC/output/<RUN-ID>/config/parsed.yaml` is a snapshot of the
fully-resolved config from a training run. **Re-pass it for
validation, novel-view rendering, and every `export-*` command** —
otherwise the dataset paths and sensor selection won't match what was
trained.

To list any other recipes shipped in the source checkout, run:

```bash
nurec_dev bazelisk run //:run -- --help
```

The image is authoritative on which recipes ship in any given release;
treat undocumented paths as unverified.

## Top-level overrides

| Override                    | Default             | Purpose                                                                                                     |
| --------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------- |
| `mode`                      | `trainval`          | `train`, `val`, or `trainval`.                                                                              |
| `dataset.path`              | (required at train) | NCore JSON inside the container (`$NUREC_SRC/dataset/<NAME>.json`).                                         |
| `out_dir`                   | `???`               | Output root inside the container; bind-mount it.                                                            |
| `logger`                    | `wandb`             | Set `logger=tensorboard` to skip the wandb prompt; `logger=dummy` to disable logging entirely.              |
| `logger.run_id`             | random              | Force a specific `<RUN-ID>` (subdirectory name). Honors `$NUREC_ENV_RUN_ID`.                                |
| `logger.offline` (W&B only) | `false`             | Run W&B in offline mode (skips network calls).                                                              |
| `seed`                      | `42`                | Global RNG seed.                                                                                            |
| `matmul_precision`          | `medium`            | `highest`, `high`, or `medium` for `torch.set_float32_matmul_precision`. Lower = faster on Tensor Cores.    |
| `resume`                    | `null`              | Path or filename to a `.ckpt`. Accepts `last`, `last.ckpt`, `epoch=4`, `epoch=4.ckpt`, or an absolute path. |
| `resume_weights_only`       | `false`             | Load weights only — drops optimizer state.                                                                  |
| `force_validate`            | `false`             | Force a validation pass at the start.                                                                       |
| `verbose`                   | `false`             | Verbose logging.                                                                                            |
| `+log_level`                | `2`                 | `0`=FATAL, `1`=ERROR, `2`=WARN, `3`=INFO, `4`=DEBUG. The `+` adds the key.                                  |

### Trainer block

| Override                                 | Default           | Purpose                                                              |
| ---------------------------------------- | ----------------- | -------------------------------------------------------------------- |
| `trainer.world_size`                     | `1`               | GPUs per node. `0` enables SLURM-style auto-detect.                  |
| `trainer.num_nodes`                      | `1`               | Number of nodes.                                                     |
| `trainer.precision`                      | `32` (AV recipes) | `16`, `16-mixed`, `16-no-cast`, `32`, `bf16`.                        |
| `trainer.max_epochs`                     | `1`               | Training epoch budget.                                               |
| `trainer.check_val_every_n_epoch`        | `1`               | Validate every N epochs.                                             |
| `trainer.log_every_n_steps`              | `50`              | Logger flush interval.                                               |
| `trainer.batch_size_scaling_factor`      | `1.0`             | Scale effective batch size in distributed training.                  |
| `trainer.training_step_scaling_factor`   | `1.0`             | Scale effective step count in distributed training (added in 25.09). |
| `trainer.relative_lr`                    | `true`            | Auto-scale LRs by world size to mimic `world_size=1`.                |
| `trainer.relative_schedule`              | `true`            | Auto-scale LR schedules by world size.                               |
| `trainer.relative_num_workers`           | `true`            | Auto-scale dataloader worker count by world size.                    |
| `trainer.nccl_timeout_minutes`           | `30`              | NCCL collective timeout. Bump for long-running jobs.                 |
| `trainer.detect_anomaly`                 | `false`           | Enable PyTorch Lightning anomaly detection (NaN/Inf).                |
| `trainer.annealing_update_every_n_steps` | `1000`            | Step interval for cosine annealing scheduler updates.                |

### Dataset block (NCore datasets)

| Override                                                                                                             | Default                        | Purpose                                                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dataset.camera_ids`                                                                                                 | recipe default                 | Camera-ID whitelist. Example: `dataset.camera_ids="['camera_front_wide_120fov','camera_cross_left_120fov']"` (no spaces).                                                |
| `dataset.lidar_ids`                                                                                                  | recipe default                 | LiDAR-ID whitelist.                                                                                                                                                      |
| `dataset.train_camera_ids` / `dataset.val_camera_ids`                                                                | `${dataset.camera_ids}`        | Decouple training / validation sensor sets.                                                                                                                              |
| `dataset.train_lidar_ids` / `dataset.val_lidar_ids`                                                                  | `${dataset.lidar_ids}`         | Same for LiDARs.                                                                                                                                                         |
| `dataset.seek_offset_sec` / `dataset.duration_sec`                                                                   | `null`                         | Restrict the time range of an NCore sequence.                                                                                                                            |
| `dataset.poses_component_group` / `intrinsics_component_group` / `masks_component_group` / `cuboids_component_group` | `default`                      | V4 component-group selectors when an NCore shard ships multiple variants of the same data.                                                                               |
| `dataset.lidar_ignore_rows`                                                                                          | `[]`                           | Per-LiDAR rows (laser indices) to drop.                                                                                                                                  |
| `dataset.max_dist_m`                                                                                                 | `150.0`                        | Scene scale for densification.                                                                                                                                           |
| `dataset.n_samples_per_epoch`                                                                                        | `30000`                        | Effective epoch length.                                                                                                                                                  |
| `dataset.n_train_sample_camera_rays`                                                                                 | `6144`                         | Camera rays per training sample.                                                                                                                                         |
| `dataset.n_train_sample_lidar_rays`                                                                                  | `2048`                         | LiDAR rays per training sample.                                                                                                                                          |
| `dataset.n_val_image_subsample`                                                                                      | recipe default                 | Sub-sample validation cameras (Waymo: `2`, NV/Alpasim/Tesla: `4`, PandaSet: `1`).                                                                                        |
| `dataset.val_camera`                                                                                                 | `true`                         | Render camera rays at val/test.                                                                                                                                          |
| `dataset.val_lidar`                                                                                                  | `false`                        | Render LiDAR rays at val/test.                                                                                                                                           |
| `dataset.val_sensor_transl_delta_m`                                                                                  | `null` (`[0,0,0]` effectively) | XYZ metres translation delta for novel views. No spaces inside the brackets.                                                                                             |
| `dataset.val_sensor_rot_delta_deg`                                                                                   | `null` (`[0,0,0]` effectively) | Roll-pitch-yaw degree shifts relative to the rig.                                                                                                                        |
| `dataset.val_camera_frame_start`                                                                                     | `0`                            | Start camera frame index for validation.                                                                                                                                 |
| `dataset.val_camera_frame_step`                                                                                      | `1`                            | Validation camera frame stride. Mutually exclusive with `val_camera_exclude_frame_step`.                                                                                 |
| `dataset.val_camera_exclude_frame_start` / `dataset.val_camera_exclude_frame_step`                                   | `null`                         | Exclude every Nth camera frame from validation.                                                                                                                          |
| `dataset.val_lidar_frame_*` (mirrors of the camera fields)                                                           | analogous                      | LiDAR frame stride / exclusion in val/test.                                                                                                                              |
| `dataset.aux_data`                                                                                                   | `true`                         | Use the auxiliary `<NAME>.aux.*.zarr.itar` shards. Disable for camera-only debugging. (Note: multi-GPU still has known crashes when this is `false`; see release notes.) |
| `dataset.camera_mask_sources`                                                                                        | `[dataset, aux]`               | Priority order for camera-mask sources.                                                                                                                                  |
| `dataset.n_camera_mask_dilation_iterations`                                                                          | `30`                           | Dilation iterations applied to camera masks.                                                                                                                             |
| `dataset.camera_mask_overrides`                                                                                      | `{}`                           | Map camera ID → host PNG path to override the per-camera mask (see Tesla mixin). Mount the PNGs into the container too.                                                  |
| `dataset.camera_mask_border`                                                                                         | `null`                         | `[top, right, bottom, left]` pixel border per camera (e.g. PandaSet's `back_camera: [0, 0, 260, 0]`).                                                                    |
| `dataset.frame_generic_data_pose_overwrite`                                                                          | `false`                        | Overwrite per-frame sensor-world poses from generic-data fields (only for separately optimized datasets).                                                                |
| `dataset.open_consolidated`                                                                                          | `true`                         | Open shards with consolidated metadata.                                                                                                                                  |
| `dataset.jpeg_backend_cpu`                                                                                           | `simplejpeg`                   | `PIL` or `simplejpeg` for CPU JPEG decoding.                                                                                                                             |
| `dataset.simplejpeg_fastdct` / `dataset.simplejpeg_fastupsample`                                                     | `false`                        | 4–5% speedup at minor quality cost.                                                                                                                                      |
| `dataset.camera_max_fov_deg`                                                                                         | `190.0`                        | Cap effective FOV of FTheta / OpenCVFisheye cameras.                                                                                                                     |
| `dataset.lidar_model_parameter_cwccw_fallback`                                                                       | `false`                        | Backwards-compat fallback for old NCore lidar spin direction.                                                                                                            |
| `dataset.lidar_model_parameter_nominal_values`                                                                       | `{}`                           | Map LiDAR ID → nominal model name (`HESAI-Pandar128`, `HESAI-AT128`).                                                                                                    |
| `dataset.use_real_lidar_rays`                                                                                        | `false`                        | Compute LiDAR rays from the actual point clouds.                                                                                                                         |

### Cuboid track classification (`dataset.cuboid_tracks_params.*`)

| Override                        | Default  | Purpose                                                                                                                                                                                       |
| ------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `track_min_distance_m`          | `1.0`    | Path-length threshold to classify dynamic.                                                                                                                                                    |
| `track_min_displacement_m`      | `1.0`    | Net displacement threshold.                                                                                                                                                                   |
| `track_speed_reduction_op`      | `median` | `median` or `max` reduction across pose-inferred speeds.                                                                                                                                      |
| `track_min_speed_ms`            | `0.1`    | Speed threshold (m/s).                                                                                                                                                                        |
| `track_min_centroid_rig_dist_m` | `3.0`    | Min centroid-to-rig distance to skip self-classifications.                                                                                                                                    |
| `track_extrapolate`             | `true`   | Extrapolate dynamic-track labels at start/end.                                                                                                                                                |
| `track_label_sources`           | `null`   | Allowed label sources from `[AUTOLABEL, EXTERNAL, GT_SYNTHETIC, GT_ANNOTATION]` (with optional `@<source-version>` suffix). Required when the dataset ships labels from more than one source. |
| `use_displacement_and_distance` | `true`   | Use displacement _and_ distance thresholds.                                                                                                                                                   |
| `camera_visibility`             | `false`  | Only use camera-visible time intervals for dynamic classification.                                                                                                                            |

### Valid measurements (`dataset.valid_measurements_method`)

| Value                           | Meaning                                                                     |
| ------------------------------- | --------------------------------------------------------------------------- |
| `EGO`                           | Mask-out ego pixels only (recipe default for Alpasim and AV/Waymo dynamic). |
| `EGO_CUBOIDTRACKS`              | Ego mask + per-cuboid mask.                                                 |
| `EGO_SCENEFLOW`                 | Ego mask + scene-flow mask.                                                 |
| `EGO_CUBOIDTRACKS_SCENEFLOW`    | All of the above.                                                           |
| `EGO_CUBOIDTRACKS_TRAFFICLIGHT` | Ego + cuboid + traffic-light dilation.                                      |
| `EGO_FRAMEMASKS`                | Ego + per-frame masks (Tesla recipe default).                               |

Tunables for each variant live under
`dataset.valid_lidarpoints_cuboid_track_params`,
`dataset.valid_pixels_cuboid_track_params`,
`dataset.valid_pixels_scene_flow_params`,
`dataset.valid_pixels_traffic_light_params`,
`dataset.valid_pixels_frame_mask_params` — see the full
`parsed.yaml` for defaults.

### LiDAR dynamic-point handling

| Override                                                    | Default                                                        | Purpose                                                                                            |
| ----------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `dataset.lidar_dynamic_points.method`                       | `dynamic_tracks`                                               | `dynamic_tracks` (re-classify at load time) or `dynamic_flag` (use NCore-internal `dynamic_flag`). |
| `dataset.camera_point_cloud_ignore_classes`                 | `[egocar, sky]`                                                | Classes filtered from the camera-derived point cloud.                                              |
| `dataset.camera_point_cloud_dynamic_classes`                | `[person, rider, car, truck, bus, train, motorcycle, bicycle]` | Classes treated as dynamic.                                                                        |
| `dataset.generate_static_rigid_cuboid_tracks.rigid_classes` | `[traffic light]`                                              | Classes treated as rigid.                                                                          |
| `dataset.generate_static_rigid_cuboid_tracks.enabled`       | `false`                                                        | Auto-generate cuboid tracks for static rigid classes.                                              |

### Model block — gaussian variants

The recipe `defaults` choose the model:

```yaml
defaults:
  - /model: gaussians/composite/dynamic_gaussians # or static_gaussians
  - /model/calib: skip # or: calib (enables calibration optim)
  - /model/background: sky_mlp # or: sky_env_map, extra_signal_sky_mlp
  - /harmonizer: harmonizer # defaults to the harmonizer_nontemporal variant
  - /system: gaussians
  - /loss: gaussians_av # or: gaussians_av_road_semantic
  # plus two post-processing entries using Hydra's package-override syntax
  # (see note below)
```

Recipes also register two post-processing variants via Hydra's
package-override syntax `group<AT>target` (where `<AT>` is a literal `@`
character). The two entries appended to `defaults:` are:

- `/model/post_processing<AT>model.post_processing.b` → `bilateral_grid_per_camera`
- `/model/post_processing<AT>model.post_processing.c` → `bilateral_grid_per_frame`

Replace `<AT>` with the `@` character when copying into a recipe.

Notable model overrides:

| Override                                                        | Default                | Purpose                                                    |
| --------------------------------------------------------------- | ---------------------- | ---------------------------------------------------------- |
| `model.background.name`                                         | recipe default         | `sky-mlp`, `sky-env-map`, or `extra-signal-sky-mlp`.       |
| `model.background.saturate_radiance`                            | `true`                 | Disable to allow over-bright sky regions (added in 25.09). |
| `model.calib.enabled`                                           | varies                 | Toggle the camera-calibration module (added in 25.09).     |
| `model.appearance_embedding.*`                                  | model default          | Per-frame appearance latents (added in 25.09).             |
| `model.layers.background.initialization.num_point_cloud_points` | `2_000_000` (PandaSet) | Number of background Gaussians at init.                    |
| `model.layers.background.optimizer.params.positions.args.lr`    | model default          | Background-position learning rate.                         |

### Harmonizer (Fixer) configuration

The Hydra group is `harmonizer`. Select
`harmonizer=public_harmonizer_nontemporal` for the public Hugging Face
`harmonizer_nontemporal.pt` checkpoint at `[576, 1024]`. Disable with
`harmonizer=disabled`. See [harmonizer.md](harmonizer.md) for the pinned URL
and the separate override syntax for standalone resolved benchmark configs.

> Harmonizer is **Built on NVIDIA Cosmos**; its weights are licensed under
> the NVIDIA Open Model License and are not redistributed by NuRec. Before
> use, accept the gated `nvidia/Cosmos-Predict2-0.6B-Text2Image` terms on
> Hugging Face and set `HF_TOKEN` (see
> [NuRec Harmonizer guide](https://docs.nvidia.com/nurec/nurec/harmonizer.html)).
> Hosted NuRec links track the latest published docs and may differ from this
> checkout.

> The group was named `difix` before the rename. The `--difix-*` **CLI
> flags** still work as deprecated aliases, but the Hydra group and its
> config keys are `harmonizer.*` now. The legacy Stable-Diffusion variant
> no longer ships.

| Override                                                                                   | Default                     | Purpose                                                                                            |
| ------------------------------------------------------------------------------------------ | --------------------------- | -------------------------------------------------------------------------------------------------- |
| `harmonizer.name`                                                                          | recipe default              | `harmonizer-nontemporal` or `disabled`.                                                            |
| `harmonizer.model_url`                                                                     | selected config URL         | Override the checkpoint URL.                                                                       |
| `harmonizer.model_filename`                                                                | matches `model_url`         | Local filename inside the cache.                                                                   |
| `harmonizer.model_resolution`                                                              | `(576, 1024)`               | (H, W) at which Harmonizer runs.                                                                   |
| `harmonizer.cache_dir`                                                                     | `~/.cache/nurec/harmonizer` | Local checkpoint cache.                                                                            |
| `harmonizer.inference.enabled`                                                             | `false`                     | Apply the Harmonizer at inference time.                                                            |
| `harmonizer.inference.use_color_transfer`                                                  | `true`                      | Re-apply input color statistics on Harmonizer output.                                              |
| `harmonizer.training.enabled`                                                              | `false`                     | Fold the Harmonizer into training (used by `*_harmonizer*.yaml` recipes and Alpasim distillation). |
| `harmonizer.training.start_step`                                                           | `20000`                     | Step at which training-time Harmonizer kicks in.                                                   |
| `harmonizer.training.p_scheduler.p_init` / `p_final` / `milestones` / `gamma`              | per-recipe                  | Probability schedule for Harmonizer application during training.                                   |
| `harmonizer.training.novel_view_poses`                                                     | `[+3m, -3m lateral]`        | List of `{translation, rotation}` deltas that Harmonizer supervises during training.               |
| `harmonizer.training.shuffle_novel_views`                                                  | `false`                     | Shuffle the novel-view list per step.                                                              |
| `harmonizer.training.num_workers`                                                          | `4`                         | Dataloader worker count for the novel-view loader.                                                 |
| `harmonizer.training.log_images` / `log_buffer_images` / `log_every_n_steps` / `debug_dir` | off                         | Per-step debug image logging.                                                                      |

### Loss block

The default loss bundle for AV is `gaussians_av`. Common overrides:

| Override                                            | Default                             | Purpose                                                                                              |
| --------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `loss.rgb.lambda_`                                  | `0.8`                               | L1 RGB loss weight.                                                                                  |
| `loss.ssim.lambda_`                                 | `0.2`                               | SSIM loss weight.                                                                                    |
| `loss.background.lambda_`                           | `0.05` (when sky head is on)        | Background MSE.                                                                                      |
| `loss.lidar.lambda_`                                | `0.0003`                            | LiDAR ray-distance L1.                                                                               |
| `loss.bilateral_grid_per_camera_tv.lambda_`         | `0.001`                             | Spatial TV regularization.                                                                           |
| `loss.bilateral_grid_per_frame_spatial_tv.lambda_`  | `0.001`                             | Per-frame spatial TV.                                                                                |
| `loss.bilateral_grid_per_frame_temporal_tv.lambda_` | `1.0`                               | Per-frame temporal TV.                                                                               |
| `loss.semantic.lambda_`                             | `0.0` (suggested `0.001` when used) | Semantic cross-entropy.                                                                              |
| `loss.intensity.lambda_`                            | `0.0` (suggested `1.0` when used)   | LiDAR intensity loss.                                                                                |
| `loss.raydrop.lambda_`                              | `0.0` (suggested `1.0`)             | LiDAR ray-drop loss.                                                                                 |
| `loss.normal.lambda_`                               | `0.0` (suggested `0.1`)             | Surface-normal loss.                                                                                 |
| `loss.gaussian_flatten.lambda_`                     | `0.0` (suggested `0.005`)           | Gaussian flatness regularization.                                                                    |
| `loss.background_in_track_gaussian.lambda_`         | `0.1`                               | Suppress background Gaussians inside dynamic tracks.                                                 |
| `loss.deform_smoothness.lambda_`                    | `0.01`                              | Deformation smoothness regularization.                                                               |
| `loss.out_of_bound.lambda_`                         | `1`                                 | Out-of-bound penalty.                                                                                |
| `loss.ppisp.lambda_`                                | `1.0`                               | Top-level PPISP loss; sub-weights live in `loss.ppisp.lambdas.*` (exposure, vignetting, color, CRF). |

### System / test block

| Override                                                          | Default                                             | Purpose                                                                                                                                           |
| ----------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `system.test.video_fps`                                           | `30` (Waymo/PandaSet/Alpasim mixins set it to `10`) | FPS for validation MP4s.                                                                                                                          |
| `system.test.metrics.cpsnr.enabled`                               | `true`                                              | Class-conditional PSNR.                                                                                                                           |
| `system.test.metrics.ssim.enabled`                                | `false`                                             | SSIM at validation time.                                                                                                                          |
| `system.test.metrics.lpips.enabled`                               | `false`                                             | LPIPS at validation time.                                                                                                                         |
| `system.test.lidar.raydrop_threshold`                             | `0.5`                                               | Drop LiDAR rays with `raydrop > T`.                                                                                                               |
| `system.test.lidar.ROI.min_m` / `max_m`                           | `null`                                              | Restrict LiDAR evaluation to a circular ROI around the rig.                                                                                       |
| `system.test.lidar.save_renders.enabled`                          | `false`                                             | Render LiDAR PLYs during val.                                                                                                                     |
| `system.test.lidar.save_filtered_pc.enabled` / `filter_threshold` | off                                                 | Save a filtered point cloud (higher threshold = fewer floaters).                                                                                  |
| `system.test.background_removal.track_ids`                        | `[]`                                                | Drop background Gaussians inside the named tracks.                                                                                                |
| `system.test.val_render_selected_nodes`                           | `{}`                                                | Per-layer-name dict of "render only this subset" overlays (e.g. `{background_only: [background], no_deformables: [background, dynamic_rigids]}`). |
| `system.test.track_orbit.*` / `track_rotation.*` / `track_ply.*`  | off                                                 | Per-actor diagnostic renders.                                                                                                                     |
| `system.test.save_extra_signals`                                  | `false`                                             | Save per-frame extra signals (semantic, intensity, …).                                                                                            |
| `system.test.save_videos`                                         | `true`                                              | Encode validation MP4s.                                                                                                                           |
| `system.test.frame_naming`                                        | `batch-index`                                       | Naming scheme for validation frames.                                                                                                              |
| `system.collect_garbage_mem_usage`                                | `null`                                              | Memory threshold (MiB) at which to call `gc.collect()`.                                                                                           |
| `system.collect_garbage_check_interval`                           | `10000`                                             | Steps between GC checks (ignored when threshold is `null`).                                                                                       |
| `system.max_split_size_mb`                                        | `null`                                              | Append `max_split_size_mb:N` to `PYTORCH_CUDA_ALLOC_CONF`.                                                                                        |

### Checkpointing

| Override                                                   | Default      | Purpose                                                                |
| ---------------------------------------------------------- | ------------ | ---------------------------------------------------------------------- |
| `checkpoint.every_n_train_steps`                           | `1000`       | Step interval between checkpoints (added as a config option in 25.09). |
| `checkpoint.save_top_k`                                    | `1`          | Keep best-K (`-1` = keep all).                                         |
| `checkpoint.monitor`                                       | `train/psnr` | Metric for `save_top_k`.                                               |
| `checkpoint.mode`                                          | `max`        | `max` or `min` for the monitored metric.                               |
| `checkpoint.save_on_train_epoch_end`                       | `true`       | Save at end of every train epoch.                                      |
| `checkpoint.save_on_preemption`                            | `false`      | Save when SLURM preempts the job.                                      |
| `checkpoint.artifact.enabled`                              | `false`      | Bundle a USDZ artifact on every save. **Required for `serve-grpc`.**   |
| `checkpoint.artifact.checkpoint.enabled`                   | `true`       | Embed `.ckpt` inside the USDZ.                                         |
| `checkpoint.artifact.parsed_config.enabled`                | `true`       | Embed the parsed config.                                               |
| `checkpoint.artifact.rig_trajectories.enabled`             | `true`       | Embed rig trajectories.                                                |
| `checkpoint.artifact.rig_trajectories.add_default_cameras` | `false`      | Add default camera calibrations.                                       |
| `checkpoint.artifact.sequence_tracks.enabled`              | `true`       | Embed cuboid/actor tracks.                                             |
| `checkpoint.artifact.sequence_tracks.controllable_only`    | `false`      | Restrict to controllable actors.                                       |
| `checkpoint.artifact.mesh.generic.enabled`                 | `false`      | Embed a Poisson mesh inside the USDZ.                                  |
| `checkpoint.artifact.mesh.ground.enabled`                  | `false`      | Embed a ground mesh inside the USDZ.                                   |

### Sensor block (LiDAR models)

`/sensor: lidar_model` exposes nominal HESAI parameters:

```yaml
lidar_models:
  HESAI_Pandar128: ${load_json:configs/sensor/hesai_pandar128.json}
  HESAI_AT128: ${load_json:configs/sensor/hesai_at128.json}
```

Use `dataset.lidar_model_parameter_nominal_values` to map a logical
LiDAR ID to one of these nominal models when your shard ships
incomplete or corrupted parameters:

```bash
... 'dataset.lidar_model_parameter_nominal_values={lidar_top:HESAI-Pandar128}'
```

### Profiling

| Override                                                                  | Default | Purpose                                                      |
| ------------------------------------------------------------------------- | ------- | ------------------------------------------------------------ |
| `profiling.enabled`                                                       | `false` | Toggle the PyTorch profiler.                                 |
| `profiling.params.start_step`                                             | unset   | Step at which to start the CUDA profiler (added in 25.09).   |
| `profiling.params.num_steps`                                              | unset   | Number of steps to profile (added in 25.09).                 |
| `scopedtimer.enabled`                                                     | `false` | Enable the `ScopedTimer` instrumentation.                    |
| `scopedtimer.verbosity`                                                   | `NONE`  | `NONE`, `SUMMARY`, `BASIC`, `DETAILS`.                       |
| `scopedtimer.profiling_backend`                                           | `NVTX`  | `NONE`, `TRACY`, `NVTX`.                                     |
| `scopedtimer.synchronize`                                                 | `false` | `cudaDeviceSynchronize` per scope.                           |
| `scopedtimer.logfile`                                                     | `null`  | Optional output log path (relative to `<out_dir>/<RUN-ID>`). |
| `scopedtimer.emit_start_step` / `emit_num_steps` / `emit_repeat_interval` | unset   | Step windows for backend emission.                           |

### Viewer (in-trainer)

| Override                                                         | Default     | Purpose                                        |
| ---------------------------------------------------------------- | ----------- | ---------------------------------------------- |
| `viewer.enabled`                                                 | `false`     | Embed the live viewer in the training process. |
| `viewer.host`                                                    | `127.0.0.1` | Bind host. Use a routable IP to share.         |
| `viewer.port`                                                    | `8080`      | Bind port.                                     |
| `viewer.remain_after_trainer_loop`                               | `false`     | Keep serving after the trainer exits.          |
| `viewer.ray_subsampling_step.high_resolution` / `low_resolution` | `4` / `16`  | Per-mode ray subsampling factors.              |
| `viewer.n_lidar_frames_displayed`                                | `15`        | Number of LiDAR frames visible.                |

## Multi-GPU and SLURM

NuRec binds to the first visible GPU by default. Two ways to scale:

```bash
# Explicit: 4 GPUs on 1 node.
... trainer.world_size=4 trainer.num_nodes=1

# Auto-detect SLURM allocation (only inside a SLURM job).
... trainer.world_size=0 trainer.num_nodes=0
```

Restrict the visible GPUs via the standard CUDA env var:

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4  # forwarded by nurec_dev
```

The `trainer.world_size=N` flag pulls the _first N_ IDs from
`CUDA_VISIBLE_DEVICES`, in order.

> Doc snippet: "If you have 6 GPUs and you specify
> `CUDA_VISIBLE_DEVICES=1,2,3,4,5,0` and pass `trainer.world_size=4`,
> NuRec uses GPUs with IDs `1,2,3,4`."

> **Practical note.** Reconstruction quality plateaus past ~6 GPUs.
> `dataset.aux_data=false` is known to crash multi-GPU training (carried
> over from 25.06–25.07). Multi-GPU + edge cases are listed in
> `references/troubleshooting` of the release-notes section.

## Novel-view synthesis

Re-run validation against `parsed.yaml` with translation / rotation
overrides:

```bash
RUN_ID=latest   # run subdirectory under $NUREC_SRC/output

nurec_dev bazelisk run //:run -- \
  main \
  --config-name=$NUREC_SRC/output/"${RUN_ID:?}"/config/parsed.yaml \
  mode=val \
  resume=$NUREC_SRC/output/"${RUN_ID:?}"/checkpoints/last.ckpt \
  out_dir=$NUREC_SRC/output \
  dataset.val_sensor_transl_delta_m="[0,2,0]" \
  dataset.val_sensor_rot_delta_deg="[0,0,5]"
```

Both list overrides must contain no spaces; quote the value to keep
the shell from splitting it.

For ad-hoc rendering without re-running validation, use the
`render` sub-command — see `references/cli-reference.md`. It
supports `--rig-translation-offset` / `--rig-rotation-offset` and an
externally authored `--custom-rig-trajectory`.

## Camera subset

```bash
... dataset.camera_ids="['camera_front_wide_120fov','camera_cross_left_120fov']"
```

Camera IDs come from the NCore JSON (`<NAME>.json`); the auxiliary-data
tool emits the IDs to stdout when `--store-meta` is set.

## Logging

- `logger=wandb` (default) prompts interactively the first time. Pick
  option `3` to skip, or pass `logger=tensorboard`.
- `logger=tensorboard` writes to `<out_dir>/<RUN-ID>/logs`.
- `logger=dummy` disables logging entirely.
- `logger.run_id=<id>` makes `<RUN-ID>` deterministic; otherwise a
  random hash is generated.
- `+log_level=4` enables DEBUG output across the training pipeline.
- Per-run logs land in `<out_dir>/<RUN-ID>/logs/`.

## Resuming a training run

| Resume value                    | Meaning                                                               |
| ------------------------------- | --------------------------------------------------------------------- |
| `resume=last`                   | Latest snapshot in `<run>/checkpoints/`.                              |
| `resume=last.ckpt`              | Same as `last`.                                                       |
| `resume="epoch=4"`              | Specific epoch. Quote to prevent shell from interpreting the `=`.     |
| `resume="epoch=4.ckpt"`         | Same.                                                                 |
| `resume=/abs/path/to/last.ckpt` | Explicit checkpoint path (does not need `<run>/checkpoints/` prefix). |
| `resume_weights_only=true`      | Load weights only — drops optimizer/scheduler state.                  |

Note: `out_dir` cannot change between the original and resumed runs.

## Environment variables

| Var                       | Effect                                                                                                                                                                                                                            |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CUDA_VISIBLE_DEVICES`    | Standard CUDA GPU mask. `trainer.world_size=N` pulls the first N IDs in order.                                                                                                                                                    |
| `NUREC_ENV_RUN_ID`        | Forces a specific `<RUN-ID>` (subdirectory name). Used by SLURM/cluster wrappers to keep outputs co-located. **Must be unset when running a `wandb` sweep agent**, otherwise every sweep run inherits the same ID.                |
| `CUDA_SYNC_DEBUG=1`       | Enables `torch.cuda.set_sync_debug_mode("warn")` to surface implicit GPU sync points that hurt training perf. Supported by NuRec training (`mode=train                                                             \| trainval`). |
| `PYTORCH_CUDA_ALLOC_CONF` | Standard PyTorch CUDA allocator knobs. NuRec appends `max_split_size_mb:N` when `system.max_split_size_mb=N` is set in the config.                                                                                                |
| `WANDB_API_KEY`           | W&B auth (alternative to `wandb login`).                                                                                                                                                                                          |
| `WANDB_MODE=disabled`     | Skip W&B even if `logger=wandb`.                                                                                                                                                                                                  |
| `HF_TOKEN`                | Used when fetching gated HuggingFace assets (e.g. PhysicalAI dataset, Harmonizer weights).                                                                                                                                        |

## Artifact layout (`<out_dir>/<RUN-ID>/`)

```text
config/parsed.yaml               # resolved Hydra config
checkpoints/{epoch=*.ckpt, last.ckpt}
val/
    metrics.yaml                 # test/psnr, test/ssim, test/lpips
    *.mp4                        # depth.mp4, opacity.mp4, segmentation.mp4
    frame_*/rgb.png, depth.png, opacity.png, segmentation.png
artifacts/last.usdz                # renderable scene (training, when checkpoint.artifact.enabled=true)
logs/                            # tensorboard / wandb sidecar logs
```

Pass `<out_dir>/<RUN-ID>/config/parsed.yaml` to `mode=val`,
`render`, `render-grpc`, `serve-grpc`, `export-gaussian-plys`,
`export-sequence-tracks`, `export-ncore-tracks`,
`export-usdz-artifact`, `gaussian-statistics`, and `export-depth`.

## Recipe override patterns

### Switch the renderer in training

```bash

```

### Swap the background head

```bash
... model/background=sky_env_map                # or sky_mlp / extra_signal_sky_mlp
```

### Disable the Harmonizer entirely

```bash
... harmonizer=disabled
```

### Force a specific variant + training-time application

```bash
... harmonizer=public_harmonizer_nontemporal harmonizer.training.enabled=true harmonizer.training.start_step=10000
```

### Save USDZ + tracks every checkpoint (so `serve-grpc` can find them)

```bash
... checkpoint.artifact.enabled=true \
    checkpoint.artifact.checkpoint.enabled=true \
    checkpoint.artifact.rig_trajectories.enabled=true \
    checkpoint.artifact.sequence_tracks.enabled=true
```

### Reduce GPU memory pressure

```bash
... model.layers.background.initialization.num_point_cloud_points=500000  # fewer bg gaussians
... dataset.n_train_sample_camera_rays=3072                                # fewer rays/sample
... trainer.precision=16-mixed                                             # mixed precision
... +system.max_split_size_mb=512                                          # PyTorch alloc tuning
```

### Camera-only debugging (no aux data, no LiDAR)

```bash
... dataset.aux_data=false dataset.lidar_ids='[]'
```

> Multi-GPU + `dataset.aux_data=false` is a known crash; stay
> single-GPU when debugging this combination.
