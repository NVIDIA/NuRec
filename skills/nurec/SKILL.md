---
name: nurec
description: >-
  Use to drive NVIDIA Omniverse NuRec / Neural Reconstruction
  Engine from released source in a development image built on the matching
  GitHub-hosted nurec_dev_deps image — train
  3DGUT Gaussian reconstructions from NCore clips, generate aux
  data, adapt an existing USDZ to an augmented target-vehicle rig
  (carline adaptation), render frames or LiDAR sweeps (local or warm
  `serve-grpc`), export PLY/depth/mesh/USDZ, edit actors, and evaluate
  metrics. Do NOT use for per-object asset capture (use
  a separate capture workflow) or unsupported sensor-to-NCore conversion.
version: "0.3.0"
tools:
  - Shell
  - Read
  - Write
license: Apache-2.0
compatibility: >-
  Linux x86_64, 1+ NVIDIA GPU (Ampere A100/A10/A40/RTX A6000, Ada
  L20/L40/L40S, Hopper H100/H20, or Blackwell RTX Pro 6000D) with
  CUDA 13 and >= 24 GB VRAM (48+ GB recommended); driver R580+ minimum.
  Docker >= 23.0.1 with Buildx + NVIDIA Container Toolkit >= 1.13.5; a matching
  released source checkout, development-dependencies image and built development image.
dependencies:
  - bash
  - docker
  - python3
metadata:
  author: NVIDIA NRS <nurec-skills@nvidia.com>
  tags:
    - nurec
    - autonomous-vehicles
    - neural-reconstruction
    - rendering
    - container
  product_page: https://www.nvidia.com/en-us/omniverse/
  source: https://github.com/NVIDIA/NuRec
  dev_deps_image: ghcr.io/nvidia/nurec/nurec_dev_deps
  hf_datasets: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
  hf_fixer_model: https://huggingface.co/nvidia/Harmonizer
  release_date: "2026-04-30"
  time-estimate: "2h"
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Omniverse NuRec (Neural Reconstruction Engine)

## Purpose

Drive the public NVIDIA Omniverse NuRec / Neural Reconstruction
Engine source targets (`//:run`, `//apps:nurec_tools`) to train a 3DGUT Gaussian
reconstruction from an NCore V4 camera+LiDAR clip, render novel
views (locally or via gRPC), generate aux data, export
PLY/depth/mesh/ego-mask/tracks, package Asset Harvester output
into a USDZ, and evaluate rendering metrics.

This skill carries the host-side toolkit around the NuRec CLI: source
setup, local render recipes,
MP4 encoding, warm `serve-grpc` boot/teardown scripts, a thin
Python gRPC client for repeated RGB renders, bundled rig JSONs,
scene-specific trajectory generation, and Bash workflow examples.

## When to Use / When NOT to Use

**Use this skill when** the user has an NCore V4 clip (or a USDZ +
NuRec artifact pair) on a Linux x86_64 host with an NVIDIA GPU, and wants to train, render, generate aux data,
export artifacts, insert/remove actors, run the gRPC server, or
evaluate metrics. Concrete triggers:

- Train a multi-camera + LiDAR AV clip into a renderable USDZ
  scene with 3DGUT Gaussians. (3DGRT training has been removed;
  `export-gaussian-plys --format _3DGRT` still exports in that format.)
- Generate NuRec auxiliary data (seg, depth, ego mask, DINOv2,
  LiDAR-seg visibility) using `//apps:nurec_tools`.
- Render frames locally (no server) along the training rig or a
  custom rig + offsets.
- Render novel views via the sensorsim gRPC API (CARLA, Isaac
  Sim, AlpaSim, custom simulator), optionally with Difix
  artifact-removal.
- Render LiDAR sweeps via `render-grpc --lidar`.
- Export PLY / ego masks / depth / Poisson mesh / ground mesh /
  point clouds / cuboid tracks / NCore tracks / custom rig
  trajectories.
- Insert / remove / replace 3D actors with `export-external-assets`
  - `render-grpc --edit-assets`.
- Render the gated HF dataset
  `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`.
- Upgrade an old USDZ once (`upgrade-artifact`).
- Inspect / evaluate (`export-parsed-config`, `gaussian-statistics`,
  `eval-rendering-metrics`, `compute-metrics`,
  `eval-ground-mesh`).
- Browse a USDZ or PLY in the in-container viewer.

**Do NOT use this skill when:**

- The user needs unsupported raw-sensor conversion. Use `nurec-onboard` when
  available to write and validate a converter for a new dataset. For raw Physical AI AV,
  use the shipped benchmark workflow or `nurec_dev bazelisk run //apps:nurec_tools -- pai-convert`;
  for other formats verify a public NCore converter is available.
- The user wants per-object 3D asset extraction from sparse views
  (use a separately available capture workflow; NuRec only _consumes_ AH outputs via
  `export-external-assets`).
- The user only needs to clean up already-rendered frames using
  the standalone Cosmos-based Fixer (follow the public Harmonizer repository). NuRec's
  inline `--enable-harmonizer` flag is still on this skill's surface
  (`--enable-difix` is a deprecated alias),
  see `references/harmonizer.md` for public weights and standalone guidance.

## Inputs

- **dataset_dir** — host directory holding the NCore shards
  (`<NAME>.zarr.itar`, `<NAME>.json`, and any pre-generated
  `<NAME>.aux.*.zarr` auxiliary shards). Required.
- **dataset_name** — basename of the NCore dataset (the part
  before `.zarr.itar`). Required.
- **output_dir** — host directory NuRec will fill with checkpoints,
  parsed config, metrics, videos, and USDZ artifacts. Required.
- **camera_ids / lidar_ids** — sensor IDs from the NCore JSON to
  include. Default: all sensors per recipe.
- **config_name** — Hydra config path resolved inside the
  container. Pick by source dataset:
  - **Waymo Open Dataset** →
    `apps/AV/Waymo/3dgut_dynamic.yaml` (and its
    `_mcmc` / `_road_semantic` / `_static` siblings). These are
    Waymo-only — they bake in the Waymo sensor rig and conventions.
  - **NVIDIA Physical AI Autonomous Vehicles (PAI)** →
    the AV Benchmark preset shipped in the checkout,
    `--config-name=$NUREC_SRC/workflows/av_benchmark/configs/av_benchmark_lidar_7cam.yaml`,
    with the `train` task arguments of
    `workflows/av_benchmark/specs/av_benchmark_7cam.yaml` (see
    [`example-workflows/bash/nurec.sh`](references/example-workflows/bash/nurec.sh)).
  - **PandaSet / NVIDIA AV (NV) / Tesla / Alpasim** → see the
    matching `apps/AV/{PandaSet,NV,Tesla}/…` or
    `apps/Alpasim/…` recipes in
    `references/configuration.md`.
  - Do **not** use the Waymo recipes for PAI clips — the
    sensor rig, validation cameras, and lidar IDs differ.
- **mode** — `train`, `val`, or `trainval`. Default: `trainval`.
- **NUREC_SRC / NUREC_DEV_DEPS_IMAGE / NUREC_IMAGE** — released checkout,
  matching published dependency image, and development image built on that base.
  See [setup](references/install.md).
- **HF_TOKEN** — only when the selected dataset/model requires authentication;
  accept its terms and set the token locally, never in chat.

## Instructions

1. **Set up source and dependencies.** Follow
   [references/install.md](references/install.md). Set `NUREC_SRC` and the
   release-matched `NUREC_DEV_DEPS_IMAGE`, then build `NUREC_IMAGE` with
   `deps/docker/Dockerfile-dev.build` and `--build-arg BASE_IMAGE`. Source
   `"$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"` in each tool shell; every application command uses
   this built development image.
2. **Validate prerequisites and build.** Run
   `python3 "$NUREC_SRC/skills/nurec/scripts/validate_setup.py" --source "$NUREC_SRC"`.
   Resolve failures, then `nurec_dev bazelisk build //:run //apps:nurec_tools`.
   Run application commands with `nurec_dev bazelisk run //:run -- <command>`;
   auxiliary data uses `nurec_dev bazelisk run //apps:nurec_tools -- ncore-aux-data`.
   Keep inputs/outputs under the checkout or add same-path extra mounts.
3. **Confirm input layout.** The dataset directory must contain
   `<NAME>.zarr.itar`, `<NAME>.json`, and any `<NAME>.aux.*.zarr`
   shards. If the NCore data is fresh, generate auxiliary data
   first — see `references/aux-data.md`.
4. **Train / validate the reconstruction.** Run the train recipe
   in [`references/cookbook.md`](references/cookbook.md) with the
   chosen Hydra `--config-name`, `mode`, `dataset.path`, and
   `out_dir`. For multi-GPU append `trainer.world_size=<N>
trainer.num_nodes=<M>` (see Workflow D). Set
   `checkpoint.artifact.enabled=true` if you intend to render or
   serve the result.
5. **Export downstream artifacts.** Use export sub-commands
   (`export-gaussian-plys`, `export-mesh`, `export-ground-mesh`,
   `export-ego-mask`, `export-depth`, `export-sequence-tracks`,
   `export-ncore-tracks`, …) — full surface in
   `references/cli-reference.md`.
6. **Render novel views — pick the backend.**
   - **Local CLI** — `nurec_dev bazelisk run //:run -- render --artifact-path <usdz>` writes
     frames on disk along the training trajectory, with optional
     rig offsets or `--custom-rig-trajectory`. No gRPC server.
     See `references/local-render.md`.
   - **Warm RGB service** — boot `serve-grpc` once with
     `scripts/session_warm_server.sh`, generate matching protobuf stubs,
     and use
     `references/NUREC_RenderClient/scripts/thin_client.py` for
     repeated single-camera or `batch_render_rgb` calls.
   - **Remote CLI / simulator integration** — `serve-grpc` +
     `render-grpc` (or your own client via `nurec.grpc.protos`).
     Use for simulator loops, in-server Harmonizer, or
     `--edit-assets`. See `references/grpc-api.md` and
     `references/physical-ai-render.md`.
7. **Edit actors (optional).** Run `export-external-assets` to
   repackage Asset-Harvester output into a new USDZ, then pass
   the produced `edit-assets.json` to `render-grpc --edit-assets`
   (with `serve-grpc --enable-editing-actors`). See
   `references/asset-editing.md`.
8. **Validate the result.** Confirm
   `<output_dir>/<RUN-ID>/artifacts/last.usdz` opens, `metrics.yaml`
   reports a reasonable `test/psnr`, and the generated MP4s
   render. For more thorough metrics use Workflow I (eval) in
   [`references/workflows.md`](references/workflows.md). Tear
   down any gRPC server (`Ctrl-C` or `docker rm -f`).

For any NuRec task expected to run 5 minutes or longer (training
or multi-clip renders), follow
`references/long-running-tasks.md`: delegate to a subagent /
background job and report compact status at least every 5
minutes.

## Examples

### Example 1 — End-to-end NCore → USDZ → render

Walk Workflow A in
[`references/workflows.md`](references/workflows.md): validate
host → generate aux data → train (cookbook recipe) → export →
local render or `serve-grpc`. Concrete commands live in the
referenced files; this index does not duplicate them.

### Example 2 — Skip training, render the gated Physical AI dataset

Walk Workflow B: download
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` from HuggingFace,
then jump to `serve-grpc` + a Python client. Coordinate-frame
conversion code is in `references/physical-ai-render.md`.

### Example 3 — Insert Asset-Harvester actors into a USDZ

Walk Workflow C: obtain compatible captured assets, then
`export-external-assets`, edit `edit-assets.json`, and call
`serve-grpc --enable-editing-actors` + `render-grpc
--edit-assets`. Schema lives in `references/asset-editing.md`.

### Example 4 — Warm-server thin-client for repeated RGB renders

Walk the warm-server quick start at the bottom of
[`references/workflows.md`](references/workflows.md). Boot
`scripts/session_warm_server.sh`, render with
`thin_client.py`, tear down with `scripts/session_teardown.sh`.

## Backend Selection

Pick the smallest backend that exposes the requested feature:

- **Local Docker, single command.** Use `nurec render`, `render-grpc`,
  or an export sub-command directly. Simplest for one-off renders,
  LiDAR sweeps, actor edits, rolling shutter, in-container video
  export, or exact `--replicate-training-views` behavior. See
  `references/local-render.md`,
  and `references/mp4-encoding.md`.
- **Local Docker, warm `serve-grpc` + thin host client.** Use for
  render-heavy RGB sessions where repeated Docker/Python/CUDA
  cold-start dominates latency, or where multiple cameras should
  be rendered through one `batch_render_rgb` RPC. See
  `references/NUREC_RenderClient/README.md` and
  `scripts/session_warm_server.sh` / `scripts/session_teardown.sh`.

## Output Format

Structured deliverables placed under `${output_dir}/${RUN_ID}/`
by the NuRec container (no JSON state file required from the
agent):

- `config/parsed.yaml` — Hydra-resolved training config.
- `checkpoints/last.ckpt` (plus periodic snapshots).
- `val/metrics.yaml` — per-frame PSNR / SSIM / LPIPS under
  `test/*`.
- `val/*.mp4`, `val/<frame>/*.png` — depth, opacity, segmentation,
  RGB visualisations.
- `artifacts/last.usdz` — USDZ containing the trained reconstruction,
  `data_info.json`, `rig_trajectories.json`,
  `sequence_tracks.json`, `parsed_config.yaml`, `checkpoint.ckpt`,
  optional `mesh.ply`, and `map.xodr`. Render with `nurec render`,
  `serve-grpc` + `render-grpc`, the in-container `viewer`, or hand
  to a downstream simulator (CARLA, AlpaSim, Isaac Sim).
- `*.ply` / `ego_mask/*` / `depth/*` / `sequence_tracks.json` /
  `ncore_tracks.json` / `mesh.ply` / `ground_mesh.ply` — produced
  by the matching export sub-command.

## Scripts

| Script                           | Purpose                                                                                             | Usage                                                                             |
| -------------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `scripts/validate_setup.py`      | Local source, dependency-image, Docker, toolkit, GPU and R580+ driver preflight; no downloads.      | `python3 scripts/validate_setup.py --source "$NUREC_SRC" [--strict]`              |
| `scripts/nurec_dev.sh`           | Source-build container helper with persistent caches and same-path mounts.                          | `source scripts/nurec_dev.sh`                                                     |
| `scripts/session_warm_server.sh` | Build `//:run`, start the session server, and wait for readiness; reuse a matching running session. | `NUREC_GRPC_USDZ_HOST_DIR=/path/to/usdz/root bash scripts/session_warm_server.sh` |
| `scripts/session_teardown.sh`    | Stop the recorded session container; preserve caches and data.                                      | `bash scripts/session_teardown.sh`                                                |

## References

Read these on demand; keep `SKILL.md` as the routing layer.

- [`references/install.md`](references/install.md) — released source, matching
  development image, setup checks, and safe `HF_TOKEN` handling.
- [`references/cookbook.md`](references/cookbook.md) — most-used
  source-built invocations: train + validate, re-validate with
  shift, local render at quarter or native res, `serve-grpc` boot,
  LiDAR sweep, in-container `--help`.
- [`references/workflows.md`](references/workflows.md) — workflows
  A – I end-to-end, plus the warm-server thin-client quick start.
- [`references/troubleshooting.md`](references/troubleshooting.md)
  — extended error matrix (`OOM`, `wandb` blocking,
  `--artifact-glob` mismatches, deprecated flags, gRPC LiDAR size,
  etc.).
- [`references/teardown.md`](references/teardown.md) — disk
  cleanup, post-teardown verification, ownership-recovery.
- `references/cli-reference.md` — full sub-command surface of the
  NuRec container (training, validation, `render`, `serve-grpc`,
  `render-grpc`, `render-novel-trajectory`, every `export-*`,
  `upgrade-config` / `upgrade-artifact`, `gaussian-statistics`,
  `eval-rendering-metrics`, `compute-metrics`, `viewer`,
  `ply_viewer`, `profile-dataloader`, `run-script`, the
  `//apps:nurec_tools` auxiliary-data and PAI conversion entry points).
- `references/configuration.md` — Hydra recipe map for Waymo / NV
  / PandaSet / Tesla / Alpasim, plus override matrix.
- `references/aux-data.md` — `//apps:nurec_tools` auxiliary-data CLI.
- `references/local-render.md` — host-side `nurec_dev bazelisk run //:run -- render`
  recipes for rig offsets and `export-custom-rig-trajectory`.
- `references/carline-adaptation.md` — augmented target-rig trajectory export,
  local render, and optional `harmonizer` handoff.
- `references/NUREC_RenderClient/README.md` — warm-server thin
  Python gRPC client.
- `references/grpc-api.md` — sensorsim gRPC server flags + Python
  client cookbook.
- `references/harmonizer.md` — public Harmonizer checkpoint and configuration.
- `references/mp4-encoding.md` — host-side ffmpeg recipe.
- `references/asset-editing.md` — `export-external-assets` +
  `edit-assets.json` schema.
- `references/physical-ai-render.md` — recipe for rendering the
  HuggingFace NuRec dataset.
- `references/example-workflows/` — local Bash workflow examples.
- `references/rig-json/` — bundled `rig.json` and
  `augmented_rig.json`.
- `references/long-running-tasks.md` — background-job + 5-minute
  status reporting convention.
- `references/nurec-skill-catalog.md` — routing table for sibling
  NuRec-stack skills.
- Public product page: <https://www.nvidia.com/en-us/omniverse/>
- HF dataset: <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec>
- HF Harmonizer model: <https://huggingface.co/nvidia/Harmonizer>

## Prerequisites

Linux x86_64 + NVIDIA GPU + Docker 23+ with Buildx + NVIDIA Container Toolkit
1.13+ + released source, matching dev-deps image and built development image. Full requirements (driver,
shm-size, file ownership, GPU-tier guidance) lives in
[`references/install.md`](references/install.md). Always verify
via `scripts/validate_setup.py` after building the development image.

## Limitations

- **Linux x86_64 only.** aarch64 (e.g. Jetson) is not supported.
- **Match the release.** Use the public source checkout and its matching
  development-dependencies image, then build the development image with that
  checkout's Dockerfile. Keep this source/base pair together when rebuilding.
- **Multi-GPU defaults are conservative.** Set
  `trainer.world_size` / `trainer.num_nodes` explicitly to scale
  out; SLURM is auto-detected when both are `0`. Quality plateaus
  past ~6 GPUs; per release notes, multi-GPU +
  `dataset.aux_data=false` is a known crash combination.
- **`--config-name` paths differ between train and val/export.**
  Training uses source-shipped recipes; validation and exports
  re-pass the `parsed.yaml` written under
  `<output_dir>/<RUN-ID>/config/`.
- **Validation may prompt for `wandb`.** Choose option 3 to skip
  in non-interactive runs, or pass `logger=tensorboard` /
  `logger=dummy`.
- **Render gRPC is data-format-pinned.** Older releases warn /
  reject artifacts that pre-date them; check release notes when
  mixing client / server versions.
- **Asset-Harvester input only.** `export-external-assets`
  requires AH outputs; raw `.ply` files won't carry the
  per-asset cuboid metadata.
- **The Harmonizer is pluggable.** The Hydra group is `harmonizer`;
  select `harmonizer=public_harmonizer_nontemporal` for public weights.
  Disable with `harmonizer=disabled`. See `references/harmonizer.md`.
- **`render` ↔ `render-grpc` overlap.** `render` runs
  in-container without a server; `render-grpc` requires an active
  `serve-grpc`. Use `render` for batch novel-view jobs and
  `render-grpc` when you need actor editing or
  a long-lived service.

## Troubleshooting (top 4)

| Error                                                   | Cause                                                | Fix                                                                                                                            |
| ------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Dependency image missing                                | Release image is not cached.                         | Pull the exact published GHCR tag/digest; see [install.md](references/install.md).                                             |
| `OOM Killed` / `CUDA out of memory` during training     | Default recipe needs >= 48 GB VRAM.                  | Reduce `dataset.camera_ids`, lower `datamodule.train_batch_size`, switch to `trainer.precision=16-mixed`, or use a 48 GB+ GPU. |
| `serve-grpc` fails to find the USDZ                     | `--artifact-glob` must end in `.usdz` and be quoted. | Use e.g. `--artifact-glob '$NUREC_SRC/output/<RUN-ID>/artifacts/last.usdz'`.                                                   |
| Edits silently ignored from `render-grpc --edit-assets` | Server started without `--enable-editing-actors`.    | Restart `serve-grpc` with that flag.                                                                                           |

Full matrix in
[`references/troubleshooting.md`](references/troubleshooting.md).

## Teardown

Full inventory, ownership-recovery, and post-teardown verification
commands live in [`references/teardown.md`](references/teardown.md).
Headline: stop only the service started for this task. Preserve images, caches,
source, datasets, and outputs unless the user requests disk cleanup.
