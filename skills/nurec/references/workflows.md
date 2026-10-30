<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec — End-to-end workflows (A – I)

Complete [source setup](install.md) before running the commands below.

Workflow A is the headline path; B – I are variants that compose
the same primitives differently. Always pair these with the CLI
recipes in [`cookbook.md`](cookbook.md).

## Workflow A — NCore clip → reconstruction → novel-view render

1. Run `python3 "$NUREC_SRC/skills/nurec/scripts/validate_setup.py" --source "$NUREC_SRC"`
   and resolve any prerequisite failures.
2. Generate auxiliary data with `nurec_dev bazelisk run //apps:nurec_tools -- ncore-aux-data`
   (see [aux-data.md](aux-data.md)). With the default archive format, confirm
   `<NAME>.aux.sseg.zarr.itar` and the other requested auxiliary shards sit next
   to the NCore manifest and source shards so training can discover them.
3. Train + validate with the [cookbook.md train recipe](cookbook.md).
   Wait for `artifacts/last.usdz` to appear and `metrics.yaml`
   `test/psnr` to stabilise.
4. `export-gaussian-plys` (optional) to emit Gaussians for offline
   tooling, or `export-mesh` / `export-ground-mesh` for downstream
   Omniverse use.
5. Render — choose **(a)** local `nurec render` for batch novel-view
   jobs without a server, or **(b)** `serve-grpc` + `render-grpc`
   (or your own Python client via `nurec.grpc.protos`) when you need
   actor editing, LiDAR rendering, or a long-lived service.

## Workflow B — Render the NVIDIA Physical AI dataset

Skip the training step entirely. Download the pre-reconstructed
USDZs from HuggingFace (gated dataset
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`), then jump straight
to `serve-grpc` + a Python client. Full code for coordinate-frame
conversion (NuRec ↔ ECEF ↔ OpenDRIVE ENU) lives in
`physical-ai-render.md`.

If you instead need to **train from scratch** on the upstream
`nvidia/PhysicalAI-Autonomous-Vehicles` clips (after running
`pai-to-ncore`), use the AV Benchmark preset shipped in the checkout — see the "Train +
validate (Physical AI Autonomous Vehicles — Hyperion-8.1)" recipe in
[`cookbook.md`](cookbook.md) and the host-side walkthrough in
[`example-workflows/bash/nurec_workflow_pai.md`](example-workflows/bash/nurec_workflow_pai.md).
Do **not** train PAI clips with the Waymo `3dgut_dynamic.yaml` recipe;
that one is for the Waymo Open Dataset only.

## Workflow C — Insert Asset-Harvester actors into a USDZ

1. Run `asset-harvester` on the relevant `track_ids` from the
   source NCore clip to produce a directory of `.ply` +
   `metadata.yaml` per asset.
2. `export-external-assets` to repackage the AH directory into a
   new USDZ and produce a stub `edit-assets.json`.
3. Edit the JSON (`replace`, `remove`, `insert`) — schema lives in
   `asset-editing.md`.
4. `serve-grpc` with `--enable-editing-actors`, then `render-grpc
--edit-assets <path>` to produce edited frames.

## Workflow D — Multi-GPU training

The example below is for the Waymo Open Dataset. Swap
`--config-name` for the recipe matching your source dataset (PAI →
the AV Benchmark preset; PandaSet / NV / Tesla → corresponding
`apps/...` path) — the multi-GPU flags are independent of
the recipe.

```bash
NAME=my-clip-name   # NCore clip basename under the dataset dir

# Explicit: 4 GPUs on 1 node.
nurec_dev bazelisk run //:run -- \
  main \
  --config-name=apps/AV/Waymo/3dgut_dynamic_mcmc.yaml \
  mode=trainval \
  dataset.path=$NUREC_SRC/dataset/"${NAME:?}".json \
  out_dir=$NUREC_SRC/output \
  logger=tensorboard \
  trainer.world_size=4 trainer.num_nodes=1
```

NuRec auto-scales LR, schedule, and dataloader workers (toggle via
`trainer.relative_lr` / `relative_schedule` / `relative_num_workers`).
On SLURM, set `trainer.world_size=0 trainer.num_nodes=0` for
auto-detect. Reconstruction quality plateaus past ~6 GPUs, so for
single-clip jobs prefer 4–6 GPUs over higher counts.

## Workflow E — Resume training

```bash
RUN_ID=latest   # run directory name under $NUREC_SRC/output

nurec_dev bazelisk run //:run -- \
  main \
  --config-name=$NUREC_SRC/output/"${RUN_ID:?}"/config/parsed.yaml \
  mode=train \
  resume=last
```

`resume=last` resolves to `<RUN-ID>/checkpoints/last.ckpt`; pass an
absolute path or `"epoch=4.ckpt"` to pick a different snapshot. Use
`resume_weights_only=true` for transfer fine-tuning. `out_dir`
cannot change between the original and resumed runs.

## Workflow F — Upgrade an old USDZ once (avoid the in-memory upgrade tax)

```bash
OLD_RUN=2025-06-01_12-00-00   # existing run dir holding the old USDZ

nurec_dev bazelisk run //:run -- \
  upgrade-artifact \
  --input  $NUREC_SRC/output/"${OLD_RUN:?}"/artifacts/last.usdz \
  --output $NUREC_SRC/output/"${OLD_RUN:?}"/artifacts/last.upgraded.usdz \
  --target-version 26.4
```

`upgrade-config` is the equivalent for a standalone YAML /
`parsed.yaml` extracted from a USDZ. To inspect the structure of a
checkpoint before/after upgrade use `export-artifact-structure`.

## Workflow G — Headless rendering of LiDAR sweeps

Combine a long-lived `serve-grpc` (e.g. on a render node) with one
or more `render-grpc --lidar` workers:

```bash
RUN_ID=latest   # run directory name under $NUREC_SRC/output

# Server (one terminal).
nurec_dev bazelisk run //:run -- \
  serve-grpc \
  --port="${NUREC_GRPC_PORT:-8080}" \
  --artifact-glob "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz" \
  --renderer default \
  --max-workers 4 \
  --cache-size 4

# Client (another terminal / job).
nurec_dev bazelisk run //:run -- \
  render-grpc \
  --port="${NUREC_GRPC_PORT:-8080}" \
  --artifact-path $NUREC_SRC/output/"${RUN_ID:?}"/artifacts/last.usdz \
  --output-dir $NUREC_SRC/output/"${RUN_ID:?}"/lidar \
  --lidar --lidar-id lidar_top \
  --lidar-format ply \
  --lidar-raydrop-threshold 0.5 \
  --lidar-opacity-threshold 0.0
```

For RGB + LiDAR + actor editing in the same loop, write a Python
client against `nurec.grpc.protos` — see `grpc-api.md`.

## Workflow H — Browse a USDZ in a browser

```bash
RUN_ID=latest   # run directory name under $NUREC_SRC/output

nurec_dev bazelisk run //:run -- \
  viewer \
  --artifact-path $NUREC_SRC/output/"${RUN_ID:?}"/artifacts/last.usdz \
  --port 8080
  # add --host 0.0.0.0 ONLY if you need to reach it from another machine;
  # the viewer is unauthenticated, so leave it on loopback otherwise.
```

Open `http://localhost:8080`. The viewer binds **loopback by default**, so a
routable IP only works if you added `--host 0.0.0.0` above — and that exposes
an unauthenticated viewer to the network. Use
`ply_viewer --ply-path <ply>` to inspect a standalone PLY (e.g.
exported via `export-gaussian-plys`).

## Workflow I — Evaluate rendering quality against ground truth

1. `export-ncore-benchmark-gt` to materialize the GT folder layout
   (`<gt-dir>/camera_images/<seq>/<ts>.<ext>` plus optional
   `<gt-dir>/camera_ego_masks/<seq>.png`).
2. Render the trained USDZ along the same trajectory (`render` or
   `render-grpc`). Use `--frame-naming frame-end-timestamp` so
   frames match the GT naming.
3. Compute metrics:

   ```bash
   RUN_ID=latest   # run directory name under $NUREC_SRC/output

   nurec_dev bazelisk run //:run -- \
     eval-rendering-metrics \
     --render-dir $NUREC_SRC/output/"${RUN_ID:?}"/renders/color \
     --gt-dir     $NUREC_SRC/output/"${RUN_ID:?}"/gt \
     --output-dir $NUREC_SRC/output/"${RUN_ID:?}"/metrics \
     --metrics psnr --metrics ssim --metrics lpips
   ```

   For one-off pairs, `compute-metrics --metric
psnr|ssim|lpips|fid|...` is a lightweight alternative.

## Warm-server thin-client quick start

Use this path when a session will render the same USDZ repeatedly
or render several cameras per pose. It keeps one `serve-grpc`
container alive and calls it from host-side Python.

```bash
# 1. Complete install.md and activate the client Python environment.

# These paths are relative to the SKILL ROOT, so declare it once. The block
# previously mixed two working directories -- NUREC_RenderClient/... and
# rig-json/... live under references/, while scripts/... lives at the skill
# root -- so no single cwd made every step resolve.
# Point SKILL_ROOT at the `nurec` skill directory (the one holding SKILL.md
# and scripts/). Set it explicitly for an installed or standalone copy:
#   export SKILL_ROOT=/path/to/nurec
# Inside a checkout of this repo the default below derives it for you.
SKILL_ROOT="${SKILL_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null)/skills/nurec}"
if [ ! -f "$SKILL_ROOT/SKILL.md" ]; then
  echo "STOP: SKILL_ROOT is not the nurec skill dir: $SKILL_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi

# 2. Generate protobuf stubs from the matching source checkout.
bash "$SKILL_ROOT/references/NUREC_RenderClient/scripts/setup_protos.sh"

export NUREC_GRPC_PORT="${NUREC_GRPC_PORT:-8080}"

# 3. Build and start the warm server for the scene directory.
NUREC_GRPC_USDZ_HOST_DIR=/path/to/usdz/root \
  bash "$SKILL_ROOT/scripts/session_warm_server.sh"

# 4. Render one or more cameras with the thin client.
python3 "$SKILL_ROOT/references/NUREC_RenderClient/scripts/thin_client.py" \
    --port="${NUREC_GRPC_PORT:-8080}" \
  --cameras camera_front_wide_120fov \
  --rig-file "$SKILL_ROOT/references/rig-json/augmented_rig.json" \
  --output-dir /tmp/nurec-out/turn-001

# 5. Encode frames to MP4 if needed, then tear down at session end.
bash "$SKILL_ROOT/scripts/session_teardown.sh"
```

Use the source `render` or `render-grpc` CLI when the turn needs LiDAR,
actor edits, rolling shutter, arbitrary custom ego trajectories,
or features the thin client deliberately leaves to the NuRec
container CLI.

## Physical AI AV benchmark

For raw PAI clips, follow the [AV quickstart](https://docs.nvidia.com/nurec/av/index.html)
for data preparation, reconstruction, and evaluation. For automated runs, prefer
the `physical-ai-av-benchmark` skill when available.
Hosted NuRec links track the latest published docs and may differ from this
checkout.
The released source also provides `//workflows/av_benchmark:run`, which keeps
download, conversion, auxiliary data, training, export, rendering, and evaluation
consistent. For runner options, read `$NUREC_SRC/workflows/av_benchmark/README.md` and inspect:

```bash
nurec_dev bazelisk run //workflows/av_benchmark:run -- --help
```

Select a bounded clip set, output directory, and run budget; inspect
`manifest.json` and the first failed stage before retrying.
