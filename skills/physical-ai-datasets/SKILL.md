---
name: physical-ai-datasets
description: >-
  Use when the user wants to find, download, or pick a NVIDIA Physical
  AI dataset on Hugging Face for autonomous-vehicle, robotics, spatial
  intelligence, manipulation, or neural-reconstruction workflows.
  Catalog of the NVIDIA `PhysicalAI-` datasets on Hugging Face as of the
  2026-04 snapshot — not guaranteed exhaustive; check the org listing for
  anything newer. Organised by domain (AV, Robotics-Manipulation,
  Robotics-GR00T, Robotics-mindmap, Robotics-NuRec, Spatial
  Intelligence, Grasping, Healthcare, Sim-Ready, Material properties),
  with per-dataset size, format, gating, license, and the downstream
  companion skill (`ncore`, `nurec`, `asset-harvester`, `harmonizer`) or
  upstream tool (Isaac Sim, CARLA, Isaac-GR00T, Cosmos-*) that
  consumes it. Do NOT use as a runtime — it routes you elsewhere.
  Trigger keywords: nvidia physical ai dataset, PhysicalAI- dataset,
  hf nvidia dataset, NCore dataset, NuRec dataset, GR00T dataset,
  GraspGen, SimReady, Cosmos-Drive-Dreams, Lyra SDG,
  Open-H-Embodiment, huggingface-cli download, physical_ai_av,
  dataset gated, RDS-HQ.
version: "0.1.1"
tools:
  - Shell
  - Read
  - Write
license: Apache-2.0
dependencies:
  - bash
  - python
  - huggingface_hub
compatibility: >-
  All listed datasets are hosted on Hugging Face and need
  `huggingface_hub` (CLI: `huggingface-cli` / new `hf` shim). Most are
  PUBLIC and pull without a token; only a small minority are GATED and
  need a token plus explicit license acceptance on the dataset page
  (the AV family in particular). Check the dataset page before assuming
  a token is required. Storage ranges from <1 GB to >100 TB (PhysicalAI-Autonomous-Vehicles, 133 TB).
metadata:
  author: NVIDIA Physical AI
  tags:
    - dataset-catalog
    - physical-ai
    - huggingface
    - autonomous-vehicles
    - robotics
  hf_org: https://huggingface.co/nvidia
  hf_collection: https://huggingface.co/collections/nvidia/physical-ai
  cosmos_dataset_search: https://build.nvidia.com/nvidia/cosmos-dataset-search
  pai_av_toolkit: https://github.com/NVlabs/physical_ai_av
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Physical AI Datasets (Hugging Face)

## Purpose

Help the agent find, evaluate, and download an NVIDIA `PhysicalAI-*`
dataset on Hugging Face that fits the user's downstream task —
autonomous-vehicle reconstruction, robotics manipulation, GR00T
post-training, spatial-intelligence research, grasping, or sim-ready
content — and then hand off to the companion skill (`ncore`, `nurec`,
`asset-harvester`, `harmonizer`) or upstream NVIDIA tool that
actually consumes it.

**Use this skill when:** the user asks "is there an NVIDIA dataset
for X?", "where do I get NCore / NuRec / GR00T sample data?", or is
shopping the Hugging Face NVIDIA org for `PhysicalAI-*` collections.

**Do NOT use this skill when:**

- The user already knows the dataset and just wants to run a
  pipeline — jump straight to the consuming skill.
- The user needs a non-NVIDIA dataset (Waymo, nuScenes, KITTI, …) —
  this catalog is NVIDIA-only.
- The user wants to train Cosmos / GR00T / Isaac Sim itself — that's
  the upstream tool's job, not this catalog's.

## Overview

Catalog of NVIDIA's open Physical AI dataset family on Hugging Face.
Pick by **task** (Section 2 § lookup table) or **family** (Sections 3–10).
Every entry lists: dataset path, size, format, license, gating, and the
downstream tool that consumes it.

> Source of truth: <https://huggingface.co/nvidia> (filter `PhysicalAI-`)
> and the curated [Physical AI collection](https://huggingface.co/collections/nvidia/physical-ai).
> When upstream cards drift, re-check the HF page; this skill mirrors
> the cards as of Apr 2026.

## Recommendation handoff

Every recommendation, including robotics/manipulation datasets, must state:

- The exact repository and selected subset, format, and license from its current card.
- Whether it is public or gated, with browser-acceptance/token steps only when needed.
- A bounded download command or file-selection procedure and an estimated storage budget.
- A concrete consuming tool/workflow and the next action with that data.

Verify a companion skill is installed before routing to it; otherwise link to
the upstream tool's actual documentation. For a Physical AI AV benchmark on a
new Brev instance, prefer `physical-ai-av-benchmark` when installed. For raw
Physical AI AV reconstruction on an existing host, use the source-built workflow
documented by `nurec`; for pretrained scene USDZ files, use its direct rendering
workflow.

## Prerequisites

- `git`, `git-lfs`, and `huggingface_hub[cli]` on PATH. **That is all you
  need for the public majority of these datasets** — no account, no token.
- **Only for the gated minority** (the `PhysicalAI-Autonomous-Vehicles*`
  family in particular):

  - a HuggingFace account with that dataset's card **opened in a browser at
    least once** and its gating checkbox accepted;
  - a HuggingFace user access token exported as `HF_TOKEN` (create at
    <https://huggingface.co/settings/tokens>).

  Check the dataset's own Gating row before assuming either is required.

- Storage room sized to the dataset you're pulling (see the per-row
  size column; some are < 1 GB, the AV dataset is 133 TB — always
  pre-filter with `--include` or `physical_ai_av`).

### Verifying secrets safely

**Always check token presence with `hf auth whoami` or a length-only
shell test; never write ad-hoc bash that interpolates `HF_TOKEN`
values.** The common one-liner

```bash
# BAD — leaks the secret to the terminal when the variable is set
echo "HF_TOKEN: ${HF_TOKEN:+yes}${HF_TOKEN:-no}"
```

prints `yes<token-value>` whenever `HF_TOKEN` is set, because
`${VAR:-no}` only falls back to "no" when `VAR` is empty — when set
it expands to `$VAR`. Use one of these instead:

```bash
hf auth whoami                              # confirms the token without echoing it
test -n "$HF_TOKEN" && echo "HF_TOKEN: set (${#HF_TOKEN} chars)" || echo "HF_TOKEN: missing"
```

Rotate any token you suspect was echoed at
<https://huggingface.co/settings/tokens>.

## Table of Contents

1. [Common download recipe](#common-download-recipe) — HF auth, gating, CLI.
2. [Filtered AV download recipe](#filtered-av-download-recipe) — default `hyperion_8.1` filter for `PhysicalAI-Autonomous-Vehicles` raw pulls.
3. [Pick a dataset by task](#pick-a-dataset-by-task) — fast lookup table.
4. [Autonomous Vehicles](#autonomous-vehicles) — 5 datasets.
5. [Robotics — Manipulation](#robotics--manipulation) — 6 datasets.
6. [Robotics — GR00T](#robotics--gr00t) — 7 datasets.
7. [Robotics — mindmap](#robotics--mindmap) — 4 datasets.
8. [Robotics — NuRec / Sim-Ready scenes](#robotics--nurec--sim-ready-scenes) — 2 datasets.
9. [Robotics — Healthcare](#robotics--healthcare) — 1 dataset.
10. [Robotics — Grasping](#robotics--grasping) — 1 dataset.
11. [Robotics — Physical / material properties](#robotics--physical--material-properties) — 2 datasets.
12. [Spatial Intelligence + SimReady scenes](#spatial-intelligence--simready-scenes) — 5 datasets.
13. [License decision tree](#license-decision-tree) — what you can do with each.
14. [Cross-skill usage map](#cross-skill-usage-map) — which skill consumes which dataset.

## Common download recipe

All NVIDIA Physical AI datasets live on `huggingface.co/datasets/nvidia/...`.
Most are **public**; only a minority are gated (see each entry's License /
Gating rows).

```bash
sudo apt -y install git git-lfs
git lfs install

uv tool install -U "huggingface_hub[cli]"   # or: pip install --upgrade "huggingface_hub[cli]"

# Only needed for the GATED minority (the AV family in particular). Most
# PhysicalAI-* repos are public and pull with no token at all -- do not log in
# reflexively.
#   export HF_TOKEN=...   # then `hf download` picks it up from the environment
```

Most `PhysicalAI-*` repos are public — a plain `hf download` works with no
token at all. A token is only needed for the gated minority.

For a **gated** dataset, the token must:

1. Belong to a HF account that has **logged in** to that dataset page in a
   browser at least once.
2. Have **accepted** the license / terms-of-use checkbox the dataset shows
   (re-accept if it has been re-gated — common for AV).

If a download 404s or returns HTML, check whether that specific dataset is
gated rather than assuming every one is.

Three download patterns:

```bash
# REQUIRED: pick a dataset from the catalog below. There is deliberately no
# default -- the previous one was GR00T-X-Embodiment-Sim, a 1.91 TB collection,
# which is not something to pull by accident.
: "${DATASET:?set DATASET to a PhysicalAI-* name from the catalog below}"

# Whole dataset -- ONLY for datasets the catalog lists as small/medium. Check
# the Size row first; for anything multi-hundred-GB use the --include form.
hf download "nvidia/${DATASET:?}" --repo-type dataset --local-dir "./${DATASET:?}"

# Sub-folder only (recommended for large multi-task collections)
hf download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --repo-type dataset \
  --include "gr1_arms_only.CanSort/**" \
  --local-dir ./gr00t_dataset

# Sparse-checkout via git-LFS (if you want incremental git-style work)
SUBFOLDER=gr1_arms_only.CanSort
git clone --filter=blob:none --no-checkout "https://huggingface.co/datasets/nvidia/${DATASET:?}"
cd "${DATASET:?}"
git sparse-checkout init --cone
git sparse-checkout set "${SUBFOLDER:?}"
git checkout main
```

Special-case downloaders:

- **PhysicalAI-Autonomous-Vehicles** (133 TB) — use the official Python
  toolkit `pip install physical_ai_av`
  ([NVlabs/physical_ai_av](https://github.com/NVlabs/physical_ai_av))
  to filter by sensor / country / split before downloading; otherwise
  you will pull TBs you don't need.

  **Default policy: pre-filter to `platform_class == 'hyperion_8.1'`
  before any bulk pull.** The dataset mixes two sensor rigs
  (`hyperion_8` and `hyperion_8.1`). NCore's PAI converter **accepts
  both** (`tools/data_converter/pai/converter.py`), but the downstream
  NuRec / Asset-Harvester chain is only _validated_ on `hyperion_8.1`. Roughly half of the 306k clips fall
  outside that platform, so the filter typically cuts the working set
  (and download size) in half.

  Apply this filter by default in any download script you generate.
  Only skip the filter when the user **explicitly** asks for it (e.g.
  "download all platforms", "include hyperion_8", "don't filter by
  platform") or when they hand you a specific clip UUID — in that
  single-clip case, `download_clip_features(clip_id=...)` is already
  scoped and no platform filter is needed.

  Canonical filtered recipe (see § [Filtered AV download recipe](#filtered-av-download-recipe)
  below for an end-to-end example):

  ```python
  from physical_ai_av import PhysicalAIAVDatasetInterface

  dataset = PhysicalAIAVDatasetInterface()
  dataset.download_metadata()
  dc = dataset.metadata['data_collection']
  hyperion_81_clip_ids = dc[dc['platform_class'] == 'hyperion_8.1'].index.tolist()
  ```

- **PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams** (3 TB) — use the
  upstream `download.py`
  ([nv-tlabs/Cosmos-Drive-Dreams](https://github.com/nv-tlabs/Cosmos-Drive-Dreams/blob/main/scripts/download.py))
  with `--file_types {hdmap,lidar,synthetic}` to select layers.
- **PhysicalAI-SpatialIntelligence-Lyra-SDG** (25 TB) — `hf download
--local-dir lyra_dataset/tar`; untar each tar yourself.
- **Spatial-Intelligence-Warehouse** — chunked TAR-GZs need a manual
  loop after download (script provided in the upstream card).

For dataset filtering / preview: NVIDIA's
[Cosmos Dataset Search (CDS)](https://build.nvidia.com/nvidia/cosmos-dataset-search)
lets you query a 41K subset of the AV dataset semantically before
downloading.

## Filtered AV download recipe

Always use this recipe (or a derivative of it) when the user asks to
download raw clips from `PhysicalAI-Autonomous-Vehicles`. The
`hyperion_8.1` filter is the **default**; it matches every downstream
skill (`ncore`, `nurec`, `asset-harvester`, `harmonizer`).

When to **skip** the platform filter:

1. The user explicitly opts out — e.g. "download all platforms",
   "include hyperion_8 too", "ignore platform_class", or asks for a
   dataset-wide statistic. In that case, drop the platform mask and
   warn them that NuRec / NCore tooling will not work on the
   `hyperion_8` clips.
2. The user gave you a **specific clip UUID**. A single-clip
   `download_clip_features(clip_id=...)` is already scoped, and
   filtering by `platform_class` for one clip is pointless.

Recipe (defaults: filter on; user may layer extra masks like country
or sensor presence on top):

```python
from physical_ai_av import PhysicalAIAVDatasetInterface

dataset = PhysicalAIAVDatasetInterface()

dataset.download_metadata()
dc = dataset.metadata['data_collection']

clip_mask = dc['platform_class'] == 'hyperion_8.1'

# Optional extra masks (only add when the user asked for them):
# clip_mask &= dc['country'] == 'US'
# sp = dataset.metadata['feature_presence']   # 26.03+; was 'sensor_presence' in 25.10
# clip_mask &= sp['lidar_top_360fov']

clip_ids = dc[clip_mask].index.tolist()
print(f"Downloading {len(clip_ids)} hyperion_8.1 clips")

dataset.download_clip_features(
    clip_id=clip_ids,
    features=["camera_front_wide_120fov", "lidar_top_360fov", "egomotion"],
    max_workers=8,
)
```

Single-clip fast path (no platform filter — the UUID is already
specific):

```python
dataset.download_clip_features(
    clip_id="<paste-clip-uuid>",
    features=["camera_front_wide_120fov", "lidar_top_360fov", "egomotion"],
)
```

Explicit opt-out (only when the user asked for it):

```python
clip_ids = dc.index.tolist()   # NO platform filter — all 306k clips
# WARNING: NCore converts both platforms, but only the hyperion_8.1
# subset is validated downstream (NuRec / Asset-Harvester).
```

## Pick a dataset by task

| Goal                                               | Recommended dataset(s)                                                                             |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| End-to-end AV training (real, multi-sensor)        | `PhysicalAI-Autonomous-Vehicles` (133 TB, 1700 h, 25 countries)                                    |
| AV in NCore V4 format (drop-in for `ncore`)        | `PhysicalAI-Autonomous-Vehicles-NCore` (~1.1k clips)                                               |
| AV photoreal Sim2Real / weather augmentation       | `PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams` (3 TB; 7 weather variants)                     |
| AV neural reconstructions ready for CARLA / NuRec  | `PhysicalAI-Autonomous-Vehicles-NuRec` (1607 USDZ scenes)                                          |
| Asset Harvester / 3DGS extraction sample clip      | `PhysicalAI-Autonomous-Vehicles-NCore`                                                             |
| GR00T post-training, broad coverage                | `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` (1.91 TB, 24 GR1 task families + bimanual + RoboCasa) |
| GR00T fine-tune on industrial tasks                | `PhysicalAI-GR00T-Tuned-Tasks` (Nut Pouring, Exhaust Pipe Sorting)                                 |
| GR00T eval images / videos                         | `PhysicalAI-Robotics-GR00T-Eval`, `PhysicalAI-Robotics-GR00T-GR1`                                  |
| Real humanoid teleop (Unitree G1)                  | `PhysicalAI-Robotics-GR00T-Teleop-G1` (1000 trajectories)                                          |
| Sim humanoid teleop (Fourier GR1)                  | `PhysicalAI-Robotics-GR00T-Teleop-Sim` (24 tasks × 1k trajectories)                                |
| Massive humanoid pretraining (44k h, DreamDojo)    | `PhysicalAI-Robotics-GR00T-Teleop-GR1` (74.3 GB)                                                   |
| Spatial-memory imitation learning (mindmap)        | `PhysicalAI-Robotics-mindmap-{Stick-in-Bin,Drill-in-Box,Cube-Stacking,Mug-in-Drawer}`              |
| Robot pick-place in kitchen (bimanual Kinova Gen3) | `PhysicalAI-Robotics-Manipulation-Kitchen`, `-Manipulation-Objects`                                |
| Robot pick-place tabletop (single Franka)          | `PhysicalAI-Robotics-Manipulation-SingleArm`                                                       |
| Cosmos-Transfer1 visual-augmented stacking         | `PhysicalAI-Robotics-Manipulation-Augmented`                                                       |
| Massive teleop in kitchen (Franka + mobile base)   | `PhysicalAI-Robotics-Manipulation-Kitchen-Demos` (600 h, 316 tasks, 55k traj)                      |
| MJCF kitchen objects + fixtures (MuJoCo)           | `PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF`                                            |
| Sim-Ready warehouse for IsaacSim                   | `PhysicalAI-SimReady-Warehouse-01` (753 USD assets)                                                |
| GR1 tabletop digital cousins (assets)              | `PhysicalAI-DigitalCousin-Assets`                                                                  |
| 3DGS / Sim-Ready indoor scenes for AMR sim         | `PhysicalAI-Robotics-NuRec` (Nova Carter labs, Zurich offices, hand-held)                          |
| Multi-cam tracking + 3D box benchmark              | `PhysicalAI-SmartSpaces` (AI City Challenge 2024 + 2025)                                           |
| 3D scene QA / VLM training (warehouses)            | `PhysicalAI-Spatial-Intelligence-Warehouse` (499k QA pairs)                                        |
| Generative 3D scene reconstruction training        | `PhysicalAI-SpatialIntelligence-Lyra-SDG` (25 TB; GEN3C-derived)                                   |
| Radiance-field photometric benchmark               | `PhysicalAI-NuRec-PPISP` (8 sequences, +/-2 EV bracketing)                                         |
| Grasping models (Franka, Robotiq-2f-140, suction)  | `PhysicalAI-Robotics-GraspGen` (57M grasps, Objaverse-LVIS)                                        |
| Healthcare / surgical robotics autonomy            | `PhysicalAI-Robotics-Open-H-Embodiment` (750 h, 4.5 TB, 30+ orgs)                                  |
| Volumetric mechanical / material properties        | `PhysicalAI-Robotics-PhysicalAssets-VoMP`, `-VoMP-Eval`                                            |

## Autonomous Vehicles

### PhysicalAI-Autonomous-Vehicles

The flagship real-world AV dataset.

| Field          | Value                                                                                                                                                                                                                                                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF             | <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>                                                                                                                                                                                                                                                                            |
| Size           | **133 TB** — 306,152 clips × 20 s = **1700 h**                                                                                                                                                                                                                                                                                                     |
| Sensors        | 7 cameras (1080p @ 30 FPS), top-360 LiDAR (10 Hz, 298k clips), up to 10 radars (160k clips), ego motion, calibration, machine labels                                                                                                                                                                                                               |
| Geography      | 25 countries, 2500+ cities (US 155k, Germany 44k, …)                                                                                                                                                                                                                                                                                               |
| Format         | Per-sensor parquet/mp4 chunks of ~100 clips; UUIDs cross-link sensors                                                                                                                                                                                                                                                                              |
| License        | **NVIDIA AV Dataset License Agreement** (gated; AV-development-only purpose; no biometric / surveillance / re-identification; expires 12 months after download)                                                                                                                                                                                    |
| Toolkit        | `pip install physical_ai_av` — direct filtered downloads + format docs. **Default to filtering `metadata['data_collection']['platform_class'] == 'hyperion_8.1'` before any bulk pull** (see [Filtered AV download recipe](#filtered-av-download-recipe)); only skip the filter on explicit user request or when downloading a specific clip UUID. |
| Versions       | 26.03 (current; offline-optimized features for 97 % of clips), 25.10 (initial)                                                                                                                                                                                                                                                                     |
| Subset preview | 41k clips searchable on [Cosmos Dataset Search](https://build.nvidia.com/nvidia/cosmos-dataset-search)                                                                                                                                                                                                                                             |
| Use with       | `ncore` (convert raw clips to NCore V4), `asset-harvester` (extract per-object Gaussian assets). NuRec workflows are only validated for `platform_class == hyperion_8.1`. Other sim/training tools: `NVlabs/alpamayo-1.5`, `NVlabs/alpasim`, CARLA.                                                                                                |

### PhysicalAI-Autonomous-Vehicles-NCore

Curated NCore V4 subset of the above.

| Field    | Value                                                                                                                                                                  |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF       | <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore>                                                                                          |
| Size     | ~1.1k clips with accurate offline calibration / egomotion / cuboids                                                                                                    |
| Format   | NCore V4 — `pai_<uuid>.json` + per-sensor `.zarr.itar` files                                                                                                           |
| License  | NVIDIA AV Dataset License Agreement (gated, same as above)                                                                                                             |
| Use with | `ncore` (drop-in), `asset-harvester` (sample clip path: `clips/02a6f330-5ab0-4e92-99d4-d19e406952f4/`)                                                                 |
| Notes    | Built via [PAI data converter](https://github.com/NVIDIA/ncore/tree/main/tools/data_converter/pai). Use this BEFORE the full AV dataset for any NCore-driven workflow. |

### PhysicalAI-Autonomous-Vehicles-NuRec

Pre-built NuRec dynamic neural reconstructions ready for IsaacSim / CARLA.

| Field          | Value                                                                                                                                   |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| HF             | <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec>                                                           |
| Size           | 1607 USDZ scenes + 1607 front-camera mp4, ~20 s each, with surface meshes. No `labels.json` in the live tree.                           |
| Reconstruction | 6 cameras (front-wide 120°, front-tele 30°, cross-L/R 120°, rear-L/R 70°)                                                               |
| Versions       | Only `sample_set/26.04_release` is present; 26.02 / 25.07 / 25.05 are not downloadable.                                                 |
| License        | NVIDIA AV Dataset License Agreement (gated)                                                                                             |
| Use with       | `harmonizer` (clean up rendered frames), `nurec` (render the USDZs locally or over `serve-grpc`), CARLA (NuRec integration in 0.9.16+). |

### PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams

Cosmos-Transfer-style synthetic + HD-map labels for diverse weather.

| Field            | Value                                                                                                                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF               | <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams>                                                                                                     |
| Size             | 3 TB total (synthetic only ~700 GB) — 5,843 RDS-HQ clips × 2 chunks × 7 weather = 81,802 synthetic videos (121 frames each)                                                                    |
| Modalities       | Cosmos-generated MP4, HDMap (lanes/lanelines/road boundaries/wait lines/crosswalks/markings/poles/lights/signs), LiDAR, vehicle pose, camera intrinsics (ftheta + pinhole), 4D object tracking |
| Cameras          | 7 (front-wide/cross-L/cross-R/rear-L/rear-R/rear-tele/front-tele)                                                                                                                              |
| Weather variants | Foggy / Golden hour / Morning / Night / Rainy / Snowy / Sunny                                                                                                                                  |
| License          | **CC-BY-4.0** (commercial OK with attribution)                                                                                                                                                 |
| Tooling          | `wget … scripts/download.py; python download.py --odir <path> --file_types hdmap,lidar,synthetic`                                                                                              |
| Paper            | <https://arxiv.org/abs/2506.09042>                                                                                                                                                             |
| Use with         | `nvidia/Cosmos-Transfer1`, `nvidia/Cosmos-Predict`, `NVlabs/alpasim`, CARLA.                                                                                                                   |

### PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic

| Field  | Value                                                                                                                          |
| ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| HF     | <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic>                                        |
| Status | **Pointer / placeholder** — content moved to `PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`. Use that. (Card is 2.59 kB.) |

## Robotics — Manipulation

All in **LeRobot v2.x** format unless noted, generated in IsaacSim with
task-and-motion planning + `scene_synthesizer` procedural scenes +
CuRobo motion generation.

### PhysicalAI-Robotics-Manipulation-Kitchen

Bimanual Kinova Gen3 in procedurally-generated kitchens.

| Field        | Value                                                                             |
| ------------ | --------------------------------------------------------------------------------- |
| HF           | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen> |
| Size         | 12 GB total                                                                       |
| Tasks        | open/close × {cabinet, dishwasher, fridge, drawer} = 8                            |
| Trajectories | ~874 episodes total (range 72–205 per task)                                       |
| Cameras      | 6 × 512² RGB+depth+segmentation (world / external / each wrist / head)            |
| License      | CC-BY-4.0                                                                         |
| Commercial   | ✅                                                                                |

### PhysicalAI-Robotics-Manipulation-Objects

Same kitchen environment, bimanual Kinova; pick / place bench / place cabinet.

| Field    | Value                                                                             |
| -------- | --------------------------------------------------------------------------------- |
| HF       | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects> |
| Size     | 4.26 GB                                                                           |
| Tasks    | `pick`, `place_bench`, `place_cabinet` (540 episodes total)                       |
| License  | **NVIDIA OneWay Noncommercial License** — NOT commercial-use                      |
| Use with | Isaac Sim / Isaac Lab.                                                            |

### PhysicalAI-Robotics-Manipulation-SingleArm

Franka Panda tabletop, procedurally generated.

| Field      | Value                                                                                                                                                |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF         | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-SingleArm>                                                                  |
| Size       | 15.3 GB                                                                                                                                              |
| Tasks      | `panda-stack-wide`, `panda-stack-platforms`, `panda-stack-platforms-texture`, `panda-open-cabinet-{left,right}`, `panda-open-drawer` (~38k episodes) |
| Modalities | World cam + wrist cam (RGB + depth on the texture/cabinet/drawer subsets)                                                                            |
| State      | 53 D (stack-wide) / 81 D (others) — proprioception + object poses                                                                                    |
| License    | CC-BY-4.0; commercial OK                                                                                                                             |

### PhysicalAI-Robotics-Manipulation-Augmented

Mimic-generated Franka cube-stacking, plus Cosmos-Transfer1 visual augmentation.

| Field    | Value                                                                                                                   |
| -------- | ----------------------------------------------------------------------------------------------------------------------- |
| HF       | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented>                                     |
| Size     | 77.9 GB                                                                                                                 |
| Episodes | 1000 mimic + 1000 Cosmos-augmented (table + wrist cams, depth + seg + normals)                                          |
| Trick    | 10 human teleops → MimicGen 1k → Cosmos Transfer1 photoreal domain randomization                                        |
| License  | CC-BY-4.0; commercial OK                                                                                                |
| Paper    | <https://arxiv.org/abs/2503.14492> (Cosmos-Transfer1)                                                                   |
| Use with | `nvidia/Cosmos-Transfer1` (legacy Transfer1 workflow), Isaac Sim / Isaac Lab (replay scripts ship in the dataset repo). |

### PhysicalAI-Robotics-Manipulation-Kitchen-Demos

Massive human-teleop dataset on Franka + Omron mobile base.

| Field    | Value                                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------------- |
| HF       | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos>                  |
| Size     | 600 h, 55k trajectories, 316 tasks                                                                       |
| Format   | LeRobot v2.x with MuJoCo `extras/` (`model.xml.gz` + raw states)                                         |
| Cameras  | left + right agentview + eye-in-hand                                                                     |
| Tasks    | `pretrain/atomic/...` × 100 traj/task (Open*, Close*, PickPlace*, Adjust*, Coffee\*, NavigateKitchen, …) |
| Use with | Pair with the MJCF assets dataset below for replay in MuJoCo.                                            |

### PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF

The MuJoCo XML assets that back the Kitchen-Demos environment.

| Field      | Value                                                                                                                                                                                                   |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF         | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF>                                                                                                          |
| Size       | 1.32 GB                                                                                                                                                                                                 |
| Categories | Objects (~58 categories from kettle to whisk) + Fixtures (12 — blender, coffee machine, dishwasher, electric kettle, fridge, microwave, oven, stand mixer, stove, toaster, toaster oven, cabinet panel) |
| Format     | Per-model `model.xml` + visual / collision OBJ + textures, zipped per category                                                                                                                          |
| Use with   | MuJoCo replay of `Manipulation-Kitchen-Demos`.                                                                                                                                                          |

## Robotics — GR00T

GR00T = NVIDIA's generalist humanoid foundation model line. Most data is
sim-generated for post-training; eval / real-robot supplements are
small.

### PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

Largest GR00T post-training corpus.

| Field        | Value                                                                                                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF           | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim>                                                                                         |
| Size         | **1.91 TB**                                                                                                                                                                 |
| Composition  | 9k cross-embodied bimanual (Panda + GR1) + 240k humanoid GR1 tabletop + 24k downsampled + 72k single-Panda RoboCasa + 102 Unitree G1 loco-manipulation = ~345k trajectories |
| Used by      | `nvidia/GR00T-N1.5-3B`, `GR00T-N1.6-3B`, `GR00T-N1.6-bridge`, `GR00T-N1.6-G1-PnPAppleToPlate`, `GR00T-N1.6-DROID`, `GR00T-N1.6-fractal`                                     |
| Download tip | Always pass `--include "<task>/**"` — full clone is 1.91 TB                                                                                                                 |

### PhysicalAI-GR00T-Tuned-Tasks

Two industrial post-training task families.

| Field   | Value                                                                      |
| ------- | -------------------------------------------------------------------------- |
| HF      | <https://huggingface.co/datasets/nvidia/PhysicalAI-GR00T-Tuned-Tasks>      |
| Size    | 26.5 GB                                                                    |
| Tasks   | Exhaust-Pipe-Sorting (1000), Nut-Pouring (1000)                            |
| Format  | HDF5 + GR00T-LeRobot, 256² first-person RGB, 26-DoF state/action, 20 Hz    |
| License | CC-BY-4.0; commercial OK                                                   |
| Models  | `nvidia/GR00T-N1-2B-tuned-Nut-Pouring-task`, `…-Exhaust-Pipe-Sorting-task` |

### PhysicalAI-Robotics-GR00T-Teleop-G1

Real-robot Unitree G1 fruit pick-and-place.

| Field        | Value                                                                            |
| ------------ | -------------------------------------------------------------------------------- |
| HF           | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1>     |
| Size         | 534 MB                                                                           |
| Trajectories | 1000 real teleop, Unitree G1 upper body + Tri-finger hands + RealSense           |
| Tasks        | Pick {apple, pear, grapes, starfruit} → basket                                   |
| Format       | MP4 + HDF5                                                                       |
| License      | CC-BY-4.0; commercial OK                                                         |
| Use with     | `Isaac-GR00T` finetune docs (`getting_started/3_0_new_embodiment_finetuning.md`) |

### PhysicalAI-Robotics-GR00T-Teleop-Sim

Simulated GR1 tabletop teleop.

| Field        | Value                                                                         |
| ------------ | ----------------------------------------------------------------------------- |
| HF           | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim> |
| Size         | 55.4 GB (39 GB LeRobot + 14 GB HDF5)                                          |
| Trajectories | 24 tasks × 1000 each                                                          |
| License      | **CC-BY-NC-4.0** (non-commercial) — different from G1 above                   |
| Format       | HDF5 + LeRobot                                                                |

### PhysicalAI-Robotics-GR00T-Teleop-GR1

DreamDojo pretraining corpus — large-scale human egocentric video.

| Field    | Value                                                                         |
| -------- | ----------------------------------------------------------------------------- |
| HF       | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1> |
| Size     | 74.3 GB                                                                       |
| License  | CC-BY-NC-4.0 — **noncommercial only**                                         |
| Coverage | 44k hours of human egocentric data (per project page)                         |
| Project  | <https://dreamdojo-world.github.io/> + <https://github.com/NVIDIA/DreamDojo>  |
| Paper    | <https://arxiv.org/abs/2602.06949>                                            |

### PhysicalAI-Robotics-GR00T-GR1

Lab-recorded GR1-T2 third-person video.

| Field   | Value                                                                  |
| ------- | ---------------------------------------------------------------------- |
| HF      | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1> |
| Size    | 142 MB                                                                 |
| Records | 92 MP4 videos (Fourier GR1-T2)                                         |
| Use     | DreamGen training reference                                            |
| License | CC-BY-4.0; commercial OK                                               |

### PhysicalAI-Robotics-GR00T-Eval

GR00T eval initial-state frames.

| Field   | Value                                                                   |
| ------- | ----------------------------------------------------------------------- |
| HF      | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Eval> |
| Size    | 180 MB                                                                  |
| Records | 123 PNG frames (GR1-T2 robot's first-person view) + per-frame TXT       |
| License | CC-BY-4.0                                                               |

## Robotics — mindmap

Spatial-memory benchmark from `nvidia-isaac/nvblox_mindmap`. Each
dataset is one task with the same multimodal layout (RGB-D + camera
intrinsics/poses + nvblox vertex features in `.zst` + robot state).
**All four are CC-BY-NC-4.0** (research only). Models trained:
[`nvidia/PhysicalAI-Robotics-mindmap-Checkpoints`](https://huggingface.co/nvidia/PhysicalAI-Robotics-mindmap-Checkpoints).

| Dataset                                                                                                                                       | Robot       | Teleop tool      | Demos provided            | Total mimic-generated        | Storage |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------- | ------------------------- | ---------------------------- | ------- |
| [`PhysicalAI-Robotics-mindmap-GR1-Stick-in-Bin`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-GR1-Stick-in-Bin)         | Fourier GR1 | Apple Vision Pro | 10 (`mindmap` fmt) + HDF5 | 200 (from 20 human teleops)  | 103 GB  |
| [`PhysicalAI-Robotics-mindmap-GR1-Drill-in-Box`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-GR1-Drill-in-Box)         | Fourier GR1 | Apple Vision Pro | 10 + HDF5                 | 200 (from 20 human teleops)  | 53.7 GB |
| [`PhysicalAI-Robotics-mindmap-Franka-Cube-Stacking`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-Franka-Cube-Stacking) | Franka      | SpaceMouse       | 10 + HDF5                 | 1000 (from 10 human teleops) | 6.12 GB |
| [`PhysicalAI-Robotics-mindmap-Franka-Mug-in-Drawer`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-Franka-Mug-in-Drawer) | Franka      | SpaceMouse       | 10 + HDF5                 | 250 (from 15 human teleops)  | 34.5 GB |

> Provided datasets ship 10 mindmap-formatted demos for storage reasons;
> regenerate the full 200/1000/250 with the upstream
> [data-generation docs](https://nvidia-isaac.github.io/nvblox_mindmap/pages/data_generation.html).

Paper: <https://arxiv.org/abs/2509.20297>. Codebase:
<https://github.com/nvidia-isaac/nvblox_mindmap>.

## Robotics — NuRec / Sim-Ready scenes

### PhysicalAI-Robotics-NuRec

Indoor 3DGUT scenes for IsaacSim AMR simulation.

| Field     | Value                                                                                                                                                                                                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF        | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec>                                                                                                                                                                                                              |
| Size      | 62.9 GB                                                                                                                                                                                                                                                                         |
| Scenes    | Nova-Carter (galileo, cafe, wormhole) — stereo, with mesh + occupancy; Zurich offices (lounge, fourth-floor iphone) — mono, no mesh; Endeavor hand-held (andoria, livingroom, wormhole) — stereo with mesh                                                                      |
| Format    | USDZ (3DGUT + mesh + occupancy) loadable in **Isaac Sim 5.1**                                                                                                                                                                                                                   |
| Workflows | [Stereo NuRec (Isaac ROS + cuSFM + FoundationStereo + nvblox + 3DGURT)](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html) for Carter; [Mono NuRec (COLMAP + 3DGURT)](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html) for Zurich |
| Gating    | Contact-info gate (no separate license)                                                                                                                                                                                                                                         |
| License   | CC-BY-4.0                                                                                                                                                                                                                                                                       |
| Use with  | `nurec` to retrain reconstructions; upstream Isaac Sim 5.1 (no in-repo skill) for AMR simulation; pair with [`MobilityGen`](https://github.com/NVlabs/MobilityGen) for AMR data generation.                                                                                     |

### PhysicalAI-NuRec-PPISP

Photometric-variation benchmark for radiance-field methods.

| Field    | Value                                                                                                                       |
| -------- | --------------------------------------------------------------------------------------------------------------------------- |
| HF       | <https://huggingface.co/datasets/nvidia/PhysicalAI-NuRec-PPISP>                                                             |
| Size     | 15.2 GB (8.1 GB COLMAP + 6.2 GB NCore V4)                                                                                   |
| Captures | 4 outdoor scenes × 3 cameras (Nikon Z7, OM-1 II, iPhone 13 Pro) = 8 sequences (~2600 photos), `+/-2 EV` exposure bracketing |
| Variants | Standard (full bracket) + auto (re-processed with auto-exposure / WB)                                                       |
| License  | CC-BY-4.0; commercial OK                                                                                                    |
| Use with | 3DGRUT / GSplat benchmarking via `nurec` (`eval-rendering-metrics`), `asset-harvester`.                                     |

## Robotics — Healthcare

### PhysicalAI-Robotics-Open-H-Embodiment

Surgical / ultrasound robotics multi-embodiment corpus.

| Field        | Value                                                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------- |
| HF           | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment>                                            |
| Size         | 4.5 TB, 750 h, 120,000 trajectories                                                                                       |
| Format       | LeRobot v2.1 — MP4 video + Parquet kinematics + JSONL manifests                                                           |
| Contributors | 30+ orgs (JHU, Stanford, UCSD, UCB, Vanderbilt, TUM, MBZUAI, …)                                                           |
| Purpose      | Healthcare autonomy + world-foundation-model training (used by `nvidia/GR00T-H` and `nvidia/Cosmos-H-Surgical-Simulator`) |
| License      | CC-BY-4.0                                                                                                                 |

## Robotics — Grasping

### PhysicalAI-Robotics-GraspGen

Sim2Real grasping at scale.

| Field    | Value                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| HF       | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen>                                                           |
| Size     | 21.6 GB                                                                                                                         |
| Coverage | **57 M+ grasps** over 8515 Objaverse-XL (LVIS) objects                                                                          |
| Grippers | Franka Panda, Robotiq-2f-140, suction (30 mm radius)                                                                            |
| Format   | WebDataset shards (`grasp_data/{franka_panda,robotiq_2f_140,single_suction_cup_30mm}/shard_{000-007}.tar`) + train/valid splits |
| License  | CC-BY-4.0; commercial OK                                                                                                        |
| Models   | [`adithyamurali/GraspGenModels`](https://huggingface.co/adithyamurali/GraspGenModels)                                           |
| Tip      | Objaverse meshes are **NOT** included — pull separately via the bundled `download_objaverse.py`                                 |

For Franka Panda, use `grasp_data/franka_panda/` and `splits/franka_panda/`.
Before selecting a shard, list the files in the chosen subset on the Hub and
confirm the exact path; download only the user-selected shard or split.

## Robotics — Physical / material properties

VoMP = Volumetric Mechanical Properties.

### PhysicalAI-Robotics-PhysicalAssets-VoMP

| Field   | Value                                                                            |
| ------- | -------------------------------------------------------------------------------- |
| HF      | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-PhysicalAssets-VoMP> |
| Size    | 65.9 GB on-disk (full data nominally 125 GB pre-compression)                     |
| Records | 1664 objects, 37,337,952 voxels, multi-view renders + VLM material annotations   |
| License | CC-BY-4.0                                                                        |

### PhysicalAI-Robotics-PhysicalAssets-VoMP-Eval

| Field   | Value                                                                                 |
| ------- | ------------------------------------------------------------------------------------- |
| HF      | <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-PhysicalAssets-VoMP-Eval> |
| Size    | 8.41 GB on-disk (full eval data nominally 125 GB pre-compression)                     |
| Use     | Held-out eval split for the VoMP model                                                |
| License | CC-BY-4.0                                                                             |

## Spatial Intelligence + SimReady scenes

### PhysicalAI-SimReady-Warehouse-01

OpenUSD warehouse scene + asset library.

| Field       | Value                                                                                               |
| ----------- | --------------------------------------------------------------------------------------------------- |
| HF          | <https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01>                           |
| Size        | 14.4 GB, 753 USD assets + master scene (`physical_ai_simready_warehouse_01.usd`)                    |
| Asset class | Prop / Assembly / Scenario; 1.1.0 adds physically-graspable subset                                  |
| Metadata    | CSV catalogue with WikiData Q-codes, mass (kg), thumbnails                                          |
| Target      | **Isaac Sim 4.x** (Properties → disable Instanceable → Physics → Rigid Body to make assets dynamic) |
| License     | CC-BY-4.0                                                                                           |
| Use with    | Isaac Sim / Isaac Lab, Omniverse SDG.                                                               |

### PhysicalAI-DigitalCousin-Assets

Companion-asset library for the GR1 tabletop sim environments.

| Field   | Value                                                                                                      |
| ------- | ---------------------------------------------------------------------------------------------------------- |
| HF      | <https://huggingface.co/datasets/nvidia/PhysicalAI-DigitalCousin-Assets>                                   |
| Size    | 270 MB                                                                                                     |
| Content | 3D meshes, textures, metadata for tabletop objects (mug, bottle, bowl, container, …) used by GR1 sim tasks |
| License | **CC-BY-NC-4.0** (non-commercial)                                                                          |

### PhysicalAI-SmartSpaces

Multi-camera tracking + 3D box benchmark (AI City Challenge).

| Field  | Value                                                                                                                               |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| HF     | <https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces>                                                                     |
| Size   | 3.53 TB total (216 GB MTMC_Tracking_2024 + 3.31 TB MTMC_Tracking_2025)                                                              |
| 2024   | 90 scenes, 212 h, 953 cameras — Person-only, 2D boxes + multi-cam IDs (52M / 135M)                                                  |
| 2025   | 23 scenes, 42 h, 504 cameras — Person/Forklift/NovaCarter/Transporter/FourierGR1T2/AgilityDigit, 3D boxes + depth maps (8.9M / 73M) |
| Splits | Warehouse (train/val/test) + Lab (val) + Hospital (val) + 4 test scenes added 2025-05-28                                            |
| Eval   | <https://eval.aicitychallenge.org/aicity2024> + <https://eval.aicitychallenge.org/aicity2025>; 3D-bbox HOTA metric for 2025         |
| Paper  | <https://arxiv.org/abs/2412.00692> (MCBLT)                                                                                          |

### PhysicalAI-Spatial-Intelligence-Warehouse

VLM-style spatial QA in warehouses.

| Field      | Value                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------- |
| HF         | <https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse>                                  |
| Size       | 261 GB                                                                                                              |
| QA pairs   | 499k train + 19k test + 1.9k val (categories: `left_right`, `multi_choice_question`, `distance` in metres, `count`) |
| Imagery    | ~95k RGB-D pairs, RLE object masks (pycoco), LLaVA-style conversations                                              |
| Annotation | Rule-based + Llama-3.1-70B-Instruct refinement                                                                      |
| Gating     | Contact-info gate (no separate license)                                                                             |
| License    | CC-BY-4.0                                                                                                           |
| Format     | `train.json` / `val.json` / `test.json` + chunked TAR-GZs of images + depths                                        |

### PhysicalAI-SpatialIntelligence-Lyra-SDG

GEN3C-derived multi-view 3D + 4D training data for `nv-tlabs/lyra`.

| Field       | Value                                                                                                                                                                        |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF          | <https://huggingface.co/datasets/nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG>                                                                                             |
| Size        | **25 TB**                                                                                                                                                                    |
| Composition | 59,031 multi-view 3D examples (354,186 videos) + 7,378 4D examples (44,268 videos), 6 trajectories per source                                                                |
| Modalities  | RGB MP4 + camera pose `.npz` + depth zip                                                                                                                                     |
| License     | CC-BY-4.0                                                                                                                                                                    |
| Paper       | <https://arxiv.org/abs/2509.19296> (Lyra)                                                                                                                                    |
| Pair with   | [`nvidia/Lyra-Testing-Example`](https://huggingface.co/datasets/nvidia/Lyra-Testing-Example) for inference; [`nv-tlabs/lyra`](https://github.com/nv-tlabs/lyra) for training |

### (See also) PhysicalAI-NuRec-PPISP

Listed in [§ 7](#robotics--nurec--sim-ready-scenes) but is also a
spatial-intelligence radiance-field benchmark.

## License decision tree

| License                                                                           | Commercial OK?                                   | Reproducible?                                                                        | Datasets                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **CC-BY-4.0**                                                                     | ✅ (with attribution)                            | ✅ (must keep notice)                                                                | most — Cosmos-Drive-Dreams, GraspGen, all Manipulation-\* (except where noted), GR00T-Teleop-G1, GR00T-Tuned-Tasks, GR00T-GR1, GR00T-Eval, GR00T-X-Embodiment-Sim, Robotics-NuRec, NuRec-PPISP, Open-H-Embodiment, SmartSpaces (CC-BY-4.0 implied via card), Spatial-Intelligence-Warehouse, SpatialIntelligence-Lyra-SDG, SimReady-Warehouse-01, VoMP / VoMP-Eval, Manipulation-Augmented |
| **CC-BY-NC-4.0** (non-commercial)                                                 | ❌                                               | ✅ research                                                                          | GR00T-Teleop-Sim, GR00T-Teleop-GR1, DigitalCousin-Assets, **all 4 mindmap** datasets (all public, no gate)                                                                                                                                                                                                                                                                                 |
| **NVIDIA AV Dataset License Agreement** (gated, AV-only purpose, 12-month expiry) | ✅ ONLY for AV / ADAS development on NVIDIA tech | ❌ — no derivative works, no redistribution, no biometric / re-id / surveillance use | `PhysicalAI-Autonomous-Vehicles`, `…-NCore`, `…-NuRec`                                                                                                                                                                                                                                                                                                                                     |

Quick rule of thumb:

1. If `Robotics-Manipulation-*` and **not** mindmap / DigitalCousin-Assets / **Objects** → CC-BY-4.0 commercial OK. (`…-Manipulation-Objects` is NVIDIA OneWay **Noncommercial**.)
2. If `mindmap-*` → research only (NC).
3. If `Autonomous-Vehicles*` → AV License only, gated.
4. Everything else → check the card.

## Cross-skill usage map

The first column names the skill that consumes each dataset; the
second lists other consumers that have no skill.

| Dataset                                                     | Skill                                                     | Other consumers                                                                              |
| ----------------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `PhysicalAI-Autonomous-Vehicles`                            | `ncore`                                                   | `NVlabs/alpamayo-1.5`, `NVlabs/alpasim`, CARLA                                               |
| `…-NCore`                                                   | `ncore`, `asset-harvester`                                | —                                                                                            |
| `…-NuRec`                                                   | `nurec`, `harmonizer`                                     | CARLA (NuRec integration 0.9.16+)                                                            |
| `…-Cosmos-Drive-Dreams`                                     | —                                                         | `nvidia/Cosmos-Transfer1`, `nvidia/Cosmos-Predict`, `NVlabs/alpasim`, CARLA                  |
| `…-Cosmos-Synthetic`                                        | —                                                         | (pointer to Cosmos-Drive-Dreams)                                                             |
| `Robotics-Manipulation-Kitchen` / `-Objects` / `-SingleArm` | —                                                         | Isaac Sim / Isaac Lab                                                                        |
| `Robotics-Manipulation-Augmented`                           | —                                                         | `nvidia/Cosmos-Transfer1` (Transfer1 path), Isaac Sim / Isaac Lab                            |
| `Robotics-Manipulation-Kitchen-Demos` + `-Kitchen-MJCF`     | —                                                         | MuJoCo direct; Isaac Sim for MJCF→USD                                                        |
| `Robotics-GR00T-X-Embodiment-Sim` / `-Tuned-Tasks`          | —                                                         | [`NVIDIA/Isaac-GR00T`](https://github.com/NVIDIA/Isaac-GR00T), Isaac Sim                     |
| `Robotics-GR00T-Teleop-G1` / `-Sim` / `-GR1` (DreamDojo)    | —                                                         | [`NVIDIA/Isaac-GR00T`](https://github.com/NVIDIA/Isaac-GR00T)                                |
| `Robotics-GR00T-GR1` (DreamGen ref) / `-Eval`               | —                                                         | reference assets only                                                                        |
| `Robotics-mindmap-*`                                        | —                                                         | [`nvidia-isaac/nvblox_mindmap`](https://github.com/nvidia-isaac/nvblox_mindmap), Isaac Lab   |
| `Robotics-NuRec`                                            | `nurec`                                                   | Isaac Sim 5.1, [`MobilityGen`](https://github.com/NVlabs/MobilityGen)                        |
| `NuRec-PPISP`                                               | `nurec` (3DGRUT / GSplat benchmarking), `asset-harvester` | —                                                                                            |
| `Robotics-Open-H-Embodiment`                                | —                                                         | `nvidia/GR00T-H`, `nvidia/Cosmos-H-Surgical-Simulator`                                       |
| `Robotics-GraspGen`                                         | —                                                         | Ships its own visualisation scripts; Isaac Sim for replay                                    |
| `Robotics-PhysicalAssets-VoMP` / `-Eval`                    | —                                                         | VoMP model                                                                                   |
| `SimReady-Warehouse-01`                                     | —                                                         | Isaac Sim 4.x, Omniverse SDG                                                                 |
| `DigitalCousin-Assets`                                      | —                                                         | Isaac Sim                                                                                    |
| `SmartSpaces`                                               | —                                                         | AI City Challenge eval server                                                                |
| `Spatial-Intelligence-Warehouse`                            | —                                                         | Warehouse VLM benchmark (no upstream skill)                                                  |
| `SpatialIntelligence-Lyra-SDG`                              | —                                                         | [`nv-tlabs/lyra`](https://github.com/nv-tlabs/lyra), `nvidia/Cosmos-Predict` (GEN3C lineage) |

## Verify

After downloading any dataset:

```bash
LOCAL_DIR=./my-downloaded-dataset
CLIP_UUID=02a6f330-5ab0-4e92-99d4-d19e406952f4

ls "${LOCAL_DIR:?}"
du -sh "${LOCAL_DIR:?}"          # confirm size matches the table above (within 10 %)

# For LeRobot v2.x:
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
           d = LeRobotDataset(repo_id='nvidia/<the-dataset>', root='${LOCAL_DIR:?}'); print(d.meta.info)"

# For NCore V4 (.zarr.itar):
# ncore_vis is not a wheel entry point -- run it from an ncore checkout.
# Bind loopback: its Viser server is unauthenticated.
bazel run //tools/ncore_vis -- \
    --host=127.0.0.1 \
    v4 --component-group="${LOCAL_DIR:?}/clips/${CLIP_UUID:?}/pai_${CLIP_UUID:?}.json"

# For USDZ scenes:
# load the .usdz in Isaac Sim 5.1 — File -> Open
```

GREEN when:

- File counts and total size match the dataset card to within ~10 %.
- For gated datasets, the download did not silently terminate at the
  license-agreement page (re-`hf auth login` if the first chunk is HTML).
- For LeRobot datasets, `meta/info.json` parses and the episode count
  matches the card.

## Troubleshooting

- **`Repo gated. Cannot access … 401`** — open the dataset URL in a
  browser, click _Agree_ on the license / contact-info form, **then** retry
  with the same token. Tokens don't get auto-refreshed when a new
  agreement appears (re-accept after major version bumps).
- **First chunk is 5 KB of HTML** — same cause as above.
- **Download hangs indefinitely on AV (133 TB)** — you almost certainly
  don't want the whole thing. Use `physical_ai_av` to filter
  `platform_class == 'hyperion_8.1'` (default — see
  [Filtered AV download recipe](#filtered-av-download-recipe)), plus
  any sensor / country / split mask, BEFORE pulling.
- **AV clip downloaded but downstream NCore / NuRec / Asset-Harvester
  rejects it** — double-check `data_collection['platform_class']` for
  that clip; only `hyperion_8.1` is validated. If it's `hyperion_8`,
  either swap to a `hyperion_8.1` clip or accept that the downstream
  reconstruction tooling will not work.
- **Cosmos-Drive-Dreams 3 TB on a small disk** — pass
  `--file_types synthetic` (700 GB) or `--file_types hdmap` (small) to
  the official `download.py`.
- **Lyra-SDG 25 TB out of disk** — `hf download --include "tar/static_*"
--exclude "tar/dynamic_*"` to take just the 3D half (or vice-versa).
- **Cosmos-Synthetic looks empty (2.59 kB)** — it's a pointer page
  redirecting to `Cosmos-Drive-Dreams`. Use that.
- **mindmap dataset only has 10 demos when the card says 200 / 1000** —
  expected; regenerate the rest from the HDF5 via the upstream
  `mindmap` data-generation docs.
- **AV license expired (12 months)** — re-accept on the HF page; this
  is by-design per the License Agreement section 7.

## Limitations

- **Catalogue-only.** This skill does not actually run any
  pipeline — it routes the agent to the consuming tool once the right
  dataset is chosen.
- **NVIDIA-only.** Non-NVIDIA datasets (Waymo, nuScenes, KITTI,
  PandaSet, …) are out of scope; the `ncore` skill covers how to
  convert _those_ into NCore V4 manually.
- **Gating is enforced by Hugging Face, not by this skill.** A minority of
  datasets — the `PhysicalAI-Autonomous-Vehicles*` family in particular —
  require _both_ license acceptance on the HF page _and_ a valid `HF_TOKEN`,
  and those acceptances expire (AV is annual) and must be re-clicked. The
  **four `mindmap` repos are public and ungated**; they are CC-BY-NC, which
  restricts _use_, not _access_. Do not assume a dataset is gated because it
  is non-commercial.
- **Footprint can be enormous.** PhysicalAI-Autonomous-Vehicles is
  ~133 TB; Lyra-SDG ~25 TB; Cosmos-Drive-Dreams ~3 TB. Always check
  free disk and use `--include` / `--exclude` filters before kicking
  off a full pull.
- **The catalogue drifts.** NVIDIA ships new `PhysicalAI-*` datasets
  regularly. If a dataset isn't in this skill, browse
  <https://huggingface.co/nvidia?search_datasets=PhysicalAI-> and
  consider opening a PR to update Section 2 / the relevant family.
- **No GPU / runtime checks here.** Storage and HF auth are the only
  prerequisites this skill validates; per-dataset compute needs
  belong to the consuming tool.

## Source links

- HF org (filter `PhysicalAI-`): <https://huggingface.co/nvidia>
- Curated collection (29 items): <https://huggingface.co/collections/nvidia/physical-ai>
- Cosmos Dataset Search (semantic preview of AV): <https://build.nvidia.com/nvidia/cosmos-dataset-search>
- AV Python toolkit: <https://github.com/NVlabs/physical_ai_av>
- NCore: <https://github.com/NVIDIA/ncore>
- Cosmos-Drive-Dreams downloader: <https://github.com/nv-tlabs/Cosmos-Drive-Dreams/blob/main/scripts/download.py>
- Lyra repo: <https://github.com/nv-tlabs/lyra>
- mindmap repo: <https://github.com/nvidia-isaac/nvblox_mindmap>
- Isaac-GR00T repo: <https://github.com/NVIDIA/Isaac-GR00T>
- DreamDojo project: <https://dreamdojo-world.github.io/>
- AI City Challenge eval: <https://eval.aicitychallenge.org/>
