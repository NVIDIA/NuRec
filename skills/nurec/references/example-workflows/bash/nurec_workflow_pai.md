<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-End Workflow: PAI → NCore → NuRec → Renderer

> Pipeline scope covered here: PAI → CDS (optional) → NCore → auxiliary
> data → NuRec → NuRec Renderer.

<!-- ========================================================================== -->
<!-- SECTION: Goal -->
<!-- ========================================================================== -->

<details open>
<summary><strong><span style="color:#76B900">Goal</span></strong></summary>

This guide covers downloading a bounded Physical AI clip set, converting it to NCore,
generating auxiliary data, reconstructing with NuRec, and rendering the result.
For a step-by-step overview, see the [AV quickstart](https://docs.nvidia.com/nurec/av/index.html).
For the automated end-to-end path, prefer the `physical-ai-av-benchmark` skill when
available, or the released `//workflows/av_benchmark:run` workflow. Its runner
options are documented in `$NUREC_SRC/workflows/av_benchmark/README.md`.
Hosted NuRec links track the latest published docs and may differ from this
checkout.

</details>

<!-- ========================================================================== -->
<!-- SECTION: Pre-requisites -->
<!-- ========================================================================== -->

<details open>
<summary><strong><span style="color:#76B900">Pre-requisites</span></strong></summary>

Complete [source setup](../../install.md): set `NUREC_SRC`, select the matching
GitHub-hosted `NUREC_DEV_DEPS_IMAGE`, build `NUREC_IMAGE`, and source
`scripts/nurec_dev.sh`.
For external dataset/output directories, add same-path mounts with
`NUREC_DOCKER_ARGS+=(--volume /data:/data)`. The separate NCore
download tools below need their own Python environment.

<!-- Accounts and Tokens (subsection of Pre-requisites) -->

<details open>
<summary><strong><span style="color:#5E5E5E">Accounts and Tokens</span></strong></summary>
You will also need the following accounts and tokens:

1. A Huggingface account with a valid HF_TOKEN (accept the selected PAI dataset terms)

</details>

<!--Hardware (subsection of Pre-requisites) -->

<details open>
<summary><strong><span style="color:#5E5E5E">Hardware</span></strong></summary>

Supported GPU families (driver baseline R580+ for the source release):

| Architecture | Boards                    | GPU codename(s) | Minimum driver |
| ------------ | ------------------------- | --------------- | -------------- |
| Ampere       | A100, A10, A40, RTX A6000 | GA100, GA102    | R580           |
| Ada Lovelace | L20, L40, L40S            | AD102           | R580           |
| Grace Hopper | H20, H100                 | GH100           | R580           |
| Blackwell    | RTX Pro 6000D             | GB202           | R580           |

</details>

</details>

<!-- ========================================================================== -->
<!-- SECTION: End-to-end Workflow -->
<!-- ========================================================================== -->

<details open>
<summary><strong><span style="color:#76B900">End-to-end Workflow</span></strong></summary>

There are 5 main steps to this pipeline, plus 1 optional steps:

| #               | Subsection                                                              |
| --------------- | ----------------------------------------------------------------------- |
| 1               | [Download raw data](#1-download-raw-data)                               |
| 1a _(Optional)_ | [Curate with CDS](#1a-optional-curate-with-cosmos-dataset-search-cds)   |
| 2               | [Convert data to NCore](#2-convert-data-to-ncore)                       |
| 3               | [Generate Auxiliary Data](#3-generate-auxiliary-data)                   |
| 4               | [Run Neural Reconstruction](#4-run-neural-reconstruction)               |
| 5               | [Run NuRec Renderer](#5-visualize-your-reconstructed-scenes-with-nurec) |

</details>

<!-- -------------------------------------------------------------------------- -->
<!-- Step 1: Download raw data -->
<!-- -------------------------------------------------------------------------- -->

## 1. Download raw data

The script for downloading the data samples is located in the NCore github repository. You'll need to clone it and install the package using the following commands:

<details><summary>▸ View command</summary>

```bash
ENV_NAME=ncore-env

git clone https://github.com/NVIDIA/ncore.git
uv venv "${ENV_NAME:?}"
source "${ENV_NAME:?}/bin/activate"
cd ncore
uv pip install nvidia-ncore
```

</details>

You'll then be able to download PAI clips (see step [1a](#1a-optional-curate-with-cosmos-dataset-search-cds) for finding a clip-id specific to your needs), see the metadata for a specific clip, or list available features using the following commands:

<details><summary>▸ View command</summary>

```bash
CLIP_ID=02a6f330-5ab0-4e92-99d4-d19e406952f4
DATA_DIR=/path/to/data/storage

# Download one or more clips to a local directory (repeat the id to add more)
bazel run \
  //tools/data_converter/pai/pai_remote:pai-clip-dl \
  -- \
  download "${CLIP_ID:?}" \
  --output-dir "${DATA_DIR:?}"

# Show clip metadata and sensor presence
bazel run \
  //tools/data_converter/pai/pai_remote:pai-clip-dl \
  -- \
  info "${CLIP_ID:?}"

# List all available features in the dataset
bazel run \
  //tools/data_converter/pai/pai_remote:pai-clip-dl \
  -- \
  list-features
```

</details>

The data downloaded has the following layout:

```
{output_dir}/{clip_id}/
├── calibration/
│   ├── camera_intrinsics.parquet
│   ├── sensor_extrinsics.parquet
│   ├── vehicle_dimensions.parquet
│   └── lidar_intrinsics.parquet      (optional)
├── labels/
│   ├── {clip_id}.egomotion.parquet
│   └── {clip_id}.obstacle.parquet    (optional)
├── camera/
│   ├── {clip_id}.{camera_id}.mp4
│   ├── {clip_id}.{camera_id}.timestamps.parquet
│   └── {clip_id}.{camera_id}.blurred_boxes.parquet
├── lidar/
│   └── {clip_id}.lidar_top_360fov.parquet
└── metadata/
    ├── sensor_presence.parquet
    ├── data_collection.parquet
    └── provenance.json               (download source, optional)
```

> Sample raw-data preview omitted from this lightweight skill copy.

<!-- -------------------------------------------------------------------------- -->
<!-- Step 1a (Optional): Curate with CDS -->
<!-- -------------------------------------------------------------------------- -->

## 1a. (Optional) Curate with Cosmos Dataset Search (CDS)

Cosmos Dataset Search (CDS) is a suite of visual search and analytics micro-services to ingest, index, search and curate multi-modal data with a focus on video understanding and temporal reasoning. For more information visit [here](https://docs.nvidia.com/cosmos/cds/latest/introduction.html).

To use CDS, run:

<details><summary>▸ View command</summary>

```bash
git clone https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search
cd cosmos-dataset-search
```

</details>

Install CDS using:

<details><summary>▸ View command</summary>

```bash
make build-docker
make install
make install-cds-cli
source .venv/bin/activate
```

</details>

Setup environment variables required for CDS bring up. Edit `deploy/standalone/.env.example` and set:

- **NVIDIA_API_KEY** — Only for this optional CDS deployment; follow the public CDS documentation for its service access requirements. This is not a NuRec prerequisite.
- **DATA_DIR** — Path to your data (e.g. `$(pwd)` if running from the cosmos-dataset-search root, or the absolute path to the directory containing `PhysicalAI-AV-sample`)

Then run:

<details><summary>▸ View command</summary>

```bash
# Copy template if .env doesn't exist
cp deploy/standalone/.env.example deploy/standalone/.env

# Edit deploy/standalone/.env and set:
#   NVIDIA_API_KEY="nvapi-your-actual-key"
#   DATA_DIR="/path/to/cosmos-dataset-search"   # or $(pwd) when run from repo root

make test-integration-up
```

</details>

Install HuggingFace client, authenticate, download sample dataset, extract videos:

<details><summary>▸ View command</summary>

```bash
pip install -U "huggingface_hub[cli]"

# `hf auth login --token ...` puts the token in argv (visible to `ps`) and
# --add-to-git-credential writes it into your git credential store. Neither is
# needed: every hf command reads HF_TOKEN from the environment.
export HF_TOKEN=...   # or: read -rs HF_TOKEN && export HF_TOKEN

# Download PAI sample dataset
hf download nvidia/PhysicalAI-Autonomous-Vehicles \
  --repo-type dataset \
  --include "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0001.zip" \
  --local-dir ./PhysicalAI-AV-sample

VIDEO_DIR=PhysicalAI-AV-sample/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0001
unzip -d $VIDEO_DIR PhysicalAI-AV-sample/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0001.zip
rm PhysicalAI-AV-sample/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0001/*parquet
```

</details>

Encode videos to H264:

<details><summary>▸ View command</summary>

```bash
VIDEO_DIR_H264=${VIDEO_DIR}_h264
mkdir -p $VIDEO_DIR_H264
ls $VIDEO_DIR | xargs -I {} ffmpeg -i ${VIDEO_DIR}/{} -c:v libx264 -preset medium -crf 23 -c:a aac -movflags +faststart ${VIDEO_DIR_H264}/{}
```

</details>

Build csv dataset (create `build_pai_csv_dataset.py`):

<details><summary>▸ View python code</summary>

```python
#!/usr/bin/env python3
# Generate a dataset file (CSV or JSONL) from a directory of video files for use
# with `make ingest --dataset-file ... --video-dir ...`. Required columns:
#   video_id, video, caption
# (or use --id-field, --video-field, --text-field to match custom names.)

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build a dataset file (CSV or JSONL) from a directory of video files."
    )
    parser.add_argument(
        "video_dir",
        type=Path,
        help="Directory containing video files (e.g. .mp4)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <video_dir>/dataset.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of videos to include (0 = no limit)",
    )
    args = parser.parse_args()

    video_dir = args.video_dir.expanduser().resolve()
    if not video_dir.is_dir():
        print(f"Error: not a directory: {video_dir}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        args.output = video_dir / "dataset.csv"
    out_path = args.output.expanduser().resolve()

    rows = []
    for p in sorted(video_dir.iterdir()):
        if p.suffix.lower() not in [".mp4"]:
            continue
        # Use first segment of stem as id (e.g. uuid from "uuid.camera_front_wide_120fov")
        video_id = p.stem.split(".")[0] if "." in p.stem else p.stem
        video_basename = p.name
        rows.append(
            {"video_id": video_id, "video": str(p), "caption": "driving clip"}
        )
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print(f"No video files found in {video_dir}", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "video", "caption"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Example ingest: make ingest INGEST_FLAGS='--dataset-file {out_path} --video-dir {video_dir} --collection-name test --limit 10'")


if __name__ == "__main__":
    main()
```

</details>

Run `build_pai_csv_dataset.py`. If there are failures, restart with `make test-integration-down && make test-integration-up`.

<details><summary>▸ View command</summary>

```bash
python build_pai_csv_dataset.py $VIDEO_DIR_H264 -o pai_dataset.csv

make ingest INGEST_FLAGS="--dataset-file pai_dataset.csv --video-dir $VIDEO_DIR_H264 --collection-name 'pai-sample' --s3-endpoint http://localstack:4566"
```

</details>

**Note:** Using `localhost` or host IP will cause "Invalid video format or request format" errors, because the cosmos-embed container cannot reach those addresses from inside Docker. Use the Docker bridge gateway (`172.17.0.1`; verify with `ip addr show docker0`). Use `--limit N` for faster testing with fewer videos.

<details><summary><strong>Troubleshooting: Ingestion fails with "Could not connect to the endpoint URL"</strong></summary>

If ingestion fails with `Could not connect to the endpoint URL: "http://localstack:4566/..."`:

1. **Ensure LocalStack is running** — Start the stack if needed:

   ```bash
   docker compose -f deploy/standalone/docker-compose.build.yml up -d
   ```

   Wait until LocalStack is healthy before re-running ingest.

2. **Use the Docker bridge gateway** — When running `make ingest` from the host, `localstack` or `localhost` may not work. Use `172.17.0.1` as shown in the ingest command above (verify with `ip addr show docker0`).

3. **Quick checks:**

   ```bash
   # Is LocalStack running?
   docker ps | grep localstack

   # Can you reach it?
   curl http://localhost:4566/_localstack/health

   # If using localstack hostname, ensure it resolves (add to /etc/hosts if needed):
   grep localstack /etc/hosts
   # If missing: echo "127.0.0.1   localstack" | sudo tee -a /etc/hosts
   ```

</details>

A successful run will look like this:

<details><summary>▸ View output</summary>

```bash
Collection ingestion completed successfully!
Collection ID: a3c19d176_7ab5_43e3_94ee_efb7dc30ab04
Collection Name: pai-sample
Pipeline: cosmos_video_search_milvus
```

</details>

Query with Cosmos Dataset Search:

<details><summary>▸ View command</summary>

```bash
cds collections list
#### grab the id from:
# INFO:root:Loading profile default
# {
#   "collections": [
#     {
#       "pipeline": "cosmos_video_search_milvus",
#       "name": "pai-sample Accuracy Test",
#       "tags": {
#         "default_index": "GPU_CAGRA"
#       },
#       "init_params": null,
#       "cameras": "camera_front_wide_120fov",
#       "id": "f7ca1ed6_8d71_4770_b7d9_7873a5961a40",
#       "created_at": "2026-03-13T23:30:29.216601"
#     }
#   ]
# }
#here the id is: "f7ca1ed6_8d71_4770_b7d9_7873a5961a40
cds search --collection-ids f7ca1ed6_8d71_4770_b7d9_7873a5961a40 --text-query "sunny day" --top-k 3
```

</details>

Which will give you your desired `id`. Here is a sample output:

<details><summary>▸ View output</summary>

```text
INFO:root:Loading profile default
{
  "retrievals": [
    {
      "id": "adf546a8e80aa7adb48263f105bdc9a16bf6456d489f25a6dbe7bec1ce4ec5bb",
      "metadata": {
        "filename": "4e53b568-3dcd-46c6-a3d9-2ece8e3ba8a1.camera_front_wide_120fov.mp4",
        "video_id": "/data/cosmos-dataset-search/PhysicalAI-AV-sample/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0001_h264/4e53b568-3dcd-46c6-a3d9-2ece8e3ba8a1.camera_front_wide_120fov.mp4",
        "source_id": "3336ccc5b83380dfe4a2612695677a6e69cfcbfbf99ce432e5af09772cda4c3b",
        "indexed_at": "2026-03-12T08:40:21.676079",
        "source_url": "http://localstack:4566/cosmos-test-bucket/pai-sample/4e53b568-3dcd-46c6-a3d9-2ece8e3ba8a1.camera_front_wide_120fov.mp4?AWSAccessKeyId=test&Signature=Uj%2FrDx%2BsRLXo2nLsbx8l4tju93A%3D&Expires=1773308421"
      },
      "collection_id": "a3c19d176_7ab5_43e3_94ee_efb7dc30ab04",
      "asset_url": null,
      "score": 0.29800325632095337,
      "content": "",
      "mime_type": "video/mp4",
      "embedding": null
    },
    {
      "id": "f5e3f376e374fd3dfcc597915ab18171ac4f711fbbe80aa804e88b8226fbf3df",
      "metadata": {
        "filename": "9571d238-9866-43dd-a9ea-a629d725a7b0.camera_front_wide_120fov.mp4",
        "video_id": "/data/cosmos-dataset-search/PhysicalAI-AV-sample/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0001_h264/9571d238-9866-43dd-a9ea-a629d725a7b0.camera_front_wide_120fov.mp4",
        "source_id": "4baede079033073c1ced9992ff94a316e79c0fcaceb989d70b884cfb96723728",
        "indexed_at": "2026-03-12T08:40:44.969826",
        "source_url": "http://localstack:4566/cosmos-test-bucket/pai-sample/9571d238-9866-43dd-a9ea-a629d725a7b0.camera_front_wide_120fov.mp4?AWSAccessKeyId=test&Signature=zH40FFDZ4yIm70R4V6PlyECyy6U%3D&Expires=1773308441"
      },
      "collection_id": "a3c19d176_7ab5_43e3_94ee_efb7dc30ab04",
      "asset_url": null,
      "score": 0.2942220866680145,
      "content": "",
      "mime_type": "video/mp4",
      "embedding": null
    },
    {
      "id": "c43a1d5026b2df204461ef41c9edb5ce4741e583ed5f74473644430c02acbbb2",
      "metadata": {
        "filename": "7daa5276-7789-4a5a-b881-ec3bfeb3c7e1.camera_front_wide_120fov.mp4",
        "video_id": "/data/cosmos-dataset-search/PhysicalAI-AV-sample/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0001_h264/7daa5276-7789-4a5a-b881-ec3bfeb3c7e1.camera_front_wide_120fov.mp4",
        "source_id": "2e150af535dfdeeea45bc75da0f8e7668dd70a8f42a4f2bcd310ea6d8dc849b1",
        "indexed_at": "2026-03-12T08:40:37.472764",
        "source_url": "http://localstack:4566/cosmos-test-bucket/pai-sample/7daa5276-7789-4a5a-b881-ec3bfeb3c7e1.camera_front_wide_120fov.mp4?AWSAccessKeyId=test&Signature=FiahMTx6MLthoAXdNoCrEDbQ2yg%3D&Expires=1773308437"
      },
      "collection_id": "a3c19d176_7ab5_43e3_94ee_efb7dc30ab04",
      "asset_url": null,
      "score": 0.28784406185150146,
      "content": "",
      "mime_type": "video/mp4",
      "embedding": null
    }
  ]
}
```

</details>

<!-- -------------------------------------------------------------------------- -->
<!-- Step 2: Convert data to NCore -->
<!-- -------------------------------------------------------------------------- -->

## 2. Convert data to NCore

NCore, in the context of NuRec, refers to NVIDIA’s standardized data format for autonomous vehicle and robotics datasets. It provides a unified schema for storing sensor data, annotations, and metadata required for neural reconstruction and simulation. The NCore format includes data types that define sensor rig configurations, camera, lidar, and radar setups in the NuRec scenarios, and metadata and session information. Converting data to the standardized NCore format supports consistent, quality reconstruction and the highest quality output through NuRec rendering.

There are two conversion modes:

- **`pai-v4`** -- converts clips previously downloaded to disk with `pai-clip-dl`.
- **`pai-stream-v4`** -- convert clips directly from HuggingFace without downloading.

### Local mode (`pai-v4`)

<details><summary>▸ View command</summary>

```bash
ROOT_DIR=/path/to/downloaded/clips  # parent folder of the clips fetched with pai-clip-dl
NCORE_OUT=/path/to/ncore/output     # folder for the converted NCore data
CLIP_ID=02a6f330-5ab0-4e92-99d4-d19e406952f4

nurec_dev bazelisk run //apps:nurec_tools -- pai-convert \
    --root-dir "${ROOT_DIR:?}" \
    --output-dir "${NCORE_OUT:?}" \
    pai-v4 \
    --clip-id "${CLIP_ID:?}"
```

</details>

### Streaming mode (`pai-stream-v4`)

Converts clips directly from HuggingFace. No prior download required.

<details><summary>▸ View command</summary>

```bash
NCORE_OUT=/path/to/ncore/output
CLIP_ID=02a6f330-5ab0-4e92-99d4-d19e406952f4

# HF_TOKEN is read from the environment (--hf-token has envvar="HF_TOKEN"),
# so the token never lands in argv where `ps` can read it.
export HF_TOKEN=...

nurec_dev bazelisk run //apps:nurec_tools -- pai-convert \
    --output-dir "${NCORE_OUT:?}" \
    pai-stream-v4 \
    --clip-id "${CLIP_ID:?}"
```

</details>

### Arguments

- `--root-dir` is required by the base CLI but ignored in streaming mode (set to any placeholder).
- `--clip-id` is **required** for both `pai-v4` and `pai-stream-v4` and can be specified multiple times.
- `--hf-token` reads from the `HF_TOKEN` environment variable if not provided on the command line.
- `--revision` selects the HuggingFace dataset branch/tag (default: `main`).
- `--seek-sec <float>` -- skip this many seconds from the start
- `--duration-sec <float>` -- restrict total duration in seconds
- `--store-type itar|directory` -- output format (default: `itar`)
- `--profile default|separate-sensors|separate-all` -- component group layout (NuRec prefers `separate-sensors`)
- `--sequence-meta / --no-sequence-meta` -- generate sequence metadata JSON (NuRec/`ncore_vis` need this; on by default)
- `--no-cameras` -- Disable exporting all camera sensors
- `--camera-id ID` -- Export only the specified camera (repeatable; defaults to all cameras)
- `--no-lidars` -- Disable exporting all lidar sensors
- `--verbose` -- Enable debug-level logging

The output is an NCore v4 shard per clip at:

```
<output-dir>/pai_<clip-id>/pai_<clip-id>.ncore4.zarr.itar
```

> The NCore visualization tool can visualize all sensor modalities. See <https://nvidia.github.io/ncore/tools/data_vis.html>.

<!-- -------------------------------------------------------------------------- -->
<!-- Step 3: Generate Auxiliary Data -->
<!-- -------------------------------------------------------------------------- -->

## 3. Generate Auxiliary Data

The Neural Reconstruction (NuRec) engine requires additional information to reconstruct scenes with real-world data. This dataset, called NuRec Auxiliary Data, includes the following required and optional data types:

1. Semantic Segmentation Data (conditionally required)
2. Depth Estimation Data (optional)
3. DINOv2 Feature Extraction (optional)
4. LiDAR Segmentation and Visibility (optional, recommended)
5. Metadata and Configuration (required)

Generate auxiliary data with the source target `//apps:nurec_tools`.

You'll then run:

<details><summary>▸ View command</summary>

```bash
SHARD_DIR=/path/to/ncore/shards
AUX_OUT=/path/to/aux/output
CLIP_ID=02a6f330-5ab0-4e92-99d4-d19e406952f4

mkdir -p "$AUX_OUT"
NUREC_DOCKER_ARGS+=(--volume "$SHARD_DIR:$SHARD_DIR" --volume "$AUX_OUT:$AUX_OUT")

# The cameras and lidar of the AV Benchmark spec's aux task.
nurec_dev bazelisk run //apps:nurec_tools -- \
    ncore-aux-data \
    --dataset-path="$SHARD_DIR/pai_${CLIP_ID:?}.json" \
    --output-dir=$AUX_OUT \
    --camera-id=camera_front_wide_120fov \
    --camera-id=camera_front_tele_30fov \
    --camera-id=camera_cross_right_120fov \
    --camera-id=camera_cross_left_120fov \
    --camera-id=camera_rear_left_70fov \
    --camera-id=camera_rear_right_70fov \
    --camera-id=camera_rear_tele_30fov \
    --lidar-id=lidar_top_360fov \
    --store-meta
```

</details>

Notes:

- Point the dataset volume mount to the directory where the `.zarr.itar` and `.json` file are saved from the previous section.
- It is recommended to write the auxiliary data to the same directory as the source NCore shard so NuRec training picks it up automatically; either pass the same host path for both `--volume` bind mounts, or move/symlink the resulting `*.aux.*.zarr.itar` files next to the original shard before training.
- Use the `--camera-id=<ID1>` `--camera-id=<ID2>` flags to pass camera IDs multiple times. If you don't pass any camera IDs, the model generates auxiliary data for all cameras.
- Camera IDs can be found in the JSON file generated by NCore.
- You can use the `--num-threads <N>` flag (or `--num-threads auto`) to increase the number of CPU threads and generate the auxiliary data faster.

Files generated by above command will have .aux added to the filename: `<clip_id>.aux.<>.zarr.itar`.

<!-- -------------------------------------------------------------------------- -->
<!-- Step 4: Run Neural Reconstruction -->
<!-- -------------------------------------------------------------------------- -->

## 4. Run Neural Reconstruction

Use the matching source checkout for reconstruction.

<details><summary>▸ Source target</summary>

```bash
nurec_dev bazelisk build //:run
```

</details>

For PAI clips, train with the **AV Benchmark preset** shipped in the
checkout, `workflows/av_benchmark/configs/av_benchmark_lidar_7cam.yaml`,
and the arguments of the `train` task in
`workflows/av_benchmark/specs/av_benchmark_7cam.yaml` — **not** the
Waymo recipe. A clip trained this way matches an AV Benchmark run of the
same source release.

To begin training the reconstruction model in the Docker container, edit the following command to reflect the correct variables and then run it:

<details><summary>▸ View command</summary>

```bash
DATASET_DIR=/path/to/ncore/shards                # NCore shard + auxiliary data directory
OUTPUT_DIR=/path/to/nurec/output                 # folder for the reconstructed scenes
CLIP_ID=02a6f330-5ab0-4e92-99d4-d19e406952f4     # clip to reconstruct
PRESET_DIR=$NUREC_SRC/workflows/av_benchmark/configs  # absolute path in the source checkout
CAMERAS='[camera_front_wide_120fov,camera_front_tele_30fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_rear_left_70fov,camera_rear_right_70fov,camera_rear_tele_30fov]'

mkdir -p "$OUTPUT_DIR"
NUREC_DOCKER_ARGS+=(--volume "$DATASET_DIR:$DATASET_DIR" --volume "$OUTPUT_DIR:$OUTPUT_DIR")
nurec_dev bazelisk run //:run -- \
  main \
  --config-name=${PRESET_DIR}/av_benchmark_lidar_7cam.yaml \
  dataset.path="$DATASET_DIR/pai_${CLIP_ID:?}.json" \
  out_dir=$OUTPUT_DIR \
  mode=train \
  logger.name=dummy \
  logger.run_id=latest \
  dataset.n_samples_per_epoch=30000 \
  "dataset.camera_ids=${CAMERAS}" \
  "dataset.train_camera_ids=${CAMERAS}" \
  "dataset.val_camera_ids=${CAMERAS}"
```

</details>

Notes:

- The preset and the spec are in the checkout under `$NUREC_SRC/workflows/av_benchmark/`; `PRESET_DIR` above is the absolute path (the helper's working directory is `$NUREC_SRC`).
- The preset writes the USDZ artifact that `serve-grpc` / `render-grpc` consume in the next step.
- The camera list must be a subset of the cameras you ran the aux step on. Dropping `camera_rear_tele_30fov` gives the six cameras of the published NuRec dataset.
- Update the `--volume <path to your NCore shard, and auxiliary data directory>:$DATASET_DIR` flag to point at the folder containing the `.zarr.itar`, `<clip_id>.aux.*.zarr.itar`, and `<clip_id>.json` files.
- The `-u "$(id -u):$(id -g)"` flag keeps all outputs (USDZ, MP4s, metrics) owned by the host user — without it everything lands as `root`.

For more information regarding Neural Reconstruction and its options please visit the [NuRec public docs](https://docs.nvidia.com/nurec/).

After completion of this step, you will have your reconstructed scene at `$OUTPUT_DIR/<RUN-ID>/artifacts/last.usdz` (the `<RUN-ID>` matches `logger.run_id`, so `latest` in the example above).

<!-- -------------------------------------------------------------------------- -->
<!-- Step 5: NuRec Renderer -->
<!-- -------------------------------------------------------------------------- -->

## 5. Visualize your reconstructed Scenes with NuRec

After [step 4](#4-run-neural-reconstruction), you have a 3D reconstructed scene at `<output-dir>/<RUN-ID>/artifacts/last.usdz`. To visualize and render it, use the NuRec gRPC renderer in two stages:

1. **Start the gRPC server** — serves your USDZ artifact so the renderer can access it.
2. **Run the renderer** — connects to the server and renders frames to disk.

**Prerequisite:** The gRPC server must be running in a separate terminal before you run the renderer.

In a fresh terminal, complete [source setup](../../install.md), then start the server:

<details><summary>▸ View command</summary>

```bash
OUTPUT_DIR=/path/to/nurec/output  # out_dir used in step 4
RUN_ID=latest                     # matches logger.run_id from step 4
ARTIFACT_PATH="$OUTPUT_DIR/$RUN_ID/artifacts/last.usdz"
test -s "$ARTIFACT_PATH" || { echo "Missing reconstructed scene: $ARTIFACT_PATH" >&2; exit 1; }
NUREC_DOCKER_ARGS+=(--volume "$OUTPUT_DIR:$OUTPUT_DIR")

nurec_dev bazelisk run //:run -- \
    serve-grpc \
    --artifact-glob "$ARTIFACT_PATH" \
    --renderer default \
    --enable-editing-actors \
    --test-scenes-are-valid
```

</details>

Notes:

- `--renderer default` selects the renderer the artifact was trained with; `--renderer gsplat` forces the GSplat rasterizer. NRend has been removed, and a legacy `*-nrend` artifact is migrated to its GSplat equivalent by the config upgrade.
- Quote `--artifact-glob` so the shell does not expand the glob before Docker sees it.

Leave this terminal running.

In another fresh terminal, complete [source setup](../../install.md), then run the renderer:

<details><summary>▸ View command</summary>

```bash
RENDER_OUT=/path/to/render/output  # host folder for the rendered frames
OUTPUT_DIR=/path/to/nurec/output   # out_dir used in step 4
RUN_ID=latest                      # matches logger.run_id from step 4
ARTIFACT_PATH="$OUTPUT_DIR/$RUN_ID/artifacts/last.usdz"
test -s "$ARTIFACT_PATH" || { echo "Missing reconstructed scene: $ARTIFACT_PATH" >&2; exit 1; }
mkdir -p "$RENDER_OUT"
NUREC_DOCKER_ARGS+=(--volume "$OUTPUT_DIR:$OUTPUT_DIR" --volume "$RENDER_OUT:$RENDER_OUT")

nurec_dev bazelisk run //:run -- \
  render-grpc \
  --artifact-path "$ARTIFACT_PATH" \
  --output-dir "$RENDER_OUT" \
  --camera-id camera_front_wide_120fov
```

</details>

Rendered frames are written to the output directory specified by `--output-dir`. Run a separate `render-grpc` invocation for each additional sensor, setting `--camera-id` and a distinct `--output-dir` for each invocation. For purely local rendering with no gRPC server, swap `render-grpc` for the in-container `render` sub-command — see the [NuRec CLI reference](https://docs.nvidia.com/nurec/).

> Reconstructed-scene preview omitted from this lightweight skill copy.
