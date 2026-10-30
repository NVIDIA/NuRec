<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec Toolkit

A collection of tools for NuRec processing.

## Overview

- **ncore-aux-data**: Data preprocessing utility
- **mask-annotator**: GUI tool for mask annotation

All tools are **enabled by default** in `.bazelrc`.

## Building

### Binary (Recommended)

```bash
# Default build (all tools, see .bazelrc)
bazel build //apps:nurec_tools
```

### Container Images

```bash
bazel build //apps:nurec_tools_image_oci
```

## Tool Selection

All tools are enabled by default in `.bazelrc`:

```bash
build --//bazel/flags:ncore_aux_data=True
build --//bazel/flags:mask_annotator=True
```

To disable specific tools, override with `=False`:

```bash
# Build without mask_annotator
bazel build //apps:nurec_tools \
    --//bazel/flags:mask_annotator=False
```

**Flag Reference**

| Flag                             | Default | Description                 |
| -------------------------------- | ------- | --------------------------- |
| `--//bazel/flags:mask_annotator` | `True`  | Include mask_annotator tool |
| `--//bazel/flags:ncore_aux_data` | `True`  | Include ncore_aux_data tool |

## Usage

### FFmpeg Selection

**Customer Notice:** This container includes an LGPL-compatible limited-codec build of FFmpeg 9.0.1 (bundled via the `imageio-ffmpeg==0.6.0+nre6` wheel). Your use of FFmpeg 9.0.1 is governed by the [LGPL-2.1 license](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html). To use the full Linux FFmpeg 7.0.2 binary distributed with upstream `imageio-ffmpeg==0.6.0`, run with `--allow-ffmpeg-download`. This will download the appropriate Linux static binary from [ImageIO's imageio-binaries repository](https://github.com/imageio/imageio-binaries/tree/master/ffmpeg). Your use of the downloaded full FFmpeg binary will be governed by the [GNU GPL v3 or later](https://www.gnu.org/licenses/gpl-3.0.html) license.

NuRec tools use the imageio-ffmpeg package for video output. By default they invoke the bundled LGPL limited-codec FFmpeg 9.0.1 build. To use a site-provided or locally validated ffmpeg binary instead, pass `--ffmpeg-exe` before the tool command or set `USER_FFMPEG_EXE`:

```bash
bazel run //apps:nurec_tools -- --ffmpeg-exe /usr/local/bin/ffmpeg ncore-aux-data --help
USER_FFMPEG_EXE=/usr/local/bin/ffmpeg bazel run //apps:nurec_tools -- ncore-aux-data --help
```

`--allow-ffmpeg-download` opts in to the full (GPL-licensed) FFmpeg binary distributed with `imageio-ffmpeg==0.6.0`, which provides broader codec coverage than the bundled LGPL build. Approval is intentionally CLI-only — there is no environment variable or config setting for this path, so it cannot be silently inherited from a shell or checked-in config. The download source and cache root are controlled by NuRec configuration:

```bash
bazel run //apps:nurec_tools -- \
    --allow-ffmpeg-download \
    ncore-aux-data --help
```

The approved download path prints the GPLv3 and FFmpeg licensing notice before it fetches or uses the cached full binary.

### Ncore Aux Data

#### Binary Execution

```bash
# Show help
bazel run //apps:nurec_tools -- ncore-aux-data --help

# Example usage
bazel run //apps:nurec_tools -- ncore-aux-data \
    --dataset-path=/path/to/input/sequence.json \
    --output-dir=/path/to/output
```

#### Container Execution

```bash
bazel run //apps:load_nurec_tools_image_oci
docker run -it --rm --gpus all \
    -v /path/to/dataset:/path/to/dataset \
    -v /path/to/output:/path/to/output \
    ghcr.io/nvidia/nurec/nurec_tools:latest \
    ncore-aux-data \
    --dataset-path=/path/to/dataset/sequence.json \
    --output-dir=/path/to/output
```

### PAI Convert

Converts raw [PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
clips to NCore V4, either from a local clip tree (`pai-v4`) or streamed from the dataset
(`pai-stream-v4`).

#### Binary Execution

```bash
# Show help
bazel run //apps:nurec_tools -- pai-convert --help

# Local clip tree
bazel run //apps:nurec_tools -- pai-convert \
    --root-dir=/path/to/clips \
    --output-dir=/path/to/output \
    --camera-id=camera_front_wide_120fov \
    pai-v4 --clip-id=<clip-uuid>

# Streamed from the dataset (needs HF_TOKEN)
bazel run //apps:nurec_tools -- pai-convert \
    --output-dir=/path/to/output \
    --camera-id=camera_front_wide_120fov \
    pai-stream-v4 --clip-id=<clip-uuid> --duration-sec=2
```

#### Container Execution

```bash
bazel run //apps:load_nurec_tools_image_oci
docker run -it --rm --gpus all \
    -v /path/to/clips:/path/to/clips \
    ghcr.io/nvidia/nurec/nurec_tools:latest \
    -- pai-convert \
    --root-dir=/path/to/clips \
    --output-dir=/path/to/output \
    --camera-id=camera_front_wide_120fov \
    pai-v4 --clip-id=<clip-uuid>
```

Append AUX shards to the result with `ncore-aux-data` before training.

### Mask Annotator

#### Binary Execution

Make sure to export `DISPLAY=:0` before running:

```bash
bazel run //apps:nurec_tools -- mask-annotator
```

#### Container Execution

```bash
bazel run //apps:load_nurec_tools_image_oci
docker run -it --rm --gpus all ghcr.io/nvidia/nurec/nurec_tools:latest \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -- mask-annotator
```

## Notes

### X11 Display Configuration

For mask-annotator GUI functionality in containers, ensure your `$DISPLAY` environment variable matches an entry in `~/.Xauthority`:

```bash
# Check available displays
xauth list

# Set DISPLAY if needed
export DISPLAY=:0
```

### GPU Support

Container runs should always include `--gpus=all` otherwise you may get some missing `.so` errors.

## Adding New Tools to the Toolkit

### 1. Tool Configuration

Start by adding your tool to the configuration file:

**File:** `apps/tools.yaml`

- Add the import path
- Specify the function to import
- Optionally provide an alias for the function

### 2. Bazel Configuration

Create the necessary Bazel configuration files:

- **Flag:** Add a flag at `//bazel/flags` for your tool
- **Condition:** Add a condition at `//bazel/conditions` for your tool, and add your flag to
  `//bazel/conditions:all_tools`
- **Default:** Add your tool to the defaults in `.bazelrc`:

```bash
build --//bazel/flags:your_tool=True
```

### 3. Container Integration

#### Update Build Dependencies

**File:** `apps/BUILD.bazel`

Add your tool to the unified select in `NUREC_TOOLS_DEPS`:

```python
select({
    "//bazel/conditions:all_tools": [..., "<tool_pylib_target>"],
    "//bazel/conditions:<tool_condition_name>": ["<tool_pylib_target>"],
    "//conditions:default": [],
})
```

#### Update Entrypoint Generation

**File:** `//apps:gen_entrypoint`

Add your tool to the unified select:

```python
select({
    "//bazel/conditions:all_tools": ["ncore-aux-data", "mask-annotator", "<your-tool>"],
    "//bazel/conditions:<tool_condition_name>": ["<tool name from scripts/tools.yaml>"],
    "//conditions:default": [],
})
```
