<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Neural Reconstruction Engine gRPC Protocol Buffers

This package contains the protocol buffer definitions and Python generation tools for the NVIDIA Neural Reconstruction Engine (NuRec) gRPC API.

## Contents

- `nurec/grpc/protos/`: Protocol buffer definitions (.proto files)
  - `common.proto`: Common data types and structures
  - `sensorsim.proto`: Sensor simulation service definitions
  - `ipc.proto`: CUDA IPC (same-node) session control-plane messages, imported by `sensorsim.proto`
  - `rdma.proto`: RDMA (cross-node, RoCEv2) session control-plane messages, imported by `sensorsim.proto`
- `nurec/grpc/update_generated.py`: Python script to generate Python code from proto files
- `nurec/requirements.txt`: Required Python dependencies

## Quick Start

### 1. Install Dependencies

Install [Python 3.13](https://www.python.org/downloads/) on the client machine.
Extract the archive and run these commands from its root, which contains `nurec/`:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r nurec/requirements.txt
```

### 2. Generate Python Code

Run the generation script to create Python files from the proto definitions:

```bash
python nurec/grpc/update_generated.py
```

This will generate the following Python files in `nurec/grpc/protos/`:

- `common_pb2.py` and `common_pb2.pyi`: Python code for common.proto
- `sensorsim_pb2.py` and `sensorsim_pb2.pyi`: Python code for sensorsim.proto
- `ipc_pb2.py` and `ipc_pb2.pyi`: Python code for ipc.proto (message types only; its RPCs are
  declared on `SensorsimService` in sensorsim.proto, not a separate service)
- `rdma_pb2.py` and `rdma_pb2.pyi`: Python code for rdma.proto (message types only, like ipc.proto)
- `common_pb2_grpc.py` and `sensorsim_pb2_grpc.py`: gRPC service code

### 3. Use in Your Code

Before upgrading, review the [LiDAR consumer changes](LIDAR_MIGRATION.md)
in the source checkout, including scaled-scene point-cloud units and validation/status changes.

After generating the Python files, you can import and use them in your Python code:

For a remote server, start `serve-grpc` with `--host <SERVER_IP>` and allow TCP port 8080, then replace `localhost:8080` below with `<SERVER_IP>:8080`.

```python
# Import the generated modules
import nurec.grpc.protos.common_pb2 as common_pb2
import nurec.grpc.protos.sensorsim_pb2 as sensorsim_pb2
import nurec.grpc.protos.sensorsim_pb2_grpc as sensorsim_pb2_grpc

# Create a gRPC channel and stub
import grpc
channel = grpc.insecure_channel(
    'localhost:8080',
    options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
    ],
)
stub = sensorsim_pb2_grpc.SensorsimServiceStub(channel)

# Discover scenes registered by the server
response = stub.get_available_scenes(common_pb2.Empty(), timeout=60)
print(list(response.scene_ids))
channel.close()
```

## Service Overview

The NuRec gRPC API provides the following main services:

### SensorsimService

- `render_camera`: Render a camera image; pass one or more `render_products` to receive
  color, distance-colorized, and/or semantics-colorized outputs in a single response.
  Optional bounding-box overlay applies to every requested product.
- `batch_render_camera`: Render multiple cameras for the same scene in one call.
- `render_lidar`: Return a compensated point cloud in metres by default for every LiDAR.
  Explicit `LIDAR_OUTPUT_PER_RAY` requests return uncompensated, layer-major per-ray
  buffers -- distance, intensity, and one packed validity mask per return layer
  -- in the ray order of `get_lidar_firing_pattern`, including rays that got no return.
  `LIDAR_OUTPUT_PER_RAY` requires a pattern-backed LiDAR. Point-cloud requests must
  leave `return_types` empty and `include_firing_pattern` false.
  In per-ray mode, `LidarRenderRequest.return_types` requests the ordered layers;
  empty means `[LIDAR_RETURN_STRONGEST]`, while FIRST and STRONGEST may be requested
  separately or together. Explicit `LIDAR_RETURN_UNSPECIFIED` is invalid.
  The current renderer models only STRONGEST, so FIRST temporarily
  repeats the same distances, intensities, and validity mask rather than representing a
  distinct predicted return. A per-ray client reconstructs points itself with
  `xyz_sensor_at_firing[layer][i] = directions_sensor[i] * ray_distances_m[layer][i]`, which
  is what lets it apply its own motion compensation.
- `get_lidar_firing_pattern`: The firing geometry of one lidar -- ray directions, per-ray
  firing times, the optional scan grid, and optional packet layout. Packet-constant firing
  times are encoded once per packet; sensors with intra-packet timing retain per-ray times.
  By default the server selects a repeating pattern from `frame_start_us`; an explicit
  `firing_pattern_id` takes precedence. Omitting both is valid only for a singleton
  pattern; multi-template lookup returns `INVALID_ARGUMENT`. Each per-ray response echoes the
  selected ID so clients can cache pattern payloads. Set
  `LidarRenderRequest.include_firing_pattern` to embed a copy in the response. An ET30X
  one-layer per-ray response without geometry fits gRPC's default 4 MiB receive limit,
  but the pattern lookup itself exceeds it: directions alone occupy 5,244,480 bytes.
  Configure a larger receive limit for lookup, two-layer returns, and embedded patterns;
  the channel example above uses 50 MiB to match the server.
- `open_ipc_session` / `close_ipc_session`: Open/close a CUDA IPC (same-node) session for
  zero-copy frame delivery to a client co-located on the same node and physical GPU. Once a
  session is open, set `use_cuda_ipc=true` and `ipc_session_id` on a `RGBRenderRequest` or
  `CameraRenderRequest` (`render_camera` supports only a single `CAMERA_RENDER_PRODUCT_COLOR`
  output this way, with no bounding-box overlay) to deliver into one of the session's
  server-owned buffers instead of `image_bytes`. `batch_render_rgb` supports
  `use_cuda_ipc=true` per item too. `batch_render_camera` accepts 1–8 color images from
  one scene and session, with dimensions fixed when the session opens. It also supports
  opt-in `EVENT_ORDERED` completion: wait on each returned buffer's exported ready event
  before reading it. The default `COMPLETION_ON_RETURN` mode needs no producer-event wait.
  Fence client reads before sending `released_buffer_id`; unmap buffers before closing.
  IPC is not supported with `--multiproc-workers`.
- `open_rdma_session` / `register_receive_buffer` / `close_rdma_session`: Set up an RDMA
  (RoCEv2) session for frame delivery to a client on another node. Once a session is open and
  buffers are registered, set `use_rdma=true`, `rdma_session_id` and `rdma_buffer_id` on a
  `RGBRenderRequest` or `CameraRenderRequest` (same `render_camera` restrictions as CUDA IPC) to
  RDMA WRITE the frame into that client buffer instead of returning `image_bytes`. Requires a
  server built with `--config=rdma`; not supported in batches or with `--multiproc-workers`.
- `get_version`: Get version information
- `get_available_scenes`: List available scenes
- `get_loaded_scenes`: List cached scene instance counts and cache capacity
- `get_available_cameras`: List available cameras for a scene
- `get_available_trajectories`: List available trajectories for a scene
- `get_available_ego_masks`: List available ego masks
- `shut_down`: Gracefully shutdown the service

`render_rgb`, `batch_render_rgb`, `RGBRenderRequest`, `RGBRenderReturn`,
`BatchRGBRenderRequest`, and `BatchRGBRenderReturn` are deprecated. Migrate to
`render_camera` / `CameraRenderRequest` with
`render_products = [CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR]` for like-for-like behavior.

Use `serve-grpc --camera-only` to skip lidar initialization in single-process serving.
`render_lidar` and `get_lidar_firing_pattern` return `FAILED_PRECONDITION`.

API 0.0.10 adds `BatchCameraRenderRequest.prefer_camera_batching`, off by default.
For IPC batches, set it to true to try rendering all views in one gsplat call.
Views must share a frame interval, camera intrinsics (including logical camera ID),
and actor overrides. Rolling-shutter scenes with dynamic actors render sequentially
because actor state can depend on each camera's rays. Other incompatible batches
also render sequentially.

To request global shutter, supply explicit intrinsics with `shutter_type=GLOBAL`;
NuRec does not change the shutter model automatically. The hint preserves item
ordering and failure handling, but does not guarantee bit-identical pixels.
Global-shutter inference samples dynamic tracks at frame start, with or without batching.

NRE logs the first occurrence of each CUDA IPC camera-dispatch outcome and every 100th dispatch,
with cumulative counts. The `shut_down` RPC also logs final counts when at least one dispatch
occurred.
`native/batched_rays` identifies the direct global-shutter path; `native/per_camera_rays` identifies
the rolling-shutter path. For phase timing, also pass `--enable-timing`, `--timing-synchronize`, and
`--timing-logfile <path>`. Synchronization is diagnostic and can reduce throughput.

### Scene prefetch

API 0.0.9 adds `prefetch_scene` for local normal or compact Gaussian USDZs. Require
`scene_prefetch.enabled=True`. Put the scene ID in the request and `x-nre-scene-id` metadata. Put the
absolute local artifact path or `file://` URI in `x-nre-scene-url` metadata. You may omit scene
metadata when the scene ID identifies one registered scene. The server prepares one scene at a time
and queues one successor key. Duplicate hints are idempotent. A distinct nonblocking hint returns
`RESOURCE_EXHAUSTED` when both positions are occupied. A successful RPC means the hint was admitted,
not that preparation finished.

With `wait_for_capacity=True`, one caller may wait for the pending slot when
`scene_prefetch.blocking_admission=True`. This requires `--max-workers=2` or greater and is bounded by
the RPC deadline and `--backend-cache-wait-timeout-s`.

A request that needs a new backend can activate matching prefetched work on CUDA. Submit and consume
hints in order. Requesting a queued successor before its predecessor discards both hints and loads
the successor synchronously. Cache hits, source changes, and shutdown can also discard hints.
Failed preparation releases its slot. Unused ready models expire after
`--backend-cache-wait-timeout-s` seconds (120 by default). Missing, stale, failed, discarded, or
timed-out preparation falls back to synchronous loading; client cancellation and deadlines still
abort the request.

Keep registered artifact contents unchanged. Metadata-only changes are accepted after a content-hash
check. CUDA IPC sessions stay mapped across scene changes. Remote artifacts and multiprocess serving
do not support prefetch.

## Development

When modifying the .proto files, always regenerate the Python code:

- via Bazel:

  ```bash
  bazel run //nurec/grpc:update_generated
  ```

- via Python:

  ```bash
  python nurec/grpc/update_generated.py
  ```

## Support

For questions about this API, please contact NVIDIA support or refer to the official documentation.

## License

Copyright (c) 2025-2026 NVIDIA CORPORATION. All rights reserved.
