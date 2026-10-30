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
- `common_pb2_grpc.py` and `sensorsim_pb2_grpc.py`: gRPC service code

### 3. Use in Your Code

After generating the Python files, you can import and use them in your Python code:

For a remote server, start `serve-grpc` with `--host <SERVER_IP>` and allow TCP port 8080, then replace `localhost:8080` below with `<SERVER_IP>:8080`.

```python
# Import the generated modules
import nurec.grpc.protos.common_pb2 as common_pb2
import nurec.grpc.protos.sensorsim_pb2 as sensorsim_pb2
import nurec.grpc.protos.sensorsim_pb2_grpc as sensorsim_pb2_grpc

# Create a gRPC channel and stub
import grpc
channel = grpc.insecure_channel('localhost:8080')
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
- `render_lidar`: Render lidar point clouds
- `open_ipc_session` / `close_ipc_session`: Open/close a CUDA IPC (same-node) session for
  zero-copy frame delivery to a client co-located on the same node and physical GPU. Once a
  session is open, set `use_cuda_ipc=true` and `ipc_session_id` on a `RGBRenderRequest` or
  `CameraRenderRequest` (`render_camera` supports only a single `CAMERA_RENDER_PRODUCT_COLOR`
  output this way, with no bounding-box overlay) to deliver into one of the session's
  server-owned buffers instead of `image_bytes`. `batch_render_rgb` supports
  `use_cuda_ipc=true` per item too (single-process mode only); `batch_render_camera` does
  not. Not supported at all with `--multiproc-workers`.
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
