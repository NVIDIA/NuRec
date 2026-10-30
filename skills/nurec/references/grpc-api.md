<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec sensorsim gRPC API

Complete [source setup](install.md) before running the commands below.

NuRec exposes its rendering pipeline as a gRPC service so external
simulators (CARLA, AlpaSim, Isaac Sim, custom clients) can request
RGB and LiDAR frames for arbitrary trajectories on a trained USDZ.

Three pieces work together:

1. **Server** — `nurec_dev bazelisk run //:run -- serve-grpc`.
2. **Client protobufs** — generated from the matching source release.
3. **Client code** — Python (or any language with a gRPC stub).

The proto definitions live in two files:

| Proto file                          | Contents                                                                                                                                  |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `nurec/grpc/protos/common.proto`    | `Empty`, `Quat`, `Vec3`, `Pose`, `DynamicState`, `AABB`, `PoseAtTime`, `StateAtTime`, `Trajectory`, `VersionId`, `AvailableScenesReturn`. |
| `nurec/grpc/protos/sensorsim.proto` | The `SensorsimService` definition and all the request / response messages it uses.                                                        |

## 1. Server

```bash
RUN_ID=latest    # run directory name under $NUREC_SRC/output

nurec_dev bazelisk run //:run -- \
  serve-grpc \
  --port="${NUREC_GRPC_PORT:-8080}" \
  --artifact-glob "$NUREC_SRC/output/${RUN_ID:?}/artifacts/last.usdz"
```

Required flag: `--artifact-glob` (must end in `.usdz`; quote it to
avoid shell expansion). Common server flags:

| Flag                                                     | Default                                                                        | Purpose                                                                                                           |
| -------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `--host`                                                 | `localhost`                                                                    | gRPC bind host. Use `0.0.0.0` to expose to LAN (with `-p 8080:8080` instead of `--net=host`).                     |
| `--port`                                                 | `8080`                                                                         | gRPC bind port.                                                                                                   |
| `--health-port`                                          | unset                                                                          | Run a dedicated `grpc.health.v1.Health` server on this port; otherwise health is multiplexed on `--port`.         |
| `--test-scenes-are-valid` / `--no-test-scenes-are-valid` | `False`                                                                        | Load + validate every scene at startup before serving.                                                            |
| `--renderer [default\|gsplat]`                           | `default`                                                                      | `default` = the renderer the artifact was trained with; `gsplat` = force the GSplat rasterizer.                   |
| `--enable-harmonizer`                                    | `False`                                                                        | Run the Harmonizer post-process in-server.                                                                        |
| `--harmonizer-url`                                       | `https://huggingface.co/nvidia/Harmonizer/blob/main/harmonizer_nontemporal.pt` | URL of the Harmonizer checkpoint.                                                                                 |
| `--harmonizer-cache`                                     | `~/.cache/nurec/harmonizer`                                                    | Local checkpoint cache dir.                                                                                       |
| `--harmonizer-model-filename`                            | `harmonizer_nontemporal.pt`                                                    | Filename inside the cache dir.                                                                                    |
| `--harmonizer-resolution`                                | `(576, 1024)`                                                                  | (H, W). Takes TWO ints: `--harmonizer-resolution 576 1024`.                                                       |
| `--enable-timing`                                        | `False`                                                                        | Print per-stage rendering timings.                                                                                |
| `--ray-chunk-size`                                       | `2^62`                                                                         | Max rays per forward pass; lower it to fit smaller GPUs.                                                          |
| `--egocar-hood-dir`                                      | unset                                                                          | Override directory of egocar hood images.                                                                         |
| `--enable-editing-actors`                                | `False`                                                                        | Required to apply `--edit-assets` later via `render-grpc`.                                                        |
| `--cache-size`                                           | `10`                                                                           | Max number of loaded backends (count-based LRU). On OOM, spare backends are auto-evicted and the load is retried. |
| `--max-workers`                                          | `1`                                                                            | gRPC server thread-pool worker count.                                                                             |
| `--download-cache-dir`                                   | `~/.cache/nurec/downloaded_scenes`                                             | Cache for scenes the server downloads on demand.                                                                  |
| `--download-cache-size`                                  | `5`                                                                            | Max downloaded scenes (LRU).                                                                                      |
| `--metrics-output-dir`                                   | unset                                                                          | Directory to write per-request rendering-time metrics.                                                            |

Deprecated (still accepted with a warning, hidden from `--help`):

- `--use-gsplat` → `--renderer gsplat`

The server raises gRPC `max_send_message_length` /
`max_receive_message_length` to **50 MB** (LiDAR returns can be large);
match those limits client-side.

## 2. Client protobufs from source

Generate client stubs from the same released checkout as the server. In a Python
3.13 virtual environment:

```bash
python -m pip install -r "$NUREC_SRC/nurec/grpc/release/requirements.txt" numpy scipy
bash "$NUREC_SRC/skills/nurec/references/NUREC_RenderClient/scripts/setup_protos.sh"
export PYTHONPATH="${NUREC_RENDER_CLIENT_PROTO_DIR:-$HOME/.cache/nurec-render-client/proto}${PYTHONPATH:+:$PYTHONPATH}"
```

Regenerate when switching source/server releases. See the
[thin-client setup](NUREC_RenderClient/README.md) for interpreter and cache options.

After that the client imports look like:

```python
import nurec.grpc.protos.common_pb2 as common_pb2
import nurec.grpc.protos.sensorsim_pb2 as sensorsim_pb2
import nurec.grpc.protos.sensorsim_pb2_grpc as sensorsim_pb2_grpc
```

## 3. Service surface — `SensorsimService`

Defined in `nurec.grpc.protos.sensorsim_pb2_grpc`. Every RPC request /
response message is in `sensorsim_pb2` (or `common_pb2` for the shared
types).

| RPC                          | Request                                            | Response                                                                                                    | Purpose                                                                                                                                                                                              |
| ---------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_version`                | `Empty`                                            | `VersionId` (server `version_id`, `git_hash`, `grpc_api_version`)                                           | Server / API version probe.                                                                                                                                                                          |
| `get_server_config`          | `Empty`                                            | `ServerConfig` (`map<string,string>`)                                                                       | Read the resolved server config (renderer, difix, cache sizes, …).                                                                                                                                   |
| `get_available_scenes`       | `Empty`                                            | `AvailableScenesReturn(scene_ids)`                                                                          | List loaded USDZ scene IDs.                                                                                                                                                                          |
| `get_available_cameras`      | `AvailableCamerasRequest(scene_id)`                | `AvailableCamerasReturn(available_cameras)`                                                                 | Camera intrinsics + `rig_to_camera` per scene.                                                                                                                                                       |
| `get_available_trajectories` | `AvailableTrajectoriesRequest(scene_id)`           | `AvailableTrajectoriesReturn(available_trajectories)`                                                       | Embedded trajectories (rig pose vs time).                                                                                                                                                            |
| `get_available_ego_masks`    | `Empty`                                            | `AvailableEgoMasksReturn(ego_mask_metadata)`                                                                | List of `(camera_logical_id, rig_config_id)` ego-mask IDs (e.g. `hyperion8.0`, `hyperion8.1`).                                                                                                       |
| `get_dynamic_objects`        | `AvailableDynamicObjectsRequest(scene_id)`         | `AvailableDynamicObjectsReturn(dynamic_objects)`                                                            | Per-track trajectory + AABB + semantic class for the scene's actors.                                                                                                                                 |
| `get_external_asset_objects` | `ExternalAssetObjectsRequest(scene_id)`            | `ExternalAssetObjectsReturn(track_ids)`                                                                     | List the external assets (Asset-Harvester PLYs) currently bundled in the artifact.                                                                                                                   |
| `edit_assets`                | `EditAssetsRequest(scene_id, replace[], insert[])` | `EditAssetsResponse(success, message)`                                                                      | One-shot mutate of the renderable model: replace tracks with a different asset and/or insert new dynamic-object tracks. Server saves the original training parameters first so they can be restored. |
| `restore_model_parameters`   | `RestoreModelParametersRequest(scene_id)`          | `Empty`                                                                                                     | Undo the last `edit_assets` call and revert to the training parameters.                                                                                                                              |
| `render_rgb`                 | `RGBRenderRequest`                                 | `RGBRenderReturn(image_bytes)`                                                                              | Render one RGB frame.                                                                                                                                                                                |
| `batch_render_rgb`           | `BatchRGBRenderRequest(items[])`                   | `BatchRGBRenderReturn(items[])`                                                                             | Multiple cameras / frames in one call (per-item `success`, `error_message`).                                                                                                                         |
| `render_lidar`               | `LidarRenderRequest`                               | `LidarRenderReturn(point_xyzs, point_intensities, num_points, point_xyzs_buffer, point_intensities_buffer)` | Render one LiDAR sweep.                                                                                                                                                                              |
| `shut_down`                  | `Empty`                                            | `Empty`                                                                                                     | Graceful server shutdown.                                                                                                                                                                            |

The server also registers the standard `grpc.health.v1.Health`
service — on the same port unless `--health-port` is set.

### Status codes the server emits

| gRPC status           | When the server raises it                                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------------------------- |
| `OUT_OF_RANGE`        | Render request's `[frame_start_us, frame_end_us)` is outside the scene's time range.                            |
| `RESOURCE_EXHAUSTED`  | Backend cache full or GPU OOM during render (volumetric effects / under-converged checkpoints often trip this). |
| `FAILED_PRECONDITION` | Cache cannot serve the scene; or model/artifact incompatibility.                                                |
| `NOT_FOUND`           | Scene or asset key missing.                                                                                     |
| `INVALID_ARGUMENT`    | Bad request fields (wrong type, out-of-range value).                                                            |
| `UNKNOWN`             | Anything else (server logs the full traceback before failing).                                                  |

`render_rgb` reports a single status; `batch_render_rgb` returns
per-item `success` / `error_message` so a partial failure does not
fail the whole batch.

## 4. Key data types

### Pose & trajectory

```python
common_pb2.Vec3(x: float, y: float, z: float)
common_pb2.Quat(w: float, x: float, y: float, z: float)
common_pb2.Pose(vec: Vec3, quat: Quat)            # translation applied first, then rotation
common_pb2.AABB(size_x: float, size_y: float, size_z: float)
common_pb2.PoseAtTime(pose: Pose, timestamp_us: fixed64)
common_pb2.Trajectory(poses: list[PoseAtTime])

sensorsim_pb2.PosePair(start_pose: Pose, end_pose: Pose)  # rolling shutter
```

> Convention: `start_pose != end_pose` ⇒ rolling-shutter. For instant
> shutters the server expects an interval `[t, t+1)` (the
> half-closed convention) — the bundled `render-grpc` client adds the
> `+1` automatically; do the same in your own clients.

### Cameras

```python
sensorsim_pb2.CameraSpec(
    logical_id: str,
    resolution_w: uint32,
    resolution_h: uint32,
    shutter_type: ShutterType,        # UNKNOWN | ROLLING_TOP_TO_BOTTOM | ROLLING_LEFT_TO_RIGHT
                                       # | ROLLING_BOTTOM_TO_TOP | ROLLING_RIGHT_TO_LEFT | GLOBAL
    # one of camera_param oneof:
    ftheta_param:           sensorsim_pb2.FthetaCameraParam,
    opencv_pinhole_param:   sensorsim_pb2.OpenCVPinholeCameraParam,   # added in 25.09
    opencv_fisheye_param:   sensorsim_pb2.OpenCVFisheyeCameraParam,
    # one of external_distortion oneof (optional):
    bivariate_windshield_model_param: sensorsim_pb2.BivariateWindshieldModelParameters,
)
```

`FthetaCameraParam` carries `principal_point_{x,y}`, the
`reference_poly` enum (`PIXELDIST_TO_ANGLE` or `ANGLE_TO_PIXELDIST`),
both forward and backward polynomial coefficient lists, `max_angle`,
and an optional `LinearCde(linear_c, linear_d, linear_e)` block for
linearization corrections.

`OpenCVPinholeCameraParam` carries `(focal_length_x, focal_length_y,
principal_point_x, principal_point_y)` plus radial / tangential /
thin-prism distortion coefficient lists.

### Ego mask

```python
sensorsim_pb2.EgoMaskId(camera_logical_id: str, rig_config_id: str)
```

The `rig_config_id` matches the `--rig-name` flag of `render-grpc`
(e.g. `hyperion8.0`, `hyperion8.1`). To enable inpainting set
`RGBRenderRequest.insert_ego_mask=True` and pass the matching
`EgoMaskId` from `get_available_ego_masks`.

### Render requests

```python
sensorsim_pb2.RGBRenderRequest(
    scene_id:           str,
    resolution_h:       uint32,
    resolution_w:       uint32,
    camera_intrinsics:  CameraSpec,
    frame_start_us:     fixed64,
    frame_end_us:       fixed64,                # half-closed: end exclusive
    sensor_pose:        PosePair,
    dynamic_objects:    list[DynamicObject],
    image_format:       ImageFormat,            # UNDEFINED | PNG | JPEG | RGB_UINT8_PLANAR | AVC | AV1
    image_quality:      float,                  # used for JPEG
    insert_ego_mask:    bool,
    ego_mask_id:        EgoMaskId,              # required when insert_ego_mask=True
)

sensorsim_pb2.LidarRenderRequest(
    scene_id:        str,
    lidar_config:    LidarSpec(lidar_type=LidarDeviceType.PANDAR128 | AT128),
    frame_start_us:  fixed64,
    frame_end_us:    fixed64,
    sensor_pose:     PosePair,
    dynamic_objects: list[DynamicObject],
    render_filter:   LidarRenderFilter,         # all four fields are optional
)
```

`LidarRenderFilter` exposes the four LiDAR post-processing knobs:

| Field                       | Default behaviour                  | Purpose                                                                                        |
| --------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| `raydrop_threshold`         | artifact config (typically `0.5`)  | Drop rays with `raydrop > threshold`.                                                          |
| `opacity_threshold`         | artifact config (typically `0.8`)  | Drop rays with `opacity <= threshold`. Set to `0.0` to disable (matches validation behaviour). |
| `enable_distance_filter`    | artifact config                    | Toggle distance-based edge filtering (removes floating points).                                |
| `distance_filter_threshold` | artifact config (typically `0.02`) | Higher = fewer points filtered.                                                                |

### Render responses

```python
sensorsim_pb2.RGBRenderReturn(image_bytes: bytes)   # encoded by image_format
sensorsim_pb2.LidarRenderReturn(
    # Either the typed lists or the raw bytes buffers may be populated
    # by the server depending on the build.
    point_xyzs:               list[float],            # [x1,y1,z1, x2,y2,z2, ...]
    point_intensities:        list[float],            # in [0, 1]
    num_points:               uint32,
    point_xyzs_buffer:        bytes,                  # float32 binary, end-of-spin LiDAR space
    point_intensities_buffer: bytes,                  # float32 binary
)
```

Coordinates are in **end-of-spin LiDAR space**.

### Asset editing

```python
sensorsim_pb2.DynamicObject(track_id: str, pose_pair: PosePair)

sensorsim_pb2.DynamicObjectTrack(
    id:             str,
    semantic_class: str,
    trajectory:     common_pb2.Trajectory,
    object_size:    common_pb2.AABB,
    asset_id:       str,                        # selects the underlying PLY asset (decoupled from track id)
)

sensorsim_pb2.ReplaceAssetAction(
    original_id:    str,                        # track in the artifact's sequence_tracks
    replacement_id: str,                        # asset in external_assets/
    object_size:    common_pb2.AABB,
)

sensorsim_pb2.EditAssetsRequest(scene_id, replace=[ReplaceAssetAction], insert=[DynamicObjectTrack])
sensorsim_pb2.EditAssetsResponse(success: bool, message: str)
sensorsim_pb2.RestoreModelParametersRequest(scene_id)
```

Send `edit_assets` **once** before sending `RGBRenderRequest`s with
matching dynamic actors; call `restore_model_parameters` to revert to
the original training parameters before re-using the same scene for
something else. (The server saves the training parameters before
applying any edits, so a single restore call is enough.)

## 5. Python client cookbook (async)

```python
import grpc.aio

from nurec.grpc.protos.common_pb2 import Empty
from nurec.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasRequest,
    AvailableTrajectoriesRequest,
    ImageFormat,
    PosePair,
    RGBRenderRequest,
)
from nurec.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub
# Use the host-only helper; generated stubs do not include nurec.grpc.pose.
import os
import sys
sys.path.insert(0, os.path.join(os.environ["NUREC_SRC"],
    "skills/nurec/references/NUREC_RenderClient/scripts"))
from thin_client import se3_to_grpc_pose  # numpy SE3 -> gRPC Pose


async def render_one_frame(scene_id, pose, timestamp_us):
    # Match the server's 50 MB message-size limits (LiDAR responses can be large).
    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ]
    channel = grpc.aio.insecure_channel("localhost:8080", options=options)
    client = SensorsimServiceStub(channel)

    scenes = await client.get_available_scenes(Empty())
    assert scene_id in scenes.scene_ids

    cameras = await client.get_available_cameras(
        AvailableCamerasRequest(scene_id=scene_id)
    )
    front = next(
        c for c in cameras.available_cameras
        if c.logical_id == "camera_front_wide_120fov"
    )

    request = RGBRenderRequest(
        scene_id=scene_id,
        resolution_h=front.intrinsics.resolution_h,
        resolution_w=front.intrinsics.resolution_w,
        camera_intrinsics=front.intrinsics,
        frame_start_us=timestamp_us,
        frame_end_us=timestamp_us + 1,                  # half-closed [t, t+1)
        sensor_pose=PosePair(
            start_pose=se3_to_grpc_pose(pose),
            end_pose=se3_to_grpc_pose(pose),
        ),
        image_format=ImageFormat.JPEG,
        image_quality=95,
    )

    response = await client.render_rgb(request)
    with open("frame.jpg", "wb") as fh:
        fh.write(response.image_bytes)
```

A synchronous variant is fine too — use
`grpc.insecure_channel("localhost:8080", options=options)` and drop
the `await` keywords.

### LiDAR client

```python
import numpy as np
from nurec.grpc.protos.sensorsim_pb2 import (
    LidarDeviceType,
    LidarRenderFilter,
    LidarRenderRequest,
    LidarSpec,
    PosePair,
)


def render_one_lidar_sweep(client, scene_id, pose, timestamp_us):
    request = LidarRenderRequest(
        scene_id=scene_id,
        lidar_config=LidarSpec(lidar_type=LidarDeviceType.PANDAR128),
        frame_start_us=timestamp_us,
        frame_end_us=timestamp_us + 100_000,            # 100 ms sweep window
        sensor_pose=PosePair(
            start_pose=se3_to_grpc_pose(pose),
            end_pose=se3_to_grpc_pose(pose),
        ),
        render_filter=LidarRenderFilter(
            raydrop_threshold=0.5,
            opacity_threshold=0.0,
        ),
    )
    response = client.render_lidar(request)
    if response.point_xyzs_buffer:
        xyz = np.frombuffer(response.point_xyzs_buffer, dtype=np.float32).reshape(-1, 3)
    else:
        xyz = np.array(response.point_xyzs, dtype=np.float32).reshape(-1, 3)
    return xyz
```

### Edit-actors client

```python
from nurec.grpc.protos.common_pb2 import AABB
from nurec.grpc.protos.sensorsim_pb2 import (
    EditAssetsRequest,
    ReplaceAssetAction,
    RestoreModelParametersRequest,
)


def replace_actor(client, scene_id, original_id, replacement_id, dims_xyz):
    edit = EditAssetsRequest(
        scene_id=scene_id,
        replace=[ReplaceAssetAction(
            original_id=original_id,
            replacement_id=replacement_id,
            object_size=AABB(size_x=dims_xyz[0], size_y=dims_xyz[1], size_z=dims_xyz[2]),
        )],
    )
    resp = client.edit_assets(edit)
    if not resp.success:
        raise RuntimeError(resp.message)

    # ... render frames here ...

    client.restore_model_parameters(RestoreModelParametersRequest(scene_id=scene_id))
```

For the full `edit_assets` flow (including `insert`) plus how to
populate the asset bank in the first place, see
`references/asset-editing.md`.

## 6. `render-grpc` CLI helper

If you just need a directory of frames on disk, skip the Python client
and use the bundled helper:

```bash
RUN_ID=latest    # run directory name under $NUREC_SRC/output

nurec_dev bazelisk run //:run -- \
  render-grpc \
  --port="${NUREC_GRPC_PORT:-8080}" \
  --artifact-path $NUREC_SRC/output/"${RUN_ID:?}"/artifacts/last.usdz \
  --output-dir $NUREC_SRC/output/"${RUN_ID:?}"/renders \
  --camera-id camera_front_wide_120fov
```

Required: `--artifact-path` and `--output-dir`. Common optional flags
(see `references/cli-reference.md` for the full matrix):

| Flag                                     | Default                    | Purpose                                                                                                    |
| ---------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `--host`                                 | `localhost`                | gRPC server host.                                                                                          |
| `--port`                                 | `8080`                     | gRPC server port.                                                                                          |
| `--height`                               | `300`                      | Image height (px).                                                                                         |
| `--camera-id`                            | `camera_front_wide_120fov` | Camera identifier.                                                                                         |
| `--image-format [png\|jpeg]`             | `jpeg`                     | Output codec.                                                                                              |
| `--frame-step`                           | `1`                        | Frame stride.                                                                                              |
| `--frame-naming`                         | `contiguous-output-index`  | `frame-end-timestamp` (microseconds, recommended for benchmarking) or `contiguous-output-index` (0,1,2,…). |
| `--disable-rolling-shutter`              | unset                      | Override per-row timestamps with the frame-end timestamp.                                                  |
| `--demo-actor-transform`                 | unset                      | Apply a demo actor transform (Z-axis rotation). Requires `--enable-editing-actors`.                        |
| `--enable-editing-actors`                | unset                      | Send dynamic actor updates each frame.                                                                     |
| `--shutdown-server-on-completion`        | unset                      | Stop the server after the run.                                                                             |
| `--rig-name`                             | unset                      | `hyperion8.0` / `hyperion8.1` for inpainted ego hood.                                                      |
| `--rig-translation-offset tx ty tz`      | `0 0 0`                    | Static rig-frame XYZ offset (m).                                                                           |
| `--rig-rotation-offset yaw -roll -pitch` | `0 0 0`                    | Static rig-frame rotation (deg).                                                                           |
| `--edit-assets`                          | unset                      | Path to `edit-assets.json` (requires `--enable-editing-actors` on the server).                             |
| `--sequential`                           | unset                      | Send requests sequentially instead of in an async batch (debugging).                                       |

LiDAR-mode flags activate when `--lidar` is set:

| Flag                                                     | Default          | Purpose                                                              |
| -------------------------------------------------------- | ---------------- | -------------------------------------------------------------------- |
| `--lidar`                                                | off              | Switch to `render_lidar` and write point clouds.                     |
| `--lidar-id`                                             | first available  | LiDAR sensor ID to use for frame timestamps.                         |
| `--lidar-format [bin\|ply]`                              | `bin`            | Raw binary (xyz then intensity) or PLY with intensity-mapped colors. |
| `--lidar-raydrop-threshold`                              | `0.5`            | Drop rays with `raydrop > T`.                                        |
| `--lidar-opacity-threshold`                              | `0.0`            | Drop rays with `opacity <= T`. `0.0` disables the filter.            |
| `--lidar-distance-filter` / `--no-lidar-distance-filter` | artifact default | Toggle distance-based edge filtering.                                |
| `--lidar-distance-filter-threshold`                      | artifact default | `[0, 1]`; higher = fewer points filtered.                            |

The helper writes `timestamps.json` next to the rendered frames (one
entry per frame with `file_name`, `render_frame_idx`,
`frame_start_timestamp_us`, `frame_end_timestamp_us`) and
`render_grpc_cli_args.json` capturing the exact CLI invocation.

## 7. Harmonizer (Fixer) post-processing inside the container

Use the public non-temporal Harmonizer checkpoint at `[576, 1024]`
(see [public model setup](harmonizer.md)). To
enable in-server post-processing add the flags below to `serve-grpc`.
The `--difix-*` flags are deprecated aliases of the `--harmonizer-*`
flags and still work, emitting a `DeprecationWarning`:

```bash
RUN_ID=latest    # run directory name under $NUREC_SRC/output

nurec_dev bazelisk run //:run -- serve-grpc \
  --port="${NUREC_GRPC_PORT:-8080}" \
    --artifact-glob $NUREC_SRC/output/"${RUN_ID:?}"/artifacts/last.usdz \
    --enable-harmonizer \
    --harmonizer-url=https://huggingface.co/nvidia/Harmonizer/resolve/f2bae9823e07d281c9dce6581d06118d675e08c3/harmonizer_nontemporal.pt \
    --harmonizer-model-filename=harmonizer_nontemporal.pt \
    --harmonizer-resolution 576 1024
```

For standalone image/video enhancement, follow the public Harmonizer
repository linked in [harmonizer.md](harmonizer.md).

## 8. Common operational tips

- **`--net=host`** is the easiest way to expose port 8080 to the
  client; alternatively use `-p 8080:8080` and bind the server to
  `--host 0.0.0.0`.
- **`--privileged`** is **not** used by default here, and is not needed
  on a normal host — `--gpus all` with the NVIDIA Container Toolkit is
  enough (`scripts/session_warm_server.sh` runs the same GPU server
  without it). A few host setups do need it for direct GPU access,
  Hyperion 8 rigs in particular. Add it only on a host you have
  confirmed requires it: it disables container isolation entirely.
- **Health checks:** the server registers `grpc.health.v1.Health` on
  the data port by default; pass `--health-port 9090` to keep
  health-checks on a separate port.
- **Coordinate frames:** the gRPC API expects poses in NuRec space
  (frame 0 is identity). When integrating with a simulator that
  ships poses in OpenDRIVE ENU or ECEF, convert them — full recipe in
  `references/physical-ai-render.md`.
- **Half-closed time intervals:** `frame_start_us < frame_end_us` is
  required; for instant shutters use `[t, t+1)`. The bundled
  `render-grpc` does this automatically; replicate the convention in
  your own clients.
- **Message-size limits:** the server raises `max_send_message_length`
  / `max_receive_message_length` to **50 MB** so LiDAR responses fit;
  set the same limits on the client channel.
- **Backend cache & OOM:** the server keeps up to `--cache-size` model
  backends in a count-based LRU. On GPU OOM during a checkout, spare
  backends are evicted and the load is retried. If every slot is
  in-use the request returns `RESOURCE_EXHAUSTED` — increase
  `--cache-size`, lower request concurrency, or wait for in-flight
  requests to drain.
- **Probing the server:** `get_version` / `get_server_config` /
  `get_available_scenes` are cheap and a good sanity check before
  issuing render calls.
- **Editing flow:** `edit_assets` mutates the renderable model;
  always call `restore_model_parameters` (or restart the server) when
  done, otherwise subsequent renders will see the edits.
- **Batched RGB:** use `batch_render_rgb` when you have multiple
  cameras / frames against the same `scene_id` — each item carries
  its own `RGBRenderRequest` plus a per-item `success` /
  `error_message` so partial failures don't fail the whole batch.
