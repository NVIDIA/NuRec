<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Render the NVIDIA Physical AI Dataset with NuRec

Complete [source setup](install.md) before running the commands below.

NVIDIA publishes pre-reconstructed NuRec USDZ scenes on HuggingFace as
the **`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`** dataset. With
those scenes you can skip the train/val step entirely and jump
straight to programmatic rendering through the sensorsim gRPC API.

This recipe covers:

1. Loading and parsing the USDZ scene package.
2. Aligning custom trajectories to the NuRec coordinate frame using
   ECEF + OpenDRIVE ENU transforms.
3. Driving the NuRec gRPC server to render frames along the new
   trajectory.

## 0. Public references

- Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
- Source target: `//:run` (see [setup](install.md))
- gRPC API surface: `references/grpc-api.md`

The dataset is gated — accept the dataset license on HuggingFace and
authenticate with `huggingface-cli login` (or set `HF_TOKEN` from
https://huggingface.co/settings/tokens).

## 1. USDZ contents

Each scenario USDZ contains:

- The neural reconstruction binaries (Gaussians).
- `data_info.json` — scenario metadata, timestamps, sequence ID.
- `rig_trajectories.json` — camera calibrations, ego trajectory,
  `T_world_base` (i.e. `T_rig_ecef` for frame 0).
- `sequence_tracks.json` — per-actor track data (poses, timestamps,
  labels).
- `map.xodr` — OpenDRIVE map with georeference proj string.

```python
import json, zipfile


def load_nurec_scenario(usdz_path: str) -> dict:
    out = {}
    with zipfile.ZipFile(usdz_path, "r") as z:
        out["metadata"] = json.loads(z.read("data_info.json"))
        out["rig_trajectories"] = json.loads(z.read("rig_trajectories.json"))
        out["sequence_tracks"] = json.loads(z.read("sequence_tracks.json"))
        out["xodr_data"] = z.read("map.xodr").decode("utf-8")
    return out


scenario = load_nurec_scenario("/path/to/scenario.usdz")
```

## 2. Parse trajectory and tracks

```python
import numpy as np


def parse_trajectory_data(scenario):
    metadata = scenario["metadata"]
    rig = scenario["rig_trajectories"]
    tracks = scenario["sequence_tracks"]

    start_time = metadata["pose-range"]["start-timestamp_us"]
    end_time = metadata["pose-range"]["end-timestamp_us"]

    ego_poses = rig["rig_trajectories"][0]["T_rig_worlds"]
    ego_timestamps = rig["rig_trajectories"][0]["T_rig_world_timestamps_us"]

    tracks_data = tracks["dummy_chunk_id"]["tracks_data"]
    track_ids = tracks_data["tracks_id"]
    object_tracks = {}
    for i, tid in enumerate(track_ids):
        object_tracks[tid] = {
            "poses": tracks_data["tracks_poses"][i],
            "timestamps": tracks_data["tracks_timestamps_us"][i],
            "label": tracks_data["tracks_label_class"][i],
        }

    return start_time, end_time, {
        "ego": {"poses": ego_poses, "timestamps": ego_timestamps},
        "objects": object_tracks,
    }
```

## 3. Coordinate frames

Four frames matter:

| Frame                 | Definition                                             |
| --------------------- | ------------------------------------------------------ |
| `ECEF`                | Earth-Centered, Earth-Fixed (WGS84).                   |
| `ENU`                 | East-North-Up, origin from the OpenDRIVE georeference. |
| `OpenDRIVE Map Space` | The local ENU frame the map is authored in.            |
| `NuRec Space`         | Reconstruction frame where frame 0 is identity.        |

The chain is **NuRec → World → ECEF → ENU (Map)**. The World hop is easy to
miss and is not optional: `world_to_nurec` is what places the reconstruction
inside the recorded world frame, and skipping it leaves the result displaced
by the scene-origin offset (53.7 m for the `048b974e` clip, 86.7 m for the
`0fd06bc3` demo clip).

```python
T_nurec_map = T_ecef_enu @ T_world_ecef @ inv(T_world_nurec)
```

where `T_world_nurec` is `world_to_nurec` (World → NuRec, so it is inverted to go
NuRec → World), `T_world_ecef` is `T_world_base` (World → ECEF), and
`T_ecef_enu` maps ECEF to the OpenDRIVE map's local ENU origin.

### Parse the georeference

```python
import xml.etree.ElementTree as ET


def parse_opendrive_georeference(xodr_data: str) -> dict:
    tree = ET.fromstring(xodr_data)
    geo_reference = tree.find(".//geoReference")
    parts = geo_reference.text.split(" ")

    geo = {}
    for part in parts:
        if part.startswith("+lat_0="):
            geo["latitude"] = float(part[7:])
        elif part.startswith("+lon_0="):
            geo["longitude"] = float(part[7:])
        elif part.startswith("+alt_0="):
            geo["altitude"] = float(part[7:])
        elif part.startswith("+proj="):
            geo["projection"] = part[6:]
    geo.setdefault("altitude", 0.0)
    return geo
```

### ECEF and ECEF→ENU

```python
def lat_lng_alt_to_ecef(lla: np.ndarray) -> np.ndarray:
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    phi = np.deg2rad(lla[:, 0])
    gamma = np.deg2rad(lla[:, 1])
    e2 = (a * a - b * b) / (a * a)
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    x = (N + lla[:, 2]) * np.cos(phi) * np.cos(gamma)
    y = (N + lla[:, 2]) * np.cos(phi) * np.sin(gamma)
    z = (N * (b * b) / (a * a) + lla[:, 2]) * np.sin(phi)
    return np.column_stack([x, y, z])


def ecef_to_enu_transform(lla: np.ndarray) -> np.ndarray:
    ecef_origin = lat_lng_alt_to_ecef(lla)
    lat = np.deg2rad(lla[0, 0])
    lon = np.deg2rad(lla[0, 1])
    R = np.array([
        [-np.sin(lon),                 np.cos(lon),                0],
        [-np.sin(lat) * np.cos(lon),  -np.sin(lat) * np.sin(lon),  np.cos(lat)],
        [ np.cos(lat) * np.cos(lon),   np.cos(lat) * np.sin(lon),  np.sin(lat)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ ecef_origin.T.flatten()
    return T


def calculate_nurec_to_map_transform(scenario, geo):
    # The rig-trajectories JSON carries BOTH `T_world_base` (world -> ECEF)
    # and `world_to_nurec` (world -> NuRec scene frame). Skipping `world_to_nurec`
    # leaves the chain in world space rather than NuRec space and displaces
    # the result by the scene origin offset -- 53.7 m for the 048b974e clip,
    # 86.7 m for the 0fd06bc3 demo clip. Invert it to go NuRec -> world first.
    def _mat(v):
        # These entries are not a uniform shape: the bundled
        # custom-rig-trajectories JSONs store world_to_nurec as
        # {"matrix": [...]} while T_world_base is a bare nested list.
        # np.array() on the dict yields a 0-D object array and
        # np.linalg.inv then fails, so unwrap "matrix" when present.
        return np.array(v["matrix"] if isinstance(v, dict) else v)

    rig = scenario["rig_trajectories"]
    T_world_ecef = _mat(rig["T_world_base"])
    T_world_nurec = _mat(rig.get("world_to_nurec", rig.get("world_to_nre")))  # pre-rename artifacts use world_to_nre
    T_nurec_world = np.linalg.inv(T_world_nurec)
    lla = np.array([[geo["latitude"], geo["longitude"], geo["altitude"]]])
    T_ecef_enu = ecef_to_enu_transform(lla)
    T_nurec_map = T_ecef_enu @ T_world_ecef @ T_nurec_world
    return T_nurec_map, T_world_ecef, T_ecef_enu
```

### Round-trip a custom trajectory through NuRec space

```python
def map_to_nurec(map_pose: np.ndarray, T_nurec_map: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_nurec_map) @ map_pose
```

Author the trajectory in map (ENU) coordinates, then convert each pose
back to NuRec before sending it to the renderer.

## 4. Stand up the gRPC server

```bash
nurec_dev bazelisk run //:run -- \
  serve-grpc \
  --artifact-glob "$NUREC_SRC/scenarios/*.usdz" \
  --renderer default
```

Useful add-ons:

- `--enable-harmonizer` to apply Harmonizer artifact removal in-server
  (`--enable-difix` is a deprecated alias for it).
- `--enable-timing` to capture per-stage rendering latency.

## 5. Render a custom trajectory

Complete the [host client setup](NUREC_RenderClient/README.md) with
protobufs generated from the matching source checkout before running this example.

```python
# Use the bundled host helper with the matching generated protobufs.
import os
import sys
sys.path.insert(0, os.path.join(os.environ["NUREC_SRC"],
    "skills/nurec/references/NUREC_RenderClient/scripts"))
from thin_client import se3_to_grpc_pose

import grpc
import numpy as np
from nurec.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasRequest,
    CameraSpec,
    ImageFormat,
    PosePair,
    RGBRenderRequest,
)
from nurec.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub


def render_trajectory(usdz_scenario, nurec_trajectory, camera_specs, out_dir):
    channel = grpc.insecure_channel("localhost:8080")
    client = SensorsimServiceStub(channel)

    name = next(iter(camera_specs))
    spec = camera_specs[name]
    cameras = client.get_available_cameras(
        AvailableCamerasRequest(scene_id=usdz_scenario["metadata"]["sequence_id"])
    )
    camera = next(cam for cam in cameras.available_cameras if cam.logical_id == name)
    grpc_camera = CameraSpec(
        logical_id=name,
        resolution_h=spec["resolution_h"],
        resolution_w=spec["resolution_w"],
        ftheta_param=spec["intrinsics"],
        shutter_type=camera.intrinsics.shutter_type,
    )

    for i, point in enumerate(nurec_trajectory):
        ts = int(point["timestamp_us"])
        # PosePair fields are protobuf Pose messages (quat + vec3), not raw
        # 4x4 matrices -- convert, or the request fails at construction time.
        pose = se3_to_grpc_pose(np.array(point["pose_matrix"]))
        request = RGBRenderRequest(
            scene_id=usdz_scenario["metadata"]["sequence_id"],
            resolution_h=spec["resolution_h"],
            resolution_w=spec["resolution_w"],
            camera_intrinsics=grpc_camera,
            frame_start_us=ts,
            frame_end_us=ts + 1,
            sensor_pose=PosePair(start_pose=pose, end_pose=pose),
            dynamic_objects=[],
            image_format=ImageFormat.JPEG,
            image_quality=95,
        )
        response = client.render_rgb(request)
        with open(f"{out_dir}/frame_{i:06d}_{ts}.jpg", "wb") as fh:
            fh.write(response.image_bytes)
```

Where `camera_specs` is built from `rig_trajectories.json`:

```python
def setup_camera_specs(scenario):
    specs = {}
    cams = scenario["rig_trajectories"]["camera_calibrations"]
    for cam_id, calib in cams.items():
        name = calib["logical_sensor_name"]
        params = calib["camera_model"]["parameters"]
        specs[name] = {
            "type": calib["camera_model"]["type"],
            "resolution_w": params["resolution"][0],
            "resolution_h": params["resolution"][1],
            "intrinsics": {
                "principal_point_x": params["principal_point"][0],
                "principal_point_y": params["principal_point"][1],
                "max_angle": params["max_angle"],
                "pixeldist_to_angle_poly": params["pixeldist_to_angle_poly"],
                "angle_to_pixeldist_poly": params["angle_to_pixeldist_poly"],
                "reference_poly": params["reference_poly"],
            },
            "T_sensor_rig": calib["T_sensor_rig"],
        }
    return specs
```

## 6. Validation tips

- Render the original `ego.poses` first and visually diff against the
  source clip — if they don't match, the coordinate chain is broken.
- Plot `T_nurec_map @ ego_pose` against the OpenDRIVE map in QGIS or a
  similar tool; the trajectory should follow the road network.
- The gRPC API expects poses in NuRec space; always convert custom
  trajectories with `np.linalg.inv(T_nurec_map)` before the RPC call.
