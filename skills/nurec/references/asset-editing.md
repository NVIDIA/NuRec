<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Edit Actors in a NuRec USDZ Scene

Complete [source setup](install.md) before running the commands below.

Use this guide to remove, insert, or replace 3D assets in a trained
NuRec USDZ by combining Asset-Harvester output with two NuRec
sub-commands: `export-external-assets` and `render-grpc --edit-assets`
(or any custom client that calls `SensorsimService.edit_assets` —
see `references/grpc-api.md`).

## Pipeline overview

```text
            ┌─────────────────────────┐
            │  Asset-Harvester output │
            │  (per-track .ply +      │
            │   metadata.yaml)        │
            └────────────┬────────────┘
                         │
                         ▼
   nurec export-external-assets
                 │
                 ├── repackages USDZ → external_assets/<track>/<track>.ply
                 ├── adds external_assets/metadata.yaml
                 └── writes edit-assets.json (stub) ── you edit this file
                         │
                         ▼
   nurec serve-grpc --enable-editing-actors --artifact-glob <new.usdz>
                         │
                         ▼
   nurec render-grpc --edit-assets <edit-assets.json>
       └── client calls SensorsimService.edit_assets(replace=[…], insert=[…])
       └── then renders RGB/LiDAR frames with the modified scene
       └── on completion, calls SensorsimService.restore_model_parameters
```

The edit operations live entirely in the renderable model on the
server — they mutate the loaded Gaussians in memory until a
`restore_model_parameters` RPC reverts them. The `edit-assets.json`
file is just a convenient JSON encoding of the same operations the
gRPC API exposes.

**Target-layer limitation:** Interpolated-table deformable layers require
a fixed Gaussian topology because their deformation tables store one
row-aligned trajectory per Gaussian. They therefore reject live asset
insertion, PLY replacement, and baked asset edits targeting the layer.
Use a supported rigid layer, or reconstruct the scene with the legacy
deformable model, when these editing workflows are required.

## Prerequisites

- A trained reconstruction USDZ at
  `<output_dir>/<RUN-ID>/artifacts/last.usdz`. It must have been trained
  with `checkpoint.artifact.{enabled, sequence_tracks.enabled,
rig_trajectories.enabled}=true` so the embedded
  `sequence_tracks.json` is available.
- Asset-Harvester output at `<AH_OUTPUT_DIR>/`, containing per-asset
  `<track_id>/<track_id>.ply` (or `<track_id>/gs.ply`) files plus a
  top-level `metadata.yaml`. Asset capture is a separate workflow; do not
  assume an `asset-harvester` skill is installed.
- The source/dev-deps environment from [install.md](install.md).

The `metadata.yaml` produced by Asset Harvester has the shape:

```yaml
assets:
  <track_id>:
    ply_file: relative/path/to/<track_id>.ply
    label_class: car
    cuboids_dims: [4.5, 2.0, 1.8] # [size_x, size_y, size_z], metres
  <other_track_id>: ...
```

The `track_id` keys are arbitrary strings (often integer-looking).
They are used as **`replacement_id` / `asset_ids`** in the edit JSON
below. Any `@<scene_id>` suffix on a track id is stripped on import for
backwards compatibility.

## Step 1 — Repackage AH output into a new USDZ

Run [`export-external-assets`](cli-reference.md#export-external-assets)
on the trained USDZ plus the Asset-Harvester output. The full source-based
invocation template, flag matrix, and required-vs-optional table all
live in the CLI reference — only four flags need to be wired to your
local paths:

- `--artifact-path` → the trained USDZ that should be repackaged.
- `--external-assets-dir` → the Asset-Harvester output directory
  containing `metadata.yaml`.
- `--output-edit-file` → where to write the stub `edit-assets.json`.
- `--output-artifact-path` → where to write the repackaged USDZ;
  **omit to only emit the JSON** (the source USDZ is left untouched).

After this completes you have:

- A new USDZ containing the original reconstruction plus, inside its
  zip layout:

  ```text
  external_assets/
    <track_id>/<track_id>.ply        # one PLY per asset
    metadata.yaml                    # mirror of the Asset-Harvester metadata
                                     # (assets list of {track_id, label_class, cuboids_dims})
  ```

- A starter `edit-assets.json` (skeleton + the assets metadata
  pre-populated for you) that you can customise.

## Step 2 — Edit `edit-assets.json`

The JSON drives the three edit operations: `remove`, `replace`, and
`insert`. The file `export-external-assets` writes for you looks like:

```json
{
  "metadata": {
    "output_artifact_path": "/path/to/output/target-external-assets.usdz",
    "external_assets_metadata": [
      { "track_id": "8", "label_class": "car", "cuboid_dims": [4.5, 2.0, 1.8] },
      {
        "track_id": "13",
        "label_class": "truck",
        "cuboid_dims": [6.0, 2.5, 2.6]
      }
    ]
  },
  "replace": [],
  "remove": [],
  "insert": { "asset_ids": [], "data": {} }
}
```

`metadata` is informational only — it lists every asset that
`export-external-assets` packaged into the USDZ so the rest of the file
can refer to them. **`external_assets_metadata` is populated
automatically by `export-external-assets`; do not edit it by hand.**

### `remove`

List of `track_id` strings from the artifact's `sequence_tracks.json`.
These tracks are filtered out at render time so dynamic actors with
those IDs are simply not rendered:

```json
"remove": ["8", "13"]
```

The `track_id`s must exist in the source artifact's
`sequence_tracks.json`; unknown IDs are silently ignored by
`render-grpc` (older builds raise an `undefined 'remove' set` error
when the JSON is missing entirely — fixed in 25.11).

### `replace`

Each entry maps an original artifact track (`original_id`) to a
harvested asset (`replacement_id`) in the USDZ's external assets:

```json
"replace": [
  {
    "original_id":   "8",
    "replacement_id": "13",
    "object_size":   [4.5, 2.0, 1.8]
  },
  {
    "original_id":   "18",
    "replacement_id": "22",
    "object_size":   []
  },
  {
    "original_id":   "6",
    "replacement_id": "7"
  }
]
```

Constraints (all enforced by `render-grpc`):

- `original_id` MUST exist in the artifact's `sequence_tracks.json`.
- `replacement_id` MUST exist in the USDZ's `external_assets/`
  directory (i.e. it must be one of the IDs listed in
  `metadata.external_assets_metadata`, or otherwise present in the
  scene's AssetBank).
- `object_size` is `[size_x, size_y, size_z]` of the replacement's AABB
  in metres. If the field is missing, an empty list, or omitted,
  `render-grpc` falls back to the `cuboid_dims` recorded in
  `metadata.external_assets_metadata` for that `replacement_id`. If
  neither source provides a 3-float dimension, the request fails with
  an `object_size must be list of 3 floats` assertion.

In gRPC terms each entry maps to a `ReplaceAssetAction(original_id,
replacement_id, object_size=AABB(size_x, size_y, size_z))`.

### `insert`

Drop a _new_ track into the scene that points to one of the imported
assets. The `data` block follows the same shape as the artifact's
`sequence_tracks.json` (i.e. the `CuboidTracks.to_dict()` layout):

```json
"insert": {
  "asset_ids": ["18"],
  "data": {
    "tracks_data": {
      "tracks_id":             ["car_18"],
      "tracks_poses":          [/* per-frame [x,y,z, qx,qy,qz,qw] poses */],
      "tracks_timestamps_us":  [/* per-frame microsecond timestamps */],
      "tracks_label_class":    ["car"]
    },
    "cuboidtracks_data": {
      "cuboids_dims": [[4.5, 2.0, 1.8]]
    }
  }
}
```

Constraints (all enforced by `render-grpc`):

- `tracks_data.tracks_id` strings MUST NOT collide with any existing
  `track_id` in the artifact's `sequence_tracks.json`.
- `asset_ids` must be the same length as `tracks_id` (1-to-1, in
  order). The `i`-th inserted track will be rendered with the
  `i`-th asset.
- Each entry in `asset_ids` must either exist in the USDZ's
  `external_assets/` (preferred — these are the IDs listed in
  `external_assets_metadata`) **or** be a path to a `.ply` file that
  exists on the server's filesystem.
- The full `data` payload is parsed via `CuboidTracks.from_dict(...)`,
  so it must be self-consistent (every `tracks_*` array indexed by
  `track_idx`, plus the matching `cuboidtracks_data.cuboids_dims`).

For a Python walk-through of the `sequence_tracks.json` /
`CuboidTracks.to_dict()` shape (poses, timestamps, labels), see the
"Load Trajectory Data" section in
[`references/physical-ai-render.md`](./physical-ai-render.md).

In gRPC terms each entry maps to one
`DynamicObjectTrack(id, semantic_class, trajectory, object_size,
asset_id)` element appended to `EditAssetsRequest.insert`.

### Combined example

```json
{
  "metadata": {
    "output_artifact_path": "/abs/path/to/output/last-edited.usdz",
    "external_assets_metadata": [
      { "track_id": "8", "label_class": "car", "cuboid_dims": [4.5, 2.0, 1.8] },
      {
        "track_id": "13",
        "label_class": "truck",
        "cuboid_dims": [6.0, 2.5, 2.6]
      },
      { "track_id": "18", "label_class": "car", "cuboid_dims": [4.6, 2.0, 1.8] }
    ]
  },
  "remove": ["55"],
  "replace": [
    {
      "original_id": "23",
      "replacement_id": "8",
      "object_size": [4.5, 2.0, 1.8]
    },
    { "original_id": "30", "replacement_id": "13", "object_size": [] }
  ],
  "insert": {
    "asset_ids": ["18"],
    "data": {
      "tracks_data": {
        "tracks_id": ["car_18"],
        "tracks_poses": [
          [
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1]
          ]
        ],
        "tracks_timestamps_us": [[1700000000000000, 1700000000100000]],
        "tracks_label_class": ["car"]
      },
      "cuboidtracks_data": {
        "cuboids_dims": [[4.6, 2.0, 1.8]]
      }
    }
  }
}
```

## Step 3 — Launch the gRPC server with editing enabled

```bash
nurec_dev bazelisk run //:run -- \
  serve-grpc \
  --artifact-glob "$NUREC_SRC/output/target-external-assets.usdz" \
  --renderer default \
  --enable-editing-actors \
  --test-scenes-are-valid
```

`--enable-editing-actors` is **required** — without it the server
ignores `--edit-assets` on subsequent `render-grpc` calls (and direct
`SensorsimService.edit_assets` calls fail with "model does not support
actor updates").

> An older doc snippet may show `--no-enable-nrend`. That flag no
> longer exists on any NuRec command — NRend was removed. Use
> `--renderer default` (or `--renderer gsplat`).

The server saves the original training parameters of the renderable
model when the first `edit_assets` call lands; that snapshot is used
by `restore_model_parameters` (and by the `--edit-assets` flow below)
to revert state between rendering passes.

## Step 4 — Render with `--edit-assets`

```bash
nurec_dev bazelisk run //:run -- \
  render-grpc \
  --artifact-path $NUREC_SRC/output/target-external-assets.usdz \
  --output-dir    $NUREC_SRC/output/renders \
  --enable-editing-actors \
  --edit-assets   $NUREC_SRC/output/edit-assets.json
```

What `render-grpc` does behind the scenes:

1. Calls `get_external_asset_objects(scene_id)` to enumerate every
   asset id available in the AssetBank.
2. Validates each `replace.replacement_id` and every `insert.asset_ids`
   element exists in that bank (or, for `insert`, points to a `.ply`
   file on disk).
3. Validates `insert.data.tracks_data.tracks_id` does not collide
   with any track in the artifact's `sequence_tracks`.
4. Sends a single `EditAssetsRequest(scene_id, replace=[…],
insert=[…])`.
5. Renders frames along the embedded training trajectory. Tracks named
   in `remove` are filtered out per-frame at request build time.
6. After the run, calls
   `restore_model_parameters(RestoreModelParametersRequest(scene_id))`
   so the server's renderable model is back to the training state.

Rendered frames land in `--output-dir` alongside `timestamps.json`
and `render_grpc_cli_args.json`. Compare against an unedited render of
the same trajectory to verify the edits applied correctly.

## Step 5 — Driving asset edits from a custom Python client

If you don't want the bundled CLI, drive the same flow yourself:

```python
import grpc
from nurec.grpc.protos.common_pb2 import AABB
from nurec.grpc.protos.sensorsim_pb2 import (
    EditAssetsRequest,
    ExternalAssetObjectsRequest,
    ReplaceAssetAction,
    RestoreModelParametersRequest,
)
from nurec.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub

channel = grpc.insecure_channel("localhost:8080", options=[
    ("grpc.max_send_message_length", 50 * 1024 * 1024),
    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
])
client = SensorsimServiceStub(channel)
scene_id = "<your scene id>"

# Sanity-check the scene's AssetBank.
asset_ids = set(client.get_external_asset_objects(
    ExternalAssetObjectsRequest(scene_id=scene_id)).track_ids)
assert "13" in asset_ids and "8" in asset_ids

# Apply the edits once.
resp = client.edit_assets(EditAssetsRequest(
    scene_id=scene_id,
    replace=[ReplaceAssetAction(
        original_id="23",
        replacement_id="13",
        object_size=AABB(size_x=4.5, size_y=2.0, size_z=1.8),
    )],
    # insert=[DynamicObjectTrack(...)]   # see references/grpc-api.md
))
if not resp.success:
    raise RuntimeError(resp.message)

try:
    # ... call render_rgb / render_lidar here ...
    pass
finally:
    # Always restore so the next user of this server isn't surprised.
    client.restore_model_parameters(
        RestoreModelParametersRequest(scene_id=scene_id))
```

`removed` track IDs are not part of the gRPC request — the bundled
`render-grpc` CLI simply skips them when it builds each frame's
`dynamic_objects` list. If you need the same behaviour from your own
client, drop those IDs locally before populating
`RGBRenderRequest.dynamic_objects` / `LidarRenderRequest.dynamic_objects`.

## Validation checklist

After Step 4 confirm:

- The frames you expected to **remove** no longer contain those actors.
- The actors you **replaced** show the new asset geometry at the
  original track's pose.
- **Inserted** actors appear at the poses you specified and have the
  cuboid dimensions you set.
- Server logs show no `replacement_id not found` / `original_id not
found` warnings — those indicate a JSON typo against the artifact's
  `sequence_tracks.json` or the AssetBank.
- A second `render-grpc` call WITHOUT `--edit-assets` reproduces the
  unedited training rig (proves `restore_model_parameters` ran).

## Troubleshooting

| Symptom                                                          | Likely cause                                                                                                        | Fix                                                                                                                                                                                          |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `undefined 'remove' set error` from `render-grpc` (older builds) | Empty / missing `edit-assets.json`.                                                                                 | Pass the file explicitly with `--edit-assets`; ensure the JSON includes the four top-level keys (`metadata`, `replace`, `remove`, `insert`) even if some are empty. Fixed upstream in 25.11. |
| `replacement_id <id> not found in external_assets`               | The repackage step (`export-external-assets`) was skipped or pointed at a USDZ that did not include that AH output. | Re-run Step 1 with the correct `--external-assets-dir` and use the resulting `target-external-assets.usdz` everywhere.                                                                       |
| `original_id <id> not found in sequence_tracks`                  | Track ID doesn't exist in the source USDZ.                                                                          | Inspect the artifact's `sequence_tracks.json` to find a valid `track_id` (e.g. via `unzip -p last.usdz sequence_tracks.json`).                                                               |
| `object_size must be list of 3 floats for replacing`             | Replacement entry has neither `object_size` nor a matching `cuboid_dims` in `metadata.external_assets_metadata`.    | Either provide `object_size: [sx, sy, sz]` explicitly, or make sure `external_assets_metadata` carries the asset's dimensions (it should, if you didn't edit that block by hand).            |
| `Conflicting track IDs between inserted and existing tracks`     | An `insert.data.tracks_data.tracks_id` value collides with an existing track.                                       | Pick a fresh `track_id` (e.g. `car_<replacement_id>`).                                                                                                                                       |
| `asset_ids length mismatch <N> with inserted track_ids <M>`      | `insert.asset_ids` and `insert.data.tracks_data.tracks_id` are not 1:1.                                             | Re-pad the lists so they are the same length and aligned by index.                                                                                                                           |
| `Missing assets (not in AssetBank or on disk): {…}`              | An `asset_ids` entry is neither in the scene's AssetBank nor a real `.ply` file.                                    | Either repackage the USDZ to include that PLY (Step 1) or supply an absolute filesystem path for the asset.                                                                                  |
| Edits silently ignored                                           | Server was started without `--enable-editing-actors`, or the model doesn't support actor updates.                   | Restart `serve-grpc` with `--enable-editing-actors`; ensure the USDZ was trained with `checkpoint.artifact.{rig_trajectories.enabled, sequence_tracks.enabled}=true`.                        |
| Subsequent renders look "wrong" / show old edits                 | `restore_model_parameters` was not called between two passes.                                                       | Either let `render-grpc` finish (it calls restore on completion) or call `RestoreModelParametersRequest(scene_id)` from your own client; alternatively restart the server.                   |
| `Trying to edit assets but model does not support actor updates` | The artifact's renderable model lacks dynamic-actor support.                                                        | Re-train with `checkpoint.artifact.{enabled, sequence_tracks.enabled}=true` (and prefer a dynamic recipe such as `apps/AV/Waymo/3dgut_dynamic.yaml`).                                        |
