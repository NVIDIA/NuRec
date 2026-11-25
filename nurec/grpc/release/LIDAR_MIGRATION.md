<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LiDAR gRPC 0.1.0: Consumer Changes

This source-versioned guide describes the interface implemented alongside it.
It is not part of the 26.10 documentation and does not assign the API to a
product release; consult the release notes for availability in published images.

This guide describes upgrading from the point-cloud API through `0.0.10` to
the LiDAR contract in API `0.1.0`. The schema additions preserve existing fields
and RPCs, but point-cloud units and some error statuses change. Pin the server
image digest and matching client bindings for reproducible deployments.

Clients enforcing a `major.minor` allowlist must explicitly accept `0.1` after
validating these changes. Retaining the point-cloud default does not mean a
strict `0.0` version check will accept this server.

## Upgrade Checklist

| Consumer                                                   | Required action                                                                                                                                        |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Existing point-cloud client omitting `output_mode`         | No output-selection change: omission returns a point cloud for every sensor, including sensors with stored firing patterns.                            |
| Client adopting per-ray output                             | Regenerate bindings and set `output_mode=LIDAR_OUTPUT_PER_RAY` explicitly. Omission returns point-cloud fields, not per-ray buffers.                   |
| Point-cloud client applying its own scene-scale correction | Remove that correction: coordinates are now metres. Earlier point-cloud serving retained NuRec length units when the world-to-NuRec scale was not one. |
| Client adopting pattern selection                          | Supply `frame_start_us` or an exact `firing_pattern_id` when looking up a multi-template sensor. Omission is accepted for a single template.           |
| Consumer branching on error codes                          | Apply the status table below; unknown sensor IDs and missing artifact capabilities are no longer generic invalid arguments.                            |
| Client reading per-ray NPZ exports                         | Read `firing_patterns.json` and resolve each frame's pattern ID. The catalog layout is the same for one or many patterns.                              |
| Repeated or multi-sensor CLI exports                       | Use a fresh output directory per invocation and sensor. Existing per-ray exports are not overwritten, merged, or resumed.                              |

Explicit point-cloud requests remain supported. `render-grpc --lidar-format bin`
and `ply` already send them; `--lidar-format per-ray` explicitly selects per-ray
output. The Python `generate_lidar_request` helper leaves `output_mode` and
`return_types` unset unless supplied. `return_types` and `include_firing_pattern`
are per-ray-only options; using them in point-cloud mode produces
`INVALID_ARGUMENT`. An explicit `firing_pattern_id` is valid in either mode.

## Units, Geometry, and Intensities

Both point-cloud coordinates and per-ray distances are in metres. If the
world-to-NuRec scale is `s`, the server divides renderer-space lengths by `s`
exactly once at serialization. For example, `5` NuRec units with `s=0.5` becomes
`10` metres in either representation. Scale-one scenes are unchanged. Internal
renderer units are not changed by this correction.

Point clouds contain valid returns only, motion-compensated into the sensor
frame at the end of the sweep. For stored patterns this is the final firing time,
which can differ from the requested pose-interval end. Per-ray output retains all emitted-ray slots and requires
the validity mask before consuming distances or intensities.

Point-cloud intensities are passed through from the model without clipping.
Per-ray intensities are clipped to `[0, 1]`. Either representation omits intensity
buffers when the model has no intensity prediction.

## Error Statuses

| Condition                                                                                                                                                     | Status                | Consumer response                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------- |
| Unknown scene, explicit lidar ID, or pattern ID                                                                                                               | `NOT_FOUND`           | Correct the ID or refresh discovery/cache.                                                                     |
| Ambiguous sensor/phase selection, invalid or duplicate return layers, unsupported output mode, per-ray-only fields in point-cloud mode, invalid pose duration | `INVALID_ARGUMENT`    | Correct the request.                                                                                           |
| Artifact lacks lidar geometry, lacks a stored pattern required by lookup/per-ray rendering, or has an unusable length scale                                   | `FAILED_PRECONDITION` | Select/re-export a capable artifact; retrying the same request does not repair it.                             |
| LiDAR rendering or pattern lookup requested from a camera-only server                                                                                         | `FAILED_PRECONDITION` | Use a server with LiDAR enabled.                                                                               |
| Backend capacity exhausted during rendering or pattern lookup                                                                                                 | `RESOURCE_EXHAUSTED`  | Apply bounded backoff. Earlier `render_lidar` mapped backend-cache capacity failures to `FAILED_PRECONDITION`. |

Malformed requests and invalid scale are rejected before ray construction or GPU
rendering where possible. A rendered-ray/template count mismatch fails the RPC;
the server must not return a successful response with misaligned arrays.

The CLI retries both renders and pattern lookups only for `RESOURCE_EXHAUSTED`,
with capped exponential backoff and at most 12 attempts. Capability errors,
unavailable transport, cancellation, and expired deadlines are not retried.

## Per-Ray Contract and Compatibility

- Existing fields and RPC names are preserved. Clients adopting per-ray output
  need bindings generated from the `0.1.0` schema.
- Output modes use `POINT_CLOUD=0` and `PER_RAY=1`. Return types use
  `UNSPECIFIED=0`, `FIRST=1`, and `STRONGEST=2`. Explicit `UNSPECIFIED` and unknown
  return types are rejected with `INVALID_ARGUMENT`; `UNSPECIFIED` is never an
  output layer.
- Empty `return_types` selects STRONGEST in per-ray mode. Explicit FIRST is
  a compatibility alias for STRONGEST, not a separate physical return. Requesting
  both duplicates the buffers intentionally; request only STRONGEST to minimize
  bandwidth.
- The server selects the pattern phase from `frame_start_us` unless an exact
  pattern ID is supplied. The explicit ID overrides phase selection, not timestamps or
  pose-window validation.
- Responses follow the pattern's ray order in layer-major layout, with per-layer
  LSB-first validity padding. Pattern metadata supports optional packet boundaries
  and either per-ray or per-packet timing. All numeric byte buffers are little-endian.
- No firing times are stretched to fill a request. A ray fires at
  `frame_start_us + timestamp_offset_us`. The accepted pose duration is from
  `min(last_firing_offset_us, frame_period_us)` through `1.05 * frame_period_us`,
  inclusive; there is no additional 5% tolerance below the lower bound.
- Pattern lookup and multi-layer/embedded responses need a sufficiently
  large gRPC receive limit; the examples use 50 MiB. Removing a duplicate return
  layer alone would not make large pattern lookups fit the stock 4 MiB limit.
- These changes do not require retraining or repackaging an otherwise compatible
  USDZ. Stored patterns and their hashes are not modified.

## Per-Ray CLI Export Layout

```text
output-dir/
  firing_patterns.json
  firing_patterns/
    pattern_000.npz
    pattern_001.npz
  repeat-000/                 # only with --repeat-count > 1
    000000.npz
    timestamps.json
  repeat-001/
    000000.npz
    timestamps.json
```

Without repeats, frames and `timestamps.json` are directly under `output-dir`.
Catalog values are always relative to `output-dir`, not to a frame's repeat
directory. Match each frame's `firing_pattern_id` to its catalog entry; do not
infer phase from `pattern_NNN.npz` numbering. A singleton uses this same layout.
The CLI refuses nonempty per-ray output directories, leaving existing exports intact.

In addition to distances, validity, optional intensities, return types, and the
pattern ID, each CLI per-ray frame contains the following request metadata:

| Field                                            | Type / shape                   | Meaning                                                                                                      |
| ------------------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `T_lidar_to_world_start`, `T_lidar_to_world_end` | float64 `(4, 4)`               | Actual request sensor-to-world poses, including mount extrinsics and CLI rig offsets.                        |
| `frame_start_us`, `frame_end_us`                 | int64 scalars                  | Actual request interval, including the CLI's one-microsecond adjustment for a zero-length recorded interval. |
| `scene_id`, `lidar_id`                           | Unicode scalars                | Requested scene and logical sensor.                                                                          |
| `return_type_names`                              | Unicode `(num_return_layers,)` | Human-readable names alongside numeric `return_types`.                                                       |

The matrices use float64 storage but represent the actual wire poses; this does
not recover precision already rounded by the protobuf pose fields.
`timestamps.json` retains recorded frame timestamps; use the NPZ's request
timestamps when reconstructing the rendering request. The low-level
`save_lidar_response` helper only adds request metadata when given its optional
`request` argument; the CLI always supplies it.

For offline world geometry, multiply a pattern direction by the valid distance
and transform it using the sensor pose at that ray's firing time. Match the
server's linear translation and quaternion-SLERP interpolation over the saved
request interval. A permitted firing-span overrun can require slight extrapolation
beyond the requested end; do not silently clamp that pose if reproducing the server.

Frames and pattern files are published atomically; the catalog is replaced only
after its referenced sidecars are complete. Unchanged sidecars are not recompressed
on every repeat. Published files respect the process umask, like ordinary file creation.
An interrupted run may leave unreferenced completed sidecars;
the catalog remains authoritative. This is not a resumable or whole-run transaction.

Repeated CLI renders explicitly pin each first-pass response's pattern ID for
later passes. They measure replay of the same geometry, not repeated timestamp
selection. Pattern caching does not mutate caller requests.

## Validation Scope

Regression coverage includes default selection for native and legacy patterns,
scaled-scene point-cloud/per-ray agreement, request validation and gRPC statuses,
malformed masks/grids, atomic-write failures, and CLI bin/PLY/per-ray orchestration
with repeated requests and saved pose metadata. These are controlled tests, not
certification of external consumer implementations. Consumers should smoke-test
their pinned image and USDZ before rollout.
