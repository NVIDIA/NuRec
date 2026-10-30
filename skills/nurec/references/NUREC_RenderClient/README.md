<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec Render Client (warm-server gRPC, host-side Python)

A small host-side Python gRPC client (`scripts/thin_client.py`) for
talking to an `nurec serve-grpc` server directly, without re-paying the
client-container cold-start (Docker boot + Python interpreter + CUDA
context init) on every render. Companion to (not a replacement for)
the bundled `nurec render` and `nurec render-grpc` CLIs.

Use this when:

- A render-heavy session renders the same scene multiple times. Boot
  one `serve-grpc` for the session, then fire the thin client per
  turn.
- A multi-camera turn needs to render N cameras as one server-side
  scheduling unit. The bundled `render-grpc` CLI is single-camera, so
  the only way to reach the proto's `batch_render_rgb` RPC is from a
  Python client.
- A turn asks for a **novel rig view** ("shift the rig 0.5m left",
  "yaw the cameras 5 degrees right"). The thin client supports
  `--rig-translation-offset X Y Z` and `--rig-rotation-offset YAW NROLL NPITCH`
  with the same axis convention as `nurec render`, so the rendered
  output is directly comparable to the per-turn-CLI baseline.
- A turn asks to **render with a different rig file** (e.g. an
  `augmented_rig.json` that adds bivariate-windshield distortion on
  top of the recorded base rig for the front-wide camera). Pass
  `--rig-file <path>` and the thin client overrides the per-camera
  intrinsics on the gRPC request: F-theta lens
  (`pixeldist_to_angle_poly`, cx/cy, optional linear-cde) and, when
  all four windshield polys are present,
  `bivariate_windshield_model_param` (forward + inverse 2D polys).
  Cameras absent from the rig file fall through to the recorded
  intrinsics, so partial overrides are fine. Combine with
  `--rig-translation-offset` / `--rig-rotation-offset` to render a
  novel viewpoint _through_ a different rig.
- A turn asks to **switch to a new scene** ("download clip XYZ and
  render it") without restarting the warm server. Stage the USDZ
  anywhere under the host-side bind-mounted root (whatever path you
  passed to `NUREC_GRPC_USDZ_HOST_DIR` when booting the warm server)
  and pass `--scene-url file:///inputs/sdg/<sub-path>.usdz` (or
  `https://...usdz`) plus a unique `--scene-id`; the server's
  `SceneDownloadInterceptor` registers / downloads the scene on the
  first RPC of the run. Subsequent runs against the same scene-id
  pay nothing — the scene is in the server's `--cache-size` LRU.
  Do **not** use this for:

- One-off renders where Docker cold-start is irrelevant. Use
  `nurec render` (covered by the main `nurec` skill and `../local-render.md`).
- Workflows that need rolling-shutter modeling, dynamic obstacle
  edits, or asset edits — the thin client uses instant-shutter framing
  and doesn't model dynamic actors / asset edits. The bundled
  `render-grpc` CLI covers all of that.

---

## Quick start

```bash
RUN_ID=latest                     # output subdir tag for this render run
SUB_PATH=clips/my-scene           # container-side subdir under /inputs/sdg
NEW_CLIP=my-clip-name             # scene-id + USDZ basename; must match [A-Za-z0-9_-]+
RIG_HOST_DIR=/path/to/rig-dir     # host dir holding augmented_rig.json

# 1. Complete ../install.md: set NUREC_SRC and build NUREC_IMAGE on NUREC_DEV_DEPS_IMAGE.
# Run this block from this README's directory in a Python 3.13 client venv.
python -m pip install -r "$NUREC_SRC/nurec/grpc/release/requirements.txt" numpy scipy

# 2. Generate stubs from the same source release as the server.
bash scripts/setup_protos.sh

export NUREC_GRPC_PORT="${NUREC_GRPC_PORT:-8080}"

# 3. Build and boot once. The helper retains the /inputs/sdg scene mount,
#    default port 8080, readiness poll, and session state for teardown.
USDZ_HOST=/path/to/your/usdz/root
NUREC_GRPC_USDZ_HOST_DIR="$USDZ_HOST" bash ../../scripts/session_warm_server.sh

# 4. Render one camera, all trajectory poses, JPEG out (recorded view).
#    NOTE: use a Python that has grpcio/protobuf/numpy/scipy installed.
#    Override the default interpreter via NUREC_CLIENT_PYTHON if
#    your venv is not activated.
python3 scripts/thin_client.py \
    --port="${NUREC_GRPC_PORT:-8080}" \
    --cameras camera_front_wide_120fov \
    --output-dir "$USDZ_HOST/outputs/${RUN_ID:?}/"

# 5. Render four cameras in one batched RPC per pose (recorded view):
python3 scripts/thin_client.py \
    --port="${NUREC_GRPC_PORT:-8080}" \
    --cameras camera_cross_left_120fov,camera_cross_right_120fov,camera_front_tele_30fov,camera_rear_right_70fov \
    --mode batch \
    --output-dir "$USDZ_HOST/outputs/${RUN_ID:?}-multicam/"

# 6. Novel rig view. Shift rig 0.5m along Y and yaw 5 degrees right
#    (axis convention matches `nurec render --rig-translation-offset` /
#    `--rig-rotation-offset`):
python3 scripts/thin_client.py \
    --port="${NUREC_GRPC_PORT:-8080}" \
    --cameras camera_front_wide_120fov \
    --rig-translation-offset 0 0.5 0 \
    --rig-rotation-offset 5 0 0 \
    --output-dir "$USDZ_HOST/outputs/${RUN_ID:?}-novel/"

# 7. Render a NEW scene that wasn't in `serve-grpc --artifact-glob`
#    at boot, without restarting the warm server. The path is
#    container-side (must be inside a bind mount the server can read,
#    e.g. anywhere under /inputs/sdg/ when the warm server bind-mounts
#    $USDZ_HOST); --scene-id is the cache key + filename, and must
#    match the server-side regex [A-Za-z0-9_-]+.
python3 scripts/thin_client.py \
    --port="${NUREC_GRPC_PORT:-8080}" \
    --cameras camera_front_wide_120fov \
    --scene-url "file:///inputs/sdg/${SUB_PATH:?}/${NEW_CLIP:?}.usdz" \
    --scene-id "${NEW_CLIP:?}" \
    --output-dir "$USDZ_HOST/outputs/${RUN_ID:?}-new-scene/"

# 8. Render with a different rig file (e.g. add windshield optics on
#    top of the recorded base rig). The rig JSON's per-camera
#    intrinsics overlay the server's recorded values; cameras absent
#    from the file fall through.
python3 scripts/thin_client.py \
    --port="${NUREC_GRPC_PORT:-8080}" \
    --cameras camera_front_wide_120fov \
    --rig-file "${RIG_HOST_DIR:?}/augmented_rig.json" \
    --output-dir "$USDZ_HOST/outputs/${RUN_ID:?}-augmented/"
```

The client writes:

- `<output-dir>/000000.jpeg ...` (single-cam) or `<output-dir>/<cam>/000000.jpeg ...` (batch).
- `<output-dir>/timestamps.json` — per-frame timing array.
- `<output-dir>/results.json` — `{setup_s, render_s, wall_s, frame_stats_excl_frame_0, ...}`.

Encode the JPEGs to MP4 with the host's `ffmpeg` — see
[`../mp4-encoding.md`](../mp4-encoding.md) for the canonical recipe
and filename conventions.

---

## Why this skill exists

The **demo-path latency floor problem.** A naive per-turn `docker
run … nurec render` repays the same fixed overhead every call:

- `docker run` container startup (cgroups, namespaces, GPU device
  passthrough)
- Python interpreter startup + heavy imports (torch, NuRec modules)
- CUDA context init + cuDNN load
- Model load (USDZ deserialization, checkpoint to GPU, any TRT JIT
  warm-up the artifact triggers on first use)
- Then the actual render

The first four are pure overhead — they don't depend on trajectory
length, camera count, or anything else the caller controls. On a
representative ~20 s NuRec clip that overhead is roughly a minute
per turn on its own, independent of clip length. The architectural
fact this skill leans on is just "everything except 'the actual
render' is cold-start tax". Parent skills that care about absolute
numbers should keep their own measured tables (see e.g.
[`../local-render.md`](../local-render.md)); this skill deliberately
stays clip- and hardware-agnostic.

Standing up one `serve-grpc` for the whole session and hitting it via
the Docker-bundled `render-grpc` cuts the **server-side** cold-start
to once per session, but **the Docker client itself still pays its
own cold-start every call** — `render-grpc` is an source-built NuRec
binary, so each invocation is a fresh container with the full Python
import + CUDA context init even though the client never loads a
model.

The thin client bypasses the client-side container entirely:

- Talk to the server's gRPC port from the host's Python directly.
- Per-call wall = server-side render time + gRPC round-trip overhead
  - (one-time-per-process) Python startup. The architectural change
    is that the **client-side Docker + Python + CUDA tax is gone every
    call after the first**. The server-side render time itself is
    unchanged (it's the same `serve-grpc` doing the same work).
- For multi-camera turns, dispatch all cameras in one
  `batch_render_rgb` RPC. Server gets to schedule the work itself
  (single CUDA context, shared scene state), instead of the host
  sequentially booting N Docker clients.

---

## How the thin client works

The gRPC server already exposes everything the client needs — there
is no NuRec-side change required:

| Setup RPC (once at start)    | Purpose                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| `get_available_scenes`       | Pick `scene_id` (or pass `--scene-id`).                                                  |
| `get_available_cameras`      | Per-camera intrinsics + `rig_to_camera` (misnamed proto field; actually `T_camera_rig`). |
| `get_available_trajectories` | Recorded ego trajectory: `Trajectory.poses` = list of `(timestamp_us, T_rig_world)`.     |

| Render RPC (per pose) | Purpose                                                                                                                                                                                                                                             |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `render_rgb`          | One camera, one pose, one image_bytes blob back. Used by `--mode single`.                                                                                                                                                                           |
| `batch_render_rgb`    | N cameras, one pose, list of image_bytes blobs back (one per camera). Each `BatchRGBRenderRequestItem` carries a full per-camera `RGBRenderRequest`, so per-camera intrinsics / poses / formats / qualities can all differ. Used by `--mode batch`. |

**`--target-fps` defaults to `30.0`, so by default the recorded trajectory is
RESAMPLED, not replayed pose-for-pose.** Pass `--target-fps 0` to iterate
`Trajectory.poses` verbatim. There is no rolling-shutter modeling either way.

For each pose `(t, T_rig_world)` the script:

1. Composes `T_camera_world = T_rig_world @ rig_offset_se3 @ T_camera_rig`
   (per camera). Composition + axis convention match
   `nurec.utils.io.export.render_grpc.py`, so with `--target-fps 0` the output
   is comparable to the per-turn-CLI baseline. At the default resampling
   rate the pose set differs, so do not expect a bytewise match.
2. Build `RGBRenderRequest` (or N items wrapped in `BatchRGBRenderRequest`).
3. Send. Write `image_bytes` to disk. Record per-call wall time.

`rig_offset_se3` is built once per invocation from
`--rig-translation-offset` / `--rig-rotation-offset` via a port of
`nurec.utils.geometry.pose_offsets_to_se3` (NCore's `euler_2_so3` is
replaced with scipy's `Rotation`; the val-mode axis-permutation hack
from NuRec is replicated client-side so the input tuple semantics —
`(yaw, -roll, -pitch)` degrees — match the CLI exactly). Default
`(0, 0, 0)` for both gives identity, i.e. the recorded view.

The thin client renders at the trajectory's natural sample points. A
sim-driven caller that wants to render at arbitrary timestamps would
need to interpolate poses between recorded samples — that's a small
addition on top of the cached trajectory but not something this skill
implements out of the box.

---

## Loading new scenes at runtime (no warm-server restart)

`nurec serve-grpc` ships a `SceneDownloadInterceptor`
(`nurec/grpc/serve.py`) that runs on every gRPC request and looks for
two metadata headers:

- `x-nre-scene-url` — `file:///container-side/path.usdz` (registers
  the local file) or `https://...usdz` (downloads via the server's
  downloader). **Plain `http://` is rejected**: the server builds its
  `Downloader` with clear-text disabled, so only `https://` and `file://`
  work.
  cache directory).
- `x-nre-scene-id` — the cache key + filename. Must match the
  server-side regex `^[A-Za-z0-9_-]+$`. Validated client-side too.

If the scene under that id isn't already in the server's backend
cache, the interceptor registers/downloads + loads it synchronously
_before_ the actual RPC continues. Subsequent calls with the same id
short-circuit. The server holds up to `--cache-size N` loaded scenes
in an LRU; new scenes evict the least-recently-used when full. The
upstream default is 10, but each loaded USDZ holds gaussian-splat
tensors plus any windshield-model state on the GPU, so parent skills
should pin a value sized to their hardware (e.g.
[`../../scripts/session_warm_server.sh`](../../scripts/session_warm_server.sh)
defaults to `--cache-size 3` for A100-40GB, raisable via
`NUREC_GRPC_CACHE_SIZE`).

The thin client exposes this via `--scene-url` + `--scene-id`. When
both are set, every RPC of the run carries the metadata, so:

- The first RPC pays the load cost (USDZ deserialization, model
  upload to GPU). Comparable to a `serve-grpc` cold-start for that
  one scene.
- Every subsequent RPC of the run, and every subsequent
  `thin_client.py` invocation against the same scene-id, is back to
  the warm-render fast path.

Practical usage from a parent skill:

1. User: "render clip XYZ from HuggingFace".
2. Parent skill downloads the USDZ to a path inside the warm
   server's bind mount (anywhere under the host-side root you
   bind-mounted into `/inputs/sdg` is fine).
3. Parent skill invokes `thin_client.py --scene-url
file:///inputs/sdg/<sub-path>/<clip>.usdz --scene-id <clip>
--cameras ...`.
4. First RPC of run = USDZ register + load (~serve-grpc cold-start
   for that one artifact). Per-pose renders thereafter run at warm
   speed.
5. Next turn against `<clip>` ("now shift left 0.5m") = pure-warm
   path; parent skill reuses `--scene-url` + `--scene-id` and the
   server's interceptor sees "already loaded", short-circuits.

If the user rotates through more than `--cache-size` distinct
scenes in a session, the server evicts the least-recently-used one
and the next turn against that scene re-pays the load cost. Bump
`--cache-size` on the server boot only if you have GPU memory
headroom AND expect lots of scene churn — the cost is GPU-resident
splat tensors per scene, not CPU/disk.

---

## Proto stubs from the matching source release

Run `scripts/setup_protos.sh` in the client Python environment after setting
`NUREC_SRC`. It generates stubs with `grpcio-tools` from the released schemas.
Set `NUREC_CLIENT_PYTHON` to select the generator's interpreter and
`NUREC_RENDER_CLIENT_PROTO_DIR` to override the default
`~/.cache/nurec-render-client/proto` output directory. Use the same proto directory
for the thin client, and regenerate when switching source/server releases.

Dependencies: Python 3.13, `grpcio`, `grpcio-tools`, `protobuf`, `numpy`, and
`scipy`. The client does not need the full NuRec Python package.

---

## Limitations (what the thin client deliberately doesn't do)

These are punted to the bundled `render-grpc` CLI (or future
iterations of this skill if a chat user actually needs them):

1. **Rolling shutter modeling.** `start_pose == end_pose` per frame;
   the server gets a `[t, t+1)` instant-shutter window. Fine for the
   demo path; not faithful to the recorded camera physics.
2. **Dynamic obstacle / actor edits, asset edits.** Out of scope.
3. **Lidar rendering.** RGB only. Lidar via `render-grpc --lidar`.
4. **Egocar hood overlay.** Not supported on either path. The
   bundled `render-grpc` CLI has `--egocar-hood-dir` plumbing but
   the thin client deliberately does not invoke it; the standalone
   `nurec render` CLI has no hood plumbing at all.
5. **Stale-server detection.** If the server dies between calls,
   the script raises a gRPC error on the next RPC. Caller
   (the parent skill) is responsible for restarting if needed.
6. **Sim-driven timestamps / interpolated poses.** Renders at the
   trajectory's natural sample points. If a parent skill needs
   `get_pose_at_timestamp`-style interpolation between recorded
   samples, implement interpolation over the cached trajectory. For the typical "novel rig view" use case,
   the rig translation/rotation offsets are sufficient — the
   trajectory itself doesn't need resampling.

---

## Integrating from another skill

A parent skill that wants the warm-server flow should:

1. Run [`scripts/setup_protos.sh`](scripts/setup_protos.sh) once per
   matching source/server release.
2. Boot the warm server via
   [`../../scripts/session_warm_server.sh`](../../scripts/session_warm_server.sh)
   (or its own equivalent `docker run … serve-grpc …` if it needs
   custom boot flags).
3. Invoke [`scripts/thin_client.py`](scripts/thin_client.py) per
   turn through the venv interpreter
   (`$NUREC_CLIENT_PYTHON`, or `python3` on PATH).

```bash
# Once per session:
NUREC_SKILL_DIR=/path/to/nurec-skill
bash "$NUREC_SKILL_DIR/references/NUREC_RenderClient/scripts/setup_protos.sh"

export NUREC_GRPC_PORT="${NUREC_GRPC_PORT:-8080}"

# Boot warm server (once per session). Either use the bundled script:
NUREC_GRPC_USDZ_HOST_DIR=/path/to/usdz/root \
    bash "$NUREC_SKILL_DIR/scripts/session_warm_server.sh"
# The helper polls for readiness before returning.

# Per-turn render. The parent skill is expected to translate user
# intent ("shift left 0.5m, yaw 5deg right") into the X Y Z and
# YAW NROLL NPITCH tuples here; pass 0 0 0 for both to render the
# recorded view.
#
# When the user asks for a NEW scene (HuggingFace clip, S3 USDZ,
# etc.), the parent skill downloads it into a path that is inside
# the warm server's bind mount, then sets $SCENE_URL + $SCENE_ID;
# the warm server loads on first RPC and caches in LRU. For the
# default sample clip (already in --artifact-glob at boot), leave
# $SCENE_URL / $SCENE_ID unset.
#
# When the parent skill wants to render with a non-recorded rig
# (e.g. an augmented_rig with windshield optics on top of the
# recorded base rig), point $RIG_FILE at the rig JSON; the thin
# client overrides the per-camera intrinsics on each request.
# Leave $RIG_FILE unset to render with the rig baked into the USDZ.
"${NUREC_CLIENT_PYTHON:-python3}" \
    "$NUREC_SKILL_DIR/references/NUREC_RenderClient/scripts/thin_client.py" \
    --port="${NUREC_GRPC_PORT:-8080}" \
    --cameras "$CAMERAS_CSV" \
    --rig-translation-offset "$TX" "$TY" "$TZ" \
    --rig-rotation-offset "$RYAW" "$RNROLL" "$RNPITCH" \
    ${RIG_FILE:+--rig-file "$RIG_FILE"} \
    ${SCENE_URL:+--scene-url "$SCENE_URL"} \
    ${SCENE_ID:+--scene-id "$SCENE_ID"} \
    --output-dir "$TURN_OUT_DIR" \
    --image-quality 95 --height 1080
# results.json is at $TURN_OUT_DIR/results.json
# (also records `rig_translation_offset_m`, `rig_rotation_offset_deg`,
# `is_novel_view`, `rig_file`, `rig_file_overrides`, `scene_url` so
# post-hoc debugging / video naming knows what was rendered against
# which scene + rig). `rig_file_overrides` is a per-camera dict:
#   {"<cam>": {"ftheta": true, "windshield": <bool>,
#              "rig_resolution": [<w>, <h>] | null}}
# `windshield: false` when the rig file was missing one or more of
# the four bivariate-windshield polys; `rig_resolution: null` when
# the rig file didn't declare a calibration resolution. Compare
# `rig_resolution` against the recorded camera resolution to see
# whether F-theta polys / cx / cy were rescaled.
# JPEGs at $TURN_OUT_DIR/000000.jpeg ... (single) or $TURN_OUT_DIR/<cam>/... (batch)

# In the parent skill's session teardown:
bash "$NUREC_SKILL_DIR/scripts/session_teardown.sh"
```

The thin client is stateless across invocations (each call opens a
fresh gRPC channel). Boot/teardown is the parent skill's responsibility.

---

## File layout

In this repo (committed) — nested under the `nurec` skill as one of
its references:

```
<nurec-skill-dir>/references/NUREC_RenderClient/
  README.md                         # this file
  scripts/
    thin_client.py                  # the host-side gRPC client
    setup_protos.sh                 # generate stubs from matching source schemas
```

Cached at runtime (per-user, regenerated when the source/server release changes):

```
~/.cache/nurec-render-client/
  proto/
    nurec/grpc/protos/
      common_pb2.py
      common_pb2_grpc.py
      sensorsim_pb2.py
      sensorsim_pb2_grpc.py
```

Use a Python 3.13 virtual environment with the client dependencies above.
Set `NUREC_CLIENT_PYTHON` to its interpreter when generating the protobufs.

---

## Related

- [`../local-render.md`](../local-render.md) — per-turn `nurec render`
  CLI flow. Use it when the thin client doesn't expose what you
  need (custom trajectory, lidar, rolling shutter, in-container
  video encode, actor edits).
- [`../../scripts/session_warm_server.sh`](../../scripts/session_warm_server.sh)
  - [`../../scripts/session_teardown.sh`](../../scripts/session_teardown.sh)
    — bundled session-scoped warm-server boot / teardown that wraps
    source build and launch with readiness checks and idempotency.
- The `nurec` skill — the bundled `nurec render` and
  `nurec serve-grpc` / `nurec render-grpc` CLI reference for the matching source release. Use the CLIs for
  one-off renders, scenes that need rolling-shutter or actor edits,
  or any non-RGB workload. See
  [`../nurec-skill-catalog.md`](../nurec-skill-catalog.md) for the
  full routing block.
- [Public Harmonizer setup](../harmonizer.md) — optional post-processing. The thin client returns raw
  `image_bytes` blobs; if the parent skill wants harmonized frames,
  it runs the fixer separately on the JPEGs.
