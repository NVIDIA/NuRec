<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec Local Docker Render Reference

Read this file when you need to render a NuRec USDZ scene on the
local host's GPU via `nurec_dev bazelisk run //:run -- render`.

See [source setup](install.md) for the released checkout and matching
dev-deps image, and [MP4 encoding](mp4-encoding.md) for video output.

## Environment expectations

- Shell execution on the host.
- `docker` with GPU support (`--gpus all`).
- `apt-get` (or equivalent) available for a one-time ffmpeg install
  (if missing).
- Set `NUREC_SRC` and `NUREC_IMAGE` per [install.md](install.md), then source
  `scripts/nurec_dev.sh`. Use same-path mounts for external inputs and outputs:

  ```bash
  NUREC_DOCKER_ARGS+=(--volume "$INPUT_USDZ_HOST:$INPUT_USDZ_HOST:ro")
  NUREC_DOCKER_ARGS+=(--volume "$OUTPUT_HOST:$OUTPUT_HOST")
  # For Option 2, after resolving INPUT_RIG_HOST:
  NUREC_DOCKER_ARGS+=(--volume "$INPUT_RIG_HOST:$INPUT_RIG_HOST:ro")
  ```

  Set the input/output variables and create the output directory before adding
  these mounts. Data already under `$NUREC_SRC` needs no additional mount.

`INPUT_USDZ_HOST` should point at a directory tree containing at
least one `.usdz` file. The canonical
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` layout is
`<root>/<clip-uuid>/<clip-uuid>.usdz` alongside the clip's
`labels.json` and `camera_front_wide_120fov.mp4` reference video;
scope a single invocation to one clip by setting
`INPUT_USDZ_HOST=<root>/<clip-uuid>`.

`INPUT_RIG_HOST` should point at a directory containing
`*_augmented.json`. The bundled example rig JSONs at
[`rig-json/`](rig-json/) can be used as a fallback when no
host-side rig dir is provided:

```bash
INPUT_RIG_HOST="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")/skills/nurec/references/rig-json"
```

and pass `augmented_rig.json` explicitly to `--rig-json` (note it does
**not** match a `*_augmented.json` glob — that is why the bundled dir
resolves through the fallback branch below). Surface the fallback to
the caller; do not use it silently.

Fail fast if a required input, path, or the image is missing — name
exactly what is missing.

---

## Two render modes

| Mode                             | When to use                                                        | Needs rig JSON? |
| -------------------------------- | ------------------------------------------------------------------ | --------------- |
| **Rig offset (simple)**          | Target view is a translation/rotation offset from the original rig | No              |
| **Custom rig trajectory (full)** | User provides an augmented rig JSON                                | Yes             |

For the rig-rotation sign convention (`(yaw, -roll, -pitch)` in
degrees), inspect `render --help` for the checked-out release. If the user
says "rotate 5° yaw", pass `5.0 0.0 0.0` — not `0 0 5`.

Default sequence id when unspecified: `manual-test`.

---

## Per-turn wall-time (measured, A100 40GB)

Reference numbers from a measured pass on A100 40GB hardware:
clips of ~600 poses @ 30 fps (~20 s of recorded driving), single
front-wide camera, 1080p JPEG, augmented rig overlay applied.
Numbers are across two clips run back-to-back in one warm-server
session.

| Path                                                                 | First-ever turn in session | Subsequent new-scene turn | Repeat turn, same scene |
| -------------------------------------------------------------------- | -------------------------- | ------------------------- | ----------------------- |
| Per-turn CLI fallback (Option 2 with `export-custom-rig-trajectory`) | ~3 min                     | ~3 min                    | ~3 min (no warm reuse)  |
| Warm gRPC + thin client                                              | ~2 min                     | ~35 s                     | ~25 s                   |

These historical measurements exclude the first source build. Generate a
trajectory for the actual scene and rig; no pre-baked trajectory cache is shipped.

Where the time goes (warm-thin path has **two** distinct one-time
costs that are easy to conflate):

- The **CLI cold-start tax** is paid once per `docker run`
  invocation: Docker boot + Python imports + CUDA context init +
  USDZ deserialization + GPU model upload, roughly ~50 s/container
  on this clip. Option 2 (rig-file swap) needs **two** invocations
  per turn — `nurec export-custom-rig-trajectory` to bake the
  augmented rig into a custom trajectory, then `nurec render
--custom-rig-trajectory <that-json>` to render — so it pays the
  tax twice (~85 s/turn cold-start total, before any pixels are
  rendered). Option 1 (rig offset only) needs one application launch. Each CLI call
  still starts a fresh process, so it cannot reuse the warm server state.

- The **server-process warmup tax** (~80–90 s) is paid once per
  `serve-grpc` process, on the first render call regardless of
  scene. It's the GPU side of cold-start: torch / CUDA / cuDNN
  context init + any TRT JIT compilation the artifact triggers on
  first use. Every subsequent render in that server's lifetime
  skips this tax entirely, including renders against new scenes.
- The **per-scene JIT-load tax** (~7–10 s, split across two
  phases) is paid the first time the `SceneDownloadInterceptor`
  sees a new `scene_id`. It's split between two RPCs in the
  thin-client flow:
  - The first metadata RPC (`get_available_*`) registers the
    USDZ with the renderer + deserialises it into the server's
    scene cache. Cost surfaces in `results.json.setup_s` (~7–9 s
    on this clip class — vs ~0.05 s when the scene is already
    cached).
  - The first render RPC pays the rest (~2–3 s) as a frame-0
    spike: model upload to GPU. Cost surfaces in
    `results.json.frame_stats_all.max_ms` for frame 0; the
    `frame_stats_excl_frame_0` view drops it.
    Every subsequent render against that same `scene_id`
    short-circuits the whole thing — `setup_s` returns to ~0.05 s
    and frame 0 looks like every other frame. Effectively cheap once
    the scene is warm — orders of magnitude smaller than the process
    warmup.
- The **warm-thin steady-state** (~22–42 ms/frame, ~25–40 s/turn
  on a 600-pose 20 s clip) is essentially the render itself plus a
  small per-pose gRPC round-trip — no per-turn container, Python,
  CUDA, or model-load tax. Rig-file swaps are free here too: the
  thin client overlays per-camera intrinsics on each gRPC request,
  no separate `export-custom-rig-trajectory` step needed.

So the realistic shape of a chat session: the first render of the
session pays both the warmup tax and the scene-load tax (~2 min
total on this clip class); every later new-scene render pays just
the scene-load tax on top of steady-state (~35 s); every repeat
render against an already-loaded scene is pure steady-state (~25 s).

**Per-frame cost varies with scene complexity.** Steady-state
frame time on the warm-thin path is dominated by visible splat
count, not USDZ size on disk. Measured medians on this hardware:

- Highway / open-road clips (sparse near-field, sky+road
  dominant): ~28–32 ms/frame, p95 ~32 ms, very tight spread.
- Urban / construction-zone clips (dense vehicles, signs,
  buildings, traffic): ~40–45 ms/frame, p95 ~~50 ms, wider spread
  (~~±15 % around median because frame-to-frame splat count swings
  with traffic / intersections).

Multi-camera batched runs (`thin-batch` mode) median around
~95–110 ms/pose for 3 cameras — ~33 ms per camera amortised, ~20 %
cheaper per camera than N sequential single-cam runs because pose
interpolation, gRPC serialisation, and JPEG IO amortise across
cameras in the batched RPC.

**Scaling.** Wall time is roughly linear in `frames × cameras` for
all three columns; multi-camera runs that exercise
`batch_render_rgb` (the thin-client batch mode) scale better than N
sequential single-cam invocations because the server schedules all
cameras in one shared CUDA context per pose. Different GPUs shift
the absolute numbers; the ratios between paths are what's
architectural.

`results.json` always carries the real measured `wall_s` /
`render_s` / `frame_stats_excl_frame_0` for the actual run — those
fields are the source of truth for any specific turn, and the
numbers above are just the rough expectation to set against them.

---

## Shared helper: `render_ground_truth`

Both options below render an unmodified-rig "ground truth" pass into
`$OUTPUT_HOST/$SEQUENCE_ID/gt/` before the adapted view. The render
command is identical in both flows, so it lives in one place here.
Source this helper into your shell first (it reads
`INPUT_USDZ_HOST`, `OUTPUT_HOST`, `SEQUENCE_ID`, `CAMERA_ID`,
`IMAGE_FORMAT`, `container_usdz_path`, and `gt_dir` from the
surrounding snippet's env):

```bash
render_ground_truth() {
  # Idempotent. This helper renders FRAMES, not an MP4 -- the old guard looked
  # for a gt_<cam>.mp4 that nothing here ever writes, so it never short-circuited
  # and re-rendered the ground truth on every call. `render` writes to
  # <output-dir>/<output-type>/<camera>, so check for frames there.
  # NOTE $gt_dir is the path INSIDE the container; check the host side.
  if compgen -G "$OUTPUT_HOST/$SEQUENCE_ID/gt/color/$CAMERA_ID/*.$IMAGE_FORMAT" > /dev/null 2>&1; then
    return 0
  fi
  mkdir -p "$OUTPUT_HOST/${SEQUENCE_ID}/gt"
  nurec_dev bazelisk run //:run -- render \
      --artifact-path="$container_usdz_path" \
      --output-dir="$gt_dir" \
      --renderer default \
      --image-format "$IMAGE_FORMAT" --frame-step 1 --height 1080 \
      --camera-id "$CAMERA_ID"
}
```

---

## Option 1: Rig offset (no rig JSON)

Use the built development image `NUREC_IMAGE` from source setup.
`--renderer default` loads the artifact's natively trained renderer.
For a non-rigid rig change use Option 2.

`--image-format jpeg` is **not** the default — `render` defaults to PNG
(`render.py:862-865`); only `render-grpc` defaults to JPEG. Passing jpeg
is still worth it for speed: typical end-to-end speedup over png is
~2-3× on long 1080p sequences. PIL still
encodes on CPU for both, but JPEG encode is ~10× cheaper and the file
is ~6× smaller at 1080p (~125 KB vs ~764 KB), so the per-frame disk
write is much faster too. PNG remains user-overridable.

`--height 1080` is used here in place of `--image-scale 1` (the CLI
requires exactly one of the two). NuRec USDZs in the PAI sample set
ship at 4K (3840×2160), so `--image-scale 1` would render every
frame at native 4K — ~4× the per-frame work and disk write of the
warm-thin path's 1080p output, with no visible benefit in the
side-by-side comparison viewer (which also pre-scales to 1080p for
parity). 1080p also matches the warm-thin client's default so a
CLI-fallback turn drops in cleanly next to a previous warm-thin
render in the same viewer set. Bump to native 4K with
`--image-scale 1` (and drop `--height`) only if a user explicitly
asks for "full-resolution" / "native" output.

```bash
set -euo pipefail

source "$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"

INPUT_USDZ_HOST=/host/path/to/usdz
OUTPUT_HOST=/host/path/to/outputs

SEQUENCE_ID=manual-test
CAMERA_ID="${CAMERA_ID:-camera_front_wide_120fov}"
IMAGE_FORMAT="${IMAGE_FORMAT:-jpeg}"
output_dir=$OUTPUT_HOST/$SEQUENCE_ID
gt_dir="$OUTPUT_HOST/${SEQUENCE_ID}/gt"

command -v ffmpeg >/dev/null || { apt-get update && apt-get install -y ffmpeg; }

# Pick a .usdz under INPUT_USDZ_HOST. Prefer the NuRec canonical layout
# `<uuid>/<uuid>.usdz`; fall back to any *.usdz the recursive search
# finds (covers the historic `<uuid>/usd_out/last.usdz` and any
# user-supplied layout).
found_usdz_path=$(find "$INPUT_USDZ_HOST" -type f -name "*.usdz" -print -quit || true)
[ -n "${found_usdz_path:-}" ] || { echo "No .usdz under $INPUT_USDZ_HOST"; exit 1; }
container_usdz_path="$found_usdz_path"

mkdir -p "$OUTPUT_HOST/$SEQUENCE_ID"
NUREC_DOCKER_ARGS+=(--volume "$INPUT_USDZ_HOST:$INPUT_USDZ_HOST:ro" --volume "$OUTPUT_HOST:$OUTPUT_HOST")

# Ground truth — uses render_ground_truth() from the shared helper above.
render_ground_truth

# Adapted — replace the offset values with whatever the user asked for.
# Translation is (x,y,z) meters, rig frame.
# Rotation is (yaw, -roll, -pitch) degrees — verify against render --help.
nurec_dev bazelisk run //:run -- render \
    --artifact-path="$container_usdz_path" \
    --output-dir="$output_dir" \
    --rig-translation-offset 0.0 0.5 0.0 \
    --rig-rotation-offset 0.0 0.0 5.0 \
    --no-replicate-training-views \
    --renderer default \
    --image-format "$IMAGE_FORMAT" --frame-step 1 --height 1080 \
    --camera-id "$CAMERA_ID"
```

Then encode MP4s with the host's ffmpeg — see
[`mp4-encoding.md`](mp4-encoding.md). If a downstream comparison
viewer is in use (e.g. `nurec-compare`-style side-by-side), follow
its `generate-manifest.sh` to refresh the manifest after the new
MP4 pair lands; the filename rules in `mp4-encoding.md` are what
those manifest scans look for.

---

## Option 2: Custom rig trajectory (full flow)

Use `export-custom-rig-trajectory` from the released source to generate
a trajectory for this scene and this rig.

`--renderer default` is the CLI default here; `--image-format jpeg` must be
passed explicitly, since `render` defaults to png
for the same reasons documented in Option 1 above.

```bash
set -euo pipefail

source "$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"

INPUT_USDZ_HOST=/host/path/to/usdz
INPUT_RIG_HOST=/host/path/to/augmented_rig
OUTPUT_HOST=/host/path/to/outputs

SEQUENCE_ID=manual-test
CAMERA_ID="${CAMERA_ID:-camera_front_wide_120fov}"
IMAGE_FORMAT="${IMAGE_FORMAT:-jpeg}"
output_dir=$OUTPUT_HOST/$SEQUENCE_ID
gt_dir="$OUTPUT_HOST/${SEQUENCE_ID}/gt"

command -v ffmpeg >/dev/null || { apt-get update && apt-get install -y ffmpeg; }

found_usdz_path=$(find "$INPUT_USDZ_HOST" -type f -name "*.usdz" -print -quit || true)
[ -n "${found_usdz_path:-}" ] || { echo "No .usdz under $INPUT_USDZ_HOST"; exit 1; }
container_usdz_path="$found_usdz_path"

# `|| true`: a missing $INPUT_RIG_HOST makes find exit non-zero, and under
# `set -euo pipefail` that aborted here -- before the fallback below could run.
mapfile -t AUGMENTED_RIG_MATCHES < <(find "$INPUT_RIG_HOST" -name "*_augmented.json" -type f -print 2>/dev/null || true)
if [ "${#AUGMENTED_RIG_MATCHES[@]}" -gt 1 ]; then
  echo "Multiple *_augmented.json under $INPUT_RIG_HOST; pass an explicit --rig-json path."
  exit 1
fi
AUGMENTED_RIG="${AUGMENTED_RIG_MATCHES[0]:-}"
if [ -z "$AUGMENTED_RIG" ]; then
  # Fallback to the bundled example rig that ships with this skill.
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
  FALLBACK_RIG_DIR="$REPO_ROOT/skills/nurec/references/rig-json"
  if [ -f "$FALLBACK_RIG_DIR/augmented_rig.json" ]; then
    echo "No *_augmented.json under $INPUT_RIG_HOST; falling back to bundled example rig at $FALLBACK_RIG_DIR/augmented_rig.json"
    INPUT_RIG_HOST="$FALLBACK_RIG_DIR"
    AUGMENTED_RIG="$FALLBACK_RIG_DIR/augmented_rig.json"
  else
    echo "No *_augmented.json under $INPUT_RIG_HOST and no bundled fallback at $FALLBACK_RIG_DIR/augmented_rig.json"; exit 1
  fi
fi
container_rig_path="$AUGMENTED_RIG"
NUREC_DOCKER_ARGS+=(--volume "$INPUT_RIG_HOST:$INPUT_RIG_HOST:ro")

mkdir -p "$OUTPUT_HOST/$SEQUENCE_ID"
NUREC_DOCKER_ARGS+=(--volume "$INPUT_USDZ_HOST:$INPUT_USDZ_HOST:ro" --volume "$OUTPUT_HOST:$OUTPUT_HOST")

# Ground truth — uses render_ground_truth() from the shared helper above.
render_ground_truth

# Generate for this exact scene and rig; do not reuse another scene's trajectory.
nurec_dev bazelisk run //:run -- export-custom-rig-trajectory \
  --artifact-path="$container_usdz_path" \
  --rig-json "$container_rig_path" \
  --output "$output_dir/custom_rig_trajectories.json"
python3 -m json.tool "$output_dir/custom_rig_trajectories.json" > /dev/null


# Render the adapted view using that trajectory
nurec_dev bazelisk run //:run -- render \
    --artifact-path="$container_usdz_path" \
    --output-dir="$output_dir" \
    --custom-rig-trajectory "$output_dir/custom_rig_trajectories.json" \
    --no-replicate-training-views \
    --renderer default \
    --image-format "$IMAGE_FORMAT" --frame-step 1 --height 1080 \
    --camera-id "$CAMERA_ID"

# Copy rig JSONs next to the render for provenance
ORIGINAL_RIG=$(find "$INPUT_RIG_HOST" -name "*.json" ! -name "*_augmented.json" -type f -print -quit)
[ -n "$ORIGINAL_RIG" ] && cp "$ORIGINAL_RIG" "$OUTPUT_HOST/$SEQUENCE_ID/$(basename "$ORIGINAL_RIG")"
cp "$AUGMENTED_RIG" "$OUTPUT_HOST/$SEQUENCE_ID/$(basename "$AUGMENTED_RIG")"
```

Then encode MP4s with the host's ffmpeg — see
[`mp4-encoding.md`](mp4-encoding.md). If a downstream comparison
viewer is in use (e.g. `nurec-compare`-style side-by-side), follow
its `generate-manifest.sh` to refresh the manifest after the new
MP4 pair lands.

---

## Output reporting (local specifics)

Outputs are written directly to the host under `OUTPUT_HOST`. Report
the host path (e.g.
`$OUTPUT_HOST/manual-test/camera_front_wide_120fov.mp4`) as the final
MP4 location.

## Failure handling (local specifics)

Watch for:

- `docker` missing, not on PATH, or no GPU support configured.
- `docker pull` fails — verify the published dependency-image tag and
  its access requirements. See [install.md](install.md).
- `apt-get install -y ffmpeg` fails — host has no network or no
  `apt`. Install ffmpeg through the host's native package manager.

State exactly what is missing, include paths/identifiers, and suggest
the smallest next action.
