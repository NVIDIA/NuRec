#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# session_warm_server.sh -- idempotent warm boot of `nurec serve-grpc`.
#
# Boot once at the start of a render-heavy session (chat agent,
# benchmark loop, batch job, etc.) so each per-turn render can go
# through the thin client (../NUREC_RenderClient/scripts/thin_client.py)
# instead of repeating Docker container start + Python interpreter +
# CUDA context init + model load on every turn.
#
# Idempotent: if a session warm server is already up under the canonical
# container name (default: nurec-grpc-session), exits 0 immediately.
# Otherwise boots one, polls for readiness, then exits.
#
# Capability fallback: if a caller needs something the thin client
# doesn't expose -- custom rig trajectory (--custom-rig-trajectory),
# lidar, actor / asset edits, rolling shutter, or
# --replicate-training-views semantics -- drop to per-turn `nurec render`
# CLI for that turn. The warm server stays up regardless; the next
# thin-client-supported turn re-uses it. Two model copies (server +
# CLI) will co-exist in GPU memory during a fallback turn -- fine on
# H100/80GB for single-clip-at-a-time usage, watch GPU memory on
# smaller cards.
#
# Source selection: set NUREC_SRC and NUREC_IMAGE built using references/install.md.
# Build once, then launch the built executable without holding the Bazel lock.
#
# Environment overrides:
#   NUREC_SRC              -- absolute path to released source checkout.
#   NUREC_IMAGE            -- built development image for this source checkout.
#   NUREC_GRPC_PORT          -- Server port (default 8080)
#   NUREC_GRPC_CONTAINER     -- Container name (default nurec-grpc-session)
#   NUREC_GRPC_ARTIFACT_GLOB -- Artifact glob inside the container
#                             (default /inputs/sdg/**/*.usdz -- a
#                             recursive search for any .usdz under
#                             the mounted SDG root, so we don't have
#                             to know the caller's exact per-clip
#                             layout. Covers at least:
#                               * <uuid>/usd_out/last.usdz
#                                 (historic clipgt-<UUID> /
#                                 nurec-fixer staging convention)
#                               * <uuid>/<uuid>.usdz
#                                 (PhysicalAI-Autonomous-Vehicles-NuRec
#                                 mirror layout)
#                               * any other ad-hoc per-clip layout.)
#   NUREC_GRPC_USDZ_HOST_DIR -- Host dir to bind-mount as /inputs/sdg.
#                             REQUIRED -- no default. Point at the
#                             root of your USDZ tree; the recursive
#                             glob below catches the .usdz wherever
#                             you stage it under this root. The glob
#                             runs against this mount once at server
#                             startup. Clips downloaded after startup
#                             are NOT auto-discovered -- the thin
#                             client picks them up at request time
#                             via --scene-url + --scene-id, which the
#                             SceneDownloadInterceptor turns into a
#                             runtime _register_local_scene call. So
#                             new clips still need to land under this
#                             host root (otherwise they're not
#                             reachable inside the container), but
#                             they don't need a warm-server restart.
#   NUREC_GRPC_BOOT_TIMEOUT  -- Boot wait seconds (default 120)
#   NUREC_GRPC_CACHE_SIZE    -- Server-side LRU size for loaded scenes
#                             (passed to `serve-grpc --cache-size`,
#                             default 3). Upstream's own default is
#                             10; we pin lower because each loaded
#                             USDZ holds gaussian-splat tensors +
#                             windshield model state on the GPU and
#                             A100-40GB-class hardware can't
#                             comfortably keep 10 scenes resident.
#                             3 covers the typical demo flow
#                             (default clip + 1-2 user-requested
#                             clips); raise it only if you have
#                             headroom and need more churn.
#
# Exit codes:
#   0  -- warm server up (or already up)
#   1  -- generic boot failure (timeout, container died)
#   2  -- precondition unmet (no image cached, USDZ dir missing)
#
# State file (per-user, 0700 dir) records {container_id, port}
# for session_teardown.sh to consume.

set -euo pipefail

: "${NUREC_SRC:?set the released source checkout}"
: "${NUREC_IMAGE:?set the built development image (see references/install.md)}"
# shellcheck source=/dev/null
source "$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"

CONTAINER_NAME="${NUREC_GRPC_CONTAINER:-nurec-grpc-session}"
PORT="${NUREC_GRPC_PORT:-8080}"
ARTIFACT_GLOB="${NUREC_GRPC_ARTIFACT_GLOB:-/inputs/sdg/**/*.usdz}"
USDZ_HOST_DIR="${NUREC_GRPC_USDZ_HOST_DIR:-}"
BOOT_TIMEOUT="${NUREC_GRPC_BOOT_TIMEOUT:-120}"
CACHE_SIZE="${NUREC_GRPC_CACHE_SIZE:-3}"
# State file. A fixed path under /tmp is unsafe: it is shared by every user on
# the host, and `>` follows a pre-planted symlink, so another user can both
# collide with us and have us overwrite a file they choose. Keep it per-user in
# a 0700 directory, and namespace it by container name so two sessions with
# different names do not fight.
STATE_DIR="${XDG_RUNTIME_DIR:-/tmp/nurec-$(id -u)}"
mkdir -p "$STATE_DIR" && chmod 700 "$STATE_DIR"
STATE_FILE="$STATE_DIR/nurec-grpc-${CONTAINER_NAME}.state"

write_state() {  # $1=cid $2=port -- atomic, never follows a symlink
    local tmp
    tmp="$(mktemp "${STATE_DIR}/.state-XXXXXX")"
    printf '%s %s\n' "$1" "$2" > "$tmp"
    chmod 600 "$tmp"
    mv -f "$tmp" "$STATE_FILE"
}

# `docker logs ... | grep -q` is unsafe under `set -euo pipefail`: grep -q exits
# at the first match and SIGPIPEs docker logs, so the pipeline reports non-zero
# and a server that IS serving reads as not-ready. Measured 2026-09-10 on a log
# that matched early and kept writing. Read into a variable; no pipeline.
# `docker ps ... | grep` has the same pipefail hazard as the logs check above:
# grep exits at the first match, SIGPIPEs docker, and a TRUE result reads false.
# At the readiness poll that made a live container look dead, which then removed
# its state and aborted. Ask docker inspect directly instead -- no pipeline.
container_exists() {   # $1 = name or id
    docker inspect "$1" >/dev/null 2>&1
}
container_running() {  # $1 = name or id
    [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || echo false)" == "true" ]]
}

logged_serving() {   # $1 = container id
    local logs
    logs="$(docker logs "$1" 2>&1 || true)"
    case "$logs" in *"Serving on"*) return 0 ;; esac
    return 1
}

log() {
    printf '[warm-server] %s\n' "$*" >&2
}

# Idempotency: if the container is up, do nothing (refresh state file
# in case it was nuked between sessions).
if container_running "$CONTAINER_NAME"; then
    cid="$(docker inspect --format '{{.Id}}' "$CONTAINER_NAME")"
    # A matching NAME does not make it OUR server. Verify the image and the
    # USDZ mount, and confirm it is actually SERVING -- "running" is not
    # "ready", and adopting a container with a different mount silently
    # renders the wrong scene.
    running_image="$(docker inspect --format '{{.Config.Image}}' "$cid" 2>/dev/null || true)"
    running_mount="$(docker inspect --format '{{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}' "$cid" 2>/dev/null || true)"
    running_args="$(docker inspect --format '{{join .Args " "}}' "$cid" 2>/dev/null || true)"
    expected_image="$NUREC_IMAGE"
    if [[ -n "$expected_image" && "$running_image" != "$expected_image" ]]; then
        log "STOP: container '$CONTAINER_NAME' runs $running_image, not the pinned $expected_image."
        log "  Run session_teardown.sh, or set NUREC_GRPC_CONTAINER to another name."
        exit 1
    fi
    running_label="$(docker inspect --format '{{index .Config.Labels "com.nvidia.nurec.warm-server"}}' "$cid" 2>/dev/null || true)"
    if [[ "$running_label" != "1" ]]; then
        log "STOP: container '$CONTAINER_NAME' does not carry the com.nvidia.nurec.warm-server"
        log "  label, so it was not started by this script. Refusing to adopt it."
        log "  Set NUREC_GRPC_CONTAINER to another name."
        exit 1
    fi
    if [[ "$running_mount" != *"$USDZ_HOST_DIR:/inputs/sdg"* ]]; then
        log "STOP: container '$CONTAINER_NAME' does not mount this session's USDZ dir."
        log "  mounts: $running_mount  (expected $USDZ_HOST_DIR:/inputs/sdg)"
        log "  Run session_teardown.sh, or set NUREC_GRPC_CONTAINER to another name."
        exit 1
    fi
    if [[ "$running_mount" != *"$NUREC_SRC:$NUREC_SRC"* ]]; then
        log "STOP: existing container uses a different source checkout; tear it down before switching releases."
        exit 1
    fi
    # Record what it is ACTUALLY serving, not what we asked for: adopting an
    # 8080 server while advertising 9090 sends every client to a dead port.
    running_port="$(printf '%s' "$running_args" | sed -nE 's/.*--port[= ]+([0-9]+).*/\1/p')"
    if [[ -n "$running_port" && "$running_port" != "$PORT" ]]; then
        log "container '$CONTAINER_NAME' serves port $running_port, not the requested $PORT; adopting $running_port"
        PORT="$running_port"
    fi
    if ! logged_serving "$cid"; then
        log "STOP: container '$CONTAINER_NAME' is running but has not logged 'Serving on'."
        log "  It is booting or wedged. Wait, or run session_teardown.sh and retry."
        exit 1
    fi
    write_state "$cid" "$PORT"
    log "container '$CONTAINER_NAME' already serving on port $PORT; nothing to do."
    exit 0
fi

# Boot-on-stale: a STOPPED container under our name. Same reasoning as above --
# the name alone does not make it ours, and `docker rm -f` on someone else's
# stopped container is not recoverable. Check the USDZ mount before removing.
if container_exists "$CONTAINER_NAME"; then
    # A /inputs/sdg mount is not proof of ownership -- any container could
    # have one. Require the label this script stamps at creation.
    stale_label="$(docker inspect --format '{{index .Config.Labels "com.nvidia.nurec.warm-server"}}' "$CONTAINER_NAME" 2>/dev/null || true)"
    if [[ "$stale_label" == "1" ]]; then
        log "removing stopped container '$CONTAINER_NAME' before re-boot"
        docker rm -f "$CONTAINER_NAME" >/dev/null
    else
        log "STOP: a stopped container named '$CONTAINER_NAME' exists but is not one of ours"
        log "  labels: $(docker inspect --format '{{json .Config.Labels}}' "$CONTAINER_NAME" 2>/dev/null || echo none)"
        log "  Remove it yourself, or set NUREC_GRPC_CONTAINER to another name."
        exit 1
    fi
fi

if [[ -z "$USDZ_HOST_DIR" ]]; then
    echo "STOP [warm-server]: NUREC_GRPC_USDZ_HOST_DIR is not set." >&2
    echo "  Export it to the host directory that holds your USDZ tree." >&2
    echo "  Example: NUREC_GRPC_USDZ_HOST_DIR=/data/usdz bash session_warm_server.sh" >&2
    exit 2
fi
if [[ ! -d "$USDZ_HOST_DIR" ]]; then
    echo "STOP [warm-server]: USDZ host dir '$USDZ_HOST_DIR' does not exist." >&2
    echo "  Re-check NUREC_GRPC_USDZ_HOST_DIR or create the directory before retrying." >&2
    exit 2
fi

# Build the source target in the prepared development image.
nurec_dev bazelisk build //:run
bazel_bin=$(nurec_dev bazelisk info bazel-bin)
log "source: $NUREC_SRC; development image: $NUREC_IMAGE"

log "booting '$CONTAINER_NAME' on port $PORT (artifact-glob=$ARTIFACT_GLOB cache-size=$CACHE_SIZE)"
docker run -d \
    --name "$CONTAINER_NAME" \
    --label com.nvidia.nurec.warm-server=1 \
    --gpus all --shm-size 64g \
    --network host \
    -v "$USDZ_HOST_DIR:/inputs/sdg:ro" \
    --user "$(id -u):$(id -g)" \
    --env HOME="$NUREC_BUILD_HOME" --env USER="$(id -un)" \
    --env CCACHE_DIR --env HF_TOKEN --env CUDA_VISIBLE_DEVICES \
    --volume "$NUREC_SRC:$NUREC_SRC" \
    --volume "$NUREC_OUTPUT_BASE:$NUREC_OUTPUT_BASE" \
    --workdir "$NUREC_SRC" \
    "$NUREC_IMAGE" "$bazel_bin/run" \
    serve-grpc \
        --artifact-glob "$ARTIFACT_GLOB" \
        --renderer default \
        --cache-size "$CACHE_SIZE" \
        --host 127.0.0.1 --port "$PORT" \
    >/dev/null

cid="$(docker inspect --format '{{.Id}}' "$CONTAINER_NAME")"
# State is deliberately NOT written yet: a half-booted container must not be
# advertised as a usable warm server. write_state runs after readiness below.

# Poll for readiness. `Serving on` is the canonical readiness marker
# emitted by `nurec serve-grpc` once the gRPC server is accepting
# connections.
log "waiting for 'Serving on' (timeout ${BOOT_TIMEOUT}s)..."
deadline=$(( $(date +%s) + BOOT_TIMEOUT ))
while (( $(date +%s) < deadline )); do
    if logged_serving "$cid"; then
        # Only now is it a usable warm server, so only now is it advertised.
        write_state "$cid" "$PORT"
        log "READY  container=$CONTAINER_NAME  port=$PORT  state=$STATE_FILE"
        exit 0
    fi
    if ! container_running "$cid"; then
        log "container died during boot; last 50 log lines:"
        docker logs --tail 50 "$cid" >&2 || true
        rm -f "$STATE_FILE"
        exit 1
    fi
    sleep 2
done

log "TIMEOUT waiting for serve-grpc readiness; last 50 log lines:"
docker logs --tail 50 "$cid" >&2 || true
# Do not leave a half-booted container and a stale state file behind for the
# next run to adopt. Remove by the CID we recorded, not by name.
log "removing the container that failed to become ready"
docker rm -f "$cid" >/dev/null 2>&1 || true
rm -f "$STATE_FILE"
exit 1
