#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# session_teardown.sh -- stop + remove the warm `nurec serve-grpc`
# container booted by session_warm_server.sh.
#
# Idempotent: noop if the container or state file is already gone.
#
# Verifies that `docker ps -a` no longer reports the container before
# returning. `docker rm -f` returns as soon as it has issued SIGKILL,
# but the daemon may take up to a few seconds to actually reap the
# container record on a busy host. Returning before reap means a
# subsequent `session_warm_server.sh` boot can race with a leftover
# container of the same name and fail with "name already in use".
#
# Environment overrides:
#   NUREC_GRPC_CONTAINER -- Container name (default nurec-grpc-session)
#   NUREC_GRPC_TEARDOWN_TIMEOUT -- Max seconds to wait for the container
#                                to actually disappear from `docker ps -a`
#                                (default 15).

set -euo pipefail

CONTAINER_NAME="${NUREC_GRPC_CONTAINER:-nurec-grpc-session}"
STATE_DIR="${XDG_RUNTIME_DIR:-/tmp/nurec-$(id -u)}"
STATE_FILE="$STATE_DIR/nurec-grpc-${CONTAINER_NAME}.state"
TEARDOWN_TIMEOUT="${NUREC_GRPC_TEARDOWN_TIMEOUT:-15}"

log() {
    printf '[teardown] %s\n' "$*" >&2
}

# Prefer the CID this session recorded: a name can have been reused by an
# unrelated container since we booted, and removing that would be wrong.
RECORDED_CID=""
if [[ -f "$STATE_FILE" && ! -L "$STATE_FILE" && -O "$STATE_FILE" ]]; then
    RECORDED_CID="$(awk '{print $1}' "$STATE_FILE" 2>/dev/null || true)"
    # A recorded CID can be STALE (container already gone, state left behind).
    # Falling through to the name is right in that case; silently doing nothing
    # would leave a live server running.
    if [[ -n "$RECORDED_CID" ]] && ! docker inspect "$RECORDED_CID" >/dev/null 2>&1; then
        log "recorded container $RECORDED_CID no longer exists; falling back to the name"
        RECORDED_CID=""
    fi
fi

# Falling back to the NAME means we no longer know it is ours -- the state file
# was missing, corrupt, or stale. Removing whatever holds a common default name
# is not recoverable, so require it to look like one of our servers (it mounts a
# USDZ dir at /inputs/sdg) before touching it.
name_is_ours() {
    # A /inputs/sdg mount proves nothing -- any container could mount one, and
    # a reused name would then be deleted. Require the ownership label that
    # session_warm_server.sh stamps at creation.
    local label
    label="$(docker inspect --format '{{index .Config.Labels "com.nvidia.nurec.warm-server"}}' \
        "$CONTAINER_NAME" 2>/dev/null || true)"
    [[ "$label" == "1" ]]
}

if [[ -n "$RECORDED_CID" ]]; then
    TARGET="$RECORDED_CID"
elif docker inspect "$CONTAINER_NAME" >/dev/null 2>&1 && ! name_is_ours; then
    log "STOP: a container named '$CONTAINER_NAME' exists but does not carry the"
    log "  com.nvidia.nurec.warm-server label, and no valid state file recorded it as ours."
    log "  Refusing to remove it. Remove it yourself if you know it is stale."
    exit 1
else
    TARGET="$CONTAINER_NAME"
fi

if docker inspect "$TARGET" >/dev/null 2>&1; then
    log "stopping + removing container '$TARGET'"
    docker rm -f "$TARGET" >/dev/null

    deadline=$(( $(date +%s) + TEARDOWN_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        if ! docker inspect "$TARGET" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if docker inspect "$TARGET" >/dev/null 2>&1; then
        log "WARN: container '$TARGET' still present after ${TEARDOWN_TIMEOUT}s"
        log "  status: $(docker inspect --format '{{.State.Status}}' "$TARGET" 2>/dev/null || echo unknown)"
        log "  next boot may collide; rerun teardown or 'docker rm -f $TARGET' manually"
        exit 1
    fi
    log "container '$TARGET' confirmed gone"
else
    log "no container named '$CONTAINER_NAME'; nothing to stop"
fi

if [[ -f "$STATE_FILE" ]]; then
    rm -f "$STATE_FILE"
fi

log "DONE"
