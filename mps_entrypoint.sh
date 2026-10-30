#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MPS bootstrap image entrypoint: start the NVIDIA MPS control daemon, wait until it is
# live, then exec the wrapped entrypoint (passed as "$@" — the image chains /app/run, and
# runtime args append after it unchanged).
#
# MPS must be up before any CUDA context is created — CUDA initializes at Python *import*
# time, so it cannot be started from inside the server process; a process already on CUDA
# silently time-slices instead of routing through MPS. Pair with `--require-mps` on
# serve-grpc so a failed bootstrap fails the launch instead of degrading silently.
#
# The daemon needs no teardown handling here: it dies with the container's PID namespace.
set -e

export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

if ! command -v nvidia-cuda-mps-control >/dev/null; then
    echo "[mps-entrypoint] FATAL: nvidia-cuda-mps-control not found — needs the nvidia container" \
         "runtime with the 'compute' driver capability (NVIDIA_DRIVER_CAPABILITIES)." >&2
    exit 1
fi

echo "[mps-entrypoint] starting MPS control daemon (pipe=$CUDA_MPS_PIPE_DIRECTORY, log=$CUDA_MPS_LOG_DIRECTORY)"
nvidia-cuda-mps-control -d

# The daemon forks before the control socket exists; wait for it so the server's MPS
# preflight cannot race the bring-up.
for _ in $(seq 50); do
    [[ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]] && break
    sleep 0.1
done
if [[ ! -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]]; then
    echo "[mps-entrypoint] FATAL: MPS control socket did not appear within 5s" >&2
    exit 1
fi
echo "[mps-entrypoint] MPS control daemon is up; exec: $*"

exec "$@"
