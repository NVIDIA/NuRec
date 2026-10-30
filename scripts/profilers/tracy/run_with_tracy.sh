#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run NuRec with Tracy profiling enabled

set -euo pipefail

# Parse command line arguments
GPU_PROFILING=false
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_PROFILING=true
            shift
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            # Not a flag, assume it's the start of the command
            break
            ;;
    esac
done

if [[ "$SHOW_HELP" = true ]]; then
    echo "Usage: $0 [OPTIONS] <command> [args...]"
    echo ""
    echo "Run NuRec with Tracy profiling enabled"
    echo "Supported commands: serve-grpc, render, render-grpc"
    echo ""
    echo "Options:"
    echo "  --gpu                Enable GPU profiling (requires CUPTI)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 serve-grpc --host 127.0.0.1"
    echo "  $0 --gpu serve-grpc"
    echo ""
    echo "Notes:"
    echo "  - Connect the Tracy 0.12.2 GUI to the application on port 8086"
    echo "  - Script automatically uses appropriate Bazel target"
    echo "  - Bazel supplies hermetic CUDA and CUPTI dependencies"
    echo "  - Run from the repository root on a CUDA-capable NVIDIA GPU host"
    exit 0
fi

# Check if command was provided
if [[ $# -eq 0 ]]; then
    echo "ERROR: No command provided" >&2
    echo "Use --help for usage information" >&2
    exit 1
fi

# These commands expose the ScopedTimer flags appended below. Hydra training
# uses config overrides instead, so passing these flags to main does not work.
case "$1" in
    serve-grpc|render|render-grpc) ;;
    *)
        echo "ERROR: Tracy wrapper supports serve-grpc, render, and render-grpc; got '$1'." >&2
        exit 1
        ;;
esac

echo "🔍 Starting Tracy profiling..."

# Setup GPU profiling if requested
if [[ "$GPU_PROFILING" = true ]]; then
    echo "🎮 GPU profiling enabled"

    # Bazel supplies hermetic CUDA/CUPTI dependencies for GPU profiling.
else
    echo "💻 CPU-only profiling"
fi

echo "Command: $*"
echo ""
echo "Tracy should be listening on port 8086"
echo "   Connect with Tracy GUI to see real-time profiling data"
echo ""

# Use the appropriate Tracy-enabled Bazel target (they have different dependencies)
if [[ "$GPU_PROFILING" = true ]]; then
    BAZEL_TARGET="//:run_with_tracy_gpu"
    PROFILE_TYPE="(with GPU profiling)"
else
    BAZEL_TARGET="//:run_with_tracy"
    PROFILE_TYPE="(CPU-only profiling)"
fi

# Build the command
printf -v BAZEL_CMD '%q ' bazel run "$BAZEL_TARGET" -- "$@" --profiling-backend=TRACY --enable-timing

echo "Building and running: $BAZEL_CMD $PROFILE_TYPE"

# Run the Tracy-enabled target (environment variables are set by the Bazel target)
exec bazel run "$BAZEL_TARGET" -- "$@" --profiling-backend=TRACY --enable-timing
