#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${NUREC_SRC:?set NUREC_SRC to the released checkout}"
source "$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"

: "${USDZ_PATH:?set USDZ_PATH to an absolute scene path}"
nurec_dev bazelisk run //:run -- serve-grpc \
    --artifact-glob="$USDZ_PATH" --host=127.0.0.1 --port="${NUREC_GRPC_PORT:-8080}" \
    --renderer=default --test-scenes-are-valid "$@"
