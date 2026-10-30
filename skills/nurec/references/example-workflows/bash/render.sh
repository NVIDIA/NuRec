#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${NUREC_SRC:?set NUREC_SRC to the released checkout}"
source "$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"

: "${USDZ_PATH:?set USDZ_PATH to an absolute scene path}"
: "${RENDER_OUTPUT_DIR:?set RENDER_OUTPUT_DIR to an absolute output directory}"
nurec_dev bazelisk run //:run -- render \
    --artifact-path="$USDZ_PATH" --output-dir="$RENDER_OUTPUT_DIR" "$@"
