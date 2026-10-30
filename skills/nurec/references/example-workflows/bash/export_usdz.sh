#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${NUREC_SRC:?set NUREC_SRC to the released checkout}"
source "$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"

: "${PARSED_CONFIG:?set PARSED_CONFIG to the training config/parsed.yaml}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to an absolute output directory}"
nurec_dev bazelisk run //:run -- export-usdz-artifact \
    --config-name="$PARSED_CONFIG" --checkpoint-name=last.ckpt --output-dir="$OUTPUT_DIR" "$@"
