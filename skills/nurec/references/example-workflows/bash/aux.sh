#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${NUREC_SRC:?set NUREC_SRC to the released checkout}"
source "$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"

: "${DATASET_JSON:?set DATASET_JSON to an absolute NCore V4 manifest path}"
: "${AUX_OUTPUT_DIR:?set AUX_OUTPUT_DIR to a same-path mounted output directory}"
nurec_dev bazelisk run //apps:nurec_tools -- ncore-aux-data \
    --dataset-path="$DATASET_JSON" --output-dir="$AUX_OUTPUT_DIR" --store-meta "$@"
