#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${NUREC_SRC:?set NUREC_SRC to the released checkout}"
source "$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"

# PAI training on already-converted NCore V4 data. Use the full benchmark for raw data.
: "${DATASET_JSON:?set DATASET_JSON to the converted PAI manifest}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to an absolute output directory}"
CAMERAS='[camera_front_wide_120fov,camera_front_tele_30fov,camera_cross_right_120fov,camera_cross_left_120fov,camera_rear_left_70fov,camera_rear_right_70fov,camera_rear_tele_30fov]'
nurec_dev bazelisk run //:run -- main \
    --config-name="$NUREC_SRC/workflows/av_benchmark/configs/av_benchmark_lidar_7cam.yaml" \
    "dataset.path=$DATASET_JSON" "out_dir=$OUTPUT_DIR" mode=train logger.name=dummy \
    "logger.run_id=${RUN_ID:-reconstruction}" dataset.n_samples_per_epoch=30000 \
    "dataset.camera_ids=$CAMERAS" "dataset.train_camera_ids=$CAMERAS" "dataset.val_camera_ids=$CAMERAS" "$@"
