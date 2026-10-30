# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AV Mask Annotator module for camera frame mask editing and annotation.

This package provides tools for annotating and editing masks in camera frames.
"""

from apps.av.avmask_annotator.mask_annotator import MaskAnnotator, run_mask_annotator


__all__ = ["MaskAnnotator", "run_mask_annotator"]
