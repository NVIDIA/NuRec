# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto


class MetricType(Enum):
    """Enum for all available metric types."""

    PSNR = auto()
    CPSNR = auto()
    LIDAR_COMMON = auto()
    PERCEPTUAL = auto()
    FID = auto()
    TEMPORAL_COHERENCE = auto()
    FEATURE_DRIFT = auto()
    NTD = auto()
    D_SKEW = auto()
    D_KURT = auto()
    FCS_ADAPTIVE = auto()
    OBJECT_LEVEL_SEMANTIC = auto()
    OBJECT_LEVEL_PERCEPTUAL = auto()
    SSIM = auto()
    LPIPS = auto()
    REF = auto()  # Road Edge Fidelity
    RLCF = auto()  # Road Local Contrast Fidelity
    LM = auto()  # Lane Markings detection
    RMTF = auto()  # Road Multi-scale Texture Fidelity
