# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from nurec.metrics.impl.cpsnr import CPSNRMetric
from nurec.metrics.impl.drift import FeatureDriftMetric
from nurec.metrics.impl.fcs_adaptive import FCSAdaptiveMetric
from nurec.metrics.impl.fid import FIDMetric
from nurec.metrics.impl.higher_order_moments import HigherOrderMomentsMetric
from nurec.metrics.impl.lidar_common import LIDARCommonMetrics
from nurec.metrics.impl.lpips import LPIPSMetric
from nurec.metrics.impl.ntd import NTDMetric
from nurec.metrics.impl.object_level_perceptual import ObjectLevelPerceptualMetric
from nurec.metrics.impl.object_level_semantic import ObjectLevelSemanticMetric
from nurec.metrics.impl.perceptual import PerceptualMetric
from nurec.metrics.impl.psnr import PSNRMetric
from nurec.metrics.impl.road_quality import LmMetric, RefMetric, RlcfMetric, RmtfMetric
from nurec.metrics.impl.ssim import SSIMMetric
from nurec.metrics.impl.temporal_coherence import TemporalCoherenceMetric
from nurec.metrics.metric import BaseMetric
from nurec.metrics.types import MetricType


MetricFactory: dict[MetricType, Callable[..., BaseMetric]] = {
    MetricType.PSNR: PSNRMetric,
    MetricType.CPSNR: CPSNRMetric,
    MetricType.SSIM: SSIMMetric,
    MetricType.LPIPS: LPIPSMetric,
    MetricType.LIDAR_COMMON: LIDARCommonMetrics,
    MetricType.FEATURE_DRIFT: FeatureDriftMetric,
    MetricType.FID: FIDMetric,
    MetricType.PERCEPTUAL: PerceptualMetric,
    MetricType.TEMPORAL_COHERENCE: TemporalCoherenceMetric,
    MetricType.NTD: NTDMetric,
    MetricType.FCS_ADAPTIVE: FCSAdaptiveMetric,
    MetricType.D_SKEW: HigherOrderMomentsMetric,
    MetricType.D_KURT: HigherOrderMomentsMetric,
    MetricType.OBJECT_LEVEL_SEMANTIC: ObjectLevelSemanticMetric,
    MetricType.OBJECT_LEVEL_PERCEPTUAL: ObjectLevelPerceptualMetric,
    MetricType.REF: RefMetric,
    MetricType.RLCF: RlcfMetric,
    MetricType.LM: LmMetric,
    MetricType.RMTF: RmtfMetric,
}
