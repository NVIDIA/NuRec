# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from types import SimpleNamespace

import torch

from nurec.metrics.impl.lidar_common import LIDARCommonMetrics
from nurec.metrics.metric import MetricResult
from nurec.metrics.utils import AggregationMethod
from nurec.utils.batch import DataBatch, LidarFrameLabels, RenderingData
from nurec.utils.types import ExtraSignal, GaussiansRenderReturn, RayFlags


class TestLIDARCommonMetrics(unittest.TestCase):
    def setUp(self):
        self.metric = LIDARCommonMetrics()
        self.dist_gt = torch.tensor([10.0, 10.0, 10.0])
        self.dist_pred = torch.tensor([10.0, 11.0, 13.0])

    def test_depth_mae_is_mean_absolute_error(self):
        metrics = self.metric._compute_depth(self.dist_gt, self.dist_pred)

        self.assertAlmostEqual(float(metrics["depth_mae"]), 4.0 / 3.0, places=6)

    def test_depth_rmse_is_root_mean_squared_error(self):
        metrics = self.metric._compute_depth(self.dist_gt, self.dist_pred)

        self.assertAlmostEqual(float(metrics["depth_rmse"]), math.sqrt(10.0 / 3.0), places=6)
        self.assertGreaterEqual(float(metrics["depth_rmse"]), float(metrics["depth_mae"]))

    def test_depth_median_relative_error_and_recall_match_known_values(self):
        metrics = self.metric._compute_depth(self.dist_gt, self.dist_pred)

        self.assertAlmostEqual(float(metrics["depth_median_l2"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["depth_mean_rel_l2"]), 1.0 / 30.0, places=6)
        self.assertAlmostEqual(float(metrics["depth_medae"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["depth_recall50"]), 1.0 / 3.0, places=6)

    def test_depth_recall50_excludes_exact_half_meter_error(self):
        metrics = self.metric._compute_depth(torch.tensor([10.0]), torch.tensor([10.5]))

        self.assertEqual(float(metrics["depth_recall50"]), 0.0)

    def test_intensity_metrics_match_known_values(self):
        intensity_gt = torch.tensor([0.0, 1.0, 3.0])
        intensity_pred = torch.tensor([0.0, 2.0, 6.0])

        metrics = self.metric._compute_intensity(intensity_gt, intensity_pred)

        self.assertAlmostEqual(float(metrics["intensity_mae"]), 4.0 / 3.0, places=6)
        self.assertAlmostEqual(float(metrics["intensity_rmse"]), math.sqrt(10.0 / 3.0), places=6)

    def test_raydrop_metrics_match_confusion_matrix(self):
        did_return_gt = torch.tensor([False, True, False, True])
        did_return_pred = torch.tensor([False, False, True, True])

        metrics = self.metric._compute_raydrop(did_return_gt, did_return_pred)

        self.assertAlmostEqual(float(metrics["raydrop_accuracy"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["raydrop_precision"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["raydrop_recall"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["raydrop_IoU"]), 1.0 / 3.0, places=6)

    def test_chamfer_distance_is_zero_for_identical_point_clouds(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

        metrics = self.metric._compute_chamfer_distance(points, points)

        self.assertAlmostEqual(float(metrics["chamfer_distance"]), 0.0, places=6)

    def test_chamfer_distance_matches_known_value_for_displaced_point_clouds(self):
        points_gt = torch.tensor([[0.0, 0.0, 0.0]])
        points_pred = torch.tensor([[1.0, 0.0, 0.0]])

        metrics = self.metric._compute_chamfer_distance(points_gt, points_pred)

        self.assertAlmostEqual(float(metrics["chamfer_distance"]), 2.0, places=6)

    def test_compute_applies_return_mask_roi_and_raydrop_threshold(self):
        metric = LIDARCommonMetrics(compute_chamfer_distance=False)
        flags = torch.tensor(
            [[[[0], [RayFlags.DROPPED], [0]]]],
            dtype=torch.int32,
        )
        data_lidar = DataBatch.Lidar(
            meta=[],
            labels=LidarFrameLabels(
                flags=flags,
                distance=torch.tensor([[[[10.0], [20.0], [30.0]]]]),
                intensity=torch.tensor([[[[1.0], [2.0], [3.0]]]]),
            ),
        )
        rendering_lidar = RenderingData(
            rays=torch.zeros((1, 1, 3, 6)),
            sensor_model_parameters=[{}],
            poses_tquat_startend=torch.zeros((1, 2, 7)),
            timestamps_startend_us=torch.zeros((1, 2), dtype=torch.int64),
            timestamps_startend_us_cpu=torch.zeros((1, 2), dtype=torch.int64),
        )
        rendered = GaussiansRenderReturn(
            opacity=torch.ones(3),
            distance=torch.tensor([10.0, 50.0, 33.0]),
            extra_ray_signals=ExtraSignal(
                intensity=torch.tensor([1.0, 5.0, 9.0]),
                raydrop=torch.tensor([0.4, 0.6, 0.4]),
            ),
        )
        config = SimpleNamespace(
            test=SimpleNamespace(
                lidar=SimpleNamespace(
                    ROI=SimpleNamespace(min_m=5.0, max_m=15.0),
                    raydrop_threshold=0.5,
                )
            )
        )

        metrics = metric.compute(data_lidar, rendering_lidar, rendered, None, config)

        self.assertEqual(float(metrics["depth_rmse"]), 0.0)
        self.assertEqual(float(metrics["intensity_mae"]), 0.0)
        self.assertAlmostEqual(float(metrics["raydrop_IoU"]), 0.5, places=6)

    def test_aggregate_averages_per_frame_depth_metrics(self):
        metric = LIDARCommonMetrics(
            compute_raydrop=False,
            compute_chamfer_distance=False,
            compute_intensity=False,
        )
        metric.append(MetricResult(values=metric._compute_depth(torch.tensor([10.0]), torch.tensor([11.0]))))
        metric.append(MetricResult(values=metric._compute_depth(torch.tensor([10.0]), torch.tensor([13.0]))))

        metrics = metric.aggregate()[AggregationMethod.MEAN]

        self.assertAlmostEqual(float(metrics["depth_mae"]), 2.0, places=6)
        self.assertAlmostEqual(float(metrics["depth_rmse"]), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
