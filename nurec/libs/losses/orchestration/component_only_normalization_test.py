# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from types import SimpleNamespace

import torch

from nurec.libs.losses.models.camera_loss_masks import CameraLossKind
from nurec.libs.losses.orchestration.component_only_normalization import (
    ComponentOnlyLossNormalizationContext,
    camera_loss_valid_mask,
    configured_camera_loss_kinds_by_name,
    configured_distance_bounds_by_name,
    rescale_camera_mean_losses_to_original_valid_pixels,
)
from nurec.libs.losses.orchestration.config import LossAggregatorReturn, LossReturn
from nurec.utils.batch import CameraFrameLabels
from nurec.utils.types import RayFlags


def _loss(name: str, reduced_value: float = 8.0) -> LossReturn:
    return LossReturn(
        name=name,
        lambda_=1.0,
        value=torch.tensor(reduced_value),
        reduce_fn=lambda value, **kwargs: value,
        _reduced_value=torch.tensor(reduced_value),
    )


def _labels(flags: torch.Tensor | None) -> CameraFrameLabels:
    metric_distance = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 0.5, 10.0, 101.0]).reshape(1, 1, 8, 1)
    return CameraFrameLabels(flags=flags, metric_distance=metric_distance)


_CAMERA_KINDS_BY_NAME = {
    "rgb_l1_mean": CameraLossKind.RGB,
    "ssim_ssim_mean": CameraLossKind.SSIM,
    "background_mse_mean": CameraLossKind.BACKGROUND,
    "semantic_cross_entropy_mean": CameraLossKind.SEMANTIC,
    "velocity_l1_mean": CameraLossKind.VELOCITY,
    "distance_l1_mean": CameraLossKind.DISTANCE,
    "normal_l1_mean": CameraLossKind.NORMAL,
}


class TestComponentOnlyCameraLossKind(unittest.TestCase):
    def test_collects_only_explicitly_declared_camera_mean_metadata(self) -> None:
        configured = [
            SimpleNamespace(name="rgb_l1_mean", component_only_camera_loss_kind=CameraLossKind.RGB),
            SimpleNamespace(name="distance_future_mean", component_only_camera_loss_kind=None),
            SimpleNamespace(name="normal_future_mean"),
            SimpleNamespace(name="background_lidar_mse_mean", component_only_camera_loss_kind=None),
        ]

        self.assertEqual(
            configured_camera_loss_kinds_by_name(configured),
            {"rgb_l1_mean": CameraLossKind.RGB},
        )


class TestComponentOnlyCameraLossValidMask(unittest.TestCase):
    def setUp(self) -> None:
        base = RayFlags.RGB_LABEL.value | RayFlags.VALID_NORMAL.value
        self.flags = torch.tensor(
            [
                base,
                base | RayFlags.INVALID.value,
                RayFlags.VALID_NORMAL.value,
                base | RayFlags.HARMONIZED.value,
                base | RayFlags.SYNTHETIC.value,
                base | RayFlags.SKY_SEMANTIC.value,
                RayFlags.RGB_LABEL.value,
                base,
            ],
            dtype=torch.int32,
        ).reshape(1, 1, 8, 1)
        self.labels = _labels(self.flags)

    def test_matches_each_concrete_camera_loss_validity_predicate(self) -> None:
        cases = {
            CameraLossKind.RGB: [True, False, False, True, True, True, True, True],
            CameraLossKind.SSIM: [True, False, True, True, True, True, True, True],
            CameraLossKind.VELOCITY: [True, False, True, True, True, True, True, True],
            CameraLossKind.BACKGROUND: [True, False, True, False, False, True, True, True],
            CameraLossKind.SEMANTIC: [True, False, False, False, True, False, True, True],
            CameraLossKind.NORMAL: [True, False, True, False, True, False, False, True],
        }
        for kind, expected in cases.items():
            with self.subTest(kind=kind):
                actual = camera_loss_valid_mask(kind, self.labels, self.flags)
                self.assertEqual(actual.flatten().tolist(), expected)

    def test_distance_includes_range_invalid_and_harmonized_predicates(self) -> None:
        actual = camera_loss_valid_mask(
            CameraLossKind.DISTANCE,
            self.labels,
            self.flags,
            distance_bounds=(1.0, 100.0),
        )
        self.assertEqual(actual.flatten().tolist(), [True, False, True, False, True, False, True, False])

    def test_distance_requires_configured_bounds(self) -> None:
        with self.assertRaisesRegex(AssertionError, "Missing configured bounds"):
            camera_loss_valid_mask(CameraLossKind.DISTANCE, self.labels, self.flags)

    def test_distance_requires_metric_labels(self) -> None:
        labels = CameraFrameLabels(flags=self.flags)
        with self.assertRaisesRegex(AssertionError, "Missing metric-distance labels"):
            camera_loss_valid_mask(
                CameraLossKind.DISTANCE,
                labels,
                self.flags,
                distance_bounds=(1.0, 100.0),
            )


class TestConfiguredDistanceBounds(unittest.TestCase):
    def test_collects_only_mean_camera_distance_losses(self) -> None:
        configured = [
            SimpleNamespace(
                name="distance_l1_mean",
                component_only_camera_loss_kind=CameraLossKind.DISTANCE,
                min_distance=1,
                max_distance=80,
            ),
            SimpleNamespace(
                name="distance_l1_sum",
                component_only_camera_loss_kind=None,
                min_distance=2,
                max_distance=90,
            ),
            SimpleNamespace(name="rgb_l1_mean", component_only_camera_loss_kind=CameraLossKind.RGB),
        ]
        self.assertEqual(configured_distance_bounds_by_name(configured), {"distance_l1_mean": (1.0, 80.0)})


class TestRescaleCameraMeanLossesToOriginalValidPixels(unittest.TestCase):
    def setUp(self) -> None:
        base = RayFlags.RGB_LABEL.value | RayFlags.VALID_NORMAL.value
        self.original_flags = torch.full((1, 1, 4, 1), base, dtype=torch.int32)
        self.masked_flags = self.original_flags.clone()
        self.masked_flags[..., 1:, :] |= RayFlags.INVALID.value
        self.labels = CameraFrameLabels(
            flags=self.masked_flags,
            metric_distance=torch.full((1, 1, 4, 1), 5.0),
        )
        self.context = ComponentOnlyLossNormalizationContext.capture(self.original_flags)

    def test_capture_owns_an_immutable_snapshot(self) -> None:
        self.original_flags.fill_(RayFlags.INVALID.value)
        self.assertFalse(bool(self.context.original_flags.eq(RayFlags.INVALID.value).any()))

    def test_rescales_all_supported_mean_aliases_but_not_sums_or_non_camera_losses(self) -> None:
        supported = [
            "rgb_l1_mean",
            "ssim_ssim_mean",
            "background_mse_mean",
            "semantic_cross_entropy_mean",
            "velocity_l1_mean",
            "distance_l1_mean",
            "normal_l1_mean",
        ]
        untouched = [
            "rgb_l1_sum",
            "lpips_lpips_mean",
            "lidar_l1_mean",
            "background_lidar_mse_mean",
            "background_in_track_gaussian_bce_with_logits_mean",
            "node_semantic_gaussians_bce_with_logits_mean",
            "calib_regularization_mean",
            "ppisp_ppisp_loss_mean",
            "distance_future_mean",
            "normal_future_mean",
        ]
        losses = {name: _loss(name) for name in supported + untouched}
        result = LossAggregatorReturn(losses)

        rescale_camera_mean_losses_to_original_valid_pixels(
            result,
            self.labels,
            self.context,
            camera_loss_kinds_by_name=_CAMERA_KINDS_BY_NAME,
            distance_bounds_by_name={"distance_l1_mean": (1.0, 100.0)},
        )

        for name in supported:
            self.assertEqual(float(losses[name].reduced_value), 2.0, name)
        for name in untouched:
            self.assertEqual(float(losses[name].reduced_value), 8.0, name)
        self.assertEqual(float(result.total_value), 2.0 * len(supported) + 8.0 * len(untouched))

    def test_zero_original_valid_population_produces_zero(self) -> None:
        invalid = torch.full((1, 1, 1, 1), RayFlags.INVALID.value, dtype=torch.int32)
        labels = CameraFrameLabels(flags=invalid)
        result = LossAggregatorReturn({"ssim_ssim_mean": _loss("ssim_ssim_mean")})
        rescale_camera_mean_losses_to_original_valid_pixels(
            result,
            labels,
            ComponentOnlyLossNormalizationContext.capture(invalid),
            camera_loss_kinds_by_name={"ssim_ssim_mean": CameraLossKind.SSIM},
            distance_bounds_by_name={},
        )
        self.assertEqual(float(result.total_value), 0.0)

    def test_masked_rgb_gradient_matches_masked_numerator_over_full_valid_denominator(self) -> None:
        prediction = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        masked_mean = prediction[:1].abs().mean()
        masked_loss = LossReturn(
            name="rgb_l1_mean",
            lambda_=1.0,
            value=masked_mean,
            reduce_fn=lambda value, **kwargs: value,
            _reduced_value=masked_mean,
        )
        result = LossAggregatorReturn({masked_loss.name: masked_loss})

        rescale_camera_mean_losses_to_original_valid_pixels(
            result,
            self.labels,
            self.context,
            camera_loss_kinds_by_name={"rgb_l1_mean": CameraLossKind.RGB},
            distance_bounds_by_name={},
        )
        result.total_value.backward()

        reference_prediction = prediction.detach().clone().requires_grad_(True)
        reference = reference_prediction[:1].abs().sum() / reference_prediction.numel()
        reference.backward()
        prediction_grad = prediction.grad
        reference_grad = reference_prediction.grad
        assert prediction_grad is not None
        assert reference_grad is not None
        self.assertTrue(torch.equal(prediction_grad, reference_grad))

    def test_positive_original_but_zero_masked_population_has_zero_gradient(self) -> None:
        prediction = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        empty_flags = self.original_flags | RayFlags.INVALID.value
        labels = CameraFrameLabels(flags=empty_flags)
        connected_zero = prediction.sum() * 0.0
        loss = LossReturn(
            name="rgb_l1_mean",
            lambda_=1.0,
            value=connected_zero,
            reduce_fn=lambda value, **kwargs: value,
            _reduced_value=connected_zero,
        )
        result = LossAggregatorReturn({loss.name: loss})

        rescale_camera_mean_losses_to_original_valid_pixels(
            result,
            labels,
            self.context,
            camera_loss_kinds_by_name={"rgb_l1_mean": CameraLossKind.RGB},
            distance_bounds_by_name={},
        )
        result.total_value.backward()
        prediction_grad = prediction.grad
        assert prediction_grad is not None
        self.assertTrue(torch.equal(prediction_grad, torch.zeros_like(prediction)))

    def test_no_supported_loss_leaves_precomputed_total_untouched(self) -> None:
        result = LossAggregatorReturn({"lpips_lpips_mean": _loss("lpips_lpips_mean")})
        sentinel = torch.tensor(123.0)
        result.total_value = sentinel
        rescale_camera_mean_losses_to_original_valid_pixels(
            result,
            self.labels,
            self.context,
            camera_loss_kinds_by_name={},
            distance_bounds_by_name={},
        )
        self.assertIs(result.total_value, sentinel)

    def test_requires_current_flags(self) -> None:
        with self.assertRaisesRegex(AssertionError, "requires component-masked camera flags"):
            rescale_camera_mean_losses_to_original_valid_pixels(
                LossAggregatorReturn({"lpips_lpips_mean": _loss("lpips_lpips_mean")}),
                _labels(None),
                self.context,
                camera_loss_kinds_by_name={},
                distance_bounds_by_name={},
            )

    def test_rejects_flag_shape_change(self) -> None:
        labels = _labels(torch.zeros((1, 1, 8, 1), dtype=torch.int32))
        with self.assertRaisesRegex(AssertionError, "flag shape changed"):
            rescale_camera_mean_losses_to_original_valid_pixels(
                LossAggregatorReturn({"lpips_lpips_mean": _loss("lpips_lpips_mean")}),
                labels,
                self.context,
                camera_loss_kinds_by_name={},
                distance_bounds_by_name={},
            )

    def test_distance_requires_bounds_mapping(self) -> None:
        with self.assertRaisesRegex(AssertionError, "Missing configured bounds"):
            rescale_camera_mean_losses_to_original_valid_pixels(
                LossAggregatorReturn({"distance_l1_mean": _loss("distance_l1_mean")}),
                self.labels,
                self.context,
                camera_loss_kinds_by_name={"distance_l1_mean": CameraLossKind.DISTANCE},
                distance_bounds_by_name={},
            )


if __name__ == "__main__":
    unittest.main()
