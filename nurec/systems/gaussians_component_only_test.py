# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the component-only wiring in ``GaussiansSystem``.

The system under test is a bare instance (``object.__new__`` +
``torch.nn.Module.__init__``) with a duck-typed ``model``: the masking
methods only read ``model._component_only_mask_spec`` and
``model.build_component_only_bbox_masks`` — everything else in
``GaussiansSystem`` is irrelevant to them and stays unconstructed.
"""

import unittest

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import torch

from nurec.config.component_only import (
    CameraFrameSelection,
    CameraLossNormalization,
    ComponentOnlyConfig,
    ComponentOnlyLayerConfig,
    ComponentState,
    MaskSource,
)
from nurec.config.trainer import TrainerConfig
from nurec.datasets.ncore import NCORETrainDataset
from nurec.libs.losses.models.camera_loss_masks import CameraLossKind
from nurec.libs.losses.models.render_losses import LPIPSLoss
from nurec.libs.losses.orchestration.component_only_normalization import ComponentOnlyLossNormalizationContext
from nurec.libs.losses.orchestration.config import LossConfig, LossItemConfig
from nurec.libs.losses.orchestration.loss_aggregator import LossAggregator
from nurec.systems.active_tile_mapping_prewarmer import ActiveTileMappingPrewarmer
from nurec.systems.base import BaseSystemSO
from nurec.systems.gaussians import GaussiansSystem
from nurec.utils.batch import CameraFrameLabels
from nurec.utils.component_only_mask import DISABLED_TRACKS_SUFFIX, ProjectedTrackRegion, build_mask_spec
from nurec.utils.types import ExtraSignal, GaussiansCompositeReturn, GaussiansRenderReturn, RayFlags


def _ns(**kwargs: Any) -> Any:
    """SimpleNamespace typed as Any: duck-typed stand-ins for batch/model structs."""
    return SimpleNamespace(**kwargs)


class _TestGaussiansSystem(GaussiansSystem):
    @property
    def trainer(self) -> Any:
        return self.__dict__["_trainer"]

    @trainer.setter
    def trainer(self, value: Any) -> None:
        self.__dict__["_trainer"] = value


def _bare_system() -> GaussiansSystem:
    system = object.__new__(_TestGaussiansSystem)
    torch.nn.Module.__init__(system)
    cast(Any, system)._trainer = SimpleNamespace()
    system._active_tile_mapping_prewarmer = ActiveTileMappingPrewarmer()
    system._in_process_instant_nurec_initialized = False
    system.cached_config = _ns(in_process_instant_nurec=None)
    system.__dict__["_device"] = torch.device("cpu")
    return system


def _bbox_spec():
    """Spec supervising one bbox-only layer (no semantics needed)."""
    config = ComponentOnlyConfig(enabled=True, layers={"dynamic_rigids": ComponentOnlyLayerConfig()})
    model_layers = {
        "background": _ns(class_labels=None, tracks=None),
        "dynamic_rigids": _ns(class_labels=None, tracks=object()),
    }
    spec = build_mask_spec(config, model_layers, None)
    assert spec is not None
    return spec


def _system_with_mask(mask: torch.Tensor) -> GaussiansSystem:
    system = _bare_system()
    depth = torch.where(mask, torch.ones(mask.shape), torch.inf)
    system.model = _ns(
        _component_only_mask_spec=_bbox_spec(),
        build_component_only_bbox_masks=lambda rendering_cam: {
            "dynamic_rigids": ProjectedTrackRegion(mask=mask, nearest_depth=depth)
        },
    )
    return system


def _system_with_occlusion_depths(selected_depth: float, disabled_depth: float) -> GaussiansSystem:
    layers = {
        "selected": _ns(class_labels=None, tracks=object()),
        "occluder": _ns(class_labels=None, tracks=object()),
    }
    config = ComponentOnlyConfig(
        enabled=True,
        layers={
            "selected": ComponentOnlyLayerConfig(masks=[MaskSource.bbox]),
            "occluder": ComponentOnlyLayerConfig(state=ComponentState.disabled, masks=[MaskSource.bbox]),
        },
    )
    spec = build_mask_spec(config, layers, None)
    assert spec is not None
    mask = torch.ones((_B, _H, _W), dtype=torch.bool)
    system = _bare_system()
    system.model = _ns(
        _component_only_mask_spec=spec,
        build_component_only_bbox_masks=lambda rendering_cam: {
            "selected": ProjectedTrackRegion(mask, torch.full(mask.shape, selected_depth)),
            "occluder" + DISABLED_TRACKS_SUFFIX: ProjectedTrackRegion(mask, torch.full(mask.shape, disabled_depth)),
        },
    )
    return system


def _batch(*, camera_labels=None, with_rendering: bool = True):
    return _ns(
        data=_ns(camera=camera_labels),
        rendering=_ns(camera=object()) if with_rendering else None,
    )


_B, _H, _W = 1, 2, 4


def _half_mask() -> torch.Tensor:
    mask = torch.zeros((_B, _H, _W), dtype=torch.bool)
    mask[0, 0, :] = True
    return mask


class TestMaybeApplyComponentOnlyMask(unittest.TestCase):
    def test_noop_when_spec_is_none(self) -> None:
        system = _bare_system()
        system.model = _ns(_component_only_mask_spec=None)
        out = _ns(rendered_cam=None)
        system._maybe_apply_component_only_mask(batch=cast(Any, None), out=out)  # must not touch batch/out

    def test_noop_when_no_rendered_camera(self) -> None:
        system = _system_with_mask(_half_mask())
        camera = _ns(labels=_ns(rgb=torch.ones((_B, _H, _W, 3)), semantic=None, flags=None))
        out = _ns(rendered_cam=None)
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out)
        self.assertTrue(torch.equal(camera.labels.rgb, torch.ones((_B, _H, _W, 3))))

    def test_noop_when_no_camera_data(self) -> None:
        system = _system_with_mask(_half_mask())
        pred = torch.ones((_B, _H, _W, 3))
        out = _ns(rendered_cam=_ns(rgb=pred))
        system._maybe_apply_component_only_mask(_batch(camera_labels=None), out)
        self.assertTrue(torch.equal(out.rendered_cam.rgb, torch.ones((_B, _H, _W, 3))))

    def test_asserts_without_rendering_rays(self) -> None:
        system = _system_with_mask(_half_mask())
        camera = _ns(labels=_ns(rgb=torch.ones((_B, _H, _W, 3)), semantic=None, flags=None))
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))
        with self.assertRaises(AssertionError):
            system._maybe_apply_component_only_mask(_batch(camera_labels=camera, with_rendering=False), out)

    def test_noop_when_rgb_missing(self) -> None:
        # Coverage still recorded (training path), but nothing to mask:
        # rendered rgb or gt rgb absent.
        for rendered_rgb, gt_rgb in [(None, torch.ones((_B, _H, _W, 3))), (torch.ones((_B, _H, _W, 3)), None)]:
            with self.subTest(rendered=rendered_rgb is not None):
                system = _system_with_mask(_half_mask())
                camera = _ns(labels=_ns(rgb=gt_rgb, semantic=None, flags=None))
                out = _ns(rendered_cam=_ns(rgb=rendered_rgb))
                system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out, record_coverage=True)
                self.assertEqual(system._component_only_probe_batches_seen, 1)

    def test_coverage_recorded_only_when_requested(self) -> None:
        # The empty-mask probe is a train-split guard: it must NOT run on
        # the validation path (record_coverage defaults to False).
        system = _system_with_mask(_half_mask())
        camera = _ns(labels=_ns(rgb=torch.ones((_B, _H, _W, 3)), semantic=None, flags=None))
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out)  # default: no record
        self.assertFalse(hasattr(system, "_component_only_probe_batches_seen"))
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out, record_coverage=True)
        self.assertEqual(system._component_only_probe_batches_seen, 1)

    def test_coverage_ignores_already_invalid_pixels(self) -> None:
        from nurec.utils.types import RayFlags

        # Supervision is True only on a pixel the camera flags already mark
        # INVALID: the guard must see zero USABLE supervision (that pixel is
        # rejected by every camera loss regardless of the mask).
        mask = torch.zeros((_B, _H, _W), dtype=torch.bool)
        mask[0, 0, 0] = True
        system = _system_with_mask(mask)
        flags = torch.zeros((_B, _H, _W, 1), dtype=torch.int32)
        flags[0, 0, 0, 0] = RayFlags.INVALID.value
        camera = _ns(labels=_ns(rgb=torch.ones((_B, _H, _W, 3)), semantic=None, flags=flags))
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out, record_coverage=True)
        self.assertEqual(system._component_only_probe_batches_seen, 1)
        self.assertEqual(system._component_only_probe_batches_with_kept, 0)  # nothing usable

    def test_masks_image_shaped_rgb(self) -> None:
        system = _system_with_mask(_half_mask())
        pred = torch.ones((_B, _H, _W, 3))
        gt = torch.full((_B, _H, _W, 3), 2.0)
        camera = _ns(labels=_ns(rgb=gt, semantic=None, flags=None))
        out = _ns(rendered_cam=_ns(rgb=pred))
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out)
        # Row 0 kept, row 1 zeroed — for both pred and gt.
        self.assertTrue(torch.equal(out.rendered_cam.rgb[0, 0], torch.ones((_W, 3))))
        self.assertTrue(torch.equal(out.rendered_cam.rgb[0, 1], torch.zeros((_W, 3))))
        self.assertTrue(torch.equal(camera.labels.rgb[0, 0], torch.full((_W, 3), 2.0)))
        self.assertTrue(torch.equal(camera.labels.rgb[0, 1], torch.zeros((_W, 3))))

    def test_masks_flat_rgb(self) -> None:
        # Rendered rgb arrives flat (N, 3) with N = B*H*W during
        # training; gt keeps its image shape.
        system = _system_with_mask(_half_mask())
        pred = torch.ones((_B * _H * _W, 3))
        gt = torch.full((_B, _H, _W, 3), 2.0)
        camera = _ns(labels=_ns(rgb=gt, semantic=None, flags=None))
        out = _ns(rendered_cam=_ns(rgb=pred))
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out)
        self.assertEqual(out.rendered_cam.rgb.shape, (_B * _H * _W, 3))
        self.assertTrue(torch.equal(out.rendered_cam.rgb[:_W], torch.ones((_W, 3))))
        self.assertTrue(torch.equal(out.rendered_cam.rgb[_W:], torch.zeros((_W, 3))))
        self.assertEqual(camera.labels.rgb.shape, (_B, _H, _W, 3))
        self.assertTrue(torch.equal(camera.labels.rgb[0, 1], torch.zeros((_W, 3))))

    def test_disabled_foreground_masks_training_and_validation_identically(self) -> None:
        from nurec.utils.types import RayFlags

        for record_coverage in (True, False):
            with self.subTest(record_coverage=record_coverage):
                system = _system_with_occlusion_depths(selected_depth=10.0, disabled_depth=5.0)
                flags = torch.full((_B, _H, _W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
                camera = _ns(
                    labels=_ns(
                        rgb=torch.full((_B, _H, _W, 3), 2.0),
                        semantic=None,
                        flags=flags,
                        _n_valid_rgb=8,
                        _n_valid=8,
                        _n_valid_bg=8,
                    )
                )
                out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))
                system._maybe_apply_component_only_mask(
                    _batch(camera_labels=camera), out, record_coverage=record_coverage
                )
                self.assertTrue((camera.labels.flags & RayFlags.INVALID.value).ne(0).all())
                self.assertTrue(torch.equal(out.rendered_cam.rgb, torch.zeros((_B, _H, _W, 3))))
                self.assertTrue(torch.equal(camera.labels.rgb, torch.zeros((_B, _H, _W, 3))))
                self.assertIsNone(camera.labels._n_valid_rgb)
                self.assertIsNone(camera.labels._n_valid)
                self.assertIsNone(camera.labels._n_valid_bg)
                self.assertEqual(hasattr(system, "_component_only_probe_batches_seen"), record_coverage)

    def test_disabled_actor_behind_selected_keeps_pixel(self) -> None:
        system = _system_with_occlusion_depths(selected_depth=5.0, disabled_depth=10.0)
        flags = torch.full((_B, _H, _W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        camera = _ns(labels=_ns(rgb=torch.ones((_B, _H, _W, 3)), semantic=None, flags=flags))
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out)
        self.assertFalse(bool((camera.labels.flags & RayFlags.INVALID.value).any()))
        self.assertTrue(bool(out.rendered_cam.rgb.all()))

    def test_disabled_pixel_is_excluded_by_the_real_camera_loss_stack(self) -> None:
        height = width = 3
        n_rays = height * width
        keep = torch.ones((1, height, width), dtype=torch.bool)
        keep[0, 1, 1] = False
        system = _system_with_mask(keep)

        flags = torch.full(
            (1, height, width, 1),
            RayFlags.RGB_LABEL.value | RayFlags.VALID_NORMAL.value,
            dtype=torch.int32,
        )
        gt_rgb = torch.zeros((1, height, width, 3))
        gt_rgb[0, 1, 1] = 1.0
        gt_normal = torch.tensor([0.0, 1.0, 0.0]).expand(1, height, width, 3).clone()
        target_distance = torch.full((1, height, width, 1), 2.0)
        semantic = torch.zeros((1, height, width, 1), dtype=torch.uint8)
        labels = CameraFrameLabels(
            flags=flags,
            rgb=gt_rgb,
            metric_distance=target_distance,
            semantic=semantic,
            normals=gt_normal,
        )

        pred_rgb = torch.zeros((n_rays, 3))
        pred_rgb[4] = -1.0
        pred_normal = gt_normal.reshape(-1, 3).clone()
        pred_normal[4] = torch.tensor([1.0, 0.0, 0.0])
        pred_distance = target_distance.reshape(-1).clone()
        pred_distance[4] = 20.0
        opacity = torch.ones(n_rays)
        opacity[4] = 0.0
        semantic_logits = torch.full((n_rays, 2), -10.0)
        semantic_logits[:, 0] = 10.0
        semantic_logits[4] = torch.tensor([-10.0, 10.0])
        rendered = GaussiansRenderReturn(
            rgb=pred_rgb,
            opacity=opacity,
            distance=pred_distance,
            normal=pred_normal,
            extra_ray_signals=ExtraSignal(semantic_logits=semantic_logits),
        )
        results = GaussiansCompositeReturn(rendered_cam=rendered)
        batch = _ns(data=_ns(camera=_ns(labels=labels)), rendering=_ns(camera=object()))
        system._maybe_apply_component_only_mask(cast(Any, batch), results)

        config = LossConfig.model_validate(
            {
                "rgb": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "ssim": {
                    "fn": "ssim",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                    "window_size": 3,
                    "channel": 3,
                    "mask": {"mode": "GT", "value": 0.0},
                },
                "semantic": {"fn": "cross_entropy", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "normal": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "background": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "distance": {
                    "fn": "l1",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                    "min_distance": 0.1,
                    "max_distance": 100.0,
                    "normalize_by_opacity": False,
                },
            }
        )
        trainer_config = TrainerConfig(
            max_epochs=1,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            precision="32",
        )
        aggregator = LossAggregator(config, trainer_config, force_disable_cuda=True)
        self.assertEqual(
            {loss.name: loss.component_only_camera_loss_kind for loss in aggregator.losses},
            {
                "rgb_l1_mean": CameraLossKind.RGB,
                "ssim_ssim_mean": CameraLossKind.SSIM,
                "semantic_cross_entropy_mean": CameraLossKind.SEMANTIC,
                "normal_l1_mean": CameraLossKind.NORMAL,
                "background_mse_mean": CameraLossKind.BACKGROUND,
                "distance_l1_mean": CameraLossKind.DISTANCE,
            },
        )
        losses = aggregator(
            step=0,
            model=cast(Any, None),
            results=results,
            target=cast(Any, batch),
        )
        self.assertEqual(
            set(losses.loss_returns),
            {
                "rgb_l1_mean",
                "ssim_ssim_mean",
                "semantic_cross_entropy_mean",
                "normal_l1_mean",
                "background_mse_mean",
                "distance_l1_mean",
            },
        )
        for name, loss in losses.loss_returns.items():
            self.assertAlmostEqual(float(loss.reduced_value), 0.0, places=6, msg=name)

        # LPIPS uses the same invalid-bit path, but inject a deterministic
        # lightweight network so this unit test exercises the real LPIPS loss
        # without loading pretrained weights.
        lpips = LPIPSLoss(
            LossItemConfig.model_validate({"fn": "lpips", "lambda_": 1.0, "reduce": {"name": "mean"}}),
            trainer_config,
        )
        network = _ns(forward=lambda image_a, image_b: (image_a - image_b).square().mean(dim=(1, 2, 3), keepdim=True))
        with mock.patch.object(lpips, "_get_network", return_value=network):
            lpips_result = lpips(results=results, target=cast(Any, batch), model=cast(Any, None))
        assert lpips_result is not None
        self.assertAlmostEqual(float(lpips_result.reduced_value), 0.0, places=6)
        self.assertIsNone(lpips.component_only_camera_loss_kind)

        sum_aggregator = LossAggregator(
            LossConfig.model_validate({"rgb": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "sum"}}}),
            trainer_config,
            force_disable_cuda=True,
        )
        self.assertIsNone(sum_aggregator.losses[0].component_only_camera_loss_kind)


class TestMaskInvalidatesRayFlags(unittest.TestCase):
    def test_pixels_outside_mask_get_invalid_flag(self) -> None:
        from nurec.utils.types import RayFlags

        system = _system_with_mask(_half_mask())
        flags = torch.full((_B, _H, _W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        camera = _ns(
            labels=_ns(
                rgb=torch.ones((_B, _H, _W, 3)),
                semantic=None,
                flags=flags,
                _n_valid_rgb=24,  # stale caches must be dropped
                _n_valid=8,
                _n_valid_bg=8,
            )
        )
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out)
        new_flags = camera.labels.flags
        # Row 0 (inside the mask): untouched. Row 1: INVALID set, other bits kept.
        self.assertTrue((new_flags[0, 0, :, 0] == RayFlags.RGB_LABEL.value).all())
        self.assertTrue((new_flags[0, 1, :, 0] & RayFlags.INVALID.value).ne(0).all())
        self.assertTrue((new_flags[0, 1, :, 0] & RayFlags.RGB_LABEL.value).ne(0).all())
        # Lazily-computed valid counts were invalidated.
        self.assertIsNone(camera.labels._n_valid_rgb)
        self.assertIsNone(camera.labels._n_valid)
        self.assertIsNone(camera.labels._n_valid_bg)

    def test_all_valid_pixels_captures_flags_before_masking(self) -> None:
        system = _system_with_mask(_half_mask())
        system.model.component_only = ComponentOnlyConfig.model_validate(
            {
                "enabled": True,
                "camera_supervision": {"loss_normalization": CameraLossNormalization.all_valid_pixels},
            }
        )
        original_flags = torch.full((_B, _H, _W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        camera = _ns(
            labels=_ns(
                rgb=torch.ones((_B, _H, _W, 3)),
                semantic=None,
                flags=original_flags,
                _n_valid_rgb=None,
                _n_valid=None,
                _n_valid_bg=None,
            )
        )
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))

        context = system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out)

        assert context is not None
        self.assertTrue(torch.equal(context.original_flags, original_flags))
        self.assertTrue((camera.labels.flags[0, 1, :, 0] & RayFlags.INVALID.value).ne(0).all())

    def test_supervised_pixels_does_not_capture_original_flags(self) -> None:
        system = _system_with_mask(_half_mask())
        system.model.component_only = ComponentOnlyConfig(enabled=True)
        flags = torch.full((_B, _H, _W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        camera = _ns(
            labels=_ns(
                rgb=torch.ones((_B, _H, _W, 3)),
                semantic=None,
                flags=flags,
                _n_valid_rgb=None,
                _n_valid=None,
                _n_valid_bg=None,
            )
        )
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))
        self.assertIsNone(system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out))

    def test_all_valid_pixels_without_flags_has_no_context(self) -> None:
        system = _system_with_mask(_half_mask())
        system.model.component_only = ComponentOnlyConfig.model_validate(
            {
                "enabled": True,
                "camera_supervision": {"loss_normalization": CameraLossNormalization.all_valid_pixels},
            }
        )
        camera = _ns(labels=_ns(rgb=torch.ones((_B, _H, _W, 3)), semantic=None, flags=None))
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))
        self.assertIsNone(system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out))

    def test_flags_untouched_when_absent(self) -> None:
        system = _system_with_mask(_half_mask())
        camera = _ns(labels=_ns(rgb=torch.ones((_B, _H, _W, 3)), semantic=None, flags=None))
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out)
        self.assertIsNone(camera.labels.flags)

    def test_supervised_only_aborts_zero_mask_before_mutating_batch(self) -> None:
        system = _system_with_mask(torch.zeros((_B, _H, _W), dtype=torch.bool))
        system.model.component_only = ComponentOnlyConfig.model_validate(
            {"enabled": True, "camera_supervision": {"frame_selection": "supervised_only"}}
        )
        flags = torch.full((_B, _H, _W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        camera = _ns(
            labels=_ns(
                rgb=torch.ones((_B, _H, _W, 3)),
                semantic=None,
                flags=flags,
                _n_valid_rgb=None,
                _n_valid=None,
                _n_valid_bg=None,
            )
        )
        out = _ns(rendered_cam=_ns(rgb=torch.ones((_B, _H, _W, 3))))

        with self.assertRaisesRegex(RuntimeError, "aborted before backward"):
            system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out, record_coverage=True)

        self.assertTrue(torch.equal(camera.labels.flags, flags))

    def test_supervised_only_aborts_when_one_frame_in_batch_has_zero_mask(self) -> None:
        mask = torch.ones((2, _H, _W), dtype=torch.bool)
        mask[0] = False
        system = _system_with_mask(mask)
        system.model.component_only = ComponentOnlyConfig.model_validate(
            {"enabled": True, "camera_supervision": {"frame_selection": "supervised_only"}}
        )
        flags = torch.full((2, _H, _W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        camera = _ns(
            labels=_ns(
                rgb=torch.ones((2, _H, _W, 3)),
                semantic=None,
                flags=flags,
                _n_valid_rgb=None,
                _n_valid=None,
                _n_valid_bg=None,
            )
        )
        out = _ns(rendered_cam=_ns(rgb=torch.ones((2, _H, _W, 3))))

        with self.assertRaisesRegex(RuntimeError, "at least one camera frame"):
            system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out, record_coverage=True)

        self.assertTrue(torch.equal(camera.labels.flags, flags))

    def test_all_frame_selection_keeps_existing_empty_mask_behavior(self) -> None:
        system = _system_with_mask(torch.zeros((_B, _H, _W), dtype=torch.bool))
        system.model.component_only = ComponentOnlyConfig(enabled=True)
        camera = _ns(labels=_ns(rgb=None, semantic=None, flags=None))
        out = _ns(rendered_cam=_ns(rgb=None))
        system._maybe_apply_component_only_mask(_batch(camera_labels=camera), out, record_coverage=True)
        self.assertEqual(system._component_only_probe_batches_with_kept, 0)


class TestMaskCoverageProbe(unittest.TestCase):
    def _probe_system(self) -> GaussiansSystem:
        system = _bare_system()
        system.model = _ns(_component_only_mask_spec=_bbox_spec())
        return system

    def test_zero_coverage_warns_not_raises(self) -> None:
        # A sampled, camera-only zero-hit over the probe window is inconclusive:
        # it must WARN, not abort a possibly-valid run.
        system = self._probe_system()
        empty = torch.zeros((_B, _H, _W), dtype=torch.bool)
        with self.assertLogs(level="WARNING") as cm:
            for _ in range(system._COMPONENT_ONLY_EMPTY_PROBE_BATCHES):
                system._record_component_only_mask_coverage(empty)  # must not raise
        self.assertTrue(any("not conclusive" in m for m in cm.output))
        self.assertEqual(system._component_only_probe_batches_with_kept, 0)

    def test_late_hit_does_not_abort(self) -> None:
        # A valid supervised batch after the probe window must not abort — the
        # probe only samples the first N batches and warns at most.
        system = self._probe_system()
        empty = torch.zeros((_B, _H, _W), dtype=torch.bool)
        with self.assertLogs(level="WARNING"):
            for _ in range(system._COMPONENT_ONLY_EMPTY_PROBE_BATCHES):
                system._record_component_only_mask_coverage(empty)
        system._record_component_only_mask_coverage(_half_mask())  # batch N+1: no-op, never raises
        self.assertEqual(system._component_only_probe_batches_seen, system._COMPONENT_ONLY_EMPTY_PROBE_BATCHES)

    def test_passes_with_single_covered_batch(self) -> None:
        system = self._probe_system()
        empty = torch.zeros((_B, _H, _W), dtype=torch.bool)
        system._record_component_only_mask_coverage(_half_mask())
        for _ in range(system._COMPONENT_ONLY_EMPTY_PROBE_BATCHES - 1):
            system._record_component_only_mask_coverage(empty)
        self.assertEqual(system._component_only_probe_batches_with_kept, 1)
        # Past the probe window: recording becomes a no-op.
        system._record_component_only_mask_coverage(_half_mask())
        self.assertEqual(system._component_only_probe_batches_seen, system._COMPONENT_ONLY_EMPTY_PROBE_BATCHES)
        self.assertEqual(system._component_only_probe_batches_with_kept, 1)


class TestOnTrainStartInventory(unittest.TestCase):
    def _system(self, resume: bool) -> tuple[GaussiansSystem, mock.MagicMock]:
        system = _bare_system()
        system.resume = cast(Any, resume)
        model = mock.MagicMock()
        system.model = model
        return system, model

    def test_logs_post_init_inventory(self) -> None:
        system, model = self._system(resume=False)
        with mock.patch.object(BaseSystemSO, "on_train_start"):
            system.on_train_start()
        model.on_train_from_scratch_start.assert_called_once_with(system)
        model._log_component_inventory.assert_called_once_with(when="post-init")

    def test_resume_skips_inventory(self) -> None:
        system, model = self._system(resume=True)
        with mock.patch.object(BaseSystemSO, "on_train_start"):
            system.on_train_start()
        model.on_train_from_scratch_start.assert_not_called()
        model._log_component_inventory.assert_not_called()


class TestOnFitStartSupervisedCameraSetup(unittest.TestCase):
    def test_builds_index_after_restore_before_train_loop(self) -> None:
        system = _bare_system()
        with (
            mock.patch.object(BaseSystemSO, "on_fit_start") as parent_hook,
            mock.patch.object(system, "_setup_component_only_camera_supervision") as setup_supervision,
        ):
            system.on_fit_start()
        parent_hook.assert_called_once_with()
        setup_supervision.assert_called_once_with()

    def test_epoch_hooks_publish_offsets_before_worker_prefetch(self) -> None:
        system = _bare_system()
        plan = mock.MagicMock()
        dataset = mock.MagicMock()
        dataset.__len__.return_value = 25
        system.datamodule = _ns(train_dataset=dataset)
        system._component_only_camera_coverage_plan = plan
        cast(Any, system).trainer = _ns(current_epoch=3)

        with mock.patch.object(BaseSystemSO, "on_train_epoch_start") as parent_start:
            system.on_train_epoch_start()
        parent_start.assert_called_once_with()
        plan.set_batch_index_offset.assert_called_once_with(75)

        plan.reset_mock()
        with mock.patch.object(BaseSystemSO, "on_train_epoch_end") as parent_end:
            system.on_train_epoch_end()
        parent_end.assert_called_once_with()
        plan.set_batch_index_offset.assert_called_once_with(100)

    def test_epoch_hooks_are_noops_without_a_supervised_coverage_plan(self) -> None:
        system = _bare_system()
        cast(Any, system).trainer = _ns(current_epoch=0)
        with (
            mock.patch.object(BaseSystemSO, "on_train_epoch_start") as parent_start,
            mock.patch.object(BaseSystemSO, "on_train_epoch_end") as parent_end,
        ):
            system.on_train_epoch_start()
            system.on_train_epoch_end()
        parent_start.assert_called_once_with()
        parent_end.assert_called_once_with()

    def test_setup_epoch_uses_restored_processed_progress_before_prefetch(self) -> None:
        system = _bare_system()
        cast(Any, system).trainer = _ns(
            current_epoch=3,
            fit_loop=_ns(epoch_progress=_ns(current=_ns(processed=4))),
        )
        self.assertEqual(system._component_only_camera_coverage_setup_epoch(), 4)

    def test_setup_epoch_falls_back_to_current_epoch(self) -> None:
        system = _bare_system()
        cast(Any, system).trainer = _ns(current_epoch=3)
        self.assertEqual(system._component_only_camera_coverage_setup_epoch(), 3)


class TestConstructorPassesComponentOnly(unittest.TestCase):
    def test_component_only_reaches_the_composite(self) -> None:
        import nurec.systems.gaussians as system_module

        config = mock.MagicMock()
        config.mode = "val"
        config.resume = False

        def fake_base_init(system_self, cfg):
            torch.nn.Module.__init__(system_self)
            system_self.config = cfg
            system_self.datamodule = mock.MagicMock()
            system_self.loss = mock.MagicMock()

        with (
            mock.patch.object(BaseSystemSO, "__init__", fake_base_init),
            mock.patch.object(system_module, "adjust_step_for_world_size", side_effect=lambda trainer, v: v),
            mock.patch.object(system_module, "DataSourceSummary") as summary_cls,
            mock.patch.object(system_module, "GaussiansComposite") as composite_cls,
            mock.patch.object(system_module, "TrainingHarmonizerController"),
        ):
            GaussiansSystem(config)
        composite_cls.assert_called_once_with(
            config.model,
            config.trainer,
            summary_cls.from_datasource.return_value,
            init_from_datasource=False,
            component_only=config.component_only,
        )


class TestSupervisedOnlyCameraSetup(unittest.TestCase):
    @staticmethod
    def _component_only(*, scale: bool = False) -> ComponentOnlyConfig:
        return ComponentOnlyConfig.model_validate(
            {
                "enabled": True,
                "camera_supervision": {
                    "frame_selection": CameraFrameSelection.supervised_only,
                    "supervised_only": {"scale_samples_per_epoch_by_frame_selection_fraction": scale},
                },
            }
        )

    def test_noop_branches(self) -> None:
        configurations = [
            (None, "train", False),
            (
                ComponentOnlyConfig.model_validate(
                    {
                        "enabled": False,
                        "camera_supervision": {"frame_selection": CameraFrameSelection.supervised_only},
                    }
                ),
                "train",
                False,
            ),
            (ComponentOnlyConfig(enabled=True), "train", False),
            (self._component_only(), "val", False),
            (self._component_only(), "train", True),
        ]
        for component_only, mode, ready in configurations:
            with self.subTest(component_only=component_only, mode=mode, ready=ready):
                system = _bare_system()
                system.model = _ns(component_only=component_only)
                system.mode = cast(Any, mode)
                system._component_only_supervised_frames_ready = ready
                system._setup_component_only_camera_supervision()

    def test_rejects_non_ncore_training_dataset(self) -> None:
        system = _bare_system()
        system.model = _ns(component_only=self._component_only())
        system.mode = "train"
        system.datamodule = _ns(train_dataset=object())
        with self.assertRaisesRegex(TypeError, "NCORETrainDataset"):
            system._setup_component_only_camera_supervision()

    def _system_and_dataset(self, *, scale: bool) -> tuple[GaussiansSystem, NCORETrainDataset]:
        system = _bare_system()
        system.model = _ns(component_only=self._component_only(scale=scale))
        system.mode = "train"
        dataset = object.__new__(NCORETrainDataset)
        dataset.configured_n_samples_per_epoch = 100
        dataset.effective_n_samples_per_epoch = 25
        dataset.apply_samples_per_epoch_scaling = mock.MagicMock(return_value=25)
        system.datamodule = _ns(train_dataset=dataset)
        system.train_batch_size = 1
        cast(Any, system).trainer = _ns(
            strategy=_ns(lr_scheduler_configs=[]),
            current_epoch=0,
            fit_loop=_ns(epoch_progress=_ns(current=_ns(processed=0))),
        )
        return system, dataset

    @staticmethod
    def _restriction() -> Any:
        return _ns(eligible_total=1, original_total=4, plan=mock.MagicMock())

    def test_builds_installs_and_optionally_scales_index(self) -> None:
        for scale in (False, True):
            with self.subTest(scale=scale):
                system, dataset = self._system_and_dataset(scale=scale)
                if scale:
                    cast(Any, system).trainer.strategy.lr_scheduler_configs = [
                        _ns(interval="step", frequency=1, scheduler=object())
                    ]
                restriction = self._restriction()
                with (
                    mock.patch("nurec.systems.gaussians.resolve_supervised_camera_samplers", return_value="samplers"),
                    mock.patch(
                        "nurec.systems.gaussians.find_local_eligible_camera_pairs", return_value=[("cam", 3)]
                    ) as find,
                    mock.patch(
                        "nurec.systems.gaussians.gather_eligible_camera_pairs", return_value={("cam", 3)}
                    ) as gather,
                    mock.patch(
                        "nurec.systems.gaussians.install_eligible_camera_plan", return_value=restriction
                    ) as install,
                    mock.patch("torch.distributed.is_available", return_value=False),
                ):
                    system._setup_component_only_camera_supervision()

                find.assert_called_once_with("samplers", system.model, device=system.device, rank=0, world_size=1)
                gather.assert_called_once_with([("cam", 3)], local_error=None)
                install.assert_called_once_with("samplers", {("cam", 3)})
                cast(mock.MagicMock, dataset.apply_samples_per_epoch_scaling).assert_called_once_with(
                    eligible_pairs=1,
                    total_pairs=4,
                    scale_samples_per_epoch_by_frame_selection_fraction=scale,
                )
                self.assertTrue(system._component_only_supervised_frames_ready)
                self.assertIs(system._component_only_camera_coverage_plan, restriction.plan)
                restriction.plan.set_batch_index_offset.assert_called_once_with(0)
                self.assertEqual(system._lr_scheduler_epoch_steps, (100, 25) if scale else None)

    def test_epoch_end_resume_seeds_the_next_coverage_segment(self) -> None:
        system, dataset = self._system_and_dataset(scale=False)
        cast(Any, system).trainer.current_epoch = 3
        cast(Any, system).trainer.fit_loop.epoch_progress.current.processed = 4
        restriction = self._restriction()
        with (
            mock.patch("nurec.systems.gaussians.resolve_supervised_camera_samplers", return_value="samplers"),
            mock.patch("nurec.systems.gaussians.find_local_eligible_camera_pairs", return_value=[("cam", 3)]),
            mock.patch("nurec.systems.gaussians.gather_eligible_camera_pairs", return_value={("cam", 3)}),
            mock.patch("nurec.systems.gaussians.install_eligible_camera_plan", return_value=restriction),
            mock.patch("torch.distributed.is_available", return_value=False),
        ):
            system._setup_component_only_camera_supervision()

        self.assertEqual(len(dataset), 25)
        restriction.plan.set_batch_index_offset.assert_called_once_with(100)

    def test_rejects_accelerated_step_reduce_on_plateau_scheduler(self) -> None:
        system, _ = self._system_and_dataset(scale=True)
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        cast(Any, system).trainer.strategy.lr_scheduler_configs = [
            _ns(interval="step", frequency=1, scheduler=scheduler)
        ]
        restriction = self._restriction()
        with (
            mock.patch("nurec.systems.gaussians.resolve_supervised_camera_samplers", return_value="samplers"),
            mock.patch("nurec.systems.gaussians.find_local_eligible_camera_pairs", return_value=[("cam", 3)]),
            mock.patch("nurec.systems.gaussians.gather_eligible_camera_pairs", return_value={("cam", 3)}),
            mock.patch("nurec.systems.gaussians.install_eligible_camera_plan", return_value=restriction),
            mock.patch("torch.distributed.is_available", return_value=False),
            self.assertRaisesRegex(ValueError, "ReduceLROnPlateau"),
        ):
            system._setup_component_only_camera_supervision()

    def test_allows_step_reduce_on_plateau_without_metric_replay(self) -> None:
        system, _ = self._system_and_dataset(scale=True)
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        cast(Any, system).trainer.strategy.lr_scheduler_configs = [
            _ns(interval="step", frequency=4, scheduler=scheduler)
        ]
        restriction = self._restriction()
        with (
            mock.patch("nurec.systems.gaussians.resolve_supervised_camera_samplers", return_value="samplers"),
            mock.patch("nurec.systems.gaussians.find_local_eligible_camera_pairs", return_value=[("cam", 3)]),
            mock.patch("nurec.systems.gaussians.gather_eligible_camera_pairs", return_value={("cam", 3)}),
            mock.patch("nurec.systems.gaussians.install_eligible_camera_plan", return_value=restriction),
            mock.patch("torch.distributed.is_available", return_value=False),
        ):
            system._setup_component_only_camera_supervision()
        self.assertEqual(system._lr_scheduler_epoch_steps, (100, 25))

    def test_uses_distributed_rank_and_world_size(self) -> None:
        system, dataset = self._system_and_dataset(scale=True)
        dataset.configured_n_samples_per_epoch = 2
        dataset.apply_samples_per_epoch_scaling = mock.MagicMock(return_value=1)
        restriction = self._restriction()
        with (
            mock.patch("nurec.systems.gaussians.resolve_supervised_camera_samplers", return_value="samplers"),
            mock.patch("nurec.systems.gaussians.find_local_eligible_camera_pairs", return_value=[]) as find,
            mock.patch("nurec.systems.gaussians.gather_eligible_camera_pairs", return_value={("cam", 3)}),
            mock.patch("nurec.systems.gaussians.install_eligible_camera_plan", return_value=restriction),
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_rank", return_value=2),
            mock.patch("torch.distributed.get_world_size", return_value=4),
        ):
            system._setup_component_only_camera_supervision()
        find.assert_called_once_with("samplers", system.model, device=system.device, rank=2, world_size=4)
        self.assertEqual(system._lr_scheduler_epoch_steps, (1, 1))

    def test_collects_scan_failure_before_raising(self) -> None:
        system, _ = self._system_and_dataset(scale=False)
        scan_error = ValueError("bad frame")
        with (
            mock.patch("nurec.systems.gaussians.resolve_supervised_camera_samplers", return_value="samplers"),
            mock.patch("nurec.systems.gaussians.find_local_eligible_camera_pairs", side_effect=scan_error),
            mock.patch(
                "nurec.systems.gaussians.gather_eligible_camera_pairs",
                side_effect=RuntimeError("synchronized failure"),
            ) as gather,
            mock.patch("torch.distributed.is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "synchronized failure"),
        ):
            system._setup_component_only_camera_supervision()
        gather.assert_called_once_with([], local_error=scan_error)


class TestHarmonizerNovelViewCompatibility(unittest.TestCase):
    def _system(
        self,
        *,
        needs_semantic: bool,
        harmonizer_enabled: bool = True,
        novel_view_poses: list[Any] | None = None,
        component_only_enabled: bool = True,
        frame_selection: CameraFrameSelection = CameraFrameSelection.all_frames,
    ) -> GaussiansSystem:
        system = _bare_system()
        mask_spec = _ns(needs_semantic=needs_semantic) if component_only_enabled else None
        component_only = (
            _ns(camera_supervision=_ns(frame_selection=frame_selection)) if component_only_enabled else None
        )
        system.model = _ns(_component_only_mask_spec=mask_spec, component_only=component_only)
        system.harmonizer_config = _ns(
            training=_ns(
                enabled=harmonizer_enabled,
                novel_view_poses=[object()] if novel_view_poses is None else novel_view_poses,
            )
        )
        return system

    def test_rejects_semantic_component_mask_for_novel_view_training(self) -> None:
        system = self._system(needs_semantic=True)
        with self.assertRaisesRegex(
            ValueError,
            "component_only requires per-pixel semantic labels.*Harmonizer novel-view training",
        ):
            system._validate_training_harmonizer_component_only_compatibility()

    def test_allows_bbox_only_component_mask_for_novel_view_training(self) -> None:
        system = self._system(needs_semantic=False)
        system._validate_training_harmonizer_component_only_compatibility()

    def test_allows_novel_view_training_without_component_only(self) -> None:
        system = self._system(needs_semantic=True, component_only_enabled=False)
        system._validate_training_harmonizer_component_only_compatibility()

    def test_allows_semantic_mask_without_novel_view_training(self) -> None:
        configurations: list[tuple[bool, list[Any]]] = [
            (False, [object()]),
            (True, []),
        ]
        for harmonizer_enabled, novel_view_poses in configurations:
            with self.subTest(
                harmonizer_enabled=harmonizer_enabled,
                novel_view_poses=novel_view_poses,
            ):
                system = self._system(
                    needs_semantic=True,
                    harmonizer_enabled=harmonizer_enabled,
                    novel_view_poses=novel_view_poses,
                )
                system._validate_training_harmonizer_component_only_compatibility()

    def test_rejects_supervised_only_for_bbox_novel_view_training(self) -> None:
        system = self._system(
            needs_semantic=False,
            frame_selection=CameraFrameSelection.supervised_only,
        )
        with self.assertRaisesRegex(ValueError, "source-view frame eligibility"):
            system._validate_training_harmonizer_component_only_compatibility()

    def test_setup_runs_compatibility_check_after_component_mask_initialization(self) -> None:
        system = _bare_system()
        system.model = mock.MagicMock()
        system._validate_training_harmonizer_component_only_compatibility = mock.MagicMock(
            side_effect=ValueError("incompatible")
        )
        with (
            mock.patch.object(BaseSystemSO, "setup"),
            self.assertRaisesRegex(ValueError, "incompatible"),
        ):
            system.setup("fit")
        system.model.initialize_strategy_and_maybe_gaussians.assert_called_once_with()
        system._validate_training_harmonizer_component_only_compatibility.assert_called_once_with()


class TestLossNormalizationCompatibility(unittest.TestCase):
    @staticmethod
    def _system(
        *,
        normalization: CameraLossNormalization = CameraLossNormalization.all_valid_pixels,
        configured_loss_names: tuple[str, ...] = (),
        component_only_enabled: bool = True,
    ) -> GaussiansSystem:
        system = _bare_system()
        component_only = ComponentOnlyConfig.model_validate(
            {
                "enabled": component_only_enabled,
                "camera_supervision": {"loss_normalization": normalization},
            }
        )
        system.mode = "train"
        system.model = _ns(component_only=component_only, _component_only_mask_spec=object())
        system.loss = _ns(
            losses=[_ns(name=name) for name in configured_loss_names],
            in_cuda=_ns(losses=[]),
        )
        return system

    def test_rejects_lpips_with_all_valid_pixels(self) -> None:
        system = self._system(configured_loss_names=("rgb_l1_mean", "lpips_lpips_mean"))
        with self.assertRaisesRegex(ValueError, "incompatible with LPIPS"):
            system._validate_component_only_loss_normalization_compatibility()

    def test_allows_pointwise_losses_with_all_valid_pixels(self) -> None:
        system = self._system(configured_loss_names=("rgb_l1_mean", "ssim_ssim_mean"))
        system._validate_component_only_loss_normalization_compatibility()

    def test_non_all_valid_pixels_modes_are_inert(self) -> None:
        for component_only_enabled, normalization in (
            (False, CameraLossNormalization.all_valid_pixels),
            (True, CameraLossNormalization.supervised_pixels),
        ):
            with self.subTest(component_only_enabled=component_only_enabled, normalization=normalization):
                system = self._system(
                    normalization=normalization,
                    configured_loss_names=("lpips_lpips_mean",),
                    component_only_enabled=component_only_enabled,
                )
                system._validate_component_only_loss_normalization_compatibility()

    def test_missing_component_only_section_is_inert(self) -> None:
        system = _bare_system()
        system.mode = "train"
        system.model = _ns(component_only=None)
        system._validate_component_only_loss_normalization_compatibility()

    def test_inference_and_missing_mask_are_inert(self) -> None:
        for mode, mask_spec in (("val", object()), ("train", None)):
            with self.subTest(mode=mode, mask_spec=mask_spec):
                system = self._system(configured_loss_names=("lpips_lpips_mean",))
                cast(Any, system).mode = mode
                cast(Any, system.model)._component_only_mask_spec = mask_spec
                system._validate_component_only_loss_normalization_compatibility()


class TestTrainingLossesAppliesMask(unittest.TestCase):
    def test_mask_applied_between_alpha_and_loss(self) -> None:
        system = _bare_system()
        recorder = mock.Mock()
        system.apply_alpha = recorder.apply_alpha
        system._maybe_apply_component_only_mask = recorder.mask
        recorder.mask.return_value = None
        system.loss = recorder.loss
        system.log = mock.MagicMock()
        recorder.regularization_loss.return_value = None
        system.model = _ns(calib=_ns(regularization_loss=recorder.regularization_loss))
        batch = _ns(
            data=_ns(camera=None),
            rendering=_ns(camera=None, lidar=None),
        )
        out = _ns()
        with mock.patch.object(GaussiansSystem, "global_step", 0):
            loss_return = system._training_losses(batch, out, batch_local_idx=0)
        self.assertIs(loss_return, recorder.loss.return_value)
        recorder.mask.assert_called_once_with(batch, out, record_coverage=True)
        recorder.regularization_loss.assert_called_once_with()
        call_order = [name for name, _, _ in recorder.mock_calls if name in ("apply_alpha", "mask", "loss")]
        self.assertEqual(call_order, ["apply_alpha", "mask", "loss"])

    def test_all_valid_pixels_context_rescales_after_loss(self) -> None:
        import nurec.systems.gaussians as system_module

        system = _bare_system()
        system.apply_alpha = mock.MagicMock()
        flags = torch.full((1, 1, 1, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32)
        context = ComponentOnlyLossNormalizationContext.capture(flags)
        system._maybe_apply_component_only_mask = mock.MagicMock(return_value=context)
        loss_return = mock.MagicMock()
        system.loss = mock.MagicMock(return_value=loss_return)
        distance_loss = _ns(
            name="distance_l1_mean",
            component_only_camera_loss_kind=CameraLossKind.DISTANCE,
            min_distance=1.0,
            max_distance=80.0,
        )
        system.loss.losses = [distance_loss]
        system.loss.in_cuda.losses = [_ns(name="rgb_l1_mean", component_only_camera_loss_kind=CameraLossKind.RGB)]
        system.model = _ns(calib=_ns(regularization_loss=mock.MagicMock(return_value=None)))
        system.log = mock.MagicMock()
        system.config = _ns(record_quality_metrics=_ns(psnr_interval=2))
        labels = CameraFrameLabels(flags=flags)
        batch = _ns(
            data=_ns(camera=_ns(labels=labels)),
            rendering=_ns(camera=None, lidar=None),
        )
        out = _ns()

        with (
            mock.patch.object(GaussiansSystem, "global_step", 1),
            mock.patch.object(system_module, "rescale_camera_mean_losses_to_original_valid_pixels") as rescale,
        ):
            actual = system._training_losses(batch, out, batch_local_idx=0)

        self.assertIs(actual, loss_return)
        rescale.assert_called_once_with(
            loss_return,
            labels,
            context,
            camera_loss_kinds_by_name={
                "distance_l1_mean": CameraLossKind.DISTANCE,
                "rgb_l1_mean": CameraLossKind.RGB,
            },
            distance_bounds_by_name={"distance_l1_mean": (1.0, 80.0)},
        )


class TestValidationStepAppliesMask(unittest.TestCase):
    def test_mask_applied_to_validation_render(self) -> None:
        system = _bare_system()
        system.config = mock.MagicMock()
        system.config.test.track_rotation.track_ids = []
        results = _ns(rendered_cam=None, rendered_lidar=None)
        system.model = _ns(
            raise_if_mcmc_publication_failed=lambda: None,
        )
        system.forward = mock.MagicMock(return_value=results)
        system.harmonizer_config = _ns(inference=_ns(enabled=False))
        system.apply_alpha = mock.MagicMock()
        system._maybe_apply_component_only_mask = mock.MagicMock()
        system.val_out_types = []
        batch = _ns(
            data=_ns(idx=0, camera=None, lidar=_ns(meta=[object()])),
            rendering=None,
        )
        system.validation_step(batch, 0)
        system._maybe_apply_component_only_mask.assert_called_once_with(batch, results, record_coverage=False)


class TestFusedGatherGenerationLifecycle(unittest.TestCase):
    def _system(self, events: list[str]) -> GaussiansSystem:
        system = _bare_system()
        system.model = _ns(
            drain_fused_gather_generation_plan_refresh=lambda: events.append("drain"),
        )
        return system

    def test_train_end_and_teardown_drain_before_super(self) -> None:
        for hook, args in (("on_train_end", ()), ("teardown", ("fit",))):
            with self.subTest(hook=hook):
                events: list[str] = []
                system = self._system(events)

                def base_hook(_self, *_args):
                    events.append("super")

                with mock.patch.object(BaseSystemSO, hook, new=base_hook):
                    getattr(system, hook)(*args)

                self.assertEqual(events, ["drain", "super"])

    def test_validation_start_drains_before_super_transition(self) -> None:
        events: list[str] = []
        system = self._system(events)

        def stop_after_super(_self):
            events.append("super")
            raise RuntimeError("stop after super")

        with (
            mock.patch.object(BaseSystemSO, "on_validation_start", new=stop_after_super),
            self.assertRaisesRegex(RuntimeError, "stop after super"),
        ):
            system.on_validation_start()

        self.assertEqual(events, ["drain", "super"])

    def test_load_resets_prewarm_before_base_load(self) -> None:
        events: list[str] = []
        system = self._system(events)
        system._active_tile_mapping_prewarmer.reset = lambda: events.append("reset")  # type: ignore[method-assign]
        expected = object()

        def base_load(_self, state_dict, *args, **kwargs):
            self.assertEqual(state_dict, {"weight": 1})
            events.append("super-load")
            return expected

        with mock.patch.object(BaseSystemSO, "load_state_dict", new=base_load):
            result = system.load_state_dict({"weight": 1})

        self.assertIs(result, expected)
        self.assertEqual(events, ["reset", "super-load"])


if __name__ == "__main__":
    unittest.main()
