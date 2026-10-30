# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase, mock

import numpy as np
import torch

from nurec.datasets.samplers.holdout import CoverageFrameSampler, CoveragePlan, CoverageSensorSampler
from nurec.datasets.samplers.image_crop import ImageCropCameraPixelSampler
from nurec.utils.batch import RectSubsampled
from nurec.utils.component_only_supervision import (
    SupervisedCameraSamplers,
    component_camera_supervision_mask,
    find_local_eligible_camera_pairs,
    gather_eligible_camera_pairs,
    install_eligible_camera_plan,
    resolve_supervised_camera_samplers,
)
from nurec.utils.types import RayFlags


def _plan() -> CoveragePlan:
    return CoveragePlan(
        sensor_ids=("cam",),
        unique_sensor_ids=("uid",),
        per_sensor_frame_idxs=(np.asarray([3, 4], dtype=np.int64),),
        prefix=np.asarray([0, 2], dtype=np.int64),
    )


def _coverage_sensor() -> CoverageSensorSampler:
    sampler = object.__new__(CoverageSensorSampler)
    sampler._coverage_plan = _plan()
    return sampler


def _coverage_frame() -> CoverageFrameSampler:
    sampler = object.__new__(CoverageFrameSampler)
    sampler._coverage_plan = _plan()
    return sampler


def _image_sampler(crop_type: str = "full_image") -> ImageCropCameraPixelSampler:
    sampler = object.__new__(ImageCropCameraPixelSampler)
    sampler.crop_type = crop_type
    return sampler


def _dataset_with_batch_sampler(**updates: Any) -> Any:
    values = dict(
        n_train_sample_camera_rays=16,
        camera_sensor_sampler=_coverage_sensor(),
        camera_frame_sampler=_coverage_frame(),
        camera_pixel_sampler=_image_sampler(),
    )
    values.update(updates)
    return SimpleNamespace(batch_samplers=[SimpleNamespace(**values)])


class TestResolveSupervisedCameraSamplers(TestCase):
    def test_accepts_exact_coverage_full_image_path(self) -> None:
        dataset = _dataset_with_batch_sampler()
        result = resolve_supervised_camera_samplers(dataset)
        self.assertIs(result.dataset, dataset)
        self.assertIs(result.plan, dataset.batch_samplers[0].camera_frame_sampler.coverage_plan)

    def test_rejects_multiple_batch_samplers(self) -> None:
        dataset = _dataset_with_batch_sampler()
        dataset.batch_samplers.append(dataset.batch_samplers[0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_supervised_camera_samplers(dataset)

    def test_rejects_batches_without_camera_rays(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera rays"):
            resolve_supervised_camera_samplers(_dataset_with_batch_sampler(n_train_sample_camera_rays=0))

    def test_rejects_noncoverage_sensor_and_frame_independently(self) -> None:
        for field in ("camera_sensor_sampler", "camera_frame_sampler"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "coverage"):
                resolve_supervised_camera_samplers(_dataset_with_batch_sampler(**{field: object()}))

    def test_rejects_sparse_and_non_full_image_samplers(self) -> None:
        for pixel_sampler in (object(), _image_sampler("random_crop")):
            with self.subTest(pixel_sampler=type(pixel_sampler)), self.assertRaisesRegex(ValueError, "full-image"):
                resolve_supervised_camera_samplers(_dataset_with_batch_sampler(camera_pixel_sampler=pixel_sampler))


class TestComponentCameraSupervisionMask(TestCase):
    def test_returns_raw_mask_when_flags_are_absent(self) -> None:
        expected = torch.tensor([[[True, False]]])
        model = SimpleNamespace(
            _component_only_mask_spec=object(),
            build_component_only_bbox_masks=mock.Mock(return_value={}),
        )
        camera = SimpleNamespace(labels=SimpleNamespace(semantic=None, flags=None))
        with mock.patch(
            "nurec.utils.component_only_supervision.build_camera_loss_mask", return_value=expected
        ) as build:
            actual = component_camera_supervision_mask(model, cast(Any, camera), object())
        self.assertIs(actual, expected)
        build.assert_called_once_with(model._component_only_mask_spec, None, {})

    def test_intersects_preexisting_invalid_pixels(self) -> None:
        raw = torch.ones((1, 1, 2), dtype=torch.bool)
        flags = torch.tensor([[[[0], [RayFlags.INVALID.value]]]], dtype=torch.int32)
        model = SimpleNamespace(
            _component_only_mask_spec=object(),
            build_component_only_bbox_masks=mock.Mock(return_value={}),
        )
        camera = SimpleNamespace(labels=SimpleNamespace(semantic=None, flags=flags))
        with mock.patch("nurec.utils.component_only_supervision.build_camera_loss_mask", return_value=raw):
            actual = component_camera_supervision_mask(model, cast(Any, camera), object())
        self.assertTrue(torch.equal(actual, torch.tensor([[[True, False]]])))

    def test_requires_active_component_mask(self) -> None:
        model = SimpleNamespace(_component_only_mask_spec=None)
        camera = SimpleNamespace(labels=SimpleNamespace(semantic=None, flags=None))
        with self.assertRaisesRegex(AssertionError, "active mask"):
            component_camera_supervision_mask(model, cast(Any, camera), object())


def _probe_bundle(sampled_pixels: Any = None) -> tuple[SupervisedCameraSamplers, Any, Any]:
    if sampled_pixels is None:
        sampled_pixels = RectSubsampled(
            i=0,
            j=0,
            height=1,
            width=1,
            subsample_factor=1.0,
            original_width=1,
            original_height=1,
        )
    pixel_sampler = mock.MagicMock()
    pixel_sampler.sample_camera_pixels.return_value = SimpleNamespace(sampled_pixels=sampled_pixels)
    camera = mock.MagicMock()
    camera.to.return_value = camera
    dataset = SimpleNamespace(
        datasource=SimpleNamespace(
            camera_frame_ranges={"cam": range(10)},
            camera_unique_ids={"cam": SimpleNamespace(id="uid")},
            cameras_frame_valid_pixels_masks={
                "cam": {
                    3: SimpleNamespace(unpacked=lambda: np.ones((1, 1), dtype=np.bool_)),
                    4: SimpleNamespace(unpacked=lambda: np.ones((1, 1), dtype=np.bool_)),
                }
            },
            cameras_all_pixels={"cam": np.zeros((1, 2), dtype=np.int32)},
            sequence_id="seq",
        ),
        n_train_sample_camera_rays=1,
        get_camera_data_batch=mock.Mock(return_value=camera),
    )
    bundle = SupervisedCameraSamplers(
        dataset=cast(Any, dataset),
        sensor_sampler=_coverage_sensor(),
        frame_sampler=_coverage_frame(),
        pixel_sampler=cast(Any, pixel_sampler),
        plan=_plan(),
    )
    model = SimpleNamespace(calib=mock.Mock(return_value=SimpleNamespace(camera=object())))
    return bundle, model, camera


class TestFindLocalEligibleCameraPairs(TestCase):
    def test_rejects_invalid_distributed_shards(self) -> None:
        bundle, model, _ = _probe_bundle()
        for rank, world_size in ((-1, 1), (0, 0), (1, 1)):
            with self.subTest(rank=rank, world_size=world_size), self.assertRaisesRegex(ValueError, "Invalid"):
                find_local_eligible_camera_pairs(
                    bundle, model, device=torch.device("cpu"), rank=rank, world_size=world_size
                )

    def test_keeps_only_nonempty_masks(self) -> None:
        bundle, model, _ = _probe_bundle()
        with mock.patch(
            "nurec.utils.component_only_supervision.component_camera_supervision_mask",
            side_effect=[torch.tensor([False]), torch.tensor([True])],
        ):
            result = find_local_eligible_camera_pairs(bundle, model, device=torch.device("cpu"))
        self.assertEqual(result, [("cam", 4)])
        get_camera_data_batch = cast(mock.Mock, bundle.dataset.get_camera_data_batch)
        self.assertEqual(get_camera_data_batch.call_count, 2)
        self.assertTrue(all(call.kwargs["load_rgb"] is False for call in get_camera_data_batch.call_args_list))
        self.assertTrue(all(call.kwargs["load_aux_geometry"] is False for call in get_camera_data_batch.call_args_list))

    def test_processes_only_this_rank_shard(self) -> None:
        bundle, model, _ = _probe_bundle()
        with mock.patch(
            "nurec.utils.component_only_supervision.component_camera_supervision_mask",
            return_value=torch.tensor([True]),
        ):
            result = find_local_eligible_camera_pairs(bundle, model, device=torch.device("cpu"), rank=1, world_size=2)
        self.assertEqual(result, [("cam", 4)])

    def test_rejects_nonrectangular_pixel_result(self) -> None:
        bundle, model, _ = _probe_bundle(sampled_pixels=np.asarray([0]))
        with self.assertRaisesRegex(TypeError, "RectSubsampled"):
            find_local_eligible_camera_pairs(bundle, model, device=torch.device("cpu"))

    def test_rejects_missing_calibrated_camera_rays(self) -> None:
        bundle, model, _ = _probe_bundle()
        model.calib.return_value = SimpleNamespace(camera=None)
        with self.assertRaisesRegex(RuntimeError, "calibrated camera rays"):
            find_local_eligible_camera_pairs(bundle, model, device=torch.device("cpu"))


class TestGatherAndInstallEligibleCameraPairs(TestCase):
    def test_gather_is_local_without_distributed_runtime(self) -> None:
        with mock.patch("torch.distributed.is_available", return_value=False):
            self.assertEqual(gather_eligible_camera_pairs([("cam", 3)]), {("cam", 3)})
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=False),
        ):
            self.assertEqual(gather_eligible_camera_pairs([("cam", 4)]), {("cam", 4)})

    def test_gather_unions_rank_shards(self) -> None:
        def gather(output, local) -> None:
            output[:] = [local, None, ([("cam", 4)], None)]

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=3),
            mock.patch("torch.distributed.all_gather_object", side_effect=gather),
        ):
            result = gather_eligible_camera_pairs([("cam", 3)])
        self.assertEqual(result, {("cam", 3), ("cam", 4)})

    def test_gather_reraises_local_failure_without_distributed_runtime(self) -> None:
        error = ValueError("bad frame")
        with (
            mock.patch("torch.distributed.is_available", return_value=False),
            self.assertRaisesRegex(ValueError, "bad frame"),
        ):
            gather_eligible_camera_pairs([], local_error=error)

    def test_gather_raises_all_rank_failure_after_collective(self) -> None:
        def gather(output, local) -> None:
            output[:] = [local, ([], "ValueError: bad frame")]

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_gather_object", side_effect=gather),
            self.assertRaisesRegex(RuntimeError, "rank 1: ValueError: bad frame"),
        ):
            gather_eligible_camera_pairs([("cam", 3)])

    def test_install_reinjects_same_restricted_plan(self) -> None:
        bundle, _, _ = _probe_bundle()
        result = install_eligible_camera_plan(bundle, {("cam", 4)})
        self.assertIs(bundle.sensor_sampler.coverage_plan, result.plan)
        self.assertIs(bundle.frame_sampler.coverage_plan, result.plan)
        self.assertEqual(list(result.plan.iter_pairs()), [("cam", 4)])
