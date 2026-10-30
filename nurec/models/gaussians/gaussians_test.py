# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import lietorch as lt
import numpy as np
import torch

from omegaconf import OmegaConf
from torch import nn

import nurec.models.gaussians  # noqa: F401  populates BaseGaussianModel.GAUSSIANS_VARIANTS
import nurec.models.gaussians.gaussians_model as gaussians_model

from nurec.config.trainer import TrainerConfig
from nurec.datasets.track_split import TrackSplitChild, TrackSplitMetadata
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.models.composite import LayerTrackIds
from nurec.models.gaussians.collect import DirectTracksCalibData, LayerDataRigid
from nurec.models.gaussians.gaussians_model import (
    DeformableGaussianModel,
    RigidGaussianModel,
    compute_stable_merging_indices,
)
from nurec.models.gaussians.utils import RGB2SH, SH2RGB, make_ppisp_crf_inverse_rgb_post_processing, ppisp_crf_inverse
from nurec.models.tracks_calib import BaseTracksCalib, DirectTracksCalib, SkipTracksCalib
from nurec.utils.batch import RenderingData


CRF_TEST_CAMERA_INDEX = 1


class _InsertionAsset:
    def __init__(self):
        self.positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        self.scales = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        self.rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        self.densities = torch.tensor([[0.5]], dtype=torch.float32)
        self.features_albedo = torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32)
        self.features_specular = torch.tensor([[0.0]], dtype=torch.float32)
        self.scale_calls = []
        self.transform_arg = None
        self.last_clone: Any = None

    def clone(self) -> "_InsertionAsset":
        """Return an independent test asset while exposing it for call assertions."""
        cloned = _InsertionAsset()
        for name in ("positions", "scales", "rotations", "densities", "features_albedo", "features_specular"):
            setattr(cloned, name, getattr(self, name).clone())
        self.last_clone = cloned
        return cloned

    def transform(self, transform: torch.Tensor) -> None:
        self.transform_arg = transform

    def scale(self, scale_factor, scale_activation, scale_activation_inv) -> None:
        self.scale_calls.append((scale_factor, scale_activation, scale_activation_inv))


class _InsertionModel:
    def __init__(self, cuboid_tracks: CuboidTracks):
        self.device = torch.device("cpu")
        self.cuboid_tracks = cuboid_tracks
        self.tracks_calib = cast(Any, object())
        self.time_embed = None
        self.track_albedos = None
        self.track_scales = None
        self.features_albedo = torch.empty((0, 3), dtype=torch.float32)
        self.fourier_features_dim = 1
        self.positions = torch.empty((0, 3), dtype=torch.float32)
        self.scales = torch.empty((0, 3), dtype=torch.float32)
        self.rotations = torch.empty((0, 4), dtype=torch.float32)
        self.densities = torch.empty((0, 1), dtype=torch.float32)
        self.features_specular = torch.empty((0, 1), dtype=torch.float32)
        self.extra_signal = torch.empty((0, 0), dtype=torch.float32)
        self.gaussian_cuboid_ids = torch.empty((0,), dtype=torch.long)
        self.scale_activation = torch.nn.Identity()
        self.scale_activation_inv = torch.nn.Identity()
        self.scale_factor_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.save_training_parameters_calls = 0

    def _compute_scale_factor(self, cuboid_dims: torch.Tensor, dims_offset: torch.Tensor) -> float:
        self.scale_factor_calls.append((cuboid_dims, dims_offset))
        return 2.0

    def get_extra_signal(self, sensor_name: str) -> torch.Tensor:
        return torch.empty((0, 0), dtype=torch.float32)

    def save_training_parameters(self) -> None:
        self.save_training_parameters_calls += 1

    def validate_external_gaussian_topology_edit(self, operation: str) -> None:
        """Match the default topology-edit capability of a real rigid model."""

    def prepare_asset_insertion(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate staged construction while retaining this lightweight test double."""
        return RigidGaussianModel.prepare_asset_insertion(cast(RigidGaussianModel, self), *args, **kwargs)

    def apply_prepared_asset_insertion(self, *args: Any, **kwargs: Any) -> None:
        """Delegate staged publication while retaining this lightweight test double."""
        RigidGaussianModel.apply_prepared_asset_insertion(cast(RigidGaussianModel, self), *args, **kwargs)


def _make_single_insert_track() -> CuboidTracks:
    identity_poses = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
    return CuboidTracks.Factory.from_numpy(
        tracks_id=["inserted-1"],
        tracks_poses=[identity_poses],
        tracks_timestamps_us=[np.array([0, 1], dtype=np.int64)],
        tracks_label_class=["other"],
        tracks_flags=[TrackFlags.DYNAMIC | TrackFlags.CONTROLLABLE],
        cuboids_dims=[np.array([7.0, 8.0, 9.0], dtype=np.float32)],
        device=torch.device("cpu"),
    )


class TestGaussiansModel(unittest.TestCase):
    def test_distributed_parameter_gather_requires_dedicated_group(self) -> None:
        local = {"positions": torch.ones((2, 3))}

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            self.assertRaisesRegex(RuntimeError, "requires the dedicated process group"),
        ):
            gaussians_model.distributed_all_gather_gaussian_parameters(local)

    def test_distributed_parameter_gather_uses_gaussian_group(self) -> None:
        group = cast(Any, object())
        local = {"positions": torch.ones((2, 3))}
        gathered = [torch.ones((4, 3))]
        queried_groups: list[object | None] = []

        def get_world_size(process_group: object | None = None) -> int:
            queried_groups.append(process_group)
            return 2

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", side_effect=get_world_size),
            patch.object(gaussians_model, "all_gather_tensor_list", return_value=gathered) as all_gather,
        ):
            result = gaussians_model.distributed_all_gather_gaussian_parameters(local, group=group)

        torch.testing.assert_close(result["positions"], gathered[0])
        self.assertEqual(queried_groups, [None, group])
        all_gather.assert_called_once()
        args, kwargs = all_gather.call_args
        self.assertEqual(args[0], 2)
        torch.testing.assert_close(args[1][0], torch.ones((2, 3)))
        self.assertIs(kwargs["group"], group)

    def test_split_boundary_mask_does_not_modify_collector_saved_mask(self):
        """Boundary ownership must leave an autograd collector's saved mask untouched."""

        class SavedMaskCollector(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, tracks_poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                inside = torch.ones(2, dtype=torch.bool, device=tracks_poses.device)
                resolved_timestamps = torch.full((2,), 4_950_000, dtype=torch.int64, device=tracks_poses.device)
                ctx.save_for_backward(inside)
                return tracks_poses, inside, resolved_timestamps

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:
                # Accessing this saved tensor raises a version-counter error if
                # production code modified the collector-owned mask in-place.
                (inside,) = ctx.saved_tensors
                output_poses_grad = cast(torch.Tensor, grad_outputs[0])
                return output_poses_grad * inside.unsqueeze(-1)

        tracks_poses = torch.zeros(2, 7, requires_grad=True)

        def interpolate_tracks_poses(_tracks_interpolation_data):
            return SavedMaskCollector.apply(tracks_poses)

        boundary_us = 5_000_000
        model = cast(
            RigidGaussianModel,
            SimpleNamespace(
                is_static=False,
                collector=SimpleNamespace(interpolate_tracks_poses=interpolate_tracks_poses),
                get_frame_timestamp=RigidGaussianModel.get_frame_timestamp,
                nearest_neighbor_track_for_lidar=False,
                _split_boundary_track_indices=torch.tensor([0, 1], dtype=torch.int64),
                _split_internal_start_timestamps_us=torch.tensor(
                    [torch.iinfo(torch.int64).min, boundary_us], dtype=torch.int64
                ),
                _split_internal_end_timestamps_us=torch.tensor(
                    [boundary_us, torch.iinfo(torch.int64).max], dtype=torch.int64
                ),
            ),
        )
        timestamps = torch.tensor([[boundary_us - 50_000, boundary_us - 50_000]], dtype=torch.int64)
        rendering_data = RenderingData(
            rays=torch.zeros(1, 1, 1, 6),
            sensor_model_parameters=cast(Any, [object()]),
            poses_tquat_startend=torch.zeros(1, 2, 7),
            timestamps_startend_us=timestamps,
            timestamps_startend_us_cpu=timestamps.cpu(),
            rays_timestamps_us=None,
        )

        interpolated_poses, final_mask, resolved_timestamps = RigidGaussianModel._get_interpolated_poses(
            model,
            rendering_data=rendering_data,
            frustum_cull=None,
            tracks_poses=tracks_poses,
            tracks_timestamps=torch.tensor([0, boundary_us, boundary_us, 2 * boundary_us], dtype=torch.int64),
            tracks_packinfo=torch.tensor([[0, 2], [2, 2]], dtype=torch.int32),
            cuboid_dims=torch.ones(2, 3),
            max_positive_timestamp_gap_us=torch.full((2,), 100_000, dtype=torch.int64),
            is_training_batch=True,
        )

        self.assertEqual(final_mask.tolist(), [True, False])
        torch.testing.assert_close(
            resolved_timestamps,
            torch.full((2,), boundary_us - 50_000, dtype=torch.int64),
        )
        interpolated_poses.sum().backward()
        self.assertIsNotNone(tracks_poses.grad)

    def test_deformable_freeze_tracks_clears_stale_mask_without_tracks_to_freeze(self) -> None:
        model = gaussians_model.DeformableGaussianModel.__new__(gaussians_model.DeformableGaussianModel)
        nn.Module.__init__(model)

        for frozen_track_indices, cuboid_tracks in (([], object()), ([0], None)):
            with self.subTest(frozen_track_indices=frozen_track_indices, cuboid_tracks=cuboid_tracks):
                model.cuboid_tracks = cast(Any, cuboid_tracks)
                model._component_only_frozen_track_mask = torch.tensor([False, True])

                model.freeze_tracks(frozen_track_indices)

                self.assertIsNone(model._component_only_frozen_track_mask)

    def test_short_timestamp_gap_fusion_config_branches(self):
        sentinel_tracks = _make_single_insert_track()
        fused_tracks = CuboidTracks.Ops.clone(sentinel_tracks)

        self.assertEqual(RigidGaussianModel._get_config_value(None, "value", 1), 1)
        self.assertEqual(RigidGaussianModel._get_config_value({"value": 2}, "value", 1), 2)
        self.assertEqual(RigidGaussianModel._get_config_value(SimpleNamespace(value=3), "value", 1), 3)

        disabled_config = SimpleNamespace(tracks_calib=SimpleNamespace())
        self.assertIs(
            RigidGaussianModel._maybe_coalesce_short_timestamp_gaps(disabled_config, sentinel_tracks),
            sentinel_tracks,
        )

        with patch.object(
            CuboidTracks.Ops,
            "coalesce_short_timestamp_gaps",
            return_value=fused_tracks,
        ) as coalesce:
            default_config = SimpleNamespace(
                tracks_calib={"short_gap_fusion": {"enabled": True}},
            )
            self.assertIs(
                RigidGaussianModel._maybe_coalesce_short_timestamp_gaps(default_config, sentinel_tracks),
                fused_tracks,
            )
            coalesce.assert_called_once_with(sentinel_tracks, 10000)

            coalesce.reset_mock()
            explicit_config = SimpleNamespace(
                tracks_calib=SimpleNamespace(
                    short_gap_fusion=SimpleNamespace(enabled=True, min_gap_us=123),
                )
            )
            self.assertIs(
                RigidGaussianModel._maybe_coalesce_short_timestamp_gaps(explicit_config, sentinel_tracks),
                fused_tracks,
            )
            coalesce.assert_called_once_with(sentinel_tracks, 123)

    def test_compute_stable_merging_indices(self):
        full_length = np.random.randint(100, 1000)
        world_size = np.random.randint(2, 32)
        device = torch.device("cuda")

        original_data = torch.arange(full_length, device=device)
        gathered_data = []
        for rank in range(world_size):
            gathered_data.append(original_data[rank::world_size])
        gathered_data = torch.cat(gathered_data)

        indices = compute_stable_merging_indices(full_length, world_size, device)
        assert torch.allclose(gathered_data[indices], original_data)

    def test_insert_asset_scales_to_cuboid_dims_using_given_offset(self):
        cuboid_tracks = _make_single_insert_track()
        model = _InsertionModel(cuboid_tracks)
        asset = _InsertionAsset()
        dims_offset = torch.tensor([1.0, 2.0, 3.0])

        # Harvested assets are unit-normalized (positions divided by max(lwh)); insert_asset
        # must rescale them to the track cuboid dimensions, computing the factor with the
        # dims_offset it is given (the caller passes a zero offset for insertion).
        RigidGaussianModel.insert_asset(
            model,
            track_id="inserted-1",
            asset=asset,
            cuboid_tracks=cuboid_tracks,
            dims_offset=dims_offset,
        )

        # Scale factor computed once, against the track's cuboid dims and the given offset.
        self.assertEqual(len(model.scale_factor_calls), 1)
        cuboid_dims_arg, dims_offset_arg = model.scale_factor_calls[0]
        torch.testing.assert_close(cuboid_dims_arg, cuboid_tracks.cuboids_dims[0])
        torch.testing.assert_close(dims_offset_arg, dims_offset)

        # Preparation transforms/scales a private clone, leaving the reusable input untouched.
        self.assertEqual(asset.scale_calls, [])
        self.assertIsNone(asset.transform_arg)
        self.assertIsNotNone(asset.last_clone)
        self.assertEqual(len(asset.last_clone.scale_calls), 1)
        self.assertEqual(asset.last_clone.scale_calls[0][0], 2.0)
        self.assertEqual(model.save_training_parameters_calls, 1)

    def test_insert_asset_construction_failure_does_not_publish_partial_parameters(self):
        """A late concatenation failure leaves every live Gaussian object untouched."""
        cuboid_tracks = _make_single_insert_track()
        model = _InsertionModel(cuboid_tracks)
        asset = _InsertionAsset()
        asset.features_specular = torch.zeros((1, 2), dtype=torch.float32)
        original_state = {
            name: getattr(model, name)
            for name in (
                "positions",
                "scales",
                "rotations",
                "densities",
                "features_albedo",
                "features_specular",
                "extra_signal",
                "gaussian_cuboid_ids",
            )
        }

        with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match"):
            RigidGaussianModel.insert_asset(
                model,
                track_id="inserted-1",
                asset=asset,
                cuboid_tracks=cuboid_tracks,
                dims_offset=torch.zeros(3),
            )

        for name, original_value in original_state.items():
            self.assertIs(getattr(model, name), original_value, name)
        self.assertEqual(model.save_training_parameters_calls, 0)
        self.assertIsNone(asset.transform_arg)
        self.assertEqual(asset.scale_calls, [])

    def test_static_interpolation_requires_two_pose_samples_per_track(self) -> None:
        model = cast(
            RigidGaussianModel,
            SimpleNamespace(is_static=True, get_frame_timestamp=lambda _rendering_data: 123),
        )

        def interpolate(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return RigidGaussianModel._get_interpolated_poses(
                model,
                rendering_data=cast(RenderingData, object()),
                frustum_cull=None,
                tracks_poses=poses,
                tracks_timestamps=torch.arange(poses.shape[0], dtype=torch.int64),
                tracks_packinfo=torch.tensor([[0, 2], [2, 2]], dtype=torch.int32),
                cuboid_dims=torch.ones(2, 3),
                max_positive_timestamp_gap_us=torch.ones(2, dtype=torch.int64),
                is_training_batch=False,
            )

        # Each track stores its two endpoint poses contiguously. Static motion
        # uses the first endpoint and keeps both tracks without interpolation.
        poses = torch.tensor(
            [[1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1], [3, 0, 0, 0, 0, 0, 1], [4, 0, 0, 0, 0, 0, 1]],
            dtype=torch.float32,
        )
        actual_poses, keep_mask, timestamps = interpolate(poses)
        torch.testing.assert_close(actual_poses, poses[[0, 2]])
        torch.testing.assert_close(keep_mask, torch.tensor([True, True]))
        torch.testing.assert_close(timestamps, torch.tensor([123, 123], dtype=torch.int64))

        for pose_count in (0, 2, 3, 5):
            with self.subTest(pose_count=pose_count):
                with self.assertRaisesRegex(ValueError, "Static interpolation requires"):
                    interpolate(torch.zeros(pose_count, 7))

    def test_get_interpolated_poses_wires_cadence_and_exposure_for_all_batch_types(self):
        class FakeLidarModelParameters:
            pass

        def make_rendering_data(sensor_model):
            timestamps = torch.tensor([[1000, 1600]], dtype=torch.int64)
            return RenderingData(
                rays=torch.zeros(1, 1, 1, 6),
                sensor_model_parameters=[sensor_model],
                poses_tquat_startend=torch.zeros(1, 2, 7),
                timestamps_startend_us=timestamps,
                timestamps_startend_us_cpu=timestamps.cpu(),
                rays_timestamps_us=None,
            )

        def captured_metadata(is_training_batch: bool, sensor_model) -> tuple[torch.Tensor, int]:
            captured_tracks_interpolation_data = []

            def interpolate_tracks_poses(tracks_interpolation_data):
                captured_tracks_interpolation_data.append(tracks_interpolation_data)
                n_tracks = tracks_interpolation_data.tracks_packinfo.shape[0]
                device = tracks_interpolation_data.tracks_poses.device
                return (
                    torch.zeros(n_tracks, 7, device=device),
                    torch.ones(n_tracks, dtype=torch.bool, device=device),
                    torch.full((n_tracks,), 1300, dtype=torch.int64, device=device),
                )

            model = cast(
                RigidGaussianModel,
                SimpleNamespace(
                    is_static=False,
                    collector=SimpleNamespace(interpolate_tracks_poses=interpolate_tracks_poses),
                    get_frame_timestamp=RigidGaussianModel.get_frame_timestamp,
                    nearest_neighbor_track_for_lidar=True,
                    _split_boundary_track_indices=torch.empty(0, dtype=torch.int64),
                ),
            )
            RigidGaussianModel._get_interpolated_poses(
                model,
                rendering_data=make_rendering_data(sensor_model),
                frustum_cull=None,
                tracks_poses=torch.zeros(4, 7),
                tracks_timestamps=torch.tensor([1000, 1100, 1150, 2150], dtype=torch.int64),
                tracks_packinfo=torch.tensor([[0, 2], [2, 2]], dtype=torch.int32),
                cuboid_dims=torch.ones(2, 3),
                max_positive_timestamp_gap_us=torch.tensor([100, 1000], dtype=torch.int64),
                is_training_batch=is_training_batch,
            )
            interpolation_data = captured_tracks_interpolation_data[-1]
            return interpolation_data.max_positive_timestamp_gap_us, interpolation_data.frame_exposure_us

        with patch.object(
            gaussians_model,
            "RowOffsetStructuredSpinningLidarModelParameters",
            FakeLidarModelParameters,
        ):
            for is_training_batch, sensor_model in (
                (False, object()),
                (True, object()),
                (False, FakeLidarModelParameters()),
            ):
                cadence, exposure = captured_metadata(is_training_batch, sensor_model)
                torch.testing.assert_close(cadence, torch.tensor([100, 1000], dtype=torch.int64))
                self.assertEqual(exposure, 600)

    def test_split_boundary_mask_prevents_tolerance_overlap(self):
        """A synthetic split boundary has one owner even while both children clamp to it."""
        boundary_us = 5_000_000
        cadence_us = 100_000
        exposure_us = 33_333
        endpoint_tolerance_us = cadence_us + exposure_us

        child_timestamps = torch.arange(0, boundary_us + cadence_us, cadence_us, dtype=torch.int64)
        tracks_timestamps = torch.cat((child_timestamps, child_timestamps + boundary_us))
        n_child_poses = child_timestamps.shape[0]
        tracks_packinfo = torch.tensor(
            [[0, n_child_poses], [n_child_poses, n_child_poses]],
            dtype=torch.int32,
        )
        tracks_poses = torch.zeros(2 * n_child_poses, 7)
        cuboid_dims = torch.ones(2, 3)
        max_positive_timestamp_gap_us = torch.full((2,), cadence_us, dtype=torch.int64)

        raw_masks: list[torch.Tensor] = []

        def interpolate_tracks_poses(tracks_interpolation_data):
            query_timestamp_us = tracks_interpolation_data.timestamps_data.timestamp
            start_indices = tracks_interpolation_data.tracks_packinfo[:, 0].long()
            end_indices = start_indices + tracks_interpolation_data.tracks_packinfo[:, 1].long() - 1
            starts_us = tracks_interpolation_data.tracks_timestamps[start_indices]
            ends_us = tracks_interpolation_data.tracks_timestamps[end_indices]
            tolerances_us = (
                tracks_interpolation_data.max_positive_timestamp_gap_us + tracks_interpolation_data.frame_exposure_us
            )
            raw_mask = (query_timestamp_us >= starts_us - tolerances_us) & (
                query_timestamp_us <= ends_us + tolerances_us
            )
            # Keep the collector-owned tensor itself: production must not mutate
            # it because the real collector saves it for backward.
            raw_masks.append(raw_mask)
            return torch.zeros(2, 7), raw_mask, torch.full((2,), query_timestamp_us, dtype=torch.int64)

        split_metadata = TrackSplitMetadata(
            children=[
                TrackSplitChild("parent", "child0", 0, boundary_us, 0),
                TrackSplitChild("parent", "child1", boundary_us, 2 * boundary_us, 1),
            ]
        )
        policy_tracks = CuboidTracks.Factory.from_numpy(
            tracks_id=["child0", "child1"],
            tracks_poses=[
                np.repeat(np.eye(4, dtype=np.float32)[None], n_child_poses, axis=0),
                np.repeat(np.eye(4, dtype=np.float32)[None], n_child_poses, axis=0),
            ],
            tracks_timestamps_us=[
                child_timestamps.numpy(),
                (child_timestamps + boundary_us).numpy(),
            ],
            tracks_label_class=["person", "person"],
            tracks_flags=[TrackFlags.DYNAMIC, TrackFlags.DYNAMIC],
            cuboids_dims=[np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
            device=torch.device("cpu"),
        )
        split_boundary_policy = RigidGaussianModel._build_split_boundary_policy(policy_tracks, split_metadata)

        model = cast(
            RigidGaussianModel,
            SimpleNamespace(
                is_static=False,
                collector=SimpleNamespace(interpolate_tracks_poses=interpolate_tracks_poses),
                get_frame_timestamp=RigidGaussianModel.get_frame_timestamp,
                nearest_neighbor_track_for_lidar=False,
                _split_boundary_track_indices=split_boundary_policy[0],
                _split_internal_start_timestamps_us=split_boundary_policy[1],
                _split_internal_end_timestamps_us=split_boundary_policy[2],
            ),
        )

        def interpolate_at(
            timestamp_us: int, is_training_batch: bool
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            frame_start_us = timestamp_us - exposure_us // 2
            timestamps = torch.tensor([[frame_start_us, frame_start_us + exposure_us]], dtype=torch.int64)
            rendering_data = RenderingData(
                rays=torch.zeros(1, 1, 1, 6),
                sensor_model_parameters=cast(Any, [object()]),
                poses_tquat_startend=torch.zeros(1, 2, 7),
                timestamps_startend_us=timestamps,
                timestamps_startend_us_cpu=timestamps.cpu(),
                rays_timestamps_us=None,
            )
            _, final_mask, resolved_timestamps = RigidGaussianModel._get_interpolated_poses(
                model,
                rendering_data=rendering_data,
                frustum_cull=None,
                tracks_poses=tracks_poses,
                tracks_timestamps=tracks_timestamps,
                tracks_packinfo=tracks_packinfo,
                cuboid_dims=cuboid_dims,
                max_positive_timestamp_gap_us=max_positive_timestamp_gap_us,
                is_training_batch=is_training_batch,
            )
            return raw_masks[-1], final_mask, resolved_timestamps

        # These are the eight 30 Hz frame centers inside the 2 * (G + E) overlap
        # window described in the Slack report.
        overlapping_frame_centers_us = (
            4_883_333,
            4_916_667,
            4_950_000,
            4_983_333,
            5_016_667,
            5_050_000,
            5_083_333,
            5_116_667,
        )
        for is_training_batch in (False, True):
            for timestamp_us in overlapping_frame_centers_us:
                with self.subTest(is_training_batch=is_training_batch, timestamp_us=timestamp_us):
                    raw_mask, final_mask, resolved_timestamps = interpolate_at(timestamp_us, is_training_batch)
                    self.assertEqual(raw_mask.tolist(), [True, True])
                    self.assertEqual(final_mask.tolist(), [timestamp_us < boundary_us, timestamp_us >= boundary_us])
                    torch.testing.assert_close(resolved_timestamps, torch.full((2,), timestamp_us, dtype=torch.int64))

            # The shared timestamp belongs to the incoming/right child.
            raw_mask, final_mask, resolved_timestamps = interpolate_at(boundary_us, is_training_batch)
            self.assertEqual(raw_mask.tolist(), [True, True])
            self.assertEqual(final_mask.tolist(), [False, True])
            torch.testing.assert_close(resolved_timestamps, torch.full((2,), boundary_us, dtype=torch.int64))

            # The original parent's outer endpoint tolerance remains unchanged.
            for timestamp_us, expected in (
                (-endpoint_tolerance_us, [True, False]),
                (-endpoint_tolerance_us - 1, [False, False]),
                (2 * boundary_us + endpoint_tolerance_us, [False, True]),
                (2 * boundary_us + endpoint_tolerance_us + 1, [False, False]),
            ):
                with self.subTest(is_training_batch=is_training_batch, timestamp_us=timestamp_us):
                    raw_mask, final_mask, resolved_timestamps = interpolate_at(timestamp_us, is_training_batch)
                    self.assertEqual(final_mask.tolist(), expected)
                    self.assertEqual(final_mask.tolist(), raw_mask.tolist())
                    torch.testing.assert_close(resolved_timestamps, torch.full((2,), timestamp_us, dtype=torch.int64))

    def test_tracks_edit_maps_cadence_with_packinfo(self):
        def make_tracks(track_ids, timestamps):
            return CuboidTracks.Factory.from_numpy(
                tracks_id=track_ids,
                tracks_poses=[
                    np.repeat(np.eye(4, dtype=np.float32)[None], len(values), axis=0) for values in timestamps
                ],
                tracks_timestamps_us=[np.asarray(values, dtype=np.int64) for values in timestamps],
                tracks_label_class=["unknown"] * len(track_ids),
                tracks_flags=[TrackFlags.NONE] * len(track_ids),
                cuboids_dims=[np.ones(3, dtype=np.float32)] * len(track_ids),
                device=torch.device("cpu"),
            )

        original = make_tracks(["a", "b", "c"], [[0, 1], [0, 1], [0, 1]])
        edited = make_tracks(["c", "a"], [[10, 30], [100, 300, 700]])
        edited.cuboids_dims.copy_(torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.float32))
        model = cast(RigidGaussianModel, SimpleNamespace(cuboid_tracks=original))
        cast(Any, model).get_motion_source = lambda: model
        tracks_edit = LayerTrackIds.Edit(cuboid_tracks=edited, mapping=torch.tensor([2, 0], dtype=torch.long))
        context = SimpleNamespace(tracks_edit=tracks_edit)
        gathered_parameters = {
            "gaussian_cuboid_ids": torch.tensor([0, 1, 2], dtype=torch.int32),
            "positions": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),
        }

        tracks_poses, tracks_timestamps, packinfo, dims, cadence = RigidGaussianModel._get_edited_motion_data(
            model, tracks_edit
        )
        # Preparing shared metadata must not filter any layer's Gaussian rows.
        self.assertEqual(gathered_parameters["gaussian_cuboid_ids"].tolist(), [0, 1, 2])
        RigidGaussianModel._filter_edited_gaussians(model, context, gathered_parameters)

        torch.testing.assert_close(tracks_poses, edited.tracks_poses.vec())
        torch.testing.assert_close(tracks_timestamps, edited.tracks_timestamps_us)
        torch.testing.assert_close(packinfo, torch.tensor([[2, 3], [0, 0], [0, 2]], dtype=torch.int32))
        torch.testing.assert_close(
            dims,
            torch.tensor([[6, 7, 8], [0, 0, 0], [3, 4, 5]], dtype=torch.float32),
        )
        torch.testing.assert_close(cadence, torch.tensor([400, 0, 20], dtype=torch.int64))
        torch.testing.assert_close(gathered_parameters["gaussian_cuboid_ids"], torch.tensor([0, 2], dtype=torch.int32))
        torch.testing.assert_close(context.kept_rows, torch.tensor([0, 2], dtype=torch.long))
        torch.testing.assert_close(
            gathered_parameters["positions"], torch.tensor([[1, 2, 3], [7, 8, 9]], dtype=torch.float32)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CuboidTracks subsetting requires CUDA")
    def test_tracks_edit_lookup_preserves_order_and_first_matching_id(self) -> None:
        device = torch.device("cuda")

        def make_tracks(track_ids: list[str]) -> CuboidTracks:
            return CuboidTracks.Factory.from_numpy(
                tracks_id=track_ids,
                tracks_poses=[np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0) for _ in track_ids],
                tracks_timestamps_us=[np.array([0, 1], dtype=np.int64) for _ in track_ids],
                tracks_label_class=["unknown"] * len(track_ids),
                tracks_flags=[TrackFlags.NONE] * len(track_ids),
                cuboids_dims=[np.ones(3, dtype=np.float32)] * len(track_ids),
                device=device,
            )

        requested = make_tracks(["c", "unknown", "a"])
        for owner_ids, selected_ids, mapping in (
            (["a", "b", "c"], ["c", "a"], [2, 0]),
            # Preserve the previous list.index first-match behavior if input IDs repeat.
            (["a", "b", "a"], ["a"], [0]),
        ):
            with self.subTest(owner_ids=owner_ids):
                model = cast(RigidGaussianModel, SimpleNamespace(cuboid_tracks=make_tracks(owner_ids)))
                cast(Any, model).get_motion_source = lambda: model
                tracks_edit = RigidGaussianModel.create_tracks_edit_from_cuboid_tracks(model, requested)
                self.assertEqual(tracks_edit.cuboid_tracks.tracks_id, selected_ids)
                torch.testing.assert_close(tracks_edit.mapping, torch.tensor(mapping, dtype=torch.long, device=device))

    def test_prepare_motion_uses_motion_source_direct_calib(self) -> None:
        owner, dependent = (RigidGaussianModel.__new__(RigidGaussianModel) for _ in range(2))
        for node, source_layer in ((owner, None), (dependent, "body")):
            nn.Module.__init__(node)
            node.config = cast(Any, SimpleNamespace(motion_source_layer=source_layer))
            node._motion_source_ref = None
            node.is_static = False
            node.nearest_neighbor_track_for_lidar = False
            node.cuboid_tracks = _make_single_insert_track()
        dependent.bind_motion_source(owner)
        dependent.cuboid_tracks.tracks_poses.data[:, :3].add_(100)
        dependent.cuboid_tracks.tracks_timestamps_us.add_(100)
        dependent.cuboid_tracks.cuboids_dims.add_(10)
        trainer_config = TrainerConfig(
            max_epochs=1,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            precision="32",
        )
        dependent.tracks_calib = SkipTracksCalib(
            OmegaConf.create({"visualize": False}), trainer_config, dependent.cuboid_tracks
        )
        base_poses = owner.cuboid_tracks.tracks_poses.vec()
        for delta_frame, left_multiply in (("local", False), ("world", True)):
            with self.subTest(delta_frame=delta_frame):
                calib = DirectTracksCalib(
                    OmegaConf.create(
                        {
                            "visualize": False,
                            "start_global_step": 0,
                            "delta_frame": delta_frame,
                            "fix_first_pose": True,
                            "fix_last_pose": False,
                        }
                    ),
                    trainer_config,
                    owner.cuboid_tracks,
                )
                nn.init.constant_(calib.tracks_delta_t, 0.25)
                owner.tracks_calib = calib
                collector_outputs: list[torch.Tensor] = []

                def calibrate_poses(poses: torch.Tensor, data: DirectTracksCalibData) -> torch.Tensor:
                    # Synthetic differentiable boundary: verify owner parameter wiring,
                    # not the native collector's SE(3) calibration math.
                    output = poses + torch.cat((data.tracks_delta_t, data.tracks_delta_q), dim=-1)
                    collector_outputs.append(output)
                    return output

                calibrate = Mock(side_effect=calibrate_poses)
                owner.collector = cast(Any, SimpleNamespace(calibrate_tracks_poses=calibrate))
                resolved_timestamps = torch.tensor([123], dtype=torch.int64)
                interpolate = Mock(
                    side_effect=lambda *args: (args[2], torch.ones(1, dtype=torch.bool), resolved_timestamps)
                )
                cast(Any, owner)._get_interpolated_poses = interpolate

                result = dependent.prepare_motion(cast(RenderingData, object()), None, True)

                calibrate.assert_called_once()
                poses, data = calibrate.call_args.args
                torch.testing.assert_close(poses, base_poses)
                self.assertIsInstance(data, DirectTracksCalibData)
                self.assertIs(data.gradient_mask, calib.gradient_mask)
                self.assertIs(data.tracks_delta_q, calib.tracks_delta_q)
                self.assertIs(data.tracks_delta_t, calib.tracks_delta_t)
                self.assertEqual(data.left_multiply, left_multiply)
                self.assertIs(result.poses, collector_outputs[0])
                self.assertIs(result.timestamps_us, resolved_timestamps)
                assert result.track_time_data is not None
                self.assertIs(
                    result.track_time_data.tracks_timestamps_us,
                    owner.cuboid_tracks.tracks_timestamps_us,
                )
                self.assertIs(result.track_time_data.tracks_packinfo, owner.cuboid_tracks.tracks_packinfo)
                self.assertIs(
                    result.track_time_data.max_positive_timestamp_gap_us,
                    owner.cuboid_tracks.max_positive_timestamp_gap_us,
                )
                interpolate.assert_called_once()
                for actual, expected in zip(
                    interpolate.call_args.args[3:7],
                    (
                        owner.cuboid_tracks.tracks_timestamps_us,
                        owner.cuboid_tracks.tracks_packinfo,
                        owner.cuboid_tracks.cuboids_dims,
                        owner.cuboid_tracks.max_positive_timestamp_gap_us,
                    ),
                ):
                    self.assertIs(actual, expected)
                result.poses.sum().backward()
                for parameter in (calib.tracks_delta_t, calib.tracks_delta_q):
                    self.assertIsNotNone(parameter.grad)
                    torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))

    def test_prepare_motion_uses_motion_source_unicycle_calib(self) -> None:
        class ResolvedUnicycleCalib:
            def __call__(self, cuboid_tracks: CuboidTracks) -> CuboidTracks:
                return cuboid_tracks

            def collect_data(self) -> object:
                return unicycle_data

            def _get_pose_from_collector(self, compute: Callable[[], torch.Tensor]) -> SimpleNamespace:
                return SimpleNamespace(vec=compute)

        base_poses = torch.zeros((2, 7), dtype=torch.float32)
        calibrated_poses = torch.ones_like(base_poses)
        cuboid_tracks = SimpleNamespace(
            tracks_poses=SimpleNamespace(vec=Mock(return_value=base_poses)),
            tracks_timestamps_us=torch.tensor([0, 1], dtype=torch.int64),
            tracks_packinfo=torch.tensor([[0, 2]], dtype=torch.int32),
            cuboids_dims=torch.ones((1, 3), dtype=torch.float32),
            max_positive_timestamp_gap_us=torch.tensor([1], dtype=torch.int64),
        )
        resolved_calib = ResolvedUnicycleCalib()
        unicycle_data = object()
        source = cast(
            RigidGaussianModel,
            SimpleNamespace(
                tracks_calib=resolved_calib,
                cuboid_tracks=cuboid_tracks,
                collector=SimpleNamespace(calibrate_tracks_poses=Mock(return_value=calibrated_poses)),
                is_static=False,
            ),
        )
        cast(Any, source)._get_calibrated_motion_data = lambda: RigidGaussianModel._get_calibrated_motion_data(source)
        resolved_timestamps = torch.tensor([123], dtype=torch.int64)
        cast(Any, source)._get_interpolated_poses = Mock(
            side_effect=lambda *args: (args[2], torch.ones(1, dtype=torch.bool), resolved_timestamps)
        )
        model = cast(
            RigidGaussianModel,
            SimpleNamespace(
                cuboid_tracks=object(),  # Dependent appearance metadata is not the calibration input.
                get_motion_source=Mock(return_value=source),
            ),
        )
        resolved_calib.collect_data = Mock(return_value=unicycle_data)

        with patch.object(gaussians_model, "ElevationAwareUnicycleTracksCalib", ResolvedUnicycleCalib):
            result = RigidGaussianModel.prepare_motion(model, cast(RenderingData, object()), None, True)

        cast(Mock, model.get_motion_source).assert_called_once_with()
        cast(Mock, resolved_calib.collect_data).assert_called_once_with()
        cast(Mock, source.collector.calibrate_tracks_poses).assert_called_once_with(base_poses, unicycle_data)
        torch.testing.assert_close(result.poses, calibrated_poses)
        self.assertIs(result.timestamps_us, resolved_timestamps)

    def test_prepare_motion_uses_edited_poses_without_calibration(self) -> None:
        owner = RigidGaussianModel.__new__(RigidGaussianModel)
        nn.Module.__init__(owner)
        owner.config = cast(Any, SimpleNamespace(motion_source_layer=None))
        owner._motion_source_ref = None
        owner.is_static = False
        owner.cuboid_tracks = _make_single_insert_track()
        edited = _make_single_insert_track()
        edited.tracks_poses.data[:, :3].add_(10)
        edited.tracks_poses.data.requires_grad_(True)
        tracks_edit = LayerTrackIds.Edit(cuboid_tracks=edited, mapping=torch.tensor([0], dtype=torch.long))
        cast(Any, owner)._get_calibrated_motion_data = Mock(
            side_effect=AssertionError("Supplied edited poses must not receive learned calibration again")
        )
        resolved_timestamps = torch.tensor([123], dtype=torch.int64)
        interpolate = Mock(side_effect=lambda *args: (args[2], torch.ones(1, dtype=torch.bool), resolved_timestamps))
        cast(Any, owner)._get_interpolated_poses = interpolate

        result = owner.prepare_motion(cast(RenderingData, object()), None, True, tracks_edit=tracks_edit)

        interpolate.assert_called_once()
        torch.testing.assert_close(result.poses, edited.tracks_poses.vec())
        self.assertIs(result.timestamps_us, resolved_timestamps)
        result.poses[:, :3].sum().backward()
        poses_grad = edited.tracks_poses.data.grad
        assert poses_grad is not None
        self.assertTrue(torch.isfinite(poses_grad).all())

    def test_static_tracks_resolve_to_frame_midpoint_timestamps(self):
        timestamps = torch.tensor([[1000, 1600]], dtype=torch.int64)
        rendering_data = RenderingData(
            rays=torch.zeros(1, 1, 1, 6),
            sensor_model_parameters=[object()],
            poses_tquat_startend=torch.zeros(1, 2, 7),
            timestamps_startend_us=timestamps,
            timestamps_startend_us_cpu=timestamps.cpu(),
            rays_timestamps_us=None,
        )
        tracks_poses = torch.arange(28, dtype=torch.float32).reshape(4, 7)
        model = cast(
            RigidGaussianModel,
            SimpleNamespace(
                is_static=True,
                get_frame_timestamp=RigidGaussianModel.get_frame_timestamp,
            ),
        )

        poses, inside, resolved_timestamps = RigidGaussianModel._get_interpolated_poses(
            model,
            rendering_data=rendering_data,
            frustum_cull=None,
            tracks_poses=tracks_poses,
            tracks_timestamps=torch.tensor([1000, 1600, 1000, 1600], dtype=torch.int64),
            tracks_packinfo=torch.tensor([[0, 2], [2, 2]], dtype=torch.int32),
            cuboid_dims=torch.ones(2, 3),
            max_positive_timestamp_gap_us=torch.tensor([600, 600], dtype=torch.int64),
            is_training_batch=False,
        )

        torch.testing.assert_close(poses, tracks_poses[::2])
        torch.testing.assert_close(inside, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(resolved_timestamps, torch.tensor([1300, 1300], dtype=torch.int64))

    def test_deformnet_uses_each_gaussians_resolved_track_timestamp(self):
        n_tracks = 2
        n_gaussians = 3
        gaussian_track_ids = torch.tensor([1, 0, 1], dtype=torch.int32)
        base_layer_data = LayerDataRigid(
            positions=torch.zeros(n_gaussians, 3),
            rotations=torch.zeros(n_gaussians, 4),
            scales=torch.zeros(n_gaussians, 3),
            densities=torch.zeros(n_gaussians, 1),
            extra_signal=torch.zeros(n_gaussians, 0),
            camera_extra_signal=torch.zeros(n_gaussians, 0),
            lidar_extra_signal=torch.zeros(n_gaussians, 0),
            features_albedo=torch.zeros(n_gaussians, 3),
            features_specular=torch.zeros(n_gaussians, 0),
            embed_data=None,
            poses=torch.zeros(n_tracks, 7),
            keep_mask=torch.ones(n_tracks, dtype=torch.bool),
            tracks_ids=gaussian_track_ids,
        )
        timestamps = torch.tensor([[1000, 1600]], dtype=torch.int64)
        context = gaussians_model.BaseGaussianModel.CollectionContext(
            rendering_data=RenderingData(
                rays=torch.zeros(1, 1, 1, 6),
                sensor_model_parameters=[object()],
                poses_tquat_startend=torch.zeros(1, 2, 7),
                timestamps_startend_us=timestamps,
                timestamps_startend_us_cpu=timestamps.cpu(),
                rays_timestamps_us=None,
            ),
            is_training_batch=False,
            tracks_edit=None,
        )
        resolved_track_timestamps = torch.tensor([1120, 1480], dtype=torch.int64)
        context.additional_data["interpolated_track_timestamps"] = (resolved_track_timestamps, False)

        def run_deform_network(embeddings):
            return {
                "positions": torch.zeros(embeddings.shape[0], 3),
                "rotations": torch.zeros(embeddings.shape[0], 4),
            }

        captured_timestamps = []

        def compute_deform_embeddings(*, xyzs, instance_idx, timestamp_us, timestamps_delta=None):
            captured_timestamps.append(timestamp_us)
            self.assertIsNone(timestamps_delta)
            torch.testing.assert_close(instance_idx, gaussian_track_ids)
            return torch.zeros(xyzs.shape[0], 5)

        model = object.__new__(DeformableGaussianModel)
        object.__setattr__(model, "use_deform_network", True)
        object.__setattr__(model, "_compute_deform_embeddings", compute_deform_embeddings)
        object.__setattr__(model, "deform_network", run_deform_network)
        object.__setattr__(
            model,
            "config",
            SimpleNamespace(
                deform_network=SimpleNamespace(
                    deform_rotations=True,
                    rotations_from_identity=True,
                    deform_scales=False,
                ),
            ),
        )
        object.__setattr__(model, "_replaced_track_indices", set())
        object.__setattr__(model, "_inserted_track_indices", set())

        with patch.object(RigidGaussianModel, "get_layer_data", return_value=base_layer_data):
            layer_data = DeformableGaussianModel.get_layer_data(model, context, {})

        self.assertEqual(len(captured_timestamps), 1)
        torch.testing.assert_close(
            captured_timestamps[0],
            resolved_track_timestamps,
        )
        self.assertIsNotNone(layer_data.deform_positions)
        self.assertIsNotNone(layer_data.deform_rotations)

    def test_deformnet_training_smoothness_uses_each_gaussians_resolved_track_timestamp(self):
        n_tracks = 2
        n_gaussians = 3
        gaussian_track_ids = torch.tensor([1, 0, 1], dtype=torch.int32)
        base_layer_data = LayerDataRigid(
            positions=torch.zeros(n_gaussians, 3),
            rotations=torch.zeros(n_gaussians, 4),
            scales=torch.zeros(n_gaussians, 3),
            densities=torch.zeros(n_gaussians, 1),
            extra_signal=torch.zeros(n_gaussians, 0),
            camera_extra_signal=torch.zeros(n_gaussians, 0),
            lidar_extra_signal=torch.zeros(n_gaussians, 0),
            features_albedo=torch.zeros(n_gaussians, 3),
            features_specular=torch.zeros(n_gaussians, 0),
            embed_data=None,
            poses=torch.zeros(n_tracks, 7),
            keep_mask=torch.ones(n_tracks, dtype=torch.bool),
            tracks_ids=gaussian_track_ids,
        )
        timestamps = torch.tensor([[1000, 1600]], dtype=torch.int64)
        context = gaussians_model.BaseGaussianModel.CollectionContext(
            rendering_data=RenderingData(
                rays=torch.zeros(1, 1, 1, 6),
                sensor_model_parameters=[object()],
                poses_tquat_startend=torch.zeros(1, 2, 7),
                timestamps_startend_us=timestamps,
                timestamps_startend_us_cpu=timestamps.cpu(),
                rays_timestamps_us=None,
            ),
            is_training_batch=True,
            tracks_edit=None,
        )
        context.additional_data["interpolated_track_timestamps"] = (
            torch.tensor([1120, 1480], dtype=torch.int64),
            False,
        )
        context.additional_data["interpolated_mask"] = (torch.tensor([True, False]), False)
        sampled_delta = torch.tensor([3, 5, 7], dtype=torch.int64)
        captured_timestamps = []
        captured_deltas = []

        def compute_deform_embeddings(*, xyzs, instance_idx, timestamp_us, timestamps_delta=None):
            captured_timestamps.append(timestamp_us)
            captured_deltas.append(timestamps_delta)
            torch.testing.assert_close(instance_idx, gaussian_track_ids)
            return torch.zeros(3 * xyzs.shape[0], 5)

        def run_deform_network(embeddings):
            self.assertEqual(embeddings.shape[0], 3 * n_gaussians)
            positions = torch.cat(
                [
                    torch.zeros(n_gaussians, 3),
                    torch.ones(n_gaussians, 3),
                    torch.full((n_gaussians, 3), 3.0),
                ]
            )
            return {
                "positions": positions,
                "rotations": torch.zeros(3 * n_gaussians, 4),
            }

        model = object.__new__(DeformableGaussianModel)
        object.__setattr__(model, "use_deform_network", True)
        object.__setattr__(model, "_compute_deform_embeddings", compute_deform_embeddings)
        object.__setattr__(model, "deform_network", run_deform_network)
        object.__setattr__(
            model,
            "config",
            SimpleNamespace(
                smoothness_frame_steps=2,
                deform_network=SimpleNamespace(
                    deform_rotations=True,
                    rotations_from_identity=True,
                    deform_scales=False,
                ),
            ),
        )
        object.__setattr__(model, "frame_step", 100)
        object.__setattr__(model, "_replaced_track_indices", set())
        object.__setattr__(model, "_inserted_track_indices", set())
        object.__setattr__(model, "_component_only_frozen_track_mask", None)

        with (
            patch.object(RigidGaussianModel, "get_layer_data", return_value=base_layer_data),
            patch.object(gaussians_model, "get_cuda_generator", return_value=None),
            patch.object(gaussians_model.torch, "randint", return_value=sampled_delta),
        ):
            layer_data = DeformableGaussianModel.get_layer_data(model, context, {})

        self.assertEqual(len(captured_timestamps), 1)
        torch.testing.assert_close(captured_timestamps[0], torch.tensor([1120, 1480], dtype=torch.int64))
        torch.testing.assert_close(captured_deltas[0], sampled_delta)
        torch.testing.assert_close(
            context.additional_data["deform_smoothness"][0],
            torch.full((n_gaussians, 3), 2.0),
        )
        torch.testing.assert_close(
            context.additional_data["deform_smoothness_mask"][0],
            torch.tensor([[0.0], [1.0], [0.0]]),
        )
        torch.testing.assert_close(layer_data.deform_positions, torch.zeros(n_gaussians, 3))
        torch.testing.assert_close(layer_data.deform_rotations, torch.zeros(n_gaussians, 4))

    def test_post_process_track_albedo_fp16_features(self):
        """Regression: post_process runs the fp32 track_albedos @ fp16 features matmul without a dtype error."""
        device = torch.device("cuda")
        n_tracks = 2
        nb_gaussians = 3
        feature_dim = 48  # 3 albedo + 45 specular

        # post_process's track-albedo branch only reads self.track_albedos, so a stand-in suffices.
        track_albedos = torch.eye(4, device=device)[:3].unsqueeze(0).repeat(n_tracks, 1, 1)  # fp32 [n_tracks, 3, 4]
        model = SimpleNamespace(track_albedos=track_albedos)

        # fp16 collector output.
        features = torch.randn(nb_gaussians, feature_dim, device=device, dtype=torch.float16)
        results = SimpleNamespace(features=features.clone())

        layer_data = LayerDataRigid(
            positions=torch.zeros(nb_gaussians, 3, device=device),
            rotations=torch.zeros(nb_gaussians, 4, device=device),
            scales=torch.zeros(nb_gaussians, 3, device=device),
            densities=torch.zeros(nb_gaussians, 1, device=device),
            extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            camera_extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            lidar_extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            features_albedo=torch.zeros(nb_gaussians, 3, device=device),
            features_specular=torch.zeros(nb_gaussians, 45, device=device),
            embed_data=None,
            poses=torch.zeros(n_tracks, 7, device=device),
            keep_mask=torch.ones(n_tracks, device=device, dtype=torch.bool),
            tracks_ids=torch.tensor([0, 1, 0], device=device, dtype=torch.int32),
        )

        RigidGaussianModel.post_process(model, 0, results, layer_data)

        self.assertEqual(results.features.dtype, torch.float16)
        self.assertTrue(torch.isfinite(results.features).all())
        # Identity track albedo through fp32 is lossless, so both halves are bit-exact.
        torch.testing.assert_close(results.features[:, :3], features[:, :3])
        torch.testing.assert_close(results.features[:, 3:], features[:, 3:])

    def test_post_process_only_touches_the_rows_the_collect_wrote(self):
        """A windowed collect leaves other rows carrying an earlier frame's transform already applied."""
        device = torch.device("cuda")
        n_tracks = 2
        nb_gaussians = 6

        # Non-identity, so applying the affine twice is visible.
        track_albedos = torch.eye(4, device=device)[:3].unsqueeze(0).repeat(n_tracks, 1, 1)
        track_albedos[:, :, :3] *= 2.0
        track_albedos[:, :, 3] = 0.5
        model = SimpleNamespace(track_albedos=track_albedos)

        features = torch.randn(nb_gaussians, 48, device=device)
        layer_data = LayerDataRigid(
            positions=torch.zeros(nb_gaussians, 3, device=device),
            rotations=torch.zeros(nb_gaussians, 4, device=device),
            scales=torch.zeros(nb_gaussians, 3, device=device),
            densities=torch.zeros(nb_gaussians, 1, device=device),
            extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            camera_extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            lidar_extra_signal=torch.zeros(nb_gaussians, 0, device=device),
            features_albedo=torch.zeros(nb_gaussians, 3, device=device),
            features_specular=torch.zeros(nb_gaussians, 45, device=device),
            embed_data=None,
            poses=torch.zeros(n_tracks, 7, device=device),
            keep_mask=torch.ones(n_tracks, device=device, dtype=torch.bool),
            tracks_ids=torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=torch.int32),
        )

        # The dense answer, for comparison: every row transformed exactly once.
        dense = SimpleNamespace(features=features.clone())
        RigidGaussianModel.post_process(model, 0, dense, layer_data)
        self.assertFalse(torch.allclose(dense.features[:, :3], features[:, :3]), "the affine must be visible")

        # Only track 1's rows were collected this frame.
        windowed = SimpleNamespace(features=features.clone())
        rows = np.array([[3, 3]], dtype=np.int64)
        RigidGaussianModel.post_process(model, 0, windowed, layer_data, rows=rows)
        torch.testing.assert_close(windowed.features[:3], features[:3])
        torch.testing.assert_close(windowed.features[3:], dense.features[3:])

        # The next frame selects the same window. The rows outside it are still the ones
        # the collect never wrote, so they must stay untransformed rather than pick up a
        # second application.
        RigidGaussianModel.post_process(model, 0, windowed, layer_data, rows=rows)
        torch.testing.assert_close(windowed.features[:3], features[:3])

        # A warm cacheable layer writes nothing at all.
        untouched = SimpleNamespace(features=features.clone())
        RigidGaussianModel.post_process(model, 0, untouched, layer_data, rows=np.zeros((0, 2), dtype=np.int64))
        torch.testing.assert_close(untouched.features, features)

    def test_insert_asset_applies_ppisp_crf_inverse_to_inserted_albedo(self):
        cuboid_tracks = _make_single_insert_track()
        model = _InsertionModel(cuboid_tracks)
        asset = _InsertionAsset()
        original_features_albedo = asset.features_albedo.clone()
        crf_params = torch.zeros(2, 3, 4, dtype=torch.float32)
        crf_params[CRF_TEST_CAMERA_INDEX, :, 0] = 0.4
        rgb_post_processing = make_ppisp_crf_inverse_rgb_post_processing(
            crf_params,
            camera_index=CRF_TEST_CAMERA_INDEX,
        )

        RigidGaussianModel.insert_asset(
            model,
            track_id="inserted-1",
            asset=asset,
            cuboid_tracks=cuboid_tracks,
            dims_offset=torch.zeros(3),
            rgb_post_processing=rgb_post_processing,
        )

        expected_features_albedo = RGB2SH(
            ppisp_crf_inverse(SH2RGB(original_features_albedo), crf_params[CRF_TEST_CAMERA_INDEX])
        )
        torch.testing.assert_close(model.features_albedo.detach(), expected_features_albedo)
        torch.testing.assert_close(asset.features_albedo, original_features_albedo)
        self.assertFalse(torch.allclose(model.features_albedo.detach(), original_features_albedo))


# ---------------------------------------------------------------------------------------------------
# Trajectory removal + save/restore must filter EVERY per-gaussian tensor, including the
# camera/lidar extra-signal tensors, by the same include-mask. Before the fix those two tensors kept
# their original row count, so a model with `camera_extra_signal_dim`/`lidar_extra_signal_dim` > 0
# tripped the collector's `nb_gaussians == camera_extra_signal.shape[0]` assertion after removal.
# ---------------------------------------------------------------------------------------------------

_PER_GAUSSIAN_TENSOR_NAMES = (
    "positions",
    "scales",
    "rotations",
    "densities",
    "extra_signal",
    "camera_extra_signal",
    "lidar_extra_signal",
    "features_albedo",
    "features_specular",
    "gaussian_cuboid_ids",
)


def _make_removal_model(
    *, extra_dim: int, camera_dim: int, lidar_dim: int, track_ids: torch.Tensor | None = None
) -> RigidGaussianModel:
    """A minimal real ``RigidGaussianModel`` (heavy config ``__init__`` bypassed) carrying only the
    per-gaussian tensors + override bookkeeping that ``remove_gaussians_in_trajectories`` /
    ``save_training_parameters`` / ``restore_training_parameters`` touch. Six gaussians: the first three
    sit inside the removal cuboid, the last three outside it (see ``_removal_trajectory``). Every element
    has a distinct value so masking and restoration can be checked row-for-row."""
    model = RigidGaussianModel.__new__(RigidGaussianModel)
    nn.Module.__init__(model)

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # inside cuboid -> removed
            [0.5, -0.4, 0.2],  # inside cuboid -> removed
            [-0.9, 0.1, 0.3],  # inside cuboid -> removed
            [5.0, 5.0, 5.0],  # outside cuboid -> kept
            [10.0, 0.0, 0.0],  # outside cuboid -> kept
            [0.0, -8.0, 0.0],  # outside cuboid -> kept
        ],
        dtype=torch.float32,
    )
    n = positions.shape[0]

    def _rows(width: int) -> torch.Tensor:
        return torch.arange(n * width, dtype=torch.float32).reshape(n, width)

    model.positions = nn.Parameter(positions)
    model.scales = nn.Parameter(_rows(3))
    model.rotations = nn.Parameter(_rows(4))
    model.densities = nn.Parameter(_rows(1))
    model.extra_signal = nn.Parameter(_rows(extra_dim))
    model.camera_extra_signal = nn.Parameter(_rows(camera_dim))
    model.lidar_extra_signal = nn.Parameter(_rows(lidar_dim))
    model.features_albedo = nn.Parameter(_rows(3))
    model.features_specular = nn.Parameter(_rows(1))
    model.gaussian_cuboid_ids = nn.Buffer(torch.arange(n, dtype=torch.long) if track_ids is None else track_ids.clone())

    # Non-tensor state is only stored/restored by reference, so sentinels round-trip cleanly.
    model.track_albedos = None
    model.track_scales = None
    model.cuboid_tracks = cast(CuboidTracks, object())
    model.tracks_calib = cast(BaseTracksCalib, object())
    model.time_embed = None

    model.overriden_parameters = {}
    model.overriden_buffers = {}
    model.overriden_state = {}
    return model


def _removal_trajectory() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    # One identity pose + a 2x2x2 cuboid (half-extent 1.0): a gaussian is removed iff every coordinate
    # is within [-1, 1], which matches the first three rows of ``_make_removal_model``.
    trajectories = [torch.eye(4, dtype=torch.float32).unsqueeze(0)]
    cuboid_dims = [torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)]
    return trajectories, cuboid_dims


class TestRemoveGaussiansExtraSignalMasking(unittest.TestCase):
    def test_removal_masks_every_per_gaussian_tensor(self):
        model = _make_removal_model(extra_dim=2, camera_dim=3, lidar_dim=5)
        originals = {name: getattr(model, name).detach().clone() for name in _PER_GAUSSIAN_TENSOR_NAMES}
        trajectories, cuboid_dims = _removal_trajectory()

        include_mask = model.remove_gaussians_in_trajectories(trajectories, cuboid_dims)

        self.assertEqual(include_mask.tolist(), [False, False, False, True, True, True])
        kept = int(include_mask.sum())
        self.assertEqual(kept, 3)
        for name in _PER_GAUSSIAN_TENSOR_NAMES:
            tensor = getattr(model, name)
            self.assertEqual(tensor.shape[0], kept, f"{name} row count != include_mask.sum()")
            # Surviving rows are exactly the kept rows of the original tensor. This is the fix: the
            # camera/lidar extra signals must be filtered by the same mask as positions.
            torch.testing.assert_close(tensor.detach(), originals[name][include_mask])

    def test_save_remove_restore_round_trip_is_exact(self):
        model = _make_removal_model(extra_dim=2, camera_dim=3, lidar_dim=5)
        originals = {name: getattr(model, name).detach().clone() for name in _PER_GAUSSIAN_TENSOR_NAMES}
        trajectories, cuboid_dims = _removal_trajectory()

        model.remove_gaussians_in_trajectories(trajectories, cuboid_dims)
        model.restore_training_parameters()

        for name in _PER_GAUSSIAN_TENSOR_NAMES:
            restored = getattr(model, name)
            self.assertEqual(restored.shape, originals[name].shape, f"{name} not restored to full shape")
            torch.testing.assert_close(restored.detach(), originals[name])

    def test_restore_tolerates_snapshot_without_extra_signal_keys(self):
        # Compatibility: an override snapshot produced before camera/lidar extra signals were tracked
        # lacks those keys. restore_training_parameters must not KeyError; the (never-masked) fields are
        # left untouched, which is the correct no-op for such an older snapshot.
        model = _make_removal_model(extra_dim=2, camera_dim=3, lidar_dim=5)
        original_camera = model.camera_extra_signal.detach().clone()
        original_lidar = model.lidar_extra_signal.detach().clone()

        model.save_training_parameters()
        del model.overriden_parameters["camera_extra_signal"]
        del model.overriden_parameters["lidar_extra_signal"]

        model.restore_training_parameters()  # must not raise KeyError

        torch.testing.assert_close(model.camera_extra_signal.detach(), original_camera)
        torch.testing.assert_close(model.lidar_extra_signal.detach(), original_lidar)


def _make_sortable_model() -> RigidGaussianModel:
    """_make_removal_model with all six rows on one track.

    A rigid node partitions its order by track before Morton-ordering inside one. The default one-row-per-track
    layout would leave it with the identity and nothing to check. The track-partitioned order itself is covered in
    gaussians_composite_test.
    """
    return _make_removal_model(extra_dim=2, camera_dim=3, lidar_dim=5, track_ids=torch.zeros(6, dtype=torch.long))


class TestMortonSortGaussians(unittest.TestCase):
    def test_master_weights_are_refused_and_absent_ones_are_not(self):
        """apex FusedAdam shadows every parameter in param_groups_master under master_weights.

        The shadow lives outside optimizer.state, so the reorder cannot permute it and refuses instead.
        Entries are None when master_weights is off, which must stay permittable.
        """
        model = _make_sortable_model()
        order = model.compute_morton_order()
        assert order is not None
        optimizer = torch.optim.Adam([model.positions])
        model.optimizers = [{"optimizer": optimizer}]

        # master_weights off: apex still builds the list, filled with None.
        optimizer.param_groups_master = [{"params": [None] * len(g["params"])} for g in optimizer.param_groups]
        model.validate_optimizer_state(order)

        # master_weights on: a real fp32 shadow of one parameter.
        shadow = optimizer.param_groups[0]["params"][0].detach().clone().float()
        optimizer.param_groups_master[0]["params"][0] = shadow
        with self.assertRaises(ValueError) as caught:
            model.validate_optimizer_state(order)
        self.assertIn("master weights", str(caught.exception))

    def test_a_live_gradient_is_permuted_with_its_rows(self):
        """Lightning runs callbacks before the module hook, so a save can land between backward and
        densification, which reads positions.grad and asserts it is not None.
        """
        model = _make_sortable_model()
        n = model.positions.shape[0]
        # A distinct gradient per row, so a wrong pairing is detectable rather than symmetric.
        model.positions.grad = torch.arange(float(n)).unsqueeze(1).repeat(1, 3)
        before = model.positions.grad.clone()

        order = model.morton_sort_gaussians()

        assert order is not None
        self.assertIsNotNone(model.positions.grad)
        torch.testing.assert_close(model.positions.grad, before[order])

    def test_sort_permutes_every_per_gaussian_tensor_by_the_same_order(self):
        model = _make_sortable_model()
        originals = {name: getattr(model, name).detach().clone() for name in _PER_GAUSSIAN_TENSOR_NAMES}

        order = model.morton_sort_gaussians()

        assert order is not None
        self.assertEqual(sorted(order.tolist()), list(range(originals["positions"].shape[0])))
        for name in _PER_GAUSSIAN_TENSOR_NAMES:
            torch.testing.assert_close(getattr(model, name).detach(), originals[name][order], msg=name)

        # An identity or a permutation of the wrong tensor would satisfy the loop above.
        # Check the order is actually spatial: sorted rows sit closer to their neighbours.
        def mean_step(p: torch.Tensor) -> torch.Tensor:
            return (p[1:] - p[:-1]).norm(dim=1).mean()

        self.assertLess(mean_step(model.positions.detach()), mean_step(originals["positions"]))

    def test_sort_moves_adam_moments_with_their_parameters(self):
        model = _make_sortable_model()
        positions = model.positions
        # Nameless group, which is what configure_optimizers emits for the group holding every parameter.
        # Matching on a group name would skip it and leave the optimizer pointing at stale tensors.
        optimizer = torch.optim.Adam([{"params": [positions]}], lr=0.1)
        exp_avg = torch.arange(positions.numel(), dtype=torch.float32).reshape(positions.shape)
        optimizer.state[positions] = {
            "step": torch.tensor(7.0),
            "exp_avg": exp_avg.clone(),
            "exp_avg_sq": exp_avg.clone() * 2.0,
        }
        model.optimizers = [{"optimizer": optimizer}]

        order = model.morton_sort_gaussians()

        assert order is not None
        sorted_positions = model.positions
        self.assertIs(sorted_positions, optimizer.param_groups[0]["params"][0])
        state = optimizer.state[sorted_positions]
        torch.testing.assert_close(state["exp_avg"], exp_avg[order])
        torch.testing.assert_close(state["exp_avg_sq"], exp_avg[order] * 2.0)
        # The step counter is per-optimizer bookkeeping, not a per-gaussian row.
        torch.testing.assert_close(state["step"], torch.tensor(7.0))

    def test_sort_drops_per_row_gradient_hooks(self):
        # Dropping the hooks is what lets the composite rebuild their row masks, so pin it here.
        model = _make_sortable_model()
        fired = []
        model.positions.register_hook(lambda grad: fired.append("stale") or grad)

        model.morton_sort_gaussians()
        model.positions.sum().backward()

        self.assertEqual(fired, [])

    def test_sort_keeps_every_member_of_a_multi_param_group(self):
        # get_model_parameters emits multi-param groups, which rebuilding the list as [new_param] would drop.
        model = _make_sortable_model()
        other = nn.Parameter(torch.arange(3.0))
        optimizer = torch.optim.Adam([{"params": [model.positions, other], "name": "grouped"}], lr=0.1)
        model.optimizers = [{"optimizer": optimizer}]

        model.morton_sort_gaussians()

        params = optimizer.param_groups[0]["params"]
        self.assertEqual(len(params), 2)
        self.assertIs(params[0], model.positions)
        self.assertIs(params[1], other)

    def test_sort_does_not_bind_a_group_to_a_same_named_attribute(self):
        # A deform-net group whose name tail is "scales" must not capture the Gaussian scales.
        # Neither its parameter nor its optimizer state may be touched.
        model = _make_sortable_model()
        foreign = nn.Parameter(torch.arange(4.0))
        optimizer = torch.optim.Adam([{"params": [foreign], "name": "deform_network.scales"}], lr=0.1)
        optimizer.state[foreign] = {"exp_avg": torch.ones(4)}
        model.optimizers = [{"optimizer": optimizer}]

        model.morton_sort_gaussians()

        self.assertIs(optimizer.param_groups[0]["params"][0], foreign)
        torch.testing.assert_close(optimizer.state[foreign]["exp_avg"], torch.ones(4))


# How to materialize each row-aligned buffer a Gaussian variant can declare.
# Interleaved track ids, because one row per track would leave a track-partitioned order as the identity.
_ROW_BUFFER_SEEDS: dict[str, Any] = {
    "barrier_object_id": lambda n: nn.Buffer(torch.arange(n, dtype=torch.long) % 4),
    "gaussian_cuboid_ids": lambda n: nn.Buffer(torch.arange(n, dtype=torch.long) % 4),
    "mcmc_position_anchor": lambda n: nn.Buffer(torch.rand(n, 3)),
    "gaussians_hit_counter": lambda n: nn.Buffer(torch.arange(n, dtype=torch.float32)),
    "gaussians_hit_mask": lambda n: nn.Buffer(torch.ones(n, dtype=torch.bool)),
}

# Row-aligned tensors that deliberately do not move with a Morton order, keyed by name with the reason.
# Empty today. An entry here is a decision, so it needs a reason next to it rather than a quiet skip.
_ROW_ALIGNED_EXEMPTIONS: dict[str, str] = {}


def _make_completeness_node(cls: type[gaussians_model.BaseGaussianModel], n: int) -> Any:
    """Build cls with exactly the row-aligned tensors it declares, heavy config __init__ bypassed.

    Every known row buffer is seeded, then the ones this class does not claim through get_additional_buffers
    are removed, so the node ends up carrying that class's real set rather than a union.
    """
    model = cls.__new__(cls)
    nn.Module.__init__(model)

    def _rows(width: int) -> torch.Tensor:
        return torch.arange(n * width, dtype=torch.float32).reshape(n, width)

    generator = torch.Generator().manual_seed(0)
    model.positions = nn.Parameter(torch.rand(n, 3, generator=generator) * 40.0)
    model.scales = nn.Parameter(_rows(3))
    model.rotations = nn.Parameter(_rows(4))
    model.densities = nn.Parameter(_rows(1))
    model.extra_signal = nn.Parameter(_rows(2))
    model.camera_extra_signal = nn.Parameter(_rows(3))
    model.lidar_extra_signal = nn.Parameter(_rows(5))
    model.features_albedo = nn.Parameter(_rows(3))
    model.features_specular = nn.Parameter(_rows(1))
    # restriction_policy gates DeformableRigidAssetModel's hook reinstall, everything else ignores the stub.
    model.config = cast(Any, SimpleNamespace(restriction_policy="none"))

    for name, seed in _ROW_BUFFER_SEEDS.items():
        setattr(model, name, seed(n))
    claimed = set(model.get_additional_buffers())
    for name in _ROW_BUFFER_SEEDS:
        if name not in claimed:
            delattr(model, name)
    return model


class TestMortonSortCoversEveryRowAlignedTensor(unittest.TestCase):
    """A row-aligned tensor that the reorder misses desynchronizes silently, since the row count never changes.

    The other Morton tests check a hand-written list of tensor names against the same list restated here, so a
    class carrying a tensor in neither list passes both. This walks what each node actually holds instead, and
    it iterates the factory registry rather than a fixed class list, so a variant added later is covered without
    anyone remembering to extend this file.
    """

    _N = 8

    def test_every_row_buffer_a_variant_declares_can_be_seeded(self) -> None:
        # The sweep below can only see tensors the fixture materialized. This is what stops a newly declared
        # buffer from being invisible to it: seeding has to be taught the name before the sweep can judge it.
        for variant, cls in sorted(gaussians_model.BaseGaussianModel.GAUSSIANS_VARIANTS.items()):
            with self.subTest(variant=variant):
                model = _make_completeness_node(cls, self._N)
                unseeded = set(model.get_additional_buffers()) - set(_ROW_BUFFER_SEEDS)
                self.assertFalse(unseeded, f"{variant} declares {sorted(unseeded)}, add a seed to _ROW_BUFFER_SEEDS")

    def test_every_row_aligned_tensor_moves_with_the_order(self) -> None:
        variants = gaussians_model.BaseGaussianModel.GAUSSIANS_VARIANTS
        # An empty or half-populated registry would make every subTest below vacuous.
        self.assertGreaterEqual(len(variants), 4, "gaussian variants are not all imported")

        for variant, cls in sorted(variants.items()):
            with self.subTest(variant=variant):
                model = _make_completeness_node(cls, self._N)
                before = {
                    name: tensor.detach().clone()
                    for name, tensor in [*model.named_parameters(), *model.named_buffers()]
                    if tensor.dim() > 0 and tensor.shape[0] == self._N
                }
                self.assertTrue(before, f"{variant} materialized no row-aligned tensors")

                order = model.morton_sort_gaussians()

                assert order is not None
                self.assertEqual(sorted(order.tolist()), list(range(self._N)))
                # An identity order would satisfy every comparison below without moving anything.
                self.assertFalse(torch.equal(order, torch.arange(self._N)))
                for name, original in before.items():
                    if name in _ROW_ALIGNED_EXEMPTIONS:
                        continue
                    current = (
                        model.get_parameter(name) if name in dict(model.named_parameters()) else model.get_buffer(name)
                    )
                    torch.testing.assert_close(current.detach(), original[order], msg=f"{variant}.{name}")


class TestDeformTablesGrowOnInsert(unittest.TestCase):
    """The instance-embedding, per-track timestamp and AABB tables are sized at construction;
    inserting a track grows the node's track table but not those, and the Slang kernel gathers
    all three by ``instance_idx`` with no bounds check."""

    @staticmethod
    def _tracks(n: int) -> CuboidTracks:
        from nurec.datasets.tracks import CuboidTracksData, TracksData

        return CuboidTracks(
            tracks_data=TracksData(
                tracks_id=[f"t{i}" for i in range(n)],
                tracks_poses=lt.SE3(torch.zeros(n * 2, 7).index_fill_(1, torch.tensor([6]), 1.0)),
                tracks_timestamps_us=torch.arange(n * 2, dtype=torch.int64) * 1000,
                tracks_packinfo=torch.tensor([[i * 2, 2] for i in range(n)], dtype=torch.int64),
                max_track_n_poses=2,
                tracks_label_class=["person"] * n,
                tracks_flags=torch.zeros(n, dtype=torch.int32),
                max_positive_timestamp_gap_us=torch.full((n,), 1_000_000, dtype=torch.int64),
            ),
            cuboidtracks_data=CuboidTracksData(
                cuboids_dims=torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
            ),
        )

    def _model(self, n_old: int):
        from nurec.models.input_embedding import (
            IndividualRemapTimeInputEmbedding,
            WeightedInstanceInputEmbedding,
        )
        from nurec.utils.types import AABB3D

        trainer_config = TrainerConfig(
            max_epochs=1,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            precision="32",
        )
        embed_cfg = OmegaConf.create(
            {
                "name": "weighted-instance-input-embedding",
                "embedding_dim": 1,
                "requires_grad": True,
                "weight_init_config": {"method": "linspace", "from_": 0.05, "to_": 0.95},
            }
        )
        time_cfg = OmegaConf.create({"name": "individual-remap-time-input-embedding", "remap_min": 0, "remap_max": 1})
        old = self._tracks(n_old)
        volume = SimpleNamespace(
            instance_input_embedding=WeightedInstanceInputEmbedding(embed_cfg, trainer_config, tracks=old),
            time_input_embedding=IndividualRemapTimeInputEmbedding(time_cfg, trainer_config, tracks=old),
        )
        model = object.__new__(DeformableGaussianModel)
        nn.Module.__init__(model)
        object.__setattr__(model, "deform_network", SimpleNamespace(feature_volume=volume))
        object.__setattr__(model, "_replaced_track_indices", set())
        object.__setattr__(model, "_inserted_track_indices", set())
        object.__setattr__(model, "_pre_insert_deform_tables", None)
        model.aabb_extent = nn.Buffer(old.cuboids_dims.clone())
        side = torch.max(model.aabb_extent, dim=1, keepdim=True).values.repeat(1, 3)
        model.aabbs = AABB3D(blb=-side / 2.0, trf=side / 2.0)
        return model, volume

    def _insert(self, model, n_tracks: int, track_id: str) -> None:
        """One track per call, as `GaussiansComposite.insert_asset` does."""
        with patch.object(gaussians_model.RigidGaussianModel, "update_tracks_calib_and_time_embed"):
            DeformableGaussianModel.update_tracks_calib_and_time_embed(model, self._tracks(n_tracks), {track_id})

    @staticmethod
    def _tables(model, volume) -> list[torch.Tensor]:
        return [
            volume.instance_input_embedding.embedding.weight,
            volume.time_input_embedding.timestamps_us_ranges,
            model.aabb_extent,
            model.aabbs.blb,
            model.aabbs.trf,
        ]

    def test_every_per_track_table_covers_the_inserted_tracks(self):
        model, volume = self._model(3)
        before = [table.detach().clone() for table in self._tables(model, volume)]

        self._insert(model, 4, "t3")
        self._insert(model, 5, "t4")

        after = self._tables(model, volume)
        self.assertEqual([table.shape[0] for table in after], [5] * len(after))
        self.assertEqual(volume.instance_input_embedding.embedding.num_embeddings, 5)
        # Rows for pre-existing tracks are untouched.
        for old, new in zip(before, after):
            torch.testing.assert_close(new[:3], old, rtol=0, atol=0)
        # The remap divides by the span, so no row may be degenerate.
        ranges = volume.time_input_embedding.timestamps_us_ranges
        self.assertTrue(bool((ranges[:, 0] < ranges[:, 1]).all()))

    def test_staged_insert_does_not_publish_deform_tables_early(self):
        """Atomic composite preparation must leave every live DeformNet table untouched."""
        model, volume = self._model(3)
        originals = self._tables(model, volume)

        prepared = model._prepare_asset_insertion_model_state(self._tracks(4), {"t3"})

        for original, current in zip(originals, self._tables(model, volume)):
            self.assertIs(current, original)
        self.assertEqual(model._inserted_track_indices, set())
        self.assertIsNone(model._pre_insert_deform_tables)

        model._apply_prepared_asset_insertion_model_state(prepared)

        self.assertEqual([table.shape[0] for table in self._tables(model, volume)], [4] * len(originals))
        self.assertEqual(volume.instance_input_embedding.embedding.num_embeddings, 4)
        self.assertEqual(model._inserted_track_indices, {3})
        self.assertEqual([original for _, _, original in model._pre_insert_deform_tables], originals)

    def test_inserted_tracks_are_marked_rigid(self):
        """The instance table is a fixed grid the field was trained against, so an inserted track
        has no coordinate of its own; it is excluded from deformation like a replaced track. Every
        inserted track must stay excluded as later ones are inserted."""
        model, _ = self._model(3)

        self._insert(model, 4, "t3")
        self._insert(model, 5, "t4")

        self.assertEqual(model._inserted_track_indices, {3, 4})

    def test_new_instance_rows_stay_inside_the_trained_range(self):
        """Rows are a linspace over ``[from_, to_]``; a new row outside it would query a part of
        the deformation field that training never reached."""
        model, volume = self._model(3)
        before = volume.instance_input_embedding.embedding.weight.detach().clone()

        self._insert(model, 4, "t3")
        self._insert(model, 5, "t4")

        new_rows = volume.instance_input_embedding.embedding.weight[3:]
        self.assertTrue(bool((new_rows >= before.min()).all() and (new_rows <= before.max()).all()))

    def test_grown_tables_stay_on_their_device(self):
        model, volume = self._model(3)
        meta = torch.device("meta")
        volume.time_input_embedding.timestamps_us_ranges = volume.time_input_embedding.timestamps_us_ranges.to(meta)

        self._insert(model, 4, "t3")

        self.assertEqual(volume.time_input_embedding.timestamps_us_ranges.device.type, meta.type)

    def test_restore_puts_the_tables_back(self):
        model, volume = self._model(3)
        originals = self._tables(model, volume)

        self._insert(model, 4, "t3")
        self._insert(model, 5, "t4")
        with patch.object(gaussians_model.RigidGaussianModel, "restore_training_parameters"):
            DeformableGaussianModel.restore_training_parameters(model)

        for original, restored in zip(originals, self._tables(model, volume)):
            self.assertIs(restored, original)
        self.assertEqual(volume.instance_input_embedding.embedding.num_embeddings, 3)
        self.assertEqual(model._inserted_track_indices, set())


if __name__ == "__main__":
    unittest.main()
