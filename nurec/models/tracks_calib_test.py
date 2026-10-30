# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from unittest.mock import MagicMock

import lietorch as lt
import numpy as np
import pytest
import torch

from omegaconf import OmegaConf
from torch import nn

import nurec.models.tracks_calib as tracks_calib_module

from nurec.config.trainer import TrainerConfig
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.models.tracks_calib import (
    BaseTracksCalib,
    DirectTracksCalib,
    ElevationAwareUnicycleTracksCalib,
    StaticTracksCalib,
)
from nurec.models.tracks_calib_data import TracksCalibResiduals


def _translation_pose(translation: list[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array(translation, dtype=np.float32)
    return pose


def _rotation_y(angle: float) -> np.ndarray:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ],
        dtype=np.float32,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        max_epochs=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        precision=32,
    )


def _static_tracks_calib(cuboid_tracks: CuboidTracks) -> StaticTracksCalib:
    return StaticTracksCalib(
        config=OmegaConf.create({"visualize": False}),
        trainer_config=_trainer_config(),
        cuboid_tracks=cuboid_tracks,
    )


def _direct_tracks_calib(
    cuboid_tracks: CuboidTracks,
    delta_frame: str | None = None,
    calib_class: type[DirectTracksCalib] = DirectTracksCalib,
    *,
    fix_first_pose: bool = False,
    fix_last_pose: bool = False,
) -> DirectTracksCalib:
    config: dict[str, object] = {
        "visualize": False,
        "start_global_step": 0,
        "fix_first_pose": fix_first_pose,
        "fix_last_pose": fix_last_pose,
    }
    if delta_frame is not None:
        config["delta_frame"] = delta_frame

    return calib_class(
        config=OmegaConf.create(config),
        trainer_config=_trainer_config(),
        cuboid_tracks=cuboid_tracks,
    )


def _unicycle_tracks_calib(
    cuboid_tracks: CuboidTracks,
    *,
    fix_first_pose: bool,
    fix_last_pose: bool,
) -> ElevationAwareUnicycleTracksCalib:
    return ElevationAwareUnicycleTracksCalib(
        config=OmegaConf.create(
            {
                "visualize": False,
                "start_global_step": -1,
                "fix_first_pose": fix_first_pose,
                "fix_last_pose": fix_last_pose,
                "pitch_slope_clamp_rad": 1.2,
                "cuboid_tracks": {
                    "forward_axis": [1.0, 0.0, 0.0],
                    "up_axis": [0.0, 0.0, 1.0],
                },
            }
        ),
        trainer_config=_trainer_config(),
        cuboid_tracks=cuboid_tracks,
    )


def _two_tracks_for_unicycle_calib(num_poses: int, device: torch.device = torch.device("cpu")) -> CuboidTracks:
    """Two tracks of different lengths on non-uniform timestamps.

    Per-track indexing (pair/triplet starts, per-track gathers) and the time scaling in the
    residuals only have observable effects with more than one track and time deltas away
    from 1 s; a single track sampled at 1 Hz makes both fixed points.
    """

    def track(count: int, offset: list[float], deltas_s: list[float]) -> tuple[np.ndarray, np.ndarray]:
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
        poses[:, 0, 3] = offset[0] + np.arange(count, dtype=np.float32)
        poses[:, 1, 3] = offset[1]
        poses[:, 2, 3] = offset[2]
        deltas = np.array([deltas_s[i % len(deltas_s)] for i in range(count - 1)])
        timestamps = np.concatenate([[0.0], np.cumsum(deltas)]) * 1e6
        return poses, timestamps.astype(np.int64)

    poses_a, timestamps_a = track(num_poses, [0.0, 0.0, 0.0], [0.4, 1.1, 0.7, 0.5, 1.3])
    poses_b, timestamps_b = track(num_poses + 3, [20.0, 5.0, 1.0], [0.9, 0.6, 1.4, 0.8])
    return CuboidTracks.Factory.from_numpy(
        tracks_id=["track_a", "track_b"],
        tracks_poses=[poses_a, poses_b],
        tracks_timestamps_us=[timestamps_a, timestamps_b],
        tracks_label_class=["unknown", "unknown"],
        tracks_flags=[TrackFlags.NONE, TrackFlags.NONE],
        cuboids_dims=[np.ones(3, dtype=np.float32), np.array([2.0, 1.5, 1.0], dtype=np.float32)],
        device=device,
    )


def _tracks_for_unicycle_calib(num_poses: int = 3, device: torch.device = torch.device("cpu")) -> CuboidTracks:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], num_poses, axis=0)
    poses[:, 0, 3] = np.arange(num_poses, dtype=np.float32)
    return CuboidTracks.Factory.from_numpy(
        tracks_id=["track"],
        tracks_poses=[poses],
        tracks_timestamps_us=[np.arange(num_poses, dtype=np.int64) * 1_000_000],
        tracks_label_class=["unknown"],
        tracks_flags=[TrackFlags.NONE],
        cuboids_dims=[np.ones(3, dtype=np.float32)],
        device=device,
    )


def _tracks_for_static_calib(device: torch.device) -> CuboidTracks:
    return CuboidTracks.Factory.from_numpy(
        tracks_id=["track_a", "track_b"],
        tracks_poses=[
            np.stack(
                [
                    _translation_pose([0.0, 0.0, 0.0]),
                    _translation_pose([2.0, 4.0, 6.0]),
                ]
            ),
            np.stack(
                [
                    _translation_pose([10.0, 1.0, -1.0]),
                    _translation_pose([14.0, 5.0, -5.0]),
                    _translation_pose([18.0, 9.0, -9.0]),
                ]
            ),
        ],
        tracks_timestamps_us=[
            np.array([0, 1], dtype=np.int64),
            np.array([10, 20, 30], dtype=np.int64),
        ],
        tracks_label_class=["unknown", "unknown"],
        tracks_flags=[TrackFlags.NONE, TrackFlags.NONE],
        cuboids_dims=[
            np.array([1, 1, 1], dtype=np.float32),
            np.array([2, 2, 2], dtype=np.float32),
        ],
        device=device,
    )


class _ParamCalib(BaseTracksCalib):
    """Minimal calib exposing one pose-, one track- and one inter-pose-indexed
    parameter to exercise ``BaseTracksCalib.freeze_tracks``'s three branches."""

    def __init__(self, n_poses: int, n_tracks: int):
        nn.Module.__init__(self)  # skip BaseTracksCalib.__init__ (no config needed)
        self.pose_p = nn.Parameter(torch.randn(n_poses, 2))
        self.track_p = nn.Parameter(torch.randn(n_tracks, 4))
        self.inter_p = nn.Parameter(torch.randn(n_poses - n_tracks))

    def forward(self, tracks: CuboidTracks) -> CuboidTracks:
        return tracks


def test_direct_tracks_calib_freeze_tracks_zeros_frozen_pose_deltas() -> None:
    # track_a: 2 poses (rows 0-1), track_b: 3 poses (rows 2-4).
    tracks = _tracks_for_static_calib(torch.device("cpu"))
    calib = _direct_tracks_calib(tracks)

    calib.freeze_tracks([1], tracks.tracks_packinfo)  # freeze track_b
    (calib.tracks_delta_t.sum() + calib.tracks_delta_q.sum()).backward()

    for param in (calib.tracks_delta_t, calib.tracks_delta_q):
        grad = param.grad
        assert grad is not None
        torch.testing.assert_close(grad[:2], torch.ones_like(grad[:2]), rtol=0, atol=0)  # track_a trains
        torch.testing.assert_close(grad[2:], torch.zeros_like(grad[2:]), rtol=0, atol=0)  # track_b frozen


def test_freeze_tracks_generic_covers_pose_track_and_inter_indexed_params() -> None:
    # 2 tracks: track 0 = 2 poses (1 inter), track 1 = 3 poses (2 inter).
    packinfo = torch.tensor([[0, 2], [2, 3]], dtype=torch.int64)
    calib = _ParamCalib(n_poses=5, n_tracks=2)

    calib.freeze_tracks([1], packinfo)  # freeze track 1
    (calib.pose_p.sum() + calib.track_p.sum() + calib.inter_p.sum()).backward()

    pose_grad, track_grad, inter_grad = calib.pose_p.grad, calib.track_p.grad, calib.inter_p.grad
    assert pose_grad is not None and track_grad is not None and inter_grad is not None
    torch.testing.assert_close(pose_grad[:2], torch.ones(2, 2), rtol=0, atol=0)
    torch.testing.assert_close(pose_grad[2:], torch.zeros(3, 2), rtol=0, atol=0)
    torch.testing.assert_close(track_grad[0], torch.ones(4), rtol=0, atol=0)
    torch.testing.assert_close(track_grad[1], torch.zeros(4), rtol=0, atol=0)
    torch.testing.assert_close(inter_grad[:1], torch.ones(1), rtol=0, atol=0)  # track 0's inter entry
    torch.testing.assert_close(inter_grad[1:], torch.zeros(2), rtol=0, atol=0)  # track 1's inter entries


def _tracks_for_direct_calib() -> CuboidTracks:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pose[:3, 3] = np.array([10.0, 0.0, 0.0], dtype=np.float32)

    return CuboidTracks.Factory.from_numpy(
        tracks_id=["track"],
        tracks_poses=[np.stack([pose, pose])],
        tracks_timestamps_us=[np.array([0, 1], dtype=np.int64)],
        tracks_label_class=["unknown"],
        tracks_flags=[TrackFlags.NONE],
        cuboids_dims=[np.ones(3, dtype=np.float32)],
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    ("delta_frame", "expected_left_multiply"),
    [
        (None, True),
        ("world", True),
        ("local", False),
    ],
)
def test_direct_tracks_calib_delta_frame(
    delta_frame: str | None,
    expected_left_multiply: bool,
) -> None:
    tracks_calib = _direct_tracks_calib(_tracks_for_direct_calib(), delta_frame)

    assert tracks_calib.delta_frame == (delta_frame or "world")
    assert tracks_calib.left_multiply is expected_left_multiply


def test_direct_tracks_calib_rejects_invalid_delta_frame() -> None:
    with pytest.raises(ValueError, match="delta_frame must be 'world' or 'local'"):
        _direct_tracks_calib(_tracks_for_direct_calib(), "invalid")


@pytest.mark.parametrize("delta_frame", ["world", "local"])
def test_direct_tracks_calib_applies_delta_in_configured_frame(delta_frame: str) -> None:
    tracks = _tracks_for_direct_calib()
    tracks_calib = _direct_tracks_calib(tracks, delta_frame)

    delta_translation = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    delta_quaternion = torch.tensor([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=torch.float32)
    with torch.no_grad():
        tracks_calib.tracks_delta_t[:] = delta_translation
        tracks_calib.tracks_delta_q[:] = delta_quaternion

    delta_pose = tracks_calib.get_tracks_delta_transform()
    expected_poses = delta_pose * tracks.tracks_poses if delta_frame == "world" else tracks.tracks_poses * delta_pose
    calibrated_tracks = tracks_calib(tracks)

    torch.testing.assert_close(calibrated_tracks.tracks_poses.matrix(), expected_poses.matrix())


def _tracks_with_rotations(device: torch.device) -> CuboidTracks:
    poses = []
    for i in range(5):
        pose = _translation_pose([2.0 * i, -1.5 * i, 0.5 * i])
        pose[:3, :3] = _rotation_z(0.3 * i) @ _rotation_y(-0.2 * i)
        poses.append(pose)
    return CuboidTracks.Factory.from_numpy(
        tracks_id=["track_a", "track_b"],
        tracks_poses=[np.stack(poses[:2]), np.stack(poses[2:])],
        tracks_timestamps_us=[
            np.array([0, 1], dtype=np.int64),
            np.array([10, 20, 30], dtype=np.int64),
        ],
        tracks_label_class=["unknown", "unknown"],
        tracks_flags=[TrackFlags.NONE, TrackFlags.NONE],
        cuboids_dims=[np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
        device=device,
    )


# The expected masks spell out the fixture's packing (rows 0-1 and 2-4), so a mask
# built per-tensor instead of per-track cannot satisfy them.
@pytest.mark.parametrize(
    ("fix_first_pose", "fix_last_pose", "expected_mask"),
    [
        (True, False, [False, True, False, True, True]),
        (False, True, [True, False, True, True, False]),
        (True, True, [False, False, False, True, False]),
    ],
)
def test_direct_gradient_mask_freezes_requested_rows(
    fix_first_pose: bool, fix_last_pose: bool, expected_mask: list[bool]
) -> None:
    tracks = _tracks_with_rotations(torch.device("cpu"))
    calib = _direct_tracks_calib(tracks, fix_first_pose=fix_first_pose, fix_last_pose=fix_last_pose)
    assert calib.gradient_mask.cpu().tolist() == expected_mask


@pytest.mark.parametrize(("fix_first_pose", "fix_last_pose"), [(True, False), (False, True), (True, True)])
def test_direct_frozen_rows_get_no_gradient(fix_first_pose: bool, fix_last_pose: bool) -> None:
    tracks = _tracks_with_rotations(torch.device("cpu"))
    calib = _direct_tracks_calib(tracks, fix_first_pose=fix_first_pose, fix_last_pose=fix_last_pose)
    n = calib.tracks_delta_t.shape[0]
    # Non-trivial deltas exercise the gradient away from the identity point.
    with torch.no_grad():
        calib.tracks_delta_t.add_(torch.linspace(-0.5, 0.5, n).view(-1, 1) * torch.tensor([[1.0, -0.5, 0.25]]))
        calib.tracks_delta_q.add_(torch.linspace(-0.2, 0.3, n).view(-1, 1) * torch.tensor([[0.1, -0.2, 0.15, 0.05]]))

    got = calib(tracks).tracks_poses.vec()
    # A generic cotangent reaches every gradient direction, so a trainable row cannot
    # hide a zero gradient by construction.
    upstream = torch.randn(got.shape, generator=torch.Generator().manual_seed(99))
    grads = torch.autograd.grad((got * upstream).sum(), [calib.tracks_delta_t, calib.tracks_delta_q])

    keep = calib.gradient_mask.tolist()
    frozen = [row for row, kept in enumerate(keep) if not kept]
    trainable = [row for row, kept in enumerate(keep) if kept]
    for grad in grads:
        assert grad[frozen].abs().max().item() == 0.0
        assert grad[trainable].abs().amax(dim=1).min().item() > 0.0


def test_static_tracks_calib_computes_grouped_mean_translation() -> None:
    tracks = _tracks_for_static_calib(torch.device("cpu"))

    tracks_calib = _static_tracks_calib(tracks)

    assert tracks_calib.static_poses.shape[0] == 2
    torch.testing.assert_close(
        tracks_calib.static_poses.vec()[:, :3],
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [14.0, 5.0, -5.0],
            ],
            dtype=torch.float32,
        ),
        rtol=0,
        atol=0,
    )

    frozen_tracks = tracks_calib(tracks)
    torch.testing.assert_close(
        frozen_tracks.max_positive_timestamp_gap_us,
        torch.full((2,), 2_000_000_030, dtype=torch.int64),
        rtol=0,
        atol=0,
    )


def test_static_tracks_calib_cuda_construction_is_repeatable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to check GPU repeatability")

    tracks = _tracks_for_static_calib(torch.device("cuda"))
    reference = _static_tracks_calib(tracks).static_poses.vec().detach().cpu()

    for _ in range(10):
        current = _static_tracks_calib(tracks).static_poses.vec().detach().cpu()
        torch.testing.assert_close(current, reference, rtol=0, atol=0)


def test_static_tracks_calib_mean_poses_follow_the_module() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to check that the mean poses move with the module")

    # Building on the CPU and moving afterwards is the path that leaves an unregistered attribute behind.
    tracks_calib = _static_tracks_calib(_tracks_for_static_calib(torch.device("cpu"))).to(torch.device("cuda"))

    assert tracks_calib.static_poses.device.type == "cuda"

    frozen_tracks = tracks_calib(_tracks_for_static_calib(torch.device("cuda")))
    assert frozen_tracks.tracks_poses.device.type == "cuda"


def test_static_tracks_calib_rejects_empty_tracks() -> None:
    with pytest.raises(AssertionError, match="requires at least one track"):
        _static_tracks_calib(CuboidTracks.Factory.empty(device=torch.device("cpu")))


@pytest.mark.parametrize(
    ("fix_first_pose", "fix_last_pose", "expected_gradient_mask"),
    [
        (True, False, [False, True, False, True, True]),
        (False, True, [True, False, True, True, False]),
        (True, True, [False, False, False, True, False]),
        (False, False, [True, True, True, True, True]),
    ],
)
def test_unicycle_tracks_calib_fixes_requested_track_endpoints(
    fix_first_pose: bool,
    fix_last_pose: bool,
    expected_gradient_mask: list[bool],
) -> None:
    tracks_calib = _unicycle_tracks_calib(
        _tracks_for_static_calib(torch.device("cpu")),
        fix_first_pose=fix_first_pose,
        fix_last_pose=fix_last_pose,
    )

    expected_gradient_mask_tensor = torch.tensor(expected_gradient_mask)
    torch.testing.assert_close(tracks_calib.pose_gradient_mask, expected_gradient_mask_tensor)

    pose_params = tracks_calib._get_pose_params()
    torch.stack([param.sum() for param in pose_params]).sum().backward()
    for param in (tracks_calib.xy, tracks_calib.height, tracks_calib.theta, tracks_calib.pitch):
        assert param.grad is not None
        per_pose_has_gradient = param.grad.reshape(param.shape[0], -1).abs().sum(dim=1) > 0
        torch.testing.assert_close(per_pose_has_gradient, expected_gradient_mask_tensor)


# Three poses leave the motion-triplet mapping empty; six populate it. Both are allocated in the
# constructor, so each length reaches a different allocation site.
@pytest.mark.parametrize("num_poses", [3, 6], ids=["no_motion_triplets", "with_motion_triplets"])
def test_unicycle_tracks_calib_builds_on_the_tracks_device(num_poses: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to check construction from GPU tracks")

    # The tracks reach the calibrator on the GPU while the module itself is still on the CPU.
    cuda_tracks = _tracks_for_unicycle_calib(num_poses=num_poses, device=torch.device("cuda"))
    cuda_calib = _unicycle_tracks_calib(cuda_tracks, fix_first_pose=True, fix_last_pose=True)

    for name in (
        "Tclc_tquat",
        "pose_gradient_mask",
        "tracks_idx",
        "pose_pair_start_indices",
        "motion_triplet_start_indices",
    ):
        assert getattr(cuda_calib, name).device == cuda_tracks.device, f"{name} stayed behind"

    assert cuda_calib.get_smoothness().device == cuda_tracks.device

    cpu_tracks = _tracks_for_unicycle_calib(num_poses=num_poses, device=torch.device("cpu"))
    cpu_calib = _unicycle_tracks_calib(cpu_tracks, fix_first_pose=True, fix_last_pose=True)
    torch.testing.assert_close(
        cuda_calib.get_pose().vec().detach().cpu(),
        cpu_calib.get_pose().vec().detach(),
        rtol=0,
        atol=1e-4,
    )


def test_unicycle_tracks_calib_reconstructs_curved_sloped_trajectory() -> None:
    num_poses = 4
    speed = 2.0
    yaw_rate = 0.25
    pitch = 0.1
    ground_yaw = 0.4
    ground_translation = np.array([3.0, -2.0, 0.5], dtype=np.float32)
    ground_rotation = _rotation_z(ground_yaw)

    yaws = -0.2 + yaw_rate * np.arange(num_poses, dtype=np.float32)
    xy = np.zeros((num_poses, 2), dtype=np.float32)
    heights = np.zeros(num_poses, dtype=np.float32)
    arc_displacement = speed * np.sinc(yaw_rate / (2.0 * np.pi))
    for pose_idx in range(num_poses - 1):
        yaw_mid = yaws[pose_idx] + 0.5 * yaw_rate
        xy[pose_idx + 1] = xy[pose_idx] + arc_displacement * np.array(
            [np.cos(yaw_mid), np.sin(yaw_mid)], dtype=np.float32
        )
        heights[pose_idx + 1] = heights[pose_idx] + speed * np.tan(pitch)

    expected_poses = []
    for current_xy, current_height, current_yaw in zip(xy, heights, yaws, strict=True):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = ground_rotation @ _rotation_z(float(current_yaw)) @ _rotation_y(-pitch)
        pose[:3, 3] = ground_translation + ground_rotation @ np.array(
            [current_xy[0], current_xy[1], current_height], dtype=np.float32
        )
        expected_poses.append(pose)
    expected_poses_np = np.stack(expected_poses)

    tracks = CuboidTracks.Factory.from_numpy(
        tracks_id=["track"],
        tracks_poses=[expected_poses_np],
        tracks_timestamps_us=[np.arange(num_poses, dtype=np.int64) * 1_000_000],
        tracks_label_class=["unknown"],
        tracks_flags=[TrackFlags.NONE],
        cuboids_dims=[np.ones(3, dtype=np.float32)],
        device=torch.device("cpu"),
    )
    tracks_calib = _unicycle_tracks_calib(tracks, fix_first_pose=False, fix_last_pose=False)

    # Install the exact ground-local oracle state instead of relying on chord-speed initialization.
    with torch.no_grad():
        tracks_calib.ground_t.copy_(torch.from_numpy(ground_translation).view(1, 3))
        tracks_calib.ground_q.copy_(torch.tensor([[0.0, 0.0, np.sin(ground_yaw / 2.0), np.cos(ground_yaw / 2.0)]]))
        tracks_calib.Tclc_tquat.copy_(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
        tracks_calib.xy.copy_(torch.from_numpy(xy))
        tracks_calib.height.copy_(torch.from_numpy(heights))
        tracks_calib.theta.copy_(torch.from_numpy(yaws))
        tracks_calib.pitch.fill_(pitch)
        tracks_calib.velocity.fill_(speed)
        tracks_calib.theta_velocity.fill_(yaw_rate)
        tracks_calib.pitch_velocity.zero_()
        tracks_calib.initial_pose_xy.copy_(tracks_calib.xy)
        tracks_calib.initial_pose_height.copy_(tracks_calib.height)

    forward_input_tracks = CuboidTracks.Factory.from_numpy(
        tracks_id=["track"],
        tracks_poses=[np.repeat(np.eye(4, dtype=np.float32)[None], num_poses, axis=0)],
        tracks_timestamps_us=[np.arange(num_poses, dtype=np.int64) * 1_000_000],
        tracks_label_class=["unknown"],
        tracks_flags=[TrackFlags.NONE],
        cuboids_dims=[np.ones(3, dtype=np.float32)],
        device=torch.device("cpu"),
    )
    calibrated_tracks = tracks_calib(forward_input_tracks)
    torch.testing.assert_close(
        calibrated_tracks.tracks_poses.matrix(),
        torch.from_numpy(expected_poses_np),
        rtol=1e-5,
        atol=1e-5,
    )

    consistency = tracks_calib.get_consistency()
    smoothness = tracks_calib.get_smoothness()
    pose_regularization = tracks_calib.get_pose_regularization()
    assert consistency.shape == (num_poses - 1, 5)
    assert smoothness.shape == (num_poses - 3, 4)
    assert pose_regularization.shape == (num_poses,)
    for residuals in (consistency, smoothness, pose_regularization):
        assert torch.isfinite(residuals).all()
        torch.testing.assert_close(residuals, torch.zeros_like(residuals), rtol=0, atol=1e-5)

    with torch.no_grad():
        tracks_calib.velocity[1] += 0.5
        tracks_calib.xy[1, 0] += 0.2

    torch.testing.assert_close(
        tracks_calib.get_smoothness(),
        torch.tensor([[-1.0, 0.0, 0.0, 0.0]]),
        rtol=0,
        atol=1e-5,
    )
    torch.testing.assert_close(
        tracks_calib.get_pose_regularization(),
        torch.tensor([0.0, 0.04, 0.0, 0.0]),
        rtol=0,
        atol=1e-6,
    )


def test_unicycle_tracks_calib_rejects_duplicate_timestamps() -> None:
    tracks = _tracks_for_static_calib(torch.device("cpu"))
    tracks.tracks_timestamps_us[1] = tracks.tracks_timestamps_us[0]

    with pytest.raises(ValueError, match="track timestamps must be strictly increasing"):
        _unicycle_tracks_calib(tracks, fix_first_pose=True, fix_last_pose=True)


def test_unwrap_angle_sequence_accepts_single_angle() -> None:
    angles = torch.tensor([0.25])

    unwrapped_angles, angle_deltas = ElevationAwareUnicycleTracksCalib._unwrap_angle_sequence(angles)

    torch.testing.assert_close(unwrapped_angles, angles)
    assert angle_deltas.shape == (0,)


def test_unicycle_tracks_calib_returns_empty_smoothness_for_short_track() -> None:
    tracks_calib = _unicycle_tracks_calib(
        _tracks_for_unicycle_calib(),
        fix_first_pose=False,
        fix_last_pose=False,
    )

    smoothness = tracks_calib.get_smoothness()

    assert smoothness.shape == (0, 4)
    assert smoothness.dtype == tracks_calib.xy.dtype
    assert smoothness.device == tracks_calib.xy.device


def test_unicycle_tracks_calib_update_step_toggles_training_and_visualizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracks = _tracks_for_unicycle_calib()
    tracks_calib = _unicycle_tracks_calib(tracks, fix_first_pose=False, fix_last_pose=False)
    tracks_calib.config.start_global_step = 2

    tracks_calib.update_step_train_batch_start(epoch=0, global_step=1, system=None)
    assert all(not parameter.requires_grad for parameter in tracks_calib.parameters())

    optimized_tracks = MagicMock()
    forward = MagicMock(return_value=optimized_tracks)
    visualize_unicycle = MagicMock()
    visualdebugger = MagicMock()
    monkeypatch.setattr(tracks_calib, "forward", forward)
    monkeypatch.setattr(tracks_calib, "visualize_unicycle", visualize_unicycle)
    monkeypatch.setattr(tracks_calib_module, "get_visualdebugger", MagicMock(return_value=visualdebugger))
    tracks_calib.vis_data.current = tracks

    result = tracks_calib.update_step_train_batch_start(epoch=0, global_step=2, system=None)

    assert result == {}
    assert all(parameter.requires_grad for parameter in tracks_calib.parameters())
    forward.assert_called_once_with(tracks)
    optimized_tracks.visualize.assert_called_once_with(1, show=False)
    visualize_unicycle.assert_called_once_with()
    visualdebugger.update.assert_called_once_with()


def test_unicycle_tracks_calib_visualizes_and_redraws_single_track() -> None:
    tracks_calib = _unicycle_tracks_calib(
        _tracks_for_unicycle_calib(),
        fix_first_pose=False,
        fix_last_pose=False,
    )

    try:
        tracks_calib.visualize_unicycle()
        tracks_calib.visualize_unicycle()

        assert len(tracks_calib.all_scatter_points) == 1
        assert len(tracks_calib.all_quiver_arrows) == 1
        assert len(tracks_calib.all_connections) == 1
    finally:
        if hasattr(tracks_calib, "figure"):
            tracks_calib_module.plt.close(tracks_calib.figure)


def test_unicycle_pose_matches_lietorch_composition() -> None:
    tracks = _two_tracks_for_unicycle_calib(num_poses=8)
    calib = _unicycle_tracks_calib(tracks, fix_first_pose=False, fix_last_pose=False)
    calib.requires_grad_(True)
    n = calib.xy.shape[0]
    # Every channel and every frame gets a non-trivial value, so a sign or order slip in any
    # part of the composition shows up in the comparison.
    with torch.no_grad():
        calib.xy.add_(torch.linspace(-0.5, 0.5, n).view(-1, 1) * torch.tensor([[1.0, -0.5]]))
        calib.height.add_(torch.linspace(0.1, 0.4, n))
        calib.theta.add_(torch.linspace(-0.3, 0.7, n))
        calib.pitch.add_(torch.linspace(-0.2, 0.2, n))
        calib.ground_t.add_(torch.tensor([[0.5, -1.0, 2.0]]))
        calib.ground_q.add_(torch.tensor([[0.1, -0.2, 0.15, 0.0]]))
        calib.Tclc_tquat.add_(torch.tensor([[0.1, -0.2, 0.3, 0.05, -0.1, 0.15, 0.02]]))

    got = calib.get_pose().vec()

    current_xy, current_height, current_theta, current_pitch = calib._get_pose_params()
    world_ground_t = calib.ground_t[calib.tracks_idx]
    world_ground_q = calib._normalize_quat_xyzw(calib.ground_q[calib.tracks_idx])
    ground_cuboid_t = torch.cat([current_xy, current_height.view(-1, 1)], dim=1)
    ground_cuboid_q = calib._get_yaw_pitch_quat(current_theta, current_pitch)
    tclc = calib.Tclc_tquat.expand(n, -1)
    expected = (
        lt.SE3.InitFromVec(torch.cat([world_ground_t, world_ground_q], dim=1))
        * lt.SE3.InitFromVec(torch.cat([ground_cuboid_t, ground_cuboid_q], dim=1))
        * lt.SE3.InitFromVec(torch.cat([tclc[:, :3], calib._normalize_quat_xyzw(tclc[:, 3:])], dim=1))
    ).vec()

    assert (got - expected).abs().max().item() < 1e-5

    # A generic cotangent exercises every gradient direction; a squared loss only reaches
    # the radial quaternion one, where any composition agrees.
    upstream = torch.randn(got.shape, generator=torch.Generator().manual_seed(99))
    channels = [calib.xy, calib.height, calib.theta, calib.pitch]
    grads_new = torch.autograd.grad((got * upstream).sum(), channels, retain_graph=True)
    grads_ref = torch.autograd.grad((expected * upstream).sum(), channels)
    for name, new, reference in zip(("xy", "height", "theta", "pitch"), grads_new, grads_ref, strict=True):
        assert (new - reference).abs().max().item() < 1e-4, name


def test_unicycle_step_cache_shares_and_invalidates() -> None:
    tracks = _tracks_for_unicycle_calib(num_poses=8)
    calib = ElevationAwareUnicycleTracksCalib(
        config=OmegaConf.create(
            {
                "visualize": False,
                "start_global_step": 0,
                "fix_first_pose": False,
                "fix_last_pose": False,
                "pitch_slope_clamp_rad": 1.2,
                "cuboid_tracks": {"up_axis": [0.0, 0.0, 1.0], "forward_axis": [1.0, 0.0, 0.0]},
            }
        ),
        trainer_config=_trainer_config(),
        cuboid_tracks=tracks,
    )
    calib.update_step_train_batch_start(epoch=0, global_step=10, system=None)

    # Two collections in one step read the same parameters, so they must receive the very same
    # tensors and accumulate gradients through one shared graph.
    first = calib(tracks).tracks_poses
    second = calib(tracks).tracks_poses
    assert first.data is second.data

    # A backward spends the shared graph; the next read must rebuild.
    (first.vec().sum() + second.vec().sum()).backward()
    assert calib.xy.grad is not None
    rebuilt = calib(tracks).tracks_poses
    assert rebuilt.data is not first.data
    rebuilt.vec().sum().backward()

    # A backward through only the pose losses also spends the graph the pose shares with
    # them; the next pose read must rebuild rather than replay the freed graph.
    shared = calib(tracks).tracks_poses
    (calib.get_consistency().square().sum() + calib.get_pose_regularization().sum()).backward()
    after_losses = calib(tracks).tracks_poses
    assert after_losses.data is not shared.data
    after_losses.vec().sum().backward()

    # An in-place parameter update (what an optimizer step is) must show up in the next read.
    with torch.no_grad():
        before = calib(tracks).tracks_poses.vec().clone()
        calib.xy.add_(1.0)
        after = calib(tracks).tracks_poses.vec()
    assert (after - before).abs().max().item() > 0.1


def test_unicycle_collector_cache_bridges_share_and_invalidate() -> None:
    calib = _unicycle_tracks_calib(
        _tracks_for_unicycle_calib(num_poses=8),
        fix_first_pose=False,
        fix_last_pose=False,
    )
    num_poses = calib.xy.shape[0]

    def make_pose_vec() -> torch.Tensor:
        identity_q = torch.zeros((num_poses, 4), dtype=calib.xy.dtype)
        identity_q[:, 3] = 1.0
        return torch.cat((calib.xy, calib.height[:, None], identity_q), dim=1)

    pose_compute = MagicMock(side_effect=make_pose_vec)
    residual_compute = MagicMock(
        side_effect=lambda: TracksCalibResiduals(
            consistency=calib.xy.sum(dim=1),
            smoothness=2.0 * calib.height,
            pose_regularization=calib.theta + calib.pitch,
        )
    )

    # Every collector call in one training graph must reuse both the tensors and the
    # computation that produced them.
    first_pose = calib._get_pose_from_collector(pose_compute)
    second_pose = calib._get_pose_from_collector(pose_compute)
    first_residuals = calib._get_residuals_from_collector(residual_compute)
    second_residuals = calib._get_residuals_from_collector(residual_compute)

    assert first_pose.data is second_pose.data
    assert all(
        left is right for left, right in zip(first_residuals.as_tuple(), second_residuals.as_tuple(), strict=True)
    )
    pose_compute.assert_called_once_with()
    residual_compute.assert_called_once_with()

    # Backward consumes the shared autograd graph, so the next collector call must
    # construct fresh outputs instead of serving tensors tied to the freed graph.
    (first_pose.vec().sum() + sum(value.sum() for value in first_residuals.as_tuple())).backward()
    calib._get_pose_from_collector(pose_compute)
    calib._get_residuals_from_collector(residual_compute)
    assert pose_compute.call_count == 2
    assert residual_compute.call_count == 2


def test_cuboid_tracks_factory_rejects_zero_pose_track() -> None:
    with pytest.raises(AssertionError, match="require at least two poses per track"):
        CuboidTracks.Factory.from_numpy(
            tracks_id=["track"],
            tracks_poses=[np.empty((0, 4, 4), dtype=np.float32)],
            tracks_timestamps_us=[np.empty((0,), dtype=np.int64)],
            tracks_label_class=["unknown"],
            tracks_flags=[TrackFlags.NONE],
            cuboids_dims=[np.array([1, 1, 1], dtype=np.float32)],
            device=torch.device("cpu"),
        )


class _CachingDirectCalib(DirectTracksCalib):
    """Opts into the base per-step cache to exercise its machinery end to end."""

    def _watched_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.tracks_delta_q, self.tracks_delta_t)

    def cached_delta(self) -> torch.Tensor:
        value = self._cache_lookup("delta")
        if value is None:
            value = torch.cat([self.tracks_delta_t, self.tracks_delta_q], dim=1)
            self._cache_store("delta", value, (value,))
        return value


def _caching_direct_calib() -> _CachingDirectCalib:
    return cast(_CachingDirectCalib, _direct_tracks_calib(_tracks_for_direct_calib(), calib_class=_CachingDirectCalib))


def test_step_cache_shares_within_a_step_and_follows_watched_versions() -> None:
    calib = _caching_direct_calib()

    first = calib.cached_delta()
    assert calib.cached_delta() is first

    with torch.no_grad():
        calib.tracks_delta_t.add_(1.0)  # a dispatcher-visible mutation bumps the version
    second = calib.cached_delta()
    assert second is not first
    torch.testing.assert_close(second[:, :3], torch.ones_like(second[:, :3]))


def test_step_cache_pops_entries_consumed_by_backward() -> None:
    calib = _caching_direct_calib()

    first = calib.cached_delta()
    first.sum().backward()

    second = calib.cached_delta()
    assert second is not first
    second.sum().backward()  # the replacement carries a live graph


def test_step_cache_keeps_an_entry_a_stale_hook_would_evict() -> None:
    calib = _caching_direct_calib()

    first = calib.cached_delta()
    with torch.no_grad():
        calib.tracks_delta_t.add_(1.0)  # supersede the entry; `first`'s hook is now stale
    second = calib.cached_delta()
    assert second is not first

    first.sum().backward()  # the stale hook fires and must leave `second` in place

    assert calib.cached_delta() is second


def test_step_cache_clears_at_train_batch_start() -> None:
    calib = _caching_direct_calib()

    first = calib.cached_delta()
    calib.tracks_delta_t.data.add_(1.0)  # fused-optimizer-shaped update: the version stays put
    assert calib.cached_delta() is first  # the token alone cannot see it

    calib.update_step_train_batch_start(epoch=0, global_step=1, system=MagicMock(logger_enabled=False))

    second = calib.cached_delta()
    assert second is not first
    torch.testing.assert_close(second[:, :3], torch.ones_like(second[:, :3]))


class _UncachedDirectCalib(DirectTracksCalib):
    """Watches nothing, so the base per-step cache stays unused."""

    def _watched_tensors(self) -> tuple[torch.Tensor, ...]:
        return ()


def test_step_cache_clear_skips_variants_without_a_cache() -> None:
    calib = _direct_tracks_calib(_tracks_for_direct_calib(), calib_class=_UncachedDirectCalib)
    # A variant watching nothing may have skipped this class's __init__, so the clear
    # must not assume a cache.
    del calib._pose_cache

    calib.update_step_train_batch_start(epoch=0, global_step=1, system=MagicMock(logger_enabled=False))


def test_step_cache_isolates_grad_no_grad_and_inference_readers() -> None:
    calib = _caching_direct_calib()

    graphed = calib.cached_delta()
    with torch.no_grad():
        plain = calib.cached_delta()
    assert plain is not graphed
    assert plain.grad_fn is None

    with torch.inference_mode():
        inference = calib.cached_delta()
    assert inference.is_inference()

    with torch.no_grad():
        after = calib.cached_delta()
    assert after is not inference
    assert not after.is_inference()


def test_step_cache_follows_requires_grad_flips() -> None:
    calib = _caching_direct_calib()

    first = calib.cached_delta()
    calib.tracks_delta_q.requires_grad_(False)
    calib.tracks_delta_t.requires_grad_(False)

    second = calib.cached_delta()
    assert second is not first
    assert second.grad_fn is None


def test_step_cache_follows_data_swapping_conversions() -> None:
    calib = _caching_direct_calib()

    first = calib.cached_delta()
    # nn.Module conversions swap `param.data`, which keeps both id and version.
    calib.double()
    second = calib.cached_delta()
    assert second is not first
    assert second.dtype == torch.float64

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to check that the device element follows a module move")

    calib.to("cuda")
    third = calib.cached_delta()
    assert third is not second
    assert third.is_cuda


def test_step_cache_declines_to_serve_autocast_readers() -> None:
    calib = _caching_direct_calib()

    plain = calib.cached_delta()
    # Setting the flag directly keeps this leg meaningful without a GPU: entering
    # `torch.autocast("cuda", ...)` there warns and disables, leaving the flag False.
    prev_cuda_autocast = torch.is_autocast_enabled("cuda")
    torch.set_autocast_enabled("cuda", True)
    try:
        cuda_autocast = calib.cached_delta()
    finally:
        torch.set_autocast_enabled("cuda", prev_cuda_autocast)
    assert cuda_autocast is not plain

    with torch.autocast("cpu", dtype=torch.bfloat16):
        cpu_autocast = calib.cached_delta()
    assert cpu_autocast is not plain

    # Neither autocast read stored anything, so the plain entry is still the one served.
    assert calib.cached_delta() is plain


def test_step_cache_separates_cuda_stream_readers() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to read the cache from a second stream")

    calib = _caching_direct_calib().cuda()
    default_stream = calib.cached_delta()
    with torch.cuda.stream(torch.cuda.Stream()):
        side_stream = calib.cached_delta()
    assert side_stream is not default_stream


class _RowViewCachingCalib(DirectTracksCalib):
    """Watches a row view, whose temporary is freed as soon as the token is built."""

    row = 0

    def _watched_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.tracks_delta_t[self.row],)

    def cached_row(self) -> torch.Tensor:
        value = self._cache_lookup("row")
        if value is None:
            value = self.tracks_delta_t[self.row] * 1.0
            self._cache_store("row", value, (value,))
        return value


def test_step_cache_does_not_serve_a_recycled_watched_id() -> None:
    calib = cast(
        _RowViewCachingCalib, _direct_tracks_calib(_tracks_for_direct_calib(), calib_class=_RowViewCachingCalib)
    )
    with torch.no_grad():
        calib.tracks_delta_t[1].add_(5.0)

    first = calib.cached_row()
    calib.row = 1
    second = calib.cached_row()

    assert second is not first
    torch.testing.assert_close(second, calib.tracks_delta_t[1] * 1.0)


def test_unicycle_update_step_clears_the_step_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    tracks = _tracks_for_unicycle_calib()
    calib = _unicycle_tracks_calib(tracks, fix_first_pose=False, fix_last_pose=False)
    # Keep the hook's requires_grad toggle a no-op so the token stays put and only the
    # step-boundary clear can invalidate the entry.
    calib.config.start_global_step = 0
    monkeypatch.setattr(calib, "_watched_tensors", lambda: (calib.xy,))
    value = calib.xy * 2.0
    calib._cache_store("probe", value, (value,))
    assert calib._cache_lookup("probe") is value

    calib.update_step_train_batch_start(epoch=0, global_step=1, system=None)

    assert calib._cache_lookup("probe") is None
