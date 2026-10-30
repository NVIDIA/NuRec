# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Type, cast

import lietorch as lt
import matplotlib.pyplot as plt
import numpy as np
import torch

from lietorch.group_ops import lietorch_backends
from matplotlib import collections as mc
from omegaconf import DictConfig
from torch import nn

from nurec.config.trainer import TrainerConfig
from nurec.datasets.tracks import CuboidTracks
from nurec.models.base import BaseModel
from nurec.models.nn_extensions import module_call_type
from nurec.models.tracks_calib_data import ElevationAwareUnicycleTracksCalibData, TracksCalibResiduals
from nurec.utils.geometry import quat_mult_xyzw, quat_rotate_xyzw, quat_to_so3_matrix, so3_matrix_to_quat
from nurec.utils.optim import OptimizerLRSchedulerConfig, configure_optimizers
from nurec.utils.trainer import adjust_step_for_world_size
from nurec.visualdebugger import get_visualdebugger


log = logging.getLogger(__name__)


def make_row_grad_hook(trainable_rows: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Gradient hook zeroing the leading-dim rows where ``trainable_rows`` is False.

    A single ``nn.Parameter`` cannot carry per-row ``requires_grad``, so the
    frozen partition's gradient rows are zeroed instead. ``trainable_rows``
    is moved to the grad's device lazily so the hook works regardless of
    where the mask was built.
    """

    def hook(grad: torch.Tensor) -> torch.Tensor:
        keep = trainable_rows.to(grad.device)
        return grad * keep.view(-1, *([1] * (grad.ndim - 1))).to(grad.dtype)

    return hook


class _PlainVecToSE3Grad(torch.autograd.Function):
    """Identity on an SE3 vec; the backward supplies lietorch's missing entry conversion.

    Lietorch keeps gradients between its ops in a tangent convention: reading a pose out
    (``vec()``) multiplies the gradient by the group's tangent projector, and building one
    (``InitFromVec``) multiplies by that projector's pseudo-inverse. A vec composed with
    plain tensor ops and handed to ``lt.SE3`` skips the build-side conversion, so the
    read-side one arrives unpaired and corrupts the gradient. The backward here applies
    the pseudo-inverse in closed form. The first six columns of the 7-by-7 projector split
    as ``[[I, A], [0, B]]`` (translation rows, rotation columns); its seventh column is
    zero. For a unit quaternion, ``B^T B = I/4``, so the block solve collapses to two small
    matrix products.
    """

    @staticmethod
    def forward(ctx, vec: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(vec)
        return vec

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> torch.Tensor:
        (vec,) = ctx.saved_tensors
        grad = grad_outputs[0].contiguous()
        projector = lietorch_backends.projector(lt.SE3.group_id, vec.detach().contiguous())
        coupling = projector[:, :3, 3:6]
        rotation = projector[:, 3:7, 3:6]
        grad_t = grad[:, :3]
        omega = grad[:, 3:6].unsqueeze(-1) - coupling.transpose(1, 2) @ grad_t.unsqueeze(-1)
        return torch.cat([grad_t, 4.0 * (rotation @ omega).squeeze(-1)], dim=1)


def se3_from_plain_vec(vec: torch.Tensor) -> lt.SE3:
    """An ``lt.SE3`` around a plainly-composed (n, 7) vec, gradient convention included.

    The vec's quaternion must be unit-norm: the backward's closed-form conversion holds
    only on the unit sphere, and off it the rotation gradient is silently wrong.
    """
    return lt.SE3(_PlainVecToSE3Grad.apply(vec))


class BaseTracksCalib(BaseModel):
    def freeze_tracks(self, frozen_track_indices: Sequence[int], tracks_packinfo: torch.Tensor) -> None:
        """Stop gradients for the given tracks' calibration parameters.

        Generic across variants: every trainable parameter is classified by
        its leading dimension against three indexings derived from
        ``tracks_packinfo`` and hooked with the matching frozen mask:

        * ``n_poses`` (total keyframe poses) — pose-indexed params
          (``DirectTracksCalib.tracks_delta_*``, ``UnicycleTracksCalib.xy``/
          ``theta``). A frozen track's contiguous pose range is zeroed.
        * ``n_tracks`` — track-indexed params.
        * ``n_poses - n_tracks`` — inter-pose params
          (``UnicycleTracksCalib.velocity``/``omega`` carry ``count - 1``
          entries per track, contiguous in track order).

        ``frozen_track_indices`` are indices into ``tracks_packinfo`` (the
        node's track subset). Variants with no optimizable params
        (Static/Skip) are a no-op. A trainable param whose leading dim
        matches none of the three is logged loudly rather than silently
        left trainable.
        """
        frozen = [int(t) for t in frozen_track_indices]
        if not frozen:
            return
        packinfo = tracks_packinfo.cpu()
        n_tracks = int(packinfo.shape[0])
        starts = [int(packinfo[t, 0]) for t in range(n_tracks)]
        counts = [int(packinfo[t, 1]) for t in range(n_tracks)]
        n_poses = sum(counts)
        n_inter = n_poses - n_tracks
        frozen_set = set(frozen)

        pose_trainable = torch.ones(n_poses, dtype=torch.bool)
        track_trainable = torch.ones(n_tracks, dtype=torch.bool)
        inter_trainable = torch.ones(max(n_inter, 0), dtype=torch.bool)
        inter_cursor = 0
        for t in range(n_tracks):
            inter_count = counts[t] - 1
            if t in frozen_set:
                pose_trainable[starts[t] : starts[t] + counts[t]] = False
                track_trainable[t] = False
                inter_trainable[inter_cursor : inter_cursor + inter_count] = False
            inter_cursor += inter_count

        masks: dict[int, torch.Tensor] = {n_poses: pose_trainable, n_tracks: track_trainable, n_inter: inter_trainable}
        for name, param in self.named_parameters(recurse=True):
            if not param.requires_grad or isinstance(param, torch.nn.parameter.UninitializedParameter):
                continue
            mask = masks.get(int(param.shape[0]))
            if mask is not None:
                param.register_hook(make_row_grad_hook(mask))
            else:
                log.warning(
                    "%s.freeze_tracks: parameter '%s' with leading dim %d is neither pose- (%d), track- (%d) "
                    "nor inter-pose-indexed (%d); frozen tracks may still update it",
                    type(self).__name__,
                    name,
                    int(param.shape[0]),
                    n_poses,
                    n_tracks,
                    n_inter,
                )

    @staticmethod
    def factory(
        name: str, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks
    ) -> BaseTracksCalib:
        variants: dict[str, Type[BaseTracksCalib]] = {
            "direct-tracks-calib": DirectTracksCalib,
            "static-tracks-calib": StaticTracksCalib,
            "elevation-aware-unicycle-tracks-calib": ElevationAwareUnicycleTracksCalib,
            "skip-tracks-calib": SkipTracksCalib,
        }
        variant_class = variants[name]
        return variant_class(config, trainer_config, cuboid_tracks)

    @dataclass(slots=False, kw_only=True)
    class CuboidTracksVisualizeData:
        """
        Data stored for visualizing cuboid tracks calibration.
        """

        init: Optional[CuboidTracks] = None
        current: Optional[CuboidTracks] = None

    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config)
        self.trainer_config = trainer_config

        self.visualize: bool = self.config.visualize
        # Volatile cache the cuboid tracks just for visualization
        self.vis_data = BaseTracksCalib.CuboidTracksVisualizeData()

        # Evaluations shared within a training step: name -> (token, value, watched tensors).
        self._pose_cache: dict[str, tuple[tuple, Any, tuple[torch.Tensor, ...]]] = {}

        if hasattr(self.config, "start_global_step"):
            self.config.start_global_step = adjust_step_for_world_size(trainer_config, self.config.start_global_step)
            log.info(f"BaseTracksCalib: start_global_step={self.config.start_global_step}")

    def _watched_tensors(self) -> tuple[torch.Tensor, ...]:
        """The tensors the per-step cache token follows.

        - a variant listing none caches nothing
        - a variant listing any must have run this class's __init__, which creates the cache
        - the cache holds what this returns for the step, so returning fresh views is safe
        """
        return ()

    def _pose_cache_token(self, watched: tuple[torch.Tensor, ...]) -> tuple:
        """What the cached evaluations depend on, cheap enough to compare on every read."""
        # The ambient state a served value would be wrong under:
        # - grad mode: a graph-free value must not be served to a caller building a graph
        # - inference mode: an inference tensor poisons any later autograd use
        # - cuda stream: a value produced on one stream is unfenced for a reader on another
        parts: list = [
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            torch.cuda.current_stream() if torch.cuda.is_available() else None,
        ]
        for tensor in watched:
            # device and dtype catch the `.data` swaps that keep both id and version
            # (module dtype conversions and device moves).
            parts.extend((id(tensor), tensor._version, tensor.requires_grad, tensor.device, tensor.dtype))
        return tuple(parts)

    @staticmethod
    def _cache_disabled() -> bool:
        """Ambient modes the cache declines to serve rather than key on.

        - autocast selects a dtype the token does not record
        - declining to share is cheaper than widening the token to carry it
        """
        return torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")

    def _cache_lookup(self, key: str) -> Any:
        watched = self._watched_tensors()
        if not watched or self._cache_disabled():
            return None
        entry = self._pose_cache.get(key)
        if entry is not None and entry[0] == self._pose_cache_token(watched):
            return entry[1]
        return None

    def _cache_store(self, key: str, value: Any, guards: tuple[torch.Tensor, ...]) -> None:
        """Keep a value for the rest of the step, with its autograd lifetime respected.

        A backward frees the graph it ran through, so an entry must not outlive the
        backward that consumed it:
        - a hook on each guard drops the entry as the gradient flows through it
        - a guard that does not require grad gets no hook
        - a value carrying a graph therefore needs at least one guard that does
        - the hook checks the entry is still the one it was registered for
        - a stale hook from a superseded entry therefore leaves the current entry alone
        - guards must be step-scoped tensors from the value's own graph, because the
          hooks are never removed and a long-lived leaf would accumulate one per store

        Readers of one key within a step share a single graph:
        - a step's losses must fold into one backward
        - a second independent backward through a shared value raises, because the first
          already freed the graph and popped the entry
        """
        watched = self._watched_tensors()
        if not watched or self._cache_disabled():
            return
        token = self._pose_cache_token(watched)
        self._pose_cache[key] = (token, value, watched)
        cache = self._pose_cache

        def clear(_grad: torch.Tensor, token: tuple = token) -> None:
            entry = cache.get(key)
            if entry is not None and entry[0] is token:
                cache.pop(key, None)

        for guard in guards:
            if guard.requires_grad:
                guard.register_hook(clear)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        # A fused optimizer writes parameters through `.data`, which no version counter
        # records, so the token cannot see the update: drop every entry at the boundary.
        if self._watched_tensors():
            self._pose_cache.clear()
        return {}

    def save_tracks_for_visualize(self, tracks: CuboidTracks) -> None:
        if self.visualize:
            self.vis_data.current = tracks

            if self.vis_data.init is None:
                self.vis_data.init = CuboidTracks.Ops.clone(tracks)

    @abstractmethod
    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks: ...

    __call__ = module_call_type(forward)


class DirectTracksCalib(BaseTracksCalib):
    """Applies configurable world- or local-frame 6-DoF pose deltas."""

    gradient_mask: nn.Buffer
    delta_frame: str
    left_multiply: bool

    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config, trainer_config, cuboid_tracks)

        self.delta_frame = str(self.config.get("delta_frame", "world"))  # fallback for old checkpoints
        if self.delta_frame not in {"world", "local"}:
            raise ValueError(
                f"{self.__class__.__name__}: delta_frame must be 'world' or 'local', got {self.delta_frame!r}"
            )
        self.left_multiply = self.delta_frame == "world"
        log.info(f"{self.__class__.__name__}: delta_frame={self.delta_frame}")

        # Called during class initialization
        assert self.config.start_global_step >= 0, (
            f"{self.__class__.__name__}: require positive global start step or zero"
        )

        n_tracks_poses: int = cuboid_tracks.tracks_poses.shape[0]

        self.tracks_delta_q = nn.Parameter(
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32).repeat(n_tracks_poses, 1)
        )
        self.tracks_delta_t = nn.Parameter(torch.zeros(n_tracks_poses, 3, device=self.device, dtype=torch.float32))

        self.gradient_mask = nn.Buffer(
            torch.ones(n_tracks_poses, device=self.device, dtype=torch.bool), persistent=False
        )
        for pose_start, pose_count in cuboid_tracks.tracks_packinfo.cpu().numpy():
            if self.config.fix_first_pose:
                self.gradient_mask[pose_start] = False
            if self.config.fix_last_pose:
                self.gradient_mask[pose_start + pose_count - 1] = False

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        super().update_step_train_batch_start(epoch, global_step, system, **kwargs)
        if global_step >= self.config.start_global_step:
            # unfreeze parameters if estimated
            self.tracks_delta_q.requires_grad_(True)
            self.tracks_delta_t.requires_grad_(True)
        else:
            self.tracks_delta_q.requires_grad_(False)
            self.tracks_delta_t.requires_grad_(False)

        with torch.no_grad():
            if getattr(system, "logger_enabled", True):
                # Logging
                LOG_TO_PROG_BAR = self.config.log_to_prog_bar

                delta_pose = self.get_tracks_delta_transform()
                delta_pose_vec = delta_pose.vec()
                avg_delta_t = delta_pose_vec[:, :3].mean(dim=0).norm()
                avg_delta_angle = 2 * torch.acos(delta_pose_vec[:, 6:].mean(dim=0).norm()) * 180 / np.pi

                system.log_dict(
                    {
                        "tracks_calib/delta_t": avg_delta_t,
                        "tracks_calib/delta_angle": avg_delta_angle,
                    },
                    prog_bar=LOG_TO_PROG_BAR,
                )

            # Visualize the optimized tracks
            if self.visualize and self.vis_data.current is not None:
                optimized_tracks = self.forward(self.vis_data.current)
                optimized_tracks.visualize(1, show=False)
                get_visualdebugger().update()

        return {}  # no new / additional parameters

    def get_tracks_delta_transform(self) -> lt.SE3:
        if self.gradient_mask.any():
            current_tracks_delta_q = torch.where(
                self.gradient_mask.view(-1, 1), self.tracks_delta_q, self.tracks_delta_q.detach()
            )
            current_tracks_delta_t = torch.where(
                self.gradient_mask.view(-1, 1), self.tracks_delta_t, self.tracks_delta_t.detach()
            )
        else:
            current_tracks_delta_q = self.tracks_delta_q
            current_tracks_delta_t = self.tracks_delta_t

        return lt.SE3.InitFromVec(
            torch.cat(
                [current_tracks_delta_t, current_tracks_delta_q / current_tracks_delta_q.norm(dim=1, keepdim=True)],
                dim=1,
            )
        )

    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks:
        self.save_tracks_for_visualize(tracks)

        current_tracks_delta_transform = self.get_tracks_delta_transform()
        tracks = CuboidTracks.Ops.transform_with_delta_poses(
            tracks,
            current_tracks_delta_transform,
            left_multiply=self.left_multiply,
        )

        return tracks

    __call__ = module_call_type(forward)

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config, self.trainer_config, self, name_prefix)


class StaticTracksCalib(BaseTracksCalib):
    """
    Replace the object poses with a mean pose that is obtained during initialization
    Since it makes no sense to optimize static poses via rendering loss, the model does not contain optimizable parameters
    """

    static_poses_vec: nn.Buffer  # (n_tracks, 7) - mean pose per track as [translation, quaternion]
    min_timestamp_us: int
    max_timestamp_us: int

    TIMESTAMP_EXTEND_US: int = 1_000_000_000  # 1000s of extension

    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config, trainer_config, cuboid_tracks)

        pose_starts, pose_counts = cuboid_tracks.tracks_packinfo[:, 0], cuboid_tracks.tracks_packinfo[:, 1]
        track_pose_starts = [int(x) for x in pose_starts.tolist()]
        track_pose_counts = [int(x) for x in pose_counts.tolist()]
        assert track_pose_counts, "StaticTracksCalib requires at least one track"
        assert all(count > 0 for count in track_pose_counts), "StaticTracksCalib requires at least one pose per track"

        # Compute the mean pose for each track.
        # This is an iterative algorithm taken from https://ethaneade.com/lie.pdf.
        # TODO[JH]: consider exposing the SE3-sample mean computation as a utility function (potentially with extensions to other moments)
        pose_rotation = lt.SO3.InitFromVec(cuboid_tracks.tracks_poses.vec()[:, 3:])
        pose_translation = cuboid_tracks.tracks_poses.vec()[:, :3]

        # Use the first pose as the initial value
        mu_q = pose_rotation[pose_starts]
        for _ in range(4):
            inv_mu_q = mu_q.inv()
            mean_logs: list[torch.Tensor] = []
            for track_idx, (start, count) in enumerate(zip(track_pose_starts, track_pose_counts, strict=True)):
                track_rotations = pose_rotation[start : start + count]
                # Keep a batch dimension for lietorch broadcasting rules.
                track_log = (inv_mu_q[track_idx : track_idx + 1] * track_rotations).log().mean(dim=0)
                mean_logs.append(track_log)
            mu_q = mu_q * lt.SO3.exp(torch.stack(mean_logs, dim=0))

        mu_t = torch.stack(
            [
                pose_translation[start : start + count].mean(dim=0)
                for start, count in zip(track_pose_starts, track_pose_counts, strict=True)
            ],
            dim=0,
        )

        # Held as a buffer (not a plain lietorch attribute) so nn.Module.to() moves it.
        self.static_poses_vec = nn.Buffer(torch.cat([mu_t, mu_q.vec()], dim=1), persistent=False)

        self.min_timestamp_us = cast(int, cuboid_tracks.tracks_timestamps_us.min().item()) - self.TIMESTAMP_EXTEND_US
        self.max_timestamp_us = cast(int, cuboid_tracks.tracks_timestamps_us.max().item()) + self.TIMESTAMP_EXTEND_US

    @property
    def static_poses(self) -> lt.SE3:
        return lt.SE3(self.static_poses_vec)

    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks:
        assert self.static_poses.shape[0] == tracks.n_tracks, "Mismatch in number of tracks"
        return CuboidTracks.Ops.freeze(
            tracks,
            torch.full((tracks.n_tracks,), self.min_timestamp_us, device=tracks.device, dtype=torch.int64),
            torch.full((tracks.n_tracks,), self.max_timestamp_us, device=tracks.device, dtype=torch.int64),
            self.static_poses,
        )


class ElevationAwareUnicycleTracksCalib(BaseTracksCalib):
    """
    Approximate vehicle motion with a non-planar extension of the unicycle model.

    Each track uses a fixed local ground frame. Planar x/y/yaw motion is regularized by unicycle
    dynamics, while per-pose height and pitch model elevation changes and slopes.

    The planar dynamics and regularization structure are adapted from HUGS:
    https://arxiv.org/abs/2403.12722. This implementation extends them with
    pitch dynamics, height–pitch consistency, and vertical/pitch smoothness.

    The learned pose state is x/y/height/yaw/pitch.
    The motion variables are horizontal speed, and yaw/pitch angular velocity.

    The associated unicycle-pose loss uses three regularization terms:
    - Consistency: Ensures adjacent pose changes agree with the learned motion variables.
    - Smoothness: Penalizes changes in horizontal, yaw angular, pitch angular, and vertical acceleration.
    - Pose regularization: Keeps learned x/y/height positions close to their initialization.
    """

    # Pose state and its initialization used for regularization.
    xy: nn.Parameter  # (n_track_poses, 2)              - xy position in the per-track ground frame
    height: nn.Parameter  # (n_track_poses, )           - z position in the per-track ground frame
    theta: nn.Parameter  # (n_track_poses, )            - yaw in radians
    pitch: nn.Parameter  # (n_track_poses, )            - nose-up pitch in radians
    initial_pose_xy: nn.Buffer  # (n_track_poses, 2)    - initial xy position
    initial_pose_height: nn.Buffer  # (n_track_poses, ) - initial z position

    # Motion variables.
    velocity: nn.Parameter  # (n_track_poses - n_tracks, )       - horizontal speed
    theta_velocity: nn.Parameter  # (n_track_poses - n_tracks, ) - yaw angular velocity
    pitch_velocity: nn.Parameter  # (n_track_poses - n_tracks, ) - pitch angular velocity

    # Time intervals used by the motion constraints.
    time_delta_s: nn.Buffer  # (n_track_poses - n_tracks, )  - time delta between poses
    time_delta_v_s: nn.Buffer  # (n_track_poses - n_tracks,) - time delta between velocities

    ground_q: nn.Buffer  # (n_tracks, 4) - ground plane orientation (z-up, y-forward, x-right)
    ground_t: nn.Buffer  # (n_tracks, 3) - ground plane center
    Tclc_tquat: nn.Buffer  # (1, 7) - T_cuboid_local_cuboid as [translation, quaternion]

    pose_pair_start_indices: nn.Buffer
    motion_triplet_start_indices: nn.Buffer
    tracks_idx: nn.Buffer
    pose_gradient_mask: nn.Buffer

    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config, trainer_config, cuboid_tracks)

        assert self.config.start_global_step >= -1, (
            f"{self.__class__.__name__}: require global start step >=0 or -1 (that never starts)"
        )

        self._track_ids = list(cuboid_tracks.tracks_id)

        # BaseModel.device is still CPU while __init__ runs, so allocations follow the input tracks.
        device = cuboid_tracks.device

        # T_cuboid_local_cuboid transforms from x-right/y-forward/z-up cuboid-local to the dataset cuboid frame.
        cuboid_x = np.asarray(self.config.cuboid_tracks.forward_axis, dtype=np.float32)
        cuboid_z = np.asarray(self.config.cuboid_tracks.up_axis, dtype=np.float32)
        cuboid_y = np.cross(cuboid_z, cuboid_x)
        cuboid_local_to_cuboid_rotation_np = np.stack([cuboid_x, cuboid_y, cuboid_z], axis=1)
        cuboid_local_to_cuboid_rotation = torch.from_numpy(cuboid_local_to_cuboid_rotation_np).float()
        cuboid_q = so3_matrix_to_quat(cuboid_local_to_cuboid_rotation)
        cuboid_q_inv = cuboid_q.clone()
        cuboid_q_inv[:3] = -cuboid_q_inv[:3]

        self.Tclc_tquat = nn.Buffer(
            torch.cat([torch.zeros(3, dtype=torch.float32), cuboid_q_inv]).to(device).unsqueeze(0),
            persistent=False,
        )

        tracks_xys: list[torch.Tensor] = []
        tracks_heights: list[torch.Tensor] = []
        tracks_thetas: list[torch.Tensor] = []
        tracks_pitches: list[torch.Tensor] = []

        tracks_velocities: list[torch.Tensor] = []
        tracks_theta_velocities: list[torch.Tensor] = []
        tracks_pitch_velocities: list[torch.Tensor] = []

        tracks_time_deltas: list[torch.Tensor] = []
        tracks_time_delta_vs: list[torch.Tensor] = []

        tracks_ground_qs: list[torch.Tensor] = []
        tracks_ground_ts: list[torch.Tensor] = []
        tracks_idx: list[torch.Tensor] = []
        tracks_pose_pair_start_indices: list[torch.Tensor] = []
        tracks_motion_triplet_start_indices: list[torch.Tensor] = []
        pose_gradient_mask = torch.ones(cuboid_tracks.tracks_poses.shape[0], device=device, dtype=torch.bool)

        current_x_idx: int = 0
        current_v_idx: int = 0

        for pose_start, pose_count in cuboid_tracks.tracks_packinfo.cpu().numpy():
            assert pose_count > 1, "Need at least 2 poses to enable unicycle model"
            if self.config.fix_first_pose:
                pose_gradient_mask[pose_start] = False
            if self.config.fix_last_pose:
                pose_gradient_mask[pose_start + pose_count - 1] = False

            all_poses = cuboid_tracks.tracks_poses[pose_start : pose_start + pose_count]
            all_timestamps_us = cuboid_tracks.tracks_timestamps_us[pose_start : pose_start + pose_count]

            time_delta_s = (all_timestamps_us[1:] - all_timestamps_us[:-1]).float() / 1e6
            if bool((time_delta_s <= 0).any().item()):
                track_idx = len(tracks_idx)
                track_id = self._track_ids[track_idx] if track_idx < len(self._track_ids) else str(track_idx)
                raise ValueError(
                    f"{self.__class__.__name__}: track timestamps must be strictly increasing: "
                    f"track_idx={track_idx} track_id={track_id} timestamps_us={all_timestamps_us.tolist()}"
                )
            self._warn_short_timestamp_gaps(
                all_timestamps_us=all_timestamps_us,
                time_delta_s=time_delta_s,
                track_idx=len(tracks_idx),
            )

            all_poses_tquat = all_poses.data
            positions = all_poses_tquat[:, :3]
            cuboid_rotations = quat_to_so3_matrix(all_poses_tquat[:, 3:], unbatch=False)
            cuboid_local_to_cuboid_rotation_current = cuboid_local_to_cuboid_rotation.to(
                device=all_poses_tquat.device,
                dtype=all_poses_tquat.dtype,
            )
            cuboid_local_rotations = cuboid_rotations @ cuboid_local_to_cuboid_rotation_current
            front_dirs = cuboid_local_rotations[:, :, 0]
            up_dirs = cuboid_local_rotations[:, :, 2]

            # Determine ground plane local coordinates.
            ground_z = torch.mean(up_dirs, dim=0)
            ground_z = ground_z / ground_z.norm()

            ground_center = torch.mean(positions, dim=0)
            ground_x = front_dirs[0]

            ground_y = torch.cross(ground_z, ground_x)
            ground_y = ground_y / ground_y.norm()

            ground_x = torch.cross(ground_y, ground_z)

            ground_rotation_matrix = torch.stack([ground_x, ground_y, ground_z], dim=1)
            tracks_ground_q = so3_matrix_to_quat(ground_rotation_matrix)

            # Initialize the pose state in local ground-frame coordinates.
            ground_inv_t = -ground_rotation_matrix.t().mv(ground_center)
            tracks_local_positions = positions.mm(ground_rotation_matrix) + ground_inv_t
            tracks_xy = tracks_local_positions[:, :2]
            tracks_height = tracks_local_positions[:, 2]

            local_front_dirs = front_dirs.mm(ground_rotation_matrix)
            local_front_dirs_xy_norm = local_front_dirs[:, :2].norm(dim=1)
            local_front_dirs_xy = local_front_dirs[:, :2] / local_front_dirs_xy_norm.view(-1, 1)
            tracks_theta = torch.atan2(local_front_dirs_xy[:, 1], local_front_dirs_xy[:, 0])
            tracks_theta, tracks_theta_delta = self._unwrap_angle_sequence(tracks_theta)
            tracks_pitch = torch.atan2(local_front_dirs[:, 2], local_front_dirs_xy_norm)

            # Initialize the motion variables from adjacent pose states.
            tracks_velocity = torch.norm(tracks_xy[1:] - tracks_xy[:-1], dim=1) / time_delta_s
            tracks_theta_velocity = tracks_theta_delta / time_delta_s
            tracks_pitch_velocity = (tracks_pitch[1:] - tracks_pitch[:-1]) / time_delta_s

            tracks_time_delta_v = (time_delta_s[1:] + time_delta_s[:-1]) / 2
            tracks_time_delta_v = torch.nn.functional.pad(tracks_time_delta_v, (0, 1), "constant", 0)

            tracks_xys.append(tracks_xy)
            tracks_heights.append(tracks_height)
            tracks_thetas.append(tracks_theta)
            tracks_pitches.append(tracks_pitch)

            tracks_velocities.append(tracks_velocity)
            tracks_theta_velocities.append(tracks_theta_velocity)
            tracks_pitch_velocities.append(tracks_pitch_velocity)

            tracks_time_deltas.append(time_delta_s)
            tracks_time_delta_vs.append(tracks_time_delta_v)

            tracks_ground_qs.append(tracks_ground_q.view(1, -1))
            tracks_ground_ts.append(ground_center.view(1, -1))

            tracks_idx.append(torch.full((pose_count,), len(tracks_idx), device=device).long())

            tracks_pose_pair_start_indices.append(torch.arange(pose_count - 1, device=device).long() + current_x_idx)
            if pose_count > 3:
                tracks_motion_triplet_start_indices.append(
                    torch.arange(pose_count - 3, device=device).long() + current_v_idx
                )

            assert current_x_idx == pose_start, "Mismatch in pose start"
            current_x_idx += pose_count
            current_v_idx += pose_count - 1

        # Pose state and its initialization used for regularization.
        self.xy = nn.Parameter(torch.cat(tracks_xys, dim=0))
        self.height = nn.Parameter(torch.cat(tracks_heights, dim=0))
        self.theta = nn.Parameter(torch.cat(tracks_thetas, dim=0))
        self.pitch = nn.Parameter(torch.cat(tracks_pitches, dim=0))
        self.initial_pose_xy = nn.Buffer(self.xy.detach().clone(), persistent=False)
        self.initial_pose_height = nn.Buffer(self.height.detach().clone(), persistent=False)

        # Motion variables.
        self.velocity = nn.Parameter(torch.cat(tracks_velocities, dim=0))
        self.theta_velocity = nn.Parameter(torch.cat(tracks_theta_velocities, dim=0))
        self.pitch_velocity = nn.Parameter(torch.cat(tracks_pitch_velocities, dim=0))

        # Time intervals used by the motion constraints.
        self.time_delta_s = nn.Buffer(torch.cat(tracks_time_deltas, dim=0))
        self.time_delta_v_s = nn.Buffer(torch.cat(tracks_time_delta_vs, dim=0))

        self.ground_q = nn.Buffer(torch.cat(tracks_ground_qs, dim=0))
        self.ground_t = nn.Buffer(torch.cat(tracks_ground_ts, dim=0))
        self.tracks_idx = nn.Buffer(torch.cat(tracks_idx, dim=0))
        self.pose_gradient_mask = nn.Buffer(pose_gradient_mask, persistent=False)
        self.pose_pair_start_indices = nn.Buffer(torch.cat(tracks_pose_pair_start_indices, dim=0))
        self.motion_triplet_start_indices = nn.Buffer(
            torch.cat(tracks_motion_triplet_start_indices, dim=0)
            if tracks_motion_triplet_start_indices
            else torch.empty(0, device=device, dtype=torch.long)
        )

    def collect_data(self) -> ElevationAwareUnicycleTracksCalibData:
        """Return the calibration state consumed by collector backends.

        The returned view refers to the module's current parameters and buffers;
        constructing it does not copy tensor storage.
        """
        return ElevationAwareUnicycleTracksCalibData(
            xy=self.xy,
            height=self.height,
            theta=self.theta,
            pitch=self.pitch,
            velocity=self.velocity,
            theta_velocity=self.theta_velocity,
            pitch_velocity=self.pitch_velocity,
            pose_gradient_mask=self.pose_gradient_mask,
            tracks_idx=self.tracks_idx,
            pose_pair_start_indices=self.pose_pair_start_indices,
            motion_triplet_start_indices=self.motion_triplet_start_indices,
            time_delta_s=self.time_delta_s,
            time_delta_v_s=self.time_delta_v_s,
            initial_pose_xy=self.initial_pose_xy,
            initial_pose_height=self.initial_pose_height,
            ground_t=self.ground_t,
            ground_q=self.ground_q,
            Tclc_tquat=self.Tclc_tquat,
            pitch_slope_clamp_rad=float(self.config.pitch_slope_clamp_rad),
        )

    @staticmethod
    def _unwrap_angle_sequence(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return continuous angles and shortest signed adjacent deltas.

        This prevents angle wrap-around from producing artificially large
        angular velocity during initialization.
        """
        angle_deltas = torch.atan2(
            torch.sin(angles[1:] - angles[:-1]),
            torch.cos(angles[1:] - angles[:-1]),
        )
        if angle_deltas.numel() == 0:
            return angles, angle_deltas
        unwrapped_angles = torch.cat([angles[:1], angles[:1] + torch.cumsum(angle_deltas, dim=0)], dim=0)
        return unwrapped_angles, angle_deltas

    @staticmethod
    def _normalize_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(quat, dim=-1, eps=1e-12)

    def _get_yaw_pitch_quat(self, theta: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        zero_column = torch.zeros(theta.size(0), device=theta.device, dtype=theta.dtype).view(-1, 1)
        yaw_quat = self._normalize_quat_xyzw(
            torch.cat(
                [
                    zero_column,
                    zero_column,
                    torch.sin(theta / 2).view(-1, 1),
                    torch.cos(theta / 2).view(-1, 1),
                ],
                dim=1,
            )
        )
        pitch_quat = self._normalize_quat_xyzw(
            torch.cat(
                [
                    zero_column,
                    -torch.sin(pitch / 2).view(-1, 1),
                    zero_column,
                    torch.cos(pitch / 2).view(-1, 1),
                ],
                dim=1,
            )
        )
        return self._normalize_quat_xyzw(quat_mult_xyzw(yaw_quat, pitch_quat))

    def _warn_short_timestamp_gaps(
        self,
        *,
        all_timestamps_us: torch.Tensor,
        time_delta_s: torch.Tensor,
        track_idx: int,
        min_gap_s: float = 0.010,
    ) -> None:
        """Warn about tiny pose gaps that can inflate velocity and angular-velocity initialization.

        LiDAR cuboids can contain near-duplicate poses near sensor seams, e.g. behind the ego car.
        These show up as sub-millisecond gaps in otherwise ~100 ms tracks; we remove them for
        unicycle calibration stability.
        """
        short_gap_mask = time_delta_s < min_gap_s
        if not bool(short_gap_mask.any().item()):
            return

        track_id = self._track_ids[track_idx] if track_idx < len(self._track_ids) else str(track_idx)
        short_gap_indices = short_gap_mask.nonzero(as_tuple=False).flatten()
        time_delta_s_cpu = time_delta_s.detach().cpu().numpy()
        timestamps_us_cpu = all_timestamps_us.detach().cpu().numpy()
        examples = []
        for segment_idx in short_gap_indices[:5].detach().cpu().tolist():
            segment_idx = int(segment_idx)
            examples.append(
                {
                    "pose_idx": segment_idx,
                    "next_pose_idx": segment_idx + 1,
                    "dt_ms": float(time_delta_s_cpu[segment_idx] * 1000.0),
                    "t0_us": int(timestamps_us_cpu[segment_idx]),
                    "t1_us": int(timestamps_us_cpu[segment_idx + 1]),
                }
            )
        log.warning(
            f"{self.__class__.__name__}: {int(short_gap_mask.sum().item())} adjacent cuboid-track "
            f"timestamp gap(s) below {min_gap_s * 1000.0:g} ms during initialization: track_idx={track_idx} "
            f"track_id={track_id} examples={examples}"
        )

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        super().update_step_train_batch_start(epoch, global_step, system, **kwargs)
        self.requires_grad_(global_step >= self.config.start_global_step >= 0)

        with torch.no_grad():
            if self.vis_data.current is not None:
                optimized_tracks = self.forward(self.vis_data.current)
                optimized_tracks.visualize(1, show=False)
                self.visualize_unicycle()
                get_visualdebugger().update()

        return {}

    def visualize_unicycle(self):
        num_tracks = min(self.ground_q.size(0), 16)

        if not hasattr(self, "figure"):
            plt.ion()
            num_rows = math.ceil(math.sqrt(num_tracks))
            num_cols = math.ceil(num_tracks / num_rows)
            figure, axes = plt.subplots(nrows=num_rows, ncols=num_cols)
            self.figure = figure
            axes = np.atleast_1d(axes).ravel()
            self.all_quiver_arrows = []
            self.all_scatter_points = []
            self.all_connections = []
            redraw = False

        else:
            redraw = True

        xy, theta, tracks_idx = (
            self.xy.detach().cpu().numpy(),
            self.theta.detach().cpu().numpy(),
            self.tracks_idx.detach().cpu().numpy(),
        )
        velocity = self.velocity.detach().cpu().numpy()
        time_delta_s = self.time_delta_s.cpu().numpy()
        initial_xy = self.initial_pose_xy.cpu().numpy()

        v_idx = 0
        for track_idx in range(num_tracks):
            track_xy = xy[tracks_idx == track_idx]
            initial_track_xy = initial_xy[tracks_idx == track_idx]
            angle = theta[tracks_idx == track_idx]
            v = velocity[v_idx : v_idx + len(track_xy) - 1]
            dt = time_delta_s[v_idx : v_idx + len(track_xy) - 1]

            if redraw:
                self.all_scatter_points[track_idx].set_offsets(track_xy)
                self.all_quiver_arrows[track_idx].set_offsets(track_xy[:-1])
                self.all_quiver_arrows[track_idx].set_UVC(v * np.cos(angle[:-1]) * dt, v * np.sin(angle[:-1]) * dt)
                self.all_connections[track_idx].set_paths(np.stack([initial_track_xy, track_xy], axis=1))

            else:
                ax = axes[track_idx]
                ax.plot(initial_track_xy[:, 0], initial_track_xy[:, 1], linestyle="--", color="gray")
                conn = mc.LineCollection(
                    np.stack([initial_track_xy, track_xy], axis=1), color="red", linestyle="-", linewidth=0.2
                )
                scatter_points = ax.scatter(track_xy[:, 0], track_xy[:, 1])
                quiver_arrows = ax.quiver(
                    track_xy[:-1, 0],
                    track_xy[:-1, 1],
                    v * np.cos(angle[:-1]) * dt,
                    v * np.sin(angle[:-1]) * dt,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                )
                ax.add_collection(conn)
                self.all_scatter_points.append(scatter_points)
                self.all_quiver_arrows.append(quiver_arrows)
                self.all_connections.append(conn)
                ax.set_title(f"Track {track_idx}")
            v_idx += len(v)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def _watched_tensors(self) -> tuple[torch.Tensor, ...]:
        # - Watched: every value tensor the cached pose and residuals read — an
        #   unwatched dependency would serve stale values after an out-of-band mutation.
        # - Left out: the index and time-delta buffers, fixed problem structure set
        #   once at construction.
        return (
            self.xy,
            self.height,
            self.theta,
            self.pitch,
            self.velocity,
            self.theta_velocity,
            self.pitch_velocity,
            self.pose_gradient_mask,
            self.initial_pose_xy,
            self.initial_pose_height,
            self.ground_t,
            self.ground_q,
            self.Tclc_tquat,
        )

    def _get_pose_from_collector(self, compute_pose_vec: Callable[[], torch.Tensor]) -> lt.SE3:
        """Return the collector-composed pose while preserving the per-step cache."""
        cached = self._cache_lookup("collector_pose")
        if cached is not None:
            return cached

        pose = se3_from_plain_vec(compute_pose_vec())
        self._cache_store("collector_pose", pose, (pose.data,))
        return pose

    def _get_residuals_from_collector(
        self,
        compute_residuals: Callable[[], TracksCalibResiduals],
    ) -> TracksCalibResiduals:
        """Return collector residuals while preserving their shared autograd graph."""
        cached = self._cache_lookup("collector_residuals")
        if cached is not None:
            return cached

        value = compute_residuals()
        self._cache_store("collector_residuals", value, value.as_tuple())
        return value

    def _get_pose_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The per-pose channels with frozen rows detached; returns (xy, height, theta, pitch).

        Cached for the step: the camera and lidar collections and the pose losses all read the
        same parameters, and sharing the graph also spares the backward the repeat.
        """
        cached = self._cache_lookup("pose_params")
        if cached is not None:
            return cached

        # One masking pass over every channel at once, where one per channel costs a kernel
        # launch each on tensors far too small to hide it.
        stacked = torch.cat([self.xy, self.height.view(-1, 1), self.theta.view(-1, 1), self.pitch.view(-1, 1)], dim=1)
        stacked = torch.where(self.pose_gradient_mask.view(-1, 1), stacked, stacked.detach())
        value = (stacked[:, :2], stacked[:, 2], stacked[:, 3], stacked[:, 4])
        self._cache_store("pose_params", value, value)
        return value

    def get_pose(self) -> lt.SE3:
        cached = self._cache_lookup("pose")
        if cached is not None:
            return cached

        current_xy, current_height, current_theta, current_pitch = self._get_pose_params()

        T_world_ground_t = self.ground_t.to(device=current_xy.device, dtype=current_xy.dtype)[self.tracks_idx]
        T_world_ground_q = self._normalize_quat_xyzw(
            self.ground_q.to(device=current_xy.device, dtype=current_xy.dtype)[self.tracks_idx]
        )
        T_ground_cuboid_local_t = torch.cat([current_xy, current_height.view(-1, 1)], dim=1)
        T_ground_cuboid_local_q = self._get_yaw_pitch_quat(current_theta, current_pitch)
        Tclc_tquat = self.Tclc_tquat.to(device=current_xy.device, dtype=current_xy.dtype).expand(current_xy.size(0), -1)

        # Composed in vec space: backward through lt.SE3.InitFromVec runs a batched SVD, far
        # costlier than these elementwise kernels, and lt.SE3() wraps the result without one.
        t_world_local = T_world_ground_t + quat_rotate_xyzw(T_world_ground_q, T_ground_cuboid_local_t)
        q_world_local = self._normalize_quat_xyzw(quat_mult_xyzw(T_world_ground_q, T_ground_cuboid_local_q))
        t = t_world_local + quat_rotate_xyzw(q_world_local, Tclc_tquat[:, :3])
        q = self._normalize_quat_xyzw(quat_mult_xyzw(q_world_local, self._normalize_quat_xyzw(Tclc_tquat[:, 3:])))
        pose = se3_from_plain_vec(torch.cat([t, q], dim=1))
        # The parameter slices guard this entry too: a backward through the loss terms reaches
        # them without reaching pose.data, yet frees the graph this pose shares with them.
        self._cache_store("pose", pose, (pose.data, current_xy, current_height, current_theta, current_pitch))
        return pose

    def get_consistency(self) -> torch.Tensor:
        """
        Consistency loss that enforces all the parameters in the unicycle model harmonize.

        Return (n_track_poses - n_tracks, 5) tensor. The columns represent:
            1. Consistency in x-direction
            2. Consistency in y-direction
            3. Consistency in theta-direction
            4. Consistency in pitch-direction
            5. Consistency between height and pitch
        """
        current_xy, current_height, current_theta, current_pitch = self._get_pose_params()

        start_idx = self.pose_pair_start_indices
        end_idx = start_idx + 1
        dtheta = self.theta_velocity * self.time_delta_s
        dpitch = self.pitch_velocity * self.time_delta_s
        theta_mid = current_theta[start_idx] + 0.5 * dtheta
        arc_length = self.velocity * self.time_delta_s * torch.sinc(dtheta / (2.0 * math.pi))
        predicted_xy = torch.stack([arc_length * torch.cos(theta_mid), arc_length * torch.sin(theta_mid)], dim=1)

        diff_xy = current_xy[end_idx] - current_xy[start_idx]
        consistency_xy = diff_xy - predicted_xy
        consistency_theta = current_theta[end_idx] - current_theta[start_idx] - dtheta

        consistency_pitch = current_pitch[end_idx] - current_pitch[start_idx] - dpitch

        pitch_mid = 0.5 * (current_pitch[start_idx] + current_pitch[end_idx])
        pitch_slope_clamp_rad = float(self.config.pitch_slope_clamp_rad)
        pitch_mid_for_slope = torch.clamp(pitch_mid, -pitch_slope_clamp_rad, pitch_slope_clamp_rad)
        diff_height = current_height[end_idx] - current_height[start_idx]
        predicted_height = self.velocity * self.time_delta_s * torch.tan(pitch_mid_for_slope)
        consistency_height = diff_height - predicted_height

        consistency = torch.stack(
            [
                consistency_xy[:, 0],
                consistency_xy[:, 1],
                consistency_theta,
                consistency_pitch,
                consistency_height,
            ],
            dim=1,
        )
        return consistency

    def get_smoothness(self) -> torch.Tensor:
        """
        Smoothness loss that enforces constant horizontal, theta, pitch, and vertical acceleration.

        Return (n_track_poses - n_tracks * 3, 4) tensor. The columns represent:
            1. Horizontal acceleration
            2. Theta angular acceleration
            3. Pitch angular acceleration
            4. Vertical acceleration
        """
        if self.motion_triplet_start_indices.numel() == 0:
            return torch.empty(0, 4, device=self.xy.device, dtype=self.xy.dtype)

        t_0: torch.Tensor = self.motion_triplet_start_indices
        t_1: torch.Tensor = self.motion_triplet_start_indices + 1
        t_2: torch.Tensor = self.motion_triplet_start_indices + 2

        _, current_height, _, _ = self._get_pose_params()
        start_idx = self.pose_pair_start_indices
        height_velocity = (current_height[start_idx + 1] - current_height[start_idx]) / self.time_delta_s

        dt = self.time_delta_v_s

        def acceleration_delta(velocity: torch.Tensor) -> torch.Tensor:
            return (velocity[t_2] - velocity[t_1]) / dt[t_1] - (velocity[t_1] - velocity[t_0]) / dt[t_0]

        smoothness = torch.stack(
            [
                acceleration_delta(self.velocity),
                acceleration_delta(self.theta_velocity),
                acceleration_delta(self.pitch_velocity),
                acceleration_delta(height_velocity),
            ],
            dim=1,
        )
        return smoothness

    def get_pose_regularization(self) -> torch.Tensor:
        """
        Squared L2 residual between the current learned position and its initialization.

        Return (n_track_poses, ) tensor.
        """
        current_xy, current_height, _, _ = self._get_pose_params()
        pose_delta = torch.cat(
            [
                current_xy - self.initial_pose_xy,
                (current_height - self.initial_pose_height).view(-1, 1),
            ],
            dim=1,
        )
        pose_regularization = pose_delta.square().sum(dim=1)
        return pose_regularization

    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks:
        self.save_tracks_for_visualize(tracks)
        return CuboidTracks.Ops.clone(tracks, tracks_poses=self.get_pose())

    __call__ = module_call_type(forward)

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config, self.trainer_config, self, name_prefix)


class SkipTracksCalib(BaseTracksCalib):
    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config, trainer_config, cuboid_tracks)

    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks:
        # return inputs as is
        return tracks

    __call__ = module_call_type(forward)


class CompositeTracksCalib(BaseTracksCalib):
    """Routes calibration based on whether tracks are original or inserted.

    Used for inference-time asset insertion/edit flows.
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        original_calib: BaseTracksCalib,
        inserted_track_ids: set[str],
        cuboid_tracks: CuboidTracks,
    ):
        # Avoid BaseTracksCalib.__init__ to prevent repeatedly mutating shared config
        # (e.g. start_global_step world-size adjustment) on chained insertions.
        BaseModel.__init__(self, config)
        self.trainer_config = trainer_config
        self.original_calib = original_calib
        self.inserted_track_ids = set(inserted_track_ids)

    def forward(self, tracks: CuboidTracks) -> CuboidTracks:
        # Edge case: no inserted tracks.
        if not self.inserted_track_ids:
            return self.original_calib(tracks)

        ordered_all_ids = tracks.tracks_id
        inserted_track_ids_ordered = [track_id for track_id in ordered_all_ids if track_id in self.inserted_track_ids]

        # No inserted track is present in this specific track set.
        if not inserted_track_ids_ordered:
            return self.original_calib(tracks)

        # Edge case: all tracks are inserted; equivalent to SkipCalib behavior.
        if len(inserted_track_ids_ordered) == len(ordered_all_ids):
            return tracks

        original_track_ids_ordered = [
            track_id for track_id in ordered_all_ids if track_id not in self.inserted_track_ids
        ]
        original_tracks = CuboidTracks.Ops.subset_from_tracks_id(tracks, original_track_ids_ordered)
        inserted_tracks = CuboidTracks.Ops.subset_from_tracks_id(tracks, inserted_track_ids_ordered)

        calibrated_original = self.original_calib(original_tracks)
        merged_tracks = CuboidTracks.Ops.concatenate([calibrated_original, inserted_tracks])

        # Preserve the original tracks order so gaussian_cuboid_ids keep a stable track-index mapping.
        reordered_tracks = CuboidTracks.Ops.subset_from_tracks_id(merged_tracks, ordered_all_ids)
        assert reordered_tracks.tracks_id == ordered_all_ids, (
            "CompositeTracksCalib: output track order must match input track order"
        )
        return reordered_tracks

    __call__ = module_call_type(forward)
