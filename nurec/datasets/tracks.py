# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Optional, Self, Tuple, cast

import dataclasses_json
import lietorch as lt
import numpy as np
import torch

from nurec.datasets.track_split import TrackSplitChild, TrackSplitMetadata, TrackSplitResolver
from nurec.libs.packed_ops.interface import packed_ops  # type: ignore
from nurec.libs.vren.interface import vren  # type: ignore
from nurec.utils.fields import field_enum
from nurec.utils.geometry import se3_matrix_to_tquat, tquat_to_se3_matrix
from nurec.utils.misc import get_pack_info_from_n
from nurec.utils.packed_ops import linstep_interleave
from nurec.utils.types import CuboidTracksData, CuboidTracksDataPack, FrameConversion, TrackFlags, TracksData
from nurec.visualdebugger import get_visualdebugger


log = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class Tracks(dataclasses_json.DataClassJsonMixin):
    """Manages time-dependent poses and related interpolation / intersection GPU operations for tracks

    This class uses the :class:`nurec.utils.types.TracksData` class for storing data.
    """

    @dataclass(slots=True)
    class Unpacked(dataclasses_json.DataClassJsonMixin):
        """'Unpacked' serialization format of the internal packed representation that is easier for human / external modification"""

        tracks_id: list[str] = field(default_factory=list)
        tracks_poses: list[list] = field(default_factory=list)
        tracks_timestamps_us: list[list] = field(default_factory=list)
        tracks_label_class: list[str] = field(default_factory=list)
        tracks_flags: list[TrackFlags] = field_enum(TrackFlags, default_factory=list)

        @staticmethod
        def from_data(data: TracksData) -> Tracks.Unpacked:
            tracks_id = []
            tracks_poses = []
            tracks_timestamps_us = []
            tracks_label_class = []
            tracks_flags = []

            # Unpack data for each track
            for track_idx, track_id in enumerate(data.tracks_id):
                tracks_id.append(track_id)
                start_idx, n_poses = data.tracks_packinfo[track_idx].tolist()
                tracks_poses.append(data.tracks_poses[start_idx : start_idx + n_poses].data.tolist())
                tracks_timestamps_us.append(data.tracks_timestamps_us[start_idx : start_idx + n_poses].data.tolist())
                tracks_label_class.append(data.tracks_label_class[track_idx])
                tracks_flags.append(TrackFlags(int(data.tracks_flags[track_idx].item())))

            return Tracks.Unpacked(
                tracks_id,
                tracks_poses,
                tracks_timestamps_us,
                tracks_label_class,
                tracks_flags,
            )

    @staticmethod
    def encoder_tracks(input: TracksData) -> dict:
        return Tracks.Unpacked.from_data(input).to_dict()

    @staticmethod
    def decoder_tracks(input: dict) -> TracksData:
        unpacked = Tracks.Unpacked.from_dict(input)
        return Tracks.Factory.from_numpy(
            unpacked.tracks_id,
            [np.array(track_poses, dtype=np.float32) for track_poses in unpacked.tracks_poses],
            [np.array(track_timestamps_us, dtype=np.int64) for track_timestamps_us in unpacked.tracks_timestamps_us],
            unpacked.tracks_label_class,
            unpacked.tracks_flags,
            "tquat",
        ).tracks_data

    tracks_data: TracksData = field(metadata=dataclasses_json.config(encoder=encoder_tracks, decoder=decoder_tracks))

    @property
    def tracks_id(self) -> list[str]:
        return self.tracks_data.tracks_id

    @property
    def max_track_n_poses(self) -> int:
        return self.tracks_data.max_track_n_poses

    @property
    def tracks_label_class(self) -> list[str]:
        return self.tracks_data.tracks_label_class

    @property
    def tracks_packinfo(self) -> torch.Tensor:
        return self.tracks_data.tracks_packinfo

    @property
    def tracks_poses(self) -> lt.SE3:
        return self.tracks_data.tracks_poses

    @tracks_poses.setter
    def tracks_poses(self, value: lt.SE3):
        """Allow updating tracks poses with consistent values"""
        assert (
            self.tracks_data.tracks_poses.shape == value.shape
            and self.tracks_data.tracks_poses.device == value.device
            and self.tracks_data.tracks_poses.dtype == value.dtype
        )
        self.tracks_data.tracks_poses = value

    @property
    def tracks_timestamps_us(self) -> torch.Tensor:
        return self.tracks_data.tracks_timestamps_us

    @property
    def max_positive_timestamp_gap_us(self) -> torch.Tensor:
        return self.tracks_data.max_positive_timestamp_gap_us

    @property
    def tracks_flags(self) -> torch.Tensor:
        return self.tracks_data.tracks_flags

    class Factory:
        @staticmethod
        def from_numpy(
            tracks_id: list[str],
            tracks_poses: list[np.ndarray],
            tracks_timestamps_us: list[np.ndarray],
            tracks_label_class: list[str],
            tracks_flags: list[TrackFlags],
            pose_format: Literal["matrix", "tquat"] = "matrix",
            device: torch.device = torch.device("cuda"),
        ) -> Tracks:
            """
            Inputs (N_tracks: number of different tracks, N_poses_i: number of poses of track i):
            - track_ids: string identifiers of each track, N_tracks [str]
            - tracks_poses: 4x4 matrix or 7-dim tquat vector track to world pose transformations, depending on 'pose_format', N_tracks x (N_poses_i x [4 x 4 | 7]) [float]
            - tracks_timestamps_us: pose timestamps, N_tracks x (N_poses_i, ) [int64]
            - tracks_label_class: semantic class of each track, N_tracks [str]
            - tracks_flags: per-track flags, N_tracks [TrackFlags]
            - pose_format: the format of the poses (either 4x4 'matrix'es or 7-dim 'tquat' vectors)
            - device: the device to store the data on [torch.device]
            """

            # Convert to compressed pose representation
            tracks_poses_sizes = [len(track_poses) for track_poses in tracks_poses]

            assert len(tracks_id) == len(tracks_poses_sizes), "Tracks: inconsistent track_ids / pose"
            assert len(tracks_id) == len(tracks_flags), "Tracks: inconsistent track_ids / flags"
            assert tracks_poses_sizes == [len(track_timestamps_us) for track_timestamps_us in tracks_timestamps_us], (
                "Tracks: inconsistent pose / timestamp pairs"
            )

            assert all(
                [
                    (track_poses.shape[-2:] == (4, 4) if pose_format == "matrix" else track_poses.shape[-1:] == (7,))
                    and track_poses.dtype == np.float32
                    for track_poses in tracks_poses
                ]
            ), "Tracks: invalid poses inputs"
            assert all([len(track_poses) >= 2 for track_poses in tracks_poses]), (
                "Tracks: require at least two poses per track"
            )
            assert all(
                [
                    len(track_timestamps_us.shape) == 1 and track_timestamps_us.dtype == np.int64
                    for track_timestamps_us in tracks_timestamps_us
                ]
            ), "Tracks: invalid tracks_timestamps_us inputs"
            assert len(tracks_id) == len(tracks_label_class), "Tracks: inconsistent track_ids / label_class"

            # create packed pose / timestamp representation and upload to GPU
            # (also handle special case of empty / non-existing tracks)
            n_poses = [track_poses.shape[0] for track_poses in tracks_poses]
            start_idxs = np.cumsum([0] + n_poses, dtype=np.int32)[:-1]

            max_track_n_poses = max(n_poses) if len(n_poses) else 0
            max_positive_timestamp_gap_us = np.asarray(
                [int(np.max(np.diff(track_timestamps_us), initial=0)) for track_timestamps_us in tracks_timestamps_us],
                dtype=np.int64,
            )

            tracks_packinfo = torch.tensor(
                np.stack((start_idxs, np.array(n_poses, dtype=np.int32)), axis=1), device=device
            )

            N_total_poses = tracks_packinfo[:, 1].sum()

            tracks_poses_array = np.empty(
                (N_total_poses, 4, 4) if pose_format == "matrix" else (N_total_poses, 7), dtype=np.float32
            )
            tracks_timestamps_array_us = np.empty((N_total_poses,), dtype=np.int64)
            if N_total_poses > 0:
                np.concatenate(tracks_poses, out=tracks_poses_array)
                np.concatenate(tracks_timestamps_us, out=tracks_timestamps_array_us)

            tracks_poses_se3: lt.SE3
            if pose_format == "matrix":
                tracks_poses_se3 = lt.SE3(se3_matrix_to_tquat(tracks_poses_array).to(device))
            else:
                tracks_poses_se3 = lt.SE3(torch.from_numpy(tracks_poses_array).to(device))

            return Tracks(
                tracks_data=TracksData(
                    tracks_id=tracks_id,
                    tracks_packinfo=tracks_packinfo,
                    tracks_poses=tracks_poses_se3,
                    tracks_timestamps_us=torch.from_numpy(tracks_timestamps_array_us).to(device),
                    tracks_flags=torch.tensor(
                        [track_flags.value for track_flags in tracks_flags], dtype=torch.int32, device=device
                    ),
                    max_track_n_poses=max_track_n_poses,
                    tracks_label_class=tracks_label_class,
                    max_positive_timestamp_gap_us=torch.from_numpy(max_positive_timestamp_gap_us).to(device),
                )
            )

    def get_mask_flags_all(self, flags: TrackFlags) -> torch.Tensor:
        """Mask indicating the tracks that have *all* flag bits of 'flags' set"""
        return torch.bitwise_and(self.tracks_flags, flags.value).eq(flags.value)

    def get_mask_flags_any(self, flags: TrackFlags) -> torch.Tensor:
        """Mask indicating the tracks that have *any* flag bits of 'flags' set"""
        return torch.bitwise_and(self.tracks_flags, flags.value).ne(0)

    def get_mask_flags_none(self, flags: TrackFlags) -> torch.Tensor:
        """Mask indicating the tracks that have *none* of the flag bits of 'flags' set"""
        return torch.bitwise_and(self.tracks_flags, flags.value).eq(0)

    @property
    def device(self) -> torch.device:
        return self.tracks_poses.device

    def to_device(self, device: torch.device) -> Self:
        return cast(Self, self.__class__(tracks_data=self.tracks_data.to_device(device)))

    @property
    def n_tracks(self) -> int:
        return len(self.tracks_id)


@dataclass(kw_only=True, slots=True)
class CuboidTracks(Tracks):
    """Manages time-dependent cuboid tracks related GPU intersection operations for tracks

    This class uses the :class:`nurec.datasets.tracks.CuboidTracks.Data` class for storing data.

    """

    @dataclass(slots=True)
    class Unpacked(dataclasses_json.DataClassJsonMixin):
        """'Unpacked' serialization format of the internal packed representation that is easier for human / external modification"""

        cuboids_dims: list[list] = field(default_factory=list)

    @staticmethod
    def encoder_cuboidtracks(input: CuboidTracksData) -> dict:
        # combine both the base part and local part
        return CuboidTracks.Unpacked(input.cuboids_dims.tolist()).to_dict()

    @staticmethod
    def decoder_cuboidtracks(input: dict) -> CuboidTracksData:
        return CuboidTracksData(
            cuboids_dims=torch.tensor(
                CuboidTracks.Unpacked.from_dict(input).cuboids_dims,
                dtype=torch.float32,
                device=torch.device("cuda"),
            ).reshape(-1, 3)
        )

    cuboidtracks_data: CuboidTracksData = field(
        metadata=dataclasses_json.config(encoder=encoder_cuboidtracks, decoder=decoder_cuboidtracks)
    )

    @property
    def cuboids_dims(self) -> torch.Tensor:
        return self.cuboidtracks_data.cuboids_dims

    class Factory:
        @staticmethod
        def empty(device: torch.device = torch.device("cuda")) -> CuboidTracks:
            """
            Constructs an empty cuboid tracks container
            """
            return CuboidTracks.Factory.from_numpy(
                tracks_id=[],
                tracks_poses=[],
                tracks_timestamps_us=[],
                tracks_label_class=[],
                tracks_flags=[],
                cuboids_dims=[],
                device=device,
            )

        @staticmethod
        def from_numpy(
            tracks_id: list[str],
            tracks_poses: list[np.ndarray],
            tracks_timestamps_us: list[np.ndarray],
            tracks_label_class: list[str],
            tracks_flags: list[TrackFlags],
            cuboids_dims: list[np.ndarray],
            device: torch.device = torch.device("cuda"),
        ) -> CuboidTracks:
            """
            Inputs (N_tracks: number of different tracks, N_poses_i: number of poses of track i):
            - track_ids: string identifiers of each track, N_tracks [str]
            - tracks_poses: 4x4 track to world pose transformations, N_tracks x (N_poses_i x 4 x 4) [float]
            - tracks_timestamps_us: pose timestamps, N_tracks x (N_poses_i, ) [int64]
            - tracks_label_class: semantic class of each track, N_tracks [str]
            - tracks_flags: per-track flags, N_tracks [TrackFlags]
            - cuboids_dims: cuboid x/y/z extents (in local track frame), N_tracks x 3 [float]
            - device: the device to store the data on [torch.device]
            """

            # construct base tracks part
            tracks = Tracks.Factory.from_numpy(
                tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, device=device
            )

            # construct cuboid-tracks-specific part
            if len(cuboids_dims_array := np.empty((len(tracks.tracks_id), 3), dtype=np.float32)):
                np.stack(cuboids_dims, out=cuboids_dims_array)

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=tracks.tracks_id,
                    tracks_packinfo=tracks.tracks_packinfo,
                    tracks_poses=tracks.tracks_poses,
                    tracks_timestamps_us=tracks.tracks_timestamps_us,
                    tracks_flags=tracks.tracks_flags,
                    max_track_n_poses=tracks.max_track_n_poses,
                    tracks_label_class=tracks.tracks_label_class,
                    max_positive_timestamp_gap_us=tracks.max_positive_timestamp_gap_us,
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=torch.tensor(cuboids_dims_array, dtype=torch.float32, device=device)
                ),
            )

        @staticmethod
        def from_pack(pack: CuboidTracksDataPack) -> CuboidTracks:
            """
            Constructs cuboid tracks from packed data components
            """

            return CuboidTracks(tracks_data=pack.tracks_data, cuboidtracks_data=pack.cuboidtracks_data)

    class Ops:
        """
        Operations on CuboidTracks returning new CuboidTracks instances.
        Compared to classmethods, this forces the semantics to be NOT inplace.
        """

        @staticmethod
        def clone(cuboid_tracks: CuboidTracks, tracks_poses: lt.SE3 | None = None):
            """Clone the cuboid tracks instance.

            When ``tracks_poses`` is provided, use that exact ``lt.SE3`` object instead of
            copying the original pose set. The clone therefore aliases the supplied pose
            object; every other mutable field is copied.
            """

            if tracks_poses is not None:
                replacement_is_compatible = (
                    tracks_poses.shape == cuboid_tracks.tracks_poses.shape
                    and tracks_poses.device == cuboid_tracks.tracks_poses.device
                    and tracks_poses.dtype == cuboid_tracks.tracks_poses.dtype
                )
                if not replacement_is_compatible:
                    raise ValueError(
                        "Replacement poses must match the source shape, device, and dtype: "
                        f"expected shape={cuboid_tracks.tracks_poses.shape}, "
                        f"device={cuboid_tracks.tracks_poses.device}, dtype={cuboid_tracks.tracks_poses.dtype}; "
                        f"got shape={tracks_poses.shape}, device={tracks_poses.device}, dtype={tracks_poses.dtype}"
                    )

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=cuboid_tracks.tracks_id[:],
                    tracks_packinfo=cuboid_tracks.tracks_packinfo.clone(),
                    tracks_poses=tracks_poses
                    if tracks_poses is not None
                    else lt.SE3(cuboid_tracks.tracks_poses.data.clone()),
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us.clone(),
                    tracks_flags=cuboid_tracks.tracks_flags.clone(),
                    max_track_n_poses=cuboid_tracks.max_track_n_poses,
                    tracks_label_class=cuboid_tracks.tracks_label_class[:],
                    max_positive_timestamp_gap_us=cuboid_tracks.max_positive_timestamp_gap_us.clone(),
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims.clone(),
                ),
            )

        @staticmethod
        def transform_with_frame_conversion(
            cuboid_tracks: CuboidTracks,
            source_to_target_frame_conversion: FrameConversion | None,
            T_local_to_common_target: np.ndarray | torch.Tensor | None,
        ) -> CuboidTracks:
            """Transforms the cuboid tracks from their current frame to a target frame"""

            if not cuboid_tracks.n_tracks:
                # no need to transform empty tracks
                return cuboid_tracks

            # Widen the operands rather than narrowing the matmul precision:
            # - A float32 CUDA matmul rounds its inputs to TF32's 10-bit mantissa under
            #   torch.set_float32_matmul_precision("high"/"medium"), which at world-scale
            #   coordinates perturbs the poses by millimetres to centimetres.
            # - The precision setting is process-global, so narrowing it here would change
            #   every other thread's matmuls for the duration of the call.
            pose_dtype = cuboid_tracks.tracks_poses.dtype
            tracks_poses = cuboid_tracks.tracks_poses.matrix().to(torch.float64)

            if isinstance(T_local_to_common_target, np.ndarray):
                T_local_to_common_target = torch.from_numpy(T_local_to_common_target)

            if T_local_to_common_target is not None:
                tracks_poses = T_local_to_common_target.to(tracks_poses) @ tracks_poses

            cuboids_dims = cuboid_tracks.cuboids_dims
            if source_to_target_frame_conversion is not None:
                T_np, S_np = source_to_target_frame_conversion.get_transformation_matrices()
                T = torch.from_numpy(T_np).to(tracks_poses)
                S = torch.from_numpy(S_np).to(tracks_poses)
                tracks_poses = T @ tracks_poses @ S

                # Apply transform's scale to the cuboid dimensions. The divisor comes back to
                # float32 because CuboidTracksData rejects dims of any other dtype.
                cuboids_dims = cuboids_dims / torch.diag(S)[None, :3].to(cuboids_dims)

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=cuboid_tracks.tracks_id,
                    tracks_packinfo=cuboid_tracks.tracks_packinfo,
                    # Store the poses back at their input dtype; nothing downstream checks it.
                    tracks_poses=lt.SE3(se3_matrix_to_tquat(tracks_poses).to(pose_dtype)),
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us,
                    tracks_flags=cuboid_tracks.tracks_flags,
                    max_track_n_poses=cuboid_tracks.max_track_n_poses,
                    tracks_label_class=cuboid_tracks.tracks_label_class,
                    max_positive_timestamp_gap_us=cuboid_tracks.max_positive_timestamp_gap_us,
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboids_dims,
                ),
            )

        @staticmethod
        def transform_with_delta_poses(
            cuboid_tracks: CuboidTracks,
            delta_poses: lt.SE3,
            left_multiply: bool = True,
        ) -> CuboidTracks:
            """Transforms the cuboid tracks according to the delta poses"""

            if left_multiply:
                tracks_poses = delta_poses * cuboid_tracks.tracks_poses
            else:
                tracks_poses = cuboid_tracks.tracks_poses * delta_poses

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=cuboid_tracks.tracks_id,
                    tracks_packinfo=cuboid_tracks.tracks_packinfo,
                    tracks_poses=tracks_poses,
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us,
                    tracks_flags=cuboid_tracks.tracks_flags,
                    max_track_n_poses=cuboid_tracks.max_track_n_poses,
                    tracks_label_class=cuboid_tracks.tracks_label_class,
                    max_positive_timestamp_gap_us=cuboid_tracks.max_positive_timestamp_gap_us,
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims,
                ),
            )

        @staticmethod
        def freeze(
            cuboid_tracks: CuboidTracks,
            min_timestamps_us: torch.Tensor,
            max_timestamps_us: torch.Tensor,
            tracks_poses_start: lt.SE3,
            tracks_poses_end: lt.SE3 | None = None,
        ) -> CuboidTracks:
            """
            Freeze each of the tracks to one pose (optionally two if `tracks_poses_end` is set).

            Inputs:
            - cuboid_tracks: the input cuboid tracks
            - min_timestamps_us: Tensor (N_tracks, ) [int64] containing the first timestamp for each track
            - max_timestamps_us: Tensor (N_tracks, ) [int64] containing the second (and last) timestamp for each track
            - tracks_poses_start: Start SE3 poses to freeze the tracks to
            - tracks_poses_end: Optional end SE3 poses to freeze the tracks to

            Returns:
            - the frozen cuboid tracks
            """

            assert min_timestamps_us.size(0) == cuboid_tracks.n_tracks, "CuboidTracks: invalid min_timestamps_us size"
            assert max_timestamps_us.size(0) == cuboid_tracks.n_tracks, "CuboidTracks: invalid max_timestamps_us size"
            assert tracks_poses_start.shape[0] == cuboid_tracks.n_tracks, (
                "CuboidTracks: invalid tracks_poses_start size"
            )
            if tracks_poses_end is not None:
                assert tracks_poses_end.shape[0] == cuboid_tracks.n_tracks, (
                    "CuboidTracks: invalid tracks_poses_end size"
                )
            else:
                tracks_poses_end = tracks_poses_start

            n_poses = torch.full((cuboid_tracks.n_tracks,), 2, dtype=torch.int32, device=cuboid_tracks.device)
            tracks_timestamps_us = torch.stack((min_timestamps_us, max_timestamps_us), dim=1).view(-1)
            tracks_poses = lt.stack([tracks_poses_start, tracks_poses_end], dim=1).view((-1,))

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=cuboid_tracks.tracks_id,
                    tracks_packinfo=get_pack_info_from_n(n_poses),
                    tracks_poses=tracks_poses,
                    tracks_timestamps_us=tracks_timestamps_us,
                    tracks_flags=cuboid_tracks.tracks_flags,
                    max_track_n_poses=2,
                    tracks_label_class=cuboid_tracks.tracks_label_class,
                    max_positive_timestamp_gap_us=torch.clamp_min(max_timestamps_us - min_timestamps_us, 0),
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims,
                ),
            )

        @staticmethod
        def concatenate(cuboid_tracks_list: Iterable[CuboidTracks]) -> CuboidTracks:
            """
            Concatenates a list of CuboidTracks instances into a new one
            """

            tracks_id_list = [cuboid_tracks.tracks_id for cuboid_tracks in cuboid_tracks_list]
            tracks_length_list = [cuboid_tracks.tracks_packinfo[:, 1] for cuboid_tracks in cuboid_tracks_list]
            tracks_poses_list = [cuboid_tracks.tracks_poses for cuboid_tracks in cuboid_tracks_list]
            tracks_timestamps_us_list = [cuboid_tracks.tracks_timestamps_us for cuboid_tracks in cuboid_tracks_list]
            tracks_flags_list = [cuboid_tracks.tracks_flags for cuboid_tracks in cuboid_tracks_list]
            cuboids_dims_list = [cuboid_tracks.cuboids_dims for cuboid_tracks in cuboid_tracks_list]
            max_track_n_poses = max([cuboid_tracks.max_track_n_poses for cuboid_tracks in cuboid_tracks_list])
            tracks_label_class_list = [cuboid_tracks.tracks_label_class for cuboid_tracks in cuboid_tracks_list]

            tracks_packinfo = get_pack_info_from_n(torch.cat(tracks_length_list, dim=0)).int()
            tracks_id: list[str] = sum(tracks_id_list, [])

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=tracks_id,
                    tracks_packinfo=tracks_packinfo,
                    tracks_poses=lt.cat(tracks_poses_list, dim=0),
                    tracks_timestamps_us=torch.cat(tracks_timestamps_us_list, dim=0),
                    tracks_flags=torch.cat(tracks_flags_list, dim=0),
                    max_track_n_poses=max_track_n_poses,
                    tracks_label_class=sum(tracks_label_class_list, []),
                    max_positive_timestamp_gap_us=torch.cat(
                        [cuboid_tracks.max_positive_timestamp_gap_us for cuboid_tracks in cuboid_tracks_list],
                        dim=0,
                    ),
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=torch.cat(cuboids_dims_list, dim=0),
                ),
            )

        @staticmethod
        def coalesce_short_timestamp_gaps(cuboid_tracks: CuboidTracks, min_gap_us: int) -> CuboidTracks:
            """
            Coalesce near-duplicate packed cuboid poses within each track.

            Consecutive poses connected by timestamp gaps smaller than ``min_gap_us`` are grouped and
            replaced by a single representative original pose. This avoids introducing new averaged
            rotations while removing pathological tiny-dt segments from downstream calibration.
            """
            if min_gap_us <= 0:
                raise ValueError(f"min_gap_us must be positive, got {min_gap_us}")
            if cuboid_tracks.n_tracks == 0:
                return CuboidTracks.Ops.clone(cuboid_tracks)

            def _dt_stats(dts_us: list[int]) -> str:
                values = np.asarray(dts_us, dtype=np.float64) / 1000.0
                return f"min={values.min():.3f}ms, median={np.median(values):.3f}ms, max={values.max():.3f}ms"

            packinfo_cpu = cuboid_tracks.tracks_packinfo.detach().cpu()
            timestamps_cpu = cuboid_tracks.tracks_timestamps_us.detach().cpu()

            kept_indices: list[int] = []
            track_counts: list[int] = []
            track_max_positive_timestamp_gaps_us: list[int] = []
            affected_track_count = 0
            fallback_track_count = 0
            remaining_short_gap_track_count = 0
            before_dts_us: list[int] = []
            after_dts_us: list[int] = []
            total_fused_groups = 0

            for track_idx, (pose_start, pose_count) in enumerate(packinfo_cpu.tolist()):
                pose_start = int(pose_start)
                pose_count = int(pose_count)
                timestamps = timestamps_cpu[pose_start : pose_start + pose_count]
                dts = (timestamps[1:] - timestamps[:-1]).tolist()
                if any(int(dt_us) <= 0 for dt_us in dts):
                    raise ValueError(
                        "track timestamps must be strictly increasing before short-gap coalescing: "
                        f"track_idx={track_idx} track_id={cuboid_tracks.tracks_id[track_idx]} "
                        f"timestamps_us={timestamps.tolist()}"
                    )
                before_dts_us.extend(int(dt) for dt in dts)

                selected_local_indices: list[int] = []
                n_fused_groups = 0
                group_start = 0
                for gap_idx, dt_us in enumerate(dts):
                    if int(dt_us) >= min_gap_us:
                        group_end = gap_idx
                        selected_local_indices.append(
                            CuboidTracks.Ops._short_gap_group_representative_index(group_start, group_end, pose_count)
                        )
                        if group_end > group_start:
                            n_fused_groups += 1
                        group_start = gap_idx + 1

                group_end = pose_count - 1
                selected_local_indices.append(
                    CuboidTracks.Ops._short_gap_group_representative_index(group_start, group_end, pose_count)
                )
                if group_end > group_start:
                    n_fused_groups += 1

                # A track with fewer than two poses is invalid for the existing track representation.
                if len(selected_local_indices) < 2:
                    selected_local_indices = [0, pose_count - 1]
                    fallback_track_count += 1

                selected_timestamps = timestamps[selected_local_indices]
                selected_dts = (selected_timestamps[1:] - selected_timestamps[:-1]).tolist()
                after_dts_us.extend(int(dt) for dt in selected_dts)
                track_max_positive_timestamp_gaps_us.append(max(int(dt) for dt in selected_dts))
                if any(int(dt) < min_gap_us for dt in selected_dts):
                    remaining_short_gap_track_count += 1

                kept_indices.extend(pose_start + int(local_idx) for local_idx in selected_local_indices)
                track_counts.append(len(selected_local_indices))

                removed_count = pose_count - len(selected_local_indices)
                if removed_count > 0:
                    affected_track_count += 1
                    total_fused_groups += n_fused_groups

            kept_indices_tensor = torch.tensor(kept_indices, device=cuboid_tracks.device, dtype=torch.long)
            track_counts_tensor = torch.tensor(track_counts, device=cuboid_tracks.device, dtype=torch.int32)
            total_before = int(cuboid_tracks.tracks_poses.shape[0])
            total_after = len(kept_indices)
            total_removed = total_before - total_after

            if total_removed > 0:
                log.info(
                    "Coalesced short cuboid-track timestamp gaps < %dus: poses %d -> %d "
                    "(removed=%d, affected_tracks=%d, fused_groups=%d), dt_before=(%s), dt_after=(%s)",
                    min_gap_us,
                    total_before,
                    total_after,
                    total_removed,
                    affected_track_count,
                    total_fused_groups,
                    _dt_stats(before_dts_us),
                    _dt_stats(after_dts_us),
                )
            else:
                log.info(
                    "No cuboid-track timestamp gaps < %dus found: poses=%d, dt=(%s)",
                    min_gap_us,
                    total_before,
                    _dt_stats(before_dts_us),
                )

            if fallback_track_count:
                log.warning(
                    "Kept first/last original pose for %d track(s) that would collapse below two poses during "
                    "short-gap fusion",
                    fallback_track_count,
                )
            if remaining_short_gap_track_count:
                log.warning(
                    "Short cuboid-track timestamp gaps < %dus remain after fusion for %d track(s)",
                    min_gap_us,
                    remaining_short_gap_track_count,
                )

            return CuboidTracks(
                tracks_data=TracksData(
                    tracks_id=cuboid_tracks.tracks_id[:],
                    tracks_packinfo=get_pack_info_from_n(track_counts_tensor).int(),
                    tracks_poses=cuboid_tracks.tracks_poses[kept_indices_tensor],
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us[kept_indices_tensor],
                    tracks_flags=cuboid_tracks.tracks_flags.clone(),
                    max_track_n_poses=max(track_counts),
                    tracks_label_class=cuboid_tracks.tracks_label_class[:],
                    max_positive_timestamp_gap_us=torch.tensor(
                        track_max_positive_timestamp_gaps_us,
                        device=cuboid_tracks.device,
                        dtype=torch.int64,
                    ),
                ),
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims.clone(),
                ),
            )

        @staticmethod
        def _short_gap_group_representative_index(group_start: int, group_end: int, pose_count: int) -> int:
            if group_start == 0:
                return group_start
            if group_end == pose_count - 1:
                return group_end
            return (group_start + group_end) // 2

        @staticmethod
        def subset_from_indices(cuboid_tracks: CuboidTracks, indices: list[int] | torch.Tensor) -> CuboidTracks:
            """
            Subsets the cuboid tracks so that the new tracks_id matches argument indices
            This operation keeps the gradient flow.
            """
            if isinstance(indices, list):
                indices_list = indices
                indices_tensor = torch.tensor(indices, device=cuboid_tracks.device, dtype=torch.long)

            else:
                indices_list = indices.cpu().numpy().tolist()
                indices_tensor = indices

            track_starts_counts = cuboid_tracks.tracks_packinfo[indices_tensor]
            track_starts, track_counts = track_starts_counts[:, 0], track_starts_counts[:, 1]
            packed_attr_ind = (
                linstep_interleave(track_starts, track_counts, 1).values
                if len(track_starts)
                else torch.empty(0, dtype=torch.long, device=cuboid_tracks.device)
            )

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=[cuboid_tracks.tracks_id[i] for i in indices_list],
                    tracks_packinfo=get_pack_info_from_n(track_counts),
                    tracks_poses=cuboid_tracks.tracks_poses[packed_attr_ind],
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us[packed_attr_ind],
                    tracks_flags=cuboid_tracks.tracks_flags[indices_tensor],
                    max_track_n_poses=cast(int, track_counts.max().item()) if len(track_counts) else 0,
                    tracks_label_class=[cuboid_tracks.tracks_label_class[i] for i in indices_list],
                    max_positive_timestamp_gap_us=cuboid_tracks.max_positive_timestamp_gap_us[indices_tensor],
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims[indices_tensor],
                ),
            )

        @staticmethod
        def subset_from_tracks_id(cuboid_tracks: CuboidTracks, tracks_id: list[str]) -> CuboidTracks:
            """
            Subsets the cuboid tracks so that the new tracks_id matches argument tracks_id
            This operation keeps the gradient flow.
            """
            indices = [cuboid_tracks.tracks_id.index(tid) for tid in tracks_id]
            return CuboidTracks.Ops.subset_from_indices(cuboid_tracks, indices)

        @staticmethod
        def subset_from_mask(cuboid_tracks: CuboidTracks, mask: torch.Tensor) -> CuboidTracks:
            """
            Subsets the cuboid tracks so that the new tracks_id matches the mask
            This operation keeps the gradient flow.
            """
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            return CuboidTracks.Ops.subset_from_indices(cuboid_tracks, indices)

        @staticmethod
        def clean_track_ids(cuboid_tracks: CuboidTracks, cleaner_fn: Callable[[str], str]) -> CuboidTracks:
            """
            Cleans the track IDs using the provided cleaner function.
            Returns a new CuboidTracks with cleaned track IDs.

            Args:
                cuboid_tracks: The cuboid tracks to clean
                cleaner_fn: A function that takes a track ID string and returns a cleaned version

            Returns:
                A new CuboidTracks with cleaned track IDs
            """
            cleaned_tracks_id = [cleaner_fn(tid) for tid in cuboid_tracks.tracks_id]

            return CuboidTracks(
                tracks_data=TracksData(
                    tracks_id=cleaned_tracks_id,
                    tracks_packinfo=cuboid_tracks.tracks_packinfo,
                    tracks_poses=cuboid_tracks.tracks_poses,
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us,
                    tracks_flags=cuboid_tracks.tracks_flags,
                    max_track_n_poses=cuboid_tracks.max_track_n_poses,
                    tracks_label_class=cuboid_tracks.tracks_label_class,
                    max_positive_timestamp_gap_us=cuboid_tracks.max_positive_timestamp_gap_us,
                ),
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims,
                ),
            )

        @staticmethod
        def mask_by_label_classes(cuboid_tracks: CuboidTracks, label_classes: Iterable[str]) -> np.ndarray:
            label_class_set = set(label_classes)
            return np.asarray(
                [track_label_class in label_class_set for track_label_class in cuboid_tracks.tracks_label_class],
                dtype=bool,
            )

        @staticmethod
        def mask_by_min_distance(cuboid_tracks: CuboidTracks, min_distance_m: float) -> np.ndarray:
            if min_distance_m < 0.0:
                raise ValueError("min_distance_m must be non-negative")

            if cuboid_tracks.n_tracks == 0:
                return np.empty((0,), dtype=bool)

            track_mask = np.zeros((cuboid_tracks.n_tracks,), dtype=bool)
            tracks_packinfo = cuboid_tracks.tracks_packinfo.cpu().numpy()
            tracks_poses = cuboid_tracks.tracks_poses.matrix().detach().cpu().numpy()

            for track_idx in range(cuboid_tracks.n_tracks):
                start_idx, n_poses = tracks_packinfo[track_idx].tolist()
                if n_poses < 2:
                    continue
                translations = tracks_poses[start_idx : start_idx + n_poses, :3, 3]
                track_mask[track_idx] = float(np.linalg.norm(np.diff(translations, axis=0), axis=1).sum()) > (
                    min_distance_m
                )

            return track_mask

        @staticmethod
        def split_by_visible_intervals(
            cuboid_tracks: CuboidTracks,
            visible_intervals_us: list[np.ndarray],
            max_visible_duration_us: int,
            split_mask: np.ndarray | torch.Tensor | list[bool],
            merge_gap_us: int = 0,
            min_child_visible_duration_us: int = 0,
            split_suffix: str = "__split",
            split_children_out: list[TrackSplitChild] | None = None,
        ) -> CuboidTracks:
            if cuboid_tracks.n_tracks == 0:
                return cuboid_tracks
            if len(visible_intervals_us) != cuboid_tracks.n_tracks:
                raise ValueError(
                    f"visible_intervals_us length ({len(visible_intervals_us)}) does not match "
                    f"n_tracks ({cuboid_tracks.n_tracks})"
                )
            if max_visible_duration_us <= 0:
                raise ValueError("max_visible_duration_us must be positive")
            if merge_gap_us < 0:
                raise ValueError("merge_gap_us must be non-negative")
            if min_child_visible_duration_us < 0:
                raise ValueError("min_child_visible_duration_us must be non-negative")

            if isinstance(split_mask, torch.Tensor):
                split_mask_np = split_mask.detach().cpu().numpy().astype(bool)
            else:
                split_mask_np = np.asarray(split_mask, dtype=bool)
            if split_mask_np.shape != (cuboid_tracks.n_tracks,):
                raise ValueError(f"split_mask shape {split_mask_np.shape} does not match n_tracks")

            def add_track_id_suffix_before_source(track_id: str, suffix: str) -> str:
                track_id_base, source_separator, source_suffix = track_id.partition("@")
                if source_separator:
                    return f"{track_id_base}{suffix}{source_separator}{source_suffix}"
                return f"{track_id}{suffix}"

            used_track_ids: set[str] = set(cuboid_tracks.tracks_id)

            def unique_track_id(track_id: str, segment_idx: int) -> str:
                split_id = add_track_id_suffix_before_source(track_id, f"{split_suffix}{segment_idx:02d}")
                if split_id not in used_track_ids:
                    used_track_ids.add(split_id)
                    return split_id

                duplicate_idx = 1
                while True:
                    candidate = add_track_id_suffix_before_source(
                        track_id, f"{split_suffix}{segment_idx:02d}_{duplicate_idx}"
                    )
                    if candidate not in used_track_ids:
                        used_track_ids.add(candidate)
                        return candidate
                    duplicate_idx += 1

            tracks_packinfo = cuboid_tracks.tracks_packinfo.cpu().numpy()
            tracks_timestamps_us = cuboid_tracks.tracks_timestamps_us.cpu().numpy()
            tracks_poses = cuboid_tracks.tracks_poses.matrix().detach().cpu().numpy()
            tracks_flags = cuboid_tracks.tracks_flags.detach().cpu().numpy()
            cuboids_dims = cuboid_tracks.cuboids_dims.detach().cpu().numpy()

            new_tracks_id: list[str] = []
            new_tracks_poses: list[np.ndarray] = []
            new_tracks_timestamps_us: list[np.ndarray] = []
            new_tracks_label_class: list[str] = []
            new_tracks_flags: list[TrackFlags] = []
            new_cuboids_dims: list[np.ndarray] = []

            for track_idx, track_id in enumerate(cuboid_tracks.tracks_id):
                start_idx, n_poses = tracks_packinfo[track_idx].tolist()
                timestamps_us = tracks_timestamps_us[start_idx : start_idx + n_poses]
                poses = tracks_poses[start_idx : start_idx + n_poses]
                track_label_class = cuboid_tracks.tracks_label_class[track_idx]
                track_flags = TrackFlags(int(tracks_flags[track_idx]))
                cuboid_dims = cuboids_dims[track_idx].astype(np.float32)

                def append_track(output_track_id: str, output_timestamps_us: np.ndarray, output_poses: np.ndarray):
                    new_tracks_id.append(output_track_id)
                    new_tracks_poses.append(output_poses.astype(np.float32))
                    new_tracks_timestamps_us.append(output_timestamps_us.astype(np.int64))
                    new_tracks_label_class.append(track_label_class)
                    new_tracks_flags.append(track_flags)
                    new_cuboids_dims.append(cuboid_dims)

                if not split_mask_np[track_idx] or n_poses < 2:
                    append_track(track_id, timestamps_us, poses)
                    continue

                track_start_us = int(timestamps_us[0])
                track_end_us = int(timestamps_us[-1])
                visible_intervals = [
                    (max(int(start_us), track_start_us), min(int(end_us), track_end_us))
                    for start_us, end_us in visible_intervals_us[track_idx]
                    if min(int(end_us), track_end_us) > max(int(start_us), track_start_us)
                ]
                visible_intervals.sort(key=lambda interval: interval[0])
                merged_visible_intervals: list[tuple[int, int]] = []
                for interval_start_us, interval_end_us in visible_intervals:
                    if (
                        not merged_visible_intervals
                        or interval_start_us > merged_visible_intervals[-1][1] + merge_gap_us
                    ):
                        merged_visible_intervals.append((interval_start_us, interval_end_us))
                    else:
                        previous_start_us, previous_end_us = merged_visible_intervals[-1]
                        merged_visible_intervals[-1] = (previous_start_us, max(previous_end_us, interval_end_us))
                visible_intervals = merged_visible_intervals
                if not any(end_us - start_us > max_visible_duration_us for start_us, end_us in visible_intervals):
                    append_track(track_id, timestamps_us, poses)
                    continue

                segment_idx = 0
                cursor_us = track_start_us

                def append_segment(segment_start_us: int, segment_end_us: int) -> None:
                    nonlocal segment_idx
                    if segment_end_us <= segment_start_us:
                        return

                    interior_mask = (timestamps_us > segment_start_us) & (timestamps_us < segment_end_us)
                    segment_timestamps_us = np.concatenate(
                        (
                            np.asarray([segment_start_us], dtype=np.int64),
                            timestamps_us[interior_mask],
                            np.asarray([segment_end_us], dtype=np.int64),
                        )
                    )
                    segment_timestamps_tensor = torch.from_numpy(segment_timestamps_us).to(
                        device=cuboid_tracks.device,
                        dtype=cuboid_tracks.tracks_timestamps_us.dtype,
                    )
                    segment_tracks_idx = torch.full(
                        (segment_timestamps_tensor.shape[0],),
                        track_idx,
                        device=cuboid_tracks.device,
                        dtype=cuboid_tracks.tracks_packinfo.dtype,
                    )
                    segment_poses = (
                        cuboid_tracks.interpolate_tracks_poses(segment_timestamps_tensor, segment_tracks_idx)
                        .matrix()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    output_track_id = unique_track_id(track_id, segment_idx)
                    append_track(output_track_id, segment_timestamps_us, segment_poses)
                    if split_children_out is not None:
                        split_children_out.append(
                            TrackSplitChild(
                                parent_track_id=track_id,
                                child_track_id=output_track_id,
                                start_timestamp_us=int(segment_timestamps_us[0]),
                                end_timestamp_us=int(segment_timestamps_us[-1]),
                                child_index=segment_idx,
                            )
                        )
                    segment_idx += 1

                def split_visible_interval(interval_start_us: int, interval_end_us: int) -> list[tuple[int, int]]:
                    visible_segments: list[tuple[int, int]] = []
                    segment_start_us = interval_start_us
                    while segment_start_us < interval_end_us:
                        segment_end_us = min(segment_start_us + max_visible_duration_us, interval_end_us)
                        visible_segments.append((segment_start_us, segment_end_us))
                        segment_start_us = segment_end_us

                    if (
                        min_child_visible_duration_us > 0
                        and len(visible_segments) > 1
                        and visible_segments[-1][1] - visible_segments[-1][0] < min_child_visible_duration_us
                    ):
                        previous_start_us, _ = visible_segments[-2]
                        _, final_end_us = visible_segments[-1]
                        visible_segments[-2:] = [(previous_start_us, final_end_us)]

                    return visible_segments

                leading_invisible_duration_us = visible_intervals[0][0] - track_start_us
                trailing_invisible_duration_us = track_end_us - visible_intervals[-1][1]
                merge_leading_invisible = 0 < leading_invisible_duration_us < min_child_visible_duration_us
                merge_trailing_invisible = 0 < trailing_invisible_duration_us < min_child_visible_duration_us

                for interval_idx, (interval_start_us, interval_end_us) in enumerate(visible_intervals):
                    visible_segments = split_visible_interval(interval_start_us, interval_end_us)
                    if interval_idx == 0 and merge_leading_invisible:
                        _, first_segment_end_us = visible_segments[0]
                        visible_segments[0] = (track_start_us, first_segment_end_us)
                    else:
                        append_segment(cursor_us, interval_start_us)

                    if interval_idx == len(visible_intervals) - 1 and merge_trailing_invisible:
                        last_segment_start_us, _ = visible_segments[-1]
                        visible_segments[-1] = (last_segment_start_us, track_end_us)

                    for segment_start_us, segment_end_us in visible_segments:
                        append_segment(segment_start_us, segment_end_us)
                    cursor_us = interval_end_us
                if not merge_trailing_invisible:
                    append_segment(cursor_us, track_end_us)

                if segment_idx == 0:
                    append_track(track_id, timestamps_us, poses)

            return CuboidTracks.Factory.from_numpy(
                tracks_id=new_tracks_id,
                tracks_poses=new_tracks_poses,
                tracks_timestamps_us=new_tracks_timestamps_us,
                tracks_label_class=new_tracks_label_class,
                tracks_flags=new_tracks_flags,
                cuboids_dims=new_cuboids_dims,
                device=cuboid_tracks.device,
            )

        @staticmethod
        def merge_split_children(
            cuboid_tracks: CuboidTracks,
            track_split_metadata: TrackSplitMetadata | dict[str, Any] | None,
        ) -> CuboidTracks:
            metadata = (
                track_split_metadata
                if isinstance(track_split_metadata, TrackSplitMetadata)
                else TrackSplitMetadata.from_dict(track_split_metadata)
            )
            if metadata.is_empty or cuboid_tracks.n_tracks == 0:
                return cuboid_tracks

            resolver = TrackSplitResolver(metadata)
            tracks_packinfo = cuboid_tracks.tracks_packinfo.cpu().numpy()
            tracks_timestamps_us = cuboid_tracks.tracks_timestamps_us.cpu().numpy()
            tracks_poses = cuboid_tracks.tracks_poses.matrix().detach().cpu().numpy()
            tracks_flags = cuboid_tracks.tracks_flags.detach().cpu().numpy()
            cuboids_dims = cuboid_tracks.cuboids_dims.detach().cpu().numpy()

            track_id_to_idx = {track_id: idx for idx, track_id in enumerate(cuboid_tracks.tracks_id)}
            child_track_ids = {
                child.child_track_id for child in metadata.children if child.child_track_id in track_id_to_idx
            }
            parent_track_ids = {child.parent_track_id for child in metadata.children}
            if not child_track_ids:
                return cuboid_tracks

            new_tracks_id: list[str] = []
            new_tracks_poses: list[np.ndarray] = []
            new_tracks_timestamps_us: list[np.ndarray] = []
            new_tracks_label_class: list[str] = []
            new_tracks_flags: list[TrackFlags] = []
            new_cuboids_dims: list[np.ndarray] = []
            emitted_parent_track_ids: set[str] = set()

            def get_track_arrays(track_idx: int) -> tuple[np.ndarray, np.ndarray]:
                start_idx, n_poses = tracks_packinfo[track_idx].tolist()
                return (
                    tracks_timestamps_us[start_idx : start_idx + n_poses],
                    tracks_poses[start_idx : start_idx + n_poses],
                )

            def append_track(
                output_track_id: str,
                source_track_idx: int,
                output_timestamps_us: np.ndarray,
                output_poses: np.ndarray,
            ) -> None:
                new_tracks_id.append(output_track_id)
                new_tracks_poses.append(output_poses.astype(np.float32))
                new_tracks_timestamps_us.append(output_timestamps_us.astype(np.int64))
                new_tracks_label_class.append(cuboid_tracks.tracks_label_class[source_track_idx])
                new_tracks_flags.append(TrackFlags(int(tracks_flags[source_track_idx])))
                new_cuboids_dims.append(cuboids_dims[source_track_idx].astype(np.float32))

            def append_merged_parent(parent_track_id: str) -> None:
                child_ids = [
                    child_id
                    for child_id in resolver.child_ids_for_parent(parent_track_id)
                    if child_id in track_id_to_idx
                ]
                if not child_ids:
                    return

                child_timestamps = []
                child_poses = []
                for child_id in child_ids:
                    timestamps_us, poses = get_track_arrays(track_id_to_idx[child_id])
                    child_timestamps.append(timestamps_us)
                    child_poses.append(poses)

                merged_timestamps_us = np.concatenate(child_timestamps)
                merged_poses = np.concatenate(child_poses)
                sorted_indices = np.argsort(merged_timestamps_us, kind="stable")
                merged_timestamps_us = merged_timestamps_us[sorted_indices]
                merged_poses = merged_poses[sorted_indices]

                dedup_timestamps_us: list[int] = []
                dedup_poses: list[np.ndarray] = []
                for timestamp_us, pose in zip(merged_timestamps_us, merged_poses):
                    if dedup_timestamps_us and int(timestamp_us) == dedup_timestamps_us[-1]:
                        dedup_poses[-1] = pose
                    else:
                        dedup_timestamps_us.append(int(timestamp_us))
                        dedup_poses.append(pose)

                append_track(
                    output_track_id=parent_track_id,
                    source_track_idx=track_id_to_idx[child_ids[0]],
                    output_timestamps_us=np.asarray(dedup_timestamps_us, dtype=np.int64),
                    output_poses=np.stack(dedup_poses).astype(np.float32),
                )
                emitted_parent_track_ids.add(parent_track_id)

            for track_idx, track_id in enumerate(cuboid_tracks.tracks_id):
                parent_track_id = resolver.to_parent_track_id(track_id)
                if track_id in child_track_ids:
                    if parent_track_id not in emitted_parent_track_ids:
                        append_merged_parent(parent_track_id)
                    continue

                if track_id in parent_track_ids:
                    parent_has_children = any(
                        child_id in track_id_to_idx for child_id in resolver.child_ids_for_parent(track_id)
                    )
                    if parent_has_children:
                        if track_id not in emitted_parent_track_ids:
                            append_merged_parent(track_id)
                        continue

                if track_id in emitted_parent_track_ids:
                    continue

                timestamps_us, poses = get_track_arrays(track_idx)
                append_track(track_id, track_idx, timestamps_us, poses)

            return CuboidTracks.Factory.from_numpy(
                tracks_id=new_tracks_id,
                tracks_poses=new_tracks_poses,
                tracks_timestamps_us=new_tracks_timestamps_us,
                tracks_label_class=new_tracks_label_class,
                tracks_flags=new_tracks_flags,
                cuboids_dims=new_cuboids_dims,
                device=cuboid_tracks.device,
            )

    def to_device(self, device):
        return CuboidTracks(
            tracks_data=self.tracks_data.to_device(device),
            cuboidtracks_data=self.cuboidtracks_data.to_device(device),
        )

    @dataclass
    class RayIntersectionResult:
        intersections_cnt: torch.Tensor  # number of intersections of each ray, N_rays [int]
        intersections_tracks_idx: (
            torch.Tensor
        )  # for each intersection, the index of the intersected track, N_rays x max_intersections_per_ray [int]
        intersections_ts: Optional[
            torch.Tensor
        ]  # for each intersection, the start / end distances of each intersection, N_rays x max_intersections_per_ray x 2 [float]

    def ray_intersection(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_timestamps_us: torch.Tensor,
        cuboids_dims_padding: torch.Tensor | None = None,
        max_intersections_per_ray: int = 32,
        with_intersections_ts: bool = True,
    ) -> CuboidTracks.RayIntersectionResult:
        """
        Computes the intersection of all cuboid tracks with timed world rays

        Inputs:
        - rays_o: ray origins / 3d world positions, N_rays x 3 [float]
        - rays_d: normalized 3d world directions, N_rays x 3 [float]
        - rays_timestamps_us: per ray timestamp, N_rays [int64]
        - cuboids_dims_padding: if non-None, 3d padding to add to cuboids, broadcastable to N_tracks x 3 [float]
        - max_intersections_per_ray: upper limit of intersections to return [int]
        - with_intersections_ts: if True, intersections_ts will be returned, otherwise only intersections_cnt and intersections_tracks_idx will be returned [bool]

        Returns:
        - RayIntersectionResult: result of the ray intersection
        """

        cuboids_dims = self.cuboids_dims

        if cuboids_dims_padding is not None:
            cuboids_dims = cuboids_dims + cuboids_dims_padding

        intersection_result = vren.ray_cuboidtracks_intersection(
            rays_o,
            rays_d,
            rays_timestamps_us,
            self.tracks_packinfo,
            self.tracks_poses.data,
            self.tracks_timestamps_us,
            cuboids_dims,
            self.max_track_n_poses,
            max_intersections_per_ray,
            with_intersections_ts,
        )

        return CuboidTracks.RayIntersectionResult(
            intersections_cnt=intersection_result[0],
            intersections_tracks_idx=intersection_result[1],
            intersections_ts=intersection_result[2] if with_intersections_ts else None,
        )

    def point_intersection(
        self,
        points: torch.Tensor,
        timestamps_us: torch.Tensor,
        cuboids_dims_padding: torch.Tensor | None = None,
        return_dense_mask: bool = True,
    ) -> torch.Tensor:
        """
        For each point, returns the mask of the tracks that it is inside

        Inputs:
        - points: 3d world positions, ... x 3 [float]
        - timestamps_us: per point or global timestamp, [...] or [1] [int64]
        - cuboids_dims_padding: if non-None, 3d padding to add to cuboids, broadcastable to N_tracks x 3 [float]
        - return_dense_mask: If true (... x N_tracks) mask is returned else (... x 1) and true if point is inside any cuboidtrack

        Returns:
        - point_cuboidtracks_intersection_mask: for each point, the mask of the tracks that it is inside (or if it is inside any cuboidtrack if return_dense_mask is False)  ... x [1 or N_tracks] [bool]
        """

        data_shape = points.shape[:-1]
        points = points.reshape(-1, 3).contiguous()
        timestamps_us = timestamps_us.reshape(-1).contiguous()

        cuboids_dims = self.cuboids_dims

        if cuboids_dims_padding is not None:
            cuboids_dims = cuboids_dims + cuboids_dims_padding

        mask = vren.point_cuboidtracks_intersection(
            points,
            timestamps_us,
            self.tracks_packinfo,
            self.tracks_poses.data,
            self.tracks_timestamps_us,
            cuboids_dims,
            self.max_track_n_poses,
            return_dense_mask,
        )
        return mask.reshape(data_shape + (mask.shape[1],))

    @dataclass
    class PointIntersectionResult:
        intersections_cnt: torch.Tensor  # number of cuboid-tracks containing each point, [...] [int]
        interpolated_poses: lt.SE3  # per-intersection interpolated cuboid pose at the point's timestamp, [..., max_intersections_per_point] [SE3] (zeroed beyond intersections_cnt)
        intersections_tracks_idx: (
            torch.Tensor
        )  # per-intersection cuboid-track index, [..., max_intersections_per_point] [int] (-1 for unused slots)
        intersections_points_local: Optional[torch.Tensor]  # per-intersection point in the cuboid-local bbox frame,
        # [..., max_intersections_per_point, 3] [float] (zeroed beyond intersections_cnt)

    def point_intersection_interpolate_pose(
        self,
        points: torch.Tensor,
        points_timestamps_us: torch.Tensor,
        cuboids_dims_padding: torch.Tensor | None = None,
        max_intersections_per_point: int = 1,
        with_local_points: bool = False,
    ) -> CuboidTracks.PointIntersectionResult:
        """
        For each point, returns up to ``max_intersections_per_point`` interpolated track poses of the
        cuboid-tracks that contain it, alongside their track indices and (optionally) the point coordinates
        in each cuboid's local bbox frame.

        Inputs:
        - points: 3D points to check for inside check, [..., 3] [float]
        - points_timestamps_us: per point timestamp, [...] [int64]
        - cuboids_dims_padding: if non-None, 3d padding to add to cuboids, broadcastable to N_tracks x 3 [float]
        - max_intersections_per_point: upper limit of intersections recorded per point [int]
        - with_local_points: if True, ``intersections_points_local`` will be populated with the per-intersection point coordinates in the cuboid-local bbox frame, otherwise it will be ``None`` [bool]

        Returns:
        - PointIntersectionResult: result of the point intersection
        """

        data_shape = points.shape[:-1]
        points = points.reshape(-1, 3).contiguous()
        points_timestamps_us = points_timestamps_us.reshape(-1).contiguous()

        cuboids_dims = self.cuboids_dims

        if cuboids_dims_padding is not None:
            cuboids_dims = cuboids_dims + cuboids_dims_padding

        outputs = vren.point_cuboidtracks_intersection_interpolate_pose(
            points=points,
            points_timestamps_us=points_timestamps_us,
            tracks_packinfo=self.tracks_packinfo,
            tracks_poses=self.tracks_poses.data,
            tracks_timestamps_us=self.tracks_timestamps_us,
            cuboids_dims=cuboids_dims,
            max_track_n_poses=self.max_track_n_poses,
            max_intersections_per_point=max_intersections_per_point,
            return_local_points=with_local_points,
        )
        intersections_cnt = outputs[0].reshape(data_shape)
        interpolated_tracks_pose_data = outputs[1]
        intersections_tracks_idx = outputs[2].reshape(data_shape + (max_intersections_per_point,))
        interpolated_poses = lt.SE3(
            interpolated_tracks_pose_data.reshape(data_shape + interpolated_tracks_pose_data.shape[-2:])
        )
        intersections_points_local = (
            outputs[3].reshape(data_shape + (max_intersections_per_point, 3)) if with_local_points else None
        )

        return CuboidTracks.PointIntersectionResult(
            intersections_cnt=intersections_cnt,
            interpolated_poses=interpolated_poses,
            intersections_tracks_idx=intersections_tracks_idx,
            intersections_points_local=intersections_points_local,
        )

    def ray_rolling_shutter_intersection(
        self,
        pixel_idxs: torch.Tensor,
        camera_rays: torch.Tensor,
        camera_poses: torch.Tensor,
        camera_timestamp_start_us: int,
        camera_timestamp_end_us: int,
        w: int,
        h: int,
        shutter_type: int,
        cuboids_dims_padding: torch.Tensor | None = None,
        max_intersections_per_ray: int = 32,
        with_intersections_ts: bool = True,
    ) -> CuboidTracks.RayIntersectionResult:
        """
        Computes the intersection of all cuboid tracks with frame-associated camera rays

        Inputs:
        - pixel_idxs: pixel indices of camera rays (i in [0, width-1] / j in [0, height-1]), N_rays x 2 [int16]
        - camera_rays: normalized 3d directions in camera space, N_rays x 3 [float]
        - camera_poses: start / end pose of camera frame, parameterized as [3d translation / 4d quaternion] 2 x 7 [float]
        - camera_timestamp_start_us: absolute frame start timestamps [int]
        - camera_timestamp_end_us: absolute frame end timestamps [int]
        - w: frame width [int]
        - h: frame height [int]
        - shutter_type: ROLLING_TOP_TO_BOTTOM = 1, ROLLING_LEFT_TO_RIGHT = 2, ROLLING_BOTTOM_TO_TOP = 3,ROLLING_RIGHT_TO_LEFT = 4, GLOBAL = 5 [int]
        - cuboids_dims_padding: if non-None, 3d padding to add to cuboids, broadcastable to N_tracks x 3 [float]
        - max_intersections_per_ray: upper limit of intersections to return [int]
        - with_intersections_ts: if True, intersections_ts will be returned, otherwise only intersections_cnt and intersections_tracks_idx will be returned [bool]

        Returns:
        - RayIntersectionResult: result of the ray intersection
        """

        cuboids_dims = self.cuboids_dims

        if cuboids_dims_padding is not None:
            cuboids_dims = cuboids_dims + cuboids_dims_padding

        intersection_result = vren.ray_cuboidtracks_rolling_shutter_intersection(
            pixel_idxs,
            camera_rays,
            camera_poses,
            camera_timestamp_start_us,
            camera_timestamp_end_us,
            w,
            h,
            shutter_type,
            self.tracks_packinfo,
            self.tracks_poses.data,
            self.tracks_timestamps_us,
            cuboids_dims,
            self.max_track_n_poses,
            max_intersections_per_ray,
            with_intersections_ts,
        )

        return CuboidTracks.RayIntersectionResult(
            intersections_cnt=intersection_result[0],
            intersections_tracks_idx=intersection_result[1],
            intersections_ts=intersection_result[2] if with_intersections_ts else None,
        )

    def interpolate_tracks_poses(
        self,
        timestamps_us: torch.Tensor,
        tracks_idx: torch.Tensor,
    ) -> lt.SE3:
        return self.interpolate_tracks_poses_ex(timestamps_us, tracks_idx)[0]

    def interpolate_tracks_poses_ex(
        self,
        timestamps_us: torch.Tensor,
        tracks_idx: torch.Tensor,
    ) -> Tuple[lt.SE3, torch.Tensor]:
        """
        Compute the interpolated pose of the tracks at the given timestamps.
        This will not check if the timestamps are within the range of the tracks!

        Inputs:
        - timestamps_us: timestamps to interpolate the pose at, N_data [int64]
        - tracks_idx: indices of the tracks to interpolate the pose for, N_data [int]

        Returns:
        - interpolated_pose: interpolated pose of the tracks at the given timestamps, N_data [SE3]
        - interpolated_mask: mask of the tracks that are valid interpolations, N_data [bool]
        """

        tidx_right = packed_ops.packed_searchsorted_indexed_vals(
            self.tracks_timestamps_us,
            self.tracks_packinfo,
            timestamps_us,
            tracks_idx,
        )
        # Make sure right index is locally [1, N] indices
        # (usually this won't happen since we already filter in the filtering stage)
        packinfo_start = self.tracks_packinfo[tracks_idx, 0]
        packinfo_end = packinfo_start + self.tracks_packinfo[tracks_idx, 1]
        tidx_right[tidx_right == packinfo_start] += 1
        tidx_right[tidx_right == packinfo_end] -= 1

        tidx_left = tidx_right - 1
        time_diff = (self.tracks_timestamps_us[tidx_right] - self.tracks_timestamps_us[tidx_left]).to(torch.float32)
        alpha = (timestamps_us - self.tracks_timestamps_us[tidx_left]).to(torch.float32) / time_diff
        interpolated_mask = torch.logical_and(time_diff != 0, torch.logical_and(alpha >= 0, alpha <= 1))
        alpha = torch.where(interpolated_mask, alpha, torch.zeros_like(alpha))

        pose_start = self.tracks_poses[tidx_left]
        pose_end = self.tracks_poses[tidx_right]

        # Perform manifold-product interpolation (to be consistent with transform-filter)
        R_start = lt.SO3.InitFromVec(pose_start.vec()[:, 3:])
        R_end = lt.SO3.InitFromVec(pose_end.vec()[:, 3:])
        t_start, t_end = pose_start.translation(), pose_end.translation()

        R_alpha = R_start * lt.SO3.exp(alpha[:, None] * (R_start.inv() * R_end).log())
        t_alpha = t_start + alpha[:, None] * (t_end - t_start)

        interpolated_pose = lt.SE3.InitFromVec(torch.cat([t_alpha[:, :3], R_alpha.vec()], dim=1))

        return (interpolated_pose, interpolated_mask)

    @torch.autocast("cuda", enabled=False)
    def warp_world_points_to_timestamps(
        self,
        world_points: torch.Tensor,
        points_timestamps_us: torch.Tensor,
        target_timestamps_us: torch.Tensor,
        cuboids_dims_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Warp the world points to the target timestamps using the cuboid tracks.

        Inputs:
        - world_points: The world points to warp. [..., 3] (float32)
        - points_timestamps_us: The timestamps of the world points. [...,] (int64)
        - target_timestamps_us: The target timestamps to warp to. [...,] (int64)

        Returns:
        - The warped world points. [..., 3] (float32)
        """
        data_shape = world_points.shape[:-1]

        world_points = world_points.reshape(-1, 3)
        points_timestamps_us = points_timestamps_us.reshape(-1)
        target_timestamps_us = target_timestamps_us.reshape(-1)

        intersection_result = self.point_intersection_interpolate_pose(
            points=world_points,
            points_timestamps_us=points_timestamps_us,
            cuboids_dims_padding=cuboids_dims_padding,
            max_intersections_per_point=1,
            with_local_points=False,
        )
        # Only consider the (single) closest hit per point.
        interpolated_poses = lt.SE3(intersection_result.interpolated_poses.data[..., 0, :])
        interpolated_tracks_idx = intersection_result.intersections_tracks_idx[..., 0]

        # Subselect dynamic points to save computation.
        new_world_points = world_points.clone()
        dynamic_mask = torch.where(interpolated_tracks_idx != -1)[0]
        if len(dynamic_mask) > 0:
            interpolated_poses = interpolated_poses[dynamic_mask]
            current_pose = self.interpolate_tracks_poses(
                timestamps_us=target_timestamps_us[dynamic_mask],
                tracks_idx=interpolated_tracks_idx[dynamic_mask],
            )
            xyz_rel_pose = current_pose * interpolated_poses.inv()
            new_world_points[dynamic_mask] = xyz_rel_pose * new_world_points[dynamic_mask]  # type: ignore

        return new_world_points.reshape(data_shape + (3,))

    def visualize(self, frame_step: int = 4, show: bool = True) -> None:
        """
        Visualize the cuboid tracks.

        Inputs:
        - frame_step: only show every 'frame_step' pose for each track [int]
        - show: if True, show the visualization window and block the execution,
        if False, prepare visualization data without blocking and can be used jointly with update to update GUI along with program execution [bool]
        """
        visualizer = get_visualdebugger()
        visualizer.set_properties(up="z_up")

        for track_idx, track_id in enumerate(self.tracks_id):
            pose_start, pose_count = self.tracks_packinfo[track_idx]
            pose_start, pose_count = pose_start.item(), pose_count.item()
            object_dim = self.cuboids_dims[track_idx][None, :].cpu().numpy()

            # Group name based on track ID
            group = f"track-{track_id}"
            group_color = np.random.rand(3)

            for pose_idx in range(pose_start, pose_start + pose_count, frame_step):
                pose = tquat_to_se3_matrix(self.tracks_poses[pose_idx].data).cpu().numpy()
                cuboid_v = np.array(
                    [
                        [-0.5, -0.5, -0.5],
                        [0.5, -0.5, -0.5],
                        [0.5, 0.5, -0.5],
                        [-0.5, 0.5, -0.5],
                        [-0.5, -0.5, 0.5],
                        [0.5, -0.5, 0.5],
                        [0.5, 0.5, 0.5],
                        [-0.5, 0.5, 0.5],
                    ]
                )
                cuboid_v *= object_dim
                cuboid_v = np.dot(cuboid_v, pose[:3, :3].T) + pose[:3, 3]

                node_name = f"t-{track_id}-pose-{pose_idx}"
                # Transform corners by the pose
                cuboid_f = np.array(
                    [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
                )
                mesh_color = group_color * ((pose_idx - pose_start) * 0.5 / pose_count + 0.5)
                visualizer.add_surface_mesh(node_name, cuboid_v, cuboid_f, color=mesh_color, group=group)

        if show:
            visualizer.show()
