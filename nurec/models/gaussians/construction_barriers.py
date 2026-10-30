# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static construction cuboids from the merged NCore detection stream."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


CONSTRUCTION_CLASSES = {"cone": 12, "barrier": 13, "pole": 16}


@dataclass(frozen=True)
class BarrierCuboids:
    """Validated full-clip static cuboids, ordered by persistent numeric track ID."""

    centers: torch.Tensor
    dimensions: torch.Tensor
    rotations: torch.Tensor
    class_ids: torch.Tensor
    track_ids: torch.Tensor

    @classmethod
    def from_ncore(cls, sequence_loader, time_range_us, world_to_nurec: np.ndarray) -> "BarrierCuboids":
        """Read static classes before the ordinary dynamic-track filters.

        Repeated observations of a track describe one fixed world-space box,
        not motion or a lifetime. Distinct overlapping tracks are preserved.
        """
        import ncore.impl.common.transformations as transformations

        observations = sequence_loader.get_cuboid_track_observations(
            timestamp_interval_us=transformations.HalfClosedInterval(start=time_range_us.start, stop=time_range_us.end)
        )

        @lru_cache(maxsize=16384)
        def to_world(reference_frame_id, timestamp_us):
            """Cache the reference-frame pose used to compare observations in world space."""
            return sequence_loader.pose_graph.evaluate_poses(
                reference_frame_id, "world", np.array(timestamp_us, dtype=np.uint64)
            )

        return cls.from_ncore_observations(observations, to_world, world_to_nurec)

    @classmethod
    def from_ncore_observations(cls, observations, to_world, world_to_nurec: np.ndarray) -> "BarrierCuboids":
        """Validate static geometry and convert world -> NuRec exactly once."""
        import ncore.impl.common.transformations as transformations

        tracks: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
        original_ids: dict[int, str] = {}
        for observation in observations:
            name = str(observation.class_id)
            if name not in CONSTRUCTION_CLASSES:
                continue
            raw_id = str(observation.track_id)
            if not raw_id.isdecimal():
                raise ValueError("Merged NCore barrier track IDs must be numeric")
            track_id = int(raw_id)
            if track_id > np.iinfo(np.int64).max or original_ids.setdefault(track_id, raw_id) != raw_id:
                raise ValueError("Ambiguous or out-of-range merged barrier track ID")
            bbox = np.asarray(observation.bbox3.to_array(), dtype=np.float64)
            bbox_world = transformations.transform_bbox(
                bbox, to_world(observation.reference_frame_id, observation.reference_frame_timestamp_us)
            )
            pose = np.asarray(transformations.bbox_pose(bbox_world), dtype=np.float64)
            dims = bbox[3:6]
            if not np.isfinite(pose).all() or not np.isfinite(dims).all() or np.any(dims <= 0):
                raise ValueError("Invalid merged barrier geometry")
            value = (CONSTRUCTION_CLASSES[name], pose, dims)
            if track_id in tracks:
                previous = tracks[track_id]
                if (
                    previous[0] != value[0]
                    or not np.allclose(previous[1], pose, rtol=0, atol=1e-4)
                    or not np.allclose(previous[2], dims, rtol=0, atol=1e-4)
                ):
                    raise ValueError(f"Merged barrier track {raw_id} is not static in world coordinates")
            else:
                tracks[track_id] = value

        transform = np.asarray(world_to_nurec, dtype=np.float64)
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
        ):
            raise ValueError("Construction-barrier initialization requires a metric rigid world_to_nurec transform")
        keys = sorted(tracks)
        poses = np.array([tracks[key][1] for key in keys], dtype=np.float64).reshape(-1, 4, 4)
        dims = np.array([tracks[key][2] for key in keys], dtype=np.float64).reshape(-1, 3)
        return cls(
            centers=torch.tensor(
                poses[:, :3, 3] @ transform[:3, :3].T + transform[:3, 3], dtype=torch.float32, device="cpu"
            ),
            dimensions=torch.tensor(dims, dtype=torch.float32, device="cpu"),
            rotations=torch.tensor(transform[:3, :3] @ poses[:, :3, :3], dtype=torch.float32, device="cpu"),
            class_ids=torch.tensor([tracks[key][0] for key in keys], dtype=torch.int64, device="cpu"),
            track_ids=torch.tensor(keys, dtype=torch.int64, device="cpu"),
        )

    def sample(self, points_per_cuboid: int, seed: int, device: torch.device) -> torch.Tensor:
        """75% class-shape samples + 25% uniform volume, exactly N per cuboid.

        This is a geometric prior, not a depth reconstruction. A local CPU RNG
        preserves the tested sample sequence without changing other layers' RNG.
        """
        if type(points_per_cuboid) is not int or points_per_cuboid <= 0:
            raise ValueError("points_per_cuboid must be a positive integer")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random = torch.rand(
            (len(self.centers), points_per_cuboid, 5), generator=generator, device="cpu", dtype=torch.float32
        )
        local = (random[..., :3] - 0.5) * self.dimensions[:, None, :]
        shape_count = points_per_cuboid - (points_per_cuboid + 3) // 4
        if shape_count:
            for index, class_id in enumerate(self.class_ids.tolist()):
                values = random[index, :shape_count]
                dims = self.dimensions[index]
                height = (torch.arange(shape_count, device="cpu") + values[:, 2]) / shape_count
                if class_id in (12, 16):
                    radius = (1 - 0.95 * height) if class_id == 12 else torch.ones_like(height)
                    radius = radius * (0.85 + 0.15 * values[:, 3])
                    angle = values[:, 4] * (2 * np.pi)
                    local[index, :shape_count, 0] = dims[0] * 0.5 * radius * angle.cos()
                    local[index, :shape_count, 1] = dims[1] * 0.5 * radius * angle.sin()
                    local[index, :shape_count, 2] = dims[2] * (height - 0.5)
                elif class_id == 13:
                    areas = torch.stack((dims[1] * dims[2], dims[0] * dims[2], dims[0] * dims[1]))
                    axis = torch.searchsorted((areas / areas.sum()).cumsum(0), values[:, 3]).clamp_max(2)
                    sign = torch.where(values[:, 4] < 0.5, -1.0, 1.0)
                    local[index, torch.arange(shape_count, device="cpu"), axis] = dims[axis] * 0.5 * sign
                else:
                    raise ValueError(f"Unsupported construction-barrier class: {class_id}")
        local = local.to(device)
        positions = (self.rotations.to(device)[:, None, :, :] * local[:, :, None, :]).sum(-1)
        return (positions + self.centers.to(device)[:, None, :]).reshape(-1, 3)
