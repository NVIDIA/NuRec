# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import numpy as np
import torch
import torch.nn as nn

import nurec.utils.cli as cli

from apps.viewer.abstract_viewer import AbstractViewer
from apps.viewer.dataset_interface import CameraTrajectoryData, CameraTrajectoryId, ViewerDatasetInterface
from apps.viewer.lock import MockLock
from apps.viewer.viewpoint import LookAtPose
from ncore.data import (
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from ncore.sensors import CameraModel, camera_model_from_parameters
from nurec.config.parse import parse_typed_config
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.utils import PLYGaussianLoader, sh_degree_to_specular_dim
from nurec.render import RenderableModel
from nurec.utils.misc import to_torch
from nurec.utils.profiling import ScopedTimer
from nurec.utils.types import FrameConversion, HalfClosedInterval, PointCloud


logger = logging.getLogger(__name__)


def compute_camera_position(
    points: np.ndarray, up_vector: np.ndarray = np.array([0.0, 1.0, 0.0]), margin_factor: float = 1.2
) -> np.ndarray:
    """
    Similar to the earlier approach, but uses only a central percentile of points
    to ignore outliers when computing the bounding box.
    """
    # Get 5th and 95th percentile bounds directly
    bounds = np.percentile(points, [5.0, 95.0], axis=0)
    center = bounds.mean(axis=0)

    # Camera distance based on largest dimension
    camera_distance = margin_factor * (bounds[1] - bounds[0]).max()
    camera_position = center + np.array([0.0, 0.0, camera_distance])

    return LookAtPose(up=up_vector, look_at=center, position=camera_position).to_se3().as_matrix()


def create_default_camera_model() -> CameraModel:
    """Creates a default pinhole camera model."""
    return camera_model_from_parameters(
        OpenCVPinholeCameraModelParameters(
            resolution=np.array([1920, 1280], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([935.12, 635.05], dtype=np.float32),
            focal_length=np.array([2059.0, 2059.0], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )
    )


@dataclass
class PLYLikeTrajectoryData(CameraTrajectoryData):
    """
    Part of the dataset-viewer compatibility layer.
    """

    parent: ViewerPLYInterface

    @staticmethod
    def create(
        *,
        trajectory_id: CameraTrajectoryId,
        camera_model: CameraModel,
        camera_unique_idx: int,
        parent: ViewerPLYInterface,
        rig_time_range_us: HalfClosedInterval,
    ) -> PLYLikeTrajectoryData:
        return PLYLikeTrajectoryData(
            trajectory_id=trajectory_id,
            camera_model=camera_model,
            camera_unique_idx=camera_unique_idx,
            time_range_us=rig_time_range_us,
            average_exposure_time_us=0,
            parent=parent,
        )

    def get_poses_world(self, timestamps_us: torch.Tensor) -> torch.Tensor:
        return self.parent._get_pose_world(self.trajectory_id, timestamps_us)

    def get_closest_frame_image(self, timestamp_us: int) -> tuple[int, torch.Tensor]:
        return self.parent._get_closest_frame_and_timestamp(self.trajectory_id, timestamp_us)

    def get_frame_timestamps(self) -> torch.Tensor:
        """
        Get all frame timestamps that have corresponding images.

        Returns:
            - frame_timestamps: (n_frames, 2) tensor of (start, end) timestamps for each frame.
              Returns empty tensor if no frames are available.
        """
        return self.parent._get_frame_timestamps(self.trajectory_id)


class ViewerPLYInterface(ViewerDatasetInterface):
    """Part of the dataset interface shared with NCORE viewer (in online mode): derived from RigTrajectories and CuboidTracks"""

    def __init__(self, camera_model: CameraModel, initial_camera_pose: torch.Tensor) -> None:
        super().__init__()

        self.camera_trajectory_id = CameraTrajectoryId(
            sequence_id="PLY reconstruction",
            camera_id="default_pinhole",
            unique_camera_id="default_pinhole",
        )

        # TODO: we can make this configurable from the viewer
        self.camera_model = camera_model
        self.initial_camera_pose = initial_camera_pose
        self.trajectory_ids = [self.camera_trajectory_id]

    def get_trajectory_ids(self) -> list[CameraTrajectoryId]:
        return self.trajectory_ids

    def _get_closest_frame_and_timestamp(
        self, trajectory_id: CameraTrajectoryId, timestamp_us: int
    ) -> tuple[int, torch.Tensor]:
        raise NotImplementedError()

    def _get_frame_timestamps(self, trajectory_id: CameraTrajectoryId) -> torch.Tensor:
        """
        Get all frame timestamps for a given trajectory.

        Returns:
            - frame_timestamps: (n_frames, 2) tensor of (start, end) timestamps for each frame.
              Returns empty tensor if no frames are available.
        """
        # PLY doesn't have frame timestamps, so return empty tensor
        return torch.empty((0, 2), dtype=torch.int64)

    def supports_get_closest_frame_image(self) -> bool:
        return False

    def get_trajectory_data(self, trajectory_id: CameraTrajectoryId) -> CameraTrajectoryData:
        return PLYLikeTrajectoryData.create(
            trajectory_id=self.camera_trajectory_id,
            camera_model=self.camera_model,
            camera_unique_idx=-1,  # -1 is treated as invalid sensor id in, e.g., bilateral grid
            parent=self,
            rig_time_range_us=HalfClosedInterval(0, 1),
        )

    def get_cuboid_tracks(self) -> Optional[CuboidTracks]:
        return None

    def distance_nurec_to_distance_world(self, distance: torch.Tensor) -> torch.Tensor:
        return distance

    def _get_pose_world(
        self,
        trajectory_id: CameraTrajectoryId,
        timestamps_us: torch.Tensor,
    ) -> torch.Tensor:
        return self.initial_camera_pose

    @property
    def world_to_nurec(self) -> FrameConversion:
        """
        Get the world to frame transform from the datamodule
        """
        return FrameConversion.from_origin_scale_axis(
            target_origin=np.zeros(3, dtype=np.float32),  # put the scene's center at the origin
            target_scale=1.0,
            target_axis=[0, 1, 2],
        )

    def get_point_cloud(
        self, color_type: Optional[Literal["camera-rgb", "semantics"]] = None, step_frame: int = 10
    ) -> PointCloud | None:  # type: ignore
        return None


class PLYViewer(AbstractViewer):
    def __init__(
        self,
        ply_file_path: Path,
        config: Path,
        host: str,
        port: int,
    ) -> None:
        super().__init__()

        self.host = host
        self.port = port
        self.system_lock = MockLock()  # no need to lock the system

        self.base_config = parse_typed_config(
            config_name=str(config),
            hydra_args=["dataset.path=", "out_dir=", "mode=val"],
        )
        self.config = self.base_config.viewer
        self.ply_data = PLYGaussianLoader(ply_file_path)

        camera_model = create_default_camera_model()
        # Initialize dummy datasource state
        datasource_state = DataSourceSummary.State(
            n_frames_per_camera=np.array([1], dtype=np.int32),
            n_frames_per_lidar=np.zeros(0, dtype=np.int32),
            sequence_tracks_all=None,
            sequence_tracks_dynamic=None,
            rig_trajectories=None,
            xform_matrices=np.zeros((0, 4), dtype=np.float32),
            aabb_blb=np.zeros((1, 3), dtype=np.float32),
            aabb_trf=np.ones((1, 3), dtype=np.float32),
        )
        datasource = DataSourceSummary(state=datasource_state, datasource=None)
        model = GaussiansComposite(
            self.base_config.model, self.base_config.trainer, datasource, init_from_datasource=False
        )

        initial_camera_pose = compute_camera_position(self.ply_data.positions.cpu().numpy())
        self._datasource_interface = ViewerPLYInterface(
            camera_model=camera_model, initial_camera_pose=to_torch(initial_camera_pose[None, ...], device="cpu")
        )

        # Pick the single gaussians layer to populate from the PLY. The PLY format does not
        # carry layer assignments, so it only makes sense for single-layer configs (e.g.
        # static_gaussians, where the layer is named "background"). Fail loudly on
        # multi-layer configs so we don't silently load every gaussian into the first layer.
        layer_ids = list(self.base_config.model.layers)
        if len(layer_ids) != 1:
            raise ValueError(
                f"ply_viewer expects a single-layer model config, but got {len(layer_ids)} "
                f"layers: {layer_ids}. Use a single-layer config (e.g. "
                f"configs/model/gaussians/composite/static_gaussians.yaml) for PLY playback."
            )
        layer_id = layer_ids[0]
        num_specular_dims = sh_degree_to_specular_dim(
            int(self.base_config.model.layers[layer_id].progressive_training.max_n_features)
        )

        nodes = model.gaussians_nodes[layer_id]
        device = self.ply_data.features_albedo.device

        # Set basic parameters
        for param_name in ["positions", "rotations", "scales", "densities", "features_albedo"]:
            setattr(nodes, param_name, nn.Parameter(getattr(self.ply_data, param_name)))

        # Handle specular features
        max_n_features = int(self.base_config.model.layers[layer_id].progressive_training.max_n_features)
        if self.ply_data.features_specular is not None:
            assert self.ply_data.features_specular.shape[1] == num_specular_dims, "Specular features dimension mismatch"
            nodes.features_specular = nn.Parameter(self.ply_data.features_specular)
            active_sh_degree = max_n_features
        else:
            nodes.features_specular = nn.Parameter(
                torch.zeros(self.ply_data.features_albedo.shape[0], num_specular_dims, device=device)
            )
            # PLY has DC only (e.g. Instant-NuRec output): tell the renderer to ignore the
            # zero-padded specular bands so the DC color (features_albedo) is used directly.
            active_sh_degree = 0
        if hasattr(nodes, "n_active_features"):
            # `setattr` (not direct assignment) because `nn.Module.__setattr__` is typed for
            # Tensor/Module values only; `n_active_features` is a plain int on SHGaussianModel.
            setattr(nodes, "n_active_features", active_sh_degree)

        # The model's extra_signal tensors are declared as UninitializedParameter and only
        # filled by the training-time initializer; PLYs don't carry them. Zero-fill at the
        # shapes the collector validates against so the renderer can read them at inference.
        num_gaussians = self.ply_data.positions.shape[0]
        particle = nodes.config.particle
        lidar_sph_dim = nodes.get_lidar_extra_signal_sph_dim()
        nodes.extra_signal = nn.Parameter(torch.zeros(num_gaussians, particle.extra_signal_dim, device=device))
        nodes.camera_extra_signal = nn.Parameter(
            torch.zeros(num_gaussians, particle.camera_extra_signal_dim, device=device)
        )
        nodes.lidar_extra_signal = nn.Parameter(
            torch.zeros(num_gaussians, particle.lidar_extra_signal_dim * lidar_sph_dim, device=device)
        )

        self.model = RenderableModel(model)
        assert isinstance(self.model._model, GaussiansComposite), "Model is not a GaussiansComposite model"
        self.model._model.to("cuda")
        self.model._model.eval()

    def get_dataset_interface(self) -> ViewerPLYInterface:
        return self._datasource_interface


@click.command("ply_viewer")
@click.option(
    "--ply-path",
    type=Path,
    help="Path to the ply file.",
    required=True,
)
@click.option(
    "--config",
    type=Path,
    help="Path to the ply file.",
    required=True,
)
@click.option(
    "--host",
    type=str,
    help="Host to serve on ",
    default="127.0.0.1",
)
@click.option(
    "--port",
    type=int,
    help="Port to serve on ",
    default=8080,
)
@cli.scopedtimer_cli_options(print_func=logger.info)
def run_ply_viewer(
    ply_path: Path,
    config: Path,
    host: str,
    port: int,
) -> None:
    ply_viewer = PLYViewer(ply_path, config=config, host=host, port=port)
    ply_viewer.start_server()

    try:
        while True:
            time.sleep(0.1)  # wait for ctrl+c to stop the server
    except KeyboardInterrupt:
        ply_viewer.stop_server()

        time.sleep(2)  # give the threads time to shut down

    ScopedTimer.print_summary()
