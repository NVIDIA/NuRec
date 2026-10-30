# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from omegaconf import OmegaConf

from nurec.config.dataset import CuboidTracksConfig
from nurec.config.model import ConstructionBarriersGaussiansLayerConfig, SHGaussiansLayerConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.ncore import NCOREDataSource
from nurec.datasets.utils import compute_cuboid_tracks
from nurec.models.gaussians.construction_barriers import BarrierCuboids
from nurec.models.gaussians.construction_barriers_model import ConstructionBarriersGaussianModel
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, SHGaussianModel
from nurec.models.gaussians.initializations import BaseInitialization, NoPointsFoundException
from nurec.utils.lidar_object_init_points import ConeInitPoints
from nurec.utils.types import FrameConversion, HalfClosedInterval


def observation(track_id="7", class_id="pole", center=(2.0, 3.0, 4.0), dims=(0.2, 0.4, 1.5), time=100):
    """Build a minimal timestamped NCore cuboid observation in the rig frame."""
    bbox = np.array([*center, *dims, 0.0, 0.0, 0.4])
    return SimpleNamespace(
        track_id=track_id,
        class_id=class_id,
        bbox3=SimpleNamespace(to_array=lambda: bbox),
        reference_frame_id="rig",
        reference_frame_timestamp_us=time,
    )


def boxes_from(rows, transform=None, to_world=None):
    """Load test observations with identity transforms unless explicitly overridden."""
    return BarrierCuboids.from_ncore_observations(
        rows, to_world or (lambda *_: np.eye(4)), np.eye(4) if transform is None else transform
    )


@pytest.mark.parametrize("count", [1, 64, 128, 256])
def test_exact_per_cuboid_shape_mixture_is_deterministic_and_inside_rotated_boxes(count):
    """Sampling preserves RNG state, exact track counts, bounds and class-shape coverage."""
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [10, 20, 30]
    boxes = boxes_from(
        [
            observation(),
            observation(time=900),
            observation("11", "cone"),
            observation("25", "barrier"),
            observation("8", "automobile"),
        ],
        transform,
    )
    assert boxes.track_ids.tolist() == [7, 11, 25]
    assert boxes.class_ids.tolist() == [16, 12, 13]
    torch.testing.assert_close(boxes.centers[0], torch.tensor([7.0, 22.0, 34.0]))
    rng = torch.random.get_rng_state()
    points = boxes.sample(count, 1043, torch.device("cpu"))
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(points, boxes.sample(count, 1043, torch.device("cpu")))
    assert points.shape == (3 * count, 3)
    local = (points.reshape(3, count, 3) - boxes.centers[:, None]) @ boxes.rotations
    assert torch.all(local.abs() <= boxes.dimensions[:, None] / 2 + 1e-5)
    if count >= 64:
        shape_count = count - (count + 3) // 4
        # Every height stratum contains a pole/cone shape-prior point.
        height = local[:2, :shape_count, 2] / boxes.dimensions[:2, None, 2] + 0.5
        lower = torch.arange(shape_count) / shape_count
        # World-space float32 translation may round a sample across a bin edge.
        assert torch.all(height >= lower - 2e-6)
        assert torch.all(height <= lower + 1 / shape_count + 2e-6)
        # Barrier shape samples lie on a box face, not all in its interior.
        normalized = local[2, :shape_count].abs() / (boxes.dimensions[2] / 2)
        torch.testing.assert_close(normalized.max(-1).values, torch.ones(shape_count))


def test_repeated_rig_observations_must_be_static_in_world_not_in_rig():
    """Rig motion must not make repeated observations of one world-static pole disagree."""
    first, second = observation(center=(2, 3, 4)), observation(center=(1, 3, 4), time=200)

    def to_world(_, timestamp):
        """Move the rig one meter between the two observations."""
        pose = np.eye(4)
        pose[0, 3] = (timestamp - 100) / 100
        return pose

    assert len(boxes_from([first, second], to_world=to_world).centers) == 1
    with pytest.raises(ValueError, match="not static"):
        boxes_from([first, second])


@pytest.mark.parametrize(
    "rows",
    [
        [observation(center=(2, 3, np.nan))],
        [observation(dims=(0.2, 0.4, 0))],
        [observation("invalid")],
        [observation(str(2**63))],
        [observation("7"), observation("007")],
        [observation(), observation(class_id="cone")],
        [observation(), observation(dims=(0.3, 0.4, 1.5))],
    ],
)
def test_invalid_geometry_identity_or_nonstatic_track_fails(rows):
    """Reject invalid dimensions, ambiguous IDs and inconsistent observations of a track."""
    with pytest.raises(ValueError):
        boxes_from(rows)


def test_nonmetric_transform_and_empty_input():
    """Reject scale-changing transforms while allowing an empty static detection stream."""
    with pytest.raises(ValueError, match="metric rigid"):
        boxes_from([observation()], np.diag([2, 1, 1, 1]))
    boxes = boxes_from([observation(class_id="person")])
    assert boxes.centers.shape == (0, 3)
    assert boxes.rotations.shape == (0, 3, 3)
    assert boxes.sample(256, 1043, torch.device("cpu")).shape == (0, 3)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_per_box_count(count):
    """Require a positive integer seed count for every cuboid."""
    with pytest.raises(ValueError, match="positive integer"):
        boxes_from([observation()]).sample(count, 1043, torch.device("cpu"))


@pytest.mark.parametrize("excluded", [[], ["cone", "barrier", "pole"]])
@pytest.mark.parametrize("serialize_observation", [True, False])
@pytest.mark.parametrize("classes", [["automobile", "cone", "person", "barrier", "pole"], [], ["cone", "pole"]])
def test_merged_class_split_preserves_ordinary_detection_rows(excluded, serialize_observation, classes):
    """Exclude only requested static classes without copying or reserializing box objects."""
    boxes = [{"center": [1, 2, 3]} if serialize_observation else SimpleNamespace(center=(1, 2, 3)) for _ in classes]
    frame = pd.DataFrame({"class_id": classes, "track_id": range(len(classes)), "bbox3": boxes})
    original = frame.copy(deep=True)
    loader = Mock()
    with (
        patch("nurec.datasets.utils.compute_cuboid_df", return_value=frame) as compute_df,
        patch("nurec.datasets.utils.consolidate_cuboid_tracks", return_value={}) as consolidate,
    ):
        _, filtered = compute_cuboid_tracks(
            loader,
            HalfClosedInterval(0, 1000),
            CuboidTracksConfig(excluded_class_names=excluded),
            tqdm_disabled=True,
            serialize_observation=serialize_observation,
        )
    compute_df.assert_called_once_with(
        loader, HalfClosedInterval(0, 1000), True, serialize_observation=serialize_observation
    )
    expected = frame.loc[~frame.class_id.isin(excluded)]
    pd.testing.assert_frame_equal(filtered, expected)
    pd.testing.assert_frame_equal(consolidate.call_args.kwargs["cuboids_df"], expected)
    pd.testing.assert_frame_equal(frame, original)  # Raw observations remain available to the static barrier path.
    for row in filtered.itertuples():
        assert row.bbox3 is boxes[row.track_id]  # Preserve native objects; do not reserialize during filtering.


def make_initialized_model(
    rows, init_overrides=None, conversion=None, *, source_overrides=None, layer_name="construction-barriers-gaussians"
):
    """Initialize a Gaussian layer from a small mocked NCore source."""
    source = Mock(spec=NCOREDataSource)
    source.sequence_id = "clip-test"
    source.train_camera_ids = ["cam"]
    source.train_lidar_ids = ["lidar"]
    source.world_to_nurec = conversion or FrameConversion(matrix=np.eye(4))
    source.time_range_us = HalfClosedInterval(0, 1000)
    source.cuboid_tracks_params = CuboidTracksConfig(excluded_class_names=["cone", "barrier", "pole"])
    source.sequence_loader = Mock()
    source.sequence_loader.get_cuboid_track_observations.return_value = rows
    source.sequence_loader.pose_graph.evaluate_poses.return_value = np.eye(4)
    source.aux_loader = Mock(
        has_object_instance_segmentation=Mock(return_value=True),
        get_object_instance_segmentation_meta=Mock(return_value={"method": "sam3-objects"}),
    )
    source.configure_mock(**(source_overrides or {}))
    summary = Mock(datasource=source)
    summary.get_aabb.return_value.get_extent.return_value = torch.ones(3)
    layer_type = SHGaussiansLayerConfig if layer_name == "sh-gaussians" else ConstructionBarriersGaussiansLayerConfig
    layer = layer_type(
        name=layer_name,
        fourier_features_dim=1,
        density_activation="sigmoid",
        scale_activation="exp",
        rotation_activation="normalize",
        progressive_training=OmegaConf.create({"max_n_features": 3, "increase_frequency": 1000, "increase_step": 1}),
        particle=OmegaConf.create(
            {
                "extra_signal_dim": 0,
                "camera_extra_signal_dim": 0,
                "lidar_extra_signal_dim": 0,
                "lidar_extra_signal_sph_degree": 0,
            }
        ),
    )
    trainer = TrainerConfig(
        max_epochs=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        precision="32",
    )
    init_config = OmegaConf.create(
        {
            "points_per_cuboid": 256,
            "seed": 1043,
            "default_scale": 0.03,
            "default_density": 0.5,
            "scale_multiplier": 1.0,
            "local_debug_dir": None,
        }
    )
    if init_overrides is not None:
        init_config = OmegaConf.merge(init_config, init_overrides)
    initializer = BaseInitialization.factory(
        init_config.get("name", "construction-barriers"), init_config, layer, trainer, [], []
    )
    node = BaseGaussianModel.factory(layer_name, layer, trainer, summary, True, initializer)
    node.maybe_initialize_gaussians()
    return node, initializer, summary


def test_initialization_identity_and_checkpoint_restore_without_detection_input():
    """Checkpoint restore preserves all Gaussian rows and cuboids without reloading inputs."""
    node, initializer, summary = make_initialized_model([observation(), observation("9", "cone")])
    assert node.get_num_gaussians() == 512
    assert node.barrier_object_id.tolist() == [7] * 256 + [9] * 256
    torch.testing.assert_close(node.get_scales(), torch.full_like(node.scales, 0.03))
    torch.testing.assert_close(node.get_densities(), torch.full_like(node.densities, 0.5))
    assert not hasattr(node, "mcmc_position_anchor")
    assert not hasattr(node, "barrier_seed_id")
    state = deepcopy(node.state_dict())
    summary.datasource.sequence_loader.get_cuboid_track_observations.side_effect = AssertionError("No input reread")
    restored = ConstructionBarriersGaussianModel(node.config, node.trainer_config, summary, False, None)
    restored.maybe_initialize_gaussians()
    restored.load_state_dict(state, assign=True)
    restored._validate_fields()
    for name, value in restored.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, state[name])
        else:
            assert value == state[name]
    assert restored._cuboid_constraint_cache is None
    assert initializer.cuboids is not None


def test_empty_static_stream_uses_native_optional_layer_skip():
    """No construction detections should follow the existing optional-layer skip path."""
    with pytest.raises(NoPointsFoundException):
        make_initialized_model([observation(class_id="automobile")])


def lidar_init_config(**overrides):
    """Select measured cone initialization with the production scale."""
    return {
        "name": "lidar-construction",
        "camera_id": None,
        "lidar_id": None,
        "default_scale": 0.02,
    } | overrides


def measured_init_points():
    """Provide measured positions with distinct row-aligned colors."""
    return ConeInitPoints(
        np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [2.0, 2.0, 3.0]]),
        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lidar_rows_and_checkpoint_restore(tmp_path, dtype):
    """Measured positions and colors survive transforms and standard SH checkpoint restore."""
    init_points = measured_init_points()
    matrix = np.eye(4, dtype=dtype)
    matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    matrix[:3, 3] = [10, 20, 30]
    matrix[3, 3] = 0.5
    conversion = FrameConversion(matrix=matrix)
    with patch(
        "nurec.models.gaussians.initializations.build_lidar_cone_init_points", return_value=init_points
    ) as build:
        node, initializer, summary = make_initialized_model(
            [], lidar_init_config(), conversion, layer_name="sh-gaussians"
        )
    build.assert_called_once_with(summary.datasource, "cam", "lidar", initializer.device)
    assert type(node) is SHGaussianModel
    assert not hasattr(node, "barrier_object_id")
    assert not hasattr(initializer, "cuboids")
    assert torch.equal(node.positions, torch.tensor(conversion.transform_points(init_points.xyz_world)).float())
    torch.testing.assert_close(node.get_scales(), torch.full_like(node.scales, 0.04))
    torch.testing.assert_close(node.get_densities(), torch.full_like(node.densities, 0.5))
    from nurec.models.gaussians.utils import SH2RGB

    torch.testing.assert_close(SH2RGB(node.features_albedo), torch.tensor(init_points.rgb / 255).float())
    checkpoint = tmp_path / "state.pt"
    torch.save(node.state_dict(), checkpoint)
    with patch(
        "nurec.models.gaussians.initializations.build_lidar_cone_init_points", side_effect=AssertionError("No reread")
    ):
        restored = SHGaussianModel(node.config, node.trainer_config, summary, False, None)
        restored.maybe_initialize_gaussians()
        restored.load_state_dict(torch.load(checkpoint, weights_only=False), assign=True)
        restored._validate_fields()
    for name, value in node.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    assert all(parameter.requires_grad for parameter in restored.parameters())


def test_lidar_initialization_preserves_all_accepted_init_points():
    """Dense cone initialization points initialize without a per-object quota or a whole-layer MCMC budget."""
    count = 65 * 512
    init_points = ConeInitPoints(np.zeros((count, 3)), np.full((count, 3), 128, dtype=np.uint8))
    with patch("nurec.models.gaussians.initializations.build_lidar_cone_init_points", return_value=init_points):
        node, _, _ = make_initialized_model([], lidar_init_config(), layer_name="sh-gaussians")
    assert node.get_num_gaussians() == count


@pytest.mark.parametrize(
    "field,value",
    [
        ("xyz_world", np.full((3, 3), np.nan)),
        ("rgb", np.zeros((3, 3), dtype=np.float32)),
        ("xyz_world", np.zeros((3, 2))),
        ("rgb", np.zeros((2, 3), dtype=np.uint8)),
    ],
)
def test_lidar_rejects_invalid_rows(field, value):
    """Reject invalid positions and colors that do not match the measured rows."""
    from dataclasses import replace

    with patch(
        "nurec.models.gaussians.initializations.build_lidar_cone_init_points",
        return_value=replace(measured_init_points(), **{field: value}),
    ):
        with pytest.raises(ValueError, match="cone initialization points require"):
            make_initialized_model([], lidar_init_config(), layer_name="sh-gaussians")


def test_lidar_empty_input_uses_optional_layer_skip():
    """A valid cache without supported returns follows the existing skip contract."""
    init_points = ConeInitPoints(np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8))
    with patch("nurec.models.gaussians.initializations.build_lidar_cone_init_points", return_value=init_points):
        with pytest.raises(NoPointsFoundException):
            make_initialized_model([], lidar_init_config(), layer_name="sh-gaussians")


@pytest.mark.parametrize("aux", [None, Mock(has_object_instance_segmentation=Mock(return_value=False))])
def test_lidar_missing_masks_skip_optional_layer(aux):
    with pytest.raises(NoPointsFoundException, match="SAM3"):
        make_initialized_model([], lidar_init_config(), source_overrides={"aux_loader": aux}, layer_name="sh-gaussians")


def test_lidar_unrelated_instance_masks_skip_optional_layer():
    aux = Mock(
        has_instance_segmentation=Mock(return_value=True),
        get_instance_segmentation_meta=Mock(
            return_value={"method": "mask2former", "resolution": [20, 20], "thing_classes": ["car", "person"]}
        ),
        has_object_instance_segmentation=Mock(return_value=False),
    )
    with patch("nurec.models.gaussians.initializations.build_lidar_cone_init_points") as build:
        with pytest.raises(NoPointsFoundException, match="No SAM3 cone masks"):
            make_initialized_model(
                [], lidar_init_config(), source_overrides={"aux_loader": aux}, layer_name="sh-gaussians"
            )
    build.assert_not_called()
    aux.has_object_instance_segmentation.assert_called_once_with("cam")
    aux.get_object_instance_segmentation_meta.assert_not_called()
    aux.has_instance_segmentation.assert_not_called()
    aux.get_instance_segmentation_meta.assert_not_called()


@pytest.mark.parametrize(
    "meta,error",
    [
        ({}, "missing 'method'"),
        ({"method": "sam3-objects"}, "missing camera metadata"),
        ({"method": "sam3-objects", "resolution": [20, 20], "thing_classes": ["car", "person"]}, "cone-only"),
    ],
)
def test_lidar_malformed_masks_still_fail(meta, error):
    config = lidar_init_config()
    aux = Mock(has_object_instance_segmentation=Mock(return_value=True))
    aux.get_object_instance_segmentation_meta.return_value = meta
    with pytest.raises(ValueError, match=error):
        make_initialized_model(
            [],
            config,
            layer_name="sh-gaussians",
            source_overrides={
                "aux_loader": aux,
            },
        )


def test_lidar_initialization_discovers_cone_camera_and_training_lidar():
    metadata = {"other_cam": {"method": "mask2former"}, "cone_cam": {"method": "sam3-objects"}}
    aux = Mock(
        has_object_instance_segmentation=Mock(side_effect=lambda camera: camera in metadata),
        get_object_instance_segmentation_meta=Mock(side_effect=metadata.__getitem__),
    )
    with patch(
        "nurec.models.gaussians.initializations.build_lidar_cone_init_points", return_value=measured_init_points()
    ) as build:
        _, initializer, summary = make_initialized_model(
            [],
            lidar_init_config(),
            layer_name="sh-gaussians",
            source_overrides={
                "aux_loader": aux,
                "train_camera_ids": ["no_masks", "other_cam", "cone_cam"],
                "train_lidar_ids": ["lidar_custom"],
            },
        )
    build.assert_called_once_with(summary.datasource, "cone_cam", "lidar_custom", initializer.device)


@pytest.mark.parametrize("sensor", ["camera", "lidar"])
def test_lidar_initialization_requires_override_for_ambiguous_sensors(sensor):
    source_overrides = {f"train_{sensor}_ids": ["first", "second"]}
    with patch(
        "nurec.models.gaussians.initializations.build_lidar_cone_init_points", return_value=measured_init_points()
    ) as build:
        with pytest.raises(ValueError, match=f"initializer's {sensor}_id"):
            make_initialized_model(
                [], lidar_init_config(), layer_name="sh-gaussians", source_overrides=source_overrides
            )
        build.assert_not_called()
        _, initializer, summary = make_initialized_model(
            [],
            lidar_init_config(**{f"{sensor}_id": "second"}),
            layer_name="sh-gaussians",
            source_overrides=source_overrides,
        )
    build.assert_called_once_with(
        summary.datasource,
        "second" if sensor == "camera" else "cam",
        "second" if sensor == "lidar" else "lidar",
        initializer.device,
    )


def test_lidar_initialization_without_training_lidar_skips_optional_layer():
    with patch("nurec.models.gaussians.initializations.build_lidar_cone_init_points") as build:
        with pytest.raises(NoPointsFoundException, match="No training LiDAR"):
            make_initialized_model(
                [], lidar_init_config(), layer_name="sh-gaussians", source_overrides={"train_lidar_ids": []}
            )
    build.assert_not_called()
