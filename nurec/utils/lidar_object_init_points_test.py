# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometric and input-contract regressions for measured cone initialization points."""

import numpy as np
import pytest

from scipy.spatial import cKDTree

from nurec.utils.lidar_object_init_points import (
    combine_tracks,
    eligible_masks,
    frame_observations,
    merge_observations,
    project_scan,
    select_frames,
    select_observation,
)


def cone() -> np.ndarray:
    return np.column_stack((np.zeros(12), np.zeros(12), np.linspace(0.12, 0.82, 12)))


def select(points: np.ndarray, distance: float):
    return select_observation(
        points,
        np.full((len(points), 2), 10.0),
        np.full(len(points), distance),
        np.ones((20, 20), dtype=bool),
        cKDTree(points[:, :2]),
    )


@pytest.mark.parametrize("distance", [20, 65, 150])
def test_no_range_cap_and_missing_ground(distance: float) -> None:
    result = select(cone(), distance)
    assert result is not None
    np.testing.assert_array_equal(result, np.arange(12))


def test_successful_ground_fit_still_removes_road_returns() -> None:
    rng = np.random.default_rng(42)
    road = np.column_stack((rng.uniform(-1, 1, (800, 2)), rng.normal(0, 0.002, 800)))
    world = np.concatenate((road, cone()))
    result = select(world, 65)
    assert result is not None
    assert np.all(result >= len(road))


def test_missing_ground_does_not_accept_flat_or_oversized_surfaces() -> None:
    assert select(np.column_stack((np.linspace(0, 0.5, 12), np.zeros((12, 2)))), 65) is None
    wide = np.column_stack((np.linspace(0, 1.1, 12), np.zeros(12), np.linspace(0.12, 0.6, 12)))
    assert select(wide, 65) is None
    assert select(np.empty((0, 3)), 65) is None


def test_nearest_compact_surface_wins_and_rgb_stays_aligned() -> None:
    near, far = cone(), np.repeat(cone() + [0, 4, 0], 3, axis=0)
    world = np.concatenate((near, far))
    xy = np.tile([4.2, 5.8], (len(world), 1))
    ranges = np.r_[np.full(len(near), 65.0), np.full(len(far), 70.0)]
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5, 4] = [255, 100, 3]
    obs = frame_observations(world, xy, ranges, image, np.ones((1, 20, 20), dtype=bool), 9)
    assert len(obs) == 1 and obs[0]["scan"] == 9
    np.testing.assert_array_equal(obs[0]["xyz"], near)
    np.testing.assert_array_equal(obs[0]["rgb"], np.tile([255, 100, 3], (len(near), 1)))


def observation(scan: int, distance: float, color: int) -> dict:
    xyz = cone()
    return {
        "xyz": xyz,
        "rgb": np.full(xyz.shape, color, dtype=np.uint8),
        "ranges": np.full(len(xyz), distance),
        "scan": scan,
    }


def test_distinct_scans_and_nearest_color_deterministic_merge() -> None:
    first, second = observation(1, 65, 12), observation(2, 60, 42)
    assert not merge_observations([first, first])
    tracks = merge_observations([first, second])
    assert len(tracks) == 1
    np.testing.assert_array_equal(tracks[0]["xyz"], cone())
    assert np.all(tracks[0]["rgb"] == 42)
    for key in ("xyz", "rgb"):
        np.testing.assert_array_equal(tracks[0][key], merge_observations([first, second])[0][key])


def test_empty_init_points_contract() -> None:
    init_points = combine_tracks([])
    assert init_points.xyz_world.shape == init_points.rgb.shape == (0, 3)
    assert init_points.rgb.dtype == np.uint8


def test_frame_selection_requires_exact_unique_timestamps() -> None:
    timestamps = np.arange(10) * 100 + 99
    assert select_frames(timestamps, [999, 199]) == [1, 9]
    for requested in ([100], [199, 199], []):
        with pytest.raises(ValueError):
            select_frames(timestamps, requested)
    with pytest.raises(ValueError):
        select_frames(np.array([199, 199]), [199])
    with pytest.raises(ValueError):
        select_frames(np.empty(0), [199])


def test_mask_filters_preserve_instances_and_apply_ego_validity() -> None:
    masks = np.zeros((4, 100, 100), dtype=bool)
    masks[0, 10:30, 10:15] = True
    masks[1, 40:60, 30:35] = True
    masks[2, :60, :60] = True  # Too much image area.
    masks[3, 70:75, 70:75] = True  # Too short.
    valid = np.ones((100, 100), dtype=bool)
    assert [i for i, _ in eligible_masks(masks, valid)] == [0, 1]
    valid[10:12, 10:15] = False  # More than 5% invalid support.
    assert [i for i, _ in eligible_masks(masks, valid)] == [1]
    assert eligible_masks(np.zeros((0, 100, 100), dtype=bool), valid) == []
    with pytest.raises(ValueError):
        eligible_masks(masks.astype(np.uint8), valid)


@pytest.mark.parametrize("xy", [[0.5, 4.2], [-0.2, 4.2], [20.0, 4.2]])
def test_projection_handles_scalar_indices_and_checks_float_bounds(monkeypatch: pytest.MonkeyPatch, xy: list) -> None:
    from types import SimpleNamespace

    import torch

    from ncore.sensors import CameraModel

    projection = SimpleNamespace(image_points=torch.tensor([xy]), valid_indices=torch.tensor(0))
    monkeypatch.setattr(
        CameraModel,
        "from_parameters",
        lambda *args, **kwargs: SimpleNamespace(
            world_points_to_image_points_shutter_pose=lambda *args, **kwargs: projection
        ),
    )
    camera = SimpleNamespace(
        frames_timestamps_us=np.array([[10, 20]]),
        model_parameters=SimpleNamespace(resolution=np.array([20, 20])),
        get_frames_T_source_sensor=lambda *args, **kwargs: np.stack([np.eye(4)] * 2),
    )

    def scan(index, **kwargs):
        assert kwargs == {"motion_compensation": True, "with_start_points": True, "return_index": 0}
        return SimpleNamespace(xyz_m_end=np.array([[2.0, 0.0, 1.0]]), xyz_m_start=np.zeros((1, 3)))

    lidar = SimpleNamespace(
        get_closest_frame_index=lambda timestamp: 3,
        get_frame_point_cloud=scan,
        get_frames_T_sensor_target=lambda *args: np.eye(4),
    )
    world, pixels, ranges, scan_index = project_scan(camera, lidar, 0, (20, 20), "cpu")
    expected = 1 if xy[0] == 0.5 else 0
    assert world.shape == (expected, 3) and pixels.shape == (expected, 2) and ranges.shape == (expected,)
    assert scan_index == 3
    if expected:
        np.testing.assert_array_equal(world, [[2.0, 0.0, 1.0]])


def test_combining_tracks_preserves_positions_and_colors():
    first = {"xyz": cone(), "rgb": np.full((12, 3), 10, dtype=np.uint8)}
    second = {"xyz": cone() + [0.02, 0.0, 0.0], "rgb": np.full((12, 3), 20, dtype=np.uint8)}
    # Deduplicate within each track; combining tracks preserves group and row order.
    init_points = combine_tracks([first, second, first])
    np.testing.assert_array_equal(init_points.xyz_world, np.concatenate((first["xyz"], second["xyz"], first["xyz"])))
    np.testing.assert_array_equal(init_points.rgb[:, 0], np.repeat([10, 20, 10], 12))


@pytest.fixture
def cached_source(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nurec.utils import lidar_object_init_points as module
    from nurec.utils.ncore_utils import AuxDataWriter, AuxShardDataLoader

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:] = [10, 20, 30]
    writer = AuxDataWriter(tmp_path, "scene", "scene", 0, 1, "directory")
    writer.store_object_instance_meta("cam", [20, 20], "sam3", "test-object-model", "stub", ["cone"], [[255, 128, 0]])
    masks = np.ones((1, 20, 20), dtype=bool)
    for timestamp in [20, 40]:
        writer.store_object_instance_segmentation("cam", timestamp, np.packbits(masks), np.array([0.8]), np.array([0]))
    writer.finalize()
    aux = AuxShardDataLoader("scene", [tmp_path / "scene.zarr"])
    meta = aux.get_object_instance_segmentation_meta("cam")
    monkeypatch.setattr(aux, "get_object_instance_segmentation_meta", lambda camera: meta)
    camera = SimpleNamespace(
        frames_timestamps_us=np.array([[10, 20], [30, 40]]), get_frame_image_array=lambda index: image
    )
    lidar = SimpleNamespace(get_closest_frame_index=lambda timestamp: {20: 0, 40: 1}[timestamp])
    source = SimpleNamespace(
        sequence_id="scene",
        aux_loader=aux,
        train_camera_ids=["cam"],
        train_lidar_ids=["lidar"],
        camera_sensors={"cam": camera},
        lidar_sensors={"lidar": lidar},
        camera_frame_ranges={"cam": range(2)},
        lidar_frame_ranges={"lidar": range(2)},
        sequence_loader=SimpleNamespace(get_camera_sensor=lambda name: camera, get_lidar_sensor=lambda name: None),
    )
    monkeypatch.setattr(
        module,
        "project_scan",
        lambda camera, lidar, index, size, device: (cone(), np.full((12, 2), 10), np.full(12, 65), index),
    )
    config = SimpleNamespace(camera_id="cam", lidar_id="lidar")
    return source, config, meta, aux


@pytest.mark.parametrize("active_range", [range(1), range(0)])
def test_builder_reads_native_instances_and_active_interval(cached_source, active_range):
    from nurec.utils.lidar_object_init_points import build_lidar_cone_init_points

    source, config, _, _ = cached_source
    init_points = build_lidar_cone_init_points(source, config.camera_id, config.lidar_id, "cpu")
    np.testing.assert_array_equal(init_points.xyz_world, cone())
    np.testing.assert_array_equal(init_points.rgb, np.tile([10, 20, 30], (12, 1)))
    source.camera_frame_ranges["cam"] = active_range
    assert len(build_lidar_cone_init_points(source, config.camera_id, config.lidar_id, "cpu").xyz_world) == 0


@pytest.mark.parametrize(
    "invalid",
    ["missing", "incomplete", "image", "timestamps", "classes", "camera", "lidar"],
)
def test_builder_rejects_mismatched_inputs(cached_source, invalid):
    from nurec.utils.lidar_object_init_points import build_lidar_cone_init_points

    source, config, meta, _ = cached_source
    if invalid == "missing":
        source.aux_loader = None
    if invalid == "incomplete":
        del meta["resolution"]
    if invalid == "image":
        meta["resolution"] = [10, 20]
    if invalid == "timestamps":
        source.camera_sensors["cam"].frames_timestamps_us[1, 1] = 41
    if invalid == "classes":
        meta["thing_classes"] = ["cone", "pole"]
    if invalid == "camera":
        source.train_camera_ids = []
    if invalid == "lidar":
        config.lidar_id = "held_out_lidar"
    with pytest.raises(ValueError):
        build_lidar_cone_init_points(source, config.camera_id, config.lidar_id, "cpu")


@pytest.mark.parametrize(
    "field,value",
    [
        ("classes", np.array([1])),
        ("scores", np.array([np.nan])),
        ("instance_masks", np.zeros((2, 20, 20), dtype=np.uint8)),
    ],
)
def test_builder_rejects_malformed_instance_rows(cached_source, monkeypatch, field, value):
    from nurec.utils.lidar_object_init_points import build_lidar_cone_init_points

    source, config, _, aux = cached_source
    data = aux.get_object_instance_segmentation("cam", 20)
    data[field] = value
    monkeypatch.setattr(aux, "get_object_instance_segmentation", lambda camera, timestamp: data)
    with pytest.raises(ValueError, match="Malformed object"):
        build_lidar_cone_init_points(source, config.camera_id, config.lidar_id, "cpu")


def test_empty_predictions_are_valid_but_missing_frames_are_errors(cached_source, monkeypatch):
    from nurec.utils.lidar_object_init_points import build_lidar_cone_init_points

    source, config, _, aux = cached_source
    empty = {
        "instance_masks": np.empty((0, 20, 20), dtype=np.uint8),
        "scores": np.empty(0),
        "classes": np.empty(0, dtype=np.int32),
    }
    monkeypatch.setattr(aux, "get_object_instance_segmentation", lambda camera, timestamp: empty)
    assert len(build_lidar_cone_init_points(source, config.camera_id, config.lidar_id, "cpu").xyz_world) == 0

    def missing(camera, timestamp):
        raise KeyError("missing frame")

    monkeypatch.setattr(aux, "get_object_instance_segmentation", missing)
    with pytest.raises(KeyError, match="missing frame"):
        build_lidar_cone_init_points(source, config.camera_id, config.lidar_id, "cpu")


@pytest.mark.parametrize("sensor", ["camera", "lidar"])
def test_builder_uses_configured_sensors_and_valid_frame_ranges(cached_source, monkeypatch, sensor):
    """Honor datasource pose overrides and intervals covering both shutter endpoints."""
    from nurec.utils import lidar_object_init_points as module

    source, config, _, _ = cached_source

    def fresh_sensor_is_wrong(*args):
        raise AssertionError("Must not reacquire an unpatched sensor from the loader")

    source.sequence_loader.get_camera_sensor = fresh_sensor_is_wrong
    source.sequence_loader.get_lidar_sensor = fresh_sensor_is_wrong
    if sensor == "camera":
        source.camera_frame_ranges["cam"] = range(1, 2)
    else:
        source.lidar_frame_ranges["lidar"] = range(1, 2)
    calls = []

    def project(camera, lidar, index, size, device):
        assert camera is source.camera_sensors["cam"] and lidar is source.lidar_sensors["lidar"]
        calls.append(index)
        return cone(), np.full((12, 2), 10), np.full(12, 65), index

    monkeypatch.setattr(module, "project_scan", project)
    assert len(module.build_lidar_cone_init_points(source, config.camera_id, config.lidar_id, "cpu").xyz_world) == 0
    assert calls == [1]
