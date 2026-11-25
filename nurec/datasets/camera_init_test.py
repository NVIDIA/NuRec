# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as image_functional

from nurec.datasets.camera_init import CameraInitFrames


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_empty_depth_skips_rgb_and_prefetch_preserves_order_and_scale(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    images = {i: np.full((4, 6, 3), i, dtype=np.uint8) for i in range(3)}
    depths = {i: np.full((2, 3), i, dtype=np.float32) for i in range(3)}
    sensor = SimpleNamespace(
        frames_timestamps_us=np.array([[0, 0], [1, 1], [2, 2]]),
        model_parameters=SimpleNamespace(resolution=(6, 4)),
        get_frame_handle=lambda i: i,
    )
    ds = SimpleNamespace(
        camera_frame_ranges={"camera": range(3)},
        aux_loader=SimpleNamespace(get_depth=lambda camera, time: depths[time]),
        decode_image_cpu=Mock(side_effect=lambda i: images[i]),
        simplejpeg_fastdct=False,
        simplejpeg_fastupsample=False,
        world_to_nurec=SimpleNamespace(target_scale=2.5),
    )
    tracks = SimpleNamespace(tracks_timestamps_us=torch.tensor([0, 1, 2]))
    with CameraInitFrames(ds, "camera", sensor, tracks, 1, False, True, device) as frames:
        items = iter(frames)
        assert next(items)[::2] == (0, None)
        for index in (1, 2):
            yielded, cuboid_idxs, depth, color = next(items)
            assert yielded == index
            np.testing.assert_array_equal(cuboid_idxs, [index])
            reference = (
                image_functional.resize(torch.from_numpy(depths[index])[None, None], [4, 6], antialias=True)[0, 0] * 2.5
            )
            assert torch.equal(depth.cpu(), reference)
            np.testing.assert_array_equal(color.cpu().numpy(), images[index])
        assert next(items, None) is None
    assert ds.decode_image_cpu.call_count == 2
    assert not frames.queue


@pytest.mark.parametrize("keep_all", [False, True])
def test_frames_without_cuboids_are_not_decoded(keep_all):
    sensor = SimpleNamespace(
        frames_timestamps_us=np.array([[0, 9], [10, 19], [20, 29]]),
        model_parameters=SimpleNamespace(resolution=(3, 2)),
    )
    get_depth = Mock(return_value=np.ones((2, 3), dtype=np.float32))
    ds = SimpleNamespace(
        camera_frame_ranges={"camera": range(3)},
        aux_loader=SimpleNamespace(get_depth=get_depth),
        world_to_nurec=SimpleNamespace(target_scale=1),
    )
    tracks = SimpleNamespace(tracks_timestamps_us=torch.tensor([12, 15]))
    with CameraInitFrames(ds, "camera", sensor, tracks, 1, keep_all, False, "cpu") as frames:
        selected = [(index, cuboid_idxs.tolist()) for index, cuboid_idxs, _, _ in frames]
    assert selected == ([(0, [0, 1]), (1, [0, 1]), (2, [0, 1])] if keep_all else [(1, [0, 1])])
    indices = [index for index, _ in selected]
    assert get_depth.call_count == len(indices)


def test_decode_failure_closes_workers():
    sensor = SimpleNamespace(frames_timestamps_us=np.array([[0, 0]]))

    def fail(*args):
        raise OSError("bad depth")

    ds = SimpleNamespace(camera_frame_ranges={"camera": range(1)}, aux_loader=SimpleNamespace(get_depth=fail))
    tracks = SimpleNamespace(tracks_timestamps_us=torch.tensor([0]))
    with pytest.raises(OSError, match="bad depth"):
        with CameraInitFrames(ds, "camera", sensor, tracks, 1, False, True, "cpu") as frames:
            next(iter(frames))
    assert frames.pool._shutdown and not frames.queue


def test_empty_lidar_frames_skip_point_and_track_processing():
    from collections import defaultdict

    import pandas as pd

    from nurec.datasets.ncore import NCOREDataSource

    ds = object.__new__(NCOREDataSource)
    sensor = Mock()
    sensor.get_frame_ray_bundle_count.return_value = 0
    ds.lidar_sensors = {"lidar": sensor}
    ds.lidar_frame_ranges = {"lidar": range(3)}
    ds.lidars_frame_valid_points_masks = defaultdict(dict)
    ds.lidars_frame_non_dynamic_points_masks = defaultdict(dict)
    ds.lidar_dynamic_points_params = SimpleNamespace(method="dynamic_tracks")
    ds.cuboidtracks_dynamic = object()
    ds.cuboids_df = pd.DataFrame()
    ds._compute_cuboid_tracks = Mock()
    ds.tqdm_disabled = True
    ds._compute_lidarpoints_dynamic()
    assert len(ds.lidars_frame_non_dynamic_points_masks["lidar"]) == 3
    assert all(mask.unpacked().shape == (0,) for mask in ds.lidars_frame_non_dynamic_points_masks["lidar"].values())
    sensor.get_frame_timestamp_us.assert_not_called()
    sensor.get_frame_point_cloud.assert_not_called()


def test_camera_masks_stop_at_first_available_source(monkeypatch):
    import nurec.datasets.ncore as ncore_module

    from nurec.datasets.ncore import NCOREDataSource

    ds = object.__new__(NCOREDataSource)
    model = SimpleNamespace(
        resolution=torch.tensor([4, 4]), pixels_to_camera_rays=lambda pixels: torch.ones(len(pixels), 3)
    )
    sensor = SimpleNamespace(model_parameters=model, get_mask_images=lambda: {"ego": object()})
    ds.camera_models = {"camera": model}
    ds.camera_sensors = {"camera": sensor}
    ds.resolved_dataset = SimpleNamespace(
        backend=SimpleNamespace(get_camera_mask_and_resolution=lambda sensor: (object(), (4, 4)))
    )
    ds.camera_mask_overrides = {}
    ds.camera_mask_sources = ["dataset", "aux"]
    ds.camera_mask_border = None
    ds.n_camera_mask_dilation_iterations = 1
    ds.aux_loader = Mock()
    mask = np.zeros((4, 4), dtype=bool)
    monkeypatch.setattr(ncore_module.ncore_utils, "get_mask_image", lambda *args: mask)
    monkeypatch.setattr(ncore_module, "compute_pixel_footprint", lambda rays: np.ones(rays.shape[:2], dtype=np.float32))
    *_, actual = ds._prepare_camera_geometry("camera")
    assert actual.unpacked().all()
    ds.aux_loader.has_egomask.assert_not_called()
    ds.aux_loader.get_egomask.assert_not_called()
