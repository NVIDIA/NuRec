# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, ordered frame preparation for camera point-cloud initialization."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
import torchvision.transforms.functional as image_functional


if TYPE_CHECKING:
    import ncore.data

    from nurec.datasets.ncore import NCOREDataSource
    from nurec.datasets.tracks import CuboidTracks


PreparedFrame = tuple[torch.Tensor | None, torch.Tensor | None, torch.cuda.Event | None]


class CameraInitFrames:
    """Decode selected camera frames, at most eight ahead, and yield them in dataset order.

    - Without ``keep_all``, a frame is selected when a cuboid pose falls in its time window.
    - With ``keep_all``, every stepped frame is selected while any track pose exists.
    - Each frame comes with the cuboid pose indices it uses.

    Use it as a context manager: the instance owns a four-worker pool, and
    exiting shuts the pool down and cancels pending loads.

    Workers perform no sampling. Each CUDA transfer records an event, and the
    consumer records tensor lifetimes on its own stream before using the data.
    """

    def __init__(
        self,
        datasource: NCOREDataSource,
        camera_id: str,
        sensor: ncore.data.CameraSensorProtocol,
        tracks: CuboidTracks,
        step: int,
        keep_all: bool,
        color: bool,
        device: torch.device,
    ) -> None:
        timestamps = tracks.tracks_timestamps_us.cpu().numpy()
        # One array shared by every frame, so keep_all does not copy it per frame.
        all_poses = np.flatnonzero(timestamps >= 0)
        self.frames: list[tuple[int, np.ndarray]] = []
        for index in datasource.camera_frame_ranges[camera_id][::step]:
            start, end = sensor.frames_timestamps_us[index]
            cuboid_idxs = all_poses if keep_all else np.flatnonzero((timestamps >= start) & (timestamps <= end))
            if cuboid_idxs.size:
                self.frames.append((index, cuboid_idxs))
        self.datasource, self.camera_id, self.sensor = datasource, camera_id, sensor
        self.color = color
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="init-frame")
        self.queue: deque[tuple[int, np.ndarray, Future[PreparedFrame]]] = deque()
        self.iterator = iter(self.frames)

    def _load(self, index: int) -> PreparedFrame:
        _, end = self.sensor.frames_timestamps_us[index]
        assert self.datasource.aux_loader is not None
        depth_array = self.datasource.aux_loader.get_depth(self.camera_id, end)
        # Resizing nonpositive finite depth cannot produce a valid positive pixel.
        if not np.any(depth_array > 0):
            return None, None, None
        width, height = self.sensor.model_parameters.resolution
        depth_array = image_functional.resize(
            torch.from_numpy(depth_array)[None, None], [int(height), int(width)], antialias=True
        )[0, 0].numpy()
        rgb = None
        if self.color:
            if self.datasource.simplejpeg_fastdct or self.datasource.simplejpeg_fastupsample:
                # decode_image_cpu would apply these fast-JPEG flags; decode exactly so point colors ignore them.
                rgb = self.sensor.get_frame_image_array(index)
            else:
                rgb = self.datasource.decode_image_cpu(self.sensor.get_frame_handle(index))
        if self.datasource.world_to_nurec.target_scale != 1:
            depth_array = depth_array * self.datasource.world_to_nurec.target_scale
        depth = torch.from_numpy(depth_array)
        color = torch.from_numpy(rgb) if rgb is not None else None
        event = None
        if self.device.type == "cuda":
            with torch.cuda.device(self.device), torch.cuda.stream(torch.cuda.Stream(device=self.device)):
                depth = depth.pin_memory().to(self.device, non_blocking=True)
                color = color.pin_memory().to(self.device, non_blocking=True) if color is not None else None
                event = torch.cuda.current_stream(self.device).record_event()
        else:
            depth = depth.to(self.device)
            color = color.to(self.device) if color is not None else None
        return depth, color, event

    def _push(self) -> None:
        frame = next(self.iterator, None)
        if frame is not None:
            index, cuboid_idxs = frame
            self.queue.append((index, cuboid_idxs, self.pool.submit(self._load, index)))

    def __enter__(self) -> CameraInitFrames:
        for _ in range(8):
            self._push()
        return self

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, torch.Tensor | None, torch.Tensor | None]]:
        """Yield ``(index, cuboid_idxs, depth, color)`` for each selected frame, in dataset order.

        - ``cuboid_idxs`` index the flat track-pose arrays.
        - ``depth`` is a [H,W] float32 map on ``device`` at sensor resolution, in scene units.
        - ``color`` is the [H,W,3] RGB frame on ``device``, or None without ``color``.
        - A frame with no positive depth yields None for both ``depth`` and ``color``.
        - Tensors are ready to use on the consumer's current stream.
        """
        while self.queue:
            index, cuboid_idxs, future = self.queue.popleft()
            depth, color, event = future.result()
            self._push()
            if event is not None:
                assert depth is not None
                stream = torch.cuda.current_stream(self.device)
                stream.wait_event(event)
                depth.record_stream(stream)
                if color is not None:
                    color.record_stream(stream)
            yield index, cuboid_idxs, depth, color

    def __exit__(self, *exc) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.queue.clear()
