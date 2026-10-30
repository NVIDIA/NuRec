# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import numpy as np

from nurec.grpc.protos.sensorsim_pb2 import CameraRenderOutput, CameraRenderReturn
from nurec.render.novel_trajectory import render_novel_trajectory as render_novel_trajectory_module
from nurec.render.novel_trajectory.render_novel_trajectory import NovelTrajectoryRenderer, Segment


N_REQUESTS_PER_SEGMENT = 3  # 1 warmup request + 2 timed requests, so s_per_request division stays defined.


class _FakeSegment(Segment):
    def __init__(self):
        pass  # skip building real camera/controller state; only generate_requests() is exercised

    def generate_requests(self):
        return [object() for _ in range(N_REQUESTS_PER_SEGMENT)]


class _FakeClientService:
    def __init__(self, image_bytes: bytes):
        self._image_bytes = image_bytes

    async def render_camera(self, request) -> CameraRenderReturn:
        return CameraRenderReturn(camera_render_outputs=[CameraRenderOutput(image_bytes=self._image_bytes)])

    async def shut_down(self, request):
        return None


def _make_renderer(tmp_path, store_video: bool, image_bytes: bytes, monkeypatch) -> NovelTrajectoryRenderer:
    renderer = NovelTrajectoryRenderer.__new__(NovelTrajectoryRenderer)
    renderer.host = "localhost"
    renderer.port = 0
    renderer.scene_id = "test_scene"
    renderer.output_dir = str(tmp_path)
    renderer.store_video = store_video
    renderer.shutdown_server_on_completion = False
    renderer.segments = [_FakeSegment()]

    fake_client_service = _FakeClientService(image_bytes)
    monkeypatch.setattr(render_novel_trajectory_module, "SensorsimServiceStub", lambda channel: fake_client_service)

    class _FakeChannel:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *exc_info):
            return False

    monkeypatch.setattr(render_novel_trajectory_module.grpc.aio, "insecure_channel", lambda target: _FakeChannel())

    return renderer


def test_render_without_store_video_writes_jpeg_files_from_camera_render_outputs(tmp_path, monkeypatch):
    image_bytes = b"fake-jpeg-bytes"
    renderer = _make_renderer(tmp_path, store_video=False, image_bytes=image_bytes, monkeypatch=monkeypatch)

    asyncio.run(renderer._render())

    written_files = sorted(os.listdir(tmp_path))
    assert written_files == ["000000.jpeg", "000001.jpeg", "000002.jpeg"]
    for fname in written_files:
        with open(tmp_path / fname, "rb") as f:
            assert f.read() == image_bytes


def test_render_with_store_video_writes_mp4_from_camera_render_outputs(tmp_path, monkeypatch):
    image_bytes = b"fake-jpeg-bytes"
    renderer = _make_renderer(tmp_path, store_video=True, image_bytes=image_bytes, monkeypatch=monkeypatch)

    decoded_bytes = []
    monkeypatch.setattr(
        render_novel_trajectory_module.imageio,
        "imread",
        lambda b: decoded_bytes.append(b) or np.zeros((4, 4, 3), dtype=np.uint8),
    )
    mimsave_calls = []
    monkeypatch.setattr(
        render_novel_trajectory_module.imageio,
        "mimsave",
        lambda path, images, fps: mimsave_calls.append((path, images, fps)),
    )

    asyncio.run(renderer._render())

    # Every response's decoded image comes from CameraRenderReturn.camera_render_outputs[0].image_bytes,
    # not from a stale full response object (the bug this test guards against).
    assert decoded_bytes == [image_bytes] * N_REQUESTS_PER_SEGMENT
    assert len(mimsave_calls) == 1
    video_path, images, fps = mimsave_calls[0]
    assert video_path == os.path.join(str(tmp_path), "render_novel_trajectory_test_scene.mp4")
    assert len(images) == N_REQUESTS_PER_SEGMENT
    assert fps == 30
