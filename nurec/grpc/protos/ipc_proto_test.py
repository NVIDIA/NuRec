# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pins the CUDA IPC-related proto wire tags.

A tag skew between the serve-grpc server and the direct-display client silently
disables the CUDA IPC fast path (`use_cuda_ipc` parses as false) and degrades the
stream to the slow encode path instead of erroring. These tests fix the wire tags
so any future renumber fails loudly in CI rather than on the wire. No hardware needed.
"""

import nurec.grpc.protos.ipc_pb2 as ipc_pb2
import nurec.grpc.protos.sensorsim_pb2 as sensorsim_pb2


def _tag(msg_type, field_name: str) -> int:
    return msg_type.DESCRIPTOR.fields_by_name[field_name].number


def test_rgb_render_request_ipc_tags():
    req = sensorsim_pb2.RGBRenderRequest
    assert _tag(req, "use_cuda_ipc") == 22
    assert _tag(req, "ipc_session_id") == 23
    assert _tag(req, "released_buffer_id") == 24


def test_rgb_render_return_buffer_id_tag():
    # The server-selected buffer that now holds the frame.
    ret = sensorsim_pb2.RGBRenderReturn
    assert _tag(ret, "format") == 2
    assert _tag(ret, "buffer_id") == 3


def test_camera_render_request_ipc_tags():
    req = sensorsim_pb2.CameraRenderRequest
    assert _tag(req, "use_cuda_ipc") == 19
    assert _tag(req, "ipc_session_id") == 20
    assert _tag(req, "released_buffer_id") == 21


def test_camera_render_output_ipc_tags():
    out = sensorsim_pb2.CameraRenderOutput
    assert _tag(out, "format") == 7
    assert _tag(out, "buffer_id") == 8


def test_camera_render_request_roundtrip_use_cuda_ipc():
    req = sensorsim_pb2.CameraRenderRequest(use_cuda_ipc=True, ipc_session_id="s1", released_buffer_id=1)
    parsed = sensorsim_pb2.CameraRenderRequest()
    parsed.ParseFromString(req.SerializeToString())
    assert parsed.use_cuda_ipc is True
    assert parsed.ipc_session_id == "s1"
    assert parsed.HasField("released_buffer_id")
    assert parsed.released_buffer_id == 1


def test_camera_render_request_released_buffer_id_absent_by_default():
    req = sensorsim_pb2.CameraRenderRequest(use_cuda_ipc=True, ipc_session_id="s1")
    parsed = sensorsim_pb2.CameraRenderRequest()
    parsed.ParseFromString(req.SerializeToString())
    assert not parsed.HasField("released_buffer_id")


def test_rgb_render_request_roundtrip_use_cuda_ipc():
    req = sensorsim_pb2.RGBRenderRequest(use_cuda_ipc=True, ipc_session_id="s1", released_buffer_id=1)
    parsed = sensorsim_pb2.RGBRenderRequest()
    parsed.ParseFromString(req.SerializeToString())
    assert parsed.use_cuda_ipc is True
    assert parsed.ipc_session_id == "s1"
    assert parsed.HasField("released_buffer_id")
    assert parsed.released_buffer_id == 1


def test_rgb_render_request_released_buffer_id_absent_by_default():
    req = sensorsim_pb2.RGBRenderRequest(use_cuda_ipc=True, ipc_session_id="s1")
    parsed = sensorsim_pb2.RGBRenderRequest()
    parsed.ParseFromString(req.SerializeToString())
    assert not parsed.HasField("released_buffer_id")


def test_ipc_session_messages_present():
    for name in (
        "OpenIpcSessionRequest",
        "OpenIpcSessionResponse",
        "CloseIpcSessionRequest",
        "CloseIpcSessionResponse",
    ):
        assert hasattr(ipc_pb2, name), f"missing ipc proto message {name}"
