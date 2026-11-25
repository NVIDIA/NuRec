# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pins the RDMA-related proto wire tags.

A tag skew between the serve-grpc server and the direct-display client silently
disables the RDMA fast path (`use_rdma` parses as false) and degrades the stream to
the slow encode path instead of erroring. These tests fix the wire tags so any future
renumber fails loudly in CI rather than on the wire. No hardware needed.
"""

import nurec.grpc.protos.rdma_pb2 as rdma_pb2
import nurec.grpc.protos.sensorsim_pb2 as sensorsim_pb2


def _tag(msg_type, field_name: str) -> int:
    return msg_type.DESCRIPTOR.fields_by_name[field_name].number


def test_rgb_render_request_rdma_tags():
    req = sensorsim_pb2.RGBRenderRequest
    assert _tag(req, "use_rdma") == 25
    assert _tag(req, "rdma_session_id") == 26
    assert _tag(req, "rdma_buffer_id") == 27


def test_rgb_render_return_rdma_tags():
    # format and buffer_id are shared with CUDA IPC delivery.
    ret = sensorsim_pb2.RGBRenderReturn
    assert _tag(ret, "format") == 2
    assert _tag(ret, "buffer_id") == 3
    assert _tag(ret, "bytes_written") == 4


def test_camera_render_request_rdma_tags():
    req = sensorsim_pb2.CameraRenderRequest
    assert _tag(req, "use_rdma") == 22
    assert _tag(req, "rdma_session_id") == 23
    assert _tag(req, "rdma_buffer_id") == 24


def test_camera_render_output_rdma_tags():
    # format and buffer_id are shared with CUDA IPC delivery.
    out = sensorsim_pb2.CameraRenderOutput
    assert _tag(out, "format") == 7
    assert _tag(out, "buffer_id") == 8
    assert _tag(out, "bytes_written") == 9


def test_rgb_render_request_roundtrip_use_rdma():
    req = sensorsim_pb2.RGBRenderRequest(use_rdma=True, rdma_session_id="s1", rdma_buffer_id=7)
    parsed = sensorsim_pb2.RGBRenderRequest()
    parsed.ParseFromString(req.SerializeToString())
    assert parsed.use_rdma is True
    assert parsed.rdma_session_id == "s1"
    assert parsed.rdma_buffer_id == 7
    assert parsed.use_cuda_ipc is False


def test_camera_render_request_roundtrip_use_rdma():
    req = sensorsim_pb2.CameraRenderRequest(use_rdma=True, rdma_session_id="s1", rdma_buffer_id=7)
    parsed = sensorsim_pb2.CameraRenderRequest()
    parsed.ParseFromString(req.SerializeToString())
    assert parsed.use_rdma is True
    assert parsed.rdma_session_id == "s1"
    assert parsed.rdma_buffer_id == 7
    assert parsed.use_cuda_ipc is False


def test_rdma_session_messages_present():
    for name in (
        "OpenRdmaSessionRequest",
        "OpenRdmaSessionResponse",
        "RegisterReceiveBufferRequest",
        "RegisterReceiveBufferResponse",
        "CloseRdmaSessionRequest",
        "CloseRdmaSessionResponse",
    ):
        assert hasattr(rdma_pb2, name), f"missing rdma proto message {name}"
