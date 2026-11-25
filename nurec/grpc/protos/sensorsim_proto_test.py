# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks the sensorsim gRPC API is additive over the last release snapshot in sensorsim_api.json.

Regenerate at each release: `bazel run //nurec/grpc/protos:sensorsim_api -- <tag>`.
"""

import copy
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile

from pathlib import Path
from typing import Any

import pytest

from google.protobuf import descriptor_pb2, descriptor_pool

import nurec.grpc.protos.sensorsim_pb2 as sensorsim_pb2


SNAPSHOT = Path(__file__).with_name("sensorsim_api.json")


def api(file: Any) -> dict[str, Any]:
    """Service name, RPC types, and every message and enum reachable from the SensorsimService RPCs."""
    service = file.services_by_name["SensorsimService"]
    messages: dict[str, Any] = {}
    enums: dict[str, Any] = {}

    def visit(message: Any) -> None:
        if message.full_name in messages:
            return
        messages[message.full_name] = {}  # placeholder for recursive types
        fields = {}
        for field in message.fields:
            nested = field.message_type or field.enum_type
            fields[field.name] = [
                field.number,
                descriptor_pb2.FieldDescriptorProto.Type.Name(field.type),
                descriptor_pb2.FieldDescriptorProto.Label.Name(field.label),
                nested.full_name if nested else "",
                field.containing_oneof.name if field.containing_oneof else "",
                field.has_presence,
            ]
            if field.message_type:
                visit(field.message_type)
            elif field.enum_type:
                enums[field.enum_type.full_name] = {v.name: v.number for v in field.enum_type.values}
        messages[message.full_name] = fields

    rpcs = {}
    for method in service.methods:
        rpcs[method.name] = [
            method.input_type.full_name,
            method.output_type.full_name,
            method.client_streaming,
            method.server_streaming,
        ]
        visit(method.input_type)
        visit(method.output_type)
    return {"service": service.full_name, "rpcs": rpcs, "messages": messages, "enums": enums}


def breaks(old: Any, new: Any, path: str = "") -> list[str]:
    """Every entry of `old` that `new` removes or changes; entries only in `new` are additions."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [b for key in old for b in breaks(old[key], new.get(key), f"{path}/{key}")]
    return [] if old == new else [f"{path}: {old} -> {new}"]


def test_api_is_additive():
    snapshot = json.loads(SNAPSHOT.read_text())
    source = snapshot.pop("source")
    assert breaks(snapshot, api(sensorsim_pb2.DESCRIPTOR)) == [], f"breaks the {source} API"


def _without(d: dict, key: str) -> dict:
    d.pop(key)
    return d


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: _without(a["rpcs"], "render_rgb"),
        lambda a: a["rpcs"].update(render_rgb=[a["rpcs"]["render_rgb"][0], "x.Other"]),
        lambda a: a["messages"]["nurec.grpc.protos.sensorsim.RGBRenderRequest"]["scene_id"].__setitem__(0, 99),
        lambda a: a["messages"]["nurec.grpc.protos.sensorsim.RGBRenderRequest"]["scene_id"].__setitem__(
            1, "TYPE_BYTES"
        ),
        lambda a: a.update(service="other.SensorsimService"),
        lambda a: _without(next(iter(a["enums"].values())), next(iter(next(iter(a["enums"].values()))))),
    ],
    ids=[
        "rpc removed",
        "rpc type changed",
        "field renumbered",
        "field retyped",
        "service renamed",
        "enum value removed",
    ],
)
def test_breaks_reports_incompatible_changes(mutate):
    released = api(sensorsim_pb2.DESCRIPTOR)
    current = copy.deepcopy(released)
    mutate(current)
    assert breaks(released, current)


def test_breaks_accepts_additions():
    released = api(sensorsim_pb2.DESCRIPTOR)
    current = copy.deepcopy(released)
    current["rpcs"]["new_rpc"] = ["a.B", "a.C"]
    current["messages"]["nurec.grpc.protos.sensorsim.RGBRenderRequest"]["new_field"] = [
        999,
        "TYPE_BOOL",
        "LABEL_OPTIONAL",
        "",
        "",
        False,
    ]
    assert breaks(released, current) == []


def _api_after(edit: Any) -> dict[str, Any]:
    """api() of the compiled protos after `edit` changes their FileDescriptorProtos (a dict keyed by file name)."""
    files: dict[str, descriptor_pb2.FileDescriptorProto] = {}

    def add(file: Any) -> None:
        for dependency in file.dependencies:
            add(dependency)
        if file.name not in files:
            files[file.name] = descriptor_pb2.FileDescriptorProto()
            file.CopyToProto(files[file.name])

    add(sensorsim_pb2.DESCRIPTOR)
    edit(files)
    pool = descriptor_pool.DescriptorPool()
    for proto in files.values():
        pool.Add(proto)
    return api(pool.FindFileByName(sensorsim_pb2.DESCRIPTOR.name))


def _message(files: dict, file: str, name: str) -> Any:
    return next(m for m in files[f"nurec/grpc/protos/{file}.proto"].message_type if m.name == name)


def _into_oneof(files: dict) -> None:
    pose = _message(files, "common", "Pose")
    pose.oneof_decl.add(name="pose")
    for field in pose.field:
        field.oneof_index = 0


def _drop_optional(files: dict) -> None:
    request = _message(files, "sensorsim", "RGBRenderRequest")
    field = next(f for f in request.field if f.name == "released_buffer_id")
    index = field.oneof_index
    field.ClearField("proto3_optional")
    field.ClearField("oneof_index")
    del request.oneof_decl[index]
    for other in request.field:
        if other.HasField("oneof_index") and other.oneof_index > index:
            other.oneof_index -= 1


def _stream(flag: str) -> Any:
    def edit(files: dict) -> None:
        service = files["nurec/grpc/protos/sensorsim.proto"].service[0]
        setattr(next(m for m in service.method if m.name == "render_rgb"), flag, True)

    return edit


def test_descriptor_copy_matches_compiled_api():
    assert _api_after(lambda files: None) == api(sensorsim_pb2.DESCRIPTOR)


@pytest.mark.parametrize(
    "edit",
    [_into_oneof, _drop_optional, _stream("server_streaming"), _stream("client_streaming")],
    ids=["fields moved into a oneof", "optional removed", "rpc made server-streaming", "rpc made client-streaming"],
)
def test_breaks_reports_descriptor_changes(edit):
    assert breaks(api(sensorsim_pb2.DESCRIPTOR), _api_after(edit))


def _released_api(ref: str) -> dict[str, Any]:
    repo = os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
    archive = subprocess.run(["git", "archive", ref, "nurec/grpc/protos"], cwd=repo, check=True, capture_output=True)
    with tempfile.TemporaryDirectory() as tmp:
        tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(tmp, filter="data")
        descriptor_set = f"{tmp}/set.pb"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{tmp}",
                "--include_imports",
                f"--descriptor_set_out={descriptor_set}",
                "nurec/grpc/protos/sensorsim.proto",
            ],
            check=True,
        )
        files = descriptor_pb2.FileDescriptorSet.FromString(Path(descriptor_set).read_bytes()).file
    pool = descriptor_pool.DescriptorPool()
    for f in files:
        pool.Add(f)
    return api(pool.FindFileByName("nurec/grpc/protos/sensorsim.proto"))


if __name__ == "__main__":
    ref = sys.argv[1]
    out = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")) / "nurec/grpc/protos/sensorsim_api.json"
    out.write_text(json.dumps({"source": ref, **_released_api(ref)}, indent=2, sort_keys=True) + "\n")
