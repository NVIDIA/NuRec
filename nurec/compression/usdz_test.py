# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tempfile
import zipfile

from pathlib import Path

import numpy as np
import pytest
import safetensors
import safetensors.torch
import torch

import nurec.compression.usdz as usdz

from nurec.config.checkpoint import ArtifactConfig
from nurec.utils.io.artifact_cache import ArtifactCache
from nurec.utils.types import NamedSerialized


_SOURCE_ZIP_DATE_TIME = (2020, 1, 2, 3, 4, 4)


def _torch_save_bytes(obj: object) -> bytes:
    with io.BytesIO() as buff:
        torch.save(obj, buff)
        return buff.getvalue()


def _read_checkpoint(data: bytes) -> dict:
    return torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    payload = usdz._read_compact_payload(path)
    with tempfile.NamedTemporaryFile(suffix=".safetensors") as tmp:
        tmp.write(payload)
        tmp.flush()
        with safetensors.safe_open(tmp.name, framework="pt", device="cpu") as f:
            return f.metadata() or {}


def _compact_payload(
    tensors: dict[str, torch.Tensor] | None = None,
    metadata: dict[str, str] | None = None,
) -> bytes:
    return safetensors.torch.save(tensors or {}, metadata=metadata or {})


def _writestr(
    zf: zipfile.ZipFile,
    name: str,
    data: bytes,
    date_time: tuple[int, int, int, int, int, int] = _SOURCE_ZIP_DATE_TIME,
) -> None:
    info = zipfile.ZipInfo(name, date_time)
    info.compress_type = zipfile.ZIP_STORED
    zf.writestr(info, data)


def _write_compact_source(
    path: Path,
    *,
    payload: bytes | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(usdz.COMPACT_PAYLOAD_NAME, payload or _compact_payload())
        zf.writestr("parsed_config.yaml", b"version: null\n")
        zf.writestr("datasource_summary.json", json.dumps({"source": "summary"}))
        zf.writestr("rig_trajectories.json", json.dumps({"rig": []}))
        zf.writestr("sequence_tracks.json", json.dumps({"tracks": []}))


def _artifact_cache() -> ArtifactCache:
    checkpoint = {
        "state_dict": {
            "model.weight": torch.tensor([1.0]),
            "other.weight": torch.tensor([2.0]),
        },
        "load_in_place": True,
        "epoch": 3,
        "global_step": 7,
        "pytorch-lightning_version": "2.0",
    }
    return ArtifactCache(
        artifact_config=ArtifactConfig(),
        checkpoint=NamedSerialized(filename="checkpoint.ckpt", serialized=_torch_save_bytes(checkpoint)),
        parsed_config=NamedSerialized(filename="parsed_config.yaml", serialized="version: null\n"),
        datasource_summary=NamedSerialized(
            filename="datasource_summary.json",
            serialized=json.dumps({"source": "summary"}, indent=4),
        ),
        rig_trajectories=[
            NamedSerialized(filename="rig_trajectories.json", serialized=json.dumps({"rig": []}, indent=4)),
        ],
        sequence_tracks=[
            NamedSerialized(filename="sequence_tracks.json", serialized=json.dumps({"tracks": []}, indent=4)),
        ],
        metadata=NamedSerialized(filename="metadata.yaml", serialized=b"metadata"),
        meshes=[
            NamedSerialized(filename="mesh_ground.ply", serialized=b"mesh"),
        ],
    )


def test_read_compact_sidecars_reject_corrupt_json_and_yaml(tmp_path):
    bad_json_path = tmp_path / "bad_json.compact.usdz"
    with zipfile.ZipFile(bad_json_path, "w") as zf:
        zf.writestr(usdz.COMPACT_PAYLOAD_NAME, _compact_payload())
        zf.writestr("parsed_config.yaml", b"version: null\n")
        zf.writestr("datasource_summary.json", b"{")
        zf.writestr("rig_trajectories.json", b"{}")
        zf.writestr("sequence_tracks.json", b"{}")

    with pytest.raises(ValueError, match="corrupt datasource_summary.json"):
        usdz._read_compact_sidecars(bad_json_path)

    bad_yaml_path = tmp_path / "bad_yaml.compact.usdz"
    with zipfile.ZipFile(bad_yaml_path, "w") as zf:
        zf.writestr(usdz.COMPACT_PAYLOAD_NAME, _compact_payload())
        zf.writestr("parsed_config.yaml", b"version: [\n")
        zf.writestr("datasource_summary.json", b"{}")
        zf.writestr("rig_trajectories.json", b"{}")
        zf.writestr("sequence_tracks.json", b"{}")

    with pytest.raises(ValueError, match="corrupt parsed_config.yaml"):
        usdz._read_compact_sidecars(bad_yaml_path)


def test_read_compact_sidecars_reject_non_object_json(tmp_path):
    path = tmp_path / "bad_json_shape.compact.usdz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(usdz.COMPACT_PAYLOAD_NAME, _compact_payload())
        zf.writestr("parsed_config.yaml", b"version: null\n")
        zf.writestr("datasource_summary.json", b"[]")
        zf.writestr("rig_trajectories.json", b"{}")
        zf.writestr("sequence_tracks.json", b"{}")

    with pytest.raises(ValueError, match="datasource_summary.json must be a JSON object"):
        usdz._read_compact_sidecars(path)


def test_read_compact_sidecars_reject_non_object_yaml(tmp_path):
    path = tmp_path / "bad_yaml_shape.compact.usdz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(usdz.COMPACT_PAYLOAD_NAME, _compact_payload())
        zf.writestr("parsed_config.yaml", b"[]")
        zf.writestr("datasource_summary.json", b"{}")
        zf.writestr("rig_trajectories.json", b"{}")
        zf.writestr("sequence_tracks.json", b"{}")

    with pytest.raises(ValueError, match="parsed_config.yaml must be a YAML object"):
        usdz._read_compact_sidecars(path)


def test_load_compact_checkpoint_returns_unpacked_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "scene.compact.usdz"
    metadata = {
        "compact_format": usdz.gaussian_codec.FORMAT_VALUE,
        "epoch": json.dumps(3),
        "global_step": json.dumps(7),
        "pytorch-lightning_version": json.dumps("2.0"),
    }
    _write_compact_source(
        path,
        payload=_compact_payload({"encoded": torch.ones(2)}, metadata),
    )

    def fake_unpack(tensors, metadata):
        assert tensors["encoded"].device.type == "cpu"
        assert metadata["compact_format"] == usdz.gaussian_codec.FORMAT_VALUE
        return {"model.weight": tensors["encoded"].float()}

    monkeypatch.setattr(usdz.gaussian_codec, "unpack", fake_unpack)

    checkpoint = usdz.load_compact_checkpoint(path)

    assert checkpoint["load_in_place"] is False
    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 7
    assert checkpoint["pytorch-lightning_version"] == "2.0"
    assert torch.equal(checkpoint["state_dict"]["model.weight"], torch.ones(2))


def test_compact_preserved_extra_signal_tensors_reads_header_metadata(tmp_path, monkeypatch):
    path = tmp_path / "scene.compact.usdz"
    metadata = {
        "compact_format": usdz.gaussian_codec.FORMAT_VALUE,
        usdz.gaussian_codec._METADATA_PRESERVED_EXTRA_SIGNAL_TENSORS: json.dumps(["lidar_extra_signal"]),
    }
    _write_compact_source(path, payload=_compact_payload({"encoded": torch.ones(1)}, metadata))

    monkeypatch.setattr(
        usdz,
        "_read_compact_payload",
        lambda _: pytest.fail("compact signal metadata should not read tensor payload"),
    )

    preserved = usdz.compact_preserved_extra_signal_tensors(path)

    assert preserved == frozenset(("lidar_extra_signal",))


def test_checkpoint_metadata_normalizes_json_scalars():
    checkpoint = {
        "epoch": np.int64(3),
        "global_step": torch.tensor(7),
        "pytorch-lightning_version": 2,
    }
    source = usdz._CompressionSource(
        state_dict={},
        parsed_config=None,
        datasource_summary={},
        rig_trajectories={},
        sequence_tracks={},
        checkpoint_metadata=usdz._checkpoint_metadata(checkpoint),
    )

    metadata = usdz._metadata_fields(source)
    restored = usdz._make_checkpoint(metadata, {})

    assert json.loads(metadata["epoch"]) == 3
    assert json.loads(metadata["global_step"]) == 7
    assert json.loads(metadata["pytorch-lightning_version"]) == "2"
    assert restored["epoch"] == 3
    assert restored["global_step"] == 7
    assert restored["pytorch-lightning_version"] == "2"


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("epoch", torch.ones(2, dtype=torch.int64)),
        ("epoch", 1.5),
        ("global_step", True),
        ("global_step", torch.tensor(1.0)),
    ),
)
def test_checkpoint_metadata_rejects_non_integer_scalars(key, value):
    with pytest.raises(ValueError, match=key):
        usdz._checkpoint_metadata({key: value})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_load_compact_checkpoint_can_unpack_on_cuda(tmp_path, monkeypatch):
    path = tmp_path / "scene.compact.usdz"
    _write_compact_source(
        path,
        payload=_compact_payload({"encoded": torch.ones(2)}),
    )

    def fake_unpack(tensors, metadata):
        assert tensors["encoded"].device.type == "cuda"
        return {"model.weight": tensors["encoded"].float()}

    monkeypatch.setattr(usdz.gaussian_codec, "unpack", fake_unpack)

    checkpoint = usdz.load_compact_checkpoint(path, device="cuda")

    assert checkpoint["state_dict"]["model.weight"].device.type == "cuda"


def test_compact_usdz_bad_zip_is_not_masked(tmp_path):
    path = tmp_path / "scene.usdz"
    path.write_bytes(b"not a zip")

    with pytest.raises(zipfile.BadZipFile):
        usdz.compress_usdz(path, tmp_path / "out.usdz")
    with pytest.raises(zipfile.BadZipFile):
        usdz.load_compact_checkpoint(path)


def test_compress_usdz_rejects_corrupt_checkpoint(tmp_path):
    input_path = tmp_path / "scene.usdz"
    with zipfile.ZipFile(input_path, "w") as zf:
        zf.writestr("checkpoint.ckpt", b"not a checkpoint")
        zf.writestr("parsed_config.yaml", b"version: null\n")
        zf.writestr("datasource_summary.json", b"{}")
        zf.writestr("rig_trajectories.json", b"{}")
        zf.writestr("sequence_tracks.json", b"{}")

    with pytest.raises(ValueError, match="corrupt checkpoint.ckpt"):
        usdz.compress_usdz(input_path, tmp_path / "scene.compact.usdz")


def test_load_compact_checkpoint_rejects_corrupt_safetensors_payload(tmp_path):
    path = tmp_path / "scene.compact.usdz"
    payload = _compact_payload({"encoded": torch.ones(2)}, {"compact_format": usdz.gaussian_codec.FORMAT_VALUE})
    _write_compact_source(path, payload=payload[:-1])

    with pytest.raises(ValueError, match="corrupt compact.safetensors"):
        usdz.load_compact_checkpoint(path)


def test_compress_usdz_rejects_compact_input(tmp_path):
    input_path = tmp_path / "scene.compact.usdz"
    _write_compact_source(input_path)

    with pytest.raises(ValueError, match="already compact"):
        usdz.compress_usdz(input_path, tmp_path / "out.usdz")


def test_compress_usdz_rejects_normal_input_written_in_place(tmp_path):
    input_path = tmp_path / "scene.usdz"
    with zipfile.ZipFile(input_path, "w") as zf:
        zf.writestr("checkpoint.ckpt", b"checkpoint")

    with pytest.raises(ValueError, match="input and output must be different files"):
        usdz.compress_usdz(input_path, input_path)


def test_compress_usdz_rejects_same_path_before_zip_inspection(tmp_path):
    input_path = tmp_path / "scene.usdz"
    input_path.write_bytes(b"not a zip")

    with pytest.raises(ValueError, match="input and output must be different files"):
        usdz.compress_usdz(input_path, input_path)


def test_compress_usdz_loads_normal_artifact_and_writes_compact_archive(tmp_path, monkeypatch):
    input_path = tmp_path / "scene.usdz"
    output_path = tmp_path / "scene.compact.usdz"
    checkpoint = {
        "state_dict": {
            "model.weight": torch.tensor([1.0]),
            "other.weight": torch.tensor([2.0]),
        },
        "load_in_place": True,
        "epoch": 3,
        "global_step": 7,
        "pytorch-lightning_version": "2.0",
    }
    with zipfile.ZipFile(input_path, "w") as zf:
        zf.writestr("checkpoint.ckpt", _torch_save_bytes(checkpoint))
        zf.writestr("parsed_config.yaml", b"version: null\n")
        zf.writestr("datasource_summary.json", json.dumps({"source": "summary"}, indent=4))
        zf.writestr("rig_trajectories.json", json.dumps({"rig": []}, indent=4))
        zf.writestr("sequence_tracks.json", json.dumps({"tracks": []}, indent=4))
        _writestr(zf, "metadata.yaml", b"metadata")
        zf.writestr("scene.usda", b"usda")

    def fake_pack(state_dict, *, opacity_threshold, compression_level, preserve_extra_signal_tensors):
        assert set(state_dict) == {"model.weight"}
        assert torch.equal(state_dict["model.weight"], torch.tensor([2.0]))
        assert opacity_threshold == 0.2
        assert compression_level == 9
        assert preserve_extra_signal_tensors == ("lidar_extra_signal", "camera_extra_signal")
        return {"packed": torch.zeros(1)}, {"compact_format": "test/v2"}

    def fake_upgrade_model(model, config, datasource_summary=None):
        assert set(model) == {"weight"}
        assert torch.equal(model["weight"], torch.tensor([1.0]))
        return {"weight": model["weight"] + 1.0}

    monkeypatch.setattr(usdz.gaussian_codec, "pack", fake_pack)
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_config", lambda cfg: cfg)
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_model", fake_upgrade_model)
    monkeypatch.setattr(
        usdz.nurec.datasets.summary.DataSourceSummary,
        "from_dict",
        staticmethod(lambda data, infer_missing=False: object()),
    )

    result = usdz.compress_usdz(
        input_path,
        opacity_threshold=0.2,
        compression_level=9,
        preserve_extra_signal_tensors=("lidar_extra_signal", "camera_extra_signal"),
        exclude_patterns=("*.usda", "*.yaml"),
    )

    assert result == output_path
    with zipfile.ZipFile(output_path, "r") as zf:
        names = set(zf.namelist())
        assert zf.namelist()[-1] == usdz.COMPACT_PAYLOAD_NAME
        assert usdz.COMPACT_PAYLOAD_NAME in names
        assert "checkpoint.ckpt" not in names
        assert "scene.usda" not in names
        assert "metadata.yaml" in names
        assert zf.getinfo("metadata.yaml").date_time == _SOURCE_ZIP_DATE_TIME
        for name in (
            "parsed_config.yaml",
            "datasource_summary.json",
            "rig_trajectories.json",
            "sequence_tracks.json",
            usdz.COMPACT_PAYLOAD_NAME,
        ):
            assert zf.getinfo(name).date_time == usdz._GENERATED_ZIP_DATE_TIME
        assert json.loads(zf.read("datasource_summary.json")) == {"source": "summary"}
        assert zf.read("rig_trajectories.json") == b'{"rig":[]}'
        assert zf.read("sequence_tracks.json") == b'{"tracks":[]}'
    metadata = _read_safetensors_metadata(output_path)

    assert "parsed_config" not in metadata
    assert "datasource_summary" not in metadata
    assert "rig_trajectories" not in metadata
    assert "sequence_tracks" not in metadata
    assert "load_in_place" not in metadata
    assert json.loads(metadata["epoch"]) == 3
    assert json.loads(metadata["global_step"]) == 7
    assert json.loads(metadata["pytorch-lightning_version"]) == "2.0"


def test_compress_usdz_rejects_checkpoint_without_state_dict(tmp_path, monkeypatch):
    input_path = tmp_path / "scene.usdz"
    with zipfile.ZipFile(input_path, "w") as zf:
        zf.writestr("checkpoint.ckpt", _torch_save_bytes({"static_layer": {}}))
        zf.writestr("parsed_config.yaml", b"version: null\n")
        zf.writestr("datasource_summary.json", b"{}")
        zf.writestr("rig_trajectories.json", b"{}")
        zf.writestr("sequence_tracks.json", b"{}")

    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_config", lambda cfg: cfg)

    monkeypatch.setattr(
        usdz.nurec.datasets.summary.DataSourceSummary,
        "from_dict",
        staticmethod(lambda data, infer_missing=False: object()),
    )

    with pytest.raises(ValueError, match="checkpoint.ckpt with a state_dict"):
        usdz.compress_usdz(input_path, tmp_path / "scene.compact.usdz")


def test_compress_artifact_cache_writes_compact_archive(tmp_path, monkeypatch):
    artifact_cache = _artifact_cache()
    output_path = tmp_path / "scene.compact.usdz"

    def fake_pack(state_dict, *, opacity_threshold, compression_level, preserve_extra_signal_tensors):
        assert set(state_dict) == {"model.weight"}
        assert torch.equal(state_dict["model.weight"], torch.tensor([2.0]))
        assert opacity_threshold == 0.2
        assert compression_level == 9
        assert preserve_extra_signal_tensors == ("lidar_extra_signal", "camera_extra_signal")
        return {"packed": torch.zeros(1)}, {"compact_format": "test/v2"}

    def fake_upgrade_model(model, config, datasource_summary=None):
        assert set(model) == {"weight"}
        assert torch.equal(model["weight"], torch.tensor([1.0]))
        return {"weight": model["weight"] + 1.0}

    monkeypatch.setattr(usdz.gaussian_codec, "pack", fake_pack)
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_config", lambda cfg: cfg)
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_model", fake_upgrade_model)
    monkeypatch.setattr(
        usdz.nurec.datasets.summary.DataSourceSummary,
        "from_dict",
        staticmethod(lambda data, infer_missing=False: object()),
    )

    result = usdz.compress_artifact_cache(
        artifact_cache,
        output_path,
        opacity_threshold=0.2,
        compression_level=9,
        preserve_extra_signal_tensors=("lidar_extra_signal", "camera_extra_signal"),
        exclude_patterns=("*.ply", "*.yaml"),
    )

    assert result == output_path
    with zipfile.ZipFile(output_path, "r") as zf:
        names = set(zf.namelist())
        assert zf.namelist()[-1] == usdz.COMPACT_PAYLOAD_NAME
        assert usdz.COMPACT_PAYLOAD_NAME in names
        assert "checkpoint.ckpt" not in names
        assert "mesh_ground.ply" not in names
        assert "metadata.yaml" in names
        assert "default.usda" in names
        assert zf.read("rig_trajectories.json") == b'{"rig":[]}'
        assert zf.read("sequence_tracks.json") == b'{"tracks":[]}'
    metadata = _read_safetensors_metadata(output_path)

    assert json.loads(metadata["epoch"]) == 3
    assert json.loads(metadata["global_step"]) == 7
    assert json.loads(metadata["pytorch-lightning_version"]) == "2.0"


def test_compress_artifact_cache_preserves_optional_members_by_default(tmp_path, monkeypatch):
    artifact_cache = _artifact_cache()
    output_path = tmp_path / "scene.compact.usdz"

    monkeypatch.setattr(usdz.gaussian_codec, "pack", lambda *args, **kwargs: ({}, {"compact_format": "test/v2"}))
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_config", lambda cfg: cfg)
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_model", lambda model, config, datasource_summary=None: model)
    monkeypatch.setattr(
        usdz.nurec.datasets.summary.DataSourceSummary,
        "from_dict",
        staticmethod(lambda data, infer_missing=False: object()),
    )

    usdz.compress_artifact_cache(artifact_cache, output_path)

    with zipfile.ZipFile(output_path, "r") as zf:
        names = set(zf.namelist())

    assert "metadata.yaml" in names
    assert "mesh_ground.ply" in names


def test_compress_artifact_cache_reports_missing_required_sidecar(tmp_path):
    artifact_cache = _artifact_cache()
    artifact_cache.rig_trajectories = None

    with pytest.raises(ValueError, match="artifact cache is missing rig_trajectories.json"):
        usdz.compress_artifact_cache(artifact_cache, tmp_path / "scene.compact.usdz")


def test_compress_artifact_cache_reports_corrupt_required_sidecars(tmp_path):
    artifact_cache = _artifact_cache()
    artifact_cache.datasource_summary = NamedSerialized(filename="datasource_summary.json", serialized=b"{")

    with pytest.raises(ValueError, match="artifact cache has corrupt datasource_summary.json"):
        usdz.compress_artifact_cache(artifact_cache, tmp_path / "bad_json.compact.usdz")

    artifact_cache = _artifact_cache()
    artifact_cache.parsed_config = NamedSerialized(filename="parsed_config.yaml", serialized="version: [\n")

    with pytest.raises(ValueError, match="artifact cache has corrupt parsed_config.yaml"):
        usdz.compress_artifact_cache(artifact_cache, tmp_path / "bad_yaml.compact.usdz")


def test_compress_artifact_cache_rejects_non_object_sidecar(tmp_path):
    artifact_cache = _artifact_cache()
    artifact_cache.datasource_summary = NamedSerialized(filename="datasource_summary.json", serialized=b"[]")

    with pytest.raises(ValueError, match="datasource_summary.json must be a JSON object"):
        usdz.compress_artifact_cache(artifact_cache, tmp_path / "bad_shape.compact.usdz")


def test_compress_artifact_cache_rejects_corrupt_checkpoint(tmp_path):
    artifact_cache = _artifact_cache()
    artifact_cache.checkpoint = NamedSerialized(filename="checkpoint.ckpt", serialized=b"not a checkpoint")

    with pytest.raises(ValueError, match="artifact cache has corrupt checkpoint.ckpt"):
        usdz.compress_artifact_cache(artifact_cache, tmp_path / "scene.compact.usdz")


def test_compress_artifact_cache_rejects_checkpoint_without_state_dict(tmp_path, monkeypatch):
    artifact_cache = _artifact_cache()
    artifact_cache.checkpoint = NamedSerialized(
        filename="checkpoint.ckpt",
        serialized=_torch_save_bytes({"static_layer": {}}),
    )
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_config", lambda cfg: cfg)
    monkeypatch.setattr(
        usdz.nurec.datasets.summary.DataSourceSummary,
        "from_dict",
        staticmethod(lambda data, infer_missing=False: object()),
    )

    with pytest.raises(ValueError, match="checkpoint.ckpt with a state_dict"):
        usdz.compress_artifact_cache(artifact_cache, tmp_path / "scene.compact.usdz")


def test_write_artifact_cache_uses_requested_path_when_enabled(tmp_path, monkeypatch):
    artifact_cache = _artifact_cache()
    artifact_cache.artifact_config.compression.enabled = True
    artifact_cache.artifact_config.compression.level = 4
    artifact_cache.artifact_config.compression.opacity_threshold = 0.2
    artifact_cache.artifact_config.compression.preserve_extra_signal_tensors = ["lidar_extra_signal"]
    artifact_cache.artifact_config.compression.exclude = ["*.ply"]
    output_path = tmp_path / "scene.usdz"

    calls = []

    def fake_compress(
        cache,
        path,
        *,
        opacity_threshold,
        compression_level,
        preserve_extra_signal_tensors,
        exclude_patterns,
    ):
        calls.append(
            (
                cache,
                path,
                opacity_threshold,
                compression_level,
                preserve_extra_signal_tensors,
                exclude_patterns,
            )
        )
        path.write_bytes(b"compact")
        return path

    monkeypatch.setattr(usdz, "compress_artifact_cache", fake_compress)

    written_path = usdz.write_artifact_cache(artifact_cache, output_path)

    assert written_path == output_path
    assert calls == [(artifact_cache, written_path, 0.2, 4, ["lidar_extra_signal"], ("*.ply",))]


def test_write_artifact_cache_uses_normal_writer_when_disabled(tmp_path):
    artifact_cache = _artifact_cache()
    output_path = tmp_path / "scene.usdz"

    written_path = usdz.write_artifact_cache(artifact_cache, output_path)

    assert written_path == output_path
    with zipfile.ZipFile(output_path, "r") as zf:
        assert "checkpoint.ckpt" in zf.namelist()


def test_compress_usdz_preserves_in_place_output_if_write_fails(tmp_path, monkeypatch):
    input_path = tmp_path / "scene.usdz"
    output_path = tmp_path / "scene.compact.usdz"
    output_path.write_bytes(b"old output")
    checkpoint = {"state_dict": {"model.weight": torch.ones(1)}}
    with zipfile.ZipFile(input_path, "w") as zf:
        zf.writestr("checkpoint.ckpt", _torch_save_bytes(checkpoint))
        zf.writestr("parsed_config.yaml", b"version: null\n")
        zf.writestr("datasource_summary.json", b"{}")
        zf.writestr("rig_trajectories.json", b"{}")
        zf.writestr("sequence_tracks.json", b"{}")
        zf.writestr("kept.bin", b"kept")

    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_config", lambda cfg: cfg)
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_model", lambda model, config, datasource_summary=None: model)
    monkeypatch.setattr(
        usdz.gaussian_codec,
        "pack",
        lambda state_dict, *, opacity_threshold, compression_level, preserve_extra_signal_tensors: (
            {"encoded": torch.zeros(1)},
            {"compact_format": usdz.gaussian_codec.FORMAT_VALUE},
        ),
    )
    monkeypatch.setattr(
        usdz.nurec.datasets.summary.DataSourceSummary,
        "from_dict",
        staticmethod(lambda data, infer_missing=False: object()),
    )

    def fail_write(input_path, output_path, payload, *, generated_members, exclude_patterns):
        output_path.write_bytes(b"partial archive")
        raise RuntimeError("write failed")

    monkeypatch.setattr(usdz, "_write_compact_archive", fail_write)

    with pytest.raises(RuntimeError, match="write failed"):
        usdz.compress_usdz(input_path, output_path, compression_level=6, opacity_threshold=0.1)

    assert output_path.read_bytes() == b"old output"
    assert not list(tmp_path.glob(".scene.compact.usdz.*.tmp"))


def test_decompress_usdz_restores_checkpoint_and_sidecars(tmp_path, monkeypatch):
    input_path = tmp_path / "scene.compact.usdz"
    output_path = tmp_path / "scene.restored.usdz"
    metadata = {
        "compact_format": usdz.gaussian_codec.FORMAT_VALUE,
        "epoch": json.dumps(3),
        "global_step": json.dumps(7),
        "pytorch-lightning_version": json.dumps("2.0"),
    }
    _write_compact_source(
        input_path,
        payload=_compact_payload({"encoded": torch.ones(1)}, metadata),
    )
    with zipfile.ZipFile(input_path, "a") as zf:
        _writestr(zf, "metadata.yaml", b"metadata")
        _writestr(zf, "kept.bin", b"kept")

    def fake_unpack(tensors, metadata):
        assert set(tensors) == {"encoded"}
        assert torch.equal(tensors["encoded"], torch.ones(1))
        return {"model.weight": torch.tensor([5.0])}

    monkeypatch.setattr(usdz.gaussian_codec, "unpack", fake_unpack)
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_config", lambda cfg: cfg)

    usdz.decompress_usdz(input_path, output_path)

    with zipfile.ZipFile(output_path, "r") as zf:
        names = set(zf.namelist())
        assert "checkpoint.ckpt" in names
        assert "metadata.yaml" in names
        assert "kept.bin" in names
        assert zf.getinfo("checkpoint.ckpt").date_time == usdz._GENERATED_ZIP_DATE_TIME
        assert zf.getinfo("parsed_config.yaml").date_time == usdz._GENERATED_ZIP_DATE_TIME
        assert zf.getinfo("datasource_summary.json").date_time == usdz._GENERATED_ZIP_DATE_TIME
        assert zf.getinfo("metadata.yaml").date_time == _SOURCE_ZIP_DATE_TIME
        assert zf.getinfo("kept.bin").date_time == _SOURCE_ZIP_DATE_TIME
        assert usdz.COMPACT_PAYLOAD_NAME not in names
        assert zf.read("datasource_summary.json") == b'{"source":"summary"}'
        checkpoint = _read_checkpoint(zf.read("checkpoint.ckpt"))

    assert checkpoint["load_in_place"] is False
    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 7
    assert checkpoint["pytorch-lightning_version"] == "2.0"
    assert torch.equal(checkpoint["state_dict"]["model.weight"], torch.tensor([5.0]))


def test_usdz_archive_round_trips_through_real_codec(tmp_path, monkeypatch):
    input_path = tmp_path / "scene.usdz"
    compact_path = tmp_path / "scene.compact.usdz"
    restored_path = tmp_path / "scene.restored.usdz"
    state_dict = _small_gaussian_state()
    checkpoint = {"state_dict": state_dict, "epoch": 1}
    with zipfile.ZipFile(input_path, "w") as zf:
        zf.writestr("checkpoint.ckpt", _torch_save_bytes(checkpoint))
        zf.writestr("parsed_config.yaml", b"version: null\n")
        zf.writestr("datasource_summary.json", b"{}")
        zf.writestr("rig_trajectories.json", b"{}")
        zf.writestr("sequence_tracks.json", b"{}")
        _writestr(zf, "metadata.yaml", b"metadata")

    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_config", lambda cfg: cfg)
    monkeypatch.setattr(usdz.nurec.utils.upgrade, "upgrade_model", lambda model, config, datasource_summary=None: model)
    monkeypatch.setattr(
        usdz.nurec.datasets.summary.DataSourceSummary,
        "from_dict",
        staticmethod(lambda data, infer_missing=False: object()),
    )

    usdz.compress_usdz(input_path, compact_path, opacity_threshold=0.0, compression_level=0)
    usdz.decompress_usdz(compact_path, restored_path)

    with zipfile.ZipFile(restored_path, "r") as zf:
        names = set(zf.namelist())
        checkpoint = _read_checkpoint(zf.read("checkpoint.ckpt"))

    assert usdz.COMPACT_PAYLOAD_NAME not in names
    assert "metadata.yaml" in names
    assert checkpoint["epoch"] == 1
    assert checkpoint["load_in_place"] is False
    restored_state = checkpoint["state_dict"]
    assert set(restored_state) == set(state_dict)
    for key, expected in state_dict.items():
        got = restored_state[key]
        assert isinstance(got, torch.Tensor)
        assert got.shape == expected.shape
        assert torch.isfinite(got.float()).all()


def _small_gaussian_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    n = 16
    prefix = "model.gaussians_nodes.background"
    rotations = torch.randn(n, 4, generator=generator)
    rotations = rotations / rotations.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return {
        f"{prefix}.positions": torch.randn(n, 3, generator=generator),
        f"{prefix}.densities": torch.randn(n, 1, generator=generator),
        f"{prefix}.scales": torch.rand(n, 3, generator=generator) + 0.01,
        f"{prefix}.rotations": rotations,
        f"{prefix}.features_specular": torch.randn(n, 45, generator=generator) * 0.1,
        f"{prefix}.features_albedo": torch.rand(n, 3, generator=generator),
    }
