# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import nurec.instant_nurec.predict.prediction_runner as prediction_runner_module
import nurec.instant_nurec.systems as systems_module
import nurec.instant_nurec.systems.gaussians_instant_nurec as gaussians_instant_nurec_module

from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.instant_nurec.models import load_checkpoint_state_dict
from nurec.instant_nurec.models.base import BaseInstantNuRec
from nurec.instant_nurec.predict.prediction_runner import (
    InstantNuRecPredictionOutput,
    InstantNuRecPredictionRunner,
    run_instant_nurec_prediction,
)
from nurec.instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from nurec.instant_nurec.systems.gaussians_instant_nurec import GaussiansInstantNuRecSystem
from nurec.utils.batch import DataAndRenderingBatch, InstantNuRecDataBatch


@dataclass
class _Context:
    index: int


@dataclass
class _SupervisionPack:
    index: int


@dataclass
class _LegacyLightningMetadata:
    value: str


class _Primitive:
    def __init__(self, index: int, events: list[str]) -> None:
        self.index = index
        self.events = events

    def preprocess_for_export(
        self,
        context: _Context,
        _config: object,
        *,
        context_rig: object,
        supervision_pack: _SupervisionPack | None,
    ) -> _Primitive:
        pack_index = None if supervision_pack is None else supervision_pack.index
        self.events.append(f"preprocess:{self.index}:{context.index}:{context_rig}:{pack_index}")
        return self


class _RecordingModel(torch.nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))
        self.events = events
        self.update_context: tuple[bool, bool] | None = None
        self.update_training: bool | None = None
        self.reconstruct_contexts: list[tuple[bool, bool]] = []
        self.reconstruct_training: list[bool] = []
        self.media_loggers: list[object | None] = []

    def update_step_train_batch_start(self, epoch: int, global_step: int, system: object | None) -> dict:
        self.events.append(f"update:{epoch}:{global_step}:{system}")
        self.update_context = (
            torch.is_inference_mode_enabled(),
            torch.is_autocast_enabled("cpu"),
        )
        self.update_training = self.training
        return {}

    def prepare_context(self, contexts: list[_Context], cuboid_tracks: list[str] | None) -> list[_Context]:
        indices = ",".join(str(context.index) for context in contexts)
        tracks = "none" if cuboid_tracks is None else ",".join(cuboid_tracks)
        self.events.append(f"prepare:{indices}:{tracks}")
        return contexts

    def reconstruct(
        self,
        contexts: list[_Context],
        _cuboid_tracks: list[str] | None,
        *,
        media_logger: object | None,
        compute_supervision_pack: bool,
    ) -> tuple[list[_Primitive], list[_SupervisionPack]]:
        indices = ",".join(str(context.index) for context in contexts)
        self.events.append(f"reconstruct:{indices}:{compute_supervision_pack}")
        self.reconstruct_contexts.append(
            (
                torch.is_inference_mode_enabled(),
                torch.is_autocast_enabled("cpu"),
            )
        )
        self.reconstruct_training.append(self.training)
        self.media_loggers.append(media_logger)
        return (
            [_Primitive(context.index, self.events) for context in contexts],
            [_SupervisionPack(context.index) for context in contexts],
        )


def _config(*, chunk_size: int = 1, preprocess: bool = True, merge: bool = False) -> InstantNuRecConfig:
    return cast(
        InstantNuRecConfig,
        SimpleNamespace(
            model=SimpleNamespace(export_preprocess=SimpleNamespace(enabled=preprocess)),
            predict=SimpleNamespace(
                chunk_size=chunk_size,
                primitive_merge=SimpleNamespace(enabled=merge),
            ),
        ),
    )


def _batch(size: int, *, supervision: bool = False, cuboid_tracks: bool = False) -> InstantNuRecDataBatch:
    return InstantNuRecDataBatch(
        context=cast(list[DataAndRenderingBatch], [_Context(index) for index in range(size)]),
        supervision=(
            cast(list[DataAndRenderingBatch], [SimpleNamespace(index=index) for index in range(size)])
            if supervision
            else None
        ),
        cuboid_tracks=cast(list[Any], [f"track-{index}" for index in range(size)]) if cuboid_tracks else None,
        context_rig=cast(list[Any], [f"rig-{index}" for index in range(size)]),
    )


@pytest.mark.parametrize("preprocess_enabled", [False, True])
@pytest.mark.parametrize("merge_enabled", [False, True])
def test_shared_prediction_preserves_chunk_order_packs_and_single_rendering_setup(
    monkeypatch: pytest.MonkeyPatch,
    merge_enabled: bool,
    preprocess_enabled: bool,
) -> None:
    events: list[str] = []
    config = _config(chunk_size=2, preprocess=preprocess_enabled, merge=merge_enabled)
    batch = _batch(3)

    def compute_rendering_data(self: InstantNuRecDataBatch, *, device: torch.device) -> None:
        assert self is batch
        assert device == torch.device("cpu")
        events.append("rendering-data")

    monkeypatch.setattr(InstantNuRecDataBatch, "maybe_compute_rendering_data", compute_rendering_data)

    def forward_chunk(chunk: InstantNuRecDataBatch):
        contexts = cast(list[_Context], chunk.context)
        indices = ",".join(str(context.index) for context in contexts)
        events.append(f"forward:{indices}")
        return (
            cast(list[BaseInstantNuRecPrimitive], [_Primitive(context.index, events) for context in contexts]),
            [_SupervisionPack(context.index) for context in contexts],
            [],
        )

    merged_batch = _batch(1)

    class _PrimitiveMerge:
        def merge_primitives_and_batch(self, primitives: list[_Primitive], actual_batch: InstantNuRecDataBatch):
            assert actual_batch is batch
            assert [primitive.index for primitive in primitives] == [0, 1, 2]
            events.append("merge:" + ",".join(str(primitive.index) for primitive in primitives))
            return _Primitive(99, events), merged_batch

    def make_primitive_merge(
        primitive_type: type,
        primitive_merge_config: object,
        export_preprocess_config: object,
    ):
        if not merge_enabled:
            raise AssertionError("Primitive merge must remain disabled")
        assert primitive_type is _Primitive
        assert primitive_merge_config is config.predict.primitive_merge
        assert export_preprocess_config is config.model.export_preprocess
        return _PrimitiveMerge()

    monkeypatch.setattr(prediction_runner_module, "make_primitive_merge", make_primitive_merge)

    output = run_instant_nurec_prediction(
        batch=batch,
        device=torch.device("cpu"),
        export_preprocess_config=config.model.export_preprocess,
        predict_config=config.predict,
        forward_chunk=forward_chunk,
    )

    expected_events = [
        "rendering-data",
        "forward:0,1",
    ]
    if preprocess_enabled:
        expected_events.extend(
            [
                "preprocess:0:0:rig-0:0",
                "preprocess:1:1:rig-1:1",
            ]
        )
    expected_events.append("forward:2")
    if preprocess_enabled:
        expected_events.append("preprocess:2:2:rig-2:2")
    if merge_enabled:
        expected_events.append("merge:0,1,2")
        assert output.batch is merged_batch
        assert [cast(_Primitive, primitive).index for primitive in output.primitives] == [99]
    else:
        assert output.batch is batch
        assert [cast(_Primitive, primitive).index for primitive in output.primitives] == [0, 1, 2]
    assert events == expected_events


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_shared_prediction_rejects_non_positive_chunk_size_before_rendering_setup(
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int,
) -> None:
    config = _config(chunk_size=chunk_size)

    def unexpected_rendering_setup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Rendering setup must not run for an invalid chunk size")

    monkeypatch.setattr(InstantNuRecDataBatch, "maybe_compute_rendering_data", unexpected_rendering_setup)
    with pytest.raises(ValueError, match=rf"predict\.chunk_size must be positive, got {chunk_size}"):
        run_instant_nurec_prediction(
            batch=_batch(1),
            device=torch.device("cpu"),
            export_preprocess_config=config.model.export_preprocess,
            predict_config=config.predict,
            forward_chunk=lambda _batch: ([], [], []),
        )


def test_runner_context_only_contract_cuboids_media_logger_and_explicit_update_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    config = _config(chunk_size=1, preprocess=True)
    model = _RecordingModel(events)
    runner = InstantNuRecPredictionRunner(config, model=cast(BaseInstantNuRec, model))
    batch = _batch(3, cuboid_tracks=True)

    def compute_rendering_data(_self: InstantNuRecDataBatch, *, device: torch.device) -> None:
        assert device == torch.device("cpu")
        events.append("rendering-data")

    def convert_cuboid(pack: str) -> str:
        events.append(f"cuboid:{pack}")
        return f"converted-{pack}"

    monkeypatch.setattr(InstantNuRecDataBatch, "maybe_compute_rendering_data", compute_rendering_data)
    monkeypatch.setattr(
        prediction_runner_module.CuboidTracks.Factory,
        "from_pack",
        staticmethod(convert_cuboid),
    )

    assert list(runner.named_children()) == [("model", model)]
    assert list(runner.state_dict()) == ["model.weight"]

    runner.train()
    assert runner.training
    assert model.training
    runner.prepare_prediction()
    assert not runner.training
    assert not model.training
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = runner(batch)

    assert model.update_context == (False, False)
    assert model.update_training is False
    assert model.reconstruct_contexts == [(True, True)] * 3
    assert model.reconstruct_training == [False] * 3
    assert model.media_loggers == [None, None, None]
    assert [cast(_Primitive, primitive).index for primitive in output.primitives] == [0, 1, 2]
    assert events == [
        "update:0:0:None",
        "rendering-data",
        "cuboid:track-0",
        "prepare:0:converted-track-0",
        "reconstruct:0:False",
        "preprocess:0:0:rig-0:0",
        "cuboid:track-1",
        "prepare:1:converted-track-1",
        "reconstruct:1:False",
        "preprocess:1:1:rig-1:1",
        "cuboid:track-2",
        "prepare:2:converted-track-2",
        "reconstruct:2:False",
        "preprocess:2:2:rig-2:2",
    ]


def test_lightning_forward_uses_shared_context_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(1)
    model = _RecordingModel([])
    media_logger = object()
    primitive = _Primitive(7, [])
    supervision_pack = _SupervisionPack(7)
    cuboid_tracks = ["converted-track"]
    seen: dict[str, object] = {}

    def record_context_reconstruction(**kwargs: object):
        seen.update(kwargs)
        return [primitive], [supervision_pack], cuboid_tracks

    monkeypatch.setattr(
        gaussians_instant_nurec_module, "reconstruct_instant_nurec_context", record_context_reconstruction
    )
    system = cast(
        GaussiansInstantNuRecSystem,
        SimpleNamespace(model=model, media_logger=media_logger),
    )

    primitives, supervision_packs, traced_returns = GaussiansInstantNuRecSystem.forward(system, batch)

    assert primitives == [primitive]
    assert supervision_packs == [supervision_pack]
    assert traced_returns == []
    assert seen == {
        "model": model,
        "batch": batch,
        "media_logger": media_logger,
        "compute_supervision_pack": False,
    }


def test_runner_rejects_supervision_before_rendering_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    runner = InstantNuRecPredictionRunner(config, model=cast(BaseInstantNuRec, _RecordingModel([])))

    def unexpected_rendering_setup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Rendering setup must not run for an unsupported supervised batch")

    monkeypatch.setattr(InstantNuRecDataBatch, "maybe_compute_rendering_data", unexpected_rendering_setup)
    with pytest.raises(ValueError, match="context-only batches without supervision"):
        runner(_batch(1, supervision=True))


@pytest.mark.parametrize("wrapped", [False, True])
def test_runner_strictly_loads_bare_and_wrapped_model_namespace(tmp_path: Path, wrapped: bool) -> None:
    model = torch.nn.Linear(2, 1)
    runner = InstantNuRecPredictionRunner(_config(), model=cast(BaseInstantNuRec, model))
    expected = {
        "model.weight": torch.full_like(model.weight, 3.0),
        "model.bias": torch.full_like(model.bias, -2.0),
    }
    checkpoint_path = tmp_path / ("wrapped.ckpt" if wrapped else "bare.ckpt")
    torch.save({"state_dict": expected} if wrapped else expected, checkpoint_path)

    runner.load_weights_only_checkpoint(checkpoint_path)

    assert torch.equal(model.weight, expected["model.weight"])
    assert torch.equal(model.bias, expected["model.bias"])


@pytest.mark.parametrize(
    "state_dict",
    [
        {"model.weight": torch.ones(1, 2)},
        {
            "model.weight": torch.ones(1, 2),
            "model.bias": torch.ones(1),
            "model.extra": torch.ones(1),
        },
        {"weight": torch.ones(1, 2), "bias": torch.ones(1)},
    ],
    ids=["missing", "extra", "unprefixed"],
)
def test_runner_rejects_non_strict_checkpoint_namespace(tmp_path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    runner = InstantNuRecPredictionRunner(_config(), model=cast(BaseInstantNuRec, torch.nn.Linear(2, 1)))
    checkpoint_path = tmp_path / "invalid.ckpt"
    torch.save({"state_dict": state_dict}, checkpoint_path)

    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
        runner.load_weights_only_checkpoint(checkpoint_path)


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    [
        ([torch.ones(1)], "Checkpoint must be a mapping"),
        ({"state_dict": [torch.ones(1)]}, "Checkpoint state_dict must be a mapping"),
    ],
)
def test_runner_rejects_malformed_weights_only_checkpoint(
    tmp_path: Path,
    checkpoint: object,
    message: str,
) -> None:
    runner = InstantNuRecPredictionRunner(_config(), model=cast(BaseInstantNuRec, torch.nn.Linear(2, 1)))
    checkpoint_path = tmp_path / "malformed.ckpt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(TypeError, match=message):
        runner.load_weights_only_checkpoint(checkpoint_path)


def test_shared_state_loader_accepts_trusted_legacy_lightning_metadata(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    expected = {
        "weight": torch.full_like(model.weight, 4.0),
        "bias": torch.full_like(model.bias, -3.0),
    }
    checkpoint_path = tmp_path / "legacy-lightning.ckpt"
    torch.save(
        {
            "state_dict": expected,
            "metadata": _LegacyLightningMetadata("not accepted by the restricted unpickler"),
        },
        checkpoint_path,
    )

    load_checkpoint_state_dict(model, checkpoint_path, torch_load_weights_only=False)

    assert torch.equal(model.weight, expected["weight"])
    assert torch.equal(model.bias, expected["bias"])


def test_system_factory_uses_shared_state_loader_with_legacy_unpickling(monkeypatch: pytest.MonkeyPatch) -> None:
    config = cast(InstantNuRecConfig, SimpleNamespace(resume_weights_only=True))
    checkpoint_path = "weights.ckpt"
    events: list[object] = []

    class _System:
        def __init__(self, actual_config: InstantNuRecConfig) -> None:
            events.append(("construct", actual_config))

    def record_load(module: object, path: str, *, torch_load_weights_only: bool) -> None:
        events.append(("load", module, path, torch_load_weights_only))

    monkeypatch.setattr(systems_module, "GaussiansInstantNuRecSystem", _System)
    monkeypatch.setattr(systems_module, "load_checkpoint_state_dict", record_load)

    system = systems_module.make("base-instant_nurec-system", config, load_from_checkpoint=checkpoint_path)

    assert isinstance(system, _System)
    assert events == [
        ("construct", config),
        ("load", system, checkpoint_path, False),
    ]


def test_system_factory_preserves_full_lightning_checkpoint_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    config = cast(InstantNuRecConfig, SimpleNamespace(resume_weights_only=False))
    checkpoint_path = "full.ckpt"
    loaded_system = object()
    events: list[object] = []

    class _System:
        @classmethod
        def load_from_checkpoint(cls, path: str, *, strict: bool, config: InstantNuRecConfig) -> object:
            events.append((path, strict, config))
            return loaded_system

    def unexpected_state_dict_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Full Lightning resume must not use the state-dict-only loader")

    monkeypatch.setattr(systems_module, "GaussiansInstantNuRecSystem", _System)
    monkeypatch.setattr(systems_module, "load_checkpoint_state_dict", unexpected_state_dict_load)

    system = systems_module.make("base-instant_nurec-system", config, load_from_checkpoint=checkpoint_path)

    assert system is loaded_system
    assert events == [(checkpoint_path, True, config)]


def test_lightning_predict_step_delegates_existing_forward_to_shared_core(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(1)
    output = InstantNuRecPredictionOutput(
        primitives=cast(list[BaseInstantNuRecPrimitive], [_Primitive(7, [])]), batch=batch
    )

    def forward_chunk(_chunk: InstantNuRecDataBatch):
        return [], [], []

    system = SimpleNamespace(
        device=torch.device("cpu"),
        cached_config=SimpleNamespace(model=SimpleNamespace(export_preprocess="export-config")),
        predict_config="predict-config",
        forward=forward_chunk,
    )
    seen: dict[str, object] = {}

    def record_shared_core(**kwargs: object) -> InstantNuRecPredictionOutput:
        seen.update(kwargs)
        return output

    monkeypatch.setattr(gaussians_instant_nurec_module, "run_instant_nurec_prediction", record_shared_core)

    actual = GaussiansInstantNuRecSystem.predict_step(cast(GaussiansInstantNuRecSystem, system), batch, 0)

    assert actual == {"primitives": output.primitives, "batch": batch}
    assert seen == {
        "batch": batch,
        "device": torch.device("cpu"),
        "export_preprocess_config": "export-config",
        "predict_config": "predict-config",
        "forward_chunk": forward_chunk,
    }
