# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import types

from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from nurec.config.checkpoint import ArtifactConfig, MortonSortConfig
from nurec.systems.gaussians import GaussiansSystem


class _ScaleNode(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scales = torch.nn.Parameter(torch.log(torch.tensor([[4.0, 2.0, 1.0]])))


class _SerializationGuardStrategy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def serialization_state_dict_overrides(self, gaussians_nodes: torch.nn.ModuleDict) -> dict[str, torch.Tensor]:
        self.calls += 1
        scales = cast(torch.Tensor, gaussians_nodes["background"].scales)
        return {"background.scales": scales.clamp(max=torch.log(torch.tensor(3.0)))}


class _GaussianModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gaussians_nodes = torch.nn.ModuleDict({"background": _ScaleNode()})
        self.gaussians_strategy = _SerializationGuardStrategy()

    def raise_if_mcmc_publication_failed(self) -> None:
        pass


def _serialized_artifact_checkpoint(compression_enabled: bool) -> dict:
    artifact_config = ArtifactConfig()
    artifact_config.checkpoint.fp16 = True
    artifact_config.compression.enabled = compression_enabled
    artifact_cache = types.SimpleNamespace(
        artifact_config=artifact_config,
        checkpoint=None,
    )
    model = types.SimpleNamespace(raise_if_mcmc_publication_failed=lambda: None)
    system = types.SimpleNamespace(artifact_cache=artifact_cache, model=model)
    checkpoint = {"state_dict": {"model.weight": torch.ones(1, dtype=torch.float32)}}

    GaussiansSystem.serialize_artifact_checkpoint(cast(GaussiansSystem, system), checkpoint)

    return torch.load(io.BytesIO(artifact_cache.checkpoint.serialized), weights_only=False, map_location="cpu")


def test_compact_artifact_checkpoint_keeps_fp32_tensors():
    checkpoint = _serialized_artifact_checkpoint(compression_enabled=True)

    assert checkpoint["state_dict"]["model.weight"].dtype == torch.float32


def test_normal_artifact_checkpoint_honors_fp16_setting():
    checkpoint = _serialized_artifact_checkpoint(compression_enabled=False)

    assert checkpoint["state_dict"]["model.weight"].dtype == torch.float16


@pytest.mark.parametrize(
    "entrypoint",
    [
        lambda system: GaussiansSystem.forward(system, cast(Any, None)),
        lambda system: GaussiansSystem.training_losses(system, cast(Any, None), 0),
        lambda system: GaussiansSystem.validation_step(system, cast(Any, None), 0),
        lambda system: GaussiansSystem.test_step(system, cast(Any, None), 0),
    ],
)
def test_render_and_loss_entrypoints_reject_failed_publication_before_work(entrypoint) -> None:
    failure = RuntimeError("unsafe MCMC state")
    gate = MagicMock(side_effect=failure)
    system = cast(GaussiansSystem, types.SimpleNamespace(_raise_if_mcmc_publication_failed=gate))

    with pytest.raises(RuntimeError, match="unsafe MCMC state"):
        entrypoint(system)

    gate.assert_called_once_with()


def test_publication_gate_uses_broadcast_exceptions_in_ddp() -> None:
    trainer = object()
    gate = MagicMock()
    system = cast(
        GaussiansSystem,
        types.SimpleNamespace(
            _trainer=trainer,
            _raise_if_mcmc_publication_failed=gate,
        ),
    )
    sentinel = object()
    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch("nurec.utils.trainer.BroadcastExceptions", return_value=sentinel) as broadcast,
    ):
        context = GaussiansSystem._mcmc_exception_context(system)

    assert context is sentinel
    broadcast.assert_called_once_with(trainer)

    GaussiansSystem.preflight_distributed_checkpoint_state_dict(system)
    gate.assert_called_once_with()


def test_publication_gate_does_not_construct_broadcast_context_without_ddp() -> None:
    system = cast(GaussiansSystem, types.SimpleNamespace(_trainer=None))

    with (
        patch.object(torch.distributed, "is_initialized", return_value=False),
        patch("nurec.utils.trainer.BroadcastExceptions") as broadcast,
        GaussiansSystem._mcmc_exception_context(system),
    ):
        pass

    broadcast.assert_not_called()


def test_root_state_dict_hook_guards_checkpoint_without_mutating_live_state():
    system = GaussiansSystem.__new__(GaussiansSystem)
    torch.nn.Module.__init__(system)
    cast(Any, system)._trainer = types.SimpleNamespace()
    system.model = _GaussianModel()
    system._serialization_state_dict_hook_handle = None
    system._register_serialization_state_dict_hook()
    system._register_serialization_state_dict_hook()

    state_dict = system.state_dict()

    scales_key = "model.gaussians_nodes.background.scales"
    expected_scales = torch.tensor([[3.0, 2.0, 1.0]])
    torch.testing.assert_close(torch.exp(state_dict[scales_key]), expected_scales)
    torch.testing.assert_close(
        torch.exp(system.model.gaussians_nodes["background"].scales),
        torch.tensor([[4.0, 2.0, 1.0]]),
    )
    assert system.model.gaussians_strategy.calls == 1

    artifact_config = ArtifactConfig()
    artifact_cache = types.SimpleNamespace(artifact_config=artifact_config, checkpoint=None)
    system.artifact_cache = artifact_cache
    system.serialize_artifact_checkpoint({"state_dict": state_dict})
    artifact_checkpoint = torch.load(
        io.BytesIO(artifact_cache.checkpoint.serialized),
        weights_only=False,
        map_location="cpu",
    )
    torch.testing.assert_close(torch.exp(artifact_checkpoint["state_dict"][scales_key]), expected_scales)


def _morton_sort_calls(morton_sort: MortonSortConfig) -> list[dict]:
    """Run the checkpoint-capture hook against a stub model and report how it was called."""
    calls: list[dict] = []
    model = types.SimpleNamespace(
        has_saved_training_parameters=lambda: False,
        morton_sort_gaussians=lambda **kwargs: calls.append(kwargs),
    )
    system = types.SimpleNamespace(
        cached_config=types.SimpleNamespace(checkpoint=types.SimpleNamespace(morton_sort=morton_sort)),
        model=model,
    )

    GaussiansSystem.maybe_morton_sort_gaussians(cast(GaussiansSystem, system))

    return calls


def test_morton_sort_hands_the_model_the_configured_settings():
    # A field the hook never forwards is silently ignored, and the run looks like it honoured the yaml.
    morton_sort = MortonSortConfig(enabled=True, sort_track_layers=True, voxel_size=0.25)

    assert _morton_sort_calls(morton_sort) == [{"sort_track_layers": True, "voxel_size": 0.25}]


def test_morton_sort_leaves_the_model_alone_when_disabled():
    assert _morton_sort_calls(MortonSortConfig(enabled=False)) == []


def test_the_checkpoint_capture_hook_reaches_the_morton_sort():
    # The Lightning-side wiring stops here, so nothing else pins that the hook delegates.
    system = Mock(spec=GaussiansSystem)

    GaussiansSystem.on_before_checkpoint_capture(system)

    system.maybe_morton_sort_gaussians.assert_called_once_with()
