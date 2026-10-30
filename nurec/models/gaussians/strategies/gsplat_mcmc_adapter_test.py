# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import gsplat
import pytest
import torch

from gsplat.strategy import MCMCOperation, MCMCOwner, MCMCOwnerUpdate, MCMCRowPlan

from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_model import _MCMC_TO_NUREC_PARAMETER, BaseGaussianModel
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import (
    add_gaussians,
    relocate_gaussians,
    relocate_gaussians_grouped,
)
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nurec.models.nn_extensions import TypedModuleDict


_ADAPTER = "nurec.models.gaussians.strategies.gsplat_mcmc_adapter"


class _Model(BaseGaussianModel):
    config: Any
    gaussian_cuboid_ids: torch.Tensor
    mcmc_position_anchor: torch.Tensor
    row_ids: torch.Tensor

    def __init__(self, count: int, base: float, device: torch.device) -> None:
        torch.nn.Module.__init__(self)
        self.config = cast(Any, SimpleNamespace(density_activation="sigmoid", scale_activation="exp"))
        self.hooks: list[str] = []
        self.hook_failure: BaseException | None = None
        self.validated_recycled_rows: list[torch.Tensor | None] = []
        self.applied_row_updates: list[tuple[torch.Tensor, int, torch.Tensor | None]] = []
        self.register_buffer("_device_indicator", torch.tensor(0, device=device), persistent=False)
        rows = torch.arange(count, dtype=torch.float32, device=device) + base
        self.positions = torch.nn.Parameter(rows[:, None].expand(-1, 3).clone())
        self.rotations = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(count, -1).clone()
        )
        self.scales = torch.nn.Parameter(torch.log((rows + 2)[:, None].expand(-1, 3).clone()))
        self.densities = torch.nn.Parameter(torch.linspace(-1, 1, count, device=device)[:, None])
        self.extra_signal = torch.nn.Parameter(rows[:, None].expand(-1, 2).clone())
        self.camera_extra_signal = torch.nn.Parameter(rows[:, None].clone())
        self.lidar_extra_signal = torch.nn.Parameter((rows + 100)[:, None].clone())
        self.features = torch.nn.Parameter(rows[:, None, None].expand(-1, 2, 3).clone())
        self.register_buffer("row_ids", rows.to(torch.int64))
        self.register_buffer("gaussian_cuboid_ids", torch.arange(count, dtype=torch.int64, device=device) // 2)
        self.register_buffer("mcmc_position_anchor", self.positions.detach().clone() + 1000.0)
        parameters = list(self.get_parameters().values())
        optimizer = torch.optim.Adam(
            [{"params": [parameter], "name": f"model.row_{index}"} for index, parameter in enumerate(parameters)]
        )
        self.optimizers = [{"optimizer": optimizer}]
        BaseModel.mark_as_sharded(self.positions)
        BaseModel.mark_as_sharded(self.row_ids)
        BaseModel.mark_as_sharded(self.gaussian_cuboid_ids)
        BaseModel.mark_as_sharded(self.mcmc_position_anchor)

    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {**super().get_parameters(), "features": self.features}

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        return {
            "row_ids": self.row_ids,
            "gaussian_cuboid_ids": self.gaussian_cuboid_ids,
            "mcmc_position_anchor": self.mcmc_position_anchor,
        }

    def register_mcmc_parameter_hooks(self, name: str, parameter: torch.nn.Parameter) -> None:
        del parameter
        if self.hook_failure is not None:
            raise self.hook_failure
        self.hooks.append(name)

    def validate_mcmc_row_update(self, recycled_row_indices: torch.Tensor | None) -> None:
        self.validated_recycled_rows.append(None if recycled_row_indices is None else recycled_row_indices.clone())

    def apply_mcmc_row_update(
        self,
        source_row_map: torch.Tensor,
        retained_count: int,
        recycled_row_indices: torch.Tensor | None,
    ) -> None:
        self.applied_row_updates.append(
            (
                source_row_map.clone(),
                retained_count,
                None if recycled_row_indices is None else recycled_row_indices.clone(),
            )
        )


def _selected(device: torch.device) -> tuple[_Model, _Model, SelectedGaussianNodes]:
    first, second = _Model(2, 1, device), _Model(3, 10, device)
    nodes: TypedModuleDict[BaseGaussianModel] = TypedModuleDict({"first": first, "second": second})
    for layer_id, model in nodes.items():
        model.refresh_mcmc_scene(layer_id)
    return first, second, SelectedGaussianNodes(nodes, exclude_layer_ids=[])


def _visibility(selected: SelectedGaussianNodes) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict(
        {
            layer_id: torch.nn.Parameter(
                torch.arange(count, dtype=torch.int32, device=selected.gaussians_nodes[layer_id].device) + 10,
                requires_grad=False,
            )
            for layer_id, count in selected.layer_num_gaussians.items()
        }
    )


def _binoms(device: torch.device) -> torch.Tensor:
    size = 51
    return torch.tensor(
        [[math.comb(n, k) if k <= n else 0 for k in range(size)] for n in range(size)],
        dtype=torch.float32,
        device=device,
    )


def _optimizer(model: _Model) -> torch.optim.Optimizer:
    return model.optimizers[0]["optimizer"]


def _has_parameter(optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter) -> bool:
    return any(candidate is parameter for group in optimizer.param_groups for candidate in group["params"])


def _fake_add(owners: tuple[MCMCOwner, ...]) -> tuple[MCMCOwnerUpdate, ...]:
    scene = owners[0].scene
    source_rows = torch.tensor([0, 1, 1], device=scene.splats["means"].device)
    for name, old in tuple(scene.splats.items()):
        replacement = torch.nn.Parameter(old[source_rows].detach(), requires_grad=old.requires_grad)
        scene.splats[name] = replacement
        for optimizer in owners[0].optimizers:
            for group in optimizer.param_groups:
                group["params"] = [replacement if parameter is old else parameter for parameter in group["params"]]
            optimizer.state.pop(old, None)
    return (MCMCOwnerUpdate(scene_id="first", retained_count=2, source_row_map=source_rows),)


def test_plan_handoff_publishes_nurec_state_and_uses_fresh_owners() -> None:
    first, second, selected = _selected(torch.device("cpu"))
    visibility = _visibility(selected)
    old_rows, old_second = first.row_ids.clone(), second.positions
    plan = object()
    owner_calls: list[tuple[MCMCOwner, ...]] = []

    def plan_addition(owners: tuple[MCMCOwner, ...], inputs: Any) -> object | None:
        owner_calls.append(owners)
        assert [owner.scene.id for owner in owners] == ["first", "second"]
        assert inputs.count == 1
        return plan if len(owner_calls) == 1 else None

    def apply(owners: tuple[MCMCOwner, ...], received: object) -> tuple[MCMCOwnerUpdate, ...]:
        assert received is plan and owners is owner_calls[-1]
        return _fake_add(owners)

    with (
        patch(f"{_ADAPTER}.plan_addition", side_effect=plan_addition),
        patch(f"{_ADAPTER}.apply_plan", side_effect=apply),
    ):
        assert add_gaussians(
            selected,
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=visibility,
        )
        assert not add_gaussians(
            SelectedGaussianNodes(selected.gaussians_nodes, exclude_layer_ids=[]),
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=visibility,
        )

    assert owner_calls[0] is not owner_calls[1] and owner_calls[0][0] is not owner_calls[1][0]
    scene = first.get_mcmc_scene()
    assert scene is not None
    for splat_name, parameter in scene.splats.items():
        assert getattr(first, _MCMC_TO_NUREC_PARAMETER.get(splat_name, splat_name)) is parameter
        assert _has_parameter(_optimizer(first), parameter)
    torch.testing.assert_close(first.row_ids, old_rows[[0, 1, 1]])
    torch.testing.assert_close(visibility["first"], torch.tensor([10, 11, 0], dtype=torch.int32))
    assert set(first.hooks) == set(first.get_parameters())
    assert first.validated_recycled_rows == [None]
    assert len(first.applied_row_updates) == 1
    assert first.applied_row_updates[0][1:] == (2, None)
    torch.testing.assert_close(first.applied_row_updates[0][0], torch.tensor([0, 1, 1]))
    assert second.validated_recycled_rows == [None]
    assert second.applied_row_updates == []
    assert BaseModel.is_sharded(first.positions) and BaseModel.is_sharded(first.row_ids)
    assert second.positions is old_second and second.hooks == []


def test_grouped_relocation_samples_per_group_and_publishes_once() -> None:
    device = torch.device("cpu")
    model = _Model(4, 1, device)
    with torch.no_grad():
        model.gaussian_cuboid_ids.copy_(torch.tensor([0, 0, 1, 1]))
    nodes: TypedModuleDict[BaseGaussianModel] = TypedModuleDict({"lights": model})
    model.refresh_mcmc_scene("lights")
    selected = SelectedGaussianNodes(nodes, exclude_layer_ids=[])
    first_dead = torch.tensor([False, True, False, False])
    second_dead = torch.tensor([False, False, False, True])
    first_candidates = torch.tensor([0], dtype=torch.int64)
    second_candidates = torch.tensor([2], dtype=torch.int64)
    old_rows = model.row_ids.clone()
    old_track_ids = model.gaussian_cuboid_ids.clone()
    old_anchors = model.mcmc_position_anchor.clone()
    planned_eligible: list[torch.Tensor] = []

    def plan(owners: tuple[MCMCOwner, ...], inputs: Any) -> MCMCRowPlan:
        assert len(owners) == 1
        dead = inputs.dead_by_owner["lights"]
        eligible = inputs.source_eligible_by_owner["lights"]
        planned_eligible.append(eligible.clone())
        source_indices = torch.where(eligible)[0]
        return MCMCRowPlan(
            operation=MCMCOperation.RELOCATION,
            keep=~dead,
            source_indices=source_indices,
            revised_opacities=torch.full((source_indices.numel(),), 0.4),
            revised_scales=torch.full((source_indices.numel(), 3), 2.0),
            storage_min_opacity=0.005,
        )

    def apply(owners: tuple[MCMCOwner, ...], combined: MCMCRowPlan) -> tuple[MCMCOwnerUpdate, ...]:
        source_rows = combined.replacement_rows
        scene = owners[0].scene
        for name, old in tuple(scene.splats.items()):
            replacement = torch.nn.Parameter(old[source_rows].detach(), requires_grad=old.requires_grad)
            scene.splats[name] = replacement
            for optimizer in owners[0].optimizers:
                for group in optimizer.param_groups:
                    group["params"] = [replacement if parameter is old else parameter for parameter in group["params"]]
                optimizer.state.pop(old, None)
        assert torch.equal(combined.keep, torch.tensor([True, False, True, False]))
        assert torch.equal(combined.source_indices, torch.tensor([0, 2]))
        return (MCMCOwnerUpdate(scene_id="lights", retained_count=2, source_row_map=source_rows),)

    with (
        patch(f"{_ADAPTER}.plan_relocation", side_effect=plan) as plan_relocation,
        patch(f"{_ADAPTER}.apply_plan", side_effect=apply) as apply_plan,
    ):
        relocated = relocate_gaussians_grouped(
            selected,
            groups=[(first_dead, first_candidates), (second_dead, second_candidates)],
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator().manual_seed(7),
            invisible_steps=None,
        )

    assert relocated == 2
    assert plan_relocation.call_count == 2
    apply_plan.assert_called_once()
    assert torch.equal(planned_eligible[0], torch.tensor([True, False, False, False]))
    assert torch.equal(planned_eligible[1], torch.tensor([False, False, True, False]))
    source_rows = torch.tensor([0, 2, 0, 2])
    assert torch.equal(model.row_ids, old_rows[source_rows])
    assert torch.equal(model.gaussian_cuboid_ids, old_track_ids[source_rows])
    torch.testing.assert_close(model.mcmc_position_anchor, old_anchors[source_rows])
    assert len(model.validated_recycled_rows) == 1
    torch.testing.assert_close(model.validated_recycled_rows[0], torch.tensor([1, 3]))
    assert len(model.applied_row_updates) == 1
    applied_source_rows, retained_count, recycled_rows = model.applied_row_updates[0]
    torch.testing.assert_close(applied_source_rows, source_rows)
    assert retained_count == 2
    torch.testing.assert_close(recycled_rows, torch.tensor([1, 3]))
    assert BaseModel.is_sharded(model.row_ids)
    assert BaseModel.is_sharded(model.gaussian_cuboid_ids)
    assert BaseModel.is_sharded(model.mcmc_position_anchor)


def test_nurec_preflight_rejects_stale_scene_and_shared_optimizer_before_sampling() -> None:
    first, _, selected = _selected(torch.device("cpu"))
    first.positions = torch.nn.Parameter(first.positions.detach().clone())
    with patch(f"{_ADAPTER}.plan_addition") as plan, pytest.raises(ValueError, match="scene is stale"):
        add_gaussians(
            selected,
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=None,
        )
    plan.assert_not_called()

    first, _, selected = _selected(torch.device("cpu"))
    first.extra_signal = torch.nn.Parameter(torch.zeros(3, 2))
    with patch(f"{_ADAPTER}.plan_addition") as plan, pytest.raises(ValueError, match="extra_signal is not row-aligned"):
        add_gaussians(
            selected,
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=None,
        )
    plan.assert_not_called()

    first, second, selected = _selected(torch.device("cpu"))
    shared = torch.optim.Adam([*first.get_parameters().values(), *second.get_parameters().values()])
    first.optimizers = second.optimizers = [{"optimizer": shared}]
    with patch(f"{_ADAPTER}.plan_addition") as plan, pytest.raises(ValueError, match="optimizer object is shared"):
        add_gaussians(
            selected,
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=None,
        )
    plan.assert_not_called()


def test_none_plan_is_a_noop_and_publication_failure_latches_all_owners() -> None:
    first, second, selected = _selected(torch.device("cpu"))
    old_positions = first.positions
    with patch(f"{_ADAPTER}.plan_addition", return_value=None), patch(f"{_ADAPTER}.apply_plan") as apply:
        assert not add_gaussians(
            selected,
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=None,
        )
    apply.assert_not_called()
    assert first.positions is old_positions

    failure = RuntimeError("publication failed")
    first.hook_failure = failure
    with (
        patch(f"{_ADAPTER}.plan_addition", return_value=object()),
        patch(f"{_ADAPTER}.apply_plan", side_effect=lambda owners, _plan: _fake_add(owners)),
        pytest.raises(RuntimeError) as caught,
    ):
        add_gaussians(
            selected,
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=None,
        )
    assert caught.value is failure
    for model in (first, second):
        with pytest.raises(RuntimeError, match="publication previously failed"):
            model.raise_if_mcmc_publication_failed()


def test_apply_failure_latches_all_owners() -> None:
    first, second, selected = _selected(torch.device("cpu"))
    failure = RuntimeError("apply failed after first owner")

    def fail_during_apply(owners: tuple[MCMCOwner, ...], plan: object) -> None:
        del plan
        _fake_add(owners)
        raise failure

    with (
        patch(f"{_ADAPTER}.plan_addition", return_value=object()),
        patch(f"{_ADAPTER}.apply_plan", side_effect=fail_during_apply),
        pytest.raises(RuntimeError) as caught,
    ):
        add_gaussians(
            selected,
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=None,
        )

    assert caught.value is failure
    for model in (first, second):
        with pytest.raises(RuntimeError, match="publication previously failed"):
            model.raise_if_mcmc_publication_failed()


def test_validation_failure_does_not_latch_owners() -> None:
    first, second, selected = _selected(torch.device("cpu"))
    failure = ValueError("row update rejected")

    with (
        patch(f"{_ADAPTER}.plan_addition", return_value=object()),
        patch(f"{_ADAPTER}.apply_plan") as apply_plan,
        patch.object(first, "validate_mcmc_row_update", side_effect=failure),
        pytest.raises(ValueError) as caught,
    ):
        add_gaussians(
            selected,
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=None,
        )

    assert caught.value is failure
    apply_plan.assert_not_called()
    first.raise_if_mcmc_publication_failed()
    second.raise_if_mcmc_publication_failed()


def test_unoptimized_trainable_rows_are_published_with_requires_grad_preserved() -> None:
    first, second, selected = _selected(torch.device("cpu"))
    for model in (first, second):
        model.optimizers = []
        model.refresh_mcmc_scene("first" if model is first else "second")
    old_positions = first.positions

    with (
        patch(f"{_ADAPTER}.plan_addition", return_value=object()),
        patch(f"{_ADAPTER}.apply_plan", side_effect=lambda owners, _plan: _fake_add(owners)),
    ):
        assert add_gaussians(
            selected,
            count=1,
            binoms=torch.ones(1, 1),
            min_opacity=0.005,
            generator=torch.Generator(),
            invisible_steps=None,
        )

    assert first.positions is not old_positions
    assert all(parameter.requires_grad for parameter in first.get_parameters().values())
    scene = first.get_mcmc_scene()
    assert scene is not None and scene.splats["means"] is first.positions


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gsplat.has_reloc() or not gsplat.has_adam(),
    reason="owner topology integration requires RELOC, ADAM, and CUDA",
)
@pytest.mark.parametrize("operation", ["addition", "relocation"])
def test_real_owner_api_keeps_model_scene_optimizer_and_sidecars_aligned(operation: str) -> None:
    device = torch.device("cuda")
    first, second, selected = _selected(device)
    visibility = _visibility(selected)
    models = {"first": first, "second": second}
    old_rows = {layer: model.row_ids.clone() for layer, model in models.items()}
    for model in models.values():
        torch.stack([parameter.sum() for parameter in model.get_parameters().values()]).sum().backward()
        _optimizer(model).step()
        _optimizer(model).zero_grad()

    if operation == "addition":
        assert add_gaussians(
            selected,
            count=2,
            binoms=_binoms(device),
            min_opacity=0.005,
            generator=torch.Generator(device=device).manual_seed(17),
            invisible_steps=visibility,
        )
        assert sum(model.get_num_gaussians() for model in models.values()) == 7
    else:
        assert relocate_gaussians(
            selected,
            dead_by_owner={
                "first": torch.tensor([False, True], device=device),
                "second": torch.zeros(3, dtype=torch.bool, device=device),
            },
            source_eligible_by_owner={
                "first": torch.zeros(2, dtype=torch.bool, device=device),
                "second": torch.ones(3, dtype=torch.bool, device=device),
            },
            binoms=_binoms(device),
            min_opacity=0.005,
            generator=torch.Generator(device=device).manual_seed(19),
            invisible_steps=visibility,
        )
        assert (first.get_num_gaussians(), second.get_num_gaussians()) == (1, 4)

    for layer_id, model in models.items():
        scene, optimizer = model.get_mcmc_scene(), _optimizer(model)
        assert scene is not None
        for splat_name, parameter in scene.splats.items():
            assert getattr(model, _MCMC_TO_NUREC_PARAMETER.get(splat_name, splat_name)) is parameter
            assert _has_parameter(optimizer, parameter)
        assert model.row_ids.shape == visibility[layer_id].shape
        assert set(model.row_ids.tolist()) <= set(old_rows[layer_id].tolist())
        appended = model.get_num_gaussians() - (2 if layer_id == "first" else 3)
        if appended > 0:
            assert torch.count_nonzero(visibility[layer_id][-appended:]) == 0
