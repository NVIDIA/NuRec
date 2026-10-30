# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

from collections.abc import Mapping
from dataclasses import dataclass

import gsplat
import torch

from gsplat.scene import GaussianScene
from gsplat.strategy import (
    MCMCAdditionInput,
    MCMCOwner,
    MCMCOwnerUpdate,
    MCMCRelocationInput,
    MCMCRowPlan,
    apply_plan,
    plan_addition,
    plan_relocation,
)

from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_model import (
    _MCMC_TO_NUREC_PARAMETER,
    _NUREC_TO_MCMC_SPLAT,
    BaseGaussianModel,
)
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes


def require_mcmc_capabilities() -> None:
    """Fail fast when the linked gsplat build lacks MCMC topology kernels."""
    if not gsplat.has_reloc():
        raise RuntimeError(
            "MCMC relocation and addition require a gsplat build with RELOC enabled. "
            "Ensure RELOC is included in --@gsplat//bazel:kernel_families (see .bazelrc)"
        )
    if not gsplat.has_3dgs():
        warnings.warn(
            "gsplat was built without 3DGS support; MCMC position perturbation will use the slower PyTorch fallback",
            RuntimeWarning,
            stacklevel=2,
        )


def mcmc_dead_mask(
    densities: torch.Tensor,
    invisible_steps: torch.Tensor | None,
    *,
    min_opacity: float,
    max_invisible_steps: int | None,
) -> torch.Tensor:
    """Select Gaussian rows for relocation through gsplat's tensor API."""
    import gsplat.strategy as gsplat_strategy

    return gsplat_strategy.compute_mcmc_dead_mask(
        densities,
        invisible_steps,
        min_opacity=min_opacity,
        max_invisible_steps=max_invisible_steps,
    )


def perturb_positions(
    *,
    positions: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    noise_scale: float,
    generator: torch.Generator,
    cuboid_dimensions: torch.Tensor | None = None,
) -> None:
    """Add covariance-shaped noise to one layer's positions in place."""
    import gsplat.strategy as gsplat_strategy

    gsplat_strategy.perturb_positions(
        positions=positions,
        quats=quats,
        scales=scales,
        opacities=opacities,
        noise_scale=noise_scale,
        generator=generator,
        cuboid_dimensions=cuboid_dimensions,
    )


def update_invisible_steps(
    counters: torch.Tensor,
    seen_mask: torch.Tensor,
    *,
    max_invisible_steps: int,
) -> None:
    """Advance one layer's invisible-step counters in place."""
    import gsplat.strategy as gsplat_strategy

    gsplat_strategy.update_invisible_steps_(
        counters,
        seen_mask,
        max_invisible_steps=max_invisible_steps,
    )


@dataclass(frozen=True, slots=True)
class _OwnerSnapshot:
    model: BaseGaussianModel
    scene: GaussianScene
    parameters: dict[str, torch.nn.Parameter]
    parameter_requires_grad: dict[str, bool]
    buffers: dict[str, torch.Tensor]
    invisible_steps: torch.nn.Parameter | None


def _build_owners(
    selected: SelectedGaussianNodes,
    invisible_steps: torch.nn.ParameterDict | None,
) -> tuple[tuple[MCMCOwner, ...], dict[str, _OwnerSnapshot]]:
    owners: list[MCMCOwner] = []
    snapshots: dict[str, _OwnerSnapshot] = {}
    seen_optimizer_owners: dict[int, str] = {}
    for layer_id in selected.layer_ids:
        model = selected.gaussians_nodes[layer_id]
        model.raise_if_mcmc_publication_failed()
        if model.config.density_activation != "sigmoid" or model.config.scale_activation != "exp":
            raise ValueError(f"{layer_id} must use sigmoid density and exponential scale activation")

        scene = model.get_mcmc_scene()
        if scene is None:
            raise RuntimeError(f"{layer_id} has no MCMC scene")
        row_parameters = model.get_mcmc_row_parameters()
        if row_parameters is None:
            raise RuntimeError(f"{layer_id} has unmaterialized MCMC row parameters")
        live_parameters = {_NUREC_TO_MCMC_SPLAT.get(name, name): value for name, value in row_parameters.items()}
        row_count = model.get_num_gaussians()
        if (
            scene.id != layer_id
            or scene.num_gaussians() != row_count
            or set(scene.splats) != set(live_parameters)
            or any(scene.splats[name] is not parameter for name, parameter in live_parameters.items())
        ):
            raise ValueError(f"{layer_id} MCMC scene is stale")

        optimizers = tuple(config["optimizer"] for config in model.optimizers)
        for optimizer in optimizers:
            previous_owner = seen_optimizer_owners.setdefault(id(optimizer), layer_id)
            if previous_owner != layer_id:
                raise ValueError(f"optimizer object is shared by MCMC owners {previous_owner!r} and {layer_id!r}")

        # gsplat interprets requires_grad=True as a promise of optimizer
        # ownership. NuRec historically permits inference/frozen row parameters
        # without optimizer coverage, so use operation-local non-trainable
        # aliases for only those leaves. Optimizer-owned leaves retain exact
        # identity, and publication restores every replacement's NuRec flag.
        optimizer_parameter_ids = {
            id(parameter)
            for optimizer in optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        owner_splats = {
            name: (
                torch.nn.Parameter(parameter.detach(), requires_grad=False)
                if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids
                else parameter
            )
            for name, parameter in live_parameters.items()
        }
        owner_scene = (
            scene
            if all(owner_splats[name] is parameter for name, parameter in live_parameters.items())
            else GaussianScene.from_splats(torch.nn.ParameterDict(owner_splats), id=layer_id)
        )
        owners.append(MCMCOwner(scene=owner_scene, optimizers=optimizers))

        buffers = dict(model.get_additional_buffers())
        counter = None if invisible_steps is None else invisible_steps[layer_id]
        row_state = (*buffers.values(), *(() if counter is None else (counter,)))
        if any(
            not model.is_mcmc_row_aligned_tensor(value, row_count, device=scene.splats["means"].device)
            for value in row_state
        ):
            raise ValueError(f"{layer_id} MCMC sidecar state is stale")
        snapshots[layer_id] = _OwnerSnapshot(
            model=model,
            scene=owner_scene,
            parameters={
                _MCMC_TO_NUREC_PARAMETER.get(splat_name, splat_name): parameter
                for splat_name, parameter in live_parameters.items()
            },
            parameter_requires_grad={name: parameter.requires_grad for name, parameter in row_parameters.items()},
            buffers=buffers,
            invisible_steps=counter,
        )
    return tuple(owners), snapshots


def _publish_owner_updates(
    owner_updates: tuple[MCMCOwnerUpdate, ...],
    snapshots: Mapping[str, _OwnerSnapshot],
    invisible_steps: torch.nn.ParameterDict | None,
    recycled_rows_by_owner: Mapping[str, torch.Tensor] | None,
) -> None:
    for update in owner_updates:
        snapshot = snapshots[update.scene_id]
        model = snapshot.model
        scene = snapshot.scene

        for splat_name, replacement in scene.splats.items():
            parameter_name = _MCMC_TO_NUREC_PARAMETER.get(splat_name, splat_name)
            old_parameter = snapshot.parameters[parameter_name]
            replacement.requires_grad_(snapshot.parameter_requires_grad[parameter_name])
            model.register_mcmc_parameter_hooks(parameter_name, replacement)
            if BaseModel.is_sharded(old_parameter):
                BaseModel.mark_as_sharded(replacement)
            setattr(model, parameter_name, replacement)

        for name, old_buffer in snapshot.buffers.items():
            replacement_buffer = old_buffer[update.source_row_map].detach()
            if BaseModel.is_sharded(old_buffer):
                BaseModel.mark_as_sharded(replacement_buffer)
            setattr(model, name, replacement_buffer)

        if invisible_steps is not None:
            old_counter = snapshot.invisible_steps
            assert old_counter is not None
            replacement_counter = old_counter[update.source_row_map].detach()
            replacement_counter[update.retained_count :] = 0
            counter_parameter = torch.nn.Parameter(replacement_counter, requires_grad=False)
            if BaseModel.is_sharded(old_counter):
                BaseModel.mark_as_sharded(counter_parameter)
            invisible_steps[update.scene_id] = counter_parameter

        # Row-shaped state was installed above. Give models with external
        # row-indexed state the same old-to-new plan before refreshing the
        # cached gsplat scene.
        recycled_rows = None if recycled_rows_by_owner is None else recycled_rows_by_owner.get(update.scene_id)
        model.apply_mcmc_row_update(update.source_row_map, update.retained_count, recycled_rows)

        # Rebind the cached owner scene to the exact installed NuRec Parameters.
        model.refresh_mcmc_scene(update.scene_id)


def _apply_and_publish(
    selected: SelectedGaussianNodes,
    owners: tuple[MCMCOwner, ...],
    snapshots: Mapping[str, _OwnerSnapshot],
    plan: MCMCRowPlan,
    invisible_steps: torch.nn.ParameterDict | None,
    recycled_rows_by_owner: Mapping[str, torch.Tensor] | None = None,
) -> None:
    # Validate representation-specific topology constraints before gsplat
    # mutates Parameters. A rejected plan leaves all owner state untouched, so
    # it must not latch the publication-failure guard used for partial writes.
    for layer_id in selected.layer_ids:
        recycled_rows = None if recycled_rows_by_owner is None else recycled_rows_by_owner.get(layer_id)
        selected.gaussians_nodes[layer_id].validate_mcmc_row_update(recycled_rows)

    try:
        owner_updates = apply_plan(owners, plan)
        _publish_owner_updates(owner_updates, snapshots, invisible_steps, recycled_rows_by_owner)
    except Exception:
        for layer_id in selected.layer_ids:
            selected.gaussians_nodes[layer_id].mark_mcmc_publication_failed()
        raise


@torch.no_grad()
def relocate_gaussians(
    selected: SelectedGaussianNodes,
    *,
    dead_by_owner: Mapping[str, torch.Tensor],
    source_eligible_by_owner: Mapping[str, torch.Tensor],
    binoms: torch.Tensor,
    min_opacity: float,
    generator: torch.Generator,
    invisible_steps: torch.nn.ParameterDict | None,
) -> bool:
    """Plan, apply, and publish one stateless multi-owner relocation."""
    owners, snapshots = _build_owners(selected, invisible_steps)
    plan = plan_relocation(
        owners,
        MCMCRelocationInput(
            dead_by_owner=dead_by_owner,
            source_eligible_by_owner=source_eligible_by_owner,
            binoms=binoms,
            min_opacity=min_opacity,
            generator=generator,
        ),
    )
    if plan is None:
        return False
    _apply_and_publish(selected, owners, snapshots, plan, invisible_steps)
    return True


@torch.no_grad()
def relocate_gaussians_grouped(
    selected: SelectedGaussianNodes,
    *,
    groups: list[tuple[torch.Tensor, torch.Tensor]],
    binoms: torch.Tensor,
    min_opacity: float,
    generator: torch.Generator,
    invisible_steps: torch.nn.ParameterDict | None,
) -> int:
    """Sample each donor group independently, then publish one owner update."""
    owners, snapshots = _build_owners(selected, invisible_steps)
    if len(owners) != 1:
        raise ValueError("grouped relocation requires exactly one MCMC owner")

    scene = owners[0].scene
    count = scene.num_gaussians()
    device = scene.splats["means"].device
    layer_id = scene.id
    combined_dead_mask = torch.zeros(count, dtype=torch.bool, device=device)
    plans: list[MCMCRowPlan] = []
    recycled_rows: list[torch.Tensor] = []
    for dead_mask, candidate_indices in groups:
        if not isinstance(dead_mask, torch.Tensor):
            raise TypeError("group dead_mask must be a torch.Tensor")
        if dead_mask.shape != (count,) or dead_mask.dtype != torch.bool or dead_mask.device != device:
            raise ValueError(f"group dead_mask must be bool ({count},) on {device}")
        if not isinstance(candidate_indices, torch.Tensor):
            raise TypeError("group candidate_indices must be a torch.Tensor")
        if candidate_indices.ndim != 1 or candidate_indices.dtype != torch.int64 or candidate_indices.device != device:
            raise ValueError(f"group candidate_indices must be int64 [C] on {device}")
        if bool((combined_dead_mask & dead_mask).any().item()):
            raise ValueError("group dead masks must not overlap")
        if not bool(dead_mask.any().item()) or candidate_indices.numel() == 0:
            continue
        if bool(dead_mask[candidate_indices].any().item()):
            raise ValueError("group candidates must identify retained rows")

        source_eligible = torch.zeros(count, dtype=torch.bool, device=device)
        source_eligible[candidate_indices] = True
        plan = plan_relocation(
            owners,
            MCMCRelocationInput(
                dead_by_owner={layer_id: dead_mask},
                source_eligible_by_owner={layer_id: source_eligible},
                binoms=binoms,
                min_opacity=min_opacity,
                generator=generator,
            ),
        )
        if plan is None:
            continue
        combined_dead_mask |= dead_mask
        plans.append(plan)
        recycled_rows.append(torch.where(dead_mask)[0])

    if not plans:
        return 0

    combined_plan = MCMCRowPlan(
        operation=plans[0].operation,
        keep=~combined_dead_mask,
        source_indices=torch.cat([plan.source_indices for plan in plans]),
        revised_opacities=torch.cat([plan.revised_opacities for plan in plans]),
        revised_scales=torch.cat([plan.revised_scales for plan in plans]),
        storage_min_opacity=plans[0].storage_min_opacity,
    )
    _apply_and_publish(
        selected,
        owners,
        snapshots,
        combined_plan,
        invisible_steps,
        recycled_rows_by_owner={layer_id: torch.cat(recycled_rows)},
    )
    return int(combined_dead_mask.sum().item())


@torch.no_grad()
def add_gaussians(
    selected: SelectedGaussianNodes,
    *,
    count: int,
    binoms: torch.Tensor,
    min_opacity: float,
    generator: torch.Generator,
    invisible_steps: torch.nn.ParameterDict | None,
) -> bool:
    """Plan, apply, and publish one stateless multi-owner addition."""
    owners, snapshots = _build_owners(selected, invisible_steps)
    plan = plan_addition(
        owners,
        MCMCAdditionInput(
            count=count,
            binoms=binoms,
            min_opacity=min_opacity,
            generator=generator,
        ),
    )
    if plan is None:
        return False
    _apply_and_publish(selected, owners, snapshots, plan, invisible_steps)
    return True
