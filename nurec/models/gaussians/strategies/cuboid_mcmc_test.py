# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import math

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gsplat.strategy import MCMCOperation, perturb_positions, sample_mcmc_relocation

from nurec.models.base import BaseModel
from nurec.models.gaussians.construction_barriers_model import ConstructionBarriersGaussianModel
from nurec.models.gaussians.gaussians_model import BaseGaussianModel
from nurec.models.gaussians.strategies.cuboid_mcmc import cuboid_topology, growth_allocations
from nurec.models.gaussians.strategies.mcmc import MCMCStrategy
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nurec.models.nn_extensions import TypedModuleDict


def test_equal_growth_shares_respect_layer_budget():
    """Repeated native growth reaches the layer cap with balanced per-cuboid populations."""
    counts = [256] * 105
    for _ in range(200):
        counts = [n + addition for n, addition in zip(counts, growth_allocations(counts, 256000))]
    assert sum(counts) == 256000
    assert max(counts) - min(counts) == 1
    assert growth_allocations([100, 110, 90], 1000) == [3, 0, 12]
    assert growth_allocations([], 256000) == []


@pytest.mark.parametrize("counts,cap", [([1], 0), ([0], 10), ([11, 1], 20), ([1.5], 10)])
def test_invalid_growth_budget(counts, cap):
    """Reject nonpositive, noninteger or already over-quota cuboid populations."""
    with pytest.raises(ValueError):
        growth_allocations(counts, cap)


def test_ineligible_cuboids_do_not_starve_live_growth_or_transfer_their_quota():
    """Suppressed cuboids receive no growth, while live ones can reach their own quota."""
    counts = [20] * 3
    eligible = [False, False, True]
    assert growth_allocations(counts, 300, eligible=eligible) == [0, 0, 3]
    for _ in range(100):
        additions = growth_allocations(counts, 300, eligible=eligible)
        counts = [n + addition for n, addition in zip(counts, additions)]
    assert counts == [20, 20, 100]
    assert growth_allocations(counts, 300, eligible=[False] * 3) == [0] * 3
    with pytest.raises(ValueError, match="eligibility"):
        growth_allocations(counts, 300, eligible=[True])
    with pytest.raises(ValueError, match="eligibility"):
        growth_allocations(counts, 300, eligible=[True, True, 1])


def _rotation(angle, device):
    """Build a tilted cuboid rotation so rejection is tested beyond world-aligned boxes."""
    c, s = math.cos(angle), math.sin(angle)
    rz = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]], device=device)
    c, s = math.cos(0.31), math.sin(0.31)
    rx = torch.tensor([[1.0, 0, 0], [0, c, -s], [0, s, c]], device=device)
    return rz @ rx


class _Model(ConstructionBarriersGaussianModel):
    """Materialized row parameters, without a dataset or renderer fixture."""

    def __init__(self, counts):
        """Create three CUDA cuboids with row identities and initialized optimizer groups."""
        torch.nn.Module.__init__(self)
        device, n = torch.device("cuda"), sum(counts)
        self.config = SimpleNamespace(
            density_activation="sigmoid", scale_activation="exp", cuboid_perturbation_scale=1.3
        )
        self.register_buffer("_device_indicator", torch.tensor(0, device=device), persistent=False)
        self.positions = torch.nn.Parameter(torch.zeros(n, 3, device=device))
        self.rotations = torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1))
        self.scales = torch.nn.Parameter(torch.full((n, 3), -1.0, device=device))
        self.densities = torch.nn.Parameter(torch.full((n, 1), -8.0, device=device))
        self.extra_signal = torch.nn.Parameter(torch.zeros(n, 2, device=device))
        self.camera_extra_signal = torch.nn.Parameter(torch.zeros(n, device=device))
        self.lidar_extra_signal = torch.nn.Parameter(torch.zeros(n, device=device))
        self.features = torch.nn.Parameter(torch.zeros(n, 2, 3, device=device))
        self.register_buffer(
            "barrier_object_id",
            torch.tensor([7, 23, 101], device=device).repeat_interleave(torch.tensor(counts, device=device)),
        )
        self._cuboid_records = [
            {
                "track_id": value,
                "class_id": 16,
                "center_nurec": [float(i) * 10, 2.0, -3.0],
                "rotation_nurec": _rotation(i * 0.6, device).tolist(),
                "dimensions_m": [2.0, 1.0, 3.0],
            }
            for i, value in enumerate([7, 23, 101])
        ]
        self._cuboid_constraint_cache = None
        groups = [
            {"params": [p], "name": "barriers." + name, "lr": 0.0016} for name, p in self.get_parameters().items()
        ]
        self.optimizers = [{"optimizer": torch.optim.Adam(groups)}]
        BaseModel.mark_as_sharded(self.positions)
        BaseModel.mark_as_sharded(self.barrier_object_id)
        self.refresh_mcmc_scene("barriers")

    def get_parameters(self):
        """Include SH features in the native Gaussian parameter set."""
        return BaseGaussianModel.get_parameters(self) | {"features": self.features}

    def register_mcmc_parameter_hooks(self, name, parameter):
        """No training hooks are needed by this materialized test fixture."""
        pass


def _geometry(model):
    """Resolve per-row cuboid geometry after any topology-driven identity reordering."""
    rows = torch.searchsorted(torch.tensor([7, 23, 101], device=model.device), model.barrier_object_id)

    def field(name):
        """Gather one cuboid-record field in the current Gaussian row order."""
        return model.positions.new_tensor([r[name] for r in model._cuboid_records])[rows]

    return field("center_nurec"), field("rotation_nurec"), field("dimensions_m") * 1.3


def _topology_kwargs(model, visibility):
    """Use the production opacity threshold and native CUDA relocation kernels."""
    return dict(
        cap=256000,
        min_opacity=0.005,
        invisible_steps=visibility,
        max_invisible_steps=100,
        generator=torch.Generator(device=model.device).manual_seed(14),
        binoms=torch.tensor(
            [[math.comb(n, k) if k <= n else 0 for k in range(51)] for n in range(51)],
            dtype=torch.float32,
            device=model.device,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native MCMC requires CUDA")
def test_rotated_rejection_matches_native_rigid_rule_and_random_sequence():
    """Expanded cuboid rejection matches rigid constraints without changing random draws."""
    model = _Model([20] * 3)
    centers, rotations, dimensions = _geometry(model)
    native_rule = inspect.unwrap(perturb_positions).__globals__["_constrain_mcmc_perturbation"]
    generator = torch.Generator(device=model.device).manual_seed(1234)
    local = (torch.rand(model.positions.shape, generator=generator, device=model.device) - 0.5) * dimensions
    local[::5, 0] = dimensions[::5, 0] * 0.75  # Already outside: do not snap back.
    delta = torch.randn(model.positions.shape, generator=generator, device=model.device) * 2
    with torch.no_grad():
        model.positions.copy_(centers + (rotations * local[:, None]).sum(-1))
    before = model.positions.detach().clone()
    expected = before + (rotations * native_rule(local, delta, dimensions)[:, None]).sum(-1)
    model.accept_cuboid_perturbation(before + (rotations * delta[:, None]).sum(-1))
    torch.testing.assert_close(model.positions, expected, atol=5e-6, rtol=1e-6)

    # 1.3x each dimension (not volume), accepting beyond the original box.
    with torch.no_grad():
        model.positions.copy_(centers)
    delta.zero_()
    delta[:, 0] = 1.2
    proposal = centers + (rotations * delta[:, None]).sum(-1)
    model.accept_cuboid_perturbation(proposal)
    assert torch.equal(model.positions, proposal)

    strategy = MCMCStrategy.__new__(MCMCStrategy)
    torch.nn.Module.__init__(strategy)
    strategy.config = SimpleNamespace(perturb=SimpleNamespace(move_outside_of_cuboid=False))
    strategy.perturb_noise_lr = {"barriers": 500.0}
    before = model.positions.detach().clone()
    reference_rng = torch.Generator(device=model.device).manual_seed(32)
    actual_rng = torch.Generator(device=model.device).manual_seed(32)
    proposal = before.clone()
    perturb_positions(
        proposal, model.rotations, model.scales, model.densities[:, 0], 0.0016 * 500, generator=reference_rng
    )
    local = ((before - centers)[:, :, None] * rotations).sum(-2)
    delta = ((proposal - before)[:, :, None] * rotations).sum(-2)
    expected = before + (rotations * native_rule(local, delta, dimensions)[:, None]).sum(-1)
    with patch("nurec.models.gaussians.strategies.mcmc.get_cuda_generator", return_value=actual_rng):
        strategy.perturb_gaussians("barriers", model)
    torch.testing.assert_close(model.positions, expected, atol=5e-6, rtol=1e-6)
    assert torch.equal(reference_rng.get_state(), actual_rng.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native MCMC requires CUDA")
def test_track_local_topology_preserves_identity_optimizer_and_visibility_alignment():
    """Relocation and growth preserve each track and publish all row-aligned state together."""
    model = _Model([20] * 3)
    # Materialize Adam state before changing the native row layout.
    sum(p.square().mean() for p in model.get_parameters().values()).backward()
    optimizer = model.optimizers[0]["optimizer"]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    centers, _, _ = _geometry(model)
    with torch.no_grad():
        model.positions.copy_(centers)
        model.densities.fill_(-15)
        model.densities[:3] = 0.5
        model.densities[40:43] = 0.5  # Middle track has no live donor.
        model.features.copy_(model.barrier_object_id[:, None, None].expand_as(model.features))
    visibility = torch.nn.ParameterDict(
        {"barriers": torch.nn.Parameter(torch.zeros(60, device=model.device, dtype=torch.int32), requires_grad=False)}
    )
    selected = SelectedGaussianNodes(TypedModuleDict({"barriers": model}), [])
    kwargs = _topology_kwargs(model, visibility)
    weak_densities = model.densities[model.barrier_object_id == 23].detach().clone()
    assert cuboid_topology(selected, operation=MCMCOperation.RELOCATION, **kwargs) == 34
    assert torch.unique(model.barrier_object_id, return_counts=True)[1].tolist() == [20] * 3
    torch.testing.assert_close(model.positions, _geometry(model)[0])
    assert torch.all(model.features[:, 0, 0] == model.barrier_object_id)
    # Do not raise the entirely weak object's opacity to make it look alive.
    assert torch.equal(model.densities[model.barrier_object_id == 23], weak_densities)
    with torch.no_grad():
        model.densities.fill_(0.5)
    assert cuboid_topology(selected, operation=MCMCOperation.ADDITION, **kwargs) == 3
    assert torch.unique(model.barrier_object_id, return_counts=True)[1].tolist() == [21] * 3
    assert visibility["barriers"].shape == (63,)
    assert torch.all(model.features[:, 0, 0] == model.barrier_object_id)
    for parameter, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert state[key].shape == parameter.shape
    centers, rotations, _ = _geometry(model)
    before = model.positions.detach().clone()
    model.accept_cuboid_perturbation(centers + (rotations * model.positions.new_tensor([2, 0, 0])).sum(-1))
    torch.testing.assert_close(model.positions, before, atol=5e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native MCMC requires CUDA")
@pytest.mark.parametrize("operation", [MCMCOperation.RELOCATION, MCMCOperation.ADDITION])
@pytest.mark.parametrize("all_weak", [False, True])
def test_topology_never_revives_cuboids_without_live_donors(operation, all_weak):
    """Suppressed/invisible rows keep exact values and Adam state, even during publication."""
    model = _Model([20] * 3)
    sum(p.square().mean() for p in model.get_parameters().values()).backward()
    optimizer = model.optimizers[0]["optimizer"]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    visibility = torch.nn.ParameterDict(
        {"barriers": torch.nn.Parameter(torch.zeros(60, device=model.device, dtype=torch.int32), requires_grad=False)}
    )
    with torch.no_grad():
        model.densities.fill_(-15)
        model.densities[20:40] = 0.5
        visibility["barriers"][20:40] = 101  # High opacity, but no visible donor.
        if not all_weak:
            model.densities[40:43] = 0.5
    # Cuboid 7 is opacity-suppressed; 23 is invisible; 101 optionally has live donors.
    untouched = model.barrier_object_id != 101 if not all_weak else torch.ones_like(model.barrier_object_id).bool()
    before = {
        name: (
            parameter.detach()[untouched].clone(),
            {key: optimizer.state[parameter][key][untouched].clone() for key in ("exp_avg", "exp_avg_sq")},
        )
        for name, parameter in model.get_parameters().items()
    }
    ids_before = model.barrier_object_id[untouched].clone()
    visibility_before = visibility["barriers"][untouched].clone()
    selected = SelectedGaussianNodes(TypedModuleDict({"barriers": model}), [])
    kwargs = _topology_kwargs(model, visibility)
    rng_before = kwargs["generator"].get_state().clone()
    with patch(
        "nurec.models.gaussians.strategies.cuboid_mcmc.sample_mcmc_relocation", wraps=sample_mcmc_relocation
    ) as sampler:
        changed = cuboid_topology(selected, operation=operation, **kwargs)
    assert changed == (0 if all_weak else (17 if operation == MCMCOperation.RELOCATION else 3))
    assert sampler.call_count == (0 if all_weak else 1)
    for call in sampler.call_args_list:
        assert torch.all(call.kwargs["opacities"] >= kwargs["min_opacity"])
        assert call.kwargs["opacities"].numel() == 3
    expected_counts = [20, 20, 23 if not all_weak and operation == MCMCOperation.ADDITION else 20]
    assert torch.unique(model.barrier_object_id, return_counts=True)[1].tolist() == expected_counts
    untouched = model.barrier_object_id != 101 if not all_weak else torch.ones_like(model.barrier_object_id).bool()
    assert torch.equal(model.barrier_object_id[untouched], ids_before)
    assert torch.equal(visibility["barriers"][untouched], visibility_before)
    for name, parameter in model.get_parameters().items():
        values, moments = before[name]
        assert torch.equal(parameter[untouched], values)
        for key, moment in moments.items():
            assert torch.equal(optimizer.state[parameter][key][untouched], moment)
    if all_weak:
        assert torch.equal(kwargs["generator"].get_state(), rng_before)
