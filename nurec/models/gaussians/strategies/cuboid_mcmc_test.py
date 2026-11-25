# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import math

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gsplat.strategy import MCMCOperation, perturb_positions, sample_mcmc_relocation

from nurec.config.model import TamingGSMCMCStrategyConfig
from nurec.models.base import BaseModel
from nurec.models.gaussians.construction_barriers_model import ConstructionBarriersGaussianModel
from nurec.models.gaussians.gaussians_model import BaseGaussianModel
from nurec.models.gaussians.strategies.cuboid_mcmc import (
    _apply_and_publish,
    _batched_topology_plan,
    cuboid_topology,
    growth_allocations,
)
from nurec.models.gaussians.strategies.mcmc import MCMCStrategy, TamingGSMCMCStrategy
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nurec.models.nn_extensions import TypedModuleDict


# gsplat's stable relocation formula can differ slightly from NRE's binomial sum.
_RELOCATION_RTOL = 1e-5


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


@pytest.fixture
def cuda():
    """Provide CUDA relocation helpers with device placement and no-grad shared by tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from nurec.libs import cuboid_relocation

    with torch.device("cuda"), torch.no_grad():
        yield cuboid_relocation


@pytest.mark.parametrize("counts", [[1], [1, 2, 1], [20, 31, 19], [256] * 120, [1, 257, 1027]])
@pytest.mark.parametrize("operation", [MCMCOperation.RELOCATION, MCMCOperation.ADDITION])
@pytest.mark.parametrize("mode", ["mixed", "all_dead", "one_live", "zero", "alive"])
def test_cuda_topology_matches_cdf_oracle_and_rng(cuda, counts, mode, operation):
    """Check shuffled plans, suppressed tracks, native corrections and repeatable batched RNG."""
    count = sum(counts)
    rng = torch.Generator(device="cuda").manual_seed(1997)
    order = torch.cat([rows.sort().values for rows in torch.randperm(count, generator=rng).split(counts)])
    opacities, scales = torch.rand(count, generator=rng), torch.rand(count, 3, generator=rng) + 0.1
    dead = torch.arange(count) % 4 == 0 if mode == "mixed" else torch.full((count,), mode != "alive")
    if mode in ("all_dead", "zero"):
        opacities.fill_(0.001 if mode == "all_dead" else 0)
    if mode in ("one_live", "zero"):
        dead[torch.stack([rows[0] for rows in order.split(counts)])] = False
    elif mode == "mixed":
        dead[order[: sum(counts[:2])]] = True
        opacities[order[counts[0] : sum(counts[:2])]] = 0
    kwargs = dict(
        binoms=torch.tensor(
            [[math.comb(n, k) if k <= n else 0 for k in range(51)] for n in range(51)], dtype=torch.float32
        ),
        min_opacity=0.005,
    )
    inputs = (order, opacities, scales, dead)
    before = tuple(value.clone() for value in inputs)
    keep, tracks = torch.ones_like(dead), []
    cap = max(counts) * len(counts) * 2
    additions = growth_allocations(counts, cap, eligible=[bool((~dead[rows]).any()) for rows in order.split(counts)])
    for rows, addition in zip(order.split(counts), additions):
        targets, candidates = rows[dead[rows]], rows[~dead[rows]]
        if operation == MCMCOperation.ADDITION:
            targets = rows.new_empty(addition)
        if targets.numel() and opacities[candidates].any():
            tracks.append((targets, candidates))
            if operation == MCMCOperation.RELOCATION:
                keep[targets] = False
    for seed in (0, 31, 1234) * 2:
        rng.manual_seed(seed)
        expected_rng = torch.Generator(device="cuda").manual_seed(seed)
        draws = torch.rand(sum(len(targets) for targets, _ in tracks), dtype=torch.float64, generator=expected_rng)
        actual = _batched_topology_plan(
            order, counts, opacities, scales, dead, generator=rng, operation=operation, cap=cap, **kwargs
        )
        torch.testing.assert_close(inputs, before, rtol=0, atol=0)
        assert torch.equal(rng.get_state(), expected_rng.get_state())
        if not tracks:
            assert actual is None
            continue
        assert actual is not None and torch.equal(actual.keep, keep)
        references = []
        for (targets, candidates), uniforms in zip(tracks, draws.split([len(rows) for rows, _ in tracks])):
            cdf = opacities[candidates].double().cumsum(0)
            with patch("torch.multinomial", return_value=torch.searchsorted(cdf, uniforms * cdf[-1], right=True)):
                references.append(
                    sample_mcmc_relocation(opacities, scales, len(targets), candidate_indices=candidates, **kwargs)
                )
        for value, field in zip(
            (actual.source_indices, actual.revised_opacities, actual.revised_scales),
            ("sampled_indices", "revised_opacities", "revised_scales"),
        ):
            rtol = _RELOCATION_RTOL if value.is_floating_point() else 0
            expected = torch.cat([getattr(ref, field) for ref in references])
            torch.testing.assert_close(value, expected, rtol=rtol, atol=0)


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
@pytest.mark.parametrize("count", [20, 1001])
def test_rotated_rejection_matches_native_rigid_rule_and_random_sequence(count):
    """Expanded cuboid rejection matches rigid constraints without changing random draws."""
    model = _Model([count] * 3)
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
@pytest.mark.parametrize("threshold", [None, 0, 100])
@pytest.mark.parametrize("relocation", [False, True])
def test_proxy_visibility_is_preserved_without_enabling_cuboid_death(threshold, relocation):
    model = _Model([20] * 3)
    with torch.no_grad():
        model.densities.fill_(0.5)
    visibility = torch.nn.ParameterDict(
        {
            "barriers": torch.nn.Parameter(
                torch.full((60,), 101, device=model.device, dtype=torch.int32), requires_grad=False
            )
        }
    )
    selected = SelectedGaussianNodes(TypedModuleDict({"barriers": model}), [])
    strategy = TamingGSMCMCStrategy.__new__(TamingGSMCMCStrategy)
    torch.nn.Module.__init__(strategy)
    strategy.config = TamingGSMCMCStrategyConfig.model_validate(
        {
            "name": "taminggs_mcmc",
            "independent_layer_max_n_gaussians": {"barriers": 256000},
            "opacity_threshold": 0.005,
            "print_stats": False,
            "relocate": {"max_invisible_steps": threshold},
            "add": {"taminggs": {"visibility_weight": 5.0}},
        }
    )
    strategy.invisible_steps = visibility
    strategy.binoms = _topology_kwargs(model, visibility)["binoms"]
    with torch.no_grad():
        changed = strategy._cuboid_topology(selected, relocation=relocation)
    expected_added = 3 if not relocation and threshold in (None, 0) else 0
    assert changed == expected_added
    assert model.get_num_gaussians() == 60 + expected_added
    assert visibility["barriers"].shape == (60 + expected_added,)
    if expected_added:
        assert torch.all(visibility["barriers"][-expected_added:] == 0)
        assert torch.any(visibility["barriers"][:60] == 101)
    else:
        assert torch.all(visibility["barriers"] == 101)


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
    before = (model.densities.detach().sigmoid().flatten(), model.scales.detach().exp())
    with patch("nurec.models.gaussians.strategies.cuboid_mcmc._apply_and_publish", wraps=_apply_and_publish) as publish:
        assert cuboid_topology(selected, operation=MCMCOperation.RELOCATION, **kwargs) == 34
    plan = publish.call_args.args[3]
    for original, actual, revised in zip(
        before,
        (model.densities.sigmoid().flatten(), model.scales.exp()),
        (plan.revised_opacities, plan.revised_scales),
    ):
        original[plan.source_indices] = revised
        torch.testing.assert_close(actual, torch.cat((original[plan.keep], revised)))
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
    changed = cuboid_topology(selected, operation=operation, **kwargs)
    assert changed == (0 if all_weak else (17 if operation == MCMCOperation.RELOCATION else 3))
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


def test_cuda_sampling_distribution_chunks_and_rng(cuda):
    """Cover unequal totals, zero weights, empty/chunked tracks and no-draw RNG."""
    weights = torch.tensor([0, 1, 3, 0] + [0] * 1025 + [1, 0, 1e-30, 3e-30, 0])
    rows, counts = torch.arange(weights.numel()), [4, 0, 1027, 3]
    rng = torch.Generator(device="cuda").manual_seed(42)
    state = rng.get_state()
    assert cuda.sample_tracks(rows, weights, counts, [(0, 0)] * 4, rng).numel() == 0
    assert torch.equal(rng.get_state(), state)
    sources = cuda.sample_tracks(rows, weights, counts, [(200000, 1), (0, 0), (513, 1), (200000, 1)], rng)
    first, chunked, last = sources.split([200000, 513, 200000])
    assert (first < 4).all() and (chunked == 1029).all() and (last >= 1031).all()
    assert (weights[sources] > 0).all()
    # More than five standard deviations from the expected 25% frequency.
    for sampled, first_positive in [(first, 1), (last, 1031)]:
        assert abs((sampled == first_positive).float().mean().item() - 0.25) < 0.005


def test_cuda_binding_endpoints_and_validation(cuda):
    """Empty/invalid CDF inputs and rounded endpoints must not admit zero-weight donors."""
    weights, offsets = torch.empty(0), torch.zeros(1, dtype=torch.int64)
    cdf = weights.double()
    cuda._cuda.track_cdf(weights, offsets, cdf)
    with pytest.raises(RuntimeError, match="dtype"):
        cuda._cuda.track_cdf(weights.double(), offsets, cdf)
    with pytest.raises(RuntimeError, match="lengths"):
        cuda._cuda.track_cdf(weights, offsets, torch.empty(1, dtype=torch.float64))
    sources = torch.empty(2, dtype=torch.int64)
    cuda._cuda.sample_tracks(
        torch.arange(4),
        torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64),
        torch.tensor([0, 4]),
        torch.tensor([0, 2]),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        sources,
    )
    assert (sources == 1).all()
    # Malformed empty-track draws must not read the CDF or overwrite outputs.
    cuda._cuda.sample_tracks(
        torch.empty(0, dtype=torch.int64),
        cdf,
        torch.tensor([0, 0]),
        torch.tensor([0, 2]),
        torch.zeros(2, dtype=torch.float64),
        sources,
    )
    assert (sources == 1).all()


@pytest.mark.parametrize("ratio_cap", [1, 3, 51])
@pytest.mark.parametrize("min_opacity", [0.005, 1e-36, 2.0**-23])
def test_cuda_correction_strides_and_graph_replay(cuda, ratio_cap, min_opacity):
    """Correction matches gsplat and captures without host readbacks on a side stream."""
    from gsplat.relocation import compute_relocation

    # Use a complete, strided permutation and repeated donors beyond the ratio cap.
    order = torch.tensor([2, 9, 0, 9, 1, 9])[::2]
    opacities = torch.tensor([0.0, 9, 0.5, 9, 1.0, 9])[::2]
    scales = torch.arange(1, 19, dtype=torch.float32).reshape(3, 6)[:, ::2]
    sources = torch.tensor([0] + [2] * ratio_cap)
    dead = torch.tensor([False, True, False])
    sorted_dead, group, metadata = cuda.prepare_tracks(order, [3], opacities, dead)
    binoms = (
        torch.tensor(
            [[math.comb(n, k) if k <= n else 0 for k in range(ratio_cap)] for n in range(ratio_cap)],
            dtype=torch.float32,
        )
        .T.contiguous()
        .T
    )
    expected = compute_relocation(
        opacities[sources],
        scales[sources],
        torch.tensor([2] + [ratio_cap] * ratio_cap),
        binoms.contiguous(),
        min_opacity,
    )
    args = (order, sorted_dead, group, metadata, sources, opacities, scales, binoms, min_opacity)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cuda.correct_relocation(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = cuda.correct_relocation(*args)
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(actual[0], ~dead, rtol=0, atol=0)
        torch.testing.assert_close(actual[1:], expected, rtol=_RELOCATION_RTOL, atol=0)
    assert torch.all(torch.isfinite(actual[2]) & (actual[2] > 0))
