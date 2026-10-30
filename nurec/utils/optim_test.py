# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SelectiveAdam helpers and the real CUDA-backed wrapper.

Pure helper tests run on CPU. Tests that call ``SelectiveAdam.step()`` use the
actual gsplat fused CUDA adam kernel and self-skip when CUDA or gsplat's adam
kernel is unavailable.
"""

import pytest
import torch
import torch.nn as nn

import nurec.utils.optim as optim_module

from nurec.utils.optim import (
    SelectiveAdam,
    classify_gaussian_optimizer_group,
    lr_scheduler_step_update_count,
    optim_fns,
)


class TestLRSchedulerStepUpdateCount:
    def test_equal_epoch_lengths_preserve_frequency(self) -> None:
        assert [
            lr_scheduler_step_update_count(
                batch_local_idx=batch_idx,
                frequency=2,
                configured_steps_per_epoch=6,
                effective_steps_per_epoch=6,
            )
            for batch_idx in range(6)
        ] == [0, 1, 0, 1, 0, 1]

    @pytest.mark.parametrize(
        ("frequency", "expected"),
        [
            (1, [3, 3, 4]),
            (2, [1, 2, 2]),
        ],
    )
    def test_short_epoch_reaches_configured_scheduler_position(self, frequency: int, expected: list[int]) -> None:
        assert [
            lr_scheduler_step_update_count(
                batch_local_idx=batch_idx,
                frequency=frequency,
                configured_steps_per_epoch=10,
                effective_steps_per_epoch=3,
            )
            for batch_idx in range(3)
        ] == expected

    @pytest.mark.parametrize(
        "overrides",
        [
            {"batch_local_idx": -1},
            {"frequency": 0},
            {"configured_steps_per_epoch": 0},
            {"effective_steps_per_epoch": 0},
        ],
    )
    def test_rejects_non_positive_inputs(self, overrides: dict[str, int]) -> None:
        kwargs = {
            "batch_local_idx": 0,
            "frequency": 1,
            "configured_steps_per_epoch": 1,
            "effective_steps_per_epoch": 1,
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError):
            lr_scheduler_step_update_count(**kwargs)

    def test_real_scheduler_reaches_same_trajectory_across_resume(self) -> None:
        def make_optimizer_and_scheduler():
            parameter = torch.nn.Parameter(torch.ones(()))
            optimizer = torch.optim.SGD([parameter], lr=1.0)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            return parameter, optimizer, scheduler

        def advance(parameter, optimizer, scheduler, batch_indices: list[int]) -> None:
            for batch_idx in batch_indices:
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                for _ in range(
                    lr_scheduler_step_update_count(
                        batch_local_idx=batch_idx,
                        frequency=1,
                        configured_steps_per_epoch=10,
                        effective_steps_per_epoch=3,
                    )
                ):
                    scheduler.step()

        full_parameter, full_optimizer, full_scheduler = make_optimizer_and_scheduler()
        advance(full_parameter, full_optimizer, full_scheduler, [0, 1, 2])

        resumed_parameter, resumed_optimizer, resumed_scheduler = make_optimizer_and_scheduler()
        advance(resumed_parameter, resumed_optimizer, resumed_scheduler, [0])
        optimizer_state = resumed_optimizer.state_dict()
        scheduler_state = resumed_scheduler.state_dict()

        restored_parameter, restored_optimizer, restored_scheduler = make_optimizer_and_scheduler()
        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)
        advance(restored_parameter, restored_optimizer, restored_scheduler, [1, 2])

        assert full_scheduler.last_epoch == restored_scheduler.last_epoch == 10
        assert full_scheduler.get_last_lr() == pytest.approx(restored_scheduler.get_last_lr())
        assert full_scheduler.get_last_lr() == pytest.approx([0.9**10])


class TestClassifyGaussianOptimizerGroup:
    """Tests for classify_gaussian_optimizer_group."""

    def test_densities_suffix_is_density(self):
        """Classify density parameter groups so density selective flags are applied."""
        assert classify_gaussian_optimizer_group([{"name": "dynamic_rigids.densities"}]) == "density"

    def test_scales_suffix_is_scale(self):
        """Classify scale parameter groups so scale selective flags are applied."""
        assert classify_gaussian_optimizer_group([{"name": "road.scales"}]) == "scale"

    def test_other_suffix_is_general(self):
        """Treat position, rotation, and feature groups as general selective parameters."""
        assert classify_gaussian_optimizer_group([{"name": "background.means"}]) == "general"
        assert classify_gaussian_optimizer_group([{"name": "background.rotations"}]) == "general"

    def test_bare_suffix_without_dot(self):
        """Support optimizer group names that are already bare parameter names."""
        # rsplit on a name with no '.' still yields the whole token as the suffix.
        assert classify_gaussian_optimizer_group([{"name": "densities"}]) == "density"
        assert classify_gaussian_optimizer_group([{"name": "scales"}]) == "scale"

    def test_empty_param_groups_is_general(self):
        """Fall back to general behavior when an optimizer has no named groups."""
        assert classify_gaussian_optimizer_group([]) == "general"

    def test_missing_name_key_is_general(self):
        """Fall back to general behavior when the group lacks a name field."""
        assert classify_gaussian_optimizer_group([{}]) == "general"

    def test_none_name_is_general(self):
        """Fall back to general behavior when the group name is explicitly None."""
        assert classify_gaussian_optimizer_group([{"name": None}]) == "general"

    def test_empty_name_is_general(self):
        """Fall back to general behavior when the group name is empty."""
        assert classify_gaussian_optimizer_group([{"name": ""}]) == "general"

    def test_only_first_group_is_inspected(self):
        """Document that one optimizer is expected to contain one Gaussian parameter kind."""
        # All groups in a single optimizer share the same Gaussian kind; the
        # classifier only reads the first group's name.
        groups = [{"name": "road.scales"}, {"name": "road.densities"}]
        assert classify_gaussian_optimizer_group(groups) == "scale"


class TestOptimFnsRegistry:
    """Tests for the optim_fns name -> optimizer registry."""

    def test_selective_adam_maps_to_wrapper(self):
        """Ensure config name ``selective_adam`` instantiates the local PL wrapper."""
        assert optim_fns["selective_adam"] is SelectiveAdam

    def test_known_optimizer_names_present(self):
        """Keep the optimizer registry compatible with existing config names."""
        assert set(optim_fns) >= {"fused_adam", "selective_adam", "adam", "adamw", "sgd"}

    def test_torch_builtin_mappings(self):
        """Verify non-Gaussian optimizer aliases still point at PyTorch implementations."""
        assert optim_fns["adam"] is torch.optim.Adam
        assert optim_fns["adamw"] is torch.optim.AdamW
        assert optim_fns["sgd"] is torch.optim.SGD


def _skip_without_real_kernel() -> None:
    """Skip tests that require the real gsplat fused CUDA adam kernel."""
    if not torch.cuda.is_available():
        pytest.skip("SelectiveAdam CUDA tests require torch.cuda.is_available()")
    if not optim_module.has_adam():
        pytest.skip("SelectiveAdam CUDA tests require gsplat's compiled adam kernel")


def _make_optimizer(
    n: int = 4,
    d: int = 3,
    selective_enabled: bool = True,
    lr: float = 0.1,
) -> tuple[SelectiveAdam, nn.Parameter]:
    """Build a SelectiveAdam over a single CUDA ``[n, d]`` parameter."""
    _skip_without_real_kernel()
    param = nn.Parameter(torch.zeros(n, d, device="cuda"))
    return SelectiveAdam([param], lr=lr, selective_enabled=selective_enabled), param


def _ones_grad(param: nn.Parameter) -> None:
    param.grad = torch.ones_like(param)


class TestSelectiveAdamConstruction:
    """Construction-guard behavior."""

    def test_raises_without_compiled_kernel(self, monkeypatch):
        """Fail at construction when gsplat's fused Adam kernel is unavailable."""
        monkeypatch.setattr(optim_module, "has_adam", lambda: False)
        with pytest.raises(RuntimeError, match="fused CUDA adam kernel"):
            SelectiveAdam([nn.Parameter(torch.zeros(4, 3))])

    def test_constructs_when_kernel_available(self):
        """Verify the wrapper initializes its mask state when the real kernel exists."""
        opt, _ = _make_optimizer(selective_enabled=True)
        assert isinstance(opt, SelectiveAdam)
        assert opt.selective_enabled is True
        assert opt._visibility is None

    def test_dense_step_runs_real_kernel(self):
        """Smoke-test the dense path against the real CUDA kernel."""
        opt, param = _make_optimizer(selective_enabled=False)
        before = param.detach().clone()
        _ones_grad(param)
        opt.step()
        torch.cuda.synchronize()
        assert not torch.equal(param.detach(), before)


class TestSelectiveAdamOnesMaskShapeGuard:
    """The dense all-ones mask requires a uniform leading Gaussian dim across all param groups.

    These are CPU-only: construction and ``_get_ones_mask`` do not invoke the
    fused CUDA kernel.
    """

    def _cpu_optimizer(self, monkeypatch, param_groups) -> SelectiveAdam:
        # Pretend the fused kernel is present so construction proceeds without CUDA; the shape
        # guard under test runs entirely on CPU tensors.
        monkeypatch.setattr(optim_module, "has_adam", lambda: True)
        return SelectiveAdam(param_groups, lr=0.1, selective_enabled=False)

    def test_uniform_leading_dim_across_groups_builds_mask(self, monkeypatch):
        """Multiple param groups sharing N produce a single (N,) all-visible mask."""
        n = 5
        groups = [
            {"params": [nn.Parameter(torch.zeros(n, 3))], "name": "background.means"},
            {"params": [nn.Parameter(torch.zeros(n, 4))], "name": "background.rotations"},
        ]
        opt = self._cpu_optimizer(monkeypatch, groups)
        mask = opt._get_ones_mask()
        assert mask.shape == (n,)
        assert mask.dtype == torch.bool
        assert bool(mask.all().item())

    def test_mismatched_leading_dim_raises_clear_assertion(self, monkeypatch):
        """A group whose leading dim differs from N must fail loudly, not mask the wrong rows."""
        groups = [
            {"params": [nn.Parameter(torch.zeros(5, 3))], "name": "background.means"},
            {"params": [nn.Parameter(torch.zeros(4, 4))], "name": "background.rotations"},
        ]
        opt = self._cpu_optimizer(monkeypatch, groups)
        with pytest.raises(AssertionError, match="same leading Gaussian dimension"):
            opt._get_ones_mask()

    def test_scalar_parameter_raises_clear_assertion(self, monkeypatch):
        """Scalar parameters are rejected before attempting to read a leading dimension."""
        groups = [
            {"params": [nn.Parameter(torch.zeros(5, 3))], "name": "background.means"},
            {"params": [nn.Parameter(torch.tensor(0.0))], "name": "background.scalar"},
        ]
        opt = self._cpu_optimizer(monkeypatch, groups)
        with pytest.raises(AssertionError, match="must have a leading Gaussian dimension"):
            opt._get_ones_mask()

    def test_mask_cache_reused_for_valid_groups(self, monkeypatch):
        """Repeated lookups reuse the cached mask once the shape guard passes (CPU path)."""
        n = 6
        groups = [{"params": [nn.Parameter(torch.zeros(n, 3))], "name": "background.means"}]
        opt = self._cpu_optimizer(monkeypatch, groups)
        first = opt._get_ones_mask()
        second = opt._get_ones_mask()
        assert first is second
        assert first.shape == (n,)

    def test_shape_guard_runs_on_every_lookup(self, monkeypatch):
        """A leading-dim change after construction (e.g. densification) is caught on next lookup.

        The check must not be one-time: replacing a parameter tensor with a different N between
        steps should be rejected, not silently masked with a stale-shaped mask.
        """
        n = 4
        param = nn.Parameter(torch.zeros(n, 3))
        groups = [{"params": [param], "name": "background.means"}]
        opt = self._cpu_optimizer(monkeypatch, groups)
        assert opt._get_ones_mask().shape == (n,)

        # Simulate a densification/relocation that swaps in a differently-sized tensor.
        opt.param_groups[0]["params"][0] = nn.Parameter(torch.zeros(n + 1, 3))
        # First (and only) param now defines N = n + 1, so a fresh uniform mask is fine here;
        # add a second, mismatched param to prove the per-lookup guard still fires.
        opt.param_groups[0]["params"].append(nn.Parameter(torch.zeros(n, 3)))
        with pytest.raises(AssertionError, match="same leading Gaussian dimension"):
            opt._get_ones_mask()


class TestSelectiveAdamNumGaussians:
    """_num_gaussians reports the shared leading dimension that gates empty layers."""

    def test_reports_leading_dim(self, monkeypatch):
        """Report the Gaussian count from a populated layer's parameters."""
        monkeypatch.setattr(optim_module, "has_adam", lambda: True)
        opt = SelectiveAdam([nn.Parameter(torch.zeros(5, 3))], selective_enabled=True)
        assert opt._num_gaussians() == 5

    def test_zero_for_empty_param(self, monkeypatch):
        """Report zero when a layer has been emptied (leading dim 0)."""
        monkeypatch.setattr(optim_module, "has_adam", lambda: True)
        opt = SelectiveAdam([nn.Parameter(torch.zeros(0, 3))], selective_enabled=True)
        assert opt._num_gaussians() == 0


class TestSelectiveAdamEmptyLayerNoOp:
    """A zero-Gaussian layer is a no-op instead of raising for a missing mask.

    Densification/relocation can prune a selective layer to empty; requiring a visibility
    mask for it would abort training. These tests are CPU-only: the empty-layer guard returns
    before gsplat's CUDA kernel is reached, so no GPU is required (``has_adam`` is mocked True).
    """

    def _empty_optimizer(self, monkeypatch, selective_enabled: bool = True) -> SelectiveAdam:
        monkeypatch.setattr(optim_module, "has_adam", lambda: True)
        return SelectiveAdam([nn.Parameter(torch.zeros(0, 3))], lr=0.1, selective_enabled=selective_enabled)

    def test_selective_empty_layer_step_is_noop(self, monkeypatch):
        """Selective mode must not raise when the layer has no Gaussians and no mask."""
        opt = self._empty_optimizer(monkeypatch, selective_enabled=True)
        assert opt.step() is None
        assert opt._visibility is None

    def test_selective_empty_layer_with_empty_mask_is_noop(self, monkeypatch):
        """An (empty) slice staged for an empty layer is consumed and cleared, not raised."""
        opt = self._empty_optimizer(monkeypatch, selective_enabled=True)
        opt.set_visibility(torch.empty(0))
        assert opt.step() is None
        assert opt._visibility is None

    def test_dense_empty_layer_step_is_noop(self, monkeypatch):
        """Dense mode also short-circuits an empty layer before the kernel."""
        opt = self._empty_optimizer(monkeypatch, selective_enabled=False)
        assert opt.step() is None

    def test_empty_layer_runs_closure_and_returns_loss(self, monkeypatch):
        """The no-op path still honors the optimizer.step(closure) contract."""
        opt = self._empty_optimizer(monkeypatch, selective_enabled=True)
        sentinel = torch.tensor(2.5)
        observed = {"grad_enabled": None}

        def closure():
            observed["grad_enabled"] = torch.is_grad_enabled()
            return sentinel

        assert opt.step(closure) is sentinel
        # step() is @torch.no_grad(), but the closure must still run with grad enabled.
        assert observed["grad_enabled"] is True

    def test_nonempty_layer_without_mask_still_raises(self, monkeypatch):
        """The guard must not mask the real error: a populated layer needs a mask."""
        monkeypatch.setattr(optim_module, "has_adam", lambda: True)
        groups = [{"params": [nn.Parameter(torch.zeros(4, 3))], "name": "background.means"}]
        opt = SelectiveAdam(groups, selective_enabled=True)
        with pytest.raises(RuntimeError, match=r"background\.means.*selective_enabled=False"):
            opt.step()


class TestSelectiveAdamMaskLifecycle:
    """set_visibility stores a mask; step consumes (clears) it exactly once."""

    def test_set_visibility_stores_mask(self):
        """Verify callers can stage a visibility mask for the next optimizer step."""
        opt, _ = _make_optimizer(selective_enabled=True)
        vis = torch.tensor([True, False, True, True], device="cuda")
        opt.set_visibility(vis)
        assert opt._visibility is vis

    def test_set_visibility_normalizes_numeric_mask(self):
        """Convert numeric visibility to the Boolean mask required by gsplat."""
        opt, _ = _make_optimizer(selective_enabled=True)
        opt.set_visibility(torch.tensor([1.0, 0.0, 1.0, 1.0], device="cuda"))
        assert opt._visibility is not None
        assert opt._visibility.dtype == torch.bool
        assert torch.equal(opt._visibility, torch.tensor([True, False, True, True], device="cuda"))

    def test_step_consumes_and_clears_mask(self):
        """Ensure a staged mask is one-shot state and cannot leak into later steps."""
        opt, param = _make_optimizer(selective_enabled=True)
        _ones_grad(param)
        opt.set_visibility(torch.tensor([1.0, 0.0, 1.0, 1.0], device="cuda"))
        opt.step()
        assert opt._visibility is None

    def test_boolean_mask_reaches_gsplat_without_another_conversion(self, monkeypatch):
        """A shared Boolean view must reach every fused optimizer without a new tensor."""

        opt, param = _make_optimizer(selective_enabled=True)
        _ones_grad(param)
        visibility = torch.tensor([True, False, True, False], device="cuda")
        captured = {}

        def capture_mask(_optimizer, mask):
            captured["mask"] = mask

        monkeypatch.setattr(optim_module._GsplatSelectiveAdam, "_selective_update", capture_mask)
        opt.set_visibility(visibility)
        opt.step()

        assert captured["mask"] is visibility


class TestSelectiveAdamSelectivePath:
    """selective_enabled=True branch."""

    def test_step_without_mask_raises(self):
        """Selective mode should fail loudly if no per-step visibility was provided."""
        opt, _ = _make_optimizer(selective_enabled=True)
        with pytest.raises(RuntimeError, match="no visibility mask"):
            opt.step()

    def test_step_with_empty_mask_raises(self):
        """Treat empty visibility tensors like missing masks in selective mode."""
        opt, _ = _make_optimizer(selective_enabled=True)
        opt.set_visibility(torch.empty(0, device="cuda"))
        with pytest.raises(RuntimeError, match="no visibility mask"):
            opt.step()

    def test_real_kernel_respects_and_does_not_mutate_shared_mask(self):
        """Prove NuRec's current gsplat call leaves shared mask storage unchanged."""
        opt, param = _make_optimizer(selective_enabled=True)
        before = param.detach().clone()
        _ones_grad(param)

        # Match the per-layer views shared by GaussiansSystem: this contiguous Boolean slice
        # starts inside a larger allocation, so checking the full backing tensor catches writes
        # both within the mask and outside the view.
        visibility_backing = torch.tensor(
            [False, True, False, True, False, True],
            dtype=torch.bool,
            device="cuda",
        )
        visibility = visibility_backing[1:5]
        assert visibility.is_contiguous()
        assert visibility.storage_offset() > 0
        backing_before = visibility_backing.clone()
        visibility_before = visibility.clone()

        opt.set_visibility(visibility)
        opt.step()
        torch.cuda.synchronize()

        assert torch.equal(visibility_backing, backing_before)
        assert torch.equal(visibility, visibility_before)
        after = param.detach()
        assert not torch.equal(after[0], before[0])
        assert torch.equal(after[1], before[1])
        assert not torch.equal(after[2], before[2])
        assert torch.equal(after[3], before[3])

    def test_selective_step_does_not_retain_visibility_reference(self):
        """The wrapper must not hold the visibility tensor after the step consumes it."""
        opt, param = _make_optimizer(selective_enabled=True)
        _ones_grad(param)
        opt.set_visibility(torch.tensor([1.0, 0.0, 1.0, 1.0], device="cuda"))
        opt.step()
        # The one-shot mask slot is cleared and no other attribute pins the mask.
        assert opt._visibility is None
        assert not hasattr(opt, "last_visibility")


class TestSelectiveAdamDensePath:
    """selective_enabled=False branch behaves like plain Adam (dense)."""

    def test_stored_mask_ignored_updates_all_rows(self):
        """Dense mode should update every row even if a restrictive mask was staged."""
        opt, param = _make_optimizer(n=4, selective_enabled=False)
        before = param.detach().clone()
        _ones_grad(param)
        opt.set_visibility(torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"))  # must be ignored
        opt.step()
        torch.cuda.synchronize()
        assert torch.all(param.detach() != before)

    def test_dense_path_matches_all_visible_selective_path_over_multiple_steps(self):
        """Dense mode should be numerically equivalent to selective mode with all rows visible."""
        dense_opt, dense_param = _make_optimizer(n=3, d=2, selective_enabled=False, lr=0.05)
        selective_opt, selective_param = _make_optimizer(n=3, d=2, selective_enabled=True, lr=0.05)
        initial = torch.tensor(
            [[0.0, 0.5], [1.0, -1.0], [2.0, -2.0]],
            device="cuda",
        )
        dense_param.data.copy_(initial)
        selective_param.data.copy_(initial)

        gradients = [
            torch.tensor([[1.0, -2.0], [0.5, 0.25], [-1.5, 3.0]], device="cuda"),
            torch.tensor([[-0.25, 0.75], [2.0, -0.5], [1.0, -1.0]], device="cuda"),
        ]
        ignored_mask = torch.tensor([1.0, 0.0, 0.0], device="cuda")
        all_visible_mask = torch.ones(3, device="cuda")
        for grad in gradients:
            dense_param.grad = grad.clone()
            selective_param.grad = grad.clone()

            # Dense mode ignores any stored visibility and synthesizes an all-ones mask.
            dense_opt.set_visibility(ignored_mask)
            dense_opt.step()

            # The equivalent selective path is the same kernel with every row visible.
            selective_opt.set_visibility(all_visible_mask)
            selective_opt.step()

        torch.cuda.synchronize()
        assert torch.allclose(dense_param.detach(), selective_param.detach(), rtol=1e-5, atol=1e-6)

        dense_state = dense_opt.state[dense_param]
        selective_state = selective_opt.state[selective_param]
        assert torch.allclose(dense_state["exp_avg"], selective_state["exp_avg"], rtol=1e-5, atol=1e-6)
        assert torch.allclose(
            dense_state["exp_avg_sq"],
            selective_state["exp_avg_sq"],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_dense_step_after_selective_updates_all_rows(self):
        """Flipping selective→dense should update every row regardless of a stored mask."""
        opt, param = _make_optimizer(selective_enabled=True)
        _ones_grad(param)
        opt.set_visibility(torch.tensor([1.0, 0.0, 1.0, 1.0], device="cuda"))
        opt.step()
        # Flip to dense and step again: the stored mask must be ignored and all rows move.
        opt.selective_enabled = False
        before = param.detach().clone()
        _ones_grad(param)
        opt.set_visibility(torch.tensor([1.0, 0.0, 1.0, 1.0], device="cuda"))
        opt.step()
        torch.cuda.synchronize()
        assert torch.all(param.detach() != before)

    def test_ones_mask_is_cached_by_shape_and_device(self):
        """Dense mode should reuse its synthetic all-visible mask for stable allocation."""
        opt, _ = _make_optimizer(n=4, selective_enabled=False)
        first = opt._get_ones_mask()
        second = opt._get_ones_mask()
        assert first is second
        assert first.shape == (4,)
        assert first.dtype == torch.bool
        assert first.device.type == "cuda"


class TestSelectiveAdamClosureAndZeroGrad:
    """Closure execution and zero_grad semantics."""

    def test_closure_runs_under_grad_and_loss_returned(self):
        """Match PyTorch/Lightning optimizer.step(closure) expectations."""
        opt, param = _make_optimizer(selective_enabled=False)
        _ones_grad(param)
        sentinel = torch.tensor(1.23, device="cuda")
        observed = {"grad_enabled": None}

        def closure():
            observed["grad_enabled"] = torch.is_grad_enabled()
            return sentinel

        loss = opt.step(closure)
        assert loss is sentinel
        # step() is @torch.no_grad(), but the closure must run with grad enabled.
        assert observed["grad_enabled"] is True

    def test_step_without_closure_returns_none(self):
        """Keep the no-closure step return value compatible with standard optimizers."""
        opt, param = _make_optimizer(selective_enabled=False)
        _ones_grad(param)
        assert opt.step() is None

    def test_zero_grad_always_sets_to_none(self):
        """Mirror FusedAdam's zero_grad behavior even if callers request zero tensors."""
        opt, _ = _make_optimizer(selective_enabled=False)
        param = opt.param_groups[0]["params"][0]
        param.grad = torch.ones_like(param)
        # Even when explicitly asked not to, the wrapper forces set_to_none=True.
        opt.zero_grad(set_to_none=False)
        assert param.grad is None


class TestSelectiveAdamMultiStep:
    """Multi-step selective behavior with one mask consumed per optimizer step."""

    def test_selective_mask_must_be_reset_each_step(self):
        """Require a fresh mask for every selective step; the prior mask must not persist."""
        opt, param = _make_optimizer(n=4, selective_enabled=True)

        first_mask = torch.tensor([1.0, 0.0, 0.0, 1.0], device="cuda")
        _ones_grad(param)
        opt.set_visibility(first_mask)
        opt.step()
        torch.cuda.synchronize()
        assert opt._visibility is None

        param.grad = torch.ones_like(param)
        with pytest.raises(RuntimeError, match="no visibility mask"):
            opt.step()

        second_mask = torch.tensor([0.0, 1.0, 1.0, 0.0], device="cuda")
        before_second_step = param.detach().clone()
        param.grad = torch.ones_like(param)
        opt.set_visibility(second_mask)
        opt.step()
        torch.cuda.synchronize()

        after_second_step = param.detach()
        assert torch.equal(after_second_step[0], before_second_step[0])
        assert not torch.equal(after_second_step[1], before_second_step[1])
        assert not torch.equal(after_second_step[2], before_second_step[2])
        assert torch.equal(after_second_step[3], before_second_step[3])
        assert opt._visibility is None
