# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-view Adam refinement loop in ``ppisp.sh_match.optimization``.

The loop is renderer-agnostic and consumes two render closures
(``reference_render``, ``live_render``) plus a ``sample_view``
callable. Tests synthesize a small linear-renderer setup where the
optimum minimising MSE against an analytical target is the
``features_albedo`` that reproduces the per-view target exactly. With
a rank-sufficient render matrix the problem is convex and convergence
in a few hundred Adam steps is verifiable.

Coverage:
* Argument validation (every fail-loud branch in
  :func:`_validate_optimization_loop_inputs`).
* Loss decreases on a convex synthetic problem.
* ``model`` parameter ``nn.Parameter`` object identity preserved
  (loop mutates ``.data``, not the Parameter object).
* Seed reproducibility: same seed produces the same parameter
  trajectory.
* ``reference_render`` and ``live_render`` are each called once per
  iteration.
* Reference render runs under ``torch.no_grad()``.
* ``progress_log_every == 0`` silences progress lines but still emits
  the final-loss line.
* ``reference_render`` / ``live_render`` returning a non-tensor and
  ``sample_view`` returning a non-:class:`TrainingViewContext` are
  rejected with clear messages.
"""

from __future__ import annotations

import logging
import unittest

from typing import Any, Callable, Dict, List, Tuple, cast
from unittest import mock

import torch
import torch.nn as nn

from nurec.utils.io.export.ppisp_export.sh_match import optimization as optimization_module
from nurec.utils.io.export.ppisp_export.sh_match.optimization import (
    PPISPSHOptimizationAdapter,
    SHFitTrainable,
    SHMatchingOptimizationConfig,
    TrainingViewContext,
    _setup_optimization_parameters,
    _validate_optimization_loop_inputs,
    optimize_sh_to_match_ppisp,
)


# =============================================================================
# Test doubles
# =============================================================================


class _MockPPISPConfig:
    """Stand-in for ``PPISP.config`` used by the adapter wrapper."""

    def __init__(self, use_controller: bool = True) -> None:
        self.use_controller = use_controller


class _MockPPISPModule(nn.Module):
    """Trained-PPISP-shaped module: identity forward, valid ``num_cameras`` / ``exposure_params``."""

    def __init__(self, num_cameras: int = 2, num_frames: int = 3) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.config = _MockPPISPConfig(use_controller=True)
        self.exposure_params = nn.Parameter(torch.zeros(num_frames), requires_grad=False)
        self.color_params = nn.Parameter(torch.zeros(num_frames, 8), requires_grad=False)
        self.vignetting_params = nn.Parameter(torch.zeros(num_cameras, 3, 5), requires_grad=False)
        self.crf_params = nn.Parameter(torch.zeros(num_cameras, 3, 4), requires_grad=False)

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        resolution: Tuple[int, int],
        camera_idx: int,
        frame_idx: int,
        exposure_prior: object = None,
    ) -> torch.Tensor:
        return rgb


class _LinearRendererModel(nn.Module):
    """Toy SH-fit-trainable model.

    The synthetic loss couples each per-view ``M @ features_albedo``
    rendering to a fixed target via MSE. ``M`` is row-normalised to
    keep the per-pixel sum well below 1.0 so the identity-with-clamp
    fit-transform never saturates on the analytic optimum.
    """

    def __init__(self, num_gaussians: int, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.features_albedo = nn.Parameter(torch.randn(num_gaussians, 3, generator=generator) * 0.01)
        self.features_specular = nn.Parameter(torch.zeros(num_gaussians, 9), requires_grad=True)
        self.densities = nn.Parameter(torch.zeros(num_gaussians, 1), requires_grad=True)


def _make_synthetic_setup(
    num_views: int = 3,
    num_pixels: int = 16,
    num_gaussians: int = 6,
    seed: int = 7,
    # This is a behavior test, not a production-quality convergence test.
    # 100 Adam steps leave ample margin for the 10% loss threshold while
    # avoiding hundreds of redundant tiny CPU optimizer steps.
    num_iterations: int = 100,
) -> Tuple[
    _LinearRendererModel,
    torch.Tensor,
    PPISPSHOptimizationAdapter,
    Dict[int, torch.Tensor],
]:
    """Build (model, target_albedo, adapter, render_matrices_by_view_id).

    ``target_albedo`` is the SH-fit value the live model should reach.
    It is the source of the ``reference_render`` output via
    ``M_v @ target_albedo`` per view.
    """
    torch.manual_seed(seed)

    target_albedo = torch.rand(num_gaussians, 3) * 0.4 + 0.1
    render_matrices: Dict[int, torch.Tensor] = {}
    for view_idx in range(num_views):
        matrix_generator = torch.Generator().manual_seed(seed + 100 + view_idx)
        render_matrices[view_idx] = torch.rand(num_pixels, num_gaussians, generator=matrix_generator) / float(
            num_gaussians
        )

    model = _LinearRendererModel(num_gaussians=num_gaussians, seed=seed + 1)

    config = SHMatchingOptimizationConfig(
        num_iterations=num_iterations,
        learning_rate_albedo=5e-2,
        learning_rate_density=1e-2,
        device="cpu",
    )
    ppisp_module = _MockPPISPModule()
    adapter = PPISPSHOptimizationAdapter(
        config=config,
        ppisp_module=ppisp_module,
        camera_id=0,
        frame_id=0,
    )
    return model, target_albedo, adapter, render_matrices


def _build_view_sampler(
    num_views: int,
) -> Callable[[torch.Generator], TrainingViewContext]:
    """Sample a view uniformly at random and return a :class:`TrainingViewContext`.

    The view's ``extra`` carries the integer view index so the render
    closures can look up the right per-view matrix.
    """

    def sample_view(generator: torch.Generator) -> TrainingViewContext:
        view_idx = int(torch.randint(0, num_views, (1,), generator=generator).item())
        return TrainingViewContext(
            pixel_coords=torch.zeros(16, 2),
            resolution=(4, 4),
            extra=view_idx,
        )

    return sample_view


def _build_reference_render(
    render_matrices: Dict[int, torch.Tensor],
    target_albedo: torch.Tensor,
) -> Callable[[TrainingViewContext], torch.Tensor]:
    """Frozen reference: ``M @ target_albedo``."""

    def reference_render(view: TrainingViewContext) -> torch.Tensor:
        if not isinstance(view.extra, int):
            raise AssertionError(f"_build_reference_render expects integer view.extra, got {view.extra!r}")
        return render_matrices[view.extra] @ target_albedo

    return reference_render


def _build_live_render(
    render_matrices: Dict[int, torch.Tensor],
    model: _LinearRendererModel,
) -> Callable[[TrainingViewContext], torch.Tensor]:
    """Live: ``M @ model.features_albedo`` -- gradients flow into the live Parameter."""

    def live_render(view: TrainingViewContext) -> torch.Tensor:
        if not isinstance(view.extra, int):
            raise AssertionError(f"_build_live_render expects integer view.extra, got {view.extra!r}")
        return render_matrices[view.extra] @ model.features_albedo

    return live_render


# =============================================================================
# Argument validation
# =============================================================================


class ValidateOptimizationLoopInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.target_albedo, self.adapter, render_matrices = _make_synthetic_setup()
        self.render_matrices = render_matrices
        self.sample_view = _build_view_sampler(num_views=len(render_matrices))
        self.reference_render = _build_reference_render(render_matrices, self.target_albedo)
        self.live_render = _build_live_render(render_matrices, self.model)

    def _call(self, **overrides: Any) -> None:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "adapter": self.adapter,
            "sample_view": self.sample_view,
            "reference_render": self.reference_render,
            "live_render": self.live_render,
            "seed": 0,
            "progress_log_every": 10,
        }
        kwargs.update(overrides)
        _validate_optimization_loop_inputs(**kwargs)

    def test_valid_inputs_pass(self) -> None:
        self._call()

    def test_adapter_wrong_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"adapter must be a PPISPSHOptimizationAdapter"):
            self._call(adapter=object())

    def test_model_without_parameters_raises(self) -> None:
        bad = mock.Mock(spec=[])
        with self.assertRaisesRegex(TypeError, r"model must expose a parameters\(\) iterator"):
            self._call(model=bad)

    def test_model_missing_required_attribute_raises(self) -> None:
        class _Incomplete(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features_albedo = nn.Parameter(torch.zeros(2, 3))
                self.features_specular = nn.Parameter(torch.zeros(2, 9))

        # Single-node fallback keys the missing-attribute report under
        # ``SINGLE_NODE_FALLBACK_KEY`` (see ``iter_gaussian_nodes``).
        with self.assertRaisesRegex(
            AttributeError,
            r"gaussian node 'gaussians' is missing required SH-fit parameter 'densities'",
        ):
            self._call(model=_Incomplete())

    def test_sample_view_not_callable_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"sample_view must be callable"):
            self._call(sample_view=42)  # type: ignore[arg-type]

    def test_reference_render_not_callable_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"reference_render must be callable"):
            self._call(reference_render=42)  # type: ignore[arg-type]

    def test_live_render_not_callable_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"live_render must be callable"):
            self._call(live_render=42)  # type: ignore[arg-type]

    def test_reference_and_live_must_differ(self) -> None:
        with self.assertRaisesRegex(ValueError, r"reference_render and live_render must be distinct"):
            self._call(reference_render=self.live_render)

    def test_seed_wrong_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"seed must be int or None"):
            self._call(seed=1.5)  # type: ignore[arg-type]

    def test_seed_bool_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, r"seed must be int or None"):
            self._call(seed=True)  # type: ignore[arg-type]

    def test_seed_none_accepted(self) -> None:
        self._call(seed=None)

    def test_progress_log_every_wrong_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, r"progress_log_every must be int"):
            self._call(progress_log_every="10")  # type: ignore[arg-type]

    def test_progress_log_every_negative_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"progress_log_every must be >= 0"):
            self._call(progress_log_every=-1)

    def test_progress_log_every_bool_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, r"progress_log_every must be int"):
            self._call(progress_log_every=True)  # type: ignore[arg-type]


# =============================================================================
# Optimization behavior
# =============================================================================


class OptimizeShToMatchPpispBehaviorTest(unittest.TestCase):
    def test_loss_decreases_on_synthetic_problem(self) -> None:
        """Starting from random ``features_albedo``, the loop drives MSE down."""
        model, target_albedo, adapter, render_matrices = _make_synthetic_setup()
        sample_view = _build_view_sampler(num_views=len(render_matrices))
        reference_render = _build_reference_render(render_matrices, target_albedo)
        live_render = _build_live_render(render_matrices, model)

        with torch.no_grad():
            initial_total = 0.0
            cached_targets = []
            for view_idx, matrix in render_matrices.items():
                ref_rgb = matrix @ target_albedo
                target = adapter.fixed_ppisp(ref_rgb, torch.zeros(16, 2), (4, 4))
                cur = adapter.apply_fit_transform(matrix @ model.features_albedo)
                initial_total += torch.nn.functional.mse_loss(cur, target).item()
                cached_targets.append((view_idx, target.detach().clone()))

        optimize_sh_to_match_ppisp(
            model=model,
            adapter=adapter,
            sample_view=sample_view,
            reference_render=reference_render,
            live_render=live_render,
            seed=42,
            progress_log_every=0,
        )

        with torch.no_grad():
            final_total = 0.0
            for view_idx, target in cached_targets:
                cur = adapter.apply_fit_transform(render_matrices[view_idx] @ model.features_albedo)
                final_total += torch.nn.functional.mse_loss(cur, target).item()

        self.assertLess(
            final_total,
            initial_total * 0.1,
            f"Loss did not decrease enough: initial={initial_total:.6e}, final={final_total:.6e}",
        )

    def test_seed_reproducibility(self) -> None:
        """Same seed -> identical parameter trajectory."""
        model_a, target_a, adapter_a, matrices_a = _make_synthetic_setup(seed=11)
        model_b, target_b, adapter_b, matrices_b = _make_synthetic_setup(seed=11)

        optimize_sh_to_match_ppisp(
            model=model_a,
            adapter=adapter_a,
            sample_view=_build_view_sampler(num_views=len(matrices_a)),
            reference_render=_build_reference_render(matrices_a, target_a),
            live_render=_build_live_render(matrices_a, model_a),
            seed=2024,
            progress_log_every=0,
        )
        optimize_sh_to_match_ppisp(
            model=model_b,
            adapter=adapter_b,
            sample_view=_build_view_sampler(num_views=len(matrices_b)),
            reference_render=_build_reference_render(matrices_b, target_b),
            live_render=_build_live_render(matrices_b, model_b),
            seed=2024,
            progress_log_every=0,
        )

        self.assertTrue(
            torch.allclose(model_a.features_albedo, model_b.features_albedo, atol=0.0),
            "Same seed must produce identical features_albedo",
        )
        self.assertTrue(
            torch.allclose(model_a.features_specular, model_b.features_specular, atol=0.0),
            "Same seed must produce identical features_specular",
        )
        self.assertTrue(
            torch.allclose(model_a.densities, model_b.densities, atol=0.0),
            "Same seed must produce identical densities",
        )

    def test_in_place_mutation_preserves_parameter_identity(self) -> None:
        """Loop mutates ``.data`` in place; ``nn.Parameter`` object identity preserved."""
        model, target_albedo, adapter, render_matrices = _make_synthetic_setup(num_iterations=1)
        albedo_id = id(model.features_albedo)
        specular_id = id(model.features_specular)
        densities_id = id(model.densities)
        optimize_sh_to_match_ppisp(
            model=model,
            adapter=adapter,
            sample_view=_build_view_sampler(num_views=len(render_matrices)),
            reference_render=_build_reference_render(render_matrices, target_albedo),
            live_render=_build_live_render(render_matrices, model),
            seed=0,
            progress_log_every=0,
        )
        self.assertEqual(id(model.features_albedo), albedo_id)
        self.assertEqual(id(model.features_specular), specular_id)
        self.assertEqual(id(model.densities), densities_id)

    def test_each_callable_called_once_per_iteration(self) -> None:
        """11 iterations -> 11 reference + 11 live render calls."""
        model, target_albedo, adapter, render_matrices = _make_synthetic_setup()
        config = SHMatchingOptimizationConfig(
            num_iterations=11,
            learning_rate_albedo=adapter.config.learning_rate_albedo,
            learning_rate_density=adapter.config.learning_rate_density,
            device="cpu",
        )
        adapter_short = PPISPSHOptimizationAdapter(
            config=config,
            ppisp_module=_MockPPISPModule(),
            camera_id=0,
            frame_id=0,
        )

        ref_inner = _build_reference_render(render_matrices, target_albedo)
        live_inner = _build_live_render(render_matrices, model)
        call_log: Dict[str, int] = {"reference": 0, "live": 0}

        def counting_reference(view: TrainingViewContext) -> torch.Tensor:
            call_log["reference"] += 1
            return ref_inner(view)

        def counting_live(view: TrainingViewContext) -> torch.Tensor:
            call_log["live"] += 1
            return live_inner(view)

        optimize_sh_to_match_ppisp(
            model=model,
            adapter=adapter_short,
            sample_view=_build_view_sampler(num_views=len(render_matrices)),
            reference_render=counting_reference,
            live_render=counting_live,
            seed=0,
            progress_log_every=0,
        )

        self.assertEqual(call_log["reference"], 11)
        self.assertEqual(call_log["live"], 11)

    def test_reference_render_runs_under_no_grad(self) -> None:
        """Reference render must execute inside the loop's ``no_grad`` context."""
        model, target_albedo, adapter, render_matrices = _make_synthetic_setup()
        config = SHMatchingOptimizationConfig(
            num_iterations=1,
            learning_rate_albedo=adapter.config.learning_rate_albedo,
            learning_rate_density=adapter.config.learning_rate_density,
            device="cpu",
        )
        adapter_short = PPISPSHOptimizationAdapter(
            config=config,
            ppisp_module=_MockPPISPModule(),
            camera_id=0,
            frame_id=0,
        )

        ref_inner = _build_reference_render(render_matrices, target_albedo)
        live_inner = _build_live_render(render_matrices, model)
        observed: Dict[str, bool] = {"reference_grad_set": True}

        def watching_reference(view: TrainingViewContext) -> torch.Tensor:
            observed["reference_grad_set"] = bool(torch.is_grad_enabled())
            return ref_inner(view)

        optimize_sh_to_match_ppisp(
            model=model,
            adapter=adapter_short,
            sample_view=_build_view_sampler(num_views=len(render_matrices)),
            reference_render=watching_reference,
            live_render=live_inner,
            seed=0,
            progress_log_every=0,
        )

        self.assertFalse(
            observed["reference_grad_set"],
            "Reference render must execute inside torch.no_grad()",
        )

    def test_reference_render_returning_non_tensor_raises(self) -> None:
        model, target_albedo, adapter, render_matrices = _make_synthetic_setup()

        def bad_reference(view: TrainingViewContext) -> Any:
            return [0.0, 0.0]

        with self.assertRaisesRegex(TypeError, r"reference_render must return a torch\.Tensor"):
            optimize_sh_to_match_ppisp(
                model=model,
                adapter=adapter,
                sample_view=_build_view_sampler(num_views=len(render_matrices)),
                reference_render=bad_reference,
                live_render=_build_live_render(render_matrices, model),
                seed=0,
                progress_log_every=0,
            )

    def test_live_render_returning_non_tensor_raises(self) -> None:
        model, target_albedo, adapter, render_matrices = _make_synthetic_setup()

        def bad_live(view: TrainingViewContext) -> Any:
            return [0.0, 0.0]

        with self.assertRaisesRegex(TypeError, r"live_render must return a torch\.Tensor"):
            optimize_sh_to_match_ppisp(
                model=model,
                adapter=adapter,
                sample_view=_build_view_sampler(num_views=len(render_matrices)),
                reference_render=_build_reference_render(render_matrices, target_albedo),
                live_render=bad_live,
                seed=0,
                progress_log_every=0,
            )

    def test_sample_view_returning_wrong_type_raises(self) -> None:
        model, target_albedo, adapter, render_matrices = _make_synthetic_setup()

        def bad_sampler(generator: torch.Generator) -> Any:
            return {"not": "a TrainingViewContext"}

        with self.assertRaisesRegex(TypeError, r"sample_view must return a TrainingViewContext"):
            optimize_sh_to_match_ppisp(
                model=model,
                adapter=adapter,
                sample_view=bad_sampler,
                reference_render=_build_reference_render(render_matrices, target_albedo),
                live_render=_build_live_render(render_matrices, model),
                seed=0,
                progress_log_every=0,
            )


# =============================================================================
# Logging
# =============================================================================


class OptimizeShToMatchPpispLoggingTest(unittest.TestCase):
    def _make_short_run(
        self, num_iterations: int
    ) -> Tuple[
        _LinearRendererModel,
        PPISPSHOptimizationAdapter,
        Callable[[torch.Generator], TrainingViewContext],
        Callable[[TrainingViewContext], torch.Tensor],
        Callable[[TrainingViewContext], torch.Tensor],
    ]:
        model, target_albedo, adapter, render_matrices = _make_synthetic_setup()
        config = SHMatchingOptimizationConfig(
            num_iterations=num_iterations,
            learning_rate_albedo=adapter.config.learning_rate_albedo,
            learning_rate_density=adapter.config.learning_rate_density,
            device="cpu",
        )
        adapter_short = PPISPSHOptimizationAdapter(
            config=config,
            ppisp_module=_MockPPISPModule(),
            camera_id=0,
            frame_id=0,
        )
        return (
            model,
            adapter_short,
            _build_view_sampler(num_views=len(render_matrices)),
            _build_reference_render(render_matrices, target_albedo),
            _build_live_render(render_matrices, model),
        )

    def test_progress_log_every_zero_silences_progress_lines(self) -> None:
        model, adapter_short, sampler, reference_render, live_render = self._make_short_run(num_iterations=10)

        with self.assertLogs(optimization_module.__name__, level=logging.INFO) as captured:
            optimize_sh_to_match_ppisp(
                model=model,
                adapter=adapter_short,
                sample_view=sampler,
                reference_render=reference_render,
                live_render=live_render,
                seed=0,
                progress_log_every=0,
            )

        progress_lines = [m for m in captured.output if "iter " in m and " mse=" in m]
        done_lines = [m for m in captured.output if "done after" in m]
        self.assertEqual(progress_lines, [], "progress_log_every=0 must silence per-iter logs")
        self.assertEqual(len(done_lines), 1, "final-loss line must always be emitted")

    def test_progress_log_every_set_emits_expected_count(self) -> None:
        model, adapter_short, sampler, reference_render, live_render = self._make_short_run(num_iterations=10)

        with self.assertLogs(optimization_module.__name__, level=logging.INFO) as captured:
            optimize_sh_to_match_ppisp(
                model=model,
                adapter=adapter_short,
                sample_view=sampler,
                reference_render=reference_render,
                live_render=live_render,
                seed=0,
                progress_log_every=5,
            )

        progress_lines = [m for m in captured.output if "iter " in m and " mse=" in m]
        # 10 iterations / 5-step cadence -> 2 progress lines (at iter 5 and 10)
        self.assertEqual(len(progress_lines), 2)


class _CompositeFake(nn.Module):
    """Two-node ``GaussiansComposite``-shaped fake for the optimizer-setup path.

    ``gaussians_nodes`` is a mapping of per-node SH-bearing modules.
    The composite itself exposes a working ``parameters()`` iterator
    because it inherits from :class:`nn.Module` and registers the
    two node submodules.
    """

    def __init__(self, num_gaussians_per_node: int = 4, *, seed: int = 0) -> None:
        super().__init__()
        self._node_a = _LinearRendererModel(num_gaussians=num_gaussians_per_node, seed=seed)
        self._node_b = _LinearRendererModel(num_gaussians=num_gaussians_per_node, seed=seed + 1)
        # Typed as Dict[str, nn.Module] (not Dict[str, _LinearRendererModel])
        # so the missing-SH-param test path can drop a bare nn.Module here
        # without static-type friction.
        self.gaussians_nodes: Dict[str, nn.Module] = {"static": self._node_a, "dynamic": self._node_b}
        # Register the nodes so ``self.parameters()`` covers them. Module names
        # mirror keys for stable debuggability.
        self.add_module("node_static", self._node_a)
        self.add_module("node_dynamic", self._node_b)


class SetupOptimizationParametersCompositeTest(unittest.TestCase):
    """``_setup_optimization_parameters`` walks per-node SH params on a composite.

    Each of the three Adam parameter groups must hold one
    :class:`torch.nn.Parameter` per node, the live Parameter objects on
    the nodes must have ``requires_grad=True`` after setup, and every
    other parameter on the composite (none in the fake, but covered
    structurally by the freeze pass) must be frozen.
    """

    def setUp(self) -> None:
        self.composite = _CompositeFake(num_gaussians_per_node=4, seed=11)
        self.config = SHMatchingOptimizationConfig(
            num_iterations=1,
            learning_rate_albedo=1.0e-2,
            learning_rate_density=2.0e-2,
            device="cpu",
        )

    def test_three_groups_each_with_per_node_parameters(self) -> None:
        optimizer = _setup_optimization_parameters(
            cast("SHFitTrainable", self.composite),  # type: ignore[arg-type]
            self.config,
        )
        groups = optimizer.param_groups
        self.assertEqual(len(groups), 3)
        for group in groups:
            self.assertEqual(len(group["params"]), 2)

    def test_per_group_lr_matches_config(self) -> None:
        optimizer = _setup_optimization_parameters(
            cast("SHFitTrainable", self.composite),  # type: ignore[arg-type]
            self.config,
        )
        albedo_group, specular_group, density_group = optimizer.param_groups
        self.assertAlmostEqual(albedo_group["lr"], self.config.learning_rate_albedo)
        self.assertAlmostEqual(specular_group["lr"], self.config.resolved_learning_rate_specular)
        self.assertAlmostEqual(density_group["lr"], self.config.learning_rate_density)

    def test_groups_reference_live_node_parameters(self) -> None:
        """Optimizer groups must hold the live Parameter objects, not copies."""
        optimizer = _setup_optimization_parameters(
            cast("SHFitTrainable", self.composite),  # type: ignore[arg-type]
            self.config,
        )
        albedo_group, specular_group, density_group = optimizer.param_groups
        node_a = self.composite.gaussians_nodes["static"]
        node_b = self.composite.gaussians_nodes["dynamic"]
        albedo_ids = {id(p) for p in albedo_group["params"]}
        self.assertSetEqual(albedo_ids, {id(node_a.features_albedo), id(node_b.features_albedo)})
        specular_ids = {id(p) for p in specular_group["params"]}
        self.assertSetEqual(specular_ids, {id(node_a.features_specular), id(node_b.features_specular)})
        density_ids = {id(p) for p in density_group["params"]}
        self.assertSetEqual(density_ids, {id(node_a.densities), id(node_b.densities)})

    def test_only_sh_fit_params_have_requires_grad(self) -> None:
        # Add an extra non-SH parameter to the composite to verify it ends up frozen.
        extra = nn.Parameter(torch.zeros(2))
        self.composite.register_parameter("extra", extra)

        _setup_optimization_parameters(
            cast("SHFitTrainable", self.composite),  # type: ignore[arg-type]
            self.config,
        )
        self.assertFalse(extra.requires_grad)
        for node in self.composite.gaussians_nodes.values():
            for attr in ("features_albedo", "features_specular", "densities"):
                self.assertTrue(getattr(node, attr).requires_grad, attr)

    def test_missing_sh_param_on_one_node_raises_with_node_id(self) -> None:
        bad_node = nn.Module()
        bad_node.features_albedo = nn.Parameter(torch.zeros(2, 3))  # type: ignore[assignment]
        bad_node.features_specular = nn.Parameter(torch.zeros(2, 9))  # type: ignore[assignment]
        # densities missing on purpose.
        self.composite.gaussians_nodes["dynamic"] = bad_node

        with self.assertRaisesRegex(
            AttributeError,
            r"gaussian node 'dynamic' is missing required SH-fit parameter 'densities'",
        ):
            _setup_optimization_parameters(
                cast("SHFitTrainable", self.composite),  # type: ignore[arg-type]
                self.config,
            )


class WarmstartCompositeTest(unittest.TestCase):
    """``adapter.warmstart`` walks ``gaussians_nodes`` and bakes per node.

    Every node's ``features_albedo`` must be mutated in place (the
    bake delegates to :func:`warmstart_sh_dc_from_ppisp` which writes
    back via ``copy_()``). The return value is the
    ``(exposure, color)`` from the last node, which is identical across
    nodes by construction so the choice does not matter.
    """

    def test_warmstart_invoked_once_per_node(self) -> None:
        composite = _CompositeFake(num_gaussians_per_node=3, seed=7)
        config = SHMatchingOptimizationConfig(
            num_iterations=1,
            learning_rate_albedo=1.0e-3,
            learning_rate_density=1.0e-3,
            device="cpu",
        )
        adapter = PPISPSHOptimizationAdapter(
            config=config,
            ppisp_module=_MockPPISPModule(),
            camera_id=0,
            frame_id=0,
        )

        observed_node_albedos: List[int] = []

        def fake_warmstart(*, model: object, **_: Any) -> Tuple[float, torch.Tensor]:
            observed_node_albedos.append(id(getattr(model, "features_albedo")))
            return 0.0, torch.zeros(8)

        with mock.patch.object(optimization_module, "warmstart_sh_dc_from_ppisp", side_effect=fake_warmstart):
            adapter.warmstart(cast("SHFitTrainable", composite))  # type: ignore[arg-type]

        expected = [
            id(composite.gaussians_nodes["static"].features_albedo),
            id(composite.gaussians_nodes["dynamic"].features_albedo),
        ]
        self.assertEqual(observed_node_albedos, expected)


if __name__ == "__main__":
    unittest.main()
