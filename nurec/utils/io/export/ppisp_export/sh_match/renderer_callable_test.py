# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic CPU-only tests for :mod:`renderer_callable`.

The fakes here replace the production :class:`GaussiansComposite` /
``BaseGaussianRenderer`` with a minimal duck-typed surface so the
tests can exercise:

* Snapshot construction / validation.
* Parameter-data swap (no model deepcopy, no model mutation).
* RNG-state synchronization between reference and live closures.
* Gradient flow through the renderer's parameter dict.

These tests do not -- and cannot -- validate that the production
parameter-driven camera renderer propagates SH gradients on a real
checkpoint; that's verified once, on the user's actual scene, by the
Phase 1 spike (see ``docs/ppisp_sh_optimization_renderer_plan.md``).
"""

from __future__ import annotations

import unittest

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, cast

import torch
import torch.nn as nn

from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.io.export.ppisp_export.sh_match.optimization import TrainingViewContext
from nurec.utils.io.export.ppisp_export.sh_match.renderer_callable import (
    SH_FIT_PARAMETER_NAMES,
    SINGLE_NODE_FALLBACK_KEY,
    CalibratedBatchLoader,
    ParameterDrivenRenderer,
    SHFitGaussianNode,
    SHMatchRenderableModel,
    TrainingViewBatch,
    iter_gaussian_nodes,
    iterate_calibrated_batches,
    make_sh_match_render_callables,
    make_training_view_sampler,
    snapshot_sh_fit_parameters,
)


# =============================================================================
# Fakes
# =============================================================================


@dataclass
class _FakeRenderReturn:
    rgb: Optional[torch.Tensor]


class _FakeNode(nn.Module, SHFitGaussianNode):
    """Bare-minimum SH-fit node.

    Three parameters with shapes plausibly matching the production
    ``SHGaussianModel``: albedo / specular as ``[N, 3]``, densities as
    ``[N, 1]``. Real shapes vary but tests only need observable values.

    Nominally inherits :class:`SHFitGaussianNode` so mypy accepts
    ``Dict[str, _FakeNode] <: Mapping[str, SHFitGaussianNode]`` at
    call sites (structural-only matching is too brittle here).
    """

    def __init__(self, num_gaussians: int, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.features_albedo = nn.Parameter(torch.randn(num_gaussians, 3, generator=generator))
        self.features_specular = nn.Parameter(torch.randn(num_gaussians, 3, generator=generator))
        self.densities = nn.Parameter(torch.randn(num_gaussians, 1, generator=generator))


class _RecordingRenderer(ParameterDrivenRenderer):
    """Renderer that returns a tensor whose value depends linearly on the SH-fit params.

    The output rgb has shape ``[num_pixels, 3]`` and is computed as a
    sum of the three SH-fit parameters projected through small fixed
    weights. The linear projection is enough to:

    * Preserve gradient w.r.t. each SH-fit parameter (gradient flow
      observable on a single ``backward()``).
    * Make the rgb a deterministic function of ``gaussian_parameters``
      so determinism tests can simply call the closure twice and
      compare with :func:`torch.equal`.

    Nominally inherits :class:`ParameterDrivenRenderer` so mypy
    accepts ``_FakeModel.gaussians_renderer`` against the production
    Protocol.
    """

    def __init__(self, num_pixels: int) -> None:
        self._num_pixels = num_pixels
        self.calls: List[Dict[str, torch.Tensor]] = []

    def render(
        self,
        rendering_data: object,
        gaussian_parameters: Dict[str, torch.Tensor],
        n_active_features: int,
        extra_ray_signal_infos: Tuple[List[str], List[int], List[Callable[..., object]]],
        frame_meta: Optional[List[object]] = None,
    ) -> _FakeRenderReturn:
        del rendering_data, n_active_features, extra_ray_signal_infos, frame_meta
        self.calls.append({k: v for k, v in gaussian_parameters.items()})

        feat = gaussian_parameters["features_albedo"]
        spec = gaussian_parameters["features_specular"]
        dens = gaussian_parameters["densities"]

        rgb_seed = feat.sum() + spec.sum() + dens.sum()
        ones = torch.ones((self._num_pixels, 3), dtype=feat.dtype, device=feat.device)
        return _FakeRenderReturn(rgb=rgb_seed * ones)


@dataclass
class _RNGProbe:
    """Counts how many times :func:`torch.rand` is sampled during a call.

    Used to assert that the live closure consumes the same number of
    random draws starting from the same seed as the reference closure.
    """

    consume_per_call: int = 0
    last_random_value: Optional[torch.Tensor] = None


class _FakeModel(SHMatchRenderableModel):
    """``GaussiansComposite``-like surface for the renderer-callable tests.

    Owns ``gaussians_nodes``, the recording renderer, and an RNG probe
    that lets tests confirm the snapshot/live closures observe the
    same random draws when the contract is honored.

    Nominally inherits :class:`SHMatchRenderableModel` so mypy accepts
    ``_FakeModel`` at call sites typed against the production Protocol.
    """

    def __init__(
        self,
        node_specs: Tuple[Tuple[str, int], ...],
        *,
        num_pixels: int = 6,
        rng_probe: Optional[_RNGProbe] = None,
    ) -> None:
        self.gaussians_nodes: Dict[str, _FakeNode] = {
            node_id: _FakeNode(num_gaussians, seed=hash(node_id) & 0xFFFF) for node_id, num_gaussians in node_specs
        }
        self.gaussians_renderer: _RecordingRenderer = _RecordingRenderer(num_pixels=num_pixels)
        self.camera_extra_ray_signal_infos: Tuple[List[str], List[int], List[Callable[..., object]]] = ([], [], [])
        self._rng_probe = rng_probe

    def parameters(self):
        for node in self.gaussians_nodes.values():
            yield from node.parameters()

    def collect_gaussian_parameters(
        self,
        rendering_data: object,
        is_training_batch: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del rendering_data, is_training_batch
        if self._rng_probe is not None and self._rng_probe.consume_per_call > 0:
            sample = torch.rand(self._rng_probe.consume_per_call)
            self._rng_probe.last_random_value = sample.detach().clone()
        nodes = list(self.gaussians_nodes.values())
        feat = torch.cat([n.features_albedo for n in nodes], dim=0)
        spec = torch.cat([n.features_specular for n in nodes], dim=0)
        dens = torch.cat([n.densities for n in nodes], dim=0)
        return {
            "features_albedo": feat,
            "features_specular": spec,
            "densities": dens,
        }

    def get_n_active_features(self) -> int:
        return 1


def _make_view(*, batch_idx: int = 0) -> TrainingViewContext:
    return TrainingViewContext(
        pixel_coords=torch.zeros((6, 2), dtype=torch.int16),
        resolution=(2, 3),
        extra=TrainingViewBatch(
            batch_idx=batch_idx,
            rendering_data=object(),
            frame_meta=object(),
        ),
    )


# =============================================================================
# snapshot_sh_fit_parameters
# =============================================================================


class SnapshotShFitParametersTest(unittest.TestCase):
    def test_returns_independent_clones_per_node(self) -> None:
        model = _FakeModel((("background", 4), ("vehicle", 2)))
        snapshot = snapshot_sh_fit_parameters(model)

        self.assertEqual(set(snapshot.per_node), {"background", "vehicle"})
        for node_id, node in model.gaussians_nodes.items():
            for name in SH_FIT_PARAMETER_NAMES:
                snap_tensor = snapshot.per_node[node_id][name]
                live_tensor = getattr(node, name)
                self.assertTrue(torch.equal(snap_tensor, live_tensor.data), name)
                self.assertNotEqual(snap_tensor.data_ptr(), live_tensor.data_ptr(), name)
                self.assertFalse(snap_tensor.requires_grad, name)

    def test_raises_when_node_missing_a_parameter(self) -> None:
        model = _FakeModel((("background", 4),))
        # Drop the densities Parameter from the node's _parameters dict so
        # ``hasattr(node, "densities")`` returns False.
        del model.gaussians_nodes["background"]._parameters["densities"]
        with self.assertRaisesRegex(AttributeError, "densities"):
            snapshot_sh_fit_parameters(model)

    def test_rejects_non_mapping_gaussians_nodes(self) -> None:
        broken = object()
        with self.assertRaisesRegex(TypeError, "gaussians_nodes"):
            snapshot_sh_fit_parameters(broken)  # type: ignore[arg-type]


# =============================================================================
# make_sh_match_render_callables -- type / surface validation
# =============================================================================


class MakeShMatchRenderCallablesValidationTest(unittest.TestCase):
    def test_rejects_non_snapshot(self) -> None:
        model = _FakeModel((("background", 2),))
        with self.assertRaisesRegex(TypeError, "SHParameterSnapshot"):
            make_sh_match_render_callables(model, snapshot=object())  # type: ignore[arg-type]

    def test_rejects_non_bool_is_training_batch(self) -> None:
        model = _FakeModel((("background", 2),))
        snapshot = snapshot_sh_fit_parameters(model)
        with self.assertRaisesRegex(TypeError, "is_training_batch"):
            make_sh_match_render_callables(model, snapshot, is_training_batch=1)  # type: ignore[arg-type]

    def test_rejects_model_missing_renderer(self) -> None:
        model = _FakeModel((("background", 2),))
        snapshot = snapshot_sh_fit_parameters(model)
        del model.gaussians_renderer
        with self.assertRaisesRegex(TypeError, "gaussians_renderer"):
            make_sh_match_render_callables(model, snapshot)

    def test_rejects_view_with_wrong_extra(self) -> None:
        model = _FakeModel((("background", 2),))
        snapshot = snapshot_sh_fit_parameters(model)
        ref, _ = make_sh_match_render_callables(model, snapshot)
        bad_view = TrainingViewContext(
            pixel_coords=torch.zeros((6, 2), dtype=torch.int16),
            resolution=(2, 3),
            extra="not a TrainingViewBatch",
        )
        with self.assertRaisesRegex(TypeError, "TrainingViewBatch"):
            ref(bad_view)


# =============================================================================
# Behavior
# =============================================================================


class LiveRenderGradFlowTest(unittest.TestCase):
    """Plan test 1: gradients flow into every SH-fit parameter."""

    def test_backward_populates_grads_on_every_node(self) -> None:
        model = _FakeModel((("background", 4), ("vehicle", 2)))
        snapshot = snapshot_sh_fit_parameters(model)
        _, live_render = make_sh_match_render_callables(model, snapshot)

        rgb = live_render(_make_view())
        loss = rgb.pow(2).sum()
        loss.backward()

        for node_id, node in model.gaussians_nodes.items():
            for name in SH_FIT_PARAMETER_NAMES:
                grad = getattr(node, name).grad
                self.assertIsNotNone(grad, f"node={node_id} {name} grad is None")
                assert grad is not None  # for mypy
                self.assertTrue(torch.isfinite(grad).all(), f"node={node_id} {name} non-finite grad")
                self.assertGreater(
                    grad.abs().max().item(),
                    0.0,
                    f"node={node_id} {name} grad is zero",
                )


class ReferenceRenderDeterminismTest(unittest.TestCase):
    """Plan test 2: same view + snapshot -> same RGB (no_grad)."""

    def test_two_calls_produce_identical_tensor(self) -> None:
        model = _FakeModel((("background", 4),))
        snapshot = snapshot_sh_fit_parameters(model)
        reference_render, _ = make_sh_match_render_callables(model, snapshot)
        view = _make_view()

        with torch.no_grad():
            rgb_a = reference_render(view).clone()
            rgb_b = reference_render(view).clone()

        self.assertTrue(torch.equal(rgb_a, rgb_b))


class ReferenceRenderDoesNotMutateLiveDataTest(unittest.TestCase):
    """Plan test 3: ``param.data`` is restored after reference_render."""

    def test_live_param_data_preserved_after_reference_call(self) -> None:
        model = _FakeModel((("background", 4),))
        snapshot = snapshot_sh_fit_parameters(model)
        reference_render, _ = make_sh_match_render_callables(model, snapshot)

        before: Dict[Tuple[str, str], Tuple[torch.Tensor, int]] = {}
        for node_id, node in model.gaussians_nodes.items():
            for name in SH_FIT_PARAMETER_NAMES:
                param = getattr(node, name)
                before[(node_id, name)] = (param.data.detach().clone(), param.data.data_ptr())

        with torch.no_grad():
            reference_render(_make_view())

        for (node_id, name), (saved_value, saved_ptr) in before.items():
            param = getattr(model.gaussians_nodes[node_id], name)
            self.assertTrue(torch.equal(param.data, saved_value), f"data mutated on {node_id}.{name}")
            self.assertEqual(
                param.data.data_ptr(),
                saved_ptr,
                f"underlying storage swapped on {node_id}.{name}",
            )

    def test_swap_restores_data_on_renderer_exception(self) -> None:
        model = _FakeModel((("background", 4),))
        snapshot = snapshot_sh_fit_parameters(model)

        class _ExplodingRenderer:
            def render(self, *_: object, **__: object) -> _FakeRenderReturn:
                raise RuntimeError("boom")

        model.gaussians_renderer = _ExplodingRenderer()  # type: ignore[assignment]
        reference_render, _ = make_sh_match_render_callables(model, snapshot)

        before = model.gaussians_nodes["background"].features_albedo.data.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with torch.no_grad():
                reference_render(_make_view())
        after = model.gaussians_nodes["background"].features_albedo.data
        self.assertTrue(torch.equal(before, after))


class ReferenceVsLiveRenderConsistencyTest(unittest.TestCase):
    """Reference render with snapshot == live render when snapshot equals live."""

    def test_reference_and_live_match_when_snapshot_is_current(self) -> None:
        model = _FakeModel((("background", 4),))
        snapshot = snapshot_sh_fit_parameters(model)
        reference_render, live_render = make_sh_match_render_callables(model, snapshot)
        view = _make_view()

        with torch.no_grad():
            ref_rgb = reference_render(view).clone()
            live_rgb = live_render(view).clone()

        self.assertTrue(torch.equal(ref_rgb, live_rgb))

    def test_reference_uses_snapshot_not_live_after_perturbation(self) -> None:
        model = _FakeModel((("background", 4),))
        snapshot = snapshot_sh_fit_parameters(model)
        reference_render, live_render = make_sh_match_render_callables(model, snapshot)
        view = _make_view()

        with torch.no_grad():
            ref_before = reference_render(view).clone()
            model.gaussians_nodes["background"].features_albedo.data.add_(10.0)
            ref_after = reference_render(view).clone()
            live_after = live_render(view).clone()

        # Reference must keep producing the snapshot-driven RGB.
        self.assertTrue(torch.equal(ref_before, ref_after))
        # Live must reflect the perturbation.
        self.assertFalse(torch.equal(ref_after, live_after))


class RNGSyncBetweenReferenceAndLiveTest(unittest.TestCase):
    """Live closure restores the RNG state captured by the reference closure."""

    def test_live_consumes_same_random_draw_as_reference(self) -> None:
        probe = _RNGProbe(consume_per_call=4)
        model = _FakeModel((("background", 4),), rng_probe=probe)
        snapshot = snapshot_sh_fit_parameters(model)
        reference_render, live_render = make_sh_match_render_callables(model, snapshot)
        view = _make_view()

        torch.manual_seed(12345)
        with torch.no_grad():
            reference_render(view)
            ref_random = probe.last_random_value
            live_render(view)
            live_random = probe.last_random_value

        self.assertIsNotNone(ref_random)
        self.assertIsNotNone(live_random)
        assert ref_random is not None  # for mypy
        assert live_random is not None
        self.assertTrue(torch.equal(ref_random, live_random))

    def test_live_render_alone_does_not_crash_when_no_reference_yet(self) -> None:
        model = _FakeModel((("background", 4),))
        snapshot = snapshot_sh_fit_parameters(model)
        _, live_render = make_sh_match_render_callables(model, snapshot)
        rgb = live_render(_make_view())
        self.assertIsInstance(rgb, torch.Tensor)
        self.assertEqual(rgb.shape, (6, 3))


class RendererInputsRoutedCorrectlyTest(unittest.TestCase):
    """Renderer receives the parameters the closures collected (snapshot vs live)."""

    def test_reference_call_passes_snapshot_features(self) -> None:
        model = _FakeModel((("background", 4),))
        snapshot = snapshot_sh_fit_parameters(model)
        reference_render, _ = make_sh_match_render_callables(model, snapshot)

        with torch.no_grad():
            reference_render(_make_view())
        last_call = model.gaussians_renderer.calls[-1]
        snap_albedo = snapshot.per_node["background"]["features_albedo"]
        self.assertTrue(torch.equal(last_call["features_albedo"], snap_albedo))

    def test_live_call_passes_live_features(self) -> None:
        model = _FakeModel((("background", 4),))
        snapshot = snapshot_sh_fit_parameters(model)
        _, live_render = make_sh_match_render_callables(model, snapshot)
        # Perturb the live parameter so the snapshot and live values differ.
        model.gaussians_nodes["background"].features_albedo.data.add_(7.5)

        live_render(_make_view())
        last_call = model.gaussians_renderer.calls[-1]
        live_albedo = model.gaussians_nodes["background"].features_albedo
        self.assertTrue(torch.equal(last_call["features_albedo"], live_albedo))


class RNGStateAdvancesAcrossIterationsTest(unittest.TestCase):
    """Across iterations the global RNG advances; within an iter ref/live match."""

    def test_two_iterations_with_same_seed_match_two_serial_runs(self) -> None:
        probe = _RNGProbe(consume_per_call=2)
        model = _FakeModel((("background", 2),), rng_probe=probe)
        snapshot = snapshot_sh_fit_parameters(model)
        reference_render, live_render = make_sh_match_render_callables(model, snapshot)
        view = _make_view()

        torch.manual_seed(7)
        with torch.no_grad():
            reference_render(view)
            iter_a_ref = probe.last_random_value
            live_render(view)
            iter_a_live = probe.last_random_value
            reference_render(view)
            iter_b_ref = probe.last_random_value
            live_render(view)
            iter_b_live = probe.last_random_value

        for v in (iter_a_ref, iter_a_live, iter_b_ref, iter_b_live):
            self.assertIsNotNone(v)
        assert iter_a_ref is not None and iter_a_live is not None  # for mypy
        assert iter_b_ref is not None and iter_b_live is not None

        self.assertTrue(torch.equal(iter_a_ref, iter_a_live))
        self.assertTrue(torch.equal(iter_b_ref, iter_b_live))
        self.assertFalse(torch.equal(iter_a_ref, iter_b_ref))


# =============================================================================
# Sampler + calibrated loader (moved from renderer_callable)
# =============================================================================


@dataclass
class _FakeRenderingData:
    """Minimal RenderingData stand-in for sampler tests."""

    w: int
    h: int
    rays: torch.Tensor


@dataclass
class _FakeFrameMeta:
    pass


@dataclass
class _FakeCameraData:
    meta: List[_FakeFrameMeta]


@dataclass
class _FakeDataField:
    camera: Optional[_FakeCameraData]


@dataclass
class _FakeRenderingField:
    camera: Optional[_FakeRenderingData]


@dataclass
class _FakeBatch:
    data: _FakeDataField
    rendering: Optional[_FakeRenderingField]

    def to(self, device: Optional[torch.device]) -> "_FakeBatch":
        if device is None:
            return self
        # Pretend the move happened; a real batch deep-moves all tensors.
        return self


def _make_calibrated_batch(*, w: int = 2, h: int = 3) -> _FakeBatch:
    rays = torch.zeros((w * h, 6))
    return _FakeBatch(
        data=_FakeDataField(camera=_FakeCameraData(meta=[_FakeFrameMeta()])),
        rendering=_FakeRenderingField(camera=_FakeRenderingData(w=w, h=h, rays=rays)),
    )


def _make_uncalibrated_batch(*, w: int = 2, h: int = 3) -> _FakeBatch:
    return _FakeBatch(
        data=_FakeDataField(camera=_FakeCameraData(meta=[_FakeFrameMeta()])),
        rendering=None,
    )


class MakeTrainingViewSamplerTest(unittest.TestCase):
    """Sampler iterates the dataloader, restarting on exhaustion."""

    def test_yields_training_view_context_with_payload(self) -> None:
        batch = _make_calibrated_batch()
        sampler = make_training_view_sampler(cast(Iterable[DataAndRenderingBatch], [batch]))

        view = sampler(torch.Generator())
        self.assertIsInstance(view, TrainingViewContext)
        self.assertEqual(view.resolution, (2, 3))
        self.assertIsInstance(view.extra, TrainingViewBatch)
        assert isinstance(view.extra, TrainingViewBatch)
        self.assertEqual(view.extra.batch_idx, 0)
        assert batch.rendering is not None and batch.data.camera is not None
        self.assertIs(view.extra.rendering_data, batch.rendering.camera)
        self.assertIs(view.extra.frame_meta, batch.data.camera.meta[0])

    def test_restarts_iterator_on_exhaustion(self) -> None:
        batch = _make_calibrated_batch()
        sampler = make_training_view_sampler(cast(Iterable[DataAndRenderingBatch], [batch]))

        view_0 = sampler(torch.Generator())
        view_1 = sampler(torch.Generator())

        self.assertIsNotNone(view_0)
        self.assertIsNotNone(view_1)
        # batch_idx increments monotonically across restarts so the loop
        # can distinguish iterations.
        assert isinstance(view_0.extra, TrainingViewBatch)
        assert isinstance(view_1.extra, TrainingViewBatch)
        self.assertEqual(view_0.extra.batch_idx, 0)
        self.assertEqual(view_1.extra.batch_idx, 1)

    def test_empty_loader_raises_value_error(self) -> None:
        sampler = make_training_view_sampler([])
        with self.assertRaisesRegex(ValueError, "training dataloader is empty"):
            sampler(torch.Generator())

    def test_uncalibrated_batch_raises_with_actionable_message(self) -> None:
        batch = _make_uncalibrated_batch()
        sampler = make_training_view_sampler(cast(Iterable[DataAndRenderingBatch], [batch]))
        with self.assertRaisesRegex(ValueError, "iterate_calibrated_batches"):
            sampler(torch.Generator())


class CalibratedBatchLoaderTest(unittest.TestCase):
    """The wrapper materialises ``batch.rendering`` via ``calib_fn``."""

    def test_calib_called_per_iteration_when_rendering_missing(self) -> None:
        raw = [_make_uncalibrated_batch(), _make_uncalibrated_batch()]
        calls: List[bool] = []

        def fake_calib(data: object, *, skip_calib: bool = False) -> _FakeRenderingField:
            calls.append(skip_calib)
            return _FakeRenderingField(camera=_FakeRenderingData(w=4, h=5, rays=torch.zeros((20, 6))))

        loader = CalibratedBatchLoader(cast(Iterable[DataAndRenderingBatch], raw), calib_fn=fake_calib)
        materialised = list(loader)

        self.assertEqual(len(materialised), 2)
        self.assertEqual(calls, [True, True])
        for batch in materialised:
            self.assertIsNotNone(batch.rendering)
            assert batch.rendering is not None
            self.assertIsNotNone(batch.rendering.camera)

    def test_calib_skipped_when_rendering_already_populated(self) -> None:
        raw = [_make_calibrated_batch()]
        calls: List[bool] = []

        def fake_calib(data: object, *, skip_calib: bool = False) -> _FakeRenderingField:
            calls.append(skip_calib)
            return _FakeRenderingField(camera=None)

        loader = CalibratedBatchLoader(cast(Iterable[DataAndRenderingBatch], raw), calib_fn=fake_calib)
        list(loader)
        self.assertEqual(calls, [], "calib must not run when batch.rendering is already populated")

    def test_re_iterable(self) -> None:
        raw = [_make_uncalibrated_batch()]

        def fake_calib(data: object, *, skip_calib: bool = False) -> _FakeRenderingField:
            return _FakeRenderingField(camera=_FakeRenderingData(w=1, h=1, rays=torch.zeros((1, 6))))

        loader = CalibratedBatchLoader(cast(Iterable[DataAndRenderingBatch], raw), calib_fn=fake_calib)
        first = list(loader)
        second = list(loader)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)

    def test_factory_alias_returns_wrapper(self) -> None:
        raw: List[_FakeBatch] = []
        loader = iterate_calibrated_batches(cast(Iterable[DataAndRenderingBatch], raw), calib_fn=lambda *_a, **_k: None)
        self.assertIsInstance(loader, CalibratedBatchLoader)

    def test_rejects_non_callable_calib_fn(self) -> None:
        with self.assertRaisesRegex(TypeError, "calib_fn"):
            CalibratedBatchLoader([], calib_fn=42)  # type: ignore[arg-type]

    def test_rejects_invalid_device(self) -> None:
        with self.assertRaisesRegex(TypeError, "device"):
            CalibratedBatchLoader([], calib_fn=lambda *a, **k: None, device="cuda")  # type: ignore[arg-type]


class SamplerWithCalibratedLoaderEndToEndTest(unittest.TestCase):
    """Sampler + calibrated wrapper compose: production import surface validated end-to-end."""

    def test_sampler_iterates_calibrated_batches_from_uncalibrated_source(self) -> None:
        raw = [_make_uncalibrated_batch(), _make_uncalibrated_batch()]
        rendering_camera = _FakeRenderingData(w=2, h=3, rays=torch.zeros((6, 6)))

        def fake_calib(data: object, *, skip_calib: bool = False) -> _FakeRenderingField:
            return _FakeRenderingField(camera=rendering_camera)

        calibrated = iterate_calibrated_batches(cast(Iterable[DataAndRenderingBatch], raw), calib_fn=fake_calib)
        sampler = make_training_view_sampler(calibrated)

        view = sampler(torch.Generator())
        self.assertIsInstance(view.extra, TrainingViewBatch)
        assert isinstance(view.extra, TrainingViewBatch)
        self.assertIs(view.extra.rendering_data, rendering_camera)


class IterGaussianNodesTest(unittest.TestCase):
    """The per-node iteration helper dispatches by ``gaussians_nodes`` presence.

    Mirrors the dispatch used by ``gaussian_usd_asset`` at the
    node-export loop and shared with the SH-match path (warm-start,
    optimizer setup, validators).
    """

    def test_composite_with_nodes_returns_items(self) -> None:
        node_a = _FakeNode(num_gaussians=4, seed=0)
        node_b = _FakeNode(num_gaussians=5, seed=1)

        class _Composite:
            def __init__(self) -> None:
                self.gaussians_nodes = {"static": node_a, "dynamic": node_b}

        result = iter_gaussian_nodes(_Composite())
        self.assertEqual([key for key, _ in result], ["static", "dynamic"])
        self.assertIs(dict(result)["static"], node_a)
        self.assertIs(dict(result)["dynamic"], node_b)

    def test_single_node_without_attr_returns_fallback(self) -> None:
        node = _FakeNode(num_gaussians=3, seed=42)
        result = iter_gaussian_nodes(node)
        self.assertEqual(len(result), 1)
        node_id, returned_node = result[0]
        self.assertEqual(node_id, SINGLE_NODE_FALLBACK_KEY)
        self.assertIs(returned_node, node)

    def test_empty_gaussians_nodes_returns_fallback(self) -> None:
        """Empty (falsy) ``gaussians_nodes`` mapping degrades to the single-node fallback.

        Mirrors the ``hasattr(...) and model.gaussians_nodes`` guard
        previously inlined in ``gaussian_usd_asset.py`` -- an empty
        mapping must not produce zero iterations.
        """

        class _EmptyComposite:
            def __init__(self) -> None:
                self.gaussians_nodes: Dict[str, object] = {}

        composite = _EmptyComposite()
        result = iter_gaussian_nodes(composite)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], SINGLE_NODE_FALLBACK_KEY)
        self.assertIs(result[0][1], composite)

    def test_non_mapping_gaussians_nodes_returns_fallback(self) -> None:
        """An attribute named ``gaussians_nodes`` without ``items()`` degrades to fallback."""

        class _Weird:
            def __init__(self) -> None:
                self.gaussians_nodes = ["not", "a", "mapping"]

        weird = _Weird()
        result = iter_gaussian_nodes(weird)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], SINGLE_NODE_FALLBACK_KEY)
        self.assertIs(result[0][1], weird)

    def test_iteration_order_stable_across_calls(self) -> None:
        node_a = _FakeNode(num_gaussians=4, seed=0)
        node_b = _FakeNode(num_gaussians=5, seed=1)

        class _Composite:
            def __init__(self) -> None:
                self.gaussians_nodes = {"a": node_a, "b": node_b}

        composite = _Composite()
        keys_first = [k for k, _ in iter_gaussian_nodes(composite)]
        keys_second = [k for k, _ in iter_gaussian_nodes(composite)]
        self.assertEqual(keys_first, keys_second)


if __name__ == "__main__":
    unittest.main()
