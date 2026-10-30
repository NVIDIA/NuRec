# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import copy
import pickle
import threading
import unittest
import weakref

from contextlib import nullcontext
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any, Mapping, cast
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import lietorch as lt
import numpy as np
import pytest
import torch

from omegaconf import OmegaConf

import nurec.models.gaussians.collect as collect

from nurec.config.model import ActiveTileMaskConfig, ActiveTileMaskStrategy, GSplatRendererConfig, MCMCStrategyConfig
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.libs.gaussian_collective_payload_ops.interface import gaussian_collective_payload_ops
from nurec.models.calib import FreePoseCalib
from nurec.models.composite import LayerTrackIds
from nurec.models.gaussians import gaussians_composite as gaussians_composite_module
from nurec.models.gaussians import gaussians_model as gaussians_model_module
from nurec.models.gaussians.collect import (
    CollectorResult,
    CreateGaussianParameterCollector,
    DensityActivation,
    HolisticRemapTimeInputEmbeddingConfig,
    LayerConfigRigid,
    LayerConfigSH,
    LayerDataRigid,
    LayerDataSH,
    LayersConfig,
    LayersData,
    RotationActivation,
    ScaleActivation,
)
from nurec.models.gaussians.fused_parameter_gather import (
    _COMMUNICATION_SCHEMA_HASH_WORD_COUNT,
    _FusedGatherResult,
    _LayerPayloadLayout,
    _PayloadIntegerRangeValidationState,
)
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel, SHGaussianModel
from nurec.models.gaussians.renderers import BaseGaussianRenderer
from nurec.models.gaussians.strategies.mcmc import MCMCStrategy, _LazyParameterDict
from nurec.models.gaussians.utils import RGBPostProcessing
from nurec.models.gaussians.vehicle_light import VehicleLightRigidGaussianModel
from nurec.models.nn_extensions import TypedModuleDict, module_call_type
from nurec.models.tracks_calib import BaseTracksCalib
from nurec.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    RenderingBatch,
    RenderingData,
)
from nurec.utils.morton import VOXEL_SIZE, morton_sort_order
from nurec.utils.tile_mask import CameraToLidarTileMapping
from nurec.utils.types import GaussiansCompositeReturn, GaussiansRenderReturn


TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GENERATION_PLAN_TEST_NODES = ("background", "road", "dynamic_rigids", "dynamic_deformables")


class _FakeGatherNode:
    def __init__(self, count: int) -> None:
        self.config = SimpleNamespace(motion_source_layer=None)
        self.parameters = {"positions": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)}
        self.get_parameters_calls = 0

    def get_parameters(self) -> dict[str, torch.Tensor]:
        self.get_parameters_calls += 1
        return self.parameters

    def get_layer_data(self, _context: object, parameters: dict[str, torch.Tensor]) -> SimpleNamespace:
        return SimpleNamespace(positions=parameters["positions"])

    def post_process(self, _offset: int, _collector_result: object, _layer_data: object, rows=None) -> None:
        pass


class _FilteringFakeGatherNode(_FakeGatherNode):
    def get_motion_source(self) -> "_FilteringFakeGatherNode":
        return self

    def create_tracks_edit_from_cuboid_tracks(self, _edited_tracks: CuboidTracks) -> LayerTrackIds.Edit:
        return LayerTrackIds.Edit(cuboid_tracks=_edited_tracks, mapping=torch.tensor([0], dtype=torch.int64))

    def prepare_motion(self, *_args: Any, **_kwargs: Any) -> object:
        return object()

    def get_layer_data(self, context: object, parameters: dict[str, torch.Tensor]) -> SimpleNamespace:
        if getattr(context, "tracks_edit", None) is not None:
            parameters["positions"] = parameters["positions"][:1]
        return SimpleNamespace(positions=parameters["positions"])


def _make_gather_composite() -> tuple[GaussiansComposite, _FakeGatherNode, _FakeGatherNode]:
    first = _FakeGatherNode(2)
    second = _FakeGatherNode(1)
    composite = GaussiansComposite.__new__(GaussiansComposite)
    torch.nn.Module.__init__(composite)
    composite.config = cast(
        Any,
        SimpleNamespace(
            renderer=SimpleNamespace(active_tile_mask=ActiveTileMaskConfig(enabled=False)),
            layers={"first": SimpleNamespace(), "second": SimpleNamespace()},
        ),
    )
    composite.gaussian_process_group = cast(Any, object())
    composite.gaussians_nodes = {"first": first, "second": second}  # type: ignore[assignment]
    composite._fused_gather_generation_plan = None
    composite._pending_fused_gather_generation_plan_refresh = None
    return composite, first, second


class TestLidarLayerVisibility(unittest.TestCase):
    @staticmethod
    def _composite(*, vehicle_lights_visible: bool) -> GaussiansComposite:
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite.config = cast(
            Any,
            SimpleNamespace(
                layers={
                    "background": SimpleNamespace(render_in_lidar=True),
                    "vehicle_lights": SimpleNamespace(render_in_lidar=vehicle_lights_visible),
                    "dynamic_rigids": SimpleNamespace(render_in_lidar=True),
                }
            ),
        )
        return composite

    def test_lidar_zeroes_only_camera_only_layer_and_its_gradient(self) -> None:
        composite = self._composite(vehicle_lights_visible=False)
        densities = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]], requires_grad=True)
        parameters = {"densities": densities}

        composite._apply_lidar_layer_visibility(
            parameters,
            ["background", "vehicle_lights", "dynamic_rigids"],
            [0, 2, 2, 4, 4, 5],
            "lidar",
        )

        torch.testing.assert_close(
            parameters["densities"],
            torch.tensor([[0.1], [0.2], [0.0], [0.0], [0.5]]),
        )
        parameters["densities"].sum().backward()
        torch.testing.assert_close(densities.grad, torch.tensor([[1.0], [1.0], [0.0], [0.0], [1.0]]))

    def test_camera_keeps_camera_only_layer_unchanged(self) -> None:
        composite = self._composite(vehicle_lights_visible=False)
        densities = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
        parameters = {"densities": densities}

        composite._apply_lidar_layer_visibility(
            parameters,
            ["background", "vehicle_lights", "dynamic_rigids"],
            [0, 2, 2, 4, 4, 5],
            "cam",
        )

        self.assertIs(parameters["densities"], densities)

    def test_lidar_keeps_all_default_visible_layers_unchanged(self) -> None:
        composite = self._composite(vehicle_lights_visible=True)
        densities = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])
        parameters = {"densities": densities}

        composite._apply_lidar_layer_visibility(
            parameters,
            ["background", "vehicle_lights", "dynamic_rigids"],
            [0, 2, 2, 4, 4, 5],
            "lidar",
        )

        self.assertIs(parameters["densities"], densities)

    def test_lidar_defaults_to_visible_without_full_model_config(self) -> None:
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        densities = torch.tensor([[0.1], [0.2]])
        parameters = {"densities": densities}

        composite._apply_lidar_layer_visibility(parameters, ["background"], [0, 2], "lidar")

        self.assertIs(parameters["densities"], densities)

    def test_lidar_visibility_uses_offsets_from_actual_collection(self) -> None:
        composite = self._composite(vehicle_lights_visible=False)
        composite.debug_viz = False
        composite._visibility_local_ranges = None
        nodes = {
            name: _FakeGatherNode(count)
            for name, count in (("background", 2), ("vehicle_lights", 2), ("dynamic_rigids", 1))
        }
        composite.gaussians_nodes = cast(Any, nodes)
        gaussian_nodes = list(nodes)
        densities = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]], requires_grad=True)
        collector_result = collect.CollectorResult(
            positions=torch.zeros((5, 3)),
            rotations=torch.zeros((5, 4)),
            scales=torch.ones((5, 3)),
            densities=densities,
            extra_signal=torch.empty((5, 0)),
            camera_extra_signal=torch.empty((5, 0)),
            lidar_extra_signal=torch.empty((5, 0)),
            features=torch.zeros((5, 3)),
        )
        composite.collector = MagicMock()
        composite.collector.collect.return_value = collector_result
        rendering_data = cast(
            RenderingData,
            SimpleNamespace(timestamps_startend_us_cpu=torch.zeros((1, 2), dtype=torch.int64)),
        )

        # Exercise the production offset producer, not a hand-authored offset list.
        # Only layer data and the CUDA collector are stand-ins in this CPU test.
        with (
            patch.object(composite, "get_gaussians_node_ids", return_value=gaussian_nodes),
            patch.object(
                composite, "_apply_lidar_layer_visibility", wraps=composite._apply_lidar_layer_visibility
            ) as apply_visibility,
        ):
            lidar = composite.collect_gaussian_parameters(rendering_data, sensor_name="lidar")
            self.assertEqual(apply_visibility.call_args.args[2], [0, 2, 2, 4, 4, 5])
            camera = composite.collect_gaussian_parameters(rendering_data, sensor_name="cam")

        torch.testing.assert_close(lidar["densities"], torch.tensor([[0.1], [0.2], [0.0], [0.0], [0.5]]))
        self.assertIs(camera["densities"], densities)
        self.assertIs(collector_result.densities, densities)
        lidar["densities"].sum().backward()
        torch.testing.assert_close(densities.grad, torch.tensor([[1.0], [1.0], [0.0], [0.0], [1.0]]))


class _FakeSortNode:
    """Node stub for the composite-level Morton orchestration tests.

    Not a RigidGaussianModel, so the default dynamic-layer skip never fires on it.
    """

    def __init__(
        self,
        order: torch.Tensor | None = None,
        error: BaseException | None = None,
        validate_error: Exception | None = None,
        apply_error: Exception | None = None,
    ) -> None:
        self._order = order
        self._error = error
        self._validate_error = validate_error
        self._apply_error = apply_error
        self.validated: list[torch.Tensor] = []
        self.applied: list[torch.Tensor] = []

    def compute_morton_order(self, voxel_size: float = VOXEL_SIZE) -> torch.Tensor | None:
        if self._error is not None:
            raise self._error
        return self._order

    def validate_optimizer_state(self, order: torch.Tensor) -> None:
        if self._validate_error is not None:
            raise self._validate_error
        self.validated.append(order)

    def apply_morton_order(self, order: torch.Tensor) -> None:
        if self._apply_error is not None:
            raise self._apply_error
        self.applied.append(order)


def _make_sort_composite(**nodes: _FakeSortNode) -> tuple[GaussiansComposite, dict[str, torch.Tensor]]:
    composite = GaussiansComposite.__new__(GaussiansComposite)
    torch.nn.Module.__init__(composite)
    composite.gaussians_nodes = nodes  # type: ignore[assignment]
    composite.component_only = None
    handed_to_strategy: dict[str, torch.Tensor] = {}
    composite.gaussians_strategy = cast(
        Any,
        SimpleNamespace(permute_gaussian_rows=handed_to_strategy.update, refresh_runtime_scenes=lambda _nodes: None),
    )
    return composite, handed_to_strategy


class TestGaussiansCompositeMortonSort(unittest.TestCase):
    def test_empty_local_sort_still_retires_the_generation_plan(self) -> None:
        composite, handed_to_strategy = _make_sort_composite(background=_FakeSortNode(order=None))
        stale_plan = object()
        count_token = object()
        composite._fused_gather_generation_plan = cast(Any, stale_plan)
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            SimpleNamespace(count_gather=count_token),
        )

        with patch.object(gaussians_composite_module, "_discard_layer_count_gather") as discard:
            composite.morton_sort_gaussians()

        self.assertIsNone(composite._fused_gather_generation_plan)
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
        self.assertEqual(handed_to_strategy, {})
        discard.assert_called_once_with(count_token)

    def test_unsortable_layer_is_skipped_without_losing_the_others(self) -> None:
        good = _FakeSortNode(order=torch.tensor([1, 0]))
        bad = _FakeSortNode(error=ValueError("positions contains non-finite values"))
        composite, handed_to_strategy = _make_sort_composite(background=good, road=bad)

        composite.morton_sort_gaussians()

        # A layer that cannot be sorted costs its own ordering and nothing else.
        self.assertEqual(len(good.applied), 1)
        self.assertEqual(bad.applied, [])
        self.assertEqual(list(handed_to_strategy), ["background"])

    def test_a_layer_that_fails_to_compute_its_order_never_aborts_the_save(self) -> None:
        good = _FakeSortNode(order=torch.tensor([1, 0]))
        # OOM is the realistic escape from the compute phase and it is a RuntimeError, not a ValueError.
        starved = _FakeSortNode(error=torch.cuda.OutOfMemoryError("CUDA out of memory"))
        composite, handed_to_strategy = _make_sort_composite(background=good, road=starved)

        composite.morton_sort_gaussians()

        # Aborting here would strand every DDP peer on the save-path all-gather until the NCCL timeout.
        self.assertEqual(len(good.applied), 1)
        self.assertEqual(starved.applied, [])
        self.assertEqual(list(handed_to_strategy), ["background"])

    def test_an_aborted_sort_leaves_every_layer_unpermuted(self) -> None:
        first = _FakeSortNode(order=torch.tensor([1, 0]))
        # The compute phase swallows exceptions, so this needs something outside that hierarchy to abort at all.
        second = _FakeSortNode(error=KeyboardInterrupt())
        composite, handed_to_strategy = _make_sort_composite(background=first, road=second)

        with self.assertRaises(KeyboardInterrupt):
            composite.morton_sort_gaussians()

        # Every order is computed before any row moves, so an abort cannot leave the model half-permuted.
        self.assertEqual(first.applied, [])
        self.assertEqual(handed_to_strategy, {})

    def test_a_failure_while_applying_an_order_propagates(self) -> None:
        oom = torch.cuda.OutOfMemoryError("CUDA out of memory")
        broken = _FakeSortNode(order=torch.tensor([1, 0]), apply_error=oom)
        composite, handed_to_strategy = _make_sort_composite(background=broken)

        with self.assertRaises(torch.cuda.OutOfMemoryError):
            composite.morton_sort_gaussians()

        # Once rows start moving there is no consistent state to fall back to, so the run stops here rather
        # than letting the writer capture a model whose layers disagree about their row order.
        self.assertEqual(handed_to_strategy, {})

    def test_unpermutable_optimizer_state_skips_only_that_layer(self) -> None:
        good = _FakeSortNode(order=torch.tensor([1, 0]))
        bad = _FakeSortNode(order=torch.tensor([1, 0]), validate_error=ValueError("leading dim"))
        composite, handed_to_strategy = _make_sort_composite(background=good, road=bad)

        composite.morton_sort_gaussians()

        # Validation runs in the compute phase, so a layer it rejects is skipped rather than
        # aborting the save, and the layers that passed are still sorted.
        self.assertEqual(len(good.applied), 1)
        self.assertEqual(bad.applied, [])
        self.assertEqual(list(handed_to_strategy), ["background"])


def _make_sort_node(n: int, rigid: bool, seed: int):
    """A minimal real Gaussian node carrying only the per-Gaussian rows, heavy config __init__ bypassed.

    rigid=False is the shape the static layers have. rigid=True builds a RigidGaussianModel, which is what the
    dynamic-layer skip keys off. It also adds the per-Gaussian track ids the static layers do not carry.
    """
    cls = RigidGaussianModel if rigid else SHGaussianModel
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    generator = torch.Generator().manual_seed(seed)
    model.positions = torch.nn.Parameter(torch.rand(n, 3, generator=generator) * 40.0)

    def _rows(width: int) -> torch.Tensor:
        return torch.arange(n * width, dtype=torch.float32).reshape(n, width)

    model.scales = torch.nn.Parameter(_rows(3))
    model.rotations = torch.nn.Parameter(_rows(4))
    model.densities = torch.nn.Parameter(_rows(1))
    model.extra_signal = torch.nn.Parameter(_rows(2))
    model.camera_extra_signal = torch.nn.Parameter(_rows(3))
    model.lidar_extra_signal = torch.nn.Parameter(_rows(5))
    model.features_albedo = torch.nn.Parameter(_rows(3))
    model.features_specular = torch.nn.Parameter(_rows(45))
    if rigid:
        # Four tracks interleaved row by row.
        # One row per track would make a track-partitioned order come out as the identity.
        model.gaussian_cuboid_ids = torch.nn.Buffer(torch.arange(n, dtype=torch.long) % 4)
    return model


def _invisible_steps(composite: GaussiansComposite, layer_id: str) -> torch.Tensor:
    """Read a strategy buffer. nn.Module.__getattr__ is typed Tensor | Module, hence the cast."""
    return cast(torch.nn.ParameterDict, composite.gaussians_strategy.invisible_steps)[layer_id]


def _make_real_sort_composite(n: int = 32) -> GaussiansComposite:
    """A static layer, a dynamic layer and a real strategy: the shape a real checkpoint has.

    Real classes throughout, so the reorder runs through the production skip, the production permute and
    MCMCStrategy.permute_gaussian_rows rather than stand-ins for any of them.
    """
    strategy = MCMCStrategy.__new__(MCMCStrategy)
    torch.nn.Module.__init__(strategy)
    strategy.invisible_steps = _LazyParameterDict(
        {
            layer: torch.nn.Parameter(torch.arange(n, dtype=torch.float32), requires_grad=False)
            for layer in ("background", "dynamic_rigids")
        }
    )
    composite = GaussiansComposite.__new__(GaussiansComposite)
    torch.nn.Module.__init__(composite)
    composite.gaussian_process_group = cast(Any, object())
    composite.component_only = None
    composite.gaussians_nodes = TypedModuleDict(
        {
            "background": _make_sort_node(n, rigid=False, seed=0),
            "dynamic_rigids": _make_sort_node(n, rigid=True, seed=1),
        }
    )
    composite.gaussians_strategy = strategy
    return composite


class TestGaussiansCompositeMortonSortIntegration(unittest.TestCase):
    """End to end through real nodes and a real strategy, not stubs.

    densities are seeded with the row index, so after a sort they spell out the permutation that was applied.
    Equality with arange means the layer was left alone.
    """

    _PER_GAUSSIAN = (
        "positions",
        "scales",
        "rotations",
        "densities",
        "extra_signal",
        "camera_extra_signal",
        "lidar_extra_signal",
        "features_albedo",
        "features_specular",
    )

    def test_default_sorts_static_layers_and_skips_the_dynamic_one(self) -> None:
        composite = _make_real_sort_composite()
        static = cast(SHGaussianModel, composite.gaussians_nodes["background"])
        dynamic = cast(RigidGaussianModel, composite.gaussians_nodes["dynamic_rigids"])
        originals = {name: getattr(static, name).detach().clone() for name in self._PER_GAUSSIAN}
        identity = torch.arange(32, dtype=torch.float32)

        composite.morton_sort_gaussians()

        order = static.densities.detach().flatten()
        self.assertFalse(torch.equal(order, identity))  # a no-op sort would prove nothing below
        for name in self._PER_GAUSSIAN:
            torch.testing.assert_close(getattr(static, name).detach(), originals[name][order.long()], msg=name)
        # Strategy buffers live outside the nodes, so only the composite can reach them.
        torch.testing.assert_close(_invisible_steps(composite, "background").detach(), order)
        # RigidGaussianModel nodes are skipped unless asked for.
        torch.testing.assert_close(dynamic.densities.detach().flatten(), identity)
        torch.testing.assert_close(_invisible_steps(composite, "dynamic_rigids").detach(), identity)

    def test_mcmc_scenes_rebind_to_the_parameters_the_sort_left_behind(self) -> None:
        # MCMC scenes adopt the row Parameters themselves rather than the node, so the reorder strands them on
        # the objects it replaced. A refresh that ran before the swap would rebind to those same doomed tensors,
        # which is why this pins the ordering and not just the call.
        composite = _make_real_sort_composite()
        static = cast(SHGaussianModel, composite.gaussians_nodes["background"])
        before_sort = static.positions
        rebound: list[torch.nn.Parameter] = []

        def _record(gaussians_nodes: Any) -> None:
            rebound.append(gaussians_nodes["background"].positions)

        composite.gaussians_strategy.refresh_runtime_scenes = _record  # type: ignore[method-assign]

        composite.morton_sort_gaussians()

        self.assertEqual(len(rebound), 1)
        self.assertIsNot(rebound[0], before_sort)
        self.assertIs(rebound[0], static.positions)

    def test_sort_track_layers_reaches_the_rigid_node(self) -> None:
        composite = _make_real_sort_composite()
        dynamic = cast(RigidGaussianModel, composite.gaussians_nodes["dynamic_rigids"])
        original_ids = dynamic.gaussian_cuboid_ids.detach().clone()
        original_positions = dynamic.positions.detach().clone()
        originals = {name: getattr(dynamic, name).detach().clone() for name in self._PER_GAUSSIAN}

        composite.morton_sort_gaussians(sort_track_layers=True)

        order = dynamic.densities.detach().flatten()
        self.assertFalse(torch.equal(order, torch.arange(32, dtype=torch.float32)))
        # A rigid node gets the track-partitioned order, not the plain Morton one the static layers get.
        plain_morton = morton_sort_order(original_positions).order.float()
        self.assertFalse(torch.equal(order, plain_morton))
        for name in self._PER_GAUSSIAN:
            torch.testing.assert_close(getattr(dynamic, name).detach(), originals[name][order.long()], msg=name)
        # Track membership has to move with the rows or the frozen-track mask goes stale.
        torch.testing.assert_close(dynamic.gaussian_cuboid_ids.detach(), original_ids[order.long()])
        torch.testing.assert_close(_invisible_steps(composite, "dynamic_rigids").detach(), order)

    def test_checkpoint_sort_retires_the_generation_plan_before_the_next_prefetch(self) -> None:
        composite = _make_real_sort_composite()
        composite.calib = _make_free_pose_calib_for_prefetch_test()
        composite.train()
        selected_nodes = ("background", "dynamic_rigids")
        stale_plan = _make_generation_plan(selected_nodes)
        composite._fused_gather_generation_plan = cast(Any, stale_plan)
        count_token = object()
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                prepared_inputs=cast(Any, object()),
                count_gather=cast(Any, count_token),
            ),
        )
        prepared = object()

        static = cast(SHGaussianModel, composite.gaussians_nodes["background"])
        dynamic = cast(RigidGaussianModel, composite.gaussians_nodes["dynamic_rigids"])
        static_apply = static.apply_morton_order
        dynamic_apply = dynamic.apply_morton_order
        events: list[str] = []

        def discard_refresh(value: object) -> None:
            self.assertIs(value, count_token)
            self.assertIsNone(composite._fused_gather_generation_plan)
            self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
            events.append("drain")

        def apply_after_invalidation(layer_id: str, apply: Any, order: torch.Tensor) -> None:
            self.assertIsNone(composite._fused_gather_generation_plan)
            self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
            self.assertEqual(events[0], "drain")
            events.append(layer_id)
            apply(order)

        with (
            patch.object(
                gaussians_composite_module,
                "_discard_layer_count_gather",
                side_effect=discard_refresh,
            ) as discard,
            patch.object(
                static,
                "apply_morton_order",
                side_effect=lambda order: apply_after_invalidation("background", static_apply, order),
            ),
            patch.object(
                dynamic,
                "apply_morton_order",
                side_effect=lambda order: apply_after_invalidation("dynamic_rigids", dynamic_apply, order),
            ),
        ):
            composite.morton_sort_gaussians(sort_track_layers=True)

        self.assertEqual(events, ["drain", *selected_nodes])
        discard.assert_called_once_with(count_token)

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(gaussians_composite_module, "_prepare_fused_gather_inputs", return_value=prepared) as warmup,
            patch.object(gaussians_composite_module, "_can_prefetch_grouped_payload", return_value=True),
            patch.object(gaussians_composite_module, "_prepare_fused_gather_inputs_from_generation_plan") as cached,
            patch.object(gaussians_composite_module, "_start_fused_payload_gather_from_generation_plan") as start,
        ):
            pending = composite._maybe_prefetch_gaussian_parameter_gather(
                is_training_batch=True,
                gaussian_nodes=None,
                track_transform_ids=None,
                track_transforms=None,
                edited_cuboid_tracks=None,
            )

        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertIsNone(pending.generation_plan)
        self.assertTrue(pending.publish_on_success)
        warmup.assert_called_once()
        cached.assert_not_called()
        start.assert_not_called()

    def test_dynamic_rows_are_track_contiguous_and_morton_ordered_inside_each_track(self) -> None:
        composite = _make_real_sort_composite()
        dynamic = cast(RigidGaussianModel, composite.gaussians_nodes["dynamic_rigids"])
        original_ids = dynamic.gaussian_cuboid_ids.detach().clone()
        original_positions = dynamic.positions.detach().clone()

        composite.morton_sort_gaussians(sort_track_layers=True)

        order = dynamic.densities.detach().flatten().long()
        ids = dynamic.gaussian_cuboid_ids.detach()
        tracks = ids.unique()
        self.assertGreater(len(tracks), 1)  # a single track makes contiguity vacuous
        # Every track's rows form one contiguous block, so a warp holds one vehicle.
        torch.testing.assert_close(ids, original_ids.sort(stable=True).values)

        # Inside a block the rows still climb the layer's Morton curve.
        # Checked as ranks: Morton codes are not translation invariant, so a per-track key is a different one.
        morton_rank = torch.empty(32, dtype=torch.long)
        morton_rank[morton_sort_order(original_positions).order] = torch.arange(32)
        interleaved = False
        for track in tracks.tolist():
            ranks = morton_rank[order[ids == track]]
            self.assertTrue(bool((ranks[1:] > ranks[:-1]).all()), f"track {track} is not in Morton order")
            interleaved |= bool((ranks[1:] - ranks[:-1] > 1).any())
        # Consecutive ranks throughout would mean a plain Morton sort produces this same order.
        self.assertTrue(interleaved)

    def test_a_non_default_voxel_size_reaches_both_orders(self) -> None:
        # Pin that the voxel size handed in is the one the Morton key is built with.
        composite = _make_real_sort_composite()
        static = cast(SHGaussianModel, composite.gaussians_nodes["background"])
        dynamic = cast(RigidGaussianModel, composite.gaussians_nodes["dynamic_rigids"])
        static_positions = static.positions.detach().clone()

        composite.morton_sort_gaussians(sort_track_layers=True, voxel_size=32.0)
        default = _make_real_sort_composite()
        default.morton_sort_gaussians(sort_track_layers=True)

        # The static layer pins the exact value: its order has to be the one 32 m produces.
        order = static.densities.detach().flatten().long()
        torch.testing.assert_close(order, morton_sort_order(static_positions, voxel_size=32.0).order)
        default_static = cast(SHGaussianModel, default.gaussians_nodes["background"])
        # Two sizes that agreed would make the assertion above hold whatever was passed.
        self.assertFalse(torch.equal(order, default_static.densities.detach().flatten().long()))
        # The rigid override takes the size too, and it reaches the Morton key inside a track.
        default_dynamic = cast(RigidGaussianModel, default.gaussians_nodes["dynamic_rigids"])
        self.assertFalse(
            torch.equal(dynamic.densities.detach().flatten(), default_dynamic.densities.detach().flatten())
        )


def _make_grouped_gather_composite(
    node_ids: tuple[str, ...] = _GENERATION_PLAN_TEST_NODES,
) -> tuple[GaussiansComposite, dict[str, _FakeGatherNode]]:
    nodes = {node_id: _FakeGatherNode(index + 1) for index, node_id in enumerate(node_ids)}
    composite = GaussiansComposite.__new__(GaussiansComposite)
    torch.nn.Module.__init__(composite)
    composite.gaussian_process_group = cast(Any, object())
    composite.config = cast(
        Any,
        SimpleNamespace(
            renderer=GSplatRendererConfig(name="3dgut-gsplat", sparse_segments=False),
            layers={node_id: SimpleNamespace() for node_id in node_ids},
        ),
    )
    composite.gaussians_nodes = nodes  # type: ignore[assignment]
    composite._fused_gather_generation_plan = None
    composite._pending_fused_gather_generation_plan_refresh = None
    return composite, nodes


def _make_free_pose_calib_for_prefetch_test() -> FreePoseCalib:
    calib = FreePoseCalib.__new__(FreePoseCalib)
    torch.nn.Module.__init__(calib)
    return calib


def _make_generation_plan_test_batch() -> SimpleNamespace:
    return SimpleNamespace(
        rendering=None,
        data=SimpleNamespace(
            camera=SimpleNamespace(meta=[object()]),
            lidar=SimpleNamespace(meta=[object()]),
        ),
    )


def _make_generation_plan(node_ids: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        layouts=tuple(SimpleNamespace(node_id=node_id) for node_id in node_ids),
        world_rank=0,
        world_size=2,
    )


def _make_fused_result(nodes: Mapping[str, _FakeGatherNode]) -> _FusedGatherResult:
    return _FusedGatherResult(
        parameters_by_node={node_id: dict(node.parameters) for node_id, node in nodes.items()},
        counts_by_node={node_id: (node.parameters["positions"].shape[0],) * 2 for node_id, node in nodes.items()},
    )


class TestGaussiansCompositeGather(unittest.TestCase):
    def test_batch_end_refreshes_generation_plan_after_mutations(self) -> None:
        """Drain pending generation work before entering a scheduled mutation."""
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite._distillation_active = False
        composite.gaussians_nodes = {}  # type: ignore[assignment]
        composite.trainer_config = cast(Any, SimpleNamespace(world_size=2))
        composite.config = cast(Any, SimpleNamespace(log_to_prog_bar=False))
        events: list[str] = []
        strategy = Mock()
        strategy.may_mutate_topology_at.return_value = True
        strategy.update_step_train_batch_end.side_effect = lambda *_args, **_kwargs: events.append("mutation")
        composite.gaussians_strategy = cast(Any, strategy)
        count_token = object()
        composite._fused_gather_generation_plan = cast(Any, object())
        composite._pending_fused_gather_generation_plan_refresh = cast(Any, SimpleNamespace(count_gather=count_token))
        system = Mock()

        with (
            patch.object(composite, "get_base_model_children", return_value=[]),
            patch.object(
                gaussians_composite_module,
                "_discard_layer_count_gather",
                side_effect=lambda _token: events.append("drain"),
            ) as discard,
            patch.object(
                composite,
                "_refresh_fused_gather_generation_plan_after_mutations",
                side_effect=lambda _step: events.append("refresh"),
            ) as refresh,
        ):
            composite.update_step_train_batch_end(0, 17, cast(Any, object()), system)

        self.assertEqual(events, ["drain", "mutation", "refresh"])
        discard.assert_called_once_with(count_token)
        strategy.may_mutate_topology_at.assert_called_once_with(17)
        refresh.assert_called_once_with(17)
        system.log.assert_called_once_with("gaussians/num_gaussians", 0, prog_bar=False)

    def test_scheduled_cuboid_routes_retire_generation_before_local_noop_or_mutation(self) -> None:
        """Relocation/growth retire both caches even for empty or no-donor pools."""
        for step in (10, 15):
            for count, changed in ((20, 1), (20, 0), (0, 0)):
                with self.subTest(step=step, count=count, changed=changed):
                    composite, _nodes = _make_grouped_gather_composite(("barriers",))
                    node = Mock()
                    node.get_num_gaussians.return_value = count
                    node.get_parameters.return_value = {"positions": torch.zeros(count, 3)}
                    composite.gaussians_nodes = cast(Any, {"barriers": node})
                    composite.calib = _make_free_pose_calib_for_prefetch_test()
                    composite._distillation_active = False
                    composite.trainer_config = cast(Any, SimpleNamespace(world_size=1))
                    composite.config.log_to_prog_bar = False
                    strategy = MCMCStrategy.__new__(MCMCStrategy)
                    torch.nn.Module.__init__(strategy)
                    strategy.config = MCMCStrategyConfig.model_validate(
                        {
                            "name": "mcmc",
                            "relocate": {"start_iteration": 1, "end_iteration": 30, "frequency": 10},
                            "add": {"start_iteration": 1, "end_iteration": 30, "frequency": 15},
                            "perturb": {"start_iteration": 30},
                            "independent_layer_max_n_gaussians": {"barriers": 100},
                            "cuboid_grouped_layer_ids": ["barriers"],
                            "print_stats": False,
                        }
                    )
                    strategy.frozen = False
                    strategy._empty_cache_after_parameter_update = False
                    composite.gaussians_strategy = strategy
                    count_token = object()
                    composite._fused_gather_generation_plan = cast(Any, _make_generation_plan(("barriers",)))
                    composite._pending_fused_gather_generation_plan_refresh = cast(
                        Any, SimpleNamespace(count_gather=count_token)
                    )

                    def mutate_pool(selected: Any, *, relocation: bool) -> int:
                        """Assert retirement at the actual cuboid dispatch boundary."""
                        self.assertEqual(selected.layer_ids, ["barriers"])
                        self.assertEqual(relocation, step == 10)
                        self.assertIsNone(composite._fused_gather_generation_plan)
                        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
                        discard.assert_called_once_with(count_token)
                        return changed

                    with (
                        patch.object(composite, "get_base_model_children", return_value=[]),
                        patch.object(gaussians_composite_module, "_discard_layer_count_gather") as discard,
                        patch.object(strategy, "_cuboid_topology", side_effect=mutate_pool) as mutate,
                    ):
                        composite.update_step_train_batch_end(0, step, cast(Any, object()), Mock())
                    self.assertEqual(mutate.call_count, int(count > 0))
                    discard.assert_called_once_with(count_token)
                    self.assertIsNone(composite._fused_gather_generation_plan)
                    self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)

                    # Cuboid MCMC remains single-GPU only. Injected stale state above
                    # tests retirement defensively; simulate an eligible gather here
                    # only to verify that neither stale generation can be reused.
                    with (
                        patch.object(torch.distributed, "is_initialized", return_value=True),
                        patch.object(torch.distributed, "get_world_size", return_value=2),
                        patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
                        patch.object(composite, "get_gaussians_node_ids", return_value=["barriers"]),
                        patch.object(gaussians_composite_module, "_prepare_fused_gather_inputs") as prepare,
                        patch.object(gaussians_composite_module, "_can_prefetch_grouped_payload", return_value=True),
                        patch.object(
                            gaussians_composite_module, "_prepare_fused_gather_inputs_from_generation_plan"
                        ) as cached,
                        patch.object(
                            gaussians_composite_module, "_start_fused_payload_gather_from_generation_plan"
                        ) as start,
                    ):
                        pending = composite._maybe_prefetch_gaussian_parameter_gather(
                            is_training_batch=True,
                            gaussian_nodes=None,
                            track_transform_ids=None,
                            track_transforms=None,
                            edited_cuboid_tracks=None,
                        )
                    assert pending is not None
                    self.assertIsNone(pending.generation_plan)
                    self.assertTrue(pending.publish_on_success)
                    prepare.assert_called_once()
                    cached.assert_not_called()
                    start.assert_not_called()

    def test_warmup_waits_until_joint_gather_and_publishes_only_after_success(self) -> None:
        composite, nodes = _make_grouped_gather_composite()
        composite.calib = _make_free_pose_calib_for_prefetch_test()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        prepared = object()
        plan = _make_generation_plan(selected_nodes)
        payload = object()
        events: list[str] = []

        def resolve_generation_plan(
            _value: object,
            *,
            process_group: torch.distributed.ProcessGroup,
        ) -> SimpleNamespace:
            self.assertIs(process_group, composite.gaussian_process_group)
            events.append("metadata")
            return plan

        def start_generation_payload(
            _value: object,
            _value_plan: object,
            *,
            process_group: torch.distributed.ProcessGroup,
        ) -> object:
            self.assertIs(process_group, composite.gaussian_process_group)
            events.append("payload")
            return payload

        def consume_warmup(*_args: Any, **_kwargs: Any) -> _FusedGatherResult:
            self.assertIsNone(getattr(composite, "_fused_gather_generation_plan", None))
            events.append("consume")
            return _make_fused_result(nodes)

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
            patch.object(gaussians_composite_module, "_prepare_fused_gather_inputs", return_value=prepared),
            patch.object(gaussians_composite_module, "_can_prefetch_grouped_payload", return_value=True),
            patch.object(gaussians_composite_module, "_start_layer_count_gather") as start_counts,
            patch.object(
                gaussians_composite_module,
                "_resolve_fused_gather_generation_plan",
                side_effect=resolve_generation_plan,
            ) as resolve,
            patch.object(
                gaussians_composite_module,
                "_start_fused_payload_gather_from_generation_plan",
                side_effect=start_generation_payload,
            ) as start_payload,
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                side_effect=consume_warmup,
            ),
        ):
            pending = composite._maybe_prefetch_gaussian_parameter_gather(
                is_training_batch=True,
                gaussian_nodes=None,
                track_transform_ids=None,
                track_transforms=None,
                edited_cuboid_tracks=None,
            )

            self.assertIsNotNone(pending)
            assert pending is not None
            self.assertIsNone(pending.generation_plan)
            self.assertIsNone(pending.prefetched_payload_gather)
            self.assertTrue(pending.publish_on_success)
            self.assertEqual(events, [])
            start_counts.assert_not_called()
            start_payload.assert_not_called()

            composite._gather_gaussian_parameters_once(list(selected_nodes), prefetched=pending)

        self.assertEqual(events, ["metadata", "payload", "consume"])
        resolve.assert_called_once_with(
            prepared,
            process_group=composite.gaussian_process_group,
        )
        self.assertIs(composite._fused_gather_generation_plan, plan)

    def test_generation_warmup_accepts_generic_compatible_node_sets(self) -> None:
        node_sets = (
            ("sky_shell", "asphalt", "vehicles", "cloth"),
            tuple(reversed(_GENERATION_PLAN_TEST_NODES)),
            ("near_field", "far_field", "actors"),
            ("static_scene", "moving_scene"),
            ("only_layer",),
            (
                "background",
                "road",
                "dynamic_rigids",
                "dynamic_deformables",
                "static_rigids",
                "traffic_light",
                "vehicle_lights",
                "static_vehicle_lights",
            ),
        )

        for selected_nodes in node_sets:
            with self.subTest(selected_nodes=selected_nodes):
                composite, nodes = _make_grouped_gather_composite(selected_nodes)
                composite.calib = _make_free_pose_calib_for_prefetch_test()
                prepared = object()
                with (
                    patch.object(torch.distributed, "is_initialized", return_value=True),
                    patch.object(torch.distributed, "get_world_size", return_value=2),
                    patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
                    patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
                    patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
                    patch.object(
                        gaussians_composite_module,
                        "_prepare_fused_gather_inputs",
                        return_value=prepared,
                    ) as prepare,
                    patch.object(
                        gaussians_composite_module,
                        "_can_prefetch_grouped_payload",
                        return_value=True,
                    ) as compatible,
                    patch.object(gaussians_composite_module, "_start_layer_count_gather") as start_counts,
                    patch.object(
                        gaussians_composite_module,
                        "_start_fused_payload_gather_from_generation_plan",
                    ) as start_payload,
                ):
                    pending = composite._maybe_prefetch_gaussian_parameter_gather(
                        is_training_batch=True,
                        gaussian_nodes=None,
                        track_transform_ids=None,
                        track_transforms=None,
                        edited_cuboid_tracks=None,
                    )

                self.assertIsNotNone(pending)
                assert pending is not None
                self.assertEqual(pending.gaussian_nodes, selected_nodes)
                self.assertEqual(tuple(pending.parameters_by_node), selected_nodes)
                self.assertIs(pending.prepared_inputs, prepared)
                self.assertIsNone(pending.generation_plan)
                self.assertIsNone(pending.prefetched_payload_gather)
                self.assertTrue(pending.publish_on_success)
                prepare.assert_called_once_with(
                    list(selected_nodes),
                    pending.parameters_by_node,
                    process_group=composite.gaussian_process_group,
                )
                compatible.assert_called_once_with(prepared)
                start_counts.assert_not_called()
                start_payload.assert_not_called()
                self.assertTrue(all(node.get_parameters_calls == 1 for node in nodes.values()))

    def test_over_capacity_and_incompatible_sets_use_direct_fused_fallback(self) -> None:
        cases = (
            (
                "over-capacity",
                tuple(f"layer_{i}" for i in range(gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS + 1)),
                True,
            ),
            ("incompatible-layout", ("static", "dynamic"), False),
        )

        for name, selected_nodes, locally_compatible in cases:
            with self.subTest(name=name):
                composite, nodes = _make_grouped_gather_composite(selected_nodes)
                composite.calib = _make_free_pose_calib_for_prefetch_test()
                prepared = object()
                fused_result = _make_fused_result(nodes)
                with (
                    patch.object(torch.distributed, "is_initialized", return_value=True),
                    patch.object(torch.distributed, "get_world_size", return_value=2),
                    patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
                    patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
                    patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
                    patch.object(
                        gaussians_composite_module,
                        "_prepare_fused_gather_inputs",
                        return_value=prepared,
                    ) as prepare,
                    patch.object(
                        gaussians_composite_module,
                        "_can_prefetch_grouped_payload",
                        return_value=locally_compatible,
                    ) as compatible,
                    patch.object(gaussians_composite_module, "_start_layer_count_gather") as start_counts,
                    patch.object(
                        gaussians_composite_module,
                        "_start_fused_payload_gather_from_generation_plan",
                    ) as start_payload,
                    patch.object(
                        gaussians_composite_module,
                        "_fused_all_gather_gaussian_parameters",
                        return_value=fused_result,
                    ) as direct_fused,
                ):
                    pending = composite._maybe_prefetch_gaussian_parameter_gather(
                        is_training_batch=True,
                        gaussian_nodes=None,
                        track_transform_ids=None,
                        track_transforms=None,
                        edited_cuboid_tracks=None,
                    )
                    self.assertIsNone(pending)
                    gathered = composite._gather_gaussian_parameters_once(list(selected_nodes))

                if name == "over-capacity":
                    prepare.assert_not_called()
                    compatible.assert_not_called()
                else:
                    prepare.assert_called_once()
                    compatible.assert_called_once_with(prepared)
                start_counts.assert_not_called()
                start_payload.assert_not_called()
                direct_fused.assert_called_once()
                self.assertEqual(tuple(gathered), selected_nodes)

    def test_stable_generation_captures_fresh_refs_without_validity_consensus(self) -> None:
        composite, nodes = _make_grouped_gather_composite()
        composite.calib = _make_free_pose_calib_for_prefetch_test()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        plan = _make_generation_plan(selected_nodes)
        composite._fused_gather_generation_plan = cast(Any, plan)
        prepared_tokens = (object(), object())
        payload_tokens = (object(), object())
        captured_parameters: list[dict[str, Mapping[str, torch.Tensor]]] = []

        def prepare_from_plan(
            node_ids: list[str],
            parameters_by_node: dict[str, Mapping[str, torch.Tensor]],
            supplied_plan: object,
            *,
            process_group: torch.distributed.ProcessGroup,
        ) -> object:
            self.assertIs(process_group, composite.gaussian_process_group)
            self.assertEqual(tuple(node_ids), selected_nodes)
            self.assertIs(supplied_plan, plan)
            captured_parameters.append(parameters_by_node)
            return prepared_tokens[len(captured_parameters) - 1]

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
            patch.object(
                gaussians_composite_module,
                "_prepare_fused_gather_inputs_from_generation_plan",
                side_effect=prepare_from_plan,
            ) as prepare_cached,
            patch.object(
                gaussians_composite_module,
                "_start_fused_payload_gather_from_generation_plan",
                side_effect=payload_tokens,
            ) as start_payload,
            patch.object(gaussians_composite_module, "_prepare_fused_gather_inputs") as prepare_metadata,
            patch.object(gaussians_composite_module, "_start_layer_count_gather") as start_counts,
            patch.object(torch.distributed, "all_reduce") as unexpected_validation_collective,
        ):
            captured = []
            for generation in range(2):
                for index, node in enumerate(nodes.values()):
                    node.parameters = {"positions": torch.full((index + 1, 3), float(10 * generation + index))}
                pending = composite._maybe_prefetch_gaussian_parameter_gather(
                    is_training_batch=True,
                    gaussian_nodes=None,
                    track_transform_ids=None,
                    track_transforms=None,
                    edited_cuboid_tracks=None,
                )
                self.assertIsNotNone(pending)
                assert pending is not None
                self.assertIs(pending.prepared_inputs, prepared_tokens[generation])
                self.assertIs(pending.prefetched_payload_gather, payload_tokens[generation])
                self.assertFalse(pending.publish_on_success)
                captured.append(pending)

        self.assertEqual(prepare_cached.call_count, 2)
        self.assertEqual(start_payload.call_count, 2)
        prepare_metadata.assert_not_called()
        start_counts.assert_not_called()
        unexpected_validation_collective.assert_not_called()
        self.assertIsNot(captured_parameters[0]["background"], captured_parameters[1]["background"])
        self.assertIs(captured[1].parameters_by_node["road"], nodes["road"].parameters)

    def test_cached_prepare_or_launch_failure_is_fatal_without_local_fallback(self) -> None:
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        for failure_site in ("prepare", "launch"):
            with self.subTest(failure_site=failure_site):
                composite, _nodes = _make_grouped_gather_composite()
                composite.calib = _make_free_pose_calib_for_prefetch_test()
                plan = _make_generation_plan(selected_nodes)
                composite._fused_gather_generation_plan = cast(Any, plan)
                prepared = object()
                prepare_error = RuntimeError("cached topology changed")
                launch_error = RuntimeError("payload launch failed")
                with (
                    patch.object(torch.distributed, "is_initialized", return_value=True),
                    patch.object(torch.distributed, "get_world_size", return_value=2),
                    patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
                    patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
                    patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
                    patch.object(
                        gaussians_composite_module,
                        "_prepare_fused_gather_inputs_from_generation_plan",
                        side_effect=prepare_error if failure_site == "prepare" else None,
                        return_value=prepared,
                    ),
                    patch.object(
                        gaussians_composite_module,
                        "_start_fused_payload_gather_from_generation_plan",
                        side_effect=launch_error if failure_site == "launch" else None,
                    ),
                    self.assertRaisesRegex(RuntimeError, "topology changed|launch failed") as raised,
                ):
                    composite._maybe_prefetch_gaussian_parameter_gather(
                        is_training_batch=True,
                        gaussian_nodes=None,
                        track_transform_ids=None,
                        track_transforms=None,
                        edited_cuboid_tracks=None,
                    )

                self.assertIsNone(composite._fused_gather_generation_plan)
                self.assertTrue(
                    any("terminate all ranks" in note for note in getattr(raised.exception, "__notes__", ()))
                )

    def test_refresh_candidate_remains_unpublished_until_successful_consumption(self) -> None:
        composite, nodes = _make_grouped_gather_composite()
        composite.calib = _make_free_pose_calib_for_prefetch_test()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        candidate = _make_generation_plan(selected_nodes)
        prepared_after_mutation = object()
        count_token = object()
        payload = object()
        composite._fused_gather_generation_plan = None
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                prepared_inputs=cast(Any, prepared_after_mutation),
                count_gather=cast(Any, count_token),
            ),
        )

        def consume_refresh(*_args: Any, **_kwargs: Any) -> _FusedGatherResult:
            self.assertIsNone(composite._fused_gather_generation_plan)
            return _make_fused_result(nodes)

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
            patch.object(gaussians_composite_module, "_finish_layer_count_gather", return_value=object()),
            patch.object(gaussians_composite_module, "_make_fused_gather_generation_plan", return_value=candidate),
            patch.object(
                gaussians_composite_module,
                "_prepare_fused_gather_inputs_from_generation_plan",
                return_value=prepared_after_mutation,
            ),
            patch.object(
                gaussians_composite_module,
                "_start_fused_payload_gather_from_generation_plan",
                return_value=payload,
            ),
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                side_effect=consume_refresh,
            ),
        ):
            pending = composite._maybe_prefetch_gaussian_parameter_gather(
                is_training_batch=True,
                gaussian_nodes=None,
                track_transform_ids=None,
                track_transforms=None,
                edited_cuboid_tracks=None,
            )
            self.assertIsNone(composite._fused_gather_generation_plan)
            self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
            assert pending is not None
            self.assertIs(pending.generation_plan, candidate)
            self.assertTrue(pending.publish_on_success)

            composite._gather_gaussian_parameters_once(list(selected_nodes), prefetched=pending)

        self.assertIs(composite._fused_gather_generation_plan, candidate)

    def test_refresh_resolution_failure_clears_pending_and_published_plans(self) -> None:
        composite, _nodes = _make_grouped_gather_composite()
        composite._fused_gather_generation_plan = cast(Any, _make_generation_plan(_GENERATION_PLAN_TEST_NODES))
        count_token = object()
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                prepared_inputs=cast(Any, object()),
                count_gather=cast(Any, count_token),
            ),
        )

        with (
            patch.object(
                gaussians_composite_module,
                "_finish_layer_count_gather",
                side_effect=RuntimeError("count refresh failed"),
            ) as finish,
            self.assertRaisesRegex(RuntimeError, "count refresh failed"),
        ):
            composite._finish_fused_gather_generation_plan_refresh()

        finish.assert_called_once_with(count_token)
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
        self.assertIsNone(composite._fused_gather_generation_plan)

    def test_refresh_consumption_failure_drains_payload_and_leaves_no_plan(self) -> None:
        composite, nodes = _make_grouped_gather_composite()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        candidate = _make_generation_plan(selected_nodes)
        payload = object()
        pending = gaussians_composite_module._PendingGaussianParameterGather(
            gaussian_nodes=selected_nodes,
            parameters_by_node={node_id: node.parameters for node_id, node in nodes.items()},
            prepared_inputs=cast(Any, object()),
            generation_plan=cast(Any, candidate),
            prefetched_payload_gather=cast(Any, payload),
            publish_on_success=True,
        )
        composite._fused_gather_generation_plan = None

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                side_effect=RuntimeError("consume failed"),
            ),
            patch.object(gaussians_composite_module, "_discard_fused_payload_gather") as discard,
            self.assertRaisesRegex(RuntimeError, "consume failed"),
        ):
            composite._gather_gaussian_parameters_once(list(selected_nodes), prefetched=pending)

        discard.assert_called_once_with(payload)
        self.assertIsNone(composite._fused_gather_generation_plan)

    def test_structural_refresh_clears_published_plan_and_drains_abandoned_candidate_once(self) -> None:
        composite, _nodes = _make_grouped_gather_composite()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        plan = _make_generation_plan(selected_nodes)
        prepared = SimpleNamespace(
            world_size=2,
            local_counts=(1, 2, 3, 4),
            device=torch.device("cpu"),
            schema_hash=tuple(range(_COMMUNICATION_SCHEMA_HASH_WORD_COUNT)),
        )
        count_token = object()
        strategy = gaussians_composite_module.MCMCStrategy.__new__(gaussians_composite_module.MCMCStrategy)
        torch.nn.Module.__init__(strategy)
        strategy.config = cast(
            Any,
            SimpleNamespace(
                relocate=SimpleNamespace(start_iteration=1, end_iteration=100, frequency=10),
                add=SimpleNamespace(start_iteration=1, end_iteration=100, frequency=20),
            ),
        )
        strategy.frozen = False
        composite.gaussians_strategy = strategy
        composite._fused_gather_generation_plan = cast(Any, plan)
        composite._pending_fused_gather_generation_plan_refresh = None

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(gaussians_composite_module, "_prepare_fused_gather_inputs", return_value=prepared),
            patch.object(gaussians_composite_module, "_can_prefetch_grouped_payload", return_value=True),
            patch.object(
                gaussians_composite_module,
                "_start_layer_count_gather",
                return_value=count_token,
            ) as start_counts,
            patch.object(gaussians_composite_module, "_discard_layer_count_gather") as discard_counts,
            patch.object(torch.distributed, "all_reduce") as unexpected_consensus,
        ):
            composite._refresh_fused_gather_generation_plan_after_mutations(global_step=9)
            self.assertIs(composite._fused_gather_generation_plan, plan)
            self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)

            composite._refresh_fused_gather_generation_plan_after_mutations(global_step=10)
            self.assertIsNone(composite._fused_gather_generation_plan)
            pending = composite._pending_fused_gather_generation_plan_refresh
            self.assertIsNotNone(pending)
            assert pending is not None
            self.assertIs(pending.prepared_inputs, prepared)
            self.assertIs(pending.count_gather, count_token)

            composite._refresh_fused_gather_generation_plan_after_mutations(global_step=20)
            composite.drain_fused_gather_generation_plan_refresh()

        start_counts.assert_called_once_with(
            2,
            (1, 2, 3, 4),
            torch.device("cpu"),
            tuple(range(_COMMUNICATION_SCHEMA_HASH_WORD_COUNT)),
            process_group=composite.gaussian_process_group,
        )
        discard_counts.assert_called_once_with(count_token)
        unexpected_consensus.assert_not_called()
        self.assertIsNone(composite._fused_gather_generation_plan)
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)

    def test_structural_refresh_wrapper_failure_drains_count_work_and_preserves_primary_error(self) -> None:
        composite, _nodes = _make_grouped_gather_composite()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        plan = _make_generation_plan(selected_nodes)
        prepared = SimpleNamespace(
            world_size=2,
            local_counts=(1, 2, 3, 4),
            device=torch.device("cpu"),
            schema_hash=tuple(range(_COMMUNICATION_SCHEMA_HASH_WORD_COUNT)),
        )
        count_token = object()
        primary = RuntimeError("refresh wrapper failed")
        composite._fused_gather_generation_plan = cast(Any, plan)

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
            patch.object(composite, "_is_fused_gather_generation_refresh_step", return_value=True),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(gaussians_composite_module, "_prepare_fused_gather_inputs", return_value=prepared),
            patch.object(gaussians_composite_module, "_can_prefetch_grouped_payload", return_value=True),
            patch.object(gaussians_composite_module, "_start_layer_count_gather", return_value=count_token),
            patch.object(
                gaussians_composite_module,
                "_PendingGaussianGenerationPlanRefresh",
                side_effect=primary,
            ),
            patch.object(
                gaussians_composite_module,
                "_discard_layer_count_gather",
                side_effect=RuntimeError("refresh count drain failed"),
            ) as discard,
            self.assertRaises(RuntimeError) as raised,
        ):
            composite._refresh_fused_gather_generation_plan_after_mutations(global_step=1)

        self.assertIs(raised.exception, primary)
        discard.assert_called_once_with(count_token)
        self.assertTrue(any("refresh count drain failed" in note for note in getattr(primary, "__notes__", ())))
        self.assertIsNone(composite._fused_gather_generation_plan)
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)

    def test_exact_strategy_schedule_matrix_and_subclasses_are_ineligible(self) -> None:
        composite, _nodes = _make_grouped_gather_composite()

        mcmc = gaussians_composite_module.MCMCStrategy.__new__(gaussians_composite_module.MCMCStrategy)
        torch.nn.Module.__init__(mcmc)
        mcmc.config = cast(
            Any,
            SimpleNamespace(
                relocate=SimpleNamespace(start_iteration=1, end_iteration=31, frequency=10),
                add=SimpleNamespace(start_iteration=1, end_iteration=31, frequency=15),
            ),
        )
        mcmc.frozen = False
        composite.gaussians_strategy = mcmc
        self.assertTrue(composite._supports_fused_gather_generation_plan())
        for step, expected in ((0, False), (1, False), (9, False), (10, True), (15, True), (30, True), (31, False)):
            with self.subTest(strategy="mcmc", step=step):
                self.assertEqual(composite._is_fused_gather_generation_refresh_step(step), expected)

        gsplat = gaussians_composite_module.GSplatStrategy.__new__(gaussians_composite_module.GSplatStrategy)
        torch.nn.Module.__init__(gsplat)
        gsplat.config = cast(
            Any,
            SimpleNamespace(
                densify=SimpleNamespace(start_iteration=1, end_iteration=41, frequency=12),
                prune=SimpleNamespace(start_iteration=1, end_iteration=41, frequency=20),
            ),
        )
        gsplat.frozen = False
        composite.gaussians_strategy = gsplat
        self.assertTrue(composite._supports_fused_gather_generation_plan())
        for step, expected in ((0, False), (1, False), (11, False), (12, True), (20, True), (40, True), (41, False)):
            with self.subTest(strategy="gsplat", step=step):
                self.assertEqual(composite._is_fused_gather_generation_refresh_step(step), expected)
        gsplat.frozen = True
        self.assertFalse(composite._is_fused_gather_generation_refresh_step(20))

        class CustomMCMCStrategy(gaussians_composite_module.MCMCStrategy):
            pass

        custom_mcmc = CustomMCMCStrategy.__new__(CustomMCMCStrategy)
        torch.nn.Module.__init__(custom_mcmc)
        custom_mcmc.config = mcmc.config
        custom_mcmc.frozen = False
        for custom in (custom_mcmc, torch.nn.Identity()):
            composite.gaussians_strategy = cast(Any, custom)
            self.assertFalse(composite._supports_fused_gather_generation_plan())

    def test_custom_strategy_falls_back_to_direct_fused_gather_at_point_of_use(self) -> None:
        composite, nodes = _make_grouped_gather_composite()
        composite.calib = _make_free_pose_calib_for_prefetch_test()
        selected_nodes = _GENERATION_PLAN_TEST_NODES

        class CustomMCMCStrategy(gaussians_composite_module.MCMCStrategy):
            pass

        custom = CustomMCMCStrategy.__new__(CustomMCMCStrategy)
        torch.nn.Module.__init__(custom)
        custom.frozen = False
        composite.gaussians_strategy = custom
        composite._fused_gather_generation_plan = cast(Any, _make_generation_plan(selected_nodes))
        fused_result = _make_fused_result(nodes)

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(gaussians_composite_module, "_prepare_fused_gather_inputs") as speculative_prepare,
            patch.object(gaussians_composite_module, "_start_layer_count_gather") as speculative_counts,
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                return_value=fused_result,
            ) as direct_fused,
        ):
            pending = composite._maybe_prefetch_gaussian_parameter_gather(
                is_training_batch=True,
                gaussian_nodes=None,
                track_transform_ids=None,
                track_transforms=None,
                edited_cuboid_tracks=None,
            )
            self.assertIsNone(pending)
            self.assertIsNone(composite._fused_gather_generation_plan)
            self.assertEqual(tuple(node.get_parameters_calls for node in nodes.values()), (0, 0, 0, 0))

            composite._gather_gaussian_parameters_once(list(selected_nodes))

        speculative_prepare.assert_not_called()
        speculative_counts.assert_not_called()
        direct_fused.assert_called_once_with(
            list(selected_nodes),
            {node_id: node.parameters for node_id, node in nodes.items()},
            process_group=composite.gaussian_process_group,
            integer_range_validation_state=mock.ANY,
            prepared_inputs=None,
            generation_plan=None,
            prefetched_payload_gather=None,
        )

    def test_load_hooks_invalidate_once_and_initialize_drain_is_idempotent(self) -> None:
        selected_nodes = _GENERATION_PLAN_TEST_NODES

        composite, _nodes = _make_grouped_gather_composite()
        composite.calib = cast(Any, torch.nn.Identity())
        composite._cam_to_lidar_tile_mappings = {7: cast(Any, object())}
        composite._tracked_rows_grouped = True
        composite.register_load_state_dict_pre_hook(composite._invalidate_fused_gather_generation_plan_before_load)
        composite.register_load_state_dict_post_hook(composite._invalidate_derived_render_caches_after_load)
        load_count_token = object()
        composite._fused_gather_generation_plan = cast(Any, _make_generation_plan(selected_nodes))
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                prepared_inputs=cast(Any, object()),
                count_gather=cast(Any, load_count_token),
            ),
        )
        with patch.object(gaussians_composite_module, "_discard_layer_count_gather") as discard_counts:
            with (
                patch.object(
                    composite,
                    "_invalidate_fused_gather_generation_plan",
                    wraps=composite._invalidate_fused_gather_generation_plan,
                ) as invalidate_fused_gather,
                patch.object(
                    composite,
                    "_invalidate_collect_caches",
                    wraps=composite._invalidate_collect_caches,
                ) as invalidate_collect_caches,
            ):
                composite.load_state_dict({}, strict=False)
            invalidate_fused_gather.assert_called_once_with()
            invalidate_collect_caches.assert_called_once_with()
            composite._invalidate_fused_gather_generation_plan()
        discard_counts.assert_called_once_with(load_count_token)
        self.assertIsNone(composite._fused_gather_generation_plan)
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
        self.assertEqual(composite._cam_to_lidar_tile_mappings, {})
        self.assertFalse(composite._tracked_rows_grouped)

        composite, nodes = _make_grouped_gather_composite()
        init_count_token = object()
        composite._fused_gather_generation_plan = cast(Any, _make_generation_plan(selected_nodes))
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                prepared_inputs=cast(Any, object()),
                count_gather=cast(Any, init_count_token),
            ),
        )
        for node in nodes.values():
            node.maybe_initialize_gaussians = Mock()  # type: ignore[attr-defined]
        composite.gaussians_strategy = cast(
            Any,
            SimpleNamespace(
                maybe_initialize_buffers=Mock(),
                refresh_runtime_scenes=Mock(),
            ),
        )
        with (
            patch.object(gaussians_composite_module, "_discard_layer_count_gather") as discard_counts,
            patch.object(composite, "_apply_component_only_background_scope"),
            patch.object(composite, "_apply_component_only_background_hooks"),
            patch.object(composite, "_apply_component_only_frozen_tracks"),
            patch.object(composite, "_apply_component_only_freeze"),
        ):
            composite.initialize_strategy_and_maybe_gaussians()
            composite._invalidate_fused_gather_generation_plan()
        discard_counts.assert_called_once_with(init_count_token)
        self.assertIsNone(composite._fused_gather_generation_plan)
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)

    def test_recursive_load_drains_fused_gather_before_descendant_failure(self) -> None:
        composite, _nodes = _make_grouped_gather_composite()
        composite.register_load_state_dict_pre_hook(composite._invalidate_fused_gather_generation_plan_before_load)

        composite._fused_gather_generation_plan = cast(Any, _make_generation_plan(_GENERATION_PLAN_TEST_NODES))
        count_token = object()
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                prepared_inputs=cast(Any, object()),
                count_gather=cast(Any, count_token),
            ),
        )
        load_probe = torch.nn.Linear(1, 1, bias=False)
        old_weight = load_probe.weight
        composite.add_module("load_probe", load_probe)
        root = torch.nn.Module()
        root.add_module("model", composite)
        events: list[str] = []

        def discard_refresh(value: object) -> None:
            self.assertIs(value, count_token)
            self.assertIsNone(composite._fused_gather_generation_plan)
            self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
            events.append("drain")

        def fail_after_descendant_load(module: torch.nn.Module, _incompatible_keys: Any) -> None:
            self.assertEqual(events, ["drain"])
            self.assertIsNot(cast(torch.nn.Linear, module).weight, old_weight)
            self.assertIsNone(composite._fused_gather_generation_plan)
            self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
            events.append("descendant_load")
            raise ValueError("injected descendant post-load failure")

        load_probe.register_load_state_dict_post_hook(fail_after_descendant_load)

        with (
            patch.object(
                gaussians_composite_module,
                "_discard_layer_count_gather",
                side_effect=discard_refresh,
            ) as discard_counts,
            pytest.raises(ValueError, match="injected descendant post-load failure"),
        ):
            root.load_state_dict(
                {"model.load_probe.weight": torch.zeros_like(load_probe.weight)},
                strict=False,
                assign=True,
            )

        discard_counts.assert_called_once_with(count_token)
        self.assertEqual(events, ["drain", "descendant_load"])
        self.assertIsNone(composite._fused_gather_generation_plan)
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)

    def test_nontraining_routes_wait_release_and_idempotently_drain_refresh(self) -> None:
        class PreparedInputs:
            pass

        cases = (
            ("not-training", False, False, False),
            ("module-eval", True, True, False),
            ("no-grad", True, False, True),
        )
        for name, is_training_batch, module_eval, no_grad in cases:
            with self.subTest(name=name):
                composite, _nodes = _make_grouped_gather_composite()
                composite.train(not module_eval)
                work = Mock()
                count_gather = gaussians_composite_module._PendingLayerCountGather(
                    local_packet=torch.empty(0),
                    gathered_packet=torch.empty(0),
                    work=work,
                    local_counts=(1,),
                    schema_hash=tuple(range(_COMMUNICATION_SCHEMA_HASH_WORD_COUNT)),
                    device=torch.device("cpu"),
                    world_rank=0,
                    world_size=2,
                    process_group=composite.gaussian_process_group,
                )
                prepared = PreparedInputs()
                prepared_ref = weakref.ref(prepared)
                composite._fused_gather_generation_plan = cast(
                    Any,
                    _make_generation_plan(_GENERATION_PLAN_TEST_NODES),
                )
                composite._pending_fused_gather_generation_plan_refresh = cast(
                    Any,
                    gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                        prepared_inputs=cast(Any, prepared),
                        count_gather=count_gather,
                    ),
                )
                del prepared

                def wait_after_release() -> None:
                    self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
                    self.assertIsNone(composite._fused_gather_generation_plan)

                work.wait.side_effect = wait_after_release

                def call() -> object:
                    kwargs: dict[str, Any] = {
                        "is_training_batch": is_training_batch,
                        "gaussian_nodes": None,
                        "track_transform_ids": None,
                        "track_transforms": None,
                        "edited_cuboid_tracks": None,
                    }
                    if no_grad:
                        with torch.no_grad():
                            return composite._maybe_prefetch_gaussian_parameter_gather(**kwargs)
                    return composite._maybe_prefetch_gaussian_parameter_gather(**kwargs)

                self.assertIsNone(call())
                work.wait.assert_called_once_with()
                self.assertEqual(count_gather.state, "discarded")
                self.assertIsNone(prepared_ref())
                self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
                self.assertIsNone(composite._fused_gather_generation_plan)

                # The state was cleared before the wait, so a repeated lifecycle
                # call neither retains inputs nor waits on the work a second time.
                self.assertIsNone(call())
                work.wait.assert_called_once_with()

    def test_nontraining_refresh_wait_failure_is_fail_closed_and_idempotent(self) -> None:
        class PreparedInputs:
            pass

        composite, _nodes = _make_grouped_gather_composite()
        work = Mock()
        work.wait.side_effect = RuntimeError("count drain failed")
        count_gather = gaussians_composite_module._PendingLayerCountGather(
            local_packet=torch.empty(0),
            gathered_packet=torch.empty(0),
            work=work,
            local_counts=(1,),
            schema_hash=tuple(range(_COMMUNICATION_SCHEMA_HASH_WORD_COUNT)),
            device=torch.device("cpu"),
            world_rank=0,
            world_size=2,
            process_group=composite.gaussian_process_group,
        )
        prepared = PreparedInputs()
        prepared_ref = weakref.ref(prepared)
        composite._fused_gather_generation_plan = cast(Any, _make_generation_plan(_GENERATION_PLAN_TEST_NODES))
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                prepared_inputs=cast(Any, prepared),
                count_gather=count_gather,
            ),
        )
        del prepared

        kwargs: dict[str, Any] = {
            "is_training_batch": False,
            "gaussian_nodes": None,
            "track_transform_ids": None,
            "track_transforms": None,
            "edited_cuboid_tracks": None,
        }
        with self.assertRaisesRegex(RuntimeError, "count drain failed"):
            composite._maybe_prefetch_gaussian_parameter_gather(**kwargs)

        work.wait.assert_called_once_with()
        self.assertEqual(count_gather.state, "discarded")
        self.assertIsNone(prepared_ref())
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
        self.assertIsNone(composite._fused_gather_generation_plan)

        # The failed wait is terminal for that token.  State remains cleared,
        # so later invalidation is safe and cannot accidentally retry it.
        self.assertIsNone(composite._maybe_prefetch_gaussian_parameter_gather(**kwargs))
        work.wait.assert_called_once_with()

    def test_prefetch_bypasses_every_non_target_route_without_reading_parameters(self) -> None:
        composite, nodes = _make_grouped_gather_composite()
        free_pose = _make_free_pose_calib_for_prefetch_test()

        class DerivedFreePoseCalib(FreePoseCalib):
            pass

        derived_free_pose = DerivedFreePoseCalib.__new__(DerivedFreePoseCalib)
        torch.nn.Module.__init__(derived_free_pose)
        selected_nodes = list(_GENERATION_PLAN_TEST_NODES)

        cases: tuple[dict[str, Any], ...] = (
            {"name": "not-training", "call": {"is_training_batch": False}},
            {"name": "module-eval", "eval": True},
            {"name": "no-grad", "no_grad": True},
            {"name": "non-FreePose", "calib": torch.nn.Identity()},
            {"name": "FreePose-subclass", "calib": derived_free_pose},
            {"name": "explicit-nodes", "call": {"gaussian_nodes": ["first", "second"]}},
            {"name": "track-ids", "call": {"track_transform_ids": ["track"]}},
            {"name": "track-transforms", "call": {"track_transforms": torch.empty(0)}},
            {"name": "edited-tracks", "call": {"edited_cuboid_tracks": object()}},
            {"name": "dist-off", "initialized": False},
            {"name": "world-one", "world_size": 1},
            {"name": "non-NCCL", "backend": "gloo"},
            {"name": "zero-nodes", "selected_nodes": []},
            {
                "name": "above-grouped-capacity",
                "selected_nodes": [
                    f"layer_{i}" for i in range(gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS + 1)
                ],
            },
            {"name": "duplicate-nodes", "selected_nodes": ["first", "first"]},
        )

        with (
            patch.object(gaussians_composite_module, "_prepare_fused_gather_inputs") as prepare,
            patch.object(gaussians_composite_module, "_start_layer_count_gather") as start,
            patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
        ):
            for case in cases:
                with self.subTest(name=case["name"]):
                    prepare.reset_mock()
                    start.reset_mock()
                    composite.train(not case.get("eval", False))
                    composite.calib = case.get("calib", free_pose)
                    count_token = object()
                    composite._fused_gather_generation_plan = cast(
                        Any,
                        _make_generation_plan(_GENERATION_PLAN_TEST_NODES),
                    )
                    composite._pending_fused_gather_generation_plan_refresh = cast(
                        Any,
                        gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                            prepared_inputs=cast(Any, object()),
                            count_gather=cast(Any, count_token),
                        ),
                    )
                    call: dict[str, Any] = {
                        "is_training_batch": True,
                        "gaussian_nodes": None,
                        "track_transform_ids": None,
                        "track_transforms": None,
                        "edited_cuboid_tracks": None,
                    }
                    call.update(case.get("call", {}))
                    with (
                        patch.object(
                            torch.distributed,
                            "is_initialized",
                            return_value=case.get("initialized", True),
                        ),
                        patch.object(
                            torch.distributed,
                            "get_world_size",
                            return_value=case.get("world_size", 2),
                        ),
                        patch.object(
                            torch.distributed,
                            "get_backend",
                            return_value=case.get("backend", torch.distributed.Backend.NCCL),
                        ),
                        patch.object(
                            composite,
                            "get_gaussians_node_ids",
                            return_value=case.get("selected_nodes", selected_nodes),
                        ),
                        patch.object(gaussians_composite_module, "_discard_layer_count_gather") as discard_counts,
                    ):
                        if case.get("no_grad", False):
                            with torch.no_grad():
                                pending = composite._maybe_prefetch_gaussian_parameter_gather(**call)
                        else:
                            pending = composite._maybe_prefetch_gaussian_parameter_gather(**call)
                    self.assertIsNone(pending)
                    discard_counts.assert_called_once_with(count_token)
                    self.assertIsNone(composite._fused_gather_generation_plan)
                    self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)
                    prepare.assert_not_called()
                    start.assert_not_called()

        composite.train()
        self.assertTrue(all(node.get_parameters_calls == 0 for node in nodes.values()))

    def test_prefetch_starts_payload_from_cached_generation_plan(self) -> None:
        composite, nodes = _make_grouped_gather_composite()
        composite.calib = _make_free_pose_calib_for_prefetch_test()
        composite.train()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        plan = _make_generation_plan(selected_nodes)
        composite._fused_gather_generation_plan = cast(Any, plan)
        prepared = object()
        payload = object()

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
            patch.object(
                gaussians_composite_module,
                "_prepare_fused_gather_inputs_from_generation_plan",
                return_value=prepared,
            ) as prepare,
            patch.object(
                gaussians_composite_module,
                "_start_fused_payload_gather_from_generation_plan",
                return_value=payload,
            ) as start_payload,
        ):
            pending = composite._maybe_prefetch_gaussian_parameter_gather(
                is_training_batch=True,
                gaussian_nodes=None,
                track_transform_ids=None,
                track_transforms=None,
                edited_cuboid_tracks=None,
            )

        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertIs(pending.generation_plan, plan)
        self.assertIs(pending.prefetched_payload_gather, payload)
        prepare.assert_called_once()
        start_payload.assert_called_once_with(
            prepared,
            plan,
            process_group=composite.gaussian_process_group,
        )
        self.assertTrue(all(node.get_parameters_calls == 1 for node in nodes.values()))

    def test_camera_only_training_forward_consumes_prefetched_full_gather(self) -> None:
        composite, _nodes = _make_grouped_gather_composite()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        pending = gaussians_composite_module._PendingGaussianParameterGather(
            gaussian_nodes=selected_nodes,
            parameters_by_node={},
            prepared_inputs=cast(Any, object()),
            generation_plan=cast(Any, _make_generation_plan(selected_nodes)),
            prefetched_payload_gather=cast(Any, object()),
        )
        gathered = object()
        collected: dict[str, torch.Tensor] = {}
        rendered = SimpleNamespace(visibility=None, cumulated_weights=None)
        composite._visibility_local_ranges = None
        composite.camera_extra_ray_signal_infos = cast(Any, ())
        composite._camera_renderer = cast(Any, object())

        with (
            patch.object(composite, "_build_frustum_cull", return_value=None),
            patch.object(composite, "_gather_gaussian_parameters_once", return_value=gathered) as gather,
            patch.object(composite, "_collect_sensor", return_value=(collected, None)) as collect_sensor,
            patch.object(composite, "_render_collected", return_value=rendered),
        ):
            result = composite.forward_gaussians(
                rendering_data_cam=cast(Any, object()),
                frame_meta_cam=cast(Any, [object()]),
                is_training_batch=True,
                prefetched_parameter_gather=pending,
            )

        gather.assert_called_once_with(list(selected_nodes), prefetched=pending)
        self.assertIs(collect_sensor.call_args.kwargs["gathered_parameters"], gathered)
        self.assertIs(result.rendered_cam, rendered)
        self.assertIsNone(result.rendered_lidar)

    def test_unconditional_gaussian_mutators_invalidate_before_mutation(self) -> None:
        composite, _nodes = _make_grouped_gather_composite()
        calls = (
            lambda: composite.remove_background_gaussians([], torch.empty(3), torch.empty(3)),
            composite.restore_training_parameters,
        )

        for index, call in enumerate(calls):
            with self.subTest(index=index):
                with (
                    patch.object(composite, "_invalidate_collect_caches") as first_mutation,
                    patch.object(
                        composite,
                        "_invalidate_fused_gather_generation_plan",
                        side_effect=RuntimeError("invalidation sentinel"),
                    ) as invalidate,
                    self.assertRaisesRegex(RuntimeError, "invalidation sentinel"),
                ):
                    call()
                invalidate.assert_called_once_with()
                first_mutation.assert_not_called()

    def test_lidar_only_forward_consumes_and_records_prefetched_full_gather(self) -> None:
        composite, nodes = _make_grouped_gather_composite()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        pending = gaussians_composite_module._PendingGaussianParameterGather(
            gaussian_nodes=selected_nodes,
            parameters_by_node={},
            prepared_inputs=cast(Any, object()),
            generation_plan=cast(Any, _make_generation_plan(selected_nodes)),
            prefetched_payload_gather=cast(Any, object()),
        )
        gathered_values = {node_id: SimpleNamespace(is_cuda=True, record_stream=Mock()) for node_id in selected_nodes}
        gathered = {
            node_id: SimpleNamespace(parameters={"value": gathered_values[node_id]}) for node_id in selected_nodes
        }
        for node in nodes.values():
            node.record_stream = Mock()  # type: ignore[attr-defined]
        collected: dict[str, torch.Tensor] = {}
        rendered = SimpleNamespace(visibility=None, cumulated_weights=None)
        rendering = SimpleNamespace(record_stream=Mock())
        frame_meta = SimpleNamespace(record_stream=Mock())
        lidar_stream = SimpleNamespace(wait_stream=Mock())
        current_stream = SimpleNamespace(wait_stream=Mock())
        composite._visibility_local_ranges = None
        composite.lidar_extra_ray_signal_infos = cast(Any, ())
        composite._lidar_renderer = cast(Any, object())

        with (
            patch.object(composite, "_build_frustum_cull", return_value=None),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(composite, "_resolve_lidar_stream", return_value=lidar_stream),
            patch.object(composite, "_gather_gaussian_parameters_once", return_value=gathered) as gather,
            patch.object(composite, "_collect_sensor", return_value=(collected, None)) as collect_sensor,
            patch.object(composite, "_render_collected", return_value=rendered),
            patch.object(torch.cuda, "current_stream", return_value=current_stream),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
        ):
            result = composite.forward_gaussians(
                rendering_data_lidar=cast(Any, rendering),
                frame_meta_lidar=cast(Any, [frame_meta]),
                is_training_batch=True,
                prefetched_parameter_gather=pending,
            )

        gather.assert_called_once_with(list(selected_nodes), prefetched=pending)
        self.assertIs(collect_sensor.call_args.kwargs["gathered_parameters"], gathered)
        for value in gathered_values.values():
            value.record_stream.assert_called_once_with(lidar_stream)
        for node in nodes.values():
            node.record_stream.assert_called_once_with(lidar_stream)  # type: ignore[attr-defined]
        lidar_stream.wait_stream.assert_called_once_with(current_stream)
        current_stream.wait_stream.assert_called_once_with(lidar_stream)
        rendering.record_stream.assert_called_once_with(lidar_stream)
        frame_meta.record_stream.assert_called_once_with(lidar_stream)
        self.assertIsNone(result.rendered_cam)
        self.assertIs(result.rendered_lidar, rendered)

    def test_lidar_label_stamp_looks_up_mapping_after_gaussian_forward(self) -> None:
        composite, _nodes = _make_grouped_gather_composite()
        composite.config.renderer.active_tile_mask = ActiveTileMaskConfig(
            enabled=True,
            strategy=ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP,
        )

        events: list[str] = []
        mapping_before = cast(Any, object())
        mapping_after = cast(Any, object())
        tile_mask = cast(Any, object())
        labels = cast(Any, SimpleNamespace(flags=object()))
        composite._cam_to_lidar_tile_mappings = {3: mapping_before}
        batch = cast(
            Any,
            SimpleNamespace(
                data=SimpleNamespace(
                    camera=SimpleNamespace(
                        meta=[SimpleNamespace(unique_sensor_idx=3)],
                        labels=SimpleNamespace(flags=None),
                    ),
                    lidar=SimpleNamespace(meta=[object()], labels=labels),
                ),
                rendering=SimpleNamespace(
                    camera=SimpleNamespace(active_tile_mask=None),
                    lidar=SimpleNamespace(active_tile_mask=tile_mask),
                ),
            ),
        )
        rendered = cast(Any, SimpleNamespace(rendered_cam=None))

        def forward_gaussians(**_kwargs: Any) -> Any:
            events.append("forward_gaussians")
            composite._cam_to_lidar_tile_mappings[3] = mapping_after
            return rendered

        with (
            patch.object(composite, "_maybe_prefetch_gaussian_parameter_gather", return_value=None),
            patch.object(composite, "forward_gaussians", side_effect=forward_gaussians),
            patch.object(
                gaussians_composite_module,
                "mark_inactive_lidar_tile_rays_inplace",
                side_effect=lambda *_args: events.append("stamp_labels"),
            ) as stamp,
        ):
            composite.forward(
                batch,
                render_background=False,
                enable_pp=False,
                clamp_output=False,
            )

        self.assertEqual(events, ["forward_gaussians", "stamp_labels"])
        stamp.assert_called_once_with(labels, tile_mask, mapping_after)

    def test_distillation_transition_invalidates_before_freezing(self) -> None:
        composite, _nodes = _make_grouped_gather_composite()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        count_token = object()
        composite._fused_gather_generation_plan = cast(Any, _make_generation_plan(selected_nodes))
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                prepared_inputs=cast(Any, object()),
                count_gather=cast(Any, count_token),
            ),
        )
        composite._distillation_start_step = 5
        composite._distillation_active = False
        composite.post_processings = cast(Any, [])
        events: list[str] = []

        with (
            patch.object(
                gaussians_composite_module,
                "_discard_layer_count_gather",
                side_effect=lambda token: events.append("drain"),
            ) as discard,
            patch.object(
                composite,
                "_freeze_scene_for_distillation",
                side_effect=lambda: events.append("freeze"),
            ) as freeze,
        ):
            additional = composite.update_step_train_batch_start(epoch=0, global_step=5, system=object())

        self.assertEqual(additional, {})
        self.assertEqual(events, ["drain", "freeze"])
        discard.assert_called_once_with(count_token)
        freeze.assert_called_once_with()
        self.assertTrue(composite._distillation_active)
        self.assertIsNone(composite._fused_gather_generation_plan)
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)

    def test_forward_payload_launch_preserves_primary_error_when_drain_fails(self) -> None:
        composite, _first, _second = _make_gather_composite()
        composite.calib = _make_free_pose_calib_for_prefetch_test()
        payload_gather = object()
        plan = _make_generation_plan(("first", "second"))
        composite._fused_gather_generation_plan = cast(Any, plan)
        pending = gaussians_composite_module._PendingGaussianParameterGather(
            gaussian_nodes=("first", "second"),
            parameters_by_node={},
            prepared_inputs=cast(Any, object()),
            generation_plan=cast(Any, plan),
            prefetched_payload_gather=cast(Any, payload_gather),
        )
        primary = RuntimeError("calibration failed")
        composite.calib.forward = Mock(side_effect=primary)  # type: ignore[method-assign]
        data = SimpleNamespace(
            camera=SimpleNamespace(meta=[object()]),
            lidar=SimpleNamespace(meta=[object()]),
        )
        batch = SimpleNamespace(rendering=None, data=data)

        with (
            patch.object(composite, "_maybe_prefetch_gaussian_parameter_gather", return_value=pending),
            patch.object(
                gaussians_composite_module,
                "_discard_fused_payload_gather",
                side_effect=RuntimeError("drain failed"),
            ) as discard,
            self.assertRaises(RuntimeError) as raised,
        ):
            composite.forward(batch, is_training_batch=True)  # type: ignore[arg-type]

        self.assertIs(raised.exception, primary)
        self.assertTrue(any("drain failed" in note for note in getattr(primary, "__notes__", ())))
        discard.assert_called_once_with(payload_gather)
        self.assertIsNone(composite._fused_gather_generation_plan)

    def test_forward_prefetch_setup_error_invalidates_and_preserves_primary_error(self) -> None:
        composite, _first, _second = _make_gather_composite()
        plan = _make_generation_plan(("first", "second"))
        count_token = object()
        composite._fused_gather_generation_plan = cast(Any, plan)
        composite._pending_fused_gather_generation_plan_refresh = cast(
            Any,
            gaussians_composite_module._PendingGaussianGenerationPlanRefresh(
                prepared_inputs=cast(Any, object()),
                count_gather=cast(Any, count_token),
            ),
        )
        primary = RuntimeError("prefetch setup failed")

        with (
            patch.object(composite, "_maybe_prefetch_gaussian_parameter_gather", side_effect=primary),
            patch.object(
                gaussians_composite_module,
                "_discard_layer_count_gather",
                side_effect=RuntimeError("count drain failed"),
            ) as discard,
            self.assertRaises(RuntimeError) as raised,
        ):
            composite.forward(cast(Any, object()), is_training_batch=True)

        self.assertIs(raised.exception, primary)
        self.assertTrue(any("count drain failed" in note for note in getattr(primary, "__notes__", ())))
        discard.assert_called_once_with(count_token)
        self.assertIsNone(composite._fused_gather_generation_plan)
        self.assertIsNone(composite._pending_fused_gather_generation_plan_refresh)

    def test_prefetch_wrapper_failure_drains_launched_payload_and_invalidates_plan(self) -> None:
        composite, nodes = _make_grouped_gather_composite()
        composite.calib = _make_free_pose_calib_for_prefetch_test()
        selected_nodes = _GENERATION_PLAN_TEST_NODES
        plan = _make_generation_plan(selected_nodes)
        composite._fused_gather_generation_plan = cast(Any, plan)
        prepared = object()
        work = Mock()
        work.wait.side_effect = RuntimeError("payload drain failed")
        payload = gaussians_composite_module._PendingFusedPayloadGather(
            local_grouped_payload=torch.empty(0),
            grouped_gathered_payload=torch.empty(0),
            work=work,
        )
        primary = RuntimeError("prefetch wrapper failed")

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(composite, "_supports_fused_gather_generation_plan", return_value=True),
            patch.object(composite, "get_gaussians_node_ids", return_value=list(selected_nodes)),
            patch.object(
                gaussians_composite_module,
                "_prepare_fused_gather_inputs_from_generation_plan",
                return_value=prepared,
            ),
            patch.object(
                gaussians_composite_module,
                "_start_fused_payload_gather_from_generation_plan",
                return_value=payload,
            ),
            patch.object(gaussians_composite_module, "_PendingGaussianParameterGather", side_effect=primary),
            self.assertRaises(RuntimeError) as raised,
        ):
            composite._maybe_prefetch_gaussian_parameter_gather(
                is_training_batch=True,
                gaussian_nodes=None,
                track_transform_ids=None,
                track_transforms=None,
                edited_cuboid_tracks=None,
            )

        self.assertIs(raised.exception, primary)
        work.wait.assert_called_once_with()
        self.assertEqual(payload.state, "discarded")
        self.assertTrue(any("payload drain failed" in note for note in getattr(primary, "__notes__", ())))
        self.assertIsNone(composite._fused_gather_generation_plan)
        self.assertEqual(tuple(node.get_parameters_calls for node in nodes.values()), (1, 1, 1, 1))

    def test_forward_orders_payload_launch_before_calibration_and_joint_gather(self) -> None:
        composite, _first, _second = _make_gather_composite()
        composite.calib = _make_free_pose_calib_for_prefetch_test()
        events: list[str] = []
        payload_gather = object()
        plan = _make_generation_plan(("first", "second"))
        pending = gaussians_composite_module._PendingGaussianParameterGather(
            gaussian_nodes=("first", "second"),
            parameters_by_node={},
            prepared_inputs=cast(Any, object()),
            generation_plan=cast(Any, plan),
            prefetched_payload_gather=cast(Any, payload_gather),
        )
        data = SimpleNamespace(
            camera=SimpleNamespace(meta=[object()]),
            lidar=SimpleNamespace(meta=[object()]),
        )
        rendering = SimpleNamespace(
            camera=SimpleNamespace(active_tile_mask=None),
            lidar=object(),
        )
        batch = SimpleNamespace(rendering=None, data=data)

        def prefetch(*_args: Any, **_kwargs: Any) -> Any:
            events.append("payload-launch")
            return pending

        def calibrate(*_args: Any, **_kwargs: Any) -> Any:
            events.append("calibration")
            return rendering

        def forward_gaussians(*_args: Any, **kwargs: Any) -> object:
            events.append("joint-gather")
            self.assertIs(kwargs["prefetched_parameter_gather"], pending)
            return object()

        composite.calib.forward = Mock(side_effect=calibrate)  # type: ignore[method-assign]
        with (
            patch.object(composite, "_maybe_prefetch_gaussian_parameter_gather", side_effect=prefetch),
            patch.object(composite, "forward_gaussians", side_effect=forward_gaussians),
            patch.object(
                gaussians_composite_module,
                "_discard_fused_payload_gather",
                side_effect=lambda _pending: events.append("discard"),
            ),
        ):
            composite.forward(
                batch,  # type: ignore[arg-type]
                is_training_batch=True,
                render_background=False,
                enable_pp=False,
                clamp_output=False,
            )

        self.assertEqual(events, ["payload-launch", "calibration", "joint-gather", "discard"])

    def test_eligible_gather_uses_fused_result_and_preserves_node_order(self) -> None:
        composite, first, second = _make_gather_composite()
        fused_parameters = {
            "first": {"positions": torch.full((3, 3), 1.0)},
            "second": {"positions": torch.full((3, 3), 2.0)},
        }
        fused_counts: dict[str, tuple[int, ...]] = {"first": (2, 1), "second": (1, 2)}
        fused_result = _FusedGatherResult(
            parameters_by_node=fused_parameters,
            counts_by_node=fused_counts,
        )

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                return_value=fused_result,
            ) as fused,
            patch.object(gaussians_composite_module, "distributed_all_gather_gaussian_parameters") as baseline,
        ):
            result = composite._gather_gaussian_parameters_once(["first", "second"])

        self.assertEqual(list(result), ["first", "second"])
        self.assertIs(result["first"].parameters, fused_parameters["first"])
        self.assertIs(result["second"].parameters, fused_parameters["second"])
        self.assertEqual((result["first"].local_count, result["second"].local_count), (2, 1))
        self.assertEqual(result["first"].counts_by_rank, (2, 1))
        self.assertEqual(result["second"].counts_by_rank, (1, 2))
        self.assertEqual((first.get_parameters_calls, second.get_parameters_calls), (1, 1))
        fused.assert_called_once_with(
            ["first", "second"],
            {"first": first.parameters, "second": second.parameters},
            process_group=composite.gaussian_process_group,
            integer_range_validation_state=mock.ANY,
            prepared_inputs=None,
            generation_plan=None,
            prefetched_payload_gather=None,
        )
        baseline.assert_not_called()

    def test_prefetched_gather_reuses_the_exact_captured_parameters(self) -> None:
        composite, first, second = _make_gather_composite()
        parameters_by_node: dict[str, Mapping[str, torch.Tensor]] = {
            "first": first.get_parameters(),
            "second": second.get_parameters(),
        }
        prepared_inputs = object()
        generation_plan = _make_generation_plan(("first", "second"))
        prefetched_payload_gather = object()
        pending = gaussians_composite_module._PendingGaussianParameterGather(
            gaussian_nodes=("first", "second"),
            parameters_by_node=parameters_by_node,
            prepared_inputs=cast(Any, prepared_inputs),
            generation_plan=cast(Any, generation_plan),
            prefetched_payload_gather=cast(Any, prefetched_payload_gather),
        )
        fused_result = _FusedGatherResult(
            parameters_by_node={
                "first": {"positions": torch.empty(3, 3)},
                "second": {"positions": torch.empty(3, 3)},
            },
            counts_by_node={"first": (2, 1), "second": (1, 2)},
        )

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                return_value=fused_result,
            ) as fused,
        ):
            composite._gather_gaussian_parameters_once(
                ["first", "second"],
                prefetched=pending,
            )

        self.assertEqual((first.get_parameters_calls, second.get_parameters_calls), (1, 1))
        fused.assert_called_once_with(
            ["first", "second"],
            parameters_by_node,
            process_group=composite.gaussian_process_group,
            integer_range_validation_state=mock.ANY,
            prepared_inputs=prepared_inputs,
            generation_plan=generation_plan,
            prefetched_payload_gather=prefetched_payload_gather,
        )

    def test_fused_error_propagates_without_per_node_fallback(self) -> None:
        composite, first, second = _make_gather_composite()

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                side_effect=RuntimeError("CUDA pack failed"),
            ) as fused,
            patch.object(gaussians_composite_module, "distributed_all_gather_gaussian_parameters") as baseline,
            self.assertRaisesRegex(RuntimeError, "CUDA pack failed"),
        ):
            composite._gather_gaussian_parameters_once(["first", "second"])

        self.assertEqual((first.get_parameters_calls, second.get_parameters_calls), (1, 1))
        fused.assert_called_once()
        baseline.assert_not_called()

    def test_single_node_uses_its_single_gather(self) -> None:
        composite, first, _second = _make_gather_composite()
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(gaussians_composite_module, "_fused_all_gather_gaussian_parameters") as fused,
            patch.object(
                gaussians_composite_module,
                "distributed_all_gather_gaussian_parameters",
                side_effect=lambda parameters, *, group: dict(parameters),
            ) as baseline,
        ):
            result = composite._gather_gaussian_parameters_once(["first"])

        self.assertEqual(list(result), ["first"])
        self.assertEqual(first.get_parameters_calls, 1)
        self.assertIs(result["first"].parameters["positions"], first.parameters["positions"])
        self.assertEqual(result["first"].local_count, 2)
        self.assertIsNone(result["first"].counts_by_rank)
        baseline.assert_called_once_with(first.parameters, group=composite.gaussian_process_group)
        fused.assert_not_called()

    def test_single_node_cached_generation_uses_grouped_fused_gather(self) -> None:
        composite, first, _second = _make_gather_composite()
        selected_nodes = ("first",)
        fused_parameters_by_node = {"first": first.parameters}
        parameters_by_node: dict[str, Mapping[str, torch.Tensor]] = {
            "first": cast(Mapping[str, torch.Tensor], first.parameters)
        }
        prepared_inputs = object()
        generation_plan = _make_generation_plan(selected_nodes)
        prefetched_payload_gather = object()
        pending = gaussians_composite_module._PendingGaussianParameterGather(
            gaussian_nodes=selected_nodes,
            parameters_by_node=parameters_by_node,
            prepared_inputs=cast(Any, prepared_inputs),
            generation_plan=cast(Any, generation_plan),
            prefetched_payload_gather=cast(Any, prefetched_payload_gather),
        )
        fused_result = _FusedGatherResult(
            parameters_by_node=fused_parameters_by_node,
            counts_by_node={"first": (2, 2)},
        )

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                return_value=fused_result,
            ) as fused,
            patch.object(gaussians_composite_module, "distributed_all_gather_gaussian_parameters") as baseline,
        ):
            result = composite._gather_gaussian_parameters_once(["first"], prefetched=pending)

        self.assertEqual(tuple(result), selected_nodes)
        self.assertEqual(result["first"].counts_by_rank, (2, 2))
        fused.assert_called_once_with(
            ["first"],
            parameters_by_node,
            process_group=composite.gaussian_process_group,
            integer_range_validation_state=mock.ANY,
            prepared_inputs=prepared_inputs,
            generation_plan=generation_plan,
            prefetched_payload_gather=prefetched_payload_gather,
        )
        baseline.assert_not_called()

    def test_visibility_counts_reuse_metadata_without_collective(self) -> None:
        composite, _first, _second = _make_gather_composite()

        with patch.object(gaussians_composite_module, "all_gather_int32") as gather_counts:
            counts = composite._resolve_visibility_counts(
                node_id="first",
                world_size=2,
                world_rank=0,
                local_count=2,
                device=torch.device("cuda"),
                gathered_counts=(2, 1),
            )

        self.assertEqual(counts, (2, 1))
        gather_counts.assert_not_called()

    def test_scene_data_compaction_concatenates_local_layer_ranges(self) -> None:
        composite, _first, _second = _make_gather_composite()
        composite._visibility_local_ranges = ((2, 2), (7, 1), (8, 0))
        visibility = torch.arange(8)
        cumulated_weights = torch.arange(16).reshape(8, 2)
        out = GaussiansRenderReturn(
            opacity=torch.zeros((8, 1)),
            visibility=visibility,
            cumulated_weights=cumulated_weights,
        )

        composite._compact_scene_data_to_local(out)

        assert out.visibility is not None
        assert out.cumulated_weights is not None
        torch.testing.assert_close(out.visibility, visibility[torch.tensor([2, 3, 7])])
        torch.testing.assert_close(out.cumulated_weights, cumulated_weights[torch.tensor([2, 3, 7])])
        self.assertTrue(out.visibility.is_contiguous())
        self.assertTrue(out.cumulated_weights.is_contiguous())

    def test_scene_data_compaction_returns_single_layer_view_without_copy(self) -> None:
        composite, _first, _second = _make_gather_composite()
        composite._visibility_local_ranges = ((2, 3),)
        visibility = torch.arange(8)
        out = GaussiansRenderReturn(opacity=torch.zeros((8, 1)), visibility=visibility)

        composite._compact_scene_data_to_local(out)

        assert out.visibility is not None
        self.assertEqual(out.visibility.untyped_storage().data_ptr(), visibility.untyped_storage().data_ptr())
        self.assertEqual(out.visibility.storage_offset(), 2)
        torch.testing.assert_close(out.visibility, visibility[2:5])

    def test_visibility_ranges_reuse_fused_counts_without_a_collective(self) -> None:
        composite, _first, _second = _make_gather_composite()
        empty = _FakeGatherNode(0)
        composite.gaussians_nodes["empty"] = cast(BaseGaussianModel, empty)
        gaussian_nodes = ["first", "second", "empty"]
        fused_result = _FusedGatherResult(
            parameters_by_node={
                "first": {"positions": torch.empty((4, 3))},
                "second": {"positions": torch.empty((4, 3))},
                "empty": {"positions": torch.empty((0, 3))},
            },
            counts_by_node={
                "first": (2, 2),
                "second": (3, 1),
                "empty": (0, 0),
            },
        )
        composite.debug_viz = False
        composite._visibility_local_ranges = None
        composite.collector = MagicMock()
        composite.collector.collect.return_value = collect.CollectorResult(
            positions=torch.empty((8, 3)),
            rotations=torch.empty((8, 4)),
            scales=torch.empty((8, 3)),
            densities=torch.empty((8, 1)),
            extra_signal=torch.empty((8, 0)),
            camera_extra_signal=torch.empty((8, 0)),
            lidar_extra_signal=torch.empty((8, 0)),
            features=torch.empty((8, 0)),
        )
        rendering_data = cast(
            RenderingData,
            SimpleNamespace(
                timestamps_startend_us_cpu=torch.zeros((1, 2), dtype=torch.int64),
            ),
        )

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=1),
            patch.object(composite, "get_gaussians_node_ids", return_value=gaussian_nodes),
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                return_value=fused_result,
            ),
            patch.object(gaussians_composite_module, "all_gather_int32") as gather_counts,
        ):
            gathered = composite._gather_gaussian_parameters_once(gaussian_nodes)
            composite.collect_gaussian_parameters(
                rendering_data,
                is_training_batch=True,
                gaussian_nodes=gaussian_nodes,
                gathered_parameters=gathered,
            )

        gather_counts.assert_not_called()
        self.assertEqual(composite._visibility_local_ranges, ((2, 2), (7, 1), (8, 0)))

    def test_visibility_ranges_reject_counts_that_do_not_cover_layer_rows(self) -> None:
        composite, _first, _second = _make_gather_composite()
        gaussian_nodes = ["first", "second"]
        fused_result = _FusedGatherResult(
            parameters_by_node={
                "first": {"positions": torch.empty((3, 3))},
                "second": {"positions": torch.empty((2, 3))},
            },
            counts_by_node={
                "first": (2, 2),
                "second": (1, 1),
            },
        )
        composite.debug_viz = False
        composite._visibility_local_ranges = None
        composite.collector = MagicMock()
        rendering_data = cast(
            RenderingData,
            SimpleNamespace(
                timestamps_startend_us_cpu=torch.zeros((1, 2), dtype=torch.int64),
            ),
        )

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(composite, "get_gaussians_node_ids", return_value=gaussian_nodes),
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                return_value=fused_result,
            ),
            patch.object(gaussians_composite_module, "all_gather_int32") as gather_counts,
            self.assertRaisesRegex(
                RuntimeError,
                "Gathered count mismatch for first: total count 4 != layer total 3",
            ),
        ):
            gathered = composite._gather_gaussian_parameters_once(gaussian_nodes)
            composite.collect_gaussian_parameters(
                rendering_data,
                is_training_batch=True,
                gaussian_nodes=gaussian_nodes,
                gathered_parameters=gathered,
            )

        gather_counts.assert_not_called()
        composite.collector.collect.assert_not_called()
        self.assertIsNone(composite._visibility_local_ranges)

    def test_edited_track_inference_does_not_reuse_prefilter_visibility_counts(self) -> None:
        filtering = _FilteringFakeGatherNode(2)
        unchanged = _FakeGatherNode(1)
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite.gaussian_process_group = cast(Any, object())
        composite.gaussians_nodes = {"filtering": filtering, "unchanged": unchanged}  # type: ignore[assignment]
        gaussian_nodes = ["filtering", "unchanged"]
        fused_result = _FusedGatherResult(
            parameters_by_node={
                "filtering": {"positions": torch.empty((4, 3))},
                "unchanged": {"positions": torch.empty((2, 3))},
            },
            counts_by_node={
                "filtering": (2, 2),
                "unchanged": (1, 1),
            },
        )
        composite.debug_viz = False
        composite._visibility_local_ranges = None
        composite.collector = MagicMock()
        composite.collector.collect.return_value = collect.CollectorResult(
            positions=torch.empty((3, 3)),
            rotations=torch.empty((3, 4)),
            scales=torch.empty((3, 3)),
            densities=torch.empty((3, 1)),
            extra_signal=torch.empty((3, 0)),
            camera_extra_signal=torch.empty((3, 0)),
            lidar_extra_signal=torch.empty((3, 0)),
            features=torch.empty((3, 0)),
        )
        rendering_data = cast(
            RenderingData,
            SimpleNamespace(
                timestamps_startend_us_cpu=torch.zeros((1, 2), dtype=torch.int64),
            ),
        )

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(composite, "get_gaussians_node_ids", return_value=gaussian_nodes),
            patch.object(gaussians_composite_module, "RigidGaussianModel", _FilteringFakeGatherNode),
            patch.object(
                gaussians_composite_module,
                "_fused_all_gather_gaussian_parameters",
                return_value=fused_result,
            ),
            patch.object(gaussians_composite_module, "all_gather_int32") as gather_counts,
        ):
            gathered = composite._gather_gaussian_parameters_once(gaussian_nodes)
            composite.collect_gaussian_parameters(
                rendering_data,
                edited_cuboid_tracks=cast(CuboidTracks, object()),
                gaussian_nodes=gaussian_nodes,
                gathered_parameters=gathered,
            )

        gather_counts.assert_not_called()
        self.assertIsNone(composite._visibility_local_ranges)
        layers_data = composite.collector.collect.call_args.args[0]
        self.assertEqual(layers_data.layers[0].positions.shape[0], 1)
        self.assertEqual(layers_data.layers[1].positions.shape[0], 2)

    def test_visibility_counts_without_metadata_keeps_collective(self) -> None:
        composite, _first, _second = _make_gather_composite()
        group = object()
        composite.gaussian_process_group = cast(Any, group)

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(gaussians_composite_module, "all_gather_int32", return_value=[2, 1]) as gather_counts,
        ):
            counts = composite._resolve_visibility_counts(
                node_id="first",
                world_size=2,
                world_rank=0,
                local_count=2,
                device=torch.device("cuda"),
                gathered_counts=None,
            )

        self.assertEqual(counts, (2, 1))
        gather_counts.assert_called_once_with(2, 2, device=torch.device("cuda"), group=group)

    def test_visibility_count_mismatch_fails_closed(self) -> None:
        composite, _first, _second = _make_gather_composite()

        with self.assertRaisesRegex(RuntimeError, "rank 0 count 1 != local count 2"):
            composite._resolve_visibility_counts(
                node_id="first",
                world_size=2,
                world_rank=0,
                local_count=2,
                device=torch.device("cuda"),
                gathered_counts=(1, 2),
            )

        with self.assertRaisesRegex(RuntimeError, "Expected 2 gathered counts for first, got 1"):
            composite._resolve_visibility_counts(
                node_id="first",
                world_size=2,
                world_rank=0,
                local_count=2,
                device=torch.device("cuda"),
                gathered_counts=(2,),
            )

    def test_duplicate_nodes_fail_closed(self) -> None:
        composite, first, _second = _make_gather_composite()
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(gaussians_composite_module, "_fused_all_gather_gaussian_parameters") as fused,
            patch.object(gaussians_composite_module, "distributed_all_gather_gaussian_parameters") as baseline,
            self.assertRaisesRegex(RuntimeError, "requires distinct Gaussian nodes"),
        ):
            composite._gather_gaussian_parameters_once(["first", "first"])

        self.assertEqual(first.get_parameters_calls, 0)
        baseline.assert_not_called()
        fused.assert_not_called()

    def test_every_scene_level_field_is_compacted_to_the_local_shard(self) -> None:
        # Under multi-GPU the renderer sees all-gathered gaussians, so every per-gaussian
        # field comes back at global length. Leaving one uncompacted desynchronizes it from
        # `visibility` downstream (the MCMC invisible-steps counters compare element counts
        # and silently fall back when they disagree).
        composite, _first, _second = _make_gather_composite()
        composite._visibility_local_ranges = ((1, 1), (3, 1))
        out = GaussiansRenderReturn(
            opacity=torch.zeros((2, 1)),
            visibility=torch.tensor([False, True, False, True]),
            frustum_visibility=torch.tensor([True, True, False, True]),
            cumulated_weights=torch.arange(4, dtype=torch.float32),
        )

        composite._compact_scene_data_to_local(out)

        for name in GaussiansRenderReturn._SCENE_LEVEL_FIELDS:
            self.assertEqual(getattr(out, name).shape, (2,), msg=name)
        torch.testing.assert_close(out.visibility, torch.tensor([True, True]))
        torch.testing.assert_close(out.frustum_visibility, torch.tensor([True, True]))
        torch.testing.assert_close(out.cumulated_weights, torch.tensor([1.0, 3.0]))

    def test_unset_scene_level_fields_stay_none(self) -> None:
        composite, _first, _second = _make_gather_composite()
        composite._visibility_local_ranges = ((0, 1), (2, 1))
        out = GaussiansRenderReturn(
            opacity=torch.zeros((2, 1)),
            visibility=torch.tensor([True, False, True, False]),
        )

        composite._compact_scene_data_to_local(out)

        torch.testing.assert_close(out.visibility, torch.tensor([True, True]))
        self.assertIsNone(out.frustum_visibility)
        self.assertIsNone(out.cumulated_weights)


class _IdentityTracksCalib(BaseTracksCalib):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, tracks: CuboidTracks) -> CuboidTracks:
        return tracks

    __call__ = module_call_type(forward)


class _FirstReadSynchronizedComposite(GaussiansComposite):
    """Make every test caller observe the same pre-initialization value."""

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name != "_lidar_stream":
            return value

        count_lock = super().__getattribute__("_lidar_first_read_count_lock")
        with count_lock:
            count = super().__getattribute__("_lidar_first_read_count") + 1
            object.__setattr__(self, "_lidar_first_read_count", count)
        barrier = super().__getattribute__("_lidar_first_read_barrier")
        if count <= barrier.parties:
            barrier.wait(timeout=5)
        return value


def _make_cuboid_tracks(
    track_ids: list[str], label_classes: list[str], *, device: torch.device = TEST_DEVICE
) -> CuboidTracks:
    identity_poses = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
    timestamps = np.array([0, 1], dtype=np.int64)
    return CuboidTracks.Factory.from_numpy(
        tracks_id=track_ids,
        tracks_poses=[identity_poses.copy() for _ in track_ids],
        tracks_timestamps_us=[timestamps.copy() for _ in track_ids],
        tracks_label_class=label_classes,
        tracks_flags=[TrackFlags.DYNAMIC | TrackFlags.CONTROLLABLE for _ in track_ids],
        cuboids_dims=[np.ones(3, dtype=np.float32) for _ in track_ids],
        device=device,
    )


class _FakeRigidGaussianModel(RigidGaussianModel):
    def __init__(self, cuboid_tracks: CuboidTracks) -> None:
        torch.nn.Module.__init__(self)
        self._motion_source_ref = None
        self.cuboid_tracks = cuboid_tracks
        self.is_static = False
        self.tracks_calib = _IdentityTracksCalib()
        self.time_embed = None
        self.insert_calls: list[tuple[str, object, CuboidTracks, torch.Tensor, RGBPostProcessing | None]] = []
        self.updated_tracks: list[tuple[CuboidTracks, set[str]]] = []
        self.save_training_parameters_calls = 0

    def prepare_asset_insertion(
        self,
        track_id: str,
        asset: object,
        cuboid_tracks: CuboidTracks,
        dims_offset: torch.Tensor,
        rgb_post_processing: RGBPostProcessing | None = None,
        *,
        target_cuboid_tracks: CuboidTracks | None = None,
        inserted_track_ids: set[str] | None = None,
    ) -> Any:
        """Stage the fake's post-insert metadata without changing its live state."""
        assert target_cuboid_tracks is not None
        assert inserted_track_ids is not None
        self.updated_tracks.append((target_cuboid_tracks, inserted_track_ids))
        # Emulate the production rebuild: a fresh time_embed whose per-instance table covers the combined
        # track count (one valid [min, max) row per track), so the composite's post-insert collector
        # refresh sees the grown table exactly as it would from BaseInputEmbedding.factory.
        n_tracks = len(target_cuboid_tracks.tracks_id)
        time_embed = SimpleNamespace(
            timestamps_us_ranges=torch.stack(
                [torch.zeros(n_tracks, dtype=torch.int64), torch.full((n_tracks,), 1000, dtype=torch.int64)],
                dim=1,
            )
        )
        return SimpleNamespace(
            track_id=track_id,
            asset=asset,
            inserted_cuboid_tracks=cuboid_tracks,
            dims_offset=dims_offset,
            rgb_post_processing=rgb_post_processing,
            cuboid_tracks=target_cuboid_tracks,
            tracks_calib=_IdentityTracksCalib(),
            time_embed=time_embed,
        )

    def apply_prepared_asset_insertion(
        self,
        prepared: Any,
        *,
        save_training_parameters: bool = True,
    ) -> None:
        """Publish staged fake state and retain the legacy call record for assertions."""
        if save_training_parameters:
            self.save_training_parameters()
        self.cuboid_tracks = prepared.cuboid_tracks
        self.tracks_calib = prepared.tracks_calib
        self.time_embed = prepared.time_embed
        self.insert_calls.append(
            (
                prepared.track_id,
                prepared.asset,
                prepared.inserted_cuboid_tracks,
                prepared.dims_offset,
                prepared.rgb_post_processing,
            )
        )

    def save_training_parameters(self) -> None:
        """Record the composite's all-layers-before-publish snapshot phase."""
        self.save_training_parameters_calls += 1


@unittest.skipUnless(torch.cuda.is_available(), "CuboidTracks subset operations require CUDA packed ops")
class TestGaussiansCompositeInsertAsset(unittest.TestCase):
    def _make_composite(self) -> tuple[GaussiansComposite, _FakeRigidGaussianModel, _FakeRigidGaussianModel]:
        existing_tracks = _make_cuboid_tracks(["car-1", "person-1"], ["automobile", "pedestrian"])
        dynamic_rigids_config = OmegaConf.create({"is_dynamic": True, "label_classes": ["automobile"]})
        dynamic_deformables_config = OmegaConf.create({"is_dynamic": True, "label_classes": ["pedestrian"]})

        dynamic_rigids_track_ids = LayerTrackIds(config=dynamic_rigids_config)
        dynamic_rigids_track_ids.initialize_from_tracks(existing_tracks)
        dynamic_deformables_track_ids = LayerTrackIds(config=dynamic_deformables_config)
        dynamic_deformables_track_ids.initialize_from_tracks(existing_tracks)

        dynamic_rigids = _FakeRigidGaussianModel(CuboidTracks.Ops.subset_from_tracks_id(existing_tracks, ["car-1"]))
        dynamic_deformables = _FakeRigidGaussianModel(
            CuboidTracks.Ops.subset_from_tracks_id(existing_tracks, ["person-1"])
        )

        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite._fused_gather_generation_plan = None
        composite._pending_fused_gather_generation_plan_refresh = None
        composite.cuboid_tracks = existing_tracks
        composite.obj_track_ids = {
            "dynamic_rigids": dynamic_rigids_track_ids,
            "dynamic_deformables": dynamic_deformables_track_ids,
        }
        composite.gaussians_nodes = TypedModuleDict[BaseGaussianModel](
            {
                "dynamic_rigids": dynamic_rigids,
                "dynamic_deformables": dynamic_deformables,
            }
        )
        # Minimal collector stand-in for the post-insert time-table refresh: one layer-config slot per
        # node (same order as gaussians_nodes). Plain namespaces carry no embed_config -> refresh no-ops;
        # tests that need the per-instance path install a real IndividualRemap config into slot 0.
        composite.collector = cast(
            Any, SimpleNamespace(layers_config=SimpleNamespace(layers=[SimpleNamespace(), SimpleNamespace()]))
        )
        return composite, dynamic_rigids, dynamic_deformables

    def test_unrecognized_insert_label_falls_back_to_dynamic_rigids(self):
        composite, dynamic_rigids, dynamic_deformables = self._make_composite()
        inserted_tracks = _make_cuboid_tracks(["cone-1"], ["other"])

        inserted = composite.insert_asset(object(), inserted_tracks, torch.zeros(3, device=TEST_DEVICE))

        self.assertTrue(inserted)
        self.assertEqual([call[0] for call in dynamic_rigids.insert_calls], ["cone-1"])
        self.assertEqual(dynamic_deformables.insert_calls, [])
        self.assertEqual(dynamic_rigids.cuboid_tracks.tracks_id, ["car-1", "cone-1"])
        self.assertEqual(dynamic_rigids.cuboid_tracks.tracks_label_class, ["automobile", "other"])
        self.assertEqual(composite.cuboid_tracks.tracks_id, ["car-1", "person-1", "cone-1"])
        self.assertIn("cone-1", composite.get_updated_cuboid_tracks().tracks_id)

    def test_repeated_unrecognized_insert_label_uses_dynamic_rigids_fallback(self):
        composite, dynamic_rigids, dynamic_deformables = self._make_composite()

        composite.insert_asset(object(), _make_cuboid_tracks(["cone-1"], ["other"]), torch.zeros(3, device=TEST_DEVICE))
        inserted = composite.insert_asset(
            object(), _make_cuboid_tracks(["cone-2"], ["other"]), torch.zeros(3, device=TEST_DEVICE)
        )

        self.assertTrue(inserted)
        self.assertEqual([call[0] for call in dynamic_rigids.insert_calls], ["cone-1", "cone-2"])
        self.assertEqual(dynamic_deformables.insert_calls, [])
        self.assertEqual(dynamic_rigids.cuboid_tracks.tracks_id, ["car-1", "cone-1", "cone-2"])

    def test_recognized_insert_label_does_not_use_dynamic_rigids_fallback(self):
        composite, dynamic_rigids, dynamic_deformables = self._make_composite()
        inserted_tracks = _make_cuboid_tracks(["person-2"], ["pedestrian"])

        inserted = composite.insert_asset(object(), inserted_tracks, torch.zeros(3, device=TEST_DEVICE))

        self.assertTrue(inserted)
        self.assertEqual(dynamic_rigids.insert_calls, [])
        self.assertEqual([call[0] for call in dynamic_deformables.insert_calls], ["person-2"])
        self.assertEqual(dynamic_deformables.cuboid_tracks.tracks_id, ["person-1", "person-2"])

    def test_insert_rejection_happens_before_track_or_scene_mutation(self):
        """A rejected insertion leaves composite and layer track state untouched."""

        composite, dynamic_rigids, dynamic_deformables = self._make_composite()
        original_scene_tracks = composite.cuboid_tracks
        original_layer_tracks = dynamic_deformables.cuboid_tracks
        dynamic_deformables.validate_external_gaussian_topology_edit = mock.MagicMock(
            side_effect=NotImplementedError("asset insertion is unsupported")
        )

        with self.assertRaisesRegex(NotImplementedError, "unsupported"):
            composite.insert_asset(
                object(),
                _make_cuboid_tracks(["person-2"], ["pedestrian"]),
                torch.zeros(3, device=TEST_DEVICE),
            )

        self.assertIs(composite.cuboid_tracks, original_scene_tracks)
        self.assertIs(dynamic_deformables.cuboid_tracks, original_layer_tracks)
        self.assertEqual(dynamic_rigids.updated_tracks, [])
        self.assertEqual(dynamic_deformables.updated_tracks, [])
        self.assertEqual(dynamic_rigids.insert_calls, [])
        self.assertEqual(dynamic_deformables.insert_calls, [])

    def test_insert_invalidates_before_layer_mutation(self):
        """A planned insertion invalidates cached gathers before changing its target layer."""

        composite, dynamic_rigids, dynamic_deformables = self._make_composite()
        original_scene_tracks = composite.cuboid_tracks
        original_layer_tracks = dynamic_deformables.cuboid_tracks

        with (
            mock.patch.object(composite, "_invalidate_collect_caches") as collect_invalidation,
            mock.patch.object(
                composite,
                "_invalidate_fused_gather_generation_plan",
                side_effect=RuntimeError("invalidation sentinel"),
            ) as fused_invalidation,
            self.assertRaisesRegex(RuntimeError, "invalidation sentinel"),
        ):
            composite.insert_asset(
                object(),
                _make_cuboid_tracks(["person-2"], ["pedestrian"]),
                torch.zeros(3, device=TEST_DEVICE),
            )

        fused_invalidation.assert_called_once_with()
        collect_invalidation.assert_not_called()
        self.assertIs(composite.cuboid_tracks, original_scene_tracks)
        self.assertIs(dynamic_deformables.cuboid_tracks, original_layer_tracks)
        self.assertEqual(dynamic_rigids.updated_tracks, [])
        self.assertEqual(len(dynamic_deformables.updated_tracks), 1)
        self.assertEqual(dynamic_rigids.insert_calls, [])
        self.assertEqual(dynamic_deformables.insert_calls, [])

    def test_insert_preparation_exception_does_not_change_layer_or_collector_metadata(self):
        """A preparation failure occurs before live layer or collector state is published."""

        composite, _, dynamic_deformables = self._make_composite()
        original_scene_tracks = composite.cuboid_tracks
        original_obj_track_ids = composite.obj_track_ids["dynamic_deformables"]
        original_layer_tracks = dynamic_deformables.cuboid_tracks
        original_tracks_calib = dynamic_deformables.tracks_calib
        original_time_table = torch.tensor([[0, 1000]], dtype=torch.int64)
        original_time_embed = SimpleNamespace(timestamps_us_ranges=original_time_table)
        dynamic_deformables.time_embed = cast(Any, original_time_embed)
        collector_embed_config = collect.IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=original_time_table,
            remap_min=0.0,
            remap_max=1.0,
        )
        composite.collector.layers_config.layers[1] = SimpleNamespace(embed_config=collector_embed_config)
        dynamic_deformables.prepare_asset_insertion = mock.MagicMock(side_effect=RuntimeError("insertion failed"))

        with self.assertRaisesRegex(RuntimeError, "insertion failed"):
            composite.insert_asset(
                object(),
                _make_cuboid_tracks(["person-2"], ["pedestrian"]),
                torch.zeros(3, device=TEST_DEVICE),
            )

        self.assertIs(composite.cuboid_tracks, original_scene_tracks)
        self.assertIs(composite.obj_track_ids["dynamic_deformables"], original_obj_track_ids)
        self.assertIs(dynamic_deformables.cuboid_tracks, original_layer_tracks)
        self.assertIs(dynamic_deformables.tracks_calib, original_tracks_calib)
        self.assertIs(dynamic_deformables.time_embed, original_time_embed)
        self.assertIs(collector_embed_config.timestamps_us_ranges, original_time_table)
        self.assertEqual(dynamic_deformables.save_training_parameters_calls, 0)
        self.assertEqual(dynamic_deformables.insert_calls, [])

    def test_second_layer_preparation_failure_does_not_publish_first_layer(self):
        """A later target failure leaves every earlier target and global track table untouched."""
        composite, dynamic_rigids, dynamic_deformables = self._make_composite()
        composite.obj_track_ids["dynamic_deformables"].config.label_classes.append("automobile")
        original_scene_tracks = composite.cuboid_tracks
        original_rigid_tracks = dynamic_rigids.cuboid_tracks
        original_deformable_tracks = dynamic_deformables.cuboid_tracks
        original_rigid_obj_track_ids = composite.obj_track_ids["dynamic_rigids"]
        original_deformable_obj_track_ids = composite.obj_track_ids["dynamic_deformables"]
        dynamic_deformables.prepare_asset_insertion = mock.MagicMock(side_effect=RuntimeError("second layer failed"))

        with (
            mock.patch.object(composite, "_invalidate_fused_gather_generation_plan") as fused_invalidation,
            mock.patch.object(composite, "_invalidate_collect_caches") as collect_invalidation,
            self.assertRaisesRegex(RuntimeError, "second layer failed"),
        ):
            composite.insert_asset(
                object(),
                _make_cuboid_tracks(["car-2"], ["automobile"]),
                torch.zeros(3, device=TEST_DEVICE),
            )

        self.assertEqual(len(dynamic_rigids.updated_tracks), 1)
        self.assertEqual(dynamic_rigids.insert_calls, [])
        self.assertEqual(dynamic_deformables.insert_calls, [])
        self.assertEqual(dynamic_rigids.save_training_parameters_calls, 0)
        self.assertEqual(dynamic_deformables.save_training_parameters_calls, 0)
        self.assertIs(dynamic_rigids.cuboid_tracks, original_rigid_tracks)
        self.assertIs(dynamic_deformables.cuboid_tracks, original_deformable_tracks)
        self.assertIs(composite.cuboid_tracks, original_scene_tracks)
        self.assertIs(composite.obj_track_ids["dynamic_rigids"], original_rigid_obj_track_ids)
        self.assertIs(composite.obj_track_ids["dynamic_deformables"], original_deformable_obj_track_ids)
        fused_invalidation.assert_not_called()
        collect_invalidation.assert_not_called()

    def test_insert_asset_forwards_rgb_post_processing(self):
        composite, dynamic_rigids, dynamic_deformables = self._make_composite()
        inserted_tracks = _make_cuboid_tracks(["cone-1"], ["other"])

        def rgb_post_processing(rgb: torch.Tensor) -> torch.Tensor:
            return rgb

        inserted = composite.insert_asset(
            object(),
            inserted_tracks,
            torch.zeros(3, device=TEST_DEVICE),
            rgb_post_processing=rgb_post_processing,
        )

        self.assertTrue(inserted)
        self.assertEqual(dynamic_deformables.insert_calls, [])
        self.assertIs(dynamic_rigids.insert_calls[0][4], rgb_post_processing)

    def test_insert_refreshes_collector_per_instance_time_table(self):
        """Insert must re-point the collector's captured per-instance table at the node's rebuilt one.

        The collector handler reads the embed_config captured at composite construction; without the
        refresh, the inserted track's instance_idx gathers past that frozen table (the kernel has no
        bounds check) — the review Finding #1 silent-garbage/OOB mode."""
        composite, dynamic_rigids, _ = self._make_composite()
        stale_table = torch.tensor([[0, 1000]], dtype=torch.int64)  # 1 row: pre-insert track count
        embed_config = collect.IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=stale_table, remap_min=0.0, remap_max=1.0
        )
        composite.collector.layers_config.layers[0] = SimpleNamespace(embed_config=embed_config)

        inserted = composite.insert_asset(
            object(), _make_cuboid_tracks(["car-2"], ["automobile"]), torch.zeros(3, device=TEST_DEVICE)
        )

        self.assertTrue(inserted)
        n_tracks = len(dynamic_rigids.cuboid_tracks.tracks_id)
        self.assertEqual(n_tracks, 2)  # car-1 + the insert
        # The handler-visible table is the rebuilt one: covers every live instance index, valid rows.
        self.assertEqual(embed_config.timestamps_us_ranges.shape, (n_tracks, 2))
        self.assertIsNot(embed_config.timestamps_us_ranges, stale_table)
        self.assertTrue(bool((embed_config.timestamps_us_ranges[:, 0] < embed_config.timestamps_us_ranges[:, 1]).all()))


class TestResolveLidarStream(unittest.TestCase):
    """Lidar stream is threaded in (serving) or lazily owned (training)."""

    def _bare_composite(self) -> GaussiansComposite:
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite._lidar_stream = None
        return composite

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA streams require a GPU")
    def test_passed_stream_is_used_and_no_fallback_created(self):
        composite = self._bare_composite()
        passed = torch.cuda.Stream()
        # Serving path: the context-owned stream is used verbatim...
        self.assertIs(composite._resolve_lidar_stream(passed), passed)
        # ...and the composite does not create its own fallback.
        self.assertIsNone(composite._lidar_stream)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA streams require a GPU")
    def test_fallback_stream_created_lazily_and_cached(self):
        composite = self._bare_composite()
        # Training/standalone path: a composite-owned stream is created on first use...
        fallback = composite._resolve_lidar_stream(None)
        self.assertIsNotNone(fallback)
        self.assertIs(composite._lidar_stream, fallback)
        # ...and reused on subsequent calls.
        self.assertIs(composite._resolve_lidar_stream(None), fallback)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA streams require a GPU")
    def test_fallback_stream_preserves_default_lidar_default_ordering(self):
        composite = self._bare_composite()
        current_stream = torch.cuda.current_stream()
        fallback_stream = composite._resolve_lidar_stream(None)
        source = torch.arange(1_000_000, device="cuda")

        # Match forward_gaussians: lidar waits for inputs on the current stream,
        # then the current stream waits before exposing the lidar result.
        fallback_stream.wait_stream(current_stream)
        with torch.cuda.stream(fallback_stream):
            result = source.square()
        current_stream.wait_stream(fallback_stream)

        self.assertTrue(torch.equal(result, source.square()))

    def test_concurrent_fallback_initialization_creates_one_stream(self):
        thread_count = 8
        composite = _FirstReadSynchronizedComposite.__new__(_FirstReadSynchronizedComposite)
        torch.nn.Module.__init__(composite)
        composite._lidar_stream = None
        composite._lidar_first_read_count = 0
        composite._lidar_first_read_count_lock = threading.Lock()
        composite._lidar_first_read_barrier = threading.Barrier(thread_count)
        fallback = object()
        constructor_calls = 0
        constructor_calls_lock = threading.Lock()

        def make_stream() -> object:
            nonlocal constructor_calls
            with constructor_calls_lock:
                constructor_calls += 1
            return fallback

        def resolve() -> object:
            return composite._resolve_lidar_stream(None)

        with mock.patch("nurec.models.gaussians.gaussians_composite.torch.cuda.Stream", side_effect=make_stream):
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(resolve) for _ in range(thread_count)]
                resolved = [future.result(timeout=5) for future in futures]

        self.assertEqual(constructor_calls, 1)
        self.assertTrue(all(stream is fallback for stream in resolved))
        self.assertIs(composite._lidar_stream, fallback)

    def test_initialization_lock_is_not_copied_or_serialized_with_module_state(self):
        composite = self._bare_composite()
        value = torch.ones((1, 1), dtype=torch.float32)
        integer_id = torch.ones((1, 1), dtype=torch.int64)
        integer_state = _PayloadIntegerRangeValidationState()
        integer_state.observe(
            (
                _LayerPayloadLayout(
                    node_id="layer",
                    keys=("value", "id"),
                    trailing_shapes=((1,), (1,)),
                    dtypes=(value.dtype, integer_id.dtype),
                    widths=(1, 1),
                    payload_dtype=torch.float32,
                ),
            ),
            ((value, integer_id),),
            tuple(range(8)),
            0,
            2,
        )
        integer_state.mark_validated()
        self.assertTrue(integer_state.local_ranges_cached)
        composite._fused_payload_integer_range_validation_state = integer_state

        recreated = (copy.deepcopy(composite), pickle.loads(pickle.dumps(composite)))

        for candidate in recreated:
            self.assertNotIn("_lidar_stream_init_lock", candidate.__dict__)
            self.assertIs(candidate._lidar_stream_init_lock, GaussiansComposite._lidar_stream_init_lock)
            recreated_integer_state = candidate._fused_payload_integer_range_validation_state
            self.assertIsInstance(recreated_integer_state, _PayloadIntegerRangeValidationState)
            self.assertIsNot(recreated_integer_state, integer_state)
            self.assertEqual(recreated_integer_state.generation, 0)
            self.assertFalse(recreated_integer_state.local_ranges_cached)


class _StubRenderer:
    """Stand-in renderer that records the fold ref and answers the eligibility query.

    The composite must ask the renderer whether a band-0 fold would be honored rather
    than re-deriving gsplat's rules, so a stub is enough to drive both answers without
    running a real render.
    """

    def __init__(self, fold_supported: bool = True):
        self.fold_supported = fold_supported
        self._fourier_fold_ref = "SENTINEL"
        self.queries: list[dict] = []

    def band0_fold_supported(self, *, is_camera, features_dim, extra_signal_dim, sensor_extra_signal_dim):
        self.queries.append(
            {
                "is_camera": is_camera,
                "features_dim": features_dim,
                "extra_signal_dim": extra_signal_dim,
                "sensor_extra_signal_dim": sensor_extra_signal_dim,
            }
        )
        return self.fold_supported


class TestSparseSegmentPlanning(unittest.TestCase):
    @staticmethod
    def _composite() -> GaussiansComposite:
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite.config = cast(  # type: ignore[assignment]
            Any,
            SimpleNamespace(
                renderer=GSplatRendererConfig(
                    name="3dgut-gsplat",
                    sparse_segments=True,
                    cache_static_gaussians=True,
                )
            ),
        )
        composite._layers_config = LayersConfig(
            layers=[],
            extra_signal_dim=0,
            camera_extra_signal_dim=0,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )
        composite.training = False
        # _ensure_tracked_rows_grouped walks the nodes on the first sparse frame.
        composite.gaussians_nodes = {}  # type: ignore[assignment]
        return composite

    def test_sparse_plan_rejects_an_installed_asset_edit_manifest(self) -> None:
        """Grouping the rows would break the manifest's row digest, so refuse instead."""
        composite = self._composite()
        composite._edit_overrides = cast(Any, object())
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "asset-edit manifest"):
            composite._sparse_segment_plan_enabled(is_training_batch=False)

    def test_the_first_sparse_frame_groups_the_tracked_rows(self) -> None:
        """Not load_state_dict: the manifest install runs after the load and reads row identity."""
        composite = self._composite()
        calls: list[bool] = []
        composite._ensure_tracked_rows_grouped = lambda: calls.append(True)  # type: ignore[method-assign]

        with torch.no_grad():
            self.assertTrue(composite._sparse_segment_plan_enabled(is_training_batch=False))
            self.assertEqual(len(calls), 1)
            # A refusal decides before any row moves.
            with self.assertRaises(ValueError):
                composite._sparse_segment_plan_enabled(is_training_batch=False, combined_sensor_forward=True)
            self.assertEqual(len(calls), 1)

    def test_grouping_is_refused_on_a_context_shell(self) -> None:
        composite = self._composite()
        composite._is_context_shell = True
        with self.assertRaisesRegex(RuntimeError, "before the first context shell"):
            composite._ensure_tracked_rows_grouped()

    def test_grouping_runs_once_per_loaded_rows(self) -> None:
        composite = self._composite()
        composite._ensure_tracked_rows_grouped()
        self.assertTrue(composite._tracked_rows_grouped)
        # A second call is a bool check; an ungrouped layer would otherwise be re-sorted
        # on every frame.
        composite._is_context_shell = True
        composite._ensure_tracked_rows_grouped()

    def test_failed_grouping_stays_retryable(self) -> None:
        # More than MIN_WINDOWS_ALLOWED interleaved rows makes this layer fragmented.
        composite = _make_real_sort_composite(n=65)
        dynamic = cast(RigidGaussianModel, composite.gaussians_nodes["dynamic_rigids"])
        dynamic.cuboid_tracks = cast(Any, SimpleNamespace(n_tracks=4))
        composite._tracked_rows_grouped = False

        with patch.object(composite, "morton_sort_gaussians") as morton_sort:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "could not group"):
                    composite._ensure_tracked_rows_grouped()
                self.assertFalse(composite._tracked_rows_grouped)

        self.assertEqual(morton_sort.call_count, 2)

    def test_sparse_plan_rejects_an_unsupported_inference_path(self) -> None:
        # The settings that are fixed for the run are rejected by GSplatRendererConfig;
        # see renderer_config_test. What is left here varies per frame.
        composite = self._composite()
        with torch.no_grad():
            self.assertTrue(composite._sparse_segment_plan_enabled(is_training_batch=False))
            with self.assertRaisesRegex(ValueError, "combined camera and LiDAR"):
                composite._sparse_segment_plan_enabled(
                    is_training_batch=False,
                    combined_sensor_forward=True,
                )
            with self.assertRaisesRegex(ValueError, "active tile masks"):
                composite._sparse_segment_plan_enabled(
                    is_training_batch=False,
                    has_active_tile_mask=True,
                )
            with self.assertRaisesRegex(ValueError, "without gradients only"):
                composite._sparse_segment_plan_enabled(is_training_batch=True)

    def test_sparse_plan_rejects_requested_extra_signals(self) -> None:
        composite = self._composite()
        camera_outputs = composite.config.renderer.outputs.camera
        lidar_outputs = composite.config.renderer.outputs.lidar

        with torch.no_grad():
            self.assertTrue(
                composite._sparse_segment_plan_enabled(is_training_batch=False),
                "enabled zero-width outputs do not create extra signals",
            )
            composite._layers_config.extra_signal_dim = 3
            with self.assertRaisesRegex(ValueError, "extra signals"):
                composite._sparse_segment_plan_enabled(is_training_batch=False)
            composite._layers_config.extra_signal_dim = 0
            composite._layers_config.camera_extra_signal_dim = 4
            with self.assertRaisesRegex(ValueError, "extra signals"):
                composite._sparse_segment_plan_enabled(is_training_batch=False)
            self.assertTrue(
                composite._sparse_segment_plan_enabled(is_training_batch=False, sensor_name="lidar"),
                "camera-only signals do not affect LiDAR",
            )
            composite._layers_config.lidar_extra_signal_dim = 6
            with self.assertRaisesRegex(ValueError, "extra signals"):
                composite._sparse_segment_plan_enabled(is_training_batch=False, sensor_name="lidar")
            camera_outputs.enable_extended_features = False
            camera_outputs.enable_sensor_features = False
            self.assertTrue(
                composite._sparse_segment_plan_enabled(is_training_batch=False),
                "stored extra-signal widths do not matter when their outputs are disabled",
            )

            composite._layers_config.lidar_extra_signal_dim = 0
            lidar_outputs.enable_extended_features = False
            lidar_outputs.enable_sensor_features = False
            self.assertTrue(composite._sparse_segment_plan_enabled(is_training_batch=False, sensor_name="lidar"))

    def test_cache_disabled_uses_dense_path_even_with_sparse_requested(self) -> None:
        composite = self._composite()
        composite.config.renderer.cache_static_gaussians = False
        with torch.no_grad():
            self.assertFalse(composite._sparse_segment_plan_enabled(is_training_batch=False))


@unittest.skipUnless(torch.cuda.is_available(), "static-cache collect + gsplat fold require CUDA")
class TestGaussiansCompositeStaticCollectCache(unittest.TestCase):
    """Regression coverage for config.renderer.cache_static_gaussians / sparse_segments.

    cache_static_gaussians caches the collect of the static layers for
    both sensors, and folds SH band-0 from a per-frame theta on whichever sensor's
    renderer reports it will honor the fold. Unsupported layouts and configurations
    fall back to the dense collect. These tests drive the production entry points
    GaussiansComposite._collect_maybe_static_cached and
    _invalidate_collect_caches on a minimally-wired composite (mirroring the
    __new__ construction of TestGaussiansCompositeInsertAsset above) and hold the
    cached path bit-identical (torch.equal) to the dense collect wherever the feature
    promises parity.
    """

    ALBEDO_DIM = 3
    SPECULAR_DIM = 45
    CAMERA_EXTRA_DIM = 20

    # ---- fixture builders (reuse the collect package fixtures, no new harness) ----
    def _layers_config(self, layer_configs):
        return LayersConfig(
            layers=layer_configs,
            extra_signal_dim=0,
            camera_extra_signal_dim=self.CAMERA_EXTRA_DIM,
            lidar_extra_signal_dim=0,
            albedo_dim=self.ALBEDO_DIM,
            specular_dim=self.SPECULAR_DIM,
        )

    def _sh_config(self):
        return LayerConfigSH(
            rotation_activation=RotationActivation.NORMALIZE,
            scale_activation=ScaleActivation.EXP,
            density_activation=DensityActivation.SIGMOID,
            fourier_features_dim=1,
            embed_config=None,
        )

    def _rigid_config(self):
        return LayerConfigRigid(
            rotation_activation=RotationActivation.NORMALIZE,
            scale_activation=ScaleActivation.EXP,
            density_activation=DensityActivation.SIGMOID,
            fourier_features_dim=1,
            embed_config=None,
        )

    def _sh_layer_data(self, n):
        return LayerDataSH(
            positions=torch.randn(n, 3, device=TEST_DEVICE),
            rotations=torch.randn(n, 4, device=TEST_DEVICE),
            scales=torch.randn(n, 3, device=TEST_DEVICE),
            densities=torch.randn(n, 1, device=TEST_DEVICE),
            extra_signal=torch.randn(n, 0, device=TEST_DEVICE),
            camera_extra_signal=torch.randn(n, self.CAMERA_EXTRA_DIM, device=TEST_DEVICE),
            lidar_extra_signal=torch.randn(n, 0, device=TEST_DEVICE),
            features_albedo=torch.randn(n, self.ALBEDO_DIM, device=TEST_DEVICE),
            features_specular=torch.randn(n, self.SPECULAR_DIM, device=TEST_DEVICE),
            embed_data=None,
        )

    def _rigid_layer_data(self, n, translation):
        # Single track, identity rotation, translation-only pose (t + quat xyzw). The
        # collected rigid positions depend on this pose, so a different translation
        # models the dynamic layer at a different frame.
        poses = torch.tensor([[translation[0], translation[1], translation[2], 0.0, 0.0, 0.0, 1.0]], device=TEST_DEVICE)
        return LayerDataRigid(
            positions=torch.randn(n, 3, device=TEST_DEVICE),
            rotations=torch.randn(n, 4, device=TEST_DEVICE),
            scales=torch.randn(n, 3, device=TEST_DEVICE),
            densities=torch.randn(n, 1, device=TEST_DEVICE),
            extra_signal=torch.randn(n, 0, device=TEST_DEVICE),
            camera_extra_signal=torch.randn(n, self.CAMERA_EXTRA_DIM, device=TEST_DEVICE),
            lidar_extra_signal=torch.randn(n, 0, device=TEST_DEVICE),
            features_albedo=torch.randn(n, self.ALBEDO_DIM, device=TEST_DEVICE),
            features_specular=torch.randn(n, self.SPECULAR_DIM, device=TEST_DEVICE),
            embed_data=None,
            poses=poses,
            keep_mask=torch.ones(1, dtype=torch.bool, device=TEST_DEVICE),
            tracks_ids=torch.zeros(n, dtype=torch.int32, device=TEST_DEVICE),
        )

    def _make_cache_composite(self, enabled, collector, fold_supported=True):
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite.config = OmegaConf.create(
            {"renderer": {"cache_static_gaussians": enabled, "sparse_segments": enabled}}
        )
        composite.collector = collector
        composite.training = False
        composite._static_collect_cache = None
        composite._static_collect_cache_stream = None
        composite._fourier_fold = None
        # Only the widths are read from this, and they are class constants, so the
        # layer list does not have to match the collector's.
        composite._layers_config = self._layers_config([])
        # Stand-in renderers so the per-renderer fold-ref plumbing and the
        # eligibility query are both observable; "SENTINEL" marks "untouched"
        # (the real renderers are nn.Modules in prod).
        composite._camera_renderer = _StubRenderer(fold_supported)
        composite._lidar_renderer = _StubRenderer(fold_supported)
        return composite

    def _assert_results_equal(self, actual, expected, msg=""):
        for field in fields(expected):
            exp = getattr(expected, field.name)
            act = getattr(actual, field.name)
            if isinstance(exp, torch.Tensor):
                self.assertTrue(
                    torch.equal(act, exp),
                    msg=f"{msg} field {field.name}: cached collect diverged from the dense collect",
                )

    # ---- invariant 1: camera parity across frame timestamps (copy-back correctness) ----
    def test_camera_sparse_matches_dense_across_timestamps(self):
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)

        static_sh = self._sh_layer_data(64)  # shared object -> static layer is frame-invariant
        frame_a = LayersData(layers=[static_sh, self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000)
        frame_b = LayersData(layers=[static_sh, self._rigid_layer_data(16, [0.0, 5.0, 0.0])], frame_timestamp_us=200000)
        # Re-render frame_a AFTER frame_b: catches a stale cache / wrong copy-back window.
        for step, frame in enumerate([frame_a, frame_b, frame_a]):
            with torch.no_grad():
                sparse, draw_plan, _ = composite._collect_maybe_static_cached(frame, None, is_camera=True)
                dense = collector.collect(frame, layer_indices=None)
                self._assert_results_equal(sparse, dense, msg=f"step {step}")
                self.assertIsNone(draw_plan, "a full selection needs no draw plan")

    # ---- invariant 2: lidar uses the cache too, and each sensor keeps its own fold ----
    def test_lidar_uses_the_cache_and_matches_dense(self):
        # A static layer's collected output is a pure function of frozen checkpoint
        # tensors, so it is identical for both sensors and the cache serves both.
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        frame = LayersData(
            layers=[self._sh_layer_data(64), self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000
        )
        composite_on = self._make_cache_composite(enabled=True, collector=collector)
        composite_off = self._make_cache_composite(enabled=False, collector=collector)
        with torch.no_grad():
            lidar_on, _, _ = composite_on._collect_maybe_static_cached(frame, None, is_camera=False)
            lidar_off, _, _ = composite_off._collect_maybe_static_cached(frame, None, is_camera=False)
            dense = collector.collect(frame, layer_indices=None)

        # Lidar output is byte-identical with the cache on vs off, and equals dense.
        self._assert_results_equal(lidar_on, lidar_off, msg="lidar cache-on vs cache-off")
        self._assert_results_equal(lidar_on, dense, msg="lidar vs dense")
        self.assertIsNotNone(composite_on._static_collect_cache, "the lidar collect must populate the cache")
        # The lidar collect must not touch the camera renderer's fold ref: theta is
        # derived from the sensor's own frame window, so the two are not shareable.
        self.assertEqual(composite_on._camera_renderer._fourier_fold_ref, "SENTINEL")

    def test_alternating_sensors_each_match_dense(self):
        # Both sensors share one cached buffer and each writes its own dynamic-layer
        # windows into it before its own render. Alternating them must therefore
        # leave every frame equal to a dense collect, or one sensor is reading the
        # other's dynamic data.
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        static_sh = self._sh_layer_data(64)
        cam_frame = LayersData(
            layers=[static_sh, self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000
        )
        lidar_frame = LayersData(
            layers=[static_sh, self._rigid_layer_data(16, [0.0, 7.0, 0.0])], frame_timestamp_us=150000
        )
        for step, (frame, is_camera) in enumerate(
            [(cam_frame, True), (lidar_frame, False), (cam_frame, True), (lidar_frame, False)]
        ):
            with torch.no_grad():
                cached, _, _ = composite._collect_maybe_static_cached(frame, None, is_camera=is_camera)
                dense = collector.collect(frame, layer_indices=None)
            self._assert_results_equal(cached, dense, msg=f"step {step} is_camera={is_camera}")

    def test_combined_sensor_forward_disables_the_cache(self):
        # A forward carrying both sensors would have the two collects write the same
        # shared buffer with no render in between, so it must take the dense path.
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        frame = LayersData(
            layers=[self._sh_layer_data(64), self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000
        )
        with torch.no_grad():
            for is_camera in (False, True):
                result, _, _ = composite._collect_maybe_static_cached(
                    frame,
                    None,
                    is_camera=is_camera,
                    combined_sensor_forward=True,
                )
                dense = collector.collect(frame, layer_indices=None)
                self._assert_results_equal(result, dense, msg=f"combined forward is_camera={is_camera}")
        self.assertIsNone(composite._static_collect_cache, "a combined cam+lidar forward must not cache")

    def test_cache_survives_a_collect_on_another_cuda_stream(self):
        # The lidar collect runs on a dedicated stream while the cache was allocated
        # on the default one. The values are fenced by forward_gaussians; this pins
        # that the cross-stream read is also recorded with the allocator instead of
        # tripping or corrupting.
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        static_sh = self._sh_layer_data(64)
        frame = LayersData(layers=[static_sh, self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000)

        with torch.no_grad():
            composite._collect_maybe_static_cached(frame, None, is_camera=True)  # populate on the default stream
        self.assertIsNotNone(composite._static_collect_cache)
        self.assertIsNotNone(composite._static_collect_cache_stream)

        other = torch.cuda.Stream()
        other.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(other):
            cached, _, _ = composite._collect_maybe_static_cached(frame, None, is_camera=False)
        torch.cuda.current_stream().wait_stream(other)
        torch.cuda.synchronize()
        with torch.no_grad():
            dense = collector.collect(frame, layer_indices=None)
        self._assert_results_equal(cached, dense, msg="cross-stream warm read")

    # ---- invariant 3: a scene mutation invalidates the cache ----
    def test_scene_mutation_invalidates_cache(self):
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        static_sh = self._sh_layer_data(64)
        frame = LayersData(layers=[static_sh, self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000)

        with torch.no_grad():
            composite._collect_maybe_static_cached(frame, None, is_camera=True)
        self.assertIsNotNone(composite._static_collect_cache, "camera render should populate the cache")

        # Mutate a frozen static-layer tensor in place (stand-in for a checkpoint / scene
        # edit). The stale cache still holds the pre-mutation collect.
        with torch.no_grad():
            static_sh.positions.add_(100.0)

        # train()/eval() toggle is one real invalidation entry point (invariant 3).
        composite.train()
        self.assertIsNone(composite._static_collect_cache, "train(True) must invalidate the cache")
        composite.eval()

        with torch.no_grad():
            sparse_after, _, _ = composite._collect_maybe_static_cached(frame, None, is_camera=True)
            dense_after = collector.collect(frame, layer_indices=None)
        # Would FAIL if invalidation were skipped: the stale static window (pre-mutation)
        # would not match the freshly collected +100 positions.
        self._assert_results_equal(sparse_after, dense_after, msg="post-invalidation")

        # _invalidate_collect_caches is the shared sink every scene-edit method
        # (track_ply_override, insert_asset_*, remove_background_gaussians, revert_edits,
        # load_state_dict) funnels through: it must clear the cache AND every renderer's
        # derived fold-ref.
        composite._camera_renderer._fourier_fold_ref = "STALE"
        composite._lidar_renderer._fourier_fold_ref = "STALE"
        composite._fourier_fold = {"theta": None}
        composite._segment_tables = {"node": object()}  # type: ignore[dict-item]
        composite._invalidate_collect_caches()
        self.assertIsNone(composite._static_collect_cache)
        self.assertEqual(composite._segment_tables, {})
        self.assertIsNone(composite._fourier_fold)
        self.assertIsNone(composite._camera_renderer._fourier_fold_ref)
        self.assertIsNone(composite._lidar_renderer._fourier_fold_ref)

    # ---- invariant 4: unsupported layouts fall back to the dense collect ----
    def test_unsupported_layouts_fall_back_to_dense(self):
        # (a) All-dynamic layout: no static prefix -> "nothing cacheable" -> plain collect.
        rigid_collector = CreateGaussianParameterCollector(self._layers_config([self._rigid_config()]))
        composite_dyn = self._make_cache_composite(enabled=True, collector=rigid_collector)
        dyn_frame = LayersData(layers=[self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000)
        with torch.no_grad():
            sparse_dyn, _, _ = composite_dyn._collect_maybe_static_cached(dyn_frame, None, is_camera=True)
            dense_dyn = rigid_collector.collect(dyn_frame, layer_indices=None)
        self.assertIsNone(composite_dyn._static_collect_cache, "all-dynamic layout must not cache")
        self._assert_results_equal(sparse_dyn, dense_dyn, msg="all-dynamic fallback")

        # (b) Static copy-SH layer with no holistic-Fourier features: cacheable but NOT
        # foldable -> no band-0 fold, fold-ref actively cleared to None so the renderer's
        # fold-divergence assertion can never fire ("no false assertion / no staleness").
        sh_collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config()]))
        composite_static = self._make_cache_composite(enabled=True, collector=sh_collector)
        static_frame = LayersData(layers=[self._sh_layer_data(64)], frame_timestamp_us=100000)
        with torch.no_grad():
            composite_static._collect_maybe_static_cached(static_frame, None, is_camera=True)  # populate cache
            sparse_static, _, _ = composite_static._collect_maybe_static_cached(static_frame, None, is_camera=True)
            dense_static = sh_collector.collect(static_frame, layer_indices=None)
        self.assertIsNotNone(composite_static._static_collect_cache)
        self._assert_results_equal(sparse_static, dense_static, msg="static-only cache")
        self.assertIsNone(
            composite_static._camera_renderer._fourier_fold_ref,
            "an unfoldable layer must not leave a band-0 fold ref set",
        )

    # ---- invariant 5: a selected-node / layer subset bypasses the cache ----
    def test_layer_subset_bypasses_sparse_path(self):
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        # layer_indices != None -> the cache guard requires it be None, so it is bypassed.
        subset = LayersData(layers=[self._sh_layer_data(64)], frame_timestamp_us=100000)
        with torch.no_grad():
            sparse, _, _ = composite._collect_maybe_static_cached(subset, layer_indices=[0], is_camera=True)
            dense = collector.collect(subset, layer_indices=[0])
        self.assertIsNone(composite._static_collect_cache, "layer-subset render must not use the cache")
        self._assert_results_equal(sparse, dense, msg="layer subset")

    # ---- invariant 6: camera dense/early returns clear a stale renderer fold ref ----
    def _canned_result(self, n):
        # Minimal cached CollectorResult so the warm fold path runs without a cold
        # Fourier collect (the fold arithmetic itself is pinned gsplat-side).
        return CollectorResult(
            positions=torch.zeros(n, 3, device=TEST_DEVICE),
            rotations=torch.zeros(n, 4, device=TEST_DEVICE),
            scales=torch.zeros(n, 3, device=TEST_DEVICE),
            densities=torch.zeros(n, 1, device=TEST_DEVICE),
            extra_signal=torch.zeros(n, 0, device=TEST_DEVICE),
            camera_extra_signal=torch.zeros(n, self.CAMERA_EXTRA_DIM, device=TEST_DEVICE),
            lidar_extra_signal=torch.zeros(n, 0, device=TEST_DEVICE),
            features=torch.zeros(n, self.ALBEDO_DIM + self.SPECULAR_DIM, device=TEST_DEVICE),
        )

    def test_camera_dense_returns_clear_stale_fold_ref(self):
        # A prior folded camera frame installs a fold ref; a later camera frame
        # that takes ANY dense/early path must not leave that stale ref on the
        # renderer (it would re-mix band-0 with the previous frame's angle,
        # silently — the divergence assertion does not fire). A lidar collect, by
        # contrast, must never touch the camera renderer's ref.
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        frame = LayersData(
            layers=[self._sh_layer_data(64), self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000
        )

        # (a) all-dynamic layout -> "nothing cacheable" all-variant dense return.
        dyn_collector = CreateGaussianParameterCollector(self._layers_config([self._rigid_config()]))
        composite_dyn = self._make_cache_composite(enabled=True, collector=dyn_collector)
        composite_dyn._camera_renderer._fourier_fold_ref = "STALE"
        dyn_frame = LayersData(layers=[self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000)
        with torch.no_grad():
            composite_dyn._collect_maybe_static_cached(dyn_frame, None, is_camera=True)
        self.assertIsNone(
            composite_dyn._camera_renderer._fourier_fold_ref,
            "all-variant dense return must clear a stale camera fold ref",
        )

        # (b) selected-node (layer_indices set) -> not-sparse-enabled dense return.
        # The subset frame's layers align with layer_indices, matching the render
        # path's selected-node collect.
        subset_frame = LayersData(layers=[self._sh_layer_data(64)], frame_timestamp_us=100000)
        composite._camera_renderer._fourier_fold_ref = "STALE"
        with torch.no_grad():
            composite._collect_maybe_static_cached(subset_frame, layer_indices=[0], is_camera=True)
        self.assertIsNone(
            composite._camera_renderer._fourier_fold_ref,
            "a selected-node dense return must clear a stale camera fold ref",
        )

        # (c) a lidar collect must NOT clear the camera ref: in a combined forward
        # it can run after the camera collect, and clearing here would drop the
        # camera's fresh fold before it renders.
        composite._camera_renderer._fourier_fold_ref = "SENTINEL"
        with torch.no_grad():
            composite._collect_maybe_static_cached(frame, None, is_camera=False)
        self.assertEqual(
            composite._camera_renderer._fourier_fold_ref,
            "SENTINEL",
            "a lidar collect must not touch the camera renderer's fold ref",
        )

    # ---- invariant 7: a holistic-Fourier layer folds and hands raw inputs to the renderer ----
    def test_holistic_fourier_layer_folds_and_hands_raw_inputs_to_renderer(self):
        # The NuRec side of the fold: a static SH layer with a 3-dim
        # features_albedo under a HOLISTIC time embedding is classified foldable,
        # excluded from the per-frame collect, and its RAW fold inputs (albedo +
        # embed range + this frame's timestamp) are handed to the camera renderer
        # (which turns them into theta). The fold arithmetic is pinned gsplat-side;
        # here we pin the composite decision, the raw handoff, and per-frame theta
        # freshness.
        embed = HolisticRemapTimeInputEmbeddingConfig(
            timestamps_us_min=1000, timestamps_us_max=9000, remap_min=-1.0, remap_max=1.0
        )
        fold_config = LayerConfigSH(
            rotation_activation=RotationActivation.NORMALIZE,
            scale_activation=ScaleActivation.EXP,
            density_activation=DensityActivation.SIGMOID,
            fourier_features_dim=3,
            embed_config=embed,
        )
        collector = CreateGaussianParameterCollector(self._layers_config([fold_config, self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)

        # Foldable static layer: 3-dim features_albedo [n, fourier_dim, albedo_dim].
        albedo = torch.randn(64, 3, self.ALBEDO_DIM, device=TEST_DEVICE)
        fold_layer = replace(self._sh_layer_data(64), features_albedo=albedo)
        # Pre-seed the cache so the warm fold path runs without a cold Fourier
        # collect: the folded layer is excluded from the warm collect, only the
        # dynamic tail is re-collected. The layout must be seeded alongside it —
        # it is what tells the warm path the cached region still matches.
        composite._static_collect_cache = self._canned_result(64 + 16)
        composite._static_collect_cache_layout = (64, 16)

        def _frame(ts):
            return LayersData(layers=[fold_layer, self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=ts)

        with torch.no_grad():
            composite._collect_maybe_static_cached(_frame(5000), None, is_camera=True)
        ref = composite._camera_renderer._fourier_fold_ref
        self.assertIsNotNone(ref, "a holistic-Fourier layer must install a fold ref on the camera renderer")
        self.assertEqual(ref["frame_timestamp_us"], 5000)
        self.assertEqual((ref["timestamps_us_min"], ref["timestamps_us_max"]), (1000, 9000))
        self.assertEqual((ref["remap_min"], ref["remap_max"]), (-1.0, 1.0))
        self.assertEqual((ref["start"], ref["size"]), (0, 64), "fold ref must window the static layer")
        self.assertTrue(torch.equal(ref["albedo"], albedo), "renderer must receive the RAW albedo, unmixed")

        # A different timestamp refreshes the angle inputs (no stale theta).
        with torch.no_grad():
            composite._collect_maybe_static_cached(_frame(7000), None, is_camera=True)
        self.assertEqual(
            composite._camera_renderer._fourier_fold_ref["frame_timestamp_us"],
            7000,
            "the fold ref must carry this frame's timestamp, not a stale one",
        )

    # ---- invariant 8: an ineligible render config falls back to a real dense collect ----
    def _fold_scene(self):
        embed = HolisticRemapTimeInputEmbeddingConfig(
            timestamps_us_min=1000, timestamps_us_max=9000, remap_min=-1.0, remap_max=1.0
        )
        fold_config = LayerConfigSH(
            rotation_activation=RotationActivation.NORMALIZE,
            scale_activation=ScaleActivation.EXP,
            density_activation=DensityActivation.SIGMOID,
            fourier_features_dim=3,
            embed_config=embed,
        )
        collector = CreateGaussianParameterCollector(self._layers_config([fold_config, self._rigid_config()]))
        albedo = torch.randn(64, 3, self.ALBEDO_DIM, device=TEST_DEVICE)
        fold_layer = replace(self._sh_layer_data(64), features_albedo=albedo)
        return collector, fold_layer

    def test_ineligible_render_config_falls_back_to_dense_collect(self):
        # When the renderer answers that it will not honor the fold, the foldable
        # layer must be collected densely every frame -- the advertised fallback --
        # rather than folded and left for the renderer's divergence assert to catch.
        # Two timestamps: a stale band-0 only shows up on the second frame.
        collector, fold_layer = self._fold_scene()
        composite = self._make_cache_composite(enabled=True, collector=collector, fold_supported=False)

        # One frame object per timestamp, reused for both collects: the rigid layer
        # carries random positions, so rebuilding it would compare two scenes.
        dynamic = self._rigid_layer_data(16, [1.0, 0.0, 0.0])
        frames = {ts: LayersData(layers=[fold_layer, dynamic], frame_timestamp_us=ts) for ts in (5000, 7000)}

        for ts, frame in frames.items():
            with torch.no_grad():
                cached, _, _ = composite._collect_maybe_static_cached(frame, None, is_camera=True)
                dense = collector.collect(frame, layer_indices=None)
            self._assert_results_equal(cached, dense, msg=f"ineligible fallback at ts={ts}")
        self.assertIsNone(
            composite._camera_renderer._fourier_fold_ref,
            "an ineligible render config must leave no fold ref for the renderer to honor",
        )
        # The composite must have asked the renderer rather than deciding itself.
        self.assertTrue(composite._camera_renderer.queries, "the composite must consult the renderer's predicate")
        query = composite._camera_renderer.queries[0]
        self.assertTrue(query["is_camera"])
        self.assertEqual(query["features_dim"], self.ALBEDO_DIM + self.SPECULAR_DIM)
        self.assertEqual(query["sensor_extra_signal_dim"], self.CAMERA_EXTRA_DIM)

    def test_dynamic_suffix_is_collected_into_the_cache_without_a_second_buffer(self):
        # The dynamic layers are a contiguous suffix, so they are collected straight
        # into the cached buffer's tail. Allocating a second full result to copy back
        # would be the largest per-frame allocation on this path -- for a scene whose
        # dynamic tail is a third of the gaussians it dwarfs everything else.
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        static_sh = self._sh_layer_data(64)
        frame_a = LayersData(layers=[static_sh, self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000)
        frame_b = LayersData(layers=[static_sh, self._rigid_layer_data(16, [0.0, 5.0, 0.0])], frame_timestamp_us=200000)

        with torch.no_grad():
            composite._collect_maybe_static_cached(frame_a, None, is_camera=True)  # cold: populate
        cache = composite._static_collect_cache
        ptrs_before = {f.name: getattr(cache, f.name).data_ptr() for f in fields(cache)}

        with torch.no_grad():
            warm, _, _ = composite._collect_maybe_static_cached(frame_b, None, is_camera=True)
            dense = collector.collect(frame_b, layer_indices=None)

        # Same buffers, written in place -- not reallocated and not copied into.
        self.assertIs(warm, cache, "the warm path must return the cached result itself")
        for f in fields(cache):
            self.assertEqual(
                getattr(cache, f.name).data_ptr(),
                ptrs_before[f.name],
                f"{f.name}: the cached buffer must be written in place, not replaced",
            )
        self._assert_results_equal(warm, dense, msg="in-place dynamic collect")

    def test_segmented_collect_writes_only_the_live_windows_through_the_cache(self):
        """With a window list, the cold and the warm collect touch only the live track rows."""
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        static = self._sh_layer_data(64)

        def two_track_rigid(translation):
            # Rows [0, 8) belong to track 0 and [8, 16) to track 1; both share the pose.
            layer = self._rigid_layer_data(16, translation)
            layer.poses = layer.poses.repeat(2, 1)
            layer.keep_mask = torch.ones(2, dtype=torch.bool, device=TEST_DEVICE)
            layer.tracks_ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.int32, device=TEST_DEVICE)
            return layer

        # Only track 1 is live: window [8, 16) of the rigid layer, the static layer whole.
        selections = [np.array([[0, 64]], dtype=np.int64), np.array([[8, 8]], dtype=np.int64)]
        static_rows, dead_rows, live_rows = slice(0, 64), slice(64, 72), slice(72, 80)

        def assert_rows_equal(actual, expected, rows, msg):
            for field in fields(expected):
                exp = getattr(expected, field.name)
                if isinstance(exp, torch.Tensor) and exp.shape[1] > 0:
                    self.assertTrue(torch.equal(getattr(actual, field.name)[rows], exp[rows]), f"{msg}: {field.name}")

        with torch.no_grad():
            first = LayersData(layers=[static, two_track_rigid([1.0, 0.0, 0.0])], frame_timestamp_us=100000)
            dense_first = collector.collect(first)
            cold, cold_plan, cold_written = composite._collect_maybe_static_cached(
                first,
                None,
                is_camera=True,
                selections=selections,
            )
            self.assertIsNotNone(cold_plan, "a partial selection must produce a draw plan")
            # post_process reads these back, so they must be the rows the collect wrote.
            assert cold_written is not None
            np.testing.assert_array_equal(cold_written[0], [[0, 64]])
            np.testing.assert_array_equal(cold_written[1], [[8, 8]])
            self.assertIsNotNone(composite._static_collect_cache, "the cold frame must populate the cache")
            assert_rows_equal(cold, dense_first, static_rows, "cold static rows")
            assert_rows_equal(cold, dense_first, live_rows, "cold live window")
            stale = cold.positions[dead_rows].clone()

            second = LayersData(layers=[static, two_track_rigid([0.0, 2.0, 0.0])], frame_timestamp_us=200000)
            dense_second = collector.collect(second)
            warm, warm_plan, warm_written = composite._collect_maybe_static_cached(
                second,
                None,
                is_camera=True,
                selections=selections,
            )
            self.assertIsNotNone(warm_plan, "a partial selection must produce a draw plan")
            assert warm_written is not None
            self.assertEqual(warm_written[0].shape, (0, 2), "the warm cacheable layer writes no rows")
            np.testing.assert_array_equal(warm_written[1], [[8, 8]])
            self.assertIs(composite._static_collect_cache, warm, "the warm frame must reuse the cached buffers")
            assert_rows_equal(warm, dense_second, static_rows, "warm static rows")
            assert_rows_equal(warm, dense_second, live_rows, "warm live window")
            self.assertTrue(
                torch.equal(warm.positions[dead_rows], stale),
                "rows outside the live window must not be rewritten by the warm collect",
            )
            self.assertFalse(
                torch.equal(warm.positions[dead_rows], dense_second.positions[dead_rows]),
                "sanity: the dense collect does move the dead rows, so the check above is not vacuous",
            )

    def test_cold_sparse_frame_initializes_every_static_row_for_later_disjoint_selections(self):
        """Static cache initialization is independent of the cold frame's draw ranges."""
        collector = CreateGaussianParameterCollector(
            self._layers_config([self._sh_config(), self._sh_config(), self._rigid_config()])
        )
        real_collect = collector.collect

        def collect_with_nan_destination(layers_data, layer_indices=None, out=None, segments=None):
            # Production deliberately leaves never-drawn rows untouched. Give the
            # cold sparse collect a visible sentinel destination so this test can
            # distinguish sparse dynamic work from an accidental dense collect.
            if out is None and segments is not None:
                template = real_collect(layers_data, layer_indices=layer_indices)
                out = CollectorResult(
                    **{
                        field.name: torch.full_like(getattr(template, field.name), float("nan"))
                        for field in fields(template)
                    }
                )
            return real_collect(layers_data, layer_indices=layer_indices, out=out, segments=segments)

        collect_mock = Mock(side_effect=collect_with_nan_destination)
        collector.collect = collect_mock  # type: ignore[method-assign]
        composite = self._make_cache_composite(enabled=True, collector=collector)
        static_a = self._sh_layer_data(6)
        static_b = self._sh_layer_data(7)

        # The second static layer and the dynamic layer start at non-zero global
        # offsets (6 and 13). The warm static ranges are disjoint from the cold ones.
        cold_ranges = [
            np.array([[0, 2]], dtype=np.int64),
            np.array([[4, 2]], dtype=np.int64),
            np.array([[3, 2]], dtype=np.int64),
        ]
        warm_ranges = [
            np.array([[4, 2]], dtype=np.int64),
            np.array([[0, 2]], dtype=np.int64),
            np.array([[3, 2]], dtype=np.int64),
        ]
        static_rows = slice(0, 13)
        dynamic_dead = torch.tensor([13, 14, 15, 18, 19, 20], dtype=torch.long, device=TEST_DEVICE)

        with torch.no_grad():
            cold_frame = LayersData(
                layers=[static_a, static_b, self._rigid_layer_data(8, [1.0, 0.0, 0.0])],
                frame_timestamp_us=100000,
            )
            dense_cold = collector.collect(cold_frame)
            cold, cold_plan, _ = composite._collect_maybe_static_cached(
                cold_frame,
                None,
                is_camera=True,
                selections=cold_ranges,
            )
            self.assertIsNotNone(cold_plan, "a partial selection must produce a draw plan")

            # Every static row in every output buffer is ready even though this frame
            # draws only two rows from each static layer. Skipped dynamic rows remain
            # a sentinel, proving the cold collect itself did not become dense.
            for field in fields(cold):
                got = getattr(cold, field.name)
                expected = getattr(dense_cold, field.name)
                self.assertTrue(torch.equal(got[static_rows], expected[static_rows]), field.name)
                self.assertTrue(torch.isnan(got[dynamic_dead]).all(), field.name)

            warm_frame = LayersData(
                layers=[static_a, static_b, self._rigid_layer_data(8, [0.0, 5.0, 0.0])],
                frame_timestamp_us=200000,
            )
            dense_warm = collector.collect(warm_frame)
            collect_mock.reset_mock()
            # Each upload owns its device tensor, so the warm plan is independent of
            # the still-live cold plan.
            warm, warm_plan, _ = composite._collect_maybe_static_cached(
                warm_frame,
                None,
                is_camera=True,
                selections=warm_ranges,
            )
            self.assertIsNotNone(warm_plan, "a partial selection must produce a draw plan")

        self.assertIs(warm, composite._static_collect_cache)
        collect_mock.assert_called_once()
        warm_call = collect_mock.call_args
        self.assertIsNone(
            warm_call.kwargs["layer_indices"],
            "the warm collect must hand the collector every layer and let zero-count segments skip the cacheable "
            "ones, not a layer_indices subset",
        )
        self.assertEqual(
            len(warm_call.args[0].layers), 3, "the warm collect must pass the full layer list, not a subset"
        )
        # The two later-visible static windows use the same global offsets in every
        # attribute buffer and match a dense collect without any static recollect.
        for field in fields(warm):
            got = getattr(warm, field.name)
            expected = getattr(dense_warm, field.name)
            for rows in (slice(4, 6), slice(6, 8)):
                self.assertTrue(torch.equal(got[rows], expected[rows]), f"{field.name} rows={rows}")
            self.assertTrue(torch.equal(got[16:18], expected[16:18]), f"{field.name} dynamic window")
            self.assertTrue(torch.isnan(got[dynamic_dead]).all(), field.name)

    def test_dynamic_resize_refreshes_only_displaced_static_layers(self):
        collector = CreateGaussianParameterCollector(
            self._layers_config([self._sh_config(), self._rigid_config(), self._sh_config()])
        )
        composite = self._make_cache_composite(enabled=True, collector=collector)
        first, last = self._sh_layer_data(6), self._sh_layer_data(7)
        selections = [np.array([[0, 2]]), np.array([[1, 2]]), np.array([[0, 2]])]
        with torch.no_grad():
            initial = LayersData(
                layers=[first, self._rigid_layer_data(8, [1.0, 0.0, 0.0]), last], frame_timestamp_us=100000
            )
            composite._collect_maybe_static_cached(
                initial,
                None,
                selections=selections,
            )
            cache = composite._static_collect_cache
            pointers = {f.name: getattr(cache, f.name).data_ptr() for f in fields(cache)}
            for n_dynamic, displaced in ((4, True), (4, False), (8, True), (8, False)):
                with self.subTest(n_dynamic=n_dynamic, displaced=displaced):
                    frame = LayersData(
                        layers=[first, self._rigid_layer_data(n_dynamic, [0.0, 5.0, 0.0]), last],
                        frame_timestamp_us=200000,
                    )
                    dense = collector.collect(frame)
                    with mock.patch.object(collector, "collect", wraps=collector.collect) as collect_call:
                        result, plan, _ = composite._collect_maybe_static_cached(
                            frame,
                            None,
                            selections=selections,
                        )
                    segments = collect_call.call_args.kwargs["segments"]
                    self.assertEqual([s.active_count for s in segments], [0, 2, 7 if displaced else 0])
                    self.assertIs(collect_call.call_args.kwargs["out"], cache)
                    self.assertIs(composite._static_collect_cache, cache)
                    self.assertIsNotNone(plan)
                    self.assertEqual(result.positions.shape[0], 13 + n_dynamic)
                    for f in fields(cache):
                        self.assertEqual(getattr(cache, f.name).data_ptr(), pointers[f.name])
                        got, expected = getattr(result, f.name), getattr(dense, f.name)
                        torch.testing.assert_close(got[:6], expected[:6], rtol=0, atol=0)
                        torch.testing.assert_close(got[7:9], expected[7:9], rtol=0, atol=0)
                        torch.testing.assert_close(got[6 + n_dynamic :], expected[6 + n_dynamic :], rtol=0, atol=0)

    def test_cache_disabled_ignores_sparse_selection(self):
        collector = CreateGaussianParameterCollector(self._layers_config([self._rigid_config()]))
        composite = self._make_cache_composite(enabled=False, collector=collector)
        frame = LayersData(layers=[self._rigid_layer_data(8, [1.0, 0.0, 0.0])], frame_timestamp_us=100000)
        with torch.no_grad(), mock.patch.object(collector, "collect", wraps=collector.collect) as collect_call:
            result, plan, _ = composite._collect_maybe_static_cached(frame, None, selections=[np.array([[1, 2]])])
            self.assertIsNone(collect_call.call_args.kwargs["segments"])
            self.assertIsNone(plan)
            self._assert_results_equal(result, collector.collect(frame))

    def test_resized_dynamic_layer_invalidates_the_cache(self):
        # Per-frame actor edits filter tracks by dropping rows, which resizes the
        # dynamic layers without going through any invalidation entry point. A warm
        # cache must not be reused across that: on a shrink the tail would still hold
        # the previous frame's gaussians (ghosts), and on a grow the copy-back would
        # fail on a size mismatch.
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        static_sh = self._sh_layer_data(64)

        frame_full = LayersData(
            layers=[static_sh, self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000
        )
        with torch.no_grad():
            composite._collect_maybe_static_cached(frame_full, None, is_camera=True)
        self.assertEqual(composite._static_collect_cache.positions.shape[0], 64 + 16)

        for label, n_dynamic in (("shrink", 4), ("grow", 32)):
            frame = LayersData(
                layers=[static_sh, self._rigid_layer_data(n_dynamic, [0.0, 2.0, 0.0])], frame_timestamp_us=200000
            )
            with torch.no_grad():
                got, _, _ = composite._collect_maybe_static_cached(frame, None, is_camera=True)
                dense = collector.collect(frame, layer_indices=None)
            self.assertEqual(got.positions.shape[0], 64 + n_dynamic, f"{label}: wrong row count")
            self._assert_results_equal(got, dense, msg=f"resized dynamic layer ({label})")

    def test_actor_edits_resize_the_dynamic_tail_without_ghost_rows(self):
        # enable_editing_actors subsets the tracks per frame (render.py _edit_actors ->
        # CuboidTracks.Ops.subset_from_tracks_id), and the tracks_edit branch in
        # gaussians_model drops the filtered gaussians' rows outright rather than
        # masking them. So the dynamic tail legitimately changes length between frames
        # with the cache warm, and none of the invalidation entry points fire. A shrunk
        # frame must not expose the previous frame's rows in the tail.
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        static_sh = self._sh_layer_data(64)

        big = LayersData(layers=[static_sh, self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000)
        small = LayersData(layers=[static_sh, self._rigid_layer_data(4, [0.0, 5.0, 0.0])], frame_timestamp_us=200000)

        with torch.no_grad():
            composite._collect_maybe_static_cached(big, None, is_camera=True)  # cold: buffer sized 64 + 16
        cache_rows = composite._static_collect_cache.positions.shape[0]
        self.assertEqual(cache_rows, 80)

        # Shrink: fewer active actors. Must return 68 rows, not 80, and match dense.
        with torch.no_grad():
            shrunk, _, _ = composite._collect_maybe_static_cached(small, None, is_camera=True)
            dense_small = collector.collect(small, layer_indices=None)
        self.assertEqual(shrunk.positions.shape[0], 68, "a shrunk frame must not hand back the stale tail")
        self._assert_results_equal(shrunk, dense_small, msg="shrunk actor set")
        self.assertEqual(
            composite._static_collect_cache.positions.shape[0],
            cache_rows,
            "shrinking must reuse the buffer, not reallocate it",
        )

        # Back up to the original size: still correct, and still the same buffer.
        with torch.no_grad():
            regrown, _, _ = composite._collect_maybe_static_cached(big, None, is_camera=True)
            dense_big = collector.collect(big, layer_indices=None)
        self.assertEqual(regrown.positions.shape[0], 80)
        self._assert_results_equal(regrown, dense_big, msg="regrown actor set")

        # Growing past the high-water mark repopulates rather than overrunning.
        bigger = LayersData(layers=[static_sh, self._rigid_layer_data(32, [0.0, 0.0, 3.0])], frame_timestamp_us=300000)
        with torch.no_grad():
            grown, _, _ = composite._collect_maybe_static_cached(bigger, None, is_camera=True)
            dense_bigger = collector.collect(bigger, layer_indices=None)
        self.assertEqual(grown.positions.shape[0], 96)
        self._assert_results_equal(grown, dense_bigger, msg="grown past capacity")

    def test_static_layer_resize_invalidates_the_cache(self):
        # A change in the cached region itself (not the dynamic tail) means the scene
        # changed, so the buffer must be rebuilt rather than sliced.
        collector = CreateGaussianParameterCollector(self._layers_config([self._sh_config(), self._rigid_config()]))
        composite = self._make_cache_composite(enabled=True, collector=collector)
        frame_a = LayersData(
            layers=[self._sh_layer_data(64), self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=100000
        )
        frame_b = LayersData(
            layers=[self._sh_layer_data(32), self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=200000
        )
        with torch.no_grad():
            composite._collect_maybe_static_cached(frame_a, None, is_camera=True)
            resized, _, _ = composite._collect_maybe_static_cached(frame_b, None, is_camera=True)
            dense_b = collector.collect(frame_b, layer_indices=None)
        self.assertEqual(resized.positions.shape[0], 48)
        self._assert_results_equal(resized, dense_b, msg="static layer resized")

    def test_cold_frame_caches_a_lone_holistic_layer(self):
        # The fold is chosen before the cache is read. If eligibility were keyed on
        # the cached result, a scene whose only cacheable layer is the holistic one
        # would never populate the cache (no cache -> no fold -> every layer variant
        # -> "nothing cacheable" early return), and the fold would never engage.
        collector, fold_layer = self._fold_scene()
        composite = self._make_cache_composite(enabled=True, collector=collector, fold_supported=True)

        def _frame(ts):
            return LayersData(layers=[fold_layer, self._rigid_layer_data(16, [1.0, 0.0, 0.0])], frame_timestamp_us=ts)

        selections = [np.array([[10, 5]], dtype=np.int64), np.array([[3, 2]], dtype=np.int64)]
        with torch.no_grad():
            cold_frame = _frame(5000)
            dense_cold = collector.collect(cold_frame)
            cold, cold_plan, _ = composite._collect_maybe_static_cached(
                cold_frame,
                None,
                is_camera=True,
                selections=selections,
            )
        self.assertIsNotNone(cold_plan, "a partial selection must produce a draw plan")
        self.assertIsNotNone(composite._static_collect_cache, "the cold frame must populate the cache")
        for field in fields(cold):
            self.assertTrue(
                torch.equal(getattr(cold, field.name)[:64], getattr(dense_cold, field.name)[:64]),
                f"folded layer must be cached whole in {field.name}",
            )

        with torch.no_grad():
            composite._collect_maybe_static_cached(
                _frame(7000),
                None,
                is_camera=True,
                selections=selections,
            )
        self.assertIsNotNone(
            composite._camera_renderer._fourier_fold_ref,
            "the warm frame must install the fold ref once the cache exists",
        )

    # ---- invariant 9: the folded band-0 tracks the dense collect to within one 8-bit level ----
    def test_folded_band0_matches_the_dense_collect_within_one_8bit_level(self):
        """Pins the fold-vs-collect parity contract this MR advertises.

        The fold does not reproduce the collector bit-for-bit, by construction. The
        collector's holistic remap runs in Slang `float`: `embed()` returns fp32
        (collector.slang:396-400) and the `(scalar_t)` cast to fp64 happens after that
        rounding, so theta is an fp64 angle computed from an fp32-rounded time
        embedding. The renderer computes the same remap in Python float64
        (renderers.py) and hands gsplat an fp64 theta. Both use an fp64 2*pi; the
        operand differs in precision, and that is the whole of the difference.

        The reference below accumulates in fp64 in the fold's fixed order and then
        returns fp32, because that is what the fold itself returns -- gsplat's
        fourier_mix_band0 ends in `(float)acc` -- and what the collector stores, so the
        comparison is fp32-to-fp32. gsplat pins its kernel against this same
        reconstruction bit-exactly (test_fourier_fold_band0_bitexact), so bounding
        collector-vs-reference here bounds collector-vs-rendered-fold by the same
        amount without a render.

        Two separate things are pinned, because one measurement cannot carry both:

        * That the remap precisions actually differ is asserted on the scalar embedding
          itself, per timestamp. That is exact and deterministic. A timestamp landing on
          an exactly representable remap (6250 for this range, ratio 21/32) agrees
          bit-for-bit and would pin nothing, so it is excluded.
        * That the total drift is image-safe is the bound at the end. Note the drift is
          NOT dominated by the remap difference: measured against a reference built from
          the fp32 remap instead, it barely moves (~7.0e-6 vs ~6.2e-6). At band-0
          magnitudes of order ten, fp32 output quantization and the ULP between libm and
          libdevice sin/cos are the larger terms. So the bound is on the total
          collector-vs-fold drift, which is what image safety depends on, and it is not
          evidence about any single contributing term.
        """
        import math

        n, fourier_dim, albedo_dim, specular_dim = 4096, 5, 3, 45
        ts_min, ts_max, remap_min, remap_max = 1000, 9000, -1.0, 1.0

        embed = HolisticRemapTimeInputEmbeddingConfig(
            timestamps_us_min=ts_min, timestamps_us_max=ts_max, remap_min=remap_min, remap_max=remap_max
        )
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=fourier_dim,
                    embed_config=embed,
                )
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=0,
            lidar_extra_signal_dim=0,
            albedo_dim=albedo_dim,
            specular_dim=specular_dim,
        )
        collector = CreateGaussianParameterCollector(layers_config)

        torch.manual_seed(0)
        albedo = torch.randn(n, fourier_dim, albedo_dim, device=TEST_DEVICE)
        layer = LayerDataSH(
            positions=torch.randn(n, 3, device=TEST_DEVICE),
            rotations=torch.randn(n, 4, device=TEST_DEVICE),
            scales=torch.randn(n, 3, device=TEST_DEVICE),
            densities=torch.randn(n, 1, device=TEST_DEVICE),
            extra_signal=torch.zeros(n, 0, device=TEST_DEVICE),
            camera_extra_signal=torch.zeros(n, 0, device=TEST_DEVICE),
            lidar_extra_signal=torch.zeros(n, 0, device=TEST_DEVICE),
            features_albedo=albedo,
            features_specular=torch.randn(n, specular_dim, device=TEST_DEVICE),
            embed_data=None,
        )

        def time_embed_fp64(frame_timestamp_us):
            # The renderer's host-side remap (renderers.py), in Python float64.
            ratio = (frame_timestamp_us - ts_min) / (ts_max - ts_min)
            return ratio * (remap_max - remap_min) + remap_min

        def time_embed_fp32(frame_timestamp_us):
            # The collector's remap, which HolisticRemapTimeInputEmbedding::embed
            # evaluates and returns in Slang `float` before anything widens it.
            f32 = torch.float32
            ratio = torch.tensor(frame_timestamp_us - ts_min, dtype=f32) / torch.tensor(ts_max - ts_min, dtype=f32)
            embedded = ratio * torch.tensor(remap_max - remap_min, dtype=f32) + torch.tensor(remap_min, dtype=f32)
            return float(embedded)

        def fold_reference(frame_timestamp_us):
            # gsplat's fourier_mix_band0: fp64 accumulation, j=0 first, then per k the
            # sin term before the cos term, returned as fp32.
            theta = 2.0 * math.pi * time_embed_fp64(frame_timestamp_us)
            stack = albedo.double()
            acc = stack[:, 0, :].clone()
            sin_theta, cos_theta = math.sin(theta), math.cos(theta)
            sin_k, cos_k = 0.0, 1.0
            for k in range(1, fourier_dim // 2 + 1):
                sin_k, cos_k = (
                    sin_k * cos_theta + cos_k * sin_theta,
                    cos_k * cos_theta - sin_k * sin_theta,
                )
                acc = acc + sin_k * stack[:, 2 * k - 1, :]
                if 2 * k < fourier_dim:
                    acc = acc + cos_k * stack[:, 2 * k, :]
            return acc.float()

        # Band 0 reaches the image through the SH DC coefficient, so a band-0 delta of
        # d shifts linear colour by d * _SH_C0. Inverting that at one 8-bit level gives
        # a bound fixed by the SH basis rather than by whatever the random albedo
        # happens to peak at.
        sh_c0 = 0.5 * math.sqrt(1.0 / math.pi)
        image_safe_band0_delta = (1.0 / 255.0) / sh_c0

        worst = 0.0
        for frame_timestamp_us in (5001, 3333, 7777, 1234, 8765):
            self.assertNotEqual(
                time_embed_fp32(frame_timestamp_us),
                time_embed_fp64(frame_timestamp_us),
                f"timestamp {frame_timestamp_us} remaps identically in fp32 and fp64, so it "
                "cannot exercise the precision difference this test bounds",
            )
            with torch.no_grad():
                collected = collector.collect(LayersData(layers=[layer], frame_timestamp_us=frame_timestamp_us))
            collected_band0 = collected.features[:, :albedo_dim].float()
            reference = fold_reference(frame_timestamp_us)
            worst = max(worst, (collected_band0 - reference).abs().max().item())
            self.assertFalse(
                torch.equal(collected_band0, reference),
                f"at timestamp {frame_timestamp_us} the collector and the fold agreed "
                "bit-for-bit in fp32; the two are separate implementations and are not "
                "expected to, so this points at the reference having converged on the "
                "collector rather than reconstructing it independently",
            )

        self.assertLess(
            worst,
            image_safe_band0_delta,
            f"folded band-0 drifted from the dense collect by {worst:.3e}, beyond the "
            f"{image_safe_band0_delta:.3e} that one 8-bit level of linear colour allows "
            "through the SH DC coefficient",
        )


class _TileGeometryLessRenderer(BaseGaussianRenderer):
    """A ``BaseGaussianRenderer`` that implements no active-tiling geometry.

    It inherits the base's active-tiling hooks (``active_tile_mask_tile_size`` and
    ``build_camera_to_lidar_tile_mapping``), which both return ``None`` — the "this renderer
    does not support active tiling" answer. That makes it the in-tree stand-in for every
    renderer that only inherits the hooks: Gaussian3DNRenderer (which deliberately implements
    neither, so an ``active_tile_mask.enabled`` config stays valid but inert for it), a future
    third backend, or a test double wrapping a renderer. The composite must degrade loudly —
    warn once, skip the masking — rather than assert on a concrete renderer class.
    """

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(active_tile_mask=ActiveTileMaskConfig(enabled=True)), None)  # type: ignore[arg-type]

    def render(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("render() is not exercised by these tests")


def _make_tileless_rendering_data(h: int, w: int) -> RenderingData:
    """Minimal camera ``RenderingData``; the tile-geometry-less renderer never inspects the
    sensor model parameters, so a placeholder suffices."""
    return RenderingData(
        rays=torch.zeros(1, h, w, 6, device=TEST_DEVICE),
        sensor_model_parameters=[object()],  # type: ignore[list-item]
        poses_tquat_startend=torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2], device=TEST_DEVICE),
        timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64, device=TEST_DEVICE),
        timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
    )


class TestActiveTilingWithoutTileGeometry(unittest.TestCase):
    """A renderer that opts out of active tiling degrades with a warning, never an assert."""

    def test_mapping_init_warns_and_caches_none(self) -> None:
        renderer = _TileGeometryLessRenderer()
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite._camera_renderer = renderer
        composite._lidar_renderer = renderer
        composite._cam_to_lidar_tile_mappings = {}

        rd_cam = _make_tileless_rendering_data(8, 8)
        rd_lidar = _make_tileless_rendering_data(4, 16)

        with self.assertLogs(gaussians_composite_module.log, level="WARNING") as logs:
            composite._maybe_init_cam_to_lidar_tile_mapping(7, rd_cam, rd_lidar)

        self.assertIn(7, composite._cam_to_lidar_tile_mappings)
        self.assertIsNone(composite._cam_to_lidar_tile_mappings[7])
        self.assertTrue(any("tile_mask" in line for line in logs.output))

    def test_mapping_init_warns_when_only_the_lidar_renderer_lacks_tile_geometry(self) -> None:
        """Camera side resolves a tile size, lidar side declines -> warn + cache None."""

        class _CameraOnlyTileGeometry(_TileGeometryLessRenderer):
            def active_tile_mask_tile_size(self, width: int, height: int) -> tuple[int, int]:
                del width, height
                return 4, 4

        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite._camera_renderer = _CameraOnlyTileGeometry()
        composite._lidar_renderer = _TileGeometryLessRenderer()
        composite._cam_to_lidar_tile_mappings = {}

        with self.assertLogs(gaussians_composite_module.log, level="WARNING") as logs:
            composite._maybe_init_cam_to_lidar_tile_mapping(
                0, _make_tileless_rendering_data(8, 8), _make_tileless_rendering_data(4, 16)
            )

        self.assertIsNone(composite._cam_to_lidar_tile_mappings[0])
        self.assertTrue(any("unsupported for this lidar renderer" in line for line in logs.output))

    def test_forward_blacking_block_warns_instead_of_asserting(self) -> None:
        """An active tile mask on a renderer with no tile grid must not trip a bare assert."""
        h, w = 8, 8
        rd_cam = _make_tileless_rendering_data(h, w)
        rd_cam.active_tile_mask = torch.zeros(1, 2, 2, dtype=torch.bool, device=TEST_DEVICE)

        rendered_rgb = torch.full((h * w, 3), 0.25, device=TEST_DEVICE)

        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite._camera_renderer = _TileGeometryLessRenderer()
        composite.post_processings = []  # type: ignore[assignment]
        composite.saturate_radiance = False
        # forward() gates the ray_flags assignment on this toggle; keep it off so the stub batch
        # (which carries active_tile_mask directly, no labels) skips that path.
        composite.config = SimpleNamespace(  # type: ignore[assignment]
            renderer=SimpleNamespace(active_tile_mask=ActiveTileMaskConfig(enabled=False))
        )

        def _forward_gaussians(**_kwargs) -> GaussiansCompositeReturn:
            return GaussiansCompositeReturn(
                rendered_cam=GaussiansRenderReturn(
                    rgb=rendered_rgb.clone(),
                    opacity=torch.ones(h * w, 1, device=TEST_DEVICE),
                )
            )

        composite.forward_gaussians = _forward_gaussians  # type: ignore[assignment, method-assign]

        batch = DataAndRenderingBatch(
            data=DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], lidar=None, camera=None),
            rendering=RenderingBatch(camera=rd_cam, lidar=None),
        )

        with self.assertLogs(gaussians_composite_module.log, level="WARNING") as logs:
            out = composite.forward(batch, render_background=False, enable_pp=False, clamp_output=False)

        self.assertTrue(any("skipping inactive-tile blacking" in line for line in logs.output))
        assert out.rendered_cam is not None and out.rendered_cam.rgb is not None
        # Nothing was blacked: without a tile size there is no pixel mask to apply.
        self.assertTrue(torch.equal(out.rendered_cam.rgb, rendered_rgb))


class _TileGeometryRenderer(_TileGeometryLessRenderer):
    """Camera renderer with a fixed 4x4 tile geometry (enough to resolve a tile grid)."""

    def active_tile_mask_tile_size(self, width: int, height: int) -> tuple[int, int]:
        del width, height
        return 4, 4


class TestInactiveTileBlacking(unittest.TestCase):
    """The blacking half of the inactive-tile contract, at composite level.

    ``CAMERA_INVALID_ONLY`` promises a skipped tile's RGB is *black*, not background-filled,
    and the blacking block promises that holds identically with and without background
    rendering. gsplat's own test pins that the renderer leaves masked tiles at zero; this
    pins the composite's half -- that it re-blacks them after ``self.background(...)`` has
    composited over the whole frame, and that it blacks exactly the inactive pixels.
    """

    def test_blacking_overrides_composited_background_at_a_non_multiple_resolution(self) -> None:
        # 6x6 at a 4x4 tile size: deliberately NOT a multiple, so the mask expansion
        # (repeat_interleave -> 8x8, cropped [:, :6, :6]) is exercised on its failure mode.
        # A checkerboard mask makes a mis-cropped or transposed expansion black the wrong
        # pixels rather than merely the wrong number of them.
        h, w = 6, 6
        rendered_value, background_value = 0.25, 0.5
        composited_value = rendered_value + background_value

        rd_cam = _make_tileless_rendering_data(h, w)
        rd_cam.active_tile_mask = torch.tensor([[[True, False], [False, True]]], dtype=torch.bool, device=TEST_DEVICE)

        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite._camera_renderer = _TileGeometryRenderer()
        composite.post_processings = []  # type: ignore[assignment]
        composite.saturate_radiance = False
        composite.config = SimpleNamespace(  # type: ignore[assignment]
            renderer=SimpleNamespace(active_tile_mask=ActiveTileMaskConfig(enabled=False))
        )

        def _forward_gaussians(**_kwargs) -> GaussiansCompositeReturn:
            return GaussiansCompositeReturn(
                rendered_cam=GaussiansRenderReturn(
                    rgb=torch.full((h * w, 3), rendered_value, device=TEST_DEVICE),
                    opacity=torch.zeros(h * w, 1, device=TEST_DEVICE),
                )
            )

        composite.forward_gaussians = _forward_gaussians  # type: ignore[assignment, method-assign]

        def _background(rays_d: torch.Tensor, out: GaussiansCompositeReturn, _is_training: bool) -> None:
            """Stand-in for a real background: composites over *every* pixel, inactive included."""
            assert out.rendered_cam is not None and out.rendered_cam.rgb is not None
            self.assertEqual(rays_d.shape, (h * w, 3))
            out.rendered_cam.rgb += background_value

        composite.background = _background  # type: ignore[assignment]

        batch = DataAndRenderingBatch(
            data=DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], lidar=None, camera=None),
            rendering=RenderingBatch(camera=rd_cam, lidar=None),
        )

        out = composite.forward(batch, render_background=True, enable_pp=False, clamp_output=False)

        assert out.rendered_cam is not None and out.rendered_cam.rgb is not None
        rgb = out.rendered_cam.rgb.reshape(h, w, 3)

        # Expected coverage written out by hand rather than re-deriving it with
        # repeat_interleave, so the test does not restate the implementation it pins.
        # Tile (0,0) -> rows 0:4, cols 0:4 (active); tile (1,1) -> rows 4:6, cols 4:6 (active,
        # and clipped by the crop). The other two tiles are inactive.
        expected_active = torch.zeros(h, w, dtype=torch.bool, device=TEST_DEVICE)
        expected_active[0:4, 0:4] = True
        expected_active[4:6, 4:6] = True

        self.assertTrue(
            torch.equal(rgb[~expected_active], torch.zeros_like(rgb[~expected_active])),
            "inactive-tile pixels must be forced to black even though the background "
            f"composited {composited_value} over them:\n{rgb[..., 0]}",
        )
        self.assertTrue(
            torch.equal(
                rgb[expected_active],
                torch.full_like(rgb[expected_active], composited_value),
            ),
            f"active-tile pixels must keep the background-composited value untouched:\n{rgb[..., 0]}",
        )

    def test_blacking_warns_and_skips_when_the_mask_does_not_cover_the_render(self) -> None:
        """A mask whose expansion does not cover the rgb rows warns; it must not assert.

        The bare ``assert`` this replaced also disappeared under ``python -O``, taking the
        shape check with it and leaving a raw IndexError from the masked assignment.
        """
        h, w = 8, 8
        rd_cam = _make_tileless_rendering_data(h, w)
        # A 1x1 tile grid expands to 4x4 pixels -- short of the 8x8 render.
        rd_cam.active_tile_mask = torch.zeros(1, 1, 1, dtype=torch.bool, device=TEST_DEVICE)

        rendered_rgb = torch.full((h * w, 3), 0.25, device=TEST_DEVICE)

        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite._camera_renderer = _TileGeometryRenderer()
        composite.post_processings = []  # type: ignore[assignment]
        composite.saturate_radiance = False
        composite.config = SimpleNamespace(  # type: ignore[assignment]
            renderer=SimpleNamespace(active_tile_mask=ActiveTileMaskConfig(enabled=False))
        )

        def _forward_gaussians(**_kwargs) -> GaussiansCompositeReturn:
            return GaussiansCompositeReturn(
                rendered_cam=GaussiansRenderReturn(
                    rgb=rendered_rgb.clone(),
                    opacity=torch.ones(h * w, 1, device=TEST_DEVICE),
                )
            )

        composite.forward_gaussians = _forward_gaussians  # type: ignore[assignment, method-assign]

        batch = DataAndRenderingBatch(
            data=DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], lidar=None, camera=None),
            rendering=RenderingBatch(camera=rd_cam, lidar=None),
        )

        with self.assertLogs(gaussians_composite_module.log, level="WARNING") as logs:
            out = composite.forward(batch, render_background=False, enable_pp=False, clamp_output=False)

        self.assertTrue(any("skipping inactive-tile blacking" in line for line in logs.output))
        assert out.rendered_cam is not None and out.rendered_cam.rgb is not None
        self.assertTrue(torch.equal(out.rendered_cam.rgb, rendered_rgb))


def _make_mapping(
    cam_tiles_y: int,
    cam_tiles_x: int,
    lidar_rows: int = 4,
    lidar_cols: int = 16,
    lidar_tile_grid_shape: tuple[int, int] = (2, 2),
) -> CameraToLidarTileMapping:
    """A structurally valid mapping for a ``cam_tiles_y x cam_tiles_x`` camera grid."""
    n_lidar_tiles = lidar_tile_grid_shape[0] * lidar_tile_grid_shape[1]
    n_camera_tiles = cam_tiles_y * cam_tiles_x
    return CameraToLidarTileMapping(
        mapping=torch.ones(n_lidar_tiles, n_camera_tiles, dtype=torch.bool, device=TEST_DEVICE),
        n_lidar_tiles=n_lidar_tiles,
        n_camera_tiles=n_camera_tiles,
        cam_tiles_y=cam_tiles_y,
        cam_tiles_x=cam_tiles_x,
        n_bins_azimuth=lidar_tile_grid_shape[1],
        n_bins_elevation=lidar_tile_grid_shape[0],
        lidar_tile_grid_shape=lidar_tile_grid_shape,
        element_to_tile=torch.zeros(lidar_rows, lidar_cols, dtype=torch.int64, device=TEST_DEVICE),
    )


class _SentinelMappingRenderer(_TileGeometryRenderer):
    """Lidar-side renderer that hands back a known mapping object, so tests can tell a rebuilt
    cache entry apart from the stale one it replaced."""

    def __init__(self, sentinel: CameraToLidarTileMapping) -> None:
        super().__init__()
        self.sentinel = sentinel
        self.build_calls = 0

    def build_camera_to_lidar_tile_mapping(self, *args, **kwargs) -> CameraToLidarTileMapping:  # type: ignore[override, no-untyped-def]
        del args, kwargs
        self.build_calls += 1
        return self.sentinel


def _make_active_tiling_composite(
    sentinel: CameraToLidarTileMapping | None = None,
    strategy: ActiveTileMaskStrategy = ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP,
) -> GaussiansComposite:
    """Stub composite exercising only the active-tile-mask path of ``forward_gaussians``.

    Defaults to the overlap strategy because that is the one with a lidar half to exercise; the
    real ``ActiveTileMaskConfig`` is used rather than a namespace so the composite's
    ``strategy is ActiveTileMaskStrategy...`` comparison sees a genuine enum member.
    """
    composite = GaussiansComposite.__new__(GaussiansComposite)
    torch.nn.Module.__init__(composite)
    composite._camera_renderer = _TileGeometryRenderer()
    composite._lidar_renderer = (
        _SentinelMappingRenderer(sentinel) if sentinel is not None else _TileGeometryLessRenderer()
    )
    composite._cam_to_lidar_tile_mappings = {}
    composite.config = SimpleNamespace(  # type: ignore[assignment]
        renderer=SimpleNamespace(active_tile_mask=ActiveTileMaskConfig(enabled=True, strategy=strategy))
    )
    return composite


def test_eager_and_lazy_active_tile_mapping_use_one_cache_builder() -> None:
    sentinel = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
    composite = _make_active_tiling_composite(sentinel)
    rd_cam = _make_tileless_rendering_data(8, 8)
    rd_lidar = _make_tileless_rendering_data(4, 16)

    composite.prewarm_camera_to_lidar_tile_mapping(3, rd_cam, rd_lidar)
    composite.prewarm_camera_to_lidar_tile_mapping(3, rd_cam, rd_lidar)
    renderer = cast(_SentinelMappingRenderer, composite._lidar_renderer)

    assert composite._cam_to_lidar_tile_mappings[3] is sentinel
    assert renderer.build_calls == 1

    composite._cam_to_lidar_tile_mappings.clear()
    lazy = composite._ensure_cam_to_lidar_tile_mapping(3, rd_cam, rd_lidar)

    assert lazy is sentinel
    assert composite._cam_to_lidar_tile_mappings[3] is sentinel
    assert renderer.build_calls == 2


def test_fresh_active_tile_mapping_geometry_mismatch_raises(caplog: pytest.LogCaptureFixture) -> None:
    invalid_mapping = _make_mapping(cam_tiles_y=3, cam_tiles_x=3)
    composite = _make_active_tiling_composite(invalid_mapping)
    rd_cam = _make_tileless_rendering_data(8, 8)
    rd_lidar = _make_tileless_rendering_data(4, 16)

    with pytest.raises(RuntimeError, match="Fresh camera-to-LiDAR tile mapping for sensor 3"):
        composite._ensure_cam_to_lidar_tile_mapping(3, rd_cam, rd_lidar)

    renderer = cast(_SentinelMappingRenderer, composite._lidar_renderer)
    assert renderer.build_calls == 1
    assert 3 not in composite._cam_to_lidar_tile_mappings
    assert "Rejecting freshly built camera-to-LiDAR mapping for sensor 3" in caplog.text


def test_first_active_tile_render_reuses_the_prewarmed_mapping() -> None:
    sentinel = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
    composite = _make_active_tiling_composite(sentinel)
    rd_cam = _make_tileless_rendering_data(8, 8)
    rd_lidar = _make_tileless_rendering_data(4, 16)
    rd_cam.active_tile_mask = torch.ones(1, 2, 2, dtype=torch.bool, device=TEST_DEVICE)

    composite.prewarm_camera_to_lidar_tile_mapping(3, rd_cam, rd_lidar)
    composite._maybe_compute_active_tile_masks(rd_cam, rd_lidar, cam_sensor_idx=3)

    renderer = cast(_SentinelMappingRenderer, composite._lidar_renderer)
    assert renderer.build_calls == 1
    assert composite._cam_to_lidar_tile_mappings[3] is sentinel
    assert rd_lidar.active_tile_mask is not None
    assert rd_lidar.active_tile_mask.shape == (1, 2, 2)


def test_unsupported_eager_mapping_preserves_the_lazy_noop_cache() -> None:
    composite = _make_active_tiling_composite()
    rd_cam = _make_tileless_rendering_data(8, 8)
    rd_lidar = _make_tileless_rendering_data(4, 16)

    composite.prewarm_camera_to_lidar_tile_mapping(3, rd_cam, rd_lidar)
    assert 3 in composite._cam_to_lidar_tile_mappings
    assert composite._cam_to_lidar_tile_mappings[3] is None
    assert composite._ensure_cam_to_lidar_tile_mapping(3, rd_cam, rd_lidar) is None


class TestActiveTilingCacheRevalidation(unittest.TestCase):
    """A cache hit is revalidated against the frame actually being rendered."""

    def _run(self, composite: GaussiansComposite, camera_mask: torch.Tensor) -> RenderingData:
        rd_cam = _make_tileless_rendering_data(8, 8)  # 8x8 px / 4x4 tiles -> 2x2 camera tiles
        rd_cam.active_tile_mask = camera_mask
        rd_lidar = _make_tileless_rendering_data(4, 16)
        composite._maybe_compute_active_tile_masks(rd_cam, rd_lidar, cam_sensor_idx=3)
        return rd_lidar

    def test_stale_camera_tile_count_is_detected_and_rebuilt(self) -> None:
        sentinel = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
        composite = _make_active_tiling_composite(sentinel)
        stale = _make_mapping(cam_tiles_y=3, cam_tiles_x=3)  # 9 tiles vs the current 4
        composite._cam_to_lidar_tile_mappings[3] = stale

        mask = torch.ones(1, 2, 2, dtype=torch.bool, device=TEST_DEVICE)
        with self.assertLogs(gaussians_composite_module.log, level="WARNING") as logs:
            self._run(composite, mask)

        self.assertIs(composite._cam_to_lidar_tile_mappings[3], sentinel)
        joined = "\n".join(logs.output)
        self.assertIn("does not match the current geometry", joined)
        self.assertIn("sensor 3", joined)
        self.assertIn("3x3", joined)  # cached grid
        self.assertIn("2x2", joined)  # current grid
        self.assertIn("tile size 4x4", joined)
        self.assertIn("Repeated clears for this sensor indicate geometry thrashing", joined)

    def test_transposed_camera_grid_with_equal_tile_count_is_detected(self) -> None:
        """The numel collision the un-guarded code accepted: 1x4 cached vs 2x2 current."""
        sentinel = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
        composite = _make_active_tiling_composite(sentinel)
        stale = _make_mapping(cam_tiles_y=1, cam_tiles_x=4)
        self.assertEqual(stale.n_camera_tiles, 4)  # identical tile *count* to the current grid
        composite._cam_to_lidar_tile_mappings[3] = stale

        mask = torch.ones(1, 2, 2, dtype=torch.bool, device=TEST_DEVICE)
        with self.assertLogs(gaussians_composite_module.log, level="WARNING") as logs:
            self._run(composite, mask)

        self.assertIs(composite._cam_to_lidar_tile_mappings[3], sentinel)
        self.assertIn("1x4", "\n".join(logs.output))

    def test_stale_lidar_ray_grid_is_detected(self) -> None:
        sentinel = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
        composite = _make_active_tiling_composite(sentinel)
        # Cached mapping was built against a different lidar ray grid.
        stale = _make_mapping(cam_tiles_y=2, cam_tiles_x=2, lidar_rows=8, lidar_cols=8)
        composite._cam_to_lidar_tile_mappings[3] = stale

        mask = torch.ones(1, 2, 2, dtype=torch.bool, device=TEST_DEVICE)
        with self.assertLogs(gaussians_composite_module.log, level="WARNING") as logs:
            self._run(composite, mask)

        self.assertIs(composite._cam_to_lidar_tile_mappings[3], sentinel)
        self.assertIn("lidar ray grid", "\n".join(logs.output))

    def test_matching_cached_mapping_is_reused_without_rebuild(self) -> None:
        sentinel = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
        composite = _make_active_tiling_composite(sentinel)
        fresh = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
        composite._cam_to_lidar_tile_mappings[3] = fresh

        mask = torch.ones(1, 2, 2, dtype=torch.bool, device=TEST_DEVICE)
        rd_lidar = self._run(composite, mask)

        self.assertIs(composite._cam_to_lidar_tile_mappings[3], fresh)
        lidar_renderer = cast(_SentinelMappingRenderer, composite._lidar_renderer)
        self.assertEqual(lidar_renderer.build_calls, 0)
        assert rd_lidar.active_tile_mask is not None
        self.assertEqual(tuple(rd_lidar.active_tile_mask.shape), (1, 2, 2))


class TestActiveTilingStrategyGate(unittest.TestCase):
    """``CAMERA_INVALID_ONLY`` masks camera tiles and stops there.

    The strategy's documented promise is that the cam->lidar mapping precompute never runs and
    lidar supervision is untouched, so the gate is asserted directly rather than inferred from
    the absence of a lidar mask.
    """

    def test_camera_invalid_only_never_builds_the_cam_to_lidar_mapping(self) -> None:
        sentinel = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
        composite = _make_active_tiling_composite(sentinel, strategy=ActiveTileMaskStrategy.CAMERA_INVALID_ONLY)

        rd_cam = _make_tileless_rendering_data(8, 8)  # 8x8 px / 4x4 tiles -> 2x2 camera tiles
        rd_cam.ray_flags = torch.zeros(1, 8, 8, 1, dtype=torch.int32, device=TEST_DEVICE)
        rd_lidar = _make_tileless_rendering_data(4, 16)

        composite._maybe_compute_active_tile_masks(rd_cam, rd_lidar, cam_sensor_idx=3)

        # The camera half still runs: the mask is derived from ray_flags at the renderer's grid.
        assert rd_cam.active_tile_mask is not None
        self.assertEqual(tuple(rd_cam.active_tile_mask.shape), (1, 2, 2))
        # The lidar half does not: no mapping built or cached, no lidar mask attached.
        self.assertIsNone(rd_lidar.active_tile_mask)
        self.assertEqual(composite._cam_to_lidar_tile_mappings, {})
        self.assertEqual(cast(_SentinelMappingRenderer, composite._lidar_renderer).build_calls, 0)

    def test_camera_invalid_only_keeps_a_stale_cached_mapping_untouched(self) -> None:
        """Not even the cache revalidation runs, so nothing can rebuild a mapping behind the
        strategy's back (a preset switched back to the overlap strategy revalidates then)."""
        sentinel = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
        composite = _make_active_tiling_composite(sentinel, strategy=ActiveTileMaskStrategy.CAMERA_INVALID_ONLY)
        stale = _make_mapping(cam_tiles_y=3, cam_tiles_x=3)
        composite._cam_to_lidar_tile_mappings[3] = stale

        rd_cam = _make_tileless_rendering_data(8, 8)
        rd_cam.active_tile_mask = torch.ones(1, 2, 2, dtype=torch.bool, device=TEST_DEVICE)
        rd_lidar = _make_tileless_rendering_data(4, 16)

        composite._maybe_compute_active_tile_masks(rd_cam, rd_lidar, cam_sensor_idx=3)

        self.assertIs(composite._cam_to_lidar_tile_mappings[3], stale)
        self.assertEqual(cast(_SentinelMappingRenderer, composite._lidar_renderer).build_calls, 0)
        self.assertIsNone(rd_lidar.active_tile_mask)

    def test_supplied_camera_mask_short_circuits_derivation(self) -> None:
        """``ActiveTileMaskConfig``'s precedence rule: a mask already on ``RenderingData`` is
        used as-is, never AND/OR-ed with one derived from ray_flags (which here would be
        all-active, so a re-derivation would be visible)."""
        composite = _make_active_tiling_composite(strategy=ActiveTileMaskStrategy.CAMERA_INVALID_ONLY)
        rd_cam = _make_tileless_rendering_data(8, 8)
        supplied = torch.zeros(1, 2, 2, dtype=torch.bool, device=TEST_DEVICE)
        rd_cam.active_tile_mask = supplied
        rd_cam.ray_flags = torch.zeros(1, 8, 8, 1, dtype=torch.int32, device=TEST_DEVICE)

        composite._maybe_compute_active_tile_masks(rd_cam, None, cam_sensor_idx=3)

        self.assertIs(rd_cam.active_tile_mask, supplied)
        self.assertFalse(bool(rd_cam.active_tile_mask.any()))


class TestActiveTilingRayFlagsPopulation(unittest.TestCase):
    """``forward()`` copies ``labels.flags`` onto ``RenderingData.ray_flags``.

    This is the camera half's only input, so it is keyed off ``enabled`` alone and must stay
    strategy-independent -- moving it under the lidar-side strategy gate would silently disable
    camera masking for ``CAMERA_INVALID_ONLY`` presets.
    """

    @staticmethod
    def _batch(rd_cam: RenderingData, flags: torch.Tensor) -> DataAndRenderingBatch:
        return DataAndRenderingBatch(
            data=DataBatch(
                idx=0,
                worker_id=None,
                sequence_id=["dummy"],
                lidar=None,
                camera=DataBatch.Camera(meta=[], labels=CameraFrameLabels(flags=flags)),
            ),
            rendering=RenderingBatch(camera=rd_cam, lidar=None),
        )

    def _composite(self, strategy: ActiveTileMaskStrategy, h: int, w: int) -> GaussiansComposite:
        composite = _make_active_tiling_composite(strategy=strategy)
        composite.post_processings = []  # type: ignore[assignment]
        composite.saturate_radiance = False

        def _forward_gaussians(**_kwargs) -> GaussiansCompositeReturn:
            return GaussiansCompositeReturn(
                rendered_cam=GaussiansRenderReturn(
                    rgb=torch.full((h * w, 3), 0.25, device=TEST_DEVICE),
                    opacity=torch.ones(h * w, 1, device=TEST_DEVICE),
                )
            )

        composite.forward_gaussians = _forward_gaussians  # type: ignore[assignment, method-assign]
        return composite

    def test_ray_flags_are_populated_under_every_strategy(self) -> None:
        h, w = 8, 8
        for strategy in ActiveTileMaskStrategy:
            with self.subTest(strategy=strategy):
                rd_cam = _make_tileless_rendering_data(h, w)
                self.assertIsNone(rd_cam.ray_flags)
                flags = torch.zeros(1, h, w, 1, dtype=torch.int32, device=TEST_DEVICE)
                composite = self._composite(strategy, h, w)

                composite.forward(
                    self._batch(rd_cam, flags), render_background=False, enable_pp=False, clamp_output=False
                )

                self.assertIs(rd_cam.ray_flags, flags)

    def test_ray_flags_are_not_populated_when_the_feature_is_off(self) -> None:
        h, w = 8, 8
        rd_cam = _make_tileless_rendering_data(h, w)
        composite = self._composite(ActiveTileMaskStrategy.CAMERA_INVALID_ONLY, h, w)
        composite.config.renderer.active_tile_mask = ActiveTileMaskConfig(enabled=False)

        composite.forward(
            self._batch(rd_cam, torch.zeros(1, h, w, 1, dtype=torch.int32, device=TEST_DEVICE)),
            render_background=False,
            enable_pp=False,
            clamp_output=False,
        )

        self.assertIsNone(rd_cam.ray_flags)

    def test_flags_not_matching_the_ray_grid_raise(self) -> None:
        """The assignment bypasses ``RenderingData.__post_init__``, so the shape agreement with
        ``rays`` is checked here or not at all."""
        rd_cam = _make_tileless_rendering_data(8, 8)
        composite = self._composite(ActiveTileMaskStrategy.CAMERA_INVALID_ONLY, 8, 8)
        mismatched = torch.zeros(1, 4, 4, 1, dtype=torch.int32, device=TEST_DEVICE)

        with self.assertRaises(ValueError) as ctx:
            composite.forward(
                self._batch(rd_cam, mismatched), render_background=False, enable_pp=False, clamp_output=False
            )

        self.assertIn("must be a 4D tensor (B, height, width, 1) matching", str(ctx.exception))
        self.assertIsNone(rd_cam.ray_flags)


class TestActiveTilingObservability(unittest.TestCase):
    """The feature announces itself, reports what it masks, and says why it did nothing."""

    def _rendering_data(self) -> tuple[RenderingData, RenderingData]:
        return _make_tileless_rendering_data(8, 8), _make_tileless_rendering_data(4, 16)

    def test_activation_and_fraction_logs_are_emitted_once_across_two_forwards(self) -> None:
        sentinel = _make_mapping(cam_tiles_y=2, cam_tiles_x=2)
        composite = _make_active_tiling_composite(sentinel)

        # Half the camera tiles inactive -> a readable 50% in the log.
        mask = torch.tensor([[[True, True], [False, False]]], device=TEST_DEVICE)

        with self.assertLogs(gaussians_composite_module.log, level="INFO") as logs:
            for _ in range(2):
                rd_cam, rd_lidar = self._rendering_data()
                rd_cam.active_tile_mask = mask.clone()
                composite._maybe_compute_active_tile_masks(rd_cam, rd_lidar, cam_sensor_idx=3)

        joined = logs.output
        self.assertEqual(sum("Active tile masking is ENABLED" in line for line in joined), 1)
        self.assertEqual(sum("First masked camera render" in line for line in joined), 1)
        self.assertEqual(sum("First derived lidar mask" in line for line in joined), 1)
        self.assertTrue(any("50.0% of the 4 camera tiles are inactive" in line for line in joined))

    def test_flag_on_without_ray_flags_warns_once_and_builds_no_mask(self) -> None:
        composite = _make_active_tiling_composite()

        with self.assertLogs(gaussians_composite_module.log, level="WARNING") as logs:
            for _ in range(2):
                rd_cam, rd_lidar = self._rendering_data()
                self.assertIsNone(rd_cam.ray_flags)
                composite._maybe_compute_active_tile_masks(rd_cam, rd_lidar, cam_sensor_idx=3)
                self.assertIsNone(rd_cam.active_tile_mask)
                self.assertIsNone(rd_lidar.active_tile_mask)

        self.assertEqual(sum("no camera ray_flags" in line for line in logs.output), 1)
        self.assertEqual(composite._cam_to_lidar_tile_mappings, {})

    def test_flag_on_without_tile_geometry_warns_once_on_the_camera_mask_path(self) -> None:
        composite = _make_active_tiling_composite()
        composite._camera_renderer = _TileGeometryLessRenderer()

        with self.assertLogs(gaussians_composite_module.log, level="WARNING") as logs:
            for _ in range(2):
                rd_cam, rd_lidar = self._rendering_data()
                rd_cam.ray_flags = torch.zeros(1, 8, 8, 1, dtype=torch.int32, device=TEST_DEVICE)
                composite._maybe_compute_active_tile_masks(rd_cam, rd_lidar, cam_sensor_idx=3)
                self.assertIsNone(rd_cam.active_tile_mask)

        self.assertEqual(sum("exposes no tile geometry" in line for line in logs.output), 1)

    def test_forward_with_flag_on_and_no_labels_does_not_crash(self) -> None:
        """Flag on, camera-only batch without labels: ray_flags stays None, forward still runs."""
        h, w = 8, 8
        rd_cam = _make_tileless_rendering_data(h, w)
        composite = _make_active_tiling_composite()
        composite.post_processings = []  # type: ignore[assignment]
        composite.saturate_radiance = False

        def _forward_gaussians(**_kwargs) -> GaussiansCompositeReturn:
            return GaussiansCompositeReturn(
                rendered_cam=GaussiansRenderReturn(
                    rgb=torch.full((h * w, 3), 0.25, device=TEST_DEVICE),
                    opacity=torch.ones(h * w, 1, device=TEST_DEVICE),
                )
            )

        composite.forward_gaussians = _forward_gaussians  # type: ignore[assignment, method-assign]

        batch = DataAndRenderingBatch(
            data=DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], lidar=None, camera=None),
            rendering=RenderingBatch(camera=rd_cam, lidar=None),
        )
        out = composite.forward(batch, render_background=False, enable_pp=False, clamp_output=False)

        self.assertIsNone(rd_cam.ray_flags)
        self.assertIsNone(rd_cam.active_tile_mask)
        assert out.rendered_cam is not None and out.rendered_cam.rgb is not None


def test_resolve_lidar_idx_uses_logical_id_not_calibration_storage_key() -> None:
    calibration = SimpleNamespace(unique_sensor_idx=7, logical_sensor_name="lidar_free")
    datasource = SimpleNamespace(
        datasource=None,
        state=SimpleNamespace(
            rig_trajectories=SimpleNamespace(lidar_calibrations={"lidar_free@sequence": calibration})
        ),
    )

    assert GaussiansComposite._resolve_lidar_idx_to_id(cast(Any, datasource)) == {7: "lidar_free"}


class _MotionCollectionBackend:
    """Deterministic CPU boundary for testing the real public collection wiring."""

    def __init__(self) -> None:
        self.interpolation_inputs: list[collect.TracksInterpolationData] = []
        self.layers: list[LayerDataRigid] = []

    def interpolate_tracks_poses(
        self, data: collect.TracksInterpolationData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.interpolation_inputs.append(data)
        n_tracks = data.tracks_packinfo.shape[0]
        if isinstance(data.timestamps_data, collect.TracksTimestampsGlobalData):
            timestamps = torch.full(
                (n_tracks,),
                data.timestamps_data.timestamp,
                dtype=torch.int64,
                device=data.tracks_poses.device,
            )
        elif isinstance(data.timestamps_data, collect.TracksTimestampsPerTrackData):
            timestamps = data.timestamps_data.timestamps
        else:
            assert isinstance(data.timestamps_data, collect.TracksTimestampsEstimationData)
            timestamps = torch.full(
                (n_tracks,),
                data.timestamps_data.default_timestamp,
                dtype=torch.int64,
                device=data.tracks_poses.device,
            )
        return (
            data.tracks_poses[data.tracks_packinfo[:, 0].long()],
            data.tracks_packinfo[:, 1] > 0,
            timestamps,
        )

    def collect(self, data: LayersData, layer_indices=None, segments=None) -> CollectorResult:
        assert segments is None
        self.layers = [cast(LayerDataRigid, layer) for layer in data.layers]
        return CollectorResult(
            positions=torch.cat([layer.positions + layer.poses[layer.tracks_ids.long(), :3] for layer in self.layers]),
            rotations=torch.cat([layer.rotations for layer in self.layers]),
            scales=torch.cat([layer.scales for layer in self.layers]),
            densities=torch.cat([layer.densities for layer in self.layers]),
            extra_signal=torch.cat([layer.extra_signal for layer in self.layers]),
            camera_extra_signal=torch.cat([layer.camera_extra_signal for layer in self.layers]),
            lidar_extra_signal=torch.cat([layer.lidar_extra_signal for layer in self.layers]),
            features=torch.cat(
                [torch.cat((layer.features_albedo, layer.features_specular), -1) for layer in self.layers]
            ),
        )


def _make_public_motion_collection(device: torch.device = torch.device("cpu"), *, vehicle_lights: bool = False):
    """Bare real nodes/composite; replace only initialization and the CPU kernel boundary."""
    tracks = _make_cuboid_tracks(["a", "b", "c"], ["automobile"] * 3, device=device)
    backend = _MotionCollectionBackend()
    delta = torch.nn.Parameter(torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3) / 10)
    nodes = {}
    parameters = {}
    for name, ids in (("body", [2, 0, 1, 2]), ("lights", [1, 2, 0])):
        model_type = VehicleLightRigidGaussianModel if vehicle_lights and name == "lights" else RigidGaussianModel
        node = model_type.__new__(model_type)
        torch.nn.Module.__init__(node)
        node.config = cast(
            Any,
            SimpleNamespace(
                motion_source_layer=None if name == "body" else "body",
                fourier_features_dim=1,
                tracks_timestamp_interpolation_cache=True,
            ),
        )
        node._motion_source_ref = None
        node.cuboid_tracks = tracks if name == "body" else copy.deepcopy(tracks)
        node.is_static = False
        node.nearest_neighbor_track_for_lidar = False
        node.tracks_timestamp_estimation_ray_stride = 1
        node._split_boundary_track_indices = torch.empty(0, dtype=torch.long, device=device)
        node._split_internal_start_timestamps_us = torch.empty(0, dtype=torch.long, device=device)
        node._split_internal_end_timestamps_us = torch.empty(0, dtype=torch.long, device=device)
        node.track_scales = None
        node.track_albedos = None
        node.collector = cast(Any, backend)
        node.tracks_calib = cast(Any, lambda value: value)
        n = len(ids)
        params = {
            "positions": torch.nn.Parameter(torch.arange(n * 3, dtype=torch.float32, device=device).reshape(n, 3) / 20),
            "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1),
            "scales": torch.zeros(n, 3, device=device),
            "densities": torch.zeros(n, 1, device=device),
            "extra_signal": torch.empty(n, 0, device=device),
            "camera_extra_signal": torch.empty(n, 0, device=device),
            "lidar_extra_signal": torch.empty(n, 0, device=device),
            "features_albedo": torch.zeros(n, 3, device=device),
            "features_specular": torch.zeros(n, 45, device=device),
            "gaussian_cuboid_ids": torch.tensor(ids, dtype=torch.int32, device=device),
        }
        cast(Any, node).get_parameters = Mock(side_effect=lambda values=params: dict(values))
        nodes[name], parameters[name] = node, params
    owner, dependent = nodes["body"], nodes["lights"]
    dependent.bind_motion_source(owner)

    def calibrated_tracks(current: CuboidTracks) -> CuboidTracks:
        translation = delta[: current.n_tracks].repeat_interleave(current.tracks_packinfo[:, 1].long(), dim=0)
        poses = current.tracks_poses.vec() + torch.cat(
            (translation, torch.zeros_like(translation[:, :1]).expand(-1, 4)), -1
        )
        return CuboidTracks.Ops.clone(current, tracks_poses=lt.SE3.InitFromVec(poses))

    # Both unedited preparation and transform-only edits must apply the same
    # nonzero calibration; leave the production motion-data helper intact.
    owner.tracks_calib = cast(Any, calibrated_tracks)
    composite = GaussiansComposite.__new__(GaussiansComposite)
    torch.nn.Module.__init__(composite)
    composite.gaussians_nodes = TypedModuleDict(nodes)
    composite.obj_track_ids = cast(Any, {name: SimpleNamespace(is_empty=lambda: False) for name in nodes})
    composite.config = cast(Any, SimpleNamespace(renderer=SimpleNamespace(cache_static_gaussians=False)))
    composite.collector = cast(Any, backend)
    composite.debug_viz = False
    composite._edit_overrides = cast(Any, object())
    composite._collected_edit_layouts = {}
    composite._visibility_local_ranges = None
    timestamps = torch.tensor([[0, 0]], dtype=torch.int64, device=device)
    rendering = RenderingData(
        rays=torch.zeros(1, 4, 4, 6, device=device),
        sensor_model_parameters=cast(Any, [object()]),
        poses_tquat_startend=torch.zeros(1, 2, 7, device=device),
        timestamps_startend_us=timestamps,
        timestamps_startend_us_cpu=timestamps.cpu(),
        rays_timestamps_us=None,
    )
    return composite, owner, dependent, rendering, delta, parameters, backend


class TestGaussiansCompositePreparedMotion(unittest.TestCase):
    @staticmethod
    def _use_native_collector(
        composite: GaussiansComposite, owner: RigidGaussianModel, dependent: RigidGaussianModel
    ) -> None:
        config = LayerConfigRigid(
            rotation_activation=RotationActivation.NORMALIZE,
            scale_activation=ScaleActivation.EXP,
            density_activation=DensityActivation.SIGMOID,
            fourier_features_dim=1,
            embed_config=None,
        )
        collector = CreateGaussianParameterCollector(
            LayersConfig(
                layers=[config, config],
                extra_signal_dim=0,
                camera_extra_signal_dim=0,
                lidar_extra_signal_dim=0,
                albedo_dim=3,
                specular_dim=45,
                sh_features_dtype="fp16",
            )
        )
        composite.collector = owner.collector = dependent.collector = collector

    def test_public_collection_shares_owner_motion_for_both_orders_and_dependent_only(self) -> None:
        composite, owner, dependent, rendering, _delta, params, backend = _make_public_motion_collection()
        for selected in (["body", "lights"], ["lights", "body"], ["lights"]):
            with (
                self.subTest(selected=selected),
                patch.object(owner, "prepare_motion", wraps=owner.prepare_motion) as prepare,
                patch.object(owner, "get_layer_data", wraps=owner.get_layer_data) as owner_data,
                patch.object(dependent, "get_layer_data", wraps=dependent.get_layer_data) as dependent_data,
            ):
                result = composite.collect_gaussian_parameters(
                    rendering, is_training_batch=True, gaussian_nodes=selected
                )
                prepare.assert_called_once_with(rendering, None, True, tracks_edit=None)
                data_calls = {"body": owner_data, "lights": dependent_data}
                contexts = [data_calls[name].call_args.args[0] for name in selected]
                self.assertIsNotNone(contexts[0].prepared_motion)
                for context in contexts:
                    self.assertIs(context.prepared_motion, contexts[0].prepared_motion)
                if len(contexts) == 2:
                    self.assertIsNot(contexts[0], contexts[1])
                    self.assertIsNot(contexts[0].additional_data, contexts[1].additional_data)
                for layer, name in zip(backend.layers, selected):
                    torch.testing.assert_close(layer.tracks_ids, params[name]["gaussian_cuboid_ids"])
                    self.assertIs(layer.poses, backend.layers[0].poses)
                    self.assertIs(layer.keep_mask, backend.layers[0].keep_mask)
                self.assertEqual(result["positions"].shape[0], sum(len(params[name]["positions"]) for name in selected))
                self.assertIs(dependent.get_motion_source(), owner)

    def test_standalone_dependent_collection_prepares_fresh_owner_motion(self) -> None:
        composite, _owner, dependent, rendering, delta, params, backend = _make_public_motion_collection()
        dependent.tracks_calib = cast(Any, Mock(side_effect=AssertionError("Dependent calibration must not run")))
        ids = params["lights"]["gaussian_cuboid_ids"]
        with torch.no_grad():
            for step in range(2):
                with self.subTest(step=step):
                    context = BaseGaussianModel.CollectionContext(
                        rendering_data=rendering, is_training_batch=False, tracks_edit=None
                    )
                    self.assertIsNone(context.prepared_motion)
                    backend.interpolation_inputs.clear()
                    layer = dependent.get_layer_data(context, dict(params["lights"]))
                    self.assertEqual(len(backend.interpolation_inputs), 1)
                    torch.testing.assert_close(layer.poses[:, :3], delta[:3])
                    torch.testing.assert_close(layer.tracks_ids, ids)
                    self.assertEqual(layer.keep_mask.tolist(), [True, True, True])
                    self.assertIs(context.additional_data["interpolated_mask"][0], layer.keep_mask)
                    self.assertFalse(layer.poses.requires_grad)
                    expected = composite.collect_gaussian_parameters(rendering, gaussian_nodes=["lights"])
                    torch.testing.assert_close(layer.positions + layer.poses[ids.long(), :3], expected["positions"])
                    delta.add_(1)

    def test_standalone_dependent_actor_edit_prepares_fresh_authoritative_motion_and_filters_rows(self) -> None:
        _composite, owner, dependent, rendering, _delta, params, backend = _make_public_motion_collection()
        previous_poses = None
        with (
            torch.no_grad(),
            patch.object(
                owner,
                "_get_calibrated_motion_data",
                side_effect=AssertionError("Supplied edits must not be calibrated"),
            ),
        ):
            for offset in (0.0, 5.0):
                with self.subTest(offset=offset):
                    # Packed edit order differs from the owner's [a, b, c].
                    edited = _make_cuboid_tracks(["c", "a"], ["automobile"] * 2, device=torch.device("cpu"))
                    poses = edited.tracks_poses.vec()
                    poses[:2, 0] = 7.0 + offset
                    poses[2:, 0] = 3.0 + offset
                    context = BaseGaussianModel.CollectionContext(
                        rendering_data=rendering,
                        is_training_batch=False,
                        tracks_edit=LayerTrackIds.Edit(cuboid_tracks=edited, mapping=torch.tensor([2, 0])),
                    )
                    self.assertIsNone(context.prepared_motion)
                    backend.interpolation_inputs.clear()
                    layer = dependent.get_layer_data(context, dict(params["lights"]))
                    self.assertEqual(len(backend.interpolation_inputs), 1)
                    self.assertIsNot(layer.poses, previous_poses)
                    previous_poses = layer.poses
                    torch.testing.assert_close(
                        layer.poses[[0, 2], :3], torch.tensor([[3.0 + offset, 0.0, 0.0], [7.0 + offset, 0.0, 0.0]])
                    )
                    torch.testing.assert_close(layer.tracks_ids, torch.tensor([2, 0], dtype=torch.int32))
                    torch.testing.assert_close(context.kept_rows, torch.tensor([1, 2]))
                    torch.testing.assert_close(layer.positions, params["lights"]["positions"][[1, 2]])
                    self.assertEqual(layer.keep_mask.tolist(), [True, False, True])
                    self.assertIs(context.additional_data["interpolated_mask"][0], layer.keep_mask)

    def _assert_vehicle_light_appearance_with_shared_motion(self, device: torch.device, *, edited: bool) -> None:
        composite, owner, light, rendering, delta, params, backend = _make_public_motion_collection(
            device, vehicle_lights=True
        )
        self.assertIsInstance(light, VehicleLightRigidGaussianModel)
        frame_times = [[10, 20, 40], [15, 25, 35, 45], [100, 130]]
        light.cuboid_tracks = CuboidTracks.Factory.from_numpy(
            tracks_id=["a", "b", "c"],
            tracks_poses=[np.repeat(np.eye(4, dtype=np.float32)[None], len(times), axis=0) for times in frame_times],
            tracks_timestamps_us=[np.asarray(times, dtype=np.int64) for times in frame_times],
            tracks_label_class=["automobile"] * 3,
            tracks_flags=[TrackFlags.DYNAMIC] * 3,
            cuboids_dims=[np.ones(3, dtype=np.float32)] * 3,
            device=device,
        )
        original_appearance = light.cuboid_tracks
        # Coalesced owner samples would choose column zero at t=20 for every track.
        owner.cuboid_tracks.tracks_timestamps_us.copy_(torch.tensor([10, 40] * 3, device=device))
        colors = torch.arange(3 * 4 * 3, dtype=torch.float32, device=device).reshape(3, 4, 3) / 100
        params["lights"]["features_albedo"] = colors
        edited_tracks = CuboidTracks.Ops.subset_from_tracks_id(owner.cuboid_tracks, ["b", "a"]) if edited else None
        if edited_tracks is not None:
            edited_tracks.tracks_poses.vec()[:, 0] = 9.0
        light_rows = torch.tensor([0, 2] if edited else [0, 1, 2], device=device)
        for timestamp, expected_slots in ((20, [0, 0, 1]), (30, [1, 0, 1])):
            with self.subTest(timestamp=timestamp), torch.no_grad():
                rendering.timestamps_startend_us.fill_(timestamp)
                rendering.timestamps_startend_us_cpu.fill_(timestamp)
                backend.interpolation_inputs.clear()
                result = composite.collect_gaussian_parameters(
                    rendering, gaussian_nodes=["lights", "body"], edited_cuboid_tracks=edited_tracks
                )
                self.assertEqual(len(backend.interpolation_inputs), 1)
                light_data, body_data = backend.layers
                self.assertIs(light_data.poses, body_data.poses)
                self.assertIs(light_data.keep_mask, body_data.keep_mask)
                if edited:
                    torch.testing.assert_close(
                        light_data.poses[:2, :3], torch.tensor([[9.0, 0.0, 0.0]] * 2, device=device)
                    )
                    torch.testing.assert_close(composite._collected_edit_layouts["cam"]["lights"].kept_rows, light_rows)
                    torch.testing.assert_close(
                        composite._collected_edit_layouts["cam"]["body"].kept_rows, torch.tensor([1, 2], device=device)
                    )
                else:
                    torch.testing.assert_close(light_data.poses[:, :3], delta[:3])
                torch.testing.assert_close(light_data.tracks_ids, params["lights"]["gaussian_cuboid_ids"][light_rows])
                expected_colors = colors[torch.arange(3, device=device), torch.tensor(expected_slots, device=device)][
                    light_rows
                ]
                torch.testing.assert_close(light_data.features_albedo, expected_colors)
                torch.testing.assert_close(result["features"][: len(light_rows), :3], expected_colors)
                self.assertIsNone(light_data.embed_data)
                self.assertIs(light.cuboid_tracks, original_appearance)

    @unittest.skipUnless(torch.cuda.is_available(), "Vehicle-light frame slots require CUDA")
    def test_public_vehicle_light_collection_uses_local_appearance_frames_with_shared_motion(self) -> None:
        self._assert_vehicle_light_appearance_with_shared_motion(torch.device("cuda"), edited=False)

    @unittest.skipUnless(torch.cuda.is_available(), "Packed track subsetting requires CUDA")
    def test_public_edited_vehicle_light_collection_keeps_local_frames_and_filtered_row_layouts(self) -> None:
        self._assert_vehicle_light_appearance_with_shared_motion(torch.device("cuda"), edited=True)

    def _assert_optimizer_freshness(self, device: torch.device, native: bool, *, transform_edits: bool = False) -> None:
        composite, owner, dependent, rendering, delta, params, _backend = _make_public_motion_collection(device)
        if native:
            self._use_native_collector(composite, owner, dependent)
        targets = [delta, params["body"]["positions"], params["lights"]["positions"]]
        optimizer = torch.optim.SGD(targets, lr=0.01)
        previous = None
        for step, selected in enumerate((["body", "lights"], ["lights", "body"], ["body", "lights"])):
            optimizer.zero_grad(set_to_none=True)
            edit_kwargs: dict[str, Any] = (
                {
                    "track_transform_ids": ["a"],
                    "track_transforms": torch.tensor([[float(step + 1), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device),
                }
                if transform_edits
                else {}
            )
            with patch.object(owner, "prepare_motion", wraps=owner.prepare_motion) as prepare:
                actual = composite.collect_gaussian_parameters(
                    rendering, is_training_batch=True, gaussian_nodes=selected, **edit_kwargs
                )
                prepare.assert_called_once()
                # Separate public collections are the recompute reference, including native layer-index routing.
                expected = [
                    composite.collect_gaussian_parameters(
                        rendering, is_training_batch=True, gaussian_nodes=[name], **edit_kwargs
                    )
                    for name in selected
                ]
                self.assertEqual(prepare.call_count, 3)
            expected_positions = torch.cat([item["positions"] for item in expected])
            torch.testing.assert_close(actual["positions"], expected_positions, rtol=0, atol=0)
            expected_grads = torch.autograd.grad(expected_positions.square().sum(), targets)
            actual["positions"].square().sum().backward()
            for target, expected_grad in zip(targets, expected_grads):
                self.assertIsNotNone(target.grad)
                torch.testing.assert_close(target.grad, expected_grad, rtol=1e-5, atol=1e-6)
            if step == 2:
                assert previous is not None
                self.assertFalse(torch.equal(actual["positions"].detach(), previous))
            if step == 0:
                previous = actual["positions"].detach().clone()
            optimizer.step()

    def test_public_collection_recomputes_after_optimizer_step_and_sums_gradients(self) -> None:
        self._assert_optimizer_freshness(torch.device("cpu"), native=False)

    def test_public_transform_edit_recomputes_after_optimizer_step_and_sums_calibration_gradients(self) -> None:
        self._assert_optimizer_freshness(torch.device("cpu"), native=False, transform_edits=True)

    @unittest.skipUnless(torch.cuda.is_available(), "Native Slang collection requires CUDA")
    def test_public_collection_native_slang_shared_pose_backward(self) -> None:
        for transform_edits in (False, True):
            with self.subTest(transform_edits=transform_edits):
                self._assert_optimizer_freshness(torch.device("cuda"), native=True, transform_edits=transform_edits)

    @unittest.skipUnless(torch.cuda.is_available(), "Native Slang collection and packed track subsetting require CUDA")
    def test_public_native_reordered_actor_edits_share_motion_and_sum_supplied_pose_gradients(self) -> None:
        device = torch.device("cuda")
        composite, owner, dependent, rendering, calibration_delta, params, _backend = _make_public_motion_collection(
            device
        )
        self._use_native_collector(composite, owner, dependent)
        edited_delta = torch.nn.Parameter(torch.arange(9, dtype=torch.float32, device=device).reshape(3, 3) + 2)
        targets = [edited_delta, params["body"]["positions"], params["lights"]["positions"]]
        optimizer = torch.optim.SGD(targets, lr=0.01)
        for track_names, selected in ((["c", "a", "b"], ["body", "lights"]), (["b", "c"], ["lights", "body"])):
            with self.subTest(track_names=track_names, selected=selected):
                optimizer.zero_grad(set_to_none=True)
                track_indices = torch.tensor(
                    [owner.cuboid_tracks.tracks_id.index(name) for name in track_names], device=device
                )

                def collect_selected(names: list[str]) -> dict[str, torch.Tensor]:
                    base = _make_cuboid_tracks(track_names, ["automobile"] * len(track_names), device=device)
                    translation = edited_delta[track_indices].repeat_interleave(2, dim=0)
                    poses = base.tracks_poses.vec() + torch.cat(
                        (translation, torch.zeros_like(translation[:, :1]).expand(-1, 4)), -1
                    )
                    edited = CuboidTracks.Ops.clone(base, tracks_poses=lt.SE3.InitFromVec(poses))
                    return composite.collect_gaussian_parameters(
                        rendering,
                        is_training_batch=True,
                        gaussian_nodes=names,
                        edited_cuboid_tracks=edited,
                        track_transform_ids=[track_names[0]],
                        track_transforms=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device),
                    )

                with (
                    patch.object(owner, "prepare_motion", wraps=owner.prepare_motion) as prepare,
                    patch.object(
                        owner,
                        "create_tracks_edit_from_cuboid_tracks",
                        wraps=owner.create_tracks_edit_from_cuboid_tracks,
                    ) as create_edit,
                    patch.object(
                        owner,
                        "get_calibrated_cuboid_tracks",
                        side_effect=AssertionError("Supplied edits must not be calibrated"),
                    ),
                    patch.object(composite.collector, "collect", wraps=composite.collector.collect) as native_collect,
                ):
                    actual = collect_selected(selected)
                    prepare.assert_called_once()
                    create_edit.assert_called_once()
                    layers = cast(LayersData, native_collect.call_args.args[0]).layers
                    self.assertIs(cast(LayerDataRigid, layers[0]).poses, cast(LayerDataRigid, layers[1]).poses)
                    self.assertIs(cast(LayerDataRigid, layers[0]).keep_mask, cast(LayerDataRigid, layers[1]).keep_mask)
                    layouts = composite._collected_edit_layouts["cam"]
                    expected_parts = []
                    for name in selected:
                        ids = params[name]["gaussian_cuboid_ids"].long()
                        kept_rows = torch.where(torch.isin(ids, track_indices))[0]
                        torch.testing.assert_close(layouts[name].kept_rows, kept_rows)
                        self.assertEqual(layouts[name].layer_size, kept_rows.numel())
                        kept_ids = ids[kept_rows]
                        transform = (kept_ids == track_indices[0])[:, None] * torch.tensor(
                            [1.0, 0.0, 0.0], device=device
                        )
                        expected_parts.append(params[name]["positions"][kept_rows] + edited_delta[kept_ids] + transform)
                    torch.testing.assert_close(actual["positions"], torch.cat(expected_parts))
                    # Separate public collections prepare independent motion graphs.
                    reference = torch.cat([collect_selected([name])["positions"] for name in selected])
                    self.assertEqual(prepare.call_count, 3)
                    torch.testing.assert_close(actual["positions"], reference, rtol=0, atol=0)
                    reference_grads = torch.autograd.grad(reference.square().sum(), targets)
                actual["positions"].square().sum().backward()
                self.assertIsNone(calibration_delta.grad)
                for target, expected_grad in zip(targets, reference_grads):
                    self.assertIsNotNone(target.grad)
                    assert target.grad is not None
                    self.assertTrue(torch.isfinite(target.grad).all())
                    self.assertGreater(torch.count_nonzero(target.grad).item(), 0)
                    torch.testing.assert_close(target.grad, expected_grad, rtol=1e-5, atol=1e-6)
                optimizer.step()

    def test_public_edited_collection_prepares_each_owner_separately(self) -> None:
        composite, owner, dependent, rendering, delta, params, backend = _make_public_motion_collection()
        other, other_owner, other_dependent, _rendering, other_delta, other_params, _backend = (
            _make_public_motion_collection()
        )
        # Deliberately reuse the track names in a different motion owner. The
        # request is common, but each owner's calibration must still be applied.
        other_dependent.config.motion_source_layer = "other_body"
        other_owner.collector = other_dependent.collector = cast(Any, backend)
        for name, node in (("other_body", other_owner), ("other_lights", other_dependent)):
            composite.gaussians_nodes[name] = node
            composite.obj_track_ids[name] = other.obj_track_ids["body" if node is other_owner else "lights"]
        with torch.no_grad():
            other_delta.add_(10)
        selected = ["lights", "other_body", "body", "other_lights"]
        previous_poses = None
        for translation in (1.0, 4.0):
            with (
                self.subTest(translation=translation),
                patch.object(owner, "prepare_motion", wraps=owner.prepare_motion) as prepare,
                patch.object(other_owner, "prepare_motion", wraps=other_owner.prepare_motion) as other_prepare,
                patch.object(
                    owner, "get_calibrated_cuboid_tracks", wraps=owner.get_calibrated_cuboid_tracks
                ) as calibrate,
                patch.object(
                    other_owner, "get_calibrated_cuboid_tracks", wraps=other_owner.get_calibrated_cuboid_tracks
                ) as other_calibrate,
                torch.no_grad(),
            ):
                backend.interpolation_inputs.clear()
                result = composite.collect_gaussian_parameters(
                    rendering,
                    gaussian_nodes=selected,
                    track_transform_ids=["a"],
                    track_transforms=torch.tensor([[translation, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                )
                prepare.assert_called_once()
                other_prepare.assert_called_once()
                calibrate.assert_called_once_with()
                other_calibrate.assert_called_once_with()
                self.assertEqual(len(backend.interpolation_inputs), 2)
                first, second, third, fourth = backend.layers
                self.assertIs(first.poses, third.poses)
                self.assertIs(second.poses, fourth.poses)
                self.assertIsNot(first.poses, second.poses)
                self.assertIsNot(first.poses, previous_poses)
                previous_poses = first.poses
                expected = []
                for name in selected:
                    local_name = name.removeprefix("other_")
                    values = other_params[local_name] if name.startswith("other_") else params[local_name]
                    calibration = other_delta if name.startswith("other_") else delta
                    ids = values["gaussian_cuboid_ids"].long()
                    offset = (ids == 0)[:, None] * torch.tensor([translation, 0.0, 0.0])
                    expected.append(values["positions"] + calibration[ids] + offset)
                torch.testing.assert_close(result["positions"], torch.cat(expected))
        self.assertIs(dependent.get_motion_source(), owner)
        self.assertIs(other_dependent.get_motion_source(), other_owner)

    def test_public_collection_respects_owner_policy_in_train_no_grad_and_inference(self) -> None:
        composite, owner, dependent, rendering, _delta, _params, backend = _make_public_motion_collection()
        rendering.rays_timestamps_us = torch.zeros(1, 4, 4, 1, dtype=torch.int64)
        owner.tracks_timestamp_estimation_ray_stride = 4
        dependent.tracks_timestamp_estimation_ray_stride = 1
        dependent.is_static = True  # Dependent sampling policy must not replace its owner's.
        for mode, grad_context, training, expected_rays in (
            ("train", nullcontext, True, 16),
            ("no_grad", torch.no_grad, False, 4),
            ("inference", torch.inference_mode, False, 4),
        ):
            with self.subTest(mode=mode), grad_context():
                composite.train(training)
                backend.interpolation_inputs.clear()
                composite.collect_gaussian_parameters(rendering, is_training_batch=training)
                self.assertEqual(len(backend.interpolation_inputs), 1)
                data = cast(collect.TracksTimestampsEstimationData, backend.interpolation_inputs[0].timestamps_data)
                self.assertEqual(data.rays.shape[0], expected_rays)
                self.assertEqual(backend.layers[0].poses.requires_grad, training)

    def test_public_lidar_collection_uses_owner_nearest_neighbor_policy_with_and_without_edits(self) -> None:
        class LidarParameters:
            pass

        composite, owner, dependent, rendering, _delta, _params, backend = _make_public_motion_collection()
        rendering.sensor_model_parameters = cast(Any, [LidarParameters()])
        with (
            patch.object(gaussians_model_module, "RowOffsetStructuredSpinningLidarModelParameters", LidarParameters),
            torch.no_grad(),
        ):
            for nearest_neighbor in (False, True):
                owner.nearest_neighbor_track_for_lidar = nearest_neighbor
                dependent.nearest_neighbor_track_for_lidar = not nearest_neighbor
                for edited in (False, True):
                    with self.subTest(nearest_neighbor=nearest_neighbor, edited=edited):
                        backend.interpolation_inputs.clear()
                        composite.collect_gaussian_parameters(
                            rendering,
                            is_camera=False,
                            track_transform_ids=["a"] if edited else None,
                            track_transforms=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]) if edited else None,
                        )
                        self.assertEqual(len(backend.interpolation_inputs), 1)
                        self.assertIs(backend.layers[0].poses, backend.layers[1].poses)
                        self.assertIs(backend.layers[0].keep_mask, backend.layers[1].keep_mask)
                        for data in backend.interpolation_inputs:
                            self.assertEqual(data.nearest_neighbor, nearest_neighbor)

    @unittest.skipUnless(torch.cuda.is_available(), "Packed track subsetting requires CUDA")
    def test_public_collection_owner_track_append_edit_and_restore_preserves_layer_rows(self) -> None:
        device = torch.device("cuda")
        composite, owner, dependent, rendering, _delta, params, backend = _make_public_motion_collection(device)
        original_appearance = dependent.cuboid_tracks
        # State after owner-only asset insertion: its metadata and Gaussian rows grow; light appearance does not.
        owner.cuboid_tracks = _make_cuboid_tracks(["a", "b", "c", "new"], ["automobile"] * 4, device=device)
        for key, value in params["body"].items():
            params["body"][key] = torch.cat((value, value[:1]), 0)
        params["body"]["gaussian_cuboid_ids"][-1] = 3
        unedited = composite.collect_gaussian_parameters(rendering, gaussian_nodes=["body", "lights"])
        self.assertEqual(backend.layers[0].poses.shape[0], 4)
        self.assertIs(backend.layers[0].poses, backend.layers[1].poses)
        edited_tracks = CuboidTracks.Ops.subset_from_tracks_id(owner.cuboid_tracks, ["b", "new"])
        backend.interpolation_inputs.clear()
        with (
            patch.object(owner, "prepare_motion", wraps=owner.prepare_motion) as prepare,
            patch.object(
                owner, "create_tracks_edit_from_cuboid_tracks", wraps=owner.create_tracks_edit_from_cuboid_tracks
            ) as create_edit,
            patch.object(dependent, "create_tracks_edit_from_cuboid_tracks") as dependent_edit,
        ):
            composite.collect_gaussian_parameters(
                rendering,
                gaussian_nodes=["body", "lights"],
                edited_cuboid_tracks=edited_tracks,
                track_transform_ids=["new"],
                track_transforms=torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device),
            )
            prepare.assert_called_once()
            create_edit.assert_called_once_with(edited_tracks)
            dependent_edit.assert_not_called()
        self.assertEqual(len(backend.interpolation_inputs), 1)
        layouts = composite._collected_edit_layouts["cam"]
        torch.testing.assert_close(layouts["body"].kept_rows, torch.tensor([2, 4], device=device))
        torch.testing.assert_close(layouts["lights"].kept_rows, torch.tensor([0], device=device))
        self.assertEqual(layouts["body"].layer_size, 2)
        self.assertEqual(layouts["lights"].layer_offset, 2)
        self.assertEqual(layouts["lights"].layer_size, 1)
        torch.testing.assert_close(backend.layers[0].tracks_ids, torch.tensor([1, 3], dtype=torch.int32, device=device))
        torch.testing.assert_close(backend.layers[1].tracks_ids, torch.tensor([1], dtype=torch.int32, device=device))
        torch.testing.assert_close(backend.layers[0].poses[3, :3], torch.tensor([9.0, 0.0, 0.0], device=device))
        edited_poses = backend.layers[0].poses
        self.assertIs(edited_poses, backend.layers[1].poses)
        self.assertIs(backend.layers[0].keep_mask, backend.layers[1].keep_mask)
        self.assertIs(dependent.cuboid_tracks, original_appearance)
        restored = composite.collect_gaussian_parameters(rendering, gaussian_nodes=["body", "lights"])
        self.assertIsNot(backend.layers[0].poses, edited_poses)
        torch.testing.assert_close(restored["positions"], unedited["positions"])
        for layout in composite._collected_edit_layouts["cam"].values():
            self.assertIsNone(layout.kept_rows)
        self.assertIs(backend.layers[0].poses, backend.layers[1].poses)

    @unittest.skipUnless(torch.cuda.is_available(), "Native Slang collection and packed track subsetting require CUDA")
    def test_public_native_collection_selecting_only_appended_actor_removes_all_light_rows(self) -> None:
        device = torch.device("cuda")
        composite, owner, dependent, rendering, _delta, params, _backend = _make_public_motion_collection(device)
        self._use_native_collector(composite, owner, dependent)
        original_appearance = dependent.cuboid_tracks
        owner.cuboid_tracks = _make_cuboid_tracks(["a", "b", "c", "new"], ["automobile"] * 4, device=device)
        for key, value in params["body"].items():
            params["body"][key] = torch.cat((value, value[:1]), 0)
        params["body"]["gaussian_cuboid_ids"][-1] = 3
        # Supplied edits are authoritative even when the owner's calibration is nonzero.
        edited_tracks = CuboidTracks.Ops.subset_from_tracks_id(owner.cuboid_tracks, ["new"])
        edited_tracks.tracks_poses.vec()[:, 0] = 9.0
        translation = torch.tensor([9.0, 0.0, 0.0], device=device)
        for selected in (["body", "lights"], ["lights", "body"], ["lights"]):
            with (
                self.subTest(selected=selected),
                patch.object(owner, "prepare_motion", wraps=owner.prepare_motion) as prepare,
                patch.object(
                    owner, "create_tracks_edit_from_cuboid_tracks", wraps=owner.create_tracks_edit_from_cuboid_tracks
                ) as create_edit,
                patch.object(dependent, "create_tracks_edit_from_cuboid_tracks") as dependent_edit,
                patch.object(owner, "get_layer_data", wraps=owner.get_layer_data) as owner_data,
                patch.object(dependent, "get_layer_data", wraps=dependent.get_layer_data) as dependent_data,
                patch.object(composite.collector, "collect", wraps=composite.collector.collect) as native_collect,
            ):
                actual = composite.collect_gaussian_parameters(
                    rendering, gaussian_nodes=selected, edited_cuboid_tracks=edited_tracks
                )
                prepare.assert_called_once()
                create_edit.assert_called_once_with(edited_tracks)
                dependent_edit.assert_not_called()
                data_calls = {"body": owner_data, "lights": dependent_data}
                layers = cast(LayersData, native_collect.call_args.args[0]).layers
                layouts = composite._collected_edit_layouts["cam"]
                offset = 0
                for name, layer in zip(selected, layers):
                    context = data_calls[name].call_args.args[0]
                    self.assertIsNotNone(context.prepared_motion)
                    self.assertIs(context.tracks_edit, prepare.call_args.kwargs["tracks_edit"])
                    torch.testing.assert_close(context.tracks_edit.mapping, torch.tensor([3], device=device))
                    rigid_layer = cast(LayerDataRigid, layer)
                    self.assertIs(rigid_layer.poses, cast(LayerDataRigid, layers[0]).poses)
                    self.assertIs(rigid_layer.keep_mask, cast(LayerDataRigid, layers[0]).keep_mask)
                    rows = torch.tensor([4] if name == "body" else [], dtype=torch.long, device=device)
                    torch.testing.assert_close(layouts[name].kept_rows, rows)
                    self.assertEqual(layouts[name].layer_offset, offset)
                    self.assertEqual(layouts[name].layer_size, rows.numel())
                    torch.testing.assert_close(
                        layouts[name].collect_keep, torch.ones(rows.numel(), dtype=torch.bool, device=device)
                    )
                    torch.testing.assert_close(rigid_layer.tracks_ids, params[name]["gaussian_cuboid_ids"][rows])
                    torch.testing.assert_close(rigid_layer.poses[3, :3], translation)
                    self.assertEqual(rigid_layer.positions.shape[0], rows.numel())
                    offset += rows.numel()
                expected = (
                    params["body"]["positions"][4:5] + translation
                    if "body" in selected
                    else params["lights"]["positions"][:0]
                )
                torch.testing.assert_close(actual["positions"], expected)
                self.assertIs(dependent.cuboid_tracks, original_appearance)
                self.assertEqual(dependent.cuboid_tracks.tracks_id, ["a", "b", "c"])

    @unittest.skipUnless(torch.cuda.is_available(), "Native Slang collection requires CUDA")
    def test_public_native_transform_only_after_owner_append_retains_original_actors_and_light_metadata(self) -> None:
        device = torch.device("cuda")
        composite, owner, dependent, rendering, delta, params, _backend = _make_public_motion_collection(device)
        self._use_native_collector(composite, owner, dependent)
        original_appearance = dependent.cuboid_tracks
        appearance_poses = original_appearance.tracks_poses.vec().clone()
        appearance_timestamps = original_appearance.tracks_timestamps_us.clone()
        appearance_packinfo = original_appearance.tracks_packinfo.clone()
        owner.cuboid_tracks = _make_cuboid_tracks(["a", "b", "c", "new"], ["automobile"] * 4, device=device)
        for key, value in params["body"].items():
            params["body"][key] = torch.cat((value, value[:1]), 0)
        params["body"]["gaussian_cuboid_ids"][-1] = 3
        with torch.no_grad():
            delta.zero_()
        translation = torch.tensor([9.0, 0.0, 0.0], device=device)
        for target, target_index in (("new", 3), ("a", 0)):
            for selected in (["body", "lights"], ["lights", "body"], ["lights"]):
                with (
                    self.subTest(target=target, selected=selected),
                    patch.object(owner, "prepare_motion", wraps=owner.prepare_motion) as prepare,
                    patch.object(
                        owner, "get_calibrated_cuboid_tracks", wraps=owner.get_calibrated_cuboid_tracks
                    ) as calibrate,
                    patch.object(owner, "get_layer_data", wraps=owner.get_layer_data) as owner_data,
                    patch.object(dependent, "get_layer_data", wraps=dependent.get_layer_data) as dependent_data,
                    patch.object(composite.collector, "collect", wraps=composite.collector.collect) as native_collect,
                ):
                    actual = composite.collect_gaussian_parameters(
                        rendering,
                        gaussian_nodes=selected,
                        track_transform_ids=[target],
                        track_transforms=torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device),
                    )
                    prepare.assert_called_once()
                    calibrate.assert_called_once_with()
                    data_calls = {"body": owner_data, "lights": dependent_data}
                    layers = cast(LayersData, native_collect.call_args.args[0]).layers
                    expected_parts = []
                    for name, layer in zip(selected, layers):
                        context = data_calls[name].call_args.args[0]
                        self.assertIsNotNone(context.prepared_motion)
                        self.assertIs(context.tracks_edit, prepare.call_args.kwargs["tracks_edit"])
                        torch.testing.assert_close(context.tracks_edit.mapping, torch.arange(4, device=device))
                        rigid_layer = cast(LayerDataRigid, layer)
                        self.assertIs(rigid_layer.poses, cast(LayerDataRigid, layers[0]).poses)
                        self.assertIs(rigid_layer.keep_mask, cast(LayerDataRigid, layers[0]).keep_mask)
                        ids = params[name]["gaussian_cuboid_ids"]
                        torch.testing.assert_close(rigid_layer.tracks_ids, ids)
                        torch.testing.assert_close(rigid_layer.poses[target_index, :3], translation)
                        layout = composite._collected_edit_layouts["cam"][name]
                        self.assertEqual(layout.layer_size, ids.numel())
                        torch.testing.assert_close(layout.kept_rows, torch.arange(ids.numel(), device=device))
                        expected_parts.append(params[name]["positions"] + (ids == target_index)[:, None] * translation)
                    torch.testing.assert_close(actual["positions"], torch.cat(expected_parts))
        self.assertIs(dependent.cuboid_tracks, original_appearance)
        self.assertEqual(original_appearance.tracks_id, ["a", "b", "c"])
        torch.testing.assert_close(original_appearance.tracks_poses.vec(), appearance_poses)
        torch.testing.assert_close(original_appearance.tracks_timestamps_us, appearance_timestamps)
        torch.testing.assert_close(original_appearance.tracks_packinfo, appearance_packinfo)
        restored = composite.collect_gaussian_parameters(rendering, gaussian_nodes=["body", "lights"])
        torch.testing.assert_close(
            restored["positions"], torch.cat([params[name]["positions"] for name in ("body", "lights")])
        )
        for layout in composite._collected_edit_layouts["cam"].values():
            self.assertIsNone(layout.kept_rows)

    def test_public_identity_actor_edit_retains_owner_sampling_and_split_policy(self) -> None:
        composite, owner, dependent, rendering, delta, params, backend = _make_public_motion_collection()
        rendering.rays_timestamps_us = torch.zeros(1, 4, 4, 1, dtype=torch.int64)
        composite.eval()
        owner.tracks_timestamp_estimation_ray_stride = 4
        owner._split_boundary_track_indices = torch.tensor([2])
        owner._split_internal_start_timestamps_us = torch.tensor([-1])
        owner._split_internal_end_timestamps_us = torch.tensor([0])
        dependent.is_static = True
        dependent.nearest_neighbor_track_for_lidar = True
        dependent.tracks_timestamp_estimation_ray_stride = 1
        dependent._split_boundary_track_indices = torch.tensor([0])
        dependent._split_internal_start_timestamps_us = torch.tensor([-1])
        dependent._split_internal_end_timestamps_us = torch.tensor([0])
        with (
            torch.no_grad(),
            patch.object(
                dependent, "_get_interpolated_poses", wraps=dependent._get_interpolated_poses
            ) as local_interpolation,
        ):
            normal = composite.collect_gaussian_parameters(rendering)
            torch.testing.assert_close(
                normal["positions"],
                torch.cat(
                    [params[name]["positions"] + delta[params[name]["gaussian_cuboid_ids"].long()] for name in params]
                ),
            )
            normal_poses = [layer.poses.clone() for layer in backend.layers]
            normal_masks = [layer.keep_mask.clone() for layer in backend.layers]
            backend.interpolation_inputs.clear()
            edited = composite.collect_gaussian_parameters(
                rendering,
                track_transform_ids=["a"],
                track_transforms=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            )
            local_interpolation.assert_not_called()
        torch.testing.assert_close(edited["positions"], normal["positions"])
        self.assertEqual(len(backend.interpolation_inputs), 1)
        self.assertIs(backend.layers[0].poses, backend.layers[1].poses)
        self.assertIs(backend.layers[0].keep_mask, backend.layers[1].keep_mask)
        for layer, poses, mask in zip(backend.layers, normal_poses, normal_masks):
            torch.testing.assert_close(layer.poses, poses)
            torch.testing.assert_close(layer.keep_mask, mask)
            self.assertEqual(mask.tolist(), [True, True, False])
        for data in backend.interpolation_inputs:
            timestamps = cast(collect.TracksTimestampsEstimationData, data.timestamps_data)
            self.assertEqual(timestamps.rays.shape[0], 4)

    def test_public_collection_empty_edit_uses_fresh_owner_motion(self) -> None:
        composite, owner, _dependent, rendering, _delta, _params, _backend = _make_public_motion_collection()
        empty = CuboidTracks.Ops.subset_from_tracks_id(owner.cuboid_tracks, [])
        expected = composite.collect_gaussian_parameters(rendering)["positions"]
        with patch.object(owner, "prepare_motion", wraps=owner.prepare_motion) as prepare:
            actual = composite.collect_gaussian_parameters(rendering, edited_cuboid_tracks=empty)["positions"]
            prepare.assert_called_once()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
