# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the component-only machinery in ``GaussiansComposite``.

Two harnesses:

* **Bare instances** (``object.__new__`` + ``torch.nn.Module.__init__``)
  for the resolved-state / mask / freeze methods, which only read the
  component-only attributes they are given.
* **Patched-factory construction** for the ``__init__`` wiring: the real
  car2sim ``ModelConfig`` and real ``CuboidTracks`` fixtures drive the
  real constructor logic, while the module factories (calib, background,
  Gaussian models, strategy, renderers, post-processing) return
  lightweight stand-ins.
"""

import copy
import unittest

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest import mock

import numpy as np
import torch

from nurec.config.component_only import (
    ComponentOnlyComponentConfig,
    ComponentOnlyConfig,
    ComponentOnlyLayerConfig,
    ComponentOnlyTrackConfig,
    ComponentState,
    MaskSource,
    ScopeRegion,
)
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.models.background import BaseBackground
from nurec.models.calib import BaseCalib
from nurec.models.gaussians.gaussians_composite import GaussiansComposite, _make_row_grad_hook
from nurec.models.gaussians.gaussians_model import BaseGaussianModel, DeformableGaussianModel, RigidGaussianModel
from nurec.models.gaussians.initializations import BaseInitialization, NoPointsFoundException
from nurec.models.gaussians.renderers import BaseGaussianRenderer
from nurec.models.gaussians.strategies import BaseGaussianStrategy
from nurec.models.nn_extensions import TypedModuleDict
from nurec.models.post_processing import BasePostProcessing
from nurec.models.tracks_calib import SkipTracksCalib
from nurec.systems.active_tile_mapping_prewarmer import ActiveTileMappingPrewarmer
from nurec.systems.gaussians import GaussiansSystem
from nurec.utils.component_only_mask import (
    DISABLED_TRACKS_SUFFIX,
    ComponentOnlyMaskSpec,
    LayerRegionSpec,
    ProjectedTrackRegion,
    build_camera_loss_mask,
    build_mask_spec,
)
from nurec.utils.types import FrameConversion


def _ns(**kwargs: Any) -> Any:
    """SimpleNamespace typed as Any: duck-typed stand-ins for configs/structs."""
    return SimpleNamespace(**kwargs)


def _grad(param: torch.nn.Parameter | torch.nn.parameter.UninitializedParameter) -> torch.Tensor:
    assert param.grad is not None
    return param.grad


_CPU = torch.device("cpu")
# CuboidTracks pose interpolation (packed_ops) runs CUDA kernels — tests
# that interpolate need a GPU; pure bookkeeping tests stay on CPU.
_DEVICE = torch.device("cuda") if torch.cuda.is_available() else _CPU
_NEEDS_CUDA = unittest.skipUnless(torch.cuda.is_available(), "CuboidTracks interpolation requires CUDA")

# Class map for specs whose residual must identify sky/ego pixels.
_CLASS_MAP = {"road": 0, "sky": 3, "egocar": 4}


def _make_tracks(
    entries: list[tuple[str, str, tuple[float, float, float]]], device: torch.device = _CPU
) -> CuboidTracks:
    """Static 2m-cube tracks: (track_id, label_class, center). Valid over [0, 1s]."""
    tracks_id, poses, timestamps, labels, flags, dims = [], [], [], [], [], []
    for track_id, label, center in entries:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = center
        tracks_id.append(track_id)
        poses.append(np.stack([pose, pose]))
        timestamps.append(np.array([0, 1_000_000], np.int64))
        labels.append(label)
        flags.append(TrackFlags.DYNAMIC)
        dims.append(np.array([2, 2, 2], dtype=np.float32))
    return CuboidTracks.Factory.from_numpy(tracks_id, poses, timestamps, labels, flags, dims, device=device)


def _rigids_tracks(device: torch.device = _CPU) -> CuboidTracks:
    # t_near sits 10m in front of the origin (+y); t_far is out of view.
    return _make_tracks(
        [("t_near", "automobile", (0.0, 10.0, 0.0)), ("t_far", "automobile", (100.0, 100.0, 100.0))], device=device
    )


def _bare(component_only: Optional[ComponentOnlyConfig] = None) -> GaussiansComposite:
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_load_state_dict_post_hook(model._refresh_mcmc_runtime_state_after_load)
    model.register_load_state_dict_post_hook(model._invalidate_derived_render_caches_after_load)
    model.register_state_dict_pre_hook(model._raise_if_mcmc_publication_failed_before_state_dict)
    model.config = _ns(strategy=_ns(name="gsplat"))
    model.gaussians_strategy = _ns(refresh_runtime_scenes=lambda gaussians_nodes: None)
    model._fused_gather_generation_plan = None
    model._pending_fused_gather_generation_plan_refresh = None
    model.component_only = component_only
    model.config = cast(Any, _ns(layers={}))
    model._component_only_layer_tracks = {}
    model._component_only_track_states = {}
    model._component_only_track_expanded = {}
    model._component_only_globally_rendered_track_ids = frozenset()
    model._component_only_bbox_groups_cache = {}
    model._component_only_bbox_group_indices_cache = {}
    model._component_only_intersection_chunk_cache = {}
    model._component_only_frozen_layers = frozenset()
    model._component_only_mask_spec = None
    model._component_only_viewpoints = (torch.zeros((0, 3)), torch.zeros((0,), dtype=torch.int64))
    model._cam_to_lidar_tile_mappings = {}
    return model


def _enabled_config(**layers: ComponentOnlyLayerConfig) -> ComponentOnlyConfig:
    return ComponentOnlyConfig(enabled=True, layers=dict(layers))


def _resolve_and_store(model: GaussiansComposite, layer_id: str, tracks: CuboidTracks) -> Optional[Any]:
    """Resolve a track partition and store it (the resolver is now pure)."""
    partition = model._resolve_component_only_track_partition(layer_id, tracks)
    if partition is not None:
        model._component_only_track_states[layer_id] = partition.states
        model._component_only_track_expanded[layer_id] = partition.expanded
    return partition


def _model_layers_ns() -> dict:
    return {
        "background": _ns(class_labels=None, tracks=None),
        "dynamic_rigids": _ns(class_labels=None, tracks=object(), cuboid_tracks_expand=0.5),
    }


def _rendering_cam(directions: list[list[float]]) -> SimpleNamespace:
    """(1, 1, W, 6) rays from the origin with the given directions, mid-frame at 0.5s."""
    rays = torch.zeros((1, 1, len(directions), 6), device=_DEVICE)
    rays[0, 0, :, 3:] = torch.tensor(directions, device=_DEVICE)
    return _ns(rays=rays, timestamps_startend_us_cpu=torch.tensor([[0, 1_000_000]]))


class _FakeRigidNode(BaseGaussianModel):
    """Node with per-row track membership, as the frozen-track hooks expect.

    ``freeze_tracks`` is a spy: the composite owns only the per-Gaussian row
    hook and delegates per-track/pose params to the node, so tests assert the
    delegation here and exercise the real freeze in tracks_calib_test /
    gaussians_model.
    """

    def __init__(self, tracks: CuboidTracks, rows_per_track: int = 3):
        torch.nn.Module.__init__(self)  # intentionally skip BaseGaussianModel.__init__
        self.config = cast(Any, _ns(motion_source_layer=None))
        n = tracks.n_tracks * rows_per_track
        self.cuboid_tracks = tracks
        self.gaussian_cuboid_ids = torch.nn.Buffer(
            torch.arange(tracks.n_tracks, dtype=torch.int64).repeat_interleave(rows_per_track)
        )
        self.positions: torch.nn.Parameter = torch.nn.Parameter(torch.randn(n, 3))
        self.rotations = torch.nn.Parameter(torch.randn(n, 4))
        self.scales = torch.nn.Parameter(torch.randn(n, 3))
        self.densities = torch.nn.Parameter(torch.randn(n, 1))
        self.extra_signal = torch.nn.Parameter(torch.empty(n, 0))
        self.camera_extra_signal = torch.nn.Parameter(torch.empty(n, 0))
        self.lidar_extra_signal = torch.nn.Parameter(torch.empty(n, 0))
        self.token = torch.nn.Parameter(torch.randn(7))  # shape mismatch: never hooked
        self.freeze_tracks_calls: list[list[int]] = []

    def get_num_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_positions(self) -> torch.nn.Parameter:
        return self.positions

    def freeze_tracks(self, frozen_track_indices: Sequence[int]) -> None:
        self.freeze_tracks_calls.append([int(t) for t in frozen_track_indices])


class _SortableRigidNode(_FakeRigidNode):
    """``_FakeRigidNode`` carrying the tensors the Morton reorder permutes.

    Positions are laid out so the two tracks interleave once sorted, which is what makes a stale mask detectable.
    Left in the old row order it would freeze the wrong rows.
    """

    def __init__(self, tracks: CuboidTracks, rows_per_track: int = 3):
        super().__init__(tracks, rows_per_track)
        n = self.positions.shape[0]
        # Interleave the two tracks along x, so a Morton sort alternates between them.
        xs = torch.cat([torch.arange(rows_per_track) * 4.0 + 2.0 * t for t in range(tracks.n_tracks)])
        self.positions = torch.nn.Parameter(torch.stack([xs, torch.zeros(n), torch.zeros(n)], dim=1))
        self.scales = torch.nn.Parameter(torch.randn(n, 3))
        self.rotations = torch.nn.Parameter(torch.randn(n, 4))
        self.densities = torch.nn.Parameter(torch.randn(n, 1))
        self.extra_signal = torch.nn.Parameter(torch.randn(n, 2))
        self.camera_extra_signal = torch.nn.Parameter(torch.randn(n, 3))
        self.lidar_extra_signal = torch.nn.Parameter(torch.randn(n, 5))

    def _permute_gaussian_rows(self, order: torch.Tensor) -> None:
        # Mirrors RigidGaussianModel, track membership has to move with the rows or the rebuilt mask is stale.
        super()._permute_gaussian_rows(order)
        self.gaussian_cuboid_ids = self._reordered_buffer_rows(self.gaussian_cuboid_ids, order)


class TestRowGradHook(unittest.TestCase):
    def test_zeroes_frozen_rows_only(self) -> None:
        param = torch.nn.Parameter(torch.ones(4, 3))
        trainable = torch.tensor([True, False, True, False])
        param.register_hook(_make_row_grad_hook(trainable))
        param.sum().backward()
        self.assertTrue(torch.equal(_grad(param)[0], torch.ones(3)))
        self.assertTrue(torch.equal(_grad(param)[1], torch.zeros(3)))
        self.assertTrue(torch.equal(_grad(param)[3], torch.zeros(3)))

    def test_handles_1d_gradients(self) -> None:
        param = torch.nn.Parameter(torch.ones(3))
        param.register_hook(_make_row_grad_hook(torch.tensor([False, True, False])))
        param.sum().backward()
        self.assertTrue(torch.equal(_grad(param), torch.tensor([0.0, 1.0, 0.0])))


class TestComponentState(unittest.TestCase):
    def test_everything_enabled_without_component_only(self) -> None:
        model = _bare(None)
        for component in ["background", "road", "sky_model", "calib", "post_processing", "anything"]:
            self.assertEqual(model._component_state(component), ComponentState.enabled)

    def test_non_layer_components(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"dynamic_rigids": ComponentOnlyLayerConfig()},
            sky_model=ComponentOnlyComponentConfig(state=ComponentState.frozen),
            calib=ComponentOnlyComponentConfig(),
        )
        model = _bare(config)
        self.assertEqual(model._component_state("sky_model"), ComponentState.frozen)
        self.assertEqual(model._component_state("calib"), ComponentState.enabled)
        self.assertEqual(model._component_state("post_processing"), ComponentState.disabled)  # unlisted

    def test_unlisted_layer_is_disabled(self) -> None:
        model = _bare(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()))
        self.assertEqual(model._component_state("road"), ComponentState.disabled)

    def test_plain_layer_state_passthrough(self) -> None:
        model = _bare(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig(state=ComponentState.frozen)))
        self.assertEqual(model._component_state("dynamic_rigids"), ComponentState.frozen)

    def test_background_aggregates_scope(self) -> None:
        cases = [
            ({ScopeRegion.residual: ComponentState.disabled}, ComponentState.disabled),
            ({ScopeRegion.residual: ComponentState.frozen}, ComponentState.frozen),
            (
                {ScopeRegion.residual: ComponentState.frozen, ScopeRegion.bbox: ComponentState.enabled},
                ComponentState.enabled,
            ),
        ]
        for scope, expected in cases:
            with self.subTest(scope=scope):
                model = _bare(
                    _enabled_config(
                        background=ComponentOnlyLayerConfig(scope=scope),
                        dynamic_rigids=ComponentOnlyLayerConfig(),
                    )
                )
                self.assertEqual(model._component_state("background"), expected)

    def test_per_track_population_beats_layer_default(self) -> None:
        # disabled layer + enabled track -> the layer exists.
        model = _bare(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.enabled)},
                )
            )
        )
        self.assertEqual(model._component_state("dynamic_rigids"), ComponentState.enabled)

    def test_per_track_all_disabled_layer_is_disabled(self) -> None:
        model = _bare(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.disabled)},
                )
            )
        )
        self.assertEqual(model._component_state("dynamic_rigids"), ComponentState.disabled)

    def test_per_track_layer_wide_freeze_requires_no_enabled_track(self) -> None:
        frozen_all = _bare(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    state=ComponentState.frozen,
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.disabled)},
                )
            )
        )
        self.assertEqual(frozen_all._component_state("dynamic_rigids"), ComponentState.frozen)
        mixed = _bare(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    state=ComponentState.frozen,
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.enabled)},
                )
            )
        )
        self.assertEqual(mixed._component_state("dynamic_rigids"), ComponentState.enabled)


class TestBackgroundScope(unittest.TestCase):
    def test_full_training_default(self) -> None:
        self.assertEqual(_bare(None)._background_scope(), {ScopeRegion.residual: ComponentState.enabled})

    def test_unlisted_background_has_no_regions(self) -> None:
        model = _bare(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()))
        self.assertEqual(model._background_scope(), {})

    def test_explicit_scope(self) -> None:
        scope = {ScopeRegion.bbox: ComponentState.enabled}
        model = _bare(
            _enabled_config(background=ComponentOnlyLayerConfig(scope=scope), dynamic_rigids=ComponentOnlyLayerConfig())
        )
        self.assertEqual(model._background_scope(), scope)

    def test_state_sugar(self) -> None:
        model = _bare(_enabled_config(background=ComponentOnlyLayerConfig(state=ComponentState.frozen)))
        self.assertEqual(model._background_scope(), {ScopeRegion.residual: ComponentState.frozen})


class TestResolveTrackPartition(unittest.TestCase):
    def test_no_component_only_or_no_entry_or_no_tracks(self) -> None:
        tracks = _rigids_tracks()
        self.assertFalse(_bare(None)._resolve_component_only_track_partition("dynamic_rigids", tracks))
        model = _bare(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()))
        self.assertFalse(model._resolve_component_only_track_partition("road", tracks))  # no entry
        self.assertFalse(model._resolve_component_only_track_partition("dynamic_rigids", tracks))  # no overrides
        self.assertEqual(model._component_only_track_states, {})

    def test_unknown_track_id_fails_loud(self) -> None:
        model = _bare(
            _enabled_config(dynamic_rigids=ComponentOnlyLayerConfig(tracks={"nope": ComponentOnlyTrackConfig()}))
        )
        with self.assertRaises(ValueError) as cm:
            model._resolve_component_only_track_partition("dynamic_rigids", _rigids_tracks())
        self.assertIn("nope", str(cm.exception))
        self.assertIn("t_near", str(cm.exception))

    def test_states_and_variants_resolved_per_track(self) -> None:
        model = _bare(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    # layer default: bbox (rigids default); one track
                    # overrides both state and box variant.
                    tracks={
                        "t_far": ComponentOnlyTrackConfig(state=ComponentState.frozen, masks=[MaskSource.bbox_expanded])
                    },
                )
            )
        )
        partition = model._resolve_component_only_track_partition("dynamic_rigids", _rigids_tracks())
        self.assertIsNotNone(partition)
        assert partition is not None
        self.assertEqual(partition.states, {"t_near": ComponentState.enabled, "t_far": ComponentState.frozen})
        self.assertEqual(partition.expanded, {"t_near": False, "t_far": True})
        # The resolver is pure — it must not have mutated the composite's maps.
        self.assertEqual(model._component_only_track_states, {})

    def test_mask_only_override_inherits_layer_state(self) -> None:
        # A per-track override that sets only ``masks`` (state omitted) must
        # inherit the layer's state, not silently become enabled.
        for layer_state in (ComponentState.frozen, ComponentState.disabled):
            model = _bare(
                _enabled_config(
                    dynamic_rigids=ComponentOnlyLayerConfig(
                        state=layer_state,
                        tracks={"t_near": ComponentOnlyTrackConfig(masks=[MaskSource.bbox_expanded])},
                    )
                )
            )
            partition = model._resolve_component_only_track_partition("dynamic_rigids", _rigids_tracks())
            assert partition is not None
            self.assertEqual(partition.states["t_near"], layer_state)  # mask-only override
            self.assertEqual(partition.states["t_far"], layer_state)  # unlisted track

    def test_empty_subset_still_validates_track_ids(self) -> None:
        # Even when the layer has no dataset tracks, an explicit track override
        # id must be validated (raise), not silently ignored — the constructor
        # calls the resolver before the empty-subset early return.
        model = _bare(
            _enabled_config(dynamic_rigids=ComponentOnlyLayerConfig(tracks={"ghost": ComponentOnlyTrackConfig()}))
        )
        empty = CuboidTracks.Factory.empty(device=_CPU)
        with self.assertRaises(ValueError) as cm:
            model._resolve_component_only_track_partition("dynamic_rigids", empty)
        self.assertIn("ghost", str(cm.exception))


class _MaskHarness(unittest.TestCase):
    """Shared setup: a bare model with rigids tracks and a bbox-only spec."""

    def _model(
        self, config: ComponentOnlyConfig, resolve_tracks: bool = False, device: torch.device = _CPU
    ) -> GaussiansComposite:
        model = _bare(config)
        model.config = _ns(layers=_model_layers_ns(), strategy=_ns(name="gsplat"))
        model._component_only_layer_tracks = {"dynamic_rigids": _rigids_tracks(device=device)}
        model._component_only_mask_spec = build_mask_spec(config, _model_layers_ns(), _CLASS_MAP)
        if resolve_tracks:
            _resolve_and_store(model, "dynamic_rigids", model._component_only_layer_tracks["dynamic_rigids"])
        return model


class TestBboxTrackGroups(_MaskHarness):
    _SPEC_BBOX = LayerRegionSpec(sources=(MaskSource.bbox,), operation=None, semantic_class_ids=())
    _SPEC_EXPANDED = LayerRegionSpec(sources=(MaskSource.bbox_expanded,), operation=None, semantic_class_ids=())

    def test_no_tracks_returns_empty(self) -> None:
        model = self._model(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()))
        self.assertEqual(model._component_only_bbox_track_groups("road", self._SPEC_BBOX), [])
        model._component_only_layer_tracks["dynamic_rigids"] = CuboidTracks.Factory.empty(device=_CPU)
        self.assertEqual(model._component_only_bbox_track_groups("dynamic_rigids", self._SPEC_BBOX), [])

    def test_layer_level_variants(self) -> None:
        model = self._model(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()))
        ((idx, expand),) = model._component_only_bbox_track_groups("dynamic_rigids", self._SPEC_BBOX)
        self.assertEqual(idx.tolist(), [0, 1])
        self.assertEqual(expand, 0.0)
        ((_, expand),) = model._component_only_bbox_track_groups("dynamic_rigids", self._SPEC_EXPANDED)
        self.assertEqual(expand, 0.5)  # layer's cuboid_tracks_expand
        # No per-track overrides: the disabled partition is empty.
        self.assertEqual(
            model._component_only_bbox_track_groups("dynamic_rigids", self._SPEC_BBOX, disabled_partition=True), []
        )

    def test_per_track_partition_and_variant_grouping(self) -> None:
        model = self._model(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    tracks={
                        "t_near": ComponentOnlyTrackConfig(masks=[MaskSource.bbox_expanded]),
                        "t_far": ComponentOnlyTrackConfig(state=ComponentState.disabled),
                    }
                )
            ),
            resolve_tracks=True,
        )
        groups = model._component_only_bbox_track_groups("dynamic_rigids", self._SPEC_BBOX)
        self.assertEqual(len(groups), 1)  # t_near only, expanded variant
        idx, expand = groups[0]
        self.assertEqual(idx.tolist(), [0])
        self.assertEqual(expand, 0.5)
        carved = model._component_only_bbox_track_groups("dynamic_rigids", self._SPEC_BBOX, disabled_partition=True)
        ((idx, expand),) = carved
        self.assertEqual(idx.tolist(), [1])  # t_far, raw variant
        self.assertEqual(expand, 0.0)

    def test_mixed_variants_in_one_partition(self) -> None:
        model = self._model(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    tracks={"t_near": ComponentOnlyTrackConfig(masks=[MaskSource.bbox_expanded])}
                )
            ),
            resolve_tracks=True,
        )
        groups = model._component_only_bbox_track_groups("dynamic_rigids", self._SPEC_BBOX)
        self.assertEqual(len(groups), 2)  # raw group (t_far) + expanded group (t_near)
        (raw_idx, raw_expand), (exp_idx, exp_expand) = groups
        self.assertEqual((raw_idx.tolist(), raw_expand), ([1], 0.0))
        self.assertEqual((exp_idx.tolist(), exp_expand), ([0], 0.5))

    def test_rendered_track_wins_over_disabled_copy_in_another_layer(self) -> None:
        model = self._model(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    tracks={"t_far": ComponentOnlyTrackConfig(state=ComponentState.disabled)}
                )
            ),
            resolve_tracks=True,
        )
        # The same cleaned track id is rendered by another overlapping layer.
        model._component_only_globally_rendered_track_ids = frozenset({"t_far"})
        self.assertEqual(
            model._component_only_bbox_track_groups("dynamic_rigids", self._SPEC_BBOX, disabled_partition=True),
            [],
        )


class TestGlobalRenderedTrackResolution(unittest.TestCase):
    def test_non_disabled_occurrence_takes_precedence_across_layers(self) -> None:
        tracks = _make_tracks([("shared", "automobile", (0.0, 10.0, 0.0))])
        model = _bare(
            _enabled_config(
                rendered_layer=ComponentOnlyLayerConfig(),
                disabled_layer=ComponentOnlyLayerConfig(state=ComponentState.disabled),
            )
        )
        model._component_only_layer_tracks = {
            "rendered_layer": tracks,
            "disabled_layer": tracks,
        }
        self.assertEqual(model._resolve_component_only_rendered_track_ids(), frozenset({"shared"}))

    def test_layer_skipped_during_construction_does_not_count_as_rendered(self) -> None:
        tracks = _make_tracks([("shared", "automobile", (0.0, 10.0, 0.0))])
        model = _bare(
            _enabled_config(
                skipped_enabled_layer=ComponentOnlyLayerConfig(),
                constructed_disabled_layer=ComponentOnlyLayerConfig(state=ComponentState.disabled),
            )
        )
        model._component_only_layer_tracks = {
            "skipped_enabled_layer": tracks,
            "constructed_disabled_layer": tracks,
        }
        self.assertEqual(
            model._resolve_component_only_rendered_track_ids({"constructed_disabled_layer"}),
            frozenset(),
        )


class TestRayCuboidDepths(unittest.TestCase):
    @staticmethod
    def _subset(depths: list[float], n_tracks: int):
        intersections_ts = torch.full((1, n_tracks, 2), -1.0)
        for i, depth in enumerate(depths):
            intersections_ts[0, i] = torch.tensor([depth, depth + 1.0])
        return _ns(
            n_tracks=n_tracks,
            cuboids_dims=torch.ones((n_tracks, 3)),
            ray_intersection=mock.Mock(
                return_value=_ns(
                    intersections_cnt=torch.tensor([len(depths)], dtype=torch.int32),
                    intersections_ts=intersections_ts,
                )
            ),
        )

    def test_unsorted_intersections_reduce_nearest_and_preserve_timestamps(self) -> None:
        model = _bare(None)
        subset = self._subset([10.0, 5.0], 2)  # deliberately not nearest-first
        rays_o = torch.zeros((1, 3))
        rays_d = torch.tensor([[0.0, 1.0, 0.0]])
        rays_ts = torch.tensor([123], dtype=torch.int64)
        with mock.patch.object(CuboidTracks.Ops, "subset_from_indices", return_value=subset):
            projected = model._component_only_ray_cuboid_hits(
                cast(Any, object()), torch.tensor([0, 1], dtype=torch.int32), 0.0, rays_o, rays_d, rays_ts
            )
        self.assertEqual(projected.mask.tolist(), [True])
        self.assertEqual(projected.nearest_depth.tolist(), [5.0])
        call = subset.ray_intersection.call_args
        self.assertEqual(call.kwargs["max_intersections_per_ray"], 2)
        self.assertTrue(call.kwargs["with_intersections_ts"])
        self.assertTrue(torch.equal(call.args[2], rays_ts))

    def test_large_population_is_chunked_and_reduced_globally(self) -> None:
        model = _bare(None)
        model._COMPONENT_ONLY_INTERSECTION_TRACK_CHUNK_SIZE = 8
        subsets = [self._subset([9.0, 6.0], 8), self._subset([4.0], 2)]
        rays_o = torch.zeros((1, 3))
        rays_d = torch.tensor([[0.0, 1.0, 0.0]])
        rays_ts = torch.tensor([0], dtype=torch.int64)
        with mock.patch.object(CuboidTracks.Ops, "subset_from_indices", side_effect=subsets) as subset_mock:
            projected = model._component_only_ray_cuboid_hits(
                cast(Any, object()), torch.arange(10, dtype=torch.int32), 0.0, rays_o, rays_d, rays_ts
            )
        self.assertEqual(projected.nearest_depth.tolist(), [4.0])
        self.assertEqual(subset_mock.call_count, 2)
        self.assertEqual(subsets[0].ray_intersection.call_args.kwargs["max_intersections_per_ray"], 8)
        self.assertEqual(subsets[1].ray_intersection.call_args.kwargs["max_intersections_per_ray"], 2)

    def test_ray_and_track_chunks_preserve_exact_global_minimum(self) -> None:
        model = _bare(None)
        model._COMPONENT_ONLY_INTERSECTION_TRACK_CHUNK_SIZE = 2
        # float32, K=2 -> 83 bytes/ray + 16 fixed; this admits two rays.
        model._COMPONENT_ONLY_INTERSECTION_TEMP_MEMORY_BUDGET_BYTES = 182
        subsets = [self._depths_from_ray_ids(offset=5.0, n_tracks=2), self._depths_from_ray_ids(offset=3.0, n_tracks=2)]
        rays_o = torch.zeros((5, 3))
        rays_o[:, 0] = torch.arange(5)
        rays_d = torch.tensor([[0.0, 1.0, 0.0]]).repeat(5, 1)
        rays_ts = torch.arange(5, dtype=torch.int64)

        with mock.patch.object(CuboidTracks.Ops, "subset_from_indices", side_effect=subsets) as subset_mock:
            projected = model._component_only_ray_cuboid_hits(
                cast(Any, object()), torch.arange(4, dtype=torch.int32), 0.0, rays_o, rays_d, rays_ts
            )

        self.assertEqual(subset_mock.call_count, 2)
        self.assertEqual(projected.mask.tolist(), [True] * 5)
        self.assertEqual(projected.nearest_depth.tolist(), [3.0, 4.0, 5.0, 6.0, 7.0])
        for subset in subsets:
            self.assertEqual([call.args[0].shape[0] for call in subset.ray_intersection.call_args_list], [2, 2, 1])

    @staticmethod
    def _depths_from_ray_ids(offset: float, n_tracks: int) -> Any:
        def intersect(rays_o: torch.Tensor, *_args: Any, **_kwargs: Any) -> Any:
            depth = rays_o[:, 0] + offset
            intersections_ts = torch.full((rays_o.shape[0], n_tracks, 2), -1.0)
            intersections_ts[:, 0, 0] = depth
            intersections_ts[:, 0, 1] = depth + 1.0
            return _ns(
                intersections_cnt=torch.ones(rays_o.shape[0], dtype=torch.int32),
                intersections_ts=intersections_ts,
            )

        return _ns(
            n_tracks=n_tracks,
            cuboids_dims=torch.ones((n_tracks, 3)),
            ray_intersection=mock.Mock(side_effect=intersect),
        )

    def test_ray_chunk_budget_bounds_production_resolution_payload(self) -> None:
        model = _bare(None)
        num_rays = 2160 * 3840
        num_tracks = model._COMPONENT_ONLY_INTERSECTION_TRACK_CHUNK_SIZE

        float32_chunk = model._component_only_intersection_ray_chunk_size(num_rays, num_tracks, 4)
        float64_chunk = model._component_only_intersection_ray_chunk_size(num_rays, num_tracks, 8)

        self.assertLess(float32_chunk, num_rays)
        self.assertLess(float64_chunk, float32_chunk)
        for chunk_size, scalar_size in ((float32_chunk, 4), (float64_chunk, 8)):
            with self.subTest(scalar_size=scalar_size):
                payload = model._component_only_intersection_temporary_bytes(chunk_size, num_tracks, scalar_size)
                next_payload = model._component_only_intersection_temporary_bytes(
                    chunk_size + 1, num_tracks, scalar_size
                )
                self.assertLessEqual(payload, model._COMPONENT_ONLY_INTERSECTION_TEMP_MEMORY_BUDGET_BYTES)
                self.assertGreater(next_payload, model._COMPONENT_ONLY_INTERSECTION_TEMP_MEMORY_BUDGET_BYTES)

    def test_track_chunks_are_cached_across_camera_batches(self) -> None:
        model = _bare(None)
        tracks = cast(Any, object())
        subset = self._subset([5.0], 2)
        tracks_idx = torch.tensor([0, 1], dtype=torch.int32)
        rays_o = torch.zeros((1, 3))
        rays_d = torch.tensor([[0.0, 1.0, 0.0]])
        rays_ts = torch.tensor([0], dtype=torch.int64)
        with mock.patch.object(CuboidTracks.Ops, "subset_from_indices", return_value=subset) as subset_mock:
            for _ in range(2):
                model._component_only_ray_cuboid_hits(
                    tracks,
                    tracks_idx,
                    0.0,
                    rays_o,
                    rays_d,
                    rays_ts,
                    track_indices_key=(0, 1),
                )
        subset_mock.assert_called_once()
        self.assertEqual(subset.ray_intersection.call_count, 2)

    def test_empty_group_is_miss_with_infinite_depth(self) -> None:
        projected = _bare(None)._component_only_ray_cuboid_hits(
            cast(Any, object()),
            torch.zeros((0,), dtype=torch.int32),
            0.0,
            torch.zeros((2, 3)),
            torch.zeros((2, 3)),
            torch.zeros((2,), dtype=torch.int64),
        )
        self.assertEqual(projected.mask.tolist(), [False, False])
        self.assertTrue(bool(torch.isinf(projected.nearest_depth).all()))

    def test_empty_ray_batch_skips_track_subset_and_intersection(self) -> None:
        with mock.patch.object(CuboidTracks.Ops, "subset_from_indices") as subset_mock:
            projected = _bare(None)._component_only_ray_cuboid_hits(
                cast(Any, object()),
                torch.tensor([0], dtype=torch.int32),
                0.0,
                torch.zeros((0, 3)),
                torch.zeros((0, 3)),
                torch.zeros((0,), dtype=torch.int64),
            )
        subset_mock.assert_not_called()
        self.assertEqual(projected.mask.numel(), 0)
        self.assertEqual(projected.nearest_depth.numel(), 0)


class TestBuildBboxMaskWiring(unittest.TestCase):
    _CAMERA = _ns(
        rays=torch.tensor([[[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]]]),
        timestamps_startend_us_cpu=torch.tensor([[0, 1_000_000]]),
    )

    @staticmethod
    def _layers(*layer_ids: str) -> dict[str, Any]:
        return {
            "background": _ns(class_labels=None, tracks=None),
            **{layer_id: _ns(class_labels=None, tracks=object(), cuboid_tracks_expand=0.5) for layer_id in layer_ids},
        }

    def test_raw_and_expanded_disabled_groups_merge_by_or_and_min(self) -> None:
        tracks = _rigids_tracks()
        layers = self._layers("actors")
        config = _enabled_config(
            actors=ComponentOnlyLayerConfig(
                state=ComponentState.disabled,
                masks=[MaskSource.bbox],
                tracks={"t_near": ComponentOnlyTrackConfig(masks=[MaskSource.bbox_expanded])},
            )
        )
        model = _bare(config)
        model.config = _ns(layers=layers)
        model._component_only_layer_tracks = {"actors": tracks}
        _resolve_and_store(model, "actors", tracks)
        model._component_only_mask_spec = build_mask_spec(
            config,
            layers,
            None,
            layer_track_states=model._component_only_track_states,
        )

        def project(*args: Any, **kwargs: Any) -> ProjectedTrackRegion:
            expand = cast(float, args[2])
            if expand:
                return ProjectedTrackRegion(torch.tensor([True, False]), torch.tensor([7.0, torch.inf]))
            return ProjectedTrackRegion(torch.tensor([True, True]), torch.tensor([9.0, 4.0]))

        with mock.patch.object(model, "_component_only_ray_cuboid_hits", side_effect=project) as project_mock:
            regions = model.build_component_only_bbox_masks(cast(Any, self._CAMERA))
        disabled = regions["actors" + DISABLED_TRACKS_SUFFIX]
        self.assertEqual(disabled.mask[0, 0].tolist(), [True, True])
        self.assertEqual(disabled.nearest_depth[0, 0].tolist(), [7.0, 4.0])
        self.assertEqual(project_mock.call_count, 2)

    def test_fully_disabled_residual_projection_is_reused_for_occlusion(self) -> None:
        tracks = _rigids_tracks()
        layers = self._layers("actors")
        config = _enabled_config(
            background=ComponentOnlyLayerConfig(),
            actors=ComponentOnlyLayerConfig(state=ComponentState.disabled, masks=[MaskSource.bbox]),
        )
        model = _bare(config)
        model.config = _ns(layers=layers)
        model._component_only_layer_tracks = {"actors": tracks}
        model._component_only_mask_spec = build_mask_spec(config, layers, _CLASS_MAP)
        projected = ProjectedTrackRegion(torch.tensor([True, False]), torch.tensor([3.0, torch.inf]))
        with mock.patch.object(model, "_component_only_ray_cuboid_hits", return_value=projected) as project_mock:
            regions = model.build_component_only_bbox_masks(cast(Any, self._CAMERA))
        pseudo = "actors" + DISABLED_TRACKS_SUFFIX
        self.assertEqual(set(regions), {"actors", pseudo})
        self.assertTrue(torch.equal(regions["actors"].mask, regions[pseudo].mask))
        self.assertTrue(torch.equal(regions["actors"].nearest_depth, regions[pseudo].nearest_depth))
        project_mock.assert_called_once()

    def test_warmed_bbox_group_cache_survives_deepcopy(self) -> None:
        tracks = _rigids_tracks()
        layers = self._layers("actors")
        config = _enabled_config(
            actors=ComponentOnlyLayerConfig(
                masks=[MaskSource.bbox],
                tracks={"t_far": ComponentOnlyTrackConfig(state=ComponentState.disabled)},
            )
        )
        model = _bare(config)
        model.config = _ns(layers=layers)
        model._component_only_layer_tracks = {"actors": tracks}
        _resolve_and_store(model, "actors", tracks)
        model._component_only_mask_spec = build_mask_spec(
            config,
            layers,
            None,
            layer_track_states=model._component_only_track_states,
        )
        projected = ProjectedTrackRegion(torch.tensor([True, False]), torch.tensor([3.0, torch.inf]))
        with mock.patch.object(model, "_component_only_ray_cuboid_hits", return_value=projected):
            model.build_component_only_bbox_masks(cast(Any, self._CAMERA))

        copied = copy.deepcopy(model)
        with mock.patch.object(copied, "_component_only_ray_cuboid_hits", return_value=projected):
            regions = copied.build_component_only_bbox_masks(cast(Any, self._CAMERA))
        self.assertEqual(set(regions), {"actors", "actors" + DISABLED_TRACKS_SUFFIX})

    def test_cross_layer_depth_ordering_in_both_actor_directions(self) -> None:
        layers = self._layers("dynamic_rigids", "dynamic_deformables")
        tracks_by_layer = {
            "dynamic_rigids": _make_tracks([("rigid", "automobile", (0.0, 5.0, 0.0))]),
            "dynamic_deformables": _make_tracks([("deformable", "person", (0.0, 10.0, 0.0))]),
        }
        for selected_layer, disabled_layer in (
            ("dynamic_deformables", "dynamic_rigids"),
            ("dynamic_rigids", "dynamic_deformables"),
        ):
            with self.subTest(selected=selected_layer, disabled=disabled_layer):
                config = _enabled_config(**{selected_layer: ComponentOnlyLayerConfig()})
                model = _bare(config)
                model.config = _ns(layers=layers)
                model._component_only_layer_tracks = tracks_by_layer
                model._component_only_globally_rendered_track_ids = model._resolve_component_only_rendered_track_ids()
                model._component_only_mask_spec = build_mask_spec(config, layers, None)

                def project(*args: Any, **kwargs: Any) -> ProjectedTrackRegion:
                    tracks = cast(CuboidTracks, args[0])
                    depth = 10.0 if tracks is tracks_by_layer[selected_layer] else 5.0
                    return ProjectedTrackRegion(torch.ones(2, dtype=torch.bool), torch.full((2,), depth))

                with mock.patch.object(model, "_component_only_ray_cuboid_hits", side_effect=project):
                    regions = model.build_component_only_bbox_masks(cast(Any, self._CAMERA))
                final = build_camera_loss_mask(cast(Any, model._component_only_mask_spec), None, regions)
                self.assertFalse(bool(final.any()))


@_NEEDS_CUDA
class TestBuildBboxMasks(_MaskHarness):
    # Pixel 0 looks straight at t_near; pixel 1 looks away from both boxes.
    _DIRECTIONS = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]

    def test_no_spec_returns_empty(self) -> None:
        model = _bare(None)
        self.assertEqual(model.build_component_only_bbox_masks(cast(Any, _rendering_cam(self._DIRECTIONS))), {})

    def test_supervised_layer_mask(self) -> None:
        model = self._model(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()), device=_DEVICE)
        masks = model.build_component_only_bbox_masks(cast(Any, _rendering_cam(self._DIRECTIONS)))
        self.assertEqual(set(masks), {"dynamic_rigids"})
        self.assertEqual(masks["dynamic_rigids"].mask[0, 0].tolist(), [True, False])
        self.assertAlmostEqual(float(masks["dynamic_rigids"].nearest_depth[0, 0, 0]), 9.0)
        self.assertTrue(torch.isinf(masks["dynamic_rigids"].nearest_depth[0, 0, 1]))

    def test_missing_tracks_yield_empty_mask(self) -> None:
        model = self._model(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()), device=_DEVICE)
        del model._component_only_layer_tracks["dynamic_rigids"]
        masks = model.build_component_only_bbox_masks(cast(Any, _rendering_cam(self._DIRECTIONS)))
        self.assertFalse(bool(masks["dynamic_rigids"].mask.any()))
        self.assertTrue(bool(torch.isinf(masks["dynamic_rigids"].nearest_depth).all()))

    def test_residual_adds_carved_layers_and_pseudo_partition(self) -> None:
        model = self._model(
            _enabled_config(
                background=ComponentOnlyLayerConfig(),
                dynamic_rigids=ComponentOnlyLayerConfig(
                    tracks={"t_near": ComponentOnlyTrackConfig(state=ComponentState.disabled)}
                ),
            ),
            resolve_tracks=True,
            device=_DEVICE,
        )
        masks = model.build_component_only_bbox_masks(cast(Any, _rendering_cam(self._DIRECTIONS)))
        pseudo = "dynamic_rigids" + DISABLED_TRACKS_SUFFIX
        self.assertEqual(set(masks), {"dynamic_rigids", pseudo})
        # Supervised partition (t_far) hits nothing; the disabled
        # partition (t_near) owns pixel 0.
        self.assertFalse(bool(masks["dynamic_rigids"].mask.any()))
        self.assertEqual(masks[pseudo].mask[0, 0].tolist(), [True, False])

    def test_disabled_partition_is_emitted_without_residual(self) -> None:
        model = self._model(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    tracks={"t_near": ComponentOnlyTrackConfig(state=ComponentState.disabled)}
                )
            ),
            resolve_tracks=True,
            device=_DEVICE,
        )
        masks = model.build_component_only_bbox_masks(cast(Any, _rendering_cam(self._DIRECTIONS)))
        pseudo = "dynamic_rigids" + DISABLED_TRACKS_SUFFIX
        self.assertEqual(set(masks), {"dynamic_rigids", pseudo})
        self.assertTrue(bool(masks[pseudo].mask[0, 0, 0]))
        self.assertAlmostEqual(float(masks[pseudo].nearest_depth[0, 0, 0]), 9.0)

    def test_semantic_only_layers_are_skipped(self) -> None:
        layers = {
            "background": _ns(class_labels=None, tracks=None),
            "road": _ns(class_labels=["road"], tracks=None),
        }
        config = _enabled_config(road=ComponentOnlyLayerConfig())
        model = _bare(config)
        model.config = _ns(layers=layers)
        model._component_only_mask_spec = build_mask_spec(config, layers, _CLASS_MAP)
        self.assertEqual(model.build_component_only_bbox_masks(cast(Any, _rendering_cam(self._DIRECTIONS))), {})

    def test_per_track_mask_groups_build_separate_box_keys(self) -> None:
        # Per-track masks split the layer into per-recipe groups; the
        # builder emits one box mask per group key, over that group's
        # tracks only. t_near = raw bbox, t_far = expanded bbox.
        config = _enabled_config(
            dynamic_rigids=ComponentOnlyLayerConfig(
                tracks={
                    "t_near": ComponentOnlyTrackConfig(masks=[MaskSource.bbox]),
                    "t_far": ComponentOnlyTrackConfig(masks=[MaskSource.bbox_expanded]),
                }
            )
        )
        model = _bare(config)
        model.config = _ns(layers=_model_layers_ns())
        model._component_only_layer_tracks = {"dynamic_rigids": _rigids_tracks(device=_DEVICE)}
        _resolve_and_store(model, "dynamic_rigids", model._component_only_layer_tracks["dynamic_rigids"])
        model._component_only_mask_spec = build_mask_spec(
            config, _model_layers_ns(), _CLASS_MAP, layer_track_states=model._component_only_track_states
        )
        masks = model.build_component_only_bbox_masks(cast(Any, _rendering_cam(self._DIRECTIONS)))
        # Two recipe groups → two distinct box-mask keys.
        self.assertEqual(set(masks), {"dynamic_rigids#0", "dynamic_rigids#1"})
        # Group 0 (t_near, raw) owns pixel 0; group 1 (t_far, out of view) is empty.
        self.assertEqual(masks["dynamic_rigids#0"].mask[0, 0].tolist(), [True, False])
        self.assertFalse(bool(masks["dynamic_rigids#1"].mask.any()))

    def test_per_ray_timestamps_gate_the_hit(self) -> None:
        # Both pixels aim straight at t_near, but the cuboid pose is
        # interpolated at each ray's own timestamp: the ray stamped inside
        # the track's [0, 1s] keyframe range hits, the one stamped past it
        # misses. This is the rolling-shutter fidelity the fused kernel
        # gives over a single mid-frame pose.
        model = self._model(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()), device=_DEVICE)
        rays = torch.zeros((1, 1, 2, 6), device=_DEVICE)
        rays[0, 0, :, 3:] = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=_DEVICE)
        cam = _ns(
            rays=rays,
            rays_timestamps_us=torch.tensor([[[[500_000], [5_000_000]]]], dtype=torch.int64, device=_DEVICE),
            timestamps_startend_us_cpu=torch.tensor([[0, 1_000_000]]),
        )
        masks = model.build_component_only_bbox_masks(cast(Any, cam))
        self.assertEqual(masks["dynamic_rigids"].mask[0, 0].tolist(), [True, False])
        self.assertTrue(torch.isinf(masks["dynamic_rigids"].nearest_depth[0, 0, 1]))

    def test_timestamp_outside_track_window_contributes_nothing(self) -> None:
        model = self._model(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()), device=_DEVICE)
        cam = _rendering_cam(self._DIRECTIONS)
        cam.timestamps_startend_us_cpu = torch.tensor([[5_000_000, 6_000_000]])  # past track end
        masks = model.build_component_only_bbox_masks(cast(Any, cam))
        self.assertFalse(bool(masks["dynamic_rigids"].mask.any()))
        self.assertTrue(bool(torch.isinf(masks["dynamic_rigids"].nearest_depth).all()))

    def test_camera_origin_inside_cuboid_has_zero_nearest_depth(self) -> None:
        model = self._model(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()), device=_DEVICE)
        tracks = model._component_only_layer_tracks["dynamic_rigids"]
        rays_o = torch.tensor([[0.0, 10.0, 0.0]], device=_DEVICE)
        rays_d = torch.tensor([[0.0, 1.0, 0.0]], device=_DEVICE)
        projected = model._component_only_ray_cuboid_hits(
            tracks,
            torch.tensor([0], dtype=tracks.tracks_packinfo.dtype, device=_DEVICE),
            0.0,
            rays_o,
            rays_d,
            torch.tensor([500_000], dtype=torch.int64, device=_DEVICE),
        )
        self.assertEqual(projected.mask.tolist(), [True])
        self.assertEqual(projected.nearest_depth.tolist(), [0.0])


class TestApplyFrozenTracks(_MaskHarness):
    def _frozen_config(self) -> ComponentOnlyConfig:
        return _enabled_config(
            dynamic_rigids=ComponentOnlyLayerConfig(
                tracks={"t_far": ComponentOnlyTrackConfig(state=ComponentState.frozen)}
            )
        )

    def test_frozen_track_rows_get_zero_gradients(self) -> None:
        model = self._model(self._frozen_config(), resolve_tracks=True)
        node = _FakeRigidNode(_rigids_tracks())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})
        model._apply_component_only_frozen_tracks()
        node.positions.sum().backward()
        rows_near = node.gaussian_cuboid_ids == 0
        self.assertTrue(torch.equal(_grad(node.positions)[rows_near], torch.ones((int(rows_near.sum()), 3))))
        self.assertTrue(torch.equal(_grad(node.positions)[~rows_near], torch.zeros((int((~rows_near).sum()), 3))))

    def test_reapply_reinstalls_frozen_track_row_hook(self) -> None:
        # Resume: after the system's load_state_dict, reapply must reinstall
        # the frozen-track per-Gaussian row hook so an optimizer step leaves the
        # frozen track's rows unchanged.
        model = self._model(self._frozen_config(), resolve_tracks=True)
        node = _FakeRigidNode(_rigids_tracks())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})
        model.reapply_component_only_runtime_state()
        before = node.positions.detach().clone()
        node.positions.sum().backward()
        opt = torch.optim.SGD([node.positions], lr=1.0)
        opt.step()
        rows_near = node.gaussian_cuboid_ids == 0  # t_near enabled; t_far frozen
        self.assertTrue(torch.equal(_grad(node.positions)[rows_near], torch.ones((int(rows_near.sum()), 3))))
        self.assertTrue(torch.equal(_grad(node.positions)[~rows_near], torch.zeros((int((~rows_near).sum()), 3))))
        self.assertTrue(torch.equal(node.positions.detach()[~rows_near], before[~rows_near]))  # frozen unchanged

    def test_morton_sort_reinstalls_frozen_track_row_hook(self) -> None:
        # The reorder replaces the parameters the hooks were bound to, and each mask was built for the old order.
        # The composite has to rebuild it from the permuted membership, or the freeze lands on other Gaussians.
        model = self._model(self._frozen_config(), resolve_tracks=True)
        node = _SortableRigidNode(_rigids_tracks())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})
        handed_to_strategy: dict[str, torch.Tensor] = {}
        model.gaussians_strategy = cast(
            Any, _ns(permute_gaussian_rows=handed_to_strategy.update, refresh_runtime_scenes=lambda _nodes: None)
        )
        model._apply_component_only_frozen_tracks()

        model.morton_sort_gaussians()

        # Strategy-owned row-aligned buffers live outside the nodes, so they get the orders too.
        self.assertEqual(list(handed_to_strategy), ["dynamic_rigids"])
        # Without an actual interleave a stale mask would still line up and prove nothing.
        self.assertFalse(torch.equal(node.gaussian_cuboid_ids, torch.tensor([0, 0, 0, 1, 1, 1])))

        before = node.positions.detach().clone()
        node.positions.sum().backward()
        torch.optim.SGD([node.positions], lr=1.0).step()
        rows_near = node.gaussian_cuboid_ids == 0  # t_near enabled; t_far frozen
        self.assertTrue(torch.equal(_grad(node.positions)[rows_near], torch.ones((int(rows_near.sum()), 3))))
        self.assertTrue(torch.equal(_grad(node.positions)[~rows_near], torch.zeros((int((~rows_near).sum()), 3))))
        self.assertTrue(torch.equal(node.positions.detach()[~rows_near], before[~rows_near]))  # frozen unchanged

    def test_morton_sort_does_not_restack_hooks_on_a_skipped_layer(self) -> None:
        # A layer the sort skips keeps its parameter objects, so it keeps the hooks already bound to them.
        # Reinstalling anyway adds a second copy on every checkpoint, and grad * keep hides it numerically.
        model = self._model(self._frozen_config(), resolve_tracks=True)
        skipped = _SortableRigidNode(_rigids_tracks())
        # Nothing to sort, which is how a layer reaches the reinstall while absent from `orders`.
        # A second layer does sort, so the reinstall runs at all.
        skipped.compute_morton_order = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        model.gaussians_nodes = TypedModuleDict(
            {"road": _SortableRigidNode(_rigids_tracks()), "dynamic_rigids": skipped}
        )
        model.gaussians_strategy = cast(
            Any, _ns(permute_gaussian_rows=lambda _orders: None, refresh_runtime_scenes=lambda _nodes: None)
        )
        model._apply_component_only_frozen_tracks()
        installed = len(skipped.positions._backward_hooks or {})
        self.assertEqual(installed, 1)  # otherwise the count below proves nothing

        for _ in range(3):
            model.morton_sort_gaussians()

        self.assertEqual(len(skipped.positions._backward_hooks or {}), installed)

    def test_morton_sort_does_not_refreeze_per_track_params(self) -> None:
        # freeze_tracks hooks track_albedos, track_scales and the calibration poses, all indexed by track.
        # A permutation leaves those alone, so re-running it would only stack a second hook on each.
        model = self._model(self._frozen_config(), resolve_tracks=True)
        node = _SortableRigidNode(_rigids_tracks())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})
        model.gaussians_strategy = cast(
            Any, _ns(permute_gaussian_rows=lambda _orders: None, refresh_runtime_scenes=lambda _nodes: None)
        )
        model._apply_component_only_frozen_tracks()
        self.assertEqual(node.freeze_tracks_calls, [[1]])  # otherwise the check below is empty

        for _ in range(3):
            model.morton_sort_gaussians(sort_track_layers=True)

        # Rows moved three times, each rebuild landing on a fresh parameter, so exactly one row hook remains.
        self.assertEqual(len(node.positions._backward_hooks or {}), 1)
        # The per-track params never moved, so the node was not asked to freeze them again.
        self.assertEqual(node.freeze_tracks_calls, [[1]])

    def test_system_load_state_dict_reapplies_frozen_track_hook(self) -> None:
        # Real integration through GaussiansSystem.load_state_dict(assign=True):
        # loading a checkpoint replaces the tensors the hooks were bound to, and
        # the system's load must trigger the model's reapply so the frozen-track
        # row hook is reinstalled on the fresh tensors. This fails if the system
        # ever stops invoking reapply (the original resume-integration bug).
        model = self._model(self._frozen_config(), resolve_tracks=True)
        node = _FakeRigidNode(_rigids_tracks())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})
        model.obj_track_ids = {}  # consumed by the composite's get/set_extra_state
        system = object.__new__(GaussiansSystem)
        torch.nn.Module.__init__(system)
        system.model = cast(Any, model)  # register as a submodule
        system._active_tile_mapping_prewarmer = ActiveTileMappingPrewarmer()
        stale_mapping = object()
        model._cam_to_lidar_tile_mappings[7] = cast(Any, stale_mapping)
        stale_collect = object()
        model._static_collect_cache = cast(Any, stale_collect)
        # assign=True replaces the parameter tensors (as a real checkpoint load).
        # ``node`` is the same module object; only its parameters are replaced.
        system.load_state_dict(system.state_dict(), strict=False, assign=True)
        self.assertEqual(model._cam_to_lidar_tile_mappings, {})
        self.assertIsNone(model._static_collect_cache)
        before = node.positions.detach().clone()
        node.positions.sum().backward()
        torch.optim.SGD([node.positions], lr=1.0).step()
        rows_near = node.gaussian_cuboid_ids == 0  # t_near enabled; t_far frozen
        self.assertTrue(torch.equal(_grad(node.positions)[~rows_near], torch.zeros((int((~rows_near).sum()), 3))))
        self.assertTrue(torch.equal(node.positions.detach()[~rows_near], before[~rows_near]))

    def test_per_track_params_freeze_is_delegated_to_the_node(self) -> None:
        # Per-track/pose params (track_albedos/scales, tracks_calib, deform)
        # are indexed by each owner's own scheme, so the composite delegates
        # their freeze to node.freeze_tracks (the real freeze is covered in
        # tracks_calib_test / gaussians_model). Here we assert the delegation
        # fires with the frozen track's local index.
        model = self._model(self._frozen_config(), resolve_tracks=True)
        node = _FakeRigidNode(_rigids_tracks())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})
        model._apply_component_only_frozen_tracks()
        # t_far is track index 1 in _rigids_tracks; t_near (0) stays trainable.
        self.assertEqual(node.freeze_tracks_calls, [[1]])

    def test_skips_when_node_absent_or_uninitialized(self) -> None:
        model = self._model(self._frozen_config(), resolve_tracks=True)
        model.gaussians_nodes = TypedModuleDict({})
        model._apply_component_only_frozen_tracks()  # layer not built: no-op

        lazy = _FakeRigidNode(_rigids_tracks())
        lazy.gaussian_cuboid_ids = cast(Any, torch.nn.parameter.UninitializedBuffer())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": lazy})
        model._apply_component_only_frozen_tracks()  # lazy buffer: no-op
        lazy.positions.sum().backward()
        self.assertTrue(bool(_grad(lazy.positions).abs().sum() > 0))

        plain = torch.nn.Module()  # no cuboid_tracks / gaussian_cuboid_ids at all
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_rigids": plain}))
        model._apply_component_only_frozen_tracks()

    def test_skips_when_no_frozen_tracks(self) -> None:
        model = self._model(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    tracks={"t_far": ComponentOnlyTrackConfig(state=ComponentState.disabled)}
                )
            ),
            resolve_tracks=True,
        )
        node = _FakeRigidNode(_rigids_tracks())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})
        model._apply_component_only_frozen_tracks()
        node.positions.sum().backward()
        self.assertTrue(bool(_grad(node.positions).abs().sum() > 0))


class TestFrozenComponentUpdateStep(unittest.TestCase):
    """``update_step_train_batch_start`` must skip whole-frozen components,
    else their per-batch schedulers re-enable ``requires_grad`` and unfreeze
    them (calib.py / tracks_calib.py re-assert grad each step)."""

    class _Rec(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stepped = False

        def update_step_train_batch_start(self, epoch: int, global_step: int, system: Any, **kw: Any) -> dict:
            self.stepped = True
            return {}

    def test_frozen_layer_and_calib_skip_update_step(self) -> None:
        model = _bare(
            ComponentOnlyConfig(
                enabled=True,
                layers={
                    "dynamic_rigids": ComponentOnlyLayerConfig(state=ComponentState.frozen),
                    "road": ComponentOnlyLayerConfig(),
                },
                calib=ComponentOnlyComponentConfig(state=ComponentState.frozen),
            )
        )
        model._component_only_frozen_layers = frozenset({"dynamic_rigids"})
        model._distillation_active = False
        model._distillation_start_step = -1
        frozen_node, enabled_node = self._Rec(), self._Rec()
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_rigids": frozen_node, "road": enabled_node}))
        model.calib = cast(Any, self._Rec())
        model.background = cast(Any, self._Rec())
        model.post_processings = cast(Any, [])
        model.update_step_train_batch_start(0, 0, object())
        self.assertFalse(frozen_node.stepped)  # frozen layer skipped
        self.assertTrue(enabled_node.stepped)  # enabled layer stepped
        self.assertFalse(cast(Any, model.calib).stepped)  # frozen calib skipped
        self.assertTrue(cast(Any, model.background).stepped)  # sky_model not frozen -> stepped


@_NEEDS_CUDA
class TestBackgroundRegionRows(_MaskHarness):
    def _scoped_model(self) -> GaussiansComposite:
        model = self._model(
            _enabled_config(
                background=ComponentOnlyLayerConfig(scope={ScopeRegion.bbox: ComponentState.enabled}),
                dynamic_rigids=ComponentOnlyLayerConfig(),
            ),
            device=_DEVICE,
        )
        model._component_only_viewpoints = (torch.zeros((1, 3)), torch.tensor([500_000]))
        return model

    def test_points_behind_sibling_boxes_are_in_region(self) -> None:
        model = self._scoped_model()
        # Behind t_near from the viewpoint; off to the side.
        positions = torch.tensor([[0.0, 20.0, 0.0], [20.0, 0.0, 0.0]], device=_DEVICE)
        self.assertEqual(model._background_bbox_region_rows(positions).tolist(), [True, False])

    def test_empty_without_spec_or_viewpoints(self) -> None:
        model = self._scoped_model()
        positions = torch.tensor([[0.0, 20.0, 0.0]], device=_DEVICE)
        model._component_only_viewpoints = (torch.zeros((0, 3)), torch.zeros((0,), dtype=torch.int64))
        self.assertEqual(model._background_bbox_region_rows(positions).tolist(), [False])
        model = self._scoped_model()
        model._component_only_mask_spec = None
        self.assertEqual(model._background_bbox_region_rows(positions).tolist(), [False])

    def test_layers_without_tracks_contribute_nothing(self) -> None:
        model = self._scoped_model()
        model._component_only_layer_tracks = {}
        positions = torch.tensor([[0.0, 20.0, 0.0]], device=_DEVICE)
        self.assertEqual(model._background_bbox_region_rows(positions).tolist(), [False])

    def test_viewpoint_outside_track_window_contributes_nothing(self) -> None:
        model = self._scoped_model()
        model._component_only_viewpoints = (torch.zeros((1, 3)), torch.tensor([9_000_000]))
        positions = torch.tensor([[0.0, 20.0, 0.0]], device=_DEVICE)
        self.assertEqual(model._background_bbox_region_rows(positions).tolist(), [False])


class _FakeBackgroundNode(BaseGaussianModel):
    def __init__(self, positions: torch.Tensor):
        torch.nn.Module.__init__(self)  # intentionally skip BaseGaussianModel.__init__
        self.positions: torch.nn.Parameter = torch.nn.Parameter(positions)
        self.token = torch.nn.Parameter(torch.randn(5))  # never row-filtered

    def get_num_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_positions(self) -> torch.nn.Parameter:
        return self.positions


@_NEEDS_CUDA
class TestApplyBackgroundScope(_MaskHarness):
    _POSITIONS = torch.tensor([[0.0, 20.0, 0.0], [20.0, 0.0, 0.0]])  # in-region; out-of-region

    def _scope_model(self, scope: dict) -> tuple[GaussiansComposite, _FakeBackgroundNode]:
        model = self._model(
            _enabled_config(
                background=ComponentOnlyLayerConfig(scope=scope), dynamic_rigids=ComponentOnlyLayerConfig()
            ),
            device=_DEVICE,
        )
        model._component_only_viewpoints = (torch.zeros((1, 3)), torch.tensor([500_000]))
        node = _FakeBackgroundNode(self._POSITIONS.clone().to(_DEVICE))
        model.gaussians_nodes = TypedModuleDict({"background": node})
        return model, node

    def test_noop_paths(self) -> None:
        # No component_only; no bbox region; node missing; empty node.
        model = _bare(None)
        model._apply_component_only_background_scope()

        model, node = self._scope_model({ScopeRegion.residual: ComponentState.enabled})
        model._apply_component_only_background_scope()
        self.assertEqual(node.get_num_gaussians(), 2)

        model, _ = self._scope_model({ScopeRegion.bbox: ComponentState.enabled})
        model.gaussians_nodes = TypedModuleDict({})
        model._apply_component_only_background_scope()

        model, node = self._scope_model({ScopeRegion.bbox: ComponentState.enabled})
        empty_node = _FakeBackgroundNode(torch.zeros((0, 3)))
        model.gaussians_nodes = TypedModuleDict({"background": empty_node})
        model._apply_component_only_background_scope()

        # Checkpoint-based val/test calls setup before state loading. The
        # checkpoint already contains the scoped background population.
        model, lazy_node = self._scope_model({ScopeRegion.bbox: ComponentState.enabled})
        lazy_node.positions = cast(Any, torch.nn.parameter.UninitializedParameter())
        model._apply_component_only_background_scope()
        self.assertIsInstance(lazy_node.positions, torch.nn.parameter.UninitializedParameter)

    def test_resume_reapplies_mixed_background_hook_after_lazy_load(self) -> None:
        # Full resume path: the mixed-background hook is skipped at init
        # (positions lazy), load(assign=True) materializes them, then the
        # system's reapply reinstalls the hook so an optimizer step leaves the
        # frozen region unchanged.
        model, node = self._scope_model(
            {ScopeRegion.bbox: ComponentState.enabled, ScopeRegion.residual: ComponentState.frozen}
        )
        node.positions = cast(Any, torch.nn.parameter.UninitializedParameter())
        model._apply_component_only_background_hooks()  # lazy -> no-op
        # load(assign=True): materialize the (trainable) parameter.
        node.positions = torch.nn.Parameter(self._POSITIONS.clone().to(_DEVICE))
        model.reapply_component_only_runtime_state()
        before = node.positions.detach().clone()
        node.positions.sum().backward()
        opt = torch.optim.SGD([node.positions], lr=1.0)
        opt.step()
        grad = _grad(node.positions)
        # bbox region (row 0) enabled -> trains; residual (row 1) frozen.
        self.assertTrue(torch.equal(grad[0].cpu(), torch.ones(3)))
        self.assertTrue(torch.equal(grad[1].cpu(), torch.zeros(3)))
        self.assertTrue(torch.equal(node.positions.detach()[1], before[1]))  # frozen row unchanged

    def test_bbox_only_scope_drops_outside_rows(self) -> None:
        model, node = self._scope_model({ScopeRegion.bbox: ComponentState.enabled})
        model._apply_component_only_background_scope()
        self.assertEqual(node.get_num_gaussians(), 1)
        self.assertTrue(torch.equal(node.get_positions().data.cpu(), self._POSITIONS[:1]))
        self.assertEqual(node.token.shape, (5,))  # non-row parameter untouched

    def test_uniform_states_keep_all_rows(self) -> None:
        model, node = self._scope_model(
            {ScopeRegion.bbox: ComponentState.enabled, ScopeRegion.residual: ComponentState.enabled}
        )
        model._apply_component_only_background_scope()
        self.assertEqual(node.get_num_gaussians(), 2)
        node.positions.sum().backward()
        self.assertTrue(bool(_grad(node.positions).abs().sum() > 0))

    def test_mixed_states_install_row_hooks(self) -> None:
        model, node = self._scope_model(
            {ScopeRegion.bbox: ComponentState.enabled, ScopeRegion.residual: ComponentState.frozen}
        )
        model._apply_component_only_background_hooks()
        node.positions.sum().backward()
        self.assertTrue(torch.equal(_grad(node.positions)[0].cpu(), torch.ones(3)))  # bbox region trains
        self.assertTrue(torch.equal(_grad(node.positions)[1].cpu(), torch.zeros(3)))  # residual frozen

    def test_mixed_states_frozen_bbox_region(self) -> None:
        model, node = self._scope_model(
            {ScopeRegion.bbox: ComponentState.frozen, ScopeRegion.residual: ComponentState.enabled}
        )
        model._apply_component_only_background_hooks()
        node.positions.sum().backward()
        self.assertTrue(torch.equal(_grad(node.positions)[0].cpu(), torch.zeros(3)))
        self.assertTrue(torch.equal(_grad(node.positions)[1].cpu(), torch.ones(3)))


class TestApplyFreeze(unittest.TestCase):
    def _model(self) -> GaussiansComposite:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"dynamic_rigids": ComponentOnlyLayerConfig(state=ComponentState.frozen)},
            sky_model=ComponentOnlyComponentConfig(state=ComponentState.frozen),
            calib=ComponentOnlyComponentConfig(state=ComponentState.frozen),
            post_processing=ComponentOnlyComponentConfig(state=ComponentState.frozen),
        )
        model = _bare(config)
        node = _FakeRigidNode(_rigids_tracks())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})
        model._component_only_frozen_layers = frozenset({"dynamic_rigids"})
        model.background = cast(Any, torch.nn.Linear(2, 2))
        model.calib = cast(Any, torch.nn.Linear(2, 2))
        model.post_processings = cast(Any, [torch.nn.Linear(2, 2)])
        return model

    def test_noop_without_component_only(self) -> None:
        model = _bare(None)
        model._apply_component_only_freeze()

    def test_freezes_all_frozen_components(self) -> None:
        model = self._model()
        model._apply_component_only_freeze()
        for module in [model.gaussians_nodes["dynamic_rigids"], model.background, model.calib] + list(
            model.post_processings
        ):
            for param in module.parameters():
                self.assertFalse(param.requires_grad)

    def test_lazy_parameters_are_skipped(self) -> None:
        model = self._model()
        lazy_holder = torch.nn.Module()
        lazy_holder.weight = torch.nn.parameter.UninitializedParameter()
        model.background = cast(Any, lazy_holder)
        model._apply_component_only_freeze()  # must not touch the lazy parameter

    def test_reapply_runtime_state_refreezes_after_checkpoint_load(self) -> None:
        # Resume: the frozen components were lazy at init (freeze no-op'd),
        # then load(assign=True) materialized them as trainable. The reapply
        # from the system's load_state_dict must turn gradients back off so
        # an optimizer step never touches the frozen tensors.
        model = self._model()
        for module in [model.gaussians_nodes["dynamic_rigids"], model.background, model.calib]:
            for param in module.parameters():
                param.requires_grad_(True)  # as load(assign=True) leaves them
        model.reapply_component_only_runtime_state()
        for module in [model.gaussians_nodes["dynamic_rigids"], model.background, model.calib] + list(
            model.post_processings
        ):
            for param in module.parameters():
                self.assertFalse(param.requires_grad)

    def test_non_frozen_components_keep_gradients(self) -> None:
        model = self._model()
        model.component_only = ComponentOnlyConfig(enabled=True, layers={"dynamic_rigids": ComponentOnlyLayerConfig()})
        model._component_only_frozen_layers = frozenset()
        model._apply_component_only_freeze()
        self.assertTrue(model.background.weight.requires_grad)
        self.assertTrue(model.calib.weight.requires_grad)
        self.assertTrue(model.post_processings[0].weight.requires_grad)


class TestComponentInventory(unittest.TestCase):
    def _model(self, component_only: Optional[ComponentOnlyConfig]) -> GaussiansComposite:
        model = _bare(component_only)
        node = torch.nn.Module()
        node.weight = torch.nn.Parameter(torch.randn(4, 3))
        node.frozen_extra = torch.nn.Parameter(torch.randn(2), requires_grad=False)
        lazy_node = torch.nn.Module()
        lazy_node.weight = torch.nn.parameter.UninitializedParameter()
        mixed_node = torch.nn.Module()
        mixed_node.weight = torch.nn.Parameter(torch.randn(3))
        mixed_node.lazy = torch.nn.parameter.UninitializedParameter()
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"a": node, "b": lazy_node, "c": mixed_node}))
        model.background = cast(Any, None)
        model.calib = cast(Any, torch.nn.Linear(2, 2))
        model.post_processings = cast(Any, [torch.nn.Linear(2, 2)])
        return model

    def test_full_mode_line(self) -> None:
        model = self._model(None)
        with self.assertLogs("nurec.models.gaussians.gaussians_composite", level="INFO") as logs:
            model._log_component_inventory()
        line = logs.output[-1]
        self.assertIn("FULL", line)
        self.assertIn("'a': 12", line)
        self.assertIn("'b': 0 (lazy)", line)
        self.assertIn("'c': 3 (+ lazy)", line)

    def test_component_only_mode_with_frozen_tag_and_when(self) -> None:
        model = self._model(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()))
        model._component_only_frozen_layers = frozenset({"a"})
        with self.assertLogs("nurec.models.gaussians.gaussians_composite", level="INFO") as logs:
            model._log_component_inventory(when="post-init")
        line = logs.output[-1]
        self.assertIn("COMPONENT-ONLY", line)
        self.assertIn("post-init", line)
        self.assertIn("'a': 12 (frozen)", line)


class TestInitializeStrategyAppliesComponentOnly(unittest.TestCase):
    def test_empty_barrier_checkpoint_restores_the_skipped_layer_inventory(self) -> None:
        """Strict recursive loading must not recreate a layer skipped during training."""

        def strategy_factory(_name, _config, _trainer, initialize, nodes):
            """Model persistent visibility state and runtime initialization separately."""
            strategy = torch.nn.Module()
            strategy.init_from_datasource = initialize
            strategy.invisible_steps = torch.nn.ParameterDict(
                {name: torch.nn.Parameter(torch.zeros(1), requires_grad=False) for name in nodes}
            )
            strategy.maybe_initialize_buffers = mock.Mock()
            strategy.refresh_runtime_scenes = mock.Mock()
            return strategy

        def make_model(empty_barrier: bool) -> GaussiansComposite:
            """Build real module state with a controllable deferred-empty barrier layer."""
            model = _bare(None)
            model.obj_track_ids = {}
            model.trainer_config = cast(Any, object())
            model._gaussians_strategy_config = _ns(name="mcmc", exclude_layer_ids=[])
            nodes = {}
            for name in ("background", "barriers"):
                node = torch.nn.Linear(1, 1, bias=False)
                node.config = _ns(motion_source_layer=None)
                node.maybe_initialize_gaussians = mock.Mock(
                    side_effect=NoPointsFoundException("no construction detections")
                    if empty_barrier and name == "barriers"
                    else None
                )
                nodes[name] = node
            model.config = cast(Any, _ns(layers={name: node.config for name, node in nodes.items()}))
            model.gaussians_nodes = TypedModuleDict(cast(Any, nodes))
            model.gaussians_strategy = cast(Any, strategy_factory("mcmc", None, None, False, nodes))
            model.register_load_state_dict_pre_hook(model._restore_initialization_skipped_layers)
            return model

        with (
            mock.patch.object(BaseGaussianStrategy, "factory", side_effect=strategy_factory),
            mock.patch.object(GaussiansComposite, "_get_layers_config", return_value=cast(Any, object())),
            mock.patch("nurec.models.gaussians.gaussians_composite.collect.CreateGaussianParameterCollector"),
        ):
            trained = make_model(empty_barrier=True)
            trained.initialize_strategy_and_maybe_gaussians()
            checkpoint = torch.nn.ModuleDict({"model": trained}).state_dict()
            self.assertEqual(checkpoint["model._extra_state"]["initialization_skipped_layer_ids"], ["barriers"])

            restored = make_model(empty_barrier=False)
            restored._initialization_skipped_layer_ids = {"previously_skipped_layer"}
            root = torch.nn.ModuleDict({"model": restored})
            root.load_state_dict(checkpoint, strict=True, assign=True)
            self.assertEqual(list(restored.gaussians_nodes), ["background"])
            self.assertEqual(
                list(cast(torch.nn.ParameterDict, restored.gaussians_strategy.invisible_steps)), ["background"]
            )
            self.assertNotIn("barriers", restored.config.layers)
            cast(mock.Mock, restored.gaussians_nodes["background"].maybe_initialize_gaussians).assert_not_called()
            cast(mock.Mock, restored.gaussians_strategy.maybe_initialize_buffers).assert_called_once_with(
                restored.gaussians_nodes
            )
            self.assertEqual(root.state_dict()["model._extra_state"], checkpoint["model._extra_state"])
            torch.testing.assert_close(
                restored.gaussians_nodes["background"].weight, trained.gaussians_nodes["background"].weight
            )
            restored_strategy = restored.gaussians_strategy
            root.load_state_dict(checkpoint, strict=True, assign=True)
            self.assertIs(restored.gaussians_strategy, restored_strategy)
            cast(mock.Mock, restored_strategy.maybe_initialize_buffers).assert_called_once()

            # No skip marker means missing state is still an error, not permission
            # to discard a configured layer from a damaged/legacy checkpoint.
            unmarked = copy.deepcopy(checkpoint)
            del unmarked["model._extra_state"]["initialization_skipped_layer_ids"]
            fresh = make_model(empty_barrier=False)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                torch.nn.ModuleDict({"model": fresh}).load_state_dict(unmarked, strict=True, assign=True)

            for key in ("model.gaussians_nodes.barriers.weight", "model.gaussians_strategy.invisible_steps.barriers"):
                with self.subTest(conflicting_key=key):
                    conflicting = copy.deepcopy(checkpoint)
                    conflicting[key] = torch.ones(1, 1)
                    with self.assertRaisesRegex(ValueError, "Skipped Gaussian layer.*checkpoint state"):
                        root.load_state_dict(conflicting, strict=True, assign=True)

            for marker in (None, "barriers", [1], [""], ["barriers.weight"]):
                with self.subTest(invalid_skip_marker=marker):
                    invalid = copy.deepcopy(checkpoint)
                    invalid["model._extra_state"]["initialization_skipped_layer_ids"] = marker
                    with self.assertRaisesRegex(ValueError, "must be a list of layer names"):
                        root.load_state_dict(invalid, strict=True, assign=True)

            explicit = make_model(empty_barrier=False)
            explicit.component_only = _enabled_config(barriers=ComponentOnlyLayerConfig())
            with self.assertRaisesRegex(ValueError, "non-disabled layer 'barriers'.*skipped"):
                torch.nn.ModuleDict({"model": explicit}).load_state_dict(checkpoint, strict=True, assign=True)

            # Existing nonempty/legacy checkpoints need no new marker, and must
            # neither rebuild the strategy nor discard the barrier parameters.
            nonempty = make_model(empty_barrier=False)
            nonempty.initialize_strategy_and_maybe_gaussians()
            nonempty_state = torch.nn.ModuleDict({"model": nonempty}).state_dict()
            self.assertNotIn("initialization_skipped_layer_ids", nonempty_state["model._extra_state"])
            for explicit_empty_marker in (False, True):
                with self.subTest(explicit_empty_marker=explicit_empty_marker):
                    incoming = copy.deepcopy(nonempty_state)
                    if explicit_empty_marker:
                        incoming["model._extra_state"]["initialization_skipped_layer_ids"] = []
                    fresh = make_model(empty_barrier=False)
                    fresh._initialization_skipped_layer_ids = {"barriers"}
                    strategy = fresh.gaussians_strategy
                    fresh_root = torch.nn.ModuleDict({"model": fresh})
                    fresh_root.load_state_dict(incoming, strict=True, assign=True)
                    self.assertEqual(list(fresh.gaussians_nodes), ["background", "barriers"])
                    self.assertIs(fresh.gaussians_strategy, strategy)
                    self.assertEqual(fresh._initialization_skipped_layer_ids, set())
                    resaved = fresh_root.state_dict()
                    self.assertNotIn("initialization_skipped_layer_ids", resaved["model._extra_state"])
                    roundtrip = make_model(empty_barrier=False)
                    torch.nn.ModuleDict({"model": roundtrip}).load_state_dict(resaved, strict=True, assign=True)
                    torch.testing.assert_close(
                        roundtrip.gaussians_nodes["barriers"].weight, nonempty.gaussians_nodes["barriers"].weight
                    )

            # Missing/None extra state still clears process-local inventory;
            # normal strict loading independently enforces required keys.
            for missing_state in ({}, {"model._extra_state": None}):
                with self.subTest(missing_extra_state=missing_state):
                    fresh = make_model(empty_barrier=False)
                    fresh._initialization_skipped_layer_ids = {"barriers"}
                    fresh._restore_initialization_skipped_layers(fresh, missing_state, "model.")
                    self.assertEqual(fresh._initialization_skipped_layer_ids, set())
                    self.assertEqual(list(fresh.gaussians_nodes), ["background", "barriers"])

    def test_initializes_nodes_and_applies_freezes(self) -> None:
        model = _bare(None)
        node = _FakeRigidNode(_rigids_tracks())
        node.maybe_initialize_gaussians = cast(Any, mock.MagicMock())
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})
        strategy_stub = _ns(
            maybe_initialize_buffers=mock.MagicMock(),
            refresh_runtime_scenes=mock.MagicMock(),
        )
        model.gaussians_strategy = strategy_stub
        with (
            mock.patch.object(model, "_apply_component_only_background_scope") as scope,
            mock.patch.object(model, "_apply_component_only_frozen_tracks") as frozen,
            mock.patch.object(model, "_apply_component_only_freeze") as freeze,
        ):
            model.initialize_strategy_and_maybe_gaussians()
        cast(mock.MagicMock, node.maybe_initialize_gaussians).assert_called_once()
        strategy_stub.maybe_initialize_buffers.assert_called_once()
        strategy_stub.refresh_runtime_scenes.assert_called_once_with(model.gaussians_nodes)
        scope.assert_called_once()
        frozen.assert_called_once()
        freeze.assert_called_once()

    def test_prunes_deferred_empty_light_layer_and_rebuilds_from_unadjusted_config(self) -> None:
        model = _bare(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(),
                vehicle_lights=ComponentOnlyLayerConfig(state=ComponentState.disabled),
            )
        )
        owner = _FakeRigidNode(_rigids_tracks())
        empty_light = _FakeRigidNode(_rigids_tracks())
        owner.config = _ns(motion_source_layer=None)
        empty_light.config = _ns(motion_source_layer="dynamic_rigids")
        owner.maybe_initialize_gaussians = cast(Any, mock.MagicMock())
        empty_light.maybe_initialize_gaussians = cast(
            Any,
            mock.MagicMock(side_effect=NoPointsFoundException("all SAM masks are empty")),
        )
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_rigids": owner, "vehicle_lights": empty_light}))
        model.obj_track_ids = {}
        model.config = cast(
            Any,
            _ns(layers={"dynamic_rigids": owner.config, "vehicle_lights": empty_light.config}),
        )
        model.trainer_config = cast(Any, object())
        unadjusted_config = _ns(
            name="mcmc",
            exclude_layer_ids=["dynamic_rigids", "vehicle_lights"],
            relocate=_ns(
                frequency=100,
                same_track_layer_ids=["vehicle_lights"],
                min_gaussians_per_track={"vehicle_lights": 16},
            ),
            add=_ns(fixed_count_layer_ids=["vehicle_lights"]),
            perturb=_ns(max_position_offset_m={"vehicle_lights": 0.02}),
        )
        adjusted_runtime_config = _ns(
            name="mcmc",
            relocate=_ns(frequency=50),
            frequency_already_scaled=True,
        )
        model._gaussians_strategy_config = unadjusted_config
        model.gaussians_strategy = _ns(
            config=adjusted_runtime_config,
            init_from_datasource=True,
            maybe_initialize_buffers=mock.MagicMock(),
        )
        rebuilt_strategy = _ns(
            maybe_initialize_buffers=mock.MagicMock(),
            refresh_runtime_scenes=mock.MagicMock(),
        )
        collector = object()

        with (
            mock.patch.object(BaseGaussianStrategy, "factory", return_value=rebuilt_strategy) as strategy_factory,
            mock.patch.object(model, "_get_layers_config", return_value=cast(Any, "surviving-layers")),
            mock.patch(
                "nurec.models.gaussians.gaussians_composite.collect.CreateGaussianParameterCollector",
                return_value=collector,
            ),
            mock.patch.object(model, "_apply_component_only_background_scope"),
            mock.patch.object(model, "_apply_component_only_background_hooks"),
            mock.patch.object(model, "_apply_component_only_frozen_tracks"),
            mock.patch.object(model, "_apply_component_only_freeze"),
        ):
            model.initialize_strategy_and_maybe_gaussians()

        self.assertEqual(list(model.gaussians_nodes), ["dynamic_rigids"])
        self.assertNotIn("vehicle_lights", model.config.layers)
        self.assertIs(owner.collector, collector)
        strategy_factory.assert_called_once()
        strategy_name, rebuilt_config, trainer_config, init_from_datasource, rebuilt_nodes = (
            strategy_factory.call_args.args
        )
        self.assertEqual(strategy_name, "mcmc")
        self.assertIsNot(rebuilt_config, unadjusted_config)
        self.assertIs(trainer_config, model.trainer_config)
        self.assertTrue(init_from_datasource)
        self.assertIs(rebuilt_nodes, model.gaussians_nodes)
        self.assertEqual(rebuilt_config.exclude_layer_ids, ["dynamic_rigids"])
        self.assertEqual(rebuilt_config.relocate.same_track_layer_ids, [])
        self.assertEqual(rebuilt_config.relocate.min_gaussians_per_track, {})
        self.assertEqual(rebuilt_config.add.fixed_count_layer_ids, [])
        self.assertEqual(rebuilt_config.perturb.max_position_offset_m, {})
        self.assertEqual(rebuilt_config.relocate.frequency, 100)
        self.assertEqual(adjusted_runtime_config.relocate.frequency, 50)
        self.assertEqual(unadjusted_config.relocate.same_track_layer_ids, ["vehicle_lights"])
        rebuilt_strategy.maybe_initialize_buffers.assert_called_once_with(model.gaussians_nodes)
        rebuilt_strategy.refresh_runtime_scenes.assert_called_once_with(model.gaussians_nodes)

    def test_non_disabled_layer_with_no_points_during_deferred_initialization_fails_loud(self) -> None:
        model = _bare(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()))
        node = _FakeRigidNode(_rigids_tracks())
        node.maybe_initialize_gaussians = cast(
            Any,
            mock.MagicMock(side_effect=NoPointsFoundException("all initialization masks are empty")),
        )
        model.gaussians_nodes = TypedModuleDict({"dynamic_rigids": node})

        with self.assertRaises(ValueError) as cm:
            model.initialize_strategy_and_maybe_gaussians()

        self.assertIn("non-disabled layer 'dynamic_rigids'", str(cm.exception))
        self.assertIn("deferred initialization found no points", str(cm.exception))

    def test_prunes_constructor_time_missing_owner_and_recursive_dependents(self) -> None:
        model = _bare(None)
        owner_config = _ns(motion_source_layer=None)
        light_config = _ns(motion_source_layer="dynamic_rigids")
        nested_config = _ns(motion_source_layer="vehicle_lights")
        background_config = _ns(motion_source_layer=None)
        model.config = cast(
            Any,
            _ns(
                layers={
                    "dynamic_rigids": owner_config,
                    "vehicle_lights": light_config,
                    "nested_lights": nested_config,
                    "background": background_config,
                }
            ),
        )
        model.obj_track_ids = cast(Any, {name: object() for name in model.config.layers})
        model._component_only_track_states = {
            "vehicle_lights": {},
            "nested_lights": {},
        }
        model._component_only_track_expanded = {
            "vehicle_lights": {},
            "nested_lights": {},
        }
        nodes = cast(
            Any,
            {
                "vehicle_lights": _ns(config=light_config),
                "nested_lights": _ns(config=nested_config),
                "background": _ns(config=background_config),
            },
        )

        removed = model._prune_unavailable_motion_dependents(
            nodes,
            unavailable_layer_ids={"dynamic_rigids"},
        )

        self.assertEqual(removed, {"dynamic_rigids", "vehicle_lights", "nested_lights"})
        self.assertEqual(list(nodes), ["background"])
        self.assertEqual(list(model.config.layers), ["background"])
        self.assertEqual(set(model.obj_track_ids), {"background"})
        self.assertNotIn("vehicle_lights", model._component_only_track_states)
        self.assertNotIn("nested_lights", model._component_only_track_expanded)

    def test_prunes_deferred_component_only_caches_and_mask_spec(self) -> None:
        model = _bare(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(state=ComponentState.frozen),
                vehicle_lights=ComponentOnlyLayerConfig(state=ComponentState.frozen),
            )
        )
        owner_config = _ns(motion_source_layer=None)
        light_config = _ns(motion_source_layer="dynamic_rigids")
        nested_config = _ns(motion_source_layer="vehicle_lights")
        background_config = _ns(motion_source_layer=None)
        model.config = cast(
            Any,
            _ns(
                layers={
                    "dynamic_rigids": owner_config,
                    "vehicle_lights": light_config,
                    "static_rigids": nested_config,
                    "background": background_config,
                }
            ),
        )
        model.obj_track_ids = cast(Any, {name: object() for name in model.config.layers})
        model._component_only_layer_tracks = cast(
            Any,
            {
                "dynamic_rigids": object(),
                "vehicle_lights": object(),
                "static_rigids": object(),
            },
        )
        model._component_only_track_states = {
            "dynamic_rigids": {},
            "vehicle_lights": {},
            "static_rigids": {},
        }
        model._component_only_track_expanded = {
            "dynamic_rigids": {},
            "vehicle_lights": {},
            "static_rigids": {},
        }
        model._component_only_frozen_layers = frozenset({"dynamic_rigids", "vehicle_lights"})
        region = LayerRegionSpec(
            sources=(MaskSource.bbox,),
            operation=None,
            semantic_class_ids=(),
        )
        model._component_only_mask_spec = ComponentOnlyMaskSpec(
            supervised_layers={
                "dynamic_rigids": (region,),
                "vehicle_lights": (region,),
            },
            carved_layers={
                "static_rigids": region,
                "vehicle_lights" + DISABLED_TRACKS_SUFFIX: region,
            },
            disabled_track_regions={
                "static_rigids" + DISABLED_TRACKS_SUFFIX: region,
                "vehicle_lights" + DISABLED_TRACKS_SUFFIX: region,
            },
            include_residual=True,
            include_sky=False,
            sky_class_id=None,
            ego_class_id=None,
        )
        old_spec = model._component_only_mask_spec
        nodes = cast(
            Any,
            {
                "dynamic_rigids": _ns(config=owner_config),
                "vehicle_lights": _ns(config=light_config),
                "static_rigids": _ns(config=nested_config),
                "background": _ns(config=background_config),
            },
        )

        removed = model._prune_unavailable_motion_dependents(
            nodes,
            unavailable_layer_ids={"dynamic_rigids"},
        )

        self.assertEqual(removed, {"dynamic_rigids", "vehicle_lights", "static_rigids"})
        self.assertEqual(model._component_only_layer_tracks, {})
        self.assertEqual(model._component_only_track_states, {})
        self.assertEqual(model._component_only_track_expanded, {})
        self.assertEqual(model._component_only_frozen_layers, frozenset())
        new_spec = model._component_only_mask_spec
        assert new_spec is not None
        self.assertIsNot(new_spec, old_spec)
        self.assertEqual(new_spec.supervised_layers, {})
        self.assertEqual(new_spec.carved_layers, {})
        self.assertEqual(new_spec.disabled_track_regions, {})
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"background": torch.nn.Linear(1, 1)}))
        model._apply_component_only_freeze()

    def test_unknown_motion_source_is_not_silently_pruned(self) -> None:
        model = _bare(None)
        bad_config = _ns(motion_source_layer="typo")
        model.config = cast(Any, _ns(layers={"bad": bad_config}))
        model.obj_track_ids = cast(Any, {"bad": object()})
        nodes = cast(Any, {"bad": _ns(config=bad_config)})

        removed = model._prune_unavailable_motion_dependents(nodes)

        self.assertEqual(removed, set())
        self.assertIn("bad", nodes)
        with self.assertRaisesRegex(ValueError, "references unknown motion source 'typo'"):
            model._bind_motion_source_layers(nodes)


class _EditOperationRigidNode(RigidGaussianModel):
    def __init__(self, motion_source_layer: Optional[str]) -> None:
        """Create a lightweight owner or dependent for edit-routing tests."""

        torch.nn.Module.__init__(self)
        self.config = cast(Any, _ns(motion_source_layer=motion_source_layer))
        self.cuboid_tracks = _make_tracks([("track", "automobile", (0.0, 10.0, 0.0))])
        self._motion_source_ref = None
        self.is_static = False
        self.nearest_neighbor_track_for_lidar = False
        self.track_ply_override = mock.MagicMock(return_value=True)
        self.insert_asset = mock.MagicMock(return_value=True)
        self.update_tracks_calib_and_time_embed = mock.MagicMock()
        self.get_track_poses_and_dims = mock.MagicMock(return_value=None)
        self.validate_external_gaussian_topology_edit = mock.MagicMock()


class TestTrackOwnershipGuards(unittest.TestCase):
    def test_track_ply_override_invalidates_before_replacement(self) -> None:
        """A matching replacement invalidates cached gathers before changing its owner."""

        model = _bare(None)
        owner = _EditOperationRigidNode(None)
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_rigids": owner}))

        with (
            mock.patch.object(model, "_invalidate_collect_caches") as collect_invalidation,
            mock.patch.object(
                model,
                "_invalidate_fused_gather_generation_plan",
                side_effect=RuntimeError("invalidation sentinel"),
            ) as fused_invalidation,
            self.assertRaisesRegex(RuntimeError, "invalidation sentinel"),
        ):
            model.track_ply_override("track", cast(Any, object()), torch.eye(4), torch.zeros(3))

        fused_invalidation.assert_called_once_with()
        collect_invalidation.assert_not_called()
        cast(mock.MagicMock, owner.track_ply_override).assert_not_called()

    def test_track_ply_override_calls_deformable_and_rigid_owners(self) -> None:
        """PLY replacement reaches every owning layer but skips dependents."""

        model = _bare(None)
        owner = _EditOperationRigidNode(None)
        dependent = _EditOperationRigidNode("dynamic_rigids")
        deformable = DeformableGaussianModel.__new__(DeformableGaussianModel)
        torch.nn.Module.__init__(deformable)
        deformable.cuboid_tracks = _make_tracks([("track", "pedestrian", (0.0, 10.0, 0.0))])
        deformable.track_ply_override = mock.MagicMock(return_value=True)
        model.gaussians_nodes = TypedModuleDict(
            cast(
                Any,
                {"dynamic_rigids": owner, "vehicle_lights": dependent, "dynamic_deformables": deformable},
            )
        )

        replaced = model.track_ply_override(
            "track",
            cast(Any, object()),
            torch.eye(4),
            torch.zeros(3),
        )

        self.assertTrue(replaced)
        cast(mock.MagicMock, owner.track_ply_override).assert_called_once()
        cast(mock.MagicMock, dependent.track_ply_override).assert_not_called()
        cast(mock.MagicMock, deformable.track_ply_override).assert_called_once()

    def test_track_ply_override_ignores_rejecting_layer_that_does_not_own_track(self) -> None:
        """A table-like layer cannot reject replacement of another layer's track."""

        model = _bare(None)
        owner = _EditOperationRigidNode(None)
        rejecting = _EditOperationRigidNode(None)
        rejecting.cuboid_tracks = _make_tracks([("other", "pedestrian", (0.0, 10.0, 0.0))])
        cast(mock.MagicMock, rejecting.validate_external_gaussian_topology_edit).side_effect = NotImplementedError
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_rigids": owner, "dynamic_deformables": rejecting}))

        replaced = model.track_ply_override("track", cast(Any, object()), torch.eye(4), torch.zeros(3))

        self.assertTrue(replaced)
        cast(mock.MagicMock, owner.track_ply_override).assert_called_once()
        cast(mock.MagicMock, rejecting.validate_external_gaussian_topology_edit).assert_not_called()
        cast(mock.MagicMock, rejecting.track_ply_override).assert_not_called()

    def test_track_ply_override_preflights_all_owners_before_replacement(self) -> None:
        """A rejecting owner prevents earlier owners from being partly replaced."""

        model = _bare(None)
        first_owner = _EditOperationRigidNode(None)
        rejecting_owner = _EditOperationRigidNode(None)
        cast(
            mock.MagicMock, rejecting_owner.validate_external_gaussian_topology_edit
        ).side_effect = NotImplementedError("track PLY replacement is unsupported")
        model.gaussians_nodes = TypedModuleDict(
            cast(Any, {"dynamic_rigids": first_owner, "dynamic_deformables": rejecting_owner})
        )

        with self.assertRaisesRegex(NotImplementedError, "unsupported"):
            model.track_ply_override("track", cast(Any, object()), torch.eye(4), torch.zeros(3))

        cast(mock.MagicMock, first_owner.track_ply_override).assert_not_called()
        cast(mock.MagicMock, rejecting_owner.track_ply_override).assert_not_called()

    def test_unbound_motion_dependent_layer_fails_fast(self) -> None:
        dependent = _EditOperationRigidNode("dynamic_rigids")
        dependent.tracks_calib = mock.MagicMock()

        with self.assertRaisesRegex(RuntimeError, "bind_motion_source was not called"):
            dependent.get_motion_tracks_calib()

    def test_component_only_motion_binding_rejects_different_track_ids_or_order(self) -> None:
        owner = _EditOperationRigidNode(None)
        owner.cuboid_tracks = _rigids_tracks()
        for entries in (
            [("t_far", "automobile", (0.0, 0.0, 0.0)), ("t_near", "automobile", (0.0, 0.0, 0.0))],
            [("t_near", "automobile", (0.0, 0.0, 0.0))],
        ):
            with self.subTest(track_ids=[entry[0] for entry in entries]):
                dependent = _EditOperationRigidNode("dynamic_rigids")
                dependent.cuboid_tracks = _make_tracks(entries)
                dependent.tracks_calib = SkipTracksCalib.__new__(SkipTracksCalib)
                torch.nn.Module.__init__(dependent.tracks_calib)
                model = _bare(
                    _enabled_config(
                        dynamic_rigids=ComponentOnlyLayerConfig(), vehicle_lights=ComponentOnlyLayerConfig()
                    )
                )
                model.config = _ns(layers={"dynamic_rigids": owner.config, "vehicle_lights": dependent.config})
                with self.assertRaisesRegex(ValueError, "match target tracks in id and order"):
                    model._bind_motion_source_layers({"dynamic_rigids": owner, "vehicle_lights": dependent})
                self.assertIsNone(dependent._motion_source_ref)

    def test_motion_binding_rejects_mismatched_interpolation_modes(self) -> None:
        for field, message in (
            ("is_static", "same static interpolation mode"),
            ("nearest_neighbor_track_for_lidar", "same LiDAR track interpolation mode"),
        ):
            for owner_value in (False, True):
                with self.subTest(field=field, owner_value=owner_value):
                    owner = _EditOperationRigidNode(None)
                    dependent = _EditOperationRigidNode("dynamic_rigids")
                    owner.cuboid_tracks = dependent.cuboid_tracks = _rigids_tracks()
                    setattr(owner, field, owner_value)
                    setattr(dependent, field, not owner_value)

                    with self.assertRaisesRegex(ValueError, message):
                        dependent.bind_motion_source(owner)

                    self.assertIsNone(dependent._motion_source_ref)

    def test_motion_binding_allows_different_timestamp_ray_strides(self) -> None:
        owner = _EditOperationRigidNode(None)
        dependent = _EditOperationRigidNode("dynamic_rigids")
        owner.cuboid_tracks = dependent.cuboid_tracks = _rigids_tracks()
        owner.tracks_timestamp_estimation_ray_stride = 4
        dependent.tracks_timestamp_estimation_ray_stride = 1

        dependent.bind_motion_source(owner)

        self.assertIs(dependent.get_motion_source(), owner)
        self.assertEqual(dependent.get_motion_source().tracks_timestamp_estimation_ray_stride, 4)
        self.assertEqual(dependent.tracks_timestamp_estimation_ray_stride, 1)

    def test_insert_asset_skips_dependent_main_and_fallback_paths(self) -> None:
        model = _bare(None)
        dependent = _EditOperationRigidNode("other_owner")
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_rigids": dependent}))
        model.obj_track_ids = {
            "dynamic_rigids": _ns(config={"is_dynamic": True}),
        }
        model.cuboid_tracks = _make_tracks([("existing", "automobile", (0.0, 10.0, 0.0))])
        inserted = _make_tracks([("inserted", "automobile", (1.0, 10.0, 0.0))])

        inserted_result = model.insert_asset(
            cast(Any, object()),
            inserted,
            torch.zeros(3),
        )

        self.assertFalse(inserted_result)
        cast(mock.MagicMock, dependent.update_tracks_calib_and_time_embed).assert_not_called()
        cast(mock.MagicMock, dependent.insert_asset).assert_not_called()

    def test_remove_background_trajectory_collection_only_queries_track_owners(self) -> None:
        model = _bare(None)
        owner = _EditOperationRigidNode(None)
        dependent = _EditOperationRigidNode("dynamic_rigids")
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_rigids": owner, "vehicle_lights": dependent}))

        model.remove_background_gaussians(
            ["track"],
            torch.zeros(3),
            torch.zeros(3),
        )

        cast(mock.MagicMock, owner.get_track_poses_and_dims).assert_called_once_with("track")
        cast(mock.MagicMock, dependent.get_track_poses_and_dims).assert_not_called()


class _RecordingCalibRigidNode(RigidGaussianModel):
    """Rigid node stub whose tracks_calib records what it is asked to calibrate."""

    def __init__(self, tracks: CuboidTracks, is_static: bool = False):
        torch.nn.Module.__init__(self)  # intentionally skip RigidGaussianModel.__init__
        self.cuboid_tracks = tracks
        self.is_static = is_static
        self._motion_source_ref = None
        self.calib_inputs: list[CuboidTracks] = []
        # Stands in for the BaseTracksCalib module, which is sized to the
        # tracks the node was constructed with.
        self.tracks_calib = cast(Any, self._record)

    def _record(self, tracks: CuboidTracks) -> CuboidTracks:
        self.calib_inputs.append(tracks)
        return tracks


@_NEEDS_CUDA
class TestGetUpdatedCuboidTracks(unittest.TestCase):
    def test_calibrates_the_node_partition_not_the_full_layer_subset(self) -> None:
        # Component-only per-track selection: the node (and its
        # tracks_calib parameter sizing) holds only the non-disabled
        # partition. The calibrated-tracks export must feed exactly that
        # partition — the full layer subset would shape-mismatch the
        # calib module (and disabled tracks were never calibrated).
        full = _rigids_tracks(device=_DEVICE)
        partition = CuboidTracks.Ops.subset_from_tracks_id(full, ["t_near"])
        node = _RecordingCalibRigidNode(partition)
        model = _bare(None)
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_rigids": node}))
        model.cuboid_tracks = full
        model.obj_track_ids = {"dynamic_rigids": _ns(get_layer_tracks=lambda tracks: full)}
        updated = model.get_updated_cuboid_tracks()
        self.assertEqual(len(node.calib_inputs), 1)
        self.assertIs(node.calib_inputs[0], partition)
        self.assertEqual(updated.tracks_id, ["t_near"])

    def test_static_rigid_nodes_are_skipped(self) -> None:
        node = _RecordingCalibRigidNode(_rigids_tracks(device=_DEVICE), is_static=True)
        model = _bare(None)
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_rigids": node}))
        updated = model.get_updated_cuboid_tracks()
        self.assertEqual(node.calib_inputs, [])
        self.assertEqual(updated.n_tracks, 0)

    def test_two_track_partition_exports_sequence_tracks(self) -> None:
        """End-to-end regression for the component-only artifact-export
        crash: a node built on a two-track partition carries a REAL
        DirectTracksCalib whose parameters are sized to the partition.
        The calibrated-tracks export must run through it and serialize
        without shape mismatches."""
        from omegaconf import DictConfig

        from nurec.config.trainer import TrainerConfig
        from nurec.models.tracks_calib import DirectTracksCalib
        from nurec.utils.io.sequence_tracks import serialize_sequence_tracks

        full = _make_tracks(
            [
                ("t_a", "pedestrian", (0.0, 10.0, 0.0)),
                ("t_b", "pedestrian", (5.0, 10.0, 0.0)),
                ("t_disabled", "pedestrian", (10.0, 10.0, 0.0)),
            ],
            device=_DEVICE,
        )
        partition = CuboidTracks.Ops.subset_from_tracks_id(full, ["t_a", "t_b"])
        trainer_config = TrainerConfig(
            max_epochs=1,
            check_val_every_n_epoch=1,
            precision="32",
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        calib = DirectTracksCalib(
            DictConfig({"visualize": False, "start_global_step": 0, "fix_first_pose": False, "fix_last_pose": False}),
            trainer_config,
            partition,
        ).to(_DEVICE)
        # The calib deltas are sized to the partition's poses — feeding
        # the full layer subset through it (the pre-fix behavior) is a
        # guaranteed shape mismatch.
        self.assertEqual(calib.tracks_delta_q.shape[0], partition.tracks_poses.shape[0])
        self.assertNotEqual(calib.tracks_delta_q.shape[0], full.tracks_poses.shape[0])

        node = _RecordingCalibRigidNode(partition)
        node.tracks_calib = cast(Any, calib)
        model = _bare(None)
        model.gaussians_nodes = TypedModuleDict(cast(Any, {"dynamic_deformables": node}))

        updated = model.get_updated_cuboid_tracks()
        self.assertEqual(updated.tracks_id, ["t_a", "t_b"])

        exported = serialize_sequence_tracks(
            sequence_id="dummy_chunk_id",
            cuboid_tracks=updated,
            excluded_tracks=None,
            formats=["json"],
        )
        self.assertEqual(len(exported), 1)


class _FakeFactoryNode(BaseGaussianModel):
    """Stand-in returned by the patched BaseGaussianModel.factory."""

    camera_extra_ray_signal_infos = cast(Any, ((), ()))
    lidar_extra_ray_signal_infos = cast(Any, ((), ()))

    def __init__(self, layer_config: Any, cuboid_subset: Optional[CuboidTracks]):
        torch.nn.Module.__init__(self)  # intentionally skip BaseGaussianModel.__init__
        self.config = layer_config
        self.cuboid_subset = cuboid_subset
        self.weight: torch.nn.Parameter = torch.nn.Parameter(torch.randn(4, 3))

    def get_num_gaussians(self) -> int:
        return self.weight.shape[0]

    def get_positions(self) -> torch.nn.Parameter:
        return self.weight


@_NEEDS_CUDA
class TestConstructorWiring(unittest.TestCase):
    """Exercises the real ``GaussiansComposite.__init__`` component-only
    wiring against a generic ``ModelConfig`` fixture, with all
    module factories patched to lightweight stand-ins."""

    cfg: Any

    _TESTDATA_CONFIG_DIR = Path(__file__).resolve().parent / "testdata"

    @classmethod
    def setUpClass(cls) -> None:
        from nurec.config.parse import parse_typed_config

        cls.cfg = parse_typed_config(
            config_name=str(cls._TESTDATA_CONFIG_DIR / "apps/gaussians_composite_component_only_config.yaml"),
            hydra_args=["dataset.path=/does/not/matter", "out_dir=/does/not/matter"],
        )

    def _datasource(
        self,
        tracks: Optional[CuboidTracks],
        with_trajectories: bool = True,
        inner_datasource: object = None,
        world_to_nurec: Optional[FrameConversion] = None,
    ) -> SimpleNamespace:
        trajectory = _ns(
            T_rig_worlds=torch.eye(4).expand(3, 4, 4).contiguous(),
            T_rig_world_timestamps_us=torch.tensor([0, 500_000, 1_000_000]),
        )
        w2n = world_to_nurec if world_to_nurec is not None else FrameConversion(matrix=np.eye(4, dtype=np.float32))
        rig = _ns(rig_trajectories=[trajectory], world_to_nurec=w2n)
        return _ns(
            get_aabb=lambda: _ns(get_extent=lambda: torch.tensor([100.0, 100.0, 20.0])),
            get_cuboid_tracks=lambda dynamic_only: tracks,
            get_rig_trajectories=lambda: rig if with_trajectories else None,
            get_n_training_frames_per_camera=lambda: torch.tensor([10]),
            datasource=inner_datasource,
        )

    def _construct(
        self,
        component_only: Optional[ComponentOnlyConfig],
        datasource: Optional[SimpleNamespace] = None,
        no_points_model_name: Optional[str] = None,
    ) -> tuple[GaussiansComposite, dict]:
        captured: dict = {}

        def fake_model_factory(name, layer_config, trainer_config, ds, init_from, initializer, precision, subset, ts):
            if name == no_points_model_name:
                raise NoPointsFoundException("test initialization found no points")
            captured.setdefault("nodes", {})[len(captured.get("nodes", {}))] = name
            return _FakeFactoryNode(layer_config, subset)

        def fake_strategy_factory(name, strategy_config, trainer_config, init_from, nodes):
            captured["strategy_config"] = strategy_config
            return _ns(maybe_initialize_buffers=lambda nodes: None)

        tracks = _make_tracks(
            [("t_car", "automobile", (0.0, 10.0, 0.0)), ("t_ped", "pedestrian", (5.0, 10.0, 0.0))], device=_DEVICE
        )
        datasource = datasource or self._datasource(tracks)

        def fake_calib_factory(name, config, trainer_config, ds):
            captured["calib_name"] = name
            captured["calib_sensor_enabled"] = (config.camera.enabled, config.lidar.enabled)
            return torch.nn.Linear(2, 2)

        def fake_background_factory(name, config, trainer_config):
            captured["background_name"] = name
            return torch.nn.Linear(2, 2)

        with (
            mock.patch("nurec.models.gaussians.gaussians_composite.model_config_compatibility_check"),
            mock.patch.object(BaseCalib, "factory", side_effect=fake_calib_factory),
            mock.patch.object(BaseBackground, "factory", side_effect=fake_background_factory),
            mock.patch.object(BaseInitialization, "factory", return_value=object()),
            mock.patch.object(BaseGaussianModel, "factory", side_effect=fake_model_factory),
            mock.patch.object(BaseGaussianStrategy, "factory", side_effect=fake_strategy_factory),
            mock.patch.object(BaseGaussianRenderer, "factory", return_value=_ns()),
            mock.patch.object(BasePostProcessing, "factory", return_value=torch.nn.Linear(2, 2)),
            mock.patch.object(GaussiansComposite, "_get_layers_config", return_value={}),
            mock.patch("nurec.models.gaussians.collect.CreateGaussianParameterCollector", return_value=None),
            mock.patch("torch.cuda.Stream", return_value=None),
        ):
            model = GaussiansComposite(
                self.cfg.model,
                self.cfg.trainer,
                cast(Any, datasource),
                init_from_datasource=True,
                component_only=component_only,
            )
        return model, captured

    def test_full_training_is_unaffected(self) -> None:
        model, captured = self._construct(None)
        self.assertIsNone(model.component_only)
        self.assertIsNone(model._component_only_mask_spec)
        self.assertEqual(model._component_only_frozen_layers, frozenset())
        self.assertEqual(set(model.gaussians_nodes.keys()), set(self.cfg.model.layers.keys()))
        self.assertEqual(captured["calib_name"], self.cfg.model.calib.name)
        self.assertEqual(captured["background_name"], self.cfg.model.background.name)
        self.assertGreater(len(model.post_processings), 0)
        # Viewpoint samples for the bbox scope are collected regardless.
        self.assertEqual(model._component_only_viewpoints[0].shape[0], 3)

    def test_registers_derived_state_invalidation_load_hooks(self) -> None:
        with (
            mock.patch.object(
                GaussiansComposite,
                "register_load_state_dict_pre_hook",
                autospec=True,
            ) as register_pre_hook,
            mock.patch.object(
                GaussiansComposite,
                "register_load_state_dict_post_hook",
                autospec=True,
            ) as register_post_hook,
        ):
            model, _ = self._construct(None)

        register_pre_hook.assert_any_call(
            model,
            model._invalidate_fused_gather_generation_plan_before_load,
        )
        register_pre_hook.assert_has_calls(
            [
                mock.call(model, model._invalidate_fused_gather_generation_plan_before_load),
                mock.call(model, model._restore_initialization_skipped_layers),
                mock.call(model, model._reset_incompatible_mcmc_visibility_state_on_load),
            ]
        )
        register_post_hook.assert_any_call(
            model,
            model._invalidate_derived_render_caches_after_load,
        )

    def test_bbox_scope_origins_transformed_world_to_nurec(self) -> None:
        # The sampled rig origins are NCore-world on the standard datasource
        # path, but must be stored in the NuRec frame the cuboid tracks / Gaussian
        # positions live in. A non-identity world_to_nurec must be applied (it is
        # identity — a no-op — on the instant_nurec path).
        w2n = FrameConversion.from_origin_scale_axis(np.array([10.0, 0.0, 0.0], dtype=np.float32), 1.0, [0, 1, 2])
        tracks = _make_tracks([("t_car", "automobile", (0.0, 10.0, 0.0))], device=_DEVICE)
        model, _ = self._construct(
            _enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()),
            datasource=self._datasource(tracks, world_to_nurec=w2n),
        )
        origins = model._component_only_viewpoints[0]
        # T_rig_worlds are identity (world origin [0,0,0]) -> world_to_nurec maps
        # each to [-10, 0, 0]; without the transform they'd stay [0,0,0].
        expected = torch.tensor([-10.0, 0.0, 0.0], dtype=origins.dtype).expand_as(origins)
        self.assertTrue(torch.allclose(origins, expected))

    def test_deformable_mixed_enabled_frozen_tracks_rejected(self) -> None:
        # A deformable layer's tracks share one deformation network, so mixing
        # enabled and frozen per-track states cannot strictly freeze the frozen
        # track — it must fail loud at construction.
        tracks = _make_tracks(
            [("p1", "pedestrian", (0.0, 10.0, 0.0)), ("p2", "pedestrian", (5.0, 10.0, 0.0))], device=_DEVICE
        )
        with self.assertRaises(ValueError) as cm:
            self._construct(
                _enabled_config(
                    dynamic_deformables=ComponentOnlyLayerConfig(
                        tracks={
                            "p1": ComponentOnlyTrackConfig(state=ComponentState.enabled),
                            "p2": ComponentOnlyTrackConfig(state=ComponentState.frozen),
                        }
                    )
                ),
                datasource=self._datasource(tracks, with_trajectories=False),
            )
        self.assertIn("deformable", str(cm.exception))
        self.assertIn("mixed enabled/frozen", str(cm.exception))

    def test_disabled_section_is_inert(self) -> None:
        model, _ = self._construct(ComponentOnlyConfig(enabled=False))
        self.assertIsNone(model.component_only)
        self.assertEqual(set(model.gaussians_nodes.keys()), set(self.cfg.model.layers.keys()))

    def test_component_only_disables_unlisted_components(self) -> None:
        tracks = _make_tracks(
            [("t_car", "automobile", (0.0, 10.0, 0.0)), ("t_ped", "pedestrian", (5.0, 10.0, 0.0))], device=_DEVICE
        )
        model, captured = self._construct(
            _enabled_config(dynamic_deformables=ComponentOnlyLayerConfig()),
            datasource=self._datasource(tracks, with_trajectories=False),
        )
        self.assertEqual(set(model.gaussians_nodes.keys()), {"dynamic_deformables"})
        self.assertEqual(captured["calib_name"], "skip-calib")
        # Disabling calib must reach the true disabled boundary — both sensor
        # flags cleared — so no trainable pose params are built/optimized.
        self.assertEqual(captured["calib_sensor_enabled"], (False, False))
        self.assertEqual(captured["background_name"], "skip-background")
        self.assertEqual(len(model.post_processings), 0)
        # No trajectories: no viewpoint samples.
        self.assertEqual(model._component_only_viewpoints[0].shape[0], 0)
        # No live datasource: no mask spec (inference-from-json path).
        self.assertIsNone(model._component_only_mask_spec)
        # Declared-but-disabled tracked layers keep their track subsets
        # for the residual carve.
        self.assertEqual(model._component_only_layer_tracks["dynamic_rigids"].tracks_id, ["t_car"])

    def test_all_layers_disabled_fails_loud(self) -> None:
        with self.assertRaises(ValueError) as cm:
            self._construct(ComponentOnlyConfig(enabled=True, layers={}))
        self.assertIn("no Gaussian layer was constructed", str(cm.exception))

    def test_non_disabled_layer_with_no_initial_points_fails_loud(self) -> None:
        tracks = _make_tracks([("t_ped", "pedestrian", (5.0, 10.0, 0.0))], device=_DEVICE)
        with self.assertRaises(ValueError) as cm:
            self._construct(
                _enabled_config(dynamic_deformables=ComponentOnlyLayerConfig()),
                datasource=self._datasource(tracks, with_trajectories=False),
                no_points_model_name="deformable-gaussians",
            )
        self.assertIn("non-disabled layer 'dynamic_deformables'", str(cm.exception))
        self.assertIn("initialization found no points", str(cm.exception))

    def test_non_disabled_track_layer_with_empty_selector_fails_loud(self) -> None:
        # Keep background constructed so the generic all-layers-disabled guard
        # cannot hide a skipped, but still supervised, deformable layer.
        tracks = _make_tracks([("t_car", "automobile", (0.0, 10.0, 0.0))], device=_DEVICE)
        with self.assertRaises(ValueError) as cm:
            self._construct(
                _enabled_config(
                    background=ComponentOnlyLayerConfig(),
                    dynamic_deformables=ComponentOnlyLayerConfig(),
                ),
                datasource=self._datasource(tracks),
            )
        self.assertIn("non-disabled track-backed layer 'dynamic_deformables'", str(cm.exception))
        self.assertIn("matched no cuboid tracks", str(cm.exception))

    def test_frozen_layer_lands_on_strategy_excludes(self) -> None:
        model, captured = self._construct(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(),
                dynamic_deformables=ComponentOnlyLayerConfig(state=ComponentState.frozen),
            )
        )
        self.assertEqual(model._component_only_frozen_layers, frozenset({"dynamic_deformables"}))
        self.assertIn("dynamic_deformables", captured["strategy_config"].exclude_layer_ids)
        for param in model.gaussians_nodes["dynamic_deformables"].parameters():
            self.assertFalse(param.requires_grad)

    def test_per_track_partition_builds_node_from_subset(self) -> None:
        tracks = _make_tracks(
            [("t_car_a", "automobile", (0.0, 10.0, 0.0)), ("t_car_b", "automobile", (5.0, 10.0, 0.0))], device=_DEVICE
        )
        model, captured = self._construct(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    tracks={"t_car_b": ComponentOnlyTrackConfig(state=ComponentState.disabled)}
                )
            ),
            datasource=self._datasource(tracks),
        )
        node = cast(_FakeFactoryNode, model.gaussians_nodes["dynamic_rigids"])
        assert node.cuboid_subset is not None
        self.assertEqual(node.cuboid_subset.tracks_id, ["t_car_a"])
        # All constructed rows are enabled: the layer keeps its
        # densification — disabled tracks are absent, and their presence
        # in the config alone must not cost the layer its MCMC.
        self.assertNotIn("dynamic_rigids", captured["strategy_config"].exclude_layer_ids)

    def test_component_only_disabled_adds_no_exclusions(self) -> None:
        _, captured = self._construct(ComponentOnlyConfig(enabled=False))
        self.assertEqual(
            set(captured["strategy_config"].exclude_layer_ids),
            set(self.cfg.model.strategy.exclude_layer_ids),
        )

    def test_enabled_only_track_partition_keeps_densification(self) -> None:
        # Layer default disabled, two tracks enabled: everything that was
        # constructed is enabled — the layer participates in MCMC.
        tracks = _make_tracks(
            [("t_car_a", "automobile", (0.0, 10.0, 0.0)), ("t_car_b", "automobile", (5.0, 10.0, 0.0))], device=_DEVICE
        )
        _, captured = self._construct(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    tracks={
                        "t_car_a": ComponentOnlyTrackConfig(state=ComponentState.enabled),
                        "t_car_b": ComponentOnlyTrackConfig(state=ComponentState.enabled),
                    },
                )
            ),
            datasource=self._datasource(tracks),
        )
        self.assertNotIn("dynamic_rigids", captured["strategy_config"].exclude_layer_ids)

    def test_mixed_enabled_frozen_tracks_exclude_layer(self) -> None:
        # MCMC is not row-aware: any frozen row (gradient hook with a
        # fixed-length row mask) sidelines the whole layer.
        tracks = _make_tracks(
            [("t_car_a", "automobile", (0.0, 10.0, 0.0)), ("t_car_b", "automobile", (5.0, 10.0, 0.0))], device=_DEVICE
        )
        _, captured = self._construct(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    tracks={"t_car_b": ComponentOnlyTrackConfig(state=ComponentState.frozen)}
                )
            ),
            datasource=self._datasource(tracks),
        )
        self.assertIn("dynamic_rigids", captured["strategy_config"].exclude_layer_ids)

    def test_all_constructed_tracks_frozen_exclude_layer(self) -> None:
        tracks = _make_tracks(
            [("t_car_a", "automobile", (0.0, 10.0, 0.0)), ("t_car_b", "automobile", (5.0, 10.0, 0.0))], device=_DEVICE
        )
        _, captured = self._construct(
            _enabled_config(
                dynamic_rigids=ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    tracks={
                        "t_car_a": ComponentOnlyTrackConfig(state=ComponentState.frozen),
                        "t_car_b": ComponentOnlyTrackConfig(state=ComponentState.frozen),
                    },
                )
            ),
            datasource=self._datasource(tracks),
        )
        self.assertIn("dynamic_rigids", captured["strategy_config"].exclude_layer_ids)

    def test_explicit_exclude_layer_ids_still_wins(self) -> None:
        model_config = self.cfg.model.model_copy(deep=True)
        model_config.strategy.exclude_layer_ids = ["dynamic_rigids"]
        tracks = _make_tracks(
            [("t_car_a", "automobile", (0.0, 10.0, 0.0)), ("t_car_b", "automobile", (5.0, 10.0, 0.0))], device=_DEVICE
        )
        with mock.patch.object(self, "cfg", SimpleNamespace(model=model_config, trainer=self.cfg.trainer)):
            _, captured = self._construct(
                _enabled_config(
                    dynamic_rigids=ComponentOnlyLayerConfig(
                        tracks={"t_car_a": ComponentOnlyTrackConfig(state=ComponentState.enabled)}
                    )
                ),
                datasource=self._datasource(tracks),
            )
        self.assertIn("dynamic_rigids", captured["strategy_config"].exclude_layer_ids)

    def test_per_track_everything_disabled_skips_layer(self) -> None:
        tracks = _make_tracks([("t_car_a", "automobile", (0.0, 10.0, 0.0))], device=_DEVICE)
        model, _ = self._construct(
            _enabled_config(
                background=ComponentOnlyLayerConfig(),
                dynamic_rigids=ComponentOnlyLayerConfig(
                    tracks={"t_car_a": ComponentOnlyTrackConfig(state=ComponentState.disabled)}
                ),
            ),
            datasource=self._datasource(tracks),
        )
        self.assertNotIn("dynamic_rigids", model.gaussians_nodes.keys())
        self.assertIn("background", model.gaussians_nodes.keys())

    def test_unknown_track_id_fails_loud_at_construction(self) -> None:
        with self.assertRaises(ValueError) as cm:
            self._construct(
                _enabled_config(dynamic_rigids=ComponentOnlyLayerConfig(tracks={"typo": ComponentOnlyTrackConfig()}))
            )
        self.assertIn("typo", str(cm.exception))

    def test_mixed_background_scope_excluded_from_densification(self) -> None:
        model, captured = self._construct(
            _enabled_config(
                background=ComponentOnlyLayerConfig(
                    scope={
                        ScopeRegion.residual: ComponentState.frozen,
                        ScopeRegion.bbox: ComponentState.enabled,
                    }
                ),
                dynamic_rigids=ComponentOnlyLayerConfig(),
            )
        )
        self.assertIn("background", captured["strategy_config"].exclude_layer_ids)
        # Mixed regions: NOT frozen layer-wide (per-row hooks handle it).
        self.assertNotIn("background", model._component_only_frozen_layers)

    def test_background_frozen_bbox_enabled_residual_excluded(self) -> None:
        # The reverse mixed orientation: bbox frozen, residual enabled.
        _, captured = self._construct(
            _enabled_config(
                background=ComponentOnlyLayerConfig(
                    scope={
                        ScopeRegion.residual: ComponentState.enabled,
                        ScopeRegion.bbox: ComponentState.frozen,
                    }
                ),
                dynamic_rigids=ComponentOnlyLayerConfig(),
            )
        )
        self.assertIn("background", captured["strategy_config"].exclude_layer_ids)

    def test_background_residual_only_keeps_densification(self) -> None:
        # residual-only enabled scope keeps all rows (no drop, no
        # partition) — background participates in MCMC.
        _, captured = self._construct(
            _enabled_config(
                background=ComponentOnlyLayerConfig(
                    scope={ScopeRegion.bbox: ComponentState.disabled, ScopeRegion.residual: ComponentState.enabled}
                ),
                dynamic_rigids=ComponentOnlyLayerConfig(),
            )
        )
        self.assertNotIn("background", captured["strategy_config"].exclude_layer_ids)

    def test_background_bbox_only_scope_keeps_densification(self) -> None:
        # bbox-only scope drops out-of-frustum rows BEFORE the strategy sizes
        # its per-row buffers (row-drop is reordered ahead of buffer init), so
        # the strategy sizes to the final row count and background stays in
        # densification/pruning (single enabled region, no per-row hooks).
        model, captured = self._construct(
            _enabled_config(
                background=ComponentOnlyLayerConfig(scope={ScopeRegion.bbox: ComponentState.enabled}),
                dynamic_rigids=ComponentOnlyLayerConfig(),
            )
        )
        self.assertNotIn("background", captured["strategy_config"].exclude_layer_ids)
        self.assertNotIn("background", model._component_only_frozen_layers)

    def test_background_all_regions_frozen_excluded(self) -> None:
        model, captured = self._construct(
            _enabled_config(
                background=ComponentOnlyLayerConfig(scope={ScopeRegion.residual: ComponentState.frozen}),
                dynamic_rigids=ComponentOnlyLayerConfig(),
            )
        )
        self.assertIn("background", captured["strategy_config"].exclude_layer_ids)
        self.assertIn("background", model._component_only_frozen_layers)

    def test_mask_spec_uses_non_ncore_datasource_without_class_map(self) -> None:
        # A live non-NCORE datasource: spec built, pure-bbox config legal.
        tracks = _make_tracks([("t_car", "automobile", (0.0, 10.0, 0.0))], device=_DEVICE)
        model, _ = self._construct(
            _enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()),
            datasource=self._datasource(tracks, inner_datasource=object()),
        )
        spec = model._component_only_mask_spec
        assert spec is not None
        self.assertIsNone(spec.sky_class_id)
        self.assertIn("dynamic_rigids", spec.supervised_layers)

    def test_mask_spec_resolves_ncore_class_map(self) -> None:
        from nurec.datasets.ncore import NCOREDataSource

        ncore = object.__new__(NCOREDataSource)
        ncore.get_semantic_classes_map = lambda camera_semantics, lidar_semantics: {
            "road": 0,
            "sky": 3,
            "egocar": 4,
        }
        tracks = _make_tracks([("t_car", "automobile", (0.0, 10.0, 0.0))], device=_DEVICE)
        model, _ = self._construct(
            _enabled_config(background=ComponentOnlyLayerConfig(), dynamic_rigids=ComponentOnlyLayerConfig()),
            datasource=self._datasource(tracks, inner_datasource=ncore),
        )
        spec = model._component_only_mask_spec
        assert spec is not None
        self.assertEqual(spec.sky_class_id, 3)
        self.assertTrue(spec.include_residual)

    def test_layer_without_tracks_config_stores_no_subset(self) -> None:
        # Layer schemas without a tracks section store no subset.
        background_cls = type(self.cfg.model.layers["background"])
        original_to_dictconfig = background_cls.to_dictconfig

        def without_tracks(config_self):
            dictconfig = original_to_dictconfig(config_self)
            dictconfig.pop("tracks", None)
            return dictconfig

        with mock.patch.object(background_cls, "to_dictconfig", without_tracks):
            model, _ = self._construct(_enabled_config(background=ComponentOnlyLayerConfig()))
        self.assertNotIn("background", model._component_only_layer_tracks)
        self.assertIn("background", model.gaussians_nodes.keys())

    def test_empty_tracks_selector_still_stores_all_track_geometry(self) -> None:
        # An explicitly declared empty selector means "all tracks". Presence,
        # not mapping truthiness, decides whether mask/occlusion geometry is
        # retained for a track-backed layer.
        dynamic_config = self.cfg.model.layers["dynamic_rigids"]
        dynamic_cls = type(dynamic_config)
        original_to_dictconfig = dynamic_cls.to_dictconfig

        def with_empty_tracks(config_self):
            dictconfig = original_to_dictconfig(config_self)
            if config_self is dynamic_config:
                dictconfig["tracks"] = {}
            return dictconfig

        with mock.patch.object(dynamic_cls, "to_dictconfig", with_empty_tracks):
            model, _ = self._construct(_enabled_config(dynamic_rigids=ComponentOnlyLayerConfig()))
        self.assertIn("dynamic_rigids", model._component_only_layer_tracks)
        self.assertEqual(model._component_only_layer_tracks["dynamic_rigids"].n_tracks, 2)

    def test_no_dataset_tracks_still_constructs(self) -> None:
        model, _ = self._construct(
            _enabled_config(background=ComponentOnlyLayerConfig()),
            datasource=self._datasource(None),
        )
        self.assertIn("background", model.gaussians_nodes.keys())
        # No dataset tracks: whatever subsets got stored are empty.
        for subset in model._component_only_layer_tracks.values():
            self.assertEqual(subset.n_tracks, 0)


if __name__ == "__main__":
    unittest.main()
