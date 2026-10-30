# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Producer -> consumer test of the asset-edit row remap.

``runtime_test.py`` hands :class:`AspectOverrideRuntime` layouts built by hand, so a
regression in what ``GaussiansComposite`` RECORDS during collect would keep it green while
edited-actors renders write baked opacities into the wrong gaussians. This file drives the
real collect and feeds what it recorded into a real ``apply()``.

It lives here, not in ``gaussians_composite_test.py``, because the contract under test is
this feature's; the composite is the other half of it. That also keeps the heavier model
dependency off ``//nurec/asset_edits:runtime_test``.
"""

import unittest

from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import MagicMock, patch

import torch

from nurec.asset_edits.runtime import (
    AspectOverrideRuntime,
    AspectScheduleRegistry,
    AspectSwitch,
    BakedCluster,
    EditManifest,
    TrackSchedule,
)
from nurec.datasets.tracks import CuboidTracks
from nurec.models.composite import LayerTrackIds
from nurec.models.gaussians import collect
from nurec.models.gaussians import gaussians_composite as gaussians_composite_module
from nurec.models.gaussians.edit_hook import CollectedLayerLayout
from nurec.models.gaussians.gaussians_composite import (
    GaussiansComposite,
    _GatheredGaussianParameterSet,
)
from nurec.models.gaussians.gaussians_model import PreparedRigidMotion
from nurec.utils.batch import RenderingData


class _FakeGatherNode:
    """Minimal gaussians node: fixed rank-local parameters, no collector behavior."""

    def __init__(self, count: int) -> None:
        """Stores rank-local positions and a call counter."""
        self.parameters = {"positions": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)}
        self.get_parameters_calls = 0

    def get_parameters(self) -> dict[str, torch.Tensor]:
        """Returns the rank-local parameter dict."""
        self.get_parameters_calls += 1
        return self.parameters

    def get_layer_data(self, _context: object, parameters: dict[str, torch.Tensor]) -> SimpleNamespace:
        """Passes the parameters straight through as layer data."""
        return SimpleNamespace(positions=parameters["positions"])

    def post_process(self, _offset: int, _collector_result: object, _layer_data: object, rows=None) -> None:
        """No post-processing in this fake."""


class _RowFilterNode(_FakeGatherNode):
    """A rigid layer that row-filters on an edited-actors collect, as RigidGaussianModel does.

    Mirrors the real producer's two published facts: the collected parameters shrink to
    the surviving rows, and ``context.kept_rows`` records which UNFILTERED rows those
    were — in the gathered index space, because the filter runs after the all-gather.
    """

    def __init__(
        self, count: int, kept_rows: Optional[torch.Tensor] = None, track_keep: Optional[torch.Tensor] = None
    ) -> None:
        """Stores the rank-local parameters, the kept rows, and the per-track keep mask."""
        super().__init__(count)
        self._kept_rows = kept_rows
        self._track_keep = track_keep

    def create_tracks_edit_from_cuboid_tracks(self, _edited_tracks: CuboidTracks) -> LayerTrackIds.Edit:
        """The fixture's single edited track puts collection on the filtered path."""
        return cast(LayerTrackIds.Edit, SimpleNamespace(mapping=torch.tensor([0], dtype=torch.long)))

    def get_motion_source(self) -> "_RowFilterNode":
        """This fixture owns the motion of its single track."""
        return self

    def prepare_motion(
        self,
        _rendering_data: RenderingData,
        _frustum_cull: Optional[collect.TracksFrustumCullData],
        _is_training_batch: bool,
        tracks_edit: LayerTrackIds.Edit | None = None,
    ) -> PreparedRigidMotion:
        """Supply the collection prepass with fixed motion, including actor edits."""
        return PreparedRigidMotion(
            poses=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            keep_mask=self._track_keep if self._track_keep is not None else torch.ones(1, dtype=torch.bool),
            timestamps_us=torch.zeros(1, dtype=torch.int64),
        )

    def get_layer_data(self, context: object, parameters: dict[str, torch.Tensor]) -> SimpleNamespace:
        """Filters the gathered rows down to ``kept_rows`` and publishes the mapping.

        Also returns the collector's real outputs: ``keep_mask`` is PER TRACK and
        ``tracks_ids`` is PER ROW, which is what the composite expands into the per-row
        mask the runtime gathers by.
        """
        if getattr(context, "tracks_edit", None) is not None and self._kept_rows is not None:
            parameters["positions"] = parameters["positions"][self._kept_rows]
            context.kept_rows = self._kept_rows  # type: ignore[attr-defined]
        rows = parameters["positions"].shape[0]
        return SimpleNamespace(
            positions=parameters["positions"],
            keep_mask=self._track_keep,
            tracks_ids=torch.zeros(rows, dtype=torch.int32) if self._track_keep is not None else None,
        )


class CollectedEditLayoutsTest(unittest.TestCase):
    """The asset-edit row-remap contract, PRODUCER through CONSUMER.

    Every other test of this mechanism hands ``AspectOverrideRuntime`` layouts built by
    hand, so a regression on the recording side keeps them all green while edited-actors
    renders write baked opacities into the wrong gaussians. These drive the real collect
    and feed what it recorded into a real ``apply()``.

    The gather is faked at world_size 2 on purpose: ``kept_rows`` indexes the ALL-GATHERED
    layer, not the rank-local one, and only a gathered-parameter producer makes that
    distinction observable.
    """

    _GATHERED = 4  # 2 ranks x 2 rows
    _LOCAL = 2

    def _collect(
        self,
        kept_rows: Optional[torch.Tensor],
        *,
        edited: bool,
        track_keep: Optional[torch.Tensor] = None,
        sensor_names: tuple = ("cam",),
    ) -> GaussiansComposite:
        """Run one collect through the real GaussiansComposite and return it."""
        node = _RowFilterNode(self._LOCAL, kept_rows, track_keep)
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite.gaussians_nodes = {"dyn": node}  # type: ignore[assignment]
        composite.debug_viz = False
        composite._visibility_local_ranges = None
        # The collect only records layouts when a hook is installed, so install one.
        composite._edit_overrides = self._runtime([0])
        collected_rows = self._GATHERED if kept_rows is None or not edited else int(kept_rows.numel())
        composite.collector = MagicMock()
        composite.collector.collect.return_value = collect.CollectorResult(
            positions=torch.empty((collected_rows, 3)),
            rotations=torch.empty((collected_rows, 4)),
            scales=torch.empty((collected_rows, 3)),
            densities=torch.empty((collected_rows, 1)),
            extra_signal=torch.empty((collected_rows, 0)),
            camera_extra_signal=torch.empty((collected_rows, 0)),
            lidar_extra_signal=torch.empty((collected_rows, 0)),
            features=torch.empty((collected_rows, 0)),
        )
        rendering_data = cast(
            RenderingData,
            SimpleNamespace(timestamps_startend_us_cpu=torch.zeros((1, 2), dtype=torch.int64)),
        )
        # Hand-built rather than gathered for real: the all-gather itself needs CUDA and
        # a process group, and what this test is about is what collect RECORDS given
        # already-gathered parameters — the gathered layer is twice the rank-local one.
        gathered = {
            "dyn": _GatheredGaussianParameterSet(
                parameters={"positions": torch.empty((self._GATHERED, 3))},
                local_count=self._LOCAL,
                counts_by_rank=(self._LOCAL, self._LOCAL),
            )
        }
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(composite, "get_gaussians_node_ids", return_value=["dyn"]),
            patch.object(gaussians_composite_module, "RigidGaussianModel", _RowFilterNode),
        ):
            for sensor_name in sensor_names:
                composite.collect_gaussian_parameters(
                    rendering_data,
                    edited_cuboid_tracks=cast(CuboidTracks, object()) if edited else None,
                    gaussian_nodes=["dyn"],
                    gathered_parameters=gathered,
                    sensor_name=sensor_name,
                )
        return composite

    @staticmethod
    def _runtime(on_indices: list[int]) -> AspectOverrideRuntime:
        """A runtime whose single cluster is shown by aspect 'a' on ('dyn', 0)."""
        manifest = EditManifest(
            clusters=[
                BakedCluster(
                    cluster_id="c",
                    edit_id="e",
                    layer_id="dyn",
                    nurec_track_id=0,
                    aspect="a",
                    on_indices=on_indices,
                    centroid_world=[0.0, 0.0, 0.0],
                )
            ]
        )
        registry = AspectScheduleRegistry()
        registry.set_many([TrackSchedule("dyn", 0, "e", [AspectSwitch(["a"], 0)])])
        return AspectOverrideRuntime(manifest=manifest, registry=registry)

    def test_recorded_kept_rows_index_the_gathered_layer_not_the_local_one(self) -> None:
        """kept_rows lives in the GATHERED layer's row space (row 3 exceeds the 2-row rank)."""
        composite = self._collect(torch.tensor([1, 3]), edited=True)
        layout = composite._collected_edit_layouts["cam"]["dyn"]
        self.assertEqual(layout.layer_size, 2)
        assert layout.kept_rows is not None
        self.assertEqual(layout.kept_rows.tolist(), [1, 3])  # 3 only exists post-gather

    def test_filtered_collect_shows_exactly_the_surviving_baked_rows(self) -> None:
        """Baked rows 1 and 3 survive the filter and land at collected positions 0 and 1."""
        composite = self._collect(torch.tensor([1, 3]), edited=True)
        params = {"densities": torch.zeros(2)}
        self._runtime([1, 3]).apply(params, 0, layouts=composite._collected_edit_layouts["cam"])
        self.assertGreater(float(params["densities"][0]), 0.9)
        self.assertGreater(float(params["densities"][1]), 0.9)

    def test_rows_dropped_by_the_filter_touch_nothing(self) -> None:
        """A cluster whose baked rows were all filtered out must not write ANY collected row."""
        composite = self._collect(torch.tensor([1, 3]), edited=True)
        params = {"densities": torch.full((2,), 0.5)}
        self._runtime([0, 2]).apply(params, 0, layouts=composite._collected_edit_layouts["cam"])
        self.assertEqual(params["densities"].tolist(), [0.5, 0.5])

    def test_unfiltered_collect_records_an_identity_layout(self) -> None:
        """Without an edited-actors filter the layer is collected whole and maps 1:1."""
        composite = self._collect(None, edited=False)
        layout = composite._collected_edit_layouts["cam"]["dyn"]
        self.assertIsNone(layout.kept_rows)
        self.assertEqual((layout.layer_offset, layout.layer_size), (0, 4))
        params = {"densities": torch.zeros(4)}
        self._runtime([1, 3]).apply(params, 0, layouts=composite._collected_edit_layouts["cam"])
        self.assertEqual([i for i, v in enumerate(params["densities"].tolist()) if v > 0.9], [1, 3])

    def test_layout_records_are_keyed_per_sensor(self) -> None:
        """A combined camera+lidar forward records one layout per sensor: consuming one
        sensor's record must leave the other's intact (one shared slot would hand the
        lidar render the camera's record)."""
        composite = self._collect(torch.tensor([1, 3]), edited=True, sensor_names=("cam", "lidar"))
        self.assertEqual(set(composite._collected_edit_layouts), {"cam", "lidar"})
        cam = composite._collected_edit_layouts.pop("cam")
        self.assertIn("dyn", cam)
        self.assertEqual(set(composite._collected_edit_layouts), {"lidar"})
        assert composite._collected_edit_layouts["lidar"]["dyn"].kept_rows is not None


class RenderHookInvocationTest(unittest.TestCase):
    """The composite actually calls the hook, and a hook fault fails the frame.

    `_render_collected` is the only place the override runtime runs in production, so
    without these the whole feature can be deleted with the suite still green.
    """

    def _composite(self, hook: Any) -> GaussiansComposite:
        """A composite with `hook` installed and one recorded camera layout."""
        composite = GaussiansComposite.__new__(GaussiansComposite)
        torch.nn.Module.__init__(composite)
        composite._edit_overrides = hook
        composite._collected_edit_layouts = {
            "cam": {"dyn": CollectedLayerLayout(layer_offset=0, layer_size=2, layer_id="dyn")}
        }
        return composite

    @staticmethod
    def _render(composite: GaussiansComposite, params: dict) -> Any:
        """Drive _render_collected with a stub renderer and minimal rendering data."""
        renderer = MagicMock()
        renderer.render.return_value = SimpleNamespace(normal=None)
        rendering_data = cast(
            RenderingData,
            SimpleNamespace(timestamps_startend_us_cpu=torch.zeros((1, 2), dtype=torch.int64)),
        )
        with patch.object(GaussiansComposite, "get_n_active_features", return_value=3, create=True):
            composite._render_collected(rendering_data, [], params, (), renderer, True, "cam")
        return renderer

    def test_hook_receives_this_sensors_layouts_and_mutates_the_collected_tensors(self) -> None:
        """The hook is invoked with the recorded layouts, and its writes reach the renderer."""
        seen: dict = {}

        class _Hook:
            def apply(self, gaussian_parameters, frame_timestamp_us, layouts):
                seen["layouts"] = layouts
                gaussian_parameters["densities"][:] = 0.75

        composite = self._composite(_Hook())
        params = {"densities": torch.zeros(2)}
        renderer = self._render(composite, params)

        self.assertIn("dyn", seen["layouts"])
        self.assertEqual(seen["layouts"]["dyn"].layer_size, 2)
        torch.testing.assert_close(params["densities"], torch.full((2,), 0.75))
        renderer.render.assert_called_once()
        # The record is consumed, so a later frame cannot silently reuse a stale layout.
        self.assertNotIn("cam", composite._collected_edit_layouts)

    def test_a_missing_layout_record_fails_the_frame(self) -> None:
        """With a hook installed, a render whose collect published no layouts is refused.

        The runtime has no fallback layout to guess with, and guessing is how a
        row-filtered layer gets written at the wrong offsets.
        """

        class _Hook:
            def apply(self, gaussian_parameters, frame_timestamp_us, layouts):
                raise AssertionError("must not be reached")

        composite = self._composite(_Hook())
        composite._collected_edit_layouts = {}
        with self.assertRaisesRegex(RuntimeError, "published no layer layouts"):
            self._render(composite, {"densities": torch.zeros(2)})

    def test_a_hook_fault_fails_the_frame_and_leaves_the_hook_installed(self) -> None:
        """A raising hook aborts the render rather than emitting a half-edited frame."""

        class _Boom:
            def apply(self, gaussian_parameters, frame_timestamp_us, layouts):
                gaussian_parameters["densities"][0] = 0.5  # partial write, then fail
                raise RuntimeError("boom")

        hook = _Boom()
        composite = self._composite(hook)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            self._render(composite, {"densities": torch.zeros(2)})
        # Still installed: the fault may be transient, and silently disabling edits for
        # every later frame is the failure mode this policy exists to avoid.
        self.assertIs(composite._edit_overrides, hook)


class CollectKeepExpansionTest(unittest.TestCase):
    """The collector's per-TRACK visibility mask reaches the runtime as a per-ROW mask.

    The runtime gathers this by collected row, so a mask left per-track would either
    gather out of bounds or silently re-show rows the collector dropped.
    """

    def test_recorded_collect_keep_is_per_collected_row(self) -> None:
        track_keep = torch.tensor([False], dtype=torch.bool)  # the single track is hidden
        composite = CollectedEditLayoutsTest()._collect(None, edited=False, track_keep=track_keep)
        layout = composite._collected_edit_layouts["cam"]["dyn"]
        collect_keep = layout.collect_keep
        assert collect_keep is not None
        self.assertEqual(int(collect_keep.shape[0]), layout.layer_size)
        self.assertFalse(bool(collect_keep.any()))
