# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the edit manifest/schedule data model and runtime hook. GPU-free.

ControlAnimation is pure transport data for external animator processes, and the
registry/schedule types are dumb selection state.
"""

import json
import tempfile
import unittest
import zipfile

from pathlib import Path

import torch

from nurec.asset_edits.runtime import (
    _SH_C0,
    _VISIBLE_OPACITY,
    EDIT_MANIFEST_FILENAME,
    AspectOverrideRuntime,
    AspectScheduleRegistry,
    AspectSwitch,
    BakedCluster,
    ControlAnimation,
    EditManifest,
    EditSchedule,
    TrackSchedule,
    _active_aspects_at,
    layer_track_digest,
    load_manifest_from_usdz,
    validate_aspect_gain,
)
from nurec.models.gaussians.edit_hook import CollectedLayerLayout


US = 1_000_000


def _appearance() -> ControlAnimation:
    """Builds a fully-populated ControlAnimation exercising every optional transport field."""
    return ControlAnimation(
        edit_id="asset_1",
        layer_id="dyn",
        nurec_track_id=5,
        control="appearancemode",
        states=["off", "sequence", "alternate"],
        default_state="off",
        cycle_us=US,
        windows_by_state={"alternate": {"a": [(0.0, 0.5)], "b": [(0.5, 1.0)]}},
        members_by_state={"alternate": ["a", "b", "static"]},
        shared_elements=["b"],
        gains_by_state={"alternate": {"a": [(0.0, 0.2), (0.5, 1.0)]}},
        cold_rgb_by_state={"alternate": {"a": [0.1, 0.2, 0.3]}},
        semantics=["APPEARANCE_MODE"],
        state_aspects={"off": "appearancemode:off", "alternate": "appearancemode:alternate"},
        element_aspects={
            "a": ["appearancemode:alternate:a"],
            "b": ["appearancemode:shared:b"],
        },
    )


class ManifestRoundTripTest(unittest.TestCase):
    """EditManifest survives JSON to_dict/from_dict round-trips, with and without controls."""

    def test_manifest_roundtrip_with_controls(self):
        """ControlAnimation entries survive a JSON round-trip through the manifest unchanged."""
        manifest = EditManifest(clusters=[], controls=[_appearance()])
        data = json.loads(json.dumps(manifest.to_dict()))
        loaded = EditManifest.from_dict(data)
        self.assertEqual(loaded.controls, manifest.controls)

    def test_manifest_without_controls(self):
        """Absent controls load as an empty list and are omitted from serialized output."""
        loaded = EditManifest.from_dict({"clusters": []})
        self.assertEqual(loaded.controls, [])
        self.assertNotIn("controls", EditManifest(clusters=[]).to_dict())


class TrackScheduleTest(unittest.TestCase):
    """TrackSchedule serialization and AspectScheduleRegistry replacement semantics."""

    def test_aspect_gains_roundtrip(self):
        """Per-aspect gains with optional cold RGB survive JSON, and switch state stays queryable."""
        fs = TrackSchedule(
            layer_id="dyn",
            nurec_track_id=5,
            edit_id="live:170",
            switches=[AspectSwitch(aspects=["panelstate:on"], t_us=0)],
            aspect_gains={"detailstate:shared:Detail/surface": (0.42, [0.1, 0.2, 0.3]), "x": (1.0, None)},
        )
        loaded = TrackSchedule.from_dict(json.loads(json.dumps(fs.to_dict())))
        self.assertEqual(loaded.aspect_gains, fs.aspect_gains)
        self.assertEqual(loaded.active_state_at(123), {"panelstate:on"})

    def test_registry_set_many_replaces(self):
        """set_many replaces the registry's whole track map instead of merging into it."""
        registry = AspectScheduleRegistry()
        a = TrackSchedule("dyn", 1, "a", [AspectSwitch(["x"], 0)])
        b = TrackSchedule("dyn", 2, "b", [AspectSwitch(["y"], 0)])
        registry.set_many([a, b])
        self.assertEqual(set(registry.tracks), {("dyn", 1), ("dyn", 2)})
        registry.set_many([a])
        self.assertEqual(set(registry.tracks), {("dyn", 1)})
        self.assertEqual(_active_aspects_at(registry.tracks, 0), {("dyn", 1): {"x"}})


class RuntimeApplyTest(unittest.TestCase):
    """CPU apply() coverage: show/hide plus sender-evaluated per-aspect gains."""

    def _runtime(self) -> AspectOverrideRuntime:
        """Builds a runtime with one two-row 'panelstate:on' cluster in layer 'dyn'."""
        cluster = BakedCluster(
            cluster_id="b",
            edit_id="fx",
            layer_id="dyn",
            nurec_track_id=0,
            aspect="panelstate:on",
            on_indices=[0, 1],
            centroid_world=[0.0, 0.0, 0.0],
        )
        return AspectOverrideRuntime(EditManifest(clusters=[cluster]))

    def _params(self) -> dict[str, torch.Tensor]:
        """Returns 4-row zeroed densities and all-ones features (DC plus one higher-order band)."""
        return {
            "densities": torch.zeros(4),
            # DC (3) + one higher-order coefficient band (3), all ones.
            "features": torch.ones(4, 6),
        }

    def test_apply_gain_scales_alpha_and_lerps_dc(self):
        """A 0.5 gain halves cluster alpha, lerps DC toward the cold color, and scales higher-order SH."""
        runtime = self._runtime()
        cold = [0.1, 0.1, 0.1]
        runtime.registry.set_many(
            [
                TrackSchedule(
                    layer_id="dyn",
                    nurec_track_id=0,
                    edit_id="fx",
                    switches=[AspectSwitch(["panelstate:on"], 0)],
                    aspect_gains={"panelstate:on": (0.5, cold)},
                )
            ]
        )
        params = self._params()
        runtime.apply(params, frame_timestamp_us=10, layouts=_layouts({"dyn": (0, 4, None)}))

        densities = params["densities"]
        self.assertAlmostEqual(float(densities[0]), _VISIBLE_OPACITY * 0.5, places=5)
        self.assertAlmostEqual(float(densities[1]), _VISIBLE_OPACITY * 0.5, places=5)
        self.assertEqual(float(densities[2]), 0.0)  # untouched row
        self.assertEqual(float(densities[3]), 0.0)  # untouched row

        cold_dc = (0.1 - 0.5) / _SH_C0
        expected_dc = cold_dc + (1.0 - cold_dc) * 0.5
        features = params["features"]
        for c in range(3):
            self.assertAlmostEqual(float(features[0, c]), expected_dc, places=5)
        self.assertAlmostEqual(float(features[0, 3]), 0.5, places=5)  # higher-order scaled by gain
        self.assertEqual(float(features[3, 0]), 1.0)  # untouched row

    def test_apply_full_gain_shows_plain(self):
        """An active aspect with no gain entry shows rows at full opacity and leaves colors untouched."""
        runtime = self._runtime()
        runtime.registry.set_many(
            [
                TrackSchedule(
                    layer_id="dyn",
                    nurec_track_id=0,
                    edit_id="fx",
                    switches=[AspectSwitch(["panelstate:on"], 0)],
                )
            ]
        )
        params = self._params()
        runtime.apply(params, frame_timestamp_us=10, layouts=_layouts({"dyn": (0, 4, None)}))
        self.assertAlmostEqual(float(params["densities"][0]), _VISIBLE_OPACITY, places=5)
        self.assertEqual(float(params["features"][0, 0]), 1.0)  # colors untouched at gain 1

    def test_apply_off_hides_on_cluster(self):
        """An explicit all-hidden switch zeroes the cluster's opacities while unrelated rows keep theirs."""
        runtime = self._runtime()
        runtime.registry.set_many(
            [
                TrackSchedule(
                    layer_id="dyn",
                    nurec_track_id=0,
                    edit_id="fx",
                    switches=[AspectSwitch([], 0)],  # explicit all-hidden
                )
            ]
        )
        params = self._params()
        params["densities"] = torch.full((4,), 0.7)
        runtime.apply(params, frame_timestamp_us=10, layouts=_layouts({"dyn": (0, 4, None)}))
        self.assertEqual(float(params["densities"][0]), 0.0)  # inactive cluster hidden
        self.assertEqual(float(params["densities"][1]), 0.0)
        self.assertAlmostEqual(float(params["densities"][3]), 0.7, places=6)  # untouched row


def _layouts(spec: dict) -> dict:
    """Build the per-layer collected layouts a real collect would hand apply().

    ``spec`` maps layer id to ``(offset, collected_size, kept_rows[, collect_keep])``.
    """
    out = {}
    for layer_id, entry in spec.items():
        offset, size, kept, *rest = entry
        out[layer_id] = CollectedLayerLayout(
            layer_offset=offset,
            layer_size=size,
            layer_id=layer_id,
            kept_rows=kept,
            collect_keep=rest[0] if rest else None,
        )
    return out


class MapRowsTest(unittest.TestCase):
    """_map_rows translates baked full-layer row indices into collected tensor positions."""

    def test_unfiltered_layer_maps_identity(self) -> None:
        """Without a row filter, rows map through unchanged and no validity mask is returned."""
        layout = CollectedLayerLayout(layer_offset=0, layer_size=4)
        rows = torch.tensor([0, 3], dtype=torch.long)
        mapped, valid = AspectOverrideRuntime(EditManifest(clusters=[]))._map_rows(layout, rows)
        self.assertIsNone(valid)
        self.assertTrue(torch.equal(mapped, rows))

    def test_filtered_layer_compresses_rows_with_validity(self) -> None:
        """Filtered-out rows are flagged invalid; survivors map to their compressed collected positions."""
        # Full layer has 10 rows; only originals 2, 3, 7 survived the row filter.
        layout = CollectedLayerLayout(layer_offset=0, layer_size=3, kept_rows=torch.tensor([2, 3, 7]))
        rows = torch.tensor([2, 5, 7], dtype=torch.long)
        mapped, valid = AspectOverrideRuntime(EditManifest(clusters=[]))._map_rows(layout, rows)
        assert valid is not None
        self.assertEqual(valid.tolist(), [True, False, True])
        self.assertEqual(mapped[valid].tolist(), [0, 2])  # collected positions

    def test_out_of_range_rows_are_invalid(self) -> None:
        """Rows outside the layer's full range are marked invalid instead of raising."""
        layout = CollectedLayerLayout(layer_offset=0, layer_size=2, kept_rows=torch.tensor([0, 1]))
        rows = torch.tensor([1, 5], dtype=torch.long)
        _, valid = AspectOverrideRuntime(EditManifest(clusters=[]))._map_rows(layout, rows)
        assert valid is not None
        self.assertEqual(valid.tolist(), [True, False])


class RowFilterApplyTest(unittest.TestCase):
    """apply() must remap baked full-layer rows through the recorded collected layout."""

    def _runtime(self) -> AspectOverrideRuntime:
        """Builds a runtime with one two-row 'on' cluster active from t=0."""
        cluster = BakedCluster(
            cluster_id="b",
            edit_id="fx",
            layer_id="dyn",
            nurec_track_id=0,
            aspect="on",
            on_indices=[0, 1],
            centroid_world=[0.0, 0.0, 0.0],
        )
        runtime = AspectOverrideRuntime(EditManifest(clusters=[cluster]))
        runtime.registry.set_many(
            [TrackSchedule(layer_id="dyn", nurec_track_id=0, edit_id="fx", switches=[AspectSwitch(["on"], 0)])]
        )
        return runtime

    def test_only_surviving_rows_are_written(self) -> None:
        """Only cluster rows that survived the filter are visible, at their remapped collected positions."""
        # Collected layer holds originals 1 and 3: baked on-row 1 -> collected row 0;
        # baked on-row 0 and off-row 2 were filtered out and must not be touched.
        layouts = _layouts({"dyn": (0, 2, torch.tensor([1, 3]))})
        params = {"densities": torch.zeros(2)}
        self._runtime().apply(params, 0, layouts=layouts)
        self.assertAlmostEqual(float(params["densities"][0]), _VISIBLE_OPACITY, places=5)
        self.assertEqual(float(params["densities"][1]), 0.0)  # original row 3: not a cluster row

    def test_fully_filtered_cluster_is_a_noop(self) -> None:
        """When every cluster row was filtered out, apply() leaves the collected tensors untouched."""
        layouts = _layouts({"dyn": (0, 1, torch.tensor([3]))})
        params = {"densities": torch.full((1,), 0.7)}
        self._runtime().apply(params, 0, layouts=layouts)
        self.assertAlmostEqual(float(params["densities"][0]), 0.7, places=6)

    def test_recorded_layouts_are_never_cached(self) -> None:
        """apply() uses each call's own layouts, so per-request row filters remap freshly."""
        runtime = self._runtime()
        params_a = {"densities": torch.zeros(2)}
        runtime.apply(params_a, 0, layouts=_layouts({"dyn": (0, 2, torch.tensor([1, 3]))}))
        self.assertAlmostEqual(float(params_a["densities"][0]), _VISIBLE_OPACITY, places=5)
        # Next request keeps originals 0 and 1: BOTH on-rows now present at 0 and 1.
        params_b = {"densities": torch.zeros(2)}
        runtime.apply(params_b, 0, layouts=_layouts({"dyn": (0, 2, torch.tensor([0, 1]))}))
        self.assertAlmostEqual(float(params_b["densities"][0]), _VISIBLE_OPACITY, places=5)
        self.assertAlmostEqual(float(params_b["densities"][1]), _VISIBLE_OPACITY, places=5)


class OnOpacitiesTest(unittest.TestCase):
    """Imported SimReady clusters restore their real per-row alphas, not a flat fill."""

    def _runtime(self, gain: float = 1.0) -> AspectOverrideRuntime:
        """Builds a runtime whose cluster bakes per-row on_opacities, optionally scheduled with a gain."""
        cluster = BakedCluster(
            cluster_id="b",
            edit_id="fx",
            layer_id="dyn",
            nurec_track_id=0,
            aspect="on",
            on_indices=[0, 1],
            on_opacities=[0.25, 0.75],
            centroid_world=[0.0, 0.0, 0.0],
        )
        runtime = AspectOverrideRuntime(EditManifest(clusters=[cluster]))
        runtime.registry.set_many(
            [
                TrackSchedule(
                    layer_id="dyn",
                    nurec_track_id=0,
                    edit_id="fx",
                    switches=[AspectSwitch(["on"], 0)],
                    aspect_gains={"on": (gain, None)} if gain != 1.0 else {},
                )
            ]
        )
        return runtime

    def test_per_row_alphas_restored(self) -> None:
        """Active rows restore their individual baked alphas rather than a flat visible opacity."""
        params = {"densities": torch.zeros(3)}
        self._runtime().apply(params, 0, layouts=_layouts({"dyn": (0, 3, None)}))
        self.assertAlmostEqual(float(params["densities"][0]), 0.25, places=6)
        self.assertAlmostEqual(float(params["densities"][1]), 0.75, places=6)

    def test_gain_scales_per_row_alphas(self) -> None:
        """A gain multiplies each row's baked alpha instead of overwriting it with a uniform value."""
        params = {"densities": torch.zeros(3)}
        self._runtime(gain=0.5).apply(params, 0, layouts=_layouts({"dyn": (0, 3, None)}))
        self.assertAlmostEqual(float(params["densities"][0]), 0.125, places=6)
        self.assertAlmostEqual(float(params["densities"][1]), 0.375, places=6)

    def test_row_filter_trims_the_value_vector(self) -> None:
        """Under a row filter, each surviving row keeps its own baked alpha after remapping."""
        layouts = _layouts({"dyn": (0, 2, torch.tensor([1, 2]))})
        params = {"densities": torch.zeros(2)}
        self._runtime().apply(params, 0, layouts=layouts)
        self.assertAlmostEqual(float(params["densities"][0]), 0.75, places=6)  # survivor keeps ITS alpha
        self.assertEqual(float(params["densities"][1]), 0.0)


class LoadManifestFromUsdzTest(unittest.TestCase):
    """load_manifest_from_usdz extracts the well-known manifest member from a USDZ zip."""

    def test_roundtrips_through_a_zip_member(self) -> None:
        """A manifest stored as the well-known zip member loads back with clusters, assets, and controls."""
        manifest = EditManifest(
            clusters=[
                BakedCluster(
                    cluster_id="b",
                    edit_id="fx",
                    layer_id="dyn",
                    nurec_track_id=1,
                    aspect="on",
                    on_indices=[5],
                    centroid_world=[1.0, 2.0, 3.0],
                )
            ],
            assets=[{"edit_id": "fx", "asset": "scene.usd"}],
            controls=[_appearance()],
        )
        with tempfile.TemporaryDirectory() as tmp:
            usdz = Path(tmp) / "baked.usdz"
            with zipfile.ZipFile(usdz, "w") as zf:
                zf.writestr("other.bin", b"x")
                zf.writestr(EDIT_MANIFEST_FILENAME, json.dumps(manifest.to_dict()))
            loaded = load_manifest_from_usdz(usdz)
        assert loaded is not None
        self.assertEqual(loaded.clusters, manifest.clusters)
        self.assertEqual(loaded.assets, manifest.assets)
        self.assertEqual(loaded.controls, manifest.controls)

    def test_plain_artifact_returns_none(self) -> None:
        """A USDZ without the manifest member yields None instead of raising."""
        with tempfile.TemporaryDirectory() as tmp:
            usdz = Path(tmp) / "plain.usdz"
            with zipfile.ZipFile(usdz, "w") as zf:
                zf.writestr("other.bin", b"x")
            self.assertIsNone(load_manifest_from_usdz(usdz))


class TrackScheduleEdgeTest(unittest.TestCase):
    """active_state_at edge cases: early queries, empty schedules, and explicit empty states."""

    def test_holds_first_switch_before_its_time(self) -> None:
        """Queries earlier than the first switch time return that first switch's aspects."""
        fs = TrackSchedule("dyn", 0, "fx", [AspectSwitch(["red"], 10), AspectSwitch(["green"], 20)])
        self.assertEqual(fs.active_state_at(5), {"red"})
        self.assertEqual(fs.active_state_at(20), {"green"})

    def test_no_switches_means_no_state(self) -> None:
        """A schedule with no switches reports None rather than an empty aspect set."""
        self.assertIsNone(TrackSchedule("dyn", 0, "fx", []).active_state_at(0))

    def test_explicit_empty_state_is_kept(self) -> None:
        """An explicit empty switch surfaces as an empty aspect set, distinct from no schedule at all."""
        registry = AspectScheduleRegistry()
        registry.set_many([TrackSchedule("dyn", 0, "fx", [AspectSwitch([], 0)])])
        self.assertEqual(_active_aspects_at(registry.tracks, 0), {("dyn", 0): set()})


class CollectKeepMaskTest(unittest.TestCase):
    """apply() must never re-show rows the collector's own keep mask hid (track outside
    its keyframe window / deactivated). The mask rides the recorded layout's 4th slot."""

    def _runtime(self, on_opacities: list[float] | None = None) -> AspectOverrideRuntime:
        """A runtime with one two-row 'on' cluster scheduled active from t=0."""
        cluster = BakedCluster(
            cluster_id="b",
            edit_id="fx",
            layer_id="dyn",
            nurec_track_id=0,
            aspect="on",
            on_indices=[0, 1],
            centroid_world=[0.0, 0.0, 0.0],
            on_opacities=on_opacities or [],
        )
        runtime = AspectOverrideRuntime(EditManifest(clusters=[cluster]))
        runtime.registry.set_many(
            [TrackSchedule(layer_id="dyn", nurec_track_id=0, edit_id="fx", switches=[AspectSwitch(["on"], 0)])]
        )
        return runtime

    def test_masked_rows_stay_hidden(self) -> None:
        """A cluster row the collector masked is never visible; unmasked cluster rows still are."""
        keep = torch.tensor([True, False, True, False])
        layouts = _layouts({"dyn": (0, 4, None, keep)})
        params = {"densities": torch.zeros(4)}
        self._runtime().apply(params, 0, layouts=layouts)
        self.assertAlmostEqual(float(params["densities"][0]), _VISIBLE_OPACITY, places=5)
        self.assertEqual(float(params["densities"][1]), 0.0)

    def test_fully_masked_cluster_is_a_noop(self) -> None:
        """A track fully outside its lifetime (all cluster rows masked) stays hidden."""
        keep = torch.tensor([False, False, True, True])
        layouts = _layouts({"dyn": (0, 4, None, keep)})
        params = {"densities": torch.zeros(4)}
        self._runtime().apply(params, 0, layouts=layouts)
        self.assertEqual(float(params["densities"].abs().sum()), 0.0)

    def test_mask_composes_with_row_filter(self) -> None:
        """The collector mask applies on top of the edited-actors row filter."""
        # Originals 1, 3 survived the filter (collected rows 0, 1); the collector
        # additionally masked collected row 0 — the cluster's only surviving row.
        keep = torch.tensor([False, True])
        layouts = _layouts({"dyn": (0, 2, torch.tensor([1, 3]), keep)})
        params = {"densities": torch.zeros(2)}
        self._runtime().apply(params, 0, layouts=layouts)
        self.assertEqual(float(params["densities"].abs().sum()), 0.0)

    def test_per_row_alphas_subset_by_mask(self) -> None:
        """The baked per-row opacity vector is trimmed to the mask survivors."""
        keep = torch.tensor([False, True, True, True])
        layouts = _layouts({"dyn": (0, 4, None, keep)})
        params = {"densities": torch.zeros(4)}
        self._runtime(on_opacities=[0.4, 0.8]).apply(params, 0, layouts=layouts)
        self.assertEqual(float(params["densities"][0]), 0.0)
        self.assertAlmostEqual(float(params["densities"][1]), 0.8, places=6)


class ScheduleValidationTest(unittest.TestCase):
    """Schedule JSON fails loud: bad time_base, missing keys, and wire-hostile gains."""

    def test_non_absolute_time_base_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_base"):
            EditSchedule.from_dict({"time_base": "clip_relative", "tracks": []})

    def test_default_time_base_accepted(self) -> None:
        self.assertEqual(EditSchedule.from_dict({"tracks": []}).time_base, "absolute_us")

    def test_missing_track_keys_contextual(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer_id"):
            TrackSchedule.from_dict({"switches": []})

    def test_missing_t_us_contextual(self) -> None:
        with self.assertRaisesRegex(ValueError, "t_us"):
            AspectSwitch.from_dict({"aspects": ["on"]})

    def test_negative_gain_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "gain"):
            TrackSchedule.from_dict({"layer_id": "dyn", "nre_track_id": 0, "aspect_gains": {"on": {"gain": -0.5}}})

    def test_nan_gain_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "gain"):
            validate_aspect_gain("on", float("nan"), None, "ctx")

    def test_overrange_cold_rgb_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cold_rgb"):
            validate_aspect_gain("on", 0.5, [2.0, 0.0, 0.0], "ctx")

    def test_valid_gain_passes_through(self) -> None:
        self.assertEqual(validate_aspect_gain("on", 0.5, [0.1, 0.2, 0.3], "ctx"), (0.5, [0.1, 0.2, 0.3]))

    def test_track_id_key_accepts_legacy_spelling(self) -> None:
        """Manifests and schedules written before the rename use ``nre_track_id``."""
        cases = {
            "old": {"nre_track_id": 4},
            "new": {"nurec_track_id": 4},
            "both_equal": {"nre_track_id": 4, "nurec_track_id": 4},
        }
        for name, keys in cases.items():
            with self.subTest(name):
                cluster = BakedCluster.from_dict(
                    {"cluster_id": "c", "edit_id": "e", "layer_id": "dyn", "aspect": "on", **keys}
                )
                self.assertEqual(cluster.nurec_track_id, 4)
                self.assertEqual(cluster.to_dict()["nurec_track_id"], 4)
                self.assertNotIn("nre_track_id", cluster.to_dict())
                schedule = TrackSchedule.from_dict({"layer_id": "dyn", "switches": [], **keys})
                self.assertEqual(schedule.nurec_track_id, 4)
                self.assertEqual(schedule.to_dict()["nurec_track_id"], 4)
                self.assertNotIn("nre_track_id", schedule.to_dict())
                anim = ControlAnimation.from_dict(
                    {"edit_id": "e", "layer_id": "dyn", "control": "c", "default_state": "off", **keys}
                )
                self.assertEqual(anim.nurec_track_id, 4)
                self.assertEqual(anim.to_dict()["nurec_track_id"], 4)
                self.assertNotIn("nre_track_id", anim.to_dict())

    def test_track_id_key_conflict_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting"):
            TrackSchedule.from_dict({"layer_id": "dyn", "nre_track_id": 4, "nurec_track_id": 5, "switches": []})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            ControlAnimation.from_dict(
                {
                    "edit_id": "e",
                    "layer_id": "dyn",
                    "control": "c",
                    "default_state": "off",
                    "nre_track_id": 4,
                    "nurec_track_id": 5,
                }
            )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            BakedCluster.from_dict(
                {
                    "cluster_id": "c",
                    "edit_id": "e",
                    "layer_id": "dyn",
                    "aspect": "on",
                    "nre_track_id": 4,
                    "nurec_track_id": 5,
                }
            )

    def test_duplicate_schedule_tracks_rejected(self) -> None:
        """The registry is keyed by (layer, track); a duplicate would silently last-win."""
        entry = {"layer_id": "dyn", "nre_track_id": 0, "switches": []}
        with self.assertRaisesRegex(ValueError, "more than once"):
            EditSchedule.from_dict({"tracks": [entry, dict(entry)]})


class LayerTrackDigestTest(unittest.TestCase):
    """The digest must cover exactly the tensor's own elements, never shared storage."""

    def test_view_of_larger_storage_hashes_only_its_elements(self) -> None:
        big = torch.arange(10, dtype=torch.int64)
        view = big[2:6]  # contiguous CPU int64 view: .to()/.contiguous() are no-ops
        owned = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
        self.assertEqual(layer_track_digest(view), layer_track_digest(owned))


class LayerRowCountsTest(unittest.TestCase):
    """layer_row_counts binds a manifest to the exact bake it addresses."""

    def test_roundtrip(self) -> None:
        again = EditManifest.from_dict(EditManifest(clusters=[], layer_row_counts={"dyn": 42}).to_dict())
        self.assertEqual(again.layer_row_counts, {"dyn": 42})

    def test_validate_mismatch_raises(self) -> None:
        manifest = EditManifest(clusters=[], layer_row_counts={"dyn": 42})
        with self.assertRaisesRegex(ValueError, "42"):
            manifest.validate_against_layers({"dyn": _ValidationNode([0] * 43)})

    def test_validate_match_and_legacy_pass(self) -> None:
        EditManifest(clusters=[], layer_row_counts={"dyn": 42}).validate_against_layers(
            {"dyn": _ValidationNode([0] * 42)}
        )  # exact match
        EditManifest(clusters=[]).validate_against_layers({"dyn": _ValidationNode([0])})  # legacy -> no check


class _ValidationNode:
    """Gaussian node exposing the two things the install gate inspects."""

    def __init__(self, cuboid_ids: list[int]) -> None:
        """Stores the per-row track ids the manifest is checked against."""
        self.gaussian_cuboid_ids = torch.tensor(cuboid_ids, dtype=torch.int32)

    def get_num_gaussians(self) -> int:
        """Row count of the layer."""
        return int(self.gaussian_cuboid_ids.shape[0])


class ValidateAgainstLayersTest(unittest.TestCase):
    """The install gate: a manifest may only address the checkpoint it was baked from.

    Every failure here writes baked opacities into unrelated gaussians rather than
    raising, so each rejection needs its own case.
    """

    def _manifest(self, on_indices=(1, 2), track=1, layer="dyn", **kw) -> EditManifest:
        """A one-cluster manifest addressing `on_indices` of `layer` as track `track`."""
        return EditManifest(
            clusters=[
                BakedCluster(
                    cluster_id="c",
                    edit_id="e",
                    layer_id=layer,
                    nurec_track_id=track,
                    aspect="a",
                    on_indices=list(on_indices),
                    centroid_world=[0.0, 0.0, 0.0],
                )
            ],
            **kw,
        )

    def test_matching_manifest_passes(self) -> None:
        node = _ValidationNode([0, 1, 1, 2])
        m = self._manifest(
            layer_row_counts={"dyn": 4}, layer_track_digest={"dyn": layer_track_digest(node.gaussian_cuboid_ids)}
        )
        m.validate_against_layers({"dyn": node})  # no raise

    def test_missing_layer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "which the loaded checkpoint does not have"):
            self._manifest().validate_against_layers({"other": _ValidationNode([0, 1])})

    def test_row_count_mismatch_is_rejected(self) -> None:
        m = self._manifest(layer_row_counts={"dyn": 99})
        with self.assertRaisesRegex(ValueError, "baked against layer"):
            m.validate_against_layers({"dyn": _ValidationNode([0, 1, 1, 2])})

    def test_reordered_layer_of_unchanged_size_is_rejected_by_the_digest(self) -> None:
        """The row count alone cannot see a permutation; the digest can."""
        baked = _ValidationNode([0, 1, 1, 2])
        reordered = _ValidationNode([1, 1, 0, 2])
        m = self._manifest(
            layer_row_counts={"dyn": 4}, layer_track_digest={"dyn": layer_track_digest(baked.gaussian_cuboid_ids)}
        )
        self.assertEqual(reordered.get_num_gaussians(), baked.get_num_gaussians())
        with self.assertRaisesRegex(ValueError, "digest"):
            m.validate_against_layers({"dyn": reordered})

    def test_rows_belonging_to_another_track_are_rejected(self) -> None:
        """Catches a stale manifest even when size and digest were never recorded."""
        with self.assertRaisesRegex(ValueError, "belong to other tracks"):
            self._manifest(on_indices=(0, 1)).validate_against_layers({"dyn": _ValidationNode([0, 1, 1, 2])})

    def test_rows_outside_the_layer_are_rejected(self) -> None:
        """Both over-range and NEGATIVE rows are refused here. This is the only wall:
        the runtime indexes the baked rows as-is, and tensor indexing would silently
        WRAP a negative row onto a valid-looking entry."""
        with self.assertRaisesRegex(ValueError, "outside layer"):
            self._manifest(on_indices=(1, 99)).validate_against_layers({"dyn": _ValidationNode([0, 1, 1, 2])})
        with self.assertRaisesRegex(ValueError, "outside layer"):
            self._manifest(on_indices=(-1, 1)).validate_against_layers({"dyn": _ValidationNode([0, 1, 1, 2])})

    def test_layer_without_cuboid_ids_is_rejected(self) -> None:
        """A cluster target without per-row track ids can never be validated — refuse it
        rather than silently skipping every row/track check for that layer."""

        class _NoIdsNode:
            def get_num_gaussians(self) -> int:
                return 4

        with self.assertRaisesRegex(ValueError, "no per-row gaussian_cuboid_ids"):
            self._manifest().validate_against_layers({"dyn": _NoIdsNode()})

    def test_duplicate_cluster_id_is_rejected(self) -> None:
        m = self._manifest()
        m.clusters.append(m.clusters[0])
        with self.assertRaisesRegex(ValueError, "duplicate cluster_id"):
            m.validate_against_layers({"dyn": _ValidationNode([0, 1, 1, 2])})

    def test_mismatched_on_opacities_length_is_rejected(self) -> None:
        m = self._manifest()
        m.clusters[0].on_opacities = [0.5]  # 1 opacity for 2 rows
        with self.assertRaisesRegex(ValueError, "on_opacities"):
            m.validate_against_layers({"dyn": _ValidationNode([0, 1, 1, 2])})


class ValidateScheduleTest(unittest.TestCase):
    """A schedule may only drive tracks and aspects the bake actually produced."""

    def _manifest(self) -> EditManifest:
        return EditManifest(
            clusters=[
                BakedCluster(
                    cluster_id="c",
                    edit_id="e",
                    layer_id="dyn",
                    nurec_track_id=2,
                    aspect="panel:on",
                    on_indices=[0],
                    centroid_world=[0.0, 0.0, 0.0],
                )
            ]
        )

    def test_matching_schedule_passes(self) -> None:
        s = EditSchedule(
            time_base="absolute_us", tracks=[TrackSchedule("dyn", 2, "e", [AspectSwitch(["panel:on"], 0)])]
        )
        self._manifest().validate_schedule(s)  # no raise

    def test_unknown_track_is_rejected(self) -> None:
        s = EditSchedule(
            time_base="absolute_us", tracks=[TrackSchedule("dyn", 7, "e", [AspectSwitch(["panel:on"], 0)])]
        )
        with self.assertRaisesRegex(ValueError, "no baked clusters"):
            self._manifest().validate_schedule(s)

    def test_unknown_aspect_is_rejected(self) -> None:
        s = EditSchedule(
            time_base="absolute_us", tracks=[TrackSchedule("dyn", 2, "e", [AspectSwitch(["panel:typo"], 0)])]
        )
        with self.assertRaisesRegex(ValueError, "no baked cluster"):
            self._manifest().validate_schedule(s)


class ZeroRowLayerTest(unittest.TestCase):
    """A layer the row filter emptied has no collected row to write into."""

    def test_apply_is_a_noop_when_the_filter_kept_nothing(self) -> None:
        manifest = EditManifest(
            clusters=[
                BakedCluster(
                    cluster_id="c",
                    edit_id="e",
                    layer_id="dyn",
                    nurec_track_id=0,
                    aspect="a",
                    on_indices=[0, 1],
                    centroid_world=[0.0, 0.0, 0.0],
                )
            ]
        )
        registry = AspectScheduleRegistry()
        registry.set_many([TrackSchedule("dyn", 0, "e", [AspectSwitch(["a"], 0)])])
        layouts = _layouts({"dyn": (0, 0, torch.tensor([], dtype=torch.long), torch.zeros(0, dtype=torch.bool))})
        params = {"densities": torch.zeros(0)}
        AspectOverrideRuntime(manifest=manifest, registry=registry).apply(params, 0, layouts=layouts)
        self.assertEqual(params["densities"].numel(), 0)


if __name__ == "__main__":
    unittest.main()
