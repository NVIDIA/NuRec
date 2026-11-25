# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``asset_edits.yaml`` schema: loading, defaulting and validation.

Standalone by construction: documents are built as in-memory dicts (plus a
tempfile YAML where PyYAML's parsing itself is under test); no scene data, no
torch, no USD.
"""

import tempfile
import unittest

from pathlib import Path
from typing import Any

from nurec.utils.edit.asset_edits_schema import (
    DEFAULT_LAYER_ID,
    AssetEditsDoc,
    AssetEditsError,
    AssetPlacement,
    Binding,
    CloneMotion,
    ControlSpec,
    EditSpec,
    _as_vec3,
    _norm_token,
    binding_key,
    is_ply_asset,
    load_asset_edits,
)


def _edit(**over: Any) -> dict[str, Any]:
    """A minimal valid edit dict; keyword args override fields."""
    d: dict[str, Any] = {
        "id": "truck_1",
        "op": "replace",
        "bind": {"type": "nre_track", "nre_track_id": 3},
        "asset": "usd/configurable_asset/scene.usd",
        "controls": {"panelState": {"default": "off", "controllable": True}},
    }
    d.update(over)
    return d


def _clone(**over: Any) -> dict[str, Any]:
    """A minimal valid clone-of-captured-track edit dict; keyword args override fields."""
    d: dict[str, Any] = {
        "id": "lead_clone",
        "op": "clone",
        "source": {"nurec_track_id": 12},
        "motion": {"inherit": True, "offset": [0.0, -2.4, 0.0]},
        "track_id": "9000221",
    }
    d.update(over)
    return d


def _exclude(**over: Any) -> dict[str, Any]:
    """A minimal valid exclude edit dict; keyword args override fields."""
    d: dict[str, Any] = {"id": "drop_1", "op": "exclude", "bind": {"type": "nurec_track", "nurec_track_id": 5}}
    d.update(over)
    return d


def _doc(**over: Any) -> dict[str, Any]:
    """A minimal valid document dict; keyword args override top-level keys."""
    d: dict[str, Any] = {"edits": [_edit()]}
    d.update(over)
    return d


class NormTokenTest(unittest.TestCase):
    """Token normalization: YAML 1.1 booleans map back to off/on, and file loading enforces non-empty docs."""

    def test_yaml_bools_map_back_to_off_on(self) -> None:
        """Python False/True (YAML 1.1 bare off/on) normalize to the tokens "off" and "on"."""
        self.assertEqual(_norm_token(False), "off")
        self.assertEqual(_norm_token(True), "on")

    def test_strings_are_stripped_and_lowercased(self) -> None:
        """Surrounding whitespace is stripped and the token lowercased."""
        self.assertEqual(_norm_token(" Sequence "), "sequence")

    def test_unquoted_off_default_roundtrips_through_yaml(self) -> None:
        """A bare `default: off` (YAML 1.1 bool) maps back to the token "off" on load."""
        # YAML 1.1 parses bare off/on as booleans; the loader must map them back
        # (an unquoted `default: off` arrives as Python False).
        text = """
        edits:
          - id: truck_1
            op: replace
            bind: {nurec_track_id: 3}
            asset: scene.usd
            controls:
              panelState: {default: off, controllable: true}
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "asset_edits.yaml"
            path.write_text(text)
            doc = load_asset_edits(path)
        self.assertEqual(doc.edits[0].controls[0].default, "off")

    def test_empty_yaml_file_rejected(self) -> None:
        """Loading an empty YAML file fails loud rather than yielding an empty document."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("")
            with self.assertRaisesRegex(AssetEditsError, "empty"):
                load_asset_edits(path)


class AsVec3Test(unittest.TestCase):
    """_as_vec3 fails loud on malformed 3-vectors."""

    def test_wrong_length_rejected(self) -> None:
        """Anything but exactly three elements is rejected."""
        with self.assertRaisesRegex(AssetEditsError, "3-element"):
            _as_vec3([1.0, 2.0], "ctx")

    def test_non_numeric_rejected(self) -> None:
        """Non-numeric components are rejected rather than coerced, naming the component."""
        with self.assertRaisesRegex(AssetEditsError, r"ctx\[1\]: expected a number"):
            _as_vec3([1.0, "a", 3.0], "ctx")


class BindingTest(unittest.TestCase):
    """Binding parsing: nurec_track and world binds, each with strict per-type fields."""

    def test_unknown_bind_type_rejected(self) -> None:
        """Bind types other than nurec_track/world are rejected."""
        with self.assertRaisesRegex(AssetEditsError, "unsupported bind type"):
            Binding.from_dict({"type": "actor_name", "nre_track_id": 1}, "ctx")

    def test_requires_track_id(self) -> None:
        """A bind without nurec_track_id is rejected."""
        with self.assertRaisesRegex(AssetEditsError, "nurec_track_id"):
            Binding.from_dict({"type": "nre_track"}, "ctx")

    def test_type_defaults_to_nurec_track(self) -> None:
        """Omitting the bind type defaults it to nurec_track."""
        self.assertEqual(Binding.from_dict({"nurec_track_id": 3}, "ctx").type, "nurec_track")

    def test_legacy_spellings_parse_to_current_ones(self) -> None:
        """Pre-rename documents use ``nre_track`` / ``nre_track_id``; the parsed binding never does."""
        cases: dict[str, dict[str, Any]] = {
            "old": {"type": "nre_track", "nre_track_id": 3},
            "new": {"type": "nurec_track", "nurec_track_id": 3},
            "mixed": {"type": "nre_track", "nurec_track_id": 3},
            "both_equal": {"nre_track_id": 3, "nurec_track_id": 3},
        }
        for name, d in cases.items():
            with self.subTest(name):
                b = Binding.from_dict(d, "ctx")
                self.assertEqual((b.type, b.nurec_track_id), ("nurec_track", 3))

    def test_unknown_field_error_names_the_legacy_spelling_the_author_wrote(self) -> None:
        with self.assertRaisesRegex(AssetEditsError, r"unknown field\(s\) \['nre_track_id'\]"):
            Binding.from_dict({"type": "world", "position": [0, 0, 0], "nre_track_id": 3}, "ctx")

    def test_conflicting_track_id_spellings_rejected(self) -> None:
        with self.assertRaisesRegex(AssetEditsError, "conflicting"):
            Binding.from_dict({"nre_track_id": 3, "nurec_track_id": 4}, "ctx")


class WorldBindingTest(unittest.TestCase):
    """World binds: a new static actor at a fixed world pose instead of a source track."""

    def test_parses_happy_path(self) -> None:
        """A fully-specified world bind parses with every field intact."""
        bind = Binding.from_dict(
            {
                "type": "world",
                "position": [10.0, -3.5, 0.25],
                "yaw_deg": 90.0,
                "anchor": "center",
                "dims": [1.0, 2.0, 0.5],
                "label_class": "obstacle",
                "track_id": "prop:custom",
            },
            "ctx",
        )
        self.assertTrue(bind.is_world)
        self.assertEqual(bind.position, [10.0, -3.5, 0.25])
        self.assertEqual((bind.yaw_deg, bind.anchor), (90.0, "center"))
        self.assertEqual((bind.dims, bind.label_class, bind.track_id), ([1.0, 2.0, 0.5], "obstacle", "prop:custom"))

    def test_minimal_world_bind_defaults(self) -> None:
        """Only position is required; anchor defaults to base and dims/track_id stay unset."""
        bind = Binding.from_dict({"type": "world", "position": [0.0, 0.0, 0.0]}, "ctx")
        self.assertEqual((bind.yaw_deg, bind.anchor), (0.0, "base"))
        self.assertIsNone(bind.dims)
        self.assertIsNone(bind.track_id)  # EditSpec fills in "prop:<edit_id>"

    def test_requires_position(self) -> None:
        """A world bind without a position is rejected."""
        with self.assertRaisesRegex(AssetEditsError, "requires 'position'"):
            Binding.from_dict({"type": "world"}, "ctx")

    def test_anchor_validated(self) -> None:
        """Anchors other than base/center are rejected."""
        with self.assertRaisesRegex(AssetEditsError, "anchor"):
            Binding.from_dict({"type": "world", "position": [0, 0, 0], "anchor": "top"}, "ctx")

    def test_dims_must_be_positive(self) -> None:
        """Zero or negative dims components are rejected."""
        with self.assertRaisesRegex(AssetEditsError, "must be > 0"):
            Binding.from_dict({"type": "world", "position": [0, 0, 0], "dims": [1.0, 0.0, 1.0]}, "ctx")

    def test_cross_type_fields_rejected(self) -> None:
        """nurec_track fields on world binds fail loud, and vice versa."""
        with self.assertRaisesRegex(AssetEditsError, "unknown field"):
            Binding.from_dict({"type": "world", "position": [0, 0, 0], "nre_track_id": 3}, "ctx")
        with self.assertRaisesRegex(AssetEditsError, "unknown field"):
            Binding.from_dict({"type": "nre_track", "nre_track_id": 3, "position": [0, 0, 0]}, "ctx")

    def test_empty_track_id_rejected(self) -> None:
        """An explicitly empty track_id is rejected rather than silently defaulted."""
        with self.assertRaisesRegex(AssetEditsError, "non-empty"):
            Binding.from_dict({"type": "world", "position": [0, 0, 0], "track_id": "  "}, "ctx")


class WorldEditInterplayTest(unittest.TestCase):
    """EditSpec-level rules that only apply when the bind is a world bind."""

    @staticmethod
    def _world_edit(**over: Any) -> dict[str, Any]:
        """A world-bind add edit; keyword arguments override its fields."""
        d: dict[str, Any] = {
            "id": "barrier_1",
            "op": "add",
            "bind": {"type": "world", "position": [150.0, -3.5, 0.0], "yaw_deg": 90.0},
            "asset": "props/barrier/Body.usd",
        }
        d.update(over)
        return d

    def test_track_id_defaults_to_prop_edit_id(self) -> None:
        """A world bind without an explicit track_id gets "prop:<edit_id>" on parse."""
        fx = EditSpec.from_dict(self._world_edit(), "edits[0]")
        assert fx.bind is not None
        self.assertEqual(fx.bind.track_id, "prop:barrier_1")

    def test_replace_on_world_bind_rejected(self) -> None:
        """A replace at a world pose is meaningless (no captured actor there) and fails loud."""
        with self.assertRaisesRegex(AssetEditsError, "replace edit binds an existing track"):
            EditSpec.from_dict(self._world_edit(op="replace"), "edits[0]")

    def test_scale_defaults_to_fixed_one(self) -> None:
        """With no explicit scale, world binds keep real-world size (fixed 1.0, not fit_length)."""
        fx = EditSpec.from_dict(self._world_edit(), "edits[0]")
        assert fx.placement is not None
        self.assertEqual((fx.placement.scale_mode, fx.placement.scale), ("fixed", 1.0))

    def test_explicit_numeric_scale_kept(self) -> None:
        """An explicit numeric scale on a world bind is honoured."""
        fx = EditSpec.from_dict(self._world_edit(placement={"scale": 2.0}), "edits[0]")
        assert fx.placement is not None
        self.assertEqual((fx.placement.scale_mode, fx.placement.scale), ("fixed", 2.0))

    def test_fit_length_requires_explicit_dims(self) -> None:
        """fit_length is circular without a source actor unless the bind gives explicit dims."""
        with self.assertRaisesRegex(AssetEditsError, "requires explicit bind.dims"):
            EditSpec.from_dict(self._world_edit(placement={"scale": "fit_length"}), "edits[0]")
        fx = EditSpec.from_dict(
            self._world_edit(
                bind={"type": "world", "position": [0, 0, 0], "dims": [2.0, 1.0, 1.0]},
                placement={"scale": "fit_length"},
            ),
            "edits[0]",
        )
        assert fx.placement is not None
        self.assertEqual(fx.placement.scale_mode, "fit_length")

    def test_nurec_track_edits_keep_fit_length_default(self) -> None:
        """The world-bind scale default does not leak into replace edits."""
        fx = EditSpec.from_dict(_edit(), "edits[0]")
        assert fx.placement is not None
        self.assertEqual(fx.placement.scale_mode, "fit_length")

    def test_attachment_keeps_real_size_by_default(self) -> None:
        """An add on a track (attachment) is not fitted to the actor's cuboid unless asked."""
        fx = EditSpec.from_dict(_edit(op="add", controls={}), "edits[0]")
        self.assertEqual((fx.placement.scale_mode, fx.placement.scale), ("fixed", 1.0))
        fitted = EditSpec.from_dict(_edit(op="add", controls={}, placement={"scale": "fit_length"}), "edits[0]")
        self.assertEqual(fitted.placement.scale_mode, "fit_length")
        again = EditSpec.from_dict(fx.to_dict(), "edits[0]")
        self.assertEqual(again, fx)

    def test_track_id_with_at_sign_rejected(self) -> None:
        """NuRec strips '@<source>' from track ids at load, so such an id would not survive."""
        with self.assertRaisesRegex(AssetEditsError, "contains '@'.*'221'"):
            EditSpec.from_dict(
                self._world_edit(bind={"type": "world", "position": [1.0, 2.0, 0.0], "track_id": "221@clone"}),
                "edits[0]",
            )
        with self.assertRaisesRegex(AssetEditsError, "defaults to 'prop:<id>'"):
            EditSpec.from_dict(self._world_edit(id="barrier@1"), "edits[0]")
        with self.assertRaisesRegex(AssetEditsError, "contains '@'"):
            EditSpec.from_dict(
                {
                    "id": "c",
                    "op": "clone",
                    "source": {"nurec_track_id": 1},
                    "motion": {"inherit": True},
                    "track_id": "221@clone",
                },
                "edits[0]",
            )


class AssetPlacementTest(unittest.TestCase):
    """Placement parsing: align/scale validation, defaults, and yaw folding into the Euler Z."""

    def test_align_validated(self) -> None:
        """align accepts only 'bbox' or 'origin'."""
        with self.assertRaisesRegex(AssetEditsError, "'bbox' or 'origin'"):
            AssetPlacement.from_dict({"align": "centroid"}, "ctx")

    def test_scale_fit_length_default(self) -> None:
        """An empty placement defaults to fit_length scale mode."""
        placement = AssetPlacement.from_dict({}, "ctx")
        self.assertEqual(placement.scale_mode, "fit_length")

    def test_numeric_scale_is_fixed_mode(self) -> None:
        """A numeric scale switches to fixed mode and stores the value."""
        placement = AssetPlacement.from_dict({"scale": 2.5}, "ctx")
        self.assertEqual((placement.scale_mode, placement.scale), ("fixed", 2.5))

    def test_bad_scale_string_rejected(self) -> None:
        """A scale string other than 'fit_length' is rejected."""
        with self.assertRaisesRegex(AssetEditsError, "fit_length"):
            AssetPlacement.from_dict({"scale": "huge"}, "ctx")

    def test_fixed_scale_must_be_positive(self) -> None:
        """Zero or negative fixed scales are rejected."""
        with self.assertRaisesRegex(AssetEditsError, "must be > 0"):
            AssetPlacement.from_dict({"scale": -1.0}, "ctx")

    def test_yaw_added_to_euler_z(self) -> None:
        """yaw_deg adds onto the Z component of rotation_deg in euler_xyz_deg()."""
        placement = AssetPlacement.from_dict({"rotation_deg": [0.0, 0.0, 10.0], "yaw_deg": 80.0}, "ctx")
        self.assertEqual(placement.euler_xyz_deg(), (0.0, 0.0, 90.0))


class ControlSpecTest(unittest.TestCase):
    """Control parsing: strict fields, real booleans only, and name/semantics normalization."""

    def test_unknown_field_rejected(self) -> None:
        """Unrecognised control fields fail loud instead of being ignored."""
        with self.assertRaisesRegex(AssetEditsError, "unknown control field"):
            ControlSpec.from_dict("panelState", {"speed": 1}, "ctx")

    def test_semantics_must_be_string_list(self) -> None:
        """A bare string for semantics is rejected; a list of strings is required."""
        with self.assertRaisesRegex(AssetEditsError, "semantics"):
            ControlSpec.from_dict("panelState", {"semantics": "PANEL"}, "ctx")

    def test_quoted_string_flag_rejected(self) -> None:
        """A quoted "false" arrives as a truthy string; it is rejected rather than coerced to True."""
        # A quoted "false" in YAML arrives as a truthy string; the schema must not
        # silently coerce it (bool("false") is True).
        with self.assertRaisesRegex(AssetEditsError, "expected a boolean"):
            ControlSpec.from_dict("panelState", {"controllable": "false"}, "ctx")

    def test_name_normalized_and_semantics_uppercased(self) -> None:
        """Control name and default state are lowercased while semantics labels are uppercased."""
        spec = ControlSpec.from_dict("PanelState", {"default": "Off", "semantics": ["panel_visibility"]}, "ctx")
        self.assertEqual(spec.name, "panelstate")
        self.assertEqual(spec.default, "off")
        self.assertEqual(spec.semantics, ["PANEL_VISIBILITY"])


class EditSpecTest(unittest.TestCase):
    """Edit parsing: required keys, strict unknown-field rejection, and control handling."""

    def test_requires_id_bind_and_asset(self) -> None:
        """Missing id, op, bind or asset each fails with an error naming the absent key."""
        with self.assertRaisesRegex(AssetEditsError, "requires an 'id'"):
            EditSpec.from_dict({"op": "add", "bind": {"nurec_track_id": 1}, "asset": "a.usd"}, "edits[0]")
        no_op = _edit()
        del no_op["op"]
        with self.assertRaisesRegex(AssetEditsError, "requires an 'op'"):
            EditSpec.from_dict(no_op, "edits[0]")
        no_bind = _edit()
        del no_bind["bind"]
        with self.assertRaisesRegex(AssetEditsError, "requires a 'bind'"):
            EditSpec.from_dict(no_bind, "edits[0]")
        with self.assertRaisesRegex(AssetEditsError, "requires an 'asset'"):
            EditSpec.from_dict(_edit(asset=None), "edits[0]")

    def test_unknown_field_rejected(self) -> None:
        """Unknown edit fields, including those from retired schema revisions, fail loud instead of being ignored."""
        # Includes fields of retired schema revisions: they must fail loud, not be ignored.
        for field_name, value in [
            ("color", "red"),
            ("kind", "vehicle"),
            ("bulbs", []),
            ("schedule", []),
            ("simulate", {"wheels": True}),
        ]:
            with self.assertRaisesRegex(AssetEditsError, "unknown edit field"):
                EditSpec.from_dict(_edit(**{field_name: value}), "edits[0]")

    def test_retired_replace_original_key_rejected(self) -> None:
        """The POC's replace_original flag is gone: the op says whether the actor is hidden."""
        with self.assertRaisesRegex(AssetEditsError, "unknown edit field.*replace_original"):
            EditSpec.from_dict(_edit(replace_original=True), "edits[0]")

    def test_unknown_op_rejected(self) -> None:
        """An op outside add/replace/clone/exclude fails, naming the supported set."""
        with self.assertRaisesRegex(AssetEditsError, "unsupported op 'swap'.*add.*replace.*clone.*exclude"):
            EditSpec.from_dict(_edit(op="swap"), "edits[0]")

    def test_op_predicates(self) -> None:
        """replace hides the captured actor and bakes an asset; add on a track is an attachment."""
        replace = EditSpec.from_dict(_edit(), "edits[0]")
        self.assertEqual(
            (replace.op, replace.hides_original, replace.has_asset, replace.is_world), ("replace", True, True, False)
        )
        attach = EditSpec.from_dict(_edit(op="add"), "edits[0]")
        self.assertEqual((attach.hides_original, attach.has_asset, attach.binds_synthetic_track), (False, True, False))
        self.assertEqual(attach.nurec_track_id, 3)
        self.assertEqual(binding_key(attach), (DEFAULT_LAYER_ID, 3))

    def test_controls_must_be_a_mapping(self) -> None:
        """A list under controls is rejected; a name-to-spec mapping is required."""
        with self.assertRaisesRegex(AssetEditsError, "expected a mapping"):
            EditSpec.from_dict(_edit(controls=["panelState"]), "edits[0]")

    def test_duplicate_control_names_after_normalization(self) -> None:
        """Control names that collide once lowercased count as duplicates."""
        with self.assertRaisesRegex(AssetEditsError, "duplicate control name"):
            EditSpec.from_dict(_edit(controls={"PanelState": {}, "panelstate": {}}), "edits[0]")

    def test_parses_happy_path(self) -> None:
        """A fully-specified edit parses with bind, flags and normalized controls intact."""
        fx = EditSpec.from_dict(_edit(placement={"align": "bbox", "scale": "fit_length"}), "edits[0]")
        self.assertEqual(fx.id, "truck_1")
        assert fx.bind is not None
        self.assertEqual(fx.bind.nurec_track_id, 3)
        self.assertTrue(fx.hides_original)
        self.assertEqual(fx.controls[0].name, "panelstate")
        self.assertTrue(fx.controls[0].controllable)

    def test_controls_are_optional(self) -> None:
        """An empty controls mapping parses to an edit with no controls."""
        fx = EditSpec.from_dict(_edit(controls={}), "edits[0]")
        self.assertEqual(fx.controls, [])


class PlyAssetEditTest(unittest.TestCase):
    """A ``.ply`` asset is a static prop: accepted bare, refused with controls or a prim_path."""

    def test_suffix_detection_is_case_insensitive(self) -> None:
        """Only a trailing .ply (any case) marks a splat file; a USD stays a USD."""
        self.assertTrue(is_ply_asset("props/cone.ply"))
        self.assertTrue(is_ply_asset("props/CONE.PLY"))
        self.assertFalse(is_ply_asset("props/cone.usd"))
        self.assertFalse(is_ply_asset("props/cone.ply.usd"))

    def test_bare_ply_parses_on_track_and_world_binds(self) -> None:
        """A controls-free .ply edit parses like any static prop, with the usual placement defaults."""
        fx = EditSpec.from_dict(_edit(asset="props/cone.ply", controls={}), "edits[0]")
        self.assertEqual(fx.controls, [])
        self.assertEqual(fx.placement.scale_mode, "fit_length")
        world = EditSpec.from_dict(
            {
                "id": "cone_1",
                "op": "add",
                "bind": {"type": "world", "position": [1.0, 2.0, 0.0], "dims": [0.4, 0.4, 0.7]},
                "asset": "props/cone.ply",
                "placement": {"scale": "fit_length"},
            },
            "edits[0]",
        )
        self.assertEqual(world.placement.scale_mode, "fit_length")
        assert world.bind is not None
        self.assertEqual(world.bind.dims, [0.4, 0.4, 0.7])

    def test_controls_rejected(self) -> None:
        """Controls on a .ply fail at parse time, naming the offending control and the fix."""
        with self.assertRaisesRegex(AssetEditsError, "takes no controls .*panelstate.*SimReady USD"):
            EditSpec.from_dict(_edit(asset="props/cone.ply"), "edits[0]")

    def test_prim_path_rejected(self) -> None:
        """A prim_path on a .ply is meaningless (no prims) and fails at parse time."""
        with self.assertRaisesRegex(AssetEditsError, r"prim_path does not apply to a \.ply"):
            EditSpec.from_dict(_edit(asset="props/cone.ply", controls={}, prim_path="/Body"), "edits[0]")


class DocResolutionTest(unittest.TestCase):
    """Document-level checks and per-edit layer resolution."""

    def test_top_level_must_be_mapping(self) -> None:
        """A non-mapping top level (e.g. a list) is rejected."""
        with self.assertRaisesRegex(AssetEditsError, "top-level"):
            AssetEditsDoc.from_dict(["edits"])  # type: ignore[arg-type]

    def test_document_requires_edits(self) -> None:
        """A document whose edits list is empty is rejected."""
        with self.assertRaisesRegex(AssetEditsError, "no edits"):
            AssetEditsDoc.from_dict(_doc(edits=[]))

    def test_layer_defaults_and_explicit_override(self) -> None:
        """Edits without layer_id resolve to DEFAULT_LAYER_ID at parse; an explicit layer_id wins."""
        doc = AssetEditsDoc.from_dict(
            _doc(
                edits=[
                    _edit(id="a"),
                    _edit(id="b", layer_id="custom_rigids", bind={"nre_track_id": 4}),
                ]
            )
        )
        self.assertEqual(doc.edits[0].layer_id, DEFAULT_LAYER_ID)
        self.assertEqual(doc.edits[1].layer_id, "custom_rigids")  # explicit layer wins
        self.assertEqual(list(dict.fromkeys(e.layer_id for e in doc.edits)), [DEFAULT_LAYER_ID, "custom_rigids"])


class FailLoudScalarsTest(unittest.TestCase):
    """Malformed scalars raise AssetEditsError (never a raw ValueError/TypeError), and
    numeric coercion never silently truncates."""

    def test_non_numeric_yaw_rejected_cleanly(self) -> None:
        with self.assertRaisesRegex(AssetEditsError, "yaw_deg.*ninety"):
            AssetPlacement.from_dict({"yaw_deg": "ninety"}, "p")

    def test_non_numeric_scale_rejected_cleanly(self) -> None:
        with self.assertRaisesRegex(AssetEditsError, "scale"):
            AssetPlacement.from_dict({"scale": {"x": 1}}, "p")

    def test_boolean_scalar_rejected(self) -> None:
        """YAML true would silently coerce to 1.0/1 via float()/int()."""
        with self.assertRaisesRegex(AssetEditsError, "yaw_deg"):
            Binding.from_dict({"type": "world", "position": [0, 0, 0], "yaw_deg": True}, "b")

    def test_fractional_track_id_rejected_not_truncated(self) -> None:
        """nurec_track_id 7.9 must not silently address track 7."""
        with self.assertRaisesRegex(AssetEditsError, "nurec_track_id.*7.9"):
            Binding.from_dict({"nre_track_id": 7.9}, "b")

    def test_integral_float_track_id_accepted(self) -> None:
        self.assertEqual(Binding.from_dict({"nre_track_id": 7.0}, "b").nurec_track_id, 7)

    def test_non_numeric_track_id_rejected_cleanly(self) -> None:
        with self.assertRaisesRegex(AssetEditsError, "nurec_track_id"):
            Binding.from_dict({"nre_track_id": "seven"}, "b")


class FailLoudStructureTest(unittest.TestCase):
    """Unknown keys and malformed structure fail loud at every level of the document."""

    def test_unknown_placement_field_rejected(self) -> None:
        """A typo'd placement key (yaw_degrees) must not silently bake with defaults."""
        with self.assertRaisesRegex(AssetEditsError, "yaw_degrees"):
            AssetPlacement.from_dict({"yaw_degrees": 90.0}, "p")

    def test_unknown_top_level_field_rejected(self) -> None:
        """A document-level stray key (e.g. a misplaced layer_id) must not silently no-op."""
        with self.assertRaisesRegex(AssetEditsError, "layer_id"):
            AssetEditsDoc.from_dict(_doc(layer_id="custom_rigids"))

    def test_edits_must_be_a_list(self) -> None:
        with self.assertRaisesRegex(AssetEditsError, "list"):
            AssetEditsDoc.from_dict({"edits": "truck_1"})

    def test_non_mapping_edit_rejected(self) -> None:
        with self.assertRaisesRegex(AssetEditsError, "edits\\[0\\].*mapping"):
            AssetEditsDoc.from_dict({"edits": ["truck_1"]})

    def test_duplicate_edit_ids_rejected(self) -> None:
        """Edit ids key the baker's bookkeeping; a duplicate would bake the wrong asset."""
        with self.assertRaisesRegex(AssetEditsError, "duplicate edit id.*truck_1"):
            AssetEditsDoc.from_dict({"edits": [_edit(), _edit(asset="other/scene.usd")]})

    def test_distinct_edit_ids_accepted(self) -> None:
        doc = AssetEditsDoc.from_dict({"edits": [_edit(), _edit(id="truck_2")]})
        self.assertEqual([e.id for e in doc.edits], ["truck_1", "truck_2"])


class DuplicateKeyTest(unittest.TestCase):
    """A repeated mapping key is rejected instead of resolving to its last value.

    YAML's last-one-wins rule is invisible to every other check in the schema, which all
    run on the parsed dict, so a repeated key silently changes what was authored.
    """

    def _write(self, tmp_path, body: str):
        path = Path(tmp_path) / "edits.yaml"
        path.write_text(body)
        return path

    def test_duplicate_bind_is_rejected(self) -> None:
        """Left unchecked this rebinds the asset to a different track."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                "edits:\n"
                "  - id: a\n"
                "    op: add\n"
                "    bind: {type: nurec_track, nurec_track_id: 7}\n"
                "    bind: {type: nurec_track, nurec_track_id: 42}\n"
                "    asset: scene.usd\n",
            )
            with self.assertRaisesRegex(AssetEditsError, "duplicate key 'bind'"):
                load_asset_edits(path)

    def test_duplicate_control_is_rejected(self) -> None:
        """Left unchecked this drops the first entry's controllable/semantics."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                "edits:\n"
                "  - id: a\n"
                "    op: add\n"
                "    bind: {type: nurec_track, nurec_track_id: 7}\n"
                "    asset: scene.usd\n"
                "    controls:\n"
                "      panel: {default: 'off', controllable: true}\n"
                "      panel: {default: 'off'}\n",
            )
            with self.assertRaisesRegex(AssetEditsError, "duplicate key 'panel'"):
                load_asset_edits(path)

    def test_malformed_yaml_becomes_a_domain_error(self) -> None:
        """A bad indent must not escape as a raw PyYAML traceback."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "edits:\n  - id: a\n   op: add\n   bind: {nurec_track_id: 1}\n")
            with self.assertRaisesRegex(AssetEditsError, "invalid YAML"):
                load_asset_edits(path)

    def test_valid_document_still_parses(self) -> None:
        """A well-formed document loads through the strict loader."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                "edits:\n  - id: a\n    op: add\n    bind: {type: nurec_track, nurec_track_id: 7}\n    asset: scene.usd\n",
            )
            self.assertEqual(load_asset_edits(path).edits[0].nurec_track_id, 7)


class NonFiniteScalarTest(unittest.TestCase):
    """Booleans and non-finite numbers are schema errors, never silent placement values."""

    def test_vec3_rejects_bool_and_non_finite_components(self) -> None:
        """A placement offset with a bool, NaN or inf component is refused."""
        for bad in ([True, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, float("inf")]):
            with self.subTest(bad=bad), self.assertRaises(AssetEditsError):
                AssetEditsDoc.from_dict(
                    {
                        "edits": [
                            {
                                "id": "a",
                                "op": "add",
                                "bind": {"type": "nre_track", "nre_track_id": 0},
                                "asset": "s.usd",
                                "placement": {"offset": bad},
                            }
                        ]
                    }
                )

    def test_non_finite_scale_and_yaw_rejected(self) -> None:
        """A NaN scale or an infinite yaw is refused."""
        for field, bad in (("scale", float("nan")), ("yaw_deg", float("inf"))):
            with self.subTest(field=field), self.assertRaises(AssetEditsError):
                AssetEditsDoc.from_dict(
                    {
                        "edits": [
                            {
                                "id": "a",
                                "op": "add",
                                "bind": {"type": "nre_track", "nre_track_id": 0},
                                "asset": "s.usd",
                                "placement": {field: bad},
                            }
                        ]
                    }
                )


class ExcludeEditTest(unittest.TestCase):
    """An exclude edit names a captured track and nothing else."""

    def test_parses_and_predicates(self) -> None:
        """An exclude parses without asset or controls and is keyed by its track."""
        fx = EditSpec.from_dict(_exclude(), "edits[0]")
        self.assertEqual((fx.op, fx.is_exclude, fx.hides_original, fx.has_asset), ("exclude", True, True, False))
        self.assertIsNone(fx.asset)
        self.assertEqual(fx.controls, [])
        self.assertEqual(fx.nurec_track_id, 5)
        self.assertEqual(binding_key(fx), (DEFAULT_LAYER_ID, 5))

    def test_extra_keys_rejected(self) -> None:
        """An exclude bakes nothing: asset, controls and placement are refused."""
        for key, value in (("asset", "a.usd"), ("controls", {}), ("placement", {}), ("prim_path", "/Car")):
            with self.subTest(key=key), self.assertRaisesRegex(AssetEditsError, "unknown exclude edit field"):
                EditSpec.from_dict(_exclude(**{key: value}), "edits[0]")

    def test_world_bind_rejected(self) -> None:
        """There is no captured actor at a world pose to exclude."""
        with self.assertRaisesRegex(AssetEditsError, "binds an existing track"):
            EditSpec.from_dict(_exclude(bind={"type": "world", "position": [0, 0, 0]}), "edits[0]")

    def test_layer_id_honoured(self) -> None:
        """An exclude keeps an explicit layer_id."""
        fx = EditSpec.from_dict(_exclude(layer_id="static_rigids"), "edits[0]")
        self.assertEqual(fx.layer_id, "static_rigids")


class CloneEditTest(unittest.TestCase):
    """Clone edits: source (captured track or another edit), motion (inherit or parked), track id."""

    def test_captured_clone_parses(self) -> None:
        """A clone of a captured track parses with inherited motion and its own track id."""
        fx = EditSpec.from_dict(_clone(), "edits[0]")
        assert fx.clone is not None
        self.assertTrue(fx.is_clone and fx.binds_synthetic_track)
        self.assertEqual((fx.has_asset, fx.hides_original, fx.is_world), (False, False, False))
        self.assertTrue(fx.clone.is_captured)
        self.assertEqual(fx.clone.source_nurec_track_id, 12)
        self.assertEqual(fx.clone.track_id, "9000221")
        self.assertEqual(fx.synthetic_track_id, "9000221")
        self.assertEqual(binding_key(fx), ("synthetic", "9000221"))
        motion = fx.clone.motion
        self.assertEqual(
            (motion.kind, motion.offset, motion.yaw_deg, motion.time_shift_us), ("inherit", [0.0, -2.4, 0.0], 0.0, 0)
        )

    def test_parked_motion_parses(self) -> None:
        """A clone with a world motion parks there, anchored at its base by default."""
        fx = EditSpec.from_dict(_clone(motion={"world": {"position": [1.0, 2.0, 0.0], "yaw_deg": 90.0}}), "edits[0]")
        assert fx.clone is not None
        motion = fx.clone.motion
        self.assertTrue(motion.is_world)
        self.assertEqual((motion.position, motion.yaw_deg, motion.anchor), ([1.0, 2.0, 0.0], 90.0, "base"))

    def test_edit_source_parses_at_document_level(self) -> None:
        """A clone of an edit takes the source's layer and resolves its appearance to it."""
        doc = AssetEditsDoc.from_dict(
            {"edits": [_edit(layer_id="custom_rigids"), _clone(id="c", source={"edit_id": "truck_1"})]}
        )
        clone = doc.edits[1]
        assert clone.clone is not None
        self.assertFalse(clone.clone.is_captured)
        self.assertEqual(clone.layer_id, "custom_rigids")
        self.assertIs(doc.appearance_of(clone), doc.edits[0])
        self.assertIs(doc.appearance_of(doc.edits[0]), doc.edits[0])

    def test_required_keys(self) -> None:
        """A clone without source, motion or track_id is refused."""
        for key in ("source", "motion", "track_id"):
            d = _clone()
            del d[key]
            with self.subTest(key=key), self.assertRaisesRegex(AssetEditsError, f"requires '{key}'"):
                EditSpec.from_dict(d, "edits[0]")

    def test_foreign_keys_rejected(self) -> None:
        """A clone copies an appearance: asset/controls/bind/placement are not its to declare."""
        for key, value in (("asset", "a.usd"), ("controls", {}), ("bind", {"nurec_track_id": 1}), ("placement", {})):
            with self.subTest(key=key), self.assertRaisesRegex(AssetEditsError, "unknown clone edit field"):
                EditSpec.from_dict(_clone(**{key: value}), "edits[0]")

    def test_source_needs_exactly_one_kind(self) -> None:
        """A clone source names one captured track or one edit id, and nothing else."""
        with self.assertRaisesRegex(AssetEditsError, "exactly one of"):
            EditSpec.from_dict(_clone(source={"nurec_track_id": 1, "edit_id": "x"}), "edits[0]")
        with self.assertRaisesRegex(AssetEditsError, "exactly one of"):
            EditSpec.from_dict(_clone(source={}), "edits[0]")
        with self.assertRaisesRegex(AssetEditsError, "unknown clone source field"):
            EditSpec.from_dict(_clone(source={"track_id": "x"}), "edits[0]")

    def test_motion_forms_validated(self) -> None:
        """Clone motion is either 'inherit: true' or a world pose, and its fields are checked."""
        with self.assertRaisesRegex(AssetEditsError, "exactly one of 'inherit: true' or 'world"):
            EditSpec.from_dict(_clone(motion={}), "edits[0]")
        with self.assertRaisesRegex(AssetEditsError, "exactly one of 'inherit: true' or 'world"):
            EditSpec.from_dict(_clone(motion={"inherit": True, "world": {"position": [0, 0, 0]}}), "edits[0]")
        with self.assertRaisesRegex(AssetEditsError, "inherit: must be true"):
            EditSpec.from_dict(_clone(motion={"inherit": False}), "edits[0]")
        with self.assertRaisesRegex(AssetEditsError, "unknown motion field"):
            EditSpec.from_dict(_clone(motion={"inherit": True, "position": [0, 0, 0]}), "edits[0]")
        with self.assertRaisesRegex(AssetEditsError, "world: requires 'position'"):
            EditSpec.from_dict(_clone(motion={"world": {"yaw_deg": 1.0}}), "edits[0]")
        with self.assertRaisesRegex(AssetEditsError, "time_shift_us.*integer"):
            EditSpec.from_dict(_clone(motion={"inherit": True, "time_shift_us": 0.5}), "edits[0]")

    def test_track_id_must_be_non_empty(self) -> None:
        """A blank clone track_id is refused."""
        with self.assertRaisesRegex(AssetEditsError, "track_id"):
            EditSpec.from_dict(_clone(track_id="  "), "edits[0]")

    def test_source_edit_must_exist_and_bake_an_asset(self) -> None:
        """A clone's source edit must be in the document and bake an asset."""
        with self.assertRaisesRegex(AssetEditsError, "not in this document"):
            AssetEditsDoc.from_dict({"edits": [_clone(source={"edit_id": "ghost"})]})
        with self.assertRaisesRegex(AssetEditsError, "is an exclude edit"):
            AssetEditsDoc.from_dict({"edits": [_exclude(), _clone(source={"edit_id": "drop_1"})]})
        with self.assertRaisesRegex(AssetEditsError, "is a clone edit"):
            AssetEditsDoc.from_dict(
                {"edits": [_clone(), _clone(id="c2", source={"edit_id": "lead_clone"}, track_id="x")]}
            )

    def test_source_edit_layer_mismatch_rejected(self) -> None:
        """A clone may not name a layer other than its source edit's."""
        with self.assertRaisesRegex(AssetEditsError, "differs from its clone source"):
            AssetEditsDoc.from_dict(
                {"edits": [_edit(), _clone(source={"edit_id": "truck_1"}, layer_id="custom_rigids")]}
            )

    def test_explicit_default_layer_mismatch_rejected(self) -> None:
        """An explicit ``layer_id`` equal to the default is still an explicit mismatch, not an omission."""
        source = _edit(layer_id="static_rigids")
        with self.assertRaisesRegex(AssetEditsError, "layer_id 'dynamic_rigids' differs from its clone source"):
            AssetEditsDoc.from_dict(
                {"edits": [source, _clone(source={"edit_id": "truck_1"}, layer_id=DEFAULT_LAYER_ID)]}
            )
        doc = AssetEditsDoc.from_dict({"edits": [source, _clone(id="c", source={"edit_id": "truck_1"})]})
        self.assertEqual(doc.edits[1].layer_id, "static_rigids")  # omitted: inherited from the source

    def test_synthetic_track_id_collisions_rejected(self) -> None:
        """World binds and clones each claim a NEW track id; two edits on one id fail."""
        world = {
            "id": "prop",
            "op": "add",
            "bind": {"type": "world", "position": [0, 0, 0], "track_id": "9000221"},
            "asset": "a.usd",
        }
        with self.assertRaisesRegex(AssetEditsError, "synthetic track id.*9000221"):
            AssetEditsDoc.from_dict({"edits": [world, _clone()]})
        with self.assertRaisesRegex(AssetEditsError, "synthetic track id"):
            AssetEditsDoc.from_dict({"edits": [_clone(), _clone(id="c2", source={"nurec_track_id": 1})]})

    def test_exclude_and_clone_of_the_same_track_coexist(self) -> None:
        """Hiding the source and cloning it elsewhere is a move; the keys do not collide."""
        doc = AssetEditsDoc.from_dict({"edits": [_exclude(bind={"nurec_track_id": 12}), _clone()]})
        self.assertNotEqual(binding_key(doc.edits[0]), binding_key(doc.edits[1]))

    def test_needs_clip_window(self) -> None:
        """Parked clones and world binds need the clip window; inheriting clones do not."""
        self.assertFalse(AssetEditsDoc.from_dict({"edits": [_clone()]}).needs_clip_window)
        parked = _clone(motion={"world": {"position": [0, 0, 0]}})
        self.assertTrue(AssetEditsDoc.from_dict({"edits": [parked]}).needs_clip_window)
        self.assertTrue(
            AssetEditsDoc.from_dict(
                {"edits": [_edit(op="add", bind={"type": "world", "position": [0, 0, 0]}, controls={})]}
            ).needs_clip_window
        )


class ToDictRoundTripTest(unittest.TestCase):
    """``to_dict`` writes the authored form with every default resolved; re-parsing it is a fixpoint."""

    def _roundtrip(self, doc_dict: dict[str, Any]) -> None:
        """Assert that the document survives to_dict and from_dict unchanged."""
        doc = AssetEditsDoc.from_dict(doc_dict)
        again = AssetEditsDoc.from_dict(doc.to_dict())
        self.assertEqual(again, doc)
        self.assertEqual(again.to_dict(), doc.to_dict())

    def test_every_op_roundtrips(self) -> None:
        """Every op, with its optional fields, round-trips through to_dict."""
        self._roundtrip(
            {
                "edits": [
                    _edit(placement={"yaw_deg": 90.0, "rotation_deg": [1.0, 2.0, 3.0]}, prim_path="/Car"),
                    _edit(id="attach", op="add", controls={"appearance": {"semantics": ["x"]}}),
                    {
                        "id": "prop",
                        "op": "add",
                        "bind": {"type": "world", "position": [1.0, 2.0, 0.0], "dims": [1, 1, 1]},
                        "asset": "p.usd",
                        "placement": {"scale": "fit_length"},
                    },
                    _clone(motion={"inherit": True, "offset": [1.0, 0.0, 0.0], "yaw_deg": 5.0, "time_shift_us": 3}),
                    _clone(
                        id="parked",
                        source={"edit_id": "truck_1"},
                        motion={"world": {"position": [0, 1, 2], "anchor": "center"}},
                        track_id="p2",
                    ),
                    _exclude(layer_id="static_rigids"),
                ]
            }
        )

    def test_world_bind_default_track_id_is_written(self) -> None:
        """to_dict writes a world bind's default track id and scale."""
        doc = AssetEditsDoc.from_dict(
            {"edits": [{"id": "prop", "op": "add", "bind": {"type": "world", "position": [0, 0, 0]}, "asset": "p.usd"}]}
        )
        self.assertEqual(doc.to_dict()["edits"][0]["bind"]["track_id"], "prop:prop")
        self.assertEqual(doc.to_dict()["edits"][0]["placement"]["scale"], 1.0)

    def test_clone_motion_dict_forms(self) -> None:
        """CloneMotion.to_dict writes every field, defaults included."""
        self.assertEqual(
            CloneMotion.from_dict({"inherit": True}, "m").to_dict(),
            {"inherit": True, "offset": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "time_shift_us": 0},
        )
        self.assertEqual(
            CloneMotion.from_dict({"world": {"position": [1, 2, 3]}}, "m").to_dict(),
            {"world": {"position": [1.0, 2.0, 3.0], "yaw_deg": 0.0, "anchor": "base"}},
        )


if __name__ == "__main__":
    unittest.main()
