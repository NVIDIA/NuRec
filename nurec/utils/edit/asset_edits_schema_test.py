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
    ControlSpec,
    EditSpec,
    _as_vec3,
    _norm_token,
    is_ply_asset,
    load_asset_edits,
)


def _edit(**over: Any) -> dict[str, Any]:
    """A minimal valid edit dict; keyword args override fields."""
    d: dict[str, Any] = {
        "id": "truck_1",
        "bind": {"type": "nre_track", "nre_track_id": 3},
        "asset": "usd/configurable_asset/scene.usd",
        "controls": {"panelState": {"default": "off", "controllable": True}},
    }
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
        d: dict[str, Any] = {
            "id": "barrier_1",
            "bind": {"type": "world", "position": [150.0, -3.5, 0.0], "yaw_deg": 90.0},
            "asset": "props/barrier/Body.usd",
        }
        d.update(over)
        return d

    def test_track_id_defaults_to_prop_edit_id(self) -> None:
        """A world bind without an explicit track_id gets "prop:<edit_id>" on parse."""
        fx = EditSpec.from_dict(self._world_edit(), "edits[0]")
        self.assertEqual(fx.bind.track_id, "prop:barrier_1")

    def test_replace_original_rejected(self) -> None:
        """replace_original on a world bind is meaningless and fails loud."""
        with self.assertRaisesRegex(AssetEditsError, "replace_original"):
            EditSpec.from_dict(self._world_edit(replace_original=True), "edits[0]")

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
        """The world-bind scale default does not leak into nurec_track edits."""
        fx = EditSpec.from_dict(_edit(), "edits[0]")
        assert fx.placement is not None
        self.assertEqual(fx.placement.scale_mode, "fit_length")


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
        """Missing id, bind or asset each fails with an error naming the absent key."""
        with self.assertRaisesRegex(AssetEditsError, "requires an 'id'"):
            EditSpec.from_dict({"bind": {"nre_track_id": 1}, "asset": "a.usd"}, "edits[0]")
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

    def test_replace_original_must_be_boolean(self) -> None:
        """A string value for replace_original is rejected; only real booleans pass."""
        with self.assertRaisesRegex(AssetEditsError, "expected a boolean"):
            EditSpec.from_dict(_edit(replace_original="no"), "edits[0]")

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
        fx = EditSpec.from_dict(
            _edit(placement={"align": "bbox", "scale": "fit_length"}, replace_original=True),
            "edits[0]",
        )
        self.assertEqual(fx.id, "truck_1")
        self.assertEqual(fx.bind.nurec_track_id, 3)
        self.assertTrue(fx.replace_original)
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
                "bind": {"type": "world", "position": [1.0, 2.0, 0.0], "dims": [0.4, 0.4, 0.7]},
                "asset": "props/cone.ply",
                "placement": {"scale": "fit_length"},
            },
            "edits[0]",
        )
        self.assertEqual(world.placement.scale_mode, "fit_length")
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
            path = self._write(tmp, "edits:\n  - id: a\n   bind: {nurec_track_id: 1}\n")
            with self.assertRaisesRegex(AssetEditsError, "invalid YAML"):
                load_asset_edits(path)

    def test_valid_document_still_parses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                "edits:\n  - id: a\n    bind: {type: nurec_track, nurec_track_id: 7}\n    asset: scene.usd\n",
            )
            self.assertEqual(load_asset_edits(path).edits[0].bind.nurec_track_id, 7)


class NonFiniteScalarTest(unittest.TestCase):
    """Booleans and non-finite numbers are schema errors, never silent placement values."""

    def test_vec3_rejects_bool_and_non_finite_components(self) -> None:
        for bad in ([True, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, float("inf")]):
            with self.subTest(bad=bad), self.assertRaises(AssetEditsError):
                AssetEditsDoc.from_dict(
                    {
                        "edits": [
                            {
                                "id": "a",
                                "bind": {"type": "nre_track", "nre_track_id": 0},
                                "asset": "s.usd",
                                "placement": {"offset": bad},
                            }
                        ]
                    }
                )

    def test_non_finite_scale_and_yaw_rejected(self) -> None:
        for field, bad in (("scale", float("nan")), ("yaw_deg", float("inf"))):
            with self.subTest(field=field), self.assertRaises(AssetEditsError):
                AssetEditsDoc.from_dict(
                    {
                        "edits": [
                            {
                                "id": "a",
                                "bind": {"type": "nre_track", "nre_track_id": 0},
                                "asset": "s.usd",
                                "placement": {field: bad},
                            }
                        ]
                    }
                )


if __name__ == "__main__":
    unittest.main()
