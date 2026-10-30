# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`read_variant_animation` (variant-state animation extraction).

Standalone by construction: tiny USD stages are authored into tempfiles with ``pxr``
(one root prim, one variantSet, a couple of element Xforms); no scene data, no GPU,
no torch.
"""

import tempfile
import unittest

from pathlib import Path
from typing import Optional

from pxr import Gf, Sdf, Usd, UsdGeom

from nurec.utils.edit.variant_animation import (
    VariantAnimation,
    VariantAnimationError,
    read_variant_animation,
)


def _author_asset(
    dir_path: str,
    *,
    animated: bool = True,
    timecodes: bool = True,
    selection: Optional[str] = "off",
    default_prim: bool = True,
) -> str:
    """Author a tiny SimReady-style asset: /Car with an ``appearanceMode`` variantSet.

    One cycle is 2.0 s (timeCodes 0..48 at 24 tps). The ``sequence`` state carries an
    element ``/Car/Detail`` with visibility on for the first half-cycle, a ``nurec:gain``
    ramp and a ``nurec:coldRgb`` color, plus a sample-less element ``/Car/Frame``.
    """
    path = str(Path(dir_path) / "car.usda")
    stage = Usd.Stage.CreateNew(path)
    if timecodes:
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(48)
        stage.SetTimeCodesPerSecond(24)
    car = UsdGeom.Xform.Define(stage, "/Car")
    if default_prim:
        stage.SetDefaultPrim(car.GetPrim())
    vset = car.GetPrim().GetVariantSets().AddVariantSet("appearanceMode")
    vset.AddVariant("off")
    vset.AddVariant("sequence")
    vset.SetVariantSelection("sequence")
    with vset.GetVariantEditContext():
        detail = UsdGeom.Xform.Define(stage, "/Car/Detail")
        if animated:
            vis = UsdGeom.Imageable(detail.GetPrim()).GetVisibilityAttr()
            vis.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(0))
            vis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(24))
            gain = detail.GetPrim().CreateAttribute("nurec:gain", Sdf.ValueTypeNames.Float, True)
            gain.Set(0.2, Usd.TimeCode(0))
            gain.Set(1.0, Usd.TimeCode(24))
            cold = detail.GetPrim().CreateAttribute("nurec:coldRgb", Sdf.ValueTypeNames.Color3f, True)
            cold.Set(Gf.Vec3f(0.1, 0.2, 0.3))
        # Membership-only element: defined by the state, but carries no timeSamples.
        UsdGeom.Xform.Define(stage, "/Car/Frame")
    if selection is None:
        vset.ClearVariantSelection()
    else:
        vset.SetVariantSelection(selection)
    stage.GetRootLayer().Save()
    return path


def _author_member_asset(dir_path: str) -> str:
    """An asset whose element /Car/Surface is defined OUTSIDE the variants (present in
    every state) but only carries visibility timeSamples inside ``sequence``."""
    path = str(Path(dir_path) / "member.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(48)
    stage.SetTimeCodesPerSecond(24)
    car = UsdGeom.Xform.Define(stage, "/Car")
    stage.SetDefaultPrim(car.GetPrim())
    surface = UsdGeom.Xform.Define(stage, "/Car/Surface")
    vset = car.GetPrim().GetVariantSets().AddVariantSet("appearance")
    vset.AddVariant("off")
    vset.AddVariant("sequence")
    vset.SetVariantSelection("sequence")
    with vset.GetVariantEditContext():
        vis = UsdGeom.Imageable(surface.GetPrim()).GetVisibilityAttr()
        vis.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(0))
        vis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(24))
    vset.SetVariantSelection("off")
    stage.GetRootLayer().Save()
    return path


def _author_shared_halves_asset(dir_path: str) -> str:
    """Two alternate surfaces /Car/Primary and /Car/Secondary defined OUTSIDE the
    variants; ``sequence`` alternates both halves with visibility timeSamples while
    ``off`` statically hides Primary and shows Secondary (sample-less overrides only)."""
    path = str(Path(dir_path) / "halves.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(48)
    stage.SetTimeCodesPerSecond(24)
    car = UsdGeom.Xform.Define(stage, "/Car")
    stage.SetDefaultPrim(car.GetPrim())
    primary = UsdGeom.Xform.Define(stage, "/Car/Primary")
    secondary = UsdGeom.Xform.Define(stage, "/Car/Secondary")
    vset = car.GetPrim().GetVariantSets().AddVariantSet("appearance")
    vset.AddVariant("off")
    vset.AddVariant("sequence")
    vset.SetVariantSelection("sequence")
    with vset.GetVariantEditContext():
        vis = UsdGeom.Imageable(primary.GetPrim()).GetVisibilityAttr()
        vis.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(0))
        vis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(24))
        dvis = UsdGeom.Imageable(secondary.GetPrim()).GetVisibilityAttr()
        dvis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(0))
        dvis.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(24))
    vset.SetVariantSelection("off")
    with vset.GetVariantEditContext():
        UsdGeom.Imageable(primary.GetPrim()).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdGeom.Imageable(secondary.GetPrim()).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    stage.GetRootLayer().Save()
    return path


def _author_parent_gated_asset(dir_path: str) -> str:
    """``sequence`` samples BOTH a parent bar Xform and its child panel: /Car/Bar drops
    invisible at half-cycle (tc 24) while /Car/Bar/Panel's own samples would keep it on
    until tc 36 — inherited visibility must clip the panel's window to the bar's."""
    path = str(Path(dir_path) / "gated.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(48)
    stage.SetTimeCodesPerSecond(24)
    car = UsdGeom.Xform.Define(stage, "/Car")
    stage.SetDefaultPrim(car.GetPrim())
    vset = car.GetPrim().GetVariantSets().AddVariantSet("appearance")
    vset.AddVariant("off")
    vset.AddVariant("sequence")
    vset.SetVariantSelection("sequence")
    with vset.GetVariantEditContext():
        bar = UsdGeom.Xform.Define(stage, "/Car/Bar")
        bvis = UsdGeom.Imageable(bar.GetPrim()).GetVisibilityAttr()
        bvis.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(0))
        bvis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(24))
        panel = UsdGeom.Xform.Define(stage, "/Car/Bar/Panel")
        lvis = UsdGeom.Imageable(panel.GetPrim()).GetVisibilityAttr()
        lvis.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(0))
        lvis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(36))
    vset.SetVariantSelection("off")
    stage.GetRootLayer().Save()
    return path


def _author_hidden_bar_asset(dir_path: str) -> str:
    """An element /Car/Bar/Surface authored as an UNREGISTERED-schema prim (like real splat
    payloads), sampled inside ``sequence``, while ``off`` hides the whole bar with one
    static ``invisible`` on the parent Xform — membership must honour the inherited
    chain, not the element's local opinion."""
    path = str(Path(dir_path) / "hidden_bar.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(48)
    stage.SetTimeCodesPerSecond(24)
    car = UsdGeom.Xform.Define(stage, "/Car")
    stage.SetDefaultPrim(car.GetPrim())
    bar = UsdGeom.Xform.Define(stage, "/Car/Bar")
    surface = stage.DefinePrim("/Car/Bar/Surface", "")
    vset = car.GetPrim().GetVariantSets().AddVariantSet("appearance")
    vset.AddVariant("off")
    vset.AddVariant("sequence")
    vset.SetVariantSelection("sequence")
    with vset.GetVariantEditContext():
        vis = surface.CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False)
        vis.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(0))
        vis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(24))
    vset.SetVariantSelection("off")
    with vset.GetVariantEditContext():
        UsdGeom.Imageable(bar.GetPrim()).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    stage.GetRootLayer().Save()
    return path


class ReadVariantAnimationTest(unittest.TestCase):
    """Behavioral tests for read_variant_animation over tempfile-authored USD variant assets."""

    def setUp(self) -> None:
        """Creates a per-test temporary directory for authored USD files."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _anim(self, path: str, variant_set: str = "appearanceMode") -> VariantAnimation:
        """Reads the variant animation from path and asserts a non-None result."""
        anim = read_variant_animation(path, variant_set)
        assert anim is not None
        return anim

    def test_unknown_variant_set_returns_none(self) -> None:
        """Asking for a variantSet the asset does not have yields None rather than an error."""
        path = _author_asset(self._tmp.name)
        self.assertIsNone(read_variant_animation(path, "noSuchSet"))

    def test_case_insensitive_set_name_returns_canonical(self) -> None:
        """A differently-cased set name matches and the result reports the canonical authored name."""
        path = _author_asset(self._tmp.name)
        anim = self._anim(path, "APPEARANCEmode")
        self.assertIsInstance(anim, VariantAnimation)
        self.assertEqual(anim.variant_set, "appearanceMode")
        self.assertEqual(anim.windows_by_state, self._anim(path).windows_by_state)

    def test_cycle_seconds_from_timecode_range(self) -> None:
        """cycle_s is the stage timeCode span divided by timeCodesPerSecond."""
        path = _author_asset(self._tmp.name)
        self.assertEqual(self._anim(path).cycle_s, 2.0)  # (48 - 0) / 24 tps

    def test_visibility_windows_normalized(self) -> None:
        """Visibility timeSamples become on-windows in normalized [0, 1) cycle fractions per state."""
        path = _author_asset(self._tmp.name)
        anim = self._anim(path)
        # inherited@0 -> invisible@24 over a 48-tc cycle: on for the first half.
        self.assertEqual(anim.windows_by_state["sequence"], {"Detail": [(0.0, 0.5)]})
        self.assertEqual(anim.windows_by_state["off"], {})
        self.assertTrue(anim.is_animated)

    def test_element_keys_are_root_relative_paths(self) -> None:
        """Window and gain dicts key elements by prim path relative to the root prim, not absolute path."""
        path = _author_asset(self._tmp.name)
        anim = self._anim(path)
        # Keys are the prim path with the root prefix and leading "/" stripped.
        self.assertEqual(sorted(anim.windows_by_state["sequence"]), ["Detail"])
        self.assertEqual(sorted(anim.gains_by_state["sequence"]), ["Detail"])

    def test_gain_curve_keyed_by_cycle_position(self) -> None:
        """nurec:gain samples are returned as (cycle-position, value) points, absent from sample-less states."""
        path = _author_asset(self._tmp.name)
        pts = self._anim(path).gains_by_state["sequence"]["Detail"]
        # Samples at tc 0 and 24 normalize to cycle positions 0.0 and 0.5 (not 1.0).
        self.assertEqual(len(pts), 2)
        self.assertEqual(pts[0][0], 0.0)
        self.assertAlmostEqual(pts[0][1], 0.2, places=6)  # float32 round-trip
        self.assertEqual(pts[1], (0.5, 1.0))
        self.assertEqual(self._anim(path).gains_by_state["off"], {})

    def test_cold_rgb_extracted(self) -> None:
        """An authored nurec:coldRgb color surfaces per state and element; states without it stay empty."""
        path = _author_asset(self._tmp.name)
        anim = self._anim(path)
        rgb = anim.cold_rgb_by_state["sequence"]["Detail"]
        for got, want in zip(rgb, (0.1, 0.2, 0.3)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(anim.cold_rgb_by_state["off"], {})

    def test_members_by_state(self) -> None:
        """Membership tracks only sampled element keys per state; sample-less prims never appear."""
        path = _author_asset(self._tmp.name)
        anim = self._anim(path)
        # Detail exists only inside sequence, so it is a member of sequence only.
        self.assertEqual(anim.members_by_state, {"off": [], "sequence": ["Detail"]})
        # Pins current behavior: membership only tracks the union of ELEMENT keys
        # (prims sampled in at least one state). /Car/Frame has no samples in any
        # state, so it never becomes a key and appears in no state's member list.
        for members in anim.members_by_state.values():
            self.assertNotIn("Frame", members)

    def test_member_without_samples_listed_in_every_composing_state(self) -> None:
        """An element sampled in one state is a member of every state that composes it, windows or not."""
        path = _author_member_asset(self._tmp.name)
        anim = self._anim(path, "appearance")
        # Surface is sampled in sequence (-> element key) and composed sample-less in off,
        # so it is a member of both states but only has windows in sequence.
        self.assertEqual(anim.members_by_state, {"off": ["Surface"], "sequence": ["Surface"]})
        self.assertEqual(anim.windows_by_state, {"off": {}, "sequence": {"Surface": [(0.0, 0.5)]}})

    def test_statically_hidden_element_not_member_of_state(self) -> None:
        """An element composed by a state WITHOUT samples joins its member list only when
        statically visible: an alternate surface hidden by the off state's sample-less visibility
        override must not be an off member (consumers treat window-less members as
        always-visible, so it would render the hidden surface while the control is off)."""
        path = _author_shared_halves_asset(self._tmp.name)
        anim = self._anim(path, "appearance")
        self.assertEqual(anim.members_by_state, {"off": ["Secondary"], "sequence": ["Primary", "Secondary"]})
        self.assertEqual(sorted(anim.windows_by_state["sequence"]), ["Primary", "Secondary"])

    def test_parent_xform_sample_clips_child_window(self) -> None:
        """A parent Xform going invisible at a timeSample hides its subtree from then on:
        the child's window follows the INHERITED chain, not its local opinion alone."""
        path = _author_parent_gated_asset(self._tmp.name)
        anim = self._anim(path, "appearance")
        # Bar: on for the first half-cycle. Panel's local samples would say (0, 0.75),
        # but the bar hides it from 0.5 — the inherited window is clipped.
        self.assertEqual(anim.windows_by_state["sequence"]["Bar"], [(0.0, 0.5)])
        self.assertEqual(anim.windows_by_state["sequence"]["Bar/Panel"], [(0.0, 0.5)])

    def test_statically_invisible_ancestor_excludes_member(self) -> None:
        """A sample-less element under a statically invisible ANCESTOR is not a member of
        that state: one 'invisible' on the parent bar prunes the whole subtree, even for
        unregistered-schema payload prims where ComputeVisibility silently degrades."""
        path = _author_hidden_bar_asset(self._tmp.name)
        anim = self._anim(path, "appearance")
        self.assertEqual(anim.members_by_state, {"off": [], "sequence": ["Bar/Surface"]})
        self.assertEqual(anim.windows_by_state["sequence"], {"Bar/Surface": [(0.0, 0.5)]})

    def test_authored_selection_restored(self) -> None:
        """Reading the animation leaves the asset's authored variant selection untouched."""
        path = _author_asset(self._tmp.name, selection="off")
        keep = Usd.Stage.Open(path)  # share the layer so mutations would be visible
        self._anim(path)
        vset = keep.GetPrimAtPath("/Car").GetVariantSets().GetVariantSet("appearanceMode")
        self.assertEqual(vset.GetVariantSelection(), "off")

    def test_missing_selection_restored_to_empty(self) -> None:
        """An asset with no authored selection still reads fully and ends with an empty selection."""
        path = _author_asset(self._tmp.name, selection=None)
        keep = Usd.Stage.Open(path)
        anim = self._anim(path)
        self.assertTrue(anim.is_animated)
        vset = keep.GetPrimAtPath("/Car").GetVariantSets().GetVariantSet("appearanceMode")
        self.assertEqual(vset.GetVariantSelection(), "")

    def test_animated_without_timecode_range_raises(self) -> None:
        """An animated variantSet on a stage without a timeCode range raises VariantAnimationError."""
        path = _author_asset(self._tmp.name, timecodes=False)
        with self.assertRaises(VariantAnimationError):
            read_variant_animation(path, "appearanceMode")

    def test_bad_root_prim_path_raises(self) -> None:
        """An explicit root_prim_path that resolves to no prim raises VariantAnimationError."""
        path = _author_asset(self._tmp.name)
        with self.assertRaises(VariantAnimationError):
            read_variant_animation(path, "appearanceMode", root_prim_path="/DoesNotExist")

    def test_missing_default_prim_raises(self) -> None:
        """Without a defaultPrim (and no explicit root) the reader raises VariantAnimationError."""
        path = _author_asset(self._tmp.name, default_prim=False)
        with self.assertRaises(VariantAnimationError):
            read_variant_animation(path, "appearanceMode")

    def test_static_variant_set_is_not_animated(self) -> None:
        """A variantSet with no timeSamples reports is_animated False with empty per-state maps."""
        path = _author_asset(self._tmp.name, animated=False)
        anim = self._anim(path)
        self.assertFalse(anim.is_animated)
        self.assertEqual(anim.windows_by_state, {"off": {}, "sequence": {}})
        self.assertEqual(anim.gains_by_state, {"off": {}, "sequence": {}})
        self.assertEqual(anim.members_by_state, {})
        self.assertEqual(anim.cycle_s, 2.0)


if __name__ == "__main__":
    unittest.main()
