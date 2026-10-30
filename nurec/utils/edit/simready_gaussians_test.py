# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SimReady Gaussian-splat importer (:mod:`simready_gaussians`).

Standalone by construction: tiny USD stages carrying hand-authored
``ParticleField3DGaussianSplat`` prims are written into tempfiles with ``pxr``;
no checked-in scene data, no GPU, no torch.
"""

import tempfile
import unittest

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from nurec.utils.edit.simready_gaussians import (
    SimReadyGaussians,
    SimReadyImportError,
    _looks_like_gaussian_prim,
    _read_gaussian_prim,
    load_simready_asset,
    payloads_equal,
)


def _load(path: str) -> SimReadyGaussians:
    """Read the first authored splat prim through the module's single-prim reader.

    A thin stand-in for the production entry (``load_simready_asset``), which reads the
    same prims via the same ``_read_gaussian_prim`` but composes them per variant state;
    reading one prim directly keeps these conversion tests exact and fixture-cheap.
    """
    stage = Usd.Stage.Open(str(path))
    prim = next(p for p in stage.Traverse() if _looks_like_gaussian_prim(p))
    return _read_gaussian_prim(stage, prim, UsdGeom.XformCache(Usd.TimeCode.Default()))


# Post-activation reference arrays (all values exactly representable in float32).
_POS = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
_OPAC = np.array([0.0, 0.5, 1.0])
_SCALES = np.array([[0.5, 1.0, 2.0], [0.25, 0.5, 4.0], [1.0, 1.0, 1.0]])
_QUATS = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]])
_SH = np.arange(3 * 4 * 3, dtype=np.float64).reshape(3, 4, 3) * 0.125  # (N=3, K=4, 3)

_OPACITY_EPS = 1.0e-6  # keep in sync with the module's clamp


def _logit(opacities: np.ndarray) -> np.ndarray:
    """Computes the reference clamped logit the loader applies to opacities."""
    clamped = np.clip(opacities, _OPACITY_EPS, 1.0 - _OPACITY_EPS)
    return np.log(clamped) - np.log1p(-clamped)


def _author_splat(
    stage: Usd.Stage,
    path: str,
    positions: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    quats_wxyz: Optional[np.ndarray],
    sh: np.ndarray,
    half: bool = False,
    degree: Optional[int] = None,
    element_size: bool = True,
    element_size_override: Optional[int] = None,
    omit: Sequence[str] = (),
    typed: bool = True,
) -> Usd.Prim:
    """Author one ``ParticleField3DGaussianSplat`` prim with raw writer-style attrs.

    ``sh`` is ``(N, K, 3)`` and is flattened to ``(N*K, 3)`` gaussian-major with
    ``elementSize=K`` metadata. ``half=True`` authors only the ``*h`` half-precision
    attribute variants. ``omit`` skips attribute base names ("positions", "scales",
    "opacities", "sh") for missing-attribute tests. ``typed=False`` authors an
    untyped prim (the loader's attribute-based detection): environments where the
    NVIDIA schema IS registered inject schema-default ``sphericalHarmonicsDegree``/
    ``elementSize`` values on typed prims, so inference-path tests must stay untyped.
    """
    prim = stage.DefinePrim(path, "ParticleField3DGaussianSplat" if typed else "")
    suffix = "h" if half else ""
    point3_type = Sdf.ValueTypeNames.Point3hArray if half else Sdf.ValueTypeNames.Point3fArray
    vec3_type = Sdf.ValueTypeNames.Half3Array if half else Sdf.ValueTypeNames.Float3Array
    scalar_type = Sdf.ValueTypeNames.HalfArray if half else Sdf.ValueTypeNames.FloatArray
    quat_type = Sdf.ValueTypeNames.QuathArray if half else Sdf.ValueTypeNames.QuatfArray
    vec3 = Gf.Vec3h if half else Gf.Vec3f
    quat = Gf.Quath if half else Gf.Quatf
    vec3_array = Vt.Vec3hArray if half else Vt.Vec3fArray
    scalar_array = Vt.HalfArray if half else Vt.FloatArray
    quat_array = Vt.QuathArray if half else Vt.QuatfArray

    if "positions" not in omit:
        prim.CreateAttribute("positions" + suffix, point3_type, True).Set(
            vec3_array([vec3(float(p[0]), float(p[1]), float(p[2])) for p in positions])
        )
    if quats_wxyz is not None:
        prim.CreateAttribute("orientations" + suffix, quat_type, True).Set(
            quat_array([quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in quats_wxyz])
        )
    if "scales" not in omit:
        prim.CreateAttribute("scales" + suffix, vec3_type, True).Set(
            vec3_array([vec3(float(s[0]), float(s[1]), float(s[2])) for s in scales])
        )
    if "opacities" not in omit:
        prim.CreateAttribute("opacities" + suffix, scalar_type, True).Set(scalar_array([float(o) for o in opacities]))
    if "sh" not in omit:
        n, k = int(sh.shape[0]), int(sh.shape[1])
        flat = np.asarray(sh, dtype=np.float64).reshape(n * k, 3)
        attr = prim.CreateAttribute("radiance:sphericalHarmonicsCoefficients" + suffix, vec3_type, True)
        attr.Set(vec3_array([vec3(float(r[0]), float(r[1]), float(r[2])) for r in flat]))
        if element_size:
            attr.SetMetadata("elementSize", element_size_override if element_size_override is not None else k)
    if degree is not None:
        prim.CreateAttribute("radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int, True).Set(degree)
    return prim


def _author_default_splat(stage: Usd.Stage, path: str, opacities: Optional[np.ndarray] = None) -> Usd.Prim:
    """Authors a splat prim from the module reference arrays, optionally overriding opacities."""
    return _author_splat(stage, path, _POS, opacities if opacities is not None else _OPAC, _SCALES, _QUATS, _SH)


def _new_stage(path: str, meters_per_unit: float = 1.0) -> Usd.Stage:
    """Creates a new stage with a /Root default prim, Z up-axis, and the given metersPerUnit."""
    stage = Usd.Stage.CreateNew(path)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    return stage


def _author_vehicle(dir_path: str, *, authored_panel: Optional[str] = "off") -> str:
    """A composed vehicle: static Body under a translated rig, a payload-swapping
    ``panelState`` variantSet, a presence-swapping ``detailState`` variantSet, and a
    marked animation preview subtree that must be skipped."""
    path = str(Path(dir_path) / "vehicle.usda")
    stage = Usd.Stage.CreateNew(path)
    vehicle = UsdGeom.Xform.Define(stage, "/Vehicle")
    stage.SetDefaultPrim(vehicle.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    rig = UsdGeom.Xform.Define(stage, "/Vehicle/BodyRig")
    rig.AddTranslateOp().Set(Gf.Vec3d(10.0, 20.0, 30.0))
    _author_default_splat(stage, "/Vehicle/BodyRig/Body")

    panel = vehicle.GetPrim().GetVariantSets().AddVariantSet("panelState")
    panel.AddVariant("off")
    panel.AddVariant("on")
    for state, opacity in (("off", 0.25), ("on", 0.75)):
        panel.SetVariantSelection(state)
        with panel.GetVariantEditContext():
            _author_default_splat(stage, "/Vehicle/Panels", opacities=np.full(3, opacity))
    if authored_panel is None:
        panel.ClearVariantSelection()
    else:
        panel.SetVariantSelection(authored_panel)

    detail = vehicle.GetPrim().GetVariantSets().AddVariantSet("detailState")
    detail.AddVariant("off")
    detail.AddVariant("on")
    detail.SetVariantSelection("on")
    with detail.GetVariantEditContext():
        _author_default_splat(stage, "/Vehicle/Detail")
    detail.SetVariantSelection("off")

    preview = UsdGeom.Xform.Define(stage, "/Vehicle/Preview")
    preview.GetPrim().CreateAttribute("nurec:lightAnimation", Sdf.ValueTypeNames.Bool, True).Set(True)
    _author_default_splat(stage, "/Vehicle/Preview/Surface")

    stage.GetRootLayer().Save()
    return path


def _author_conflict(dir_path: str) -> str:
    """/Car/X is payload-swapped by ``setA`` AND is an animated-visibility element of
    ``setB`` (whose over on ``visibility`` is not shadowed by setA's defs)."""
    path = str(Path(dir_path) / "conflict.usda")
    stage = Usd.Stage.CreateNew(path)
    car = UsdGeom.Xform.Define(stage, "/Car")
    stage.SetDefaultPrim(car.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    set_a = car.GetPrim().GetVariantSets().AddVariantSet("setA")
    set_a.AddVariant("s1")
    set_a.AddVariant("s2")
    for state, opacity in (("s1", 0.25), ("s2", 0.75)):
        set_a.SetVariantSelection(state)
        with set_a.GetVariantEditContext():
            _author_default_splat(stage, "/Car/X", opacities=np.full(3, opacity))
    set_a.SetVariantSelection("s1")
    set_b = car.GetPrim().GetVariantSets().AddVariantSet("setB")
    set_b.AddVariant("t1")
    set_b.AddVariant("t2")
    set_b.SetVariantSelection("t2")
    with set_b.GetVariantEditContext():
        over = stage.OverridePrim("/Car/X")
        vis = over.CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False)
        vis.Set("inherited", Usd.TimeCode(0))
        vis.Set("invisible", Usd.TimeCode(24))
    set_b.SetVariantSelection("t1")
    stage.GetRootLayer().Save()
    return path


def _author_visibility_asset(dir_path: str) -> str:
    """A root with four splats exercising static-visibility handling: a plain visible
    one, an authored-invisible one, one under an invisible Xform, and one whose
    invisibility is TIME-SAMPLED (an animation element that must stay in)."""
    path = str(Path(dir_path) / "visibility.usda")
    stage = Usd.Stage.CreateNew(path)
    root = UsdGeom.Xform.Define(stage, "/Board")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    _author_default_splat(stage, "/Board/Body")

    off_panel = _author_default_splat(stage, "/Board/PanelOn")
    off_panel.CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False).Set("invisible")

    hidden_rig = UsdGeom.Xform.Define(stage, "/Board/HiddenRig")
    hidden_rig.GetPrim().CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False).Set("invisible")
    _author_default_splat(stage, "/Board/HiddenRig/Surface")

    animated = _author_default_splat(stage, "/Board/Cycle")
    vis = animated.CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False)
    vis.Set("invisible", Usd.TimeCode(0))
    vis.Set("inherited", Usd.TimeCode(12))

    stage.GetRootLayer().Save()
    return path


class StaticVisibilityTest(unittest.TestCase):
    """Authored static visibility prunes hidden dressing; time-sampled visibility never does.

    This is what lets a controls-free bake of an on/off asset come out empty: the
    composed state's ``invisible`` payload halves (e.g. alternate panel splats while ``off``
    is selected) must not bake as visible static parts.
    """

    def test_statically_invisible_prims_are_skipped(self) -> None:
        """Authored-invisible splats and splats under invisible ancestors do not load."""
        with tempfile.TemporaryDirectory() as tmp:
            asset = load_simready_asset(_author_visibility_asset(tmp))
        paths = sorted(p.prim_path for p in asset.static_parts)
        self.assertNotIn("/Board/PanelOn", paths)
        self.assertNotIn("/Board/HiddenRig/Surface", paths)
        self.assertIn("/Board/Body", paths)

    def test_time_sampled_visibility_is_kept(self) -> None:
        """A splat with visibility timeSamples is an animation element, never pruned."""
        with tempfile.TemporaryDirectory() as tmp:
            asset = load_simready_asset(_author_visibility_asset(tmp))
        self.assertIn("/Board/Cycle", [p.prim_path for p in asset.static_parts])

    def test_all_invisible_asset_fails_loud(self) -> None:
        """An asset whose every splat is statically hidden fails rather than baking nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "hidden.usda")
            stage = Usd.Stage.CreateNew(path)
            root = UsdGeom.Xform.Define(stage, "/HiddenAsset")
            stage.SetDefaultPrim(root.GetPrim())
            prim = _author_default_splat(stage, "/HiddenAsset/Hidden")
            prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False).Set("invisible")
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(SimReadyImportError, "no Gaussian-bearing prims"):
                load_simready_asset(path)

    def test_all_hidden_state_imports_with_empty_parts(self) -> None:
        """A control state that hides EVERY splat is legitimate authoring (an explicit
        all-hidden 'off'): the import succeeds and that state's part list is empty."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "hidden_state.usda")
            stage = Usd.Stage.CreateNew(path)
            root = UsdGeom.Xform.Define(stage, "/Panel")
            stage.SetDefaultPrim(root.GetPrim())
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            _author_default_splat(stage, "/Panel/Surface")
            power = root.GetPrim().GetVariantSets().AddVariantSet("power")
            power.AddVariant("off")
            power.AddVariant("on")
            power.SetVariantSelection("off")
            with power.GetVariantEditContext():
                over = stage.OverridePrim("/Panel/Surface")
                over.CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False).Set("invisible")
            stage.GetRootLayer().Save()
            asset = load_simready_asset(path, control_sets=["power"])
        control = asset.controls["power"]
        self.assertEqual(control.parts_by_state["off"], [])
        self.assertEqual([p.prim_path for p in control.parts_by_state["on"]], ["/Panel/Surface"])
        # The authored baseline is the all-hidden state, so nothing bakes statically.
        self.assertEqual(asset.static_parts, [])

    def test_variant_authored_invisibility_prunes_per_state(self) -> None:
        """A panel whose 'on' half is invisible in the off STATE (and vice versa) loads
        only the selected state's visible half — the on/off pair never bakes together."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "panel.usda")
            stage = Usd.Stage.CreateNew(path)
            root = UsdGeom.Xform.Define(stage, "/Panel")
            stage.SetDefaultPrim(root.GetPrim())
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            on_half = _author_default_splat(stage, "/Panel/on")
            off_half = _author_default_splat(stage, "/Panel/off")
            power = root.GetPrim().GetVariantSets().AddVariantSet("power")
            power.AddVariant("off")
            power.AddVariant("on")
            for state, hidden_path in (("off", "/Panel/on"), ("on", "/Panel/off")):
                power.SetVariantSelection(state)
                with power.GetVariantEditContext():
                    over = stage.OverridePrim(hidden_path)
                    over.CreateAttribute("visibility", Sdf.ValueTypeNames.Token, False).Set("invisible")
            power.SetVariantSelection("off")
            stage.GetRootLayer().Save()
            del on_half, off_half

            # No controls: the authored 'off' composition bakes only the off half.
            asset = load_simready_asset(path)
            self.assertEqual([p.prim_path for p in asset.static_parts], ["/Panel/off"])

            # With the control expanded, each state carries exactly its own half.
            asset = load_simready_asset(path, control_sets=["power"])
            control = asset.controls["power"]
            self.assertEqual([p.prim_path for p in control.parts_by_state["off"]], ["/Panel/off"])
            self.assertEqual([p.prim_path for p in control.parts_by_state["on"]], ["/Panel/on"])


class LoadSimReadyGaussiansTest(unittest.TestCase):
    """Behavioral tests for the single-prim reader on hand-authored splat prims."""

    def setUp(self) -> None:
        """Creates a per-test temporary directory for authored USD files."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _path(self, name: str = "splat.usda") -> str:
        """Returns an absolute path for a USD file inside the test's temporary directory."""
        return str(Path(self._tmp.name) / name)

    def test_pre_activation_conversion(self) -> None:
        """Post-activation attrs convert to pre-activation params: log scales and finite clamped-logit opacities."""
        stage = _new_stage(self._path())
        _author_default_splat(stage, "/Root/Splat")
        stage.GetRootLayer().Save()
        g = _load(self._path())
        self.assertEqual(g.count, 3)
        np.testing.assert_allclose(g.positions, _POS)
        np.testing.assert_allclose(g.scales_log, np.log(_SCALES))
        np.testing.assert_allclose(g.densities_logit, _logit(_OPAC).reshape(3, 1))
        # Opacity endpoints 0.0 and 1.0 are clamped, not +-inf.
        self.assertTrue(np.isfinite(g.densities_logit).all())
        self.assertEqual(g.densities_logit.shape, (3, 1))
        self.assertEqual(g.up_axis, "Z")
        self.assertEqual(g.meters_per_unit, 1.0)

    def test_quaternion_wxyz_round_trip(self) -> None:
        """Orientations authored as Gf.Quat come back as (w, x, y, z) rows unchanged."""
        stage = _new_stage(self._path())
        _author_default_splat(stage, "/Root/Splat")
        stage.GetRootLayer().Save()
        g = _load(self._path())
        # Authored as Gf.Quat(w, x, y, z); reader returns (w, x, y, z) rows.
        np.testing.assert_allclose(g.rotations, _QUATS, atol=1e-7)

    def test_sh_split_and_degree_attr(self) -> None:
        """Flattened SH coefficients split into sh_dc (band 0) and sh_rest, with degree from the attribute."""
        stage = _new_stage(self._path())
        _author_splat(stage, "/Root/Splat", _POS, _OPAC, _SCALES, _QUATS, _SH, degree=1)
        stage.GetRootLayer().Save()
        g = _load(self._path())
        self.assertEqual(g.sh_degree, 1)
        np.testing.assert_allclose(g.sh_dc, _SH[:, 0, :])
        self.assertEqual(g.sh_rest.shape, (3, 3, 3))
        np.testing.assert_allclose(g.sh_rest, _SH[:, 1:, :])

    def test_sh_degree_from_element_size_when_degree_attr_missing(self) -> None:
        """Without a degree attribute, sh_degree is inferred from the elementSize metadata (sqrt(K) - 1)."""
        stage = _new_stage(self._path())
        # Untyped prim: a registered schema would inject a default degree attribute,
        # hiding the elementSize inference under test.
        _author_splat(stage, "/Root/Splat", _POS, _OPAC, _SCALES, _QUATS, _SH, typed=False)
        stage.GetRootLayer().Save()
        g = _load(self._path())
        self.assertEqual(g.sh_degree, 1)  # round(sqrt(4)) - 1

    def test_sh_count_from_degree_attr_without_element_size(self) -> None:
        """Without elementSize, K comes from the degree attribute; an inconsistent degree raises."""
        stage = _new_stage(self._path())
        _author_splat(stage, "/Root/Splat", _POS, _OPAC, _SCALES, _QUATS, _SH, degree=1, element_size=False)
        stage.GetRootLayer().Save()
        g = _load(self._path())
        self.assertEqual(g.sh_rest.shape, (3, 3, 3))  # K = (1 + 1)^2 = 4
        # A degree inconsistent with the coefficient count raises.
        stage2 = _new_stage(self._path("bad_degree.usda"))
        _author_splat(stage2, "/Root/Splat", _POS, _OPAC, _SCALES, _QUATS, _SH, degree=2, element_size=False)
        stage2.GetRootLayer().Save()
        with self.assertRaises(SimReadyImportError):
            _load(self._path("bad_degree.usda"))

    def test_sh_count_by_divisibility_fallback(self) -> None:
        """With neither degree nor elementSize, K falls back to coefficient count divided by gaussian count."""
        stage = _new_stage(self._path())
        # Untyped prim: a registered schema would inject default degree/elementSize,
        # hiding the divisibility fallback under test.
        _author_splat(stage, "/Root/Splat", _POS, _OPAC, _SCALES, _QUATS, _SH, element_size=False, typed=False)
        stage.GetRootLayer().Save()
        g = _load(self._path())
        self.assertEqual(g.sh_rest.shape, (3, 3, 3))  # K = 12 coeffs / 3 gaussians
        self.assertEqual(g.sh_degree, 1)

    def test_sh_element_size_mismatch_raises(self) -> None:
        """An elementSize that disagrees with the coefficient count raises SimReadyImportError."""
        stage = _new_stage(self._path())
        _author_splat(stage, "/Root/Splat", _POS, _OPAC, _SCALES, _QUATS, _SH, element_size_override=3)
        stage.GetRootLayer().Save()
        with self.assertRaises(SimReadyImportError):
            _load(self._path())

    def test_sh_indivisible_count_raises(self) -> None:
        """A coefficient count not divisible by the gaussian count (no elementSize) raises SimReadyImportError."""
        stage = _new_stage(self._path())
        prim = _author_splat(stage, "/Root/Splat", _POS, _OPAC, _SCALES, _QUATS, _SH, omit=("sh",))
        attr = prim.CreateAttribute("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array, True)
        attr.Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)] * 7))  # 7 % 3 != 0, no elementSize
        stage.GetRootLayer().Save()
        with self.assertRaises(SimReadyImportError):
            _load(self._path())

    def test_half_precision_asset(self) -> None:
        """An asset authored only with half-precision *h attribute variants loads to the same values."""
        positions = np.array([[0.5, 1.5, -2.0], [0.25, 0.0, 1.0]])
        opacities = np.array([0.5, 0.25])
        scales = np.array([[0.5, 0.25, 1.0], [2.0, 1.0, 0.5]])
        quats = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        sh = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3) * 0.125
        stage = _new_stage(self._path())
        _author_splat(stage, "/Root/Splat", positions, opacities, scales, quats, sh, half=True)
        stage.GetRootLayer().Save()
        g = _load(self._path())
        self.assertEqual(g.count, 2)
        np.testing.assert_allclose(g.positions, positions, atol=1e-2)
        np.testing.assert_allclose(g.rotations, quats, atol=1e-2)
        np.testing.assert_allclose(g.scales_log, np.log(scales), atol=1e-2)
        np.testing.assert_allclose(g.densities_logit, _logit(opacities).reshape(2, 1), atol=1e-2)
        np.testing.assert_allclose(g.sh_dc, sh[:, 0, :], atol=1e-2)
        np.testing.assert_allclose(g.sh_rest, sh[:, 1:, :], atol=1e-2)

    def test_meters_per_unit_folds_into_positions_and_scales(self) -> None:
        """Stage metersPerUnit scales positions and (log) scales into metres on load."""
        stage = _new_stage(self._path(), meters_per_unit=0.01)
        _author_default_splat(stage, "/Root/Splat")
        stage.GetRootLayer().Save()
        g = _load(self._path())
        self.assertAlmostEqual(g.meters_per_unit, 0.01)
        np.testing.assert_allclose(g.positions, _POS * 0.01)
        np.testing.assert_allclose(g.scales_log, np.log(_SCALES * 0.01))

    def test_ancestor_transform_folds_into_root_frame(self) -> None:
        """Ancestor Xform ops are baked into positions and rotations, expressing gaussians in the root frame."""
        stage = _new_stage(self._path())
        rig = UsdGeom.Xform.Define(stage, "/Root/Rig")
        rig.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        rig.AddRotateZOp().Set(90.0)
        positions = np.array([[1.0, 0.0, 0.0]])
        quats = np.array([[1.0, 0.0, 0.0, 0.0]])
        _author_splat(stage, "/Root/Rig/Splat", positions, np.array([0.5]), np.ones((1, 3)), quats, _SH[:1])
        stage.GetRootLayer().Save()
        g = _load(self._path())
        # (1, 0, 0) rotated 90 deg about Z then translated by (1, 2, 3).
        np.testing.assert_allclose(g.positions, [[1.0, 3.0, 3.0]], atol=1e-6)
        expected = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])  # Z-rotation premultiplied
        self.assertTrue(
            np.allclose(g.rotations[0], expected, atol=1e-6) or np.allclose(g.rotations[0], -expected, atol=1e-6)
        )
        np.testing.assert_allclose(g.scales_log, np.zeros((1, 3)), atol=1e-6)  # uniform scale 1

    def test_time_sampled_ancestor_xform_op_raises(self) -> None:
        """A pose authored only as xformOp timeSamples fails loud: Default-time resolution
        ignores samples, so the splat would silently import at its rest/identity pose."""
        stage = _new_stage(self._path())
        rig = UsdGeom.Xform.Define(stage, "/Root/Rig")
        op = rig.AddTranslateOp()
        op.Set(Gf.Vec3d(0.0, 0.0, 0.0), Usd.TimeCode(0))
        op.Set(Gf.Vec3d(5.0, 0.0, 0.0), Usd.TimeCode(24))
        _author_default_splat(stage, "/Root/Rig/Splat")
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(SimReadyImportError, "time-sampled xformOp"):
            _load(self._path())

    def test_non_uniform_ancestor_scale_raises(self) -> None:
        """A non-uniform ancestor scale raises: per-gaussian scales cannot follow it, so
        folding it would knowingly distort the splats behind a scrolling warning."""
        stage = _new_stage(self._path())
        rig = UsdGeom.Xform.Define(stage, "/Root/Rig")
        rig.AddScaleOp().Set(Gf.Vec3f(1.0, 2.0, 3.0))
        _author_default_splat(stage, "/Root/Rig/Splat")
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(SimReadyImportError, "non-uniform scale"):
            _load(self._path())

    def test_uniform_ancestor_scale_folds_into_scales(self) -> None:
        """A uniform ancestor scale folds into positions AND per-gaussian scales."""
        stage = _new_stage(self._path())
        rig = UsdGeom.Xform.Define(stage, "/Root/Rig")
        rig.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 2.0))
        _author_default_splat(stage, "/Root/Rig/Splat")
        stage.GetRootLayer().Save()
        g = _load(self._path())
        np.testing.assert_allclose(g.positions, _POS * 2.0, atol=1e-6)
        np.testing.assert_allclose(g.scales_log, np.log(_SCALES * 2.0), atol=1e-6)

    def test_missing_required_attributes_raise(self) -> None:
        """Omitting any of positions, scales, opacities, or SH coefficients raises SimReadyImportError."""
        for missing in ("positions", "scales", "opacities", "sh"):
            with self.subTest(missing=missing):
                path = self._path(f"missing_{missing}.usda")
                stage = _new_stage(path)
                _author_splat(stage, "/Root/Splat", _POS, _OPAC, _SCALES, _QUATS, _SH, omit=(missing,))
                stage.GetRootLayer().Save()
                with self.assertRaises(SimReadyImportError):
                    _load(path)

    def test_missing_orientations_fall_back_to_identity(self) -> None:
        """A splat without an orientations attribute loads with identity quaternions for every gaussian."""
        stage = _new_stage(self._path())
        _author_splat(stage, "/Root/Splat", _POS, _OPAC, _SCALES, None, _SH)
        stage.GetRootLayer().Save()
        g = _load(self._path())
        np.testing.assert_array_equal(g.rotations, np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1)))


class LoadSimReadyAssetTest(unittest.TestCase):
    """Behavioral tests for load_simready_asset: part discovery, variant controls, and selection restore."""

    def setUp(self) -> None:
        """Creates a per-test temporary directory for authored USD files."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_static_part_folded_into_root_frame(self) -> None:
        """Non-variant splats become static parts with rig transforms folded in, plus correct stage metadata."""
        path = _author_vehicle(self._tmp.name)
        vehicle = load_simready_asset(path, control_sets=["panelState", "detailState"])
        self.assertEqual([p.prim_path for p in vehicle.static_parts], ["/Vehicle/BodyRig/Body"])
        np.testing.assert_allclose(
            vehicle.static_parts[0].gaussians.positions, _POS + np.array([10.0, 20.0, 30.0]), atol=1e-6
        )
        self.assertEqual(vehicle.root_prim_path, "/Vehicle")
        self.assertEqual(vehicle.up_axis, "Z")
        self.assertEqual(vehicle.meters_per_unit, 1.0)
        self.assertEqual(vehicle.variant_sets, {"panelState": ["off", "on"], "detailState": ["off", "on"]})

    def test_root_prim_transform_divided_out(self) -> None:
        """Parts come out relative to the ROOT prim: the root's own authored transform and
        everything above a nested root_prim_path must not leak into part positions."""
        path = str(Path(self._tmp.name) / "posed.usda")
        stage = Usd.Stage.CreateNew(path)
        world = UsdGeom.Xform.Define(stage, "/World")
        world.AddTranslateOp().Set(Gf.Vec3d(100.0, 0.0, 0.0))
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        asset_root = UsdGeom.Xform.Define(stage, "/World/Asset")
        asset_root.AddTranslateOp().Set(Gf.Vec3d(0.0, 50.0, 0.0))
        rig = UsdGeom.Xform.Define(stage, "/World/Asset/Rig")
        rig.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        _author_default_splat(stage, "/World/Asset/Rig/Body")
        stage.GetRootLayer().Save()
        asset = load_simready_asset(path, root_prim_path="/World/Asset")
        # Only the rig translate (below the root prim) survives; /World and the root's
        # own op are divided out.
        np.testing.assert_allclose(
            asset.static_parts[0].gaussians.positions, _POS + np.array([1.0, 2.0, 3.0]), atol=1e-6
        )

    def test_time_sampled_xform_on_root_prim_tolerated(self) -> None:
        """timeSamples on the ROOT prim's own xformOps (an animated asset pose) cancel
        exactly in the root-relative frame: parts import at their authored layout."""
        path = str(Path(self._tmp.name) / "posed_root.usda")
        stage = Usd.Stage.CreateNew(path)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        op = root.AddTranslateOp()
        op.Set(Gf.Vec3d(0.0, 0.0, 0.0), Usd.TimeCode(0))
        op.Set(Gf.Vec3d(7.0, 0.0, 0.0), Usd.TimeCode(24))
        _author_default_splat(stage, "/Asset/Body")
        stage.GetRootLayer().Save()
        asset = load_simready_asset(path)
        np.testing.assert_allclose(asset.static_parts[0].gaussians.positions, _POS, atol=1e-6)

    def test_time_sampled_xform_below_root_raises(self) -> None:
        """A time-sampled xformOp BELOW the root (a rigged part) raises: Default-time
        resolution would silently bake the part at its rest/identity pose."""
        path = str(Path(self._tmp.name) / "rigged.usda")
        stage = Usd.Stage.CreateNew(path)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        rig = UsdGeom.Xform.Define(stage, "/Asset/Rig")
        op = rig.AddTranslateOp()
        op.Set(Gf.Vec3d(0.0, 0.0, 0.0), Usd.TimeCode(0))
        op.Set(Gf.Vec3d(5.0, 0.0, 0.0), Usd.TimeCode(24))
        _author_default_splat(stage, "/Asset/Rig/Body")
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(SimReadyImportError, "time-sampled xformOp"):
            load_simready_asset(path)

    def test_panel_control_swaps_payload(self) -> None:
        """A prim present in both states with differing payloads is a control part, not a static part."""
        path = _author_vehicle(self._tmp.name)
        vehicle = load_simready_asset(path, control_sets=["panelState", "detailState"])
        control = vehicle.controls["panelState"]
        self.assertEqual(control.states, ["off", "on"])
        self.assertEqual([p.prim_path for p in control.parts_by_state["off"]], ["/Vehicle/Panels"])
        self.assertEqual([p.prim_path for p in control.parts_by_state["on"]], ["/Vehicle/Panels"])
        off = control.parts_by_state["off"][0].gaussians
        on = control.parts_by_state["on"][0].gaussians
        self.assertFalse(payloads_equal(off, on))
        self.assertFalse(np.array_equal(off.densities_logit, on.densities_logit))
        # The swapped prim must not also be a static part.
        self.assertNotIn("/Vehicle/Panels", [p.prim_path for p in vehicle.static_parts])

    def test_detail_present_only_in_on_state(self) -> None:
        """A prim defined inside only one variant appears in that state's part list and nowhere else."""
        path = _author_vehicle(self._tmp.name)
        vehicle = load_simready_asset(path, control_sets=["panelState", "detailState"])
        control = vehicle.controls["detailState"]
        self.assertEqual(control.parts_by_state["off"], [])
        self.assertEqual([p.prim_path for p in control.parts_by_state["on"]], ["/Vehicle/Detail"])

    def test_preview_subtree_skipped(self) -> None:
        """Splats under a marked animation preview subtree are excluded from every part list."""
        path = _author_vehicle(self._tmp.name)
        vehicle = load_simready_asset(path, control_sets=["panelState", "detailState"])
        everywhere = [p.prim_path for p in vehicle.static_parts]
        for control in vehicle.controls.values():
            for parts in control.parts_by_state.values():
                everywhere += [p.prim_path for p in parts]
        self.assertNotIn("/Vehicle/Preview/Surface", everywhere)

    def test_default_state_matches_authored_selection(self) -> None:
        """Each control's default_state reflects the variant selection authored in the asset."""
        path = _author_vehicle(self._tmp.name, authored_panel="on")
        vehicle = load_simready_asset(path, control_sets=["panelState", "detailState"])
        self.assertEqual(vehicle.controls["panelState"].default_state, "on")
        self.assertEqual(vehicle.controls["detailState"].default_state, "off")

    def test_case_insensitive_control_set_resolution(self) -> None:
        """A differently-cased control-set name resolves to the canonical authored variantSet name."""
        path = _author_vehicle(self._tmp.name)
        vehicle = load_simready_asset(path, control_sets=["panelstate"])
        self.assertEqual(list(vehicle.controls), ["panelState"])
        self.assertEqual(vehicle.controls["panelState"].variant_set, "panelState")

    def test_unknown_control_set_raises(self) -> None:
        """Requesting a control set the asset does not define raises SimReadyImportError."""
        path = _author_vehicle(self._tmp.name)
        with self.assertRaises(SimReadyImportError):
            load_simready_asset(path, control_sets=["noSuchControl"])

    def test_prim_swapped_by_two_variant_sets_raises(self) -> None:
        """A prim whose payload varies under two different variantSets is rejected with a clear error."""
        path = _author_conflict(self._tmp.name)
        with self.assertRaises(SimReadyImportError) as ctx:
            load_simready_asset(path, control_sets=["setA", "setB"])
        self.assertIn("more than one variantSet", str(ctx.exception))

    def test_authored_selections_restored_after_load(self) -> None:
        """Loading iterates variants but leaves the authored selections on the shared layer intact."""
        path = _author_vehicle(self._tmp.name)
        keep = Usd.Stage.Open(path)  # share the layer so mutations would be visible
        load_simready_asset(path, control_sets=["panelState", "detailState"])
        vsets = keep.GetPrimAtPath("/Vehicle").GetVariantSets()
        self.assertEqual(vsets.GetVariantSet("panelState").GetVariantSelection(), "off")
        self.assertEqual(vsets.GetVariantSet("detailState").GetVariantSelection(), "off")

    def test_unauthored_selection_cleared_after_load(self) -> None:
        """A set with no authored selection defaults to its first state and ends up cleared, not pinned."""
        # With NO authored selection the loader must clear its in-memory selection
        # after iterating, leaving the shared layer exactly as authored (empty).
        path = _author_vehicle(self._tmp.name, authored_panel=None)
        keep = Usd.Stage.Open(path)
        vehicle = load_simready_asset(path, control_sets=["panelState"])
        self.assertEqual(vehicle.controls["panelState"].default_state, "off")  # first state
        vsets = keep.GetPrimAtPath("/Vehicle").GetVariantSets()
        self.assertEqual(vsets.GetVariantSet("panelState").GetVariantSelection(), "")


def _payload(n: int = 3, **over: np.ndarray) -> SimReadyGaussians:
    """Builds a deterministic n-gaussian SimReadyGaussians, with keyword overrides for individual fields."""
    fields: dict[str, np.ndarray] = {
        "positions": np.arange(n * 3, dtype=np.float64).reshape(n, 3),
        "rotations": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        "scales_log": np.zeros((n, 3)),
        "densities_logit": np.zeros((n, 1)),
        "sh_dc": np.ones((n, 3)),
        "sh_rest": np.zeros((n, 3, 3)),
    }
    fields.update(over)
    return SimReadyGaussians(
        positions=fields["positions"],
        rotations=fields["rotations"],
        scales_log=fields["scales_log"],
        densities_logit=fields["densities_logit"],
        sh_dc=fields["sh_dc"],
        sh_rest=fields["sh_rest"],
        up_axis="Z",
        meters_per_unit=1.0,
        sh_degree=1,
    )


class PayloadsEqualTest(unittest.TestCase):
    """Behavioral tests for payloads_equal comparing SimReadyGaussians field-by-field."""

    _FIELDS = ("positions", "rotations", "scales_log", "densities_logit", "sh_dc", "sh_rest")

    def test_equal_payloads(self) -> None:
        """Two independently built identical payloads compare equal."""
        self.assertTrue(payloads_equal(_payload(), _payload()))

    def test_each_field_flip_breaks_equality(self) -> None:
        """Perturbing any single array field makes payloads_equal return False."""
        for name in self._FIELDS:
            with self.subTest(field=name):
                flipped = _payload()
                mutated = getattr(flipped, name).copy()
                mutated.flat[0] += 1.0
                setattr(flipped, name, mutated)
                self.assertFalse(payloads_equal(_payload(), flipped))

    def test_count_mismatch(self) -> None:
        """Payloads with different gaussian counts are unequal."""
        self.assertFalse(payloads_equal(_payload(n=2), _payload(n=3)))


if __name__ == "__main__":
    unittest.main()
