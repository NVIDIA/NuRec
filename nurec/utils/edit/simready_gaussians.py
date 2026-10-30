# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importer for SimReady OpenUSD Gaussian-splat assets (asset-edits bake).

This inverts NuRec's own ``GaussianLightFieldWriter`` (``gaussian_usd_writer.py``):
a SimReady asset's splat payloads ship as a single ``ParticleField3DGaussianSplat``
prim (a ``UsdVol`` schema) carrying ``positions``, ``orientations``, ``scales``,
``opacities`` and ``radiance:sphericalHarmonicsCoefficients`` (+ degree). Those are
**post-activation** values (linear-metre scales, ``[0, 1]`` opacities) and raw SH
coefficients in the same convention as NuRec's ``features_albedo`` (DC) /
``features_specular`` (higher orders).

:class:`SimReadyGaussians` carries one such prim's payload as **pre-activation**
numpy arrays ready for the ``compile-asset-edits`` compiler to transform into a
bound track's cuboid-local frame and append to the layer tensors:

* ``positions``       ``(N, 3)``  asset-local metres
* ``rotations``       ``(N, 4)``  unit quaternion ``(w, x, y, z)`` (NuRec convention)
* ``scales_log``      ``(N, 3)``  ``log`` of the linear scales (``exp`` activation inverse)
* ``densities_logit`` ``(N, 1)``  ``logit`` of the opacities (``sigmoid`` activation inverse)
* ``sh_dc``           ``(N, 3)``  SH DC coefficient (-> ``features_albedo`` DC)
* ``sh_rest``         ``(N, K-1, 3)`` higher-order SH (-> ``features_specular``)

:func:`load_simready_asset` reads a **composed multi-prim** SimReady asset (e.g. a
configurable prop: body + frame + panels, each its own ``ParticleField3DGaussianSplat``
prim, nested under transformed Xforms). Every part is returned in the **root prim's
frame**: ancestor transforms are folded via one shared ``XformCache`` per import pass
and the root prim's own local-to-world transform is divided out (identity in the
common case), so a ``root_prim_path`` nested under transformed ancestors still yields
root-relative parts. The asset's
**variantSets are its native control surface**: each requested set is expanded into
per-state Gaussian clusters (the prims whose payload the variant swaps), while prims
unaffected by any control are returned once as **static parts**.

The module is intentionally pure (``pxr`` + ``numpy`` only, no ``torch``) so it can
be unit-tested cheaply and reused by the compiler.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from pxr import Gf, Usd, UsdGeom


log = logging.getLogger(__name__)

# Typed prim schema NuRec / Kit emit for 3DGS LightField assets.
PARTICLE_FIELD_TYPE = "ParticleField3DGaussianSplat"

# Candidate attribute names: float first, then the half-precision (``*h``) variant the
# writer emits when ``half_geometry`` / ``half_features`` is set.
_POSITIONS_ATTRS = ("positions", "positionsh")
_ORIENTATIONS_ATTRS = ("orientations", "orientationsh")
_SCALES_ATTRS = ("scales", "scalesh")
_OPACITIES_ATTRS = ("opacities", "opacitiesh")
_SH_COEFFS_ATTRS = (
    "radiance:sphericalHarmonicsCoefficients",
    "radiance:sphericalHarmonicsCoefficientsh",
)
_SH_DEGREE_ATTR = "radiance:sphericalHarmonicsDegree"

# Clamp opacities away from the open interval ends so ``logit`` stays finite.
_OPACITY_EPS = 1.0e-6
# Floor scales before ``log`` so a degenerate zero scale does not become ``-inf``.
_SCALE_EPS = 1.0e-9

# Marker for USD-native animation PREVIEW clips an asset may author. Their
# element prims reference the same Gaussian payloads the variantSets swap — a playable
# preview representation, not extra geometry — so the importer must skip those subtrees
# or the payloads would bake twice (once via the variants, once as bogus static parts).
_ANIMATION_PREVIEW_ATTR = "nurec:lightAnimation"


def in_animation_preview_subtree(prim: Usd.Prim) -> bool:
    """True when ``prim`` sits under a marked animation preview subtree (skipped)."""
    p = prim
    while p and p.IsValid():
        attr = p.GetAttribute(_ANIMATION_PREVIEW_ATTR)
        if attr and attr.Get():
            return True
        p = p.GetParent()
    return False


def _classify_visibility(prim: Usd.Prim, root: Usd.Prim) -> tuple[bool, bool]:
    """One ancestor-chain walk classifying a prim's visibility: ``(animated, hidden)``.

    ``animated``: some level up to the asset root carries ``visibility`` timeSamples —
    the prim is a variant ANIMATION element the schedule machinery owns, even when its
    Gaussian payload is byte-identical across states (a static bake would leave it
    permanently visible instead of following the schedule).

    ``hidden``: a STATIC ``invisible`` is authored somewhere on the chain. USD visibility
    is pruning, so one opinion hides the subtree; such prims are the composed state's
    hidden dressing (e.g. the ``on`` panel halves while ``off`` is selected) and must not
    bake. The animation exemption is per LEVEL: a sampled level contributes no static
    hide, but does not excuse a static ``invisible`` authored further down — a hidden panel
    nested under an animated assembly is still statically hidden in this state.

    Reads the raw ``visibility`` attribute instead of
    ``UsdGeom.Imageable.ComputeVisibility()``: with the ParticleField schema not
    registered (any stock pxr build), computed visibility silently degrades to
    ``inherited`` on authored-invisible prims.
    """
    animated = hidden = False
    p = prim
    while p and p.IsValid():
        attr = p.GetAttribute("visibility")
        if attr:
            if attr.GetNumTimeSamples() > 0:
                animated = True  # schedule-owned at this level; no static hide from it
            elif attr.HasAuthoredValue() and str(attr.Get()) == "invisible":
                hidden = True
        if p.GetPath() == root.GetPath():
            break
        p = p.GetParent()
    return animated, hidden


def open_stage_root(
    usd_path: str, root_prim_path: Optional[str], error_cls: Optional[type] = None
) -> tuple[Usd.Stage, Usd.Prim]:
    """Open a USD stage and resolve its root prim (explicit path, else defaultPrim).

    Shared by the geometry importer and the variant-animation extractor so the two
    resolve assets identically. ``error_cls`` selects the caller's domain error.
    """
    err = error_cls or SimReadyImportError
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise err(f"could not open USD stage: {usd_path}")
    if root_prim_path:
        root = stage.GetPrimAtPath(root_prim_path)
        if not root or not root.IsValid():
            raise err(f"root prim {root_prim_path!r} not found in {usd_path}")
    else:
        root = stage.GetDefaultPrim()
        if not root or not root.IsValid():
            raise err(f"{usd_path} has no defaultPrim; pass root_prim_path explicitly")
    return stage, root


class SimReadyImportError(ValueError):
    """Raised when a SimReady USD asset cannot be parsed as a Gaussian payload."""


@dataclass
class SimReadyGaussians:
    """Pre-activation Gaussian attributes imported from a SimReady USD asset."""

    positions: np.ndarray  # (N, 3) float64, asset-local metres
    rotations: np.ndarray  # (N, 4) float64, unit quaternion (w, x, y, z)
    scales_log: np.ndarray  # (N, 3) float64, log of linear scales
    densities_logit: np.ndarray  # (N, 1) float64, logit of opacities
    sh_dc: np.ndarray  # (N, 3) float64, SH DC coefficient
    sh_rest: np.ndarray  # (N, K-1, 3) float64, higher-order SH coefficients
    up_axis: str  # stage up axis ("Z" or "Y")
    meters_per_unit: float  # stage metersPerUnit (already folded into positions/scales)
    sh_degree: int  # SH degree of the asset

    @property
    def count(self) -> int:
        """Number of Gaussians in this payload."""
        return int(self.positions.shape[0])


def _get_attr(prim: Usd.Prim, names: tuple[str, ...]) -> Optional[Usd.Attribute]:
    """First attribute of ``names`` on ``prim`` with an authored value (else any defined
    one, else ``None``). Returns the ATTRIBUTE, not its value."""
    for name in names:
        attr = prim.GetAttribute(name)
        if attr and attr.HasAuthoredValue():
            return attr
    # Fall back to any defined attribute even if the value is only a default.
    for name in names:
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid() and attr.Get() is not None:
            return attr
    return None


def _looks_like_gaussian_prim(prim: Usd.Prim) -> bool:
    """Heuristic: does this prim carry the Gaussian-splat payload attributes?"""
    if not prim or not prim.IsValid():
        return False
    if prim.GetTypeName() == PARTICLE_FIELD_TYPE:
        return True
    # Be tolerant of API-schema variants: require positions + SH coefficients.
    has_positions = _get_attr(prim, _POSITIONS_ATTRS) is not None
    has_sh = _get_attr(prim, _SH_COEFFS_ATTRS) is not None
    return has_positions and has_sh


def _read_quaternions(attr: Usd.Attribute, expected_n: int) -> np.ndarray:
    """Return ``(N, 4)`` quaternions as ``(w, x, y, z)`` (NuRec convention).

    USD stores quaternions real-last when viewed as a numpy array
    (``[i, j, k, real]``); the writer builds ``Gf.Quat(real, i, j, k)`` from NuRec's
    ``(w, x, y, z)`` rotations, so we reorder back to real-first here.
    """
    value = attr.Get()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4 or arr.shape[0] != expected_n:
        raise SimReadyImportError(f"orientations attribute has shape {arr.shape}, expected ({expected_n}, 4)")
    # numpy view of a Vt.Quat*Array is [x, y, z, w] -> reorder to [w, x, y, z].
    rotations = arr[:, [3, 0, 1, 2]]
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    if not (norms > 1e-12).all():
        # A zero quaternion is corrupt authoring; passing it through would silently
        # collapse those splats' orientations.
        raise SimReadyImportError(f"orientations contain {int((norms <= 1e-12).sum())} zero-norm quaternion(s)")
    return rotations / norms


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``(w, x, y, z)`` quaternions (broadcasting)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _time_sampled_xform_ops(prim: Usd.Prim, frame_prim: Optional[Usd.Prim]) -> list[str]:
    """``prim.attr`` names of time-sampled ``xformOp:*`` attributes affecting ``prim``'s pose.

    Walks the RAW attributes up the ancestor chain (same philosophy as the visibility
    walks: no reliance on the schema registry). When ``frame_prim`` is given the walk
    stops BELOW it: ops on the frame prim itself or above cancel exactly when the
    result is re-expressed relative to that prim, time-sampled or not.
    """
    out: list[str] = []
    p = prim
    while p and p.IsValid():
        if frame_prim is not None and p.GetPath() == frame_prim.GetPath():
            break
        for name in p.GetAuthoredPropertyNames():
            if name.startswith("xformOp:"):
                attr = p.GetAttribute(name)
                if attr and attr.GetNumTimeSamples() > 0:
                    out.append(f"{p.GetPath()}.{name}")
        p = p.GetParent()
    return out


def _apply_prim_transform(
    prim: Usd.Prim,
    positions: np.ndarray,
    rotations: np.ndarray,
    scales_linear: np.ndarray,
    xf_cache: UsdGeom.XformCache,
    frame_prim: Optional[Usd.Prim] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold the prim's transform relative to ``frame_prim`` into positions/rotations/scales.

    Identity for a directly-opened payload (the common case). For a composed wrapper
    where the Gaussian prim is nested under a transformed parent this re-expresses the
    cloud in ``frame_prim``'s frame (the stage frame when ``None``): the prim's
    local-to-world with the frame prim's own local-to-world divided out, so nothing
    authored on or above the frame prim leaks into the parts.

    Uses the caller's shared ``XformCache`` (one per import pass) rather than
    ``UsdGeom.Xformable(prim)`` so ancestor Xform transforms still fold when the
    Gaussian prim's own schema type is not registered in the running USD build
    (``Xformable()`` is falsy then, which would silently skip e.g. a wheel's
    suspension translate/rotate).

    Fail-loud contracts (neither pattern occurs in a conforming SimReady asset, so
    both raise instead of silently degrading):

    * Transforms resolve at ``Usd.TimeCode.Default()``, which IGNORES xformOp
      timeSamples: a pose authored only as samples would silently import at its
      rest/identity pose. A time-sampled ``xformOp:*`` on the prim or any ancestor
      below ``frame_prim`` raises :class:`SimReadyImportError` (ops on/above
      ``frame_prim`` cancel in the relative transform and are tolerated).
    * Only rigid + uniform-scale transforms fold exactly: positions could take the
      full affine, but a Gaussian's per-axis scales cannot follow a non-uniform
      ancestor scale, which would knowingly distort the geometry. A non-uniform
      decomposed scale raises :class:`SimReadyImportError`.
    """
    sampled = _time_sampled_xform_ops(prim, frame_prim)
    if sampled:
        raise SimReadyImportError(
            f"prim {prim.GetPath()} has time-sampled xformOp(s) ({', '.join(sampled)}); "
            "the importer resolves transforms at Usd.TimeCode.Default(), which ignores "
            "timeSamples, so the prim would silently import at its rest/identity pose. "
            "Author the rest pose as a default (non-sampled) xformOp value."
        )
    mat = xf_cache.GetLocalToWorldTransform(prim)
    if frame_prim is not None:
        mat = mat * xf_cache.GetLocalToWorldTransform(frame_prim).GetInverse()
    mat_np = np.array(mat, dtype=np.float64)  # 4x4, USD row-vector convention (v' = v @ M)
    if np.allclose(mat_np, np.eye(4)):
        return positions, rotations, scales_linear

    # Positions: full affine via homogeneous row vectors.
    homog = np.concatenate([positions, np.ones((positions.shape[0], 1))], axis=1)
    positions_w = (homog @ mat_np)[:, :3]

    # Decompose for the rotation (applied to orientations) and uniform scale (to scales).
    transform = Gf.Transform(mat)
    rot_quat = transform.GetRotation().GetQuat()
    rw = float(rot_quat.GetReal())
    ri = rot_quat.GetImaginary()
    rq = np.array([rw, float(ri[0]), float(ri[1]), float(ri[2])], dtype=np.float64)
    rotations_w = quat_mul_wxyz(np.broadcast_to(rq, rotations.shape), rotations)

    scale_vec = np.array(transform.GetScale(), dtype=np.float64)
    if not np.allclose(scale_vec, scale_vec[0]):
        raise SimReadyImportError(
            f"prim {prim.GetPath()}: composed transform has non-uniform scale "
            f"{scale_vec.tolist()}; per-gaussian scales cannot follow a non-uniform "
            "ancestor scale (positions would, silently distorting the splats). "
            "SimReady Gaussian assets require rigid + uniform-scale prim transforms."
        )
    scales_w = scales_linear * float(scale_vec[0])
    return positions_w, rotations_w, scales_w


def _read_gaussian_prim(
    stage: Usd.Stage,
    prim: Usd.Prim,
    xf_cache: UsdGeom.XformCache,
    frame_prim: Optional[Usd.Prim] = None,
) -> SimReadyGaussians:
    """Read one Gaussian-bearing prim into pre-activation arrays (frame-prim/stage frame).

    The prim's composed local-to-world transform RELATIVE to ``frame_prim`` (the
    stage's pseudo-root when ``None``) is folded in, so a prim nested under
    transformed ancestors (e.g. a wheel under a translated/rotated suspension Xform)
    comes out correctly placed in the requested frame — and nothing authored on or
    above ``frame_prim`` leaks in. ``xf_cache`` is the caller's shared per-pass
    transform cache; a cache must not outlive a variant-selection change (selections
    recompose ancestor transforms).
    """
    pos_attr = _get_attr(prim, _POSITIONS_ATTRS)
    orient_attr = _get_attr(prim, _ORIENTATIONS_ATTRS)
    scale_attr = _get_attr(prim, _SCALES_ATTRS)
    opacity_attr = _get_attr(prim, _OPACITIES_ATTRS)
    sh_attr = _get_attr(prim, _SH_COEFFS_ATTRS)
    if pos_attr is None or sh_attr is None:
        raise SimReadyImportError(f"prim {prim.GetPath()} is missing positions and/or SH coefficients")

    positions = np.asarray(pos_attr.Get(), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise SimReadyImportError(f"positions has shape {positions.shape}, expected (N, 3)")
    n = int(positions.shape[0])

    if scale_attr is None:
        raise SimReadyImportError(f"prim {prim.GetPath()} is missing scales")
    scales_linear = np.asarray(scale_attr.Get(), dtype=np.float64)
    if scales_linear.shape != (n, 3):
        raise SimReadyImportError(f"scales has shape {scales_linear.shape}, expected ({n}, 3)")

    if opacity_attr is None:
        raise SimReadyImportError(f"prim {prim.GetPath()} is missing opacities")
    opacities = np.asarray(opacity_attr.Get(), dtype=np.float64).reshape(-1)
    if opacities.shape[0] != n:
        raise SimReadyImportError(f"opacities has {opacities.shape[0]} entries, expected {n}")

    if orient_attr is not None:
        rotations = _read_quaternions(orient_attr, n)
    else:
        log.warning("SimReady asset has no orientations; using identity quaternions.")
        rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    # SH coefficients: flat (N*K, 3), gaussian-major; coeff 0 = DC, rest = higher orders.
    sh_flat = np.asarray(sh_attr.Get(), dtype=np.float64)
    if sh_flat.ndim != 2 or sh_flat.shape[1] != 3:
        raise SimReadyImportError(f"SH coefficients has shape {sh_flat.shape}, expected (M, 3)")
    k = _resolve_sh_count(prim, sh_attr, n, sh_flat.shape[0])
    sh = sh_flat.reshape(n, k, 3)
    sh_dc = sh[:, 0, :]
    sh_rest = sh[:, 1:, :]

    degree_attr = prim.GetAttribute(_SH_DEGREE_ATTR)
    sh_degree = int(degree_attr.Get()) if degree_attr and degree_attr.Get() is not None else int(round(k**0.5)) - 1

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Y")

    # Fold any ancestor/prim transform, then normalize units to metres.
    positions, rotations, scales_linear = _apply_prim_transform(
        prim, positions, rotations, scales_linear, xf_cache, frame_prim
    )
    if meters_per_unit != 1.0:
        positions = positions * meters_per_unit
        scales_linear = scales_linear * meters_per_unit

    scales_log = np.log(np.clip(scales_linear, _SCALE_EPS, None))
    opacities_clamped = np.clip(opacities, _OPACITY_EPS, 1.0 - _OPACITY_EPS)
    densities_logit = (np.log(opacities_clamped) - np.log1p(-opacities_clamped)).reshape(n, 1)

    log.info(
        "SimReady import: %d gaussians, sh_degree=%d (K=%d), up_axis=%s, metersPerUnit=%g",
        n,
        sh_degree,
        k,
        up_axis,
        meters_per_unit,
    )
    return SimReadyGaussians(
        positions=positions,
        rotations=rotations,
        scales_log=scales_log,
        densities_logit=densities_logit,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        up_axis=up_axis,
        meters_per_unit=meters_per_unit,
        sh_degree=sh_degree,
    )


def _resolve_sh_count(prim: Usd.Prim, sh_attr: Usd.Attribute, n: int, m: int) -> int:
    """Number of SH coefficients per Gaussian (K), from elementSize / degree / divisibility."""
    element_size = sh_attr.GetMetadata("elementSize")
    if element_size:
        k = int(element_size)
    else:
        degree_attr = prim.GetAttribute(_SH_DEGREE_ATTR)
        if degree_attr and degree_attr.Get() is not None:
            k = (int(degree_attr.Get()) + 1) ** 2
        elif m % n == 0:
            k = m // n
        else:
            raise SimReadyImportError(f"cannot determine SH coefficients-per-gaussian: {m} coeffs for {n} gaussians")
    if k * n != m:
        raise SimReadyImportError(f"SH coefficient count {m} != gaussians {n} x per-gaussian {k}")
    return k


def payloads_equal(a: SimReadyGaussians, b: SimReadyGaussians) -> bool:
    """Full payload equality — a variant may swap ONLY opacity, scale, orientation or
    higher-order SH (e.g. an emissive surface with unchanged geometry/DC), and such a
    prim must still classify as variant-driven, not static. Also used by the bake to
    deduplicate identical element clusters across variant states."""
    return a.count == b.count and all(
        np.array_equal(getattr(a, f), getattr(b, f))
        for f in ("positions", "rotations", "scales_log", "densities_logit", "sh_dc", "sh_rest")
    )


# ---------------------------------------------------------------------------------
# Composed multi-prim assets (variantSets as the native control surface)
# ---------------------------------------------------------------------------------
@dataclass
class SimReadyPart:
    """One Gaussian-bearing prim of a composed asset, expressed in the ROOT PRIM's frame
    (the root prim's own local-to-world transform is divided out; see :class:`SimReadyAsset`)."""

    prim_path: str
    gaussians: SimReadyGaussians


@dataclass
class SimReadyVariantControl:
    """One variantSet exposed as a control: per-state clusters for the prims it swaps.

    ``parts_by_state`` holds ONLY the variant-affected prims (e.g. ``/asset/Panels``);
    prims identical across every state are returned as the asset's static parts instead.
    A prim absent in some state simply contributes no parts for that state.
    """

    variant_set: str
    states: list[str]
    default_state: str  # the asset's authored selection (else the first state)
    parts_by_state: dict[str, list[SimReadyPart]]

    def gaussian_count(self, state: str) -> int:
        """Total gaussians composed by ``state`` across its parts (0 for an all-hidden state)."""
        return sum(p.gaussians.count for p in self.parts_by_state.get(state, []))


@dataclass
class SimReadyAsset:
    """A composed multi-prim SimReady asset: static parts + variant-driven controls.

    All parts (static and per-state) are in the ROOT PRIM's frame: ancestor transforms
    are folded and the root prim's own local-to-world transform is divided out
    (identity in the common case, so a stage-level root is unaffected — but a
    ``root_prim_path`` nested under transformed ancestors, or a root with its own
    xformOps, still yields root-relative parts). Units are normalized to metres, so
    one shared assembly transform places the whole asset without breaking part
    alignment.
    """

    asset_path: str
    root_prim_path: str
    static_parts: list[SimReadyPart]
    controls: dict[str, SimReadyVariantControl]  # keyed by variantSet name
    variant_sets: dict[str, list[str]]  # EVERY variantSet on the root prim (discovery)
    up_axis: str
    meters_per_unit: float

    @property
    def static_count(self) -> int:
        """Number of static (non-variant) parts in the asset."""
        return sum(p.gaussians.count for p in self.static_parts)


def load_simready_asset(
    usd_path: Union[str, Path],
    control_sets: Optional[list[str]] = None,
    root_prim_path: Optional[str] = None,
) -> SimReadyAsset:
    """Load a composed multi-prim SimReady asset, expanding variantSets into controls.

    Args:
        usd_path: Path to the composed stage (e.g. ``scene.usd`` referencing body /
            wheel / panel payloads).
        control_sets: VariantSet names (on the root prim) to expand into per-state
            clusters — the asset's native controls. Sets not listed stay at their
            authored selection; naming a set the asset does not have is an error.
        root_prim_path: Optional explicit root prim; defaults to the stage's defaultPrim.

    Each requested set is expanded by selecting every state in turn and re-reading the
    Gaussian prims under the root; prims whose payload is identical across all states
    are static, the rest belong to that control. Prims swapped by MORE than one control
    are not supported (would need per-combination clusters). A state that composes ZERO
    visible gaussians (an explicit all-hidden state whose entire payload is the other
    states' hidden dressing) is legitimate: it imports with an empty part list. Only an
    asset with no visible gaussians anywhere fails loud.
    """
    usd_path = str(usd_path)
    stage, root = open_stage_root(usd_path, root_prim_path)

    vsets = root.GetVariantSets()
    variant_sets = {name: list(root.GetVariantSet(name).GetVariantNames()) for name in vsets.GetNames()}
    # Resolve requested control names case-insensitively (the YAML loader normalizes
    # tokens to lowercase, but USD variantSet names are case-sensitive).
    canonical_by_lower = {n.lower(): n for n in variant_sets}
    if control_sets and len(canonical_by_lower) != len(variant_sets):
        raise SimReadyImportError(
            f"variantSet names on {root.GetPath()} collide case-insensitively "
            f"({sorted(variant_sets)}); cannot resolve controls unambiguously"
        )
    requested: list[str] = []
    for name in control_sets or []:
        match = canonical_by_lower.get(str(name).lower())
        if match is None:
            raise SimReadyImportError(
                f"variantSet {name!r} not found on {root.GetPath()} "
                f"(asset variantSets: {sorted(variant_sets) or 'none'})"
            )
        if not variant_sets[match]:
            raise SimReadyImportError(f"variantSet {match!r} on {root.GetPath()} has no variants")
        if match not in requested:
            requested.append(match)

    # Authored selections BEFORE we mutate anything (these are the control defaults).
    # USD returns an authored selection verbatim even when it names no existing variant;
    # accepting one would bake every real state hidden (or crash later on a bare KeyError).
    authored_selection = {name: vsets.GetVariantSet(name).GetVariantSelection() for name in requested}
    for name, selection in authored_selection.items():
        if selection and selection not in variant_sets[name]:
            raise SimReadyImportError(
                f"variantSet {name!r} on {root.GetPath()} has authored selection {selection!r}, "
                f"which is not one of its variants {variant_sets[name]}; fix the asset's "
                "authored selection."
            )

    def _read_all_parts() -> tuple[dict[str, SimReadyGaussians], set[str]]:
        """All VISIBLE Gaussian parts under the root (in the root prim's frame), plus
        the subset with animated visibility.

        Statically-invisible prims are the composed state's hidden dressing (e.g. the
        ``on`` panel halves while ``off`` is selected) and are excluded — so they drop
        out of the state's reading and classify as variant-affected, never as static
        parts. Time-sampled prims stay in (the schedule machinery owns them).

        MAY return an empty reading: a state whose entire payload is the other states'
        hidden dressing (an explicit all-hidden ``off``) legitimately composes zero
        visible gaussians. The caller decides when emptiness is fatal (an asset with
        no visible gaussians in ANY reading is).
        """
        out: dict[str, SimReadyGaussians] = {}
        animated: set[str] = set()
        # ONE transform cache per pass, shared across the pass's prims so ancestor
        # local-to-world matrices are computed once. Scoped to the pass because it is
        # only valid for the CURRENT variant selection (selections recompose xforms).
        xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        # TraverseInstanceProxies: an instanceable reference's prims are otherwise
        # invisible to traversal, silently dropping the whole subtree from the bake.
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if _looks_like_gaussian_prim(prim) and not in_animation_preview_subtree(prim):
                is_animated, is_hidden = _classify_visibility(prim, root)
                if is_hidden:
                    continue
                path = str(prim.GetPath())
                out[path] = _read_gaussian_prim(stage, prim, xf_cache, frame_prim=root)
                if is_animated:
                    animated.add(path)
        return out, animated

    # Read the full prim set once per state of each requested control. Each control's
    # authored selection is restored after iterating its states, so other controls'
    # reads (and the returned static baseline) always see authored defaults. Variant
    # selections here are in-memory only — nothing is written back to the asset.
    readings: dict[str, dict[str, dict[str, SimReadyGaussians]]] = {}
    animated_by_control: dict[str, set[str]] = {}
    if not requested:
        readings_base, _ = _read_all_parts()
        if not readings_base:
            raise SimReadyImportError(f"no Gaussian-bearing prims found under {root.GetPath()} in {usd_path}")
    else:
        for name in requested:
            vset = root.GetVariantSet(name)
            per_state: dict[str, dict[str, SimReadyGaussians]] = {}
            animated_per_state: list[set[str]] = []
            try:
                for state in variant_sets[name]:
                    if not vset.SetVariantSelection(state):
                        raise SimReadyImportError(f"could not select variant {name}={state!r} on {root.GetPath()}")
                    per_state[state], animated = _read_all_parts()
                    if not per_state[state]:
                        # Legitimate (see load_simready_asset docstring): explicit all-hidden state.
                        log.info(
                            "variantSet %s=%r on %s composes no visible gaussians; the state bakes all-hidden.",
                            name,
                            state,
                            root.GetPath(),
                        )
                    animated_per_state.append(animated)
            finally:
                # Restore the authored selection (or clear it) even when a state fails
                # to read: the mutation hits the process-shared stage session layer,
                # and a raise must not leave the stage stuck on the last iterated state.
                if authored_selection[name]:
                    vset.SetVariantSelection(authored_selection[name])
                else:
                    vset.ClearVariantSelection()
            readings[name] = per_state
            # Animated in SOME states but not all. A prim animated in every state is
            # animated regardless of this control's selection — its samples come from
            # elsewhere (base level, or another control's authored state) — so attributing
            # it here would claim a prim this variantSet does not drive, and an asset with
            # two controls would then look like one prim swapped by both.
            animated_union: set[str] = set().union(*animated_per_state) if animated_per_state else set()
            animated_all: set[str] = set.intersection(*animated_per_state) if animated_per_state else set()
            animated_by_control[name] = animated_union - animated_all
        if not any(reading for per_state in readings.values() for reading in per_state.values()):
            # Only ALL states hidden everywhere is fatal (single all-hidden states are fine).
            raise SimReadyImportError(
                f"no Gaussian-bearing prims found under {root.GetPath()} in {usd_path} (in any requested state)"
            )
        # Static baseline from the first control's AUTHORED state (not the first variant
        # name, which is alphabetical), so a misc static prim is baked as authored.
        base_name = requested[0]
        base_state = authored_selection[base_name] or variant_sets[base_name][0]
        readings_base = readings[base_name][base_state]

    # A prim belongs to a control iff its payload changes (or it appears/disappears)
    # across that control's states, OR it is an animated-visibility element of one of
    # the control's states (identical payload in every state, but schedule-driven).
    affected_by: dict[str, list[str]] = {}
    for name in requested:
        per_state = readings[name]
        all_paths = sorted(set().union(*[set(d) for d in per_state.values()]))
        for path in all_paths:
            gs = [per_state[s].get(path) for s in variant_sets[name]]
            present = [g for g in gs if g is not None]
            if (
                len(present) < len(gs)
                or any(not payloads_equal(present[0], g) for g in present[1:])
                or path in animated_by_control.get(name, ())
            ):
                affected_by.setdefault(path, []).append(name)

    multi = {p: cs for p, cs in affected_by.items() if len(cs) > 1}
    if multi:
        raise SimReadyImportError(f"prim(s) swapped by more than one variantSet are not supported: {multi}")

    static_parts = [
        SimReadyPart(prim_path=p, gaussians=g) for p, g in sorted(readings_base.items()) if p not in affected_by
    ]

    controls: dict[str, SimReadyVariantControl] = {}
    for name in requested:
        states = variant_sets[name]
        own_paths = sorted(p for p, cs in affected_by.items() if cs == [name])
        if not own_paths:
            log.warning(
                "variantSet %s on %s swaps no Gaussian prims; the control will have no visible effect.",
                name,
                root.GetPath(),
            )
        parts_by_state = {
            state: [
                SimReadyPart(prim_path=p, gaussians=readings[name][state][p])
                for p in own_paths
                if p in readings[name][state]
            ]
            for state in states
        }
        controls[name] = SimReadyVariantControl(
            variant_set=name,
            states=list(states),
            default_state=authored_selection[name] or states[0],
            parts_by_state=parts_by_state,
        )

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Y")
    log.info(
        "SimReady asset import: %s root=%s, %d static part(s) (%d gaussians), controls: %s, "
        "up_axis=%s, metersPerUnit=%g",
        Path(usd_path).name,
        root.GetPath(),
        len(static_parts),
        sum(p.gaussians.count for p in static_parts),
        {n: {s: c.gaussian_count(s) for s in c.states} for n, c in controls.items()} or "none",
        up_axis,
        meters_per_unit,
    )
    return SimReadyAsset(
        asset_path=usd_path,
        root_prim_path=str(root.GetPath()),
        static_parts=static_parts,
        controls=controls,
        variant_sets=variant_sets,
        up_axis=up_axis,
        meters_per_unit=meters_per_unit,
    )


__all__ = [
    "PARTICLE_FIELD_TYPE",
    "SimReadyImportError",
    "SimReadyGaussians",
    "SimReadyPart",
    "SimReadyVariantControl",
    "SimReadyAsset",
    "in_animation_preview_subtree",
    "load_simready_asset",
    "open_stage_root",
    "payloads_equal",
    "quat_mul_wxyz",
]
