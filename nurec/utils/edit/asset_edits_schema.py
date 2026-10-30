# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoring schema + loader for ``asset_edits.yaml``.

This module owns the *human-facing* declarative format only: the dataclasses a
user edits by hand, a YAML loader, and strict validation. It is intentionally
free of ``torch`` / heavy NuRec imports so it can be parsed and unit-tested
cheaply; ``nurec compile-asset-edits`` (``asset_edits.py``) consumes it and bakes
the geometry.

An edit is one instruction: "use this SimReady asset in place of whatever the
capture has on this cuboid track" — or, with a ``world`` bind, "introduce this
asset as a NEW static actor at this world pose". The bake imports the composed
asset with one shared placement transform, bakes each controllable variantSet
state as its own hidden Gaussian cluster, and records how to address them; the
asset's native USD variantSets are its controls. NuRec never learns what the asset
*is* — aspects are opaque strings end to end, so the same mechanism serves any
controllable object. A STATIC prop may instead be a 3D Gaussian Splatting ``.ply``
(the AssetBank / ``edit_assets`` format): one splat cloud with no variantSets, so it
takes no ``controls`` and no ``prim_path``. There is **no** document-level
``layer_id``: edits default to the ``dynamic_rigids`` layer and may target another
Gaussian layer explicitly with their own ``layer_id``.

Format
------
::

    edits:
      # The whole asset (static parts + variant-swapped parts) is imported with one
      # shared placement transform and rides the bound track. `controllable: true`
      # bakes every state's clusters and writes the control's animation table into
      # the manifest, so an external animator drives it per frame over gRPC
      # (AspectState on the render request). `semantics` binds proto enum value names
      # to the control so animators need no name heuristics. Controls without
      # `controllable` bake only their default state (static dressing).
      - id: truck_1
        bind: { type: nurec_track, nurec_track_id: 3 }
        asset: usd/configurable_asset/scene.usd
        placement: { align: bbox, scale: fit_length, yaw_deg: 0.0, offset: [0.0, 0.0, 0.0] }
        replace_original: true                # hide the captured actor on this track
        controls:
          panelState:                         # must be a variantSet on the asset
            default: off                      # state shown when undriven
            controllable: true
            semantics: [PANEL_VISIBILITY]
          appearanceMode:
            default: off
            controllable: true
            semantics: [APPEARANCE_MODE]

      # A `world` bind introduces a NEW static actor instead of replacing one: the
      # bake synthesizes a constant-pose cuboid track spanning the clip (so the prop
      # is a real track everywhere downstream — visible in playback, listed by
      # get_dynamic_objects, aspect-drivable later like any other edit). `position`
      # is world metres; `anchor: base` (default) rests the asset's base there,
      # `center` puts the cuboid center there. The cuboid dims default to the
      # placed asset's bbox; `scale` defaults to fixed 1.0 (real-world size).
      - id: barrier_1
        bind: { type: world, position: [150.0, -3.5, 0.0], yaw_deg: 90.0 }
        asset: usd/props/barrier_waterfilled_01/Body.usd

      # A static prop may also be a 3DGS `.ply`, read exactly as `edit_assets` reads
      # one (Y-up file, rotated to Z-up). Bank PLYs are unit-normalised, so on a world
      # bind give the real-world `dims` and fit to them; a track bind's default
      # `fit_length` needs nothing extra. No `controls`, no `prim_path`.
      - id: cone_1
        bind: { type: world, position: [152.0, -3.5, 0.0], dims: [0.4, 0.4, 0.7] }
        asset: ply/traffic_cone/9000014.ply
        placement: { scale: fit_length }

Placement offsets are in the bound track's **cuboid-local** frame (the compile
log's ``[placement]`` line reports that frame's axis roles and extents).
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from nurec.utils.renamed_keys import normalize_renamed_key, normalize_renamed_value


# Layer an edit bakes into when it does not set its own ``layer_id``. Targeting
# another layer requires an explicit per-edit ``layer_id``.
DEFAULT_LAYER_ID = "dynamic_rigids"
# Current bind type and track-index key spellings.
TRACK_BIND_TYPE = "nurec_track"
TRACK_ID_KEY = "nurec_track_id"
# Currently supported binding types.
_SUPPORTED_BIND_TYPES = {TRACK_BIND_TYPE, "world"}
# Spellings written by asset-edit documents that predate the nre -> nurec rename.
_LEGACY_TRACK_BIND_TYPE = "nre_track"
_LEGACY_TRACK_ID_KEY = "nre_track_id"
# What a world bind's ``position`` names on the placed asset's cuboid.
_WORLD_ANCHORS = {"base", "center"}
# Label class a world bind's synthetic track carries unless overridden; must route
# the track to the edit's target layer (the bake asserts this).
DEFAULT_WORLD_LABEL_CLASS = "protruding_object"
# Suffix marking an edit's asset as a static 3DGS splat file rather than a SimReady USD.
PLY_SUFFIX = ".ply"


class AssetEditsError(ValueError):
    """Raised for malformed ``asset_edits.yaml`` documents."""


def _norm_token(value: Any) -> str:
    """Normalise an aspect / key token to lowercase string.

    YAML 1.1 (PyYAML's safe_load) parses the bare words ``off``/``on``/``yes``/``no``
    as booleans, so an unquoted ``aspect: on`` or a bare ``off`` in an aspect list
    would arrive here as a Python ``bool``. Map those back to their intended strings
    so authors can write the natural ``off`` / ``on`` without quoting.
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value).strip().lower()


def _as_vec3(value: Any, ctx: str) -> list[float]:
    """Validate and coerce a YAML ``[x, y, z]`` into a FINITE float triple; fail loud otherwise.

    Components go through :func:`_as_float`, so booleans, ``.nan`` and ``.inf`` are
    rejected instead of silently baking an invalid placement.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AssetEditsError(f"{ctx}: expected a 3-element [x, y, z], got {value!r}")
    return [_as_float(v, f"{ctx}[{i}]") for i, v in enumerate(value)]


def _as_float(value: Any, ctx: str) -> float:
    """Coerce a YAML scalar to float; fail loud (as AssetEditsError) on anything else.

    Rejects YAML booleans explicitly: ``float(True)`` would silently become 1.0.
    """
    if isinstance(value, bool):
        raise AssetEditsError(f"{ctx}: expected a number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssetEditsError(f"{ctx}: expected a number, got {value!r}") from exc
    if not math.isfinite(result):
        # YAML parses bare .nan/.inf; a non-finite placement is never intended.
        raise AssetEditsError(f"{ctx}: expected a finite number, got {value!r}")
    return result


def _as_int(value: Any, ctx: str) -> int:
    """Coerce a YAML scalar to int; reject non-integral numbers instead of truncating.

    ``int(7.9)`` silently addressing track 7 is exactly the mistake this guards.
    """
    f = _as_float(value, ctx)
    if not f.is_integer():
        raise AssetEditsError(f"{ctx}: expected an integer, got {value!r}")
    return int(f)


def _as_bool(value: Any, ctx: str) -> bool:
    """Require an actual YAML boolean; reject truthy strings like a quoted "false".

    YAML already parses bare true/false/on/off to bool; anything else (e.g. the
    quoted string "false", which is truthy) is an authoring mistake — fail loud.
    """
    if not isinstance(value, bool):
        raise AssetEditsError(f"{ctx}: expected a boolean (true/false), got {value!r}")
    return value


def is_ply_asset(asset: Any) -> bool:
    """Whether an edit's ``asset`` names a static 3DGS ``.ply`` rather than a SimReady USD."""
    return str(asset).lower().endswith(PLY_SUFFIX)


@dataclass
class Binding:
    """Where an edit's asset goes.

    ``nurec_track``: an existing layer-local cuboid track whose actor the asset
    replaces (or rides). ``world``: no source actor — the bake synthesizes a NEW
    constant-pose cuboid track at the given world pose and the asset rides that.
    """

    type: str
    # nurec_track binds only: layer-local cuboid track index.
    nurec_track_id: int = -1
    # world binds only ------------------------------------------------------
    position: Optional[list[float]] = None  # [x, y, z] world metres (required)
    yaw_deg: float = 0.0  # heading about world +Z
    anchor: str = "base"  # what `position` names: cuboid base center | cuboid center
    dims: Optional[list[float]] = None  # explicit cuboid [dx, dy, dz]; default: placed asset bbox
    label_class: str = DEFAULT_WORLD_LABEL_CLASS  # routes the synthetic track's layer partition
    track_id: Optional[str] = None  # string track id; defaults to "prop:<edit_id>" (set on parse)

    @property
    def is_world(self) -> bool:
        return self.type == "world"

    @staticmethod
    def from_dict(d: Mapping[str, Any], ctx: str) -> "Binding":
        """Parse and validate a ``bind:`` mapping (type defaults to ``nurec_track``).

        The pre-rename spellings ``nre_track`` / ``nre_track_id`` are accepted; the parsed
        binding always carries the current spelling.
        """
        if not isinstance(d, Mapping):
            raise AssetEditsError(f"{ctx}: bind must be a mapping, got {d!r}")
        original_keys = set(d)
        try:
            d = normalize_renamed_key(d, TRACK_ID_KEY, _LEGACY_TRACK_ID_KEY)
        except ValueError as e:
            raise AssetEditsError(f"{ctx}: {e}") from None
        btype = normalize_renamed_value(
            str(d.get("type", TRACK_BIND_TYPE)).strip().lower(), TRACK_BIND_TYPE, _LEGACY_TRACK_BIND_TYPE
        )
        if btype not in _SUPPORTED_BIND_TYPES:
            raise AssetEditsError(
                f"{ctx}: unsupported bind type {btype!r} (expected one of {sorted(_SUPPORTED_BIND_TYPES)})"
            )
        known = {"type"} | (
            {TRACK_ID_KEY}
            if btype == TRACK_BIND_TYPE
            else {"position", "yaw_deg", "anchor", "dims", "label_class", "track_id"}
        )
        # Report the spelling the author wrote, not the normalized one.
        unknown = sorted(
            k for k in original_keys if k not in known and not (k == _LEGACY_TRACK_ID_KEY and TRACK_ID_KEY in known)
        )
        if unknown:
            raise AssetEditsError(
                f"{ctx}: unknown field(s) {unknown} for bind type {btype!r} (supported: {sorted(known)})"
            )
        if btype == TRACK_BIND_TYPE:
            if TRACK_ID_KEY not in d:
                raise AssetEditsError(f"{ctx}: bind requires '{TRACK_ID_KEY}'")
            return Binding(type=btype, nurec_track_id=_as_int(d[TRACK_ID_KEY], f"{ctx}.nurec_track_id"))

        if "position" not in d:
            raise AssetEditsError(f"{ctx}: world bind requires 'position' ([x, y, z] world metres)")
        anchor = str(d.get("anchor", "base")).strip().lower()
        if anchor not in _WORLD_ANCHORS:
            raise AssetEditsError(f"{ctx}.anchor: expected one of {sorted(_WORLD_ANCHORS)}, got {anchor!r}")
        dims = d.get("dims")
        if dims is not None:
            dims = _as_vec3(dims, f"{ctx}.dims")
            if any(v <= 0.0 for v in dims):
                raise AssetEditsError(f"{ctx}.dims: components must be > 0, got {dims}")
        track_id = d.get("track_id")
        if track_id is not None:
            track_id = str(track_id).strip()
            if not track_id:
                raise AssetEditsError(f"{ctx}.track_id: must be a non-empty string")
        return Binding(
            type=btype,
            position=_as_vec3(d["position"], f"{ctx}.position"),
            yaw_deg=_as_float(d.get("yaw_deg", 0.0), f"{ctx}.yaw_deg"),
            anchor=anchor,
            dims=dims,
            label_class=str(d.get("label_class", DEFAULT_WORLD_LABEL_CLASS)).strip(),
            track_id=track_id,
        )


@dataclass
class AssetPlacement:
    """How a composed asset is placed into its bound track's cuboid-local frame.

    One SHARED transform is applied to the whole assembly (never per-part), preserving
    the parts' relative layout. ``align: bbox`` centers the assembly bbox on the cuboid
    in x/y and rests the bbox bottom on the cuboid floor (wheels on the ground);
    ``align: origin`` maps the asset's root-frame origin to the cuboid center.
    ``scale: fit_length`` (default) uniformly scales the assembly so its bbox length
    matches the cuboid length; a number is a fixed uniform scale instead.
    ``offset`` is a final cuboid-local trim applied after align/scale/rotation.
    """

    align: str = "bbox"  # bbox | origin
    scale_mode: str = "fit_length"  # fit_length | fixed
    scale: float = 1.0  # uniform scale; used when scale_mode == fixed
    yaw_deg: float = 0.0  # rotation about the cuboid up-axis (asset +x forward -> cuboid +length)
    rotation_deg: Optional[list[float]] = None  # full [rx, ry, rz] euler (deg) if yaw isn't enough
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # cuboid-local metres

    def euler_xyz_deg(self) -> tuple[float, float, float]:
        """Effective [rx, ry, rz] euler (deg); ``yaw_deg`` is added to the Z (up) component."""
        rx, ry, rz = self.rotation_deg or [0.0, 0.0, 0.0]
        return float(rx), float(ry), float(rz) + float(self.yaw_deg)

    @staticmethod
    def from_dict(d: Mapping[str, Any], ctx: str) -> "AssetPlacement":
        """Parse and validate a ``placement:`` mapping; unknown fields fail loud."""
        if not isinstance(d, Mapping):
            raise AssetEditsError(f"{ctx}: placement must be a mapping (align/scale/yaw_deg/offset), got {d!r}")
        known = {"align", "scale", "yaw_deg", "rotation_deg", "offset"}
        unknown = sorted(set(d) - known)
        if unknown:
            raise AssetEditsError(f"{ctx}: unknown placement field(s) {unknown} (supported: {sorted(known)})")
        align = str(d.get("align", "bbox")).strip().lower()
        if align not in {"bbox", "origin"}:
            raise AssetEditsError(f"{ctx}.align: expected 'bbox' or 'origin', got {align!r}")
        raw_scale = d.get("scale", "fit_length")
        if isinstance(raw_scale, str):
            if raw_scale.strip().lower() != "fit_length":
                raise AssetEditsError(f"{ctx}.scale: expected a number or 'fit_length', got {raw_scale!r}")
            scale_mode, scale = "fit_length", 1.0
        else:
            scale_mode, scale = "fixed", _as_float(raw_scale, f"{ctx}.scale")
            if scale <= 0.0:
                raise AssetEditsError(f"{ctx}.scale must be > 0 (got {scale})")
        rotation = d.get("rotation_deg")
        return AssetPlacement(
            align=align,
            scale_mode=scale_mode,
            scale=scale,
            yaw_deg=_as_float(d.get("yaw_deg", 0.0), f"{ctx}.yaw_deg"),
            rotation_deg=_as_vec3(rotation, f"{ctx}.rotation_deg") if rotation is not None else None,
            offset=_as_vec3(d.get("offset", [0.0, 0.0, 0.0]), f"{ctx}.offset"),
        )


@dataclass
class ControlSpec:
    """One native control of the asset: a variantSet driven at render time.

    ``name`` must be a variantSet on the asset's root prim (the compiler validates this
    against the asset and fails on an unknown set or state). ``default`` is the state
    baked VISIBLE — what renders when the control is not driven; ``None`` means the
    asset's authored selection.

    ``controllable: true`` bakes ALL states' clusters and emits the control's animation
    table into the manifest, so an external animator can drive it over gRPC.
    ``semantics`` optionally binds the control to sender-defined enum value names
    (e.g. ``PANEL_VISIBILITY``) so animators map incoming semantic state to this
    control without name heuristics. These names are illustrative, not built-in enums.
    """

    # AUTHORED FIELDS ONLY: a parsed document means the same thing at every point in the
    # pipeline. Derived data travels separately — variant animation read from the asset
    # and passed into AssetEditsBaker, bake results accumulated on the baker — so nothing
    # here is back-filled as the bake progresses.
    name: str
    default: Optional[str] = None
    # Bake all states for external driving over gRPC; see class docstring.
    controllable: bool = False
    # Proto enum value names this control answers to (uppercased on load).
    semantics: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(name: Any, d: Mapping[str, Any], ctx: str) -> "ControlSpec":
        """Parse and validate one ``controls:`` entry; unknown fields fail loud."""
        if not isinstance(d, Mapping):
            raise AssetEditsError(f"{ctx}: control entry must be a mapping (default/controllable/semantics), got {d!r}")
        known = {"default", "controllable", "semantics"}
        unknown = sorted(set(d) - known)
        if unknown:
            raise AssetEditsError(f"{ctx}: unknown control field(s) {unknown} (supported: {sorted(known)})")
        semantics = d.get("semantics") or []
        if not isinstance(semantics, (list, tuple)) or not all(isinstance(s, str) for s in semantics):
            raise AssetEditsError(f"{ctx}.semantics: expected a list of enum value names, got {semantics!r}")
        return ControlSpec(
            name=_norm_token(name),
            default=_norm_token(d["default"]) if d.get("default") is not None else None,
            controllable=_as_bool(d.get("controllable", False), f"{ctx}.controllable"),
            semantics=[s.strip().upper() for s in semantics],
        )


@dataclass
class EditSpec:
    """One asset edit: bind a SimReady asset (or a static ``.ply`` prop) to an existing
    cuboid track (replacing its actor) or to a fixed world pose (introducing a new one)."""

    id: str
    bind: Binding
    asset: str  # composed stage (e.g. scene.usd) or a 3DGS .ply; relative to the YAML's directory
    # Gaussian layer to bake this edit into (resolved at parse: an absent YAML
    # ``layer_id`` becomes ``DEFAULT_LAYER_ID``, so consumers never re-resolve it).
    layer_id: str = DEFAULT_LAYER_ID
    # Always present: from_dict builds one from `placement: {}` when the key is absent.
    placement: AssetPlacement = field(default_factory=AssetPlacement)
    controls: list[ControlSpec] = field(default_factory=list)
    # Hide the captured (trained) Gaussians already riding the bound track, so the imported
    # asset REPLACES the original actor instead of overlapping it.
    replace_original: bool = False
    prim_path: Optional[str] = None  # optional explicit root prim inside the asset

    @staticmethod
    def from_dict(d: Mapping[str, Any], ctx: str) -> "EditSpec":
        """Parse and validate one ``edits:`` entry; unknown fields fail loud."""
        if not isinstance(d, Mapping):
            raise AssetEditsError(f"{ctx}: edit entry must be a mapping, got {d!r}")
        if "id" not in d:
            raise AssetEditsError(f"{ctx}: edit requires an 'id'")
        fid = str(d["id"])
        if "bind" not in d:
            raise AssetEditsError(f"{ctx} ({fid}): edit requires a 'bind'")
        if not d.get("asset"):
            raise AssetEditsError(
                f"{ctx} ({fid}): edit requires an 'asset' (a composed SimReady USD, or a .ply for a static prop)"
            )

        known = {"id", "bind", "asset", "layer_id", "placement", "controls", "replace_original", "prim_path"}
        unknown = sorted(set(d) - known)
        if unknown:
            raise AssetEditsError(f"{ctx} ({fid}): unknown edit field(s) {unknown} (supported: {sorted(known)})")

        raw_controls = d.get("controls") or {}
        if not isinstance(raw_controls, Mapping):
            raise AssetEditsError(f"{ctx} ({fid}).controls: expected a mapping of variantSet -> control spec")
        controls = [
            ControlSpec.from_dict(name, spec or {}, f"{ctx} ({fid}).controls[{name}]")
            for name, spec in raw_controls.items()
        ]
        seen_names = [c.name for c in controls]
        dupes = sorted({n for n in seen_names if seen_names.count(n) > 1})
        if dupes:
            raise AssetEditsError(f"{ctx} ({fid}).controls: duplicate control name(s) after normalization: {dupes}")

        if is_ply_asset(d["asset"]):
            # One splat cloud, no variantSets: nothing for a control to switch, no prim to pick.
            if controls:
                raise AssetEditsError(
                    f"{ctx} ({fid}): a .ply asset is a static splat cloud with no variantSets, so it takes no "
                    f"controls (got {[c.name for c in controls]}); controllable edits need a SimReady USD asset"
                )
            if d.get("prim_path") is not None:
                raise AssetEditsError(f"{ctx} ({fid}): prim_path does not apply to a .ply asset (it has no prims)")

        layer_id = DEFAULT_LAYER_ID
        if d.get("layer_id") is not None:
            layer_id = str(d["layer_id"]).strip()
            if not layer_id:
                raise AssetEditsError(f"{ctx} ({fid}).layer_id: must be a non-empty layer id")

        bind = Binding.from_dict(d["bind"], f"{ctx} ({fid}).bind")
        raw_placement = d.get("placement") or {}
        placement = AssetPlacement.from_dict(raw_placement, f"{ctx} ({fid}).placement")
        replace_original = _as_bool(d.get("replace_original", False), f"{ctx}.replace_original")

        if bind.is_world:
            if replace_original:
                raise AssetEditsError(
                    f"{ctx} ({fid}): replace_original is meaningless on a world bind (there is no source actor)"
                )
            if bind.track_id is None:
                bind.track_id = f"prop:{fid}"
            # A world bind has no source actor to fit, so the props keep their
            # authored real-world size unless the author scales explicitly.
            if "scale" not in raw_placement:
                placement.scale_mode, placement.scale = "fixed", 1.0
            elif placement.scale_mode == "fit_length" and bind.dims is None:
                raise AssetEditsError(
                    f"{ctx} ({fid}).placement.scale: 'fit_length' on a world bind requires explicit bind.dims"
                    " (there is no source actor to fit)"
                )

        return EditSpec(
            id=fid,
            bind=bind,
            asset=str(d["asset"]),
            layer_id=layer_id,
            placement=placement,
            controls=controls,
            replace_original=replace_original,
            prim_path=str(d["prim_path"]) if d.get("prim_path") is not None else None,
        )


@dataclass
class AssetEditsDoc:
    """A parsed, validated ``asset_edits.yaml`` (every edit's ``layer_id`` is resolved)."""

    edits: list[EditSpec]

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AssetEditsDoc":
        """Parse and validate a whole document (top-level ``edits:`` list)."""
        if not isinstance(data, Mapping):
            raise AssetEditsError("top-level document must be a mapping")
        unknown = sorted(set(data) - {"edits"})
        if unknown:
            raise AssetEditsError(f"unknown top-level field(s) {unknown} (supported: ['edits'])")
        raw_edits = data.get("edits", [])
        if not isinstance(raw_edits, (list, tuple)):
            raise AssetEditsError(f"'edits' must be a list of edit mappings, got {raw_edits!r}")
        edits = [EditSpec.from_dict(e, f"edits[{i}]") for i, e in enumerate(raw_edits)]
        if not edits:
            raise AssetEditsError("document has no edits")
        # Edit ids key the baker's bookkeeping (asset cache, schedules, manifests):
        # a duplicate would silently bake the first edit's asset in both places.
        seen_ids = [e.id for e in edits]
        dupes = sorted({i for i in seen_ids if seen_ids.count(i) > 1})
        if dupes:
            raise AssetEditsError(f"duplicate edit id(s) {dupes}: every edit needs a unique 'id'")
        return AssetEditsDoc(edits=edits)


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses a repeated mapping key.

    YAML resolves a repeated key to its last value, which no later validation can see: a
    repeated ``bind:`` binds the asset to a different track, a repeated control entry drops
    the first one's ``controllable`` and ``semantics``. Both are silent.
    """


def _no_duplicate_keys(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    """Construct a mapping, raising on any key that appears more than once."""
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            mark = key_node.start_mark
            raise AssetEditsError(
                f"duplicate key {key!r} at line {mark.line + 1}, column {mark.column + 1}: "
                "YAML would silently keep only the last one"
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)  # type: ignore[arg-type]


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def load_asset_edits(path: Path) -> AssetEditsDoc:
    """Parse and validate an ``asset_edits.yaml`` document."""
    with open(path, "r") as f:
        try:
            data = yaml.load(f, Loader=_StrictLoader)  # noqa: S506 - _StrictLoader derives from SafeLoader
        except yaml.YAMLError as exc:
            # PyYAML's scanner/parser errors are not AssetEditsError, and the CLI only
            # translates AssetEditsError — without this a bad indent prints a traceback.
            raise AssetEditsError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        raise AssetEditsError(f"{path} is empty")
    return AssetEditsDoc.from_dict(data)


__all__ = [
    "DEFAULT_LAYER_ID",
    "DEFAULT_WORLD_LABEL_CLASS",
    "AssetEditsError",
    "Binding",
    "AssetPlacement",
    "ControlSpec",
    "EditSpec",
    "AssetEditsDoc",
    "load_asset_edits",
]
