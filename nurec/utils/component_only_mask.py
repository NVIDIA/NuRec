# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss-mask assembly for component-only optimization.

Combines the per-layer mask sources configured in
``nurec.config.component_only`` into the final per-batch camera loss
mask:

    final = union of regions of all non-disabled pixel-owning components
              (layers per their masks; sky if non-disabled;
               background's residual if that scope region is non-disabled)
            minus disabled tracked actors in front of supervised tracked actors
            minus ego (always)

State semantics recap (see ``nurec.config.component_only``):
``enabled`` and ``frozen`` components both contribute their region to
the loss union (frozen renders participate; enabled overlapping content
trains around them). ``disabled`` components' regions are in NOBODY's
mask: they are excluded from the union, carved out of background's
residual, and subtracted from supervised tracked regions when their
nearest cuboid depth is in front — nothing may train on them.

Split of responsibilities: this module is pure tensor logic. The
system wiring resolves the dataset class-id map and projects per-layer
cuboid hit masks plus nearest depths
(``GaussiansComposite.build_component_only_bbox_masks``, via the shared
``CuboidTracks.ray_intersection`` kernel) per camera frame, and hands
both in. ``build_mask_spec`` is called once at construction (resolving
class names -> ids, fail-loud); ``build_camera_loss_mask`` runs per
batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from torch import Tensor

from nurec.config.component_only import (
    BACKGROUND_LAYER_ID,
    ComponentOnlyConfig,
    ComponentState,
    MasksOperation,
    MaskSource,
    ScopeRegion,
    _entry_has_disabled_partition,
    _entry_has_non_disabled_population,
    resolve_layer_masks,
)


# Pseudo carved-entry key suffix for the disabled-track partition of a
# supervised layer (per-track subsetting): those tracks' boxes are in
# nobody's mask and must be carved out of the residual — a disabled
# track is never baked into the background.
DISABLED_TRACKS_SUFFIX = "@disabled_tracks"


# Semantic class that is unconditionally excluded from every loss mask.
# The ego car is masked by the dedicated ego mask earlier in the
# pipeline; excluding it here as well is redundant-but-safe.
EGO_CLASS_NAME = "egocar"

# The sky rendering module's implicit semantic class (it has no
# ``masks`` config — it is semantically defined, like road).
SKY_CLASS_NAME = "sky"

# Cuboid ray distances are measured in NuRec metres because camera ray
# directions are normalized. Treat a disabled actor within one millimetre of
# a rendered actor as the foreground owner. This is comfortably above
# float32 slab/interpolation jitter while negligible relative to actor boxes.
ACTOR_DEPTH_TIE_TOLERANCE_M = 1e-3


@dataclass(frozen=True)
class ProjectedTrackRegion:
    """Projected track geometry for one component-only region.

    Both tensors have shape ``(B, H, W)``. ``mask`` is bool and
    ``nearest_depth`` is floating point, with ``+inf`` wherever no cuboid is
    hit. Keeping them together prevents mask/depth partitions from drifting.
    """

    mask: Tensor
    nearest_depth: Tensor


@dataclass(frozen=True)
class LayerRegionSpec:
    """Resolved mask recipe for one region group (class names bound to ids).

    A supervised layer maps to a *tuple* of these — one per distinct
    effective mask recipe among its non-disabled tracks (see
    ``build_mask_spec``). A layer without per-track ``masks`` overrides
    has a single group covering the whole layer.
    """

    sources: tuple[MaskSource, ...]
    operation: Optional[MasksOperation]
    # Dataset class ids for the ``semantic`` source; empty when the
    # group's recipe has no semantic component.
    semantic_class_ids: tuple[int, ...]
    # Key into the per-batch projected ``track_regions`` mapping this group's
    # bbox source reads. Empty falls back to the layer id (single-group / carved).
    bbox_key: str = ""
    # Dataset track ids whose boxes form this group's bbox mask. None =
    # the layer's whole non-disabled population (no per-track ``masks``
    # override); a tuple = exactly these tracks (per-recipe group).
    bbox_track_ids: Optional[tuple[str, ...]] = None
    # When set, the ``semantic`` source is intersected with this bbox
    # key's box mask, so a class mask only supervises the *enabled*
    # instances (instance-awareness). Set only when the layer has disabled
    # tracks — otherwise the plain whole-class mask is used.
    semantic_bbox_key: str = ""
    # Key whose projected cuboid depth orders this supervised actor region.
    # Empty for non-track/static regions. This is set even for semantic-only
    # tracked regions so only their semantic-and-cuboid pixels gain depth.
    track_depth_key: str = ""

    @property
    def needs_bbox(self) -> bool:
        return MaskSource.bbox in self.sources or MaskSource.bbox_expanded in self.sources

    @property
    def needs_semantic(self) -> bool:
        return MaskSource.semantic in self.sources


@dataclass(frozen=True)
class ComponentOnlyMaskSpec:
    """Static (per-run) decision of how to assemble the loss mask.

    ``supervised_layers`` are the non-disabled (enabled + frozen)
    layers whose regions join the loss union. ``carved_layers`` are the
    disabled/unlisted-but-existing layers whose regions must still be
    carved out of the residual. Background appears in neither — its
    participation is captured by ``include_residual``.
    """

    # Each supervised layer maps to a tuple of region groups (one per
    # distinct per-track mask recipe; single group without per-track
    # ``masks`` overrides). Their regions are unioned for the layer.
    supervised_layers: Mapping[str, tuple[LayerRegionSpec, ...]]
    carved_layers: Mapping[str, LayerRegionSpec]
    # Disabled track populations whose geometry can occlude supervised actors.
    # This is independent of ``include_residual``: a missing actor must never
    # leak into an actor-only loss either.
    disabled_track_regions: Mapping[str, LayerRegionSpec]
    include_residual: bool
    include_sky: bool
    sky_class_id: Optional[int]
    ego_class_id: Optional[int]

    @property
    def needs_semantic(self) -> bool:
        """Whether per-pixel semantic labels are required to build the mask."""
        if self.include_sky:
            return True
        if self.include_residual:
            # The residual must identify sky (in nobody's mask when the
            # sky model is disabled) and carve semantic siblings.
            return True
        return any(group.needs_semantic for groups in self.supervised_layers.values() for group in groups)


def _resolve_class_ids(
    layer_id: str,
    class_names: list[str],
    semantic_class_id_map: Optional[Mapping[str, int]],
) -> tuple[int, ...]:
    if semantic_class_id_map is None:
        raise ValueError(
            f"component_only: layer '{layer_id}' requires per-pixel semantic labels (mask source "
            f"'semantic'), but the dataset provides no semantic class map. Enable NCORE "
            f"aux_data + semantic aux, or drop the semantic mask source."
        )
    if not class_names:
        raise ValueError(
            f"component_only: layer '{layer_id}' uses a semantic mask source but "
            f"model.layers.{layer_id}.class_labels is empty — the semantic region would be all "
            f"false, silently dropping the layer from supervision. Bind classes or drop the "
            f"semantic mask source."
        )
    missing = sorted(name for name in class_names if name not in semantic_class_id_map)
    if missing:
        raise ValueError(
            f"component_only: layer '{layer_id}' declares semantic class(es) {missing} that the "
            f"dataset's semantic class map does not contain. Available classes: "
            f"{sorted(semantic_class_id_map)}. Fix model.layers.{layer_id}.class_labels or use a "
            f"dataset whose semantic vocabulary matches."
        )
    return tuple(semantic_class_id_map[name] for name in class_names)


def _region_spec(
    layer_id: str,
    entry: Any,
    model_layers: Mapping[str, Any],
    semantic_class_id_map: Optional[Mapping[str, int]],
    resolve_semantic: bool,
) -> LayerRegionSpec:
    """Build a layer's region spec, resolving class names -> ids.

    ``resolve_semantic=False`` skips the id resolution (and thereby the
    dataset-capability check) — used for carved layers when no residual
    region consumes their semantic classes.
    """
    sources, operation = resolve_layer_masks(layer_id, entry)
    semantic_ids: tuple[int, ...] = ()
    if MaskSource.semantic in sources and resolve_semantic:
        class_labels = getattr(model_layers[layer_id], "class_labels", None) or []
        semantic_ids = _resolve_class_ids(layer_id, list(class_labels), semantic_class_id_map)
    return LayerRegionSpec(sources=sources, operation=operation, semantic_class_ids=semantic_ids, bbox_key=layer_id)


def _has_track_mask_override(entry: Any) -> bool:
    """True iff any per-track override sets its own ``masks`` recipe.

    Per-track ``state`` alone (enable/disable/freeze) keeps every track
    on the layer recipe — that stays a single region group. Only a
    per-track ``masks`` override splits the layer into per-recipe groups.
    """
    tracks = getattr(entry, "tracks", None)
    return tracks is not None and any(t.masks is not None for t in tracks.values())


def _supervised_region_specs(
    layer_id: str,
    entry: Any,
    model_layers: Mapping[str, Any],
    semantic_class_id_map: Optional[Mapping[str, int]],
    track_states: Optional[Mapping[str, ComponentState]],
    track_backed: bool,
) -> tuple[LayerRegionSpec, ...]:
    """One region group per distinct effective mask recipe in the layer.

    Without per-track ``masks`` overrides (or without resolved track
    states) the layer is a single group on the layer recipe. With
    overrides, each non-disabled track's effective recipe
    (``resolve_layer_masks(layer, track)``) defines a group; the group's
    supervised region is (semantic class mask and/or the group tracks'
    boxes, combined by the group's operation), and the layer region is
    their union. Grouping by recipe — not per individual track — keeps
    the box masks compact (raw vs expanded already differ by recipe).
    """

    def semantic_ids_for(sources: tuple[MaskSource, ...]) -> tuple[int, ...]:
        if MaskSource.semantic not in sources:
            return ()
        class_labels = getattr(model_layers[layer_id], "class_labels", None) or []
        return _resolve_class_ids(layer_id, list(class_labels), semantic_class_id_map)

    # When any track is disabled, a semantic source must be instance-scoped to
    # the enabled tracks' boxes (a class mask alone can't exclude a disabled
    # same-class instance). With no disabled track the plain class mask stands.
    has_disabled = track_states is not None and any(s == ComponentState.disabled for s in track_states.values())

    if not _has_track_mask_override(entry) or track_states is None:
        sources, operation = resolve_layer_masks(layer_id, entry)
        return (
            LayerRegionSpec(
                sources=sources,
                operation=operation,
                semantic_class_ids=semantic_ids_for(sources),
                bbox_key=layer_id,
                bbox_track_ids=None,
                semantic_bbox_key=(layer_id if has_disabled and MaskSource.semantic in sources else ""),
                track_depth_key=(layer_id if track_backed else ""),
            ),
        )

    layer_track_cfgs = getattr(entry, "tracks", None) or {}
    groups: dict[tuple[tuple[MaskSource, ...], Optional[MasksOperation]], list[str]] = {}
    for track_id, state in track_states.items():
        if state == ComponentState.disabled:
            continue
        recipe = resolve_layer_masks(layer_id, entry, layer_track_cfgs.get(track_id))
        groups.setdefault(recipe, []).append(track_id)

    specs: list[LayerRegionSpec] = []
    for i, (recipe, track_ids) in enumerate(groups.items()):
        sources, operation = recipe
        bbox_key = f"{layer_id}#{i}"
        specs.append(
            LayerRegionSpec(
                sources=sources,
                operation=operation,
                semantic_class_ids=semantic_ids_for(sources),
                bbox_key=bbox_key,
                bbox_track_ids=tuple(sorted(track_ids)),
                semantic_bbox_key=(bbox_key if has_disabled and MaskSource.semantic in sources else ""),
                track_depth_key=(bbox_key if track_backed else ""),
            )
        )
    return tuple(specs)


def build_mask_spec(
    component_only: ComponentOnlyConfig,
    model_layers: Mapping[str, Any],
    semantic_class_id_map: Optional[Mapping[str, int]],
    layer_track_states: Optional[Mapping[str, Mapping[str, ComponentState]]] = None,
) -> Optional[ComponentOnlyMaskSpec]:
    """Resolve the config into a static mask-assembly spec.

    Returns ``None`` when component-only mode is off (BC path: no
    masking).

    ``layer_track_states`` (per layer, per dataset-track effective state)
    is supplied by the model once track partitions are resolved. It lets
    a layer with per-track ``masks`` overrides split into per-recipe
    region groups so each track's mask reaches the loss mask. Without it
    (e.g. pure-config callers), layers fall back to a single layer-recipe
    group.

    ``semantic_class_id_map`` may be ``None`` for datasets without
    per-pixel semantics: a pure-bbox run (no semantic mask sources
    needed, sky disabled, no residual) is legal without semantics. The
    dataset-capability check is folded in here — the map is only
    demanded (fail-loud) where semantic classes are actually consumed:
    supervised layers with a semantic source, carved layers when the
    residual must carve their regions, and the sky class when the sky
    model participates.
    """
    if not component_only.enabled:
        return None

    background_entry = component_only.layers.get(BACKGROUND_LAYER_ID)
    background_scope: dict[ScopeRegion, ComponentState] = {}
    if background_entry is not None:
        if background_entry.scope is not None:
            background_scope = dict(background_entry.scope)
        else:
            # A background entry without an explicit scope is sugar for
            # "residual region with the entry's state".
            background_scope = {ScopeRegion.residual: background_entry.state}
    include_residual = background_scope.get(ScopeRegion.residual, ComponentState.disabled) != ComponentState.disabled

    sky_state = component_only.sky_model.state if component_only.sky_model is not None else ComponentState.disabled
    include_sky = sky_state != ComponentState.disabled

    supervised: dict[str, tuple[LayerRegionSpec, ...]] = {}
    carved: dict[str, LayerRegionSpec] = {}
    disabled_track_regions: dict[str, LayerRegionSpec] = {}
    for layer_id in model_layers:
        if layer_id == BACKGROUND_LAYER_ID:
            continue
        entry = component_only.layers.get(layer_id)
        track_states = layer_track_states.get(layer_id) if layer_track_states is not None else None
        track_backed = getattr(model_layers[layer_id], "tracks", None) is not None
        # A layer is supervised if ANY of its population is non-disabled
        # — with per-track overrides the layer default may be disabled
        # while individual tracks are enabled.
        has_non_disabled_population = (
            any(state != ComponentState.disabled for state in track_states.values())
            if track_states is not None
            else entry is not None and _entry_has_non_disabled_population(entry)
        )
        if entry is not None and has_non_disabled_population:
            # Supervised layers always need their semantic ids. Per-track
            # ``masks`` overrides split the layer into per-recipe groups
            # (each supervising its own tracks' boxes / class mask).
            supervised[layer_id] = _supervised_region_specs(
                layer_id,
                entry,
                model_layers,
                semantic_class_id_map,
                track_states,
                track_backed,
            )
        else:
            # Carved (fully-disabled / unlisted) layer. Its semantic region is
            # only consumed by the residual — without it, don't demand the map.
            carved[layer_id] = _region_spec(
                layer_id, entry, model_layers, semantic_class_id_map, resolve_semantic=include_residual
            )

        # A disabled partition — a disabled track under a supervised layer, or
        # ALL tracks of a fully-disabled layer — carries boxes that are in
        # nobody's supervised mask and must be carved from the residual,
        # regardless of the layer's semantic-vs-bbox recipe (a disabled
        # bbox-override track's box is not covered by a semantic parent recipe).
        # The composite builds the disabled partition's boxes per each disabled
        # track's own variant under this pseudo key.
        if include_residual and entry is not None and _entry_has_disabled_partition(entry):
            carved[layer_id + DISABLED_TRACKS_SUFFIX] = LayerRegionSpec(
                sources=(MaskSource.bbox,),
                operation=None,
                semantic_class_ids=(),
                bbox_key=layer_id + DISABLED_TRACKS_SUFFIX,
            )

        # Occlusion geometry is required for every disabled tracked actor,
        # including unlisted/fully-disabled layers and actor-only runs with no
        # residual. Production supplies resolved dataset track states; the
        # config-only fallback may over-emit an empty partition, which is safe.
        if track_backed:
            has_disabled_tracks = (
                any(state == ComponentState.disabled for state in track_states.values())
                if track_states is not None
                else entry is None or _entry_has_disabled_partition(entry)
            )
            if has_disabled_tracks:
                sources, operation = resolve_layer_masks(layer_id, entry)
                disabled_key = layer_id + DISABLED_TRACKS_SUFFIX
                disabled_track_regions[disabled_key] = LayerRegionSpec(
                    sources=sources,
                    operation=operation,
                    semantic_class_ids=(),
                    bbox_key=disabled_key,
                    track_depth_key=disabled_key,
                )

    sky_class_id = semantic_class_id_map.get(SKY_CLASS_NAME) if semantic_class_id_map is not None else None
    if (include_sky or include_residual) and sky_class_id is None:
        available = sorted(semantic_class_id_map) if semantic_class_id_map is not None else "none (no semantic aux)"
        reason = (
            "sky_model is non-disabled (its pixels join the loss mask)"
            if include_sky
            else "background's residual region is non-disabled (sky pixels must be excluded from "
            "the residual — they are in nobody's mask when the sky model is disabled)"
        )
        raise ValueError(
            f"component_only: {reason}, but the dataset's semantic class map has no "
            f"'{SKY_CLASS_NAME}' class (available: {available}) — sky pixels cannot be identified."
        )

    return ComponentOnlyMaskSpec(
        supervised_layers=supervised,
        carved_layers=carved,
        disabled_track_regions=disabled_track_regions,
        include_residual=include_residual,
        include_sky=include_sky,
        sky_class_id=sky_class_id,
        # Ego exclusion via semantics is best-effort in pure-bbox runs
        # (no map): the dedicated ego mask earlier in the pipeline
        # remains the primary ego handling either way.
        ego_class_id=semantic_class_id_map.get(EGO_CLASS_NAME) if semantic_class_id_map is not None else None,
    )


def _semantic_region(semantic: Tensor, class_ids: tuple[int, ...]) -> Tensor:
    ids = torch.tensor(sorted(class_ids), dtype=semantic.dtype, device=semantic.device)
    return torch.isin(semantic, ids)


def _layer_region(
    layer_id: str,
    spec: LayerRegionSpec,
    semantic: Optional[Tensor],
    track_regions: Mapping[str, ProjectedTrackRegion],
) -> Tensor:
    """Combine a layer's mask sources into its (B, H, W) region."""
    parts: list[Tensor] = []
    for source in spec.sources:
        if source == MaskSource.semantic:
            assert semantic is not None, (
                f"layer '{layer_id}' has a semantic mask source but no per-pixel semantic labels "
                f"were provided — the capability check at construction should have caught this"
            )
            region = _semantic_region(semantic, spec.semantic_class_ids)
            if spec.semantic_bbox_key:
                # Instance-scope the class mask to the enabled tracks' boxes so a
                # disabled same-class instance is not supervised via semantics.
                if spec.semantic_bbox_key not in track_regions:
                    raise KeyError(
                        f"region '{spec.semantic_bbox_key}' scopes a semantic mask but no bbox pixel "
                        f"mask was provided for it (available: {sorted(track_regions)})"
                    )
                region = region & track_regions[spec.semantic_bbox_key].mask
            parts.append(region)
        else:  # bbox / bbox_expanded — variant already applied by the producer
            bbox_key = spec.bbox_key or layer_id
            if bbox_key not in track_regions:
                raise KeyError(
                    f"region '{bbox_key}' has a bbox mask source but no bbox pixel mask was "
                    f"provided for it (available: {sorted(track_regions)})"
                )
            parts.append(track_regions[bbox_key].mask)
    assert parts, f"layer '{layer_id}' resolved to no mask sources"
    region = parts[0]
    for part in parts[1:]:
        region = region & part if spec.operation == MasksOperation.intersection else region | part
    return region


def build_camera_loss_mask(
    spec: ComponentOnlyMaskSpec,
    semantic: Optional[Tensor],
    track_regions: Mapping[str, ProjectedTrackRegion],
) -> Tensor:
    """Assemble the final (B, H, W) bool camera loss mask.

    Args:
        spec: static spec from ``build_mask_spec``.
        semantic: (B, H, W, 1) or (B, H, W) integer per-pixel class ids,
            or None when ``spec.needs_semantic`` is False.
        track_regions: projected cuboid mask and nearest depth per required
            supervised, carved, and disabled-track region. Mask recipes keep
            their configured raw/expanded geometry; misses have ``+inf`` depth.

    Returns:
        (B, H, W) bool — True where the pixel contributes to the loss.
    """
    if spec.needs_semantic:
        assert semantic is not None, (
            "component-only mask requires per-pixel semantic labels — the dataset capability "
            "check at construction should have refused to train"
        )
    if semantic is not None and semantic.ndim == 4:
        semantic = semantic.squeeze(-1)
    if semantic is not None:
        semantic = semantic.to(torch.int64)

    for key, projected in track_regions.items():
        if not isinstance(projected, ProjectedTrackRegion):
            raise TypeError(
                f"component-only projected region '{key}' must be ProjectedTrackRegion, got {type(projected).__name__}"
            )
        if projected.mask.dtype != torch.bool:
            raise TypeError(f"component-only projected region '{key}' mask must be bool")
        if not projected.nearest_depth.is_floating_point():
            raise TypeError(f"component-only projected region '{key}' nearest_depth must be floating point")

    assert semantic is not None or track_regions, "cannot infer mask shape: no semantic labels and no track regions"
    reference = semantic if semantic is not None else next(iter(track_regions.values())).mask
    shape = reference.shape
    device = reference.device

    for key, projected in track_regions.items():
        if projected.mask.shape != shape or projected.nearest_depth.shape != shape:
            raise ValueError(
                f"component-only projected region '{key}' shape mismatch: expected {shape}, "
                f"got mask={projected.mask.shape}, depth={projected.nearest_depth.shape}"
            )
        if projected.mask.device != device or projected.nearest_depth.device != device:
            raise ValueError(f"component-only projected region '{key}' must be on {device}")

    def _empty() -> Tensor:
        return torch.zeros(shape, dtype=torch.bool, device=device)

    def _infinite_depth() -> Tensor:
        return torch.full(shape, torch.inf, dtype=torch.float32, device=device)

    def _required_track_region(key: str) -> ProjectedTrackRegion:
        if key not in track_regions:
            raise KeyError(
                f"supervised tracked region '{key}' requires projected cuboid depth, but no region "
                f"was provided (available: {sorted(track_regions)})"
            )
        return track_regions[key]

    supervised_union = _empty()
    supervised_track_depth = _infinite_depth()
    for layer_id, layer_groups in spec.supervised_layers.items():
        for group in layer_groups:
            group_region = _layer_region(layer_id, group, semantic, track_regions)
            supervised_union |= group_region
            if group.track_depth_key:
                projected = _required_track_region(group.track_depth_key)
                group_has_depth = group_region & projected.mask & torch.isfinite(projected.nearest_depth)
                group_depth = torch.where(group_has_depth, projected.nearest_depth, torch.inf)
                supervised_track_depth = torch.minimum(supervised_track_depth, group_depth)

    disabled_track_depth = _infinite_depth()
    for disabled_key in spec.disabled_track_regions:
        projected = _required_track_region(disabled_key)
        depth = torch.where(
            projected.mask & torch.isfinite(projected.nearest_depth), projected.nearest_depth, torch.inf
        )
        disabled_track_depth = torch.minimum(disabled_track_depth, depth)

    has_depth_ordering = torch.isfinite(supervised_track_depth) & torch.isfinite(disabled_track_depth)
    disabled_in_front = has_depth_ordering & (
        disabled_track_depth <= supervised_track_depth + ACTOR_DEPTH_TIE_TOLERANCE_M
    )

    sky_region = _empty()
    if spec.sky_class_id is not None and semantic is not None:
        sky_region = semantic == spec.sky_class_id

    final = supervised_union & ~disabled_in_front
    if spec.include_sky:
        final |= sky_region

    if spec.include_residual:
        claimed = supervised_union.clone()
        for layer_id, layer_spec in spec.carved_layers.items():
            claimed |= _layer_region(layer_id, layer_spec, semantic, track_regions)
        # Sky pixels never belong to the residual: they are the sky
        # module's region when it exists and nobody's when it doesn't.
        claimed |= sky_region
        final |= ~claimed

    if spec.ego_class_id is not None and semantic is not None:
        final &= semantic != spec.ego_class_id

    return final


def mask_has_any_kept_pixels(mask: Tensor) -> bool:
    """Cheap probe: True iff the mask supervises at least one pixel.

    Used by the training loop's early-stop accumulator: if the first N
    batches all yield zero kept pixels, the chosen components have
    nothing to train on in this dataset.
    """
    return bool(mask.any().item())


def apply_zero_mask(pred: Tensor, gt: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Zero ``pred`` and ``gt`` wherever ``mask`` is False.

    Both tensors are zeroed identically so any per-pixel loss (L1, L2,
    SSIM, LPIPS) produces zero contribution outside the mask:
    ``loss(0, 0) = 0`` for value losses; structural losses see the same
    all-zero pattern in both inputs. Returns new tensors.
    """
    if mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    keep = mask.to(pred.dtype)
    return pred * keep, gt * keep
