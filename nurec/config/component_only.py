# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration schema for component-only optimization.

Component-only optimization trains an arbitrary subset of the
composite's components (Gaussian layers, sky model, calibration,
post-processing) in isolation, with the photometric loss masked to
exactly the pixels the enabled components are responsible for.

The ``component_only`` section is the single source of truth for what
*exists in a run*; the ``model.*`` sections keep describing what each
component *is* and are never modified by component-only overlays.

Semantics overview:

* ``component_only.enabled: false`` (default) — section is inert,
  everything trains, no masking. Backward-compatible path.
* ``component_only.enabled: true`` — opt-in mode. A layer that is
  declared in ``model.layers`` but NOT listed under
  ``component_only.layers`` is **disabled**. A key listed here that
  does not exist in ``model.layers`` is a validation error (typo
  protection). Layers absent from ``model.layers`` altogether simply
  don't exist: they contribute nothing anywhere and their image
  content is residual (owned by ``background``), exactly as in full
  training.

Camera-supervision controls are nested under
``component_only.camera_supervision``:

* ``frame_selection: all_frames`` (default) preserves the existing sampling
  distribution. Frames whose final component mask is empty still consume
  a training step.
* ``frame_selection: supervised_only`` samples uniformly from the eligible
  ``(camera, frame)`` pairs whose full-image component mask is non-empty.
  Eligibility is computed once at startup. If calibration or track updates later
  make a selected frame's mask empty, training fails before backward instead of
  executing a zero-supervision optimizer step.
* ``loss_normalization: supervised_pixels`` (default) averages camera losses
  over the component mask. ``all_valid_pixels`` instead keeps the numerator
  masked but uses the original valid-pixel population as denominator, matching
  full-training gradient scale. That scale does not make a frozen or
  overlapping render equivalent to removing it from the composite.
* ``supervised_only.scale_samples_per_epoch_by_frame_selection_fraction: true``
  scales the configured samples per epoch by the eligible/all reachable-frame
  fraction. It is only meaningful with ``frame_selection: supervised_only``.
  Step-interval LR schedulers are advanced to the same position per epoch as
  with the configured sample count; epoch-interval LR schedulers are unchanged.
  Other global-step schedules retain their configured step coordinates.

Component states:

* ``enabled`` — constructed, rendered, optimized; its mask region is
  part of the loss union.
* ``frozen`` — constructed and rendered, but not optimized (no
  gradients, excluded from densification/pruning); its mask region
  REMAINS in the loss union so enabled components train around its
  render. Note: enabled components receive gradient to compensate for
  a frozen render's errors inside its region — freezing a poorly
  trained component invites bleed.
* ``disabled`` — absent. Not constructed, nothing rendered; its mask
  region is in NOBODY's mask (unsupervised), and in particular it is
  NOT part of ``background``'s residual.
"""

from __future__ import annotations

from enum import Enum, StrEnum, auto
from typing import Any, Mapping, Optional

from pydantic import model_validator

from nurec.config.base_schema import BaseConfigSchema, Field


class ComponentState(str, Enum):
    """Tri-state of a component under component-only optimization."""

    enabled = "enabled"
    disabled = "disabled"
    frozen = "frozen"


class CameraFrameSelection(StrEnum):
    """Which camera/frame pairs component-only training may sample."""

    all_frames = auto()
    supervised_only = auto()


class CameraLossNormalization(StrEnum):
    """Denominator used by component-masked camera losses."""

    supervised_pixels = auto()
    all_valid_pixels = auto()


class MaskSource(str, Enum):
    """Sources a layer's loss-mask region can be built from.

    * ``semantic`` — pixels of the layer's semantic classes, read from
      the layer's existing ``class_labels`` field in ``model.layers``
      (only legal for layers that declare it, e.g. ``road``,
      ``traffic_light``).
    * ``bbox`` — pixels inside the layer's projected cuboid tracks
      (only legal for track-attached layers).
    * ``bbox_expanded`` — like ``bbox`` but with each cuboid scaled by
      the layer's ``cuboid_tracks_expand`` factor (only legal for
      layers that declare it, i.e. deformables). Exactly one of
      ``bbox`` / ``bbox_expanded`` may appear in a masks list: one
      strictly contains the other, so combining them is degenerate.
    """

    semantic = "semantic"
    bbox = "bbox"
    bbox_expanded = "bbox_expanded"


class MasksOperation(str, Enum):
    """How multiple mask sources combine into the layer's region."""

    intersection = "intersection"
    union = "union"


class ScopeRegion(str, Enum):
    """Regions of the ``background`` layer's Gaussian population.

    ``background`` has no mask source of its own — it is the residual
    bucket by definition — so its participation is expressed as a
    mapping of population *regions* to states:

    * ``residual`` — the complement region: everything no other
      component claims. Carve-out is state-independent: enabled,
      frozen AND disabled siblings' regions are all carved out
      (disabled means *nobody* owns a region, not that background
      does). Contributes the complement pixels to the loss mask.
    * ``bbox`` — the union of rays through the pixels of the
      non-disabled track-attached siblings' bbox masks. IMPORTANT:
      ``bbox`` follows each sibling's configured masks variant — a
      deformables sibling configured with ``bbox_expanded`` contributes
      EXPANDED frustums, one configured with raw ``bbox`` contributes
      raw frustums. Invariant: the scope region is the unprojection of
      exactly the pixels the siblings' masks supervise; this guarantees
      every supervised box pixel has background candidates and no
      background Gaussian sits in never-supervised space. This region
      contributes NO extra pixels to the loss mask — it exists so that
      box-interior background content (ground under a car, wall behind
      it) has a legitimate trainable owner instead of bleeding into the
      dynamic layers' Gaussians.

    Region-state independence: a sibling's state decides only whether
    its frustum is PART OF the bbox region; the state of background
    Gaussians inside that region comes solely from this mapping. (A
    frozen sibling still contributes its frustum — its box pixels are
    supervised, and the non-object content behind the frozen render
    needs a trainable owner.) Scope regions are deliberately the
    finest background partition — there are no per-track scope states.
    """

    residual = "residual"
    bbox = "bbox"


# Default masks per canonical layer id. Used in two places:
# (a) an enabled/frozen component_only entry that omits ``masks``,
# (b) the complement / scope contributions of layers that are unlisted
#     or disabled (their region must still be well-defined so it can be
#     carved out of ``background``'s residual).
# A layer listed with explicit ``masks`` uses those in both roles.
#
# Rationale for the choices:
# dynamics are bbox-only because no per-layer pixel-class contract
# exists (the static/dynamic rigids split is per-cuboid motion, not
# expressible in semantics); deformables default to the expanded box
# because that is where the deform network can actually place
# Gaussians; traffic_light is the only layer with both vocabularies
# declared, so it gets the most conservative default (union).
DEFAULT_LAYER_MASKS: dict[str, tuple[tuple[MaskSource, ...], Optional[MasksOperation]]] = {
    "road": ((MaskSource.semantic,), None),
    "dynamic_rigids": ((MaskSource.bbox,), None),
    "static_rigids": ((MaskSource.bbox,), None),
    "vehicle_lights": ((MaskSource.bbox,), None),
    "static_vehicle_lights": ((MaskSource.bbox,), None),
    "dynamic_deformables": ((MaskSource.bbox_expanded,), None),
    "traffic_light": ((MaskSource.semantic, MaskSource.bbox), MasksOperation.union),
    "loaded_ply_assets": ((MaskSource.bbox,), None),
}

# The one layer whose region is derived (complement) rather than
# declared; the only layer on which ``scope`` is legal.
BACKGROUND_LAYER_ID = "background"


def _validate_masks_fields(
    masks: Optional[list[MaskSource]],
    masks_operation: Optional[MasksOperation],
    context: str,
) -> None:
    """Shared masks/masks_operation consistency rules for layer and track entries."""
    if masks is not None:
        if len(masks) == 0:
            raise ValueError(f"{context}: 'masks' must not be empty; omit the field to use the layer default")
        if len(set(masks)) != len(masks):
            raise ValueError(f"{context}: duplicate entries in 'masks' {[m.value for m in masks]}")
        if MaskSource.bbox in masks and MaskSource.bbox_expanded in masks:
            raise ValueError(
                f"{context}: 'bbox' and 'bbox_expanded' cannot be combined — one strictly "
                f"contains the other (intersection degenerates to bbox, union to bbox_expanded); "
                f"pick exactly one"
            )
        if len(masks) > 1 and masks_operation is None:
            raise ValueError(f"{context}: 'masks_operation' is required when more than one mask source is listed")
        if len(masks) == 1 and masks_operation is not None:
            raise ValueError(f"{context}: 'masks_operation' is meaningless for a single mask source; remove it")
    elif masks_operation is not None:
        raise ValueError(f"{context}: 'masks_operation' without 'masks' — either set both or neither")


class ComponentOnlyTrackConfig(BaseConfigSchema):
    """Per-track override inside a track-attached layer's entry.

    The shape mirrors ``ComponentOnlyLayerConfig`` one level down:
    omitted fields inherit from the layer entry. Selection is by
    explicit track id only (deterministic — no random-K subsetting).
    Unselected tracks follow the layer's ``state`` default; a track
    that ends up disabled is a disabled partition: its box pixels are
    in nobody's mask and are carved out of ``background``'s residual
    (a disabled track is never baked into the background).
    """

    # None (the default) inherits the layer entry's state, so a mask-only
    # override under a frozen/disabled layer keeps that state instead of
    # silently becoming enabled. The resolver applies the inheritance.
    state: Optional[ComponentState] = None
    masks: Optional[list[MaskSource]] = Field(
        default=None,
        description=(
            "Per-track mask override; None inherits the layer entry. Only meaningful where the "
            "layer has options (deformables: bbox XOR bbox_expanded; traffic_light: semantic|bbox)."
        ),
    )
    masks_operation: Optional[MasksOperation] = None

    @model_validator(mode="after")
    def _check_masks(self) -> "ComponentOnlyTrackConfig":
        _validate_masks_fields(self.masks, self.masks_operation, "component_only track entry")
        return self


class ComponentOnlyLayerConfig(BaseConfigSchema):
    """Per-layer entry under ``component_only.layers``."""

    state: ComponentState = ComponentState.enabled

    masks: Optional[list[MaskSource]] = Field(
        default=None,
        description=(
            "Mask sources defining this layer's loss-mask region; None uses DEFAULT_LAYER_MASKS. "
            "'semantic' requires the layer to declare class_labels in model.layers; "
            "'bbox'/'bbox_expanded' require a track-attached layer."
        ),
    )
    masks_operation: Optional[MasksOperation] = Field(
        default=None,
        description="Required iff len(masks) > 1.",
    )

    scope: Optional[dict[ScopeRegion, ComponentState]] = Field(
        default=None,
        description=(
            "Background only: mapping of population region -> state. See ScopeRegion for the "
            "region semantics, the sibling-variant rule for 'bbox', and the region-state "
            "independence rule. A background entry with 'state' and no 'scope' is sugar for "
            "scope == {residual: <state>}."
        ),
    )

    tracks: Optional[dict[str, ComponentOnlyTrackConfig]] = Field(
        default=None,
        description=(
            "Track-attached layers only: per-track overrides keyed by track id. 'state' above "
            "is the default for unlisted tracks."
        ),
    )

    @model_validator(mode="after")
    def _check_masks(self) -> "ComponentOnlyLayerConfig":
        _validate_masks_fields(self.masks, self.masks_operation, "component_only layer entry")
        if self.scope is not None and len(self.scope) == 0:
            raise ValueError(
                "component_only layer entry: empty 'scope' — omit the entry entirely to disable background"
            )
        return self


class ComponentOnlyComponentConfig(BaseConfigSchema):
    """Entry for the non-layer components (sky_model, calib, post_processing).

    These components own no pixel region except ``sky_model``, which is
    treated like a semantically-defined layer with the implicit class
    ``"sky"``: enabled/frozen -> sky pixels are part of the final loss
    mask; disabled -> sky pixels are in nobody's mask (there is no sky
    to render, so nothing may train on them — not even background).
    ``post_processing`` toggles the whole chain (no per-entry toggle).
    """

    state: ComponentState = ComponentState.enabled


class ComponentOnlySupervisedOnlyConfig(BaseConfigSchema):
    """Controls that apply only when supervised camera frames are selected."""

    scale_samples_per_epoch_by_frame_selection_fraction: bool = Field(
        default=False,
        description=(
            "Scale dataset.n_samples_per_epoch by the eligible/all reachable camera-frame "
            "fraction. False preserves the configured training-step budget. The shorter "
            "epoch changes the number of optimizer steps. Step-interval LR schedulers preserve "
            "their configured progress per epoch; other global-step schedules are unchanged."
        ),
    )


class ComponentOnlyCameraSupervisionConfig(BaseConfigSchema):
    """Camera sampling and loss scaling for component-only optimization."""

    frame_selection: CameraFrameSelection = Field(
        default=CameraFrameSelection.all_frames,
        description=(
            "'all_frames' preserves the normal camera-frame distribution, including empty component "
            "masks. 'supervised_only' samples only full images with a non-empty final component mask."
        ),
    )
    loss_normalization: CameraLossNormalization = Field(
        default=CameraLossNormalization.supervised_pixels,
        description=(
            "'supervised_pixels' averages over the component mask. 'all_valid_pixels' keeps the "
            "masked numerator but uses the original valid-pixel denominator, matching the loss "
            "scale of full training."
        ),
    )
    supervised_only: ComponentOnlySupervisedOnlyConfig = Field(
        default_factory=ComponentOnlySupervisedOnlyConfig,
        description="Controls used only by frame_selection='supervised_only'.",
    )


class ComponentOnlyConfig(BaseConfigSchema):
    """Top-level ``component_only`` section."""

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch. False: this section is inert and training behaves exactly as "
            "without it. True: opt-in mode — every component not listed here is disabled."
        ),
    )

    layers: dict[str, ComponentOnlyLayerConfig] = Field(
        default_factory=dict,
        description=(
            "Per-layer entries keyed by model.layers id. Keys that do not exist in model.layers "
            "are a validation error; existing-but-unlisted layers are disabled."
        ),
    )

    camera_supervision: ComponentOnlyCameraSupervisionConfig = Field(
        default_factory=ComponentOnlyCameraSupervisionConfig,
        description=(
            "Controls camera-frame eligibility and component-masked camera-loss normalization. "
            "Defaults preserve existing component-only behavior."
        ),
    )

    sky_model: Optional[ComponentOnlyComponentConfig] = Field(
        default=None, description="Sky/background rendering module. Unlisted -> disabled (SkipBackground)."
    )
    calib: Optional[ComponentOnlyComponentConfig] = Field(
        default=None, description="Calibration component. Unlisted -> disabled (skip-calib)."
    )
    post_processing: Optional[ComponentOnlyComponentConfig] = Field(
        default=None, description="Post-processing chain (whole chain). Unlisted -> disabled (empty chain)."
    )


def resolve_layer_masks(
    layer_id: str,
    entry: Optional[ComponentOnlyLayerConfig],
    track_entry: Optional[ComponentOnlyTrackConfig] = None,
) -> tuple[tuple[MaskSource, ...], Optional[MasksOperation]]:
    """Resolve the effective mask sources for a layer (or one of its tracks).

    Precedence: track entry -> layer entry -> ``DEFAULT_LAYER_MASKS``.
    Used both for enabled/frozen layers' loss-mask regions and for the
    complement/scope contributions of unlisted or disabled layers.
    """
    if track_entry is not None and track_entry.masks is not None:
        return tuple(track_entry.masks), track_entry.masks_operation
    if entry is not None and entry.masks is not None:
        return tuple(entry.masks), entry.masks_operation
    if layer_id in DEFAULT_LAYER_MASKS:
        return DEFAULT_LAYER_MASKS[layer_id]
    raise ValueError(
        f"No default masks known for layer '{layer_id}' and none configured. "
        f"Layers with defaults: {sorted(DEFAULT_LAYER_MASKS)}. "
        f"Set component_only.layers.{layer_id}.masks explicitly."
    )


def _effective_track_state(entry: ComponentOnlyLayerConfig, track_entry: ComponentOnlyTrackConfig) -> ComponentState:
    """Resolve a per-track state, inheriting the layer's ``state`` when omitted (``None``)."""
    return track_entry.state if track_entry.state is not None else entry.state


def _configured_track_state(entry: ComponentOnlyLayerConfig, track_id: str) -> ComponentState:
    """Resolve one configured track id, falling back to the layer default."""
    track_entry = entry.tracks.get(track_id) if entry.tracks is not None else None
    return _effective_track_state(entry, track_entry) if track_entry is not None else entry.state


def _validate_motion_source_component_states(
    component_only: ComponentOnlyConfig,
    model_layers: Mapping[str, Any],
) -> None:
    """Protect frozen motion-dependent populations from a trainable owner."""
    for dependent_layer_id, dependent_layer_config in model_layers.items():
        source_layer_id = getattr(dependent_layer_config, "motion_source_layer", None)
        if source_layer_id is None:
            continue

        dependent_entry = component_only.layers.get(dependent_layer_id)
        source_entry = component_only.layers.get(source_layer_id)
        # An unlisted layer is disabled in component-only mode. If either
        # side is absent there is no frozen dependent rendering through an
        # enabled owner (unavailable dependents are pruned at construction).
        if dependent_entry is None or source_entry is None:
            continue

        populations: list[tuple[str | None, ComponentState, ComponentState]] = [
            (None, dependent_entry.state, source_entry.state)
        ]
        explicit_track_ids = sorted(set(dependent_entry.tracks or {}) | set(source_entry.tracks or {}))
        populations.extend(
            (
                track_id,
                _configured_track_state(dependent_entry, track_id),
                _configured_track_state(source_entry, track_id),
            )
            for track_id in explicit_track_ids
        )

        for track_id, dependent_state, source_state in populations:
            if dependent_state != ComponentState.frozen or source_state != ComponentState.enabled:
                continue
            population = "default track population" if track_id is None else f"track '{track_id}'"
            raise ValueError(
                f"component_only.layers.{dependent_layer_id}: frozen {population} depends on enabled "
                f"motion source '{source_layer_id}'. The dependent layer reuses "
                f"model.layers.{source_layer_id}.tracks_calib, so its rendered motion could change "
                f"and gradients could update the enabled owner. Freeze the matching motion-source "
                f"population, or enable/disable the dependent population."
            )


def _entry_has_non_disabled_population(entry: ComponentOnlyLayerConfig) -> bool:
    """True if any part of the layer's population is enabled or frozen.

    With per-track subsetting the layer's ``state`` is only the default
    for unlisted tracks — a layer with ``state: disabled`` but enabled
    track overrides still contributes those tracks' regions.
    """
    if entry.state != ComponentState.disabled:
        return True
    if entry.tracks:
        return any(_effective_track_state(entry, track) != ComponentState.disabled for track in entry.tracks.values())
    return False


def _entry_has_disabled_partition(entry: ComponentOnlyLayerConfig) -> bool:
    """True if some of the layer's population is disabled (a partition to carve).

    Config-derivable (no dataset needed): the layer default is ``disabled``
    (so unlisted/non-overridden tracks are disabled), or an explicit per-track
    override resolves to disabled. An over-emitted carve for a partition that
    turns out empty against the dataset yields an empty mask — harmless.
    """
    if entry.state == ComponentState.disabled:
        return True
    if entry.tracks:
        return any(_effective_track_state(entry, t) == ComponentState.disabled for t in entry.tracks.values())
    return False


def _has_bbox_masked_population(layer_id: str, entry: ComponentOnlyLayerConfig) -> bool:
    """True if any non-disabled part of the layer supplies a bbox-type mask.

    A sibling counts toward background's ``bbox`` scope frustum region if
    its non-disabled population resolves to ``bbox``/``bbox_expanded`` —
    either at the layer level OR via an enabled per-track override. The
    layer-level masks alone are not sufficient: a layer with
    semantic-only layer masks but an enabled ``bbox`` track override does
    supply the frustum (matching how the background bbox region is
    actually built from siblings' per-track bbox masks).
    """
    bbox_sources = (MaskSource.bbox, MaskSource.bbox_expanded)
    if entry.state != ComponentState.disabled and any(
        m in bbox_sources for m in resolve_layer_masks(layer_id, entry)[0]
    ):
        return True
    if entry.tracks:
        for track_entry in entry.tracks.values():
            if _effective_track_state(entry, track_entry) != ComponentState.disabled and any(
                m in bbox_sources for m in resolve_layer_masks(layer_id, entry, track_entry)[0]
            ):
                return True
    return False


def _layer_has_tracks(layer_config: Any) -> bool:
    return getattr(layer_config, "tracks", None) is not None


def _layer_has_class_labels(layer_config: Any) -> bool:
    return bool(getattr(layer_config, "class_labels", None))


def _layer_has_expand_factor(layer_config: Any) -> bool:
    return getattr(layer_config, "cuboid_tracks_expand", None) is not None


def _validate_layer_mask_sources(
    layer_id: str,
    masks: tuple[MaskSource, ...],
    layer_config: Any,
    context: str,
) -> None:
    """Mask sources must be backed by declarations on the model-side layer config."""
    if MaskSource.semantic in masks and not _layer_has_class_labels(layer_config):
        raise ValueError(
            f"{context}: 'semantic' mask on layer '{layer_id}' requires model.layers.{layer_id}."
            f"class_labels to be declared (no per-layer pixel-class contract exists otherwise)"
        )
    if (MaskSource.bbox in masks or MaskSource.bbox_expanded in masks) and not _layer_has_tracks(layer_config):
        raise ValueError(
            f"{context}: bbox mask on layer '{layer_id}' requires a track-attached layer "
            f"(model.layers.{layer_id} declares no tracks)"
        )
    if MaskSource.bbox_expanded in masks and not _layer_has_expand_factor(layer_config):
        raise ValueError(
            f"{context}: 'bbox_expanded' on layer '{layer_id}' requires model.layers.{layer_id}."
            f"cuboid_tracks_expand (deformable layers only)"
        )


def validate_component_only(
    component_only: ComponentOnlyConfig,
    model_layers: Mapping[str, Any],
) -> None:
    """Cross-section validation of ``component_only`` against ``model.layers``.

    Dataset-independent checks only; semantic class names are validated
    against the dataset's class map at model construction (fail-loud,
    same pattern as ``NodeSemanticGaussiansLoss.initialize``).

    Raises ``ValueError`` with the list of valid alternatives on every
    failure — never warn-and-continue.
    """
    if not component_only.enabled:
        return

    camera_supervision = component_only.camera_supervision
    if (
        camera_supervision.supervised_only.scale_samples_per_epoch_by_frame_selection_fraction
        and camera_supervision.frame_selection != CameraFrameSelection.supervised_only
    ):
        raise ValueError(
            "component_only.camera_supervision.supervised_only."
            "scale_samples_per_epoch_by_frame_selection_fraction requires "
            "component_only.camera_supervision.frame_selection='supervised_only'"
        )

    available = sorted(model_layers.keys())

    for layer_id, entry in component_only.layers.items():
        ctx = f"component_only.layers.{layer_id}"

        if layer_id not in model_layers:
            raise ValueError(
                f"{ctx}: layer does not exist in model.layers (available: {available}). "
                f"Component-only entries must reference declared layers — this catches typos "
                f"and prevents silently training against a non-existent layer."
            )
        layer_config = model_layers[layer_id]

        if layer_id == BACKGROUND_LAYER_ID:
            if entry.masks is not None or entry.tracks is not None:
                raise ValueError(
                    f"{ctx}: background has no mask source of its own (it is the residual "
                    f"bucket) and no tracks; configure its participation via 'scope'"
                )
            continue

        if entry.scope is not None:
            raise ValueError(
                f"{ctx}: 'scope' is only legal on '{BACKGROUND_LAYER_ID}' — other layers' "
                f"regions are declared via 'masks', not derived"
            )

        masks, _ = resolve_layer_masks(layer_id, entry)
        _validate_layer_mask_sources(layer_id, masks, layer_config, ctx)

        if entry.tracks is not None:
            if not _layer_has_tracks(layer_config):
                raise ValueError(f"{ctx}: per-track overrides on a layer without tracks")
            for track_id, track_entry in entry.tracks.items():
                track_masks, _ = resolve_layer_masks(layer_id, entry, track_entry)
                _validate_layer_mask_sources(layer_id, track_masks, layer_config, f"{ctx}.tracks.{track_id}")

    _validate_motion_source_component_states(component_only, model_layers)

    background = component_only.layers.get(BACKGROUND_LAYER_ID)
    if background is not None and background.scope is not None and ScopeRegion.bbox in background.scope:
        bbox_masked_siblings = [
            layer_id
            for layer_id, entry in component_only.layers.items()
            if layer_id != BACKGROUND_LAYER_ID
            and _entry_has_non_disabled_population(entry)
            and _has_bbox_masked_population(layer_id, entry)
        ]
        if not bbox_masked_siblings:
            raise ValueError(
                "component_only.layers.background.scope: 'bbox' region requires at least one "
                "non-disabled sibling with a bbox-type mask — with none, the frustum region is "
                "empty and no background Gaussian could ever receive supervision there"
            )
