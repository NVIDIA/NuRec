# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nurec.utils.component_only_mask`` (pure tensor logic).

Model layers are duck-typed SimpleNamespace stand-ins exposing the
``class_labels`` and track metadata used by spec assembly. Projected
track regions carry both cuboid hit masks and nearest depths, mirroring
the production wiring contract.
"""

import unittest

from types import SimpleNamespace
from typing import cast

import torch

from nurec.config.component_only import (
    ComponentOnlyComponentConfig,
    ComponentOnlyConfig,
    ComponentOnlyLayerConfig,
    ComponentOnlyTrackConfig,
    ComponentState,
    MasksOperation,
    MaskSource,
    ScopeRegion,
)
from nurec.utils.component_only_mask import (
    ACTOR_DEPTH_TIE_TOLERANCE_M,
    DISABLED_TRACKS_SUFFIX,
    ProjectedTrackRegion,
    apply_zero_mask,
    build_mask_spec,
    mask_has_any_kept_pixels,
)
from nurec.utils.component_only_mask import (
    build_camera_loss_mask as _build_camera_loss_mask,
)


# Semantic vocabulary of the fake dataset: 2x4 image layouts below use
# these ids directly.
CLASS_MAP = {"road": 0, "car": 1, "building": 2, "sky": 3, "egocar": 4, "traffic light": 5}


def _model_layers() -> dict:
    return {
        "background": SimpleNamespace(class_labels=None, tracks=None),
        "road": SimpleNamespace(class_labels=["road"], tracks=None),
        "dynamic_rigids": SimpleNamespace(class_labels=None, tracks=object()),
        "traffic_light": SimpleNamespace(class_labels=["traffic light"], tracks=object()),
    }


def _semantic() -> torch.Tensor:
    # (1, 2, 4): row 0 = [road, car, building, sky], row 1 = [egocar, road, building, traffic light]
    return torch.tensor([[[0, 1, 2, 3], [4, 0, 2, 5]]], dtype=torch.uint8)


def _rigids_bbox() -> torch.Tensor:
    # Covers the 'car' pixel (0,1) plus the 'building' pixel behind it (0,2).
    mask = torch.zeros((1, 2, 4), dtype=torch.bool)
    mask[0, 0, 1] = True
    mask[0, 0, 2] = True
    return mask


def _spec(config: ComponentOnlyConfig):
    spec = build_mask_spec(config, _model_layers(), CLASS_MAP)
    assert spec is not None
    return spec


def _projected(mask: torch.Tensor, depth: float | torch.Tensor = 1.0) -> ProjectedTrackRegion:
    if isinstance(depth, torch.Tensor):
        nearest_depth = depth.to(dtype=torch.float32)
    else:
        nearest_depth = torch.full(mask.shape, depth, dtype=torch.float32, device=mask.device)
    nearest_depth = torch.where(mask, nearest_depth, torch.inf)
    return ProjectedTrackRegion(mask=mask, nearest_depth=nearest_depth)


def build_camera_loss_mask(spec, semantic, masks):
    """Adapt legacy Boolean fixtures while keeping production strict.

    Existing Boolean-assembly tests should remain about their original mask
    recipes, so disabled test geometry defaults to empty and supplied disabled
    partitions default far behind rendered actors. New depth-ordering tests call
    the production function directly with explicit depths.
    """
    regions = {
        key: value
        if isinstance(value, ProjectedTrackRegion)
        else _projected(value, 1_000.0 if key.endswith(DISABLED_TRACKS_SUFFIX) else 1.0)
        for key, value in masks.items()
    }
    if semantic is not None:
        shape = semantic.shape[:-1] if semantic.ndim == 4 else semantic.shape
        device = semantic.device
    elif regions:
        first = next(iter(regions.values()))
        shape, device = first.mask.shape, first.mask.device
    else:
        shape, device = (1, 1, 1), torch.device("cpu")
    empty = torch.zeros(shape, dtype=torch.bool, device=device)
    for groups in spec.supervised_layers.values():
        for group in groups:
            # Legacy semantic-only fixtures had no projected-box input. The
            # production builder now supplies their cuboid depth explicitly;
            # use an empty projection here so these tests stay recipe-focused.
            if group.track_depth_key and not group.needs_bbox and not group.semantic_bbox_key:
                regions.setdefault(group.track_depth_key, _projected(empty))
    for key in spec.disabled_track_regions:
        regions.setdefault(key, _projected(empty))
    return _build_camera_loss_mask(spec, semantic, regions)


class TestBuildMaskSpec(unittest.TestCase):
    def test_disabled_returns_none(self) -> None:
        self.assertIsNone(build_mask_spec(ComponentOnlyConfig(enabled=False), _model_layers(), CLASS_MAP))

    def test_layer_states_partition(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "road": ComponentOnlyLayerConfig(),
                "dynamic_rigids": ComponentOnlyLayerConfig(state=ComponentState.frozen),
            },
        )
        spec = _spec(config)
        # road enabled + rigids frozen → both supervised; traffic_light
        # unlisted → carved. Background never appears in either.
        self.assertEqual(set(spec.supervised_layers), {"road", "dynamic_rigids"})
        self.assertEqual(set(spec.carved_layers), {"traffic_light"})
        self.assertFalse(spec.include_residual)
        self.assertFalse(spec.include_sky)

    def test_unknown_semantic_class_fails_loud(self) -> None:
        layers = _model_layers()
        layers["road"] = SimpleNamespace(class_labels=["road", "no_such_class"], tracks=None)
        config = ComponentOnlyConfig(enabled=True, layers={"road": ComponentOnlyLayerConfig()})
        with self.assertRaises(ValueError) as cm:
            build_mask_spec(config, layers, CLASS_MAP)
        self.assertIn("no_such_class", str(cm.exception))
        self.assertIn("road", str(cm.exception))

    def test_semantic_layer_with_empty_class_labels_fails_loud(self) -> None:
        # A semantic mask source with no bound classes would resolve to an
        # all-false region and silently drop the layer from supervision.
        layers = _model_layers()
        layers["road"] = SimpleNamespace(class_labels=[], tracks=None)
        config = ComponentOnlyConfig(enabled=True, layers={"road": ComponentOnlyLayerConfig()})
        with self.assertRaises(ValueError) as cm:
            build_mask_spec(config, layers, CLASS_MAP)
        self.assertIn("class_labels", str(cm.exception))
        self.assertIn("road", str(cm.exception))

    def test_sky_without_sky_class_fails_loud(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"road": ComponentOnlyLayerConfig()},
            sky_model=ComponentOnlyComponentConfig(),
        )
        class_map = {k: v for k, v in CLASS_MAP.items() if k != "sky"}
        with self.assertRaises(ValueError) as cm:
            build_mask_spec(config, _model_layers(), class_map)
        self.assertIn("sky", str(cm.exception))

    def test_background_state_sugar_sets_residual(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"background": ComponentOnlyLayerConfig(state=ComponentState.enabled)},
        )
        self.assertTrue(_spec(config).include_residual)

    def test_background_scope_residual_frozen_counts_as_included(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"background": ComponentOnlyLayerConfig(scope={ScopeRegion.residual: ComponentState.frozen})},
        )
        self.assertTrue(_spec(config).include_residual)

    def test_needs_semantic_false_for_pure_bbox_run(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"dynamic_rigids": ComponentOnlyLayerConfig()},
        )
        self.assertFalse(_spec(config).needs_semantic)

    def test_needs_semantic_true_when_residual_carves_semantic_sibling(self) -> None:
        # background residual must carve road's semantic region even
        # though road is disabled → semantics required.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(),
                "background": ComponentOnlyLayerConfig(),
            },
        )
        self.assertTrue(_spec(config).needs_semantic)

    def test_pure_bbox_run_builds_without_class_map(self) -> None:
        # No semantic sources needed anywhere → dataset without
        # per-pixel semantics is legal (map=None).
        config = ComponentOnlyConfig(enabled=True, layers={"dynamic_rigids": ComponentOnlyLayerConfig()})
        spec = build_mask_spec(config, _model_layers(), None)
        assert spec is not None
        self.assertFalse(spec.needs_semantic)
        self.assertIsNone(spec.ego_class_id)

    def test_semantic_layer_without_class_map_fails_loud(self) -> None:
        config = ComponentOnlyConfig(enabled=True, layers={"road": ComponentOnlyLayerConfig()})
        with self.assertRaises(ValueError) as cm:
            build_mask_spec(config, _model_layers(), None)
        self.assertIn("aux_data", str(cm.exception))

    def test_residual_without_class_map_fails_loud(self) -> None:
        # Residual requires semantics (sky exclusion + carving semantic
        # siblings like road) — the first consumer to resolve raises.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(),
                "background": ComponentOnlyLayerConfig(),
            },
        )
        with self.assertRaises(ValueError) as cm:
            build_mask_spec(config, _model_layers(), None)
        self.assertIn("semantic", str(cm.exception))

    def test_per_track_enabled_track_under_disabled_layer_is_supervised(self) -> None:
        # The layer default is only the default for unlisted tracks: an
        # enabled track override keeps the layer in the supervised set.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.enabled)},
                ),
            },
        )
        spec = _spec(config)
        self.assertIn("dynamic_rigids", spec.supervised_layers)

    def test_per_track_pseudo_carved_entry_emitted_with_residual(self) -> None:
        # With a residual, the disabled-track partition must be carved
        # via the pseudo entry (bbox sources only).
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "background": ComponentOnlyLayerConfig(),
                "dynamic_rigids": ComponentOnlyLayerConfig(
                    tracks={"t9": ComponentOnlyTrackConfig(state=ComponentState.disabled)},
                ),
            },
        )
        spec = _spec(config)
        pseudo = "dynamic_rigids" + DISABLED_TRACKS_SUFFIX
        self.assertIn(pseudo, spec.carved_layers)
        self.assertTrue(spec.carved_layers[pseudo].needs_bbox)
        self.assertFalse(spec.carved_layers[pseudo].needs_semantic)

    def test_per_track_no_pseudo_entry_without_residual(self) -> None:
        # Without a residual there is nothing to carve — the supervised
        # mask already excludes the disabled partition by construction.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(
                    tracks={"t9": ComponentOnlyTrackConfig(state=ComponentState.disabled)},
                ),
            },
        )
        spec = _spec(config)
        self.assertNotIn("dynamic_rigids" + DISABLED_TRACKS_SUFFIX, spec.carved_layers)

    def test_per_track_disabled_box_carved_even_under_semantic_layer(self) -> None:
        # A disabled track's box must be carved from the residual even when
        # the layer recipe is semantic-only — the box is the disabled
        # instance's geometry, and its pixels belong to nobody.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "background": ComponentOnlyLayerConfig(),
                "traffic_light": ComponentOnlyLayerConfig(
                    masks=[MaskSource.semantic],
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.disabled)},
                ),
            },
        )
        spec = _spec(config)
        self.assertIn("traffic_light", spec.supervised_layers)
        pseudo = "traffic_light" + DISABLED_TRACKS_SUFFIX
        self.assertIn(pseudo, spec.carved_layers)
        self.assertTrue(spec.carved_layers[pseudo].needs_bbox)

    def test_fully_disabled_layer_carves_per_track_bbox_override(self) -> None:
        # A FULLY disabled semantic-default layer with a disabled bbox-override
        # track: the track's box must still be carved from the residual (the
        # parent semantic recipe alone doesn't cover it).
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "background": ComponentOnlyLayerConfig(),
                "traffic_light": ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    masks=[MaskSource.semantic],
                    tracks={"t_off": ComponentOnlyTrackConfig(masks=[MaskSource.bbox])},
                ),
            },
        )
        spec = _spec(config)
        self.assertNotIn("traffic_light", spec.supervised_layers)  # fully disabled
        pseudo = "traffic_light" + DISABLED_TRACKS_SUFFIX
        self.assertIn(pseudo, spec.carved_layers)  # its box is still carved
        self.assertTrue(spec.carved_layers[pseudo].needs_bbox)

    def test_per_track_fully_disabled_layer_is_carved_whole(self) -> None:
        # Layer default disabled and every override disabled → no population;
        # the layer's base recipe is carved under its real key, and the
        # disabled partition's boxes (per each track's own variant) are carved
        # under the pseudo key so per-track overrides are honored.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "background": ComponentOnlyLayerConfig(),
                "road": ComponentOnlyLayerConfig(),
                "dynamic_rigids": ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.disabled)},
                ),
            },
        )
        spec = _spec(config)
        self.assertNotIn("dynamic_rigids", spec.supervised_layers)
        self.assertIn("dynamic_rigids", spec.carved_layers)
        self.assertIn("dynamic_rigids" + DISABLED_TRACKS_SUFFIX, spec.carved_layers)

    def test_residual_with_bbox_only_siblings_still_needs_sky_class(self) -> None:
        # Even when every existing sibling is bbox-masked, the residual
        # must identify sky pixels (nobody's mask when sky disabled).
        layers = {
            "background": SimpleNamespace(class_labels=None, tracks=None),
            "dynamic_rigids": SimpleNamespace(class_labels=None, tracks=object()),
        }
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(),
                "background": ComponentOnlyLayerConfig(),
            },
        )
        with self.assertRaises(ValueError) as cm:
            build_mask_spec(config, layers, None)
        self.assertIn("sky", str(cm.exception))


class TestBuildCameraLossMask(unittest.TestCase):
    def test_bbox_only_layer(self) -> None:
        config = ComponentOnlyConfig(enabled=True, layers={"dynamic_rigids": ComponentOnlyLayerConfig()})
        mask = build_camera_loss_mask(_spec(config), _semantic(), {"dynamic_rigids": _rigids_bbox()})
        # Exactly the bbox pixels (car + building-behind-it).
        expected = _rigids_bbox()
        self.assertTrue(torch.equal(mask, expected))

    def test_semantic_layer(self) -> None:
        config = ComponentOnlyConfig(enabled=True, layers={"road": ComponentOnlyLayerConfig()})
        # traffic_light unlisted (carved) but no residual → carve set unused.
        mask = build_camera_loss_mask(
            _spec(config), _semantic(), {"traffic_light": torch.zeros((1, 2, 4), dtype=torch.bool)}
        )
        expected = torch.zeros((1, 2, 4), dtype=torch.bool)
        expected[0, 0, 0] = True  # road pixel
        expected[0, 1, 1] = True  # road pixel
        self.assertTrue(torch.equal(mask, expected))

    def test_intersection_operation(self) -> None:
        # traffic_light with semantic ∩ bbox: bbox covers row 1 entirely;
        # semantic 'traffic light' is only pixel (1,3) → intersection = (1,3).
        tl_bbox = torch.zeros((1, 2, 4), dtype=torch.bool)
        tl_bbox[0, 1, :] = True
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    masks=[MaskSource.semantic, MaskSource.bbox],
                    masks_operation=MasksOperation.intersection,
                )
            },
        )
        mask = build_camera_loss_mask(_spec(config), _semantic(), {"traffic_light": tl_bbox})
        expected = torch.zeros((1, 2, 4), dtype=torch.bool)
        expected[0, 1, 3] = True
        self.assertTrue(torch.equal(mask, expected))

    def test_union_operation(self) -> None:
        tl_bbox = torch.zeros((1, 2, 4), dtype=torch.bool)
        tl_bbox[0, 0, 0] = True  # some box pixel that isn't traffic-light class
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    masks=[MaskSource.semantic, MaskSource.bbox],
                    masks_operation=MasksOperation.union,
                )
            },
        )
        mask = build_camera_loss_mask(_spec(config), _semantic(), {"traffic_light": tl_bbox})
        expected = torch.zeros((1, 2, 4), dtype=torch.bool)
        expected[0, 0, 0] = True  # bbox member
        expected[0, 1, 3] = True  # semantic member
        self.assertTrue(torch.equal(mask, expected))

    def test_sky_included_when_sky_model_on(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"road": ComponentOnlyLayerConfig()},
            sky_model=ComponentOnlyComponentConfig(),
        )
        mask = build_camera_loss_mask(
            _spec(config), _semantic(), {"traffic_light": torch.zeros((1, 2, 4), dtype=torch.bool)}
        )
        self.assertTrue(bool(mask[0, 0, 3]))  # sky pixel supervised

    def test_residual_carves_disabled_layers_and_sky_and_ego(self) -> None:
        # background residual + rigids enabled; road + traffic_light
        # unlisted (carved); sky module disabled.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(),
                "background": ComponentOnlyLayerConfig(),
            },
        )
        tl_bbox = torch.zeros((1, 2, 4), dtype=torch.bool)
        tl_bbox[0, 1, 3] = True
        mask = build_camera_loss_mask(
            _spec(config),
            _semantic(),
            {"dynamic_rigids": _rigids_bbox(), "traffic_light": tl_bbox},
        )
        expected = torch.zeros((1, 2, 4), dtype=torch.bool)
        expected[0, 0, 1] = True  # rigids bbox
        expected[0, 0, 2] = True  # rigids bbox
        # Residual: NOT(rigids bbox ∪ road semantic ∪ tl region ∪ sky) minus ego.
        # road pixels (0,0),(1,1) carved; sky (0,3) out; ego (1,0) out;
        # tl (1,3) carved. Remaining residual: building (1,2) only —
        # building (0,2) is already claimed by the rigids bbox.
        expected[0, 1, 2] = True
        self.assertTrue(torch.equal(mask, expected))

    def test_residual_carves_disabled_track_partition_via_pseudo_key(self) -> None:
        # rigids supervised with one disabled track; the disabled
        # track's box pixels must be in NOBODY's mask: excluded from the
        # supervised region (the wiring's per-track partition already
        # leaves them out of the real key's mask) AND carved from the
        # residual via the pseudo key.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "background": ComponentOnlyLayerConfig(),
                "dynamic_rigids": ComponentOnlyLayerConfig(
                    tracks={"t9": ComponentOnlyTrackConfig(state=ComponentState.disabled)},
                ),
            },
        )
        supervised_bbox = torch.zeros((1, 2, 4), dtype=torch.bool)
        supervised_bbox[0, 0, 1] = True  # enabled tracks' boxes
        disabled_bbox = torch.zeros((1, 2, 4), dtype=torch.bool)
        disabled_bbox[0, 1, 2] = True  # disabled track's box (building pixel)
        tl_bbox = torch.zeros((1, 2, 4), dtype=torch.bool)
        tl_bbox[0, 1, 3] = True
        mask = build_camera_loss_mask(
            _spec(config),
            _semantic(),
            {
                "dynamic_rigids": supervised_bbox,
                "dynamic_rigids" + DISABLED_TRACKS_SUFFIX: disabled_bbox,
                "traffic_light": tl_bbox,
            },
        )
        expected = torch.zeros((1, 2, 4), dtype=torch.bool)
        expected[0, 0, 1] = True  # supervised rigids box
        # Residual: NOT(rigids supervised ∪ disabled partition ∪ road
        # semantic ∪ tl ∪ sky) minus ego → building (0,2) survives,
        # building (1,2) is carved by the disabled track's box.
        expected[0, 0, 2] = True
        self.assertTrue(torch.equal(mask, expected))
        self.assertFalse(bool(mask[0, 1, 2]))

    def test_ego_always_excluded(self) -> None:
        # Even a residual-everything run never supervises ego pixels.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "road": ComponentOnlyLayerConfig(),
                "background": ComponentOnlyLayerConfig(),
            },
            sky_model=ComponentOnlyComponentConfig(),
        )
        tl_bbox = torch.zeros((1, 2, 4), dtype=torch.bool)
        mask = build_camera_loss_mask(
            _spec(config),
            _semantic(),
            {"dynamic_rigids": torch.zeros((1, 2, 4), dtype=torch.bool), "traffic_light": tl_bbox},
        )
        self.assertFalse(bool(mask[0, 1, 0]))  # egocar pixel

    def test_pure_bbox_run_without_semantic_tensor(self) -> None:
        config = ComponentOnlyConfig(enabled=True, layers={"dynamic_rigids": ComponentOnlyLayerConfig()})
        spec = _spec(config)
        mask = build_camera_loss_mask(spec, None, {"dynamic_rigids": _rigids_bbox()})
        self.assertTrue(torch.equal(mask, _rigids_bbox()))

    def test_missing_bbox_mask_raises(self) -> None:
        config = ComponentOnlyConfig(enabled=True, layers={"dynamic_rigids": ComponentOnlyLayerConfig()})
        with self.assertRaises(KeyError) as cm:
            build_camera_loss_mask(_spec(config), _semantic(), {})
        self.assertIn("dynamic_rigids", str(cm.exception))

    def test_4d_semantic_squeezed(self) -> None:
        config = ComponentOnlyConfig(enabled=True, layers={"road": ComponentOnlyLayerConfig()})
        semantic_4d = _semantic().unsqueeze(-1)
        mask = build_camera_loss_mask(
            _spec(config), semantic_4d, {"traffic_light": torch.zeros((1, 2, 4), dtype=torch.bool)}
        )
        self.assertEqual(mask.shape, (1, 2, 4))
        self.assertTrue(bool(mask[0, 0, 0]))


class TestPerTrackMaskRegions(unittest.TestCase):
    """Per-track ``masks`` overrides must reach the supervised loss mask.

    traffic_light has both semantic and bbox mask options, so it's the
    layer where a per-track semantic-vs-bbox override is meaningful.
    """

    def _spec_with_states(self, config, states):
        spec = build_mask_spec(config, _model_layers(), CLASS_MAP, layer_track_states=states)
        assert spec is not None
        return spec

    def test_single_group_without_per_track_masks(self) -> None:
        # Per-track STATE overrides (no masks) keep one layer-recipe group.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.enabled)}
                )
            },
        )
        spec = self._spec_with_states(config, {"traffic_light": {"t1": ComponentState.enabled}})
        groups = spec.supervised_layers["traffic_light"]
        self.assertEqual(len(groups), 1)
        self.assertIsNone(groups[0].bbox_track_ids)  # layer-level population
        self.assertEqual(groups[0].bbox_key, "traffic_light")

    def test_resolved_all_disabled_population_is_not_supervised(self) -> None:
        # The layer default applies only to unlisted dataset tracks. If every
        # track actually present in the resolved subset is explicitly disabled,
        # there is no rendered population even when the config default is
        # enabled for hypothetical unlisted tracks.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.disabled)}
                )
            },
        )
        spec = self._spec_with_states(config, {"traffic_light": {"t1": ComponentState.disabled}})
        self.assertNotIn("traffic_light", spec.supervised_layers)
        self.assertIn("traffic_light", spec.carved_layers)
        self.assertIn("traffic_light" + DISABLED_TRACKS_SUFFIX, spec.disabled_track_regions)

    def test_per_track_bbox_override_becomes_its_own_group(self) -> None:
        # Layer masks [semantic], one enabled bbox track: the group's
        # recipe is [bbox] over exactly that track.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    masks=[MaskSource.semantic],
                    tracks={"t1": ComponentOnlyTrackConfig(masks=[MaskSource.bbox])},
                )
            },
        )
        spec = self._spec_with_states(config, {"traffic_light": {"t1": ComponentState.enabled}})
        (group,) = spec.supervised_layers["traffic_light"]
        self.assertEqual(group.sources, (MaskSource.bbox,))
        self.assertEqual(group.bbox_track_ids, ("t1",))
        self.assertEqual(group.bbox_key, "traffic_light#0")
        self.assertFalse(group.needs_semantic)

    def test_bbox_track_under_semantic_layer_supervises_the_box(self) -> None:
        # The reported P1: a per-track [bbox] under a [semantic] layer
        # must supervise the box pixels, NOT the layer's semantic region.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    masks=[MaskSource.semantic],
                    # Explicit enabled: an omitted state now inherits the layer's
                    # disabled state (per the state-inheritance fix).
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.enabled, masks=[MaskSource.bbox])},
                )
            },
        )
        spec = self._spec_with_states(config, {"traffic_light": {"t1": ComponentState.enabled}})
        # The track's box sits on the 'car' pixel (0,1) — deliberately NOT
        # the traffic-light semantic pixel (1,3).
        box = torch.zeros((1, 2, 4), dtype=torch.bool)
        box[0, 0, 1] = True
        mask = build_camera_loss_mask(spec, _semantic(), {"traffic_light#0": box})
        self.assertTrue(bool(mask[0, 0, 1]))  # box pixel supervised
        self.assertFalse(bool(mask[0, 1, 3]))  # traffic-light semantic pixel NOT supervised

    def test_mixed_per_track_recipes_union_both_regions(self) -> None:
        # One track by bbox, one by semantic → two groups; the loss mask
        # is the union (the box ∪ the traffic-light class pixels).
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    tracks={
                        "t1": ComponentOnlyTrackConfig(state=ComponentState.enabled, masks=[MaskSource.bbox]),
                        "t2": ComponentOnlyTrackConfig(state=ComponentState.enabled, masks=[MaskSource.semantic]),
                    },
                )
            },
        )
        spec = self._spec_with_states(
            config, {"traffic_light": {"t1": ComponentState.enabled, "t2": ComponentState.enabled}}
        )
        groups = spec.supervised_layers["traffic_light"]
        self.assertEqual(len(groups), 2)
        bbox_group = next(g for g in groups if g.needs_bbox)
        box = torch.zeros((1, 2, 4), dtype=torch.bool)
        box[0, 0, 1] = True  # 'car' pixel
        mask = build_camera_loss_mask(spec, _semantic(), {bbox_group.bbox_key: box})
        self.assertTrue(bool(mask[0, 0, 1]))  # t1 box
        self.assertTrue(bool(mask[0, 1, 3]))  # t2 semantic (traffic-light pixel)

    def test_semantic_excludes_disabled_same_class_track(self) -> None:
        # Two same-class (traffic-light) tracks, one disabled: the class mask
        # must be instance-scoped to the enabled track's box so the disabled
        # instance's class pixels are NOT supervised.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    masks=[MaskSource.semantic],
                    tracks={
                        "t_on": ComponentOnlyTrackConfig(state=ComponentState.enabled),
                        "t_off": ComponentOnlyTrackConfig(state=ComponentState.disabled),
                    },
                )
            },
        )
        spec = self._spec_with_states(
            config, {"traffic_light": {"t_on": ComponentState.enabled, "t_off": ComponentState.disabled}}
        )
        (group,) = spec.supervised_layers["traffic_light"]
        self.assertEqual(group.semantic_bbox_key, "traffic_light")
        # Two traffic-light (class 5) pixels; the enabled track's box covers
        # only (0,0). The disabled instance's class pixel (0,1) must drop out.
        semantic = torch.tensor([[[5, 5]]], dtype=torch.uint8)
        enabled_box = torch.zeros((1, 1, 2), dtype=torch.bool)
        enabled_box[0, 0, 0] = True
        mask = build_camera_loss_mask(spec, semantic, {"traffic_light": enabled_box})
        self.assertTrue(bool(mask[0, 0, 0]))  # enabled instance supervised
        self.assertFalse(bool(mask[0, 0, 1]))  # disabled same-class instance NOT supervised

    def test_residual_carves_disabled_bbox_track_under_semantic_layer(self) -> None:
        # A disabled bbox-override track under a semantic-default layer with
        # residual background: its box must be carved from the residual, so
        # residual background does not supervise inside that disabled box.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "background": ComponentOnlyLayerConfig(),
                "traffic_light": ComponentOnlyLayerConfig(
                    masks=[MaskSource.semantic],
                    tracks={"t_off": ComponentOnlyTrackConfig(state=ComponentState.disabled, masks=[MaskSource.bbox])},
                ),
            },
        )
        spec = self._spec_with_states(config, {"traffic_light": {"t_off": ComponentState.disabled}})
        pseudo = "traffic_light" + DISABLED_TRACKS_SUFFIX
        self.assertIn(pseudo, spec.carved_layers)
        self.assertTrue(spec.carved_layers[pseudo].needs_bbox)
        # 'building' (class 2) is owned by nobody → residual. The disabled
        # track's box sits on (0,1); it must be carved out of the residual.
        semantic = torch.tensor([[[2, 2]]], dtype=torch.uint8)
        disabled_box = torch.zeros((1, 1, 2), dtype=torch.bool)
        disabled_box[0, 0, 1] = True
        empty = torch.zeros((1, 1, 2), dtype=torch.bool)
        mask = build_camera_loss_mask(
            spec,
            semantic,
            {pseudo: disabled_box, "dynamic_rigids": empty, "traffic_light": empty},
        )
        self.assertTrue(bool(mask[0, 0, 0]))  # residual building pixel supervised
        self.assertFalse(bool(mask[0, 0, 1]))  # inside disabled box → carved from residual

    def test_no_semantic_group_when_no_track_uses_semantic(self) -> None:
        # Layer default disabled + only a bbox track enabled → the layer
        # must NOT contribute its semantic class mask (no semantic group).
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    tracks={"t1": ComponentOnlyTrackConfig(state=ComponentState.enabled, masks=[MaskSource.bbox])},
                )
            },
        )
        spec = self._spec_with_states(
            config, {"traffic_light": {"t1": ComponentState.enabled, "t2": ComponentState.disabled}}
        )
        groups = spec.supervised_layers["traffic_light"]
        self.assertTrue(all(not g.needs_semantic for g in groups))
        self.assertFalse(spec.needs_semantic)


class TestDepthAwareActorOcclusion(unittest.TestCase):
    _SHAPE = (1, 1, 3)

    def _bbox_spec(
        self,
        *,
        selected_state: ComponentState = ComponentState.enabled,
        include_residual: bool = False,
    ):
        layers = {
            "background": SimpleNamespace(class_labels=None, tracks=None),
            "selected": SimpleNamespace(class_labels=None, tracks=object()),
            "occluder": SimpleNamespace(class_labels=None, tracks=object()),
        }
        config_layers = {
            "selected": ComponentOnlyLayerConfig(state=selected_state, masks=[MaskSource.bbox]),
            "occluder": ComponentOnlyLayerConfig(state=ComponentState.disabled, masks=[MaskSource.bbox]),
        }
        if include_residual:
            config_layers["background"] = ComponentOnlyLayerConfig()
        spec = build_mask_spec(ComponentOnlyConfig(enabled=True, layers=config_layers), layers, CLASS_MAP)
        assert spec is not None
        return spec

    def _regions(
        self,
        selected_depth: torch.Tensor,
        disabled_depth: torch.Tensor,
        *,
        include_carve: bool = False,
    ) -> dict[str, ProjectedTrackRegion]:
        selected_mask = torch.isfinite(selected_depth)
        disabled_mask = torch.isfinite(disabled_depth)
        regions = {
            "selected": _projected(selected_mask, selected_depth),
            "occluder" + DISABLED_TRACKS_SUFFIX: _projected(disabled_mask, disabled_depth),
        }
        if include_carve:
            regions["occluder"] = _projected(disabled_mask, disabled_depth)
        return regions

    def test_selected_actor_in_front_is_kept(self) -> None:
        spec = self._bbox_spec()
        regions = self._regions(
            torch.tensor([[[5.0, torch.inf, torch.inf]]]), torch.tensor([[[10.0, torch.inf, torch.inf]]])
        )
        mask = _build_camera_loss_mask(spec, None, regions)
        self.assertEqual(mask.tolist(), [[[True, False, False]]])

    def test_disabled_actor_in_front_is_removed(self) -> None:
        spec = self._bbox_spec()
        regions = self._regions(
            torch.tensor([[[10.0, torch.inf, torch.inf]]]), torch.tensor([[[5.0, torch.inf, torch.inf]]])
        )
        mask = _build_camera_loss_mask(spec, None, regions)
        self.assertEqual(mask.tolist(), [[[False, False, False]]])

    def test_frozen_actor_is_rendered_not_disabled(self) -> None:
        spec = self._bbox_spec(selected_state=ComponentState.frozen)
        self.assertEqual(set(spec.disabled_track_regions), {"occluder" + DISABLED_TRACKS_SUFFIX})
        regions = self._regions(torch.tensor([[[5.0, torch.inf, torch.inf]]]), torch.full(self._SHAPE, torch.inf))
        self.assertTrue(bool(_build_camera_loss_mask(spec, None, regions)[0, 0, 0]))

    def test_nearest_depth_wins_across_multiple_populations(self) -> None:
        layers = {
            name: SimpleNamespace(class_labels=None, tracks=object())
            for name in ("selected_a", "selected_b", "disabled_a", "disabled_b")
        }
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "selected_a": ComponentOnlyLayerConfig(masks=[MaskSource.bbox]),
                "selected_b": ComponentOnlyLayerConfig(masks=[MaskSource.bbox]),
                "disabled_a": ComponentOnlyLayerConfig(state=ComponentState.disabled, masks=[MaskSource.bbox]),
                "disabled_b": ComponentOnlyLayerConfig(state=ComponentState.disabled, masks=[MaskSource.bbox]),
            },
        )
        spec = build_mask_spec(config, layers, None)
        assert spec is not None
        one = torch.ones((1, 1, 1), dtype=torch.bool)
        regions = {
            "selected_a": _projected(one, 10.0),
            "selected_b": _projected(one, 5.0),
            "disabled_a" + DISABLED_TRACKS_SUFFIX: _projected(one, 12.0),
            "disabled_b" + DISABLED_TRACKS_SUFFIX: _projected(one, 7.0),
        }
        self.assertTrue(bool(_build_camera_loss_mask(spec, None, regions)))  # rendered nearest: 5 < 7
        regions["selected_b"] = _projected(one, 9.0)
        self.assertFalse(bool(_build_camera_loss_mask(spec, None, regions)))  # disabled nearest: 7 < 9

    def test_no_intersections_preserve_existing_behavior(self) -> None:
        spec = self._bbox_spec()
        selected_depth = torch.tensor([[[torch.inf, 4.0, torch.inf]]])
        regions = self._regions(selected_depth, torch.full(self._SHAPE, torch.inf))
        self.assertEqual(_build_camera_loss_mask(spec, None, regions).tolist(), [[[False, True, False]]])

    def test_exact_and_near_ties_are_conservatively_removed(self) -> None:
        spec = self._bbox_spec()
        selected = torch.full(self._SHAPE, 5.0)
        disabled = torch.tensor(
            [[[5.0, 5.0 + 0.5 * ACTOR_DEPTH_TIE_TOLERANCE_M, 5.0 + 1.1 * ACTOR_DEPTH_TIE_TOLERANCE_M]]]
        )
        mask = _build_camera_loss_mask(spec, None, self._regions(selected, disabled))
        self.assertEqual(mask.tolist(), [[[False, False, True]]])

    def test_disabled_foreground_does_not_leak_back_through_residual(self) -> None:
        spec = self._bbox_spec(include_residual=True)
        selected = torch.tensor([[[10.0, torch.inf, torch.inf]]])
        disabled = torch.tensor([[[5.0, 5.0, torch.inf]]])
        regions = self._regions(selected, disabled, include_carve=True)
        # Pixel 0 is an occluded selected region; pixel 1 is disabled-only.
        # Both remain invalid rather than falling into the residual complement.
        mask = _build_camera_loss_mask(spec, torch.tensor([[[2, 2, 2]]]), regions)
        self.assertEqual(mask.tolist(), [[[False, False, True]]])

    def test_semantic_only_static_region_is_unchanged(self) -> None:
        layers = {
            "road": SimpleNamespace(class_labels=["road"], tracks=None),
            "occluder": SimpleNamespace(class_labels=None, tracks=object()),
        }
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "road": ComponentOnlyLayerConfig(masks=[MaskSource.semantic]),
                "occluder": ComponentOnlyLayerConfig(state=ComponentState.disabled, masks=[MaskSource.bbox]),
            },
        )
        spec = build_mask_spec(config, layers, CLASS_MAP)
        assert spec is not None
        disabled = torch.tensor([[[1.0, torch.inf, torch.inf]]])
        regions = {"occluder" + DISABLED_TRACKS_SUFFIX: _projected(torch.isfinite(disabled), disabled)}
        mask = _build_camera_loss_mask(spec, torch.tensor([[[0, 2, 2]]]), regions)
        self.assertEqual(mask.tolist(), [[[True, False, False]]])

    def test_semantic_intersection_and_union_keep_recipe_domains(self) -> None:
        layers = {
            "target": SimpleNamespace(class_labels=["road"], tracks=object()),
            "occluder": SimpleNamespace(class_labels=None, tracks=object()),
        }
        semantic = torch.tensor([[[0, 0, 2]]])
        target_mask = torch.tensor([[[False, True, True]]])
        target_depth = torch.tensor([[[torch.inf, 10.0, 10.0]]])
        disabled_depth = torch.tensor([[[5.0, 5.0, 5.0]]])
        for operation, expected in (
            (MasksOperation.intersection, [[[False, False, False]]]),
            # Pixel 0 is semantic-only: it has no target cuboid depth and must
            # not gain/invent depth ordering from the union's bbox source.
            (MasksOperation.union, [[[True, False, False]]]),
        ):
            with self.subTest(operation=operation):
                config = ComponentOnlyConfig(
                    enabled=True,
                    layers={
                        "target": ComponentOnlyLayerConfig(
                            masks=[MaskSource.semantic, MaskSource.bbox], masks_operation=operation
                        ),
                        "occluder": ComponentOnlyLayerConfig(state=ComponentState.disabled, masks=[MaskSource.bbox]),
                    },
                )
                spec = build_mask_spec(config, layers, CLASS_MAP)
                assert spec is not None
                regions = {
                    "target": ProjectedTrackRegion(target_mask, target_depth),
                    "occluder" + DISABLED_TRACKS_SUFFIX: _projected(torch.isfinite(disabled_depth), disabled_depth),
                }
                self.assertEqual(_build_camera_loss_mask(spec, semantic, regions).tolist(), expected)

    def test_missing_or_malformed_required_depth_fails_loud(self) -> None:
        spec = self._bbox_spec()
        disabled_key = "occluder" + DISABLED_TRACKS_SUFFIX
        one = torch.ones((1, 1, 1), dtype=torch.bool)
        with self.assertRaisesRegex(KeyError, "selected"):
            _build_camera_loss_mask(spec, None, {disabled_key: _projected(one, 1.0)})
        with self.assertRaisesRegex(TypeError, "ProjectedTrackRegion"):
            malformed_regions = cast(
                dict[str, ProjectedTrackRegion],
                {"selected": one, disabled_key: _projected(one, 1.0)},
            )
            _build_camera_loss_mask(spec, None, malformed_regions)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            _build_camera_loss_mask(
                spec,
                None,
                {
                    "selected": _projected(one, 2.0),
                    disabled_key: ProjectedTrackRegion(one, torch.ones((1, 1, 2))),
                },
            )


class TestHelpers(unittest.TestCase):
    def test_mask_has_any_kept_pixels(self) -> None:
        self.assertTrue(mask_has_any_kept_pixels(torch.ones((2, 2), dtype=torch.bool)))
        self.assertFalse(mask_has_any_kept_pixels(torch.zeros((2, 2), dtype=torch.bool)))

    def test_apply_zero_mask(self) -> None:
        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        gt = pred * 10.0
        mask = torch.tensor([[[True, False], [False, True]]])
        new_pred, new_gt = apply_zero_mask(pred, gt, mask)
        self.assertTrue(torch.equal(new_pred, torch.tensor([[[1.0, 0.0], [0.0, 4.0]]])))
        self.assertTrue(torch.equal(new_gt, torch.tensor([[[10.0, 0.0], [0.0, 40.0]]])))

    def test_apply_zero_mask_broadcasts_over_channels(self) -> None:
        # (B, H, W) mask against (B, H, W, C) tensors: the mask gains a
        # trailing channel axis and zeroes whole pixels.
        pred = torch.ones((1, 2, 2, 3))
        gt = torch.full((1, 2, 2, 3), 10.0)
        mask = torch.tensor([[[True, False], [False, True]]])
        new_pred, new_gt = apply_zero_mask(pred, gt, mask)
        self.assertTrue(torch.equal(new_pred[0, 0, 0], torch.ones(3)))
        self.assertTrue(torch.equal(new_pred[0, 0, 1], torch.zeros(3)))
        self.assertTrue(torch.equal(new_gt[0, 1, 0], torch.zeros(3)))
        self.assertTrue(torch.equal(new_gt[0, 1, 1], torch.full((3,), 10.0)))


if __name__ == "__main__":
    unittest.main()
