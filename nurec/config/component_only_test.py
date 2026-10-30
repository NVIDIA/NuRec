# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema + cross-validation tests for ``nurec.config.component_only``.

Cross-validation is duck-typed against ``model.layers`` values (reads
``class_labels`` / ``tracks`` / ``cuboid_tracks_expand`` attributes),
so tests use ``SimpleNamespace`` stand-ins instead of constructing
full ``LayerConfigType`` models.
"""

import unittest

from types import SimpleNamespace

from pydantic import ValidationError

from nurec.config.component_only import (
    DEFAULT_LAYER_MASKS,
    CameraFrameSelection,
    CameraLossNormalization,
    ComponentOnlyCameraSupervisionConfig,
    ComponentOnlyConfig,
    ComponentOnlyLayerConfig,
    ComponentOnlyTrackConfig,
    ComponentState,
    MasksOperation,
    MaskSource,
    ScopeRegion,
    resolve_layer_masks,
    validate_component_only,
)


def _road_cfg() -> SimpleNamespace:
    return SimpleNamespace(class_labels=["road"], tracks=None)


def _rigids_cfg() -> SimpleNamespace:
    return SimpleNamespace(class_labels=None, tracks=SimpleNamespace(label_classes=["Car"]))


def _deformables_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        class_labels=None,
        tracks=SimpleNamespace(label_classes=["Pedestrian"]),
        cuboid_tracks_expand=0.3,
    )


def _traffic_light_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        class_labels=["traffic light"],
        tracks=SimpleNamespace(label_classes=["generated traffic light"]),
    )


def _background_cfg() -> SimpleNamespace:
    return SimpleNamespace(class_labels=None, tracks=None)


def _model_layers() -> dict:
    return {
        "background": _background_cfg(),
        "road": _road_cfg(),
        "dynamic_rigids": _rigids_cfg(),
        "dynamic_deformables": _deformables_cfg(),
        "traffic_light": _traffic_light_cfg(),
    }


def _model_layers_with_motion_dependent() -> dict:
    layers = _model_layers()
    layers["vehicle_lights"] = SimpleNamespace(
        class_labels=None,
        tracks=SimpleNamespace(label_classes=["Car"]),
        motion_source_layer="dynamic_rigids",
    )
    return layers


class TestDefaultsTable(unittest.TestCase):
    def test_all_canonical_layers_covered(self) -> None:
        self.assertEqual(
            set(DEFAULT_LAYER_MASKS),
            {
                "road",
                "dynamic_rigids",
                "static_rigids",
                "vehicle_lights",
                "static_vehicle_lights",
                "dynamic_deformables",
                "traffic_light",
                "loaded_ply_assets",
            },
        )

    def test_agreed_defaults(self) -> None:
        self.assertEqual(DEFAULT_LAYER_MASKS["road"], ((MaskSource.semantic,), None))
        self.assertEqual(DEFAULT_LAYER_MASKS["dynamic_rigids"], ((MaskSource.bbox,), None))
        self.assertEqual(DEFAULT_LAYER_MASKS["static_rigids"], ((MaskSource.bbox,), None))
        self.assertEqual(DEFAULT_LAYER_MASKS["vehicle_lights"], ((MaskSource.bbox,), None))
        self.assertEqual(DEFAULT_LAYER_MASKS["static_vehicle_lights"], ((MaskSource.bbox,), None))
        self.assertEqual(DEFAULT_LAYER_MASKS["dynamic_deformables"], ((MaskSource.bbox_expanded,), None))
        self.assertEqual(
            DEFAULT_LAYER_MASKS["traffic_light"],
            ((MaskSource.semantic, MaskSource.bbox), MasksOperation.union),
        )
        self.assertEqual(DEFAULT_LAYER_MASKS["loaded_ply_assets"], ((MaskSource.bbox,), None))


class TestCameraSupervisionSchema(unittest.TestCase):
    def test_defaults_preserve_existing_behavior(self) -> None:
        supervision = ComponentOnlyConfig().camera_supervision

        self.assertEqual(supervision.frame_selection, CameraFrameSelection.all_frames)
        self.assertEqual(supervision.loss_normalization, CameraLossNormalization.supervised_pixels)
        self.assertFalse(supervision.supervised_only.scale_samples_per_epoch_by_frame_selection_fraction)

    def test_all_explicit_values_parse_from_config_shape(self) -> None:
        config = ComponentOnlyConfig.model_validate(
            {
                "camera_supervision": {
                    "frame_selection": "supervised_only",
                    "loss_normalization": "all_valid_pixels",
                    "supervised_only": {"scale_samples_per_epoch_by_frame_selection_fraction": True},
                }
            }
        )

        self.assertEqual(config.camera_supervision.frame_selection, CameraFrameSelection.supervised_only)
        self.assertEqual(config.camera_supervision.loss_normalization, CameraLossNormalization.all_valid_pixels)
        self.assertTrue(config.camera_supervision.supervised_only.scale_samples_per_epoch_by_frame_selection_fraction)

    def test_explicit_default_enum_values_parse(self) -> None:
        supervision = ComponentOnlyCameraSupervisionConfig.model_validate(
            {
                "frame_selection": "all_frames",
                "loss_normalization": "supervised_pixels",
                "supervised_only": {"scale_samples_per_epoch_by_frame_selection_fraction": False},
            }
        )

        self.assertEqual(supervision.frame_selection, CameraFrameSelection.all_frames)
        self.assertEqual(supervision.loss_normalization, CameraLossNormalization.supervised_pixels)
        self.assertFalse(supervision.supervised_only.scale_samples_per_epoch_by_frame_selection_fraction)

    def test_unknown_frame_selection_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyCameraSupervisionConfig.model_validate({"frame_selection": "visible_if_convenient"})

    def test_unknown_loss_normalization_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyCameraSupervisionConfig.model_validate({"loss_normalization": "masked_sum"})

    def test_scale_samples_per_epoch_by_frame_selection_fraction_rejected_with_all_frames(self) -> None:
        config = ComponentOnlyConfig.model_validate(
            {
                "enabled": True,
                "camera_supervision": {
                    "supervised_only": {"scale_samples_per_epoch_by_frame_selection_fraction": True}
                },
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "camera_supervision.supervised_only.*scale_samples_per_epoch_by_frame_selection_fraction.*"
            "frame_selection='supervised_only'",
        ):
            validate_component_only(config, _model_layers())

    def test_scale_samples_per_epoch_by_frame_selection_fraction_accepted_with_supervised_only(self) -> None:
        config = ComponentOnlyConfig.model_validate(
            {
                "enabled": True,
                "camera_supervision": {
                    "frame_selection": "supervised_only",
                    "supervised_only": {"scale_samples_per_epoch_by_frame_selection_fraction": True},
                },
            },
        )

        validate_component_only(config, _model_layers())

    def test_disabled_component_only_keeps_camera_supervision_inert(self) -> None:
        config = ComponentOnlyConfig.model_validate(
            {
                "enabled": False,
                "camera_supervision": {
                    "supervised_only": {"scale_samples_per_epoch_by_frame_selection_fraction": True}
                },
            }
        )

        validate_component_only(config, _model_layers())


class TestEntrySchema(unittest.TestCase):
    def test_default_state_is_enabled(self) -> None:
        entry = ComponentOnlyLayerConfig()
        self.assertEqual(entry.state, ComponentState.enabled)
        self.assertIsNone(entry.masks)

    def test_masks_operation_required_for_multiple_sources(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyLayerConfig(masks=[MaskSource.semantic, MaskSource.bbox])

    def test_masks_operation_forbidden_for_single_source(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyLayerConfig(masks=[MaskSource.bbox], masks_operation=MasksOperation.union)

    def test_masks_operation_without_masks_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyLayerConfig(masks_operation=MasksOperation.union)

    def test_bbox_and_bbox_expanded_mutually_exclusive(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyLayerConfig(
                masks=[MaskSource.bbox, MaskSource.bbox_expanded],
                masks_operation=MasksOperation.union,
            )

    def test_empty_masks_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyLayerConfig(masks=[])

    def test_duplicate_masks_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyLayerConfig(masks=[MaskSource.bbox, MaskSource.bbox], masks_operation=MasksOperation.union)

    def test_empty_scope_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyLayerConfig(scope={})

    def test_valid_two_source_entry(self) -> None:
        entry = ComponentOnlyLayerConfig(
            masks=[MaskSource.semantic, MaskSource.bbox],
            masks_operation=MasksOperation.intersection,
        )
        self.assertEqual(entry.masks_operation, MasksOperation.intersection)

    def test_track_entry_shares_masks_rules(self) -> None:
        with self.assertRaises(ValidationError):
            ComponentOnlyTrackConfig(
                masks=[MaskSource.bbox, MaskSource.bbox_expanded], masks_operation=MasksOperation.union
            )


class TestResolveLayerMasks(unittest.TestCase):
    def test_defaults_when_unlisted(self) -> None:
        masks, op = resolve_layer_masks("dynamic_deformables", None)
        self.assertEqual(masks, (MaskSource.bbox_expanded,))
        self.assertIsNone(op)

    def test_entry_overrides_default(self) -> None:
        entry = ComponentOnlyLayerConfig(masks=[MaskSource.bbox])
        masks, _ = resolve_layer_masks("dynamic_deformables", entry)
        self.assertEqual(masks, (MaskSource.bbox,))

    def test_track_overrides_entry(self) -> None:
        entry = ComponentOnlyLayerConfig(masks=[MaskSource.bbox_expanded])
        track = ComponentOnlyTrackConfig(masks=[MaskSource.bbox])
        masks, _ = resolve_layer_masks("dynamic_deformables", entry, track)
        self.assertEqual(masks, (MaskSource.bbox,))

    def test_track_without_masks_inherits_entry(self) -> None:
        entry = ComponentOnlyLayerConfig(masks=[MaskSource.bbox])
        track = ComponentOnlyTrackConfig(state=ComponentState.frozen)
        masks, _ = resolve_layer_masks("dynamic_deformables", entry, track)
        self.assertEqual(masks, (MaskSource.bbox,))

    def test_unknown_layer_without_masks_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            resolve_layer_masks("my_custom_layer", None)
        self.assertIn("my_custom_layer", str(cm.exception))


class TestCrossValidation(unittest.TestCase):
    def test_disabled_section_skips_all_checks(self) -> None:
        config = ComponentOnlyConfig(
            enabled=False,
            layers={"no_such_layer": ComponentOnlyLayerConfig()},
        )
        validate_component_only(config, _model_layers())  # no raise

    def test_unknown_layer_key_fails_loud(self) -> None:
        config = ComponentOnlyConfig(enabled=True, layers={"traffic_lihgt": ComponentOnlyLayerConfig()})
        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers())
        self.assertIn("traffic_lihgt", str(cm.exception))
        self.assertIn("dynamic_rigids", str(cm.exception))  # lists available layers

    def test_unknown_key_fails_even_when_disabled(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"no_such": ComponentOnlyLayerConfig(state=ComponentState.disabled)},
        )
        with self.assertRaises(ValueError):
            validate_component_only(config, _model_layers())

    def test_semantic_requires_class_labels(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"dynamic_rigids": ComponentOnlyLayerConfig(masks=[MaskSource.semantic])},
        )
        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers())
        self.assertIn("class_labels", str(cm.exception))

    def test_bbox_requires_tracks(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"road": ComponentOnlyLayerConfig(masks=[MaskSource.bbox])},
        )
        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers())
        self.assertIn("track", str(cm.exception))

    def test_bbox_expanded_requires_expand_factor(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"dynamic_rigids": ComponentOnlyLayerConfig(masks=[MaskSource.bbox_expanded])},
        )
        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers())
        self.assertIn("cuboid_tracks_expand", str(cm.exception))

    def test_scope_on_non_background_rejected(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"dynamic_rigids": ComponentOnlyLayerConfig(scope={ScopeRegion.residual: ComponentState.enabled})},
        )
        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers())
        self.assertIn("scope", str(cm.exception))

    def test_masks_on_background_rejected(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"background": ComponentOnlyLayerConfig(masks=[MaskSource.semantic])},
        )
        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers())
        self.assertIn("residual", str(cm.exception))

    def test_bbox_scope_requires_bbox_masked_sibling(self) -> None:
        # Only road (semantic) enabled → bbox scope has no frustum source.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "road": ComponentOnlyLayerConfig(),
                "background": ComponentOnlyLayerConfig(scope={ScopeRegion.bbox: ComponentState.enabled}),
            },
        )
        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers())
        self.assertIn("bbox", str(cm.exception))

    def test_bbox_scope_with_frozen_sibling_ok(self) -> None:
        # Frozen siblings feed the scope (their box pixels are supervised;
        # non-object content behind the frozen render needs a trainable owner).
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(state=ComponentState.frozen),
                "background": ComponentOnlyLayerConfig(scope={ScopeRegion.bbox: ComponentState.enabled}),
            },
        )
        validate_component_only(config, _model_layers())  # no raise

    def test_bbox_scope_with_disabled_sibling_only_rejected(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(state=ComponentState.disabled),
                "background": ComponentOnlyLayerConfig(scope={ScopeRegion.bbox: ComponentState.enabled}),
            },
        )
        with self.assertRaises(ValueError):
            validate_component_only(config, _model_layers())

    def test_bbox_scope_satisfied_by_per_track_bbox_override(self) -> None:
        # A sibling with semantic-only LAYER masks but an enabled bbox
        # track override does supply the frustum region — must not be
        # rejected (the frustum is built from siblings' per-track bbox
        # masks, so this config is valid).
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    masks=[MaskSource.semantic],
                    tracks={"t1": ComponentOnlyTrackConfig(masks=[MaskSource.bbox])},
                ),
                "background": ComponentOnlyLayerConfig(scope={ScopeRegion.bbox: ComponentState.enabled}),
            },
        )
        validate_component_only(config, _model_layers())  # no raise

    def test_bbox_scope_semantic_only_sibling_without_bbox_track_rejected(self) -> None:
        # Semantic-only layer masks AND no bbox track override → no
        # frustum region → still rejected.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(masks=[MaskSource.semantic]),
                "background": ComponentOnlyLayerConfig(scope={ScopeRegion.bbox: ComponentState.enabled}),
            },
        )
        with self.assertRaises(ValueError):
            validate_component_only(config, _model_layers())

    def test_tracks_on_untracked_layer_rejected(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={"road": ComponentOnlyLayerConfig(tracks={"abc": ComponentOnlyTrackConfig()})},
        )
        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers())
        self.assertIn("without tracks", str(cm.exception))

    def test_per_track_mask_sources_validated(self) -> None:
        # Track override asking for semantic on a layer without class_labels.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(
                    tracks={"abc": ComponentOnlyTrackConfig(masks=[MaskSource.semantic])}
                )
            },
        )
        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers())
        self.assertIn("abc", str(cm.exception))

    def test_frozen_motion_dependent_rejects_enabled_owner(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(state=ComponentState.enabled),
                "vehicle_lights": ComponentOnlyLayerConfig(
                    state=ComponentState.frozen,
                    masks=[MaskSource.bbox],
                ),
            },
        )

        with self.assertRaises(ValueError) as cm:
            validate_component_only(config, _model_layers_with_motion_dependent())

        message = str(cm.exception)
        self.assertIn("frozen default track population", message)
        self.assertIn("enabled motion source 'dynamic_rigids'", message)
        self.assertIn("tracks_calib", message)

    def test_frozen_motion_dependent_accepts_frozen_owner(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(state=ComponentState.frozen),
                "vehicle_lights": ComponentOnlyLayerConfig(
                    state=ComponentState.frozen,
                    masks=[MaskSource.bbox],
                ),
            },
        )

        validate_component_only(config, _model_layers_with_motion_dependent())

    def test_frozen_dependent_track_rejects_inherited_enabled_owner_track(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(state=ComponentState.enabled),
                "vehicle_lights": ComponentOnlyLayerConfig(
                    state=ComponentState.enabled,
                    masks=[MaskSource.bbox],
                    tracks={"car-1": ComponentOnlyTrackConfig(state=ComponentState.frozen)},
                ),
            },
        )

        with self.assertRaisesRegex(ValueError, "frozen track 'car-1'.*enabled motion source"):
            validate_component_only(config, _model_layers_with_motion_dependent())

    def test_frozen_dependent_track_accepts_matching_frozen_owner_track(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(
                    state=ComponentState.enabled,
                    tracks={"car-1": ComponentOnlyTrackConfig(state=ComponentState.frozen)},
                ),
                "vehicle_lights": ComponentOnlyLayerConfig(
                    state=ComponentState.enabled,
                    masks=[MaskSource.bbox],
                    tracks={"car-1": ComponentOnlyTrackConfig(state=ComponentState.frozen)},
                ),
            },
        )

        validate_component_only(config, _model_layers_with_motion_dependent())

    def test_frozen_dependent_default_rejects_enabled_owner_track_override(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_rigids": ComponentOnlyLayerConfig(
                    state=ComponentState.frozen,
                    tracks={"car-1": ComponentOnlyTrackConfig(state=ComponentState.enabled)},
                ),
                "vehicle_lights": ComponentOnlyLayerConfig(
                    state=ComponentState.frozen,
                    masks=[MaskSource.bbox],
                ),
            },
        )

        with self.assertRaisesRegex(ValueError, "frozen track 'car-1'.*enabled motion source"):
            validate_component_only(config, _model_layers_with_motion_dependent())

    def test_representative_valid_config(self) -> None:
        # Example 8 from the design discussion: per-track deformables +
        # frustum-scoped background.
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "dynamic_deformables": ComponentOnlyLayerConfig(
                    state=ComponentState.disabled,
                    masks=[MaskSource.bbox_expanded],
                    tracks={
                        "t1": ComponentOnlyTrackConfig(state=ComponentState.enabled),
                        "t2": ComponentOnlyTrackConfig(state=ComponentState.enabled, masks=[MaskSource.bbox]),
                        "t3": ComponentOnlyTrackConfig(state=ComponentState.frozen),
                    },
                ),
                "background": ComponentOnlyLayerConfig(scope={ScopeRegion.bbox: ComponentState.enabled}),
            },
        )
        validate_component_only(config, _model_layers())  # no raise

    def test_traffic_light_full_choice(self) -> None:
        config = ComponentOnlyConfig(
            enabled=True,
            layers={
                "traffic_light": ComponentOnlyLayerConfig(
                    masks=[MaskSource.semantic, MaskSource.bbox],
                    masks_operation=MasksOperation.intersection,
                )
            },
        )
        validate_component_only(config, _model_layers())  # no raise


if __name__ == "__main__":
    unittest.main()
