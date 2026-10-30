# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from nurec.config.base_schema import config_to_primitive, resolve_per_sensor
from nurec.config.model import (
    ActiveTileMaskConfig,
    ActiveTileMaskStrategy,
    AntialiasingConfig,
    CameraLidarTilingConfig,
    CameraToLidarTileMappingConfig,
    CullingConfig,
    GSplatRendererConfig,
    OutputsConfig,
    ProfilingRendererConfig,
    ProjectionConfig,
    RenderModeConfig,
    SensorOutputConfig,
    TileSizeStrategy,
    TilingConfig,
)
from nurec.utils.tile_mask import (
    DEFAULT_DEPTH_SHELLS,
    DEFAULT_N_POSE_SAMPLES,
    DEFAULT_POSE_JITTER_ROTATION_RAD,
    DEFAULT_POSE_JITTER_TRANSLATION_M,
    DEFAULT_RAY_SUBSAMPLE,
)


def test_projection_config_defaults() -> None:
    config = ProjectionConfig()
    assert config.n_rolling_shutter_iterations == 5
    assert config.ut_dim == 3
    assert config.ut_alpha == 1.0
    assert config.ut_beta == 2.0
    assert config.ut_kappa == 0.0
    assert config.ut_require_all_sigma_points is False
    assert config.image_margin_factor == 0.1
    assert config.min_projected_ray_radius == pytest.approx(0.5477225575051661)


def test_projection_config_custom() -> None:
    config = ProjectionConfig.model_validate({"ut_dim": 5, "ut_alpha": 0.5, "n_rolling_shutter_iterations": 10})
    assert config.ut_dim == 5
    assert config.ut_alpha == 0.5
    assert config.n_rolling_shutter_iterations == 10


def test_culling_config_defaults() -> None:
    config = CullingConfig()
    assert config.near_clip_distance == 0.01
    assert config.near_far_z_culling is False
    assert config.rect_bounding is False
    assert config.enable_ray_based_culling is False
    assert config.require_positive_hit_t is False


def test_culling_config_custom() -> None:
    config = CullingConfig.model_validate(
        {
            "near_clip_distance": 0.5,
            "rect_bounding": True,
            "enable_ray_based_culling": True,
            "require_positive_hit_t": True,
        }
    )
    assert config.near_clip_distance == 0.5
    assert config.rect_bounding is True
    assert config.enable_ray_based_culling is True
    assert config.require_positive_hit_t is True


def test_culling_config_legacy_far_clip_distance() -> None:
    """Legacy ``far_clip_distance`` should be promoted to both sensor-specific fields."""
    config = CullingConfig.model_validate({"far_clip_distance": 500.0})
    assert config.far_clip_distance_camera == 500.0
    assert config.far_clip_distance_lidar == 500.0


def test_culling_config_legacy_far_clip_distance_no_override() -> None:
    """Sensor-specific values take precedence over legacy ``far_clip_distance``."""
    config = CullingConfig.model_validate({"far_clip_distance": 500.0, "far_clip_distance_camera": 200.0})
    assert config.far_clip_distance_camera == 200.0
    assert config.far_clip_distance_lidar == 500.0


def test_tiling_config_lidar_defaults() -> None:
    """Base TilingConfig has lidar with defaults."""
    config = TilingConfig()
    assert config.lidar.tile_size_elevation == 16
    assert config.lidar.tile_size_azimuth == 16


def test_tiling_config_lidar_custom() -> None:
    config = TilingConfig.model_validate({"lidar": {"tile_size_elevation": 8, "tile_size_azimuth": 8}})
    assert config.lidar.tile_size_azimuth == 8


def test_camera_lidar_tiling_config() -> None:
    """CameraLidarTilingConfig has both camera and lidar."""
    config = CameraLidarTilingConfig.model_validate(
        {
            "strategy": "custom",
            "camera": {"tile_width": 16, "tile_height": 16},
            "lidar": {"tile_size_elevation": 8, "tile_size_azimuth": 8},
        }
    )
    assert config.camera.tile_width == 16
    assert config.lidar.tile_size_azimuth == 8


def test_camera_lidar_tiling_config_defaults() -> None:
    """CameraLidarTilingConfig has sensible defaults for both sensors."""
    config = CameraLidarTilingConfig()
    assert config.camera.tile_width == 16
    assert config.lidar.tile_size_elevation == 16
    assert config.strategy == TileSizeStrategy.RESOLUTION


def test_camera_lidar_tiling_config_legacy_stale_tile_width_loads_resolution() -> None:
    """Legacy artifacts with explicit stale tile_width default to RESOLUTION (no migration)."""
    config = CameraLidarTilingConfig.model_validate({"camera": {"tile_width": 32, "tile_height": 7}})
    assert config.strategy == TileSizeStrategy.RESOLUTION
    # Stale values are kept on the field but the renderer ignores them in RESOLUTION mode.
    assert config.camera.tile_width == 32
    assert config.camera.tile_height == 7


@pytest.mark.parametrize(
    "strategy_input, expected",
    [
        ("resolution", TileSizeStrategy.RESOLUTION),
        ("RESOLUTION", TileSizeStrategy.RESOLUTION),
        ("custom", TileSizeStrategy.CUSTOM),
        ("CUSTOM", TileSizeStrategy.CUSTOM),
    ],
)
def test_camera_lidar_tiling_config_strategy_accepts_value_and_name(
    strategy_input: str, expected: TileSizeStrategy
) -> None:
    """OmegaConf round-trips Enums through their ``.name`` ("RESOLUTION"), pydantic
    only accepts the ``.value`` ("resolution"). The strategy field must take either."""
    config = CameraLidarTilingConfig.model_validate(
        {"strategy": strategy_input, "camera": {"tile_width": 16, "tile_height": 16}}
    )
    assert config.strategy == expected


def test_camera_lidar_tiling_config_strategy_survives_omegaconf_round_trip() -> None:
    """Regression test: a parsed.yaml written by OmegaConf must reload cleanly.

    OmegaConf serializes Enum members as their ``.name``; before the strategy
    field_validator existed, the resulting yaml failed pydantic's enum check on
    reload (NuRec-style train -> dump -> resume cycle)."""
    serialized = OmegaConf.to_yaml(DictConfig({"strategy": TileSizeStrategy.RESOLUTION}))
    assert "strategy: RESOLUTION" in serialized
    config = CameraLidarTilingConfig.model_validate(OmegaConf.to_container(OmegaConf.create(serialized)))
    assert config.strategy == TileSizeStrategy.RESOLUTION


def test_camera_lidar_tiling_config_strategy_rejects_unknown_value() -> None:
    """Random strings still get rejected by the default enum validator."""
    with pytest.raises(ValidationError):
        CameraLidarTilingConfig.model_validate({"strategy": "not-a-strategy"})


def test_camera_lidar_tiling_config_custom_rejects_unsupported_tile_size() -> None:
    """CUSTOM strategy rejects tile sizes outside the gsplat-supported set."""
    with pytest.raises(ValidationError, match="Unsupported gsplat tile size"):
        CameraLidarTilingConfig.model_validate(
            {
                "strategy": "custom",
                "camera": {"tile_width": 32, "tile_height": 32},
            }
        )


def test_camera_lidar_tiling_config_strategy_name_custom_rejects_unsupported_tile_size() -> None:
    """CUSTOM-by-name (OmegaConf .name form) must also reject unsupported tile sizes.

    Pydantic v2 runs model_validator(mode="before") before field_validator(mode="before"),
    so the model validator must recognize the uppercase .name form on its own."""
    with pytest.raises(ValidationError, match="Unsupported gsplat tile size"):
        CameraLidarTilingConfig.model_validate(
            {
                "strategy": "CUSTOM",
                "camera": {"tile_width": 32, "tile_height": 32},
            }
        )


def test_render_mode_config() -> None:
    config = RenderModeConfig.model_validate({"mode": "kbuffer", "k_buffer_size": 16})
    assert config.mode == "kbuffer"
    assert config.k_buffer_size == 16
    assert config.enable_warp_atomic_optim is True


def test_active_tile_mask_strategy_values() -> None:
    """The ``.value`` strings are the config files' contract: the option yamls select a
    strategy by name (``strategy: valid_camera_lidar_overlap``), so they must stay stable
    and the member set must stay closed."""
    assert ActiveTileMaskStrategy.CAMERA_INVALID_ONLY.value == "camera_invalid_only"
    assert ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP.value == "valid_camera_lidar_overlap"
    assert {member.value for member in ActiveTileMaskStrategy} == {
        "camera_invalid_only",
        "valid_camera_lidar_overlap",
    }


def test_camera_to_lidar_tile_mapping_config_defaults() -> None:
    """Defaults must stay equal to the ``DEFAULT_*`` module constants in
    ``nurec/utils/tile_mask.py``, which the precompute entry point applies when a caller
    passes no explicit knobs. Pinning the constants (not literals) keeps the config and
    the module from drifting apart."""
    config = CameraToLidarTileMappingConfig()
    assert config.n_pose_samples == DEFAULT_N_POSE_SAMPLES
    assert config.depth_shells == DEFAULT_DEPTH_SHELLS
    assert config.ray_subsample == DEFAULT_RAY_SUBSAMPLE
    assert config.pose_jitter_translation_m == DEFAULT_POSE_JITTER_TRANSLATION_M
    assert config.pose_jitter_rotation_rad == DEFAULT_POSE_JITTER_ROTATION_RAD


def test_active_tile_mask_config_custom() -> None:
    """Every knob is overridable through ``active_tile_mask.mapping_params``; a yaml list is
    accepted for ``depth_shells`` and the strategy takes its yaml ``.value`` form."""
    config = ActiveTileMaskConfig.model_validate(
        {
            "enabled": True,
            "strategy": "valid_camera_lidar_overlap",
            "mapping_params": {
                "n_pose_samples": 8,
                "depth_shells": [10.0, 50.0],
                "ray_subsample": 1,
                "pose_jitter_translation_m": 0.1,
                "pose_jitter_rotation_rad": 0.01,
            },
        }
    )
    assert config.enabled is True
    assert config.strategy is ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP
    assert config.mapping_params.n_pose_samples == 8
    assert config.mapping_params.depth_shells == (10.0, 50.0)
    assert config.mapping_params.ray_subsample == 1
    assert config.mapping_params.pose_jitter_translation_m == 0.1
    assert config.mapping_params.pose_jitter_rotation_rad == 0.01


@pytest.mark.parametrize(
    "strategy_input, expected",
    [
        ("camera_invalid_only", ActiveTileMaskStrategy.CAMERA_INVALID_ONLY),
        ("CAMERA_INVALID_ONLY", ActiveTileMaskStrategy.CAMERA_INVALID_ONLY),
        ("valid_camera_lidar_overlap", ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP),
        ("VALID_CAMERA_LIDAR_OVERLAP", ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP),
    ],
)
def test_active_tile_mask_config_strategy_accepts_value_and_name(
    strategy_input: str, expected: ActiveTileMaskStrategy
) -> None:
    """OmegaConf round-trips Enums through their ``.name`` ("CAMERA_INVALID_ONLY"), pydantic
    only accepts the ``.value`` ("camera_invalid_only"). The strategy field must take either."""
    config = ActiveTileMaskConfig.model_validate({"enabled": True, "strategy": strategy_input})
    assert config.strategy is expected


def test_active_tile_mask_config_strategy_survives_omegaconf_round_trip() -> None:
    """Regression test: a parsed.yaml written by OmegaConf must reload cleanly.

    OmegaConf serializes Enum members as their ``.name``; without the strategy normalizer
    the resulting yaml fails pydantic's enum check on reload (NuRec-style train -> dump ->
    resume cycle), which would silently break resuming any preset that selects a
    non-default strategy."""
    serialized = OmegaConf.to_yaml(
        DictConfig({"enabled": True, "strategy": ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP})
    )
    assert "strategy: VALID_CAMERA_LIDAR_OVERLAP" in serialized
    config = ActiveTileMaskConfig.model_validate(OmegaConf.to_container(OmegaConf.create(serialized)))
    assert config.enabled is True
    assert config.strategy is ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP


def test_active_tile_mask_config_strategy_rejects_unknown_value() -> None:
    """Random strings still get rejected by the default enum validator."""
    with pytest.raises(ValidationError):
        ActiveTileMaskConfig.model_validate({"strategy": "not-a-strategy"})


@pytest.mark.parametrize("bad_input", ["camera_invalid_only", None, [1, 2], 7])
def test_active_tile_mask_config_rejects_non_mapping_input(bad_input: object) -> None:
    """Non-mapping input must fall through the strategy normalizer to pydantic's own error.

    The normalizer runs before core validation, so it has to tolerate anything a caller can
    pass; without the mapping guard it dies with an AttributeError instead of a
    ValidationError."""
    with pytest.raises(ValidationError):
        ActiveTileMaskConfig.model_validate(bad_input)


def test_active_tile_mask_config_strategy_normalizes_readonly_dictconfig() -> None:
    """A read-only DictConfig carrying the ``.name`` form must still normalize.

    OmegaConf configs are routinely flagged read-only before being handed around, and a
    frozen resume config is exactly the case this normalizer exists for; rewriting the
    strategy in place would raise ReadonlyConfigError."""
    data = OmegaConf.create({"enabled": True, "strategy": "VALID_CAMERA_LIDAR_OVERLAP"})
    OmegaConf.set_readonly(data, True)
    config = ActiveTileMaskConfig.model_validate(data)
    assert config.strategy is ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP
    assert data.strategy == "VALID_CAMERA_LIDAR_OVERLAP"


def test_active_tile_mask_config_strategy_does_not_mutate_caller_mapping() -> None:
    """The normalization is written to a shallow copy, so the caller's dict is left alone."""
    data = {"enabled": True, "strategy": "VALID_CAMERA_LIDAR_OVERLAP"}
    config = ActiveTileMaskConfig.model_validate(data)
    assert config.strategy is ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP
    assert data == {"enabled": True, "strategy": "VALID_CAMERA_LIDAR_OVERLAP"}


@pytest.mark.parametrize(
    "overrides",
    [
        # n_pose_samples < 1 would be silently absorbed by the max(1, ...) clamp in tile_mask.py.
        {"n_pose_samples": 0},
        {"n_pose_samples": -1},
        # An empty depth_shells tuple skips the projection loop entirely, yielding an all-False
        # mapping that marks every assigned lidar ray INVALID and zeroes the lidar loss silently.
        {"depth_shells": ()},
        # ray_subsample < 1 dies far from the cause, inside torch.arange.
        {"ray_subsample": 0},
        {"ray_subsample": -2},
        # Negative jitter half-widths are meaningless.
        {"pose_jitter_translation_m": -0.1},
        {"pose_jitter_rotation_rad": -0.01},
    ],
)
def test_active_tile_mask_config_rejects_out_of_range(overrides: dict[str, object]) -> None:
    """Out-of-range knobs must fail at config validation, not silently degrade at runtime.

    Exercised through the nested ``active_tile_mask.mapping_params`` path -- the only path a
    preset can reach these knobs by -- as well as on the embedded config directly. The nested
    form is the load-bearing one: ``BaseConfigSchema`` inherits pydantic's ``extra="ignore"``,
    so a knob written one level too high is silently dropped rather than validated."""
    with pytest.raises(ValidationError):
        ActiveTileMaskConfig.model_validate({"mapping_params": overrides})
    with pytest.raises(ValidationError):
        CameraToLidarTileMappingConfig.model_validate(overrides)


@pytest.mark.parametrize(
    "config",
    [
        # ``active_tile_mask`` sits on BaseRendererConfig because the composite reads it
        # renderer-agnostically.
        GSplatRendererConfig.model_validate({"name": "3dgut-gsplat"}),
        GSplatRendererConfig.model_validate({"name": "3dgs-gsplat"}),
    ],
)
def test_renderer_config_active_tile_mask_defaults(config: GSplatRendererConfig) -> None:
    """Every renderer config carries the mask config, off by default, with the
    supervision-neutral strategy preselected and the shared precompute knobs."""
    assert config.active_tile_mask.enabled is False
    assert config.active_tile_mask.strategy is ActiveTileMaskStrategy.CAMERA_INVALID_ONLY
    assert config.active_tile_mask.mapping_params == CameraToLidarTileMappingConfig()


def test_sensor_output_config_defaults() -> None:
    config = SensorOutputConfig()
    assert config.enable_features is True
    assert config.enable_normals is True
    assert config.enable_extended_features is True
    assert config.enable_sensor_features is True
    assert config.enable_ray_gradients is False


def test_outputs_config_defaults() -> None:
    config = OutputsConfig()
    assert config.camera.enable_features is True
    assert config.lidar.enable_normals is True
    assert config.scene.enable_cumulated_weights is False


def test_profiling_config_defaults() -> None:
    config = ProfilingRendererConfig()
    assert config.frequency == 0.0


def test_antialiasing_config_defaults() -> None:
    config = AntialiasingConfig()
    assert config.lidar_divergence == 0.002


def test_gsplat_renderer_config_minimal() -> None:
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgut-gsplat",
            "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
            "background": {"color": 0.0},
        }
    )
    assert config.name == "3dgut-gsplat"
    assert config.lidar_unstructured is False
    assert config.profiling.frequency == 0.0
    assert config.antialiasing.lidar_divergence == 0.002
    assert config.tiling is not None
    assert config.tiling.camera is not None
    assert config.tiling.camera.tile_width == 16


def test_gsplat_renderer_tensor_cache_sizes() -> None:
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgut-gsplat",
            "max_camera_tensor_cache_entries": 8192,
            "max_lidar_geometry_cache_entries": 2048,
            "max_lidar_frame_context_cache_entries": 512,
        }
    )
    assert config.max_camera_tensor_cache_entries == 8192
    assert config.max_lidar_geometry_cache_entries == 2048
    assert config.max_lidar_frame_context_cache_entries == 512

    with pytest.raises(ValidationError):
        GSplatRendererConfig.model_validate({"name": "3dgut-gsplat", "max_camera_tensor_cache_entries": 0})
    with pytest.raises(ValidationError):
        GSplatRendererConfig.model_validate({"name": "3dgut-gsplat", "max_lidar_geometry_cache_entries": 0})
    with pytest.raises(ValidationError):
        GSplatRendererConfig.model_validate({"name": "3dgut-gsplat", "max_lidar_frame_context_cache_entries": 0})


def test_gsplat_renderer_config_with_sub_configs() -> None:
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgut-gsplat",
            "lidar_unstructured": False,
            "projection": {"ut_dim": 5, "ut_alpha": 0.5},
            "culling": {"near_clip_distance": 0.3, "enable_ray_based_culling": True},
            "tiling": {
                "strategy": "custom",
                "camera": {"tile_width": 16, "tile_height": 16},
                "lidar": {"tile_size_elevation": 16, "tile_size_azimuth": 16},
            },
            "render": {"mode": "kbuffer", "k_buffer_size": 0},
            "profiling": {"frequency": 100.0},
            "antialiasing": {"lidar_divergence": 0.005},
            "background": {"color": 0.0},
        }
    )
    assert config.projection is not None
    assert config.projection.ut_dim == 5
    assert config.culling is not None
    assert config.culling.enable_ray_based_culling is True
    assert config.render is not None
    assert config.render.mode == "kbuffer"
    assert config.profiling.frequency == 100.0
    assert config.antialiasing.lidar_divergence == 0.005


def test_profiling_defaults_on_renderer() -> None:
    """Profiling and antialiasing should have sensible defaults without explicit config."""
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgs-gsplat",
            "lidar_unstructured": False,
            "background": {"color": 0.0},
        }
    )
    assert config.profiling.frequency == 0.0
    assert config.antialiasing.lidar_divergence == 0.002


def test_gsplat_renderer_config_batch_policy() -> None:
    """renderer_config defaults to mixed_batch; parallel_batch accepted for 3dgut-gsplat but rejected for 3dgs-gsplat."""
    config = GSplatRendererConfig.model_validate(
        {"name": "3dgut-gsplat", "lidar_unstructured": False, "background": {"color": 0.0}}
    )
    assert config.renderer_config == "mixed_batch"

    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgut-gsplat",
            "lidar_unstructured": False,
            "background": {"color": 0.0},
            "renderer_config": "parallel_batch",
        }
    )
    assert config.renderer_config == "parallel_batch"

    with pytest.raises(ValidationError):
        GSplatRendererConfig.model_validate(
            {
                "name": "3dgs-gsplat",
                "lidar_unstructured": False,
                "background": {"color": 0.0},
                "renderer_config": "parallel_batch",
            }
        )


def test_sparse_segments_do_not_require_static_gaussian_cache() -> None:
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgut-gsplat",
            "sparse_segments": True,
            "cache_static_gaussians": False,
        }
    )

    assert config.sparse_segments
    assert not config.cache_static_gaussians


def test_sparse_segments_require_the_3dgut_renderer() -> None:
    with pytest.raises(ValidationError, match="3dgut-gsplat"):
        GSplatRendererConfig.model_validate({"name": "3dgs-gsplat", "sparse_segments": True})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"packed": True}, "packed rendering"),
        ({"sparse_grad": True}, "sparse gradients"),
        ({"absgrad": True}, "absolute gradients"),
        ({"outputs": {"scene": {"enable_visibility": True}}}, "scene visibility output"),
        ({"lidar_unstructured": True}, "unstructured LiDAR"),
        ({"lidar_unstructured": {"lidar:front": False, "lidar:top": True}}, "unstructured LiDAR"),
    ],
)
def test_sparse_segments_reject_unsupported_static_settings(overrides: dict[str, Any], message: str) -> None:
    """Every setting the sparse path cannot render and that a run cannot change."""
    with pytest.raises(ValidationError, match=message):
        GSplatRendererConfig.model_validate({"name": "3dgut-gsplat", "sparse_segments": True, **overrides})


def test_sparse_segments_name_every_unsupported_setting_at_once() -> None:
    with pytest.raises(ValidationError) as excinfo:
        GSplatRendererConfig.model_validate(
            {"name": "3dgut-gsplat", "sparse_segments": True, "packed": True, "absgrad": True}
        )

    assert "packed rendering" in str(excinfo.value)
    assert "absolute gradients" in str(excinfo.value)


def test_unsupported_settings_are_free_while_sparse_segments_are_off() -> None:
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgut-gsplat",
            "packed": True,
            "absgrad": True,
            "lidar_unstructured": True,
        }
    )

    assert not config.sparse_segments


def test_gsplat_renderer_name_literal() -> None:
    """GSplatRendererConfig.name must be one of the supported names."""
    with pytest.raises(ValidationError):
        GSplatRendererConfig.model_validate({"name": "unsupported-renderer"})


# --- config_to_primitive() round-trip tests ---


def test_tiling_config_to_primitive_lidar_only() -> None:
    """Base TilingConfig serializes with lidar defaults only (no camera key)."""
    config = TilingConfig()
    d = config_to_primitive(config)
    assert "lidar" in d
    assert d["lidar"]["tile_size_elevation"] == 16
    assert "camera" not in d


def test_camera_lidar_tiling_config_to_primitive() -> None:
    """CameraLidarTilingConfig serializes with both camera and lidar."""
    config = CameraLidarTilingConfig()
    d = config_to_primitive(config)
    assert "camera" in d
    assert "lidar" in d
    assert d["camera"]["tile_width"] == 16
    assert d["lidar"]["tile_size_elevation"] == 16


def test_culling_config_to_primitive_far_clip_migration() -> None:
    """Legacy far_clip_distance produces sensor-specific keys after serialization."""
    config = CullingConfig.model_validate({"far_clip_distance": 500.0})
    d = config_to_primitive(config)
    assert d["far_clip_distance_camera"] == 500.0
    assert d["far_clip_distance_lidar"] == 500.0


@pytest.mark.parametrize(
    "config",
    [
        GSplatRendererConfig.model_validate({"name": "3dgut-gsplat"}),
        GSplatRendererConfig.model_validate({"name": "3dgs-gsplat"}),
    ],
)
def test_renderer_config_to_primitive_strategy_is_plain_str(
    config: GSplatRendererConfig,
) -> None:
    """``ActiveTileMaskStrategy`` must stay a ``str`` subclass to survive serialization.

    ``config_to_primitive`` dumps in python mode, so ``strategy`` survives as a live enum
    member rather than its value. That dict is what gets written to ``parsed.yaml`` and
    reloaded from an artifact; a plain ``Enum`` member would not round-trip as its
    ``.value`` through a plain-text serializer, and string comparisons against the value
    would stop holding. The ``str`` mixin on the enum is what makes the member behave as
    its ``.value``.

    The field is always dumped, mask enabled or not, so the invariant is checked on the
    defaults. ``active_tile_mask`` lives on ``BaseRendererConfig``, so it is shared by
    every renderer name.
    """
    strategy = config_to_primitive(config)["active_tile_mask"]["strategy"]
    assert isinstance(strategy, str)
    assert strategy == ActiveTileMaskStrategy.CAMERA_INVALID_ONLY.value


def test_outputs_config_to_primitive_nested_shape() -> None:
    """Outputs sub-structure must have the nested shape expected by the renderer."""
    config = OutputsConfig()
    d = config_to_primitive(config)
    assert "camera" in d
    assert "lidar" in d
    assert "scene" in d
    assert d["camera"]["enable_features"] is True
    assert d["camera"]["enable_normals"] is True
    assert d["lidar"]["enable_sensor_features"] is True
    assert d["scene"]["enable_cumulated_weights"] is False


# box_method_lidar_unstructured is PerSensor[Literal["ut", "pbf", "hybrid"]]. The
# Literal makes pydantic reject bad values at config load instead of letting them
# reach the gsplat renderer -- but it also has to survive the shapes hydra actually
# delivers (plain dict, DictConfig via to_object) and both PerSensor forms.


@pytest.mark.parametrize("value", ["ut", "pbf", "hybrid"])
def test_box_method_accepts_each_valid_scalar(value: str) -> None:
    config = CullingConfig.model_validate({"box_method_lidar_unstructured": value})
    assert resolve_per_sensor(config.box_method_lidar_unstructured, "lidar_a") == value


def test_box_method_default_is_ut() -> None:
    config = CullingConfig.model_validate({})
    assert resolve_per_sensor(config.box_method_lidar_unstructured, "lidar_a") == "ut"


def test_box_method_accepts_per_sensor_dict() -> None:
    """The dict form must still resolve per sensor after the Literal narrowing."""
    config = CullingConfig.model_validate({"box_method_lidar_unstructured": {"lidar_a": "ut", "lidar_b": "hybrid"}})
    assert resolve_per_sensor(config.box_method_lidar_unstructured, "lidar_a") == "ut"
    assert resolve_per_sensor(config.box_method_lidar_unstructured, "lidar_b") == "hybrid"


def test_box_method_rejects_invalid_scalar() -> None:
    with pytest.raises(ValidationError):
        CullingConfig.model_validate({"box_method_lidar_unstructured": "bogus"})


def test_box_method_rejects_invalid_per_sensor_value() -> None:
    """A typo inside the dict must fail at load, not at render time."""
    with pytest.raises(ValidationError):
        CullingConfig.model_validate({"box_method_lidar_unstructured": {"lidar_a": "nope"}})


def test_box_method_survives_omegaconf_round_trip() -> None:
    """Hydra hands the config over as DictConfig, so validate what to_object yields."""
    cfg = OmegaConf.create({"box_method_lidar_unstructured": {"lidar_a": "pbf"}})
    config = CullingConfig.model_validate(OmegaConf.to_object(cfg))
    assert resolve_per_sensor(config.box_method_lidar_unstructured, "lidar_a") == "pbf"


def test_box_method_resolves_from_dictconfig_value() -> None:
    """resolve_per_sensor also sees raw DictConfig mappings (unconverted hydra nodes)."""
    dict_config = OmegaConf.create({"lidar_a": "hybrid"})
    assert isinstance(dict_config, DictConfig)
    assert resolve_per_sensor(dict_config, "lidar_a") == "hybrid"


def test_box_method_serializes_to_primitive() -> None:
    """config_to_primitive must emit a plain str the renderer/gsplat can consume."""
    config = CullingConfig.model_validate({"box_method_lidar_unstructured": "hybrid"})
    assert config_to_primitive(config)["box_method_lidar_unstructured"] == "hybrid"
