# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

from collections.abc import ItemsView, KeysView
from unittest import TestCase

import pytest

from pydantic import TypeAdapter, ValidationError

from nurec.config.fixed_camera_render import FixedCameraRenderConfig
from nurec.config.image_resolution import (
    AbsoluteCameraImageResolutionConfig,
    AbsoluteImageResolutionConfig,
    CameraImageResolutionGroupConfig,
    ImageResolutionConfig,
    RelativeImageResolutionConfig,
    parse_camera_image_resolution_config,
    parse_image_resolution_config,
    resolve_camera_image_resolution_transforms,
)
from nurec.utils.image_resolution import ResolutionRounding


_IMAGE_RESOLUTION_ADAPTER: TypeAdapter[ImageResolutionConfig] = TypeAdapter(ImageResolutionConfig)


@pytest.mark.parametrize("rounding", ["round", "floor"])
def test_camera_absolute_and_relative_reuse_consumer_rounding(rounding: ResolutionRounding) -> None:
    def resolve(config):
        return resolve_camera_image_resolution_transforms(
            config,
            active_camera_ids=["front"],
            source_resolutions={"front": (10, 7)},
            field_name="test",
            rounding=rounding,
        )

    relative = resolve(RelativeImageResolutionConfig(subsample=2))
    absolute = resolve(
        AbsoluteCameraImageResolutionConfig(
            camera_groups=[CameraImageResolutionGroupConfig(camera_ids=["front"], width=5)]
        )
    )
    assert absolute == relative


def test_legacy_warning_is_visible_when_python_warnings_are_disabled() -> None:
    with warnings.catch_warnings(), TestCase().assertLogs("nurec.config.image_resolution", level="WARNING") as logs:
        warnings.simplefilter("ignore")
        parse_camera_image_resolution_config(None, legacy_subsample=2, field_name="dataset.n_val_image_resolution")
    assert "dataset.n_val_image_subsample is deprecated; use dataset.n_val_image_resolution" in logs.output[0]


class _MappingLike:
    """Minimal DictConfig-style mapping protocol without Mapping inheritance."""

    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def keys(self) -> KeysView[str]:
        return self._values.keys()

    def items(self) -> ItemsView[str, object]:
        return self._values.items()

    def __getitem__(self, key: str) -> object:
        return self._values[key]


@pytest.mark.parametrize(
    ("values", "expected_resolution", "expected_scale"),
    [
        ({"mode": "relative", "subsample": 2}, (4, 3), 0.5),
        ({"mode": "absolute", "width": 4, "keep_aspect_ratio": True}, (4, 3), 0.5),
        ({"mode": "absolute", "width": None, "height": 3, "keep_aspect_ratio": True}, (4, 3), 0.5),
        ({"mode": "absolute", "width": 3, "height": 2, "keep_aspect_ratio": False}, (3, 2), (0.375, 1.0 / 3.0)),
    ],
)
def test_resolution_config_resolves_supported_forms(
    values: dict,
    expected_resolution: tuple[int, int],
    expected_scale: float | tuple[float, float],
) -> None:
    transform = _IMAGE_RESOLUTION_ADAPTER.validate_python(values).resolve((8, 6))

    assert transform.output_resolution == expected_resolution
    assert transform.image_domain_scale == expected_scale


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, "Unable to extract tag"),
        ({"mode": "relative", "subsample": 2, "width": 4}, "Extra inputs are not permitted"),
        ({"mode": "absolute", "subsample": 2}, "Extra inputs are not permitted"),
        (
            {"mode": "absolute", "width": 4, "height": 3, "keep_aspect_ratio": True},
            "Exactly one of width or height",
        ),
        ({"mode": "absolute", "width": 4, "keep_aspect_ratio": False}, "Both width and height"),
        (
            {"mode": "absolute", "width": None, "height": 3, "keep_aspect_ratio": False},
            "Both width and height",
        ),
    ],
)
def test_resolution_config_rejects_unsupported_forms(values: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _IMAGE_RESOLUTION_ADAPTER.validate_python(values)


def test_select_resolution_config_accepts_legacy_subsample() -> None:
    with TestCase().assertLogs("nurec.config.image_resolution", level="WARNING"):
        selected = parse_camera_image_resolution_config(
            None,
            legacy_subsample=2,
            field_name="dataset.n_val_image_resolution",
        )

    assert selected == RelativeImageResolutionConfig(subsample=2)


def test_parse_camera_resolution_legacy_subsample_overrides_relative_default() -> None:
    with TestCase().assertLogs("nurec.config.image_resolution", level="WARNING"):
        selected = parse_camera_image_resolution_config(
            {"mode": "relative", "subsample": 4},
            legacy_subsample=2,
            field_name="dataset.n_val_image_resolution",
        )

    assert selected == RelativeImageResolutionConfig(subsample=2)


def test_parse_camera_resolution_accepts_absolute_camera_groups() -> None:
    selected = parse_camera_image_resolution_config(
        {
            "mode": "absolute",
            "camera_groups": [{"camera_ids": ["front"], "width": 4}],
        },
        legacy_subsample=None,
        field_name="dataset.n_val_image_resolution",
    )

    assert selected == AbsoluteCameraImageResolutionConfig(
        camera_groups=[CameraImageResolutionGroupConfig(camera_ids=["front"], width=4)]
    )


def test_parse_camera_resolution_rejects_inactive_relative_field() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse_camera_image_resolution_config(
            {"mode": "absolute", "subsample": None, "camera_groups": [{"camera_ids": ["front"], "width": 4}]},
            legacy_subsample=None,
            field_name="dataset.n_val_image_resolution",
        )


def test_resolution_parsers_accept_dictconfig_style_mappings() -> None:
    camera_resolution = parse_camera_image_resolution_config(
        _MappingLike({"mode": "relative", "subsample": 2}),
        legacy_subsample=None,
        field_name="dataset.n_val_image_resolution",
    )
    image_resolution = parse_image_resolution_config(
        _MappingLike({"mode": "relative", "subsample": 2}),
        legacy_subsample=None,
        field_name="fixed_camera_render",
        default_subsample=2,
    )

    assert camera_resolution == RelativeImageResolutionConfig(subsample=2)
    assert image_resolution == RelativeImageResolutionConfig(subsample=2)


def test_parse_camera_resolution_rejects_unshipped_bare_camera_group_list() -> None:
    with pytest.raises(ValidationError):
        parse_camera_image_resolution_config(
            [{"camera_ids": ["front"], "width": 4}],
            legacy_subsample=None,
            field_name="dataset.n_val_image_resolution",
        )


@pytest.mark.parametrize(
    ("resolution", "legacy_subsample", "message"),
    [
        (None, None, "must specify either relative or absolute"),
        ({"mode": "absolute", "camera_groups": [{"camera_ids": ["front"]}]}, 2, "cannot combine"),
    ],
)
def test_parse_camera_resolution_rejects_missing_or_conflicting_selection(
    resolution: object,
    legacy_subsample: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_camera_image_resolution_config(
            resolution,
            legacy_subsample=legacy_subsample,
            field_name="dataset.n_val_image_resolution",
        )


def test_parse_camera_resolution_requires_non_empty_groups_and_camera_ids() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        parse_camera_image_resolution_config(
            {"mode": "absolute", "camera_groups": [{"camera_ids": []}]},
            legacy_subsample=None,
            field_name="dataset.n_val_image_resolution",
        )


def test_parse_fixed_resolution_defaults_to_relative_resolution() -> None:
    assert parse_image_resolution_config(
        None,
        legacy_subsample=None,
        field_name="fixed_camera_render",
        default_subsample=2,
    ) == RelativeImageResolutionConfig(subsample=2)


def test_parse_fixed_resolution_legacy_subsample_overrides_relative_default() -> None:
    with TestCase().assertLogs("nurec.config.image_resolution", level="WARNING"):
        selected = parse_image_resolution_config(
            {"mode": "relative", "subsample": 2},
            legacy_subsample=3,
            field_name="fixed_camera_render",
            default_subsample=2,
        )

    assert selected == RelativeImageResolutionConfig(subsample=3)


def test_parse_fixed_resolution_rejects_absolute_and_legacy_subsample() -> None:
    with pytest.raises(ValueError, match="cannot combine"):
        parse_image_resolution_config(
            {"mode": "absolute", "width": 4},
            legacy_subsample=2,
            field_name="fixed_camera_render",
            default_subsample=2,
        )


def test_parse_fixed_resolution_accepts_canonical_absolute_resolution() -> None:
    selected = parse_image_resolution_config(
        {"mode": "absolute", "width": 4},
        legacy_subsample=None,
        field_name="fixed_camera_render",
        default_subsample=2,
    )

    assert selected == AbsoluteImageResolutionConfig(width=4)


def test_parse_fixed_resolution_rejects_inactive_relative_field() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse_image_resolution_config(
            {"mode": "absolute", "subsample": None, "width": 4},
            legacy_subsample=None,
            field_name="fixed_camera_render",
            default_subsample=2,
        )


def test_legacy_camera_resolution_parser_serializes_only_canonical_form() -> None:
    with TestCase().assertLogs("nurec.config.image_resolution", level="WARNING"):
        selected = parse_camera_image_resolution_config(
            None,
            legacy_subsample=2,
            field_name="dataset.n_val_image_resolution",
        )

    assert selected.model_dump() == {"mode": "relative", "subsample": 2}


def test_relative_resolution_config_rejects_absolute_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RelativeImageResolutionConfig(subsample=2, width=4)  # type: ignore[call-arg]


def test_resolve_camera_image_resolution_transforms_preserves_relative_broadcast() -> None:
    transforms = resolve_camera_image_resolution_transforms(
        RelativeImageResolutionConfig(subsample=2),
        active_camera_ids=["front", "rear"],
        source_resolutions={"front": (8, 6), "rear": (10, 6)},
        field_name="resolution",
        require_divisible=True,
        rounding="floor",
    )

    assert transforms["front"].output_resolution == (4, 3)
    assert transforms["rear"].output_resolution == (5, 3)


def test_resolve_camera_image_resolution_transforms_supports_homogeneous_group() -> None:
    transforms = resolve_camera_image_resolution_transforms(
        AbsoluteCameraImageResolutionConfig(
            camera_groups=[CameraImageResolutionGroupConfig(camera_ids=["front", "rear"], width=4)]
        ),
        active_camera_ids=["front", "rear"],
        source_resolutions={"front": (8, 6), "rear": (8, 6), "unused": (12, 8)},
        field_name="resolution",
    )

    assert {camera_id: transform.output_resolution for camera_id, transform in transforms.items()} == {
        "front": (4, 3),
        "rear": (4, 3),
    }


def test_resolve_camera_image_resolution_transforms_supports_separate_source_resolution_groups() -> None:
    transforms = resolve_camera_image_resolution_transforms(
        AbsoluteCameraImageResolutionConfig(
            camera_groups=[
                CameraImageResolutionGroupConfig(camera_ids=["front"], width=4),
                CameraImageResolutionGroupConfig(camera_ids=["rear"], width=None, height=3),
            ]
        ),
        active_camera_ids=["front", "rear"],
        source_resolutions={"front": (8, 6), "rear": (10, 6)},
        field_name="resolution",
    )

    assert transforms["front"].output_resolution == (4, 3)
    assert transforms["rear"].output_resolution == (5, 3)


@pytest.mark.parametrize(
    "resolution",
    [
        AbsoluteCameraImageResolutionConfig(
            camera_groups=[CameraImageResolutionGroupConfig(camera_ids=["front", "front", "front"], width=4)]
        ),
        AbsoluteCameraImageResolutionConfig(
            camera_groups=[
                CameraImageResolutionGroupConfig(camera_ids=["front"], width=4),
                CameraImageResolutionGroupConfig(camera_ids=["front"], width=4),
            ]
        ),
    ],
)
def test_resolve_camera_image_resolution_transforms_rejects_duplicate_cameras(
    resolution: AbsoluteCameraImageResolutionConfig,
) -> None:
    with pytest.raises(ValueError, match=r"duplicate cameras are \['front'\]"):
        resolve_camera_image_resolution_transforms(
            resolution,
            active_camera_ids=["front"],
            source_resolutions={"front": (8, 6)},
            field_name="resolution",
        )


@pytest.mark.parametrize(
    ("resolution", "message"),
    [
        (
            AbsoluteCameraImageResolutionConfig(
                camera_groups=[CameraImageResolutionGroupConfig(camera_ids=["front"], width=4)]
            ),
            r"missing cameras \['rear'\]",
        ),
        (
            AbsoluteCameraImageResolutionConfig(
                camera_groups=[
                    CameraImageResolutionGroupConfig(camera_ids=["front"], width=4),
                    CameraImageResolutionGroupConfig(camera_ids=["rear"], width=4),
                    CameraImageResolutionGroupConfig(camera_ids=["unused"], width=4),
                ]
            ),
            r"unknown cameras \['unused'\]",
        ),
    ],
)
def test_resolve_camera_image_resolution_transforms_requires_exact_group_coverage(
    resolution: AbsoluteCameraImageResolutionConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_camera_image_resolution_transforms(
            resolution,
            active_camera_ids=["front", "rear"],
            source_resolutions={"front": (8, 6), "rear": (8, 6)},
            field_name="resolution",
        )


def test_resolve_camera_image_resolution_transforms_rejects_active_camera_missing_from_data() -> None:
    with pytest.raises(ValueError, match="not present in the data"):
        resolve_camera_image_resolution_transforms(
            AbsoluteCameraImageResolutionConfig(
                camera_groups=[CameraImageResolutionGroupConfig(camera_ids=["front", "rear"], width=4)]
            ),
            active_camera_ids=["front", "rear"],
            source_resolutions={"front": (8, 6)},
            field_name="resolution",
        )


def test_resolve_camera_image_resolution_transforms_rejects_mixed_source_resolution_group() -> None:
    with pytest.raises(ValueError, match="split cameras with different source resolutions"):
        resolve_camera_image_resolution_transforms(
            AbsoluteCameraImageResolutionConfig(
                camera_groups=[CameraImageResolutionGroupConfig(camera_ids=["front", "rear"], width=4)]
            ),
            active_camera_ids=["front", "rear"],
            source_resolutions={"front": (8, 6), "rear": (10, 6)},
            field_name="resolution",
        )


def test_absolute_resolution_config_rejects_relative_subsample() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AbsoluteImageResolutionConfig(subsample=2, width=None, height=None)  # type: ignore[call-arg]


def test_absolute_resolution_config_defaults_to_original_width() -> None:
    config = AbsoluteImageResolutionConfig()

    assert config.width == "original"
    assert config.height is None
    assert config.keep_aspect_ratio is True
    assert config.resolve((8, 6)).output_resolution == (8, 6)


def test_absolute_resolution_config_supports_original_height() -> None:
    config = AbsoluteImageResolutionConfig(
        width=None,
        height="original",
    )

    assert config.resolve((8, 6)).output_resolution == (8, 6)


def test_absolute_resolution_config_rejects_unknown_dimension_sentinel() -> None:
    with pytest.raises(ValidationError):
        AbsoluteImageResolutionConfig(width="source")  # type: ignore[arg-type]


def test_absolute_resolution_config_rejects_upsampling_by_default() -> None:
    with pytest.raises(ValueError, match="allow_upsampling=true"):
        AbsoluteImageResolutionConfig(width=16).resolve((8, 6))


def test_absolute_resolution_config_allows_explicit_upsampling() -> None:
    config = AbsoluteImageResolutionConfig(
        width=16,
        allow_upsampling=True,
    )

    assert config.resolve((8, 6)).output_resolution == (16, 12)


def test_fixed_camera_render_supports_absolute_resolution() -> None:
    config = FixedCameraRenderConfig.model_validate(
        {
            "resolution": {
                "mode": "absolute",
                "width": 1920,
                "height": 1080,
                "keep_aspect_ratio": False,
            }
        }
    )

    assert config.resolution.resolve((3840, 2160)).output_resolution == (1920, 1080)
    assert config.model_dump()["resolution"] == {
        "mode": "absolute",
        "width": 1920,
        "height": 1080,
        "keep_aspect_ratio": False,
        "allow_upsampling": False,
    }


def test_fixed_camera_render_defaults_to_relative_resolution() -> None:
    assert FixedCameraRenderConfig().resolution == RelativeImageResolutionConfig(subsample=2)


def test_fixed_camera_render_legacy_normalizer_preserves_non_mapping_input() -> None:
    assert FixedCameraRenderConfig.normalize_legacy_resolution(42) == 42  # type: ignore[operator]


def test_fixed_camera_render_normalizes_legacy_relative_resolution() -> None:
    with TestCase().assertLogs("nurec.config.image_resolution", level="WARNING"):
        config = FixedCameraRenderConfig.model_validate({"subsample": 2})

    assert config.resolution == RelativeImageResolutionConfig(subsample=2)
    assert "subsample" not in config.model_dump()


@pytest.mark.parametrize("field_name", ["width", "height", "keep_aspect_ratio", "allow_upsampling"])
def test_fixed_camera_render_rejects_unshipped_flat_resolution_fields(field_name: str) -> None:
    with pytest.raises(ValidationError, match="Unsupported flat fixed-camera resolution fields"):
        FixedCameraRenderConfig.model_validate(
            {
                "resolution": {"mode": "relative", "subsample": 2},
                field_name: True if field_name in {"keep_aspect_ratio", "allow_upsampling"} else 1920,
            }
        )
