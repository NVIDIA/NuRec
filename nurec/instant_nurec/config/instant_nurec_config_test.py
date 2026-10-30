# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from subprocess import run
from sys import executable
from unittest.mock import MagicMock, call, patch

import pytest

from omegaconf import DictConfig

from nurec.instant_nurec.config.parse import (
    _download_pretrained_instant_nurec_artifacts,
    materialize_pretrained_instant_nurec_config,
    parse_pretrained_instant_nurec_config,
    parse_untyped_config,
    parse_untyped_instant_nurec_config,
)
from nurec.utils.model_registry import ModelRegistry, ModelRegistryError


def test_absolute_primary_config_resolves_from_its_parent(tmp_path: Path) -> None:
    """instant_nurec composes a primary config by absolute path (the parser parent-roots it)."""
    config_root = tmp_path / "configs"
    config_root.mkdir()
    config_file = config_root / "app.yaml"
    config_file.write_text("# @package _global_\n\nsource: absolute_path\n")

    config, _ = parse_untyped_config(str(config_file), [])

    assert config.source == "absolute_path"


def test_absolute_primary_config_resolves_relative_defaults_from_its_parent(tmp_path: Path) -> None:
    """An absolute primary's relative defaults resolve from its own directory."""
    primary_root = tmp_path / "primary"
    primary_root.mkdir()
    (primary_root / "shared.yaml").write_text("# @package _global_\n\nshared_value: from_shared_dir\n")

    primary_config = primary_root / "app.yaml"
    primary_config.write_text(
        "# @package _global_\n\ndefaults:\n  - shared\n  - _self_\n\nprimary_value: from_absolute_config\n"
    )

    config, _ = parse_untyped_config(str(primary_config), [])

    assert config.primary_value == "from_absolute_config"
    assert config.shared_value == "from_shared_dir"


def test_importing_instant_nurec_schema_does_not_load_parser_stack() -> None:
    result = run(
        [
            executable,
            "-c",
            "import sys; from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig; "
            "assert InstantNuRecConfig; "
            "assert not {'hydra', 'nurec.config.resolvers', 'nurec.instant_nurec.config.parse', "
            "'nurec.utils.model_registry'}.intersection(sys.modules)",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_pretrained_descriptor_errors_name_the_contract() -> None:
    with pytest.raises(TypeError, match="in_process_instant_nurec.shared_config.pretrained must be a config"):
        materialize_pretrained_instant_nurec_config(DictConfig({"shared_config": {"pretrained": "invalid"}}))

    with pytest.raises(ValueError, match="model_config, model_cache_dir"):
        _download_pretrained_instant_nurec_artifacts(DictConfig({"model_url": "https://example.com/model.ckpt"}))


@pytest.mark.parametrize(
    ("plaintext_field", "plaintext_url"),
    [
        ("model_url", "http://api.ngc.nvidia.com/models/model.ckpt"),
        ("model_config", "http://gitlab.example.com/models/model.yaml"),
    ],
)
def test_pretrained_descriptor_rejects_plaintext_registry_urls(
    plaintext_field: str,
    plaintext_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Register the host so the URL reaches the plaintext-http check instead of "Unsupported domain".
    monkeypatch.setenv("NUREC_GITLAB_DOMAINS", "gitlab.example.com")
    cache_dir = tmp_path / "model-version"
    cache_dir.mkdir()
    # A valid cached checkpoint keeps the model_config case local and proves
    # that no authenticated request is needed before its URL is rejected.
    (cache_dir / "model.ckpt").write_bytes(b"x" * ModelRegistry.MIN_FILE_SIZE)
    descriptor = DictConfig(
        {
            "model_url": "https://api.ngc.nvidia.com/models/model.ckpt",
            "model_config": str(tmp_path / "model.yaml"),
            "model_cache_dir": str(cache_dir),
        }
    )
    descriptor[plaintext_field] = plaintext_url

    with (
        patch("nurec.utils.model_registry.requests.Session.request") as request,
        pytest.raises(ModelRegistryError, match="Invalid model URL"),
    ):
        _download_pretrained_instant_nurec_artifacts(descriptor)

    request.assert_not_called()


@patch("nurec.instant_nurec.config.parse.create_model_registry")
@patch("nurec.instant_nurec.config.parse.parse_untyped_config")
def test_pretrained_models_resolve_through_the_registry(
    parse_untyped_config: MagicMock,
    create_model_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """instant_nurec resolves pretrained checkpoint + config URLs through the model registry."""
    cache_dir = tmp_path / "model-version"
    checkpoint_path = cache_dir / "model.ckpt"
    model_config_path = cache_dir / "model.yaml"
    parse_untyped_config.return_value = (
        DictConfig(
            {
                "model_url": "https://api.ngc.nvidia.com/models/model.ckpt",
                "model_config": "https://api.ngc.nvidia.com/models/model.yaml",
                "model_cache_dir": str(cache_dir),
                "model_config_overrides": ["model.from_descriptor=true"],
            }
        ),
        "unused-search-path",
    )
    create_model_registry.side_effect = [
        MagicMock(get_model=MagicMock(return_value=str(checkpoint_path))),
        MagicMock(get_model=MagicMock(return_value=str(model_config_path))),
    ]

    config_name, hydra_args = parse_pretrained_instant_nurec_config(
        "pretrained.yaml",
        ["model.from_caller=true"],
    )

    assert config_name == str(model_config_path)
    parse_untyped_config.assert_called_once_with("pretrained.yaml", [])
    assert hydra_args == [
        "model.from_descriptor=true",
        "++resume_weights_only=true",
        "model.from_caller=true",
        f"++resume={checkpoint_path}",
    ]
    assert create_model_registry.call_args_list == [
        call("https://api.ngc.nvidia.com/models/model.ckpt", cache_dir),
        call("https://api.ngc.nvidia.com/models/model.yaml", cache_dir),
    ]


@patch("nurec.instant_nurec.config.parse.create_model_registry")
@patch("nurec.instant_nurec.config.parse.parse_untyped_config")
def test_nested_pretrained_config_materializes_then_applies_in_process_overlay(
    parse_untyped_config: MagicMock,
    create_model_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """The downloaded YAML is the base; the already-composed in-process subtree wins."""
    cache_dir = tmp_path / "model-version"
    checkpoint_path = cache_dir / "model.ckpt"
    model_config_path = cache_dir / "model.yaml"
    in_process_overlay = DictConfig(
        {
            "shared_config": {
                "pretrained": {
                    "model_url": "https://api.ngc.nvidia.com/models/model.ckpt",
                    "model_config": "https://api.ngc.nvidia.com/models/model.yaml",
                    "model_cache_dir": str(cache_dir),
                    "model_config_overrides": ["model.from_descriptor=true"],
                }
            },
            "system": {"predict_num_workers": 7},
        }
    )
    downloaded = DictConfig(
        {
            "resume": str(checkpoint_path),
            "resume_weights_only": True,
            "model": {"freeze_encoder": True, "from_descriptor": True},
            "system": {"predict_num_workers": 1},
        }
    )
    parse_untyped_config.return_value = (downloaded, "unused-search-path")
    create_model_registry.side_effect = [
        MagicMock(get_model=MagicMock(return_value=str(checkpoint_path))),
        MagicMock(get_model=MagicMock(return_value=str(model_config_path))),
    ]

    materialized = materialize_pretrained_instant_nurec_config(in_process_overlay)

    assert materialized.resume == str(checkpoint_path)
    assert materialized.resume_weights_only is True
    assert materialized.model.freeze_encoder is True
    assert materialized.model.from_descriptor is True
    assert materialized.system.predict_num_workers == 7
    parse_untyped_config.assert_called_once_with(
        str(model_config_path),
        [
            "model.from_descriptor=true",
            "++resume_weights_only=true",
            f"++resume={checkpoint_path}",
        ],
    )
    assert create_model_registry.call_args_list == [
        call("https://api.ngc.nvidia.com/models/model.ckpt", cache_dir),
        call("https://api.ngc.nvidia.com/models/model.yaml", cache_dir),
    ]


@patch("nurec.instant_nurec.config.parse.parse_pretrained_instant_nurec_config")
@patch("nurec.instant_nurec.config.parse.parse_untyped_config")
def test_standalone_pretrained_config_upgrades_legacy_names(
    parse_untyped_config: MagicMock,
    parse_pretrained: MagicMock,
) -> None:
    parse_pretrained.return_value = ("downloaded.yaml", [])
    parse_untyped_config.return_value = (
        DictConfig({"dataset": {"name": "nrm-ncore"}, "system": {"name": "base-nrm-system"}}),
        "unused-search-path",
    )

    config = parse_untyped_instant_nurec_config("pretrained.yaml", [])

    assert config.dataset.name == "instant_nurec-ncore"
    assert config.system.name == "base-instant_nurec-system"


@patch("nurec.instant_nurec.config.parse.create_model_registry")
@patch("nurec.instant_nurec.config.parse.parse_untyped_config")
def test_nested_pretrained_config_upgrades_legacy_names(
    parse_untyped_config: MagicMock,
    create_model_registry: MagicMock,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "model-version"
    parse_untyped_config.return_value = (
        DictConfig({"dataset": {"name": "nrm-ncore"}, "system": {"name": "base-nrm-system"}}),
        "unused-search-path",
    )
    create_model_registry.side_effect = [
        MagicMock(get_model=MagicMock(return_value=str(cache_dir / "model.ckpt"))),
        MagicMock(get_model=MagicMock(return_value=str(cache_dir / "model.yaml"))),
    ]
    overlay = DictConfig(
        {
            "shared_config": {
                "pretrained": {
                    "model_url": "https://api.ngc.nvidia.com/models/model.ckpt",
                    "model_config": "https://api.ngc.nvidia.com/models/model.yaml",
                    "model_cache_dir": str(cache_dir),
                }
            }
        }
    )

    config = materialize_pretrained_instant_nurec_config(overlay)

    assert config.dataset.name == "instant_nurec-ncore"
    assert config.system.name == "base-instant_nurec-system"
