# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import hydra

from omegaconf import DictConfig, OmegaConf, open_dict

# Register the common OmegaConf resolvers before composing instant_nurec configs.
import nurec.config.resolvers  # noqa: F401

from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.repo_root import __reporoot__
from nurec.utils.model_registry import create_model_registry
from nurec.utils.upgrade.config import upgrade_config


def parse_untyped_config(
    config_name: str,
    hydra_args: list[str],
) -> tuple[DictConfig, str]:
    """Compose an untyped instant_nurec config against the repository's public ``configs/`` tree.

    Configuration names may be relative to ``configs/`` or relative to the repository
    root beneath it. For an absolute configuration name, the file's parent
    is the composition root. The ``configs/`` root is also added to Hydra's search path.

    Args:
        config_name: Configuration name relative to ``configs/``, a path relative to
            the repository root that lies beneath it, or an absolute path.
        hydra_args: Hydra overrides applied during composition.

    Returns:
        The composed configuration and the Hydra search-path override appended during
        composition.
    """

    config_registry_dir = __reporoot__ / "configs"
    config_name_path = Path(config_name)
    if not config_name_path.is_absolute():
        config_name_from_repo_root = __reporoot__ / config_name_path
        if config_name_from_repo_root.is_relative_to(config_registry_dir):
            config_name_path = config_name_from_repo_root.relative_to(config_registry_dir)
        config_name_path = config_registry_dir / config_name_path

    # Adding repo config path to hydra search path so we can refer to those in overrides
    # Example override: +instant_nurec/dataset/concrete@dataset.predict=ncore_local_low_res
    search_path_config: str = f"hydra.searchpath=['{config_registry_dir}']"

    # Use hydra context for clean-up.
    # We always assume that the main config dir is the one containing config_name, and rest configs are found via search paths.
    with hydra.initialize_config_dir(str(config_name_path.parent.resolve()), version_base=None):
        config = hydra.compose(config_name=str(config_name_path.name), overrides=hydra_args + [search_path_config])

    return config, search_path_config


def _download_pretrained_instant_nurec_artifacts(config: DictConfig) -> tuple[str, str, list[str]]:
    """Resolve a pretrained descriptor through the existing model registry."""

    required_fields = ("model_url", "model_config", "model_cache_dir")
    missing_fields = [field for field in required_fields if config.get(field) is None]
    if missing_fields:
        raise ValueError(f"Pretrained instant_nurec descriptor is missing required fields: {', '.join(missing_fields)}")

    model_url = str(config["model_url"])
    model_config = str(config["model_config"])
    model_cache_dir = Path(config["model_cache_dir"])

    model_path = create_model_registry(model_url, model_cache_dir).get_model()
    if model_config.startswith("http"):
        model_config = create_model_registry(model_config, model_cache_dir).get_model()

    return model_config, model_path, list(config.get("model_config_overrides", []))


def parse_pretrained_instant_nurec_config(
    config_name: str,
    hydra_args: list[str],
) -> tuple[str, list[str]]:
    """Resolve a pretrained instant_nurec config and checkpoint when requested.

    A pretrained configuration may retrieve its checkpoint and associated configuration
    through the model registry. When that happens, this function inserts
    ``++resume_weights_only=true`` into the caller's ``hydra_args`` list.

    Args:
        config_name: Configuration to inspect for pretrained model references.
        hydra_args: Mutable Hydra overrides for the resolved configuration.

    Returns:
        The effective configuration name and Hydra arguments, including configured
        model overrides and resume overrides for a pretrained model.

    Raises:
        AssertionError: If a pretrained configuration is combined with a caller-provided
            ``resume`` override.
    """

    # load basic config without overwrites
    config, _ = parse_untyped_config(config_name, [])

    if config.get("model_url", None) is not None and config.get("model_config", None) is not None:
        assert not any([param.startswith("resume=") for param in hydra_args]), (
            "resume cannot be used with pretrained model configs"
        )

        model_config, model_path, model_config_overrides = _download_pretrained_instant_nurec_artifacts(config)

        # Models stored in the registry often only have the state dict kept.
        hydra_args.insert(0, "++resume_weights_only=true")

        # Allow for overrides to the downloaded model config
        hydra_args = model_config_overrides + hydra_args

        # use the downloaded checkpoint as the main entrypoint for the downstream config
        hydra_args.append(f"++resume={model_path}")  # points to the absolute path of the downloaded checkpoint
        config_name = (
            model_config  # points to the absolute path of the checkpoint-associated config to load, or a local config
        )

    return config_name, hydra_args


def materialize_pretrained_instant_nurec_config(config: DictConfig) -> DictConfig:
    """Build a complete in-process instant_nurec subtree from a pretrained descriptor.

    Standalone instant_nurec can replace its Hydra root with the downloaded model YAML in
    :func:`parse_pretrained_instant_nurec_config`. In-process instant_nurec is nested inside an NuRec
    config, so replacing the root would discard the surrounding NuRec run. This
    function applies the same pretrained-model flow only to the nested subtree.
    """

    # The in-process preset is intentionally only an overlay. It relocates the
    # existing standalone pretrained descriptor here so the outer Hydra compose
    # can retain the NuRec config as its root.
    pretrained_config = OmegaConf.select(config, "shared_config.pretrained")
    # A caller may instead supply an already-complete local instant_nurec subtree. There
    # is no registry-backed base to materialize in that case.
    if pretrained_config is None:
        return config
    # Fail at the in-process-instant_nurec contract boundary rather than passing a malformed
    # descriptor to the shared registry helper and producing an indirect error.
    if not isinstance(pretrained_config, DictConfig):
        raise TypeError("in_process_instant_nurec.shared_config.pretrained must be a config")

    # Reuse standalone instant_nurec's registry boundary so both entrypoints resolve and
    # cache the checkpoint and model YAML while preserving descriptor-owned
    # compatibility overrides.
    model_config, model_path, model_config_overrides = _download_pretrained_instant_nurec_artifacts(pretrained_config)
    # The in-process runtime supports the pretrained weights-only load boundary, not
    # restoration of standalone instant_nurec trainer state. Mirror standalone pretrained
    # parsing: apply descriptor overrides first, then force the cached checkpoint
    # through that supported path. `++` works whether the downloaded YAML already
    # declares the two resume fields or leaves them absent.
    pretrained_overrides = [
        *model_config_overrides,
        "++resume_weights_only=true",
        f"++resume={model_path}",
    ]
    # The downloaded YAML may contain a Hydra defaults list and repo config-group
    # references, so it must be composed through the normal instant_nurec Hydra search path
    # rather than loaded as a standalone YAML mapping.
    downloaded_config, _ = parse_untyped_config(model_config, pretrained_overrides)

    # Detach a copy before consuming the temporary descriptor so this helper does
    # not mutate the caller's composed NuRec tree. `resolve=False` also preserves
    # unresolved values when this helper is called independently; they are
    # resolved after the final merge below.
    in_process_overlay = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    # The descriptor has now served its only purpose. Remove just that temporary
    # metadata while preserving other shared_config values used by relocated instant_nurec
    # options (for example, shared_config.n_chunks). The scoped mutation remains
    # valid if the OmegaConf subtree is structured.
    with open_dict(in_process_overlay.shared_config):
        del in_process_overlay.shared_config.pretrained

    # The downloaded YAML supplies the complete model/system/dataset base. The
    # already-composed in-process overlay is merged second so native Hydra group
    # and in_process_instant_nurec.* CLI choices retain normal caller-wins precedence. Hydra may
    # mark composed configs as structured, hence the scoped open_dict mutation.
    with open_dict(downloaded_config):
        downloaded_config.merge_with(in_process_overlay)
    # Guarantee a resolved result for typed InstantNuRecConfig validation. When an
    # independently supplied overlay still contains interpolations, resolving
    # here lets them observe the complete merged in-process instant_nurec tree.
    OmegaConf.resolve(downloaded_config)
    return upgrade_config(downloaded_config)


def parse_untyped_instant_nurec_config(
    config_name: str,
    hydra_args: list[str],
) -> DictConfig:
    """Compose and resolve an untyped instant_nurec configuration.

    The instant_nurec search root allows Hydra overrides to reference other instant_nurec configuration
    groups.

    Args:
        config_name: Primary instant_nurec configuration to compose.
        hydra_args: Hydra overrides applied during composition.

    Returns:
        The composed and resolved configuration.
    """

    # Load pretrained config/checkpoint if specified
    config_name, hydra_args = parse_pretrained_instant_nurec_config(config_name, hydra_args)

    # Load the config
    config, _ = parse_untyped_config(config_name, hydra_args)

    OmegaConf.resolve(config)

    return upgrade_config(config)


def parse_typed_instant_nurec_config(
    config_name: str,
    hydra_args: list[str] | tuple[str],
) -> InstantNuRecConfig:
    """Compose and validate an instant_nurec configuration."""
    untyped_config = parse_untyped_instant_nurec_config(
        config_name,
        # convert to list in case we have a tuple
        list(hydra_args),
    )
    typed_config = InstantNuRecConfig.model_validate(untyped_config, context={"config_name": config_name})
    return typed_config
