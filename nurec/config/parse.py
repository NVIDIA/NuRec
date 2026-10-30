# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import hydra

from omegaconf import DictConfig, OmegaConf, open_dict

from nurec.config.base_schema import BaseConfigSchema
from nurec.config.nurec import NuRecConfig
from nurec.config.resolvers import xor as xor
from nurec.instant_nurec.config.parse import materialize_pretrained_instant_nurec_config
from nurec.repo_root import __reporoot__
from nurec.utils.misc import rank_zero_only


@rank_zero_only
def dump_config(path: str, config: BaseConfigSchema) -> None:
    save_config = config.to_dictconfig()

    # Add the hydra specific part that prevents creating new directories automatically
    with open_dict(save_config):
        save_config["hydra"] = {"output_subdir": None, "job": {"chdir": False}, "run": {"dir": "."}}

    # Add the macro that tells hydra that the path to the config should not be included in the dict structure
    with open(path, "w") as fp:
        fp.write("# @package _global_ \n\n")
        OmegaConf.save(config=save_config, f=fp)


def parse_untyped_config(
    config_name: str,
    hydra_args: list[str] | tuple[str],
) -> DictConfig:
    """Compose and resolve an untyped Hydra configuration.

    Args:
        config_name: Configuration name relative to the repository's ``configs/`` tree, a
            path relative to the repository root that lies beneath it, or an absolute
            configuration path. For an absolute path, the file's parent becomes the
            composition root.
        hydra_args: Hydra overrides applied during composition.

    Returns:
        The composed and resolved configuration.
    """
    hydra_args_list: list[str] = list(hydra_args)  # convert to list in case we have a tuple

    config_dir_path = __reporoot__ / "configs"
    config_name_path = Path(str(config_name))

    # This enables defining relative configs from repo root (to enable autocomplete)
    if not config_name_path.is_absolute():
        config_name_from_repo_root = __reporoot__ / config_name_path
        if config_name_from_repo_root.is_relative_to(config_dir_path):
            config_name_path = config_name_from_repo_root.relative_to(config_dir_path)

    config_registry_dir = str(config_dir_path)

    if config_name_path.is_absolute():
        # Use config_name's directory as the composition root
        compose_root = str(config_name_path.parent)
        # Now config name is simply the file name
        config_name = config_name_path.parts[-1]

        # We still want to add NuRec's config registry to the search path
        hydra_args_list.append(f"hydra.searchpath=[{config_registry_dir}]")
    else:
        # Config name must be relative to config_registry_dir
        compose_root = config_registry_dir
        config_name = str(config_name_path)

    # Using Hydra context below for proper Hydra cleanup, so that subsequent calls to main()
    # e.g. from multiple tests or other scripts work properly.
    with hydra.initialize_config_dir(str(compose_root), version_base=None):
        config = hydra.compose(config_name=str(config_name), overrides=hydra_args_list)

    OmegaConf.resolve(config)

    return config


def assert_no_out_dir_override_in_resume(hydra_args: list[str]) -> None:
    """Checks hydra overrides to prevent changing the out_dir if resume is set and mode=train"""

    resume_true = any([param.startswith("resume=") for param in hydra_args])
    # Don't fail when using resume=auto with wandb sweeps
    resume_auto = any([(param.startswith("resume=") and "auto" in param) for param in hydra_args])
    mode_is_train = any([(param.startswith("mode=") and "train" in param) for param in hydra_args])
    if (
        (resume_true and not resume_auto)
        and mode_is_train
        and any([param.startswith("out_dir=") for param in hydra_args])
    ):
        raise AssertionError("out_dir cannot be changed when resuming training")


def parse_typed_config(config_name: str, hydra_args: list[str]) -> NuRecConfig:
    """Compose and validate an NuRec configuration against the repository ``configs/`` tree."""
    untyped_config = parse_untyped_config(config_name, hydra_args)
    if "in_process_instant_nurec" in untyped_config and untyped_config.in_process_instant_nurec is not None:
        # Resolve the effective outer run state before touching the pretrained
        # producer. In particular, NuRecConfig turns resume=auto into either a real
        # checkpoint or None. Validation/test and resumed runs already own their
        # Gaussian tensors, so they must not require a stale nested instant_nurec cache.
        in_process_instant_nurec = untyped_config.in_process_instant_nurec
        untyped_config.in_process_instant_nurec = None
        outer_config = NuRecConfig.model_validate(untyped_config, context={"config_name": config_name})
        if "train" not in outer_config.mode or outer_config.resume is not None:
            return outer_config

        # TODO: Support trainer.world_size > 1 after pretrained artifact
        # materialization and the in-memory Gaussian handoff have one end-to-end
        # validated rank-ownership protocol. The typed outer config is the
        # pre-materialization topology contract; externally launched processes
        # must provide matching trainer values. The initialized process group's
        # actual world size is checked again at the runtime boundary.
        if outer_config.trainer.world_size != 1:
            raise ValueError(
                f"in_process_instant_nurec requires trainer.world_size=1; resolved to {outer_config.trainer.world_size}. "
                "Set trainer.world_size=1 or remove in_process_instant_nurec."
            )

        # Carry only the effective resume decision back to the original Hydra
        # tree. Revalidating the normalized outer model would repeat non-idempotent
        # nested initialization such as WandB path and tag setup.
        untyped_config.resume = outer_config.resume
        untyped_config.in_process_instant_nurec = materialize_pretrained_instant_nurec_config(in_process_instant_nurec)
    typed_config = NuRecConfig.model_validate(untyped_config, context={"config_name": config_name})
    return typed_config
