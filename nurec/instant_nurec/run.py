# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from typing import Any

import click
import click_default_group
import pytorch_lightning as pl
import yaml

from lightning_utilities.core.rank_zero import rank_zero_info
from omegaconf import OmegaConf

import nurec.instant_nurec.systems

from apps.viewer.lightning_viewer import maybe_create_lightning_viewer
from nurec.config.parse import assert_no_out_dir_override_in_resume, dump_config
from nurec.config.version import get_version
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.instant_nurec.config.parse import parse_typed_instant_nurec_config
from nurec.instant_nurec.systems.base import BaseInstantNuRecSystem
from nurec.instant_nurec.utils.run_helpers import resolve_has_full_init, resolve_strategy, setup_environment_and_logger
from nurec.utils.callbacks import TimingLogger, TQDMProgressBar, make_logger
from nurec.utils.misc import rank_zero_only, unpack_optional


def launch_predict(config: InstantNuRecConfig, system: BaseInstantNuRecSystem, logger: logging.Logger) -> None:
    pl_logger = make_logger(config.logger)

    callbacks = [TQDMProgressBar(refresh_rate=1), TimingLogger()]

    # Make a visualizer callback if enabled
    if (viewer_callback := maybe_create_lightning_viewer(system, config.viewer)) is not None:
        callbacks.append(viewer_callback)

    precision: Any = config.system.precision
    strategy = resolve_strategy(config)

    trainer = pl.Trainer(
        devices=unpack_optional(config.system.device_count),
        callbacks=callbacks,
        logger=pl_logger,
        strategy=strategy,
        precision=precision,
        log_every_n_steps=config.system.log_every_n_steps,
        enable_progress_bar=True,
        num_nodes=config.system.num_nodes,
    )

    ckpt_path = config.resume if (config.resume and not config.resume_weights_only) else None

    # When there's no checkpoint but init weights are provided, call the model's weight-init hook directly
    # (it otherwise only runs at the start of a training loop). Set call_train_from_scratch_hook_for_validation=False
    # to skip this, e.g. if the hook has train-only side effects.
    has_full_init = resolve_has_full_init(config.model)
    if config.call_train_from_scratch_hook_for_validation and ckpt_path is None and has_full_init:
        system.model.on_train_from_scratch_start(system)

    # Set return_predictions to False since we return primitives which is memory-consuming.
    trainer.predict(system, datamodule=system.datamodule, ckpt_path=ckpt_path, return_predictions=False)


@click.command("main")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.argument("hydra-args", nargs=-1)
def main(config_name: str, hydra_args: list[str]) -> None:
    """Main entry point for NuRec-instant_nurec inference (predict)"""

    assert_no_out_dir_override_in_resume(hydra_args)
    config = parse_typed_instant_nurec_config(config_name=config_name, hydra_args=hydra_args)

    # Save the parsed config at early stage
    rank_zero_only(os.makedirs)(config.config_dir, exist_ok=True)
    dump_config(os.path.join(config.config_dir, "parsed.yaml"), config)

    logger = setup_environment_and_logger(config, distributed_log_label="Distributed inference detected")
    #   Other rank might have different run_ids, but this does not seem to be a problem since wandb is lazy-init and
    # will hence only take effect on rank_0.
    #   [Otherwise it might be hard to all_gather run_id since NCCL is not fully set up until this point]
    rank_zero_info("instant_nurec RUN 🆔: %s", config.run_id)

    # Make the system
    checkpoint = None if (not config.resume_weights_only or not config.resume) else config.resume
    system = nurec.instant_nurec.systems.make(
        config.system.name,
        config,
        load_from_checkpoint=checkpoint,
    )

    launch_predict(config, system, logger)


@click.command("parse_config")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.argument("hydra-args", nargs=-1)
def parse_config(config_name: str, hydra_args: list[str]) -> None:
    """Parse and print the instant_nurec configuration to surface any configuration errors"""

    assert_no_out_dir_override_in_resume(hydra_args)
    config = parse_typed_instant_nurec_config(config_name=config_name, hydra_args=hydra_args)

    # Convert config to a dictionary and print as YAML
    config_dict = OmegaConf.to_container(config.to_dictconfig(), resolve=True)
    print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
    print("\n✓ Configuration parsed successfully!")


@click.group(cls=click_default_group.DefaultGroup, default="main", default_if_no_args=True)
@click.version_option(version=str(unpack_optional(get_version(), default="version-not-available")))
def cli():
    pass


cli.add_command(main)
cli.add_command(parse_config)

if __name__ == "__main__":
    cli(show_default=True)
