# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import platform
import sys

from contextlib import contextmanager, nullcontext
from typing import Any, Generator

import click
import pytorch_lightning as pl
import torch

from pytorch_lightning.plugins.precision.amp import MixedPrecision
from pytorch_lightning.strategies import SingleDeviceStrategy, Strategy
from torch.cuda.amp.grad_scaler import GradScaler

import nurec.systems

from apps.viewer.lightning_viewer import maybe_create_lightning_viewer
from nurec import __version__
from nurec.config.nurec import NuRecConfig
from nurec.config.parse import assert_no_out_dir_override_in_resume, dump_config, parse_typed_config
from nurec.models.gaussians.initialization_pipeline import configure_initialization, parallel_initialization_eligible
from nurec.systems.base import BaseSystem, DDPStrategySO
from nurec.systems.gaussians import InProcessInstantNuRecInitializer
from nurec.utils.callbacks import PreemptionInterruptException, make_callbacks, make_logger
from nurec.utils.glibc_malloc import tune_glibc_malloc
from nurec.utils.initialization import InitializationResources
from nurec.utils.misc import rank_zero_info, rank_zero_only, unpack_optional
from nurec.utils.prober import get_global_prober
from nurec.utils.profiling import ScopedTimer, create_profiler
from nurec.utils.trainer import adjust_step_for_world_size


def setup_environment_and_logger(config: NuRecConfig) -> logging.Logger:
    # Configure PyTorch matmul precision for better performance
    torch.set_float32_matmul_precision(config.matmul_precision)
    # Reduced precision is meant for CUDA (TF32); on aarch64 it would also run CPU matmuls in bf16.
    torch.backends.mkldnn.matmul.fp32_precision = "ieee"  # type: ignore[attr-defined]

    if (local_rank_env := os.environ.get("LOCAL_RANK")) is not None:
        local_rank = int(local_rank_env)
        torch.cuda.set_device(local_rank)
        logging.getLogger(__name__).info(
            f"[info] Distributed training detected. LOCAL_RANK = {local_rank}, device = {torch.cuda.current_device()}"
        )
    else:
        local_rank = 0

    # Warn if the CPU frequency governor is not 'performance'. ARM hosts (e.g. GB300/
    # Grace) are host-dispatch bound for this workload, where the 'ondemand' governor
    # pins the latency-bound dispatch thread near the minimum clock and inflates
    # per-step time. Restricted to ARM since that's where it dominates in practice.
    if platform.machine().lower() in ("aarch64", "arm64") and local_rank == 0:
        governor: str | None = None
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as governor_file:
                governor = governor_file.read().strip()
        except OSError:
            pass  # cpufreq not exposed (VM/container/non-Linux)
        if governor is not None and governor != "performance":
            logging.getLogger(__name__).warning(
                "CPU frequency governor is '%s', which is expected to significantly "
                "slow training on host-dispatch-bound platforms (e.g GB300/Grace). "
                "Fix: sudo cpupower frequency-set -g performance",
                governor,
            )

    # If set, configure torch's default allocation block size - appending to any existing key:value configs.
    if (split_size := config.system.max_split_size_mb) is not None:
        new_entry = f"max_split_size_mb:{split_size}"
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_entry if conf is None else ",".join([conf, new_entry])

    logger = logging.getLogger(__name__)
    if config.verbose:
        logger.setLevel(logging.DEBUG)

    if config.version != __version__:
        logger.warning(f"Using config for different NuRec version ({config.version=}) than current {__version__=}.")

    # Use the same seed for model initialization across all processes
    pl.seed_everything(config.seed, workers=True)

    logger.info("Setting up TensorProber with config")
    get_global_prober().configure(config.prober)

    if config.scopedtimer is not None:
        ScopedTimer.set_global_config(config.scopedtimer)
        if config.scopedtimer.logfile is not None and config.scopedtimer.enabled:
            logfilename = (
                config.scopedtimer.logfile
                if os.path.isabs(config.scopedtimer.logfile)
                else os.path.join(config.out_dir, config.run_id, config.scopedtimer.logfile)
            )
            ScopedTimer.set_global_logfile(logfilename)
        else:
            ScopedTimer.set_global_print_func(logger.info)

    return logger


class _DevicePreservingStrategy(Strategy):
    """Keep model and optimizer tensors on their devices when a run finishes."""

    def teardown(self) -> None:
        """Release plugins without offloading live tensors.

        Tensor memory is released when their owners are released or the process exits.
        Trainer still performs loop, logger, signal, and user-hook cleanup.

        Returns:
            None.

        Raises:
            AssertionError: If the strategy has no accelerator attached.
        """
        self.precision_plugin.teardown()
        assert self.accelerator is not None
        self.accelerator.teardown()
        self.checkpoint_io.teardown()


class _SingleDeviceStrategySO(SingleDeviceStrategy, _DevicePreservingStrategy):
    """Preserve model and optimizer devices for single-device runs."""


class _DDPDevicePreservingStrategySO(DDPStrategySO, _DevicePreservingStrategy):
    """Run SO, DDP, and parallel cleanup before device-preserving cleanup.

    The shared Strategy base places _DevicePreservingStrategy before Strategy in
    the MRO, replacing only its final model and optimizer CPU offload.
    """


class MixedPrecisionNoCastPlugin(MixedPrecision):
    """
    This plugin uses the behavior of the original `MixedPrecision`, 
    but only uses the grad scaler and skip the global autocasting to half precision.
    Local half-precision casting is still possible when this plugin is enabled.
    
    Global autocasting to half is bad because:
    - We would never know which module breaks or misbehaves in half precision. \
        There will always be unknown unpredictable omission. \
    - When we add new modules, global autocasting make it difficult to debug, \
        while some existing modules must use grad scaler.
    So instead, we should disable autocast globally, and only enable it locally in known modules.
    
    Known issues on half precision (where you should not use half):
    - For rays, poses and coordinates, half precision introduce very large errors. We should at least use float.
    - Some of the pytorch module is not autocast-safe (e.g. bce, exp, ...)
    - ... TBA
    """

    def __init__(self, *args, **kwargs) -> None:
        # Use a smaller value of `init_scale` then the default `2**16==65536`
        kwargs["scaler"] = GradScaler(init_scale=128.0)
        super().__init__(*args, **kwargs)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Skip the global autocast in the original plugin."""
        yield


def cap_cpu_threads_per_rank(world_size: int, devices_per_node: int) -> None:
    """Split the CPUs evenly between the ranks on a node, unless OMP_NUM_THREADS is already set.

    Lightning starts the other ranks after rank 0 has built the system, and each rank then builds its own at the same
    time; uncapped, every rank's CPU thread pools span all cores and the ranks can miss the process-group timeout.
    Ranks inherit the variable.
    """
    if world_size <= 1 or "OMP_NUM_THREADS" in os.environ:
        return
    threads = max(1, len(os.sched_getaffinity(0)) // devices_per_node)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)


def launch_trainer_loop(config: NuRecConfig, system: BaseSystem, logger: logging.Logger) -> None:
    pl_logger = make_logger(config.logger)

    # Make the pytorch lightning callback functions
    callbacks = make_callbacks(config)

    # Make a visualizer callback if enabled
    if (viewer_callback := maybe_create_lightning_viewer(system, config.viewer)) is not None:
        callbacks.append(viewer_callback)

    # Optionally, skip autocast and only use grad_scaler when using half precision
    precision: Any = config.trainer.precision
    plugins = config.trainer.plugins

    # Optionally, skip autocast and only use grad_scaler when using half precision
    if config.trainer.precision == "16-no-cast":
        precision = "16-mixed"  # Change it back to `16` or `16-mixed`, otherwise PL complains.
        if plugins is None:
            plugins = []
        plugins.append(MixedPrecisionNoCastPlugin("16-mixed", "cuda"))

    profiler = create_profiler(config.to_dictconfig()) if config.profiling.enabled else None

    if config.trainer.world_size > 1:
        # Customize DDP strategy for Scene Optimization
        # TODO: We should not find unused parameters here. Check later if there is a way around it. Maybe by doing some dummy operation or setting LR to 0.
        import datetime

        nccl_timeout = datetime.timedelta(minutes=config.trainer.nccl_timeout_minutes)
        strategy = _DDPDevicePreservingStrategySO(find_unused_parameters=True, timeout=nccl_timeout)  # type: ignore
    else:
        strategy = _SingleDeviceStrategySO(device=torch.device("cuda", 0))  # type: ignore

    trainer = pl.Trainer(
        devices=unpack_optional(config.trainer.device_count),
        callbacks=callbacks,
        logger=pl_logger,
        strategy=strategy,
        limit_val_batches=1.0 if "val" in config.mode else 0.0,
        profiler=profiler,
        reload_dataloaders_every_n_epochs=system.update_n_epochs,
        precision=precision,
        plugins=plugins,
        max_epochs=config.trainer.max_epochs,
        max_steps=config.trainer.max_steps,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        val_check_interval=config.trainer.val_check_interval,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_progress_bar=config.trainer.enable_progress_bar,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        num_nodes=config.trainer.num_nodes,
        detect_anomaly=config.trainer.detect_anomaly,
    )

    if config.resume is not None and not os.path.exists(config.resume):
        raise FileNotFoundError(f"Checkpoint {config.resume=} to be resumed does not exist")

    # NuRec checkpoints carry enums and RNG/loop state, which torch's weights-only default rejects.
    if "train" in config.mode or "trainval" in config.mode:
        if config.resume and not config.resume_weights_only:
            trainer.fit(system, datamodule=system.datamodule, ckpt_path=config.resume, weights_only=False)
        else:
            trainer.fit(system, datamodule=system.datamodule)
    elif "val" in config.mode:
        trainer.validate(system, datamodule=system.datamodule, ckpt_path=config.resume, weights_only=False)
    elif "test" in config.mode:
        trainer.test(system, datamodule=system.datamodule, ckpt_path=config.resume, weights_only=False)


@click.command("main")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.argument("hydra-args", nargs=-1)
def main(
    config_name: str,
    hydra_args: list[str],
) -> None:
    """Main entry point for Neural Reconstruction Engine training, validation and testing"""

    # The mmap threshold only applies to later allocations, so set it before initialization starts allocating.
    tune_glibc_malloc()
    assert_no_out_dir_override_in_resume(hydra_args)
    config = parse_typed_config(config_name=config_name, hydra_args=hydra_args)

    # Check the resolved path before initialization, without changing export-only config parsing.
    # Keep the trainer-loop check in case the checkpoint disappears during startup.
    if config.resume is not None and not os.path.exists(config.resume):
        raise FileNotFoundError(f"Checkpoint {config.resume=} to be resumed does not exist")

    in_process_instant_nurec_initializer: InProcessInstantNuRecInitializer | None = None
    # Keep the instant_nurec dependency lazy so runs without in-process instant_nurec retain the existing
    # startup behavior. Binding the callback here does not run instant_nurec: the system
    # invokes it later from setup, after Lightning assigns the runtime device. The
    # Bazel run target still carries the optional producer code.
    if config.in_process_instant_nurec is not None:
        from nurec.run.in_process_instant_nurec import initialize_from_instant_nurec_in_memory

        in_process_instant_nurec_initializer = initialize_from_instant_nurec_in_memory

    parallel_init = parallel_initialization_eligible(config)
    with InitializationResources() if parallel_init else nullcontext() as resources:
        if resources is not None:
            configure_initialization(resources, config)
        _run(config, in_process_instant_nurec_initializer=in_process_instant_nurec_initializer)


def _run(
    config: NuRecConfig, *, in_process_instant_nurec_initializer: InProcessInstantNuRecInitializer | None = None
) -> None:
    # Save the parsed config now in case we need to verify it.
    rank_zero_only(os.makedirs)(config.config_dir, exist_ok=True)
    dump_config(os.path.join(config.config_dir, "parsed.yaml"), config)
    logger = setup_environment_and_logger(config)

    # Other rank might have different run_ids, but this does not seem to be a problem since wandb is lazy-init and
    # will hence only take effect on rank_0. Otherwise it might be hard to all_gather run_id since NCCL is not
    # fully set up until this point.
    rank_zero_info("RUN 🆔: %s", config.run_id)

    # By default Pytorch Lightning stores the checkpoints at the end of validation. If no validation is performed
    # the checkpoints are never stored. Checkpoint callback has a parameter 'save_on_train_epoch_end' that is used
    # to circumvent this and we set it to true. However the parameter is ignored if 'check_val_every_n_epoch' is not 1.
    # This ugly hack sets 'check_val_every_n_epoch' to 1 when we are not doing validation (the parameter is not used anyways)
    if "val" not in config.mode:
        config.trainer.check_val_every_n_epoch = 1
        logger.warning(
            "The parameter 'check_val_every_n_epoch' overridden to 1 to enable storing checkpoints. "
            "This has no effect on the training process and can be safely ignored."
        )

    # If checkpointing is enabled, check that the parameters are compatible. If 'every_n_epochs == None', PL saves a checkpoint
    # at the end of every epoch (equivalent to 'every_n_epochs == 1').
    if config.checkpoint.every_n_train_steps is not None and config.checkpoint.every_n_train_steps > 0:
        # Adjust the checkpointing interval to ensure that the last checkpoint is saved
        # Estimate the number of checkpoints to save
        n_global_steps = config.trainer.max_epochs * config.dataset.n_samples_per_epoch
        n_checkpoints_to_save = (
            n_global_steps + config.checkpoint.every_n_train_steps - 1
        ) // config.checkpoint.every_n_train_steps
        # Recompute the checkpoint interval to ensure that the last checkpoint is saved
        n_actual_steps = config.trainer.max_epochs * adjust_step_for_world_size(
            config.trainer, config.dataset.n_samples_per_epoch
        )
        config.checkpoint.every_n_train_steps = (n_actual_steps + n_checkpoints_to_save - 1) // n_checkpoints_to_save
        logger.info(f"Number of checkpoints to save: {n_checkpoints_to_save}")
        logger.info(f"Number of training steps: {n_actual_steps}")
        logger.info(f"Checkpoint interval: {config.checkpoint.every_n_train_steps}")

        if n_actual_steps % config.checkpoint.every_n_train_steps != 0:
            raise ValueError(
                f"The product of 'max_epochs' and 'n_samples_per_epoch' ({n_actual_steps}) is not divisible by "
                f"'checkpoint.every_n_train_steps' ({config.checkpoint.every_n_train_steps}). This means no "
                f"checkpoint will be saved at the end of training."
            )
        if config.checkpoint.every_n_train_steps > n_actual_steps:
            raise ValueError(
                f"The parameter 'checkpoint.every_n_train_steps' ({config.checkpoint.every_n_train_steps}) is larger "
                f"than the product of 'max_epochs' ({config.trainer.max_epochs}) and 'n_samples_per_epoch' "
                f"({config.dataset.n_samples_per_epoch}). This means no checkpoint will be saved at all."
            )

    # Sanity check if multi-GPU configuration is correct. Do this before system is created to avoid loading data unnecessarily.
    if config.trainer.world_size < 0:
        raise ValueError(
            f"The parameter 'trainer.world_size' ({config.trainer.world_size}) is negative. This indicates that "
            f"'trainer.world_size' is configured incorrectly. Please set 'trainer.world_size' correctly to the number "
            f"of total GPUs across all nodes to use for training."
        )
    if config.trainer.num_nodes < 0:
        raise ValueError(
            f"The parameter 'trainer.num_nodes' ({config.trainer.num_nodes}) is negative. This indicates that "
            f"'trainer.num_nodes' is configured incorrectly. Please set 'trainer.num_nodes' correctly to the number "
            f"of nodes to use for training."
        )

    if unpack_optional(config.trainer.device_count) > torch.cuda.device_count():
        raise ValueError(
            f"The parameter 'trainer.device_count' ({config.trainer.device_count}) is larger than the number of "
            f"available GPUs ({torch.cuda.device_count()}). If multi-node training is intended, please set "
            f"'+trainer.device_count' to the number of GPUs per node and 'trainer.num_nodes' to the number of nodes "
            f"in the command line."
        )

    cap_cpu_threads_per_rank(config.trainer.world_size, unpack_optional(config.trainer.device_count))

    # Make the system
    checkpoint = None if (not config.resume_weights_only or not config.resume) else config.resume
    system = nurec.systems.make(
        config.system.name,
        config,
        load_from_checkpoint=checkpoint,
        in_process_instant_nurec_initializer=in_process_instant_nurec_initializer,
    )

    try:
        launch_trainer_loop(config, system, logger)
    except PreemptionInterruptException as e:
        rank_zero_info(f"Preemption detected: {e}. Exiting with code 1.")
        sys.exit(1)

    ScopedTimer.print_summary()
