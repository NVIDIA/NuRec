# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import logging
import os
import warnings

from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Generator, Generic, Literal, Mapping, Tuple, TypeVar, Union, cast

import numpy as np
import point_cloud_utils as pcu
import torch

from ppisp.report import export_ppisp_report  # type: ignore[import-not-found]
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer, do_nothing_closure
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig as PL_OptimizerLRSchedulerConfig
from scipy.spatial.transform import Rotation
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nurec.datamodules

from apps.av.road_mesh.ground_mesh import (
    GroundMeshReconstructionSettings,
    get_nominal_ground_point_under_lidar,
    reconstruct_ground_mesh_from_points,
)
from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from ncore.sensors import OpenCVPinholeCameraModel
from nurec.artifact import SEMANTIC_CLASS_COLORS_FILENAME, SEMANTIC_CLASS_NAMES_FILENAME, TRACK_SPLIT_METADATA_FILENAME
from nurec.compression.usdz import write_artifact_cache
from nurec.config.base_schema import config_to_primitive
from nurec.config.model import PPISPPostProcessingConfig
from nurec.config.nurec import NuRecConfig
from nurec.config.systems import (
    BaseSystemConfig,
    GaussiansSystemConfig,
)
from nurec.datamodules.base import PrefetchDataLoader, SODataModule
from nurec.datasets.base import RigTrajectoriesProvider
from nurec.datasets.ncore import NCOREDataSource
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.harmonizer.model import HarmonizerModelFactory
from nurec.libs.losses.orchestration.config import LossAggregatorBatchReturn, LossAggregatorReturn, LossReturnType
from nurec.libs.losses.orchestration.loss_aggregator import LossAggregator
from nurec.models.background import SkyEnvMapBackground
from nurec.models.base import BaseModel
from nurec.models.calib import BaseCalib
from nurec.models.composite import CompositeModel
from nurec.models.gaussians.gaussians_composite import (
    GaussiansComposite,
    reset_incompatible_mcmc_visibility_state,
)
from nurec.models.gaussians.gaussians_model import distributed_all_gather_gaussian_parameters
from nurec.models.post_processing import get_ppisp_from_model
from nurec.systems.utils import (
    system_config_compatibility_check,
    update_datamodule_epoch,
    update_module_step,
)
from nurec.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    RenderingBatch,
    RenderingData,
)
from nurec.utils.cuda_rng import (
    CHECKPOINT_STATE_KEY as CUDA_RNG_CHECKPOINT_STATE_KEY,
)
from nurec.utils.cuda_rng import (
    get_cuda_generator,
    seed_cuda_generator,
)
from nurec.utils.cuda_rng import (
    read_checkpoint_state as read_cuda_rng_checkpoint_state,
)
from nurec.utils.geometry import se3_matrix_to_tquat
from nurec.utils.io.artifact_cache import ArtifactCache, publish_existing_usdz
from nurec.utils.io.checkpoint import reduce_precision_to_fp16, upcast_fp16_to_fp32
from nurec.utils.io.firing_patterns import populate_firing_patterns
from nurec.utils.io.mesh import Mesh, get_usd_references, mesh_from_point_cloud, serialize_mesh, smooth_mesh
from nurec.utils.io.metadata import get_metadata, serialize_metadata
from nurec.utils.io.rig_trajectories import rig_trajectories_time_range, serialize_rig_trajectories
from nurec.utils.io.sequence_tracks import serialize_sequence_tracks
from nurec.utils.metrics import MetricsCollector, create_metric_sample
from nurec.utils.misc import (
    compute_process_local_rng_seed,
    distributed_all_gather_nested,
    rank_zero_only,
    sync_objects,
    to_torch,
    unpack_optional,
)
from nurec.utils.optim import configure_optimizers, lr_scheduler_step_update_count
from nurec.utils.profiling import (
    ScopedTimer,
    backward_nvtx_ranges,
    loss_backward_nvtx_active,
    warn_backward_nvtx_detail,
)
from nurec.utils.trainer import BroadcastExceptions, TrainerConfig, adjust_step_for_world_size
from nurec.utils.training_rng import capture_global_rng_state, preserve_training_rng_state, restore_global_rng_state
from nurec.utils.types import (
    Checkpoint,
    ExtraSignal,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    NamedSerialized,
    PointCloud,
)
from nurec.utils.visualize import log_image as log_image_from_utils_visualize


warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

_GLOBAL_RNG_STATES_KEY = "nre.rng.global_states"


def _ground_mesh_settings_from_artifact_config(ground_mesh_config: Any) -> GroundMeshReconstructionSettings:
    return GroundMeshReconstructionSettings.from_mapping(ground_mesh_config)


def _loss_roots_by_name(
    loss_return: LossAggregatorReturn | LossAggregatorBatchReturn,
) -> dict[str, list[torch.Tensor]]:
    """Group individual loss roots by stable loss name."""
    roots_by_name: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    loss_returns = (
        loss_return.batch_loss_returns if isinstance(loss_return, LossAggregatorBatchReturn) else [loss_return]
    )
    for aggregate_return in loss_returns:
        for name, individual_return in aggregate_return.loss_returns.items():
            roots_by_name[name].append(individual_return.reduced_value)
    return dict(roots_by_name)


R = TypeVar("R")


class DDPStrategySO(DDPStrategy):
    """Custom DDP strategy for Scene Optimization.

    This strategy owns the dedicated Gaussian process group for exactly the
    lifetime of WORLD. :class:`BaseSystemSO` lends the group to its Gaussian
    model, which borrows it without taking ownership. Manual Gaussian
    collectives must use this group; WORLD is reserved for DDP and collectives
    that run at rank-symmetric, pre-backward program points.
    """

    _sharded_params_and_buffers: list[str] | None = None
    _ddp_configured = False
    _verbose = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gaussian_process_group: torch.distributed.ProcessGroup | None = None

    # NOTE: This class overrides several pytorch-lightning Strategy/DDPStrategy internals and relies
    # on a private torch DDP API. Rather than pin an exact PL version, assert that the specific entry
    # points the overrides depend on still exist, so a pytorch-lightning / torch upgrade that renames
    # or removes one fails loudly here instead of silently no-op'ing.
    import pytorch_lightning

    _REQUIRED_PL_OVERRIDES = (
        "setup",
        "setup_distributed",
        "teardown",
        "lightning_module_state_dict",
        "optimizer_state",
        "load_model_state_dict",
        "load_optimizer_state_dict",
    )
    assert all(hasattr(DDPStrategy, _m) for _m in _REQUIRED_PL_OVERRIDES), (
        f"pytorch-lightning DDPStrategy must define {_REQUIRED_PL_OVERRIDES} for DDPStrategySO to "
        "override; one is missing — revisit DDPStrategySO after this pytorch-lightning upgrade"
    )
    assert hasattr(DistributedDataParallel, "_set_params_and_buffers_to_ignore_for_model"), (
        "torch DistributedDataParallel no longer exposes "
        "_set_params_and_buffers_to_ignore_for_model — revisit DDPStrategySO"
    )

    # The capability checks above guard structure (methods exist) but not semantics. These are the
    # pytorch-lightning versions DDPStrategySO has actually been validated against; warn (don't fail)
    # on anything else so an untested upgrade gets a human re-check of the overridden internals.
    _TESTED_PL_VERSIONS = ("2.4.0", "2.6.5")
    if pytorch_lightning.__version__ not in _TESTED_PL_VERSIONS:
        logger.warning(
            "DDPStrategySO has been validated on pytorch-lightning %s but is running %s; "
            "re-check the overridden internals after this upgrade.",
            _TESTED_PL_VERSIONS,
            pytorch_lightning.__version__,
        )

    def setup_distributed(self) -> None:
        """Initialize DDP, then create the Gaussian communicator."""
        super().setup_distributed()

        # new_group is collective, so create the Gaussian communicator here
        # while every rank is still in lockstep and before DDP starts work.
        # Keep ownership on this strategy instance so a later same-process
        # distributed initialization cannot reuse a group from an older WORLD.
        self.gaussian_process_group = None
        if torch.distributed.get_world_size() > 1:
            self.gaussian_process_group = torch.distributed.new_group(backend="nccl", timeout=self._timeout)

    def setup(self, trainer: "pytorch_lightning.Trainer") -> None:
        """Setup DDP Strategy for Scene Optimization.
        Note that configure_ddp is only called for training, not for validation or testing. Therefore we need to overload the higher-level setup method.
        Reference:
        - https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/ddp.py#L280-L284
        - https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/ddp.py#L157
        """
        logger.info(f"Configuring DDP for {self.model.__class__.__name__}")
        assert isinstance(self.model, torch.nn.Module), "Module must have been initialized before calling configure_ddp"
        # Exclude all per-Gaussian parameters from the DDP, since their shapes vary during training,
        # which significantly slows down DDP. We will handle them manually by:
        #
        # Each rank will host a subset of Gaussians, which will be broadcasted to all ranks (all_gather)
        # every iteration before rendering happens.
        #
        # https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/parallel/distributed.py#L2261
        #
        # Usage: https://github.com/pytorch/pytorch/blob/07df6ba7f5597488a93b3855d52d2ead55675125/torch/nn/parallel/distributed.py#L2285
        # - Need to call this function again on the top level model, otherwise the DDP will not ignore the parameters and buffers
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(self.model, self.sharded_params_and_buffers)
        if self._sharded_params_and_buffers is not None:
            DDPStrategySO._pretty_log_keys(
                "DDP - excluding following parameters and buffers",
                self._sharded_params_and_buffers,
                print_func=logger.info,
            )
        self._ddp_configured = True
        # Call the original method to continue the setup process
        super().setup(trainer)

    def teardown(self) -> None:
        """Release the Gaussian communicator before Lightning tears down WORLD."""
        group = self.gaussian_process_group
        try:
            if group is not None:
                system = getattr(self, "_model", None)
                if isinstance(system, DistributedDataParallel):
                    system = system.module
                gaussian_model = getattr(system, "model", None)
                if isinstance(gaussian_model, GaussiansComposite):
                    gaussian_model.gaussian_process_group = None
        finally:
            try:
                if group is not None and torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group(group)
            finally:
                self.gaussian_process_group = None
                super().teardown()

    @property
    def base_system(self) -> "BaseSystem[Any]":
        """Query the base system safely"""
        system = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        assert isinstance(system, BaseSystem), "Model must be an instance of BaseSystem"
        return system

    @property
    def sharded_params_and_buffers(self) -> list[str]:
        """Retrieve the sharded model parameters and buffers"""
        if self._sharded_params_and_buffers is None:
            self._sharded_params_and_buffers = self.base_system.configure_sharded_params_and_buffers()
        return self._sharded_params_and_buffers

    def sharded_optimizer_state_indices(
        self, optimizer: torch.optim.Optimizer, state_dict: dict[str, Any]
    ) -> list[int]:
        """Retrieve the the sharded optimizer state indices of the given optimizer"""
        self.base_system.configure_sharded_params_and_buffers()  # Make sure sharded flags are available
        # At this stage, configure_ddp() should have been called, so the model is a DistributedDataParallel instance
        assert isinstance(self.model, DistributedDataParallel), "Model must be an instance of DistributedDataParallel"
        # We then sync the state of the optimizer for the parameters that are not sharded
        indices = []
        for group, group_dict in zip(optimizer.param_groups, state_dict["param_groups"]):
            for p, i in zip(group["params"], group_dict["params"]):
                if BaseModel.is_sharded(p):
                    indices.append(i)
        return indices

    def split_params_and_buffers(self, full_states: dict[str, Any]) -> dict[str, Any]:
        """Split the parameters and buffers"""
        world_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        # Make a deep copy of the full states to avoid modifying the original states
        sharded_states = copy.deepcopy(full_states)
        reset_incompatible_mcmc_visibility_state(
            sharded_states,
            warning=logger.warning if world_rank == 0 else None,
        )
        # Split the parameters and buffers
        for p in self.sharded_params_and_buffers:
            if p not in sharded_states:
                assert self.lightning_module is not None
                try:
                    live_parameter = self.lightning_module.get_parameter(p)
                except AttributeError as error:
                    raise KeyError(f"Missing sharded model state {p!r}") from error
                if not isinstance(live_parameter, torch.nn.UninitializedParameter):
                    raise KeyError(f"Missing materialized sharded model state {p!r}")
                continue
            sharded_states[p] = sharded_states[p][world_rank::world_size]
        return sharded_states

    def split_optimizer_states(self, optimizer: torch.optim.Optimizer, full_states: dict[str, Any]) -> dict[str, Any]:
        """Split the optimizer states"""
        world_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        # Make a deep copy of the full states to avoid modifying the original states
        sharded_states = copy.deepcopy(full_states)
        # Split the optimizer states
        sharded_indices = self.sharded_optimizer_state_indices(optimizer, full_states)
        for i in sharded_indices:
            s = sharded_states["state"][i]
            for k, v in s.items():
                # Scalar state entries (e.g. Adam's "step" counter) are per-optimizer
                # bookkeeping, not per-Gaussian tensors — leave them unsharded.
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    continue
                s[k] = v[world_rank::world_size]
        return sharded_states

    def lightning_module_state_dict(self) -> dict[str, Any]:
        """Returns model state.
        Reference: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/strategy.py#L473-L476
        """
        assert self.lightning_module is not None
        # Run all-rank checks before a local state_dict hook can desynchronize peers.
        self.base_system.preflight_distributed_checkpoint_state_dict()
        state_dict = copy.deepcopy(self.lightning_module.state_dict())

        # Extract the sharded parameters and buffers. Because `sharded_params_and_buffers` contains the names of different gaussian nodes,
        # we need to perform the all-gather operation for each gaussian node separately due to their different gaussian counts.
        sharded_states: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for name in self.sharded_params_and_buffers:
            if name in state_dict:
                state = state_dict[name]
                module_name = DDPStrategySO._get_module_name(name)
                parent_module = self.lightning_module.get_submodule(module_name)
                # ParameterDict entries can represent independent row owners. Their
                # local lengths can differ by rank after strided DDP sharding, so
                # grouping by local tensor length would give ranks different
                # collective sequences and deadlock. Give each entry a stable group.
                group = name if isinstance(parent_module, torch.nn.ParameterDict) else module_name
                sharded_states[group][name] = state

        # All-gather the sharded parameters and buffers
        for sharded_module_states in sharded_states.values():
            full_module_states = distributed_all_gather_gaussian_parameters(
                sharded_module_states,
                stable=True,
                group=self.gaussian_process_group,
            )
            state_dict.update(full_module_states)

        # Log the keys that are being all-gathered
        if sharded_states and DDPStrategySO._verbose:
            DDPStrategySO._pretty_log_keys(
                f"Gathering Module State for {self.lightning_module.__class__.__name__}",
                list(sharded_states),
            )

        return state_dict

    def optimizer_state(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Return the optimizer state
        Reference: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/strategy.py#L173-L189
        """
        state_dict = copy.deepcopy(
            super().optimizer_state(optimizer)
        )  # We first get the regular state dict for the optimizer

        # We also need to access the optimizer directly
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer
        assert not hasattr(optimizer, "consolidate_state_dict"), "ZeroRedundancyOptimizer is not supported"

        # Collect the sharded states for this parameter group
        sharded_states: dict[str, Any] = {}
        for i in self.sharded_optimizer_state_indices(optimizer, state_dict):
            s = state_dict["state"][i]
            assert isinstance(s, dict), "State must be a dict"
            for k, v in s.items():
                # Scalar state entries (e.g. Adam's "step" counter) are per-optimizer
                # bookkeeping, not per-Gaussian tensors. Skip the all-gather for them
                # so the gather receives a uniform first-dim length across keys.
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    continue
                sharded_states[f"{i}/{k}"] = v

        # All-gather the sharded states (We know that different gaussian nodes have different optimizers, so we don't
        # need to worry about gaussian count mismatch).
        full_states = distributed_all_gather_gaussian_parameters(
            sharded_states,
            stable=True,
            group=self.gaussian_process_group,
        )
        for key, value in full_states.items():
            si, k = key.split("/")
            state_dict["state"][int(si)][k] = value  # type: ignore[index]

        # We log the keys that are being all-gathered
        if sharded_states and DDPStrategySO._verbose:
            DDPStrategySO._pretty_log_keys(
                f"Gathering Optimizer State for {self.lightning_module.__class__.__name__}", list(sharded_states.keys())
            )

        return state_dict

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        """The function is called by the trainer to actually load the model state_dict.
        Reference: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/strategy.py#L369-L371
        """
        # Actually load the sharded states into the model (use assign=True to initialize Gaussians parameters)
        sharded_states = self.split_params_and_buffers(checkpoint["state_dict"])
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(sharded_states, strict=strict, assign=True)
        # Restore sharded flags are still valid after assignments
        self.base_system.configure_sharded_params_and_buffers()
        self._resumed_with_full_state_dict = True
        logger.info(f"Loaded model state dict for {self.lightning_module.__class__.__name__}")

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Load the optimizer state_dict.
        Reference: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/strategies/strategy.py#L373-L377
        """
        from lightning_fabric.utilities.optimizer import _optimizer_to_device  # type: ignore[import-untyped]

        optimizer_states = checkpoint["optimizer_states"]

        # The SelectiveAdam split changed the Gaussian optimizer topology (density and scale
        # became their own optimizers), so a checkpoint saved before the split has a different
        # number of optimizer states than the current model has optimizers. zip() would
        # silently truncate to the shorter list, positionally restoring state to a misaligned
        # subset and leaving the rest cold with no signal. The old combined param groups do not
        # correspond 1:1 to the new density/scale/general optimizers, so a positional partial
        # load is unsafe (it could apply the wrong state or fail on a group-count mismatch).
        # Detect the mismatch, warn loudly, and skip optimizer-state restoration entirely: the
        # model weights are already restored (load_model_state_dict), so training resumes with
        # correct parameters but cold Adam moments for all optimizers.
        if len(self.optimizers) != len(optimizer_states):
            logger.warning(
                "Optimizer count mismatch while resuming: model has %d optimizers but the "
                "checkpoint has %d optimizer states. This usually means the checkpoint predates "
                "the Gaussian optimizer split (density/scale became separate optimizers). "
                "Skipping optimizer-state restoration: model weights are restored, but ALL "
                "optimizers start with cold Adam state (no exp_avg / exp_avg_sq carry-over).",
                len(self.optimizers),
                len(optimizer_states),
            )
            return

        for opt, opt_state in zip(self.optimizers, optimizer_states):
            opt.load_state_dict(self.split_optimizer_states(opt, opt_state))
            _optimizer_to_device(opt, self.root_device)
        logger.info(f"Loaded optimizer state dict for {self.lightning_module.__class__.__name__}")

    @staticmethod
    def _get_module_name(name: str) -> str:
        """Get the module name from the full name"""
        return ".".join(name.split(".")[:-1])

    @staticmethod
    def _pretty_log_keys(
        banner: str, keys: list[str], *, indent: int = 0, print_func: Callable[[str], None] = tqdm.write
    ) -> None:
        print_func(f"{' ' * indent}{banner}:")
        for k in keys[:-1]:
            print_func(f"{' ' * (indent + 4)}|─{k}")
        print_func(f"{' ' * (indent + 4)}└─{keys[-1]}")


class BaseSystem(LightningModule, ABC, Generic[LossReturnType]):
    model: BaseModel
    config: BaseSystemConfig
    device: torch.device

    def __init__(self, config: NuRecConfig) -> None:
        super().__init__()

        self.trainer_config: TrainerConfig = config.trainer

        # handle the training loop iteration ourselves (e.g., to support multiple optimizers)
        self.automatic_optimization = False
        self._lr_scheduler_epoch_steps: tuple[int, int] | None = None

        self.save_hyperparameters(config_to_primitive(config.to_dictconfig()))

        # save what we need from the config
        self.mode = config.mode
        steps_per_epoch = adjust_step_for_world_size(config.trainer, config.dataset.n_samples_per_epoch)
        config.trainer.max_steps = config.trainer.max_epochs * steps_per_epoch

        # If PPISP controller distillation is configured, extend the training
        # budget and compute the controller activation ratio for the library.
        if config.model.post_processing is not None:
            for pp_cfg in config.model.post_processing.values():
                if not isinstance(pp_cfg, PPISPPostProcessingConfig):
                    continue
                distillation_steps = pp_cfg.controller_distillation_steps
                if distillation_steps > 0:
                    base_steps = config.trainer.max_steps
                    pp_cfg.controller_activation_ratio = base_steps / (base_steps + distillation_steps)
                    config.trainer.max_steps += distillation_steps
                    extra_epochs = -(-distillation_steps // steps_per_epoch)  # ceil division
                    config.trainer.max_epochs += extra_epochs
                    logger.info(
                        "PPISP distillation: extending training by %d steps (%d extra epochs). "
                        "Total: %d steps, %d epochs.",
                        distillation_steps,
                        extra_epochs,
                        config.trainer.max_steps,
                        config.trainer.max_epochs,
                    )
                    break

        self.update_n_epochs = config.datamodule.update_n_epochs
        self.out_dir = config.out_dir
        self.run_id = config.run_id
        self.resume = config.resume
        self.resume_weights_only = config.resume_weights_only
        self.harmonizer_config = config.harmonizer

        self.config = config.system
        self.trainer_config = config.trainer
        self.logger_enabled = config.logger.enabled
        self._backward_nvtx_detail = config.scopedtimer.backward_nvtx_detail
        warn_backward_nvtx_detail(self._backward_nvtx_detail, config.scopedtimer)

        self.datamodule = nurec.datamodules.make(config.datamodule.name, config)
        assert config.loss is not None
        self.loss = LossAggregator(config.loss, config.trainer)
        self.training_step_outputs: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
        self.validation_step_outputs: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

        self.last_checkpoint_step = -1  # prevent re-serialization of already processed steps

        # Map config mode to metrics collector expected values
        if config.mode == "train":
            metrics_mode: Literal["train", "val", "grpc_server", "unknown"] = "train"
        elif config.mode == "val":
            metrics_mode = "val"
        else:
            metrics_mode = "unknown"

        self._validation_metrics_collector = MetricsCollector(
            train_config_name=config.train_config_name,
            mode=metrics_mode,
            run_id=config.run_id,
            version=config.version,
        )

        self.outer_loop_timer = ScopedTimer("BaseSystem/outer_loop")
        self.inner_loop_timer = ScopedTimer("BaseSystem/training_step")

    def configure_sharded_params_and_buffers(self) -> list[str]:
        return [f"model.{p}" for p in self.model.configure_sharded_params_and_buffers()]

    def preflight_distributed_checkpoint_state_dict(self) -> None:
        """Run all-rank checks before DDP starts collecting checkpoint state."""

    @abstractmethod
    def forward(self, batch: DataAndRenderingBatch) -> Any:
        """Forward function for the system."""
        ...

    def setup(self, stage: str) -> None:
        # The `setup` function is called by the `pl.Trainer` fit/validate/test functions. At this point,
        # the DDP environment is already initialized, unlike in the `__init__` function. Therefore,
        # we move the initialization of the model to the `setup` function.
        super().setup(stage)
        self.model.to(self.device)

        # Per-rank seed the cuda_rng singleton so DDP ranks decorrelate.
        # `pl.seed_everything(config.seed, workers=True)` (called in run/main.py)
        # only seeds torch's default generators; the user-allocated cuda_rng
        # singleton needs explicit per-rank seeding here, after DDP setup has
        # populated PL_GLOBAL_SEED + rank info that
        # `compute_process_local_rng_seed()` reads. Without this, every rank
        # draws the same `torch.randn(generator=cuda_rng_gen)` sequence and
        # MCMC perturbation noise becomes rank-correlated.
        if self.device.type == "cuda":
            seed_cuda_generator(self.device, compute_process_local_rng_seed())

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfers a batch to the target device with non-blocking copy to avoid
        cudaStreamSynchronize being added under the hood.

        The `dataloader_idx` argument is kept to match the
        Lightning hook signature.
        Ref: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.DataHooks.html#lightning.pytorch.core.hooks.DataHooks.transfer_batch_to_device
        """
        return batch.to(device, non_blocking=True)

    def training_step(self, batch: DataAndRenderingBatch, batch_local_idx: int):
        """Evaluate one external training-loop iteration.

        Overrides must preserve both coordinated exception gates and the
        optimizer-preparation lifecycle: stage during ``training_losses``,
        launch exactly once after the forward gate and before backward, then
        resolve exactly once after backward and before optimizer steps. These
        hooks may issue WORLD collectives and therefore must execute in the
        same order on every rank.
        """
        self.outer_loop_timer.stop(force=True)
        # Set step BEFORE starting inner_loop_timer so scopedtimer_window range
        # properly encompasses training_step range (correct NVTX nesting)
        ScopedTimer.set_step(int(self.global_step))
        self.inner_loop_timer.start()

        # Due to a barrier inside manual_backward below,
        # we need to ensure that either all process or none enters it.
        # If an exception is raised, it'll be broadcasted to all processes
        # and training will abort instead of hang.
        with BroadcastExceptions(self.trainer):
            # 0. Construct list of optimizers (also for single optimizer case)
            pl_optimizers = self.optimizers(
                # PL optimizers will automatically handle precision and profiling
                use_pl_optimizer=True
            )
            pl_optimizers = pl_optimizers if isinstance(pl_optimizers, list) else [pl_optimizers]

            # 1. Invalidate gradients in all optimizers
            for pl_optimizer in pl_optimizers:
                # note: this assumes to set None (not zero-values) to '.grad' properties of parameter tensors, but we can't
                # request this explicitly here as the "default" 'set_to_none' keyword property is not supported by all
                # optimizers, e.g., 'apex.FusedAdam.zero_grad()' is doesn't respect a 'set_to_none' keyword parameter
                pl_optimizer.optimizer.zero_grad()

            # 2. Evaluate *system-level* training losses for given batch
            loss_return = self.training_losses(batch, batch_local_idx)
            self.on_training_losses_return(loss_return)

        # Broadcast exceptions raised during optimization.
        # Note: this is adding synchronization overhead, which might be significant if the
        #       workload is uneven across processes. Ideally one would only broadcast exceptions
        #       when the processes are idle, e.g., right after backward pass
        #       (where the gradients are synchronized).
        #       I couldn't implement this idea. We'd have to call self.manual_backward() with a possibly
        #       non-existing loss_return. I've tried to create dummy total_value Tensor:
        #           1. empty tensor: resulted in error (no gradient found in element 0)
        #           2. 0.0 value, with gradient: assertion error
        with BroadcastExceptions(self.trainer):
            # Launch system-specific asynchronous optimizer preparation only
            # after the preceding coordinated forward-error gate. This keeps
            # WORLD collective ordering symmetric if a rank fails in forward.
            self._launch_optimizers_async_preparation()

            # 3. Backprop gradients of *system-level* training losses to parameters
            # Note 1: there's a inter-process barrier here to gather the gradients of
            #         all subprocesses.  All processes or none must call this function
            #         or else training will hang indefinitely.
            # Note 2: we're assuming that no exceptions occur in the code
            #         leading to the barrier inside manual_backward. If they do, process will hang.
            #         I don't think this is likely to happen, though (famous last words).
            # Partial test systems skip __init__, so the flag may be absent.
            backward_attribution = (
                backward_nvtx_ranges(_loss_roots_by_name(loss_return))
                if loss_backward_nvtx_active(getattr(self, "_backward_nvtx_detail", "off"))
                else nullcontext()
            )
            with ScopedTimer("BaseSystem/training_step/manual_backward"):
                with backward_attribution:
                    self.manual_backward(loss_return.total_value)

            # Let systems finish optimizer-specific asynchronous preparation
            # after gradients are synchronized but immediately before any
            # optimizer consumes them.
            self._prepare_optimizers_after_backward()

            # 4. Step optimizers to update parameters
            # Manually maintain by ourselves the trainer's optimizer step progress, to temporarily fix https://github.com/Lightning-AI/pytorch-lightning/issues/17958
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()
            self.trainer.profiler.start("optimizer_step")  # type: ignore[attr-defined]

            with ScopedTimer("BaseSystem/training_step/optimizer_step"):
                for pl_optimizer in pl_optimizers:
                    # skip step if none of the parameters has associated gradients after backpropagation (e.g., because all parameters are frozen)
                    # circumventing gradient scaler shortcomings not able to handle this setting
                    if all([p.grad is None for group in pl_optimizer.optimizer.param_groups for p in group["params"]]):
                        continue
                    if pl_optimizer._strategy is not None:
                        pl_optimizer._strategy.optimizer_step(pl_optimizer._optimizer, do_nothing_closure)

            self.trainer.profiler.stop("optimizer_step")  # type: ignore[attr-defined]
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

            # 5. Handle lr-schedulers
            for lr_scheduler_config in self.trainer.strategy.lr_scheduler_configs:
                scheduler_update_count = 1
                # check if we need to step this lr-scheduler / exit early if not
                match interval := lr_scheduler_config.interval:
                    case "epoch":
                        if not (
                            self.trainer.is_last_batch
                            and (self.trainer.current_epoch + 1) % lr_scheduler_config.frequency == 0
                        ):
                            continue  # step not required
                    case "step":
                        configured_steps, effective_steps = self._lr_scheduler_epoch_steps or (1, 1)
                        scheduler_update_count = lr_scheduler_step_update_count(
                            batch_local_idx=batch_local_idx,
                            frequency=lr_scheduler_config.frequency,
                            configured_steps_per_epoch=configured_steps,
                            effective_steps_per_epoch=effective_steps,
                        )
                        if scheduler_update_count == 0:
                            continue  # step not required
                    case _:
                        raise RuntimeError(f"Invalid scheduler interval '{interval}'")

                # step lr-scheduler depending on type
                if not isinstance(
                    scheduler := lr_scheduler_config.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    # step a regular scheduler
                    for _ in range(scheduler_update_count):
                        scheduler.step()
                else:
                    if scheduler_update_count != 1:
                        raise RuntimeError(
                            "Accelerated component-only LR progression does not support a step-interval "
                            "ReduceLROnPlateau scheduler because replaying one metric would change its semantics."
                        )
                    # incorporate the monitored metric for ReduceLROnPlateau
                    monitor = lr_scheduler_config.monitor
                    assert monitor is not None, "Missing metric to monitor for ReduceLROnPlateau scheduler"

                    if (monitor_metric := self.trainer.callback_metrics.get(monitor)) is not None:
                        # step the scheduler with available metric
                        scheduler.step(monitor_metric)
                    else:
                        # monitored metric is not available - warn / error out depending on strict flag
                        if lr_scheduler_config.strict:
                            raise RuntimeError(f"Missing monitored metric {monitor} for ReduceLROnPlateau scheduler")
                        else:
                            logger.warning(
                                f"Missing monitored metric {monitor} for ReduceLROnPlateau scheduler - skipping scheduler update"
                            )

        self.inner_loop_timer.stop()
        self.outer_loop_timer.start()

    @abstractmethod
    def training_losses(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> LossReturnType:
        """Evaluate system-specific losses for a single training batch.

        Implementations may execute unconditional rank-symmetric forward
        collectives and may stage optimizer preparation, but must not launch
        conditionally staged optimizer WORLD work here. :meth:`training_step`
        launches that work only after its coordinated forward-error gate and
        consumes it after backward. Subclasses overriding ``training_step``
        must preserve that exact rank-symmetric ordering.
        """
        ...

    @property
    def is_trainer_attached(self) -> bool:
        """Check if pytorch_lightning.Trainer is hooked up (we're inside .fit loop)"""
        # There seems to be no official way to check if trainer is attached
        # https://github.com/Lightning-AI/pytorch-lightning/issues/17517#issuecomment-1532325528
        return self._trainer is not None

    def on_training_losses_return(self, loss_return: LossReturnType) -> None:
        """Callback for when training losses are returned."""
        pass

    def _launch_optimizers_async_preparation(self) -> None:
        """Launch staged optimizer work once, after coordinated forward errors.

        Implementations may issue WORLD collectives, so every rank must call
        this hook in the same order. It must remain after the first
        :class:`BroadcastExceptions` gate and before ``manual_backward``.
        """
        pass

    def _prepare_optimizers_after_backward(self) -> None:
        """Consume staged optimizer work once after backward, before stepping.

        Implementations must resolve and clear work launched by
        :meth:`_launch_optimizers_async_preparation`; this hook must remain
        before every optimizer can consume its final synchronized gradients.
        """
        pass

    def on_before_checkpoint_capture(self) -> None:
        """Called right before Lightning snapshots the model into a checkpoint dict.

        Anything that has to reach the saved file has to happen here. on_save_checkpoint is too late, Lightning
        reads the state dict and the optimizer state before it runs. A model mutation there lands in memory but
        not on disk.
        """

    def on_validation_epoch_end(self) -> None:
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """

    def on_test_epoch_end(self):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """

    def configure_optimizers(self) -> list[PL_OptimizerLRSchedulerConfig]:
        """
        Returns all system-associated as well as model-associated optimizers
        """
        system_optims = configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self.model)
        model_local_optims = self.model.configure_optimizers()
        return [cast(PL_OptimizerLRSchedulerConfig, x) for x in system_optims + model_local_optims]

    def on_validation_start(self) -> None:
        super().on_validation_start()

        val_path = os.path.join(self.out_dir, self.run_id, "val")
        if self.config.test.separate_val_dir_per_step:
            val_path = os.path.join(val_path, f"{self.global_step:06d}")
        # NOTE: Make sure all ranks have the same val_dir (should we?)
        self.val_dir = sync_objects(lambda: val_path)
        os.makedirs(self.val_dir, exist_ok=True)

        if self.harmonizer_config.inference.enabled:
            self.harmonizer_model = HarmonizerModelFactory.get(
                self.harmonizer_config.model_url,
                self.harmonizer_config.cache_dir,
                self.harmonizer_config.model_filename,
                self.harmonizer_config.model_resolution,
                cosmos_base_model_url=self.harmonizer_config.cosmos_base_model_url,
                cosmos_tokenizer_url=self.harmonizer_config.cosmos_tokenizer_url,
                cosmos_cache_dir=self.harmonizer_config.cosmos_cache_dir,
            )

    def on_validation_end(self):
        super().on_validation_end()
        if self.harmonizer_config.inference.enabled:
            HarmonizerModelFactory.clear_cache()

    def on_test_end(self):
        super().on_test_end()

    def on_test_start(self) -> None:
        # NOTE: Make sure all ranks have the same test_dir (should we?)
        self.test_dir = sync_objects(lambda: os.path.join(self.out_dir, self.run_id, "test"))

        if self.harmonizer_config.inference.enabled:
            logger.warning("Harmonizer not supported in test mode. Continuing without Harmonizer application.")

    def on_train_batch_start(self, batch: DataAndRenderingBatch, batch_local_idx: int):
        super().on_train_batch_start(batch, batch_local_idx)

        with ScopedTimer("BaseSystem/update_step_train_batch_start"):
            additional_parameters: dict[str, torch.Tensor] = update_module_step(
                self.model, self.current_epoch, self.global_step, self
            )
            additional_parameters |= update_module_step(self.loss, self.current_epoch, self.global_step, self)
            # We iterate over the additional parameters, if the same parameter group already exist we
            # append the parameter to the group, else we initialize a new parameter group.
            for key, value in additional_parameters.items():
                for param_group in self.trainer.optimizers[0].param_groups:
                    if param_group["name"] == key:
                        param_group["params"].append(value)
                        break
                else:
                    self.trainer.optimizers[0].add_param_group({"params": value, "name": key})

    def on_train_batch_end(
        self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_local_idx: int
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_local_idx)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.outer_loop_timer.start()

    def on_train_epoch_end(self) -> None:
        self.outer_loop_timer.stop()

        # Open the post-training window here rather than in on_train_end so the final
        # checkpoint save lands inside it: Lightning runs Checkpoint callbacks in the
        # monitoring pass *after* this hook, so artifact population and the .ckpt/.usdz
        # writes would otherwise be attributed to the last training step.
        #
        # This must precede timer.start(). ScopedTimer.start() and stop() each
        # re-evaluate capture gating independently, so switching windows inside an open
        # range pops an NVTX zone that was never pushed. Same constraint the set_step()
        # call in training_step() observes.
        if self._is_final_train_epoch():
            ScopedTimer.begin_post_train_window()

        timer = ScopedTimer("BaseSystem/on_train_epoch_end")
        timer.start()

        super().on_train_epoch_end()
        if self.update_n_epochs != 0 and (self.current_epoch + 1) % self.update_n_epochs == 0:
            update_datamodule_epoch(self.datamodule, self.current_epoch, self)

        timer.stop()

    def _is_final_train_epoch(self) -> bool:
        """Whether the fit loop will exit after the epoch currently ending.

        Mirrors the limit checks in Lightning's `_FitLoop.done` instead of reading that
        property, which logs a "`Trainer.fit` stopped" line and mutates epoch progress
        as a side effect. `current_epoch` still reports the finishing epoch here because
        Lightning increments `completed` after this hook. Review on Lightning upgrades;
        a false negative only costs a narrower capture window, since on_train_end opens
        the window as a fallback.
        """
        if not self.is_trainer_attached:
            return False

        trainer = self.trainer
        if trainer.should_stop:
            return True

        max_steps = trainer.max_steps
        if max_steps != -1 and trainer.global_step >= max_steps:
            return True

        max_epochs = trainer.max_epochs
        return max_epochs is not None and max_epochs != -1 and trainer.current_epoch + 1 >= max_epochs

    def on_validation_batch_start(
        self, batch: DataAndRenderingBatch, batch_local_idx: int, dataloader_idx: int = 0
    ) -> None:
        update_module_step(self.model, self.current_epoch, self.global_step, self)
        torch.cuda.empty_cache()

    def on_test_batch_start(self, batch: DataAndRenderingBatch, batch_local_idx: int, dataloader_idx: int = 0) -> None:
        update_module_step(self.model, self.current_epoch, self.global_step, self)

    def on_predict_batch_start(
        self, batch: DataAndRenderingBatch, batch_local_idx: int, dataloader_idx: int = 0
    ) -> None:
        update_module_step(self.model, self.current_epoch, self.global_step, self)

    def decode_value(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError("Scalar specification only supports list, got", type(value))
            if len(value) == 3:
                value = [0] + value
            assert len(value) == 4
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = self.global_step
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
                )
            elif isinstance(end_step, float):
                current_step = self.current_epoch
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
                )
        return value

    def log_image(self, image: np.ndarray | torch.Tensor, caption: str, step: int) -> None:
        """
        Log an image to the logger.
        """
        log_image_from_utils_visualize(self.logger, image, caption, step)

    def on_load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        The issue here is that the buffers can change in size so we cannot initialize them in advance and load them directly from the checkpoint
        """
        if self.is_trainer_attached:
            # Manually load all the loops as otherwise global_step and current epoch are not set correctly in the validation mode
            # see: https://github.com/Lightning-AI/lightning/issues/17127
            self.trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"])
            self.trainer.validate_loop.load_state_dict(checkpoint["loops"]["validate_loop"])
            self.trainer.test_loop.load_state_dict(checkpoint["loops"]["test_loop"])
            self.trainer.predict_loop.load_state_dict(checkpoint["loops"]["predict_loop"])

        # Upcast fp16 tensors back to fp32 so CUDA/Slang kernels receive the expected dtype.
        # This must happen before any loading path (DDP or non-DDP).
        checkpoint["state_dict"] = upcast_fp16_to_fp32(checkpoint["state_dict"])

        # Load the remainder of the state_dict. The reason here is that the buffers can change in size so we cannot initialize them in
        # advance and load them directly from the checkpoint.
        # NOTE: We are loading the full state_dict here. The DDPStrategySO will handle the sharding of the parameters.
        # Load the remainder of the state_dict
        if self.is_trainer_attached and isinstance(self.trainer.strategy, DDPStrategySO):
            pass
        else:
            self.load_state_dict(checkpoint["state_dict"], assign=True)

        if "trainer.world_size" in checkpoint:
            ckpt_world_size = checkpoint["trainer.world_size"]
        else:
            ckpt_world_size = 1

        # Check if the world size matches. This is a sanity check.
        if self.trainer_config.world_size != ckpt_world_size:
            logger.warning(
                f"Loading checkpoints produced by a different world size. "
                f"This might cause some minor discrepancies with the training. "
                f"World size mismatch: {self.trainer_config.world_size} != {ckpt_world_size}."
            )

        # Restore other useful information.
        self.last_checkpoint_step = checkpoint["system.checkpoint.last_checkpoint_step"]
        self.training_step_outputs = defaultdict(list, checkpoint["system.checkpoint.training_step_outputs"])

        global_states = checkpoint.get(_GLOBAL_RNG_STATES_KEY)
        # Only ranks in the saved world have streams to restore.
        if not self.resume_weights_only and global_states is not None and self.global_rank < ckpt_world_size:
            if self.global_rank >= len(global_states):
                raise ValueError(f"Checkpoint has no global RNG state for rank {self.global_rank}")
            restore_global_rng_state(global_states[self.global_rank], self.device)

        # Restore the cuda_rng singleton's `(seed, offset)` so the post-resume
        # training trajectory continues the captured RNG sequence rather than
        # rewinding to `(seed=0, offset=0)`. Lightning's `_collect_rng_states`
        # auto-saves torch's default generators only -- a user-allocated
        # `torch.Generator(device='cuda')` is invisible to it -- so the matching
        # save in `on_save_checkpoint` and this restore form an explicit pair.
        # `read_checkpoint_state` returns None for checkpoints saved before this
        # hook landed and accepts the pre-rename `nre.` key spelling.
        #
        # Two shapes are accepted for backward-compat:
        #   - list[Tensor]  -> DDP-gathered states indexed by global rank
        #                      (current format; preserves per-rank decorrelation
        #                      established at trainer init by
        #                      `compute_process_local_rng_seed()`)
        #   - Tensor        -> single state from earlier code paths (pre-DDP
        #                      gather); applied to every rank, which re-correlates
        #                      ranks but is the documented behavior for that
        #                      legacy shape.
        cuda_rng_state = read_cuda_rng_checkpoint_state(checkpoint)
        if cuda_rng_state is not None and self.device.type == "cuda":
            if isinstance(cuda_rng_state, list):
                # Each rank must continue its own independent custom-generator stream.
                rank = self.global_rank
                rank_state = cuda_rng_state[rank] if rank < len(cuda_rng_state) else None
                if rank_state is not None:
                    get_cuda_generator(self.device).set_state(rank_state)
            else:
                # A legacy bare tensor represents the same custom stream on every rank.
                get_cuda_generator(self.device).set_state(cuda_rng_state)

    def on_save_checkpoint(self, checkpoint: Checkpoint) -> None:
        # The outer list preserves one whole state record per rank during the gather.
        checkpoint[_GLOBAL_RNG_STATES_KEY] = distributed_all_gather_nested(
            [capture_global_rng_state(self.device)], only_on_rank_zero=False
        )

        # First call for this step - do the aggregation across all ranks
        with ScopedTimer("BaseSystem/distributed_all_gather_nested"):
            aggregated_training_step_outputs = distributed_all_gather_nested(
                dict(self.training_step_outputs),
                only_on_rank_zero=False,  # We need it on all ranks for checkpoint
            )

        checkpoint["system.checkpoint.training_step_outputs"] = aggregated_training_step_outputs
        checkpoint["trainer.world_size"] = self.trainer_config.world_size

        self.last_checkpoint_step = self.trainer.global_step
        # TODO Check if needed once is_resumable feature is checked
        checkpoint["system.checkpoint.last_checkpoint_step"] = self.last_checkpoint_step

        if self.device.type == "cuda":
            # The custom generator has state independent of the default CUDA generator.
            local_state = get_cuda_generator(self.device).get_state()
            checkpoint[CUDA_RNG_CHECKPOINT_STATE_KEY] = distributed_all_gather_nested(
                [local_state],
                only_on_rank_zero=False,
            )

    def log(self, *args, **kwargs) -> None:
        """
        Log metrics if logging is enabled in the configuration.

        This method is a wrapper around the PyTorch Lightning's log method
        that respects the logger configuration setting.

        Args:
            *args: Arguments to pass to PyTorch Lightning's log method
            **kwargs: Keyword arguments to pass to PyTorch Lightning's log method
        """
        if self.logger_enabled:
            super().log(*args, **kwargs)
        else:
            pass

    def on_train_start(self) -> None:
        """Start of the training loop"""
        super().on_train_start()

    def on_train_end(self) -> None:
        """Clean up resources when training ends."""
        super().on_train_end()

        # Fallback for runs that end without a final on_train_epoch_end (the usual
        # opener). Idempotent, so it is a no-op when that window is already open.
        ScopedTimer.begin_post_train_window()

        # Clean up train dataloader if exists
        try:
            train_loader = self.datamodule.train_dataloader()

            # PrefetchDataLoader runs a background thread in the main process that prefetches data
            # from worker processes and transfers it to the GPU. This thread needs to be explicitly
            # terminated to prevent resource leaks and ensure clean process shutdown.
            if isinstance(train_loader, PrefetchDataLoader):
                train_loader.shutdown()
        except (AttributeError, NotImplementedError, TypeError):
            # Safely handle cases where train_dataloader is not available
            pass


class BaseSystemSO(BaseSystem[LossAggregatorReturn]):
    model: GaussiansComposite  # to be initialized by subclasses
    datamodule: SODataModule
    config: GaussiansSystemConfig

    def __init__(self, config: NuRecConfig) -> None:
        system_config_compatibility_check(config.to_dictconfig())

        super().__init__(config)
        assert isinstance(self.datamodule, SODataModule)
        self.train_num_rays = config.model.train_num_rays
        # For each artifact type that does not get trained, cache its data after the first export
        self.artifact_cache = ArtifactCache(
            artifact_config=config.checkpoint.artifact,
        )
        self._last_step_artifact_path: Path | None = None
        self._ppisp_report_executor: ThreadPoolExecutor | None = None
        self._ppisp_report_future: Future[None] | None = None

        self._validation_metrics_collector = MetricsCollector(
            train_config_name=config.train_config_name,
            mode="val",
            run_id=self.run_id,
        )

        # Value of every_n_train_steps is adjusted in main.py for DDP to ensure that roughly the same number of checkpoints are saved
        self.checkpoint_every_n_train_steps = config.checkpoint.every_n_train_steps

        self.dataset_path = config.dataset.path if config.dataset.name == "ncore" else None

        # TODO: Clean this up:
        # These are needed by the harmonizer sampler, sanity_dataset, serializer, and metadata
        self.cached_config = config

        # These are needed by the harmonizer sampler
        self.train_batch_size = config.datamodule.train_batch_size
        self.dataset_name = config.dataset.name

    def setup(self, stage: str) -> None:
        """Lend the strategy-owned Gaussian communicator to the model."""
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            strategy = self.trainer.strategy
            if not isinstance(strategy, DDPStrategySO) or strategy.gaussian_process_group is None:
                raise RuntimeError("Distributed Gaussian setup requires the dedicated process group")
            self.model.gaussian_process_group = strategy.gaussian_process_group
        super().setup(stage)

    @abstractmethod
    def forward(self, batch: DataAndRenderingBatch) -> Any: ...

    def validation_step(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> None:
        pass

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self._validation_metrics_collector.reset()

    def on_validation_end(self):
        super().on_validation_end()
        self._validation_metrics_collector.write_metrics_to_yaml(
            self.val_dir,
            write_parquet=self.config.test.save_metrics_parquet,
        )

    def test_step(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> None:
        pass

    @torch.amp.autocast("cuda")
    def precompute_extra_signals(
        self,
        mode: Literal["val", "test"],
        forward_closure: Callable,
        map_location: torch.device,
        sample_every_n_frames: int,
        n_rays_per_frame: int,
    ) -> list[ExtraSignal]:
        """
        Precompute a subset of the extra signals to be used in, e.g., the visualization.
        This is usually called in on_validation_start/on_test_start.
        """

        class EveryNSampler(torch.utils.data.Sampler):
            def __init__(self, data_source, N: int):
                """
                Args:
                    data_source (Dataset): The dataset to sample from.
                    N (int): The interval for sampling every N-th item.
                """
                self.data_source = data_source
                self.N = N

            def __iter__(self):
                # Create an iterator that yields every N-th index
                return iter(range(0, len(self.data_source), self.N))

            def __len__(self):
                # The number of samples that will be drawn
                return len(self.data_source) // self.N

        dataloader_sampler = EveryNSampler(self.datamodule.val_dataset, sample_every_n_frames)
        if mode == "val":
            dataloader = self.datamodule.val_dataloader(dataloader_sampler=dataloader_sampler)
        elif mode == "test":
            dataloader = self.datamodule.test_dataloader(dataloader_sampler=dataloader_sampler)
        else:
            raise ValueError(f"Invalid mode {mode}")

        extra_signals: list[ExtraSignal] = []
        for batch in tqdm(dataloader, desc="Precomputing extra signals"):
            if batch.rays_cam is None:
                continue

            batch_skip = batch.rays_cam.size(0) // n_rays_per_frame
            batch = batch.to(self.device)[::batch_skip]

            results = forward_closure(batch)
            rendered: GaussiansRenderReturn = unpack_optional(results.rendered_cam)
            assert isinstance(rendered, GaussiansRenderReturn)
            extra_signals.append(unpack_optional(rendered.extra_ray_signals).to(map_location))

        return extra_signals

    def on_train_batch_start(self, batch: DataAndRenderingBatch, batch_local_idx: int):
        super().on_train_batch_start(batch, batch_local_idx)

    def _get_cached_ground_mesh_artifact(self, ground_mesh_config: Any) -> Mesh | None:
        if not isinstance(self.model, GaussiansComposite):
            return None

        requested_settings = _ground_mesh_settings_from_artifact_config(ground_mesh_config)
        candidates: list[tuple[str, Mesh]] = []
        for layer_id, gaussian_model in self.model.gaussians_nodes.items():
            ground_mesh_artifact = getattr(gaussian_model, "ground_mesh_artifact", None)
            ground_mesh_artifact_settings = getattr(gaussian_model, "ground_mesh_artifact_settings", None)
            if not isinstance(ground_mesh_artifact, Mesh):
                continue
            if ground_mesh_artifact_settings != requested_settings:
                continue
            if ground_mesh_config.colored and ground_mesh_artifact.colors is None:
                continue
            candidates.append((layer_id, ground_mesh_artifact))

        if not candidates:
            return None

        def _rank(candidate: tuple[str, Mesh]) -> tuple[int, int]:
            _, mesh = candidate
            # Prefer the mesh with the most faces (best ground coverage), then the one
            # with colors available.
            return (len(mesh.faces), int(mesh.colors is not None))

        candidates.sort(key=_rank, reverse=True)
        best_layer_id, best_mesh = candidates[0]

        if len(candidates) > 1:
            shapes = [(lid, len(m.faces), len(m.vertices)) for lid, m in candidates]
            distinct = {(faces, verts) for _, faces, verts in shapes}
            if len(distinct) > 1:
                logger.warning(
                    "Multiple cached ground mesh artifacts match the requested settings but disagree "
                    "in shape (layer, faces, vertices): %s. Picking %r as the best candidate.",
                    shapes,
                    best_layer_id,
                )
            else:
                logger.info(
                    "Multiple cached ground mesh artifacts match the requested settings; "
                    "all share shape %s. Reusing %r.",
                    next(iter(distinct)),
                    best_layer_id,
                )

        colors = best_mesh.colors if ground_mesh_config.colored else None
        logger.info(
            "Reusing ground mesh reconstructed during Gaussian initialization for artifact export "
            "(layer=%r): %d vertices, %d faces, colors=%s",
            best_layer_id,
            len(best_mesh.vertices),
            len(best_mesh.faces),
            "yes" if colors is not None else "no",
        )
        return Mesh(vertices=best_mesh.vertices, faces=best_mesh.faces, colors=colors)

    def populate_artifact_cache(self, checkpoint: Checkpoint) -> None:
        """Populate artifact cache fields based on available and selected entries"""

        # Torch checkpoint
        if self.artifact_cache.artifact_config.checkpoint.enabled:
            self.serialize_artifact_checkpoint(checkpoint)

        # Parsed config
        if self.artifact_cache.artifact_config.parsed_config.enabled and self.artifact_cache.parsed_config is None:
            self.artifact_cache.parsed_config = NamedSerialized.from_config(self.cached_config.to_dictconfig())

        datasource = self.datamodule.get_datasource()
        track_split_metadata = (
            datasource.get_track_split_metadata() if isinstance(datasource, NCOREDataSource) else None
        )
        clean_track_split_metadata = (
            track_split_metadata.map_track_ids(DataSourceSummary._clean_track_id_str)
            if track_split_metadata is not None
            else None
        )

        # If available, compute global timestamp offset as the global start timestamp of _all_ rig trajectories
        # of the datasource (required for USD timestamp offsetting to maintain floating point precision).
        # Use zero timestamp offset if datasource is not providing rig-trajectories
        usd_timestamp_offset_us = (
            rig_trajectories_time_range(datasource.get_rig_trajectories()).start
            if isinstance(datasource, RigTrajectoriesProvider)
            else 0
        )

        # Datasource summary
        if self.artifact_cache.datasource_summary is None:
            self.artifact_cache.datasource_summary = NamedSerialized(
                filename="datasource_summary.json",
                serialized=DataSourceSummary.from_datasource(datasource).to_json(),
            )

        if self.artifact_cache.track_split_metadata is None and clean_track_split_metadata is not None:
            self.artifact_cache.track_split_metadata = NamedSerialized(
                filename=TRACK_SPLIT_METADATA_FILENAME,
                serialized=json.dumps(clean_track_split_metadata.to_dict(), indent=2),
            )

        # Camera semantics class names (for semantics in serve / clients). Requires camera aux semantics.
        # TODO: also export lidar class names if lidar semantics rendering is added.
        if self.artifact_cache.semantic_class_names is None and isinstance(datasource, NCOREDataSource):
            semantic_map = datasource.get_semantic_classes_map(camera_semantics=True, lidar_semantics=False)
            if semantic_map:
                # Invert {name: id} to {id: name}; JSON object keys must be strings, so the
                # artifact reader converts them back to int on load.
                self.artifact_cache.semantic_class_names = NamedSerialized(
                    filename=SEMANTIC_CLASS_NAMES_FILENAME,
                    serialized=json.dumps({str(idx): name for name, idx in semantic_map.items()}, indent=2),
                )
                semantic_colormap = datasource.get_semantic_colormap(camera_semantics=True, lidar_semantics=False)
                if semantic_colormap is not None:
                    self.artifact_cache.semantic_class_colors = NamedSerialized(
                        filename=SEMANTIC_CLASS_COLORS_FILENAME,
                        serialized=json.dumps(semantic_colormap.tolist(), indent=2),
                    )

        # Certain fields are only supported if we use the NCOREDataSource
        if isinstance(datasource, NCOREDataSource) and self.dataset_path:
            if os.path.isfile(self.dataset_path):
                # Dataset info
                if self.artifact_cache.data_info is None:
                    with open(self.dataset_path, "r") as f:
                        self.artifact_cache.data_info = NamedSerialized(
                            filename="data_info.json", serialized=json.dumps(json.load(f), indent=4)
                        )

                # Metadata:
                # Do not cache the metadata, since we export the PSNR, which changes during training.
                metadata = get_metadata(self.cached_config, Path(self.dataset_path), self.training_step_outputs)
                self.artifact_cache.metadata = serialize_metadata(metadata)

            # Mesh
            # Warn if mesh_path is provided but mesh export is disabled
            generic_mesh_config = self.artifact_cache.artifact_config.mesh.generic
            if generic_mesh_config.mesh_path is not None and not generic_mesh_config.enabled:
                logger.warning("Mesh path provided but mesh.generic.enabled is False")

            # Do not recompute the mesh if it is already cached, because meshing is expensive.
            # Side effect: the mesh will be computed with unoptimized poses even if the calibration module is enabled.
            if self.artifact_cache.meshes is None:
                if generic_mesh_config.enabled:
                    mesh_path = generic_mesh_config.mesh_path
                    if mesh_path is not None:
                        try:
                            verts, faces = pcu.load_mesh_vf(mesh_path)
                            mesh = Mesh(vertices=verts, faces=faces)
                        except Exception as e:
                            logger.warning(f"Failed to load mesh from {mesh_path}: {e}")
                            # Fall back to point cloud generation
                            mesh_path = None

                    if mesh_path is None:
                        point_cloud = PointCloud.collate_fn(
                            [
                                pc
                                for pc in datasource.get_point_clouds(
                                    device=torch.device("cpu"),
                                    lidar_ids=generic_mesh_config.lidar_ids,
                                    camera_ids=generic_mesh_config.camera_ids,
                                    valid_points_only=False,  # BUG:5195915 WAR
                                    non_dynamic_points_only=True,
                                    color_type=None,
                                    step_frame=generic_mesh_config.step_frame,
                                )
                            ],
                            device=torch.device("cpu"),
                        )
                        max_num_points = generic_mesh_config.max_num_points
                        if max_num_points is not None and max_num_points > 0 and point_cloud.n_points > max_num_points:
                            random_idxs = np.random.choice(point_cloud.n_points, max_num_points, replace=False)
                            point_cloud = point_cloud[torch.from_numpy(random_idxs)]

                        nurec_to_world = datasource.world_to_nurec.inverse()
                        mesh = mesh_from_point_cloud(
                            point_cloud,
                            n_neighbors=generic_mesh_config.n_neighbors,
                            trim_distance=generic_mesh_config.trim_distance,
                            apply_road_segmentation=generic_mesh_config.apply_road_segmentation,
                            source_to_target=nurec_to_world,
                        )

                    serialized_mesh = serialize_mesh(
                        mesh,
                        export_disjoint_meshes=generic_mesh_config.export_disjoint_meshes,
                        filename="mesh",
                        formats=generic_mesh_config.formats,
                    )

                    if self.artifact_cache.meshes is None:
                        self.artifact_cache.meshes = serialized_mesh
                    else:
                        self.artifact_cache.meshes.extend(serialized_mesh)

                    # Proxy meshes are extracted before smoothed mesh is added -- smoothed mesh is never a proxy for the NeRF.
                    self.artifact_cache.proxy_meshes = get_usd_references(serialized_mesh)

                    if generic_mesh_config.smooth:
                        rig_trajectories = datasource.get_rig_trajectories()
                        smoothed_mesh = smooth_mesh(mesh, rig_trajectories)
                        self.artifact_cache.meshes.extend(
                            serialize_mesh(
                                smoothed_mesh,
                                export_disjoint_meshes=generic_mesh_config.export_disjoint_meshes,
                                filename="mesh_smoothed",
                                formats=generic_mesh_config.formats,
                            )
                        )

                ground_mesh_config = self.artifact_cache.artifact_config.mesh.ground

                if ground_mesh_config.enabled:
                    ground_mesh = self._get_cached_ground_mesh_artifact(ground_mesh_config)
                    if ground_mesh is None:
                        if ground_mesh_config.lidar_ids is not None and len(ground_mesh_config.lidar_ids) != 1:
                            raise ValueError("Ground mesh reconstruction supports exactly one LiDAR sensor")
                        point_cloud_generator = datasource.get_point_clouds(
                            device=torch.device("cpu"),
                            lidar_ids=ground_mesh_config.lidar_ids,
                            camera_ids=ground_mesh_config.camera_ids,
                            valid_points_only=False,  # BUG:5195915 WAR
                            non_dynamic_points_only=True,
                            color_type="camera-rgb" if ground_mesh_config.colored else None,
                            step_frame=ground_mesh_config.step_frame,
                        )

                        nurec_to_world = datasource.world_to_nurec.inverse()

                        # Reconstruct the ground mesh.
                        ground_vertices, ground_triangles, _, _, _, _, _, ground_vertex_colors, _ = (
                            reconstruct_ground_mesh_from_points(
                                point_cloud_generator=point_cloud_generator,
                                min_ray_length=ground_mesh_config.min_ray_length,
                                ground_control_point=get_nominal_ground_point_under_lidar(
                                    datasource,
                                    rig_z_offset=ground_mesh_config.z_offset,
                                    lidar_id=(
                                        ground_mesh_config.lidar_ids[0]
                                        if ground_mesh_config.lidar_ids is not None
                                        else None
                                    ),
                                ),
                                ground_compat_max_distance=ground_mesh_config.ground_compat_max_distance,
                                ground_compat_max_angle_deg=ground_mesh_config.ground_compat_max_angle_deg,
                                num_plane_hypotheses=ground_mesh_config.num_plane_hypotheses,
                                plane_max_distance=ground_mesh_config.plane_max_distance,
                                plane_max_angle_deg=ground_mesh_config.plane_max_angle_deg,
                                plane_min_eval_points=ground_mesh_config.plane_min_eval_points,
                                plane_max_eval_points=ground_mesh_config.plane_max_eval_points,
                                plane_min_inliers=ground_mesh_config.plane_min_inliers,
                                enable_plane_extension=ground_mesh_config.enable_plane_extension,
                                voxel_size=ground_mesh_config.voxel_size,
                                min_points_per_voxel=ground_mesh_config.min_points_per_voxel,
                                smoothing_passes=ground_mesh_config.smoothing_passes,
                                grid_hole_fill_min_neighbors=getattr(
                                    ground_mesh_config, "grid_hole_fill_min_neighbors", 6
                                ),
                                points_to_world_transf=nurec_to_world,
                                verbose=ground_mesh_config.verbose,
                                cuboid_tracks=datasource.get_cuboid_tracks(world_frame=True)
                                if ground_mesh_config.extend_to_cuboids.enabled
                                else None,
                                cuboids_base_margin=ground_mesh_config.extend_to_cuboids.base_margin,
                                cuboids_exclude_points_radius=ground_mesh_config.extend_to_cuboids.exclude_points_radius,
                                cuboids_exclude_points_knns=ground_mesh_config.extend_to_cuboids.exclude_points_knns,
                                cuboid_point_merge_cell_size=(
                                    ground_mesh_config.extend_to_cuboids.cuboid_point_merge_cell_size
                                ),
                            )
                        )
                        ground_mesh = Mesh(
                            vertices=ground_vertices,
                            faces=ground_triangles,
                            colors=ground_vertex_colors,
                        )

                    # Add the ground mesh to the artifact cache.
                    serialized_ground_mesh = serialize_mesh(
                        mesh=ground_mesh,
                        export_disjoint_meshes=False,
                        filename="mesh_ground",
                        formats=ground_mesh_config.formats,
                    )

                    if self.artifact_cache.meshes is None:
                        self.artifact_cache.meshes = serialized_ground_mesh
                    else:
                        self.artifact_cache.meshes.extend(serialized_ground_mesh)

            # Sequence tracks
            # These tracks can be optimized during training, so update them in the cache, otherwise the tracks saved to
            # USDZ will not be the fully optimized ones, potentially causing actor pose discrepancies when rendering.
            if self.artifact_cache.artifact_config.sequence_tracks.enabled:
                logger.debug("populate_artifact_cache() triggered caching of cuboid tracks")
                self.artifact_cache.sequence_tracks = []

                # export the raw datasource tracks
                cuboid_tracks = datasource.get_cuboid_tracks(dynamic_only=False, world_frame=True)
                usd_excluded_tracks = datasource.cuboidtracks_dynamic

                # if available merge the updated tracks from the model with the datasource tracks
                if isinstance(self.model, CompositeModel):
                    model_cuboid_tracks = CuboidTracks.Ops.transform_with_frame_conversion(
                        self.model.get_updated_cuboid_tracks(), datasource.world_to_nurec.inverse(), None
                    )
                    model_cuboid_tracks.tracks_data.tracks_flags |= TrackFlags.CONTROLLABLE

                    # Compare tracks using canonical IDs to avoid mismatches between raw IDs like
                    # "<id>@<source>" and cleaned IDs like "<id>".
                    clean_track_id = DataSourceSummary._clean_track_id_str
                    datasource_tracks_by_clean_id: dict[str, list[str]] = {}
                    for track_id in cuboid_tracks.tracks_id:
                        datasource_tracks_by_clean_id.setdefault(clean_track_id(track_id), []).append(track_id)

                    model_tracks_by_clean_id: dict[str, list[str]] = {}
                    for track_id in model_cuboid_tracks.tracks_id:
                        model_tracks_by_clean_id.setdefault(clean_track_id(track_id), []).append(track_id)

                    tracks_in_both_clean_ids = set(datasource_tracks_by_clean_id) & set(model_tracks_by_clean_id)
                    tracks_in_both = [
                        track_id
                        for clean_id in tracks_in_both_clean_ids
                        for track_id in model_tracks_by_clean_id[clean_id]
                    ]
                    tracks_datasource_only = [
                        track_id
                        for clean_id, track_ids in datasource_tracks_by_clean_id.items()
                        if clean_id not in tracks_in_both_clean_ids
                        for track_id in track_ids
                    ]

                    # exclude the tracks that are not in the model from USD export
                    usd_excluded_tracks = CuboidTracks.Ops.subset_from_tracks_id(cuboid_tracks, tracks_datasource_only)

                    if self.artifact_cache.artifact_config.sequence_tracks.controllable_only:
                        cuboid_tracks = model_cuboid_tracks
                    else:
                        cuboid_tracks = CuboidTracks.Ops.concatenate(
                            [
                                CuboidTracks.Ops.subset_from_tracks_id(cuboid_tracks, tracks_datasource_only),
                                CuboidTracks.Ops.subset_from_tracks_id(model_cuboid_tracks, tracks_in_both),
                            ]
                        )

                if clean_track_split_metadata is not None and not clean_track_split_metadata.is_empty:
                    cuboid_tracks = CuboidTracks.Ops.clean_track_ids(
                        cuboid_tracks,
                        DataSourceSummary._clean_track_id_str,
                    )
                    usd_excluded_tracks = CuboidTracks.Ops.clean_track_ids(
                        usd_excluded_tracks,
                        DataSourceSummary._clean_track_id_str,
                    )
                    cuboid_tracks = CuboidTracks.Ops.merge_split_children(
                        cuboid_tracks,
                        clean_track_split_metadata,
                    )
                    usd_excluded_tracks = CuboidTracks.Ops.merge_split_children(
                        usd_excluded_tracks,
                        clean_track_split_metadata,
                    )

                # Alpasim depends on the sequence tracks being keyed with `dummy_chunk_id`
                self.artifact_cache.sequence_tracks += serialize_sequence_tracks(
                    sequence_id="dummy_chunk_id",
                    cuboid_tracks=cuboid_tracks,
                    excluded_tracks=usd_excluded_tracks,
                    usd_timestamp_offset_us=usd_timestamp_offset_us,
                )

                self.artifact_cache.usd_sequence_tracks = get_usd_references(self.artifact_cache.sequence_tracks)

        # Rig trajectories and model-free lidar firing patterns use the same calibrated
        # trajectories, but either artifact member can be enabled independently.
        if isinstance(datasource, RigTrajectoriesProvider) and (
            self.artifact_cache.artifact_config.rig_trajectories.enabled
            or self.artifact_cache.artifact_config.firing_patterns.enabled
        ):
            # Need to export both start and end frame timestamps to enable rendering input views with rolling shutter,
            # otherwise there will be a time sync misalignment between rendered and input views.
            rig_trajectories = datasource.get_rig_trajectories()
            calib = getattr(self.model, "calib", None)
            if isinstance(calib, BaseCalib):
                rig_trajectories = calib.apply_to_rig_trajectories(rig_trajectories)
            if self.artifact_cache.artifact_config.rig_trajectories.enabled:
                self.artifact_cache.rig_trajectories = serialize_rig_trajectories(
                    rig_trajectories,
                    usd_timestamp_offset=usd_timestamp_offset_us,
                    add_default_cameras=self.artifact_cache.artifact_config.rig_trajectories.add_default_cameras,
                )

            populate_firing_patterns(
                self.artifact_cache,
                rig_trajectories,
                getattr(datasource, "lidar_sensors", {}),
            )

    def serialize_artifact_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Serialize the checkpoint for the artifact. Subclasses must override to set load_in_place and apply transforms."""
        raise NotImplementedError("Subclasses must implement serialize_artifact_checkpoint")

    def on_train_end(self) -> None:
        super().on_train_end()

        # Write the final artifact and PPISP report at end of training (rank 0 only).
        # The artifact cache was populated during the last on_save_checkpoint call.
        report_start_error: BaseException | None = None
        try:
            self._start_ppisp_report_export_rank_zero()
        except BaseException as error:
            report_start_error = error
        try:
            if self.artifact_cache.artifact_config.enabled:
                self._write_last_artifact_rank_zero()
        except BaseException:
            if report_start_error is not None:
                logger.error(
                    "PPISP report startup failed before artifact publication also failed.",
                    exc_info=(type(report_start_error), report_start_error, report_start_error.__traceback__),
                )
            self._finish_ppisp_report_export_rank_zero(suppress_exceptions=True)
            raise
        if report_start_error is not None:
            raise report_start_error
        self._finish_ppisp_report_export_rank_zero()

    def on_save_checkpoint(self, checkpoint: Checkpoint) -> None:
        # Artifact subsampling must not change the next training random draw.
        with preserve_training_rng_state(self.device):
            super().on_save_checkpoint(checkpoint)

            if self.artifact_cache.artifact_config.enabled:
                self._populate_and_save_artifacts_rank_zero(checkpoint)

            # Apply training-checkpoint options after artifact serialization so they do not affect the artifact.
            checkpoint_config = self.cached_config.checkpoint
            if checkpoint_config.strip_optimizer:
                for key in ("loops", "callbacks", "optimizer_states", "lr_schedulers", "MixedPrecisionPlugin"):
                    checkpoint.pop(key, None)
            if checkpoint_config.fp16:
                checkpoint["state_dict"] = reduce_precision_to_fp16(checkpoint["state_dict"])

    @rank_zero_only
    def _populate_and_save_artifacts_rank_zero(self, checkpoint: Checkpoint) -> None:
        os.makedirs(artifact_dir := os.path.join(self.out_dir, self.run_id, "artifacts"), exist_ok=True)

        self._last_step_artifact_path = None

        # Populate artifact cache
        self.populate_artifact_cache(checkpoint)

        # Write artifact for this step
        if (
            self.checkpoint_every_n_train_steps is not None
            and self.trainer.global_step % self.checkpoint_every_n_train_steps == 0
        ):
            self._last_step_artifact_path = Path(os.path.join(artifact_dir, f"{self.last_checkpoint_step:06d}.usdz"))
            try:
                write_artifact_cache(self.artifact_cache, self._last_step_artifact_path)
            except BaseException:
                # Never let on_train_end publish a stale or partially written
                # numbered artifact if a caller catches the checkpoint error.
                self._last_step_artifact_path = None
                self._finish_ppisp_report_export_rank_zero(suppress_exceptions=True)
                raise

    @rank_zero_only
    def _write_last_artifact_rank_zero(self) -> None:
        artifact_dir = os.path.join(self.out_dir, self.run_id, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        last_artifact_path = Path(os.path.join(artifact_dir, "last.usdz"))
        if self._last_step_artifact_path is not None and self._last_step_artifact_path.is_file():
            publish_existing_usdz(self._last_step_artifact_path, last_artifact_path)
        else:
            write_artifact_cache(self.artifact_cache, last_artifact_path)

    @rank_zero_only
    def _start_ppisp_report_export_rank_zero(self) -> None:
        if self._ppisp_report_future is not None:
            return
        ppisp_module = get_ppisp_from_model(self.model)
        if ppisp_module is None:
            return
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppisp-report")
        try:
            future = executor.submit(self._maybe_export_ppisp_report_rank_zero, ppisp_module)
        except BaseException:
            try:
                executor.shutdown(wait=True)
            except BaseException:
                logger.exception("PPISP report executor shutdown failed after submission failed.")
            raise
        self._ppisp_report_executor = executor
        self._ppisp_report_future = future

    @rank_zero_only
    def _finish_ppisp_report_export_rank_zero(self, *, suppress_exceptions: bool = False) -> None:
        if self._ppisp_report_future is None:
            return
        report_failed = False
        try:
            self._ppisp_report_future.result()
        except BaseException:
            report_failed = True
            if not suppress_exceptions:
                raise
            logger.exception("PPISP report export failed while handling another exception.")
        finally:
            assert self._ppisp_report_executor is not None
            executor = self._ppisp_report_executor
            self._ppisp_report_executor = None
            self._ppisp_report_future = None
            try:
                executor.shutdown(wait=True)
            except BaseException:
                if not suppress_exceptions and not report_failed:
                    raise
                logger.exception("PPISP report executor shutdown failed while handling another exception.")

    @rank_zero_only
    def _maybe_export_ppisp_report_rank_zero(self, ppisp_module: Any | None = None) -> None:
        """Export PPISP PDF report at the end of training (rank 0 only)."""
        if ppisp_module is None:
            ppisp_module = get_ppisp_from_model(self.model)
        if ppisp_module is None:
            return

        datasource = self.datamodule.get_datasource()
        assert isinstance(datasource, NCOREDataSource), "PPISP report export requires an NCOREDataSource"

        frames_per_camera = datasource.get_n_training_frames_per_camera().tolist()
        camera_names = list(datasource.all_camera_ids)
        output_dir = Path(os.path.join(self.out_dir, self.run_id, "ppisp_report"))

        pdf_paths = export_ppisp_report(
            ppisp=ppisp_module,
            frames_per_camera=frames_per_camera,
            output_dir=output_dir,
            camera_names=camera_names,
        )

        logger.info("PPISP report exported: %s", [str(p) for p in pdf_paths])

    def log_and_inpaint_env_map(self, env_map: SkyEnvMapBackground) -> None:
        """
        Logs and potentially inpaints the textures used by SkyEnvMapBackground to Wandb/Tensorboard.
        The inpainting is just used at inference time and should be restored by calling env_map.restore_original_textures()
        before resuming training.
        """

        image_group = "sky_textures-step_" + str(self.global_step) + "-rank_" + str(self.global_rank)
        textures = env_map.flattened_textures().clamp(0, 1)

        self.log_image((textures * 255).byte().cpu().numpy(), image_group, self.global_step)

        inpaint_result = env_map.maybe_inpaint()
        if inpaint_result is not None:
            inpainted_textures, inpaint_mask = inpaint_result

            mask_image_group = "sky_textures-mask-step_" + str(self.global_step) + "-rank_" + str(self.global_rank)
            inpainted_image_group = (
                "sky_textures-inpainted-step_" + str(self.global_step) + "-rank_" + str(self.global_rank)
            )

            self.log_image((inpaint_mask * 255).byte().cpu().numpy(), mask_image_group, self.global_step)
            self.log_image(
                (inpainted_textures * 255).byte().cpu().numpy(),
                inpainted_image_group,
                self.global_step,
            )

    def get_track_orbit_views(
        self,
        track: CuboidTracks,
        distance: float,
        width: int,
        height: int,
        n_frames: int,
    ) -> Generator[DataAndRenderingBatch, None, None]:
        """
        Generates views that orbit around a specified track.
        Args:
            track: singleton cuboid track
            distance: orbit distance from track
            width: render width
            height: render height
            n_frames: number of views to render
        Returns: Batches for the orbit views
        """
        assert track.n_tracks == 1, "expected single track id"
        # Derive a fov that roughly captures the object and some background
        fov = 3 * np.rad2deg(np.arctan2(0.5 * track.cuboids_dims.max().item(), distance))

        camera_model, camera_params, target_pixels = self._get_pinhole_camera(width, height, fov)
        timestamp_us = int(track.tracks_timestamps_us[track.tracks_poses.shape[0] // 2])

        data_batch = DataBatch(
            camera=DataBatch.Camera(
                meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0, training_frame_idx=-1)],
                labels=CameraFrameLabels(),  # empty labels
            ),
        )

        track_pose = track.tracks_poses[track.tracks_poses.shape[0] // 2].matrix()
        for degree in np.arange(0, 360, 360 / n_frames):
            c2w_object = torch.eye(4, device=self.device)
            rotation = Rotation.from_euler("xz", [-90, degree], degrees=True)
            c2w_object[:3, :3] = to_torch(rotation.as_matrix(), device=self.device)
            c2w_object[:3, 3] = to_torch(rotation.apply(np.array([0, 0, -distance])), device=self.device)

            c2w_nurec = track_pose @ c2w_object

            # Keep GPU version for GPU operations, create CPU copy to avoid .item() calls
            timestamps_gpu = torch.tensor([timestamp_us, timestamp_us], dtype=torch.int64, device=self.device).reshape(
                1, 2
            )
            timestamps_cpu = torch.tensor([timestamp_us, timestamp_us], dtype=torch.int64, device="cpu").reshape(1, 2)
            rendering_batch = RenderingBatch(
                camera=RenderingData(
                    rays=camera_model.pixels_to_world_rays_static_pose(target_pixels, c2w_nurec).world_rays.reshape(
                        1, height, width, 6
                    ),
                    sensor_model_parameters=[camera_params],
                    poses_tquat_startend=se3_matrix_to_tquat(c2w_nurec).unsqueeze(0).expand(2, -1).reshape(1, 2, 7),
                    timestamps_startend_us=timestamps_gpu,  # on GPU
                    rays_timestamps_us=torch.full(
                        (target_pixels.shape[0],), timestamp_us, dtype=torch.int64, device=self.device
                    ).reshape(1, height, width, 1),
                    timestamps_startend_us_cpu=timestamps_cpu,  # on CPU
                )
            )
            yield DataAndRenderingBatch(data=data_batch, rendering=rendering_batch)

    def _get_pinhole_camera_parameters(
        self, image_width: int, image_height: int, fov: float
    ) -> OpenCVPinholeCameraModelParameters:
        """
        Generates pinhole camera parameters with the desired width, height, and fov
        """
        image_dim = np.array([image_width, image_height], dtype=np.uint64)
        target_principal_point = image_dim.astype(np.float32) / 2
        target_focal_length = np.min(image_dim.astype(np.float32)) / (2.0 * np.tan(np.deg2rad(fov) * 0.5))
        camera_params = OpenCVPinholeCameraModelParameters(
            resolution=image_dim,
            shutter_type=ShutterType.GLOBAL,
            focal_length=np.array([target_focal_length, target_focal_length], dtype=np.float32),
            principal_point=target_principal_point,
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )
        return camera_params

    def _get_pinhole_camera(
        self, image_width: int, image_height: int, fov: float
    ) -> Tuple[OpenCVPinholeCameraModel, OpenCVPinholeCameraModelParameters, torch.Tensor]:
        """
        Generates a pinhole camera model with the desired width, height, and fov
        """
        camera_params = self._get_pinhole_camera_parameters(image_width, image_height, fov)
        camera_model = OpenCVPinholeCameraModel(camera_params, device=self.device)

        pixels_x = torch.arange(image_width, dtype=torch.int16, device=self.device)
        pixels_y = torch.arange(image_height, dtype=torch.int16, device=self.device)
        grid_x, grid_y = torch.meshgrid(pixels_x, pixels_y, indexing="xy")
        target_pixels = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        return camera_model, camera_params, target_pixels

    def apply_alpha(self, batch: DataAndRenderingBatch, out: GaussiansCompositeReturn) -> None:
        """Perform alpha-compositing if alpha labels are given - updates batch in-place!"""

        if (
            out.rendered_cam is None
            or out.rendered_cam.extra_ray_signals is None
            or out.rendered_cam.extra_ray_signals.rgb_background is None
            or batch.data.camera is None
            or batch.data.camera.labels.alpha is None
        ):
            return

        batch.data.camera.labels.rgb = unpack_optional(batch.data.camera.labels.rgb) * batch.data.camera.labels.alpha[
            ..., None
        ] + out.rendered_cam.extra_ray_signals.rgb_background * (1 - batch.data.camera.labels.alpha[..., None])

    def collect_metric(
        self,
        name: str,
        value: Any,
        is_lidar: bool = False,
        frame_meta: FrameMeta | None = None,
        timestamps_startend_us: torch.Tensor | None = None,
        sequence_id: Union[list[str], str, None] = None,
        reduce_fx: Callable[[torch.Tensor], torch.Tensor] | None = None,
        timing_method: str | None = None,
    ) -> None:
        """
        Overrides the log method to retain some metrics for later saving
        """
        if timing_method is not None:
            self._validation_metrics_collector.set_metric_metadata(name, timing_method=timing_method)

        unique_sensor_id = None

        if frame_meta is not None:
            unique_sensor_idx = frame_meta.unique_sensor_idx
            unique_sensor_id = (
                self.datamodule.get_datasource().get_camera_sensor_ids(unique_sensors=True)[unique_sensor_idx]
                if not is_lidar
                else self.datamodule.get_datasource().get_lidar_sensor_ids(unique_sensors=True)[unique_sensor_idx]
            )

        self._validation_metrics_collector.save_metric(
            create_metric_sample(name, value, frame_meta, timestamps_startend_us),
            unique_sensor_id=unique_sensor_id,
            sequence_id=sequence_id,
            reduce_fx=reduce_fx,
            is_lidar=is_lidar,
        )

        # We can't `self.log()` inside `on_validation_end`. HINT: Can still log directly to
        # the logger by using `self.logger.experiment`.
        try:
            self.log(name, value, prog_bar=False)
        except MisconfigurationException as e:
            logger.warning(f"Failed to log metric {name}: {e}")
        except Exception as e:
            raise e
