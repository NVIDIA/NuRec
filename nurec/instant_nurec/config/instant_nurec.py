# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import re

from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import model_validator

from nurec.config.base_schema import BaseConfigSchema, Field
from nurec.config.logger import BatchMediaLoggerConfigMixin, LoggerConfigType
from nurec.config.trainer import infer_slurm_environment
from nurec.config.version import Version, get_version
from nurec.config.viewer import ViewerConfig
from nurec.instant_nurec.config.dataset import (
    BaseNCoreInstantNuRecDatasetConfig,
    InstantNuRecEpochSplitConfig,
    InstantNuRecMixedDatasetConfig,
    InstantNuRecSplitsConfig,
)
from nurec.instant_nurec.config.models import CelsiusModelConfig, KelvinModelConfig
from nurec.instant_nurec.config.predict import PredictConfig


SENTINEL = "<sentinel>"

current_version = get_version()

cmd_logger = logging.getLogger(__name__)


class InstantNuRecSystemConfig(BatchMediaLoggerConfigMixin, BaseConfigSchema):
    """
    Currently we only have one instant_nurec system, and this is the base class for it.
    In the future we might want to have different subclasses for this.

    This is inference-only: it carries just the fields needed to build a model and run
    predict. Training-only fields (optimizer/scheduler, max_epochs, train/val/test loader
    settings, etc.) live on `TrainInstantNuRecSystemConfig` in a separate downstream training package.
    """

    enable_render_global_step: int = Field(
        default=0,
        description="Enable rendering (and hence losses) at this global step. <=0 means always enable.",
    )

    log_every_n_steps: int = Field(
        default=50,
        description=(
            "Number of steps between logging numerical metrics. Defaulted here for predict, where logging "
            "cadence is largely inconsequential; `TrainInstantNuRecSystemConfig` re-declares this as required."
        ),
    )

    # Number of devices per node.
    device_count: int = Field(
        default=0,
        description="Number of devices per node. If set to 0, will be inferred automatically.",
    )

    num_nodes: int = Field(
        default=0,
        description="Number of nodes to use for distributed inference. If set to 0, will be inferred automatically.",
    )
    strategy: Literal["auto", "ddp_find_unused_parameters"] = Field(
        default="auto",
        description="Strategy to use for distributed inference. If set to 'auto', will be inferred automatically.",
    )

    precision: int | str = Field(
        description="https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision",
    )

    predict_num_workers: int = Field(default=0, description="Number of workers for the predict dataloader per-node.")
    predict_batch_size: int = Field(default=1, description="Batch size for the predict dataloader. Typically set to 1.")

    @property
    def world_size(self) -> int:
        world_size = self.device_count * self.num_nodes
        assert world_size > 0, "Please ensure that device_count and num_nodes are set correctly."
        return world_size

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)

        # Use SLURM environment variables to set device_count and num_nodes if not set
        # If SLURM is not detected, assuming single-node.
        slurm_environment = infer_slurm_environment()

        if self.device_count == 0:
            self.device_count = slurm_environment.num_tasks_per_node if slurm_environment else 1

        if self.num_nodes == 0:
            self.num_nodes = slurm_environment.num_nodes if slurm_environment else 1


class GaussiansInstantNuRecSystemConfig(InstantNuRecSystemConfig):
    """
    System config for the Gaussians instant_nurec system.
    """

    name: Literal["base-instant_nurec-system"]


class InstantNuRecConfig(BaseConfigSchema):
    """
    Top-level configuration for instant_nurec inference (predict mode).
    """

    seed: int = Field(default=38, description="Random seed.")
    enable_cudnn_benchmark: bool = Field(
        default=True,
        description=(
            "Enable cuDNN convolution autotuning when the training dataset has static shapes. "
            "Disable this for recipes whose cold autotuning workspace would consume unsafe HBM."
        ),
    )
    mode: Literal["train", "val", "test", "trainval", "predict"] = Field(
        default="predict",
        description=(
            "Widened here (rather than `Literal['predict']`) so that `TrainInstantNuRecConfig` (in a "
            "separate downstream training package) can narrow/require it without a "
            "Liskov-violating type override. "
            "Plain `InstantNuRecConfig` instances (as opposed to `TrainInstantNuRecConfig` ones) are still restricted "
            "to `mode='predict'` at runtime -- see `_validate_predict_only_mode` below."
        ),
    )

    resume: str | None
    resume_weights_only: bool
    call_train_from_scratch_hook_for_validation: bool = Field(
        default=False,
        description=(
            "When True, the model's on_train_from_scratch_start hook is also called when running predict "
            "without a checkpoint but with init weights. Set to False if the hook contains train-only logic that "
            "must not run during eval (e.g. writing training state)."
        ),
    )
    verbose: bool = Field(default=False, description="Verbose mode.")
    force_validate: bool = Field(default=False, description="Force validation of the config.")

    out_dir: str
    logger: LoggerConfigType = Field(discriminator="name")

    viewer: ViewerConfig = Field(default_factory=ViewerConfig, description="Web-based viewer configuration.")

    system: GaussiansInstantNuRecSystemConfig
    dataset: InstantNuRecSplitsConfig = Field(discriminator="name")

    model: CelsiusModelConfig | KelvinModelConfig = Field(discriminator="name")

    # Predict configuration
    predict: PredictConfig = Field(
        default_factory=PredictConfig,
        description="Configuration for predict-time-only functionality such as primitive merging",
    )

    version: Version | None = Field(
        default=current_version,
        description="Not to be set by the user. Used to detect NuRec version mismatch when loading old configs. Not available in sandboxed test executions",
    )

    save_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where images are saved during validation phase. If left unchanged, defaults to `out_dir/save`"
        ),
    )
    ckpt_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where model checkpoints are saved during training. If left "
            "unchanged, defaults to `out_dir/checkpoints`"
        ),
    )
    config_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where the config (with all the auto-generated and default "
            "fields) will be stored. If left unchanged, defaults to `out_dir/config`"
        ),
    )
    run_id: str = Field(
        default=SENTINEL,
        description=(
            "A unique identifier of the training run. If left unchanged, will be auto-generated. "
            "If resuming training from a checkpoint, the previous run_id will be restored."
        ),
    )

    class OverrideConfig(BaseConfigSchema):
        selector: list[str] = Field(
            description="Selector for the config to override. One can use regex to match the items."
        )
        config: Any = Field(description="The actual config to be merged with the selected configs.")

    shared_config: Any | None = Field(
        default=None,
        description="Place to put common configs to be shared via YAML's `<<: *shared_config` syntax. "
        "This will not come to the final config, just for handy purposes",
    )
    override_config: dict[str, OverrideConfig] = Field(
        default_factory=dict,
        description="Override config for the entire config. Can be used in junction with shared_config to create various combinations.",
    )

    @model_validator(mode="before")
    @classmethod
    def merge_override_config(cls, objects: DictConfig) -> DictConfig:
        # Recursively print out keys from the config to perform matching
        def iter_key_paths(d: dict[str, Any], prefix: str = ""):
            for k, v in d.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    yield path
                    yield from iter_key_paths(v, path)

        # Apply the overrides one by one
        if "override_config" in objects:
            for override_name, override_item in objects.override_config.items():
                override_model = cls.OverrideConfig.model_validate(override_item)

                matched_paths: list[str] = []
                for path in iter_key_paths(OmegaConf.to_container(objects)):  # type: ignore
                    for sel_regex in override_model.selector:
                        if re.fullmatch(sel_regex, path):
                            matched_paths.append(path)
                            break

                # Override the config paths
                for path in matched_paths:
                    cmd_logger.info(f"Overriding {path} with override key '{override_name}'.")
                    config_items = OmegaConf.select(objects, path)
                    assert isinstance(config_items, DictConfig)
                    with open_dict(config_items):
                        config_items.merge_with(override_model.config)
            OmegaConf.update(objects, "override_config", {}, merge=False)

        # Remove config overrides from the final config.
        if "shared_config" in objects:
            OmegaConf.update(objects, "shared_config", None)
        return objects

    @model_validator(mode="after")
    def _validate_predict_only_mode(self) -> InstantNuRecConfig:
        # Runtime enforcement of the invariant that `mode`'s Literal used to encode statically
        # (`Literal["predict"]`): plain `InstantNuRecConfig` (predict-only) instances must have
        # mode == "predict". `TrainInstantNuRecConfig` (in a separate downstream training package)
        # re-declares `mode` with its own (still-restricted) Literal, so it is deliberately
        # exempted via the exact-type check below rather than `isinstance`.
        if type(self) is InstantNuRecConfig and self.mode != "predict":
            raise ValueError(
                f"InstantNuRecConfig.mode must be 'predict', got {self.mode!r}. Use TrainInstantNuRecConfig for other modes."
            )
        return self

    def resolve_ncore_train_dataset_config(self) -> BaseNCoreInstantNuRecDatasetConfig | None:
        if not isinstance(self.model, (CelsiusModelConfig, KelvinModelConfig)) or "train" not in self.mode:
            return None

        train_cfg = self.dataset.train
        if isinstance(train_cfg, InstantNuRecEpochSplitConfig):
            train_cfg = train_cfg.last_milestone()
        if isinstance(train_cfg, InstantNuRecMixedDatasetConfig):
            components = [(name, comp) for name, comp in train_cfg.mixture.items() if comp.sample_ratio > 0]
            if not components:
                return None
            train_cfg = max(components, key=lambda item: item[1].sample_ratio)[1].config

        if not isinstance(train_cfg, BaseNCoreInstantNuRecDatasetConfig):
            return None

        return train_cfg

    def has_static_shapes(self) -> bool:
        train_dataset = self.resolve_ncore_train_dataset_config()
        return train_dataset is not None and train_dataset.has_static_shapes()

    def _setup_resume_path(self) -> None:
        if self.resume is None:
            return

        # When user use resume=auto, we will try to resume from the last checkpoint with the same run_id.
        # If the checkpoint does not exist, we will train from scratch.
        # This is useful for long-running training jobs on clusters where the job is constantly interrupted & restarted.
        if self.resume == "auto":
            is_auto_resume = True
            self.resume = "last"
        else:
            is_auto_resume = False

        # Add .ckpt if config.resume doesn't include it. Useful to run resume=last or resume=best
        if not self.resume.endswith(".ckpt"):
            self.resume += ".ckpt"
        if not os.path.exists(self.resume):
            if self.ckpt_dir == SENTINEL:
                if is_auto_resume:
                    # Automatically find out the checkpoint directory
                    self.ckpt_dir = os.path.join(self.out_dir, self.logger.run_id, "checkpoints")
                    cmd_logger.info(f"Automatically resuming from {self.ckpt_dir}")
                else:
                    raise FileNotFoundError("config.ckpt_dir is not provided, can't guess the resume path.")

            self.resume = os.path.join(self.ckpt_dir, self.resume)
            if not os.path.exists(self.resume):
                if is_auto_resume:
                    # If checkpoint not found, training from scratch!
                    self.resume = None
                    cmd_logger.info("Checkpoint not found, training from scratch.")
                else:
                    raise FileNotFoundError(f"Checkpoint {self.resume=} does not exist")

    def _setup_run_id(self) -> None:
        """propagate run_id set in self.logger to other fields"""
        run_id = self.logger.run_id
        self.run_id = run_id
        self.save_dir = os.path.join(self.out_dir, run_id, "save")
        self.ckpt_dir = os.path.join(self.out_dir, run_id, "checkpoints")
        self.config_dir = os.path.join(self.out_dir, run_id, "config")

    def model_post_init(self, __context) -> None:
        self._setup_resume_path()
        self._setup_run_id()  # modifies self.ckpt_dir so needs to be called AFTER _setup_resume_path
