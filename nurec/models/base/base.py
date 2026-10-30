# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf

from nurec import __version__
from nurec.config import BaseConfigSchema
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.misc import map_optional, strip_none_from_config
from nurec.utils.optim import OptimizerLRSchedulerConfig


class BaseModel(nn.Module):
    post_processings: Iterable[nn.Module]

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        # In PyTorch, any registered buffer in an nn.Module will be automatically moved to the correct device when `to(device)` is
        # called on the module. Thus, we can create a transient marker to track the device of the model. Because it's a registered
        # buffer, it is moved alongside the rest of the model. Thus, we can reliably query self._device_indicator.device to determine
        # which device the model currently resides on.
        self._device_indicator = nn.Buffer(torch.tensor(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_indicator.device

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        """
        Hook function for system, invoked at system.on_train_batch_start(), i.e. the very start of each train step.
        Useful for e.g. update_module_step()
        """
        return {}

    def update_step_train_batch_end(
        self, epoch: int, global_step: int, batch: DataAndRenderingBatch, system, **kwargs
    ) -> None:
        """
        Hook function for system, invoked at system.on_train_batch_end(), i.e. the very end of each train step.
        """
        pass

    def on_train_from_scratch_start(self, system, **kwargs) -> None:
        """
        Hook function for system, invoked at system.on_train_start() only when system.resume is False.
        Useful for e.g. setting scalers, shape initialization process for sdf models, etc.
        """
        pass

    def train(self, mode: bool = True) -> BaseModel:
        return super().train(mode=mode)

    def eval(self) -> BaseModel:
        return super().eval()

    @torch.no_grad()
    def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict[str, Any]:
        def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
            # default conversion to half for single precision tensor
            tensor = tensor.to(dtype=torch.float16) if tensor.dtype == torch.float32 else tensor
            return tensor.flatten().cpu().numpy().tobytes()

        def tensor_shape_to_list(tensor: torch.Tensor) -> list[int]:
            return list(tensor.size())

        # Convert config to plain dict for JSON serialization
        if isinstance(self.config, BaseConfigSchema):
            # Typed config (Pydantic), need deep conversion for nested DictConfigs in Any fields
            omegaconf_dictconfig = self.config.to_dictconfig()
            config_dict = OmegaConf.to_container(omegaconf_dictconfig, resolve=False, enum_to_str=True)
        else:
            # Untyped config (DictConfig)
            config_dict = OmegaConf.to_container(self.config, enum_to_str=True)

        config_dict = strip_none_from_config(config_dict)

        # refers to for versioning and format details
        json_dict = {
            "nurec_data": {
                "version": map_optional(__version__, lambda v: v.semantic_string()),
                "model": "nurec",  # base nurec model
                "config": config_dict,
                "state_dict": {f".{key}": value for key, value in self.state_dict().items()} if with_state_dict else {},
            }
        }

        assert isinstance(json_dict["nurec_data"]["state_dict"], dict)

        if with_state_dict:
            # add shape entries for every tensor
            json_dict["nurec_data"]["state_dict"].update(
                {
                    key + ".shape": tensor_shape_to_list(value)
                    for key, value in json_dict["nurec_data"]["state_dict"].items()
                    if isinstance(value, torch.Tensor) and isinstance(key, str)
                }
            )
            # convert tensor to bytes
            json_dict["nurec_data"]["state_dict"] = {
                key: tensor_to_bytes(value) if isinstance(value, torch.Tensor) else value
                for key, value in json_dict["nurec_data"]["state_dict"].items()
            }

        # remove unserializable object
        # TODO : this is not filtering pybind object.
        def serializable_value(value) -> bool:
            return not inspect.isclass(value)

        if with_state_dict:
            json_dict["nurec_data"]["state_dict"] = {
                key: value for key, value in json_dict["nurec_data"]["state_dict"].items() if serializable_value(value)
            }

        return json_dict

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Returns a list of module-owned configured optimizers (optimizers paired with an optional LR scheduler),
        which will be stepped in the main training loop, allowing the module to interact with it's owned optimizers"""
        return []

    def get_base_model_children(self) -> Iterator[BaseModel]:
        """Yields all children that are instances of BaseModel"""
        for child in self.children():
            if isinstance(child, BaseModel):
                yield child

    def configure_sharded_params_and_buffers(self) -> list[str]:
        """Return the stable names of state sharded across ranks.

        Implementations also mark currently materialized tensors as sharded.
        Lazy state still returns its name so DDP can exclude it before storage
        is materialized.
        """
        return []

    @staticmethod
    def mark_as_sharded(param: torch.Tensor) -> None:
        setattr(param, "_sharded", True)

    @staticmethod
    def is_sharded(param: torch.Tensor) -> bool:
        return hasattr(param, "_sharded") and param._sharded
