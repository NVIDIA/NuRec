# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import torch

from omegaconf import DictConfig, ListConfig, OmegaConf

import ncore.data

from ncore.sensors import maybe_lidar_model_from_parameters
from nurec.libs.vren.lidars import (  # type: ignore
    preprocess_lidar_raygen_only,
)
from nurec.ncore_backends import (
    DatasetBackend,
    UnsupportedBackendOperationError,
)
from nurec.utils.misc import unpack_optional


class LidarModelBundle:
    """
    Wrapper around the meta data for rendering of a Lidar model.
    """

    @classmethod
    def load_from_config(cls, lidar_config: dict) -> LidarModelBundle:
        # Handle the case where lidar_config is a dictionary, but might contain OmegaConf objects
        processed_config = {}

        for key, value in lidar_config.items():
            if isinstance(value, (ListConfig, DictConfig)):
                processed_config[key] = OmegaConf.to_container(value, resolve=True)
            else:
                processed_config[key] = value

        data = json.dumps(processed_config)

        lidar_parameters = ncore.data.RowOffsetStructuredSpinningLidarModelParameters.from_json(data)
        return cls(lidar_parameters)

    def __init__(self, lidar_parameters: ncore.data.ConcreteLidarModelParametersUnion):
        self.lidar_parameters = lidar_parameters
        self.lidar_model = unpack_optional(maybe_lidar_model_from_parameters(lidar_parameters))
        # This model is just for converting elements to world rays, hard code the parameters for tile.
        self.vren_lidar = preprocess_lidar_raygen_only(lidar_parameters, device=torch.device("cuda"))
        self.elements = np.stack(
            np.meshgrid(
                np.arange(lidar_parameters.n_rows, dtype=np.uint16),
                np.arange(lidar_parameters.n_columns, dtype=np.uint16),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 2))


def get_lidar_model_parameters_with_fallbacks(
    lidar_sensor: ncore.data.LidarSensorProtocol,
    cwccw_fallback: bool = True,
    *,
    backend: DatasetBackend,
) -> ncore.data.ConcreteLidarModelParametersUnion | None:
    """
    Robustly get lidar model parameters from the sensor, applying optional fallbacks if enabled
    """

    try:
        lidar_model_parameters = backend.get_lidar_model_parameters(lidar_sensor)
    except AssertionError as e:
        # Temporary WAR to support datasets generated with wrong conventions (prior to nrs/ncore!363)
        if not cwccw_fallback:
            raise AssertionError(
                f"{e} - Lidar model parameters may follow outdated conventions. "
                f"Consider enabling 'lidar_model_parameter_cwccw_fallback' in the config to apply a fallback."
            )

        try:
            lidar_model_parameters_dict = backend.get_lidar_model_metadata(lidar_sensor)
        except UnsupportedBackendOperationError as backend_error:
            raise ValueError(
                f"CW/CCW fallback is not supported by the {backend.name!r} dataset backend"
            ) from backend_error

        # Flip orientation of the spin direction (data was saved in the exact opposite order)
        lidar_model_parameters_dict["spinning_direction"] = (
            "cw" if lidar_model_parameters_dict["spinning_direction"] == "ccw" else "ccw"
        )

        # Remove obsolve FOV parameters
        for fov_key in [
            "fov_horiz_start_rad",
            "fov_horiz_end_rad",
            "fov_vert_start_rad",
            "fov_vert_end_rad",
            "fov_horiz_min_rad",
            "fov_vert_min_rad",
            "fov_horiz_max_rad",
            "fov_vert_max_rad",
        ]:
            lidar_model_parameters_dict.pop(fov_key, None)

        lidar_model_parameters = ncore.data.RowOffsetStructuredSpinningLidarModelParameters.from_dict(
            lidar_model_parameters_dict
        )

    return lidar_model_parameters
