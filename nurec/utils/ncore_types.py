# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrowing of NCore's abstract parameter types to the concrete models this repository handles

NCore types its reader APIs against the abstract parameter bases, so a model registered outside
NCore can flow through them. This repository is written against the concrete models and reads their
model-specific fields throughout, so it converts once at the boundary rather than threading the
abstract type through every signature.

Deliberately a leaf module: `nurec.utils.ncore_utils` imports `nurec.ncore_backends`, and the backends
need this narrowing, so putting these here keeps that from becoming an import cycle.
"""

from __future__ import annotations

import ncore.data

from nurec.utils.misc import get_union_types


def require_concrete_camera_parameters(
    parameters: ncore.data.CameraModelParameters,
) -> ncore.data.ConcreteCameraModelParametersUnion:
    """Narrow NCore's abstract camera parameters to the models this repository handles

    NCore types its reader APIs against the abstract bases since 19.7.0, so a model registered
    outside NCore can flow through them. This repository is written against the four in-tree models
    and reads their model-specific fields throughout, so it converts once, here, rather than
    threading the abstract type through every signature and narrowing again at each use.

    Raises:
        TypeError: If the parameters are not one of the camera models this repository handles.
    """
    if not isinstance(parameters, get_union_types(ncore.data.ConcreteCameraModelParametersUnion)):
        raise TypeError(f"Unsupported NCore camera model for this repository: {type(parameters).__name__}")
    return parameters


def require_concrete_lidar_parameters(
    parameters: ncore.data.LidarModelParameters | None,
) -> ncore.data.ConcreteLidarModelParametersUnion | None:
    """Narrow NCore's abstract lidar parameters to the models this repository handles

    `None` is passed through: a sensor without lidar model parameters is normal.

    Raises:
        TypeError: If the parameters are not one of the lidar models this repository handles.
    """
    if parameters is None:
        return None
    if not isinstance(parameters, get_union_types(ncore.data.ConcreteLidarModelParametersUnion)):
        raise TypeError(f"Unsupported NCore lidar model for this repository: {type(parameters).__name__}")
    return parameters


def require_concrete_external_distortion_parameters(
    parameters: ncore.data.ExternalDistortionParameters | None,
) -> ncore.data.ConcreteExternalDistortionParametersUnion | None:
    """Narrow NCore's abstract external distortion parameters to the models this repository handles

    `CameraModelParameters.external_distortion_parameters` is declared against the abstract base
    since the serialized form began carrying its concrete type, so it needs narrowing before the
    windshield-specific fields are read. `None` is passed through: a camera without external
    distortion is normal.

    Raises:
        TypeError: If the parameters are not one of the models this repository handles.
    """
    if parameters is None:
        return None
    if not isinstance(parameters, get_union_types(ncore.data.ConcreteExternalDistortionParametersUnion)):
        raise TypeError(f"Unsupported NCore external distortion model for this repository: {type(parameters).__name__}")
    return parameters
