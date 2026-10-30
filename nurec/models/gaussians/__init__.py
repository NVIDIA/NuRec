# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.models.gaussians.construction_barriers_model import ConstructionBarriersGaussianModel
from nurec.models.gaussians.gaussians_model import (
    BaseGaussianModel,
    DeformableGaussianModel,
    RigidGaussianModel,
    SHGaussianModel,
)
from nurec.models.gaussians.initializations import (
    AccumulatedPointCloudInitialization,
    BaseInitialization,
    DynamicTracksInitialization,
    LidarRigTrajInitialization,
    RandomInitialization,
)
from nurec.models.gaussians.interpolated_table_gaussians import InterpolatedTableDeformableGaussianModel
from nurec.models.gaussians.renderers import BaseGaussianRenderer
from nurec.models.gaussians.vehicle_light import (
    VehicleLightRigidGaussianModel,
    VehicleLightTracksInitialization,
)


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseGaussianModel",
    "SHGaussianModel",
    "ConstructionBarriersGaussianModel",
    "RigidGaussianModel",
    "DeformableGaussianModel",
    "InterpolatedTableDeformableGaussianModel",
    "VehicleLightRigidGaussianModel",
    "BaseGaussianRenderer",
    "BaseInitialization",
    "RandomInitialization",
    "LidarRigTrajInitialization",
    "DynamicTracksInitialization",
    "VehicleLightTracksInitialization",
    "AccumulatedPointCloudInitialization",
]
