# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from apps.viewer.abstract_viewer import RasterArtifacts, RenderAction, RenderStateMachine
from apps.viewer.dataset_interface import CameraTrajectoryData, CameraTrajectoryId, ViewerDatasetInterface
from apps.viewer.lightning_viewer import LightningInstantNuRecViewer, LightningSOViewer, ObserverCallback
from apps.viewer.ncore_dataset_interface import ViewerNCOREInterface
from apps.viewer.viewpoint import LookAtPose, Viewpoint


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "CameraTrajectoryId",
    "CameraTrajectoryData",
    "ViewerDatasetInterface",
    "ViewerNCOREInterface",
    "ObserverCallback",
    "RenderAction",
    "RasterArtifacts",
    "RenderStateMachine",
    "LightningSOViewer",
    "LightningInstantNuRecViewer",
    "LookAtPose",
    "Viewpoint",
]
