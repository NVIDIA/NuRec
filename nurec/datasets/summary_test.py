# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from collections import OrderedDict

import numpy as np
import torch

from nurec.datasets.summary import DataSourceSummary
from nurec.utils.types import FrameConversion, RigTrajectories


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = (x, y, z)
    return matrix


def _state_dict(rig_key: str) -> dict:
    rig = RigTrajectories(
        T_world_base=torch.eye(4, dtype=torch.float64),
        world_to_nurec=FrameConversion(matrix=_translation(1.0, 2.0, 3.0)),
        rig_trajectories=[],
        camera_calibrations=OrderedDict(),
        lidar_calibrations=OrderedDict(),
    ).to_dict()
    rig[rig_key] = rig.pop("world_to_nurec")
    return {
        "n_frames_per_camera": [1],
        "sequence_tracks_all": None,
        "sequence_tracks_dynamic": None,
        "rig_trajectories": rig,
        "xform_matrices": [[0.0, 0.0, 0.0, 1.0]],
        "aabb_blb": [[0.0, 0.0, 0.0]],
        "aabb_trf": [[1.0, 1.0, 1.0]],
    }


class LegacyRigTrajectoriesKeyTest(unittest.TestCase):
    """`datasource_summary.json` written before the rename nests `world_to_nre` inside `rig_trajectories`."""

    def test_old_and_new_key_deserialize(self) -> None:
        for key in ("world_to_nre", "world_to_nurec"):
            with self.subTest(key):
                state = DataSourceSummary.State.from_dict(_state_dict(key))
                assert state.rig_trajectories is not None
                self.assertEqual(state.rig_trajectories.world_to_nurec.matrix[0, 3], 1.0)
                rig = state.to_dict()["rig_trajectories"]
                assert isinstance(rig, dict)
                self.assertIn("world_to_nurec", rig)
                self.assertNotIn("world_to_nre", rig)

    def test_from_json_goes_through_the_same_path(self) -> None:
        import json

        state = DataSourceSummary.State.from_json(json.dumps(_state_dict("world_to_nre")))
        assert state.rig_trajectories is not None
        self.assertEqual(state.rig_trajectories.world_to_nurec.matrix[1, 3], 2.0)

    def test_conflicting_duplicates_are_rejected(self) -> None:
        d = _state_dict("world_to_nre")
        d["rig_trajectories"]["world_to_nurec"] = FrameConversion(matrix=np.eye(4, dtype=np.float32)).to_dict()
        with self.assertRaisesRegex(ValueError, "world_to_nurec"):
            DataSourceSummary.State.from_dict(d)


if __name__ == "__main__":
    unittest.main()
