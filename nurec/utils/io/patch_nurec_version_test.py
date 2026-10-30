# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from nurec.utils.io.patch_nurec_version import patch_gaussian_primitives_node


def _data(model: str) -> dict:
    return {
        "nre_data": {
            "version": "25.7.5",
            "model": model,
            "config": {"name": "gaussians_primitive", "layers": {"background": {}}},
            "state_dict": {".gaussians_nodes.background.x": 1},
        }
    }


class ModelDiscriminatorTest(unittest.TestCase):
    def test_old_and_new_discriminators_are_patched(self) -> None:
        for model in ("nre", "nurec"):
            with self.subTest(model):
                data = _data(model)
                patch_gaussian_primitives_node("0.2.715", data)
                self.assertNotIn("nre_data", data)
                self.assertEqual(list(data["nurec_data"]["config"]["layers"]), ["gaussians"])
                self.assertIn(".gaussians_nodes.gaussians.x", data["nurec_data"]["state_dict"])

    def test_new_envelope_is_patched(self) -> None:
        data = _data("nurec")
        data["nurec_data"] = data.pop("nre_data")
        patch_gaussian_primitives_node("0.2.715", data)
        self.assertEqual(list(data["nurec_data"]["config"]["layers"]), ["gaussians"])

    def test_other_discriminator_is_left_alone(self) -> None:
        data = _data("other")
        patch_gaussian_primitives_node("0.2.715", data)
        self.assertNotIn("nre_data", data)
        self.assertEqual(list(data["nurec_data"]["config"]["layers"]), ["background"])


if __name__ == "__main__":
    unittest.main()
