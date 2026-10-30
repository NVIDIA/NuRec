# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from omegaconf import DictConfig

from nurec.models.base import BaseModel


class MockModel(BaseModel):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)


class TestBaseModel(unittest.TestCase):
    def test_base_class_instantiation(self):
        model = MockModel(config=DictConfig({}))
        self.assertIsInstance(model, BaseModel)
