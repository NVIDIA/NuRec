# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from typing import Any

import torch

from nurec.utils.cuda_rng import CHECKPOINT_STATE_KEY, LEGACY_CHECKPOINT_STATE_KEY, read_checkpoint_state


class ReadCheckpointStateTest(unittest.TestCase):
    STATE = [torch.tensor([1, 2, 3], dtype=torch.uint8)]

    def test_missing_returns_none(self) -> None:
        self.assertIsNone(read_checkpoint_state({"other": 1}))

    def test_new_key(self) -> None:
        self.assertIs(read_checkpoint_state({CHECKPOINT_STATE_KEY: self.STATE}), self.STATE)

    def test_legacy_key(self) -> None:
        self.assertIs(read_checkpoint_state({LEGACY_CHECKPOINT_STATE_KEY: self.STATE}), self.STATE)

    def test_equal_duplicates(self) -> None:
        ckpt = {CHECKPOINT_STATE_KEY: self.STATE, LEGACY_CHECKPOINT_STATE_KEY: [self.STATE[0].clone()]}
        self.assertIs(read_checkpoint_state(ckpt), self.STATE)

    def test_conflicting_duplicates_are_rejected(self) -> None:
        ckpt: dict[str, Any] = {
            CHECKPOINT_STATE_KEY: self.STATE,
            LEGACY_CHECKPOINT_STATE_KEY: [torch.tensor([9], dtype=torch.uint8)],
        }
        with self.assertRaisesRegex(ValueError, "conflicting"):
            read_checkpoint_state(ckpt)
        ckpt = {CHECKPOINT_STATE_KEY: self.STATE, LEGACY_CHECKPOINT_STATE_KEY: self.STATE[0]}
        with self.assertRaisesRegex(ValueError, "conflicting"):
            read_checkpoint_state(ckpt)

    def test_written_key_is_the_new_spelling(self) -> None:
        self.assertEqual(CHECKPOINT_STATE_KEY, "nurec.cuda_rng.singleton_state")
        self.assertEqual(LEGACY_CHECKPOINT_STATE_KEY, "nre.cuda_rng.singleton_state")


if __name__ == "__main__":
    unittest.main()
