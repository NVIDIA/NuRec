# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from nurec.utils.renamed_keys import adopt_renamed_key, normalize_renamed_key, normalize_renamed_value


class NormalizeRenamedKeyTest(unittest.TestCase):
    def test_old_key_is_moved_to_new_key(self) -> None:
        out = normalize_renamed_key({"nre_track_id": 3, "x": 1}, "nurec_track_id", "nre_track_id")
        self.assertEqual(out, {"nurec_track_id": 3, "x": 1})

    def test_new_key_passes_through(self) -> None:
        out = normalize_renamed_key({"nurec_track_id": 3}, "nurec_track_id", "nre_track_id")
        self.assertEqual(out, {"nurec_track_id": 3})

    def test_equal_duplicates_collapse(self) -> None:
        out = normalize_renamed_key({"nurec_track_id": 3, "nre_track_id": 3}, "nurec_track_id", "nre_track_id")
        self.assertEqual(out, {"nurec_track_id": 3})

    def test_conflicting_duplicates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting values for 'nurec_track_id'"):
            normalize_renamed_key({"nurec_track_id": 3, "nre_track_id": 4}, "nurec_track_id", "nre_track_id")

    def test_custom_equality(self) -> None:
        out = normalize_renamed_key(
            {"a": [1.0], "b": [1.0 + 1e-12]}, "a", "b", equal=lambda x, y: abs(x[0] - y[0]) < 1e-6
        )
        self.assertEqual(out, {"a": [1.0]})

    def test_input_is_not_mutated(self) -> None:
        src = {"nre_track_id": 3}
        normalize_renamed_key(src, "nurec_track_id", "nre_track_id")
        self.assertEqual(src, {"nre_track_id": 3})


class AdoptRenamedKeyTest(unittest.TestCase):
    def test_old_envelope_is_renamed_in_place(self) -> None:
        src = {"nre_data": {"version": "25.7.5"}, "type": "model"}
        adopt_renamed_key(src, "nurec_data", "nre_data")
        self.assertEqual(src, {"nurec_data": {"version": "25.7.5"}, "type": "model"})

    def test_new_envelope_passes_through(self) -> None:
        src = {"nurec_data": {"model": "nurec"}}
        adopt_renamed_key(src, "nurec_data", "nre_data")
        self.assertEqual(src, {"nurec_data": {"model": "nurec"}})


class NormalizeRenamedValueTest(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(normalize_renamed_value("nre", "nurec", "nre"), "nurec")
        self.assertEqual(normalize_renamed_value("nurec", "nurec", "nre"), "nurec")
        self.assertEqual(normalize_renamed_value("other", "nurec", "nre"), "other")


if __name__ == "__main__":
    unittest.main()
