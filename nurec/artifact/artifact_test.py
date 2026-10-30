# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Artifact.semantic_class_names and Artifact.semantic_class_colors."""

import hashlib
import io
import json
import tempfile
import unittest
import zipfile

from pathlib import Path

import numpy as np

from nurec.artifact.artifact import SEMANTIC_CLASS_COLORS_FILENAME, SEMANTIC_CLASS_NAMES_FILENAME, Artifact


def _make_usdz(entries: dict[str, bytes]) -> Path:
    """Write an in-memory zip as a temporary .usdz file and return its path."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("metadata.json", json.dumps({"scene_id": "test_scene"}))
        for name, data in entries.items():
            zf.writestr(name, data)
    tmp = tempfile.NamedTemporaryFile(suffix=".usdz", delete=False)
    tmp.write(buf.getvalue())
    tmp.close()
    return Path(tmp.name)


class TestSha256Footprint(unittest.TestCase):
    def test_computes_and_caches_source_digest(self):
        path = _make_usdz({"payload.bin": b"original"})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)

        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(artifact.sha256_footprint, expected)

        path.write_bytes(b"changed")
        self.assertEqual(artifact.sha256_footprint, expected)

        artifact.clear_cache()
        self.assertEqual(artifact.sha256_footprint, hashlib.sha256(b"changed").hexdigest())


class TestCheckpointAccess(unittest.TestCase):
    def test_checkpoint_reads_normal_usdz(self):
        path = _make_usdz({"checkpoint.ckpt": b"checkpoint"})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)

        self.assertFalse(artifact.is_compact)
        self.assertEqual(artifact.checkpoint.read(), b"checkpoint")
        self.assertEqual(artifact.checkpoint.read(), b"checkpoint")

    def test_checkpoint_rejects_compact_usdz(self):
        path = _make_usdz({"compact.safetensors": b"payload"})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)

        self.assertTrue(artifact.is_compact)
        with self.assertRaisesRegex(ValueError, "compact USDZ artifacts do not contain checkpoint.ckpt"):
            artifact.checkpoint

    def test_missing_checkpoint_keeps_zip_error(self):
        path = _make_usdz({})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)

        self.assertFalse(artifact.is_compact)
        with self.assertRaises(KeyError):
            artifact.checkpoint


class TestFiringPatterns(unittest.TestCase):
    """Measured firing patterns ride in the artifact as firing_patterns/<logical id>.npz."""

    def test_absent_gives_an_empty_mapping(self):
        path = _make_usdz({"payload.bin": b"x"})
        self.addCleanup(path.unlink, missing_ok=True)

        self.assertEqual(Artifact(path).firing_patterns, {})

    def test_keyed_by_logical_lidar_id(self):
        # A logical id can carry colons (e.g. "lidar:top:center"), which name a zip entry fine.
        path = _make_usdz({"firing_patterns/lidar_fu.npz": b"one", "firing_patterns/lidar:top:center.npz": b"two"})
        self.addCleanup(path.unlink, missing_ok=True)

        self.assertEqual(Artifact(path).firing_patterns, {"lidar_fu": b"one", "lidar:top:center": b"two"})

    def test_unexpected_entries_are_skipped(self):
        path = _make_usdz({"firing_patterns/nested/deep.npz": b"a", "firing_patterns/notes.txt": b"b"})
        self.addCleanup(path.unlink, missing_ok=True)

        with self.assertLogs(level="WARNING"):
            self.assertEqual(Artifact(path).firing_patterns, {})


class TestSemanticClassNames(unittest.TestCase):
    def test_absent_returns_none(self):
        path = _make_usdz({})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        self.assertIsNone(a.semantic_class_names)

    def test_id_to_name_dict_format(self):
        """Current on-disk format: ``{"<id>": "name", ...}``; sparse ids are preserved."""
        payload = {"0": "bg", "1": "car", "5": "pedestrian"}
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps(payload).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        self.assertEqual(a.semantic_class_names, {0: "bg", 1: "car", 5: "pedestrian"})

    def test_legacy_list_format(self):
        names = ["bg", "car", "pedestrian"]
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps(names).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        self.assertEqual(a.semantic_class_names, {0: "bg", 1: "car", 2: "pedestrian"})

    def test_legacy_dict_class_names_format(self):
        names = ["bg", "car"]
        payload = {"class_names": names, "extra_field": "ignored"}
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps(payload).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        self.assertEqual(a.semantic_class_names, {0: "bg", 1: "car"})

    def test_unrecognized_format_logs_warning_and_returns_none(self):
        """A dict without 'class_names' key should warn and return None."""
        payload = {"labels": ["bg", "car"]}
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps(payload).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        with self.assertLogs("nurec.artifact.artifact", level="WARNING"):
            result = a.semantic_class_names
        self.assertIsNone(result)

    def test_empty_list_logs_warning_and_returns_none(self):
        """An empty top-level list is treated as malformed and warns once."""
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: b"[]"})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        with self.assertLogs("nurec.artifact.artifact", level="WARNING"):
            result = a.semantic_class_names
        self.assertIsNone(result)

    def test_dict_class_names_non_list_logs_warning_and_returns_none(self):
        """{"class_names": <not-a-list>} should warn and return None."""
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps({"class_names": "not_a_list"}).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        with self.assertLogs("nurec.artifact.artifact", level="WARNING"):
            result = a.semantic_class_names
        self.assertIsNone(result)

    def test_dict_class_names_empty_list_logs_warning_and_returns_none(self):
        """{"class_names": []} should warn and return None."""
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps({"class_names": []}).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        with self.assertLogs("nurec.artifact.artifact", level="WARNING"):
            result = a.semantic_class_names
        self.assertIsNone(result)

    def test_id_to_name_dict_non_int_key_logs_warning_and_returns_none(self):
        """An id->name dict with a non-integer key is treated as unrecognized and warns."""
        payload = {"0": "bg", "not_an_int": "car"}
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps(payload).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        with self.assertLogs("nurec.artifact.artifact", level="WARNING"):
            result = a.semantic_class_names
        self.assertIsNone(result)

    def test_id_to_name_dict_sparse_ids(self):
        """Non-contiguous (sparse) ids are preserved as-is."""
        payload = {"0": "bg", "2": "car", "10": "pedestrian"}
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps(payload).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        self.assertEqual(a.semantic_class_names, {0: "bg", 2: "car", 10: "pedestrian"})

    def test_result_is_a_copy(self):
        """Modifying the returned dict must not affect the internal cache."""
        payload = {"0": "bg", "1": "car"}
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps(payload).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        result = a.semantic_class_names
        assert result is not None
        result[99] = "MUTATED"
        self.assertEqual(a.semantic_class_names, {0: "bg", 1: "car"})

    def test_cached_after_first_access(self):
        """The zip is opened at most once; second call returns cached value without re-opening."""
        payload = {"0": "bg"}
        path = _make_usdz({SEMANTIC_CLASS_NAMES_FILENAME: json.dumps(payload).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        first = a.semantic_class_names
        path.unlink()
        second = a.semantic_class_names
        self.assertEqual(first, second)


class TestSemanticClassColors(unittest.TestCase):
    def test_absent_returns_none(self):
        path = _make_usdz({})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        self.assertIsNone(a.semantic_class_colors)

    def test_list_of_rgb_triples(self):
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        path = _make_usdz({SEMANTIC_CLASS_COLORS_FILENAME: json.dumps(colors).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        result = a.semantic_class_colors
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[0], [255, 0, 0])

    def test_dict_colors_format(self):
        colors = [[10, 20, 30]]
        payload = {"colors": colors}
        path = _make_usdz({SEMANTIC_CLASS_COLORS_FILENAME: json.dumps(payload).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        result = a.semantic_class_colors
        self.assertIsNotNone(result)
        assert result is not None
        np.testing.assert_array_equal(result[0], [10, 20, 30])

    def test_invalid_shape_logs_warning_and_returns_none(self):
        """A flat list (wrong shape) should log a warning and return None."""
        colors = [255, 0, 0]  # (3,) instead of (N, 3)
        path = _make_usdz({SEMANTIC_CLASS_COLORS_FILENAME: json.dumps(colors).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        with self.assertLogs("nurec.artifact.artifact", level="WARNING"):
            result = a.semantic_class_colors
        self.assertIsNone(result)

    def test_empty_list_logs_warning_and_returns_none(self):
        """An empty top-level list is treated as malformed and warns once."""
        path = _make_usdz({SEMANTIC_CLASS_COLORS_FILENAME: b"[]"})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        with self.assertLogs("nurec.artifact.artifact", level="WARNING"):
            result = a.semantic_class_colors
        self.assertIsNone(result)

    def test_dict_without_colors_key_logs_warning_and_returns_none(self):
        """A dict missing the 'colors' key should warn and return None."""
        path = _make_usdz({SEMANTIC_CLASS_COLORS_FILENAME: json.dumps({"other": []}).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        with self.assertLogs("nurec.artifact.artifact", level="WARNING"):
            result = a.semantic_class_colors
        self.assertIsNone(result)

    def test_dict_colors_empty_list_logs_warning_and_returns_none(self):
        """{"colors": []} should warn and return None."""
        path = _make_usdz({SEMANTIC_CLASS_COLORS_FILENAME: json.dumps({"colors": []}).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        with self.assertLogs("nurec.artifact.artifact", level="WARNING"):
            result = a.semantic_class_colors
        self.assertIsNone(result)

    def test_result_is_a_copy(self):
        """Mutating the returned array must not affect the internal cache."""
        colors = [[100, 150, 200]]
        path = _make_usdz({SEMANTIC_CLASS_COLORS_FILENAME: json.dumps(colors).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        result = a.semantic_class_colors
        assert result is not None
        result[0, 0] = 0
        second = a.semantic_class_colors
        assert second is not None
        self.assertEqual(int(second[0, 0]), 100)

    def test_cached_after_first_access(self):
        colors = [[100, 150, 200]]
        path = _make_usdz({SEMANTIC_CLASS_COLORS_FILENAME: json.dumps(colors).encode()})
        self.addCleanup(path.unlink, missing_ok=True)
        a = Artifact(path)
        first = a.semantic_class_colors
        path.unlink()
        second = a.semantic_class_colors
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
