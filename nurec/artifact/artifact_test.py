# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Artifact.semantic_class_names and Artifact.semantic_class_colors."""

import hashlib
import io
import json
import os
import tempfile
import time
import unittest
import zipfile

from pathlib import Path
from unittest import mock

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


def _settled_footprint(artifact: Artifact) -> str:
    """Hash as if the file had been untouched for a minute, so the digest is trusted on its stat.

    ctime cannot be back-dated the way mtime can, so the racy-clean window is cleared by hashing
    "in the future" rather than by ageing the file.
    """
    with mock.patch("time.time", return_value=time.time() + 60.0):
        return artifact.sha256_footprint


def _rewrite_in_a_later_clock_tick(path: Path, data: bytes) -> None:
    """Write ``data`` in place so that the file's timestamps differ from before.

    File timestamps come from the kernel's coarse clock, so a rewrite within the same tick as the
    previous write leaves mtime and ctime bit-identical; retry until the tick has moved.
    """
    ctime_ns = path.stat().st_ctime_ns
    while True:
        path.write_bytes(data)
        if path.stat().st_ctime_ns != ctime_ns:
            return
        time.sleep(0.005)


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

    def test_footprint_is_current_needs_a_cached_digest(self):
        path = _make_usdz({"payload.bin": b"x"})
        self.addCleanup(path.unlink, missing_ok=True)

        self.assertFalse(Artifact(path).footprint_is_current())

    def test_footprint_is_current_for_an_untouched_file(self):
        path = _make_usdz({"payload.bin": b"x"})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)
        _settled_footprint(artifact)

        self.assertTrue(artifact.footprint_is_current())
        artifact.clear_cache()
        self.assertFalse(artifact.footprint_is_current())

    def test_footprint_is_racy_right_after_the_file_was_written(self):
        # Written and hashed within the racy window: timestamp granularity could hide a rewrite in between.
        path = _make_usdz({"payload.bin": b"x"})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)
        artifact.sha256_footprint

        self.assertFalse(artifact.footprint_is_current())

    def test_footprint_is_stale_after_a_same_size_rewrite(self):
        path = _make_usdz({"payload.bin": b"x"})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)
        digest = _settled_footprint(artifact)

        _rewrite_in_a_later_clock_tick(path, path.read_bytes()[:-1] + b"?")  # same size, other content

        self.assertFalse(artifact.footprint_is_current())
        self.assertNotEqual(Artifact(path).sha256_footprint, digest)

    def test_footprint_is_stale_after_a_replacement_that_preserves_size_and_mtime(self):
        # `cp -p` / `rsync -t` / rename into place: same size and mtime, but a new inode and ctime.
        path = _make_usdz({"payload.bin": b"x"})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)
        digest = _settled_footprint(artifact)
        before = path.stat()

        replacement = Path(tempfile.NamedTemporaryFile(dir=path.parent, delete=False).name)
        replacement.write_bytes(path.read_bytes()[:-1] + b"?")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, path)

        after = path.stat()
        self.assertEqual((after.st_size, after.st_mtime_ns), (before.st_size, before.st_mtime_ns))
        self.assertFalse(artifact.footprint_is_current())
        self.assertNotEqual(Artifact(path).sha256_footprint, digest)

    def test_footprint_is_stale_after_an_in_place_rewrite_with_restored_mtime(self):
        # Same inode, size and mtime (`touch -r` after the write): only ctime tells.
        path = _make_usdz({"payload.bin": b"x"})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)
        _settled_footprint(artifact)
        before = path.stat()

        _rewrite_in_a_later_clock_tick(path, path.read_bytes()[:-1] + b"?")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

        after = path.stat()
        self.assertEqual(
            (after.st_ino, after.st_size, after.st_mtime_ns), (before.st_ino, before.st_size, before.st_mtime_ns)
        )
        self.assertFalse(artifact.footprint_is_current())

    def test_footprint_is_stale_when_the_file_is_gone(self):
        path = _make_usdz({"payload.bin": b"x"})
        artifact = Artifact(path)
        _settled_footprint(artifact)

        path.unlink()

        self.assertFalse(artifact.footprint_is_current())


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

    def test_open_checkpoint_reads_a_stored_member_in_place(self):
        payload = bytes(range(256)) * 4096 + b"tail"
        path = _make_usdz({"checkpoint.ckpt": payload, "after.bin": b"z" * 100})
        self.addCleanup(path.unlink, missing_ok=True)
        artifact = Artifact(path)

        with artifact.open_checkpoint() as checkpoint:
            self.assertEqual(checkpoint.seek(0, io.SEEK_END), len(payload))
            self.assertEqual(checkpoint.tell(), len(payload))
            checkpoint.seek(0)
            self.assertEqual(checkpoint.read(), payload)
            checkpoint.seek(1000)
            buffer = bytearray(5000)
            self.assertEqual(checkpoint.readinto(buffer), 5000)
            self.assertEqual(bytes(buffer), payload[1000:6000])
            checkpoint.seek(-4, io.SEEK_END)
            self.assertEqual(checkpoint.read(), b"tail")
            self.assertEqual(checkpoint.read(1), b"")
        self.assertTrue(checkpoint.closed)
        self.assertEqual(artifact.checkpoint.read(), payload)

    def test_open_checkpoint_rejects_a_corrupted_stored_member(self):
        # One flipped payload byte; sizes, headers and the recorded CRC are untouched. The clean
        # member of the test above passes the same check.
        payload = bytes(range(256)) * 4096
        path = _make_usdz({"checkpoint.ckpt": payload})
        self.addCleanup(path.unlink, missing_ok=True)
        data = bytearray(path.read_bytes())
        start = data.index(payload)
        data[start + 12345] ^= 0xFF
        path.write_bytes(data)

        with self.assertRaisesRegex(zipfile.BadZipFile, "Bad CRC-32 for file 'checkpoint.ckpt'"):
            Artifact(path).open_checkpoint()
        with Artifact(path).open_checkpoint(verify_crc=False) as checkpoint:
            self.assertEqual(checkpoint.read(), bytes(data[start : start + len(payload)]))

    def test_open_checkpoint_falls_back_to_a_copy_for_a_compressed_member(self):
        payload = b"compressible " * 1000
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("checkpoint.ckpt", payload)
        tmp = tempfile.NamedTemporaryFile(suffix=".usdz", delete=False)
        tmp.write(buf.getvalue())
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink, missing_ok=True)

        with Artifact(path).open_checkpoint() as checkpoint:
            self.assertEqual(checkpoint.read(), payload)

    def test_open_checkpoint_rejects_compact_usdz(self):
        path = _make_usdz({"compact.safetensors": b"payload"})
        self.addCleanup(path.unlink, missing_ok=True)

        with self.assertRaisesRegex(ValueError, "compact USDZ artifacts do not contain checkpoint.ckpt"):
            Artifact(path).open_checkpoint()

    def test_open_checkpoint_keeps_zip_error_when_missing(self):
        path = _make_usdz({})
        self.addCleanup(path.unlink, missing_ok=True)

        with self.assertRaises(KeyError):
            Artifact(path).open_checkpoint()


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
