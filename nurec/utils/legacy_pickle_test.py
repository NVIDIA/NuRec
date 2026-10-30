# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import pickle
import tempfile
import unittest

from unittest import mock

import torch

from nurec.utils.legacy_pickle import (
    LEGACY_PICKLE_MODULE,
    LegacyModuleUnpickler,
    install_torch_load_module_remap,
    remap_module,
)
from nurec.utils.renamed_keys import normalize_renamed_value


_NEW = b"nurec.utils.renamed_keys"
_OLD = b"nre.utils.renamed_keys"


def _legacy_checkpoint_bytes() -> bytes:
    """A torch checkpoint whose pickled globals still name the `nre` package.

    Protocol 2 writes GLOBAL as text, and the legacy (non-zip) container has no
    checksum, so the module name can be rewritten byte for byte.
    """
    buf = io.BytesIO()
    torch.save(
        {"fn": normalize_renamed_value, "t": torch.tensor([1.0, 2.0])}, buf, _use_new_zipfile_serialization=False
    )
    data = buf.getvalue()
    assert _NEW in data
    return data.replace(_NEW, _OLD)


class RemapModuleTest(unittest.TestCase):
    def test_prefix_is_rewritten(self) -> None:
        self.assertEqual(remap_module("nre"), "nurec")
        self.assertEqual(remap_module("nre.config.model"), "nurec.config.model")

    def test_other_modules_pass_through(self) -> None:
        for module in ("nurec.config.model", "nrend.x", "torch", "mynre.x", "nre_ext"):
            with self.subTest(module=module):
                self.assertEqual(remap_module(module), module)


class LegacyModuleUnpicklerTest(unittest.TestCase):
    def test_nre_global_resolves_to_nurec(self) -> None:
        data = pickle.dumps(normalize_renamed_value, protocol=2).replace(_NEW, _OLD)
        with self.assertRaises(ModuleNotFoundError):
            pickle.loads(data)
        self.assertIs(LegacyModuleUnpickler(io.BytesIO(data)).load(), normalize_renamed_value)
        self.assertIs(LEGACY_PICKLE_MODULE.loads(data), normalize_renamed_value)


class TorchLoadRemapTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_load = torch.load
        self.addCleanup(setattr, torch, "load", self._original_load)
        install_torch_load_module_remap()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False)
        self.addCleanup(os.unlink, self.tmp.name)
        self.tmp.write(_legacy_checkpoint_bytes())
        self.tmp.close()

    def test_install_is_idempotent(self) -> None:
        installed = torch.load
        install_torch_load_module_remap()
        self.assertIs(torch.load, installed)

    def test_default_full_unpickle_remaps(self) -> None:
        with mock.patch.dict(os.environ, {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}, clear=False):
            os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
            loaded = torch.load(self.tmp.name)
        self.assertIs(loaded["fn"], normalize_renamed_value)
        self.assertTrue(torch.equal(loaded["t"], torch.tensor([1.0, 2.0])))

    def test_explicit_weights_only_false_remaps(self) -> None:
        loaded = torch.load(self.tmp.name, weights_only=False)
        self.assertIs(loaded["fn"], normalize_renamed_value)

    def test_explicit_pickle_module_keeps_caller_policy(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            torch.load(self.tmp.name, pickle_module=pickle, weights_only=False)

    def test_weights_only_true_is_untouched(self) -> None:
        with self.assertRaises(pickle.UnpicklingError):
            torch.load(self.tmp.name, weights_only=True)

    def test_forced_weights_only_env_is_untouched(self) -> None:
        with mock.patch.dict(os.environ, {"TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}, clear=False):
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(self.tmp.name, weights_only=False)


if __name__ == "__main__":
    unittest.main()
