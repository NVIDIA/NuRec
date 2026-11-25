# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
import tempfile
import threading
import types
import unittest

from unittest import mock


if "slangtorch" not in sys.modules:
    sys.modules["slangtorch"] = types.SimpleNamespace(  # type: ignore[assignment]
        util=types.SimpleNamespace(wrapModule=lambda module: module),
    )

from nurec.libs.slang_gaussians.collector import collector as collector_module
from nurec.libs.slang_gaussians.collector.codegen import CollectorConfiguration, CollectorKernelCode


class TestCollectorKernelCache(unittest.TestCase):
    def setUp(self):
        self._cached_kernels_backup = collector_module._cached_kernels.copy()
        self._prebuilt_configs_loaded_backup = collector_module._prebuilt_configs_loaded
        collector_module._cached_kernels.clear()
        collector_module._prebuilt_configs_loaded = False

    def tearDown(self):
        collector_module._cached_kernels.clear()
        collector_module._cached_kernels.update(self._cached_kernels_backup)
        collector_module._prebuilt_configs_loaded = self._prebuilt_configs_loaded_backup
        collector_module._jit_build_inputs_digest.cache_clear()

    def test_load_prebuilt_configs_raises_for_duplicate_configuration(self):
        configuration = CollectorConfiguration(parameters=("Collector_Copy<3>",))
        collector_module._cached_kernels[configuration] = collector_module.CollectorKernel(
            slang_module=object(), kernels=(object(), object())
        )
        fake_extension = object()

        with tempfile.NamedTemporaryFile("w", suffix=".json") as tf:
            json.dump(
                [
                    {
                        "configuration": list(configuration.parameters),
                        "kernel_name": "collect_parameters_0",
                    }
                ],
                tf,
            )
            tf.flush()
            with mock.patch.object(
                collector_module.slangtorch.util,
                "wrapModule",
                return_value=types.SimpleNamespace(collect_parameters_0=object()),
            ):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    collector_module.load_prebuilt_configs(fake_extension, tf.name)

    def test_get_slang_kernels_compiles_once_for_concurrent_callers(self):
        configuration = CollectorConfiguration(parameters=("Collector_Copy<3>",))
        kernel_object = object()
        compile_calls = 0
        compile_gate = threading.Event()
        compile_started = threading.Event()
        results = []

        def fake_generate(configurations):
            self.assertEqual(configurations, [configuration])
            return CollectorKernelCode(code="// fake slang", kernel_names=("collect_parameters_0",))

        def fake_load_module(*args, **kwargs):
            nonlocal compile_calls
            compile_calls += 1
            compile_started.set()
            compile_gate.wait(timeout=5)
            return types.SimpleNamespace(
                collect_parameters_0=kernel_object, collect_parameters_0_segmented=kernel_object
            )

        def worker():
            results.append(collector_module.get_slang_kernels([configuration]))

        with (
            mock.patch.object(collector_module, "_load_prebuilt_configs_if_not_loaded", return_value=None),
            mock.patch.object(collector_module, "generate_collector_code", side_effect=fake_generate),
            mock.patch.object(collector_module, "add_ninja_to_path", return_value=None),
            mock.patch.object(collector_module, "get_slang_module_path", return_value="/tmp/collector.slang-module"),
            mock.patch.object(collector_module, "_jit_build_inputs_digest", return_value="0" * 64),
            mock.patch.object(collector_module.slangtorch, "loadModule", side_effect=fake_load_module, create=True),
        ):
            first = threading.Thread(target=worker)
            second = threading.Thread(target=worker)
            first.start()
            self.assertTrue(compile_started.wait(timeout=5))
            second.start()
            compile_gate.set()
            first.join(timeout=5)
            second.join(timeout=5)

        self.assertEqual(compile_calls, 1)
        self.assertEqual(len(results), 2)
        self.assertIs(results[0][0], results[1][0])

    def test_generated_file_name_tracks_slang_module_contents(self):
        """Identical generated code gets a new file when the pre-built collector module differs."""
        configuration = CollectorConfiguration(parameters=("Collector_Copy<3>",))
        kernel_code = CollectorKernelCode(code="// fake slang", kernel_names=("collect_parameters_0",))
        loaded_names: list[str] = []

        def fake_load_module(path, *args, **kwargs):
            loaded_names.append(os.path.basename(path))
            return types.SimpleNamespace(collect_parameters_0=object(), collect_parameters_0_segmented=object())

        def generated_file_name(module_bytes: bytes) -> str:
            collector_module._cached_kernels.clear()
            collector_module._jit_build_inputs_digest.cache_clear()
            with (
                tempfile.NamedTemporaryFile(suffix=".slang-module") as module_file,
                tempfile.TemporaryDirectory() as cache_dir,
                mock.patch.object(collector_module, "_load_prebuilt_configs_if_not_loaded", return_value=None),
                mock.patch.object(collector_module, "generate_collector_code", return_value=kernel_code),
                mock.patch.object(collector_module, "add_ninja_to_path", return_value=None),
                mock.patch.object(collector_module, "get_slang_module_path", return_value=module_file.name),
                mock.patch.object(collector_module, "get_jit_cache_dir", return_value=cache_dir),
                mock.patch.object(collector_module.slangtorch, "loadModule", side_effect=fake_load_module, create=True),
            ):
                module_file.write(module_bytes)
                module_file.flush()
                collector_module.get_slang_kernels([configuration])
            return loaded_names.pop()

        first = generated_file_name(b"collector module A")
        self.assertRegex(first, r"^collector_[0-9a-f]{8}\.slang$")
        self.assertEqual(generated_file_name(b"collector module A"), first)
        self.assertNotEqual(generated_file_name(b"collector module B"), first)


if __name__ == "__main__":
    unittest.main()
