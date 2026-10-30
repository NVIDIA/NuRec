# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import shutil
import tempfile
import threading
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from omegaconf import OmegaConf

import nurec.harmonizer.model

from nurec.config.harmonizer import TrainingHarmonizerConfig
from nurec.harmonizer.model import (
    BaseHarmonizerModel,
    HarmonizerModelFactory,
    HarmonizerSessionClosedError,
    HarmonizerState,
    NonTemporalHarmonizerModelTraced,
    TemporalHarmonizerModelCheckpointed,
    TemporalHarmonizerModelTraced,
)
from nurec.harmonizer.training_controller import TrainingHarmonizerController
from nurec.systems.utils import system_config_compatibility_check
from nurec.utils.model_registry import create_model_registry


def test_transformer_engine_attention_imports_with_pinned_torch():
    """Check the DotProductAttention class to confirm the attention module imports."""
    attention = importlib.import_module("transformer_engine.pytorch.attention")
    assert attention.DotProductAttention is not None


class _CallableStubModule(torch.nn.Module):
    """Stand-in for a loaded model: callable and delegates to `fn`, with a buffer so
    `BaseHarmonizerModel._model_device()` resolves a real (CPU) device and `.to()` works
    (the non-temporal forward aligns the model to the input device on first use)."""

    def __init__(self, fn):
        super().__init__()
        self.register_buffer("_device_anchor", torch.zeros(1))
        self._fn = fn

    def forward(self, x):
        return self._fn(x)


class TestNonTemporalHarmonizerModelTraced(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory that will be writable in Bazel test environment
        self.cache_dir = tempfile.mkdtemp()
        # Fixture URL; create_model_registry is patched, so nothing is fetched. The .tar.gz
        # basename is required by test_download_checkpoint_tar_gz.
        self.model_url = "https://example.com/harmonizer.tar.gz"
        self.batch_size = 1
        self.ckpt_name = "harmonizer.pt"
        self.default_cache_dir_child = ".cache/nurec/harmonizer"
        self.model_resolution = (544, 960)

    def tearDown(self):
        # Clean up the temporary directory after each test
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_init(self):
        self.maxDiff = None
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        self.assertEqual(harmonizer_model.model_url, self.model_url)
        self.assertIsNone(harmonizer_model.model)
        self.assertEqual(harmonizer_model.cache_dir, self.cache_dir)
        self.assertEqual(harmonizer_model.model_filename, self.ckpt_name)
        self.assertEqual(type(harmonizer_model._lock), type(threading.Lock()))

    def test_create_model_registry_stays_patchable(self):
        """``model.py`` must bind ``create_model_registry`` as a module-level name for patches."""
        self.assertIs(nurec.harmonizer.model.create_model_registry, create_model_registry)
        with patch("nurec.harmonizer.model.create_model_registry") as patched:
            self.assertIs(nurec.harmonizer.model.create_model_registry, patched)

    @patch("nurec.harmonizer.model.create_model_registry")
    @patch("os.path.exists")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch("shutil.move")
    def test_download_checkpoint_tar_gz(self, mock_move, mock_remove, mock_tarfile, mock_exists, mock_create_registry):
        """Test download_checkpoint method with a tar.gz file"""
        # Set up mocks
        mock_registry = MagicMock()
        tarball_path = os.path.join(self.cache_dir, "harmonizer.tar.gz")
        mock_registry.get_model.return_value = tarball_path
        mock_create_registry.return_value = mock_registry

        # Mock .tar.gz handling - provide values for calls to os.path.exists
        # 1. Initial check if cached_checkpoint_file exists - False means we need to extract
        # 2. Check if downloaded_file exists - True to pass the assertion
        # 3. Check if cached_checkpoint_file exists after extraction - True means extraction worked
        mock_exists.side_effect = lambda path: (
            path == tarball_path or path == os.path.join(self.cache_dir, self.ckpt_name) and mock_move.called
        )

        # Create mock tar file with a single .pt file member
        mock_tar_instance = MagicMock()
        mock_member = MagicMock()
        mock_member.name = "harmonizer.pt"
        mock_tar_instance.getmembers.return_value = [mock_member]
        mock_tarfile.return_value.__enter__.return_value = mock_tar_instance

        # Create instance and call method
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        result = harmonizer_model._download_checkpoint(self.model_url, self.cache_dir, self.ckpt_name)

        # Checks
        mock_create_registry.assert_called_once_with(self.model_url, Path(self.cache_dir))
        mock_registry.get_model.assert_called_once()
        mock_tarfile.assert_called_once_with(tarball_path, "r:gz")
        mock_tar_instance.extract.assert_called_once_with(mock_member, path=self.cache_dir)
        mock_move.assert_called_once_with(
            os.path.join(self.cache_dir, mock_member.name), os.path.join(self.cache_dir, self.ckpt_name)
        )
        mock_remove.assert_called_once_with(tarball_path)
        self.assertEqual(result, os.path.join(self.cache_dir, self.ckpt_name))

    @patch("nurec.harmonizer.model.create_model_registry")
    @patch("os.path.exists")
    @patch("shutil.move")
    def test_download_checkpoint_checkpoint_exists(self, mock_move, mock_exists, mock_create_registry):
        """Test download_checkpoint method when checkpoint already exists"""
        # Mock checkpoint already exists
        mock_exists.return_value = True

        # Create instance and call method
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        result = harmonizer_model._download_checkpoint(self.model_url, self.cache_dir, self.ckpt_name)

        # Checks
        mock_create_registry.assert_not_called()
        mock_move.assert_not_called()
        self.assertEqual(result, os.path.join(self.cache_dir, self.ckpt_name))

    @patch("nurec.harmonizer.model.create_model_registry")
    @patch("os.path.exists")
    @patch("shutil.move")
    def test_download_checkpoint_non_tar_file(self, mock_move, mock_exists, mock_create_registry):
        """Test download_checkpoint method with a non-tar file (e.g. direct checkpoint)"""
        # Set up mocks
        mock_registry = MagicMock()
        checkpoint_path = os.path.join(self.cache_dir, "some_direct_checkpoint.pt")
        mock_registry.get_model.return_value = checkpoint_path
        mock_create_registry.return_value = mock_registry

        # Mock the file existence check - first False for cached_checkpoint_file check,
        # then True for downloaded_file existence check
        mock_exists.side_effect = [False, True, True]

        # Create instance and call method
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        result = harmonizer_model._download_checkpoint(self.model_url, self.cache_dir, self.ckpt_name)

        # Checks
        mock_create_registry.assert_called_once_with(self.model_url, Path(self.cache_dir))
        mock_registry.get_model.assert_called_once()
        mock_move.assert_called_once_with(checkpoint_path, os.path.join(self.cache_dir, self.ckpt_name))
        self.assertEqual(result, os.path.join(self.cache_dir, self.ckpt_name))

    @patch("nurec.harmonizer.model.create_model_registry")
    @patch("os.path.exists")
    @patch("shutil.move")
    def test_download_checkpoint_non_tar_file_already_at_target(self, mock_move, mock_exists, mock_create_registry):
        """Test direct checkpoint downloads that are already cached at the requested filename."""
        mock_registry = MagicMock()
        checkpoint_path = os.path.join(self.cache_dir, self.ckpt_name)
        mock_registry.get_model.return_value = checkpoint_path
        mock_create_registry.return_value = mock_registry
        mock_exists.side_effect = [False, True]

        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        result = harmonizer_model._download_checkpoint(self.model_url, self.cache_dir, self.ckpt_name)

        mock_create_registry.assert_called_once_with(self.model_url, Path(self.cache_dir))
        mock_registry.get_model.assert_called_once()
        mock_move.assert_not_called()
        self.assertEqual(result, checkpoint_path)

    @patch("torch.jit.load")
    @patch.object(BaseHarmonizerModel, "_download_checkpoint")
    def test_load_pretrained_model(self, mock_download_checkpoint, mock_torch_load):
        """Test the _load_pretrained_model method"""
        # Set up mocks
        mock_model = MagicMock()
        mock_torch_load.return_value = mock_model
        checkpoint_path = os.path.join(self.cache_dir, self.ckpt_name)
        mock_download_checkpoint.return_value = checkpoint_path

        # Create instance and call method
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        result = harmonizer_model._load_pretrained_model()

        # Checks
        mock_download_checkpoint.assert_called_once_with(self.model_url, self.cache_dir, self.ckpt_name)
        mock_torch_load.assert_called_once_with(checkpoint_path)
        self.assertEqual(result, mock_model.cuda.return_value)

    def _make_cpu_dummy(self):
        """Create a CPU dummy tensor matching model_resolution, bypassing CUDA requirement."""
        return torch.zeros(1, 3, *self.model_resolution)

    def test_detect_needs_temporal_dim_false(self):
        """Model that accepts 4-D input sets needs_temporal_dim=False."""
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        mock_model = MagicMock()
        mock_model.side_effect = lambda x: x
        harmonizer_model.model = mock_model

        with patch("nurec.harmonizer.model.torch.zeros", return_value=self._make_cpu_dummy()):
            harmonizer_model._detect_needs_temporal_dim()

        self.assertFalse(harmonizer_model.needs_temporal_dim)
        mock_model.assert_called_once()

    def test_detect_needs_temporal_dim_true(self):
        """Model that rejects 4-D but accepts 5-D input sets needs_temporal_dim=True."""
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        mock_model = MagicMock()

        def accept_only_5d(x):
            if x.dim() == 4:
                raise RuntimeError("expected 5-D input")
            return x

        mock_model.side_effect = accept_only_5d
        harmonizer_model.model = mock_model

        with patch("nurec.harmonizer.model.torch.zeros", return_value=self._make_cpu_dummy()):
            harmonizer_model._detect_needs_temporal_dim()

        self.assertTrue(harmonizer_model.needs_temporal_dim)
        self.assertEqual(mock_model.call_count, 2)

    def test_detect_needs_temporal_dim_true_expected_5d_no_hyphen(self):
        """Covers the 'expected 5d' token (no-hyphen variant of 5-D-input errors)."""
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        mock_model = MagicMock()

        def accept_only_5d(x):
            if x.dim() == 4:
                raise RuntimeError("expected 5D input, got 4D")
            return x

        mock_model.side_effect = accept_only_5d
        harmonizer_model.model = mock_model

        with patch("nurec.harmonizer.model.torch.zeros", return_value=self._make_cpu_dummy()):
            harmonizer_model._detect_needs_temporal_dim()

        self.assertTrue(harmonizer_model.needs_temporal_dim)
        self.assertEqual(mock_model.call_count, 2)

    def test_detect_needs_temporal_dim_true_dim_out_of_range(self):
        """Regression: 'Dimension out of range' error must trigger 5-D fallback via 'dimension' token."""
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        mock_model = MagicMock()

        def accept_only_5d(x):
            if x.dim() == 4:
                raise RuntimeError("Dimension out of range (expected to be in range of [-4, 3], but got 4)")
            return x

        mock_model.side_effect = accept_only_5d
        harmonizer_model.model = mock_model

        with patch("nurec.harmonizer.model.torch.zeros", return_value=self._make_cpu_dummy()):
            harmonizer_model._detect_needs_temporal_dim()

        self.assertTrue(harmonizer_model.needs_temporal_dim)
        self.assertEqual(mock_model.call_count, 2)

    def test_detect_needs_temporal_dim_both_fail(self):
        """If both 4-D and 5-D inputs fail, the error from the 5-D attempt propagates."""
        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("unsupported model")
        harmonizer_model.model = mock_model

        with patch("nurec.harmonizer.model.torch.zeros", return_value=self._make_cpu_dummy()):
            with self.assertRaises(RuntimeError):
                harmonizer_model._detect_needs_temporal_dim()

    def test_detect_needs_temporal_dim_buffer_only_model(self):
        """Regression for NVBug 6237195: a scripted module that stores its weights as buffers
        rather than nn.Parameters (e.g. some torch.jit.load checkpoints like cosmos_3dgut.pt)
        must still resolve a device for the dummy probe — previously raised StopIteration from
        next(model.parameters())."""

        class BufferOnlyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.zeros(3))

            def forward(self, x):
                return x

        real_model = torch.jit.script(BufferOnlyModule())

        harmonizer_model = NonTemporalHarmonizerModelTraced(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        harmonizer_model.model = real_model

        with patch("nurec.harmonizer.model.torch.zeros", return_value=self._make_cpu_dummy()):
            harmonizer_model._detect_needs_temporal_dim()

        self.assertFalse(harmonizer_model.needs_temporal_dim)

    def test_init_rejects_invalid_model_resolution(self):
        """model_resolution must be a (int, int) tuple (validated in BaseHarmonizerModel.__init__)."""
        for bad in [(8,), (8, 8, 8), (8.0, 8), [8, 8]]:
            with self.assertRaises(ValueError):
                NonTemporalHarmonizerModelTraced(self.model_url, self.cache_dir, self.ckpt_name, bad)

    def _forward_model(self, stub, needs_temporal_dim, model_resolution=(8, 8)):
        """A NonTemporalHarmonizerModelTraced with a stub JIT model injected (no checkpoint / CUDA)."""
        model = NonTemporalHarmonizerModelTraced(self.model_url, self.cache_dir, self.ckpt_name, model_resolution)
        model.model = _CallableStubModule(stub)
        model.needs_temporal_dim = needs_temporal_dim
        return model

    def test_forward_raises_when_model_not_loaded(self):
        model = NonTemporalHarmonizerModelTraced(self.model_url, self.cache_dir, self.ckpt_name, (8, 8))
        model.model = None
        with self.assertRaises(ValueError):
            model.forward(torch.rand(8 * 8, 3), torch.Size((8, 8)), False)

    def test_forward_roundtrip_4d_model(self):
        # needs_temporal_dim=False: a 4-D identity model + matched resolution round-trips the input.
        torch.manual_seed(0)
        model = self._forward_model(lambda x: x, needs_temporal_dim=False)
        inp = torch.rand(8 * 8, 3)
        out = model.forward(inp, torch.Size((8, 8)), False)
        self.assertEqual(tuple(out.shape), (8 * 8, 3))
        self.assertTrue(torch.allclose(out, inp, atol=1e-3))

    def test_forward_adds_and_strips_temporal_dim_for_5d_model(self):
        # needs_temporal_dim=True: forward unsqueezes a temporal axis before the model and
        # squeezes it after. The stub records that it was called with 5-D input.
        torch.manual_seed(0)
        seen_dims = []

        def stub(x):
            seen_dims.append(x.dim())
            return x

        model = self._forward_model(stub, needs_temporal_dim=True)
        inp = torch.rand(8 * 8, 3)
        out = model.forward(inp, torch.Size((8, 8)), False)
        self.assertEqual(seen_dims, [5])  # temporal axis was added before the model call
        self.assertEqual(tuple(out.shape), (8 * 8, 3))
        self.assertTrue(torch.allclose(out, inp, atol=1e-3))


class TestHarmonizerState(unittest.TestCase):
    """Pure-tensor tests for the per-session frame history (no model / CUDA needed)."""

    C, H, W = 3, 2, 2

    def _frame(self, val):
        """A (C, H, W) frame filled with a constant, so we can identify it by value."""
        return torch.full((self.C, self.H, self.W), float(val))

    def _current(self, val):
        """A (1, C, H, W) current frame (the shape temporal input builders expect)."""
        return torch.full((1, self.C, self.H, self.W), float(val))

    def _slot_values(self, stacked):
        """Per-T-slot scalar of a (1, C, T, H, W) stack of constant frames."""
        return [stacked[0, 0, t, 0, 0].item() for t in range(stacked.shape[2])]

    def test_push_evicts_oldest_at_capacity(self):
        state = HarmonizerState(max_history=4)
        for v in range(6):  # push 0,1,2,3,4,5
            state.push(self._frame(v))
        self.assertEqual(len(state.history), 4)
        # FIFO: the four most-recent survive, oldest-first.
        self.assertEqual([f[0, 0, 0].item() for f in state.history], [2, 3, 4, 5])

    def test_push_detaches_frame(self):
        state = HarmonizerState()
        state.push(torch.full((self.C, self.H, self.W), 1.0, requires_grad=True))
        self.assertFalse(state.history[-1].requires_grad)

    def test_push_clones_view_to_avoid_retaining_full_temporal_output(self):
        state = HarmonizerState()
        temporal_length = 5
        full_output = torch.rand((1, self.C, temporal_length, self.H, self.W))
        frame_view = full_output[:, :, 0, :, :].squeeze(0)

        state.push(frame_view)

        stored_frame = state.history[-1]
        self.assertTrue(torch.equal(stored_frame, frame_view))
        self.assertNotEqual(stored_frame.untyped_storage().data_ptr(), full_output.untyped_storage().data_ptr())
        self.assertEqual(
            stored_frame.untyped_storage().nbytes(),
            stored_frame.numel() * stored_frame.element_size(),
        )
        self.assertEqual(
            full_output.untyped_storage().nbytes(),
            temporal_length * stored_frame.untyped_storage().nbytes(),
        )

    def test_build_fixed_length_temporal_input_ordering_full_history(self):
        # Outputs pushed in chronological order o1..o4; the stack must read
        # [current, t-1, t-2, t-3, t-4] = [current, o4, o3, o2, o1].
        state = HarmonizerState(max_history=4)
        for v in (1, 2, 3, 4):
            state.push(self._frame(v))
        out = state.build_fixed_length_temporal_input(self._current(0))
        self.assertEqual(tuple(out.shape), (1, self.C, 5, self.H, self.W))
        self.assertEqual(self._slot_values(out), [0, 4, 3, 2, 1])

    def test_build_fixed_length_temporal_input_cold_start_pads_with_current(self):
        # Two real outputs (o1=1, o2=2); the two oldest slots are padded with current.
        state = HarmonizerState(max_history=4)
        state.push(self._frame(1))
        state.push(self._frame(2))
        out = state.build_fixed_length_temporal_input(self._current(0))
        self.assertEqual(self._slot_values(out), [0, 2, 1, 0, 0])

    def test_build_fixed_length_temporal_input_empty_history_all_current(self):
        state = HarmonizerState(max_history=4)
        out = state.build_fixed_length_temporal_input(self._current(7))
        self.assertEqual(self._slot_values(out), [7, 7, 7, 7, 7])

    def test_build_fixed_length_temporal_input_shape_fixed_regardless_of_history(self):
        # T is always 1 + max_history, for any amount of accumulated history.
        for n in range(5):  # 0..4 prior frames
            state = HarmonizerState(max_history=4)
            for v in range(n):
                state.push(self._frame(v + 1))
            out = state.build_fixed_length_temporal_input(self._current(0))
            self.assertEqual(out.shape[2], 5, f"T should be 5 with {n} history frames")

    def test_close_marks_closed_and_clears_history(self):
        state = HarmonizerState(max_history=4)
        state.push(self._frame(1))
        state.close()
        self.assertTrue(state.closed)
        self.assertEqual(len(state.history), 0)

    def test_reset_clears_history_without_ending_session(self):
        state = HarmonizerState(max_history=4)
        state.push(self._frame(1))

        state.reset()

        self.assertFalse(state.closed)
        self.assertEqual(len(state.history), 0)

    def test_build_variable_length_temporal_input_cold_start_uses_current_only(self):
        state = HarmonizerState(max_history=4)
        state.push(self._frame(1))
        state.push(self._frame(2))
        out = state.build_variable_length_temporal_input(self._current(0))
        self.assertEqual(tuple(out.shape), (1, self.C, 1, self.H, self.W))
        self.assertEqual(self._slot_values(out), [0])

    def test_build_variable_length_temporal_input_full_history(self):
        state = HarmonizerState(max_history=4)
        for v in (1, 2, 3, 4):
            state.push(self._frame(v))
        out = state.build_variable_length_temporal_input(self._current(0))
        self.assertEqual(tuple(out.shape), (1, self.C, 5, self.H, self.W))
        self.assertEqual(self._slot_values(out), [0, 4, 3, 2, 1])


class TestTemporalHarmonizerModelTraced(unittest.TestCase):
    """forward() tests with a stub JIT model."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.model_url = "https://example.com/harmonizer_temporal.pt"
        self.ckpt_name = "harmonizer_temporal.pt"
        self.model_resolution = (8, 8)
        self.img_size = torch.Size(self.model_resolution)

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _model_with_stub(self, stub):
        model = TemporalHarmonizerModelTraced(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        # Wrap the stub fn so it plays the JIT graph's role and exposes a device (via
        # _model_device()), as the real model does.
        model.model = _CallableStubModule(stub)
        return model

    def _input(self):
        # Wrapper contract: (H*W, C) image in [0, 1] at model_resolution (so resize is a no-op).
        return torch.rand(self.model_resolution[0] * self.model_resolution[1], 3)

    def test_forward_raises_when_model_not_loaded(self):
        model = TemporalHarmonizerModelTraced(self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution)
        model.model = None
        with self.assertRaises(ValueError):
            model.forward(self._input(), self.img_size, HarmonizerState())

    def test_forward_roundtrip_identity(self):
        # An identity model + matched resolution means preprocess/postprocess invert,
        # so the refined frame equals the input — verifying the full pipeline + slot-0 == current.
        torch.manual_seed(0)
        model = self._model_with_stub(lambda x: x)
        inp = self._input()
        out = model.forward(inp, self.img_size, HarmonizerState())
        self.assertEqual(tuple(out.shape), (self.model_resolution[0] * self.model_resolution[1], 3))
        self.assertTrue(torch.allclose(out, inp, atol=1e-3))

    def test_forward_logs_hint_on_traced_device_constant_mismatch(self):
        # If the JIT model raises a cross-device error while input and model are co-located, the
        # cause is device constants baked into the trace (not the caller). forward must log a clear
        # hint and re-raise the *original* error untouched.
        def stub(x):
            raise RuntimeError("Expected all tensors to be on the same device, but found cuda:0 and cuda:1")

        model = self._model_with_stub(stub)
        with self.assertLogs("nurec.harmonizer.model", level="ERROR") as cm:
            with self.assertRaises(RuntimeError) as ctx:
                model.forward(self._input(), self.img_size, HarmonizerState())
        self.assertIn("hard-codes a cuda device", "\n".join(cm.output).lower())
        # Original error is preserved, not replaced.
        self.assertIn("same device", str(ctx.exception).lower())

    def test_forward_propagates_unrelated_runtime_error(self):
        # A non-device RuntimeError must propagate unchanged and not be misattributed to the trace.
        def stub(x):
            raise RuntimeError("some unrelated failure")

        model = self._model_with_stub(stub)
        with self.assertRaises(RuntimeError) as ctx:
            model.forward(self._input(), self.img_size, HarmonizerState())
        self.assertIn("unrelated", str(ctx.exception))

    def test_forward_takes_slot_zero(self):
        # Stub fills T-slot t with the constant t (in [-1, 1] model space). forward must
        # return slot 0 (=> 0.0 -> postprocess -> 0.5), not any later slot (which would be != 0.5).
        def stub(x):  # x: (1, C, T, H, W)
            out = torch.zeros_like(x)
            for t in range(x.shape[2]):
                out[:, :, t] = float(t)
            return out

        model = self._model_with_stub(stub)
        out = model.forward(self._input(), self.img_size, HarmonizerState())
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.5), atol=1e-4))

    def test_forward_updates_state_and_caps_history(self):
        model = self._model_with_stub(lambda x: x)
        state = HarmonizerState(max_history=4)
        model.forward(self._input(), self.img_size, state)
        self.assertEqual(len(state.history), 1)
        # History must store the model-space frame (C, H_model, W_model), NOT the
        # postprocessed (H*W, C) output — conditioning has to stay in model space.
        self.assertEqual(tuple(state.history[-1].shape), (3, *self.model_resolution))
        for _ in range(5):
            model.forward(self._input(), self.img_size, state)
        self.assertEqual(len(state.history), 4)  # capped at max_history

    def test_forward_rejects_color_transfer(self):
        # External color transfer is legacy difix / non-temporal only; the temporal path must refuse it.
        model = self._model_with_stub(lambda x: x)
        with self.assertRaises(ValueError):
            model.forward(self._input(), self.img_size, HarmonizerState(), use_color_transfer=True)

    def test_forward_raises_when_session_closed(self):
        # A render targeting an ended session must abort, not silently re-open it.
        model = self._model_with_stub(lambda x: x)
        state = HarmonizerState()
        state.close()
        with self.assertRaises(HarmonizerSessionClosedError):
            model.forward(self._input(), self.img_size, state)


class TestTemporalHarmonizerModelCheckpointed(unittest.TestCase):
    """Checkpoint-backed temporal wrapper tests with a stub model (no Cosmos import / CUDA needed)."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.model_url = "https://example.com/diffusion_harmonizer.pkl"
        self.ckpt_name = "diffusion_harmonizer.pkl"
        self.model_resolution = (8, 8)
        self.img_size = torch.Size(self.model_resolution)

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _input(self):
        return torch.rand(self.model_resolution[0] * self.model_resolution[1], 3)

    def _model_with_stub(self, stub):
        model = TemporalHarmonizerModelCheckpointed(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )
        model.model = _CallableStubModule(stub)
        return model

    def _cosmos_dir(self):
        return Path(self.cache_dir) / "nvidia" / "Cosmos-Predict2-0.6B-Text2Image"

    def _write_cosmos_weights(self, cosmos_dir):
        (cosmos_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
        (cosmos_dir / "model.pt").write_text("model")
        (cosmos_dir / "tokenizer" / "tokenizer.pth").write_text("tokenizer")

    @patch("nurec.harmonizer.model.BaseHarmonizerModel._download_checkpoint")
    def test_ensure_cosmos_weights_downloads_missing_weights(self, mock_download):
        model = TemporalHarmonizerModelCheckpointed(
            self.model_url,
            self.cache_dir,
            self.ckpt_name,
            self.model_resolution,
            cosmos_base_model_url="https://huggingface.co/nvidia/Cosmos/resolve/main/model.pt",
            cosmos_tokenizer_url="https://huggingface.co/nvidia/Cosmos/resolve/main/tokenizer/tokenizer.pth",
        )

        def fake_download(model_url, cache_dir, model_filename):
            path = Path(cache_dir) / model_filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(model_url)
            return str(path)

        mock_download.side_effect = fake_download
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NUREC_HARMONIZER_COSMOS_DIR", None)
            ensured_cosmos_dir = model._ensure_cosmos_weights()

        cosmos_dir = self._cosmos_dir()
        self.assertTrue((cosmos_dir / "model.pt").exists())
        self.assertTrue((cosmos_dir / "tokenizer" / "tokenizer.pth").exists())
        self.assertEqual(mock_download.call_count, 2)
        mock_download.assert_any_call(model.cosmos_base_model_url, str(cosmos_dir), "model.pt")
        mock_download.assert_any_call(model.cosmos_tokenizer_url, str(cosmos_dir / "tokenizer"), "tokenizer.pth")
        self.assertEqual(ensured_cosmos_dir, cosmos_dir)

    @patch("nurec.harmonizer.model.BaseHarmonizerModel._download_checkpoint")
    def test_ensure_cosmos_weights_uses_existing_cache(self, mock_download):
        cosmos_dir = self._cosmos_dir()
        self._write_cosmos_weights(cosmos_dir)
        model = TemporalHarmonizerModelCheckpointed(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NUREC_HARMONIZER_COSMOS_DIR", None)
            ensured_cosmos_dir = model._ensure_cosmos_weights()

        mock_download.assert_not_called()
        self.assertEqual(ensured_cosmos_dir, cosmos_dir)

    def test_ensure_cosmos_weights_resolves_relative_cache_path(self):
        relative_root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, relative_root, ignore_errors=True)
        cosmos_dir = relative_root / "cosmos-weights"
        self._write_cosmos_weights(cosmos_dir)
        model = TemporalHarmonizerModelCheckpointed(
            self.model_url,
            self.cache_dir,
            self.ckpt_name,
            self.model_resolution,
            cosmos_cache_dir="cosmos-weights",
        )

        old_cwd = Path.cwd()
        try:
            os.chdir(relative_root)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NUREC_HARMONIZER_COSMOS_DIR", None)
                ensured_cosmos_dir = model._ensure_cosmos_weights()
        finally:
            os.chdir(old_cwd)

        self.assertEqual(ensured_cosmos_dir, cosmos_dir)

    def test_ensure_cosmos_weights_requires_urls_for_missing_weights(self):
        model = TemporalHarmonizerModelCheckpointed(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NUREC_HARMONIZER_COSMOS_DIR", None)
            with self.assertRaisesRegex(FileNotFoundError, "cosmos_base_model_url"):
                model._ensure_cosmos_weights()

    @patch("nurec.harmonizer.model.torch.cuda.current_device", return_value=0)
    @patch.object(TemporalHarmonizerModelCheckpointed, "_ensure_cosmos_weights")
    @patch.object(TemporalHarmonizerModelCheckpointed, "_download_checkpoint")
    @patch("nurec.harmonizer.model.importlib.import_module")
    def test_load_pretrained_model_passes_absolute_cosmos_dir(
        self, mock_import_module, mock_download_checkpoint, mock_ensure_cosmos_weights, mock_current_device
    ):
        cosmos_dir = Path(self.cache_dir) / "cosmos"
        mock_ensure_cosmos_weights.return_value = cosmos_dir
        mock_download_checkpoint.return_value = str(Path(self.cache_dir) / self.ckpt_name)
        model_instance = MagicMock()
        model_instance.to.return_value = model_instance
        pix2pix_turbo = MagicMock(return_value=model_instance)
        mock_import_module.return_value = SimpleNamespace(Pix2Pix_Turbo=pix2pix_turbo)
        model = TemporalHarmonizerModelCheckpointed(
            self.model_url, self.cache_dir, self.ckpt_name, self.model_resolution
        )

        working_dir = Path.cwd()
        self.assertIs(model._load_pretrained_model(), model_instance)

        self.assertEqual(Path.cwd(), working_dir)
        self.assertEqual(pix2pix_turbo.call_args.kwargs["cosmos_dir"], str(cosmos_dir))

    def test_forward_uses_single_frame_until_full_history(self):
        seen_t = []

        def stub(x):
            seen_t.append(x.shape[2])
            return x

        model = self._model_with_stub(stub)
        state = HarmonizerState(max_history=4)
        for _ in range(4):
            model.forward(self._input(), self.img_size, state)

        self.assertEqual(seen_t, [1, 1, 1, 1])
        self.assertEqual(len(state.history), 4)

    def test_forward_converts_input_to_checkpoint_dtype(self):
        seen_dtypes = []

        def stub(x):
            seen_dtypes.append(x.dtype)
            return x

        model = self._model_with_stub(stub)
        state = HarmonizerState(max_history=4)
        for _ in range(5):
            model.forward(self._input(), self.img_size, state)

        self.assertEqual(seen_dtypes, [torch.bfloat16] * 5)

    def test_forward_uses_full_temporal_stack_after_history_is_available(self):
        seen_slots = []

        def stub(x):
            seen_slots.append(x.shape[2])
            return x

        model = self._model_with_stub(stub)
        state = HarmonizerState(max_history=4)
        for _ in range(5):
            model.forward(self._input(), self.img_size, state)

        self.assertEqual(seen_slots, [1, 1, 1, 1, 5])
        self.assertEqual(len(state.history), 4)

    def test_forward_uses_current_only_path_when_checkpoint_supports_it(self):
        seen_slots = []

        class CurrentOnlyStub(_CallableStubModule):
            def forward(self, x):
                raise AssertionError("checkpoint wrapper should use forward_current_only")

            def forward_current_only(self, x):
                seen_slots.append(x.shape[2])
                return x[:, :, :1]

        model = self._model_with_stub(lambda x: x)
        model.model = CurrentOnlyStub(lambda x: x)
        state = HarmonizerState(max_history=4)
        for _ in range(5):
            model.forward(self._input(), self.img_size, state)

        self.assertEqual(seen_slots, [1, 1, 1, 1, 5])
        self.assertEqual(len(state.history), 4)


class TestHarmonizerModelFactory(unittest.TestCase):
    def setUp(self):
        HarmonizerModelFactory._initialized_model = None
        self.model_url = "https://example.com/model.tar.gz"
        self.cache_dir = "~/.cache/nurec/harmonizer"
        self.model_filename = "harmonizer.pt"
        self.model_resolution = (544, 960)
        # MagicMock(spec=...) satisfies the factory's isinstance() cross-type guard.
        self.mock_instance = MagicMock(spec=NonTemporalHarmonizerModelTraced)

    def tearDown(self):
        HarmonizerModelFactory._initialized_model = None

    @patch("nurec.harmonizer.model.torch.distributed.is_initialized", return_value=False)
    @patch.object(NonTemporalHarmonizerModelTraced, "_detect_needs_temporal_dim")
    @patch.object(NonTemporalHarmonizerModelTraced, "_load_pretrained_model")
    @patch("nurec.harmonizer.model.BaseHarmonizerModel._download_checkpoint")
    def test_get_creates_new_instance(self, mock_download, mock_load, mock_detect, mock_is_initialized):
        # Patch the I/O methods (not the class) so a real NonTemporalHarmonizerModelTraced is
        # constructed — the factory's isinstance() guard needs a real type, not a Mock.
        mock_download.return_value = "/path/to/checkpoint"
        mock_loaded_model = MagicMock()
        mock_load.return_value = mock_loaded_model

        result = HarmonizerModelFactory.get(self.model_url, self.cache_dir, self.model_filename, self.model_resolution)

        # A real non-temporal instance is returned with the loaded JIT model attached.
        self.assertIsInstance(result, NonTemporalHarmonizerModelTraced)
        self.assertEqual(result.model, mock_loaded_model)
        # Download + load + temporal-dim probe each ran exactly once.
        mock_download.assert_called_once_with(self.model_url, self.cache_dir, self.model_filename)
        mock_load.assert_called_once()
        mock_detect.assert_called_once()
        mock_is_initialized.assert_called_once_with()

    def test_get_returns_existing_instance(self):
        # A cached instance is returned as-is, without constructing a new one.
        HarmonizerModelFactory._initialized_model = self.mock_instance
        result = HarmonizerModelFactory.get(self.model_url, self.cache_dir, self.model_filename, self.model_resolution)
        self.assertEqual(result, self.mock_instance)

    @patch("nurec.harmonizer.model.BaseHarmonizerModel._download_checkpoint")
    @patch("nurec.harmonizer.model.torch.distributed.get_world_size", return_value=2)
    @patch("nurec.harmonizer.model.torch.distributed.is_initialized", return_value=True)
    def test_get_rejects_multi_gpu_before_model_download(self, mock_is_initialized, mock_get_world_size, mock_download):
        with self.assertRaisesRegex(RuntimeError, r"does not support multi-GPU execution \(world_size=2\)"):
            HarmonizerModelFactory.get(
                self.model_url,
                self.cache_dir,
                self.model_filename,
                self.model_resolution,
            )

        mock_is_initialized.assert_called_once_with()
        mock_get_world_size.assert_called_once_with()
        mock_download.assert_not_called()

    def test_clear_cache(self):
        HarmonizerModelFactory._initialized_model = self.mock_instance
        HarmonizerModelFactory.clear_cache()
        self.assertIsNone(HarmonizerModelFactory._initialized_model)

    def test_get_rejects_unvalidated_checkpoint_suffix(self):
        with self.assertRaises(ValueError):
            HarmonizerModelFactory.get(
                self.model_url,
                self.cache_dir,
                "diffusion_harmonizer.pth",
                self.model_resolution,
                temporal=True,
            )

    @patch.object(NonTemporalHarmonizerModelTraced, "_detect_needs_temporal_dim")
    @patch.object(NonTemporalHarmonizerModelTraced, "_load_pretrained_model")
    @patch("nurec.harmonizer.model.BaseHarmonizerModel._download_checkpoint")
    def test_thread_safety(self, mock_download, mock_load, mock_detect):
        HarmonizerModelFactory.clear_cache()
        mock_download.return_value = "/path/to/checkpoint"
        mock_load.return_value = MagicMock()
        instances = []

        def create_instance():
            instances.append(
                HarmonizerModelFactory.get(self.model_url, self.cache_dir, self.model_filename, self.model_resolution)
            )

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Singleton: every thread observes the same instance, loaded exactly once.
        self.assertEqual(len({id(i) for i in instances}), 1)
        self.assertEqual(mock_download.call_count, 1)
        self.assertEqual(mock_load.call_count, 1)

    @patch.object(TemporalHarmonizerModelTraced, "_load_pretrained_model")
    @patch("nurec.harmonizer.model.BaseHarmonizerModel._download_checkpoint")
    def test_get_temporal_creates_temporal_instance(self, mock_download, mock_load):
        # temporal=True builds a TemporalHarmonizerModelTraced and skips the (non-temporal-only)
        # dimension probe — if the factory wrongly probed it, the missing method would raise.
        mock_download.return_value = "/path/to/checkpoint"
        mock_load.return_value = MagicMock()

        result = HarmonizerModelFactory.get(
            self.model_url, self.cache_dir, self.model_filename, self.model_resolution, temporal=True
        )

        self.assertIsInstance(result, TemporalHarmonizerModelTraced)
        self.assertEqual(result.model, mock_load.return_value)

    @patch.object(TemporalHarmonizerModelCheckpointed, "_load_pretrained_model")
    @patch("nurec.harmonizer.model.BaseHarmonizerModel._download_checkpoint")
    def test_get_temporal_checkpoint_creates_checkpointed_temporal_instance(self, mock_download, mock_load):
        mock_download.return_value = "/path/to/diffusion_harmonizer.pkl"
        mock_load.return_value = MagicMock()

        result = HarmonizerModelFactory.get(
            self.model_url,
            self.cache_dir,
            "diffusion_harmonizer.pkl",
            self.model_resolution,
            temporal=True,
            cosmos_base_model_url="https://huggingface.co/nvidia/Cosmos/resolve/main/model.pt",
            cosmos_tokenizer_url="https://huggingface.co/nvidia/Cosmos/resolve/main/tokenizer/tokenizer.pth",
            cosmos_cache_dir="/tmp/cosmos",
        )

        self.assertIsInstance(result, TemporalHarmonizerModelCheckpointed)
        self.assertEqual(result.model, mock_load.return_value)
        self.assertEqual(result.cosmos_base_model_url, "https://huggingface.co/nvidia/Cosmos/resolve/main/model.pt")
        self.assertEqual(
            result.cosmos_tokenizer_url, "https://huggingface.co/nvidia/Cosmos/resolve/main/tokenizer/tokenizer.pth"
        )
        self.assertEqual(result.cosmos_cache_dir, "/tmp/cosmos")

    def test_get_rejects_nontemporal_checkpoint(self):
        with self.assertRaises(ValueError):
            HarmonizerModelFactory.get(
                self.model_url, self.cache_dir, "diffusion_harmonizer.pkl", self.model_resolution, temporal=False
            )

    def test_get_rejects_temporal_request_when_nontemporal_cached(self):
        # MagicMock(spec=...) satisfies the factory's isinstance() guard.
        HarmonizerModelFactory._initialized_model = MagicMock(spec=NonTemporalHarmonizerModelTraced)
        with self.assertRaises(RuntimeError):
            HarmonizerModelFactory.get(
                self.model_url, self.cache_dir, self.model_filename, self.model_resolution, temporal=True
            )

    def test_get_rejects_nontemporal_request_when_temporal_cached(self):
        HarmonizerModelFactory._initialized_model = MagicMock(spec=TemporalHarmonizerModelTraced)
        with self.assertRaises(RuntimeError):
            HarmonizerModelFactory.get(
                self.model_url, self.cache_dir, self.model_filename, self.model_resolution, temporal=False
            )


class TestHarmonizerConfigCompatibility(unittest.TestCase):
    """Tests for Harmonizer configuration validation in system_config_compatibility_check."""

    def setUp(self):
        """Set up test configuration with valid training-time Harmonizer settings."""
        # Create a mock config with valid harmonizer training settings
        self.config = OmegaConf.create(
            {
                "dataset": {
                    "name": "ncore",
                    "n_train_sample_camera_rays": 1024,
                    "val_lidar": False,
                    "n_train_sample_lidar_rays": 0,
                    "samplers": {
                        "batch_sampler": {
                            "name": "default",
                            "ratio_camera_samples": 1.0,
                            "update_n_epochs": 1,
                            "camera_pixel_sampler": {"name": "image-crop"},
                        }
                    },
                },
                "harmonizer": {
                    "name": "harmonizer",
                    "model_url": "https://example.com/model.tar.gz",
                    "model_resolution": [544, 960],
                    "model_filename": "harmonizer.pt",
                    "cache_dir": "~/.cache/nurec/harmonizer",
                    "training": {
                        "enabled": True,
                        "p_scheduler": {
                            "p_init": 0.5,
                        },
                        "start_step": 15000,
                    },
                    "inference": {
                        "enabled": False,
                    },
                },
                "model": {
                    "calib": {
                        "name": "skip-calib"  # Skip calibration check
                    },
                    # Add minimal model config
                    "gaussians": {
                        "name": "default-gaussians"  # Not "sh-gaussians" to avoid extra assertions
                    },
                },
                "trainer": {
                    "precision": 32  # Required for gaussian checks
                },
                "system": {
                    "name": "default-system"  # For lidar validation check
                },
            }
        )

    def test_valid_harmonizer_training_config(self):
        """Test valid training-time Harmonizer configuration passes checks."""
        # Should not raise any exceptions for valid config
        system_config_compatibility_check(self.config)


class TestTrainingHarmonizerController(unittest.TestCase):
    """Guard rail: training is non-temporal only."""

    def test_temporal_training_is_rejected(self):
        # Enabling temporal training must fail loudly (the temporal harmonizer is gRPC-inference only).
        # A spec'd mock satisfies the controller's isinstance(cfg, TrainingHarmonizerConfig) assert,
        # and the guard fires before any heavy initialization (RNG / scheduler / model load).
        cfg = MagicMock(spec=TrainingHarmonizerConfig)
        cfg.enabled = True
        cfg.temporal = True
        system = MagicMock()
        system.harmonizer_config.training = cfg
        with self.assertRaises(ValueError):
            TrainingHarmonizerController(system)
