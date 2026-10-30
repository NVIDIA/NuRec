# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test to prove that ScopedTimer has minimal execution overhead.

This test creates functions with multiple ScopedTimer decorations and measures
the actual number of executed instructions at runtime to prove
that ScopedTimer has minimal execution overhead.
"""

import logging
import sys
import time

from typing import Callable
from unittest.mock import patch

import nurec.utils.profiling as profile

from nurec.config.scopedtimer import ProfilerBackend, ScopedTimerConfig, VerbosityLevel
from nurec.utils.profiling import (
    ProfileColor,
    ScopedTimer,
    TimingTag,
    backward_nvtx_ranges,
    loss_backward_nvtx_active,
    warn_backward_nvtx_detail,
)


# We'll test Tracy through the profile interface instead of importing directly


def undecorated_func() -> None:
    pass


def count_executed_instructions(func: Callable[[], None]) -> int:
    """Count the actual number of instructions executed when running a function."""
    instruction_count = 0

    def trace_calls(frame, event, arg):
        nonlocal instruction_count
        if event == "line":
            instruction_count += 1
        return trace_calls

    # Set the trace function
    old_trace = sys.gettrace()
    sys.settrace(trace_calls)

    try:
        # Execute the function
        func()
    finally:
        # Restore the old trace function
        sys.settrace(old_trace)

    return instruction_count


def create_decorated_function(num_decorations: int = 1000) -> Callable[[], None]:
    """Create a function with many ScopedTimer decorations that use global config."""

    # Start with the base function
    def base_function():
        undecorated_func()

    # Apply multiple ScopedTimer decorations that will use the global config
    decorated_func = base_function
    for i in range(num_decorations):
        # Don't pass config - let it use the global config
        timer = ScopedTimer(f"timer_{i}")
        decorated_func = timer(decorated_func)

    return decorated_func


class _SharedOutputFunction(profile.torch.autograd.Function):
    """Autograd function whose two outputs share one backward Node."""

    @staticmethod
    def forward(ctx, value):
        """Return two scaled copies of the input."""
        return value * 2, value * 3

    @staticmethod
    def backward(ctx, first_grad, second_grad):
        """Combine both incoming gradients into one input gradient."""
        return 2 * first_grad + 3 * second_grad


class _FailingBackwardFunction(profile.torch.autograd.Function):
    """Autograd function whose backward raises."""

    @staticmethod
    def forward(ctx, value):
        """Return the squared input."""
        return value.square()

    @staticmethod
    def backward(ctx, grad_output):
        """Raise to simulate a failed backward."""
        raise RuntimeError("expected backward failure")


class TestBackwardNVTXAttribution:
    """Test direct loss ownership ranges around autograd Node execution."""

    def test_backward_node_owners_are_truthful_for_unique_and_shared_paths(self):
        """Unique losses stay distinct and shared nodes collect both owners."""
        value = profile.torch.tensor(2.0, requires_grad=True)
        shared = value.square()
        first_loss = (shared * 3).sin()
        second_loss = (shared + 5).cos()

        owners = profile._collect_backward_node_owners({"first loss": [first_loss], "second/loss": [second_loss]})

        assert owners[first_loss.grad_fn] == {"first loss"}
        assert owners[second_loss.grad_fn] == {"second/loss"}
        assert owners[shared.grad_fn] == {"first loss", "second/loss"}

    def test_multi_output_node_is_labeled_shared(self):
        """A multi-output Node shared by two losses is labeled shared."""
        value = profile.torch.tensor(2.0, requires_grad=True)
        first_output, second_output = _SharedOutputFunction.apply(value)
        first_loss = first_output.square()
        second_loss = second_output.sin()

        owners = profile._collect_backward_node_owners({"first": [first_loss], "second": [second_loss]})
        shared_node = first_output.grad_fn

        assert shared_node is second_output.grad_fn
        assert owners[shared_node] == {"first", "second"}
        assert profile._backward_nvtx_label(shared_node, owners[shared_node]).startswith(
            "NRE/backward/shared/2/first+second/"
        )

    def test_backward_ranges_balance_remove_hooks_and_preserve_gradients(self):
        """Ranges balance, hooks come off, and gradients match the baseline."""
        baseline = profile.torch.tensor(2.0, requires_grad=True)
        (baseline.square() * 3 + baseline.sin()).backward()
        expected_gradient = baseline.grad.clone()

        value = profile.torch.tensor(2.0, requires_grad=True)
        first_loss = value.square() * 3
        second_loss = value.sin()
        total_loss = first_loss + second_loss
        with (
            patch.object(profile.torch.cuda.nvtx, "range_push", return_value=0) as range_push,
            patch.object(profile.torch.cuda.nvtx, "range_pop", return_value=0) as range_pop,
        ):
            with backward_nvtx_ranges(
                {"quadratic": [first_loss], "trigonometric": [second_loss]},
            ):
                total_loss.backward(retain_graph=True)

            calls_after_context = range_push.call_count
            value.grad = None
            total_loss.backward()

        assert calls_after_context > 0
        assert range_pop.call_count == calls_after_context
        assert range_push.call_count == calls_after_context
        assert profile.torch.equal(value.grad, expected_gradient)
        labels = [call.args[0] for call in range_push.call_args_list]
        assert any(label.startswith("NRE/backward/loss/quadratic/") for label in labels)
        assert any(label.startswith("NRE/backward/shared/2/") for label in labels)

    def test_backward_ranges_remove_hooks_when_context_raises(self):
        """Leaving the context before backward removes hooks and emits nothing."""
        value = profile.torch.tensor(2.0, requires_grad=True)
        loss = value.square()
        with (
            patch.object(profile.torch.cuda.nvtx, "range_push", return_value=0) as range_push,
            patch.object(profile.torch.cuda.nvtx, "range_pop", return_value=0) as range_pop,
        ):
            try:
                with backward_nvtx_ranges({"loss": [loss]}):
                    raise RuntimeError("stop before backward")
            except RuntimeError:
                pass
            loss.backward()

        range_push.assert_not_called()
        range_pop.assert_not_called()

    def test_backward_range_hooks_do_not_replace_gradient_tuples(self):
        """Pre and post hooks push ranges without replacing gradient tuples."""
        node_range = profile._NodeNVTXRange("NRE/backward/loss/test/TestNode")
        gradients = (profile.torch.tensor(1.0), None)
        with (
            patch.object(profile.torch.cuda.nvtx, "range_push", return_value=0),
            patch.object(profile.torch.cuda.nvtx, "range_pop", return_value=0),
        ):
            assert node_range.prehook(gradients) is None
            assert node_range.posthook(gradients, gradients) is None

    def test_backward_labels_are_stable_sanitized_and_bounded(self):
        """Shared labels are sanitized, sorted, and length-bounded."""
        value = profile.torch.tensor(2.0, requires_grad=True)
        label = profile._backward_nvtx_label(
            value.square().grad_fn,
            {"z" * 100, "a/b c", "third"},
        )

        assert label.startswith("NRE/backward/shared/3/a_b_c+third+")
        assert len(label) <= profile._BACKWARD_NVTX_LABEL_LENGTH
        assert " " not in label

    def test_backward_ranges_remove_partial_registration_on_failure(self):
        """A failed hook registration removes handles already installed."""

        class Handle:
            """Hook handle that records removal."""

            removed = False

            def remove(self):
                """Record that this handle was removed."""
                self.removed = True

        class Node:
            """Node double whose second hook registration fails."""

            prehook_handle = Handle()

            def name(self):
                """Return a stable Node name."""
                return "TestNode"

            def register_prehook(self, hook):
                """Install the prehook and return its handle."""
                return self.prehook_handle

            def register_hook(self, hook):
                """Fail post-hook registration."""
                raise RuntimeError("registration failed")

        node = Node()
        try:
            with (
                patch.object(profile, "_collect_backward_node_owners", return_value={node: {"loss"}}),
                backward_nvtx_ranges({}),
            ):
                pass
        except RuntimeError:
            pass

        assert node.prehook_handle.removed

    def test_backward_failure_removes_handles_but_cannot_pop_failed_node(self):
        """A failed backward removes hooks and does not pop the failed Node."""
        value = profile.torch.tensor(2.0, requires_grad=True)
        loss = _FailingBackwardFunction.apply(value).sum()
        pushes: list[str] = []
        pops: list[None] = []
        with (
            patch.object(profile.torch.cuda.nvtx, "range_push", side_effect=lambda label: pushes.append(label) or 0),
            patch.object(profile.torch.cuda.nvtx, "range_pop", side_effect=lambda: pops.append(None) or 0),
        ):
            try:
                with backward_nvtx_ranges({"failure": [loss]}):
                    loss.backward()
            except RuntimeError:
                pass
            calls_after_context = (len(pushes), len(pops))
            try:
                loss.backward()
            except RuntimeError:
                pass

        assert calls_after_context == (2, 1)
        assert (len(pushes), len(pops)) == calls_after_context

    def test_loss_backward_nvtx_stays_off_when_scopedtimer_disabled_without_window(self):
        """A loss request does not walk the graph when ScopedTimer is off and no window is set."""
        previous = (
            ScopedTimer._global_config,
            ScopedTimer._global_backend,
            ScopedTimer._step_window,
            ScopedTimer._capture_state,
        )
        try:
            ScopedTimer.set_global_config(
                ScopedTimerConfig(enabled=False, profiling_backend=ProfilerBackend.NONE, backward_nvtx_detail="loss")
            )
            assert not loss_backward_nvtx_active("loss")
            assert not loss_backward_nvtx_active("off")
        finally:
            (
                ScopedTimer._global_config,
                ScopedTimer._global_backend,
                ScopedTimer._step_window,
                ScopedTimer._capture_state,
            ) = previous

    def test_loss_backward_nvtx_follows_nvtx_backend_and_emit_window(self):
        """Ranges turn on only for NVTX steps inside the emit window."""
        previous = (
            ScopedTimer._global_config,
            ScopedTimer._global_backend,
            ScopedTimer._step_window,
            ScopedTimer._emit_post_train,
            ScopedTimer._capture_state,
        )
        events: list[str] = []

        class FakeCudart:
            @staticmethod
            def cudaProfilerStart() -> None:
                events.append("start")

            @staticmethod
            def cudaProfilerStop() -> None:
                events.append("stop")

        try:
            with patch.object(profile.torch.cuda, "cudart", return_value=FakeCudart()):
                ScopedTimer.set_global_config(
                    ScopedTimerConfig(
                        enabled=True,
                        profiling_backend=ProfilerBackend.NVTX,
                        backward_nvtx_detail="loss",
                        emit_start_step=10,
                        emit_num_steps=2,
                    )
                )
                ScopedTimer.set_step(9)
                assert not loss_backward_nvtx_active("loss")
                ScopedTimer.set_step(10)
                assert loss_backward_nvtx_active("loss")
                ScopedTimer._capture_state.active = False
                ScopedTimer.set_global_config(
                    ScopedTimerConfig(enabled=True, profiling_backend=ProfilerBackend.NONE, backward_nvtx_detail="loss")
                )
                assert not loss_backward_nvtx_active("loss")
        finally:
            ScopedTimer._capture_state.active = False
            (
                ScopedTimer._global_config,
                ScopedTimer._global_backend,
                ScopedTimer._step_window,
                ScopedTimer._emit_post_train,
                ScopedTimer._capture_state,
            ) = previous

    def test_backward_nvtx_detail_warns_when_disabled_or_unwindowed(self, caplog):
        """Startup warns if the flag cannot record, or if it would walk every step."""
        with caplog.at_level(logging.WARNING, logger="nurec.utils.profiling"):
            warn_backward_nvtx_detail(
                "loss",
                ScopedTimerConfig(enabled=False, profiling_backend=ProfilerBackend.NONE, backward_nvtx_detail="loss"),
            )
            warn_backward_nvtx_detail(
                "loss",
                ScopedTimerConfig(
                    enabled=True,
                    profiling_backend=ProfilerBackend.NVTX,
                    backward_nvtx_detail="loss",
                ),
            )
            warn_backward_nvtx_detail("off", ScopedTimerConfig())

        messages = [record.getMessage() for record in caplog.records]
        assert any("stays off" in message for message in messages)
        assert any("no emit window" in message for message in messages)
        assert len(messages) == 2


class TestScopedTimerOverhead:
    """Test class to prove ScopedTimer has minimal execution overhead when verbosity is NONE."""

    def test_scoped_timer_execution_overhead(self):
        """Test that ScopedTimer executes fewer instructions when disabled vs enabled."""
        num_decorations = 64

        results = {}

        for enabled in [True, False]:
            ScopedTimer.set_global_config(ScopedTimerConfig(enabled=enabled, verbosity=VerbosityLevel.NONE))

            # Count executed instructions for undecorated function
            undecorated_executed = count_executed_instructions(undecorated_func)

            # Count executed instructions for decorated function
            decorated_func = create_decorated_function(num_decorations=num_decorations)
            decorated_executed = count_executed_instructions(decorated_func)

            # Calculate overhead per decoration
            overhead_per_decoration = (decorated_executed - undecorated_executed) / num_decorations
            print(
                f"\nEnabled={enabled}: undecorated={undecorated_executed}, decorated={decorated_executed}, overhead={overhead_per_decoration:.2f} instructions/decoration"
            )

            results[enabled] = {
                "undecorated": undecorated_executed,
                "decorated": decorated_executed,
                "overhead": overhead_per_decoration,
            }

        # The key test: disabled should execute fewer instructions than enabled
        disabled_overhead = results[False]["overhead"]
        enabled_overhead = results[True]["overhead"]

        print(f"\nCRITERION CHECK: disabled_overhead={disabled_overhead}  < enabled_overhead={enabled_overhead}")

        assert disabled_overhead < enabled_overhead, (
            f"FAILED: Disabled mode should execute fewer instructions than enabled mode. "
            f"Disabled: {disabled_overhead:.2f}, Enabled: {enabled_overhead:.2f}"
        )

        disabled_overhead_threshold = 10
        assert disabled_overhead < disabled_overhead_threshold, (
            f"FAILED: Disabled mode should execute fewer instructions (ccu) mode. "
            f"Disabled: {disabled_overhead:.2f}, Threshold: {disabled_overhead_threshold:.2f}"
        )

        print(
            f"✓ SUCCESS: Disabled mode executes {enabled_overhead - disabled_overhead:.2f} fewer instructions per decoration than enabled mode"
        )

    def test_profiling_backend_initialization(self):
        """Test that profiling backends can be initialized without errors."""
        # Test different backend values
        backends_to_test = [ProfilerBackend.NONE, ProfilerBackend.TRACY, ProfilerBackend.NVTX]

        for backend in backends_to_test:
            print(f"\nTesting profiling backend: {backend.value}")

            # This should not raise any exceptions
            try:
                profile.initialize(profiling_backend=backend)
                print(f"Successfully initialized {backend.value} backend")
            except (ImportError, ModuleNotFoundError, RuntimeError) as e:
                # For optional backends like Tracy/NVTX, initialization might fail
                # if the dependencies aren't available, but it shouldn't crash
                print(f"Backend {backend.value} initialization returned: {e}")
                # We allow this to pass since Tracy/NVTX might not be available in test environment

    def test_backend_switching(self):
        """Test that ScopedTimer works correctly when switching between backends."""
        print("\nTesting backend switching...")

        # Test with none backend
        profile.initialize(profiling_backend=ProfilerBackend.NONE)

        with ScopedTimer("test_none_backend"):
            time.sleep(0.001)

        print("✓ None backend timer completed")

        # Test tracy backend if available
        try:
            profile.initialize(profiling_backend=ProfilerBackend.TRACY)
            with ScopedTimer("test_tracy_backend"):
                time.sleep(0.001)
            print("✓ Tracy backend timer completed")
        except (ImportError, RuntimeError):
            print("✓ Tracy backend not available, skipping")

        # Test nvtx backend
        profile.initialize(profiling_backend=ProfilerBackend.NVTX)

        with ScopedTimer("test_nvtx_backend"):
            time.sleep(0.001)

        print("✓ NVTX backend timer completed")

        # Switch back to none
        profile.initialize(profiling_backend=ProfilerBackend.NONE)

        with ScopedTimer("test_none_again"):
            time.sleep(0.001)

        print("✓ None backend timer completed after switch")

        print("Backend switching tests passed!")

    def test_parameter_order(self):
        """Test that ScopedTimer parameters work correctly: name, tag, color, config."""
        print("\nTesting parameter order...")

        # Test positional arguments
        with ScopedTimer("test_positional"):
            time.sleep(0.001)
        print("✓ Positional name argument works")

        # Test positional name and tag
        with ScopedTimer("test_positional_tag", TimingTag.DATALOADER):
            time.sleep(0.001)
        print("✓ Positional name and tag arguments work")

        # Test keyword arguments
        with ScopedTimer(name="test_keyword"):
            time.sleep(0.001)
        print("✓ Keyword name argument works")

        # Test with color (keyword-only)
        with ScopedTimer("test_color", color=ProfileColor.GREEN):
            time.sleep(0.001)
        print("✓ Color keyword argument works")

        # Test with config (keyword-only)
        test_config = ScopedTimerConfig(enabled=True, verbosity=VerbosityLevel.BASIC)
        with ScopedTimer("test_config", config=test_config):
            time.sleep(0.001)
        print("✓ Config keyword argument works")

        # Test all arguments together
        with ScopedTimer("test_all", TimingTag.DATALOADER, color=ProfileColor.BLUE, config=test_config):
            time.sleep(0.001)
        print("✓ All arguments together work")

        print("Parameter order tests passed!")


class TestScopedTimerWindows:
    """Test profiler capture ranges emitted by window mode."""

    def setup_method(self) -> None:
        self._reset_global_state()

    def teardown_method(self) -> None:
        self._reset_global_state()

    @staticmethod
    def _reset_global_state() -> None:
        ScopedTimer._global_config = ScopedTimerConfig()
        ScopedTimer._global_backend = ProfilerBackend.NONE
        ScopedTimer._step_window = None
        ScopedTimer._emit_post_train = False
        ScopedTimer._capture_state = profile.CaptureState()

    @staticmethod
    def _fake_cudart(events: list[str]) -> object:
        class FakeCudart:
            @staticmethod
            def cudaProfilerStart() -> None:
                events.append("start")

            @staticmethod
            def cudaProfilerStop() -> None:
                events.append("stop")

        return FakeCudart()

    def test_initialization_precedes_step_zero_window(self) -> None:
        events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_initialization=True,
            emit_start_step=0,
            emit_num_steps=2,
        )

        with patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)):
            ScopedTimer.set_global_config(config)
            assert ScopedTimer._capture_state.in_initialization_window
            assert ScopedTimer._backend_active_for_step()

            ScopedTimer.set_step(0)
            ScopedTimer.set_step(2)

        assert events == ["start", "stop", "start", "stop"]

    def test_initialization_precedes_delayed_step_window(self) -> None:
        events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_initialization=True,
            emit_start_step=10,
            emit_num_steps=2,
        )

        with patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)):
            ScopedTimer.set_global_config(config)
            assert ScopedTimer._backend_active_for_step()

            ScopedTimer.set_step(0)
            assert not ScopedTimer._backend_active_for_step()

            ScopedTimer.set_step(10)
            ScopedTimer.set_step(12)

        assert events == ["start", "stop", "start", "stop"]

    def test_initialization_capture_is_disabled_by_default(self) -> None:
        events: list[str] = []
        range_events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_start_step=0,
            emit_num_steps=2,
        )

        assert not config.emit_initialization
        with (
            patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)),
            patch.object(profile.torch.cuda.nvtx, "range_push", side_effect=range_events.append),
            patch.object(profile.torch.cuda.nvtx, "range_pop", side_effect=lambda: range_events.append("pop")),
        ):
            ScopedTimer.set_global_config(config)
            assert not ScopedTimer._capture_state.in_initialization_window
            assert not ScopedTimer._capture_state.active
            assert ScopedTimer._capture_state.current_step is None
            assert not ScopedTimer._backend_active_for_step()

            with ScopedTimer("initialization"):
                pass
            assert range_events == []

            ScopedTimer.set_step(0)
            with ScopedTimer("training"):
                pass
            ScopedTimer.set_step(2)

        assert events == ["start", "stop"]
        assert range_events == ["training", "pop"]

    def test_initialization_capture_requires_enabled_backend_and_windows(self) -> None:
        events: list[str] = []

        with patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)):
            ScopedTimer.set_global_config(
                ScopedTimerConfig(
                    enabled=True,
                    profiling_backend=ProfilerBackend.NONE,
                    emit_initialization=True,
                    emit_start_step=0,
                    emit_num_steps=2,
                )
            )
            ScopedTimer.set_step(0)

            ScopedTimer.set_global_config(
                ScopedTimerConfig(
                    enabled=False,
                    profiling_backend=ProfilerBackend.NVTX,
                    emit_initialization=True,
                    emit_start_step=0,
                    emit_num_steps=2,
                )
            )
            ScopedTimer.set_step(0)

            ScopedTimer.set_global_config(
                ScopedTimerConfig(
                    enabled=True,
                    profiling_backend=ProfilerBackend.NVTX,
                    emit_initialization=True,
                )
            )
            ScopedTimer.set_step(0)

        assert events == []

    def test_rejected_reconfiguration_preserves_active_capture_backend(self) -> None:
        events: list[str] = []
        active_config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_initialization=True,
            emit_start_step=10,
            emit_num_steps=2,
        )

        with patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)):
            ScopedTimer.set_global_config(active_config)

            try:
                ScopedTimer.set_global_config(ScopedTimerConfig())
            except RuntimeError as error:
                assert "capture is active" in str(error)
            else:
                raise AssertionError("Expected active capture to reject reconfiguration")

            assert ScopedTimer._global_config is active_config
            assert ScopedTimer._global_backend == ProfilerBackend.NVTX
            ScopedTimer.set_step(0)

        assert events == ["start", "stop"]

    def test_print_summary_closes_initialization_capture(self) -> None:
        events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_initialization=True,
            emit_start_step=10,
            emit_num_steps=2,
        )

        with (
            patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)),
            patch.object(ScopedTimer, "_logger", None),
            patch.object(ScopedTimer, "maybe_close_global_logfile"),
        ):
            ScopedTimer.set_global_config(config)
            ScopedTimer.print_summary()

            assert not ScopedTimer._capture_state.in_initialization_window
            assert not ScopedTimer._capture_state.active
            ScopedTimer.set_global_config(ScopedTimerConfig())

        assert events == ["start", "stop"]

    def test_post_train_follows_closed_step_window(self) -> None:
        events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_post_train=True,
            emit_start_step=0,
            emit_num_steps=2,
        )

        with (
            patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)),
            patch.object(ScopedTimer, "_logger", None),
            patch.object(ScopedTimer, "maybe_close_global_logfile"),
        ):
            ScopedTimer.set_global_config(config)
            assert not ScopedTimer._capture_state.in_post_train_window

            ScopedTimer.set_step(0)
            ScopedTimer.set_step(2)
            assert events == ["start", "stop"]

            ScopedTimer.begin_post_train_window()
            assert ScopedTimer._capture_state.in_post_train_window
            assert ScopedTimer._backend_active_for_step()

            ScopedTimer.print_summary()

        assert events == ["start", "stop", "start", "stop"]

    def test_post_train_closes_still_open_step_window(self) -> None:
        events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_post_train=True,
            emit_start_step=0,
            emit_num_steps=4,
        )

        with (
            patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)),
            patch.object(ScopedTimer, "_logger", None),
            patch.object(ScopedTimer, "maybe_close_global_logfile"),
        ):
            ScopedTimer.set_global_config(config)

            # Training ends while the step window is still active.
            ScopedTimer.set_step(0)
            ScopedTimer.set_step(1)
            assert events == ["start"]

            ScopedTimer.begin_post_train_window()
            ScopedTimer.print_summary()

        assert events == ["start", "stop", "start", "stop"]

    def test_post_train_follows_initialization_and_step_windows(self) -> None:
        events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_initialization=True,
            emit_post_train=True,
            emit_start_step=0,
            emit_num_steps=2,
        )

        with (
            patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)),
            patch.object(ScopedTimer, "_logger", None),
            patch.object(ScopedTimer, "maybe_close_global_logfile"),
        ):
            ScopedTimer.set_global_config(config)
            ScopedTimer.set_step(0)
            ScopedTimer.set_step(2)
            ScopedTimer.begin_post_train_window()
            ScopedTimer.print_summary()

        # Initialization, then the requested training window, then post-training.
        assert events == ["start", "stop", "start", "stop", "start", "stop"]

    def test_post_train_capture_is_disabled_by_default(self) -> None:
        events: list[str] = []
        range_events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_start_step=0,
            emit_num_steps=2,
        )

        assert not config.emit_post_train
        with (
            patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)),
            patch.object(profile.torch.cuda.nvtx, "range_push", side_effect=range_events.append),
            patch.object(profile.torch.cuda.nvtx, "range_pop", side_effect=lambda: range_events.append("pop")),
        ):
            ScopedTimer.set_global_config(config)
            ScopedTimer.set_step(0)
            ScopedTimer.set_step(2)

            ScopedTimer.begin_post_train_window()
            assert not ScopedTimer._capture_state.in_post_train_window
            assert not ScopedTimer._backend_active_for_step()

            with ScopedTimer("artifact_export"):
                pass

        assert events == ["start", "stop"]
        assert range_events == []

    def test_post_train_capture_requires_enabled_backend_and_windows(self) -> None:
        events: list[str] = []

        with patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)):
            ScopedTimer.set_global_config(
                ScopedTimerConfig(
                    enabled=True,
                    profiling_backend=ProfilerBackend.NONE,
                    emit_post_train=True,
                    emit_start_step=0,
                    emit_num_steps=2,
                )
            )
            ScopedTimer.begin_post_train_window()

            ScopedTimer.set_global_config(
                ScopedTimerConfig(
                    enabled=False,
                    profiling_backend=ProfilerBackend.NVTX,
                    emit_post_train=True,
                    emit_start_step=0,
                    emit_num_steps=2,
                )
            )
            ScopedTimer.begin_post_train_window()

            ScopedTimer.set_global_config(
                ScopedTimerConfig(
                    enabled=True,
                    profiling_backend=ProfilerBackend.NVTX,
                    emit_post_train=True,
                )
            )
            ScopedTimer.begin_post_train_window()

        assert events == []

    def test_post_train_window_is_idempotent(self) -> None:
        events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_post_train=True,
            emit_start_step=0,
            emit_num_steps=2,
        )

        with (
            patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)),
            patch.object(ScopedTimer, "_logger", None),
            patch.object(ScopedTimer, "maybe_close_global_logfile"),
        ):
            ScopedTimer.set_global_config(config)
            ScopedTimer.set_step(0)
            ScopedTimer.set_step(2)

            ScopedTimer.begin_post_train_window()
            ScopedTimer.begin_post_train_window()

            ScopedTimer.print_summary()
            ScopedTimer.print_summary()

            assert not ScopedTimer._capture_state.in_post_train_window
            assert not ScopedTimer._capture_state.active

        assert events == ["start", "stop", "start", "stop"]

    def test_post_train_window_emits_ranges_for_out_of_window_steps(self) -> None:
        """Mirrors the call order in BaseSystem.on_train_epoch_end.

        The final training step is normally outside every configured window, so ranges
        emit only because the post-training window is open. start() and stop() evaluate
        gating independently, so the window must flip while no range is open, otherwise
        the pop has no matching push.
        """
        events: list[str] = []
        range_events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_post_train=True,
            emit_start_step=0,
            emit_num_steps=2,
        )

        with (
            patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)),
            patch.object(profile.torch.cuda.nvtx, "range_push", side_effect=range_events.append),
            patch.object(profile.torch.cuda.nvtx, "range_pop", side_effect=lambda: range_events.append("pop")),
            patch.object(ScopedTimer, "_logger", None),
            patch.object(ScopedTimer, "maybe_close_global_logfile"),
        ):
            ScopedTimer.set_global_config(config)
            ScopedTimer.set_step(0)
            ScopedTimer.set_step(5)

            with ScopedTimer("BaseSystem/outer_loop"):
                pass
            assert range_events == []

            ScopedTimer.begin_post_train_window()

            with ScopedTimer("BaseSystem/on_train_epoch_end"):
                pass
            with ScopedTimer("checkpoint_save"):
                pass

            ScopedTimer.print_summary()

        assert events == ["start", "stop", "start", "stop"]
        assert range_events == ["BaseSystem/on_train_epoch_end", "pop", "checkpoint_save", "pop"]

    def test_print_summary_closes_capture_left_open_by_step_window(self) -> None:
        """A run ending inside a step window sets no window flag, so print_summary must stop."""
        events: list[str] = []
        config = ScopedTimerConfig(
            enabled=True,
            profiling_backend=ProfilerBackend.NVTX,
            emit_start_step=0,
            emit_num_steps=50,
        )

        with (
            patch.object(profile.torch.cuda, "cudart", return_value=self._fake_cudart(events)),
            patch.object(ScopedTimer, "_logger", None),
            patch.object(ScopedTimer, "maybe_close_global_logfile"),
        ):
            ScopedTimer.set_global_config(config)
            ScopedTimer.set_step(0)

            assert events == ["start"]
            assert ScopedTimer._capture_state.active
            assert not ScopedTimer._capture_state.in_initialization_window
            assert not ScopedTimer._capture_state.in_post_train_window

            ScopedTimer.print_summary()
            ScopedTimer.print_summary()

            assert not ScopedTimer._capture_state.active

        assert events == ["start", "stop"]


class TestScopedTimerThreading:
    """Test ScopedTimer behavior with multiple threads."""

    def test_thread_local_storage_independence(self):
        """Test that timer measurements are independent across threads using TLS."""
        import threading

        print("\nTesting thread-local storage...")

        # Configure with BASIC verbosity to enable timer state tracking
        # Without proper TLS, this should cause "Timer already running" errors
        test_config = ScopedTimerConfig(
            enabled=True,
            verbosity=VerbosityLevel.BASIC,
            profiling_backend=ProfilerBackend.NONE,  # Use basic timer (time.perf_counter_ns)
        )

        # Create a single shared timer instance to stress-test TLS
        timer_name = "shared_timer_name"
        shared_timer = ScopedTimer(timer_name, config=test_config)

        results = {}
        lock = threading.Lock()

        def tls_worker(thread_id: int, barrier: threading.Barrier, expected_duration: float):
            """Worker that measures its own duration and stores the result."""
            try:
                # Synchronize start to maximize overlap and stress-test TLS
                barrier.wait()

                # All threads use the same timer instance with BASIC verbosity
                # Without TLS, this will throw "Timer already running" RuntimeError
                with shared_timer:
                    time.sleep(expected_duration)

                # Get the actual elapsed time from the timer itself (returns ms)
                actual_duration = shared_timer._bp_elapsed[timer_name] / 1000

                with lock:
                    results[thread_id] = {
                        "expected": expected_duration,
                        "actual": actual_duration,
                    }

            except Exception as e:
                with lock:
                    results[thread_id] = {"error": str(e)}

        # Create threads with different durations to verify independence
        num_threads = 5
        expected_durations = [0.01 * (i + 1) for i in range(num_threads)]
        barrier = threading.Barrier(num_threads)
        threads = []

        for i in range(num_threads):
            t = threading.Thread(target=tls_worker, args=(i, barrier, expected_durations[i]))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all threads completed successfully
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"

        # Verify no errors occurred
        for thread_id, result in results.items():
            assert "error" not in result, f"Thread {thread_id} failed: {result['error']}"

        # Print the timer values
        for thread_id, result in results.items():
            expected = result["expected"]
            actual = result["actual"]

            print(f"  Thread {thread_id}: expected {expected:.4f}s, got {actual:.4f}s")
