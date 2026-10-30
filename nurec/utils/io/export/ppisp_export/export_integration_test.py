# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic CPU-only tests for :mod:`ppisp_export.export_integration`.

The integration helpers live outside the CLI orchestrator so they
can be exercised directly without dragging in checkpoint loading,
Hydra configs, or the GaussiansSystem. These tests only cover the
helpers' decision logic; SH-match orchestration is tested via the
optimization loop's own unit tests, and SPG sidecar production is
tested via the writer modules.
"""

import unittest

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

from nurec.utils.io.export.ppisp_export.controller.writer import (
    get_ppisp_auto_spg_files,
)
from nurec.utils.io.export.ppisp_export.export_integration import (
    PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
    PPISP_INTEGRATION_MODE_SPG_RUNTIME,
    ppisp_has_controller,
    resolve_ppisp_controller_export_enabled,
    resolve_sh_optimization_config,
    run_sh_match_optimization,
    select_spg_files_for_export,
    validate_export_integration_args,
)
from nurec.utils.io.export.ppisp_export.sh_match.optimization import SHMatchingOptimizationConfig
from nurec.utils.types import NamedSerialized


class TestValidateExportIntegrationArgs(unittest.TestCase):
    """Static fail-loud validation of the export integration arguments.

    The PPISP-presence checks that require a loaded model live in
    the orchestrator and are exercised by the orchestrator dispatch
    tests, so they are out of scope here.
    """

    def _kwargs(self, **overrides: Any) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
            scene_radiance_scale=1.0,
            ppisp_responsivity=1.0,
            sh_optimization_config=None,
            enable_ppisp_controller_export=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_default_kwargs_accepted(self) -> None:
        validate_export_integration_args(**self._kwargs())

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ppisp_integration_mode must be one of"):
            validate_export_integration_args(**self._kwargs(ppisp_integration_mode="unsupported"))

    def test_non_positive_radiance_scale_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "scene_radiance_scale"):
            validate_export_integration_args(**self._kwargs(scene_radiance_scale=0.0))
        with self.assertRaisesRegex(ValueError, "scene_radiance_scale"):
            validate_export_integration_args(**self._kwargs(scene_radiance_scale=-1.0))

    def test_non_finite_responsivity_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ppisp_responsivity"):
            validate_export_integration_args(**self._kwargs(ppisp_responsivity=float("inf")))
        with self.assertRaisesRegex(ValueError, "ppisp_responsivity"):
            validate_export_integration_args(**self._kwargs(ppisp_responsivity=float("nan")))

    def test_non_numeric_scale_rejected(self) -> None:
        with self.assertRaises(TypeError):
            validate_export_integration_args(**self._kwargs(scene_radiance_scale="1.0"))  # type: ignore[arg-type]

    def test_sh_optimized_without_config_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sh_optimization_config"):
            validate_export_integration_args(
                **self._kwargs(
                    ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
                    sh_optimization_config=None,
                )
            )

    def test_sh_optimized_with_config_accepted(self) -> None:
        config = SHMatchingOptimizationConfig(num_iterations=1, device="cpu")
        validate_export_integration_args(
            **self._kwargs(
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
                sh_optimization_config=config,
            )
        )

    def test_controller_export_with_sh_optimized_rejected(self) -> None:
        config = SHMatchingOptimizationConfig(num_iterations=1, device="cpu")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_export_integration_args(
                **self._kwargs(
                    ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
                    sh_optimization_config=config,
                    enable_ppisp_controller_export=True,
                )
            )


class TestResolvePpispControllerExportEnabled(unittest.TestCase):
    """The omitted CLI flag defaults to controller export when available."""

    def test_default_uses_controller_when_present_in_spg_runtime(self) -> None:
        ppisp = SimpleNamespace(config=SimpleNamespace(use_controller=True), controllers=[object()])
        self.assertTrue(ppisp_has_controller(ppisp))
        self.assertTrue(
            resolve_ppisp_controller_export_enabled(
                requested=None,
                ppisp_module=ppisp,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
            )
        )

    def test_default_uses_static_when_controller_absent(self) -> None:
        ppisp = SimpleNamespace(config=SimpleNamespace(use_controller=False), controllers=[])
        self.assertFalse(
            resolve_ppisp_controller_export_enabled(
                requested=None,
                ppisp_module=ppisp,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
            )
        )

    def test_explicit_disable_wins_when_controller_present(self) -> None:
        ppisp = SimpleNamespace(config=SimpleNamespace(use_controller=True), controllers=[object()])
        self.assertFalse(
            resolve_ppisp_controller_export_enabled(
                requested=False,
                ppisp_module=ppisp,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
            )
        )

    def test_sh_optimized_never_auto_enables_controller(self) -> None:
        ppisp = SimpleNamespace(config=SimpleNamespace(use_controller=True), controllers=[object()])
        self.assertFalse(
            resolve_ppisp_controller_export_enabled(
                requested=None,
                ppisp_module=ppisp,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
            )
        )


class TestResolveShOptimizationConfig(unittest.TestCase):
    """``resolve_sh_optimization_config`` picks the effective SH-match config.

    The SH-match learning rates and device are not exposed on the
    CLI by design, but ``num_iterations`` is, so the resolver
    chooses between four states: off (non ``sh-optimized`` mode),
    an explicit Python-supplied config (escape hatch for callers
    like tests forcing a CPU device), an iteration-count override
    layered on top of either the default or the explicit config, or
    a default :class:`SHMatchingOptimizationConfig` carrying the
    tuned algorithm defaults.
    """

    def test_non_sh_optimized_returns_none(self) -> None:
        self.assertIsNone(
            resolve_sh_optimization_config(
                explicit_config=None,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
            )
        )

    def test_non_sh_optimized_returns_none_even_with_explicit_config(self) -> None:
        explicit = SHMatchingOptimizationConfig(num_iterations=1, device="cpu")
        self.assertIsNone(
            resolve_sh_optimization_config(
                explicit_config=explicit,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
            )
        )

    def test_explicit_config_wins_in_sh_optimized(self) -> None:
        explicit = SHMatchingOptimizationConfig(num_iterations=42, device="cpu")
        resolved = resolve_sh_optimization_config(
            explicit_config=explicit,
            ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
        )
        self.assertIs(resolved, explicit)

    def test_default_config_built_when_explicit_omitted(self) -> None:
        # No explicit config + sh-optimized -> the resolver constructs
        # a fresh ``SHMatchingOptimizationConfig`` whose dataclass
        # defaults are the algorithm-tuned values exposed via the
        # config module's DEFAULT_* constants.
        resolved = resolve_sh_optimization_config(
            explicit_config=None,
            ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
        )
        assert resolved is not None
        self.assertIsInstance(resolved, SHMatchingOptimizationConfig)
        # Compare against a freshly-constructed default to lock in
        # "use the dataclass defaults" without hard-coding the
        # numeric values here.
        defaults = SHMatchingOptimizationConfig()
        self.assertEqual(resolved.num_iterations, defaults.num_iterations)
        self.assertEqual(resolved.learning_rate_albedo, defaults.learning_rate_albedo)
        self.assertEqual(resolved.learning_rate_specular, defaults.learning_rate_specular)
        self.assertEqual(resolved.learning_rate_density, defaults.learning_rate_density)
        self.assertEqual(resolved.device, defaults.device)

    def test_num_iterations_override_on_default_config(self) -> None:
        resolved = resolve_sh_optimization_config(
            explicit_config=None,
            ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
            num_iterations_override=123,
        )
        assert resolved is not None
        defaults = SHMatchingOptimizationConfig()
        self.assertEqual(resolved.num_iterations, 123)
        # Other fields fall back to the dataclass defaults.
        self.assertEqual(resolved.learning_rate_albedo, defaults.learning_rate_albedo)
        self.assertEqual(resolved.device, defaults.device)

    def test_num_iterations_override_on_explicit_config_preserves_other_fields(self) -> None:
        explicit = SHMatchingOptimizationConfig(num_iterations=42, device="cpu")
        resolved = resolve_sh_optimization_config(
            explicit_config=explicit,
            ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
            num_iterations_override=7,
        )
        assert resolved is not None
        # The override wins on top of the explicit config: same
        # device, but iteration count replaced.
        self.assertEqual(resolved.num_iterations, 7)
        self.assertEqual(resolved.device, "cpu")
        # Crucially, the resolver must not mutate the caller's
        # config in place; it should return a fresh instance.
        self.assertIsNot(resolved, explicit)
        self.assertEqual(explicit.num_iterations, 42)

    def test_num_iterations_override_rejected_in_non_sh_optimized(self) -> None:
        # A bare CLI typo (``--ppisp-integration-mode spg-runtime
        # --sh-optimization-num-iterations 100``) must not silently
        # become a no-op.
        with self.assertRaisesRegex(ValueError, "sh_optimization_num_iterations"):
            resolve_sh_optimization_config(
                explicit_config=None,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
                num_iterations_override=10,
            )

    def test_num_iterations_override_rejects_zero_and_negative(self) -> None:
        for bad in (0, -1, -100):
            with self.assertRaisesRegex(ValueError, "sh_optimization_num_iterations"):
                resolve_sh_optimization_config(
                    explicit_config=None,
                    ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
                    num_iterations_override=bad,
                )

    def test_num_iterations_override_rejects_non_int(self) -> None:
        with self.assertRaises(TypeError):
            resolve_sh_optimization_config(
                explicit_config=None,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
                num_iterations_override=10.0,  # type: ignore[arg-type]
            )
        # bool is an ``int`` subclass and must be rejected explicitly.
        with self.assertRaises(TypeError):
            resolve_sh_optimization_config(
                explicit_config=None,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
                num_iterations_override=True,  # type: ignore[arg-type]
            )


class TestSelectSpgFilesForExport(unittest.TestCase):
    """SPG sidecar selection mirrors the prim authored by the writer.

    Static path -> 3 ``ppisp_usd_spg.*`` files. Controller path ->
    generated embedded-weight controller files + 3
    ``ppisp_usd_spg_auto.*`` files (no static files, since no static
    prim is authored).
    """

    def test_disabled_controller_returns_static_sidecars(self) -> None:
        files = select_spg_files_for_export(enable_ppisp_controller_export=False)
        names = sorted(f.filename for f in files)
        self.assertEqual(len(files), 3)
        for n in names:
            self.assertTrue(n.startswith("ppisp_usd_spg."))

    def test_enabled_controller_returns_controller_and_auto_sidecars(self) -> None:
        embedded_controller_sidecars = [
            NamedSerialized(filename="ppisp_controller_0.cu", serialized=b"cu0"),
            NamedSerialized(filename="ppisp_controller_0.cu.lua", serialized=b"lua0"),
            NamedSerialized(filename="ppisp_controller_1.cu", serialized=b"cu1"),
            NamedSerialized(filename="ppisp_controller_1.cu.lua", serialized=b"lua1"),
        ]
        with mock.patch(
            "nurec.utils.io.export.ppisp_export.spg_files.get_ppisp_embedded_controller_spg_files",
            return_value=embedded_controller_sidecars,
        ) as controller_mock:
            files = select_spg_files_for_export(
                enable_ppisp_controller_export=True,
                ppisp_module=object(),
                camera_indices=[0, 1],
            )
        controller_mock.assert_called_once_with(mock.ANY, [0, 1])
        names = sorted(f.filename for f in files)
        expected_auto = sorted(f.filename for f in get_ppisp_auto_spg_files())
        expected_controller = sorted(f.filename for f in embedded_controller_sidecars)
        self.assertEqual(set(names), set(expected_controller) | set(expected_auto))

    def test_enabled_controller_deduplicates_camera_indices_preserving_order(self) -> None:
        with mock.patch(
            "nurec.utils.io.export.ppisp_export.spg_files.get_ppisp_embedded_controller_spg_files",
            return_value=[],
        ) as controller_mock:
            select_spg_files_for_export(
                enable_ppisp_controller_export=True,
                ppisp_module=object(),
                camera_indices=[2, 0, 2, 1, 0],
            )
        controller_mock.assert_called_once_with(mock.ANY, [2, 0, 1])

    def test_enabled_controller_requires_ppisp_module(self) -> None:
        with self.assertRaisesRegex(ValueError, "ppisp_module and camera_indices"):
            select_spg_files_for_export(enable_ppisp_controller_export=True, camera_indices=[0])

    def test_enabled_controller_requires_camera_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "ppisp_module and camera_indices"):
            select_spg_files_for_export(enable_ppisp_controller_export=True, ppisp_module=object())

    def test_disabled_controller_rejects_ppisp_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be omitted"):
            select_spg_files_for_export(
                enable_ppisp_controller_export=False,
                ppisp_module=object(),
            )
        with self.assertRaisesRegex(ValueError, "must be omitted"):
            select_spg_files_for_export(
                enable_ppisp_controller_export=False,
                camera_indices=[0],
            )

    def test_no_overlap_between_static_and_controller_sidecars(self) -> None:
        # Static and controller-mode sidecars must not collide on
        # filenames or the USDZ writer would silently shadow one set.
        static = {f.filename for f in select_spg_files_for_export(enable_ppisp_controller_export=False)}
        with mock.patch(
            "nurec.utils.io.export.ppisp_export.spg_files.get_ppisp_embedded_controller_spg_files",
            return_value=[
                NamedSerialized(filename="ppisp_controller_0.cu", serialized=b"cu"),
                NamedSerialized(filename="ppisp_controller_0.cu.lua", serialized=b"lua"),
            ],
        ):
            controller = {
                f.filename
                for f in select_spg_files_for_export(
                    enable_ppisp_controller_export=True,
                    ppisp_module=object(),
                    camera_indices=[0],
                )
            }
        self.assertEqual(static & controller, set())


class TestRunSHMatchOptimizationOrder(unittest.TestCase):
    """Pin the call order: snapshot BEFORE warm-start.

    The reference target of the SH-match optimization must encode the
    *original-checkpoint* SH state, not the warm-started state. The
    warm-start is a DC-only fold of PPISP into ``features_albedo``;
    snapshotting *after* warm-start makes the reference render a
    degraded approximation of what lavoisier's two-model baseline
    (``threedgrut/export/usd/post_processing_sh_bake.py`` with
    ``reference_model = original`` and ``baked_model = clone``) drives
    the live SH toward, producing visible color/detail regressions
    in production exports (valiant).

    This test pins the order by spying on the two collaborators
    whose relative ordering is the contract: the orchestrator must
    call ``snapshot_sh_fit_parameters`` strictly before
    ``adapter.warmstart``.
    """

    def _spy_targets(self) -> Dict[str, mock.MagicMock]:
        # Patch only what's needed to drive the orchestrator past the
        # call-order point without dragging in a real model, dataset,
        # or optimization loop. Downstream collaborators are inert.
        patchers = {
            "snapshot": mock.patch(
                "nurec.utils.io.export.ppisp_export.export_integration.snapshot_sh_fit_parameters",
                return_value=mock.MagicMock(name="snapshot"),
            ),
            "adapter_cls": mock.patch(
                "nurec.utils.io.export.ppisp_export.export_integration.PPISPSHOptimizationAdapter",
            ),
            "make_callables": mock.patch(
                "nurec.utils.io.export.ppisp_export.export_integration.make_sh_match_render_callables",
                return_value=(mock.MagicMock(name="ref_render"), mock.MagicMock(name="live_render")),
            ),
            "data_loader": mock.patch(
                "nurec.utils.io.export.ppisp_export.export_integration.DataLoader",
            ),
            "iter_batches": mock.patch(
                "nurec.utils.io.export.ppisp_export.export_integration.iterate_calibrated_batches",
            ),
            "make_sampler": mock.patch(
                "nurec.utils.io.export.ppisp_export.export_integration.make_training_view_sampler",
            ),
            "optimize": mock.patch(
                "nurec.utils.io.export.ppisp_export.export_integration.optimize_sh_to_match_ppisp",
            ),
        }
        return {name: p.start() for name, p in patchers.items()}

    def test_snapshot_called_strictly_before_warmstart(self) -> None:
        spies = self._spy_targets()
        self.addCleanup(mock.patch.stopall)

        call_order: List[str] = []

        def _record_snapshot(*args: object, **kwargs: object) -> mock.MagicMock:
            call_order.append("snapshot")
            return mock.MagicMock()

        def _record_warmstart(*args: object, **kwargs: object) -> None:
            call_order.append("warmstart")

        spies["snapshot"].side_effect = _record_snapshot
        adapter_instance = spies["adapter_cls"].return_value
        adapter_instance.warmstart.side_effect = _record_warmstart
        adapter_instance.device = "cpu"

        system = SimpleNamespace(
            model=SimpleNamespace(calib=mock.MagicMock(name="calib")),
        )
        ppisp_module = mock.MagicMock(name="ppisp_module")
        train_dataset = mock.MagicMock(name="train_dataset")
        config = SHMatchingOptimizationConfig(num_iterations=1, device="cpu")

        run_sh_match_optimization(
            system=system,
            ppisp_module=ppisp_module,
            train_dataset=train_dataset,
            config=config,
            reference_camera_id=0,
            reference_frame_id=0,
        )

        self.assertEqual(
            call_order,
            ["snapshot", "warmstart"],
            "snapshot_sh_fit_parameters must run strictly before adapter.warmstart "
            "so the reference render encodes the original-checkpoint SH, not the "
            "warm-start-degraded SH (lavoisier-parity).",
        )

    def test_optimize_loop_receives_pre_warmstart_snapshot(self) -> None:
        # End-to-end consistency check: the snapshot object handed to
        # ``make_sh_match_render_callables`` (which builds the reference
        # closure) must be the value returned by ``snapshot_sh_fit_parameters``
        # at its *first* invocation, before any ``warmstart`` mutates
        # ``features_albedo`` on the underlying model.
        spies = self._spy_targets()
        self.addCleanup(mock.patch.stopall)

        sentinel_snapshot = mock.MagicMock(name="pre_warmstart_snapshot")
        spies["snapshot"].return_value = sentinel_snapshot
        adapter_instance = spies["adapter_cls"].return_value
        adapter_instance.device = "cpu"

        system = SimpleNamespace(
            model=SimpleNamespace(calib=mock.MagicMock(name="calib")),
        )
        run_sh_match_optimization(
            system=system,
            ppisp_module=mock.MagicMock(),
            train_dataset=mock.MagicMock(),
            config=SHMatchingOptimizationConfig(num_iterations=1, device="cpu"),
            reference_camera_id=0,
            reference_frame_id=0,
        )

        # The snapshot passed to make_sh_match_render_callables must be
        # the one taken before warmstart (the sentinel above).
        _args, kwargs = spies["make_callables"].call_args
        passed_snapshot = kwargs.get("snapshot") if "snapshot" in kwargs else spies["make_callables"].call_args.args[1]
        self.assertIs(
            passed_snapshot,
            sentinel_snapshot,
            "make_sh_match_render_callables must receive the pre-warmstart snapshot.",
        )


if __name__ == "__main__":
    unittest.main()
