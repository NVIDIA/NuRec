# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for get_rendered_visibility_mask.

The function only reads ``rendered_cam`` / ``rendered_lidar`` and their
``visibility`` / ``cumulated_weights``, but we build the real
``GaussiansRenderReturn`` / ``GaussiansCompositeReturn`` dataclasses (cheap,
CPU-only) rather than ad-hoc stand-ins so the tests stay coupled to the actual
types: both are ``slots=True``, so a renamed field would fail construction here
instead of silently passing.
"""

import logging

from typing import Optional

import pytest
import torch

from nurec.libs.losses.models.utils import (
    MissingRenderedVisibilityError,
    _maybe_update_mcmc_visibility_counters,
    get_rendered_visibility_mask,
)
from nurec.utils.types import GaussiansCompositeReturn, GaussiansRenderReturn


DEVICE = torch.device("cpu")
DTYPE = torch.float32

# opacity/distance are required by GaussiansRenderReturn but unused by the
# function under test; a single throwaway value satisfies the dataclass.
_UNUSED = torch.zeros(1, device=DEVICE, dtype=DTYPE)


def _rendered(
    visibility: Optional[torch.Tensor] = None,
    cumulated_weights: Optional[torch.Tensor] = None,
    frustum_visibility: Optional[torch.Tensor] = None,
) -> GaussiansRenderReturn:
    return GaussiansRenderReturn(
        opacity=_UNUSED,
        distance=_UNUSED,
        visibility=visibility,
        cumulated_weights=cumulated_weights,
        frustum_visibility=frustum_visibility,
    )


def _results(
    rendered_cam: Optional[GaussiansRenderReturn] = None,
    rendered_lidar: Optional[GaussiansRenderReturn] = None,
) -> GaussiansCompositeReturn:
    return GaussiansCompositeReturn(rendered_cam=rendered_cam, rendered_lidar=rendered_lidar)


def _call(results: GaussiansCompositeReturn, element_count: int, **kwargs) -> torch.Tensor:
    return get_rendered_visibility_mask(
        results=results,
        element_count=element_count,
        device=DEVICE,
        dtype=DTYPE,
        **kwargs,
    )


class TestGetRenderedVisibilityMask:
    def test_filter_disabled_returns_all_ones(self):
        """Visibility filtering off means Gaussian regularizers should see every element."""
        out = _call(_results(), element_count=5, visibility_filter=False)
        assert torch.equal(out, torch.ones(5, device=DEVICE, dtype=DTYPE))

    def test_camera_only_visibility(self):
        """Use camera renderer visibility directly when it is the only available source."""
        vis = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        out = _call(_results(rendered_cam=_rendered(visibility=vis)), element_count=5, visibility_filter=True)
        assert torch.equal(out, vis)

    def test_lidar_only_visibility(self):
        """Use lidar renderer visibility directly when camera output is absent."""
        vis = torch.tensor([1.0, 0.0, 1.0])
        out = _call(_results(rendered_lidar=_rendered(visibility=vis)), element_count=3, visibility_filter=True)
        assert torch.equal(out, vis)

    def test_camera_and_lidar_union_is_elementwise_maximum(self):
        """Combine camera and lidar visibility so either sensor can mark a Gaussian visible."""
        cam = torch.tensor([0.0, 1.0, 0.0, 0.0])
        lidar = torch.tensor([0.0, 0.0, 1.0, 0.0])
        out = _call(
            _results(rendered_cam=_rendered(visibility=cam), rendered_lidar=_rendered(visibility=lidar)),
            element_count=4,
            visibility_filter=True,
        )
        assert torch.equal(out, torch.tensor([0.0, 1.0, 1.0, 0.0]))

    def test_occlusion_aware_binarizes_cumulated_weights(self):
        """Occlusion-aware filtering should use cumulated weights as a binary visibility mask."""
        weights = torch.tensor([0.0, 0.3, 0.0, 2.0, 0.1])
        out = _call(
            _results(rendered_cam=_rendered(cumulated_weights=weights)),
            element_count=5,
            visibility_filter=True,
            occlusion_aware=True,
        )
        assert torch.equal(out, torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]))

    def test_missing_sources_non_occlusion_raises_value_error(self):
        """Fail loudly when non-occlusion filtering is requested without visibility outputs."""
        with pytest.raises(ValueError, match="visibility"):
            _call(_results(), element_count=5, visibility_filter=True, occlusion_aware=False)

    def test_missing_sources_occlusion_raises_value_error(self):
        """Fail loudly when occlusion-aware filtering lacks cumulated-weight outputs."""
        with pytest.raises(ValueError, match="cumulated_weights"):
            _call(_results(), element_count=5, visibility_filter=True, occlusion_aware=True)

    def test_missing_plain_visibility_raises_specific_error(self):
        """Missing plain visibility raises the specific MissingRenderedVisibilityError subtype.

        Callers (GaussiansSystem) catch precisely this to distinguish an expected
        missing-output fallback from a genuine tensor/logic ValueError.
        """
        with pytest.raises(MissingRenderedVisibilityError, match="visibility"):
            _call(_results(), element_count=5, visibility_filter=True, occlusion_aware=False)

    def test_missing_occlusion_weights_raises_specific_error(self):
        """Missing occlusion-aware weights raises the same specific MissingRenderedVisibilityError."""
        with pytest.raises(MissingRenderedVisibilityError, match="cumulated_weights"):
            _call(_results(), element_count=5, visibility_filter=True, occlusion_aware=True)

    def test_missing_visibility_error_is_value_error_subclass(self):
        """The specific error must remain a ValueError for backwards-compatible except blocks."""
        assert issubclass(MissingRenderedVisibilityError, ValueError)
        with pytest.raises(ValueError):
            _call(_results(), element_count=5, visibility_filter=True, occlusion_aware=False)

    def test_shape_mismatch_falls_back_to_all_ones(self, caplog):
        """Shape mismatches should warn and disable filtering rather than misindexing rows."""
        vis = torch.tensor([0.0, 1.0, 0.0, 1.0])  # length 4, but element_count is 5
        with caplog.at_level(logging.WARNING):
            out = _call(_results(rendered_cam=_rendered(visibility=vis)), element_count=5, visibility_filter=True)
        assert torch.equal(out, torch.ones(5, device=DEVICE, dtype=DTYPE))
        assert "visibility shape mismatch" in caplog.text


class TestMcmcCountersUseFrustumVisibility:
    """MCMC invisible-steps counting must ignore active-tile-mask narrowing: a gaussian
    in the frustum whose tiles are all masked is intentionally unsupervised, not dead.
    The narrowed ``visibility`` still drives SelectiveAdam and the regularizer filters;
    only the counter update swaps to the pre-mask ``frustum_visibility`` union."""

    @staticmethod
    def _stub_composite(calls: list) -> "object":
        from types import SimpleNamespace

        from nurec.models.gaussians.gaussians_composite import GaussiansComposite

        model = object.__new__(GaussiansComposite)
        torch.nn.Module.__init__(model)
        model.gaussians_strategy = SimpleNamespace(  # type: ignore[assignment]
            update_visibility_counters=lambda **kwargs: calls.append(kwargs)
        )
        model.gaussians_nodes = {}  # type: ignore[assignment]
        return model

    def test_counter_uses_frustum_union_when_narrowed(self):
        """Narrowed render: counters get the pre-mask visibility, not the masked one."""
        calls: list = []
        narrowed = torch.tensor([0.0, 0.0, 1.0])
        frustum = torch.tensor([1.0, 0.0, 1.0])
        results = _results(rendered_cam=_rendered(visibility=narrowed, frustum_visibility=frustum))
        _maybe_update_mcmc_visibility_counters(
            model=self._stub_composite(calls),
            results=results,
            visibility=narrowed,  # type: ignore[arg-type]
        )
        assert len(calls) == 1
        assert torch.equal(calls[0]["visibility"], frustum)

    def test_counter_unions_frustum_cam_with_plain_lidar(self):
        """Only the narrowed render carries frustum_visibility; the other contributes
        its plain visibility to the union."""
        calls: list = []
        results = _results(
            rendered_cam=_rendered(visibility=torch.tensor([0.0, 0.0]), frustum_visibility=torch.tensor([1.0, 0.0])),
            rendered_lidar=_rendered(visibility=torch.tensor([0.0, 1.0])),
        )
        _maybe_update_mcmc_visibility_counters(
            model=self._stub_composite(calls),
            results=results,
            visibility=torch.tensor([0.0, 0.0]),  # type: ignore[arg-type]
        )
        assert torch.equal(calls[0]["visibility"], torch.tensor([1.0, 1.0]))

    def test_counter_keeps_passed_visibility_without_narrowing(self):
        """No frustum_visibility anywhere (mask off, or a renderer that never sets
        frustum_visibility): passed visibility is used."""
        calls: list = []
        passed = torch.tensor([0.0, 1.0, 0.0])
        results = _results(rendered_cam=_rendered(visibility=passed))
        _maybe_update_mcmc_visibility_counters(
            model=self._stub_composite(calls),
            results=results,
            visibility=passed,  # type: ignore[arg-type]
        )
        assert torch.equal(calls[0]["visibility"], passed)

    def test_counter_falls_back_on_shape_mismatch(self):
        """A frustum union whose numel disagrees with the passed mask is discarded."""
        calls: list = []
        passed = torch.tensor([0.0, 1.0, 0.0])
        results = _results(rendered_cam=_rendered(visibility=torch.zeros(2), frustum_visibility=torch.ones(2)))
        _maybe_update_mcmc_visibility_counters(
            model=self._stub_composite(calls),
            results=results,
            visibility=passed,  # type: ignore[arg-type]
        )
        assert torch.equal(calls[0]["visibility"], passed)
