# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from nurec.utils.unstructured_lidar import fov_from_bounds


def _deg(*values):
    return tuple(math.radians(v) for v in values)


class TestFovFromBounds:
    """The one place that decides periodicity and the +/-pi seam, for training and serving."""

    def test_a_limited_arc_keeps_its_bounds(self):
        az_lo, az_hi = _deg(-55.0, 55.0)
        fov = fov_from_bounds(az_lo, az_hi, *_deg(-11.0, 11.0), source="test")

        assert fov.periodic_az is False
        assert fov.az_lo == pytest.approx(az_lo) and fov.az_hi == pytest.approx(az_hi)

    def test_a_near_full_span_becomes_the_whole_circle(self):
        # A spinning unit's measured azimuth all but closes; keeping the measured bounds would
        # put a hard edge at the seam instead of wrapping.
        fov = fov_from_bounds(*_deg(-179.9, 179.9), *_deg(-25.0, 15.0), source="test")

        assert fov.periodic_az is True
        assert fov.az_lo == pytest.approx(-math.pi) and fov.az_hi == pytest.approx(math.pi)

    def test_a_descending_azimuth_pair_is_refused(self):
        # [170, -170] deg is the 20 deg arc through 180. Ordering it would silently turn it into
        # the 340 deg arc through 0, which the linear binning would then represent wrongly.
        with pytest.raises(ValueError, match="crosses the"):
            fov_from_bounds(*_deg(170.0, -170.0), *_deg(-10.0, 10.0), source="test")

    def test_azimuth_outside_the_range_is_refused(self):
        with pytest.raises(ValueError, match="leaves the"):
            fov_from_bounds(*_deg(-200.0, -100.0), *_deg(-10.0, 10.0), source="test")

    def test_a_descending_elevation_pair_is_ordered(self):
        # Elevation has no seam, so this really is just an ordering.
        fov = fov_from_bounds(*_deg(-55.0, 55.0), *_deg(11.0, -11.0), source="test")

        assert fov.el_lo < fov.el_hi
