# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np

from nurec.libs.fisheye import opencv_fisheye_domain, opencv_fisheye_max_angle


class FisheyeDomainTest(unittest.TestCase):
    def test_monotonic_calibration_preserves_field_of_view(self):
        for coeffs in ([0.0] * 4, [0.1, 0.01, 0.001, 0.0001], [0.01, -0.001, 0.0001, 0.0]):
            self.assertEqual(opencv_fisheye_max_angle(coeffs, 2.0), 2.0)

    def test_first_stationary_point_limits_inverse_domain(self):
        # g(theta) = theta - theta**3/3 has its first maximum at theta=1.
        bound = opencv_fisheye_max_angle([-1.0 / 3.0, 0.0, 0.0, 0.0], 2.0)
        self.assertLess(bound, 1.0)
        self.assertAlmostEqual(bound, 1.0, delta=3e-6)
        self.assertEqual(opencv_fisheye_max_angle([-1.0 / 3.0, 0.0, 0.0, 0.0], 0.8), 0.8)

    def test_selects_first_stationary_point_even_if_endpoint_slope_is_positive(self):
        # g'(theta) = (1-theta**2)*(1-theta**2/4), positive again after theta=2.
        coeffs = [-1.25 / 3.0, 0.25 / 5.0, 0.0, 0.0]
        bound = opencv_fisheye_max_angle(coeffs, 3.0)
        self.assertAlmostEqual(bound, 1.0, delta=3e-6)
        self.assertLess(bound, 1.0)

    def test_repeated_stationary_point_is_a_domain_boundary(self):
        # An even-multiplicity stationary point is still a singular inverse.
        bound = opencv_fisheye_max_angle([-2.0 / 3.0, 1.0 / 5.0, 0.0, 0.0], 2.0)
        self.assertLessEqual(bound, 1.001)

    def test_exact_repeated_root_keeps_float32_slope_margin(self):
        # These coefficients are exactly representable in float32:
        # g'(theta) = (1 - 15*theta**2)**2. Direct roots of g' can acquire
        # a tiny imaginary part, so an exact-real filter can miss this singularity.
        coeffs = np.array([-10.0, 45.0, 0.0, 0.0], dtype=np.float32)
        original = coeffs.copy()
        bound = opencv_fisheye_max_angle(coeffs, 2.0)
        singular_angle = 1.0 / math.sqrt(15.0)
        self.assertLess(bound, singular_angle)
        self.assertLess(singular_angle - bound, 3e-4)
        theta2 = np.linspace(0.0, bound, 10001, dtype=np.float32) ** 2
        derivative = np.float32(1.0) + theta2 * (np.float32(-30.0) + theta2 * np.float32(225.0))
        self.assertTrue(np.all(derivative > 1e-6))
        np.testing.assert_array_equal(coeffs, original)

    def test_configured_fov_and_inverse_limit_are_separate(self):
        domain = opencv_fisheye_domain([-1.0 / 3.0, 0.0, 0.0, 0.0], 0.8)
        self.assertEqual(domain.configured_max_angle, 0.8)
        self.assertEqual(domain.max_angle, 0.8)
        self.assertGreater(domain.inverse_monotonic_limit, domain.max_angle)
        self.assertAlmostEqual(domain.inverse_monotonic_limit, 1.0, delta=3e-6)

    def test_configured_fov_just_below_limit_cannot_round_above_it(self):
        for coeffs in ([-1.0 / 3.0, 0.0, 0.0, 0.0], [-10.0, 45.0, 0.0, 0.0]):
            with self.subTest(coeffs=coeffs):
                limit = opencv_fisheye_domain(coeffs, 2.0).inverse_monotonic_limit
                configured_fov = float(np.nextafter(limit, 0.0))
                # These simple and repeated-root cases round up in float32,
                # despite the requested FOV being strictly below the limit.
                self.assertGreaterEqual(float(np.float32(configured_fov)), limit)
                domain = opencv_fisheye_domain(coeffs, configured_fov)
                self.assertEqual(domain.configured_max_angle, configured_fov)
                self.assertLess(float(np.float32(domain.max_angle)), domain.inverse_monotonic_limit)
                self.assertLess(domain.max_angle, configured_fov)

    def test_pi_cap_is_safe_after_float32_conversion(self):
        domain = opencv_fisheye_domain([0.0] * 4, math.pi)
        self.assertEqual(domain.configured_max_angle, math.pi)
        self.assertEqual(domain.inverse_monotonic_limit, math.pi)
        self.assertLess(float(np.float32(domain.max_angle)), math.pi)
        self.assertAlmostEqual(domain.max_angle, math.pi, delta=2e-7)

    def test_invalid_calibrations_are_rejected(self):
        for coeffs, angle in [
            ([0.0] * 3, 1.0),
            ([math.nan, 0.0, 0.0, 0.0], 1.0),
            ([math.inf, 0.0, 0.0, 0.0], 1.0),
            ([1e100, 0.0, 0.0, 0.0], 1.0),
            ([0.0] * 4, 0.0),
            ([0.0] * 4, -1.0),
            ([0.0] * 4, math.nan),
            ([0.0] * 4, math.inf),
            ([0.0] * 4, math.pi + 0.01),
        ]:
            with self.subTest(coeffs=coeffs, angle=angle), self.assertRaises(ValueError):
                opencv_fisheye_max_angle(coeffs, angle)

    def test_bound_is_on_increasing_side_in_float32(self):
        # Rounded report coefficients are illustrative, not the customer fixture.
        coeffs = np.array([0.152, -0.071, 0.022, -0.003], dtype=np.float32)
        bound = opencv_fisheye_max_angle(coeffs, math.pi)
        samples = np.linspace(0.0, bound, 1001)
        derivative = np.polynomial.polynomial.polyval(samples**2, [1.0, *(coeffs * [3, 5, 7, 9])])
        self.assertTrue(np.all(derivative > 1e-6))
        # The conditioning margin removes only a few microradians for this
        # illustrative simple fold, while the projection coefficients stay intact.
        self.assertAlmostEqual(bound, 2.0112270576, delta=4e-6)


if __name__ == "__main__":
    unittest.main()
