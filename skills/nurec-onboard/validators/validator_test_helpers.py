# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for the validator unit tests in ``validators/``.

Provides :class:`ValidatorTestCase`, a :class:`unittest.TestCase` subclass
with two assertion mixins matching the validators' return shape (a list
of diagnostic strings):

* :meth:`ValidatorTestCase.assertOneDiagnosticMatching` -- exactly one
  diagnostic contains ``substring``.
* :meth:`ValidatorTestCase.assertNoDiagnosticMatching` -- no diagnostic
  contains ``substring``.

These two patterns recur dozens of times across the validator tests
(each loophole-closure test asserts the new diagnostic fires exactly
once; each happy-path test asserts a class of diagnostic does not
fire). Centralizing them keeps the assertion style consistent across
all validator tests and gives the failure message a uniform shape.

The filename follows the ``<purpose>_test_helpers.py`` shape: the
``_test_`` infix marks this file as test-side support rather than a
test case itself, and the ``validator_`` prefix names what it helps
with.
"""

from __future__ import annotations

import unittest


class ValidatorTestCase(unittest.TestCase):
    """``TestCase`` with diagnostic-list assertion helpers."""

    def assertOneDiagnosticMatching(self, diagnostics: list[str], *substrings: str) -> str:
        """Assert exactly one diagnostic contains all ``substrings`` and return it.

        Args:
            diagnostics: Validator-produced diagnostic strings.
            substrings: One or more substrings that must all appear in
                the same diagnostic (intersection match, not union).
                Passing one substring is the common case; passing two or
                more disambiguates when several diagnostics share a
                token -- e.g. ``("validate_run.exit_code", "must be an
                int")`` pins down the type-check diagnostic without
                accidentally matching the ``... is missing`` variant on
                the same key.

        Returns:
            The single matching diagnostic, for follow-up assertions on
            its full content.
        """
        hits = [d for d in diagnostics if all(s in d for s in substrings)]
        self.assertEqual(
            len(hits),
            1,
            msg=f"expected exactly 1 diagnostic matching all of {substrings!r}, got {len(hits)}: {hits}",
        )
        return hits[0]

    def assertNoDiagnosticMatching(self, diagnostics: list[str], *substrings: str) -> None:
        """Assert no diagnostic contains all ``substrings`` (same intersection semantics).

        Args:
            diagnostics: Validator-produced diagnostic strings.
            substrings: One or more substrings whose intersection must
                not appear in any diagnostic. With one substring this is
                "no diagnostic mentions ``s``"; with two or more it is
                "no diagnostic mentions all of these together".
        """
        hits = [d for d in diagnostics if all(s in d for s in substrings)]
        self.assertEqual(
            hits,
            [],
            msg=f"expected no diagnostics matching all of {substrings!r}, got {hits}",
        )
