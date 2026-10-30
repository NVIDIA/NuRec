# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

from unittest import mock

from nurec.config.logger import GENERATE, BaseLoggerConfig


class RunIdEnvTest(unittest.TestCase):
    def test_nurec_env_run_id_wins(self) -> None:
        with mock.patch.dict(os.environ, {"NUREC_ENV_RUN_ID": "new", "NRE_ENV_RUN_ID": "old"}):
            with self.assertNoLogs("nurec.config.logger", level="WARNING"):
                self.assertEqual(BaseLoggerConfig().run_id, "new")

    def test_nre_env_run_id_falls_back_with_deprecation_warning(self) -> None:
        with mock.patch.dict(os.environ, {"NRE_ENV_RUN_ID": "old"}):
            os.environ.pop("NUREC_ENV_RUN_ID", None)
            with self.assertLogs("nurec.config.logger", level="WARNING") as logs:
                self.assertEqual(BaseLoggerConfig().run_id, "old")
        self.assertIn("NRE_ENV_RUN_ID is deprecated", logs.output[0])

    def test_generated_without_env(self) -> None:
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("NUREC_ENV_RUN_ID", None)
            os.environ.pop("NRE_ENV_RUN_ID", None)
            self.assertNotEqual(BaseLoggerConfig().run_id, GENERATE)
            self.assertNotEqual(BaseLoggerConfig().run_id, BaseLoggerConfig().run_id)


if __name__ == "__main__":
    unittest.main()
