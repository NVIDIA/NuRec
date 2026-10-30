# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for auxiliary-data generation."""

from apps.av.aux_gen.ncore_aux_data import cli


if __name__ == "__main__":
    cli()
