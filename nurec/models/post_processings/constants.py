# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPISP constants.

Also runs as a script to emit the -D flags the PPISP Slang module is compiled with;
the constants_txt target captures that output.
"""

# PPISP constants
NUM_VIGNETTING_ALPHA_TERMS = 3
DEFAULT_SOURCE_CHROMS_VALUES: list[list[float]] = [
    [0.0, 0.0],  # pure blue
    [1.0, 0.0],  # pure red
    [0.0, 1.0],  # pure green
    [1 / 3, 1 / 3],  # neutral gray
]
NUM_SOURCE_CHROMS = len(DEFAULT_SOURCE_CHROMS_VALUES)
EPSILON = 1e-5


def main():
    """Print the constants used by ppisp_legacy.slang as -D flags for slangc."""
    defines = [
        f"-DNUM_VIGNETTING_ALPHA_TERMS={NUM_VIGNETTING_ALPHA_TERMS}",
        f"-DEPSILON={EPSILON}",
    ]

    print(" ".join(defines))


if __name__ == "__main__":
    main()
