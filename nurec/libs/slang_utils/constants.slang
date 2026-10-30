// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace constants {

[ForceInline]
__generic<T: __BuiltinFloatingPointType>
T pi() { return (T)3.14159265358979323846264338327950288; }

} // namespace constants