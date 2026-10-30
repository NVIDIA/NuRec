// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Helper variadic template to be able to use lambda expressions in std::visit
template <class... Ts>
struct OverloadVisitor : Ts... {
    using Ts::operator()...;
};

template <class... Ts>
OverloadVisitor(Ts...) -> OverloadVisitor<Ts...>;
