// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdlib>
#include <cstring>
#include <type_traits>
#include "opus/opus.hpp"

namespace aiter {

inline bool fp8_fn_compat_enabled()
{
    const char* mode = std::getenv("AITER_MIMO_RMSNORM_FP8_FN_COMPAT");
    return mode != nullptr && std::strcmp(mode, "1") == 0;
}

template <typename DType>
__device__ __forceinline__ float quant_inv_dtype_max(bool fn_compat)
{
#if defined(__gfx942__)
    if constexpr(std::is_same_v<DType, opus::fp8_t>)
    {
        // Half of the E4M3FN endpoint aligns the per-128 FNUZ grid with FN.
        if(fn_compat)
            return 1.0f / 224.0f;
    }
#endif
    return 1.0f / static_cast<float>(opus::finfo<DType>::max());
}

} // namespace aiter
