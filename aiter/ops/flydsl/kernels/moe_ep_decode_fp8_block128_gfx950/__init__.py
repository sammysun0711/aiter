# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP decode FP8 Block128 kernels.

Execution: sort -> stage1 -> stage2_down -> optional reduce.
Stage 1 is either fused_fp8_quant (one GEMM/SiLU/FP8-quant kernel) or
separate_fp8_quant (GEMM to FP32 workspace, then SiLU/FP8-quant). Shared arithmetic and the routing ABI live in common.py;
configuration.py owns canonical host choices and AITER CSV decoding.
"""
