# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU implementation of the gfx950 EP decode FP8 Block128 MoE.

Execution order: sort -> stage1_gate_up -> stage2_down -> optional reduce.
stage1_gate_up contains fused and split first-stage alternatives.
common contains only shared low-level primitives.
The public host entry lives one level above kernels/.
"""
