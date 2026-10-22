# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""MXFP4 weight conversion for FP8 blockscale GEMM and MoE."""

import torch


def mxfp4_to_fp8_128block(packed_w, scales, *, out=None, out_scale=None):
    """Return FP8 Q[E,N,K] and FP32 S[E,ceil(N/128),ceil(K/128)].

    Contiguous ROCm uint8 inputs: packed E2M1 [E,N,K/2] and E8M0
    [E,N,K/32], positive dimensions, K divisible by 32. Partial N/K tiles
    use only valid weights. gfx942 produces E4M3FNUZ; gfx950 E4M3FN.

    Like per_oc, use the conservative bound B=6*max(finite E8M0 scales),
    now over each 128x128 tile, and S=max(2**ceil(log2(B/FP8_MAX)), 2**-126).
    This is a bound from input scales, not the actual weight amax. Exponent
    arithmetic avoids FP32 overflow. Q=round_fp8(W/S); reconstruct using
    Q[e,n,k]*S[e,n//128,k//128]. In GEMM, rescale each K tile's partial sum
    before accumulation; S cannot be applied once after the complete dot.

    Finite inputs produce finite FP8 (magnitude <=192 FNUZ / <=384 FN).
    Tiny relative values may round to zero. Rounding is nearest-even.
    E8M0 255 produces NaNs only in its own K=32 group and is excluded from
    the maximum. All-NaN tiles return S=2**-126. Scales are normal finite
    positive FP32; reconstructing extreme weights can still overflow FP32.

    Optional outputs must have the documented shapes, dtype and device,
    be contiguous, four-byte aligned, and share no storage with inputs or
    each other. Preallocate both outputs for allocation-free graph replay.
    The launcher runs scale reduction and contiguous conversion on the same
    stream; out_scale is also the intermediate buffer, with no extra scratch.
    """
    if packed_w.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("packed_w and scales must have dtype uint8")
    if (
        not torch.version.hip
        or not packed_w.is_cuda
        or scales.device != packed_w.device
    ):
        raise ValueError("packed_w and scales must be on the same ROCm GPU")
    if not packed_w.is_contiguous() or not scales.is_contiguous():
        raise ValueError("packed_w and scales must be contiguous")
    if packed_w.ndim != 3 or min(packed_w.shape) <= 0 or packed_w.shape[-1] % 16:
        raise ValueError(
            "Expected packed_w[E,N,K/2] with positive dimensions and K divisible by 32"
        )
    e, n, half_k = packed_w.shape
    if tuple(scales.shape) != (e, n, half_k // 16):
        raise ValueError("Expected scales[E,N,K/32]")
    if packed_w.data_ptr() % 4:
        raise ValueError("packed_w must be aligned to four bytes")
    arch = torch.cuda.get_device_properties(packed_w.device).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        raise ValueError("Requires gfx942 or gfx950")
    fnuz = arch == "gfx942"
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    shape = (e, n, half_k * 2)
    scale_shape = (e, (n + 127) // 128, (half_k * 2 + 127) // 128)
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=packed_w.device)
    if out_scale is None:
        out_scale = torch.empty(
            scale_shape, dtype=torch.float32, device=packed_w.device
        )
    for name, tensor, expected_shape, expected_dtype in (
        ("out", out, shape, dtype),
        ("out_scale", out_scale, scale_shape, torch.float32),
    ):
        if (
            tuple(tensor.shape) != expected_shape
            or tensor.dtype != expected_dtype
            or tensor.device != packed_w.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous with the expected shape, dtype and device"
            )
        others = (packed_w, scales, out_scale if name == "out" else out)
        if tensor.data_ptr() % 4 or any(
            tensor.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in others
        ):
            raise ValueError(
                f"{name} must be aligned to four bytes and must not alias inputs or the other output"
            )
    from aiter.ops.flydsl.kernels.mxfp4_fp8_128block import (
        compile_mxfp4_to_fp8_128block,
    )

    with torch.cuda.device(packed_w.device):
        launch = compile_mxfp4_to_fp8_128block(experts=e, n=n, k=half_k * 2, fnuz=fnuz)
        launch(
            packed_w.data_ptr(),
            scales.data_ptr(),
            out.data_ptr(),
            out_scale.data_ptr(),
            torch.cuda.current_stream(),
        )
    return out, out_scale
