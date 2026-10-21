# SPDX-License-Identifier: MIT
"""Guarded BF16/FP8 allocations for isolated CK regression/tuning subprocesses.

The C++ allocator reserves inaccessible space after each physical buffer and
selects high or low address bits for GLOBAL_LOAD_LDS or BUFFER_LOAD. Test processes exit
explicitly after synchronization; this also releases their HIP mappings.
"""

import argparse
import ctypes as C
import json
import os
from pathlib import Path

import torch

from aiter import dtypes, mha_batch_prefill_func, per_tensor_quant


class DLDevice(C.Structure):
    _fields_ = [("device_type", C.c_int), ("device_id", C.c_int)]


class DLDataType(C.Structure):
    _fields_ = [("code", C.c_uint8), ("bits", C.c_uint8), ("lanes", C.c_uint16)]


class DLTensor(C.Structure):
    _fields_ = [
        ("data", C.c_void_p),
        ("device", DLDevice),
        ("ndim", C.c_int),
        ("dtype", DLDataType),
        ("shape", C.POINTER(C.c_int64)),
        ("strides", C.POINTER(C.c_int64)),
        ("byte_offset", C.c_uint64),
    ]


class DLManagedTensor(C.Structure):
    pass


DELETER = C.CFUNCTYPE(None, C.POINTER(DLManagedTensor))
DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", C.c_void_p),
    ("deleter", DELETER),
]
keep_alive = []


def guarded_tensor(shape, library, dtype=torch.bfloat16, load_mode="global"):
    # DLPack's uint8 representation lets us expose FP8 bytes without depending
    # on a particular DLPack float8 type-code version.
    if dtype == torch.bfloat16:
        code, bits = 4, 16
    elif dtype == torch.uint8:
        code, bits = 1, 8
    else:
        raise ValueError(f"Unsupported guarded storage dtype: {dtype}")
    lib = C.CDLL(str(library))
    if load_mode not in {"global", "buffer"}:
        raise ValueError(f"Unsupported load mode: {load_mode}")
    allocate = getattr(
        lib,
        "allocate_guarded_hip"
        if load_mode == "global"
        else "allocate_guarded_hip_buffer",
    )
    allocate.argtypes = [C.c_size_t, C.c_int, C.POINTER(C.c_void_p)]
    allocate.restype = C.c_int
    n = 1
    for d in shape:
        n *= d
    ptr = C.c_void_p()
    status = allocate(n * (bits // 8), torch.cuda.current_device(), C.byref(ptr))
    assert status == 0, f"hipMem allocation error {status}"
    sizes = (C.c_int64 * len(shape))(*shape)
    strides = []
    stride = 1
    for d in reversed(shape):
        strides.insert(0, stride)
        stride *= d
    stride_arr = (C.c_int64 * len(shape))(*strides)
    deleter = DELETER(lambda _: None)
    managed = DLManagedTensor(
        DLTensor(
            ptr,
            DLDevice(10, torch.cuda.current_device()),
            len(shape),
            DLDataType(code, bits, 1),
            sizes,
            stride_arr,
            0,
        ),
        None,
        deleter,
    )
    keep_alive.extend([sizes, stride_arr, managed, deleter])
    C.pythonapi.PyCapsule_New.restype = C.py_object
    C.pythonapi.PyCapsule_New.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
    capsule = C.pythonapi.PyCapsule_New(C.addressof(managed), b"dltensor", None)
    return torch.utils.dlpack.from_dlpack(capsule)


def make_case(
    library,
    q_lens,
    kv_lens,
    qk_dim=192,
    v_dim=128,
    permute=False,
    cache_dtype="bf16",
    load_mode="global",
):
    torch.manual_seed(12345)
    assert len(q_lens) == len(kv_lens)
    assert all(0 < q <= k for q, k in zip(q_lens, kv_lens))
    q = torch.randn(sum(q_lens), 16, qk_dim, dtype=torch.bfloat16, device="cuda")
    k_ref = torch.randn(sum(kv_lens), 1, qk_dim, dtype=torch.bfloat16, device="cuda")
    v_ref = torch.randn(sum(kv_lens), 1, v_dim, dtype=torch.bfloat16, device="cuda")
    scales = {}
    if cache_dtype == "fp8":
        q, qs = per_tensor_quant(q, quant_dtype=dtypes.fp8)
        k_quant, ks = per_tensor_quant(k_ref * 0.7, quant_dtype=dtypes.fp8)
        v_quant, vs = per_tensor_quant(v_ref * 0.3, quant_dtype=dtypes.fp8)
        scales = {"q_descale": qs, "k_descale": ks, "v_descale": vs}
        k_ref = k_quant.float() * ks
        v_ref = v_quant.float() * vs
        k = guarded_tensor(
            (sum(kv_lens), 1, qk_dim), library, torch.uint8, load_mode
        ).view(dtypes.fp8)
        v = guarded_tensor(
            (sum(kv_lens), 1, v_dim), library, torch.uint8, load_mode
        ).view(dtypes.fp8)
    elif cache_dtype == "bf16":
        k = guarded_tensor((sum(kv_lens), 1, qk_dim), library, load_mode=load_mode)
        v = guarded_tensor((sum(kv_lens), 1, v_dim), library, load_mode=load_mode)
    else:
        raise ValueError(f"Unsupported cache dtype: {cache_dtype}")
    ids = (
        torch.randperm(sum(kv_lens), device="cuda")
        if permute
        else torch.arange(sum(kv_lens), device="cuda")
    ).to(torch.int32)
    if cache_dtype == "fp8":
        k.view(torch.uint8)[ids.long()] = k_quant.view(torch.uint8)
        v.view(torch.uint8)[ids.long()] = v_quant.view(torch.uint8)
    else:
        k[ids.long()] = k_ref
        v[ids.long()] = v_ref
    cuq = torch.tensor([0] + q_lens, device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    cuk = torch.tensor([0] + kv_lens, device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    sink = torch.linspace(-1, 1, 16, device="cuda", dtype=torch.float32)
    args = (q, k, v, cuq, cuk, ids, max(q_lens), max(kv_lens))
    return args, sink, k_ref, v_ref, scales


def reference(q, k, v, q_lens, kv_lens, sink, window):
    result = torch.empty(
        (*q.shape[:2], v.shape[-1]), device="cuda", dtype=torch.float32
    )
    q_offset = k_offset = 0
    for nq, nk in zip(q_lens, kv_lens):
        for start in range(0, nq, 128):
            stop = min(start + 128, nq)
            key_start = max(0, nk - nq + start - window) if window >= 0 else 0
            key_stop = nk - nq + stop
            qc = q[q_offset + start : q_offset + stop].transpose(0, 1).float()
            kc = k[k_offset + key_start : k_offset + key_stop].transpose(0, 1).float()
            vc = v[k_offset + key_start : k_offset + key_stop].transpose(0, 1).float()
            scores = qc @ kc.transpose(-1, -2) * q.shape[-1] ** -0.5
            qp = (nk - nq + torch.arange(start, stop, device="cuda"))[:, None]
            kp = torch.arange(key_start, key_stop, device="cuda")[None]
            mask = kp <= qp
            if window >= 0:
                mask &= kp >= qp - window
            scores.masked_fill_(~mask[None], -torch.inf)
            scores = torch.cat(
                (scores, sink[:, None, None].expand(-1, stop - start, 1)), -1
            )
            result[q_offset + start : q_offset + stop] = (
                scores.softmax(-1)[..., :-1] @ vc
            ).transpose(0, 1)
        q_offset += nq
        k_offset += nk
    return result


def check_accuracy(actual, expected, cache_dtype):
    tolerance = 0.055 if cache_dtype == "fp8" else 0.02
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)
    delta = actual.float() - expected
    relative_l2 = (delta.norm() / expected.norm().clamp_min(1e-12)).item()
    if cache_dtype == "fp8":
        # Also reject collapsed outputs when a long-context reference has a
        # small absolute magnitude. The reference already includes Q/K/V quantization.
        assert relative_l2 < 0.05, f"FP8 relative L2 error {relative_l2}"
    return {
        "max_abs_error": delta.abs().max().item(),
        "relative_l2_error": relative_l2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--q-lens", type=int, nargs="+", default=[3968])
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[16384])
    parser.add_argument("--qk-dim", type=int, default=192)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--permute", action="store_true")
    parser.add_argument("--cache-dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--load-mode", choices=["global", "buffer"], default="global")
    ns = parser.parse_args()
    args, sink, k_ref, v_ref, scales = make_case(
        ns.library,
        ns.q_lens,
        ns.kv_lens,
        ns.qk_dim,
        ns.v_dim,
        ns.permute,
        ns.cache_dtype,
        ns.load_mode,
    )
    torch.cuda.synchronize()
    print(
        f"K={hex(args[1].data_ptr())} V={hex(args[2].data_ptr())} q={ns.q_lens} kv={ns.kv_lens}",
        flush=True,
    )
    q_ref = args[0].float() * scales.get("q_descale", 1.0)
    expected = reference(q_ref, k_ref, v_ref, ns.q_lens, ns.kv_lens, sink, ns.window)
    # Reuse LDS across repeated launches, including mixed valid/padded lanes.
    for _ in range(3):
        actual = mha_batch_prefill_func(
            *args, causal=True, window_size=(ns.window, -1), sink_ptr=sink, **scales
        )
        torch.cuda.synchronize()
        errors = check_accuracy(actual, expected, ns.cache_dtype)
    print(
        json.dumps(
            {
                "passed": True,
                **errors,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
    # Do not tear down Python ctypes callbacks ahead of the DLPack tensors.
    os._exit(0)
