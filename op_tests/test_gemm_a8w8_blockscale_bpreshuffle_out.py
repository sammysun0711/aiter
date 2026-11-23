# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_mod.get_gfx() not in ("gfx942", "gfx950"),
    reason="the blockscale bpreshuffle asm kernels ship for gfx942 and gfx950",
)

N, K = 2560, 6144
PAD = 192
SENTINEL = 7.0
ASM_KERNEL = "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E"


def make_inputs(m, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(m, K, device="cuda") * 0.5).to(dtypes.fp8)
    w = (torch.randn(N, K, device="cuda") * 0.5).to(dtypes.fp8)
    # bytes column-major [K/128, M], shaped (M, K/128), as the per-1x128 quant emits
    x_scale = (
        (torch.rand(m, K // 128, device="cuda") + 0.5).t().contiguous().view(m, -1)
    )
    w_scale = torch.rand(N // 128, K // 128, device="cuda") + 0.5
    return x, shuffle_weight(w, layout=(16, 16)), x_scale, w_scale


def run_backend(backend, x, w, x_scale, w_scale, out):
    if backend == "ck":
        aiter.gemm_a8w8_blockscale_bpreshuffle_ck(x, w, x_scale, w_scale, out)
    elif backend == "cktile":
        aiter.gemm_a8w8_blockscale_bpreshuffle_cktile(x, w, x_scale, w_scale, out)
    else:
        split_k = int(backend.removeprefix("asm_sk"))
        aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
            x, w, out, x_scale, w_scale, splitK=split_k, kernelName=ASM_KERNEL
        )


@pytest.mark.parametrize("m", [32, 120])
@pytest.mark.parametrize("backend", ["ck", "cktile", "asm_sk1", "asm_sk4"])
def test_row_strided_out_matches_contiguous(backend, m):
    x, w, x_scale, w_scale = make_inputs(m)
    ref = torch.empty(m, N, dtype=dtypes.bf16, device="cuda")
    run_backend(backend, x, w, x_scale, w_scale, ref)

    wide = torch.full((m, N + PAD), SENTINEL, dtype=dtypes.bf16, device="cuda")
    out = wide[:, :N]
    run_backend(backend, x, w, x_scale, w_scale, out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=0.5, rtol=0.02)
    assert bool((wide[:, N:] == SENTINEL).all()), "wrote outside the out view"


@pytest.mark.parametrize("backend", ["ck", "asm_sk1", "asm_sk4"])
def test_single_row_out_ignores_size_one_stride(backend):
    x, w, x_scale, w_scale = make_inputs(1)
    ref = torch.empty(1, N, dtype=dtypes.bf16, device="cuda")
    run_backend(backend, x, w, x_scale, w_scale, ref)

    out = torch.empty(N, 1, dtype=dtypes.bf16, device="cuda").t()
    assert out.stride() == (1, 1)
    run_backend(backend, x, w, x_scale, w_scale, out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=0.5, rtol=0.02)


@pytest.mark.parametrize("backend", ["ck", "asm_sk1", "asm_sk4"])
def test_overlapping_out_rows_raise(backend):
    m = 32
    x, w, x_scale, w_scale = make_inputs(m)
    storage = torch.empty(m * N, dtype=dtypes.bf16, device="cuda")
    out = storage.as_strided((m, N), (N - 128, 1))
    with pytest.raises(RuntimeError):
        run_backend(backend, x, w, x_scale, w_scale, out)


def test_split_k_out_needs_four_byte_alignment():
    m = 32
    x, w, x_scale, w_scale = make_inputs(m)
    wide = torch.empty(m, N + 2, dtype=dtypes.bf16, device="cuda")
    with pytest.raises(RuntimeError):
        run_backend("asm_sk4", x, w, x_scale, w_scale, wide[:, 1 : N + 1])


def test_fp16_fallback_warns_once_per_shape(monkeypatch):
    asm_row = {"libtype": "asm", "splitK": 4, "kernelName": ASM_KERNEL}
    monkeypatch.setattr(gemm_mod, "get_CKGEMM_config", lambda *args, **kwargs: asm_row)
    messages = []
    monkeypatch.setattr(
        gemm_mod.logger, "warning", lambda msg, *a, **kw: messages.append(msg)
    )
    gemm_mod._warn_blockscale_asm_row_not_bf16.cache_clear()

    for m in (16, 32, 48):
        x, w, x_scale, w_scale = make_inputs(m)
        gemm_mod.gemm_a8w8_blockscale_bpreshuffle(x, w, x_scale, w_scale, dtypes.fp16)

    assert len([msg for msg in messages if "asm" in msg]) == 1


def test_fp16_output_on_asm_row_uses_ck(monkeypatch):
    m = 32
    x, w, x_scale, w_scale = make_inputs(m)
    asm_row = {"libtype": "asm", "splitK": 4, "kernelName": ASM_KERNEL}
    monkeypatch.setattr(gemm_mod, "get_CKGEMM_config", lambda *args, **kwargs: asm_row)
    asm_calls = []
    real_asm = gemm_mod.gemm_a8w8_blockscale_bpreshuffle_asm

    def spy_asm(*args, **kwargs):
        asm_calls.append(kwargs.get("splitK"))
        return real_asm(*args, **kwargs)

    monkeypatch.setattr(gemm_mod, "gemm_a8w8_blockscale_bpreshuffle_asm", spy_asm)

    ref = gemm_mod.gemm_a8w8_blockscale_bpreshuffle(x, w, x_scale, w_scale, dtypes.bf16)
    assert asm_calls == [4], "bf16 output should keep using the tuned asm row"

    y = gemm_mod.gemm_a8w8_blockscale_bpreshuffle(x, w, x_scale, w_scale, dtypes.fp16)
    torch.cuda.synchronize()
    assert y.dtype == dtypes.fp16
    assert asm_calls == [4], "fp16 output must not reach the bf16-only asm kernel"
    torch.testing.assert_close(y.float(), ref.float(), atol=0.5, rtol=0.02)


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_mod.get_gfx() != "gfx942",
    reason="CDNA3 FNUZ kernel",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m,n,k,tile,variant",
    [(128, 256, 384, "128x128", ""), (32, 192, 128, "32x64", "")]
    + [
        (128, 256, k, "128x128", "_" + v)
        for k in (128, 256, 384)
        for v in ("lds_m", "lds_n", "reg_a")
    ]
    # The 2x4 LDS kernel also serves partial M/N tiles and M = 1.
    + [
        (m, n, k, "256x256", "_2x4_lds_pf1")
        for m, n, k in ((128, 256, 384), (257, 272, 256), (1, 528, 128))
    ],
)
def test_gfx942_flydsl_packed_out(monkeypatch, dtype, m, n, k, tile, variant):
    torch.manual_seed(942)
    # A view may be contiguous while missing the variant's load alignment.
    offset = 8 if variant else 1
    x = torch.randn(m * k + offset, device="cuda").to(dtypes.fp8)[offset:].view(m, k)
    w = torch.randn(n, k, device="cuda").to(dtypes.fp8)
    sx = torch.rand(m, k // 128, device="cuda") + 0.25
    sw = torch.rand((n + 127) // 128, k // 128, device="cuda") + 0.25
    ref = (x.double() * sx.repeat_interleave(128, 1)) @ (
        w.double() * sw.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    ).T
    outer = torch.full((m, n + 8), SENTINEL, dtype=dtype, device="cuda")
    out = outer[:, :n]
    out.fill_(float("nan"))
    row = {
        "libtype": "flydsl",
        "kernelName": f"flydsl_blockscale_gfx942_{tile}x128_ps1{variant}",
    }
    monkeypatch.setattr(gemm_mod, "get_CKGEMM_config", lambda *a, **kw: row)

    def unexpected_ck(*args, **kwargs):
        pytest.fail("a tuned FlyDSL row dispatched to CK")

    monkeypatch.setattr(gemm_mod, "gemm_a8w8_blockscale_bpreshuffle_ck", unexpected_ck)
    y = gemm_mod.gemm_a8w8_blockscale_bpreshuffle(
        x, shuffle_weight(w, layout=(16, 16)), sx.T.contiguous().T, sw, dtype, out=out
    )
    assert y.data_ptr() == out.data_ptr()
    assert bool((outer[:, n:] == SENTINEL).all())
    assert checkAllclose(ref.float(), out.float(), rtol=0.01, atol=0.01) == 0

    if variant in ("_reg_a", "_2x4_lds_pf1"):
        # The vector epilogue also accepts a contiguous but offset output view.
        storage = torch.full((m * n + 1,), SENTINEL, dtype=dtype, device="cuda")
        offset_out = storage[1:].view(m, n)
        y = gemm_mod.gemm_a8w8_blockscale_bpreshuffle(
            x,
            shuffle_weight(w, layout=(16, 16)),
            sx.T.contiguous().T,
            sw,
            dtype,
            out=offset_out,
        )
        assert y.data_ptr() == offset_out.data_ptr() and storage[0].item() == SENTINEL
        assert checkAllclose(ref.float(), y.float(), rtol=0.01, atol=0.01) == 0


def test_gfx942_flydsl_padded_lookup(tmp_path):
    table = tmp_path / "flydsl.csv"
    table.write_text(
        "gfx,cu_num,M,N,K,libtype,kernelName\n"
        "gfx942,304,128,256,384,flydsl,flydsl_blockscale_gfx942_128x128x128_ps1\n"
    )

    def lookup(m):
        return gemm_mod.get_CKGEMM_config(
            m, 256, 384, str(table), gfx="gfx942", cu_num=304
        )

    assert lookup(128)["libtype"] == "flydsl"
    assert lookup(127) is None


def test_gfx942_flydsl_2x4_lds_padded_lookup(tmp_path):
    # The 2x4 LDS kernel range-checks M, so a row tuned at a padded M serves the smaller M too.
    name = "flydsl_blockscale_gfx942_256x256x128_ps1_2x4_lds_pf1"
    table = tmp_path / "flydsl.csv"
    table.write_text(
        f"gfx,cu_num,M,N,K,libtype,kernelName\ngfx942,304,128,256,384,flydsl,{name}\n"
    )

    def lookup(m):
        return gemm_mod.get_CKGEMM_config(
            m, 256, 384, str(table), gfx="gfx942", cu_num=304
        )

    assert lookup(128)["kernelName"] == name
    assert lookup(127)["kernelName"] == name


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_mod.get_gfx() != "gfx942",
    reason="CDNA3 FNUZ kernel",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,n,k", [(128, 256, 384), (257, 272, 256), (1, 528, 128)])
def test_gfx942_flydsl_2x4_lds_plain(monkeypatch, dtype, m, n, k):
    # Plain B: gemm_a8w8_blockscale takes row-major x_scale[M, K/128].
    torch.manual_seed(942)
    x = torch.randn(m, k, device="cuda").to(dtypes.fp8)
    w = torch.randn(n, k, device="cuda").to(dtypes.fp8)
    sx = torch.rand(m, k // 128, device="cuda") + 0.25
    sw = torch.rand((n + 127) // 128, k // 128, device="cuda") + 0.25
    ref = (x.double() * sx.repeat_interleave(128, 1)) @ (
        w.double() * sw.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    ).T
    row = {
        "libtype": "flydsl",
        "kernelName": "flydsl_blockscale_gfx942_256x256x128_ps0_2x4_lds_pf1",
    }
    monkeypatch.setattr(gemm_mod, "get_CKGEMM_config", lambda *a, **kw: row)

    def unexpected_ck(*args, **kwargs):
        pytest.fail("a tuned FlyDSL row dispatched to CK")

    monkeypatch.setattr(gemm_mod, "gemm_a8w8_blockscale_ck", unexpected_ck)
    y = gemm_mod.gemm_a8w8_blockscale(x, w, sx, sw, dtype)
    assert y.dtype == dtype and tuple(y.shape) == (m, n)
    assert checkAllclose(ref.float(), y.float(), rtol=0.01, atol=0.01) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
