# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""DFlash-shaped coverage for native FlyDSL paged BF16 SWA."""

import itertools
import os

import pytest
import torch

from aiter.ops.flydsl import flydsl_paged_attention_swa_bf16
from aiter.ops.flydsl.kernels.mha_pa_swa_bf16 import (
    _static_compile_signature,
)

PAGE_SIZE = 64
HEAD_DIM = 128
VALUE_HEAD_DIM = 128
WINDOW_LEFT = 1023


def test_static_compile_signature_ignores_dynamic_extents():
    stream = type("FakeStream", (), {"cuda_stream": 1})()

    def make_args(query_tokens, page_indices, max_q, batch=1, dtype=torch.bfloat16):
        return (
            torch.empty(query_tokens, dtype=dtype),
            torch.empty(8, 1, 24, 64, 8, dtype=torch.bfloat16),
            torch.empty(8, 1, 8, 128, 8, dtype=torch.bfloat16),
            torch.empty(query_tokens, dtype=torch.bfloat16),
            torch.empty(query_tokens, dtype=torch.float32),
            torch.empty(batch + 1, dtype=torch.int32),
            torch.empty(batch + 1, dtype=torch.int32),
            torch.empty(page_indices, dtype=torch.int32),
            torch.empty(batch, dtype=torch.int32),
            torch.empty(query_tokens, dtype=torch.float32),
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, dtype=torch.float32),
            torch.empty(16, dtype=torch.float32),
            16,
            1,
            8,
            batch,
            max_q,
            192,
            3072,
            192,
            2048,
            128,
            127,
            True,
            True,
            False,
            192**-0.5,
            32,
            32,
            950,
            stream,
        )

    first_tail = make_args(32683, 1024, 32683)
    second_tail = make_args(32636, 1023, 32636)
    assert _static_compile_signature(first_tail) == _static_compile_signature(
        second_tail
    )

    different_query_length = make_args(65536, 2048, 65536)
    assert _static_compile_signature(first_tail) == _static_compile_signature(
        different_query_length
    )

    different_batch = make_args(32683, 1024, 32683, batch=2)
    assert _static_compile_signature(first_tail) == _static_compile_signature(
        different_batch
    )

    different_dtype = make_args(32683, 1024, 32683, dtype=torch.float16)
    assert _static_compile_signature(first_tail) != _static_compile_signature(
        different_dtype
    )


def test_short_q_default_tile_selection(monkeypatch):
    from aiter.ops.flydsl import fmha_kernels

    selected = []

    class FakeOp:
        def __call__(self, *args, **kwargs):
            return kwargs["out"]

    def fake_get(*args):
        selected.append((args[-2], args[-1]))
        return FakeOp()

    monkeypatch.setattr(fmha_kernels, "_get_paged_attention_swa_bf16", fake_get)

    q = torch.empty(8, 16, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 1, 16, 64, 8, dtype=torch.bfloat16)
    v = torch.empty(1, 1, 8, 128, 8, dtype=torch.bfloat16)
    i32 = lambda values: torch.tensor(values, dtype=torch.int32)
    one = torch.ones(1, dtype=torch.float32)
    sinks = torch.zeros(16, dtype=torch.float32)
    out = torch.empty_like(q)
    common = {
        "window_left": WINDOW_LEFT,
        "kv_last_page_lens": i32([64]),
        "q_descale": one,
        "k_descale": one,
        "v_descale": one,
        "sink_ptr": sinks,
        "out": out,
    }

    fmha_kernels.flydsl_paged_attention_swa_bf16(
        q, k, v, i32([0, 8]), i32([0, 1]), i32([0]), 8, 64, **common
    )
    fmha_kernels.flydsl_paged_attention_swa_bf16(
        q,
        k,
        v,
        i32([0, 8]),
        i32([0, 1]),
        i32([0]),
        8,
        64,
        block_n=32,
        query_tile=32,
        **common,
    )

    assert selected == [(64, 16), (32, 32)]


def _gpu_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]


requires_native_swa = pytest.mark.skipif(
    _gpu_arch() not in ("gfx942", "gfx950"),
    reason="native FlyDSL paged BF16 SWA requires gfx942 or gfx950",
)
requires_large_cache_test = pytest.mark.skipif(
    os.environ.get("AITER_RUN_LARGE_CACHE_TESTS", "0") != "1",
    reason="set AITER_RUN_LARGE_CACHE_TESTS=1 for the >2**31-element cache test",
)


def _make_case(head_dim=HEAD_DIM, q_lens=(4, 3), kv_lens=(4097, 2051)):
    torch.manual_seed(20260917)
    num_q_heads, num_kv_heads = 16, 1
    page_counts = tuple((length + PAGE_SIZE - 1) // PAGE_SIZE for length in kv_lens)
    num_pages = sum(page_counts)

    q = (torch.randn(sum(q_lens), num_q_heads, head_dim, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    k_linear = (
        torch.randn(num_pages, PAGE_SIZE, num_kv_heads, head_dim, device="cuda") * 0.2
    ).to(torch.bfloat16)
    v_linear = (
        torch.randn(num_pages, PAGE_SIZE, num_kv_heads, VALUE_HEAD_DIM, device="cuda")
        * 0.2
    ).to(torch.bfloat16)
    k = (
        k_linear.view(num_pages, PAGE_SIZE, num_kv_heads, head_dim // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v = (
        v_linear.view(num_pages, PAGE_SIZE // 8, 8, num_kv_heads, VALUE_HEAD_DIM)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )

    cu_seqlens_q = torch.tensor(
        [0, *itertools.accumulate(q_lens)], device="cuda", dtype=torch.int32
    )
    kv_indptr = torch.tensor(
        [0, *itertools.accumulate(page_counts)], device="cuda", dtype=torch.int32
    )
    kv_page_indices = torch.arange(num_pages, device="cuda", dtype=torch.int32)
    kv_last_page_lens = torch.tensor(
        [(length - 1) % PAGE_SIZE + 1 for length in kv_lens],
        device="cuda",
        dtype=torch.int32,
    )
    one = torch.ones(1, device="cuda", dtype=torch.float32)
    sinks = torch.linspace(-0.4, 0.3, num_q_heads, device="cuda", dtype=torch.float32)
    out = torch.empty(
        sum(q_lens), num_q_heads, VALUE_HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "k_linear": k_linear,
        "v_linear": v_linear,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "page_counts": page_counts,
        "cu_seqlens_q": cu_seqlens_q,
        "kv_indptr": kv_indptr,
        "kv_page_indices": kv_page_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "one": one,
        "sinks": sinks,
        "out": out,
    }


def _reference(case):
    outputs = []
    q_offset = 0
    page_offset = 0
    num_q_heads = case["q"].shape[1]
    num_kv_heads = case["k_linear"].shape[2]
    head_dim = case["q"].shape[-1]
    scale = head_dim**-0.5

    for q_len, kv_len, page_count in zip(
        case["q_lens"], case["kv_lens"], case["page_counts"]
    ):
        q = case["q"][q_offset : q_offset + q_len].float()
        q_offset += q_len
        k = (
            case["k_linear"][page_offset : page_offset + page_count]
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .float()
        )
        v = (
            case["v_linear"][page_offset : page_offset + page_count]
            .reshape(-1, num_kv_heads, VALUE_HEAD_DIM)[:kv_len]
            .float()
        )
        page_offset += page_count
        k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q, k) * scale
        q_positions = torch.arange(kv_len - q_len, kv_len, device=q.device)
        k_positions = torch.arange(kv_len, device=q.device)
        visible = (k_positions[None, :] <= q_positions[:, None]) & (
            k_positions[None, :] >= (q_positions - WINDOW_LEFT)[:, None]
        )
        scores.masked_fill_(~visible.unsqueeze(0), float("-inf"))
        sink_scores = case["sinks"][:, None, None].expand(num_q_heads, q_len, 1)
        probabilities = torch.softmax(torch.cat((scores, sink_scores), dim=-1), dim=-1)[
            ..., :-1
        ]
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, v))

    return torch.cat(outputs)


def _run(case):
    return flydsl_paged_attention_swa_bf16(
        case["q"],
        case["k"],
        case["v"],
        case["cu_seqlens_q"],
        case["kv_indptr"],
        case["kv_page_indices"],
        max(case["q_lens"]),
        max(case["kv_lens"]),
        window_left=WINDOW_LEFT,
        kv_last_page_lens=case["kv_last_page_lens"],
        q_descale=case["one"],
        k_descale=case["one"],
        v_descale=case["one"],
        sink_ptr=case["sinks"],
        out=case["out"],
    )


def _run_op(op, case):
    return op(
        case["q"],
        case["k"],
        case["v"],
        case["cu_seqlens_q"],
        None,
        case["kv_indptr"],
        case["kv_page_indices"],
        max(case["q_lens"]),
        max(case["kv_lens"]),
        True,
        case["one"],
        case["one"],
        case["one"],
        case["kv_last_page_lens"],
        out=case["out"],
        sink_ptr=case["sinks"],
    )


@requires_native_swa
def test_compile_cache_reuses_runtime_query_dimensions(monkeypatch):
    import aiter.ops.flydsl.kernels.mha_pa_swa_bf16 as swa

    swa.PagedAttention.cache_clear()
    compile_calls = []
    real_compile = swa.flyc.compile

    def counted_compile(*args, **kwargs):
        compile_calls.append(args)
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(swa.flyc, "compile", counted_compile)
    op = swa.PagedAttention(
        16,
        1,
        HEAD_DIM,
        VALUE_HEAD_DIM,
        PAGE_SIZE,
        window_left=WINDOW_LEFT,
        has_sink=True,
    )

    for query_len in (17, 23, 33):
        case = _make_case(q_lens=(query_len,), kv_lens=(65,))
        expected = _reference(case)
        actual = _run_op(op, case)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)

    batch_two = _make_case(q_lens=(9, 8), kv_lens=(33, 32))
    expected = _reference(batch_two)
    actual = _run_op(op, batch_two)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)

    assert len(compile_calls) == 1


@requires_native_swa
@pytest.mark.parametrize("head_dim", [128, 192])
def test_dflash_window_sink_and_graph_replay(head_dim):
    case = _make_case(head_dim)
    expected = _reference(case)

    actual = _run(case)
    torch.cuda.synchronize()
    assert actual.data_ptr() == case["out"].data_ptr()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
    eager = actual.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = _run(case)
    graph.replay()
    torch.cuda.synchronize()

    assert graph_out.data_ptr() == case["out"].data_ptr()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)


@requires_native_swa
@requires_large_cache_test
def test_page_relative_addressing_over_signed_int32_elements():
    # FlyDSL 0.3.2 encodes dynamic tensor dimensions as signed int32. Keep K/V
    # rank-5 so each dimension remains representable even when the contiguous
    # backing allocation contains more than 2**31 elements.
    num_pages = 262145
    num_q_heads, num_kv_heads = 16, 1
    q = torch.randn(1, num_q_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(
        num_pages,
        num_kv_heads,
        HEAD_DIM // 8,
        PAGE_SIZE,
        8,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.empty(
        num_pages,
        num_kv_heads,
        PAGE_SIZE // 8,
        VALUE_HEAD_DIM,
        8,
        device="cuda",
        dtype=torch.bfloat16,
    )
    assert k.numel() > 2**31
    assert v.numel() > 2**31
    k[-1].normal_()
    v[-1].normal_()

    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    kv_page_indices = torch.tensor([num_pages - 1], device="cuda", dtype=torch.int32)
    kv_last_page_lens = torch.tensor([PAGE_SIZE], device="cuda", dtype=torch.int32)
    one = torch.ones(1, device="cuda", dtype=torch.float32)
    sinks = torch.zeros(num_q_heads, device="cuda", dtype=torch.float32)
    out = torch.empty_like(q)

    flydsl_paged_attention_swa_bf16(
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        1,
        PAGE_SIZE,
        window_left=WINDOW_LEFT,
        kv_last_page_lens=kv_last_page_lens,
        q_descale=one,
        k_descale=one,
        v_descale=one,
        sink_ptr=sinks,
        out=out,
    )
    torch.cuda.synchronize()

    k_linear = k[-1].permute(2, 0, 1, 3).reshape(PAGE_SIZE, num_kv_heads, HEAD_DIM)
    v_linear = (
        v[-1].permute(1, 3, 0, 2).reshape(PAGE_SIZE, num_kv_heads, VALUE_HEAD_DIM)
    )
    k_linear = k_linear.float().repeat_interleave(num_q_heads, dim=1)
    v_linear = v_linear.float().repeat_interleave(num_q_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k_linear) * HEAD_DIM**-0.5
    sink_scores = sinks[:, None, None]
    probabilities = torch.softmax(torch.cat((scores, sink_scores), dim=-1), dim=-1)[
        ..., :-1
    ]
    expected = torch.einsum("hqk,khd->qhd", probabilities, v_linear)
    torch.testing.assert_close(out.float(), expected, rtol=0.02, atol=0.02)
