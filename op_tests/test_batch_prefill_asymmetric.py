import math

import pytest
import torch

import aiter
from aiter import dtypes, per_tensor_quant


def _vectorize_k(
    k: torch.Tensor, page_size: int, vector_width: int
) -> torch.Tensor:
    num_pages, _, num_heads, head_dim = k.shape
    return (
        k.view(
            num_pages,
            page_size,
            num_heads,
            head_dim // vector_width,
            vector_width,
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )


def _vectorize_v(
    v: torch.Tensor, page_size: int, vector_width: int
) -> torch.Tensor:
    num_pages, _, num_heads, value_head_dim = v.shape
    return (
        v.view(
            num_pages,
            page_size // vector_width,
            vector_width,
            num_heads,
            value_head_dim,
        )
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "cache_dtype,layout,use_swa_sink",
    [
        ("bf16", "linear", False),
        ("fp8", "linear", False),
        ("fp8", "vectorized", False),
        ("bf16", "linear", True),
        ("fp8", "linear", True),
        ("bf16", "vectorized", False),
        ("bf16", "vectorized", True),
        ("fp8", "vectorized", True),
    ],
)
def test_batch_prefill_qk192_v128(cache_dtype, layout, use_swa_sink):
    torch.manual_seed(20260817)
    qo_len = 128
    kv_len = 256
    num_q_heads = 16
    num_kv_heads = 1
    qk_head_dim = 192
    value_head_dim = 128
    page_size = 64
    num_pages = math.ceil(kv_len / page_size)

    q_bf16 = torch.randn(
        qo_len,
        num_q_heads,
        qk_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.1
    k_bf16 = torch.randn(
        kv_len,
        num_kv_heads,
        qk_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.1
    v_bf16 = torch.randn(
        kv_len,
        num_kv_heads,
        value_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.1

    if cache_dtype == "fp8":
        q, q_descale = per_tensor_quant(q_bf16, quant_dtype=dtypes.fp8)
        k, k_descale = per_tensor_quant(k_bf16, quant_dtype=dtypes.fp8)
        v, v_descale = per_tensor_quant(v_bf16, quant_dtype=dtypes.fp8)
        q_ref = q.float() * q_descale
        k_ref = k.float() * k_descale
        v_ref = v.float() * v_descale
    else:
        q, k, v = q_bf16, k_bf16, v_bf16
        q_descale = k_descale = v_descale = None
        q_ref, k_ref, v_ref = q.float(), k.float(), v.float()

    cu_seqlens_q = torch.tensor([0, qo_len], dtype=torch.int32, device="cuda")
    if layout == "vectorized":
        k_pages = k.view(num_pages, page_size, num_kv_heads, qk_head_dim)
        v_pages = v.view(num_pages, page_size, num_kv_heads, value_head_dim)
        vector_width = 16 // k.element_size()
        k_kernel = _vectorize_k(k_pages, page_size, vector_width)
        v_kernel = _vectorize_v(v_pages, page_size, vector_width)
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
        kv_indices = torch.nn.functional.pad(
            torch.arange(num_pages, dtype=torch.int32), (0, 256)
        ).to("cuda")
        kv_last_page_lens = torch.tensor(
            [page_size], dtype=torch.int32, device="cuda"
        )
    else:
        k_kernel, v_kernel = k, v
        kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")
        kv_indices = torch.arange(kv_len, dtype=torch.int32, device="cuda")
        kv_last_page_lens = None

    window_left = 128 if use_swa_sink else -1
    sink_ptr = (
        torch.linspace(
            -1.0,
            -0.25,
            num_q_heads,
            dtype=torch.float32,
            device="cuda",
        )
        if use_swa_sink
        else None
    )

    output = aiter.mha_batch_prefill_func(
        q,
        k_kernel,
        v_kernel,
        cu_seqlens_q,
        kv_indptr,
        kv_indices,
        max_seqlen_q=qo_len,
        max_seqlen_k=kv_len,
        causal=True,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_last_page_lens=kv_last_page_lens,
        window_size=(window_left, -1),
        sink_ptr=sink_ptr,
    )

    k_ref = k_ref.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    v_ref = v_ref.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    rows = torch.arange(qo_len, device="cuda").unsqueeze(1)
    cols = torch.arange(kv_len, device="cuda").unsqueeze(0)
    absolute_query_positions = kv_len - qo_len + rows
    causal_mask = cols <= absolute_query_positions
    if window_left >= 0:
        causal_mask &= cols >= absolute_query_positions - window_left

    logits = torch.einsum("qhd,khd->hqk", q_ref, k_ref) / math.sqrt(
        qk_head_dim
    )
    logits.masked_fill_(~causal_mask.unsqueeze(0), float("-inf"))
    if sink_ptr is not None:
        virtual_sink = sink_ptr.view(num_q_heads, 1, 1).expand(-1, qo_len, 1)
        probabilities = torch.softmax(
            torch.cat((logits, virtual_sink), dim=-1), dim=-1
        )[..., :-1]
    else:
        probabilities = torch.softmax(logits, dim=-1)
    reference = torch.einsum("hqk,khd->qhd", probabilities, v_ref)

    assert output.shape == (qo_len, num_q_heads, value_head_dim)
    assert torch.isfinite(output).all()
    tolerance = 0.055 if cache_dtype == "fp8" else 0.025
    torch.testing.assert_close(
        output.float(), reference.float(), rtol=tolerance, atol=tolerance
    )
