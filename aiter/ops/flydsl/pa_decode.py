# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Full-attention PA decode with BF16 or FP8 vectorized KV caches.

Packed query/output rows have a uniform query length per request. Context
lengths include those query tokens; masking is bottom-right causal.
Small physical pages (16/64) are supported. The optimized gfx950 Q8 path
uses GQA16, page64 and D128/V128 or D192/V128.

Metadata and KV caches must be dense on the query device. Pass GPU scalar
FP8 scales and preallocated partials before graph capture. Explicit NP16 is
recommended for the measured B96/C65536 BF16 workload; automatic splits retain
the existing cap of eight. FP8 storage uses BF16 computation for contexts <=512
on the optimized Q8 path. No SWA, sinks or variable per-request Q lengths.

This module does not change AITER's existing attention dispatcher.
"""

import torch
from packaging.version import Version

from . import _base_version

if _base_version < Version("0.3.2"):
    raise ImportError("FlyDSL PA decode requires flydsl >= 0.3.2")

from flydsl.runtime.device import get_rocm_arch

from .kernels.pa_decode_common import cdiv
from .kernels.pa_decode_tile import compile_pa_decode_tile
from .kernels.tensor_shim import _run_compiled

KV_COMPUTE_BLOCK = 256
_PA_DECODE_PS_SMALL_BLOCK_SIZES = (16, 64)


def _is_current_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def _prepare_scale_tensor(
    name: str,
    scale,
    *,
    device: torch.device,
    is_graph_capturing: bool,
) -> torch.Tensor:
    if isinstance(scale, torch.Tensor):
        if is_graph_capturing:
            if scale.device != device:
                raise ValueError(
                    f"CUDA graph capture requires `{name}` to already be on {device}, "
                    f"got {scale.device}."
                )
            if scale.dtype != torch.float32:
                raise ValueError(
                    f"CUDA graph capture requires `{name}` to already be float32, "
                    f"got {scale.dtype}."
                )
            return scale
        return scale.to(device=device, dtype=torch.float32)

    if is_graph_capturing:
        raise ValueError(
            f"CUDA graph capture requires `{name}` to be passed as a pre-created "
            "float32 tensor on the target device."
        )

    return torch.tensor(
        [float(1.0 if scale is None else scale)], device=device, dtype=torch.float32
    )


def _require_stream_scale_tensors(device, key_scale, value_scale):
    for name, scale in (("key_scale", key_scale), ("value_scale", value_scale)):
        if (
            not isinstance(scale, torch.Tensor)
            or scale.device != device
            or scale.dtype != torch.float32
        ):
            raise ValueError(
                f"Explicit-stream PA decode requires pre-created float32 {name} on {device}"
            )


def _get_output_dtype_str(output: torch.Tensor) -> str:
    if output.dtype == torch.bfloat16:
        return "bf16"
    if output.dtype == torch.float16:
        return "f16"
    if output.dtype == torch.float32:
        return "f32"
    raise ValueError(
        f"Unsupported output dtype for pa_decode_ps_launch reduce: {output.dtype}. "
        "Expected bf16, f16, or f32."
    )


def get_recommended_splits(
    num_sequences: int,
    num_kv_heads: int,
    split_kv_blocks: int = 1,
    *,
    sliding_window: int = 0,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    query_length: int = 1,
) -> int:
    """Recommend ``max_context_partition_num`` for PS partitioned paths.

    For sliding-window PS, this includes the old
    ``get_sw_ps_max_context_partition_num`` token-window calculation. For
    non-sliding PS, this mirrors ``get_recommended_splits`` in
    ``aiter/ops/triton/gluon/pa_decode_gluon.py`` so FlyDSL callers do not need
    to depend on aiter for the host-side split count.
    """
    if sliding_window > 0:
        window_token_count = sliding_window + query_length
        return cdiv(window_token_count - 1, context_partition_size) + 1

    props = torch.cuda.get_device_properties(torch.device("cuda"))
    # Reference uses occupancy = 2 (see `get_occupancy()` in the Gluon module).
    occupancy = 2
    num_sm = props.multi_processor_count * occupancy
    denom = max(1, num_sequences * num_kv_heads * split_kv_blocks)
    n = cdiv(num_sm, denom) * split_kv_blocks
    return max(4, min(n, 8))


def _validate_metadata(query, block_tables, context_lengths):
    if context_lengths.ndim != 1 or context_lengths.numel() == 0:
        raise ValueError(
            "pa_decode_tile: context_lengths must be a non-empty 1D tensor, "
            f"got shape {tuple(context_lengths.shape)}"
        )
    num_seqs = context_lengths.shape[0]
    if (
        block_tables.ndim != 2
        or block_tables.shape[0] != num_seqs
        or block_tables.shape[1] == 0
    ):
        raise ValueError(
            f"pa_decode_tile: block_tables must have shape [{num_seqs}, max_blocks > 0], "
            f"got {tuple(block_tables.shape)}"
        )
    for name, metadata in (
        ("block_tables", block_tables),
        ("context_lengths", context_lengths),
    ):
        if metadata.device != query.device:
            raise ValueError(
                f"pa_decode_tile: {name} must be on {query.device}, got {metadata.device}"
            )
        if not metadata.is_contiguous():
            raise ValueError(
                f"pa_decode_tile: {name} must be contiguous, got strides {metadata.stride()}"
            )


def pa_decode_tile(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    key_scale: float | torch.Tensor | None,
    value_scale: float | torch.Tensor | None,
    softmax_scale: float | None = None,
    stream=None,
    *,
    num_partitions: int | None = None,
    pmax: torch.Tensor | None = None,
    psum: torch.Tensor | None = None,
    pout: torch.Tensor | None = None,
) -> None:
    """Host entry point. See module docstring for tensor layouts.

    ``num_partitions``/``pmax``/``psum``/``pout`` are optional caller overrides
    (e.g. for CUDA-graph capture, where nothing may be allocated on-the-fly);
    when omitted they are picked/allocated here.
    """
    _validate_metadata(query, block_tables, context_lengths)
    if query.ndim != 3 or output.ndim != 3 or min(query.shape) <= 0:
        raise ValueError("pa_decode_tile: query/output must be non-empty 3D tensors")
    if query.device.type != "cuda" or query.device.index != torch.cuda.current_device():
        raise ValueError("pa_decode_tile: query must be on the current CUDA/HIP device")
    for name, tensor in (
        ("output", output),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
    ):
        if tensor.device != query.device:
            raise ValueError(f"pa_decode_tile: {name} must be on {query.device}")
    for name, tensor in (("key_cache", key_cache), ("value_cache", value_cache)):
        if not tensor.is_contiguous():
            raise ValueError(f"pa_decode_tile: {name} must be contiguous")
    if key_cache.ndim != 5 or value_cache.ndim not in (4, 5):
        raise ValueError("pa_decode_tile: expected 5D K and 4D/5D V caches")
    if key_cache.shape[0] != value_cache.shape[0] or min(key_cache.shape) <= 0:
        raise ValueError(
            "pa_decode_tile: K/V must have matching, non-empty page counts"
        )
    if num_partitions is not None and (
        isinstance(num_partitions, bool)
        or not isinstance(num_partitions, int)
        or num_partitions < 0
    ):
        raise ValueError(
            "pa_decode_tile: num_partitions must be a positive integer, or None/0 for automatic splits"
        )
    if stream is not None and (
        not isinstance(stream, torch.cuda.Stream) or stream.device != query.device
    ):
        raise ValueError(
            "pa_decode_tile: stream must be a torch.cuda.Stream on the query device"
        )
    is_graph_capturing = _is_current_stream_capturing()
    num_seqs = context_lengths.shape[0]

    total_q_rows, num_q_heads, head_dim = query.shape
    value_head_dim = output.shape[-1]
    assert (
        total_q_rows % num_seqs == 0
    ), f"query.shape[0] ({total_q_rows}) must be a multiple of context_lengths.shape[0] ({num_seqs})"
    query_length = total_q_rows // num_seqs
    _, num_kv_heads, num_hgroups, block_size, hgroup_width = key_cache.shape
    arch = str(get_rocm_arch()).split(":", 1)[0]
    fp8_dtype = torch.float8_e4m3fn if "gfx95" in arch else torch.float8_e4m3fnuz
    if key_cache.dtype not in (torch.bfloat16, fp8_dtype):
        raise ValueError(
            f"pa_decode_tile supports BF16 or {fp8_dtype} KV cache on {arch}, got {key_cache.dtype}"
        )
    is_bf16_kv = key_cache.dtype == torch.bfloat16
    kv_dtype = "bf16" if is_bf16_kv else "fp8"

    expected_vector = 8 if is_bf16_kv else 16
    assert (
        num_hgroups == head_dim // expected_vector and hgroup_width == expected_vector
    ), (
        f"key_cache shape {tuple(key_cache.shape)} does not match head_dim={head_dim}, "
        f"vector_width={expected_vector}"
    )
    assert block_size in (
        16,
        64,
    ), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"

    trans_v = value_cache.dim() == 5
    if trans_v:
        _, v_num_kv_heads, v_subblocks, v_head_dim, v_width = value_cache.shape
        assert (
            v_head_dim == value_head_dim
            and v_width == expected_vector
            and v_subblocks == block_size // expected_vector
        ), (
            f"value_cache shape {tuple(value_cache.shape)} doesn't match "
            f"block_size={block_size}, value_head_dim={value_head_dim}"
        )
    else:
        assert not is_bf16_kv, "BF16 KV requires the vectorized-5D value-cache layout"
        _, v_num_kv_heads, v_head_dim, v_block_size = value_cache.shape
        assert v_head_dim == value_head_dim and v_block_size == block_size, (
            f"value_cache shape {tuple(value_cache.shape)} doesn't match "
            f"block_size={block_size}, value_head_dim={value_head_dim}"
        )
    assert v_num_kv_heads == num_kv_heads
    assert (
        value_cache.dtype == key_cache.dtype
    ), f"key/value cache dtype mismatch: {key_cache.dtype} vs {value_cache.dtype}"
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by "
        f"num_kv_heads ({num_kv_heads})"
    )
    assert (
        block_tables.dtype == torch.int32
    ), f"block_tables must be int32, got {block_tables.dtype}"
    assert (
        context_lengths.dtype == torch.int32
    ), f"context_lengths must be int32, got {context_lengths.dtype}"
    query_group_size = num_q_heads // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]
    if query.dtype == torch.bfloat16:
        query_dtype = "bf16"
    elif query.dtype == torch.float16:
        query_dtype = "f16"
    else:
        raise ValueError(
            f"pa_decode_tile only supports f16/bf16 query, got {query.dtype}"
        )
    assert (
        output.dtype == query.dtype
    ), f"pa_decode_tile requires output.dtype == query.dtype, got {output.dtype} vs {query.dtype}"
    assert output.shape == (total_q_rows, num_q_heads, value_head_dim), (
        "pa_decode_tile output must be [total_q_rows, num_q_heads, value_head_dim], "
        f"got {tuple(output.shape)}"
    )

    assert (
        query.stride(2) == 1
    ), f"pa_decode_tile requires a contiguous head_dim axis, got strides {query.stride()}"
    assert output.stride(2) == 1, (
        "pa_decode_tile requires a contiguous value_head_dim axis, "
        f"got strides {output.stride()}"
    )

    dev = query.device
    if is_bf16_kv:
        if key_scale is not None or value_scale is not None:
            raise ValueError(
                "BF16 KV is unscaled; key_scale and value_scale must be None"
            )
        # The BF16 specialization does not dereference scale pointers. Reuse an
        # existing tensor so the host path stays allocation- and graph-free.
        key_scale_t = query
        value_scale_t = query
        per_token_kv = False
    else:
        if key_scale is None or value_scale is None:
            raise ValueError("FP8 KV requires key_scale and value_scale")
        if stream is not None:
            _require_stream_scale_tensors(dev, key_scale, value_scale)
        if is_graph_capturing and not all(
            isinstance(scale, torch.Tensor) for scale in (key_scale, value_scale)
        ):
            raise ValueError(
                "CUDA graph capture requires pre-created FP8 scale tensors"
            )
        key_scale_t = (
            key_scale
            if isinstance(key_scale, torch.Tensor)
            else torch.tensor([float(key_scale)], dtype=torch.float32, device=dev)
        )
        value_scale_t = (
            value_scale
            if isinstance(value_scale, torch.Tensor)
            else torch.tensor([float(value_scale)], dtype=torch.float32, device=dev)
        )
        per_token_kv = key_scale_t.dim() > 1
        for name, scale in (("key_scale", key_scale_t), ("value_scale", value_scale_t)):
            if not scale.is_contiguous():
                raise ValueError(f"pa_decode_tile: {name} must be contiguous")
            if not per_token_kv and scale.numel() != 1:
                raise ValueError(
                    f"pa_decode_tile: scalar {name} must contain one element"
                )
    if per_token_kv:
        assert (
            value_scale_t.dim() > 1
        ), "value_scale must also be per-token (dim>1) when key_scale is per-token"
        assert (
            key_scale_t.shape == value_scale_t.shape
        ), f"key_scale/value_scale shape mismatch: {tuple(key_scale_t.shape)} vs {tuple(value_scale_t.shape)}"
        assert key_scale_t.shape == (key_cache.shape[0], num_kv_heads, block_size), (
            "per-token key_scale/value_scale must be [num_blocks, num_kv_heads, block_size] "
            f"matching the KV cache, got {tuple(key_scale_t.shape)}"
        )
        stride_ks_block = int(key_scale_t.stride(0))
        stride_ks_head = int(key_scale_t.stride(1))
    else:
        stride_ks_block = 0
        stride_ks_head = 0
    if not is_bf16_kv:
        assert (
            key_scale_t.dtype == torch.float32 and key_scale_t.device == dev
        ), f"key_scale tensor must be float32 on {dev}, got {key_scale_t.dtype} on {key_scale_t.device}"
        assert (
            value_scale_t.dtype == torch.float32 and value_scale_t.device == dev
        ), f"value_scale tensor must be float32 on {dev}, got {value_scale_t.dtype} on {value_scale_t.device}"

    if not num_partitions:
        blocks_per_partition = KV_COMPUTE_BLOCK // block_size
        num_partitions = get_recommended_splits(
            num_seqs,
            num_kv_heads,
            split_kv_blocks=blocks_per_partition,
        )

    supplied = (pmax, psum, pout)
    if (
        num_partitions > 1
        and any(t is not None for t in supplied)
        and not all(t is not None for t in supplied)
    ):
        raise ValueError("pa_decode_tile: provide all of pmax/psum/pout, or none")
    for name, tensor, dtype in (
        ("pmax", pmax, torch.float32),
        ("psum", psum, torch.float32),
        ("pout", pout, output.dtype),
    ):
        if tensor is not None and (
            tensor.device != dev
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
            or tensor.numel() == 0
        ):
            raise ValueError(
                f"pa_decode_tile: {name} must be non-empty, contiguous {dtype} on {dev}"
            )

    # A full-query wave-local CTA saves KV reloads, but underfills small grids.
    if (
        arch == "gfx950"
        and is_bf16_kv
        and query_dtype == "bf16"
        and query_length == 8
        and query_group_size == 16
        and block_size == 64
        and head_dim in (128, 192)
        and value_head_dim == 128
        and num_seqs * num_kv_heads * num_partitions >= 256
        and (softmax_scale is None or 0.0 < softmax_scale < float("inf"))
    ):
        from aiter.ops.flydsl.kernels.pa_decode_bf16_wave import (
            compile_pa_decode_bf16_wave,
        )

        compiled = compile_pa_decode_bf16_wave(head_dim, num_partitions, softmax_scale)
    elif (
        arch == "gfx950"
        and kv_dtype == "fp8"
        and query_dtype == "bf16"
        and query_length == 8
        and query_group_size == 16
        and block_size == 64
        and head_dim in (128, 192)
        and value_head_dim == 128
        and not per_token_kv
        and trans_v
        and (softmax_scale is None or 0.0 < softmax_scale < float("inf"))
    ):
        if num_seqs * num_kv_heads * num_partitions <= 64:
            from aiter.ops.flydsl.kernels.pa_decode_fp8_small import (
                compile_pa_decode_fp8_small,
            )

            compiled = compile_pa_decode_fp8_small(
                head_dim, num_partitions, softmax_scale
            )
        else:
            from aiter.ops.flydsl.kernels.pa_decode_fp8_wave import (
                compile_pa_decode_fp8_wave,
            )

            compiled = compile_pa_decode_fp8_wave(
                head_dim, num_partitions, softmax_scale
            )
    else:
        compiled = compile_pa_decode_tile(
            head_dim=head_dim,
            v_head_dim=value_head_dim,
            query_group_size=query_group_size,
            block_size=int(block_size),
            num_partitions=num_partitions,
            softmax_scale=softmax_scale,
            query_dtype=query_dtype,
            per_token_kv=per_token_kv,
            query_length=query_length,
            trans_v=trans_v,
            kv_dtype=kv_dtype,
        )
    if num_partitions == 1:
        # NP==1 writes output directly; partials unused (caller buffers ignored).
        if pmax is None:
            if is_graph_capturing:
                raise ValueError(
                    "CUDA graph capture requires preallocated `pmax`/`psum`/`pout` "
                    "even when num_partitions==1 (nothing may be allocated mid-capture)."
                )
            pmax = psum = pout = torch.empty(1, dtype=torch.float32, device=dev)
        else:
            # The launch still needs dummy pointer arguments; NP1 never reads them.
            psum = pmax if psum is None else psum
            pout = pmax if pout is None else pout
    else:
        total_rows = query_length * query_group_size
        expected_scalar_shape = (num_seqs, num_kv_heads, num_partitions, total_rows)
        if pmax is None or psum is None or pout is None:
            if is_graph_capturing:
                raise ValueError(
                    "CUDA graph capture requires preallocated `pmax`/`psum`/`pout` "
                    "for num_partitions>1 (nothing may be allocated mid-capture)."
                )
            pmax = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
            psum = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
            pout = torch.empty(
                *expected_scalar_shape,
                value_head_dim,
                dtype=output.dtype,
                device=dev,
            )
        else:
            assert (
                pmax.shape == expected_scalar_shape
            ), f"pmax shape {tuple(pmax.shape)} != {expected_scalar_shape}"
            assert (
                psum.shape == expected_scalar_shape
            ), f"psum shape {tuple(psum.shape)} != {expected_scalar_shape}"
            assert pout.shape == (
                *expected_scalar_shape,
                value_head_dim,
            ), (
                f"pout shape {tuple(pout.shape)} != "
                f"{(*expected_scalar_shape, value_head_dim)}"
            )
    s = stream or torch.cuda.current_stream()
    if stream is not None:
        for tensor in (
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale_t,
            value_scale_t,
            pmax,
            psum,
            pout,
        ):
            tensor.record_stream(s)

    _run_compiled(
        compiled["launch"],
        output,
        pmax.view(-1),
        psum.view(-1),
        pout.view(-1),
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        key_scale_t,
        value_scale_t,
        int(max_blocks_per_seq),
        int(num_seqs),
        int(num_kv_heads),
        stride_ks_block,
        stride_ks_head,
        int(query.stride(0)),
        int(query.stride(1)),
        s,
    )
    if num_partitions > 1:
        from aiter.ops.flydsl.kernels.pa_decode_reduce import (
            compile_pa_decode_sw_reduce,
        )

        reduce_compiled = compile_pa_decode_sw_reduce(
            max_context_partition_num=num_partitions,
            query_seq_len=query_length,
            query_group_size=query_group_size,
            head_size=value_head_dim,
            output_dtype_str=_get_output_dtype_str(output),
            logits_dtype_str=_get_output_dtype_str(pout),
        )
        _run_compiled(
            reduce_compiled["launch"],
            output.data_ptr(),
            psum.data_ptr(),  # exp_sums
            pmax.data_ptr(),  # max_logits
            pout.data_ptr(),  # logits (already-normalized query_dtype partials)
            query_length
            * output.stride(
                0
            ),  # stride_output_bs: per TRUE seq, spans all query_length rows
            output.stride(
                0
            ),  # stride_output_len: per MTP position (query_length==1: unused, multiplied by 0)
            query_group_size * output.stride(1),
            output.stride(1),
            pmax.stride(0),
            pmax.stride(1),
            pmax.stride(2),
            pout.stride(0),
            pout.stride(1),
            pout.stride(2),
            pout.stride(3),
            int(num_seqs),
            int(num_kv_heads),
            s,
        )


def pa_decode_ps_launch(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_page_indices: torch.Tensor,  # [total_pages] int32
    kv_indptr: torch.Tensor,  # [num_seqs + 1] int32
    softmax_scale: float,
    key_scale: torch.Tensor = None,
    value_scale: torch.Tensor = None,
    *,
    sliding_window: int = 0,
    metadata: dict = None,
    block_tables: torch.Tensor = None,  # [num_seqs, max_blocks_per_seq] i32
    max_context_partition_num: int = 0,
    exp_sums: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    temporary_output: torch.Tensor = None,
    stream=None,
) -> str:
    """Small-page compatibility entrypoint using the normalized-partial ABI.

    ``block_tables`` is authoritative; CSR indices/indptr and ``metadata`` are
    accepted for call-site compatibility but are not read or constructed.
    """
    if sliding_window != 0 or key_cache.shape[-2] not in (16, 64):
        raise ValueError(
            "BF16 KV and asymmetric value dimensions currently require "
            "block_size 16 or 64 with sliding_window=0; this API is full-attention only."
        )
    if block_tables is None:
        raise ValueError("pa_decode_ps_launch: page size 16/64 requires `block_tables`")
    _validate_metadata(query, block_tables, context_lengths)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    batch_size = context_lengths.shape[0]
    if query.shape[0] % batch_size != 0:
        raise ValueError(
            f"query.shape[0] ({query.shape[0]}) must be divisible by "
            f"batch_size ({batch_size})"
        )
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )
    block_size = key_cache.shape[-2]
    dev = query.device
    is_graph_capturing = _is_current_stream_capturing()
    s = stream or torch.cuda.current_stream()
    if stream is not None and key_cache.dtype != torch.bfloat16:
        _require_stream_scale_tensors(dev, key_scale, value_scale)

    if max_context_partition_num <= 0:
        raise ValueError("max_context_partition_num must be positive.")

    # Small physical pages use the standalone tile kernel. Dispatch before the
    # FP8 metadata/SW setup so BF16 KV stays unscaled and asymmetric V uses the
    # tile wrapper's value-sized workspace allocation and validation.
    if block_size in _PA_DECODE_PS_SMALL_BLOCK_SIZES and sliding_window == 0:
        if block_tables is None:
            raise ValueError(
                f"pa_decode_ps_launch: block_size={block_size} requires `block_tables` "
                "(per-sequence physical block index table)."
            )

        tile_key_scale = key_scale
        tile_value_scale = value_scale
        if key_cache.dtype != torch.bfloat16:
            tile_key_scale = _prepare_scale_tensor(
                "key_scale",
                key_scale,
                device=dev,
                is_graph_capturing=is_graph_capturing,
            )
            tile_value_scale = _prepare_scale_tensor(
                "value_scale",
                value_scale,
                device=dev,
                is_graph_capturing=is_graph_capturing,
            )
            if tile_key_scale.ndim > 1:
                num_blocks = key_cache.shape[0]
                tile_key_scale = tile_key_scale.reshape(
                    num_blocks, num_kv_heads, block_size
                )
                tile_value_scale = tile_value_scale.reshape(
                    num_blocks, num_kv_heads, block_size
                )

        pa_decode_tile(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            tile_key_scale,
            tile_value_scale,
            softmax_scale=softmax_scale,
            stream=s,
            num_partitions=max_context_partition_num,
            pmax=max_logits,
            psum=exp_sums,
            pout=temporary_output,
        )
        return "ps_small_block"

    raise ValueError(
        "pa_decode_ps_launch requires full attention with page size 16 or 64"
    )


flydsl_pa_decode_tile = pa_decode_tile
flydsl_pa_decode_ps = pa_decode_ps_launch

__all__ = ["flydsl_pa_decode_tile", "flydsl_pa_decode_ps"]
