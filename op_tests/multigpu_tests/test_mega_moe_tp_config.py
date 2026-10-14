# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-only tests for bounded MiMo TP8 A8W4 dynamic configuration."""

import pytest

from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp import MegaMoeTPEngine
from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp_config import (
    DYN_MAX,
    MIMO_A8_DYN_MAX,
    dynamic_token_limit,
    kernel_config,
)


def engine(comm_bf16=False):
    e = MegaMoeTPEngine.__new__(MegaMoeTPEngine)
    e.H, e.I, e.E, e.K, e.tp, e.activation = 6144, 256, 384, 8, 8, "silu"
    e.a8, e.ar, e.comm_bf16 = True, False, comm_bf16
    e.a8_lb = True
    e.n_cta, e.mmax, e.lb_min, e.ag_auto = 256, 256, 0, 512
    e.mode = "ag_rs"
    e.lb_mt, e.lb_mt_table, e.lb_mt_small = 4, [(1024, 2)], 2
    e.lb_small_max, e.lb_npp, e.lb_q = 512, 1, 3
    e.dyn_limit = dynamic_token_limit(
        e.H, e.I, e.E, e.K, e.tp, e.ar, e.activation, e.a8
    )
    e.dyn_max = e.dyn_limit
    e._cfgs = {}
    return e


@pytest.mark.parametrize("tokens", [256, 264, 512, 520, 1024, 1032, 2048])
@pytest.mark.parametrize("comm_bf16", [False, True])
def test_extended_layout(tokens, comm_bf16):
    e = engine(comm_bf16)
    cfg = e.config(tokens // 8)
    rch, nch = e._chunks(cfg.mt)
    kc = kernel_config(
        H=e.H,
        I=e.I,
        E=e.E,
        TOPK=e.K,
        tp=e.tp,
        TMAX=2048,
        MT=cfg.mt,
        nab=e._nab(cfg.mt),
        npp=cfg.npp,
        a8=True,
        comm_bf16=comm_bf16,
        ll_rs=cfg.ll,
        ll_route=cfg.llr,
        route_fp8=cfg.ll and cfg.llr,
        rch=rch,
        nch=nch,
        dyn_p=cfg.dp,
        dyn_limit=e.dyn_limit,
    )
    assert kc.NRIX == 2048
    assert kc.SCAN_IT == 16
    assert kc.L_WT - kc.L_RIX == 2048 * 4
    assert kc.L_CTL - kc.L_WT == 2048 * 4
    assert kc.LDS_BYTES <= 160 * 1024
    if tokens > DYN_MAX:
        assert cfg.mt == 1 and cfg.npp == 2 and cfg.dp == 1
        assert not cfg.ll and not cfg.llr and not cfg.lb
        assert kc.NPC == 1
        assert (nch + e.n_cta - 1) // e.n_cta <= 8


def test_bound_and_shape_scope():
    e = engine()
    with pytest.raises(ValueError, match="2056 tokens"):
        e.config(257)
    assert dynamic_token_limit(6144, 256, 384, 8, 8, False, "silu", True) == 2048
    for args in (
        (6144, 256, 384, 8, 8, True, "silu", True),
        (6144, 256, 384, 8, 4, False, "silu", True),
        (6144, 256, 384, 8, 8, False, "silu", False),
        (6144, 256, 384, 8, 8, False, "swiglu", True),
        (6144, 384, 384, 8, 8, False, "silu", True),
    ):
        assert dynamic_token_limit(*args) == DYN_MAX


def test_limit_enters_kernel_identity():
    kw = {
        "H": 6144,
        "I": 256,
        "E": 384,
        "TOPK": 8,
        "tp": 8,
        "TMAX": 256,
        "MT": 1,
        "a8": True,
    }
    original = kernel_config(**kw)
    extended = kernel_config(**kw, dyn_limit=MIMO_A8_DYN_MAX)
    assert original.name != extended.name
    assert original.NRIX == extended.NRIX == 256
    with pytest.raises(AssertionError):
        kernel_config(**kw, ar=True, dyn_limit=MIMO_A8_DYN_MAX)


@pytest.mark.parametrize("mt", [1, 2, 3, 4])
@pytest.mark.parametrize("prefetch", [False, True])
def test_a8_lb_layout(mt, prefetch):
    e = engine()
    rch, nch = e._chunks(mt, lb=True)
    kw = {
        "H": 6144,
        "I": 256,
        "E": 384,
        "TOPK": 8,
        "tp": 8,
        "TMAX": 2048,
        "MT": mt,
        "a8": True,
        "lb": True,
        "rch": rch,
        "nch": nch,
        "nab": 3,
        "npp": 2,
        "lbp": 1,
        "lbpf": prefetch,
        "dyn_limit": 2048,
    }
    kc = kernel_config(**kw)
    assert kc.IB == 256 and kc.NRIX == 256 and kc.NPC == 1
    assert kc.LBPF == prefetch
    assert kc.LDS_BYTES <= 160 * 1024
    assert kc.LB_PS == [1]
    assert kc.nch <= 2048
    with pytest.raises(AssertionError):
        kernel_config(**{**kw, "ar": True})


def test_mimo_retained_schedule_and_tail_contract():
    e = engine()
    e.lb_min = 513
    for tokens, mt, lb in (
        (256, 1, False),
        (512, 1, False),
        (520, 2, True),
        (1024, 2, True),
        (1032, 4, True),
        (2048, 4, True),
    ):
        cfg = e.config(tokens // 8)
        assert cfg.mt == mt and cfg.lb == lb
        assert cfg.npp == 2
        if lb:
            assert cfg.lp == 1 and not cfg.pf
        assert not e.tail_ok(tokens // 8)
    assert e.lb_q == 3
    with pytest.raises(ValueError, match="2056 tokens"):
        e.config(257)
